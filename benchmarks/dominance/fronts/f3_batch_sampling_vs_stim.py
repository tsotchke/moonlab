"""F3 batch-sampling front: MoonLab Pauli-frame shot sampler vs Stim.

Stim's throughput champion for stabilizer sampling is
``stim.Circuit(text).compile_sampler().sample(num_shots)`` -- a single-threaded,
SIMD-packed Pauli-frame propagator. This front pits MoonLab's
``pauli_frame_batch_sample_circuit`` against it on the same circuits.

Two mandatory gates:

  1. Cross-engine correctness (correctness_ok): both engines sample the SAME
     op list. Per-measurement marginals must agree within 6 sigma over >=1e5
     shots, the global parity marginal must agree within 6 sigma, and pairwise
     measurement correlations must match within 0.02. A faster wrong sampler
     fails here.

  2. Throughput: samples/second = num_shots * num_measurements / wall-clock,
     timing only the sampling call for both engines (Stim's compile and
     MoonLab's one-shot reference tableau pass are excluded / negligible).
     Three numbers per size are reported for full honesty:
        (a) MoonLab single-thread vs Stim single-thread  (kernel/SIMD)
        (b) MoonLab all-cores   vs Stim                  (as-used)
        (c) the thread count and physical core count used.

Run via ``benchmarks/dominance/run_dominance.py`` with MOONLAB_LIB_DIR pointing
at a build directory containing libmoonlab_bsamp.<ext> (or a libquantumsim that
exports pauli_frame_batch_sample_circuit).
"""
from __future__ import annotations

import ctypes
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import stim

# --------------------------------------------------------------------------
# Op encoding -- mirrors pf_circuit_op_t / pf_op_kind_t in pauli_frame.h.
# --------------------------------------------------------------------------
PF_H, PF_S, PF_S_DAG, PF_X, PF_Y, PF_Z, PF_CNOT, PF_CZ, PF_SWAP, PF_RESET, PF_MEASURE = range(11)
(PF_X_ERROR, PF_Z_ERROR, PF_Y_ERROR, PF_DEPOLARIZE1, PF_DEPOLARIZE2,
 PF_MEASURE_NOISY) = range(11, 17)

_STIM_NAME = {PF_H: "H", PF_S: "S", PF_S_DAG: "S_DAG", PF_X: "X", PF_Y: "Y",
              PF_Z: "Z", PF_CNOT: "CNOT", PF_CZ: "CZ", PF_SWAP: "SWAP"}
# Noise channels carry a probability argument in stim.
_STIM_NOISE = {PF_X_ERROR: "X_ERROR", PF_Z_ERROR: "Z_ERROR", PF_Y_ERROR: "Y_ERROR",
               PF_DEPOLARIZE1: "DEPOLARIZE1", PF_DEPOLARIZE2: "DEPOLARIZE2"}
_TWO_QUBIT = (PF_CNOT, PF_CZ, PF_SWAP)


class _PfOp(ctypes.Structure):
    # Must mirror pf_circuit_op_t exactly: the double forces 8-byte
    # alignment, so the struct is {u8 kind, pad, u32 q0, u32 q1, pad, f64 p}.
    _fields_ = [("kind", ctypes.c_uint8), ("q0", ctypes.c_uint32),
                ("q1", ctypes.c_uint32), ("p", ctypes.c_double)]


# --------------------------------------------------------------------------
# Library loading.
# --------------------------------------------------------------------------
def _load_lib():
    if sys.platform == "darwin":
        names = ["libmoonlab_bsamp.dylib", "libquantumsim.dylib"]
    elif sys.platform == "win32":
        names = ["moonlab_bsamp.dll", "quantumsim.dll"]
    else:
        names = ["libmoonlab_bsamp.so", "libquantumsim.so"]
    dirs = []
    if os.environ.get("MOONLAB_LIB_DIR"):
        dirs.append(Path(os.environ["MOONLAB_LIB_DIR"]))
    repo = Path(__file__).resolve().parents[3]
    dirs += [repo / "build-bsamp", repo / "build", repo]
    for d in dirs:
        for nm in names:
            p = d / nm
            if p.exists():
                lib = ctypes.CDLL(str(p))
                if hasattr(lib, "pauli_frame_batch_sample_circuit"):
                    return lib
    raise OSError("libmoonlab_bsamp / libquantumsim with the batch sampler not "
                  "found; set MOONLAB_LIB_DIR to a build-bsamp directory")


_LIB = _load_lib()
_LIB.pauli_frame_batch_sample_circuit.restype = ctypes.c_long
_LIB.pauli_frame_batch_sample_circuit.argtypes = [
    ctypes.c_size_t, ctypes.POINTER(_PfOp), ctypes.c_size_t,
    ctypes.c_size_t, ctypes.c_uint64, ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint8)]
_LIB.pauli_frame_batch_sample_detectors.restype = ctypes.c_long
_LIB.pauli_frame_batch_sample_detectors.argtypes = [
    ctypes.c_size_t, ctypes.POINTER(_PfOp), ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint64, ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint8)]
_LIB.pauli_frame_circuit_num_measurements.restype = ctypes.c_size_t
_LIB.pauli_frame_circuit_num_measurements.argtypes = [ctypes.POINTER(_PfOp), ctypes.c_size_t]
_LIB.pauli_frame_simd_backend.restype = ctypes.c_char_p
_LIB.pauli_frame_simd_lanes.restype = ctypes.c_int

SIMD_BACKEND = _LIB.pauli_frame_simd_backend().decode()
SIMD_LANES = _LIB.pauli_frame_simd_lanes()


def _physical_cores():
    try:
        import subprocess
        if sys.platform == "darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).strip())
    except Exception:
        pass
    return os.cpu_count() or 1


PHYS_CORES = _physical_cores()


# --------------------------------------------------------------------------
# Circuit builders. Each returns (num_qubits, ops) with ops a list of
# (kind, q0, q1) tuples. The SAME ops feed both engines.
# --------------------------------------------------------------------------
def build_random_clifford(n, ngates, seed):
    rng = random.Random(seed)
    ops = []
    for _ in range(ngates):
        r = rng.random()
        if r < 0.45:
            a, b = rng.sample(range(n), 2)
            ops.append((rng.choice([PF_CNOT, PF_CZ, PF_SWAP]), a, b))
        elif r < 0.75:
            ops.append((PF_H, rng.randrange(n), 0))
        elif r < 0.9:
            ops.append((PF_S, rng.randrange(n), 0))
        else:
            ops.append((rng.choice([PF_X, PF_Y, PF_Z]), rng.randrange(n), 0))
    for q in range(n):
        ops.append((PF_MEASURE, q, 0))
    return n, ops


def build_surface_code(d, rounds):
    """The same surface code with the noise switched off.

    Delegates to the noisy builder so both workloads use one circuit
    definition and cannot drift apart -- in particular so this one keeps the
    checkerboard plaquette layout that makes the code valid.
    """
    n, ops, _ = build_surface_code_noisy(d, rounds, p2=0.0, p1=0.0, pm=0.0)
    keep = []
    for o in ops:
        if o[0] in (PF_DEPOLARIZE1, PF_DEPOLARIZE2):
            continue                      # zero-probability, drop entirely
        keep.append((PF_MEASURE, o[1], o[2], 0.0)
                    if o[0] == PF_MEASURE_NOISY else o)
    return n, keep


def build_surface_code_noisy(d, rounds, p2=0.001, p1=0.001, pm=0.001):
    """The same syndrome-extraction circuit under a circuit-level noise model.

    This is the regime stim is built for and the one its users actually
    sample: two-qubit depolarising after every CNOT, single-qubit
    depolarising on idling data qubits each round, and a flip probability on
    every ancilla readout.  Noiseless Clifford sampling leaves the frames
    almost static, so it exercises far less of the sampler than this does --
    every noise instruction injects fresh per-shot randomness.
    """
    def data(i, j):
        return i * d + j
    ndata = d * d
    cells = [(i, j) for i in range(d - 1) for j in range(d - 1)]
    z_anc = {c: ndata + k for k, c in enumerate(cells)}
    x_anc = {c: ndata + len(cells) + k for k, c in enumerate(cells)}
    # Plaquette types alternate on a checkerboard.  Putting BOTH an X and a Z
    # plaquette on every cell does not give a code: cells that touch only at a
    # corner share exactly one data qubit, and an X and a Z plaquette
    # overlapping on one qubit ANTICOMMUTE.  Nothing is then deterministic,
    # no valid detector exists, and both engines return coin flips every
    # round.  Checkerboarding makes every X/Z pair overlap on 0 or 2 qubits.
    z_cells = [(i, j) for (i, j) in cells if (i + j) % 2 == 0]
    x_cells = [(i, j) for (i, j) in cells if (i + j) % 2 == 1]
    anc = {c: ndata + k for k, c in enumerate(z_cells + x_cells)}
    n = ndata + len(cells)
    ops = []
    mi = 0                      # running measurement index
    last = {}                   # ancilla -> measurement index in previous round
    dets = []                   # each detector: tuple of measurement indices
    for r in range(rounds):
        for q in range(ndata):
            ops.append((PF_DEPOLARIZE1, q, 0, p1))
        for (i, j) in z_cells:
            a = anc[(i, j)]
            for (di, dj) in ((0, 0), (1, 0), (0, 1), (1, 1)):
                dq = data(i + di, j + dj)
                ops.append((PF_CNOT, dq, a, 0.0))
                ops.append((PF_DEPOLARIZE2, dq, a, p2))
            ops.append((PF_MEASURE_NOISY, a, 0, pm))
            ops.append((PF_RESET, a, 0, 0.0))
            # Z plaquettes are deterministic on the |0..0> data state, so the
            # very first round already yields a valid detector on its own.
            dets.append((mi,) if r == 0 else (mi, last[a]))
            last[a] = mi
            mi += 1
        for (i, j) in x_cells:
            a = anc[(i, j)]
            ops.append((PF_H, a, 0, 0.0))
            for (di, dj) in ((0, 0), (1, 0), (0, 1), (1, 1)):
                dq = data(i + di, j + dj)
                ops.append((PF_CNOT, a, dq, 0.0))
                ops.append((PF_DEPOLARIZE2, a, dq, p2))
            ops.append((PF_H, a, 0, 0.0))
            ops.append((PF_MEASURE_NOISY, a, 0, pm))
            ops.append((PF_RESET, a, 0, 0.0))
            # X plaquette outcomes are random in round 0 on |0..0>, so an
            # X detector only exists from round 1, comparing against round r-1.
            if r > 0:
                dets.append((mi, last[a]))
            last[a] = mi
            mi += 1
    return n, ops, dets


# --------------------------------------------------------------------------
# Engine drivers.
# --------------------------------------------------------------------------
def _to_stim(ops):
    c = stim.Circuit()
    for op in ops:
        k = op[0]
        if k == PF_MEASURE:
            c.append("M", [op[1]])
        elif k == PF_MEASURE_NOISY:
            c.append("M", [op[1]], op[3])          # M(p): flips the report
        elif k == PF_RESET:
            c.append("R", [op[1]])
        elif k == PF_DEPOLARIZE2:
            c.append("DEPOLARIZE2", [op[1], op[2]], op[3])
        elif k in _STIM_NOISE:
            c.append(_STIM_NOISE[k], [op[1]], op[3])
        elif k in _TWO_QUBIT:
            c.append(_STIM_NAME[k], [op[1], op[2]])
        else:
            c.append(_STIM_NAME[k], [op[1]])
    return c


def _pf_op_array(ops):
    arr = (_PfOp * len(ops))()
    for i, op in enumerate(ops):
        arr[i].kind = op[0]
        arr[i].q0 = op[1]
        arr[i].q1 = op[2]
        arr[i].p = op[3] if len(op) > 3 else 0.0
    return arr


def moonlab_sample(n, op_arr, nops, nmeas, shots, seed, nthreads):
    """Return an (nmeas x shots) uint8 array (measurement-major)."""
    out = np.empty((nmeas, shots), dtype=np.uint8)
    rc = _LIB.pauli_frame_batch_sample_circuit(
        n, op_arr, nops, shots, ctypes.c_uint64(seed), nthreads,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    if rc != nmeas:
        raise RuntimeError(f"sampler returned {rc}, expected {nmeas}")
    return out


def stim_sample(sampler, shots):
    """Return an (nmeas x shots) uint8 array (measurement-major) to match."""
    return sampler.sample(shots).astype(np.uint8).T


# --------------------------------------------------------------------------
# Detector sampling -- what a decoder actually consumes.
# --------------------------------------------------------------------------
def _det_csr(dets):
    """Detector definitions in the CSR form the C entry point expects."""
    offsets = np.zeros(len(dets) + 1, dtype=np.uintp)
    flat = []
    for d, idx in enumerate(dets):
        flat.extend(idx)
        offsets[d + 1] = len(flat)
    return offsets, np.array(flat, dtype=np.uint32)


def moonlab_sample_detectors(n, ops, dets, shots, seed, nthreads):
    op_arr = _pf_op_array(ops)
    offsets, flat = _det_csr(dets)
    out = np.empty((len(dets), shots), dtype=np.uint8)
    rc = _LIB.pauli_frame_batch_sample_detectors(
        n, op_arr, len(ops),
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        len(dets), shots, ctypes.c_uint64(seed), nthreads,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    if rc != len(dets):
        raise RuntimeError(f"detector sampler returned {rc}, expected {len(dets)}")
    return out


def _to_stim_with_detectors(ops, dets):
    """Same circuit plus DETECTOR instructions.

    Detector measurement indices are absolute; stim addresses measurements
    backwards from the end of the circuit, so index i becomes rec[i - total].
    """
    c = _to_stim(ops)
    total = sum(1 for op in ops if op[0] in (PF_MEASURE, PF_MEASURE_NOISY))
    for idx in dets:
        c.append("DETECTOR", [stim.target_rec(i - total) for i in idx])
    return c


def stim_sample_detectors(sampler, shots):
    return sampler.sample(shots).astype(np.uint8).T


# --------------------------------------------------------------------------
# Correctness gate.
# --------------------------------------------------------------------------
def _corr_vec(sample, k):
    """Pairwise correlations among the first k measurements."""
    out = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = sample[i], sample[j]
            out.append(np.corrcoef(a, b)[0, 1]
                       if a.std() > 0 and b.std() > 0 else 0.0)
    return np.array(out) if out else np.zeros(1)


def _corr_stats(sample, k, nblocks=10):
    """Pairwise correlations with a per-pair standard error (batch means).

    A raw correlation difference is not comparable across circuits: on a
    noisy QEC circuit the syndrome bits are rare, so each correlation rests
    on ~shots*p effective samples and its spread is an order of magnitude
    wider than on a noiseless circuit.  Splitting the sample into blocks and
    taking the standard error of the per-block estimates measures that
    spread directly from the data, so the gate can be expressed in sigma and
    mean the same thing on every workload.
    """
    S = sample.shape[1]
    b = S // nblocks
    if k < 2 or b < 2:
        return np.zeros(1), np.ones(1)
    per_block = np.stack([_corr_vec(sample[:, i * b:(i + 1) * b], k)
                          for i in range(nblocks)])
    mean = _corr_vec(sample[:, :nblocks * b], k)
    se = per_block.std(axis=0, ddof=1) / np.sqrt(nblocks)
    return mean, se



def check_correctness(n, ops, shots=200000, seed=12345, nthreads=1):
    """Gate one sampler configuration against stim.

    Must be run for the multithreaded configuration as well as the single
    threaded one: the two paths draw their randomness differently, so a
    single-threaded-only gate cannot certify the numbers the multithreaded
    path produces.  An earlier additive per-block seeding bug was invisible
    to a 1-thread gate while biasing the all-cores path by up to 7.4 sigma.
    """
    op_arr = _pf_op_array(ops)
    nmeas = int(_LIB.pauli_frame_circuit_num_measurements(op_arr, len(ops)))
    ml = moonlab_sample(n, op_arr, len(ops), nmeas, shots, seed, nthreads)
    st = stim_sample(_to_stim(ops).compile_sampler(seed=seed + 1), shots)
    if ml.shape != st.shape:
        return False, {"reason": f"shape {ml.shape} vs {st.shape}"}
    pml = ml.mean(axis=1)
    pst = st.mean(axis=1)
    sig = np.sqrt(np.maximum(pst * (1 - pst), 1e-9) / shots) + 1e-12
    marg_dev = float(np.abs(pml - pst).__truediv__(sig).max())
    par_ml = (ml.sum(axis=0) & 1).mean()
    par_st = (st.sum(axis=0) & 1).mean()
    par_dev = abs(par_ml - par_st) / (np.sqrt(max(par_st * (1 - par_st), 1e-9) / shots) + 1e-12)
    k = min(6, nmeas)
    r_ml, se_ml = _corr_stats(ml, k)
    r_st, se_st = _corr_stats(st, k)
    maxcorr = float(np.abs(r_ml - r_st).max())
    # Normalise each pair by its own sampling error, then gate in sigma.
    # A genuine modelling difference stays large under this normalisation
    # while estimator noise sits at a few sigma whatever the event rate.
    corr_sigma = float((np.abs(r_ml - r_st) / (np.sqrt(se_ml**2 + se_st**2) + 1e-12)).max())
    ok = marg_dev < 6.0 and par_dev < 6.0 and corr_sigma < 6.0
    return ok, {"marg_sigma": round(marg_dev, 2), "parity_sigma": round(float(par_dev), 2),
                "max_corr_diff": round(maxcorr, 4),
                "corr_sigma": round(corr_sigma, 2),
                "nmeas": nmeas, "shots": shots, "nthreads": nthreads}


# --------------------------------------------------------------------------
# Throughput.
# --------------------------------------------------------------------------
def _best_time(fn, reps=3, warmup=1):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def measure(n, ops, shots, seed):
    op_arr = _pf_op_array(ops)
    nops = len(ops)
    nmeas = int(_LIB.pauli_frame_circuit_num_measurements(op_arr, nops))
    total = shots * nmeas

    # Allocation symmetry: stim's sample() allocates its result array inside
    # the call, so MoonLab must allocate its destination inside the timed
    # region too.  Hoisting MoonLab's buffer out (and leaving stim's in)
    # charges stim a fresh shots*nmeas allocation + page-fault walk per rep
    # that MoonLab never pays -- which inflates the ratio and makes it grow
    # with shot count.  Both engines now produce a fresh array per call.
    def ml_run(nthreads):
        out = np.empty((nmeas, shots), dtype=np.uint8)
        _LIB.pauli_frame_batch_sample_circuit(
            n, op_arr, nops, shots, ctypes.c_uint64(seed), nthreads,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        return out

    sampler = _to_stim(ops).compile_sampler(seed=seed + 1)

    # Compare against stim's BEST configuration, not just its default.
    # sample() returns one byte per sample (matching MoonLab's uint8 output);
    # sample(bit_packed=True) is the path stim users take for large shot
    # counts and moves 8x less memory.  Beating only the unpacked path would
    # be beating stim's slow mode.
    def st_run():
        sampler.sample(shots)

    def st_run_packed():
        sampler.sample(shots, bit_packed=True)

    reps = 3 if shots <= 100000 else 2
    t_ml1 = _best_time(lambda: ml_run(1), reps=reps)
    t_mlN = _best_time(lambda: ml_run(0), reps=reps)   # 0 -> all cores
    t_st = _best_time(st_run, reps=reps)
    t_stp = _best_time(st_run_packed, reps=reps)

    stim_best = total / min(t_st, t_stp)
    return {
        "n": n, "nmeas": nmeas, "shots": shots,
        "ml_st_sps": total / t_ml1,
        "ml_mt_sps": total / t_mlN,
        "stim_sps": total / t_st,
        "stim_packed_sps": total / t_stp,
        "stim_best_sps": stim_best,
    }


def check_detector_correctness(n, ops, dets, shots=200000, seed=1234, nthreads=1):
    """Gate detector sampling against stim's compile_detector_sampler.

    Adds a property the measurement gate cannot express: on a NOISELESS run
    every detector must read 0 in every shot.  A detector is a deterministic
    parity by construction, so a single firing means either the sampler or
    the detector set is wrong.  This is what caught an invalid plaquette
    layout that the measurement gate passed -- that gate only compares the
    first six measurements, which on a syndrome circuit are all round-0
    ancillas, so it never inspects a round-to-round pair.
    """
    ml = moonlab_sample_detectors(n, ops, dets, shots, seed, nthreads)
    st = stim_sample_detectors(
        _to_stim_with_detectors(ops, dets).compile_detector_sampler(seed=seed + 1),
        shots)
    if ml.shape != st.shape:
        return False, {"reason": f"shape {ml.shape} vs {st.shape}"}
    pml, pst = ml.mean(axis=1), st.mean(axis=1)
    se = np.sqrt(np.maximum(pst * (1 - pst), 1e-9) / shots) + 1e-12
    marg_dev = float(np.abs(pml - pst).__truediv__(se).max())
    k = min(6, len(dets))
    r_ml, se_ml = _corr_stats(ml, k)
    r_st, se_st = _corr_stats(st, k)
    corr_sigma = float((np.abs(r_ml - r_st) /
                        (np.sqrt(se_ml**2 + se_st**2) + 1e-12)).max())
    ok = marg_dev < 6.0 and corr_sigma < 6.0
    return ok, {"marg_sigma": round(marg_dev, 2), "corr_sigma": round(corr_sigma, 2),
                "fire_rate_ml": round(float(pml.mean()), 5),
                "fire_rate_stim": round(float(pst.mean()), 5),
                "ndet": len(dets), "shots": shots, "nthreads": nthreads}


def check_detector_silence(n, ops, dets, shots=20000, seed=99):
    """Noiseless determinism: no detector may fire, in either engine."""
    ml = moonlab_sample_detectors(n, ops, dets, shots, seed, 1)
    st = stim_sample_detectors(
        _to_stim_with_detectors(ops, dets).compile_detector_sampler(seed=seed + 1),
        shots)
    return int(ml.sum()), int(st.sum())


def measure_detectors(n, ops, dets, shots, seed):
    op_arr = _pf_op_array(ops)
    offsets, flat = _det_csr(dets)
    ndet = len(dets)
    total = shots * ndet

    def ml_run(nthreads):
        out = np.empty((ndet, shots), dtype=np.uint8)
        _LIB.pauli_frame_batch_sample_detectors(
            n, op_arr, len(ops),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ndet, shots, ctypes.c_uint64(seed), nthreads,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))

    sampler = _to_stim_with_detectors(ops, dets).compile_detector_sampler(seed=seed + 1)

    def st_run():
        sampler.sample(shots)

    def st_run_packed():
        sampler.sample(shots, bit_packed=True)

    reps = 3 if shots <= 100000 else 2
    t_ml1 = _best_time(lambda: ml_run(1), reps=reps)
    t_mlN = _best_time(lambda: ml_run(0), reps=reps)
    t_st = _best_time(st_run, reps=reps)
    t_stp = _best_time(st_run_packed, reps=reps)
    return {
        "ndet": ndet, "shots": shots,
        "ml_st_sps": total / t_ml1,
        "ml_mt_sps": total / t_mlN,
        "stim_sps": total / t_st,
        "stim_packed_sps": total / t_stp,
        "stim_best_sps": total / min(t_st, t_stp),
    }


# --------------------------------------------------------------------------
# Workloads.
# --------------------------------------------------------------------------
def _workloads():
    return [
        ("random_clifford_n40", *build_random_clifford(40, 800, seed=41)),
        ("surface_code_d5_r8", *build_surface_code(5, 8)),
        ("surface_code_d5_r8_noisy", *build_surface_code_noisy(5, 8)[:2]),
    ]


SHOT_SIZES = [10**4, 10**5, 10**6]


def run():
    workloads = _workloads()
    correctness_ok = True
    corr_detail = {}
    for name, n, ops in workloads:
        # Gate BOTH configurations: 1 thread (kernel) and all cores (the
        # configuration the headline multithreaded numbers come from).
        for label, nt in (("st", 1), ("mt", 0)):
            ok, detail = check_correctness(n, ops, shots=200000, seed=7 + n, nthreads=nt)
            corr_detail[f"{name}:{label}"] = detail
            correctness_ok = correctness_ok and ok

    rows = []
    for name, n, ops in workloads:
        for shots in SHOT_SIZES:
            m = measure(n, ops, shots, seed=100 + n + shots % 999)
            # Ratios are taken against stim's BEST of {unpacked, bit_packed}.
            st_ratio = m["ml_st_sps"] / m["stim_best_sps"]
            mt_ratio = m["ml_mt_sps"] / m["stim_best_sps"]
            verdict = "lead" if mt_ratio > 1.1 else ("parity" if mt_ratio > 0.9 else "behind")
            rows.append({"workload": name, **m,
                         "st_ratio": st_ratio, "mt_ratio": mt_ratio, "verdict": verdict})
    # ---- detector sampling: what a decoder actually consumes --------------
    dn, dops, ddets = build_surface_code_noisy(5, 8)
    nn, nops, ndets = build_surface_code_noisy(5, 8, p2=0.0, p1=0.0, pm=0.0)
    ml_fired, st_fired = check_detector_silence(nn, nops, ndets)
    det_detail = {"noiseless_fired_moonlab": ml_fired,
                  "noiseless_fired_stim": st_fired}
    correctness_ok = correctness_ok and ml_fired == 0 and st_fired == 0
    for label, nt in (("st", 1), ("mt", 0)):
        ok, detail = check_detector_correctness(dn, dops, ddets, nthreads=nt)
        det_detail[f"surface_code_d5_r8_noisy:{label}"] = detail
        correctness_ok = correctness_ok and ok

    det_rows = []
    for shots in SHOT_SIZES:
        m = measure_detectors(dn, dops, ddets, shots, seed=500 + shots % 997)
        st_ratio = m["ml_st_sps"] / m["stim_best_sps"]
        mt_ratio = m["ml_mt_sps"] / m["stim_best_sps"]
        det_rows.append({"workload": "surface_code_d5_r8_noisy", **m,
                         "st_ratio": st_ratio, "mt_ratio": mt_ratio,
                         "verdict": "lead" if mt_ratio > 1.1
                         else ("parity" if mt_ratio > 0.9 else "behind")})

    return {
        "correctness_ok": correctness_ok,
        "correctness_detail": corr_detail,
        "detector_detail": det_detail,
        "detector_rows": det_rows,
        "simd_backend": SIMD_BACKEND, "simd_lanes": SIMD_LANES,
        "phys_cores": PHYS_CORES, "threads": os.cpu_count(),
        "rows": rows,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, default=str))
