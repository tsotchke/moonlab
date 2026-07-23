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

_STIM_NAME = {PF_H: "H", PF_S: "S", PF_S_DAG: "S_DAG", PF_X: "X", PF_Y: "Y",
              PF_Z: "Z", PF_CNOT: "CNOT", PF_CZ: "CZ", PF_SWAP: "SWAP"}


class _PfOp(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_uint8), ("q0", ctypes.c_uint32), ("q1", ctypes.c_uint32)]


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
    """A surface-code-flavoured syndrome-extraction circuit: a d x d grid of
    data qubits with a weight-4 Z- and X-plaquette on every unit cell, each
    read by its own ancilla via local CNOTs and measure+reset, repeated for
    `rounds` rounds. Local two-qubit gates and ancilla MR only."""
    def data(i, j):
        return i * d + j
    ndata = d * d
    cells = [(i, j) for i in range(d - 1) for j in range(d - 1)]
    z_anc = {c: ndata + k for k, c in enumerate(cells)}
    x_anc = {c: ndata + len(cells) + k for k, c in enumerate(cells)}
    n = ndata + 2 * len(cells)
    ops = []
    for _ in range(rounds):
        # Z-plaquettes: parity of 4 data qubits into ancilla.
        for (i, j) in cells:
            a = z_anc[(i, j)]
            for (di, dj) in ((0, 0), (1, 0), (0, 1), (1, 1)):
                ops.append((PF_CNOT, data(i + di, j + dj), a))
            ops.append((PF_MEASURE, a, 0))
            ops.append((PF_RESET, a, 0))
        # X-plaquettes: ancilla in X basis, CNOT ancilla->data, back to Z.
        for (i, j) in cells:
            a = x_anc[(i, j)]
            ops.append((PF_H, a, 0))
            for (di, dj) in ((0, 0), (1, 0), (0, 1), (1, 1)):
                ops.append((PF_CNOT, a, data(i + di, j + dj)))
            ops.append((PF_H, a, 0))
            ops.append((PF_MEASURE, a, 0))
            ops.append((PF_RESET, a, 0))
    return n, ops


# --------------------------------------------------------------------------
# Engine drivers.
# --------------------------------------------------------------------------
def _to_stim(ops):
    c = stim.Circuit()
    for op in ops:
        k = op[0]
        if k == PF_MEASURE:
            c.append("M", [op[1]])
        elif k == PF_RESET:
            c.append("R", [op[1]])
        elif k in (PF_CNOT, PF_CZ, PF_SWAP):
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
# Correctness gate.
# --------------------------------------------------------------------------
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
    maxcorr = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            a, b = ml[i], ml[j]
            c, dd = st[i], st[j]
            cml = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else 0.0
            cst = np.corrcoef(c, dd)[0, 1] if c.std() > 0 and dd.std() > 0 else 0.0
            maxcorr = max(maxcorr, abs(cml - cst))
    ok = marg_dev < 6.0 and par_dev < 6.0 and maxcorr < 0.02
    return ok, {"marg_sigma": round(marg_dev, 2), "parity_sigma": round(float(par_dev), 2),
                "max_corr_diff": round(maxcorr, 4), "nmeas": nmeas, "shots": shots,
                "nthreads": nthreads}


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


# --------------------------------------------------------------------------
# Workloads.
# --------------------------------------------------------------------------
def _workloads():
    return [
        ("random_clifford_n40", *build_random_clifford(40, 800, seed=41)),
        ("surface_code_d5_r8", *build_surface_code(5, 8)),
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
    return {
        "correctness_ok": correctness_ok,
        "correctness_detail": corr_detail,
        "simd_backend": SIMD_BACKEND, "simd_lanes": SIMD_LANES,
        "phys_cores": PHYS_CORES, "threads": os.cpu_count(),
        "rows": rows,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, default=str))
