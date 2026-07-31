"""Differential tests of moonlab's `.stim` reader against Stim itself.

Three levels of agreement are checked on the QEC community's own
circuits (``stim.Circuit.generated`` surface and repetition codes at
distances 3, 5 and 7):

1. STRUCTURE.  Our measurement / detector / observable counts and our
   detector coordinates match Stim's, and ``parse -> to_text -> parse``
   is semantically a fixed point (identical lowered op list, channel
   arguments, detector and observable CSRs, coordinates) in both the
   REPEAT-preserving and the flattened spelling.

2. DETERMINISM.  Noiseless circuits give all-zero detectors from both
   samplers.  Circuits whose noise is forced to probability 1.0 on a
   chosen subset are fully deterministic, and our detector and
   observable output is then bit-identical to Stim's -- including for
   PAULI_CHANNEL_1 and PAULI_CHANNEL_2, where a single unit-probability
   argument pins down the channel's argument order.

3. DISTRIBUTION.  With real noise, every detector's firing fraction
   agrees with Stim's inside a 5-sigma two-proportion interval over
   20000 shots per side.
"""

import math

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from moonlab.qec import StimCircuit, StimFormatError  # noqa: E402


NOISE_P = 0.006
STAT_SHOTS = 20000
SIGMA = 5.0

CODES = [
    "surface_code:rotated_memory_z",
    "surface_code:rotated_memory_x",
    "surface_code:unrotated_memory_z",
    "repetition_code:memory",
]
DISTANCES = [3, 5, 7]

#: (code, distance) pairs used by the fast structural tests.
ALL_CASES = [(code, d) for code in CODES for d in DISTANCES]

#: smaller subset for the shot-hungry statistical tests.
STAT_CASES = [
    ("surface_code:rotated_memory_z", 3),
    ("surface_code:rotated_memory_x", 3),
    ("repetition_code:memory", 5),
]

SINGLE_PAULI_NOISE = {"X_ERROR", "Y_ERROR", "Z_ERROR"}
ALL_NOISE = SINGLE_PAULI_NOISE | {
    "DEPOLARIZE1", "DEPOLARIZE2", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2",
}
MEASUREMENTS = {"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"}


def generated(code: str, d: int, p: float = NOISE_P) -> "stim.Circuit":
    """A noisy code-capacity + circuit-level memory experiment."""
    return stim.Circuit.generated(
        code, distance=d, rounds=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p)


def noiseless(code: str, d: int) -> "stim.Circuit":
    return stim.Circuit.generated(code, distance=d, rounds=d)


def reset_noise_only(code: str, d: int, p: float = NOISE_P) -> "stim.Circuit":
    """Only ``after_reset_flip_probability``, so all noise is X_ERROR."""
    return stim.Circuit.generated(
        code, distance=d, rounds=d, after_reset_flip_probability=p)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def ops_fingerprint(ops):
    """Field-wise bytes of a lowered op list.

    ``pf_circuit_op_t`` has interior padding, and the C struct copies
    carry whatever was in those bytes, so a raw ``tobytes()`` would
    compare uninitialised padding rather than the op list.
    """
    return tuple(ops[field].tobytes() for field in ("kind", "q0", "q1", "p"))


def semantic_fingerprint(circuit: StimCircuit):
    """Everything a lowered circuit is: ops, channels, CSRs, coordinates."""
    ops, chan = circuit.lower()
    det_off, det_idx = circuit.detector_csr()
    obs_off, obs_idx = circuit.observable_csr()
    return {
        "counts": (circuit.num_qubits, circuit.num_measurements,
                   circuit.num_detectors, circuit.num_observables,
                   circuit.num_ticks),
        "ops": ops_fingerprint(ops),
        "chan": chan.tobytes(),
        "det_csr": (det_off.tobytes(), det_idx.tobytes()),
        "obs_csr": (obs_off.tobytes(), obs_idx.tobytes()),
        "inversions": circuit.measurement_inversions().tobytes(),
        "qubit_coords": tuple(
            tuple(circuit.qubit_coords(q)) for q in range(circuit.num_qubits)),
        "detector_coords": tuple(
            tuple(circuit.detector_coords(d))
            for d in range(circuit.num_detectors)),
    }


def assert_same_semantics(a: StimCircuit, b: StimCircuit, label: str) -> None:
    fa, fb = semantic_fingerprint(a), semantic_fingerprint(b)
    for key in fa:
        assert fa[key] == fb[key], f"{label}: {key} differs after round trip"


def force_faults(circuit: "stim.Circuit", keep: set) -> "stim.Circuit":
    """Drop every noise instruction except ``keep``, forced to p = 1.

    The result has no randomness left in it, so both samplers must agree
    bit for bit.  ``keep`` indexes single-Pauli noise instructions in
    flattened circuit order (DEPOLARIZE at p=1 is still random over
    three Paulis, so it is never a candidate).
    """
    out = stim.Circuit()
    seen = 0
    for inst in circuit.flattened():
        name = inst.name
        if name in ALL_NOISE:
            if name in SINGLE_PAULI_NOISE:
                if seen in keep:
                    out.append(name, inst.targets_copy(), 1.0)
                seen += 1
            continue
        if name in MEASUREMENTS and inst.gate_args_copy():
            out.append(name, inst.targets_copy(), [])
            continue
        out.append(inst)
    return out


def count_single_pauli_noise(circuit: "stim.Circuit") -> int:
    return sum(1 for inst in circuit.flattened()
               if inst.name in SINGLE_PAULI_NOISE)


def inject_faults(circuit: "stim.Circuit", faults) -> "stim.Circuit":
    """Splice unit-probability Pauli faults into a noiseless circuit.

    ``faults`` maps a flattened instruction index to a list of
    ``(name, qubit)`` pairs inserted just before it.  One qubit at a
    time, which is how a real fault-injection study probes a code.
    """
    out = stim.Circuit()
    for i, inst in enumerate(circuit.flattened()):
        for name, qubit in faults.get(i, ()):
            out.append(name, [qubit], 1.0)
        out.append(inst)
    return out


def stim_detectors(circuit: "stim.Circuit", shots: int, seed: int):
    """Stim's detector sampler, transposed to moonlab's detector-major."""
    det, obs = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)
    return (np.ascontiguousarray(det.astype(np.uint8).T),
            np.ascontiguousarray(obs.astype(np.uint8).T))


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> float:
    """Pooled two-proportion z; 0 when both sides are degenerate."""
    pooled = (k1 + k2) / (n1 + n2)
    if pooled <= 0.0 or pooled >= 1.0:
        return 0.0 if k1 / n1 == k2 / n2 else math.inf
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    return (k1 / n1 - k2 / n2) / se


# ---------------------------------------------------------------------------
# 1. Structure.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code,d", ALL_CASES)
def test_counts_match_stim(code, d):
    reference = generated(code, d)
    ours = StimCircuit.from_text(str(reference))
    assert ours.num_measurements == reference.num_measurements
    assert ours.num_detectors == reference.num_detectors
    assert ours.num_observables == reference.num_observables
    assert ours.num_qubits == reference.num_qubits
    assert ours.num_ticks == reference.num_ticks


@pytest.mark.parametrize("code,d", ALL_CASES)
def test_detector_coordinates_match_stim(code, d):
    """SHIFT_COORDS accumulation across REPEAT iterations, checked
    against the coordinates stim itself resolves."""
    reference = generated(code, d)
    ours = StimCircuit.from_text(str(reference))
    expected = reference.get_detector_coordinates()
    for index, coords in expected.items():
        got = ours.detector_coords(index)
        assert list(got) == pytest.approx(list(coords)), (
            f"{code} d={d}: detector {index} coords {list(got)} != {coords}")


@pytest.mark.parametrize("code,d", ALL_CASES)
def test_qubit_coordinates_match_stim(code, d):
    reference = generated(code, d)
    ours = StimCircuit.from_text(str(reference))
    for qubit, coords in reference.get_final_qubit_coordinates().items():
        got = ours.qubit_coords(qubit)
        assert list(got) == pytest.approx(list(coords)), (
            f"{code} d={d}: qubit {qubit} coords {list(got)} != {coords}")


@pytest.mark.parametrize("code,d", ALL_CASES)
def test_round_trip_preserves_semantics(code, d):
    """parse -> to_text -> parse is a fixed point, blocks or flattened."""
    text = str(generated(code, d))
    original = StimCircuit.from_text(text)

    blocked = StimCircuit.from_text(original.to_text(flatten=False))
    assert_same_semantics(original, blocked, f"{code} d={d} (REPEAT kept)")

    flattened = StimCircuit.from_text(original.to_text(flatten=True))
    assert_same_semantics(original, flattened, f"{code} d={d} (flattened)")

    # A second serialisation of the block form must be byte-stable.
    assert blocked.to_text(flatten=False) == original.to_text(flatten=False)


@pytest.mark.parametrize("code,d", ALL_CASES)
def test_flattened_text_reparses_to_the_same_circuit_as_stim_flattening(code, d):
    """Our REPEAT expansion agrees with stim's own ``flattened()``."""
    reference = generated(code, d)
    ours = StimCircuit.from_text(str(reference))
    via_stim = StimCircuit.from_text(str(reference.flattened()))
    assert_same_semantics(ours, via_stim, f"{code} d={d} vs stim.flattened()")


def test_detector_csr_matches_stim_record_targets():
    """Detector parity sets resolve rec[-k] the way stim does.

    Checked on a flattened circuit, where stim's own instruction stream
    gives the absolute measurement indices to compare against.
    """
    reference = generated("repetition_code:memory", 3).flattened()
    ours = StimCircuit.from_text(str(reference))
    offsets, indices = ours.detector_csr()

    expected = []
    measured = 0
    for inst in reference:
        name = inst.name
        if name in MEASUREMENTS:
            measured += sum(1 for t in inst.targets_copy() if t.is_qubit_target)
        elif name == "MPAD":
            measured += len(inst.targets_copy())
        elif name == "DETECTOR":
            expected.append(sorted(
                measured + t.value for t in inst.targets_copy()))

    assert len(expected) == ours.num_detectors
    for d, want in enumerate(expected):
        got = sorted(int(x) for x in indices[offsets[d]:offsets[d + 1]])
        assert got == want, f"detector {d}: {got} != {want}"


# ---------------------------------------------------------------------------
# 2. Determinism.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code,d", ALL_CASES)
def test_noiseless_detectors_are_all_zero_on_both_sides(code, d):
    reference = noiseless(code, d)
    ours = StimCircuit.from_text(str(reference))

    det, obs = ours.sample_detectors(64, seed=12345)
    assert det.shape == (reference.num_detectors, 64)
    assert not det.any(), f"{code} d={d}: moonlab fired a detector with no noise"
    assert not obs.any(), f"{code} d={d}: moonlab flipped an observable"

    stim_det, stim_obs = stim_detectors(reference, 64, seed=12345)
    assert not stim_det.any()
    assert not stim_obs.any()


@pytest.mark.parametrize("code,d", ALL_CASES)
def test_forced_unit_probability_faults_are_bit_identical(code, d):
    """Noise forced to p=1 on a subset: no randomness, so exact equality."""
    noisy = reset_noise_only(code, d)
    total = count_single_pauli_noise(noisy)
    assert total > 0, f"{code} d={d}: no single-Pauli noise to force"

    for stride, offset in ((3, 0), (5, 2), (total, 1)):
        keep = set(range(offset, total, stride))
        forced = force_faults(noisy, keep)
        ours = StimCircuit.from_text(str(forced))

        det, obs = ours.sample_detectors(16, seed=1)
        stim_det, stim_obs = stim_detectors(forced, 16, seed=1)

        assert det.shape == stim_det.shape
        np.testing.assert_array_equal(
            det, stim_det,
            err_msg=f"{code} d={d} keep={sorted(keep)[:8]}...: detectors differ")
        np.testing.assert_array_equal(
            obs, stim_obs,
            err_msg=f"{code} d={d}: observables differ")
        # Deterministic really means deterministic: every shot identical.
        assert (det == det[:, :1]).all()


@pytest.mark.parametrize("code,d", ALL_CASES)
def test_injected_single_qubit_faults_are_bit_identical(code, d):
    """Surgical p=1 faults on individual qubits, mid-circuit.

    Complements the instruction-level forcing above: these land on one
    qubit at one moment, which is what a fault-injection study probes a
    code with, and every Pauli basis is exercised.
    """
    base = noiseless(code, d)
    flat = base.flattened()
    n_inst = len(flat)
    n_qubits = base.num_qubits
    rng = np.random.default_rng(0x5EED + d)

    for trial in range(3):
        faults = {}
        for _ in range(4):
            where = int(rng.integers(1, n_inst))
            pauli = ["X_ERROR", "Y_ERROR", "Z_ERROR"][int(rng.integers(3))]
            qubit = int(rng.integers(n_qubits))
            faults.setdefault(where, []).append((pauli, qubit))

        injected = inject_faults(base, faults)
        ours = StimCircuit.from_text(str(injected))
        assert ours.num_detectors == injected.num_detectors

        det, obs = ours.sample_detectors(8, seed=trial + 1)
        stim_det, stim_obs = stim_detectors(injected, 8, seed=trial + 1)
        np.testing.assert_array_equal(
            det, stim_det,
            err_msg=f"{code} d={d} trial {trial}: detectors differ "
                    f"for faults {faults}")
        np.testing.assert_array_equal(
            obs, stim_obs,
            err_msg=f"{code} d={d} trial {trial}: observables differ")


PAULI_CHANNEL_2_ORDER = [
    "IX", "IY", "IZ",
    "XI", "XX", "XY", "XZ",
    "YI", "YX", "YY", "YZ",
    "ZI", "ZX", "ZY", "ZZ",
]


@pytest.mark.parametrize("index,label", list(enumerate(["X", "Y", "Z"])))
def test_pauli_channel_1_argument_order_matches_stim(index, label):
    """PAULI_CHANNEL_1(px, py, pz) with one unit argument is deterministic."""
    args = [0.0, 0.0, 0.0]
    args[index] = 1.0
    arg_text = ", ".join(f"{a:g}" for a in args)
    text = (
        "R 0 1\n"
        "H 0\n"
        "CX 0 1\n"
        f"PAULI_CHANNEL_1({arg_text}) 0\n"
        "CX 0 1\n"
        "H 0\n"
        "M 0 1\n"
        "DETECTOR rec[-1]\n"
        "DETECTOR rec[-2]\n"
        "OBSERVABLE_INCLUDE(0) rec[-2]\n"
    )
    reference = stim.Circuit(text)
    ours = StimCircuit.from_text(text)

    det, obs = ours.sample_detectors(8, seed=99)
    stim_det, stim_obs = stim_detectors(reference, 8, seed=99)
    np.testing.assert_array_equal(
        det, stim_det, err_msg=f"PAULI_CHANNEL_1 {label} detectors differ")
    np.testing.assert_array_equal(
        obs, stim_obs, err_msg=f"PAULI_CHANNEL_1 {label} observables differ")


@pytest.mark.parametrize("index,label",
                         list(enumerate(PAULI_CHANNEL_2_ORDER)))
def test_pauli_channel_2_argument_order_matches_stim(index, label):
    """Each of the 15 PAULI_CHANNEL_2 slots, driven to 1.0 in turn.

    The first letter acts on the first target, the second on the second;
    driving one argument to unity makes the circuit deterministic, so a
    mismatched argument order shows up as a bit difference against stim.
    """
    args = [0.0] * 15
    args[index] = 1.0
    arg_text = ", ".join(f"{a:g}" for a in args)
    text = (
        "R 0 1 2 3\n"
        "H 0 1\n"
        "CX 0 2 1 3\n"
        f"PAULI_CHANNEL_2({arg_text}) 0 1\n"
        "CX 0 2 1 3\n"
        "H 0 1\n"
        "M 0 1 2 3\n"
        "DETECTOR rec[-1]\n"
        "DETECTOR rec[-2]\n"
        "DETECTOR rec[-3]\n"
        "DETECTOR rec[-4]\n"
        "OBSERVABLE_INCLUDE(0) rec[-4]\n"
    )
    reference = stim.Circuit(text)
    ours = StimCircuit.from_text(text)

    det, obs = ours.sample_detectors(8, seed=7)
    stim_det, stim_obs = stim_detectors(reference, 8, seed=7)
    np.testing.assert_array_equal(
        det, stim_det,
        err_msg=f"PAULI_CHANNEL_2 arg {index} ({label}) detectors differ")
    np.testing.assert_array_equal(
        obs, stim_obs,
        err_msg=f"PAULI_CHANNEL_2 arg {index} ({label}) observables differ")


def test_pauli_channel_1_unit_x_equals_x_error_one():
    """PAULI_CHANNEL_1(1, 0, 0) is X_ERROR(1), through the sampler."""
    body = ("R 0\nH 0\n{noise}\nH 0\nM 0\n"
            "DETECTOR rec[-1]\n")
    a = StimCircuit.from_text(body.format(noise="PAULI_CHANNEL_1(1, 0, 0) 0"))
    b = StimCircuit.from_text(body.format(noise="X_ERROR(1) 0"))
    np.testing.assert_array_equal(a.sample_detectors(8, seed=3)[0],
                                  b.sample_detectors(8, seed=3)[0])


def test_measurement_inversion_only_moves_raw_records():
    """`!` flips reported records; detectors are unaffected by design."""
    text = ("R 0 1\nX 0\nM !0 1\n"
            "DETECTOR rec[-2]\nDETECTOR rec[-1]\n")
    ours = StimCircuit.from_text(text)
    reference = stim.Circuit(text)

    inversions = ours.measurement_inversions()
    assert list(inversions) == [1, 0]

    meas = ours.sample_measurements(8, seed=5)
    stim_meas = np.ascontiguousarray(
        reference.compile_sampler(seed=5).sample(8).astype(np.uint8).T)
    np.testing.assert_array_equal(meas, stim_meas)

    det, _ = ours.sample_detectors(8, seed=5)
    stim_det, _ = stim_detectors(reference, 8, seed=5)
    np.testing.assert_array_equal(det, stim_det)


# ---------------------------------------------------------------------------
# 3. Distribution.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code,d", STAT_CASES)
def test_noisy_detector_firing_fractions_agree_with_stim(code, d):
    """Per-detector firing fraction inside a 5-sigma binomial interval."""
    reference = generated(code, d)
    ours = StimCircuit.from_text(str(reference))

    det, obs = ours.sample_detectors(STAT_SHOTS, seed=0xA5A5)
    stim_det, stim_obs = stim_detectors(reference, STAT_SHOTS, seed=0xA5A5)

    ours_hits = det.sum(axis=1, dtype=np.int64)
    stim_hits = stim_det.sum(axis=1, dtype=np.int64)
    assert ours_hits.shape == stim_hits.shape

    worst = 0.0
    worst_at = -1
    failures = []
    for i in range(len(ours_hits)):
        z = two_proportion_z(int(ours_hits[i]), STAT_SHOTS,
                             int(stim_hits[i]), STAT_SHOTS)
        if abs(z) > abs(worst):
            worst, worst_at = z, i
        if abs(z) > SIGMA:
            failures.append(
                f"detector {i}: moonlab {int(ours_hits[i])}/{STAT_SHOTS}, "
                f"stim {int(stim_hits[i])}/{STAT_SHOTS}, z={z:.2f}")

    print(f"[{code} d={d}] detectors={len(ours_hits)} shots={STAT_SHOTS} "
          f"max|z|={abs(worst):.2f} at detector {worst_at}; "
          f"mean firing moonlab={ours_hits.mean() / STAT_SHOTS:.5f} "
          f"stim={stim_hits.mean() / STAT_SHOTS:.5f}")

    assert not failures, (
        f"{code} d={d}: {len(failures)} detectors outside {SIGMA} sigma:\n  "
        + "\n  ".join(failures[:10]))

    # Observables too: same noise model has to give the same logical rate.
    for i in range(obs.shape[0]):
        z = two_proportion_z(int(obs[i].sum()), STAT_SHOTS,
                             int(stim_obs[i].sum()), STAT_SHOTS)
        assert abs(z) <= SIGMA, (
            f"{code} d={d}: observable {i} moonlab {int(obs[i].sum())} vs "
            f"stim {int(stim_obs[i].sum())} of {STAT_SHOTS}, z={z:.2f}")


@pytest.mark.parametrize("code,d", STAT_CASES)
def test_measurement_marginals_agree_with_stim(code, d):
    """Raw measurement records, not just their detector parities."""
    reference = generated(code, d)
    ours = StimCircuit.from_text(str(reference))

    shots = STAT_SHOTS
    meas = ours.sample_measurements(shots, seed=0x1234)
    stim_meas = np.ascontiguousarray(
        reference.compile_sampler(seed=0x1234).sample(shots).astype(np.uint8).T)
    assert meas.shape == stim_meas.shape == (reference.num_measurements, shots)

    ours_hits = meas.sum(axis=1, dtype=np.int64)
    stim_hits = stim_meas.sum(axis=1, dtype=np.int64)
    worst = 0.0
    for i in range(len(ours_hits)):
        z = two_proportion_z(int(ours_hits[i]), shots,
                             int(stim_hits[i]), shots)
        worst = z if abs(z) > abs(worst) else worst
        assert abs(z) <= SIGMA, (
            f"{code} d={d}: measurement {i} moonlab {int(ours_hits[i])} vs "
            f"stim {int(stim_hits[i])} of {shots}, z={z:.2f}")
    print(f"[{code} d={d}] measurements={len(ours_hits)} shots={shots} "
          f"max|z|={abs(worst):.2f}")


# ---------------------------------------------------------------------------
# Rejections: unsupported constructs must fail loudly, never be skipped.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,token", [
    ("MPP X0*Z1\n", "MPP"),
    ("SPP X0\n", "SPP"),
    ("SPP_DAG X0\n", "SPP_DAG"),
    ("E(0.01) X0\n", "E"),
    ("CORRELATED_ERROR(0.01) X0\n", "CORRELATED_ERROR"),
    ("HERALDED_ERASE(0.01) 0\n", "HERALDED_ERASE"),
    ("HERALDED_PAULI_CHANNEL_1(0.01, 0, 0, 0) 0\n",
     "HERALDED_PAULI_CHANNEL_1"),
    ("II 0 1\n", "II"),
    ("II_ERROR(0.01) 0 1\n", "II_ERROR"),
    ("NOT_A_GATE 0\n", "NOT_A_GATE"),
    ("CX sweep[0] 1\n", "sweep"),
    ("H 0\nM !0 1\nDETECTOR rec[-1]\nX !0\n", "!"),
])
def test_unsupported_constructs_are_rejected(text, token):
    with pytest.raises(StimFormatError) as excinfo:
        StimCircuit.from_text(text)
    err = excinfo.value
    assert err.line > 0, f"{token}: rejection carried no line number"
    assert token.lower() in err.message.lower(), (
        f"{token}: message {err.message!r} does not name the offending token")


def test_parse_error_reports_the_failing_line():
    text = "H 0\nS 1\nCX 0\n"
    with pytest.raises(StimFormatError) as excinfo:
        StimCircuit.from_text(text)
    assert excinfo.value.line == 3
