"""Differential tests of moonlab's detector-error-model bridge.

The DEM is the interchange format the QEC community decodes against, so
both directions have to hold up against the reference implementations:

- IMPORT.  A Stim DEM produced with ``decompose_errors=True`` imports to
  the same edge list and the same mechanism correlation links as the
  reference converter
  ``benchmarks/dominance/fronts/f3_decoder_vs_pymatching.py::dem_to_edges``
  -- the same converter the dominance front's published numbers use.

- EXPORT.  Our DEM text parses in Stim and loads in PyMatching, and
  ``edges -> text -> parse -> edges`` is a fixed point, so a moonlab
  noise model survives a trip through the community's tooling.

- DECODING.  Corrections from moonlab's union-find decoder and from
  PyMatching both satisfy the observed syndrome exactly (zero residual),
  and their logical error rates agree to within a two-proportion z-test.
"""

import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

from moonlab.qec import (  # noqa: E402
    UF_BOUNDARY,
    DetectorErrorModel,
    StimFormatError,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_PATH = (REPO_ROOT / "benchmarks" / "dominance" / "fronts"
                  / "f3_decoder_vs_pymatching.py")

SHOTS = 20000
Z_LIMIT = 4.0

#: (code, distance, rounds, noise) for the decoding comparisons.
DECODE_CASES = [
    ("repetition_code:memory", 5, 5, 0.02),
    ("surface_code:rotated_memory_z", 3, 3, 0.006),
    ("surface_code:rotated_memory_x", 3, 3, 0.006),
]


# ---------------------------------------------------------------------------
# The reference converter, loaded from the benchmark it ships in.
# ---------------------------------------------------------------------------

def _load_reference():
    if not REFERENCE_PATH.exists():
        pytest.skip(f"reference converter missing: {REFERENCE_PATH}")
    if "MOONLAB_LIB_DIR" not in os.environ:
        from moonlab import core as _core
        if getattr(_core, "_lib_path", None) is not None:
            os.environ["MOONLAB_LIB_DIR"] = str(Path(_core._lib_path).parent)
    spec = importlib.util.spec_from_file_location(
        "_moonlab_f3_reference", REFERENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dem_to_edges():
    return _load_reference().dem_to_edges


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def noisy_circuit(code: str, d: int, rounds: int, p: float) -> "stim.Circuit":
    return stim.Circuit.generated(
        code, distance=d, rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p)


def edge_map(model: DetectorErrorModel):
    """{(a, b, obs mask): probability} plus the index of each key."""
    ea, eb, ew, eo, ep = model.edges()
    keys = [(int(a), int(b), int(o)) for a, b, o in zip(ea, eb, eo)]
    assert len(set(keys)) == len(keys), "edge keys are not unique"
    return ({k: float(p) for k, p in zip(keys, ep)},
            keys,
            {k: float(w) for k, w in zip(keys, ew)})


def reference_edge_map(dem_to_edges, dem):
    ea, eb, ew, eo, hyper, ep, ca, cb, cq = dem_to_edges(dem)
    keys = [(int(a), int(b), int(o)) for a, b, o in zip(ea, eb, eo)]
    return {
        "prob": {k: float(p) for k, p in zip(keys, ep)},
        "weight": {k: float(w) for k, w in zip(keys, ew)},
        "keys": keys,
        "hyper": int(hyper),
        "corr": {
            tuple(sorted((keys[int(u)], keys[int(v)]))): float(q)
            for u, v, q in zip(ca, cb, cq)
        },
    }


def correlation_map(model: DetectorErrorModel, keys):
    ca, cb, cq = model.correlations()
    out = {}
    for u, v, q in zip(ca, cb, cq):
        pair = tuple(sorted((keys[int(u)], keys[int(v)])))
        assert pair not in out, f"duplicate correlation link {pair}"
        out[pair] = float(q)
    return out


def incidence_matrix(ea, eb, num_detectors: int) -> np.ndarray:
    """GF(2) detector-by-edge incidence of the matching graph."""
    h = np.zeros((num_detectors, len(ea)), dtype=np.uint8)
    for i, (a, b) in enumerate(zip(ea, eb)):
        h[int(a), i] ^= 1
        if int(b) != UF_BOUNDARY:
            h[int(b), i] ^= 1
    return h


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> float:
    pooled = (k1 + k2) / (n1 + n2)
    if pooled <= 0.0 or pooled >= 1.0:
        return 0.0 if k1 / n1 == k2 / n2 else math.inf
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    return (k1 / n1 - k2 / n2) / se


# ---------------------------------------------------------------------------
# Import: our edge list is the reference edge list.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code,d,rounds,p", DECODE_CASES)
def test_edges_match_reference_converter(dem_to_edges, code, d, rounds, p):
    circuit = noisy_circuit(code, d, rounds, p)
    dem = circuit.detector_error_model(decompose_errors=True)
    ours = DetectorErrorModel.from_text(str(dem))
    reference = reference_edge_map(dem_to_edges, dem)

    probs, keys, weights = edge_map(ours)
    assert set(probs) == set(reference["prob"]), (
        f"{code} d={d}: edge key sets differ "
        f"(ours {len(probs)}, reference {len(reference['prob'])}); "
        f"only ours: {sorted(set(probs) - set(reference['prob']))[:5]}; "
        f"only reference: "
        f"{sorted(set(reference['prob']) - set(probs))[:5]}")

    for key, p_ref in reference["prob"].items():
        assert probs[key] == pytest.approx(p_ref, rel=1e-12, abs=1e-15), (
            f"{code} d={d}: edge {key} p={probs[key]!r} != {p_ref!r}")
        assert weights[key] == pytest.approx(
            math.log((1 - p_ref) / p_ref), rel=1e-12), (
            f"{code} d={d}: edge {key} weight is not ln((1-p)/p)")

    assert ours.num_hyperedges == reference["hyper"]
    assert ours.num_detectors == circuit.num_detectors
    assert ours.num_observables == circuit.num_observables

    ours_corr = correlation_map(ours, keys)
    assert set(ours_corr) == set(reference["corr"]), (
        f"{code} d={d}: correlation link sets differ "
        f"(ours {len(ours_corr)}, reference {len(reference['corr'])})")
    for pair, q_ref in reference["corr"].items():
        assert ours_corr[pair] == pytest.approx(q_ref, rel=1e-12, abs=1e-15), (
            f"{code} d={d}: link {pair} q={ours_corr[pair]!r} != {q_ref!r}")

    print(f"[{code} d={d}] edges={len(probs)} links={len(ours_corr)} "
          f"hyperedges={ours.num_hyperedges} "
          f"detectorless={ours.num_detectorless}")


@pytest.mark.parametrize("code,d,rounds,p", DECODE_CASES)
def test_decomposed_dem_has_no_hyperedges(code, d, rounds, p):
    """`decompose_errors=True` is meant to leave nothing non-graphlike."""
    dem = noisy_circuit(code, d, rounds, p).detector_error_model(
        decompose_errors=True)
    assert DetectorErrorModel.from_text(str(dem)).num_hyperedges == 0


def test_undecomposed_dem_counts_hyperedges_rather_than_dropping_them():
    dem = noisy_circuit("surface_code:rotated_memory_z", 3, 3, 0.01)\
        .detector_error_model(decompose_errors=False)
    ours = DetectorErrorModel.from_text(str(dem))
    expected = 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        if p <= 0 or p >= 1:
            continue
        dets = sum(1 for t in inst.targets_copy()
                   if t.is_relative_detector_id())
        if dets > 2:
            expected += 1
    assert expected > 0, "expected an undecomposed DEM to have hyperedges"
    assert ours.num_hyperedges == expected


# ---------------------------------------------------------------------------
# Export: our text is Stim's and PyMatching's.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code,d,rounds,p", DECODE_CASES)
def test_exported_text_loads_in_stim_and_pymatching(code, d, rounds, p):
    circuit = noisy_circuit(code, d, rounds, p)
    dem = circuit.detector_error_model(decompose_errors=True)
    ours = DetectorErrorModel.from_text(str(dem))

    text = ours.to_text()
    round_tripped = stim.DetectorErrorModel(text)
    assert round_tripped.num_detectors == dem.num_detectors
    assert round_tripped.num_observables == dem.num_observables

    matching = pymatching.Matching.from_detector_error_model(round_tripped)
    assert matching.num_detectors == dem.num_detectors
    assert matching.num_fault_ids == dem.num_observables


@pytest.mark.parametrize("code,d,rounds,p", DECODE_CASES)
def test_edge_list_round_trip_is_a_fixed_point(code, d, rounds, p):
    """edges -> text -> parse -> edges, probabilities to 1e-12."""
    dem = noisy_circuit(code, d, rounds, p).detector_error_model(
        decompose_errors=True)
    ours = DetectorErrorModel.from_text(str(dem))

    ea, eb, ew, eo, ep = ours.edges()
    ca, cb, cq = ours.correlations()
    rebuilt = DetectorErrorModel.from_edges(
        num_detectors=ours.num_detectors,
        num_observables=ours.num_observables,
        edge_a=ea, edge_b=eb, edge_prob=ep, edge_obs=eo,
        corr_a=ca, corr_b=cb, corr_joint_p=cq)

    assert rebuilt.num_detectors == ours.num_detectors
    assert rebuilt.num_observables == ours.num_observables

    before, keys_before, _ = edge_map(ours)
    after, keys_after, _ = edge_map(rebuilt)
    assert set(before) == set(after), (
        f"{code} d={d}: edge set changed across the export round trip")
    for key, value in before.items():
        assert after[key] == pytest.approx(value, rel=1e-12, abs=1e-15), (
            f"{code} d={d}: edge {key} drifted {value!r} -> {after[key]!r}")

    corr_before = correlation_map(ours, keys_before)
    corr_after = correlation_map(rebuilt, keys_after)
    assert set(corr_before) == set(corr_after)
    for pair, value in corr_before.items():
        assert corr_after[pair] == pytest.approx(value, rel=1e-12, abs=1e-15)

    # And the exported text is what stim and pymatching read.
    stim.DetectorErrorModel(rebuilt.to_text())
    pymatching.Matching.from_detector_error_model(
        stim.DetectorErrorModel(rebuilt.to_text()))


# ---------------------------------------------------------------------------
# Decoding.
# ---------------------------------------------------------------------------

def test_residual_syndrome_is_zero_for_both_decoders():
    """Both decoders return a correction whose boundary IS the syndrome.

    Each edge is tagged with its own logical observable, so the decoded
    observable mask names exactly which edges the decoder used.  The
    residual syndrome is then computable directly: H c XOR s must be 0.
    PyMatching is checked on the same graph via its own
    ``decode_to_edges_array``.
    """
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=3, rounds=1,
        after_clifford_depolarization=0.02,
        before_measure_flip_probability=0.02,
        after_reset_flip_probability=0.02)
    dem = circuit.detector_error_model(decompose_errors=True)
    ours = DetectorErrorModel.from_text(str(dem))
    ea, eb, ew, eo, ep = ours.edges()
    n_edges = len(ea)
    n_det = ours.num_detectors
    assert 0 < n_edges <= 64, (
        f"edge tagging needs <= 64 edges, this model has {n_edges}")

    tags = np.array([1 << i for i in range(n_edges)], dtype=np.uint64)
    tagged = DetectorErrorModel.from_edges(
        num_detectors=n_det, num_observables=n_edges,
        edge_a=ea, edge_b=eb, edge_prob=ep, edge_obs=tags)
    assert tagged.num_edges == n_edges
    tag_a, tag_b, _, tag_o, _ = tagged.edges()
    # Recover the tagged model's own edge order for the incidence matrix.
    order = np.array([int(np.log2(int(o))) for o in tag_o])
    assert sorted(order.tolist()) == list(range(n_edges))
    h = incidence_matrix(tag_a, tag_b, n_det)

    shots = 4000
    det, obs = circuit.compile_detector_sampler(seed=31337).sample(
        shots, separate_observables=True)
    det_u8 = np.ascontiguousarray(det.astype(np.uint8).T)

    decoder = tagged.make_decoder(correlated=False)
    used = decoder.decode_batch(det_u8)          # (n_edges, shots)
    assert used.shape == (n_edges, shots)
    residual = (h.astype(np.int64) @ used.astype(np.int64)) % 2
    residual ^= det_u8.astype(np.int64)
    assert not residual.any(), (
        f"moonlab left a residual syndrome on "
        f"{int((residual.any(axis=0)).sum())} of {shots} shots")

    # PyMatching, on the same graph, checked through its own edge output.
    matching = pymatching.Matching.from_detector_error_model(
        stim.DetectorErrorModel(tagged.to_text()))
    pm_residual_bad = 0
    for shot in range(min(shots, 500)):
        syndrome = det_u8[:, shot]
        pairs = matching.decode_to_edges_array(syndrome)
        resid = syndrome.astype(np.int64).copy()
        for u, v in pairs:
            if u >= 0:
                resid[int(u)] ^= 1
            if v >= 0:
                resid[int(v)] ^= 1
        if resid.any():
            pm_residual_bad += 1
    assert pm_residual_bad == 0, (
        f"pymatching left a residual syndrome on {pm_residual_bad} shots")

    print(f"[residual syndrome] edges={n_edges} detectors={n_det} "
          f"shots={shots}: moonlab 0 residual, pymatching 0 residual")


@pytest.mark.parametrize("code,d,rounds,p", DECODE_CASES)
def test_logical_error_rate_agrees_with_pymatching(code, d, rounds, p):
    """Two-proportion z-test on identical shots, both decoder modes."""
    circuit = noisy_circuit(code, d, rounds, p)
    dem = circuit.detector_error_model(decompose_errors=True)
    ours = DetectorErrorModel.from_text(str(dem))

    det, obs = circuit.compile_detector_sampler(seed=0xBEEF + d).sample(
        SHOTS, separate_observables=True)
    truth = obs[:, 0].astype(np.uint8)
    det_u8 = np.ascontiguousarray(det.astype(np.uint8).T)

    matching = pymatching.Matching.from_detector_error_model(dem)
    pm_pred = matching.decode_batch(det)[:, 0].astype(np.uint8)

    plain = ours.make_decoder(correlated=False).decode_batch(det_u8)[0]
    correlated = ours.make_decoder(correlated=True).decode_batch(det_u8)[0]

    pm_err = int((pm_pred != truth).sum())
    plain_err = int((plain != truth).sum())
    corr_err = int((correlated != truth).sum())

    z_plain = two_proportion_z(plain_err, SHOTS, pm_err, SHOTS)
    z_corr = two_proportion_z(corr_err, SHOTS, pm_err, SHOTS)

    print(f"[{code} d={d} p={p}] shots={SHOTS} "
          f"pymatching={pm_err / SHOTS:.5f} "
          f"moonlab_uf={plain_err / SHOTS:.5f} (z={z_plain:+.2f}) "
          f"moonlab_uf_correlated={corr_err / SHOTS:.5f} (z={z_corr:+.2f})")

    assert abs(z_plain) < Z_LIMIT, (
        f"{code} d={d}: plain union-find {plain_err}/{SHOTS} vs pymatching "
        f"{pm_err}/{SHOTS} is {z_plain:.2f} sigma apart")
    assert abs(z_corr) < Z_LIMIT, (
        f"{code} d={d}: correlated union-find {corr_err}/{SHOTS} vs "
        f"pymatching {pm_err}/{SHOTS} is {z_corr:.2f} sigma apart")


def test_decoder_recovers_planted_single_edge_syndromes():
    """Every edge, lit on its own, decodes back to that edge's observables."""
    dem = noisy_circuit("repetition_code:memory", 5, 5, 0.01)\
        .detector_error_model(decompose_errors=True)
    ours = DetectorErrorModel.from_text(str(dem))
    ea, eb, ew, eo, ep = ours.edges()
    n_det = ours.num_detectors

    det = np.zeros((n_det, len(ea)), dtype=np.uint8)
    for i, (a, b) in enumerate(zip(ea, eb)):
        det[int(a), i] ^= 1
        if int(b) != UF_BOUNDARY:
            det[int(b), i] ^= 1

    predicted = ours.make_decoder(correlated=False).decode_batch(det)
    for i in range(len(ea)):
        want = int(eo[i]) & ((1 << ours.num_observables) - 1)
        got = 0
        for bit in range(ours.num_observables):
            got |= int(predicted[bit, i]) << bit
        assert got == want, (
            f"edge {i} ({int(ea[i])}, {int(eb[i])}) decoded to observable "
            f"mask {got:#x}, expected {want:#x}")


def test_make_decoder_matches_manual_edge_construction():
    dem = noisy_circuit("repetition_code:memory", 3, 3, 0.01)\
        .detector_error_model(decompose_errors=True)
    ours = DetectorErrorModel.from_text(str(dem))
    decoder = ours.make_decoder(correlated=True)
    assert decoder.num_edges == ours.num_edges
    assert decoder.num_detectors == ours.num_detectors
    assert decoder.num_observables == ours.num_observables


def test_bit_packed_decoding_matches_dense_decoding():
    """sinter's bit-packed contract agrees with the dense path."""
    circuit = noisy_circuit("surface_code:rotated_memory_z", 3, 3, 0.006)
    dem = circuit.detector_error_model(decompose_errors=True)
    ours = DetectorErrorModel.from_text(str(dem))
    decoder = ours.make_decoder(correlated=True)

    shots = 512
    det, obs = circuit.compile_detector_sampler(seed=4242).sample(
        shots, separate_observables=True, bit_packed=False)
    det_u8 = np.ascontiguousarray(det.astype(np.uint8).T)
    dense = decoder.decode_batch(det_u8)

    packed = np.packbits(det.astype(np.uint8), axis=1, bitorder="little")
    packed_out = decoder.decode_shots_bit_packed(packed)
    unpacked = np.unpackbits(packed_out, axis=1, count=ours.num_observables,
                             bitorder="little")
    np.testing.assert_array_equal(unpacked.T, dense)

    # And against what stim itself packs.
    stim_packed = circuit.compile_detector_sampler(seed=4242).sample(
        shots, separate_observables=True, bit_packed=True)[0]
    np.testing.assert_array_equal(stim_packed, packed)


# ---------------------------------------------------------------------------
# Parsing surface: coordinates, repeats, shifts, and rejections.
# ---------------------------------------------------------------------------

def test_detector_coordinates_match_stim():
    dem = noisy_circuit("surface_code:rotated_memory_z", 3, 3, 0.006)\
        .detector_error_model(decompose_errors=True)
    ours = DetectorErrorModel.from_text(str(dem))
    for index, coords in dem.get_detector_coordinates().items():
        assert list(ours.detector_coords(index)) == pytest.approx(
            list(coords)), f"detector {index} coordinates differ"


def test_repeat_and_shift_detectors_resolve_to_absolute_indices():
    text = (
        "error(0.1) D0 D1\n"
        "repeat 3 {\n"
        "    error(0.2) D0 D1 L0\n"
        "    shift_detectors 2\n"
        "}\n"
        "error(0.3) D0\n"
    )
    ours = DetectorErrorModel.from_text(text)
    reference = DetectorErrorModel.from_text(str(stim.DetectorErrorModel(text)
                                                 .flattened()))
    assert ours.num_detectors == reference.num_detectors
    probs_a, _, _ = edge_map(ours)
    probs_b, _, _ = edge_map(reference)
    assert set(probs_a) == set(probs_b)
    for key in probs_a:
        assert probs_a[key] == pytest.approx(probs_b[key], rel=1e-12)


def test_parallel_mechanisms_merge_by_probability():
    p1, p2 = 0.1, 0.25
    ours = DetectorErrorModel.from_text(
        f"error({p1}) D0 D1\nerror({p2}) D0 D1\n")
    assert ours.num_edges == 1
    _, _, _, _, ep = ours.edges()
    assert float(ep[0]) == pytest.approx(p1 * (1 - p2) + p2 * (1 - p1),
                                         rel=1e-14)


def test_degenerate_probabilities_are_skipped_like_the_reference():
    ours = DetectorErrorModel.from_text(
        "error(0) D0 D1\nerror(1) D1 D2\nerror(0.1) D0 D2\n")
    assert ours.num_edges == 1


def test_detectorless_components_are_counted_separately():
    ours = DetectorErrorModel.from_text(
        "error(0.1) L0\nerror(0.2) D0 D1 L0\n")
    assert ours.num_edges == 1
    assert ours.num_detectorless == 1
    assert ours.num_hyperedges == 0


def test_observable_index_beyond_64_is_rejected():
    with pytest.raises(StimFormatError) as excinfo:
        DetectorErrorModel.from_text("error(0.1) D0 L64\n")
    assert excinfo.value.code == -702, (
        f"expected MOONLAB_STIM_ERR_UNSUPPORTED, got {excinfo.value.code}")
    assert "64" in excinfo.value.message


@pytest.mark.parametrize("text,line", [
    ("error(0.1) D0 D1\nerror(bogus) D2\n", 2),
    ("error(0.1) D0 D1\nnot_an_instruction D2\n", 2),
    ("error(0.1) D0 D1\nerror(0.2) Q3\n", 2),
    ("error(0.1) D0 D1\nerror(0.2) D1 }\n", 2),
    # An unterminated block can reasonably be reported at the opening
    # brace or at end of input; only a non-zero line is required.
    ("repeat 3 {\n    error(0.1) D0\n", None),
])
def test_syntax_errors_carry_a_line_number(text, line):
    with pytest.raises(StimFormatError) as excinfo:
        DetectorErrorModel.from_text(text)
    if line is None:
        assert excinfo.value.line > 0
    else:
        assert excinfo.value.line == line, (
            f"expected line {line}, got {excinfo.value.line}: "
            f"{excinfo.value.message}")
    assert excinfo.value.message


def test_from_edges_rejects_inconsistent_correlations():
    """A link stronger than the edge it touches is a caller error."""
    with pytest.raises(StimFormatError) as excinfo:
        DetectorErrorModel.from_edges(
            num_detectors=2, num_observables=1,
            edge_a=np.array([0, 1], dtype=np.uint32),
            edge_b=np.array([1, UF_BOUNDARY], dtype=np.uint32),
            edge_prob=np.array([0.001, 0.001]),
            edge_obs=np.array([1, 0], dtype=np.uint64),
            corr_a=np.array([0], dtype=np.uint32),
            corr_b=np.array([1], dtype=np.uint32),
            corr_joint_p=np.array([0.4]))
    assert excinfo.value.code == -703, (
        f"expected MOONLAB_STIM_ERR_BAD_ARG, got {excinfo.value.code}")
