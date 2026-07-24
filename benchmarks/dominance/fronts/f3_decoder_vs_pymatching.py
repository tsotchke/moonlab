"""F3 decoder front: MoonLab union-find + exact cluster matching vs PyMatching 2.

PyMatching 2 (sparse blossom) computes exact minimum-weight perfect matching,
which is optimal FOR THE MATCHING DECODER CLASS -- but MWPM is not the
maximum-likelihood decoder.  MoonLab scores pairings by ``-ln P`` within the
pairing approximation: path weight minus the log of the number of
minimum-weight paths realising it (degeneracy), which MWPM cannot see.

CORRELATED TWO-PASS DECODING goes further: ``decompose_errors=True`` emits
mechanisms like ``error(p) D1 D2 ^ D3 D4`` -- one physical fault whose
graphlike components always fire together.  Matching (PyMatching's standard
use AND MoonLab's plain mode) treats the components as independent edges and
discards the correlation.  MoonLab's correlated decoder decodes twice: when
pass 1's correction uses one component, the partners' probabilities are
replaced by their conditionals given the used component and pass 2
re-matches.  Both MoonLab columns are reported; correlated is the headline.

Gates:

  1. Accuracy (correctness_ok): on identical shots, a paired McNemar test per
     workload for BOTH MoonLab modes vs PyMatching.  The front FAILS if the
     correlated decoder is significantly WORSE (z > 1.96 with PyMatching
     ahead) anywhere.  ``corr_better_points`` counts workloads where it is
     significantly better.
  2. Throughput: decode-only wall clock on the same detector data,
     PyMatching ``decode_batch`` (single-threaded) vs MoonLab 1-thread and
     all-cores, plain and correlated.

All decoders consume the SAME merged edge model (parallel mechanisms
combined by probability, the merge PyMatching applies internally), so the
comparison is decoder-vs-decoder, not model-vs-model.
"""
from __future__ import annotations

import ctypes
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pymatching
import stim

BOUNDARY = 0xFFFFFFFF


def _load_lib():
    name = {"darwin": "libquantumsim.dylib",
            "win32": "quantumsim.dll"}.get(sys.platform, "libquantumsim.so")
    dirs = []
    if os.environ.get("MOONLAB_LIB_DIR"):
        dirs.append(Path(os.environ["MOONLAB_LIB_DIR"]))
    repo = Path(__file__).resolve().parents[3]
    dirs += [repo / "build-campaign", repo / "build", repo]
    for dd in dirs:
        p = dd / name
        if p.exists():
            lib = ctypes.CDLL(str(p))
            if hasattr(lib, "moonlab_uf_decode_batch"):
                return lib
    raise OSError("libquantumsim with moonlab_uf_decode_batch not found; "
                  "set MOONLAB_LIB_DIR")


_LIB = _load_lib()
_LIB.moonlab_uf_decoder_new.restype = ctypes.c_void_p
_LIB.moonlab_uf_decoder_new.argtypes = [
    ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_size_t]
_LIB.moonlab_uf_decoder_new_correlated.restype = ctypes.c_void_p
_LIB.moonlab_uf_decoder_new_correlated.argtypes = [
    ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
_LIB.moonlab_uf_decoder_free.argtypes = [ctypes.c_void_p]
_LIB.moonlab_uf_decode_batch.restype = ctypes.c_long
_LIB.moonlab_uf_decode_batch.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ctypes.c_int, ctypes.POINTER(ctypes.c_uint8)]


def dem_to_edges(dem):
    """Flatten a DEM to graph edges plus the mechanism correlation links.

    ``decompose_errors=True`` splits non-graphlike mechanisms into graphlike
    components separated by ``^``; each component is an edge.  Parallel
    mechanisms on the same (pair, observables) are merged by combining
    probabilities -- p = p1(1-p2) + p2(1-p1) -- which is what PyMatching does
    internally, so both decoders see one model.  Residual hyperedges are
    counted, not silently dropped.

    The mechanism grouping survives the merge: each multi-component
    mechanism contributes every pairwise link between the merged edge
    indices of its components, and links repeated across mechanisms (many
    mechanisms share component edges) have their joint probabilities
    XOR-combined the same way parallel edges are.  Returns
    (ea, eb, ew, eo, hyper, ep, corr_a, corr_b, corr_q).
    """
    raw = []
    mechs = []   # (p, [component key, ...]) per multi-component mechanism
    hyper = 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        if p <= 0 or p >= 1:
            continue
        components, cur_d, cur_o = [], [], 0
        for t in inst.targets_copy():
            if t.is_separator():
                components.append((cur_d, cur_o)); cur_d, cur_o = [], 0
            elif t.is_relative_detector_id():
                cur_d.append(t.val)
            elif t.is_logical_observable_id():
                cur_o |= (1 << t.val)
        components.append((cur_d, cur_o))
        mech_keys = []
        for dets, obsm in components:
            if not dets:
                continue
            if len(dets) > 2:
                hyper += 1
                continue
            a = dets[0]
            b = dets[1] if len(dets) == 2 else BOUNDARY
            key = ((a, b) if a <= b else (b, a)) + (obsm,)
            raw.append((key, p))
            mech_keys.append(key)
        if len(mech_keys) >= 2:
            mechs.append((p, mech_keys))
    acc = {}
    for key, p in raw:
        q = acc.get(key, 0.0)
        acc[key] = p * (1 - q) + q * (1 - p)
    keys = list(acc)
    index = {k: i for i, k in enumerate(keys)}
    corr = {}
    for p, mech_keys in mechs:
        idx = sorted({index[k] for k in mech_keys})
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                pair = (idx[i], idx[j])
                q = corr.get(pair, 0.0)
                corr[pair] = p * (1 - q) + q * (1 - p)
    ea = np.array([k[0] for k in keys], dtype=np.uint32)
    eb = np.array([k[1] for k in keys], dtype=np.uint32)
    ew = np.array([math.log((1 - acc[k]) / acc[k]) for k in keys], dtype=np.float64)
    eo = np.array([k[2] for k in keys], dtype=np.uint64)
    ep = np.array([acc[k] for k in keys], dtype=np.float64)
    pairs = sorted(corr)
    corr_a = np.array([u for u, _ in pairs], dtype=np.uint32)
    corr_b = np.array([v for _, v in pairs], dtype=np.uint32)
    corr_q = np.array([corr[pr] for pr in pairs], dtype=np.float64)
    return ea, eb, ew, eo, hyper, ep, corr_a, corr_b, corr_q


def _decoder(ndet, nobs, ea, eb, ew, eo, corr=None):
    if corr is None:
        h = _LIB.moonlab_uf_decoder_new(
            ndet, nobs,
            ea.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            eb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ew.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            eo.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            len(ea))
    else:
        ep, corr_a, corr_b, corr_q = corr
        h = _LIB.moonlab_uf_decoder_new_correlated(
            ndet, nobs,
            ea.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            eb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ew.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            eo.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            len(ea),
            ep.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            corr_a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            corr_b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            corr_q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(corr_a))
    if not h:
        raise RuntimeError("moonlab_uf_decoder_new failed")
    return h


def _ml_decode(h, det_u8, shots, nobs, threads):
    out = np.empty((nobs, shots), dtype=np.uint8)
    buf = det_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    outp = out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    t0 = time.perf_counter()
    rc = _LIB.moonlab_uf_decode_batch(h, buf, shots, threads, outp)
    dt = time.perf_counter() - t0
    if rc != shots:
        raise RuntimeError("moonlab_uf_decode_batch failed")
    return out[0], dt


def _mcnemar(bad_a, bad_b):
    """z for a paired McNemar test; positive favours A (fewer failures)."""
    only_a = int((bad_a & ~bad_b).sum())
    only_b = int((bad_b & ~bad_a).sum())
    n = only_a + only_b
    z = (abs(only_a - only_b) - 1) / math.sqrt(n) if n else 0.0
    return only_a, only_b, z if only_a < only_b else -z if only_a > only_b else 0.0


def run(shots=1000000, workloads=None, seed_offset=0):
    if workloads is None:
        workloads = [(5, 0.002), (7, 0.002), (5, 0.005), (7, 0.003)]
    rows = []
    ok = True
    corr_better = 0
    for d, p in workloads:
        c = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=d, rounds=d,
            after_clifford_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p)
        dem = c.detector_error_model(decompose_errors=True)
        ea, eb, ew, eo, hyper, ep, ca, cb, cq = dem_to_edges(dem)
        ndet, nobs = c.num_detectors, c.num_observables
        det, obs = c.compile_detector_sampler(seed=97 + d + seed_offset).sample(
            shots, separate_observables=True)
        truth = obs[:, 0]

        pm = pymatching.Matching.from_detector_error_model(dem)
        t0 = time.perf_counter()
        pm_pred = pm.decode_batch(det)[:, 0]
        pm_t = time.perf_counter() - t0

        det_u8 = np.ascontiguousarray(det.astype(np.uint8).T)

        h = _decoder(ndet, nobs, ea, eb, ew, eo)
        ml_pred, ml_t1 = _ml_decode(h, det_u8, shots, nobs, 1)
        _, ml_tn = _ml_decode(h, det_u8, shots, nobs, 0)
        _LIB.moonlab_uf_decoder_free(h)

        h = _decoder(ndet, nobs, ea, eb, ew, eo, corr=(ep, ca, cb, cq))
        mlc_pred, mlc_t1 = _ml_decode(h, det_u8, shots, nobs, 1)
        _, mlc_tn = _ml_decode(h, det_u8, shots, nobs, 0)
        _LIB.moonlab_uf_decoder_free(h)

        pm_bad = pm_pred != truth
        ml_bad = ml_pred != truth
        mlc_bad = mlc_pred != truth

        # plain MoonLab vs PyMatching (positive z: MoonLab better)
        only_ml, only_pm, z_u = _mcnemar(ml_bad, pm_bad)
        # correlated MoonLab vs PyMatching -- the gate
        only_mlc, only_pm_c, z_c = _mcnemar(mlc_bad, pm_bad)
        # correlated vs plain MoonLab: the isolated correlation gain
        only_c2, only_u2, z_cu = _mcnemar(mlc_bad, ml_bad)

        if z_c <= -1.96:
            verdict = "pymatching_better"
            ok = False
        elif z_c >= 1.96:
            verdict = "moonlab_corr_better"
            corr_better += 1
        else:
            verdict = "tied"
        pm_err = int(pm_bad.sum())
        rows.append({
            "distance": d, "p": p, "shots": shots, "ndet": ndet,
            "hyperedges_skipped": hyper, "corr_links": int(len(ca)),
            "pm_err": pm_err / shots,
            "ml_err": int(ml_bad.sum()) / shots,
            "mlc_err": int(mlc_bad.sum()) / shots,
            "corr_reduction_vs_pm_pct":
                round(100.0 * (1.0 - int(mlc_bad.sum()) / pm_err), 2)
                if pm_err else 0.0,
            "uncorr_vs_pm": {"only_ml": only_ml, "only_pm": only_pm,
                             "z": round(z_u, 2)},
            "corr_vs_pm": {"only_mlc": only_mlc, "only_pm": only_pm_c,
                           "z": round(z_c, 2)},
            "corr_vs_uncorr": {"only_corr": only_c2, "only_uncorr": only_u2,
                               "z": round(z_cu, 2)},
            "accuracy": verdict,
            "pm_sps": shots / pm_t,
            "ml_st_sps": shots / ml_t1,
            "ml_mt_sps": shots / ml_tn,
            "mlc_st_sps": shots / mlc_t1,
            "mlc_mt_sps": shots / mlc_tn,
            "mt_ratio": (shots / ml_tn) / (shots / pm_t),
            "mlc_mt_ratio": (shots / mlc_tn) / (shots / pm_t),
        })
    return {"correctness_ok": ok, "corr_better_points": corr_better,
            "rows": rows}


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
