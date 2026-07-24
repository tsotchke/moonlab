"""F3 decoder front: MoonLab union-find + exact cluster matching vs PyMatching 2.

PyMatching 2 (sparse blossom) computes exact minimum-weight perfect matching,
which is optimal FOR THE MATCHING DECODER CLASS -- but MWPM is not the
maximum-likelihood decoder.  MoonLab scores pairings by ``-ln P`` within the
pairing approximation: path weight minus the log of the number of
minimum-weight paths realising it (degeneracy), which MWPM cannot see.  The
achievable outcomes are therefore: tie MWPM where degeneracy is irrelevant,
beat it where degeneracy discriminates.

Gates:

  1. Accuracy (correctness_ok): on identical shots, a paired McNemar test per
     workload.  The front FAILS if MoonLab is significantly WORSE (z > 1.96
     with PyMatching ahead) anywhere.  ``better``/``tied`` both pass.
  2. Throughput: decode-only wall clock on the same detector data,
     PyMatching ``decode_batch`` (single-threaded) vs MoonLab 1-thread and
     all-cores.

Both decoders consume the SAME merged edge model (parallel mechanisms
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
_LIB.moonlab_uf_decoder_free.argtypes = [ctypes.c_void_p]
_LIB.moonlab_uf_decode_batch.restype = ctypes.c_long
_LIB.moonlab_uf_decode_batch.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ctypes.c_int, ctypes.POINTER(ctypes.c_uint8)]


def dem_to_edges(dem):
    """Flatten a DEM to graph edges.

    ``decompose_errors=True`` splits non-graphlike mechanisms into graphlike
    components separated by ``^``; each component is an edge.  Parallel
    mechanisms on the same (pair, observables) are merged by combining
    probabilities -- p = p1(1-p2) + p2(1-p1) -- which is what PyMatching does
    internally, so both decoders see one model.  Residual hyperedges are
    counted, not silently dropped.
    """
    raw = []
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
        for dets, obsm in components:
            if not dets:
                continue
            if len(dets) > 2:
                hyper += 1
                continue
            a = dets[0]
            b = dets[1] if len(dets) == 2 else BOUNDARY
            raw.append((a, b, p, obsm))
    acc = {}
    for a, b, p, o in raw:
        key = ((a, b) if a <= b else (b, a)) + (o,)
        q = acc.get(key, 0.0)
        acc[key] = p * (1 - q) + q * (1 - p)
    keys = list(acc)
    ea = np.array([k[0] for k in keys], dtype=np.uint32)
    eb = np.array([k[1] for k in keys], dtype=np.uint32)
    ew = np.array([math.log((1 - acc[k]) / acc[k]) for k in keys], dtype=np.float64)
    eo = np.array([k[2] for k in keys], dtype=np.uint64)
    return ea, eb, ew, eo, hyper


def _decoder(ndet, nobs, ea, eb, ew, eo):
    h = _LIB.moonlab_uf_decoder_new(
        ndet, nobs,
        ea.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        eb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ew.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        eo.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        len(ea))
    if not h:
        raise RuntimeError("moonlab_uf_decoder_new failed")
    return h


def run(shots=1000000):
    workloads = [(5, 0.002), (7, 0.002), (5, 0.005), (7, 0.003)]
    rows = []
    ok = True
    for d, p in workloads:
        c = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=d, rounds=d,
            after_clifford_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p)
        dem = c.detector_error_model(decompose_errors=True)
        ea, eb, ew, eo, hyper = dem_to_edges(dem)
        ndet, nobs = c.num_detectors, c.num_observables
        det, obs = c.compile_detector_sampler(seed=97 + d).sample(
            shots, separate_observables=True)
        truth = obs[:, 0]

        pm = pymatching.Matching.from_detector_error_model(dem)
        t0 = time.perf_counter()
        pm_pred = pm.decode_batch(det)[:, 0]
        pm_t = time.perf_counter() - t0

        det_u8 = np.ascontiguousarray(det.astype(np.uint8).T)
        h = _decoder(ndet, nobs, ea, eb, ew, eo)
        out = np.empty((nobs, shots), dtype=np.uint8)
        buf = det_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        outp = out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        t0 = time.perf_counter()
        _LIB.moonlab_uf_decode_batch(h, buf, shots, 1, outp)
        ml_t1 = time.perf_counter() - t0
        t0 = time.perf_counter()
        _LIB.moonlab_uf_decode_batch(h, buf, shots, 0, outp)
        ml_tn = time.perf_counter() - t0
        _LIB.moonlab_uf_decoder_free(h)
        ml_pred = out[0]

        pm_bad = pm_pred != truth
        ml_bad = ml_pred != truth
        only_pm = int((pm_bad & ~ml_bad).sum())
        only_ml = int((ml_bad & ~pm_bad).sum())
        n_disc = only_pm + only_ml
        z = (abs(only_ml - only_pm) - 1) / math.sqrt(n_disc) if n_disc else 0.0
        if z < 1.96:
            verdict = "tied"
        elif only_ml < only_pm:
            verdict = "moonlab_better"
        else:
            verdict = "pymatching_better"
            ok = False
        rows.append({
            "distance": d, "p": p, "shots": shots, "ndet": ndet,
            "hyperedges_skipped": hyper,
            "pm_err": int(pm_bad.sum()) / shots,
            "ml_err": int(ml_bad.sum()) / shots,
            "only_pm": only_pm, "only_ml": only_ml,
            "mcnemar_z": round(z, 2), "accuracy": verdict,
            "pm_sps": shots / pm_t,
            "ml_st_sps": shots / ml_t1,
            "ml_mt_sps": shots / ml_tn,
            "mt_ratio": (shots / ml_tn) / (shots / pm_t),
        })
    return {"correctness_ok": ok, "rows": rows}


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
