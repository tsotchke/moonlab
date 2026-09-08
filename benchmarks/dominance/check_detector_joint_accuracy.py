"""Exact two-detector accuracy audit (not a performance or superiority claim).

The reference is obtained from Stim's *independent* detector error model.  For
each independent error ``e`` with probability p, Walsh moments are multiplied
by ``1-2*p`` when e flips an odd number of the queried detectors.  This gives
the exact four joint cells without Monte Carlo reference noise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import stim


ALPHA_6SIGMA = math.erfc(6 / math.sqrt(2))


def _error_terms(dem):
    """Yield (probability, frozenset(detectors)) for independent DEM errors."""
    for inst in dem.flattened():
        if inst.type == "gauge_detector":
            raise ValueError("gauge detectors are not supported by this exact reference")
        if inst.type != "error":
            continue
        ds = set()
        for target in inst.targets_copy():
            if target.is_relative_detector_id():
                if target.val in ds:
                    ds.remove(target.val)
                else:
                    ds.add(target.val)
        probability = float(inst.args_copy()[0])
        if not 0 <= probability <= 1:
            raise ValueError('invalid independent error probability')
        # Logical-only faults do not alter detector outcomes and are irrelevant.
        if ds:
            yield probability, frozenset(ds)


def joint_reference(dem, first: int, second: int) -> np.ndarray:
    """Exact [p00,p01,p10,p11] for a pair under an independent DEM."""
    if first == second or first < 0 or second < 0:
        raise ValueError("detector pair must contain two distinct non-negative ids")
    m1 = m2 = m12 = 1.0
    for p, ds in _error_terms(dem):
        a, b = first in ds, second in ds
        if a:
            m1 *= 1 - 2 * p
        if b:
            m2 *= 1 - 2 * p
        if a ^ b:
            m12 *= 1 - 2 * p
    # E[(-1)^D1], E[(-1)^D2], E[(-1)^(D1+D2)] Walsh inversion.
    cells = np.array([(1 + m1 + m2 + m12), (1 + m1 - m2 - m12),
                      (1 - m1 + m2 - m12), (1 - m1 - m2 + m12)]) / 4
    if not np.all(np.isfinite(cells)) or np.any(cells < -1e-12) or np.any(cells > 1+1e-12):
        raise ValueError('invalid analytic joint distribution')
    return np.clip(cells, 0.0, 1.0)  # only roundoff at exact 0/1 boundaries


def cells_from_bits(first, second) -> np.ndarray:
    """Raw counts in [00,01,10,11] order; no smoothing or renormalization."""
    return np.bincount(np.asarray(first, dtype=np.uint8) * 2 + np.asarray(second, dtype=np.uint8),
                       minlength=4).astype(np.int64)


def kl_bernoulli(q: float, p: float) -> float:
    """Bernoulli KL, including exact 0/1 boundaries."""
    if not 0 <= q <= 1 or not 0 <= p <= 1:
        raise ValueError('Bernoulli probabilities must be finite and in [0,1]')
    if q == p:
        return 0.0
    if p <= 0:
        return math.inf if q > 0 else 0.0
    if p >= 1:
        return math.inf if q < 1 else 0.0
    out = 0.0
    if q:
        out += q * math.log(q / p)
    if q < 1:
        out += (1 - q) * math.log((1 - q) / (1 - p))
    return out


def chernoff_tail(count: int, shots: int, expected: float) -> float:
    """Two-sided binomial KL-tail Chernoff bound, capped at one.

    Bounds P(N*KL(p_hat||p) >= N*KL(q||p)), not a symmetric absolute-error
    tail. This is a conservative bound, not an exact p-value. It is strict for
    impossible events: one observation against expected probability zero gives
    bound zero (and therefore fails an accuracy audit).
    """
    if shots <= 0 or not 0 <= count <= shots or not 0 <= expected <= 1:
        raise ValueError("count must be in [0, shots]")
    q = count / shots
    if q == expected:
        return 1.0
    k = kl_bernoulli(q, expected)
    return 0.0 if math.isinf(k) else min(1.0, 2 * math.exp(-shots * k))


def pair_record(actual: np.ndarray, expected: np.ndarray, shots: int) -> dict:
    counts = np.asarray(actual)
    if (counts.shape != (4,) or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0) or counts.sum() != shots):
        raise ValueError("actual must contain four raw cell counts summing to shots")
    expected = np.asarray(expected)
    if (expected.shape != (4,) or not np.all(np.isfinite(expected))
            or np.any(expected < 0) or np.any(expected > 1)
            or abs(float(expected.sum())-1) > 1e-12):
        raise ValueError('expected must be a normalized four-cell distribution')
    bounds = np.array([chernoff_tail(int(c), shots, float(p)) for c, p in zip(counts, expected)])
    # The audit threshold is alpha, with no Gaussian replacement for rare cells.
    return {"counts": counts.tolist(), "expected": np.asarray(expected).tolist(),
            "chernoff_bound": bounds.tolist(), "alpha": ALPHA_6SIGMA,
            "passed": bool(np.all(bounds >= ALPHA_6SIGMA))}


def detector_pairs(ndet: int, temporal_stride: int = 16):
    """Bounded all-detector coverage: spatial neighbors plus temporal stride."""
    pairs = set()
    for i in range(ndet):
        if i + 1 < ndet:
            pairs.add((i, i + 1))
        if i + temporal_stride < ndet:
            pairs.add((i, i + temporal_stride))
    return sorted(pairs)


def _load_front():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.dominance.fronts import f3_batch_sampling_vs_stim as front
    return front


def run_audit(shots=200_000, seeds=(1234,), threads=(1,)):
    front = _load_front()
    n, ops, dets = front.build_surface_code_noisy(5, 8)
    circuit = front._to_stim_with_detectors(ops, dets)
    dem = circuit.detector_error_model(approximate_disjoint_errors=False)
    pairs = detector_pairs(len(dets), 16)
    references = {(i, j): joint_reference(dem, i, j) for i, j in pairs}
    out = []
    for seed in seeds:
        for nthreads in threads:
            ml = front.moonlab_sample_detectors(n, ops, dets, shots, seed, nthreads)
            st = front.stim_sample_detectors(circuit.compile_detector_sampler(seed=seed + 1), shots)
            rows = []
            for i, j in pairs:
                ref = references[i, j]
                rows.append({"pair": [i, j], "moonlab": pair_record(cells_from_bits(ml[i], ml[j]), ref, shots),
                             "stim": pair_record(cells_from_bits(st[i], st[j]), ref, shots)})
            out.append({"seed": seed, "threads": nthreads, "pairs": rows,
                        "passed": all(r["moonlab"]["passed"] and r["stim"]["passed"] for r in rows)})
    identity = json.loads(subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve().parents[2] / "scripts" /
                                 "moonlab_source_identity.py"), "--repo-root",
             str(Path(__file__).resolve().parents[2])], text=True))
    lib_hash = hashlib.sha256(Path(front._LIB._name).resolve().read_bytes()).hexdigest()
    return {"schema": "moonlab.detector_joint_accuracy.v1", "detectors": len(dets),
            "pairs": len(pairs), "shots": shots, "runs": out,
            "per_cell_alpha": ALPHA_6SIGMA,
            "familywise_false_rejection_bound_under_iid": min(1.0, len(pairs)*4*2*len(out)*ALPHA_6SIGMA),
            "scope": "selected pairwise distributions, not all higher-order correlations",
            "stim_version": stim.__version__, "numpy_version": np.__version__,
            "platform": platform.platform(), "source_identity": identity,
            "library_sha256": lib_hash,
            "reference": "Stim detector_error_model(approximate_disjoint_errors=False), no gauges"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots", type=int, default=200_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234])
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    args = ap.parse_args()
    if not 1 <= args.shots <= 1_000_000 or not 1 <= len(args.seeds) <= 32:
        ap.error("shots must be 1..1000000 and seeds 1..32")
    if any(not 1 <= t <= 4 for t in args.threads):
        ap.error("threads must be 1..4")
    if any(s < 0 or s >= 2**64 - 1 for s in args.seeds):
        ap.error("seeds must be unsigned 64-bit values leaving room for seed+1")
    result = run_audit(args.shots, args.seeds, args.threads)
    result["passed"] = all(r["passed"] for r in result["runs"])
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
