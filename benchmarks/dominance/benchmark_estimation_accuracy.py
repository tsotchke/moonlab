"""Bounded equal-budget detector-marginal estimation benchmark.

This is an estimation experiment, not a throughput or superiority claim.  A
wall-clock budget starts after circuit/library preparation.  A batch is counted
only when both sampling and reduction finish before the deadline; a late final
batch is explicitly reported and never contributes to the estimate.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


MAX_BATCH = 200_000
MAX_OUTPUT_BYTES = 256 * 1024 * 1024


def batch_seed(base_seed: int, batch: int) -> int:
    """Domain-separated streams; adjacent experiment seeds must not overlap."""
    material = base_seed.to_bytes(8, 'little') + batch.to_bytes(8, 'little')
    return int.from_bytes(hashlib.blake2b(material, digest_size=8,
                                         person=b'MLBenchSeed').digest(), 'little')


def unpack_detector_output(raw, engine: str, detectors: int, shots: int) -> np.ndarray:
    """Normalize all three engine layouts to detector-major uint8 bits."""
    a = np.asarray(raw)
    if engine == "moonlab":
        expected = (detectors, shots)
        if a.shape != expected:
            raise ValueError(f"Moonlab output shape {a.shape}, expected {expected}")
        return a
    if engine == "stim-unpacked":
        expected = (shots, detectors)
        if a.shape != expected:
            raise ValueError(f"Stim output shape {a.shape}, expected {expected}")
        return a.T  # bool counts need no uint8 allocation/conversion
    if engine == "stim-packed":
        expected = (shots, (detectors + 7) // 8)
        if a.shape != expected:
            raise ValueError(f"Stim packed output shape {a.shape}, expected {expected}")
        # Stim packs detector bits within each shot, low bit first.
        if a.dtype != np.uint8:
            raise ValueError('packed output must contain uint8 bytes')
        return np.unpackbits(a, axis=1, count=detectors, bitorder="little").T
    raise ValueError(f"unknown engine: {engine}")


def marginal_rmse(counts: np.ndarray, shots: int, reference: np.ndarray) -> float:
    if shots <= 0 or counts.shape != reference.shape:
        raise ValueError("counts/reference shape or shots invalid")
    estimate = np.asarray(counts, dtype=np.float64) / shots
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


def iid_expected_mse(reference: np.ndarray, shots: int) -> float:
    """Expected squared RMSE (mean squared error) for fixed-N IID draws.

    Deadline stopping is optional-stopping data, so this is a descriptive
    fixed-count IID baseline, not a conditional guarantee for the run.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    return float(np.mean(reference * (1.0 - reference) / shots))


def aggregate_until_deadline(sample_batch, reference, budget_s, batch_size=100_000,
                              clock=time.perf_counter, seed=1234, reduce_batch=None):
    """Sample/reduce batches under a deadline; sampling and reduction are injectable."""
    reference = np.asarray(reference, dtype=np.float64)
    if (not 0 < budget_s <= 1 or not isinstance(batch_size, int)
            or not 0 < batch_size <= MAX_BATCH):
        raise ValueError("budget must be <=1s and batch_size <=200000")
    if (reference.ndim != 1 or reference.size == 0
            or not np.all(np.isfinite(reference)) or np.any(reference < 0) or np.any(reference > 1)):
        raise ValueError("reference must be a non-empty vector")
    if not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if reference.size * batch_size > MAX_OUTPUT_BYTES:
        raise ValueError("one batch output exceeds 256 MB")
    start = clock()
    deadline = start + budget_s
    counts = np.zeros(reference.size, dtype=np.int64)
    completed = 0
    batches = 0
    late = False
    discarded = 0
    last_checkpoint = start
    def count_bits(raw):
        bits = np.asarray(raw)
        if bits.shape != (reference.size, batch_size):
            raise ValueError(f'backend returned {bits.shape}, expected {(reference.size, batch_size)}')
        if bits.dtype.kind not in 'bu' or np.any(bits > 1):
            raise ValueError('backend must return binary bits')
        return np.sum(bits, axis=1, dtype=np.int64)
    reduce_batch = reduce_batch or count_bits
    while clock() < deadline:
        n = int(batch_size)
        raw = sample_batch(n, batch_seed(seed, batches))
        delta = np.asarray(reduce_batch(raw))
        if (delta.shape != counts.shape or delta.dtype.kind not in 'iu'
                or np.any(delta < 0) or np.any(delta > n)):
            raise ValueError('reducer must return one valid integer count per detector')
        proposed = counts + delta.astype(np.int64, copy=False)
        end = clock()  # includes the actual sum and aggregation, not just a cast
        if end > deadline:
            late = True
            discarded += n
            break
        counts = proposed
        completed += n
        batches += 1
        last_checkpoint = end
    actual = max(0.0, clock() - start)
    return {"shots": completed, "batches": batches, "late_batch_discarded": late,
            "discarded_shots": discarded, "last_accepted_checkpoint_time_s": last_checkpoint - start,
            "counts": counts, "actual_time_s": actual,
            "rmse": marginal_rmse(counts, completed, reference) if completed else None,
            "iid_expected_mse": iid_expected_mse(reference, completed) if completed else None}


def _rss_bytes():
    x = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(x if sys.platform == "darwin" else x * 1024)


def _make_backend(engine, front, n, ops, detectors, threads, seed):
    if engine == "moonlab":
        op_array = front._pf_op_array(ops)
        offsets, indices = front._det_csr(detectors)
        def sample(shots, shot_seed):
            out = np.empty((len(detectors), shots), dtype=np.uint8)
            got = front._LIB.pauli_frame_batch_sample_detectors(
                n, op_array, len(ops), offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
                indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), len(detectors), shots,
                shot_seed, threads, out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
            if got != len(detectors):
                raise RuntimeError(f"sampler returned {got}, expected {len(detectors)}")
            return out
        return sample
    sampler = front._to_stim_with_detectors(ops, detectors).compile_detector_sampler(seed=seed)
    def sample(shots, _shot_seed):
        return unpack_detector_output(sampler.sample(shots, bit_packed=engine == "stim-packed"),
                                       engine, len(detectors), shots)
    return sample


def run_one(engine, library, seed, budget_s=1.0, batch_size=100_000, distance=5, rounds=8):
    if engine not in ('moonlab', 'stim-unpacked', 'stim-packed'):
        raise ValueError('unknown estimation backend')
    prepare_start = time.perf_counter()
    os.environ["MOONLAB_LIB_DIR"] = str(Path(library).resolve().parent)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import stim
    from benchmarks.dominance.fronts import f3_batch_sampling_vs_stim as front
    loaded = Path(front._LIB._name).resolve()
    claimed = Path(library).resolve()
    if loaded != claimed:
        raise RuntimeError(f"loaded library {loaded} differs from claimed {claimed}")
    n, ops, detectors = front.build_surface_code_noisy(distance, rounds, .001, .001, .001)
    circuit = front._to_stim_with_detectors(ops, detectors)
    reference = front.detector_reference_probabilities(circuit)
    sample = _make_backend(engine, front, n, ops, detectors, 1, seed)
    prepare_time_s = time.perf_counter() - prepare_start
    rec = aggregate_until_deadline(sample, reference, budget_s, batch_size, seed=seed)
    peak_rss = _rss_bytes()
    repo = Path(__file__).resolve().parents[2]
    source = json.loads(subprocess.check_output(
        [sys.executable, str(repo/'scripts/moonlab_source_identity.py'), '--repo-root', str(repo)],
        text=True))
    rec.update({"engine": engine, "seed": seed, "distance": distance, "rounds": rounds,
                "noise": .001, "detectors": len(detectors), "budget_s": budget_s,
                "batch_size": batch_size, "stim_version": stim.__version__,
                "numpy_version": np.__version__, "platform": platform.platform(),
                "library": str(Path(library).resolve()),
                "library_sha256": hashlib.sha256(Path(library).read_bytes()).hexdigest(),
                "process_peak_rss_bytes": peak_rss, "source": source,
                "requested_threads": 1, "moonlab_batch_seed_scheme": "BLAKE2b-MLBenchSeed(base,counter)",
                "scope": "estimate checkpoint available by deadline; native final call may overrun; steady-state budget excludes preparation",
                "reference_marginals": reference.tolist(),
                "prepare_time_s": prepare_time_s,
                "identity": "Moonlab Pauli-frame detector sampler" if engine == "moonlab" else
                            f"Stim compile_detector_sampler ({engine})"})
    rec["counts"] = rec["counts"].tolist()
    rec["deadline_overshoot_s"] = max(0.0, rec["actual_time_s"] - budget_s)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--engines", nargs="+", default=["moonlab", "stim-unpacked", "stim-packed"],
                    choices=["moonlab", "stim-unpacked", "stim-packed"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1234, 2345, 3456, 4567, 5678])
    ap.add_argument("--budget-s", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=100_000)
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if (len(args.seeds) > 10 or any(s < 0 or s >= 2**64 - 1 for s in args.seeds)
            or len(set(args.seeds)) != len(args.seeds)):
        ap.error("use 1..10 distinct uint64 seeds below 2**64-1")
    if len(set(args.engines)) != len(args.engines):
        ap.error('engines must be distinct')
    if args._child and (len(args.engines) != 1 or len(args.seeds) != 1):
        ap.error('child mode requires exactly one engine and seed')
    if not 0 < args.budget_s <= 1 or not 0 < args.batch_size <= MAX_BATCH:
        ap.error("budget must be <=1s and batch-size <=200000")
    if not args._child and len(args.engines) * len(args.seeds) > 1:
        # A clean interpreter and freshly loaded shared library per cell keeps
        # allocator/JIT/cache state from one engine or seed affecting another.
        runs = []
        for index, seed in enumerate(args.seeds):
            shift = index % len(args.engines)
            for engine in args.engines[shift:] + args.engines[:shift]:
                cmd = [sys.executable, __file__, "--library", str(args.library),
                       "--engines", engine, "--seeds", str(seed),
                       "--budget-s", str(args.budget_s), "--batch-size", str(args.batch_size),
                       "--_child"]
                child = json.loads(subprocess.check_output(cmd, text=True))
                runs.extend(child["runs"])
    else:
        runs = [run_one(args.engines[0], args.library, args.seeds[0],
                        args.budget_s, args.batch_size)]
    print(json.dumps({"schema": "moonlab.detector_marginal_estimation.v1",
                      "budget_s": args.budget_s, "runs": runs}, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
