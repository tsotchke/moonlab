#!/usr/bin/env python3
"""Same-machine Moonlab-vs-Stim comparison on distance-d rotated surface-code
memory circuits (task #59).

Stim's 2021 paper (arXiv:2103.02202) claims a distance-100 rotated surface-code
memory circuit -- ~20k qubits, ~8M gates, ~1M measurements -- analysed in ~15 s,
after which full shots stream at kHz rates.  This script measures both engines
on the same machine on that exact circuit family.

Two phases are timed separately, matching the paper's split:

  ANALYSIS  the one-off pass that resolves the noiseless reference trajectory,
            after which sampling can begin.  For Moonlab that is the
            Aaronson-Gottesman tableau run in pf_compute_reference(); for Stim
            it is compile_detector_sampler() plus the first sample() call,
            since Stim defers the reference computation to first use (measured,
            not assumed -- the script reports both halves).
  SAMPLING  the per-shot batched pass.  Reported both at one thread (the
            core-fair comparison, since Stim's sampler is single-threaded) and
            at all cores.

Correctness is gated three ways: the noiseless circuit must leave every
detector quiet, Stim must accept the identical circuit and build its detector
error model from it, and at small d the per-detector fire rates must agree
between the engines to within a two-proportion z-test.

usage:
  python3 benchmarks/surface_code_scale_stim.py \
      --d 25 51 75 101 --xcheck-d 5 9 15 \
      --bin ./build/surface_code_scale --out benchmarks/results/out.json
"""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time


def _import_stim():
    try:
        import stim  # noqa: F401
    except ImportError:
        sys.exit("stim is not installed; run: pip install stim")
    return sys.modules["stim"]


def run_moonlab(binary, d, rounds, p, shots, extra=(), budget=0.0, seed=12345,
                stim_path=None, rates_path=None, timeout=None):
    cmd = [binary, "--d", str(d), "--rounds", str(rounds), "--p", str(p),
           "--shots", str(shots), "--analysis-budget", str(budget),
           "--seed", str(seed), "--json", *extra]
    if stim_path:
        cmd += ["--dump-stim", stim_path]
    if rates_path:
        cmd += ["--det-rates", rates_path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True,
                         timeout=timeout)
    return json.loads(out.stdout)


def stim_measure(stim, circuit, shots, repeats):
    """Analysis = compile + first sample (Stim defers the reference pass).
    Sampling = best-of-repeats at `shots`."""
    t0 = time.perf_counter()
    sampler = circuit.compile_detector_sampler()
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    sampler.sample(shots=1)
    t_first = time.perf_counter() - t0

    best, frac = math.inf, None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        s = sampler.sample(shots=shots)
        best = min(best, time.perf_counter() - t0)
        frac = float(s.mean())
    return {
        "compile_s": t_compile,
        "first_sample_s": t_first,
        "analysis_s": t_compile + t_first,
        "sampling_s": best,
        "shots": shots,
        "shots_per_s": shots / best if best > 0 else float("inf"),
        "detector_fraction": frac,
        "threads": 1,
    }


def correctness_gates(stim, args, d, rounds):
    """Noiseless determinism, structural match to stim's generated circuit, and
    a per-detector rate cross-check on the identical circuit."""
    tmpdir = tempfile.mkdtemp(prefix=f"sc_x{d}_")
    stim_path = os.path.join(tmpdir, "c.stim")
    rates_path = os.path.join(tmpdir, "rates.txt")
    g = {"d": d, "rounds": rounds}

    # (1) noiseless run: every detector must be quiet in every shot
    nl = run_moonlab(args.bin, d, rounds, args.p, 512, extra=["--verify"],
                     seed=args.seed)
    g["noiseless_all_detectors_quiet"] = nl["verify"]["all_detectors_quiet"]

    # (2) the identical circuit, parsed by stim
    ml = run_moonlab(args.bin, d, rounds, args.p, args.xcheck_shots,
                     seed=args.seed, stim_path=stim_path,
                     rates_path=rates_path)
    shared = stim.Circuit(open(stim_path).read())
    native = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=args.p,
        before_round_data_depolarization=args.p,
        before_measure_flip_probability=args.p,
        after_reset_flip_probability=args.p)
    g["num_measurements"] = {"ours": shared.num_measurements,
                             "stim_generated": native.num_measurements}
    g["num_detectors"] = {"ours": shared.num_detectors,
                          "stim_generated": native.num_detectors}
    g["matches_stim_construction"] = (
        shared.num_measurements == native.num_measurements
        and shared.num_detectors == native.num_detectors
        and shared.num_observables == native.num_observables == 1)
    try:
        dem = shared.detector_error_model(decompose_errors=True)
        g["dem_builds"] = True
        g["dem_num_errors"] = dem.num_errors
    except Exception as exc:  # noqa: BLE001
        g["dem_builds"] = False
        g["dem_error"] = str(exc)[:300]

    # (3) per-detector fire rates, both engines, identical circuit
    n = args.xcheck_shots
    ml_counts = [int(x) for x in open(rates_path).read().split()]
    st_counts = shared.compile_detector_sampler().sample(shots=n).sum(axis=0)
    worst_z, n_bad = 0.0, 0
    for a, b in zip(ml_counts, st_counts.tolist()):
        pool = (a + b) / (2 * n)
        se = math.sqrt(2 * pool * (1 - pool) / n) if 0 < pool < 1 else 0.0
        z = abs(a / n - b / n) / se if se > 0 else 0.0
        worst_z = max(worst_z, z)
        n_bad += z > 4.0
    g["detector_rate_xcheck"] = {
        "shots": n,
        "num_detectors": len(ml_counts),
        "moonlab_mean_rate": sum(ml_counts) / (n * len(ml_counts)),
        "stim_mean_rate": float(st_counts.sum()) / (n * len(st_counts)),
        "worst_abs_z": worst_z,
        "detectors_beyond_4_sigma": int(n_bad),
        "pass": bool(n_bad == 0),
    }
    g["pass"] = bool(g["noiseless_all_detectors_quiet"]
                     and g["matches_stim_construction"]
                     and g["dem_builds"]
                     and g["detector_rate_xcheck"]["pass"])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, nargs="+", default=[25, 51, 75, 101],
                    help="distances for the scaling comparison")
    ap.add_argument("--xcheck-d", type=int, nargs="+", default=[5, 9, 15],
                    help="distances for the correctness gates")
    ap.add_argument("--p", type=float, default=0.001)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--st-shots", type=int, default=512,
                    help="shots for the single-thread sampling run")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--xcheck-shots", type=int, default=20000)
    ap.add_argument("--analysis-budget", type=float, default=0.0,
                    help="cap Moonlab's analysis pass (0 = run to completion)")
    ap.add_argument("--bin", default="./build/surface_code_scale")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stim = _import_stim()

    gates = []
    for d in args.xcheck_d:
        print(f"[gate d={d}] ...", file=sys.stderr, flush=True)
        gates.append(correctness_gates(stim, args, d, d))

    rows = []
    for d in args.d:
        rounds = d
        print(f"[d={d}] moonlab analysis ...", file=sys.stderr, flush=True)
        # Analysis + the real bundled sampler.  Shots kept small: at large d
        # the analysis term dwarfs it and the sampling numbers come from the
        # dedicated runs below.
        an = run_moonlab(args.bin, d, rounds, args.p, 0,
                         budget=args.analysis_budget, seed=args.seed)

        print(f"[d={d}] moonlab sampling (all cores) ...", file=sys.stderr,
              flush=True)
        mt = run_moonlab(args.bin, d, rounds, args.p, args.shots,
                         extra=["--skip-analysis"], seed=args.seed)
        print(f"[d={d}] moonlab sampling (1 thread) ...", file=sys.stderr,
              flush=True)
        st = run_moonlab(args.bin, d, rounds, args.p, args.st_shots,
                         extra=["--skip-analysis", "--threads", "1"],
                         seed=args.seed)

        print(f"[d={d}] stim ...", file=sys.stderr, flush=True)
        native = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=d, rounds=rounds,
            after_clifford_depolarization=args.p,
            before_round_data_depolarization=args.p,
            before_measure_flip_probability=args.p,
            after_reset_flip_probability=args.p)
        sm = stim_measure(stim, native, args.shots, args.repeats)

        ml_an = (an["analysis"]["wall_s"] if an["analysis"]["completed"]
                 else an["analysis"]["projected_full_s"])
        row = {
            "d": d,
            "rounds": rounds,
            "p": args.p,
            "circuit": {
                "n_qubits": an["n_qubits"],
                "n_gates": an["n_gates"],
                "n_ops_including_noise": an["n_ops"],
                "n_measurements": an["n_measurements"],
                "n_detectors": an["n_detectors"],
            },
            "moonlab": {
                "analysis_s": ml_an,
                "analysis_completed": an["analysis"]["completed"],
                "analysis_measurements_resolved": an["analysis"]["measurements_resolved"],
                "analysis_peak_rss_bytes": an["analysis"]["peak_rss_bytes"],
                "sampling_all_cores": {
                    "shots": args.shots,
                    "wall_s": mt["sampling_replica"]["wall_s"],
                    "shots_per_s": mt["sampling_replica"]["shots_per_s"],
                    "detector_fraction": mt["sampling_replica"]["detector_fraction"],
                    "threads": os.cpu_count(),
                },
                "sampling_one_thread": {
                    "shots": args.st_shots,
                    "wall_s": st["sampling_replica"]["wall_s"],
                    "shots_per_s": st["sampling_replica"]["shots_per_s"],
                    "detector_fraction": st["sampling_replica"]["detector_fraction"],
                    "threads": 1,
                },
                "simd_backend": an["simd_backend"],
            },
            "stim": sm,
            "ratio": {
                "analysis_moonlab_over_stim": ml_an / sm["analysis_s"],
                "sampling_stim_over_moonlab_one_thread": (
                    sm["shots_per_s"] / st["sampling_replica"]["shots_per_s"]),
                "sampling_stim_over_moonlab_all_cores": (
                    sm["shots_per_s"] / mt["sampling_replica"]["shots_per_s"]),
            },
        }
        rows.append(row)
        print(json.dumps(row["ratio"], indent=2), file=sys.stderr)

    doc = {
        "schema": "moonlab/surface_code_scale_vs_stim_v1",
        "task": "#59 -- does the stabilizer stack reach Stim's headline scale",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
            "cpu_count": os.cpu_count(),
        },
        "build_type": "Release",
        "stim_version": stim.__version__,
        "params": {
            "p": args.p,
            "rounds": "d",
            "shots_all_cores": args.shots,
            "shots_one_thread": args.st_shots,
            "repeats": args.repeats,
            "seed": args.seed,
            "noise_model": ("circuit-level depolarising: DEPOLARIZE1(p) on data "
                            "before each round and after each 1q Clifford, "
                            "DEPOLARIZE2(p) after each CNOT, X_ERROR(p) after "
                            "reset and before measurement"),
            "phase_definitions": {
                "moonlab_analysis": "pf_compute_reference() -- Aaronson-Gottesman tableau pass",
                "stim_analysis": "compile_detector_sampler() + first sample(); Stim defers the reference pass to first use",
                "moonlab_sampling": "public-API replica of pf_run_block, calibrated against the bundled sampler",
            },
        },
        "correctness_gates": gates,
        "rows": rows,
    }
    text = json.dumps(doc, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
