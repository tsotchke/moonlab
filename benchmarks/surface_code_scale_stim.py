#!/usr/bin/env python3
"""Same-machine Stim baseline for the distance-d rotated surface-code memory
benchmark (task #59).

The point of this script is that Stim and Moonlab run the *identical* circuit.
`benchmarks/surface_code_scale.c --dump-stim` writes the circuit it built in
Stim's text format; this script loads that file, so any difference in the
numbers is engine performance, not a difference in what was simulated.

Two phases are timed, matching the split in Stim's 2021 paper:

  analysis  -- `circuit.compile_detector_sampler()`, the one-off setup that
               resolves the noiseless reference trajectory.
  sampling  -- `sampler.sample(shots)`, the per-shot batched pass.

The script also runs Stim's own `Circuit.generated("surface_code:rotated_memory_z")`
at the same distance so the construction can be compared against the canonical
one, and cross-checks Moonlab's per-detector fire rates against Stim's on the
shared circuit.

usage:
  python3 benchmarks/surface_code_scale_stim.py --d 5 9 15 25 --p 0.001 \
      --shots 1024 --bin ./build/surface_code_scale --out results.json
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


def best_of(fn, repeats):
    """Return (best_seconds, last_result)."""
    best = math.inf
    res = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        res = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, res


def run_moonlab(binary, d, rounds, p, shots, budget, seed, stim_path, rates_path):
    """Run the C harness, returning its JSON dict."""
    cmd = [
        binary, "--d", str(d), "--rounds", str(rounds), "--p", str(p),
        "--shots", str(shots), "--analysis-budget", str(budget),
        "--seed", str(seed), "--json",
    ]
    if stim_path:
        cmd += ["--dump-stim", stim_path]
    if rates_path:
        cmd += ["--det-rates", rates_path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def stim_phases(stim, circuit, shots, repeats):
    """Time stim's analysis and sampling phases on a circuit object."""
    analysis_s, sampler = best_of(circuit.compile_detector_sampler, repeats)
    sample_s, samples = best_of(lambda: sampler.sample(shots=shots), repeats)
    return {
        "analysis_s": analysis_s,
        "sampling_s": sample_s,
        "shots_per_s": shots / sample_s if sample_s > 0 else float("inf"),
        "detector_fraction": float(samples.mean()),
        "samples_shape": list(samples.shape),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=None,
                    help="rounds (default: equal to d, matching Stim's paper)")
    ap.add_argument("--p", type=float, default=0.001)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--analysis-budget", type=float, default=60.0,
                    help="cap on Moonlab's analysis pass, seconds")
    ap.add_argument("--bin", default="./build/surface_code_scale")
    ap.add_argument("--out", default=None)
    ap.add_argument("--xcheck-shots", type=int, default=20000,
                    help="shots used for the per-detector cross-check")
    ap.add_argument("--xcheck-max-d", type=int, default=9,
                    help="largest d to run the per-detector cross-check at")
    ap.add_argument("--skip-moonlab", action="store_true")
    args = ap.parse_args()

    stim = _import_stim()

    rows = []
    for d in args.d:
        rounds = args.rounds if args.rounds is not None else d
        row = {"d": d, "rounds": rounds, "p": args.p, "shots": args.shots}
        tmpdir = tempfile.mkdtemp(prefix=f"sc_d{d}_")
        stim_path = os.path.join(tmpdir, "circuit.stim")
        rates_path = os.path.join(tmpdir, "moonlab_det_rates.txt")

        # ---- Moonlab (also emits the shared circuit) ----
        if not args.skip_moonlab:
            print(f"[d={d}] moonlab ...", file=sys.stderr, flush=True)
            ml = run_moonlab(args.bin, d, rounds, args.p, args.shots,
                             args.analysis_budget, args.seed, stim_path, None)
            row["moonlab"] = ml
        else:
            # Still need the circuit file; ask for it with a trivial run.
            run_moonlab(args.bin, d, rounds, args.p, 1, 0.001, args.seed,
                        stim_path, None)

        # ---- Stim on the IDENTICAL circuit ----
        print(f"[d={d}] stim (shared circuit) ...", file=sys.stderr, flush=True)
        shared = stim.Circuit(open(stim_path).read())
        row["circuit"] = {
            "num_qubits_indexed": shared.num_qubits,
            "num_measurements": shared.num_measurements,
            "num_detectors": shared.num_detectors,
            "num_observables": shared.num_observables,
            "num_operations": len(shared.flattened()),
        }
        row["stim_shared_circuit"] = stim_phases(stim, shared, args.shots,
                                                 args.repeats)

        # ---- Stim's own canonical construction at the same parameters ----
        print(f"[d={d}] stim (native generated) ...", file=sys.stderr, flush=True)
        native = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=d, rounds=rounds,
            after_clifford_depolarization=args.p,
            before_round_data_depolarization=args.p,
            before_measure_flip_probability=args.p,
            after_reset_flip_probability=args.p,
        )
        row["stim_native"] = stim_phases(stim, native, args.shots, args.repeats)
        row["stim_native_circuit"] = {
            "num_qubits_indexed": native.num_qubits,
            "num_measurements": native.num_measurements,
            "num_detectors": native.num_detectors,
            "num_observables": native.num_observables,
        }

        # ---- Correctness gates ----
        gates = {}
        # (a) the shared circuit's detectors and observable are deterministic:
        #     if they were not, detector_error_model() raises.
        try:
            dem = shared.detector_error_model(decompose_errors=True)
            gates["dem_builds"] = True
            gates["dem_num_errors"] = dem.num_errors
        except Exception as exc:  # noqa: BLE001
            gates["dem_builds"] = False
            gates["dem_error"] = str(exc)[:400]

        # (b) construction matches stim's canonical rotated_memory_z on the
        #     structural invariants
        gates["matches_stim_construction"] = (
            shared.num_measurements == native.num_measurements
            and shared.num_detectors == native.num_detectors
            and shared.num_observables == native.num_observables
        )

        # (c) per-detector firing rates agree between the engines on the
        #     shared circuit (only at small d, where enough shots are cheap)
        if d <= args.xcheck_max_d and not args.skip_moonlab:
            print(f"[d={d}] per-detector cross-check ...", file=sys.stderr,
                  flush=True)
            n = args.xcheck_shots
            run_moonlab(args.bin, d, rounds, args.p, n, args.analysis_budget,
                        args.seed, None, rates_path)
            ml_counts = [int(x) for x in open(rates_path).read().split()]
            st = shared.compile_detector_sampler().sample(shots=n)
            st_counts = st.sum(axis=0).tolist()
            assert len(ml_counts) == len(st_counts), "detector count mismatch"

            # Two-proportion z-test per detector; report the worst deviation
            # and how many detectors exceed 4 sigma (expect ~0 of a few hundred).
            worst_z, n_bad = 0.0, 0
            for a, b in zip(ml_counts, st_counts):
                pa, pb = a / n, b / n
                pool = (a + b) / (2 * n)
                se = math.sqrt(2 * pool * (1 - pool) / n) if 0 < pool < 1 else 0.0
                z = abs(pa - pb) / se if se > 0 else 0.0
                worst_z = max(worst_z, z)
                if z > 4.0:
                    n_bad += 1
            gates["detector_rate_xcheck"] = {
                "shots": n,
                "num_detectors": len(ml_counts),
                "moonlab_mean_rate": sum(ml_counts) / (n * len(ml_counts)),
                "stim_mean_rate": sum(st_counts) / (n * len(st_counts)),
                "worst_abs_z": worst_z,
                "detectors_beyond_4_sigma": n_bad,
                "pass": n_bad == 0,
            }
        row["gates"] = gates

        # ---- Head-to-head ratios ----
        ml = row.get("moonlab")
        if ml:
            st = row["stim_shared_circuit"]
            ml_an = (ml["analysis"]["wall_s"] if ml["analysis"]["completed"]
                     else ml["analysis"]["projected_full_s"])
            row["ratio"] = {
                "analysis_moonlab_over_stim": ml_an / st["analysis_s"],
                "analysis_moonlab_completed": ml["analysis"]["completed"],
                "sampling_stim_over_moonlab_shots_per_s": (
                    st["shots_per_s"] / ml["sampling"]["shots_per_s"]
                    if ml["sampling"]["ran"] and ml["sampling"]["shots_per_s"] > 0
                    else None),
            }
        rows.append(row)
        print(json.dumps(row.get("ratio", {}), indent=2), file=sys.stderr)

    doc = {
        "schema": "moonlab/surface_code_scale_vs_stim_v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "stim_version": stim.__version__,
        "params": {
            "p": args.p, "shots": args.shots, "repeats": args.repeats,
            "seed": args.seed, "rounds": "d" if args.rounds is None else args.rounds,
            "noise_model": ("circuit-level depolarising: DEPOLARIZE1(p) on data "
                            "before each round and after each 1q Clifford, "
                            "DEPOLARIZE2(p) after each CNOT, X_ERROR(p) after "
                            "reset and before measurement"),
        },
        "rows": rows,
    }
    text = json.dumps(doc, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
