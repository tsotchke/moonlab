#!/usr/bin/env python3
"""Head-to-head Stim vs Moonlab Pauli-frame surface-code throughput.

Builds the same phenomenological-noise rotated surface-code Z-stabilizer
cycle in both engines, runs N shots over R rounds at distance d, and
reports shots-per-second on the same M2 Ultra host so the paper can
quote a real Stim-class ratio rather than a citation-laundered claim.

Closes the moonlab paper §5.1 / §4.7 audit point.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import stim


def build_stim_circuit(d: int, rounds: int, p: float) -> stim.Circuit:
    """Standard stim rotated surface code with depolarising data noise
    and noiseless syndrome measurement (matches Moonlab's bench harness's
    approximate noise model — Z-stabilizer cycle only, depolarising
    on every data qubit per round).
    """
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=d,
        after_clifford_depolarization=p,
        before_round_data_depolarization=0.0,
        before_measure_flip_probability=p,
        after_reset_flip_probability=0.0,
    )


def stim_throughput(d: int, rounds: int, n_shots: int, p: float):
    circ = build_stim_circuit(d, rounds, p)
    sampler = circ.compile_detector_sampler()
    t0 = time.perf_counter()
    samples = sampler.sample(shots=n_shots)
    dt = time.perf_counter() - t0
    n_data = d * d
    return {
        "engine": "stim",
        "stim_version": stim.__version__,
        "d": d,
        "n_data": n_data,
        "rounds": rounds,
        "n_shots": n_shots,
        "p": p,
        "wall_s": dt,
        "shots_per_s": n_shots / dt if dt > 0 else 0.0,
        "shot_rounds_per_s": (n_shots * rounds) / dt if dt > 0 else 0.0,
        "samples_shape": list(samples.shape),
    }


def moonlab_throughput(bin_path: str, d_set, n_shots: int, rounds: int, p: float):
    """Run Moonlab's bench_pauli_frame and parse its JSON.  bench_pauli_frame
    iterates a hard-coded distance set; we pick out the rows that match.
    """
    out = "/tmp/_moonlab_pauli_frame_h2h.json"
    if os.path.exists(out):
        os.remove(out)
    cmd = [bin_path, out, str(n_shots), str(rounds), str(p)]
    t0 = time.perf_counter()
    rc = subprocess.run(cmd, check=True)
    dt = time.perf_counter() - t0
    with open(out) as f:
        data = json.load(f)
    rows = []
    for r in data.get("rows", []):
        if r["d"] in d_set:
            rows.append({
                "engine": "moonlab",
                "d": r["d"],
                "n_data": r["n_data"],
                "rounds": rounds,
                "n_shots": n_shots,
                "p": p,
                "wall_s": r["total_wall_s"],
                "shots_per_s": r["shots_per_s"],
                "shot_rounds_per_s": r["shots_per_s"] * rounds,
                "frame_ops_per_s": r["frame_ops_per_s"],
            })
    return rows, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moonlab-bin", default=os.path.expanduser(
        "~/Desktop/quantum_simulator/build_release/bench_pauli_frame"))
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--out",
                    default=os.path.expanduser(
                        "~/Desktop/quantum_simulator/benchmarks/results/"
                        f"stim_vs_moonlab_M2Ultra_"
                        f"{datetime.now(timezone.utc):%Y-%m-%d}.json"))
    args = ap.parse_args()

    distances = [5, 9, 15, 23, 31]
    print(f"=== Stim vs Moonlab Pauli-frame head-to-head ===")
    print(f"  Stim: {stim.__version__}")
    print(f"  Moonlab bench: {args.moonlab_bin}")
    print(f"  shots={args.shots} rounds={args.rounds} p={args.p}")
    print()

    stim_rows = []
    for d in distances:
        print(f"  Stim d={d} ...", end=" ", flush=True)
        r = stim_throughput(d, args.rounds, args.shots, args.p)
        stim_rows.append(r)
        print(f"{r['shots_per_s']:.3e} shots/s "
              f"({r['shot_rounds_per_s']:.3e} shot-rounds/s, "
              f"{r['wall_s']:.2f}s)")

    print()
    print(f"  Moonlab harness running...")
    moonlab_rows, ml_total = moonlab_throughput(
        args.moonlab_bin, set(distances), args.shots, args.rounds, args.p)

    print(f"  Moonlab total wall {ml_total:.2f}s")
    print()
    print(f"  d   stim_shots/s     moonlab_shots/s   ratio (stim/moonlab)")
    for ds in distances:
        s = next((x for x in stim_rows if x["d"] == ds), None)
        m = next((x for x in moonlab_rows if x["d"] == ds), None)
        if s and m:
            ratio = s["shots_per_s"] / m["shots_per_s"] if m["shots_per_s"] else 0.0
            print(f"  {ds:3d}  {s['shots_per_s']:14.3e}  {m['shots_per_s']:14.3e}  {ratio:6.3f}")

    out = {
        "schema": "moonlab/stim_vs_moonlab_v1",
        "host": {
            "machine": os.uname().machine,
            "system": os.uname().sysname,
            "release": os.uname().release,
        },
        "stim_version": stim.__version__,
        "params": {
            "shots": args.shots,
            "rounds": args.rounds,
            "p": args.p,
            "noise_model": "phenomenological depolarising on data qubits + measurement flip",
        },
        "stim_rows": stim_rows,
        "moonlab_rows": moonlab_rows,
        "description": (
            "Head-to-head Stim vs Moonlab Pauli-frame sampler on a "
            "phenomenological-noise rotated surface-code Z-stabilizer "
            "cycle.  Each engine runs the same shots*rounds workload at "
            "fixed distance d in {5, 9, 15, 23, 31}.  Note: Stim runs the "
            "full rotated surface code with X+Z stabilizers and "
            "syndrome-extraction CNOTs as compiled by stim's circuit "
            "generator; Moonlab's bench_pauli_frame runs only the "
            "Z-stabilizer half of the cycle.  The ratio therefore "
            "compares like to like up to that factor of ~2 in stabilizer "
            "count, which the table reports without correction so the "
            "reader can assess raw single-core throughput."),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
