#!/usr/bin/env python3
"""Diff two benchmark-corpus directories.

Emits per-bench speedup / slowdown ratios relative to a reference.
Designed to be run in CI as a regression gate: a flag of --fail-on
SPEED will exit non-zero if any bench regressed by more than the
given factor.

Usage:
  tools/bench/diff_corpus.py REFERENCE_DIR CANDIDATE_DIR
                             [--fail-on PCT]
                             [--emit-json PATH]

Example:
  tools/bench/diff_corpus.py \\
      tools/bench/canonical/m2-ultra-0.2.0 \\
      artifacts/bench-20260421T190000Z-abc123 \\
      --fail-on 20

The format used by every manifest is defined in
src/utils/manifest.h; this tool extracts the "metrics" sub-object,
walks any "rows" arrays inside, and compares the `mean_us` field
(from bench_stats_to_json) when present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _walk_timings(metrics: Any, path: str = "") -> dict[str, float]:
    """Flatten a metrics blob into {path: mean_us} pairs."""
    out: dict[str, float] = {}
    if metrics is None:
        return out
    if isinstance(metrics, dict):
        if "mean_us" in metrics:
            out[path or "metrics"] = float(metrics["mean_us"])
        if "rows" in metrics and isinstance(metrics["rows"], list):
            for i, row in enumerate(metrics["rows"]):
                key_parts = []
                for k in ("M", "K", "N", "L", "chi", "n_sym"):
                    if k in row:
                        key_parts.append(f"{k}={row[k]}")
                row_key = ",".join(key_parts) or f"row[{i}]"
                for field in ("cpu_stats", "gpu_stats", "timing_us"):
                    if field in row and isinstance(row[field], dict):
                        out[f"{path}/{row_key}/{field}"] = float(
                            row[field].get("mean_us", 0.0))
                for field in ("cpu_us", "gpu_us", "ws_us", "legacy_us",
                              "wall_s", "time_ms"):
                    if field in row and isinstance(row[field], (int, float)):
                        out[f"{path}/{row_key}/{field}"] = float(row[field])
        if "wall_s" in metrics:
            out[f"{path or 'metrics'}/wall_s"] = float(metrics["wall_s"])
    return out


def load(path: Path) -> dict[str, dict[str, float]]:
    """Load all manifest.json files in a directory; return
    {bench_label: {timing_key: mean_us}}."""
    by_bench: dict[str, dict[str, float]] = {}
    for p in sorted(path.glob("*.manifest.json")):
        data = json.loads(p.read_text())
        metrics = data.get("metrics")
        timings = _walk_timings(metrics)
        by_bench[p.name.replace(".manifest.json", "")] = timings
    return by_bench


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--fail-on", type=float, default=None,
                    help="Fail (exit 1) if any bench regressed by more "
                         "than this percentage.")
    ap.add_argument("--emit-json", type=Path, default=None,
                    help="Write the full ratio table to this JSON file.")
    args = ap.parse_args(argv)

    if not args.reference.exists():
        print(f"reference dir does not exist: {args.reference}",
              file=sys.stderr)
        return 2
    if not args.candidate.exists():
        print(f"candidate dir does not exist: {args.candidate}",
              file=sys.stderr)
        return 2

    ref = load(args.reference)
    cand = load(args.candidate)

    all_benches = sorted(set(ref) | set(cand))
    worst_regression = 1.0
    ratios: dict[str, dict[str, float]] = {}
    for bench in all_benches:
        ref_t = ref.get(bench, {})
        cand_t = cand.get(bench, {})
        all_keys = sorted(set(ref_t) | set(cand_t))
        bench_ratios: dict[str, float] = {}
        for k in all_keys:
            if k in ref_t and k in cand_t and ref_t[k] > 0:
                r = cand_t[k] / ref_t[k]
                bench_ratios[k] = r
                if r > worst_regression:
                    worst_regression = r
        ratios[bench] = bench_ratios

    # Print a compact report.
    for bench, bench_ratios in ratios.items():
        print(f"== {bench} ==")
        if not bench_ratios:
            print("   (no overlapping timing keys)")
            continue
        for k, r in sorted(bench_ratios.items()):
            marker = "SLOW" if r > 1.10 else ("fast" if r < 0.90 else "")
            print(f"   {k:60s}  {r:6.3f}x  {marker}")
        print()

    if args.emit_json:
        args.emit_json.write_text(json.dumps(ratios, indent=2))

    if args.fail_on is not None:
        limit = 1.0 + args.fail_on / 100.0
        if worst_regression > limit:
            print(f"FAIL: worst regression {worst_regression:.2f}x "
                  f"exceeds limit {limit:.2f}x", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
