"""Runner for the dominance benchmark harness.

Discovers front modules in `fronts/`, runs each whose incumbent import
succeeds (skipping loudly with a printed reason otherwise), collects the
`BenchmarkResult`s, emits them to `results/dominance_results.jsonl`, and
prints a summary table.

Usage (from repo root, with the shared library built):

    MOONLAB_LIB_DIR=$(pwd)/build-bench \
    PYTHONPATH=$(pwd)/bindings/python \
    python3 benchmarks/dominance/run_dominance.py
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import harness  # noqa: E402

# (module basename, human label, incumbent import name) for each front.
_FRONTS = [
    ("f3_clifford_vs_stim", "F3 Clifford/stabilizer", "stim"),
]


def _incumbent_available(import_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(import_name)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _fmt_rate(v: float) -> str:
    if v >= 1e9:
        return f"{v / 1e9:.3f}G/s"
    if v >= 1e6:
        return f"{v / 1e6:.3f}M/s"
    if v >= 1e3:
        return f"{v / 1e3:.3f}k/s"
    return f"{v:.3f}/s"


def main() -> int:
    results = []
    ran_any = False

    for mod_name, label, incumbent in _FRONTS:
        available, reason = _incumbent_available(incumbent)
        if not available:
            print(f"[SKIP] {label}: incumbent '{incumbent}' not importable "
                  f"({reason}). Install it (e.g. 'pip install {incumbent}').")
            continue

        try:
            mod = importlib.import_module(f"fronts.{mod_name}")
        except Exception:  # noqa: BLE001
            print(f"[SKIP] {label}: could not import fronts.{mod_name}:")
            traceback.print_exc()
            continue

        try:
            front_results = mod.benchmark()
        except Exception:  # noqa: BLE001
            print(f"[FAIL] {label}: benchmark() raised:")
            traceback.print_exc()
            continue

        results.extend(front_results)
        ran_any = True

    if not ran_any:
        print("\nNo fronts ran. Nothing emitted.")
        return 1

    path = harness.emit(results)

    # Summary table.
    print("\n" + "=" * 100)
    print("DOMINANCE BENCHMARK SUMMARY")
    print("=" * 100)
    header = (f"{'front':<5} {'name':<34} {'incumbent':<8} "
              f"{'moonlab':>12} {'incumbent':>12} {'speedup':>9} {'verdict':>10}")
    print(header)
    print("-" * 100)
    for r in results:
        if r.metric.endswith("per_second"):
            mv, iv = _fmt_rate(r.moonlab_value), _fmt_rate(r.incumbent_value)
        else:
            mv, iv = f"{r.moonlab_value:.4g}", f"{r.incumbent_value:.4g}"
        speed = "n/a" if r.verdict == "incorrect" else f"{r.speedup:.3f}x"
        print(f"{r.front:<5} {r.name:<34} {r.incumbent:<8} "
              f"{mv:>12} {iv:>12} {speed:>9} {r.verdict:>10}")
    print("-" * 100)
    print(f"metric: gate_applications_per_second (higher is better); "
          f"speedup = moonlab / incumbent (>1 => MoonLab ahead)")

    # Correctness + provenance banner.
    print()
    for r in results:
        gate = "PASS" if r.correctness_ok else "FAIL"
        print(f"[correctness {gate}] {r.name}: {r.correctness_detail}")
        break  # shared correctness gate across the F3 sizes
    fp = results[0].source_fingerprint
    dirty = results[0].dirty
    print(f"\nsource_fingerprint: {fp}  (dirty={dirty}, git_head={results[0].git_head})")
    print(f"results written to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
