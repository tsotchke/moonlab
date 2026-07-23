"""Shared contract for the dominance benchmark harness.

Every front benchmark produces one BenchmarkResult that pairs MoonLab against a
named incumbent on a defining metric, with a correctness gate and the clean
source fingerprint so the ICC gate can bind the result to the exact source that
produced it. See benchmarks/dominance/README.md.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]


# Verdict thresholds on the speedup ratio (incumbent_metric / moonlab_metric,
# expressed so >1 means MoonLab is faster/larger). Within +/-10% is parity.
_PARITY_BAND = 0.10


def source_fingerprint() -> dict:
    """Return the clean source identity dict, or a marker on failure.

    The benchmark result is only certifiable when this succeeds and dirty is
    False; the runner records it either way so a dirty run is never silently
    presented as gate-bindable.
    """
    try:
        out = subprocess.run(
            [sys.executable, str(_REPO_ROOT / "scripts" / "moonlab_source_identity.py")],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
        ).stdout
        return json.loads(out)
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError) as exc:
        return {"source_fingerprint": None, "error": repr(exc)}


@dataclass
class BenchmarkResult:
    """One head-to-head result, ready to bind to the ICC evidence gate."""

    front: str                      # e.g. "F3"
    name: str                       # e.g. "clifford_throughput_vs_stim"
    incumbent: str                  # e.g. "stim"
    metric: str                     # e.g. "gate_applications_per_second"
    higher_is_better: bool
    moonlab_value: float
    incumbent_value: float
    correctness_ok: bool            # both engines produced the same answer
    correctness_detail: str         # how correctness was checked
    workload: str                   # human-readable workload description
    repeats: int
    verdict: str = "unknown"        # lead | parity | behind | incorrect
    speedup: float = 0.0            # moonlab advantage on the metric (>1 = ahead)
    source_fingerprint: Optional[str] = None
    git_head: Optional[str] = None
    dirty: Optional[bool] = None
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def finalize(self) -> "BenchmarkResult":
        """Compute verdict + speedup and stamp the source fingerprint."""
        if not self.correctness_ok:
            self.verdict = "incorrect"
        else:
            m, inc = self.moonlab_value, self.incumbent_value
            if self.higher_is_better:
                ratio = (m / inc) if inc > 0 else float("inf")
            else:
                ratio = (inc / m) if m > 0 else float("inf")
            self.speedup = ratio
            if ratio >= 1.0 + _PARITY_BAND:
                self.verdict = "lead"
            elif ratio >= 1.0 - _PARITY_BAND:
                self.verdict = "parity"
            else:
                self.verdict = "behind"
        ident = source_fingerprint()
        self.source_fingerprint = ident.get("source_fingerprint")
        self.git_head = ident.get("git_head")
        self.dirty = ident.get("dirty")
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def timed(fn: Callable[[], object], repeats: int = 5, warmup: int = 1):
    """Run fn warmup+repeats times; return (best_seconds, last_result).

    Best-of captures the achievable throughput with the least scheduler noise;
    the returned result lets the caller cross-check the two engines' answers.
    """
    for _ in range(max(0, warmup)):
        fn()
    best = float("inf")
    result = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, result


def emit(results, out_dir: Optional[Path] = None) -> Path:
    """Write results to benchmarks/dominance/results/<front>.jsonl."""
    out_dir = out_dir or (_REPO_ROOT / "benchmarks" / "dominance" / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dominance_results.jsonl"
    with path.open("w") as fh:
        for r in results:
            fh.write(r.to_json() + "\n")
    return path
