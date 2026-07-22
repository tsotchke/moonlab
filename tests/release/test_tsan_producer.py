#!/usr/bin/env python3
"""Fail-closed contract tests for scripts/run_tsan.sh."""

from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCER = REPO_ROOT / "scripts" / "run_tsan.sh"
SUPPRESSION = REPO_ROOT / "tests" / "concurrency" / "tsan.supp"


class TSanProducerContractTest(unittest.TestCase):
    def run_producer(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(PRODUCER), *args],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_absolute_suppression_path_is_not_double_prefixed(self) -> None:
        result = self.run_producer("--internal-print-suppression-options")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(), f"halt_on_error=0:suppressions={SUPPRESSION}"
        )
        self.assertEqual(result.stdout.count(str(REPO_ROOT)), 1)

    def test_zero_races_with_nonzero_exit_is_fail(self) -> None:
        result = self.run_producer("--internal-classify-run-result", "0", "134")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.strip(), "FAIL")

    def test_zero_races_with_zero_exit_is_pass(self) -> None:
        result = self.run_producer("--internal-classify-run-result", "0", "0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "PASS")

    def test_race_summary_is_fail_even_with_zero_exit(self) -> None:
        result = self.run_producer("--internal-classify-run-result", "1", "0")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.strip(), "FAIL")


if __name__ == "__main__":
    unittest.main()
