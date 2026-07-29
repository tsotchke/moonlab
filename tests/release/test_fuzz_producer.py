#!/usr/bin/env python3
"""Fail-closed contract tests for scripts/run_fuzz.sh and its corpus hygiene.

Two invariants are locked here, both of which the lane violated in production:

  * A soak must not mutate the worktree it fingerprints.  libFuzzer writes new
    coverage-increasing units into the corpus directory and crash artifacts
    into crashes-pending/; while those were non-ignored untracked files,
    scripts/moonlab_source_identity.py hashed them, the end-of-lane source
    check saw a moved fingerprint, and every soak job failed itself by
    running.  The ignore rule must swallow exactly the engine's output and
    nothing a human would ever hand-write as a seed.

  * fuzz_corpus_clean must carry one verdict per run.  The lane used to emit
    the umbrella PASS at the end of the campaign and then a contradicting FAIL
    from the source check, so one check name reported two answers.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCER = REPO_ROOT / "scripts" / "run_fuzz.sh"
CORPORA = REPO_ROOT / "tests" / "fuzz" / "corpora"
TRACE = REPO_ROOT / "scripts" / "icc_traces" / "moonlab_fuzz.jsonl"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "fuzz.yml"

# libFuzzer names each unit it writes by the SHA-1 of the unit's contents:
# 40 lowercase hex characters, no extension (FuzzerUtil Hash + WriteToFile).
GENERATED_UNIT = "7af50fc0d03ce482f99d491c4b5de000bd04115e"


def is_ignored(relative: str) -> bool:
    """True iff `relative` matches a Git ignore rule (the path need not exist)."""
    return (
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", "--no-index", relative],
            check=False,
        ).returncode
        == 0
    )


def tracked_seeds() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", "tests/fuzz/corpora"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    return [line for line in out.splitlines() if line]


class CorpusIgnoreContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = sorted(p.name for p in CORPORA.iterdir() if p.is_dir())
        self.assertTrue(self.targets, "no fuzz corpus directories found")

    def test_engine_written_units_are_ignored_for_every_surface(self) -> None:
        for target in self.targets:
            with self.subTest(target=target):
                self.assertTrue(
                    is_ignored(f"tests/fuzz/corpora/{target}/{GENERATED_UNIT}"),
                    "libFuzzer output would enter the source fingerprint and the "
                    "soak would fail itself by running",
                )

    def test_crash_quarantine_is_ignored_for_every_surface(self) -> None:
        for target in self.targets:
            with self.subTest(target=target):
                self.assertTrue(
                    is_ignored(
                        f"tests/fuzz/corpora/{target}/crashes-pending/crash-deadbeef"
                    )
                )

    def test_crash_gate_reads_the_filesystem_not_git(self) -> None:
        # Ignoring the quarantine is only safe because the artifact count comes
        # from `find`, which does not care about .gitignore.
        producer = PRODUCER.read_text(encoding="utf-8")
        self.assertRegex(
            producer,
            r'arts="\$\(find "\$quarantine".*-name \'crash-\*\'',
            "crash detection must not depend on Git's view of the quarantine",
        )

    def test_tracked_seeds_are_never_ignored(self) -> None:
        seeds = tracked_seeds()
        self.assertGreaterEqual(len(seeds), 53)
        for seed in seeds:
            with self.subTest(seed=seed):
                self.assertFalse(
                    is_ignored(seed),
                    "a tracked seed is source: it must stay in the fingerprint",
                )

    def test_every_surface_still_ships_seeds(self) -> None:
        seeds = tracked_seeds()
        for target in self.targets:
            with self.subTest(target=target):
                self.assertTrue(
                    any(s.startswith(f"tests/fuzz/corpora/{target}/") for s in seeds)
                )

    def test_hand_written_seed_names_are_not_ignored(self) -> None:
        # The rule is deliberately narrow rather than `corpora/**`, so a
        # developer adding a real seed is still seen by `git status` and still
        # moves the fingerprint -- including names made only of hex letters.
        for name in (
            "valid_new_case.json",
            "regress-embedded-nul-hang",
            "boundary_badqubits.txt",
            "ca_mps.bin",
            "decade.bin",
            "beef",
            GENERATED_UNIT[:39],
            GENERATED_UNIT + "0",
            GENERATED_UNIT + ".bin",
        ):
            with self.subTest(name=name):
                self.assertFalse(is_ignored(f"tests/fuzz/corpora/config_parse_fuzz/{name}"))


class UmbrellaVerdictContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.producer = PRODUCER.read_text(encoding="utf-8")

    def test_fuzz_corpus_clean_has_exactly_one_emit_site(self) -> None:
        sites = re.findall(r"^\s*emit fuzz_corpus_clean\b", self.producer, re.M)
        self.assertEqual(
            len(sites),
            1,
            "the umbrella verdict must be written from one place, after the "
            "source check, so a run can never record two values for it",
        )

    def test_the_only_write_path_is_idempotent(self) -> None:
        # Both the normal tail and the abort trap route through
        # record_umbrella, which refuses to write a second time.
        body = self.producer.split("record_umbrella() {", 1)[1].split("\n}", 1)[0]
        self.assertIn('[ "$UMBRELLA_EMITTED" -eq 0 ] || return 0', body)
        self.assertIn("UMBRELLA_EMITTED=1", body)
        self.assertIn("emit fuzz_corpus_clean", body)

    def test_umbrella_is_recorded_after_the_source_check(self) -> None:
        check = self.producer.index('if [ "$FINAL_FINGERPRINT" != "$SOURCE_FINGERPRINT" ]')
        tail_calls = [
            m.start() for m in re.finditer(r"^record_umbrella$", self.producer, re.M)
        ]
        self.assertEqual(len(tail_calls), 1)
        self.assertLess(check, tail_calls[0])

    def test_source_check_names_the_paths_that_moved(self) -> None:
        self.assertIn("drift=", self.producer)

    def test_unknown_target_is_a_usage_error_that_preserves_evidence(self) -> None:
        before = TRACE.read_bytes() if TRACE.exists() else None
        result = subprocess.run(
            ["bash", str(PRODUCER), "soak", "5", "no_such_fuzz_target"],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        after = TRACE.read_bytes() if TRACE.exists() else None
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("unknown target", result.stderr)
        self.assertEqual(before, after, "a usage error must not truncate the trace")


class EngineArgvContractTest(unittest.TestCase):
    """The soak argv must build under every bash the fleet actually runs.

    Only three of the six surfaces ship a dictionary.  The optional argument
    used to be spliced in from a possibly-empty array as "${dict_arg[@]}",
    which is an unbound-variable abort under `set -u` on bash < 4.4 -- the
    version macOS ships.  Every dict-less soak died immediately after the
    trace was truncated, so the lane produced an empty trace and no verdict.
    """

    SHELLS = [
        shell
        for shell in dict.fromkeys(
            [s for s in (shutil.which("bash"), "/bin/bash") if s]
        )
        if Path(shell).exists()
    ]

    def targets(self) -> list[str]:
        producer = PRODUCER.read_text(encoding="utf-8")
        block = producer.split("TARGETS=(", 1)[1].split(")", 1)[0]
        return [line.strip() for line in block.splitlines() if line.strip()]

    def engine_argv(self, shell: str, target: str) -> list[str]:
        result = subprocess.run(
            [shell, str(PRODUCER), "--internal-engine-args", target, "60"],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"{shell} {target}: {result.stderr}")
        self.assertNotIn("unbound variable", result.stderr)
        return result.stdout.split()

    def test_every_surface_builds_argv_under_every_available_bash(self) -> None:
        self.assertTrue(self.SHELLS)
        for shell in self.SHELLS:
            for target in self.targets():
                with self.subTest(shell=shell, target=target):
                    argv = self.engine_argv(shell, target)
                    self.assertEqual(argv[-1], f"tests/fuzz/corpora/{target}")
                    self.assertIn(
                        f"-artifact_prefix=tests/fuzz/corpora/{target}/crashes-pending/",
                        argv,
                    )
                    self.assertIn("-max_total_time=60", argv)

    def test_dictionary_is_passed_exactly_when_one_exists(self) -> None:
        shell = self.SHELLS[0]
        for target in self.targets():
            with self.subTest(target=target):
                argv = self.engine_argv(shell, target)
                dict_path = REPO_ROOT / "tests" / "fuzz" / "dicts" / f"{target}.dict"
                expected = [f"-dict=tests/fuzz/dicts/{target}.dict"] if dict_path.exists() else []
                self.assertEqual([a for a in argv if a.startswith("-dict=")], expected)


class AbortedLaneContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.producer = PRODUCER.read_text(encoding="utf-8")

    def test_exit_trap_is_armed(self) -> None:
        self.assertIn("trap on_exit EXIT", self.producer)

    def test_lane_start_is_marked_wherever_the_trace_is_truncated(self) -> None:
        # Truncating the trace is the point at which the lane owes a verdict.
        truncations = re.findall(r"^\s*: > \"\$TRACE\"\s*$", self.producer, re.M)
        marks = re.findall(r"^\s*LANE_STARTED=1\s*$", self.producer, re.M)
        self.assertTrue(truncations)
        self.assertEqual(len(marks), len(truncations))

    def test_abort_after_lane_start_records_a_failure(self) -> None:
        handler = self.producer.split("on_exit() {", 1)[1].split("\n}", 1)[0]
        self.assertIn('[ "$LANE_STARTED" -eq 1 ]', handler)
        self.assertIn('[ "$UMBRELLA_EMITTED" -eq 0 ]', handler)
        self.assertIn("umbrella FAIL", handler)
        self.assertIn("record_umbrella", handler)


class WorkflowBindingTest(unittest.TestCase):
    def test_nightly_workflow_runs_this_contract(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("python3 tests/release/test_fuzz_producer.py", workflow)


if __name__ == "__main__":
    unittest.main()
