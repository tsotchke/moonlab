#!/usr/bin/env python3
"""Static fail-closed contracts for candidate-build and release-promotion workflows."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_release_certificate import (  # noqa: E402
    REQUIRED_HOSTED_LINUX_JOBS,
    REQUIRED_HOSTED_RAW_JOBS,
)


class ReleaseWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.release_text = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        cls.linux_text = (ROOT / ".github/workflows/linux-compatibility.yml").read_text(
            encoding="utf-8"
        )
        cls.release = yaml.safe_load(cls.release_text)
        cls.linux = yaml.safe_load(cls.linux_text)

    def test_workflow_has_candidate_and_tag_promotion_entrypoints(self) -> None:
        self.assertRegex(self.release_text, r"(?m)^  workflow_dispatch:$")
        self.assertRegex(self.release_text, r"(?m)^  push:$")
        self.assertIn("candidate-seal", self.release["jobs"])
        self.assertIn("promotion-verify", self.release["jobs"])

    def test_candidate_builders_cannot_publish(self) -> None:
        self.assertEqual(self.release["permissions"]["contents"], "read")
        candidate_jobs = {
            "native-unix", "native-windows", "debian", "python-wheels",
            "npm-packages", "rust-crates", "linux-compatibility", "candidate-seal",
        }
        for name in candidate_jobs:
            with self.subTest(job=name):
                self.assertEqual(
                    self.release["jobs"][name].get("if"),
                    "github.event_name == 'workflow_dispatch'",
                )
                self.assertNotEqual(
                    self.release["jobs"][name].get("permissions", {}).get("contents"),
                    "write",
                )
        cargo_publish = [
            line.strip()
            for line in self.release_text.splitlines()
            if "cargo publish" in line and not line.lstrip().startswith("#")
        ]
        self.assertTrue(cargo_publish)
        self.assertTrue(all("--dry-run" in line or "echo" in line for line in cargo_publish))

    def test_promotion_queries_github_and_reuses_exact_candidate_bytes(self) -> None:
        promotion = self.release["jobs"]["promotion-verify"]
        rendered = str(promotion)
        for required in (
            "gh run view",
            "databaseId,url,headSha,conclusion,event,workflowName,jobs",
            "verify-run",
            "gh run download",
            "verify_release_candidate.py verify",
            "promoted-release-artifacts",
        ):
            self.assertIn(required, rendered)
        self.assertIn("steps.candidate.outputs.run-id", rendered)
        self.assertIn("github.sha", rendered)

    def test_every_external_mutation_is_downstream_of_fail_closed_readiness(self) -> None:
        jobs = self.release["jobs"]

        def needs(name: str) -> set[str]:
            raw = jobs[name].get("needs", [])
            return {raw} if isinstance(raw, str) else set(raw)

        def ancestors(name: str) -> set[str]:
            result: set[str] = set()
            pending = list(needs(name))
            while pending:
                parent = pending.pop()
                if parent not in result:
                    result.add(parent)
                    pending.extend(needs(parent))
            return result

        mutating = {
            "draft-release", "publish-python", "publish-npm", "publish-rust",
            "update-homebrew", "finalize-release",
        }
        for name in mutating:
            with self.subTest(job=name):
                self.assertIn("publication-readiness", ancestors(name))
        readiness = str(jobs["publication-readiness"])
        self.assertIn("exit 1", readiness)
        self.assertIn("prebuilt .crate", readiness)

    def test_hosted_linux_job_contract_matches_entire_live_matrix(self) -> None:
        matrix = self.linux["jobs"]["linux-matrix"]["strategy"]["matrix"]["include"]
        workflow_jobs = {
            "linux-portability-evidence-contract",
            "linux-portability-aggregate",
            *(f"linux-{entry['name']}" for entry in matrix),
        }
        self.assertEqual(workflow_jobs, set(REQUIRED_HOSTED_LINUX_JOBS))
        self.assertEqual(len(matrix), 17)

    def test_hosted_candidate_job_contract_matches_release_matrix(self) -> None:
        jobs = self.release["jobs"]
        native_unix = {
            f"native-{entry['name']}"
            for entry in jobs["native-unix"]["strategy"]["matrix"]["include"]
        }
        native_windows = {
            f"native-{entry['name']}"
            for entry in jobs["native-windows"]["strategy"]["matrix"]["include"]
        }
        debian = {
            f"debian-{entry['arch']}"
            for entry in jobs["debian"]["strategy"]["matrix"]["include"]
        }
        wheels = {
            f"wheel-{entry['name']}"
            for entry in jobs["python-wheels"]["strategy"]["matrix"]["include"]
        }
        exact = {
            "preflight", "npm-packages", "rust-crates", "candidate-seal",
            *native_unix,
            *native_windows,
            *debian,
            *wheels,
            *(f"linux-compatibility / {name}" for name in REQUIRED_HOSTED_LINUX_JOBS),
        }
        self.assertEqual(exact, set(REQUIRED_HOSTED_RAW_JOBS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
