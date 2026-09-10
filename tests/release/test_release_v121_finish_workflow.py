#!/usr/bin/env python3
"""Contract checks for the one-time v1.2.1 completion workflow.

Registry publication (PyPI, npm, crates.io) already happened outside this
workflow: PyPI and crates.io from release-v121-recovery.yml run 34363783550,
npm by hand from the exact certified tarballs. This workflow only verifies
those registries against the certified manifest and then finishes the
release (Homebrew tap, GitHub release) -- it must never publish anywhere.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/release-v121-finish.yml"
EXPECTED_JOBS = {"verify-registries", "update-homebrew", "finalize-release"}
EXPECTED_NEEDS = {
    "update-homebrew": ["verify-registries"],
    "finalize-release": ["verify-registries", "update-homebrew"],
}
FORBIDDEN_PUBLISH_SNIPPETS = (
    "twine upload",
    "twine check",
    "npm publish",
    "publish_crate_exact_bytes.py publish",
    "cargo publish",
)


class FinishWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = WORKFLOW.read_text(encoding="utf-8")
        cls.document = yaml.safe_load(cls.text)

    def test_finish_has_only_the_completion_jobs(self) -> None:
        self.assertEqual(set(self.document["jobs"]), EXPECTED_JOBS)
        for job, needs in EXPECTED_NEEDS.items():
            actual = self.document["jobs"][job]["needs"]
            self.assertEqual(actual if isinstance(actual, list) else [actual], needs, job)

    def test_dispatch_requires_fixed_confirmation(self) -> None:
        trigger = self.document["on"]
        self.assertEqual(
            trigger["workflow_dispatch"]["inputs"]["confirmation"]["options"],
            ["finish-v1.2.1"],
        )
        self.assertEqual(
            self.document["jobs"]["verify-registries"]["if"],
            "inputs.confirmation == 'finish-v1.2.1'",
        )

    def test_all_source_checkouts_pin_the_release_tag(self) -> None:
        for job in self.document["jobs"].values():
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("actions/checkout@"):
                    self.assertEqual(step.get("with", {}).get("ref"), "refs/tags/v1.2.1")

    def test_verify_registries_pins_the_certified_commit(self) -> None:
        self.assertGreaterEqual(
            self.text.count("a15dabcf79a93b195ed062c6ed338cb4bc2c733b"), 4
        )

    def test_verify_registries_resolves_the_manifest_via_the_tag_binding(self) -> None:
        self.assertIn("_tag_candidate_binding", self.text)
        self.assertIn("verify_release_candidate.py verify-run", self.text)
        self.assertIn("release-candidate-manifest", self.text)

    def test_verify_registries_checks_pypi_crates_and_npm(self) -> None:
        self.assertIn("pypi.org/pypi/moonlab/1.2.1/json", self.text)
        self.assertIn("crates.io/api/v1/crates/", self.text)
        self.assertIn("registry.npmjs.org", self.text)
        self.assertIn("wheel-", self.text)
        for crate_name in ("moonlab-sys", "moonlab-tui"):
            self.assertIn(crate_name, self.text)
        for package_name in (
            "@tsotchkecorp/moonlab",
            "@tsotchkecorp/moonlab-algorithms",
            "@tsotchkecorp/moonlab-viz",
            "@tsotchkecorp/moonlab-react",
            "@tsotchkecorp/moonlab-vue",
        ):
            self.assertIn(package_name, self.text)

    def test_workflow_never_publishes_to_any_registry(self) -> None:
        for snippet in FORBIDDEN_PUBLISH_SNIPPETS:
            self.assertNotIn(snippet, self.text)

    def test_update_homebrew_is_copied_from_the_recovery_workflow(self) -> None:
        job = self.document["jobs"]["update-homebrew"]
        self.assertEqual(job["runs-on"], "macos-15")
        step_names = [step.get("name") for step in job["steps"]]
        self.assertIn("Materialize stable formula", step_names)
        self.assertIn("Prepare tap checkout", step_names)
        self.assertIn("Audit and install formula from source", step_names)
        self.assertIn("Push formula to tap", step_names)
        self.assertIn("HOMEBREW_TAP_TOKEN", self.text)

    def test_finalize_release_publishes_the_draft_as_latest(self) -> None:
        job = self.document["jobs"]["finalize-release"]
        self.assertEqual(job["permissions"]["contents"], "write")
        self.assertIn(
            'gh release edit "v1.2.1" --repo "${{ github.repository }}" --draft=false --latest',
            self.text,
        )

    def test_concurrency_matches_the_recovery_workflow_convention(self) -> None:
        concurrency = self.document["concurrency"]
        self.assertEqual(concurrency["group"], "release-refs/tags/v1.2.1")
        self.assertFalse(concurrency["cancel-in-progress"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
