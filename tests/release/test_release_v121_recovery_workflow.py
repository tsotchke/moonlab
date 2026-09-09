#!/usr/bin/env python3
"""Contract checks for the one-time v1.2.1 recovery workflow."""

from __future__ import annotations

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/release-v121-recovery.yml"
EXPECTED_JOBS = {
    "preflight", "promotion-verify", "release-certificate-verify",
    "publication-readiness", "publication-credentials", "draft-release",
    "publish-python", "publish-npm", "publish-rust", "update-homebrew",
    "finalize-release",
}
EXPECTED_NEEDS = {
    "promotion-verify": ["preflight"],
    "release-certificate-verify": ["preflight", "promotion-verify"],
    "publication-readiness": ["preflight", "promotion-verify", "release-certificate-verify"],
    "publication-credentials": ["preflight", "publication-readiness"],
    "draft-release": ["preflight", "promotion-verify", "publication-readiness", "publication-credentials"],
    "publish-python": ["preflight", "promotion-verify", "draft-release"],
    "publish-npm": ["preflight", "promotion-verify", "draft-release"],
    "publish-rust": ["preflight", "promotion-verify", "draft-release"],
    "update-homebrew": ["preflight", "draft-release", "publish-python", "publish-npm", "publish-rust"],
    "finalize-release": ["preflight", "draft-release", "publish-python", "publish-npm", "publish-rust", "update-homebrew"],
}


class RecoveryWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    def test_recovery_has_only_original_publication_jobs(self) -> None:
        self.assertEqual(set(self.document["jobs"]), EXPECTED_JOBS)
        for job, needs in EXPECTED_NEEDS.items():
            actual = self.document["jobs"][job]["needs"]
            self.assertEqual(actual if isinstance(actual, list) else [actual], needs, job)

    def test_dispatch_requires_fixed_confirmation(self) -> None:
        trigger = self.document["on"]
        self.assertEqual(trigger["workflow_dispatch"]["inputs"]["confirmation"]["options"], ["publish-v1.2.1"])
        self.assertEqual(self.document["jobs"]["preflight"]["if"], "inputs.confirmation == 'publish-v1.2.1'")

    def test_all_source_checkouts_pin_the_release_tag(self) -> None:
        for job in self.document["jobs"].values():
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("actions/checkout@"):
                    self.assertEqual(step.get("with", {}).get("ref"), "refs/tags/v1.2.1")

    def test_promotion_and_helper_contract(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertGreaterEqual(text.count("a15dabcf79a93b195ed062c6ed338cb4bc2c733b"), 4)
        self.assertIn("git show \"${{ github.sha }}:scripts/materialize_release_evidence.py\" > build-promotion/materialize_release_evidence.py", text)
        self.assertIn("build-promotion/", text)
        self.assertNotIn("candidate-seal:", text)
        self.assertNotIn("cargo package", text)
        self.assertNotIn("cmake --build", text)

    def test_manual_dispatch_draft_targets_existing_release_tag(self) -> None:
        steps = self.document["jobs"]["draft-release"]["steps"]
        draft = next(step for step in steps if step.get("uses", "").startswith("softprops/action-gh-release@"))
        self.assertEqual(draft["with"]["tag_name"], "v1.2.1")
        self.assertTrue(draft["with"]["draft"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
