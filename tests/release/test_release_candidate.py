#!/usr/bin/env python3
"""Tests for exact-byte release candidate sealing and promotion verification."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_release_certificate import RELEASE_ARTIFACT_SPECS, REQUIRED_HOSTED_RAW_JOBS  # noqa: E402
from verify_release_candidate import (  # noqa: E402
    CandidateError,
    seal_candidate,
    verify_candidate,
    verify_hosted_run,
)


WHEEL_FILENAMES = {
    "wheel-linux-x64": "moonlab-1.2.0-cp311-cp311-manylinux_2_28_x86_64.whl",
    "wheel-linux-arm64": "moonlab-1.2.0-cp311-cp311-manylinux_2_28_aarch64.whl",
    "wheel-macos-arm64": "moonlab-1.2.0-cp311-cp311-macosx_11_0_arm64.whl",
    "wheel-macos-x64": "moonlab-1.2.0-cp311-cp311-macosx_10_15_x86_64.whl",
    "wheel-windows-x64": "moonlab-1.2.0-cp311-cp311-win_amd64.whl",
    "wheel-windows-arm64": "moonlab-1.2.0-cp311-cp311-win_arm64.whl",
}
HEAD = "a" * 40


def _filename(kind: str) -> str:
    if kind in WHEEL_FILENAMES:
        return WHEEL_FILENAMES[kind]
    pattern = RELEASE_ARTIFACT_SPECS[kind][2].pattern
    return pattern.removeprefix("^").removesuffix("$").replace(r"\.", ".")


class ReleaseCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dist = self.root / "dist"
        self.dist.mkdir()
        for kind in RELEASE_ARTIFACT_SPECS:
            path = self.dist / kind / _filename(kind)
            path.parent.mkdir()
            path.write_bytes(f"{kind}\n".encode())
        self.manifest = self.root / "candidate.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def seal(self) -> None:
        seal_candidate(self.dist, self.manifest, 12345, HEAD, "1.2.0")

    def test_complete_candidate_seals_and_verifies(self) -> None:
        self.seal()
        document = verify_candidate(self.dist, self.manifest, 12345, HEAD, "1.2.0")
        self.assertEqual(len(document["artifacts"]), 22)

    def test_tampered_artifact_is_rejected(self) -> None:
        self.seal()
        path = next(self.dist.rglob("*.deb"))
        path.write_bytes(b"tampered\n")
        with self.assertRaisesRegex(CandidateError, "hashes or exact identities"):
            verify_candidate(self.dist, self.manifest, 12345, HEAD, "1.2.0")

    def test_missing_duplicate_and_unexpected_artifacts_are_rejected(self) -> None:
        cases = ("missing", "duplicate", "unexpected")
        for case in cases:
            with self.subTest(case=case):
                self.tearDown()
                self.setUp()
                if case == "missing":
                    next(self.dist.rglob("*.deb")).unlink()
                elif case == "duplicate":
                    source = next(self.dist.rglob("*.deb"))
                    duplicate = self.dist / "duplicate" / source.name
                    duplicate.parent.mkdir()
                    duplicate.write_bytes(source.read_bytes())
                else:
                    (self.dist / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
                with self.assertRaises(CandidateError):
                    seal_candidate(self.dist, self.manifest, 12345, HEAD, "1.2.0")

    def test_run_identity_mismatch_is_rejected(self) -> None:
        self.seal()
        with self.assertRaisesRegex(CandidateError, "identity mismatch"):
            verify_candidate(self.dist, self.manifest, 12346, HEAD, "1.2.0")

    def test_manifest_inside_dist_and_symlink_directories_are_rejected(self) -> None:
        inside = self.dist / "candidate.json"
        with self.assertRaisesRegex(CandidateError, "outside"):
            verify_candidate(self.dist, inside, 12345, HEAD, "1.2.0")
        (self.root / "elsewhere").mkdir()
        (self.dist / "link").symlink_to(self.root / "elsewhere", target_is_directory=True)
        with self.assertRaisesRegex(CandidateError, "symlink"):
            seal_candidate(self.dist, self.manifest, 12345, HEAD, "1.2.0")

    def test_duplicate_manifest_keys_are_rejected(self) -> None:
        self.manifest.write_text('{"schema":"a","schema":"b"}\n', encoding="utf-8")
        with self.assertRaisesRegex(CandidateError, "duplicate JSON key"):
            verify_candidate(self.dist, self.manifest, 12345, HEAD, "1.2.0")

    def test_hosted_run_identity_and_complete_jobs_are_enforced(self) -> None:
        path = self.root / "run.json"
        run = {
            "databaseId": 12345,
            "url": "https://github.com/tsotchke/moonlab/actions/runs/12345",
            "headSha": HEAD,
            "conclusion": "success",
            "event": "workflow_dispatch",
            "workflowName": "Release",
            "jobs": [
                {"name": name, "conclusion": "success"}
                for name in sorted(REQUIRED_HOSTED_RAW_JOBS)
            ],
        }
        path.write_text(json.dumps(run) + "\n", encoding="utf-8")
        self.assertEqual(verify_hosted_run(path, 12345, HEAD)["databaseId"], 12345)
        run["jobs"].pop()
        path.write_text(json.dumps(run) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(CandidateError, "job matrix"):
            verify_hosted_run(path, 12345, HEAD)


if __name__ == "__main__":
    unittest.main(verbosity=2)
