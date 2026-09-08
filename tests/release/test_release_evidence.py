#!/usr/bin/env python3
"""Tests for the orphan release-evidence transport."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from seal_release_evidence import (  # noqa: E402
    BRANCH,
    CERTIFICATE,
    EvidenceSealError,
    seal_evidence,
)
from materialize_release_evidence import (  # noqa: E402
    MaterializationError,
    materialize,
    rehydratable_bindings,
)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


class ReleaseEvidenceTests(unittest.TestCase):
    @staticmethod
    def _certificate_document() -> dict:
        digest = hashlib.sha256(b"data").hexdigest()
        return {
            "version": "1.2.1",
            "release_artifacts": [
                {"file": {"path": f"artifact-{index}.bin", "size_bytes": 4, "sha256": digest}}
                for index in range(24)
            ],
            "portability": {
                "aggregate": {"path": "aggregate.json", "size_bytes": 4, "sha256": digest},
                "bundles": [
                    {"file": {"path": f"bundle-{index}.tar.gz", "size_bytes": 4, "sha256": digest}}
                    for index in range(10)
                ],
            },
        }

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name) / "repo"
        self.repo.mkdir()
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.name", "Evidence Test")
        _git(self.repo, "config", "user.email", "evidence@example.invalid")
        (self.repo / "source.txt").write_text("source\n", encoding="utf-8")
        _git(self.repo, "add", "source.txt")
        _git(self.repo, "commit", "-q", "-m", "source")
        self.bundle = self.repo / "ignored-evidence"
        self.bundle.mkdir()
        (self.bundle / CERTIFICATE).write_text(
            json.dumps(self._certificate_document()), encoding="utf-8"
        )
        (self.bundle / "codebase_index.json").write_bytes(b"index\n")
        for relative in rehydratable_bindings(self._certificate_document()):
            (self.bundle / relative).write_bytes(b"data")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_seal_updates_only_the_dedicated_orphan_branch(self) -> None:
        source_head = _git(self.repo, "rev-parse", "HEAD")
        commit, digest = seal_evidence(self.repo, self.bundle)
        self.assertEqual(_git(self.repo, "rev-parse", f"refs/heads/{BRANCH}"), commit)
        self.assertEqual(digest, hashlib.sha256((self.bundle / CERTIFICATE).read_bytes()).hexdigest())
        self.assertEqual(_git(self.repo, "rev-parse", "HEAD"), source_head)
        self.assertEqual(
            set(_git(self.repo, "ls-tree", "--name-only", BRANCH).splitlines()),
            {CERTIFICATE, "codebase_index.json"},
        )
        self.assertEqual(_git(self.repo, "status", "--short"), "?? ignored-evidence/")

    def test_existing_evidence_branch_is_rejected(self) -> None:
        seal_evidence(self.repo, self.bundle)
        with self.assertRaisesRegex(EvidenceSealError, "already exists"):
            seal_evidence(self.repo, self.bundle)

    def test_seal_rejects_a_symlink_bundle_root(self) -> None:
        link = self.repo / "bundle-link"
        link.symlink_to(self.bundle, target_is_directory=True)
        with self.assertRaisesRegex(EvidenceSealError, "real directory"):
            seal_evidence(self.repo, link)

    def test_materializer_rehydrates_only_exact_declared_files(self) -> None:
        certificate = self.bundle / CERTIFICATE
        # A compact synthetic certificate is enough to exercise the materializer
        # shape guard; the full 24+11 certificate is tested by the certificate
        # fixture suite.
        document = self._certificate_document()
        certificate.write_text(json.dumps(document), encoding="utf-8")
        bindings = rehydratable_bindings(document)
        source = self.repo / "downloaded"
        source.mkdir()
        for relative in bindings:
            (source / Path(relative).name).write_bytes(b"data")
        for relative in bindings:
            (self.bundle / relative).unlink(missing_ok=True)
        self.assertEqual(len(materialize(certificate, [source])), 35)

    def test_materializer_rejects_duplicate_basename_sources(self) -> None:
        certificate = self.bundle / CERTIFICATE
        document = self._certificate_document()
        certificate.write_text(json.dumps(document), encoding="utf-8")
        source = self.repo / "downloaded"
        duplicate = source / "duplicate"
        source.mkdir()
        duplicate.mkdir()
        for relative in rehydratable_bindings(document):
            (source / Path(relative).name).write_bytes(b"data")
        (duplicate / "artifact-0.bin").write_bytes(b"data")
        for relative in rehydratable_bindings(document):
            (self.bundle / relative).unlink(missing_ok=True)
        with self.assertRaisesRegex(MaterializationError, "ambiguous"):
            materialize(certificate, [source])

    def test_materializer_rejects_a_symlink_source_root(self) -> None:
        certificate = self.bundle / CERTIFICATE
        source = self.repo / "downloaded"
        source.mkdir()
        link = self.repo / "downloaded-link"
        link.symlink_to(source, target_is_directory=True)
        with self.assertRaisesRegex(MaterializationError, "real directory"):
            materialize(certificate, [link])

    def test_materializer_rejects_a_symlink_certificate(self) -> None:
        certificate = self.repo / "certificate-link.json"
        certificate.symlink_to(self.bundle / CERTIFICATE)
        with self.assertRaisesRegex(MaterializationError, "real file"):
            materialize(certificate, [self.repo / "downloaded"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
