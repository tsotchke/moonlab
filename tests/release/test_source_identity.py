#!/usr/bin/env python3
"""Focused contract tests for scripts/moonlab_source_identity.py."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = REPO_ROOT / "scripts" / "moonlab_source_identity.py"
SPEC = importlib.util.spec_from_file_location("moonlab_source_identity", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


class SourceIdentityContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="moonlab-source-identity-")
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.email", "source-identity@test.invalid")
        self.git("config", "user.name", "Source Identity Test")
        self.git("config", "core.filemode", "true")
        (self.root / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
        (self.root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
        trace_dir = self.root / "scripts" / "icc_traces"
        trace_dir.mkdir(parents=True)
        (trace_dir / "tracked.jsonl").write_text("old trace\n", encoding="utf-8")
        os.symlink("tracked.txt", self.root / "tracked-link")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    def identity(self) -> dict[str, object]:
        return HELPER.source_identity(self.root)

    def test_clean_shape_and_cli_json(self) -> None:
        imported = self.identity()
        cli = json.loads(
            subprocess.run(
                [sys.executable, str(HELPER_PATH), "--repo-root", str(self.root)],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout
        )
        self.assertFalse(imported["dirty"])
        self.assertRegex(str(imported["git_head"]), r"^[0-9a-f]{40,64}$")
        self.assertRegex(str(imported["git_tree"]), r"^[0-9a-f]{40,64}$")
        self.assertRegex(str(imported["source_fingerprint"]), r"^[0-9a-f]{64}$")
        for field in ("git_head", "git_tree", "dirty", "source_fingerprint"):
            self.assertEqual(imported[field], cli[field])

    def test_live_mode_is_content_bound_and_dirty(self) -> None:
        before = self.identity()
        tracked = self.root / "tracked.txt"
        tracked.chmod(stat.S_IMODE(tracked.stat().st_mode) | stat.S_IXUSR)
        after = self.identity()
        self.assertNotEqual(before["source_fingerprint"], after["source_fingerprint"])
        self.assertTrue(after["dirty"])

    def test_symlink_target_is_content_bound(self) -> None:
        before = self.identity()
        link = self.root / "tracked-link"
        link.unlink()
        os.symlink("missing-target", link)
        after = self.identity()
        self.assertNotEqual(before["source_fingerprint"], after["source_fingerprint"])
        self.assertTrue(after["dirty"])

    def test_untracked_source_is_included_and_dirty(self) -> None:
        before = self.identity()
        (self.root / "new-source.c").write_text("int new_source;\n", encoding="utf-8")
        after = self.identity()
        self.assertNotEqual(before["source_fingerprint"], after["source_fingerprint"])
        self.assertTrue(after["dirty"])

    def test_ignored_file_is_excluded(self) -> None:
        before = self.identity()
        (self.root / "ignored.txt").write_text("ignored build output\n", encoding="utf-8")
        after = self.identity()
        self.assertEqual(before["source_fingerprint"], after["source_fingerprint"])
        self.assertFalse(after["dirty"])

    def test_trace_directory_is_excluded_even_when_tracked_or_untracked(self) -> None:
        before = self.identity()
        trace_dir = self.root / "scripts" / "icc_traces"
        (trace_dir / "tracked.jsonl").write_text("new trace bytes\n", encoding="utf-8")
        (trace_dir / "untracked.jsonl").write_text("runtime only\n", encoding="utf-8")
        after = self.identity()
        self.assertEqual(before["source_fingerprint"], after["source_fingerprint"])
        self.assertFalse(after["dirty"])

    def test_tracked_content_change_sets_dirty(self) -> None:
        before = self.identity()
        (self.root / "tracked.txt").write_text("changed\n", encoding="utf-8")
        after = self.identity()
        self.assertNotEqual(before["source_fingerprint"], after["source_fingerprint"])
        self.assertTrue(after["dirty"])


if __name__ == "__main__":
    unittest.main()
