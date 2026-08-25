#!/usr/bin/env python3
"""Fail-closed contracts for Moonlab's typed mesh file transport."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


def function_body(script: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^{re.escape(name)}\(\) \{{\n(?P<body>.*?)(?=^\}}\n)",
        script,
    )
    if match is None:
        raise AssertionError(f"missing shell function {name}")
    return match.group("body")


class MeshTransportProducerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.release_mesh = (ROOT / "scripts/run_mesh_release_smoke.sh").read_text(
            encoding="utf-8"
        )
        cls.seeded_mesh = (ROOT / "scripts/run_seeded_shots_mesh_gate.sh").read_text(
            encoding="utf-8"
        )

    def test_release_mesh_posix_uploads_use_typed_mesh_transport(self) -> None:
        upload = function_body(self.release_mesh, "mesh_upload_posix")
        runner = function_body(self.release_mesh, "run_posix_target")
        self.assertIn('"$MESH_BIN" exec "$target"', upload)
        self.assertIn("cat >", upload)
        self.assertNotRegex(runner, r"\bscp\b")
        self.assertEqual(runner.count("mesh_upload_posix"), 2)

    def test_seeded_mesh_round_trip_uses_typed_mesh_transport(self) -> None:
        upload = function_body(self.seeded_mesh, "mesh_upload_posix")
        download = function_body(self.seeded_mesh, "mesh_download_posix")
        runner = function_body(self.seeded_mesh, "run_target")
        self.assertIn('"$MESH_BIN" exec "$target"', upload)
        self.assertIn("cat >", upload)
        self.assertIn('"$MESH_BIN" exec "$target"', download)
        self.assertIn("cat ", download)
        self.assertNotRegex(runner, r"\bscp\b")
        self.assertEqual(runner.count("mesh_upload_posix"), 2)
        self.assertEqual(runner.count("mesh_download_posix"), 1)

    def test_seeded_bounded_runner_wraps_shell_helpers_as_executables(self) -> None:
        bounded = function_body(self.seeded_mesh, "run_bounded")
        self.assertIn('declare -F "${1:-}"', bounded)
        self.assertIn('export -f "${bounded_function?}" quote_sh', bounded)
        self.assertIn('set -- bash -c', bounded)


if __name__ == "__main__":
    unittest.main()
