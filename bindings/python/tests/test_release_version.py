"""Release-version contract tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_version_manifest_matches_every_release_surface() -> None:
    """VERSION.txt must agree with all native, Python, Rust, and npm metadata."""
    version = (ROOT / "VERSION.txt").read_text().strip()
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/version_tool.py"), "check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"semver={version}" in result.stdout
