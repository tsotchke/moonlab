#!/usr/bin/env python3
"""Canonical exact-worktree provenance for Moonlab evidence producers.

Runtime traces are evidence about source, so ``scripts/icc_traces`` is the one
excluded subtree.  Tracked and non-ignored untracked source bytes are included;
ignored build outputs are not.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Sequence


TRACE_EXCLUDE_PATHSPEC = ":(exclude)scripts/icc_traces/**"


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def source_identity(repo_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Return the canonical identity of the current Moonlab worktree.

    The fingerprint hashes sorted relative paths, live permission modes, entry
    types, and live file/symlink bytes.  A tracked deletion is represented by a
    ``MISSING`` marker.  This intentionally differs from a Git tree hash: dirty
    and non-ignored untracked source is content-bound too.
    """

    root = Path(repo_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)

    head = _git(root, "rev-parse", "HEAD").decode().strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").decode().strip()
    raw_paths = _git(
        root,
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        ".",
        TRACE_EXCLUDE_PATHSPEC,
    )
    paths = sorted(
        path.decode("utf-8", "surrogateescape")
        for path in raw_paths.split(b"\0")
        if path
    )

    digest = hashlib.sha256()
    for relative in paths:
        path = root / relative
        relative_bytes = relative.encode("utf-8", "surrogateescape")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        try:
            info = path.lstat()
        except FileNotFoundError:
            digest.update(b"MISSING\0")
            continue

        digest.update(f"{stat.S_IMODE(info.st_mode):04o}".encode())
        digest.update(b"\0")
        if path.is_symlink():
            payload = os.readlink(path).encode("utf-8", "surrogateescape")
            digest.update(b"SYMLINK\0")
        elif path.is_file():
            payload = path.read_bytes()
            digest.update(b"FILE\0")
        else:
            payload = b""
            digest.update(b"OTHER\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    dirty = bool(
        _git(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            ".",
            TRACE_EXCLUDE_PATHSPEC,
        ).strip()
    )
    captured_at = (
        dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    return {
        "git_head": head,
        "git_tree": tree,
        "dirty": dirty,
        "source_fingerprint": digest.hexdigest(),
        "captured_at": captured_at,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Git worktree root (defaults to this script's repository)",
    )
    parser.add_argument("--pretty", action="store_true", help="indent JSON output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(
        json.dumps(
            source_identity(args.repo_root),
            sort_keys=True,
            indent=2 if args.pretty else None,
            separators=None if args.pretty else (",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
