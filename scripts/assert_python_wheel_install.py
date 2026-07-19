#!/usr/bin/env python3
"""Assert that a QSIM_PYTHON_WHEEL CMake install contains only its native payload."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import stat
import sys


NATIVE_LIBRARY = re.compile(r"^(?:libquantumsim\.(?:so|dylib)|quantumsim\.dll)$")


def assert_install_tree(root: Path) -> str:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"install root does not exist: {root}")
    entries = sorted(path for path in root.rglob("*") if not path.is_dir())
    if len(entries) != 1:
        raise ValueError(f"wheel CMake install must contain exactly one file, found {len(entries)}")
    payload = entries[0]
    info = payload.lstat()
    relative = payload.relative_to(root).as_posix()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
        or payload.parent.relative_to(root).as_posix() != "moonlab/.libs"
        or NATIVE_LIBRARY.fullmatch(payload.name) is None
    ):
        raise ValueError(f"unexpected wheel CMake install payload: {relative}")
    return relative


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        payload = assert_install_tree(arguments.install_root)
    except (OSError, ValueError) as exc:
        print(f"moonlab-python-wheel-install: {exc}", file=sys.stderr)
        return 2
    print(f"moonlab-python-wheel-install: PASS {payload}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
