#!/usr/bin/env python3
"""Rehydrate only certificate-declared candidate/portability bindings.

The evidence branch intentionally omits the large release artifacts and
portability bundles.  This command restores those exact bytes from downloaded
candidate-run artifacts before the strict certificate validator runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile


MAX_REHYDRATED_BYTES = 2 * 1024 * 1024 * 1024


class MaterializationError(RuntimeError):
    """Raised when a declared certificate binding cannot be restored exactly."""


def _read_certificate(path: Path) -> dict[str, object]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise MaterializationError(f"certificate contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"cannot read release certificate: {exc}") from exc
    if not isinstance(value, dict) or value.get("version") != "1.2.1":
        raise MaterializationError("certificate is not strict v1.2.1")
    return value


def _binding(value: object, label: str) -> tuple[str, int, str]:
    if not isinstance(value, dict):
        raise MaterializationError(f"{label} is not a binding")
    raw_path = value.get("path")
    size = value.get("size_bytes")
    digest = value.get("sha256")
    pure = PurePosixPath(raw_path) if isinstance(raw_path, str) else PurePosixPath("")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or pure.is_absolute()
        or ".." in pure.parts
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or size > MAX_REHYDRATED_BYTES
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise MaterializationError(f"{label} binding is invalid")
    return pure.as_posix(), size, digest


def rehydratable_bindings(document: dict[str, object]) -> dict[str, tuple[int, str]]:
    bindings: dict[str, tuple[int, str]] = {}
    values: list[tuple[str, object]] = []
    artifacts = document.get("release_artifacts")
    if not isinstance(artifacts, list):
        raise MaterializationError("certificate release_artifacts is missing")
    values.extend(
        (f"release artifact {index}", item.get("file"))
        for index, item in enumerate(artifacts)
        if isinstance(item, dict)
    )
    portability = document.get("portability")
    if not isinstance(portability, dict):
        raise MaterializationError("certificate portability is missing")
    values.append(("portability aggregate", portability.get("aggregate")))
    bundles = portability.get("bundles")
    if not isinstance(bundles, list):
        raise MaterializationError("certificate portability bundles are missing")
    values.extend(
        (f"portability bundle {index}", item.get("file"))
        for index, item in enumerate(bundles)
        if isinstance(item, dict)
    )
    for label, value in values:
        path, size, digest = _binding(value, label)
        prior = bindings.get(path)
        if prior is not None and prior != (size, digest):
            raise MaterializationError(f"certificate has conflicting binding for {path}")
        bindings[path] = (size, digest)
    if len(bindings) != 35:
        raise MaterializationError("certificate must declare exactly 24 release and 11 portability bindings")
    return bindings


def _source_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_symlink():
            raise MaterializationError(f"materializer source must be a real directory: {root}")
        root = root.resolve()
        if not root.is_dir() or root.is_symlink():
            raise MaterializationError(f"materializer source must be a real directory: {root}")
        for current, directories, names in os.walk(root, followlinks=False):
            current_path = Path(current)
            for name in directories:
                path = current_path / name
                if path.is_symlink() or not stat.S_ISDIR(path.lstat().st_mode):
                    raise MaterializationError(f"materializer source contains a symlink directory: {path}")
            for name in names:
                path = current_path / name
                if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
                    raise MaterializationError(f"materializer source contains a non-file: {path}")
                files.append(path)
    return files


def _digest(path: Path) -> tuple[int, str]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > MAX_REHYDRATED_BYTES:
        raise MaterializationError(f"candidate source is not a bounded regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise MaterializationError(f"candidate source changed while hashing: {path}")
    return before.st_size, digest.hexdigest()


def _copy_exact(source: Path, destination: Path, expected_size: int, expected_digest: str) -> None:
    if destination.exists() or destination.is_symlink():
        raise MaterializationError(f"certificate destination already exists: {destination}")
    actual_size, actual_digest = _digest(source)
    if (actual_size, actual_digest) != (expected_size, expected_digest):
        raise MaterializationError(f"candidate basename exists but its exact binding does not match: {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            shutil.copyfileobj(input_file, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        copied_size, copied_digest = _digest(temporary)
        if (copied_size, copied_digest) != (expected_size, expected_digest):
            raise MaterializationError(f"candidate bytes changed while materializing {source.name}")
        source_size, source_digest = _digest(source)
        if (source_size, source_digest) != (expected_size, expected_digest):
            raise MaterializationError(f"candidate bytes changed while materializing {source.name}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def materialize(certificate: Path, roots: list[Path]) -> list[Path]:
    if certificate.is_symlink():
        raise MaterializationError(f"certificate must be a real file: {certificate}")
    certificate = certificate.resolve()
    if not certificate.is_file():
        raise MaterializationError(f"certificate must be a real file: {certificate}")
    certificate_root = certificate.parent
    bindings = rehydratable_bindings(_read_certificate(certificate))
    files = _source_files(roots)
    by_name: dict[str, list[Path]] = {}
    for path in files:
        by_name.setdefault(path.name, []).append(path)
    materialized: list[Path] = []
    for relative, (size, digest) in sorted(bindings.items()):
        destination = certificate_root / Path(*PurePosixPath(relative).parts)
        resolved = destination.resolve()
        if resolved != certificate_root and certificate_root not in resolved.parents:
            raise MaterializationError(f"certificate binding escapes its root: {relative}")
        if destination.exists() or destination.is_symlink():
            raise MaterializationError(f"rehydratable binding is unexpectedly present in thin evidence: {relative}")
        candidates = by_name.get(PurePosixPath(relative).name, [])
        if len(candidates) != 1:
            raise MaterializationError(
                f"candidate evidence basename is ambiguous or missing for {relative}: {candidates}"
            )
        _copy_exact(candidates[0], destination, size, digest)
        materialized.append(destination)
    return materialized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    arguments = parser.parse_args(argv)
    try:
        materialized = materialize(arguments.certificate, arguments.source)
    except (MaterializationError, OSError) as exc:
        print(f"moonlab-release-evidence-materializer: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"value": "PASS", "materialized_count": len(materialized)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
