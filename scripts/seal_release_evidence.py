#!/usr/bin/env python3
"""Seal a thin ignored release-evidence bundle into an orphan Git branch.

The command intentionally updates only the dedicated evidence ref and a
temporary index.  It never stages or modifies the release source worktree.
The 24 release artifacts and 11 portability bindings declared by the
certificate are intentionally omitted; tag-push materializes their exact
bytes from the bound candidate workflow run.
Print the two tag-message bindings after the command, then create the
annotated v1.2.1 release tag with those exact lines:

    Moonlab-Release-Evidence-Branch: release-evidence/v1.2.1
    Moonlab-Release-Evidence-Commit: <commit printed here>
    Moonlab-Release-Certificate-SHA256: <digest printed here>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile


BRANCH = "release-evidence/v1.2.1"
CERTIFICATE = "moonlab-v1.2.1-release-certificate.json"
MAX_GIT_FILE_BYTES = 90 * 1024 * 1024
MAX_GIT_TOTAL_BYTES = 500 * 1024 * 1024


class EvidenceSealError(RuntimeError):
    """Raised when an evidence bundle cannot be sealed safely."""


def _git(repo: Path, *arguments: str, env: dict[str, str] | None = None) -> str:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=merged,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "").strip()
        raise EvidenceSealError(f"git {' '.join(arguments)} failed: {detail}") from exc


def _read_certificate(path: Path) -> dict[str, object]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceSealError(f"certificate contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceSealError(f"cannot read release certificate: {exc}") from exc
    if not isinstance(value, dict) or value.get("version") != "1.2.1":
        raise EvidenceSealError("evidence bundle must contain a v1.2.1 certificate")
    return value


def _rehydratable_bindings(document: dict[str, object]) -> dict[str, tuple[int, str]]:
    bindings: dict[str, tuple[int, str]] = {}
    values: list[object] = []
    artifacts = document.get("release_artifacts")
    if isinstance(artifacts, list):
        values.extend(
            item.get("file") for item in artifacts if isinstance(item, dict)
        )
    portability = document.get("portability")
    if isinstance(portability, dict):
        values.append(portability.get("aggregate"))
        bundles = portability.get("bundles")
        if isinstance(bundles, list):
            values.extend(item.get("file") for item in bundles if isinstance(item, dict))
    for value in values:
        if not isinstance(value, dict):
            raise EvidenceSealError("certificate has a malformed rehydratable binding")
        raw_path = value.get("path")
        size = value.get("size_bytes")
        digest = value.get("sha256")
        pure = PurePosixPath(raw_path) if isinstance(raw_path, str) else PurePosixPath("")
        if (
            not isinstance(raw_path, str)
            or pure.is_absolute()
            or ".." in pure.parts
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise EvidenceSealError("certificate has an invalid rehydratable binding")
        normalized = pure.as_posix()
        if normalized in bindings and bindings[normalized] != (size, digest):
            raise EvidenceSealError(f"certificate has conflicting binding for {normalized}")
        bindings[normalized] = (size, digest)
    if len(bindings) != 35:
        raise EvidenceSealError(
            "certificate must declare exactly 24 release and 11 portability bindings"
        )
    return bindings


def _validate_bundle(bundle: Path, certificate_name: str) -> tuple[Path, dict[str, tuple[int, str]]]:
    if not bundle.is_dir() or bundle.is_symlink():
        raise EvidenceSealError(f"evidence bundle must be a real directory: {bundle}")
    certificate = bundle / certificate_name
    if not certificate.is_file() or certificate.is_symlink():
        raise EvidenceSealError(f"evidence bundle is missing {certificate_name}")
    for path in bundle.rglob("*"):
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            raise EvidenceSealError(f"evidence bundle contains a symlink or non-file: {path}")
    return certificate, _rehydratable_bindings(_read_certificate(certificate))


def _thin_bundle(
    bundle: Path,
    staging: Path,
    rehydratable: dict[str, tuple[int, str]],
) -> None:
    total = 0
    for source in sorted(bundle.rglob("*")):
        if not source.is_file() or source.is_symlink():
            continue
        relative = source.relative_to(bundle).as_posix()
        if relative in rehydratable:
            expected_size, expected_digest = rehydratable[relative]
            info = source.lstat()
            digest = hashlib.sha256()
            with source.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            after = source.lstat()
            if (
                info.st_size != expected_size
                or digest.hexdigest() != expected_digest
                or (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            ):
                raise EvidenceSealError(f"rehydratable binding changed or mismatched: {relative}")
            continue
        info = source.lstat()
        if info.st_size <= 0 or info.st_size > MAX_GIT_FILE_BYTES:
            raise EvidenceSealError(
                f"local evidence file exceeds the Git-safe size limit: {relative}"
            )
        total += info.st_size
        if total > MAX_GIT_TOTAL_BYTES:
            raise EvidenceSealError("local evidence exceeds the Git-safe total size limit")
        destination = staging / PurePosixPath(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        after = source.lstat()
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise EvidenceSealError(f"local evidence changed while it was copied: {relative}")


def seal_evidence(
    repo: Path,
    bundle: Path,
    branch: str = BRANCH,
    certificate_name: str = CERTIFICATE,
) -> tuple[str, str]:
    repo = repo.resolve()
    if bundle.is_symlink():
        raise EvidenceSealError(f"evidence bundle must be a real directory: {bundle}")
    bundle = bundle.resolve()
    if branch != BRANCH:
        raise EvidenceSealError(f"evidence branch must be exactly {BRANCH}")
    certificate, rehydratable = _validate_bundle(bundle, certificate_name)
    if _git(repo, "rev-parse", "--is-inside-work-tree") != "true":
        raise EvidenceSealError(f"not a Git worktree: {repo}")
    try:
        subprocess.run(
            ["git", "-C", str(repo), "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        pass
    else:
        raise EvidenceSealError(f"evidence branch already exists: {branch}")

    descriptor, index_name = tempfile.mkstemp(prefix="moonlab-evidence-index-")
    os.close(descriptor)
    index = Path(index_name)
    index.unlink()
    with tempfile.TemporaryDirectory(prefix="moonlab-thin-evidence-") as staging_name:
        staging = Path(staging_name)
        _thin_bundle(bundle, staging, rehydratable)
        try:
            env = {"GIT_INDEX_FILE": str(index)}
            _git(repo, "read-tree", "--empty", env=env)
            _git(
                repo,
                "--work-tree",
                str(staging),
                "add",
                "--all",
                "--",
                ".",
                env=env,
            )
            tree = _git(repo, "write-tree", env=env)
            commit = _git(
                repo,
                "commit-tree",
                tree,
                "-m",
                "Moonlab v1.2.1 release evidence",
                env={
                    **env,
                    "GIT_AUTHOR_NAME": "Moonlab Release Evidence",
                    "GIT_AUTHOR_EMAIL": "release-evidence@moonlab.invalid",
                    "GIT_COMMITTER_NAME": "Moonlab Release Evidence",
                    "GIT_COMMITTER_EMAIL": "release-evidence@moonlab.invalid",
                },
            )
            _git(repo, "update-ref", f"refs/heads/{branch}", commit)
        finally:
            index.unlink(missing_ok=True)
    digest = hashlib.sha256(certificate.read_bytes()).hexdigest()
    return commit, digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--certificate-name", default=CERTIFICATE)
    arguments = parser.parse_args(argv)
    try:
        commit, digest = seal_evidence(
            arguments.repo,
            arguments.bundle,
            arguments.branch,
            arguments.certificate_name,
        )
    except (EvidenceSealError, OSError) as exc:
        print(f"moonlab-release-evidence: {exc}", file=sys.stderr)
        return 2
    print(f"Moonlab-Release-Evidence-Branch: {BRANCH}")
    print(f"Moonlab-Release-Evidence-Commit: {commit}")
    print(f"Moonlab-Release-Certificate-SHA256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
