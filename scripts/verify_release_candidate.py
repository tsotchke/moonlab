#!/usr/bin/env python3
"""Seal or verify the exact immutable artifact set for a Moonlab release candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any

from validate_release_certificate import (
    RELEASE_ARTIFACT_SPECS,
    REQUIRED_HOSTED_RAW_JOBS,
    SHA256,
)


SCHEMA = "moonlab.release_candidate.v1"
REPOSITORY = "tsotchke/moonlab"
WORKFLOW = "Release"
OID = re.compile(r"^[0-9a-f]{40}$")
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024


class CandidateError(RuntimeError):
    """Raised when candidate artifacts are incomplete or mutable."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def _regular_file(path: Path) -> tuple[str, int]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise CandidateError(f"cannot stat candidate artifact {path}: {exc}") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
        or info.st_size > MAX_ARTIFACT_BYTES
    ):
        raise CandidateError(f"candidate artifact must be a bounded nonempty regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    after = path.lstat()
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise CandidateError(f"candidate artifact changed while hashing: {path}")
    return digest.hexdigest(), info.st_size


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise CandidateError(f"duplicate JSON key {key!r}")
        value[key] = child
    return value


def _reject_nonfinite(value: str) -> Any:
    raise CandidateError(f"non-finite JSON number {value}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CandidateError(f"{label} must be a JSON object")
    return value


def _identity(kind: str, path: Path) -> dict[str, Any]:
    platform, package, pattern = RELEASE_ARTIFACT_SPECS[kind]
    if pattern.fullmatch(path.name) is None:
        raise CandidateError(f"candidate artifact {path.name} does not match {kind}")
    sha256, size_bytes = _regular_file(path)
    return {
        "kind": kind,
        "platform": platform,
        "package": package,
        "filename": path.name,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def scan_candidate(directory: Path) -> list[dict[str, Any]]:
    try:
        root_info = directory.lstat()
    except OSError as exc:
        raise CandidateError(f"cannot stat candidate artifact directory: {exc}") from exc
    if not stat.S_ISDIR(root_info.st_mode) or directory.is_symlink():
        raise CandidateError(f"candidate artifact directory does not exist: {directory}")
    directory = directory.resolve()
    paths: list[Path] = []
    for current, raw_directories, raw_files in os.walk(directory, followlinks=False):
        current_path = Path(current)
        for name in raw_directories:
            child = current_path / name
            if child.is_symlink() or not stat.S_ISDIR(child.lstat().st_mode):
                raise CandidateError(f"candidate contains a non-directory or symlink directory: {child}")
        for name in raw_files:
            child = current_path / name
            if not stat.S_ISREG(child.lstat().st_mode):
                raise CandidateError(f"candidate contains a non-regular entry: {child}")
            paths.append(child)
    paths.sort()
    identities: list[dict[str, Any]] = []
    matched_paths: set[Path] = set()
    for kind, (_, _, pattern) in RELEASE_ARTIFACT_SPECS.items():
        matches = [path for path in paths if pattern.fullmatch(path.name)]
        if len(matches) != 1:
            raise CandidateError(
                f"candidate requires exactly one {kind} artifact; found {[path.name for path in matches]}"
            )
        resolved = matches[0].resolve()
        if resolved in matched_paths:
            raise CandidateError(f"candidate artifact is ambiguous across kinds: {matches[0].name}")
        matched_paths.add(resolved)
        identities.append(_identity(kind, matches[0]))
    unexpected = [path.relative_to(directory).as_posix() for path in paths if path.resolve() not in matched_paths]
    if unexpected:
        raise CandidateError(f"candidate contains unexpected files: {unexpected}")
    return sorted(identities, key=lambda item: item["kind"])


def _validate_run_identity(run_id: int, head: str, version: str) -> None:
    if isinstance(run_id, bool) or run_id <= 0:
        raise CandidateError("candidate run ID must be positive")
    if OID.fullmatch(head) is None:
        raise CandidateError("candidate head must be a full lowercase Git SHA")
    if version != "1.2.0":
        raise CandidateError("candidate version must be exactly 1.2.0")


def seal_candidate(
    directory: Path,
    output: Path,
    run_id: int,
    head: str,
    version: str,
) -> dict[str, Any]:
    _validate_run_identity(run_id, head, version)
    resolved_dist = directory.resolve()
    resolved_output = output.resolve()
    if resolved_output == resolved_dist or resolved_dist in resolved_output.parents:
        raise CandidateError("candidate manifest must live outside the artifact directory")
    document = {
        "schema": SCHEMA,
        "repository": REPOSITORY,
        "workflow": WORKFLOW,
        "run_id": run_id,
        "head_sha": head,
        "version": version,
        "artifacts": scan_candidate(directory),
    }
    encoded = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    output = resolved_output
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return document


def verify_candidate(
    directory: Path,
    manifest: Path,
    run_id: int,
    head: str,
    version: str,
) -> dict[str, Any]:
    _validate_run_identity(run_id, head, version)
    resolved_dist = directory.resolve()
    resolved_manifest = manifest.resolve()
    if resolved_manifest == resolved_dist or resolved_dist in resolved_manifest.parents:
        raise CandidateError("candidate manifest must live outside the artifact directory")
    document = _read_json(manifest, "candidate manifest")
    expected_keys = {"schema", "repository", "workflow", "run_id", "head_sha", "version", "artifacts"}
    if not isinstance(document, dict) or set(document) != expected_keys:
        raise CandidateError("candidate manifest has an unsupported shape")
    if (
        document["schema"] != SCHEMA
        or document["repository"] != REPOSITORY
        or document["workflow"] != WORKFLOW
        or document["run_id"] != run_id
        or document["head_sha"] != head
        or document["version"] != version
    ):
        raise CandidateError("candidate manifest run/head/version identity mismatch")
    actual = scan_candidate(directory)
    declared = document["artifacts"]
    if not isinstance(declared, list) or declared != actual:
        raise CandidateError("candidate artifact hashes or exact identities do not match the manifest")
    for artifact in declared:
        if not isinstance(artifact.get("sha256"), str) or SHA256.fullmatch(artifact["sha256"]) is None:
            raise CandidateError("candidate manifest contains an invalid artifact digest")
    return document


def verify_hosted_run(run_json: Path, run_id: int, head: str) -> dict[str, Any]:
    _validate_run_identity(run_id, head, "1.2.0")
    run = _read_json(run_json, "hosted candidate run")
    expected_keys = {
        "databaseId", "url", "headSha", "conclusion", "event", "workflowName", "jobs"
    }
    if set(run) != expected_keys:
        raise CandidateError("hosted candidate run has an unsupported shape")
    if (
        run["databaseId"] != run_id
        or run["url"] != f"https://github.com/{REPOSITORY}/actions/runs/{run_id}"
        or run["headSha"] != head
        or run["conclusion"] != "success"
        or run["event"] != "workflow_dispatch"
        or run["workflowName"] != WORKFLOW
    ):
        raise CandidateError("hosted candidate run repository/workflow/head/status mismatch")
    jobs = run["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise CandidateError("hosted candidate run is missing complete jobs JSON")
    successful: set[str] = set()
    raw_names: set[str] = set()
    for raw_job in jobs:
        if not isinstance(raw_job, dict) or set(raw_job) != {"name", "conclusion"}:
            raise CandidateError("hosted candidate job has an unsupported shape")
        name = raw_job["name"]
        conclusion = raw_job["conclusion"]
        if (
            not isinstance(name, str)
            or not name
            or name in raw_names
            or conclusion not in {"success", "skipped"}
        ):
            raise CandidateError("hosted candidate jobs are duplicated or non-successful")
        raw_names.add(name)
        if conclusion == "success":
            successful.add(name)
    if successful != REQUIRED_HOSTED_RAW_JOBS:
        raise CandidateError("hosted candidate successful job matrix is not exact")
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("seal", "verify"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--dist", type=Path, required=True)
        subparser.add_argument("--run-id", type=int, required=True)
        subparser.add_argument("--head", required=True)
        subparser.add_argument("--version", required=True)
        if name == "seal":
            subparser.add_argument("--output", type=Path, required=True)
        else:
            subparser.add_argument("--manifest", type=Path, required=True)
    run_parser = subparsers.add_parser("verify-run")
    run_parser.add_argument("--run-json", type=Path, required=True)
    run_parser.add_argument("--run-id", type=int, required=True)
    run_parser.add_argument("--head", required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "seal":
            document = seal_candidate(
                arguments.dist, arguments.output, arguments.run_id, arguments.head, arguments.version
            )
        elif arguments.command == "verify":
            document = verify_candidate(
                arguments.dist, arguments.manifest, arguments.run_id, arguments.head, arguments.version
            )
        else:
            run = verify_hosted_run(arguments.run_json, arguments.run_id, arguments.head)
            document = {"run_id": run["databaseId"], "head_sha": run["headSha"], "artifacts": []}
    except CandidateError as exc:
        print(f"moonlab-release-candidate: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "kind": "moonlab_release_candidate",
        "name": f"candidate_{arguments.command}",
        "value": "PASS",
        "run_id": document["run_id"],
        "head_sha": document["head_sha"],
        "artifact_count": len(document["artifacts"]),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
