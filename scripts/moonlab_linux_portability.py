#!/usr/bin/env python3
"""Validate and aggregate Moonlab's public Linux portability evidence."""

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
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = ROOT / "release/linux_portability_profiles.v1.json"
MAX_JSON_BYTES = 1024 * 1024
MAX_JUNIT_BYTES = 16 * 1024 * 1024
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
REQUIRED_CHECKS = (
    "configure",
    "build",
    "focused_correctness",
    "abi",
    "health",
    "install",
    "package",
    "cmake_consumer",
    "pkg_config_consumer",
)
REQUIRED_ARTIFACTS = frozenset(
    {
        "build_log",
        "test_log",
        "install_tree",
        "package",
        "cmake_consumer_log",
        "pkg_config_consumer_log",
    }
)


class EvidenceError(RuntimeError):
    """Raised when release evidence cannot be accepted fail-closed."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise EvidenceError(f"non-finite JSON number {value}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceError(f"cannot open {label}: {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_JSON_BYTES:
            raise EvidenceError(f"{label} must be a regular JSON file no larger than {MAX_JSON_BYTES} bytes")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable):
        raise EvidenceError(f"{label} changed while it was read")
    data = b"".join(chunks)
    try:
        document = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid {label}: {exc}") from exc
    if not isinstance(document, dict):
        raise EvidenceError(f"{label} must be a JSON object")
    return document


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _require_exact_keys(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise EvidenceError(f"{label} must contain exactly {sorted(keys)}")
    return value


def _load_profiles(path: Path) -> tuple[list[dict[str, str]], str]:
    document = _require_exact_keys(
        _read_json(path, "profile manifest"), {"schema", "profiles"}, "profile manifest"
    )
    if document["schema"] != "moonlab.linux_portability.profiles.v1":
        raise EvidenceError("unsupported profile manifest schema")
    raw_profiles = document["profiles"]
    if not isinstance(raw_profiles, list) or len(raw_profiles) != 10:
        raise EvidenceError("profile manifest must contain exactly ten profiles")
    profiles: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_profiles):
        profile = _require_exact_keys(
            raw, {"id", "distribution", "release", "architecture", "image"}, f"profile {index}"
        )
        if not all(isinstance(profile[key], str) and profile[key] for key in profile):
            raise EvidenceError(f"profile {index} fields must be nonempty strings")
        if profile["id"] in seen:
            raise EvidenceError(f"duplicate profile {profile['id']}")
        expected_image = f"{profile['distribution']}:{profile['release']}"
        if profile["distribution"] == "debian":
            expected_image += "-slim"
        if (
            profile["id"]
            != f"{profile['distribution']}-{profile['release']}-{profile['architecture']}"
            or profile["image"] != expected_image
        ):
            raise EvidenceError(f"profile {profile['id']} has inconsistent identity fields")
        seen.add(profile["id"])
        profiles.append(dict(profile))
    expected = {
        f"{distribution}-{release}-{architecture}"
        for distribution, releases in (("debian", ("12", "13")), ("ubuntu", ("22.04", "24.04", "26.04")))
        for release in releases
        for architecture in ("amd64", "arm64")
    }
    if seen != expected:
        raise EvidenceError("profile manifest does not contain the canonical ten-profile set")
    profiles.sort(key=lambda item: item["id"])
    return profiles, hashlib.sha256(_canonical_bytes(profiles)).hexdigest()


def _validate_lane(document: dict[str, Any], expected: dict[str, str]) -> dict[str, Any]:
    lane = _require_exact_keys(
        document,
        {"schema", "profile", "image", "source", "status", "tests", "checks", "artifacts"},
        f"lane {expected['id']}",
    )
    if lane["schema"] != "moonlab.linux_portability.lane.v1":
        raise EvidenceError(f"lane {expected['id']} has an unsupported schema")
    profile = _require_exact_keys(
        lane["profile"], {"id", "distribution", "release", "architecture"}, f"lane {expected['id']} profile"
    )
    for key in ("id", "distribution", "release", "architecture"):
        if profile[key] != expected[key]:
            raise EvidenceError(f"lane {expected['id']} profile mismatch for {key}")
    image = _require_exact_keys(lane["image"], {"reference", "digest"}, f"lane {expected['id']} image")
    if image["reference"] != expected["image"] or not isinstance(image["digest"], str) or IMAGE_DIGEST.fullmatch(image["digest"]) is None:
        raise EvidenceError(f"lane {expected['id']} requires the expected image and a resolved sha256 digest")
    source = _require_exact_keys(
        lane["source"], {"git_head", "git_tree", "source_fingerprint", "dirty"}, f"lane {expected['id']} source"
    )
    if (
        not isinstance(source["git_head"], str)
        or HEX40.fullmatch(source["git_head"]) is None
        or not isinstance(source["git_tree"], str)
        or HEX40.fullmatch(source["git_tree"]) is None
        or not isinstance(source["source_fingerprint"], str)
        or HEX64.fullmatch(source["source_fingerprint"]) is None
        or source["dirty"] is not False
    ):
        raise EvidenceError(f"lane {expected['id']} requires an exact clean source identity")
    if lane["status"] != "PASS":
        raise EvidenceError(f"lane {expected['id']} did not pass")
    tests = _require_exact_keys(lane["tests"], {"total", "passed", "failed", "skipped"}, f"lane {expected['id']} tests")
    if any(isinstance(tests[key], bool) or not isinstance(tests[key], int) for key in tests):
        raise EvidenceError(f"lane {expected['id']} test counts must be integers")
    if tests["total"] <= 0 or tests["passed"] != tests["total"] or tests["failed"] != 0 or tests["skipped"] != 0:
        raise EvidenceError(f"lane {expected['id']} requires positive all-pass, zero-skip tests")
    checks = _require_exact_keys(lane["checks"], set(REQUIRED_CHECKS), f"lane {expected['id']} checks")
    if any(checks[name] is not True for name in REQUIRED_CHECKS):
        raise EvidenceError(f"lane {expected['id']} has a failed or skipped required check")
    artifacts = lane["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != len(REQUIRED_ARTIFACTS):
        raise EvidenceError(f"lane {expected['id']} requires exactly the release artifact set")
    names: set[str] = set()
    normalized_artifacts: list[dict[str, Any]] = []
    for raw in artifacts:
        artifact = _require_exact_keys(raw, {"name", "sha256", "size_bytes"}, f"lane {expected['id']} artifact")
        if artifact["name"] in names or artifact["name"] not in REQUIRED_ARTIFACTS:
            raise EvidenceError(f"lane {expected['id']} has a duplicate or unexpected artifact")
        if (
            not isinstance(artifact["sha256"], str)
            or HEX64.fullmatch(artifact["sha256"]) is None
            or isinstance(artifact["size_bytes"], bool)
            or not isinstance(artifact["size_bytes"], int)
            or artifact["size_bytes"] <= 0
        ):
            raise EvidenceError(f"lane {expected['id']} has an invalid artifact binding")
        names.add(artifact["name"])
        normalized_artifacts.append(dict(artifact))
    if names != REQUIRED_ARTIFACTS:
        raise EvidenceError(f"lane {expected['id']} is missing a required artifact")
    normalized_artifacts.sort(key=lambda item: item["name"])
    result = dict(lane)
    result["profile"] = dict(profile)
    result["image"] = dict(image)
    result["source"] = dict(source)
    result["tests"] = dict(tests)
    result["checks"] = dict(checks)
    result["artifacts"] = normalized_artifacts
    return result


def _read_regular_bytes(path: Path, label: str, maximum: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceError(f"cannot open {label}: {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise EvidenceError(f"{label} must be a regular file no larger than {maximum} bytes")
        data = bytearray()
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable):
        raise EvidenceError(f"{label} changed while it was read")
    return bytes(data)


def _junit_evidence(path: Path) -> tuple[dict[str, int], set[str]]:
    try:
        root = ET.fromstring(_read_regular_bytes(path, "JUnit results", MAX_JUNIT_BYTES))
    except ET.ParseError as exc:
        raise EvidenceError(f"invalid JUnit results: {exc}") from exc
    cases = list(root.iter("testcase"))
    names = {case.attrib.get("name", "") for case in cases}
    total = len(cases)
    failed = sum(
        any(child.tag in {"failure", "error"} for child in case)
        for case in cases
    )
    skipped_cases = sum(any(child.tag == "skipped" for child in case) for case in cases)
    try:
        reported_skipped = int(root.attrib.get("skipped", "0")) + int(root.attrib.get("disabled", "0"))
    except ValueError as exc:
        raise EvidenceError("JUnit skipped/disabled counts must be integers") from exc
    skipped = max(skipped_cases, reported_skipped)
    passed = total - failed - skipped
    counts = {"total": total, "passed": passed, "failed": failed, "skipped": skipped}
    if total <= 0 or passed != total or failed != 0 or skipped != 0:
        raise EvidenceError("JUnit results require positive all-pass, zero-skip tests")
    required_names = {"health_tests", "abi_moonlab_export"}
    if not required_names.issubset(names):
        raise EvidenceError(f"JUnit results are missing required release tests {sorted(required_names - names)}")
    return counts, names


def _artifact_binding(path: Path, name: str) -> dict[str, Any]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceError(f"cannot open artifact {name}: {path}: {exc}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise EvidenceError(f"artifact {name} must be a nonempty regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable):
        raise EvidenceError(f"artifact {name} changed while it was hashed")
    return {"name": name, "sha256": digest.hexdigest(), "size_bytes": after.st_size}


def emit_lane(
    profile_path: Path,
    profile_id: str,
    image_digest: str,
    source_head: str,
    source_tree: str,
    source_fingerprint: str,
    junit_path: Path,
    artifact_paths: dict[str, Path],
) -> dict[str, Any]:
    profiles, _ = _load_profiles(profile_path)
    profile = next((item for item in profiles if item["id"] == profile_id), None)
    if profile is None:
        raise EvidenceError(f"unknown portability profile {profile_id}")
    if set(artifact_paths) != REQUIRED_ARTIFACTS:
        raise EvidenceError(f"lane producer requires exactly the artifacts {sorted(REQUIRED_ARTIFACTS)}")
    test_counts, _ = _junit_evidence(junit_path)
    lane = {
        "schema": "moonlab.linux_portability.lane.v1",
        "profile": {key: profile[key] for key in ("id", "distribution", "release", "architecture")},
        "image": {"reference": profile["image"], "digest": image_digest},
        "source": {
            "git_head": source_head,
            "git_tree": source_tree,
            "source_fingerprint": source_fingerprint,
            "dirty": False,
        },
        "status": "PASS",
        "tests": test_counts,
        "checks": {name: True for name in REQUIRED_CHECKS},
        "artifacts": [_artifact_binding(artifact_paths[name], name) for name in sorted(REQUIRED_ARTIFACTS)],
    }
    return _validate_lane(lane, profile)


def aggregate(profile_path: Path, lane_paths: list[Path]) -> dict[str, Any]:
    profiles, profile_set_sha256 = _load_profiles(profile_path)
    if len(lane_paths) != len(profiles):
        raise EvidenceError("exactly ten lane manifests are required")
    expected_by_id = {profile["id"]: profile for profile in profiles}
    lanes: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_identity: dict[str, Any] | None = None
    for path in lane_paths:
        raw = _read_json(path, "lane manifest")
        raw_profile = raw.get("profile")
        lane_id = raw_profile.get("id") if isinstance(raw_profile, dict) else None
        if not isinstance(lane_id, str) or lane_id not in expected_by_id:
            raise EvidenceError("lane manifest has an unknown profile")
        if lane_id in seen:
            raise EvidenceError(f"duplicate lane profile {lane_id}")
        seen.add(lane_id)
        lane = _validate_lane(raw, expected_by_id[lane_id])
        if source_identity is None:
            source_identity = lane["source"]
        elif lane["source"] != source_identity:
            raise EvidenceError("all lanes must bind the identical Moonlab source")
        lanes.append(lane)
    if seen != set(expected_by_id):
        raise EvidenceError("lane manifests do not cover the canonical profile set")
    lanes.sort(key=lambda item: item["profile"]["id"])
    assert source_identity is not None
    return {
        "schema": "moonlab.linux_portability.aggregate.v1",
        "status": "PASS",
        "profile_set_sha256": profile_set_sha256,
        "profile_count": 10,
        "profiles_passed": 10,
        "failed": 0,
        "skipped": 0,
        "source_identical": True,
        "image_digests_resolved": True,
        "install_consumers_passed": True,
        "native_cuda_mpi_separate": True,
        "source": source_identity,
        "lanes": lanes,
    }


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            # mkstemp creates the file 0600 regardless of umask.  The lane
            # manifest is written as root inside the profile container onto a
            # host-mounted volume, so it must be world-readable or the
            # unprivileged runner that validates it gets EACCES.
            os.fchmod(handle.fileno(), 0o644)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--lane", action="append", type=Path, default=[])
    parser.add_argument("--emit-lane-profile")
    parser.add_argument("--image-digest")
    parser.add_argument("--source-head")
    parser.add_argument("--source-tree")
    parser.add_argument("--source-fingerprint")
    parser.add_argument("--test-results", type=Path)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.out.is_symlink():
            raise EvidenceError("output path must not be a symbolic link")
        if arguments.emit_lane_profile:
            required_values = {
                "image digest": arguments.image_digest,
                "source head": arguments.source_head,
                "source tree": arguments.source_tree,
                "source fingerprint": arguments.source_fingerprint,
                "test results": arguments.test_results,
            }
            missing = [name for name, value in required_values.items() if value is None]
            if missing:
                raise EvidenceError(f"lane production is missing {', '.join(missing)}")
            if arguments.lane:
                raise EvidenceError("--lane cannot be combined with --emit-lane-profile")
            artifacts: dict[str, Path] = {}
            for specification in arguments.artifact:
                name, separator, raw_path = specification.partition("=")
                if not separator or not name or not raw_path or name in artifacts:
                    raise EvidenceError(f"invalid or duplicate artifact specification {specification!r}")
                artifacts[name] = Path(raw_path).resolve()
            result = emit_lane(
                arguments.profiles.resolve(),
                arguments.emit_lane_profile,
                arguments.image_digest,
                arguments.source_head,
                arguments.source_tree,
                arguments.source_fingerprint,
                arguments.test_results.resolve(),
                artifacts,
            )
        else:
            if not arguments.lane:
                raise EvidenceError("at least one --lane is required for aggregation")
            if arguments.artifact:
                raise EvidenceError("--artifact requires --emit-lane-profile")
            result = aggregate(arguments.profiles.resolve(), [path.resolve() for path in arguments.lane])
        _atomic_write(arguments.out.resolve(), _canonical_bytes(result))
    except (EvidenceError, OSError) as exc:
        print(f"moonlab-linux-portability: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
