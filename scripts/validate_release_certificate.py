#!/usr/bin/env python3
"""Produce and validate Moonlab's fail-closed v1.2.0 release certificate."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any

from moonlab_linux_portability import (
    EvidenceError as PortabilityError,
    _load_profiles,
    _validate_lane,
)
from moonlab_source_identity import source_identity as canonical_source_identity


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CERTIFICATE = (
    ROOT / "scripts/icc_traces/release-certificate/moonlab-v1.2.0-release-certificate.json"
)
DEFAULT_ICC_INDEXES = (
    ROOT.parent / "infinite_context_coder/artifacts/repos/moonlab/codebase_index.json",
    ROOT / ".icc/codebase_index.json",
)
MAX_JSON_BYTES = 64 * 1024 * 1024
OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RFC3339_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
REQUIRED_RUNTIME_KINDS = frozenset(
    {
        "release_smoke", "oracles", "fuzz", "differential", "statistical",
        "mpi", "tsan", "numerical", "scaling",
    }
)
EXPECTED_EVENTS = {
    "release_smoke": frozenset({
        "full_suite_green", "asan_ubsan_clean", "ci_all_green",
        "gpu_host_sync_contract", "tdvp_projector_splitting", "zero_phantom_api",
        "zero_odr_collision", "hidden_visibility_abi", "docs_apis_exist",
        "examples_all_build", "binding_versions_synced", "bindings_suites_green",
        "mpi_sharded_gpu_works", "qrng_certification_wired", "mlkem_official_kat",
        "all_discovered_bugs_closed", "tsan_clean", "numerical_edge_clean",
        "uninit_clean", "scaling_differential_clean", "source_identity_stable",
    }),
    "oracles": frozenset({
        "backend_differential_oracle", "gradient_oracle", "measurement_statistics_oracle",
        "edge_matrix_oracle", "property_invariants_oracle", "corpus_artifacts_validated",
    }),
    "fuzz": frozenset({
        "control_plane_protocol_fuzz", "circuit_deserialize_fuzz", "config_parse_fuzz",
        "mlkem_decode_fuzz", "entropy_input_fuzz", "abi_boundary_fuzz", "fuzz_corpus_clean",
    }),
    "differential": frozenset({
        "cross_backend_differential", "reference_oracle_agreement", "cross_binding_python",
        "cross_binding_rust", "cross_binding_js", "moonlab_differential",
        "corpus_artifacts_validated",
    }),
    "statistical": frozenset({
        "qrng_statistical_battery", "qrng_bias_positive_control", "mlkem_negative_fuzz",
        "mlkem_avalanche", "constant_time_variance", "entropy_health_rejects_bad",
    }),
    "mpi": frozenset({"mpi_sharded_gpu_works"}),
    "tsan": frozenset({
        "control_plane_steady", "entropy_pool_steady", "entropy_pool_toggle",
        "audit_buffer_mpmc", "scheduler", "clifford_measurement", "grover_gates_omp",
        "core_init_lazy_init", "control_plane_config_fields",
        "audit_buffer_destroy_deadlock", "tsan_clean",
    }),
    "numerical": frozenset({"numerical_edge_clean", "uninit_clean"}),
    "scaling": frozenset({
        "reference_oracle_selftest", "scaling_circuit_differential",
        "scaling_clifford_stabilizer", "scaling_var_d", "scaling_dmrg_tdvp",
        "scaling_differential_clean",
    }),
    "mesh": frozenset({"mesh_release_smoke_green"}),
}
REQUIRED_MESH_NODES = frozenset({"atlas", "enki", "xavier", "cosbox", "old-donkey"})
RELEASE_ARTIFACT_SPECS = {
    "native-linux-x64": ("linux-x64", "native", re.compile(r"^moonlab-v1\.2\.0-linux-x64\.tar\.gz$")),
    "native-linux-arm64": ("linux-arm64", "native", re.compile(r"^moonlab-v1\.2\.0-linux-arm64\.tar\.gz$")),
    "native-macos-arm64": ("macos-arm64", "native", re.compile(r"^moonlab-v1\.2\.0-macos-arm64\.tar\.gz$")),
    "native-macos-x64": ("macos-x64", "native", re.compile(r"^moonlab-v1\.2\.0-macos-x64\.tar\.gz$")),
    "native-windows-x64": ("windows-x64", "native", re.compile(r"^moonlab-v1\.2\.0-windows-x64\.zip$")),
    "native-windows-arm64": ("windows-arm64", "native", re.compile(r"^moonlab-v1\.2\.0-windows-arm64\.zip$")),
    "debian-amd64": ("linux-amd64", "debian", re.compile(r"^moonlab_1\.2\.0_amd64\.deb$")),
    "debian-arm64": ("linux-arm64", "debian", re.compile(r"^moonlab_1\.2\.0_arm64\.deb$")),
    "wheel-linux-x64": ("linux-x64", "moonlab", re.compile(r"^moonlab-1\.2\.0-cp311-cp311-(?:manylinux|musllinux)[A-Za-z0-9_.-]*x86_64\.whl$")),
    "wheel-linux-arm64": ("linux-arm64", "moonlab", re.compile(r"^moonlab-1\.2\.0-cp311-cp311-(?:manylinux|musllinux)[A-Za-z0-9_.-]*aarch64\.whl$")),
    "wheel-macos-arm64": ("macos-arm64", "moonlab", re.compile(r"^moonlab-1\.2\.0-cp311-cp311-macosx_[A-Za-z0-9_.-]*_arm64\.whl$")),
    "wheel-macos-x64": ("macos-x64", "moonlab", re.compile(r"^moonlab-1\.2\.0-cp311-cp311-macosx_[A-Za-z0-9_.-]*_x86_64\.whl$")),
    "wheel-windows-x64": ("windows-x64", "moonlab", re.compile(r"^moonlab-1\.2\.0-cp311-cp311-win_amd64\.whl$")),
    "wheel-windows-arm64": ("windows-arm64", "moonlab", re.compile(r"^moonlab-1\.2\.0-cp311-cp311-win_arm64\.whl$")),
    "rust-moonlab-sys": ("source", "moonlab-sys", re.compile(r"^moonlab-sys-1\.2\.0\.crate$")),
    "rust-moonlab": ("source", "moonlab", re.compile(r"^moonlab-1\.2\.0\.crate$")),
    "rust-moonlab-tui": ("source", "moonlab-tui", re.compile(r"^moonlab-tui-1\.2\.0\.crate$")),
    "npm-core": ("source", "@moonlab/quantum-core", re.compile(r"^moonlab-quantum-core-1\.2\.0\.tgz$")),
    "npm-algorithms": ("source", "@moonlab/quantum-algorithms", re.compile(r"^moonlab-quantum-algorithms-1\.2\.0\.tgz$")),
    "npm-vue": ("source", "@moonlab/quantum-vue", re.compile(r"^moonlab-quantum-vue-1\.2\.0\.tgz$")),
    "npm-viz": ("source", "@moonlab/quantum-viz", re.compile(r"^moonlab-quantum-viz-1\.2\.0\.tgz$")),
    "npm-react": ("source", "@moonlab/quantum-react", re.compile(r"^moonlab-quantum-react-1\.2\.0\.tgz$")),
}
REQUIRED_RELEASE_ARTIFACT_KINDS = frozenset(RELEASE_ARTIFACT_SPECS)
REQUIRED_HOSTED_CANDIDATE_JOBS = frozenset({
    "preflight",
    "native-linux-x64", "native-linux-arm64", "native-macos-arm64", "native-macos-x64",
    "native-windows-x64", "native-windows-arm64", "debian-amd64", "debian-arm64",
    "wheel-linux-x64", "wheel-linux-arm64", "wheel-macos-arm64", "wheel-macos-x64",
    "wheel-windows-x64", "wheel-windows-arm64", "npm-packages", "rust-crates",
    "linux-portability-evidence-contract",
    "linux-debian-12-amd64", "linux-debian-12-arm64",
    "linux-debian-13-amd64", "linux-debian-13-arm64",
    "linux-ubuntu-22.04-amd64", "linux-ubuntu-22.04-arm64",
    "linux-ubuntu-24.04-amd64", "linux-ubuntu-24.04-arm64",
    "linux-ubuntu-26.04-amd64", "linux-ubuntu-26.04-arm64",
    "linux-almalinux-8", "linux-almalinux-9", "linux-almalinux-10",
    "linux-fedora-current", "linux-arch-current", "linux-opensuse-current",
    "linux-alpine-current",
    "linux-portability-aggregate", "candidate-seal",
})
REQUIRED_HOSTED_LINUX_JOBS = frozenset(
    name for name in REQUIRED_HOSTED_CANDIDATE_JOBS if name.startswith("linux-")
)
REQUIRED_HOSTED_RAW_JOBS = frozenset(
    (f"linux-compatibility / {name}" if name in REQUIRED_HOSTED_LINUX_JOBS else name)
    for name in REQUIRED_HOSTED_CANDIDATE_JOBS
)
MAX_TAR_MEMBER_BYTES = 512 * 1024 * 1024
MAX_TAR_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
PORTABILITY_ARTIFACT_FILES = {
    "build_log": "build.log",
    "test_log": "test.log",
    "install_tree": "install-tree.tar.gz",
    "package": "package.tar.gz",
    "cmake_consumer_log": "cmake-consumer.log",
    "pkg_config_consumer_log": "pkg-config-consumer.log",
}


class CertificateError(RuntimeError):
    """Raised when a release certificate cannot be trusted."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CertificateError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise CertificateError(f"non-finite JSON number {value}")


def _decode_json(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CertificateError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CertificateError(f"{label} must be a JSON object")
    return value


def _read_regular(path: Path, label: str, maximum: int = MAX_JSON_BYTES) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CertificateError(f"cannot open {label}: {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > maximum:
            raise CertificateError(f"{label} must be a nonempty regular file no larger than {maximum} bytes")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in stable):
        raise CertificateError(f"{label} changed while it was read")
    return b"".join(chunks)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    return _decode_json(_read_regular(path, label), label)


def _exact(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise CertificateError(f"{label} must contain exactly {sorted(keys)}")
    return value


def _git(repo: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"").decode("utf-8", "replace").strip()
        raise CertificateError(f"git {' '.join(arguments)} failed: {detail}") from exc


def source_identity(repo: Path) -> dict[str, Any]:
    """Return the shared canonical identity without observation-time metadata."""
    identity = canonical_source_identity(repo)
    return {
        key: identity[key]
        for key in ("git_head", "git_tree", "dirty", "source_fingerprint")
    }


def _resolve_bound_path(certificate_path: Path, raw_path: object, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path or "\0" in raw_path:
        raise CertificateError(f"{label} path must be a nonempty string")
    pure = PurePosixPath(raw_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise CertificateError(f"{label} path must remain below the certificate directory")
    root = certificate_path.resolve().parent
    resolved = (root / Path(*pure.parts)).resolve()
    if resolved != root and root not in resolved.parents:
        raise CertificateError(f"{label} path escapes the certificate directory")
    return resolved


def _binding(
    value: object, certificate_path: Path, label: str
) -> tuple[dict[str, Any], Path, bytes]:
    binding = _exact(value, {"path", "sha256", "size_bytes"}, label)
    if not isinstance(binding["sha256"], str) or SHA256.fullmatch(binding["sha256"]) is None:
        raise CertificateError(f"{label} sha256 is invalid")
    if (
        isinstance(binding["size_bytes"], bool)
        or not isinstance(binding["size_bytes"], int)
        or binding["size_bytes"] <= 0
    ):
        raise CertificateError(f"{label} size_bytes is invalid")
    path = _resolve_bound_path(certificate_path, binding["path"], label)
    data = _read_regular(path, label, maximum=2 * 1024 * 1024 * 1024)
    if len(data) != binding["size_bytes"] or hashlib.sha256(data).hexdigest() != binding["sha256"]:
        raise CertificateError(f"{label} content hash or size does not match")
    return binding, path, data


def _source(value: object, label: str) -> dict[str, Any]:
    source = _exact(value, {"git_head", "git_tree", "dirty", "source_fingerprint"}, label)
    if (
        not isinstance(source["git_head"], str)
        or OID.fullmatch(source["git_head"]) is None
        or not isinstance(source["git_tree"], str)
        or OID.fullmatch(source["git_tree"]) is None
        or source["dirty"] is not False
        or not isinstance(source["source_fingerprint"], str)
        or SHA256.fullmatch(source["source_fingerprint"]) is None
    ):
        raise CertificateError(f"{label} requires valid Git OIDs, dirty=false, and a sha256 fingerprint")
    return source


def _assert_same_tree(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key in ("git_head", "git_tree", "dirty", "source_fingerprint"):
        if actual.get(key) != expected[key]:
            raise CertificateError(f"{label} source mismatch for {key}")


def _validate_portability(
    value: object, certificate_path: Path, expected_source: dict[str, Any]
) -> None:
    portability = _exact(value, {"aggregate", "bundles"}, "portability")
    _, _, aggregate_bytes = _binding(
        portability["aggregate"], certificate_path, "portability aggregate"
    )
    aggregate = _decode_json(aggregate_bytes, "portability aggregate")
    if (
        aggregate.get("schema") != "moonlab.linux_portability.aggregate.v1"
        or aggregate.get("status") != "PASS"
        or aggregate.get("profile_count") != 10
        or aggregate.get("profiles_passed") != 10
        or aggregate.get("failed") != 0
        or aggregate.get("skipped") != 0
        or aggregate.get("source_identical") is not True
    ):
        raise CertificateError("portability aggregate is not an exact ten-profile, zero-skip PASS")
    aggregate_source = _source(aggregate.get("source"), "portability aggregate source")
    _assert_same_tree(aggregate_source, expected_source, "portability aggregate")
    lanes = aggregate.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != 10:
        raise CertificateError("portability aggregate must contain ten lanes")
    lane_by_id: dict[str, dict[str, Any]] = {}
    profiles, _ = _load_profiles(ROOT / "release/linux_portability_profiles.v1.json")
    expected_profiles = {profile["id"]: profile for profile in profiles}
    for raw_lane in lanes:
        raw_profile = raw_lane.get("profile") if isinstance(raw_lane, dict) else None
        lane_id = raw_profile.get("id") if isinstance(raw_profile, dict) else None
        if not isinstance(lane_id, str) or lane_id not in expected_profiles or lane_id in lane_by_id:
            raise CertificateError("portability aggregate contains an unknown or duplicate lane")
        try:
            lane = _validate_lane(raw_lane, expected_profiles[lane_id])
        except PortabilityError as exc:
            raise CertificateError(str(exc)) from exc
        _assert_same_tree(lane["source"], expected_source, f"portability lane {lane_id}")
        lane_by_id[lane_id] = lane

    bundles = portability["bundles"]
    if not isinstance(bundles, list) or len(bundles) != 10:
        raise CertificateError("portability certificate must bind exactly ten lane bundles")
    seen: set[str] = set()
    for raw_bundle in bundles:
        bundle = _exact(raw_bundle, {"profile_id", "file"}, "portability bundle")
        profile_id = bundle["profile_id"]
        if not isinstance(profile_id, str) or profile_id not in lane_by_id or profile_id in seen:
            raise CertificateError("portability bundles contain an unknown or duplicate profile")
        seen.add(profile_id)
        _, _, bundle_bytes = _binding(bundle["file"], certificate_path, f"bundle {profile_id}")
        with tempfile.TemporaryFile() as handle:
            handle.write(bundle_bytes)
            handle.seek(0)
            try:
                archive = tarfile.open(fileobj=handle, mode="r:gz")
            except tarfile.TarError as exc:
                raise CertificateError(f"bundle {profile_id} is not a valid tar.gz") from exc
            with archive:
                members: dict[str, bytes] = {}
                total_size = 0
                for member in archive.getmembers():
                    name = member.name.removeprefix("./")
                    pure = PurePosixPath(name)
                    if (
                        not member.isfile()
                        or pure.is_absolute()
                        or ".." in pure.parts
                        or len(pure.parts) != 1
                        or name in members
                    ):
                        raise CertificateError(f"bundle {profile_id} has unsafe or duplicate member {name}")
                    if member.size <= 0 or member.size > MAX_TAR_MEMBER_BYTES:
                        raise CertificateError(f"bundle {profile_id} member {name} exceeds the size policy")
                    total_size += member.size
                    if total_size > MAX_TAR_TOTAL_BYTES:
                        raise CertificateError(f"bundle {profile_id} exceeds the extracted-size policy")
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise CertificateError(f"bundle {profile_id} member {name} is unreadable")
                    members[name] = extracted.read()
        manifest_name = f"lane-{profile_id}.json"
        required_members = set(PORTABILITY_ARTIFACT_FILES.values()) | {
            manifest_name,
            "test-results.xml",
        }
        if set(members) != required_members:
            raise CertificateError(f"bundle {profile_id} does not contain the exact lane artifact set")
        bundled_lane = _decode_json(members[manifest_name], f"bundle {profile_id} lane")
        try:
            bundled_lane = _validate_lane(bundled_lane, expected_profiles[profile_id])
        except PortabilityError as exc:
            raise CertificateError(str(exc)) from exc
        if bundled_lane != lane_by_id[profile_id]:
            raise CertificateError(f"bundle {profile_id} lane differs from the aggregate")
        artifacts = {item["name"]: item for item in bundled_lane["artifacts"]}
        for artifact_name, filename in PORTABILITY_ARTIFACT_FILES.items():
            data = members[filename]
            declared = artifacts[artifact_name]
            if len(data) != declared["size_bytes"] or hashlib.sha256(data).hexdigest() != declared["sha256"]:
                raise CertificateError(f"bundle {profile_id} artifact {artifact_name} hash mismatch")
    if seen != set(lane_by_id):
        raise CertificateError("portability bundles do not cover the canonical profile set")


def _json_records(data: bytes, label: str) -> list[dict[str, Any]]:
    stripped = data.strip()
    if not stripped:
        raise CertificateError(f"{label} is empty")
    if stripped.startswith(b"["):
        try:
            values = json.loads(
                stripped.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CertificateError(f"invalid {label}: {exc}") from exc
        if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
            raise CertificateError(f"{label} JSON array must contain objects")
        return values
    if stripped.startswith(b"{"):
        try:
            return [_decode_json(stripped, label)]
        except CertificateError as single_error:
            if "Extra data" not in str(single_error):
                raise
    records: list[dict[str, Any]] = []
    for number, line in enumerate(data.splitlines(), 1):
        if line.strip():
            records.append(_decode_json(line, f"{label} line {number}"))
    if not records:
        raise CertificateError(f"{label} has no records")
    return records


def _declared_hashes(value: Any) -> set[str]:
    hashes: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("sha256"):
                values = child if isinstance(child, list) else [child]
                if not values or any(
                    not isinstance(item, str) or SHA256.fullmatch(item) is None
                    for item in values
                ):
                    raise CertificateError(f"declared {key} is not a sha256 digest or digest list")
                hashes.update(values)
            else:
                hashes.update(_declared_hashes(child))
    elif isinstance(value, list):
        for child in value:
            hashes.update(_declared_hashes(child))
    return hashes


def _validate_evidence_entry(
    entry: dict[str, Any], certificate_path: Path, expected_source: dict[str, Any]
) -> list[dict[str, Any]]:
    kind = entry["kind"]
    if entry["status"] != "PASS":
        raise CertificateError(f"{kind} evidence did not pass")
    source = _source(entry["source"], f"{kind} source")
    _assert_same_tree(source, expected_source, kind)
    _, _, manifest_bytes = _binding(entry["manifest"], certificate_path, f"{kind} manifest")
    records = _json_records(manifest_bytes, f"{kind} manifest")
    declared_hashes: set[str] = set()
    event_names: list[str] = []
    for record in records:
        record_source = {
            key: record.get(key)
            for key in ("git_head", "git_tree", "dirty", "source_fingerprint")
        }
        _source(record_source, f"{kind} manifest record source")
        _assert_same_tree(record_source, expected_source, f"{kind} manifest record")
        verdicts = (record.get("status"), record.get("value"))
        if any(value in {"FAIL", "ERROR", False} for value in verdicts):
            raise CertificateError(f"{kind} manifest contains a failing record")
        if not any(value in {"PASS", True} for value in verdicts):
            raise CertificateError(f"{kind} manifest contains a non-passing record")
        if not isinstance(record.get("name"), str) or not record["name"]:
            raise CertificateError(f"{kind} manifest record is missing an event name")
        event_names.append(record["name"])
        declared_hashes.update(_declared_hashes(record))
    artifacts = entry["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise CertificateError(f"{kind} evidence must bind at least one artifact")
    bound_hashes: set[str] = set()
    names: set[str] = set()
    for raw_artifact in artifacts:
        artifact = _exact(raw_artifact, {"name", "file"}, f"{kind} artifact")
        if not isinstance(artifact["name"], str) or not artifact["name"] or artifact["name"] in names:
            raise CertificateError(f"{kind} artifact names must be unique and nonempty")
        names.add(artifact["name"])
        binding, _, _ = _binding(artifact["file"], certificate_path, f"{kind} artifact {artifact['name']}")
        bound_hashes.add(binding["sha256"])
    if not declared_hashes or not declared_hashes.issubset(bound_hashes):
        raise CertificateError(f"{kind} declared artifact hashes are not all content-verified")
    assertions = entry["assertions"]
    expected_names = EXPECTED_EVENTS[kind]
    if kind == "mpi":
        mpi = _exact(
            assertions,
            {
                "expected_event_count", "expected_event_names", "exact_n", "routine_n",
                "ranks", "hosts", "slots", "gpu_endpoints", "halo_swaps",
                "local_qubits", "executable_sha256", "rank_local_executable_sha256",
            },
            "MPI assertions",
        )
        if (
            mpi["exact_n"] != 33 or mpi["routine_n"] != 12 or mpi["ranks"] != 4
            or mpi["hosts"] != 2 or mpi["slots"] != "2+2"
            or isinstance(mpi["gpu_endpoints"], bool) or not isinstance(mpi["gpu_endpoints"], int)
            or mpi["gpu_endpoints"] < 2
            or isinstance(mpi["halo_swaps"], bool) or not isinstance(mpi["halo_swaps"], int)
            or mpi["halo_swaps"] <= 0 or mpi["local_qubits"] != 31
            or mpi["executable_sha256"] not in bound_hashes
            or not isinstance(mpi["rank_local_executable_sha256"], list)
            or len(mpi["rank_local_executable_sha256"]) != 4
            or any(value != mpi["executable_sha256"] for value in mpi["rank_local_executable_sha256"])
        ):
            raise CertificateError("MPI topology/rank-local proof is inconsistent")
        record = records[0]
        if (
            record.get("n") != mpi["exact_n"]
            or record.get("routine_n") != mpi["routine_n"]
            or record.get("ranks") != mpi["ranks"]
            or record.get("hosts") != mpi["hosts"]
            or record.get("host_slots_2x2") != 1
            or mpi["slots"] != "2+2"
            or record.get("gpu_endpoints") != mpi["gpu_endpoints"]
            or record.get("halo_swaps") != mpi["halo_swaps"]
            or record.get("local_qubits") != mpi["local_qubits"]
            or record.get("executable_sha256") != mpi["executable_sha256"]
            or record.get("rank_hash_count") != 4
            or isinstance(record.get("rank_hash_host_count"), bool)
            or not isinstance(record.get("rank_hash_host_count"), int)
            or record["rank_hash_host_count"] < 2
            or record.get("rank_local_executable_sha256")
            != mpi["rank_local_executable_sha256"]
        ):
            raise CertificateError("MPI assertions do not match the content-bound fleet manifest")
    else:
        _exact(assertions, {"expected_event_count", "expected_event_names"}, f"{kind} assertions")
    if (
        assertions["expected_event_count"] != len(expected_names)
        or assertions["expected_event_names"] != sorted(expected_names)
        or len(event_names) != len(expected_names)
        or set(event_names) != expected_names
        or len(set(event_names)) != len(event_names)
    ):
        raise CertificateError(f"{kind} event-name/count assertions do not match the canonical lane contract")
    return records


def _validate_hosted_ci(value: object, certificate_path: Path, expected_head: str) -> int:
    hosted = _exact(value, {"run"}, "hosted_ci")
    _, _, run_bytes = _binding(hosted["run"], certificate_path, "hosted CI run artifact")
    run = _exact(
        _decode_json(run_bytes, "hosted CI run artifact"),
        {"databaseId", "url", "headSha", "conclusion", "event", "workflowName", "jobs"},
        "hosted CI run artifact",
    )
    if (
        not isinstance(run["databaseId"], int) or isinstance(run["databaseId"], bool)
        or run["databaseId"] <= 0
        or run["url"] != f"https://github.com/tsotchke/moonlab/actions/runs/{run['databaseId']}"
        or run["headSha"] != expected_head or run["conclusion"] != "success"
        or run["event"] != "workflow_dispatch" or run["workflowName"] != "Release"
    ):
        raise CertificateError("hosted CI run/head/status attestation is invalid")
    jobs = run["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise CertificateError("hosted CI must attest required jobs")
    names: set[str] = set()
    for raw_job in jobs:
        job = _exact(raw_job, {"name", "conclusion"}, "hosted CI job")
        if (
            not isinstance(job["name"], str)
            or not job["name"]
            or job["name"] in names
            or job["conclusion"] not in {"success", "skipped"}
        ):
            raise CertificateError("hosted CI jobs must be unique PASS attestations")
        if job["conclusion"] == "success":
            names.add(job["name"])
    if names != REQUIRED_HOSTED_RAW_JOBS:
        missing = sorted(REQUIRED_HOSTED_RAW_JOBS - names)
        unexpected = sorted(names - REQUIRED_HOSTED_RAW_JOBS)
        raise CertificateError(
            f"hosted CI candidate job matrix mismatch; missing={missing}, unexpected={unexpected}"
        )
    return run["databaseId"]


def _validate_release_artifacts(value: object, certificate_path: Path) -> None:
    if not isinstance(value, list) or len(value) < len(REQUIRED_RELEASE_ARTIFACT_KINDS):
        raise CertificateError("release_artifacts does not cover every publishable output family")
    kinds: set[str] = set()
    for raw in value:
        artifact = _exact(raw, {"kind", "platform", "package", "version", "file"}, "release artifact")
        if (
            artifact["kind"] not in REQUIRED_RELEASE_ARTIFACT_KINDS
            or artifact["kind"] in kinds
            or not isinstance(artifact["platform"], str) or not artifact["platform"]
            or not isinstance(artifact["package"], str) or not artifact["package"]
            or artifact["version"] != "1.2.0"
        ):
            raise CertificateError("release artifact identity/version is invalid or duplicated")
        expected_platform, expected_package, filename_pattern = RELEASE_ARTIFACT_SPECS[artifact["kind"]]
        raw_file = artifact["file"]
        filename = (
            PurePosixPath(raw_file.get("path", "")).name
            if isinstance(raw_file, dict) and isinstance(raw_file.get("path"), str)
            else ""
        )
        if (
            artifact["platform"] != expected_platform
            or artifact["package"] != expected_package
            or filename_pattern.fullmatch(filename) is None
        ):
            raise CertificateError(f"release artifact {artifact['kind']} has the wrong identity or filename")
        kinds.add(artifact["kind"])
        _binding(artifact["file"], certificate_path, f"release artifact {artifact['kind']}")
    if kinds != REQUIRED_RELEASE_ARTIFACT_KINDS:
        raise CertificateError("release_artifacts must match the exact release.yml output families")


def _tag_candidate_binding(repo: Path, name: str) -> tuple[int, str]:
    message = _git(repo, "for-each-ref", "--format=%(contents)", f"refs/tags/{name}").decode(
        "utf-8", "strict"
    )
    run_values = re.findall(r"(?m)^Moonlab-Release-Candidate-Run: ([1-9][0-9]*)$", message)
    head_values = re.findall(r"(?m)^Moonlab-Release-Candidate-Head: ([0-9a-f]{40})$", message)
    if len(run_values) != 1 or len(head_values) != 1:
        raise CertificateError("release tag annotation must bind exactly one candidate run and head")
    return int(run_values[0]), head_values[0]


def _validate_tag(value: object, repo: Path, expected_head: str, expected_run_id: int) -> None:
    tag = _exact(
        value,
        {"name", "annotated", "object", "target", "candidate_run_id", "candidate_head"},
        "tag",
    )
    if (
        tag["name"] != "v1.2.0"
        or tag["annotated"] is not True
        or not isinstance(tag["object"], str)
        or OID.fullmatch(tag["object"]) is None
        or tag["target"] != expected_head
        or tag["candidate_run_id"] != expected_run_id
        or tag["candidate_head"] != expected_head
    ):
        raise CertificateError("release tag assertion is invalid")
    actual_type = _git(repo, "cat-file", "-t", f"refs/tags/{tag['name']}").decode().strip()
    actual_object = _git(repo, "rev-parse", f"refs/tags/{tag['name']}").decode().strip()
    actual_target = _git(repo, "rev-parse", f"refs/tags/{tag['name']}^{{commit}}").decode().strip()
    if actual_type != "tag" or actual_object != tag["object"] or actual_target != expected_head:
        raise CertificateError("annotated release tag object/target does not match the certificate")
    actual_run, actual_head = _tag_candidate_binding(repo, tag["name"])
    if actual_run != expected_run_id or actual_head != expected_head:
        raise CertificateError("annotated release tag candidate binding does not match hosted CI")


def _validate_icc(
    value: object,
    certificate_path: Path,
    live_index_path: Path,
    expected_head: str,
) -> None:
    icc = _exact(value, {"index", "source_drift"}, "icc")
    index_binding, _, index_bytes = _binding(icc["index"], certificate_path, "ICC index snapshot")
    live_index_bytes = _read_regular(live_index_path, "live ICC index")
    if (
        len(live_index_bytes) != index_binding["size_bytes"]
        or hashlib.sha256(live_index_bytes).hexdigest() != index_binding["sha256"]
    ):
        raise CertificateError("bound ICC index snapshot does not match the live ICC index")
    index = _decode_json(index_bytes, "ICC index snapshot")
    index_files = index.get("files")
    if not isinstance(index_files, list) or not all(isinstance(item, dict) for item in index_files):
        raise CertificateError("ICC index snapshot is missing file records")
    fingerprint_rows = [
        {
            "path": item.get("path"),
            "size_bytes": item.get("size_bytes"),
            "sha1": item.get("sha1"),
            "skipped": item.get("skipped"),
        }
        for item in index_files
        if item.get("temporal_class") != "runtime_evidence"
    ]
    fingerprint_rows.sort(key=lambda item: str(item.get("path") or ""))
    stored_index_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if index.get("source_fingerprint") != stored_index_fingerprint:
        raise CertificateError("bound ICC index source fingerprint does not match its file records")
    _, _, drift_bytes = _binding(icc["source_drift"], certificate_path, "ICC source-drift artifact")
    drift = _decode_json(drift_bytes, "ICC source-drift artifact")
    staleness = drift.get("staleness")
    summary = drift.get("summary")
    if not isinstance(staleness, dict) or not isinstance(summary, dict):
        raise CertificateError("ICC source-drift artifact is missing staleness/summary")
    index_fingerprint = staleness.get("index_source_fingerprint")
    current_fingerprint = staleness.get("current_source_fingerprint")
    if (
        drift.get("ok") is not True
        or drift.get("repo") != "moonlab"
        or not isinstance(drift.get("index"), str)
        or Path(drift["index"]).resolve() != live_index_path.resolve()
        or drift.get("stale") is not False
        or staleness.get("is_stale") is not False
        or staleness.get("git_sha_stale") is not False
        or staleness.get("source_fingerprint_stale") is not False
        or staleness.get("index_git_sha") != expected_head
        or staleness.get("current_git_sha") != expected_head
        or index.get("git_head_sha") != expected_head
        or not isinstance(index_fingerprint, str)
        or SHA256.fullmatch(index_fingerprint) is None
        or current_fingerprint != index_fingerprint
        or summary.get("changed_file_count") != 0
        or summary.get("added_file_count") != 0
        or summary.get("modified_file_count") != 0
        or summary.get("deleted_file_count") != 0
    ):
        raise CertificateError("ICC index/source-drift proof is stale or does not bind the certified head")


def validate_certificate(
    certificate_path: Path, repo: Path, icc_index_path: Path
) -> dict[str, Any]:
    certificate_path = certificate_path.resolve()
    repo = repo.resolve()
    trace_root = (repo / "scripts/icc_traces").resolve()
    if trace_root not in certificate_path.parents:
        raise CertificateError("release certificate must live below excluded scripts/icc_traces")
    try:
        certificate_relative = certificate_path.relative_to(repo).as_posix()
    except ValueError as exc:
        raise CertificateError("release certificate must live inside the Moonlab repository") from exc
    tracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", certificate_relative],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if tracked.returncode == 0:
        raise CertificateError("release certificate must be untracked to avoid source-identity circularity")
    document = _exact(
        _read_json(certificate_path, "release certificate"),
        {
            "schema",
            "version",
            "generated_at",
            "source",
            "icc",
            "portability",
            "runtime_evidence",
            "release_artifacts",
            "mesh",
            "hosted_ci",
            "tag",
        },
        "release certificate",
    )
    if document["schema"] != "moonlab.release_certificate.v1" or document["version"] != "1.2.0":
        raise CertificateError("release certificate schema/version is unsupported")
    if not isinstance(document["generated_at"], str) or RFC3339_UTC.fullmatch(document["generated_at"]) is None:
        raise CertificateError("release certificate generated_at must be UTC RFC3339 seconds")
    source = _source(document["source"], "release certificate source")
    live = source_identity(repo)
    if live != source:
        raise CertificateError("release certificate does not bind the exact current clean source")
    _validate_icc(document["icc"], certificate_path, icc_index_path, source["git_head"])
    _validate_portability(document["portability"], certificate_path, source)
    runtime = document["runtime_evidence"]
    if not isinstance(runtime, list) or len(runtime) != len(REQUIRED_RUNTIME_KINDS):
        raise CertificateError("release certificate requires the exact runtime evidence set")
    kinds: set[str] = set()
    for raw_entry in runtime:
        entry = _exact(
            raw_entry,
            {"kind", "status", "source", "manifest", "artifacts", "assertions"},
            "runtime evidence entry",
        )
        if entry["kind"] not in REQUIRED_RUNTIME_KINDS or entry["kind"] in kinds:
            raise CertificateError("runtime evidence kind is unknown or duplicated")
        kinds.add(entry["kind"])
        _validate_evidence_entry(entry, certificate_path, source)
    if kinds != REQUIRED_RUNTIME_KINDS:
        raise CertificateError("runtime evidence does not cover every required lane")
    mesh = _exact(
        document["mesh"],
        {"status", "source", "manifest", "artifacts", "nodes_total", "nodes_passed", "node_ids"},
        "mesh",
    )
    mesh_entry = {
        "kind": "mesh",
        "status": mesh["status"],
        "source": mesh["source"],
        "manifest": mesh["manifest"],
        "artifacts": mesh["artifacts"],
        "assertions": {
            "expected_event_count": 1,
            "expected_event_names": ["mesh_release_smoke_green"],
        },
    }
    mesh_records = _validate_evidence_entry(mesh_entry, certificate_path, source)
    if (
        isinstance(mesh["nodes_total"], bool)
        or not isinstance(mesh["nodes_total"], int)
        or mesh["nodes_total"] != 5 or mesh["nodes_passed"] != 5
        or not isinstance(mesh["node_ids"], list)
        or set(mesh["node_ids"]) != REQUIRED_MESH_NODES
        or len(mesh["node_ids"]) != 5
        or not all(isinstance(node, str) and node for node in mesh["node_ids"])
    ):
        raise CertificateError("mesh evidence requires atlas,enki,xavier,cosbox,old-donkey to pass")
    mesh_record = mesh_records[0]
    targets = mesh_record.get("targets")
    if (
        mesh_record.get("expected_target_count") != mesh["nodes_total"]
        or mesh_record.get("target_count") != mesh["nodes_passed"]
        or not isinstance(targets, list)
        or len(targets) != 5
    ):
        raise CertificateError("mesh assertions do not match the content-bound fleet manifest")
    target_ids: list[str] = []
    for raw_target in targets:
        if not isinstance(raw_target, dict):
            raise CertificateError("mesh manifest target records must be objects")
        target_id = raw_target.get("target")
        if (
            not isinstance(target_id, str)
            or not target_id
            or raw_target.get("status") != "PASS"
        ):
            raise CertificateError("mesh manifest target identity/status is invalid")
        target_ids.append(target_id)
    if len(set(target_ids)) != 5 or set(target_ids) != set(mesh["node_ids"]):
        raise CertificateError("mesh certificate node IDs do not match the manifest targets")
    _validate_release_artifacts(document["release_artifacts"], certificate_path)
    candidate_run_id = _validate_hosted_ci(
        document["hosted_ci"], certificate_path, source["git_head"]
    )
    _validate_tag(document["tag"], repo, source["git_head"], candidate_run_id)
    return document


def _binding_for(path: Path, certificate_path: Path) -> dict[str, Any]:
    data = _read_regular(path, "certificate input", maximum=2 * 1024 * 1024 * 1024)
    try:
        relative = path.resolve().relative_to(certificate_path.resolve().parent)
    except ValueError as exc:
        raise CertificateError("certificate inputs must be below the certificate directory") from exc
    return {"path": relative.as_posix(), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _fill_bindings(value: Any, certificate_path: Path) -> None:
    if isinstance(value, dict):
        if set(value) == {"path"}:
            path = _resolve_bound_path(certificate_path, value["path"], "draft binding")
            value.clear()
            value.update(_binding_for(path, certificate_path))
            return
        for child in value.values():
            _fill_bindings(child, certificate_path)
    elif isinstance(value, list):
        for child in value:
            _fill_bindings(child, certificate_path)


def _fill_event_assertions(document: dict[str, Any]) -> None:
    runtime = document.get("runtime_evidence")
    if not isinstance(runtime, list):
        return
    for entry in runtime:
        if not isinstance(entry, dict) or entry.get("kind") not in EXPECTED_EVENTS:
            continue
        names = sorted(EXPECTED_EVENTS[entry["kind"]])
        assertions = entry.setdefault("assertions", {})
        if isinstance(assertions, dict):
            assertions["expected_event_count"] = len(names)
            assertions["expected_event_names"] = names


def produce_certificate(
    draft_path: Path, certificate_path: Path, repo: Path, icc_index_path: Path
) -> dict[str, Any]:
    certificate_path = certificate_path.resolve()
    trace_root = (repo.resolve() / "scripts/icc_traces").resolve()
    if trace_root not in certificate_path.parents:
        raise CertificateError("release certificate must be emitted below excluded scripts/icc_traces")
    document = _read_json(draft_path, "release certificate draft")
    source = source_identity(repo)
    if source["dirty"]:
        raise CertificateError("a final release certificate cannot be produced from a dirty source")
    document["schema"] = "moonlab.release_certificate.v1"
    document["version"] = "1.2.0"
    document["generated_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    document["source"] = source
    if not isinstance(document.get("icc"), dict):
        raise CertificateError("release certificate draft must bind ICC index and source-drift artifacts")
    tag_object = _git(repo, "rev-parse", "refs/tags/v1.2.0").decode().strip()
    tag_target = _git(repo, "rev-parse", "refs/tags/v1.2.0^{commit}").decode().strip()
    document["tag"] = {
        "name": "v1.2.0",
        "annotated": True,
        "object": tag_object,
        "target": tag_target,
        "candidate_run_id": _tag_candidate_binding(repo, "v1.2.0")[0],
        "candidate_head": _tag_candidate_binding(repo, "v1.2.0")[1],
    }
    _fill_event_assertions(document)
    _fill_bindings(document, certificate_path)
    encoded = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    certificate_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{certificate_path.name}.", dir=certificate_path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, certificate_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return validate_certificate(certificate_path, repo, icc_index_path)


def _default_icc_index() -> Path:
    override = os.environ.get("MOONLAB_ICC_INDEX")
    if override:
        return Path(override)
    return next((path for path in DEFAULT_ICC_INDEXES if path.is_file()), DEFAULT_ICC_INDEXES[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", type=Path, default=DEFAULT_CERTIFICATE)
    parser.add_argument("--icc-index", type=Path, default=None)
    parser.add_argument("--emit-from", type=Path, help="Populate hashes/source/tag from a draft and emit the certificate")
    arguments = parser.parse_args(argv)
    certificate = arguments.certificate.resolve()
    icc_index = (arguments.icc_index or _default_icc_index()).resolve()
    try:
        if arguments.emit_from:
            document = produce_certificate(arguments.emit_from.resolve(), certificate, ROOT, icc_index)
        else:
            document = validate_certificate(certificate, ROOT, icc_index)
    except (CertificateError, OSError, PortabilityError) as exc:
        print(f"moonlab-release-certificate: {exc}", file=sys.stderr)
        return 2
    source = document["source"]
    print(json.dumps({
        "kind": "moonlab_release_certificate",
        "name": "release_certificate_valid",
        "value": "PASS",
        "version": document["version"],
        "git_head": source["git_head"],
        "git_tree": source["git_tree"],
        "dirty": False,
        "source_fingerprint": source["source_fingerprint"],
        "certificate_sha256": hashlib.sha256(_read_regular(certificate, "release certificate")).hexdigest(),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
