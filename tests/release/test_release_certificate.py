#!/usr/bin/env python3
"""Adversarial tests for the Moonlab release-certificate trust boundary."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_release_certificate import (  # noqa: E402
    EXPECTED_EVENTS,
    PORTABILITY_ARTIFACT_FILES,
    RELEASE_ARTIFACT_SPECS,
    REQUIRED_HOSTED_RAW_JOBS,
    REQUIRED_RELEASE_ARTIFACT_KINDS,
    REQUIRED_RUNTIME_KINDS,
    CertificateError,
    source_identity,
    validate_certificate,
)


def _run(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _binding(path: Path, root: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


class CertificateFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name) / "repo"
        self.repo.mkdir()
        _run(self.repo, "init", "-q")
        _run(self.repo, "config", "user.email", "release-test@example.invalid")
        _run(self.repo, "config", "user.name", "Release Test")
        (self.repo / "README.md").write_text("Moonlab release fixture\n", encoding="utf-8")
        _run(self.repo, "add", "README.md")
        _run(self.repo, "commit", "-q", "-m", "fixture")
        candidate_head = _run(self.repo, "rev-parse", "HEAD")
        _run(
            self.repo,
            "tag",
            "-a",
            "v1.2.0",
            "-m",
            "Moonlab v1.2.0\n\nMoonlab-Release-Candidate-Run: 12345\n"
            f"Moonlab-Release-Candidate-Head: {candidate_head}",
        )
        self.evidence_root = self.repo / "scripts/icc_traces/release-certificate"
        self.evidence_root.mkdir(parents=True)
        self.certificate_path = self.evidence_root / "certificate.json"
        self.source = source_identity(self.repo)
        self.icc_index = self.evidence_root / "icc-index.json"
        icc_files = [{
            "path": "README.md",
            "size_bytes": len("Moonlab release fixture\n"),
            "sha1": hashlib.sha1(b"Moonlab release fixture\n").hexdigest(),
            "skipped": None,
            "temporal_class": "source",
        }]
        fingerprint_rows = [
            {key: item.get(key) for key in ("path", "size_bytes", "sha1", "skipped")}
            for item in icc_files
        ]
        icc_fingerprint = hashlib.sha256(
            json.dumps(fingerprint_rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.icc_index.write_text(
            json.dumps(
                {
                    "source_fingerprint": icc_fingerprint,
                    "git_head_sha": self.source["git_head"],
                    "files": icc_files,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.icc_drift = self.evidence_root / "icc-source-drift.json"
        self.icc_drift.write_text(
            json.dumps({
                "ok": True,
                "repo": "moonlab",
                "index": str(self.icc_index),
                "stale": False,
                "staleness": {
                    "index_git_sha": self.source["git_head"],
                    "current_git_sha": self.source["git_head"],
                    "git_sha_stale": False,
                    "index_source_fingerprint": "e" * 64,
                    "current_source_fingerprint": "e" * 64,
                    "source_fingerprint_stale": False,
                    "is_stale": False,
                },
                "summary": {
                    "changed_file_count": 0,
                    "added_file_count": 0,
                    "modified_file_count": 0,
                    "deleted_file_count": 0,
                },
            }, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.aggregate_path, bundles = self._portability()
        runtime = [self._runtime_entry(kind) for kind in sorted(REQUIRED_RUNTIME_KINDS)]
        mpi = next(item for item in runtime if item["kind"] == "mpi")
        mpi_hash = mpi["artifacts"][0]["file"]["sha256"]
        mpi["assertions"] = {
            "expected_event_count": 1,
            "expected_event_names": ["mpi_sharded_gpu_works"],
            "exact_n": 33,
            "routine_n": 12,
            "ranks": 4,
            "hosts": 2,
            "slots": "2+2",
            "gpu_endpoints": 2,
            "halo_swaps": 1,
            "local_qubits": 31,
            "executable_sha256": mpi_hash,
            "rank_local_executable_sha256": [mpi_hash] * 4,
        }
        hosted_run = self.evidence_root / "github-run.json"
        hosted_run.write_text(json.dumps({
            "databaseId": 12345,
            "url": "https://github.com/tsotchke/moonlab/actions/runs/12345",
            "headSha": self.source["git_head"],
            "conclusion": "success",
            "event": "workflow_dispatch",
            "workflowName": "Release",
            "jobs": [
                {"name": name, "conclusion": "success"}
                for name in sorted(REQUIRED_HOSTED_RAW_JOBS)
            ],
        }, sort_keys=True) + "\n", encoding="utf-8")
        self.document = {
            "schema": "moonlab.release_certificate.v1",
            "version": "1.2.0",
            "generated_at": "2026-07-19T01:00:00Z",
            "source": self.source,
            "icc": {
                "index": _binding(self.icc_index, self.evidence_root),
                "source_drift": _binding(self.icc_drift, self.evidence_root),
            },
            "portability": {
                "aggregate": _binding(self.aggregate_path, self.evidence_root),
                "bundles": bundles,
            },
            "runtime_evidence": runtime,
            "release_artifacts": self._release_artifacts(),
            "mesh": self._mesh(),
            "hosted_ci": {"run": _binding(hosted_run, self.evidence_root)},
            "tag": {
                "name": "v1.2.0",
                "annotated": True,
                "object": _run(self.repo, "rev-parse", "refs/tags/v1.2.0"),
                "target": self.source["git_head"],
                "candidate_run_id": 12345,
                "candidate_head": self.source["git_head"],
            },
        }
        self.write()

    def close(self) -> None:
        self.temporary.cleanup()

    def write(self) -> None:
        self.certificate_path.write_text(
            json.dumps(self.document, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def rebind(self, binding: dict, path: Path) -> None:
        binding.clear()
        binding.update(_binding(path, self.evidence_root))

    def _portability(self) -> tuple[Path, list[dict]]:
        profiles = json.loads(
            (ROOT / "release/linux_portability_profiles.v1.json").read_text(encoding="utf-8")
        )["profiles"]
        lanes: list[dict] = []
        bundles: list[dict] = []
        for index, profile in enumerate(profiles):
            members = {
                filename: f"{profile['id']}:{name}\n".encode()
                for name, filename in PORTABILITY_ARTIFACT_FILES.items()
            }
            lane = {
                "schema": "moonlab.linux_portability.lane.v1",
                "profile": {key: profile[key] for key in ("id", "distribution", "release", "architecture")},
                "image": {"reference": profile["image"], "digest": "sha256:" + f"{index + 1:064x}"},
                "source": self.source,
                "status": "PASS",
                "tests": {"total": 2, "passed": 2, "failed": 0, "skipped": 0},
                "checks": {name: True for name in (
                    "configure", "build", "focused_correctness", "abi", "health", "install",
                    "package", "cmake_consumer", "pkg_config_consumer",
                )},
                "artifacts": [
                    {
                        "name": name,
                        "sha256": hashlib.sha256(members[filename]).hexdigest(),
                        "size_bytes": len(members[filename]),
                    }
                    for name, filename in sorted(PORTABILITY_ARTIFACT_FILES.items())
                ],
            }
            lanes.append(lane)
            members["test-results.xml"] = (
                b'<testsuite tests="2" failures="0" skipped="0">'
                b'<testcase name="health_tests"/><testcase name="abi_moonlab_export"/></testsuite>\n'
            )
            manifest_name = f"lane-{profile['id']}.json"
            members[manifest_name] = (
                json.dumps(lane, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            bundle_path = self.evidence_root / f"bundle-{profile['id']}.tar.gz"
            with tarfile.open(bundle_path, "w:gz") as archive:
                for name, data in sorted(members.items()):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(data))
            bundles.append({"profile_id": profile["id"], "file": _binding(bundle_path, self.evidence_root)})
        aggregate = {
            "schema": "moonlab.linux_portability.aggregate.v1",
            "status": "PASS",
            "profile_set_sha256": "e" * 64,
            "profile_count": 10,
            "profiles_passed": 10,
            "failed": 0,
            "skipped": 0,
            "source_identical": True,
            "image_digests_resolved": True,
            "install_consumers_passed": True,
            "native_cuda_mpi_separate": True,
            "source": self.source,
            "lanes": lanes,
        }
        path = self.evidence_root / "portability-aggregate.json"
        path.write_text(json.dumps(aggregate, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        return path, bundles

    def _runtime_entry(self, kind: str) -> dict:
        artifact = self.evidence_root / f"{kind}.artifact"
        artifact.write_bytes(f"{kind} artifact\n".encode())
        artifact_binding = _binding(artifact, self.evidence_root)
        manifest = self.evidence_root / f"{kind}.jsonl"
        records = []
        for name in sorted(EXPECTED_EVENTS[kind]):
            record = {
                    "kind": kind,
                    "name": name,
                    "value": "PASS",
                    **self.source,
                    "artifact_sha256": artifact_binding["sha256"],
                }
            if kind == "mpi":
                record.update({
                    "n": 33,
                    "routine_n": 12,
                    "ranks": 4,
                    "hosts": 2,
                    "host_slots_2x2": 1,
                    "gpu_endpoints": 2,
                    "halo_swaps": 1,
                    "local_qubits": 31,
                    "executable_sha256": artifact_binding["sha256"],
                    "rank_hash_count": 4,
                    "rank_hash_host_count": 2,
                    "rank_local_executable_sha256": [artifact_binding["sha256"]] * 4,
                })
            if kind == "mesh":
                record.update({
                    "source_snapshot_sha256": artifact_binding["sha256"],
                    "expected_target_count": 5,
                    "target_count": 5,
                    "targets": [
                        {
                            "target": target,
                            "status": "PASS",
                            "log": f"{target}.log",
                            "log_sha256": artifact_binding["sha256"],
                        }
                        for target in ("atlas", "enki", "xavier", "cosbox", "old-donkey")
                    ],
                })
            records.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
        manifest.write_text("\n".join(records) + "\n", encoding="utf-8")
        return {
            "kind": kind,
            "status": "PASS",
            "source": self.source,
            "manifest": _binding(manifest, self.evidence_root),
            "artifacts": [{"name": f"{kind}_artifact", "file": artifact_binding}],
            "assertions": {
                "expected_event_count": len(EXPECTED_EVENTS[kind]),
                "expected_event_names": sorted(EXPECTED_EVENTS[kind]),
            },
        }

    def _release_artifacts(self) -> list[dict]:
        artifacts = []
        for kind in sorted(REQUIRED_RELEASE_ARTIFACT_KINDS):
            platform, package, pattern = RELEASE_ARTIFACT_SPECS[kind]
            examples = {
                "wheel-linux-x64": "moonlab-1.2.0-cp311-cp311-manylinux_2_28_x86_64.whl",
                "wheel-linux-arm64": "moonlab-1.2.0-cp311-cp311-manylinux_2_28_aarch64.whl",
                "wheel-macos-arm64": "moonlab-1.2.0-cp311-cp311-macosx_11_0_arm64.whl",
                "wheel-macos-x64": "moonlab-1.2.0-cp311-cp311-macosx_10_15_x86_64.whl",
                "wheel-windows-x64": "moonlab-1.2.0-cp311-cp311-win_amd64.whl",
                "wheel-windows-arm64": "moonlab-1.2.0-cp311-cp311-win_arm64.whl",
            }
            filename = examples.get(kind)
            if filename is None:
                literal = pattern.pattern.removeprefix("^").removesuffix("$")
                filename = literal.replace(r"\.", ".")
            path = self.evidence_root / filename
            path.write_bytes(f"release {kind}\n".encode())
            artifacts.append({
                "kind": kind,
                "platform": platform,
                "package": package,
                "version": "1.2.0",
                "file": _binding(path, self.evidence_root),
            })
        return artifacts

    def _mesh(self) -> dict:
        entry = self._runtime_entry("mesh")
        return {
            "status": "PASS",
            "source": entry["source"],
            "manifest": entry["manifest"],
            "artifacts": entry["artifacts"],
            "nodes_total": 5,
            "nodes_passed": 5,
            "node_ids": ["atlas", "enki", "xavier", "cosbox", "old-donkey"],
        }


class ReleaseCertificateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = CertificateFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def validate(self) -> dict:
        return validate_certificate(
            self.fixture.certificate_path,
            self.fixture.repo,
            self.fixture.icc_index,
        )

    def assert_rejected(self, contains: str) -> None:
        self.fixture.write()
        with self.assertRaisesRegex(CertificateError, contains):
            self.validate()

    def test_complete_certificate_passes(self) -> None:
        document = self.validate()
        self.assertEqual(document["version"], "1.2.0")

    def test_dirty_source_is_rejected(self) -> None:
        (self.fixture.repo / "dirty.txt").write_text("dirty\n", encoding="utf-8")
        self.assert_rejected("exact current clean source")

    def test_stale_head_tree_and_fingerprint_are_rejected(self) -> None:
        for key, value in (
            ("git_head", "a" * 40),
            ("git_tree", "b" * 40),
            ("source_fingerprint", "c" * 64),
        ):
            with self.subTest(key=key):
                original = self.fixture.document["source"][key]
                self.fixture.document["source"][key] = value
                self.assert_rejected("exact current clean source")
                self.fixture.document["source"][key] = original

    def test_missing_and_hash_mismatched_artifacts_are_rejected(self) -> None:
        entry = self.fixture.document["runtime_evidence"][0]
        path = self.fixture.evidence_root / entry["artifacts"][0]["file"]["path"]
        path.unlink()
        self.assert_rejected("cannot open")

    def test_artifact_content_hash_mismatch_is_rejected(self) -> None:
        entry = self.fixture.document["runtime_evidence"][0]
        path = self.fixture.evidence_root / entry["artifacts"][0]["file"]["path"]
        path.write_bytes(b"tampered artifact\n")
        self.assert_rejected("content hash or size")

    def test_wrong_portability_lane_count_is_rejected(self) -> None:
        aggregate = json.loads(self.fixture.aggregate_path.read_text(encoding="utf-8"))
        aggregate["profile_count"] = 9
        self.fixture.aggregate_path.write_text(json.dumps(aggregate) + "\n", encoding="utf-8")
        self.fixture.rebind(self.fixture.document["portability"]["aggregate"], self.fixture.aggregate_path)
        self.assert_rejected("ten-profile")

    def test_hosted_ci_head_mismatch_is_rejected(self) -> None:
        binding = self.fixture.document["hosted_ci"]["run"]
        path = self.fixture.evidence_root / binding["path"]
        run = json.loads(path.read_text(encoding="utf-8"))
        run["headSha"] = "a" * 40
        path.write_text(json.dumps(run) + "\n", encoding="utf-8")
        self.fixture.rebind(binding, path)
        self.assert_rejected("hosted CI")

    def test_hosted_ci_repository_and_exact_job_matrix_are_enforced(self) -> None:
        binding = self.fixture.document["hosted_ci"]["run"]
        path = self.fixture.evidence_root / binding["path"]
        run = json.loads(path.read_text(encoding="utf-8"))
        run["url"] = f"https://github.com/attacker/moonlab/actions/runs/{run['databaseId']}"
        path.write_text(json.dumps(run) + "\n", encoding="utf-8")
        self.fixture.rebind(binding, path)
        self.assert_rejected("hosted CI")

        self.fixture.close()
        self.fixture = CertificateFixture()
        binding = self.fixture.document["hosted_ci"]["run"]
        path = self.fixture.evidence_root / binding["path"]
        run = json.loads(path.read_text(encoding="utf-8"))
        run["jobs"].pop()
        path.write_text(json.dumps(run) + "\n", encoding="utf-8")
        self.fixture.rebind(binding, path)
        self.assert_rejected("job matrix mismatch")

    def test_mesh_node_count_mismatch_is_rejected(self) -> None:
        self.fixture.document["mesh"]["nodes_passed"] = 4
        self.assert_rejected("mesh evidence")

    def test_mesh_self_assertion_cannot_override_manifest_targets(self) -> None:
        binding = self.fixture.document["mesh"]["manifest"]
        path = self.fixture.evidence_root / binding["path"]
        record = json.loads(path.read_text(encoding="utf-8"))
        record["targets"][0]["target"] = "imposter"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        self.fixture.rebind(binding, path)
        self.assert_rejected("node IDs")

    def test_mpi_topology_mismatch_is_rejected(self) -> None:
        mpi = next(item for item in self.fixture.document["runtime_evidence"] if item["kind"] == "mpi")
        mpi["assertions"]["ranks"] = 3
        self.assert_rejected("MPI topology")

    def test_mpi_self_assertion_cannot_override_manifest_topology(self) -> None:
        mpi = next(item for item in self.fixture.document["runtime_evidence"] if item["kind"] == "mpi")
        binding = mpi["manifest"]
        path = self.fixture.evidence_root / binding["path"]
        record = json.loads(path.read_text(encoding="utf-8"))
        record["n"] = 32
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        self.fixture.rebind(binding, path)
        self.assert_rejected("content-bound fleet manifest")

    def test_same_head_stale_icc_fingerprint_is_rejected(self) -> None:
        drift = json.loads(self.fixture.icc_drift.read_text(encoding="utf-8"))
        drift["staleness"]["current_source_fingerprint"] = "d" * 64
        drift["staleness"]["source_fingerprint_stale"] = True
        drift["staleness"]["is_stale"] = True
        drift["stale"] = True
        self.fixture.icc_drift.write_text(json.dumps(drift) + "\n", encoding="utf-8")
        self.fixture.rebind(self.fixture.document["icc"]["source_drift"], self.fixture.icc_drift)
        self.assert_rejected("ICC index/source-drift proof")

    def test_release_artifact_identity_and_filename_are_enforced(self) -> None:
        artifact = self.fixture.document["release_artifacts"][0]
        artifact["platform"] = "wrong"
        self.assert_rejected("wrong identity or filename")

    def test_tag_target_mismatch_is_rejected(self) -> None:
        self.fixture.document["tag"]["target"] = "a" * 40
        self.assert_rejected("tag assertion")

    def test_certificate_must_be_external_and_untracked(self) -> None:
        _run(
            self.fixture.repo,
            "add",
            "-f",
            self.fixture.certificate_path.relative_to(self.fixture.repo).as_posix(),
        )
        with self.assertRaisesRegex(CertificateError, "must be untracked"):
            self.validate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
