#!/usr/bin/env python3
"""Adversarial tests for Moonlab Linux portability aggregation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/moonlab_linux_portability.py"
PROFILES = ROOT / "release/linux_portability_profiles.v1.json"
ARTIFACTS = (
    "build_log",
    "test_log",
    "install_tree",
    "package",
    "cmake_consumer_log",
    "pkg_config_consumer_log",
)
CHECKS = (
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


class PortabilityFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        profile_document = json.loads(PROFILES.read_text(encoding="utf-8"))
        self.profiles = profile_document["profiles"]
        self.lanes: list[Path] = []
        for index, profile in enumerate(self.profiles):
            lane = {
                "schema": "moonlab.linux_portability.lane.v1",
                "profile": {key: profile[key] for key in ("id", "distribution", "release", "architecture")},
                "image": {"reference": profile["image"], "digest": "sha256:" + f"{index + 1:064x}"},
                "source": {
                    "git_head": "a" * 40,
                    "git_tree": "b" * 40,
                    "source_fingerprint": "c" * 64,
                    "dirty": False,
                },
                "status": "PASS",
                "tests": {"total": 3, "passed": 3, "failed": 0, "skipped": 0},
                "checks": {name: True for name in CHECKS},
                "artifacts": [
                    {"name": name, "sha256": hashlib.sha256(f"{profile['id']}:{name}".encode()).hexdigest(), "size_bytes": 1}
                    for name in ARTIFACTS
                ],
            }
            path = self.root / f"lane-{index}.json"
            path.write_text(json.dumps(lane) + "\n", encoding="utf-8")
            self.lanes.append(path)
        self.output = self.root / "aggregate.json"

    def close(self) -> None:
        self.temporary.cleanup()

    def read_lane(self, index: int) -> dict:
        return json.loads(self.lanes[index].read_text(encoding="utf-8"))

    def write_lane(self, index: int, lane: dict) -> None:
        self.lanes[index].write_text(json.dumps(lane) + "\n", encoding="utf-8")

    def run(self, lanes: list[Path] | None = None) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(SCRIPT), "--profiles", str(PROFILES), "--out", str(self.output)]
        for lane in self.lanes if lanes is None else lanes:
            command.extend(["--lane", str(lane)])
        return subprocess.run(command, capture_output=True, text=True, check=False)


class PortabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = PortabilityFixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def assert_rejected(self) -> None:
        self.fixture.output.write_text("preserve\n", encoding="utf-8")
        result = self.fixture.run()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(self.fixture.output.read_text(encoding="utf-8"), "preserve\n")

    def test_full_matrix_passes_deterministically(self) -> None:
        first = self.fixture.run()
        self.assertEqual(first.returncode, 0, first.stderr)
        first_bytes = self.fixture.output.read_bytes()
        document = json.loads(first_bytes)
        self.assertEqual(document["status"], "PASS")
        self.assertEqual(document["profile_count"], 10)
        self.assertEqual(document["profiles_passed"], 10)
        self.assertEqual(document["failed"], 0)
        self.assertEqual(document["skipped"], 0)
        self.assertTrue(document["source_identical"])
        self.assertTrue(document["image_digests_resolved"])
        second = self.fixture.run(list(reversed(self.fixture.lanes)))
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(self.fixture.output.read_bytes(), first_bytes)

    def test_missing_duplicate_and_extra_profiles_are_rejected(self) -> None:
        for lanes in (
            self.fixture.lanes[:-1],
            self.fixture.lanes[:-1] + [self.fixture.lanes[0]],
            self.fixture.lanes + [self.fixture.lanes[0]],
        ):
            result = self.fixture.run(lanes)
            self.assertEqual(result.returncode, 2, result.stderr)

    def test_dirty_and_source_mismatch_are_rejected(self) -> None:
        lane = self.fixture.read_lane(0)
        lane["source"]["dirty"] = True
        self.fixture.write_lane(0, lane)
        self.assert_rejected()
        lane["source"]["dirty"] = False
        lane["source"]["git_tree"] = "d" * 40
        self.fixture.write_lane(0, lane)
        self.assert_rejected()

    def test_mutable_or_missing_image_digest_is_rejected(self) -> None:
        lane = self.fixture.read_lane(0)
        lane["image"]["digest"] = "latest"
        self.fixture.write_lane(0, lane)
        self.assert_rejected()

    def test_failed_or_skipped_checks_are_rejected(self) -> None:
        lane = self.fixture.read_lane(0)
        lane["checks"]["health"] = False
        self.fixture.write_lane(0, lane)
        self.assert_rejected()
        lane["checks"]["health"] = True
        lane["tests"]["skipped"] = 1
        lane["tests"]["passed"] = 2
        self.fixture.write_lane(0, lane)
        self.assert_rejected()

    def test_bad_artifact_digest_is_rejected(self) -> None:
        lane = self.fixture.read_lane(0)
        lane["artifacts"][0]["sha256"] = "not-a-digest"
        self.fixture.write_lane(0, lane)
        self.assert_rejected()

    def test_duplicate_json_key_is_rejected(self) -> None:
        self.fixture.lanes[0].write_text('{"schema":"x","schema":"y"}\n', encoding="utf-8")
        self.assert_rejected()


class LaneEmitterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "lane.json"
        self.junit = self.root / "test-results.xml"
        self.junit.write_text(
            '<testsuite tests="3" failures="0" errors="0" skipped="0" disabled="0">'
            '<testcase name="health_tests"/><testcase name="abi_moonlab_export"/>'
            '<testcase name="gate_test"/></testsuite>\n',
            encoding="utf-8",
        )
        self.artifacts: dict[str, Path] = {}
        for name in ARTIFACTS:
            path = self.root / f"{name}.artifact"
            path.write_bytes(f"{name}\n".encode())
            self.artifacts[name] = path

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def invoke_emitter(self, artifacts: dict[str, Path] | None = None) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(SCRIPT),
            "--profiles",
            str(PROFILES),
            "--emit-lane-profile",
            "debian-12-amd64",
            "--image-digest",
            "sha256:" + "d" * 64,
            "--source-head",
            "a" * 40,
            "--source-tree",
            "b" * 40,
            "--source-fingerprint",
            "c" * 64,
            "--test-results",
            str(self.junit),
            "--out",
            str(self.output),
        ]
        for name, path in (self.artifacts if artifacts is None else artifacts).items():
            command.extend(["--artifact", f"{name}={path}"])
        return subprocess.run(command, capture_output=True, text=True, check=False)

    def test_lane_emitter_binds_real_tests_and_artifacts(self) -> None:
        result = self.invoke_emitter()
        self.assertEqual(result.returncode, 0, result.stderr)
        lane = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(lane["tests"], {"total": 3, "passed": 3, "failed": 0, "skipped": 0})
        self.assertEqual({item["name"] for item in lane["artifacts"]}, set(ARTIFACTS))
        self.assertTrue(all(lane["checks"].values()))

    def test_lane_emitter_rejects_skips_and_missing_required_tests(self) -> None:
        self.junit.write_text(
            '<testsuite tests="3" failures="0" skipped="1">'
            '<testcase name="health_tests"/><testcase name="abi_moonlab_export"/>'
            '<testcase name="gate_test"><skipped/></testcase></testsuite>\n',
            encoding="utf-8",
        )
        result = self.invoke_emitter()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.output.exists())

        self.junit.write_text(
            '<testsuite tests="2" failures="0" skipped="0">'
            '<testcase name="abi_moonlab_export"/><testcase name="gate_test"/></testsuite>\n',
            encoding="utf-8",
        )
        result = self.invoke_emitter()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.output.exists())

    def test_lane_emitter_requires_the_exact_artifact_set(self) -> None:
        incomplete = dict(self.artifacts)
        incomplete.pop("pkg_config_consumer_log")
        result = self.invoke_emitter(incomplete)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.output.exists())


class WorkflowWiringTests(unittest.TestCase):
    def test_canonical_profiles_are_produced_and_aggregated_in_ci(self) -> None:
        workflow = (ROOT / ".github/workflows/linux-compatibility.yml").read_text(encoding="utf-8")
        profiles = json.loads(PROFILES.read_text(encoding="utf-8"))["profiles"]
        for profile in profiles:
            block = f"- name: {profile['id']}\n"
            self.assertEqual(workflow.count(block), 1, profile["id"])
        self.assertIn("python3 scripts/run_linux_portability_lane.py", workflow)
        self.assertIn("python3 scripts/moonlab_linux_portability.py", workflow)
        self.assertIn("pattern: linux-portability-lane-*", workflow)
        self.assertIn('[[ "${#bundles[@]}" -eq 10 ]]', workflow)
        self.assertIn("python3 tests/release/test_moonlab_linux_portability.py", workflow)

    def test_health_test_is_part_of_the_bounded_linux_label_set(self) -> None:
        registrations = (ROOT / "cmake/tests.cmake").read_text(encoding="utf-8")
        self.assertIn('set_tests_properties(health_tests PROPERTIES LABELS "health")', registrations)


if __name__ == "__main__":
    unittest.main(verbosity=2)
