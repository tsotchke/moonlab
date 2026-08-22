#!/usr/bin/env python3
"""Contract checks for the fail-closed Eshkol v1.3.4 evidence producer."""

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCER = REPO_ROOT / "scripts" / "run_eshkol_v134_compat_gate.sh"


class EshkolCompatibilityProducerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = PRODUCER.read_text(encoding="utf-8")

    def test_published_probe_matrix_is_exact(self) -> None:
        expected = (
            "quantum_smoke_test",
            "bell_chsh_test",
            "quantum_surface_coverage_test",
            "vqe_test",
            "vqe_ad_test",
            "vqe_ad_adversarial",
            "pqc_mlkem_test",
        )
        matrix = re.search(r"TESTS=\(([^)]*)\)", self.source, re.S)
        self.assertIsNotNone(matrix)
        self.assertEqual(tuple(matrix.group(1).split()), expected)

    def test_published_tag_ref_is_commit_pinned(self) -> None:
        self.assertIn('ESH_TAG="v1.3.4-evolve"', self.source)
        self.assertIn(
            'EXPECTED_ESH_COMMIT="694c31798f3f89d55015492bffd027c01951f7bf"',
            self.source,
        )
        self.assertIn('show-ref --verify --quiet "refs/tags/$ESH_TAG"', self.source)
        self.assertIn('rev-parse --verify "refs/tags/$ESH_TAG^{commit}"', self.source)
        self.assertIn('archive --format=tar "$ESH_COMMIT"', self.source)

    def test_build_is_cpu_only_and_uses_clean_source_override(self) -> None:
        for flag in (
            "-DESHKOL_QUANTUM_ENABLED=ON",
            "-DESHKOL_GPU_ENABLED=OFF",
            "-DQSIM_ENABLE_METAL=OFF",
            "-DQSIM_ENABLE_CUDA=OFF",
            "-DQSIM_ENABLE_CUQUANTUM=OFF",
            "-DQSIM_ENABLE_OPENCL=OFF",
            "-DQSIM_ENABLE_VULKAN=OFF",
            "-DQSIM_ENABLE_WEBGPU=OFF",
            '-DFETCHCONTENT_SOURCE_DIR_MOONLAB="$REPO_ROOT"',
        ):
            self.assertIn(flag, self.source)
        self.assertIn('git -C "$ESH_REPO" archive', self.source)
        self.assertNotIn("cmake -S \"$ESH_REPO\"", self.source)

    def test_verdict_requires_pass_and_rejects_error(self) -> None:
        self.assertIn("grep -Eq '(^|[^[:alpha:]])PASS([^[:alpha:]]|$)'", self.source)
        self.assertIn("grep -q 'ERROR'", self.source)
        self.assertIn("status", self.source)
        self.assertIn("value", self.source)
        self.assertIn('"kind": "moonlab_eshkol_compatibility"', self.source)
        self.assertIn('"name": "eshkol_v134_quantum_consumer"', self.source)


if __name__ == "__main__":
    unittest.main()
