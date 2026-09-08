"""Real Stim model + injected-sampler negative controls for the F3 gate.

Requires NumPy, Stim, and MOONLAB_LIB_DIR pointing at a built sampler library.
"""
import unittest
from unittest.mock import patch

import numpy as np
import stim

from benchmarks.dominance.fronts import f3_batch_sampling_vs_stim as front


class DetectorGateTests(unittest.TestCase):
    def test_exact_independent_fault_parity(self):
        circuit = stim.Circuit('X_ERROR(0.1) 0\nX_ERROR(0.2) 0\nM 0\nDETECTOR rec[-1]')
        np.testing.assert_allclose(front.detector_reference_probabilities(circuit), [.26])

    def test_duplicate_detector_targets_cancel(self):
        circuit = stim.Circuit('X_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1] rec[-1]')
        np.testing.assert_array_equal(front.detector_reference_probabilities(circuit), [0])

    def test_gauge_detector_is_not_silently_approximated(self):
        circuit = stim.Circuit('H 0\nM 0\nDETECTOR rec[-1]')
        with self.assertRaises(ValueError):
            front.detector_reference_probabilities(circuit)

    def test_matching_wrong_engines_still_fail(self):
        n, ops, dets = front.build_surface_code_noisy(3, 2, p2=0, p1=0, pm=0)
        wrong = np.ones((len(dets), 100), dtype=np.uint8)
        with patch.object(front, 'moonlab_sample_detectors', return_value=wrong), \
             patch.object(front, 'stim_sample_detectors', return_value=wrong):
            ok, detail = front.check_detector_correctness(n, ops, dets, shots=100)
        self.assertFalse(ok)
        self.assertEqual(detail['marg_sigma'], 0)
        self.assertEqual(detail['analytic_moonlab_sigma'], float('inf'))
        self.assertEqual(detail['analytic_stim_sigma'], float('inf'))

    def test_single_impossible_detector_event_fails(self):
        n, ops, dets = front.build_surface_code_noisy(3, 2, p2=0, p1=0, pm=0)
        good = np.zeros((len(dets), 100), dtype=np.uint8)
        wrong = good.copy()
        wrong[0, 0] = 1
        with patch.object(front, 'moonlab_sample_detectors', return_value=wrong), \
             patch.object(front, 'stim_sample_detectors', return_value=good):
            ok, detail = front.check_detector_correctness(n, ops, dets, shots=100)
        self.assertFalse(ok)
        self.assertEqual(detail['analytic_moonlab_sigma'], float('inf'))

    def test_real_issue21_workload_against_analytic_reference(self):
        n, ops, dets = front.build_surface_code_noisy(5, 8)
        ok, detail = front.check_detector_correctness(n, ops, dets, shots=200000, seed=1234)
        self.assertTrue(ok, detail)
        # Stim does not promise identical seeded samples across versions or
        # SIMD builds. The frozen counts are tested separately as pure data.
        self.assertIn('legacy_marg_sigma', detail)
        self.assertLess(detail['marg_sigma'], 6)
        self.assertLess(detail['analytic_moonlab_sigma'], 6)
        self.assertLess(detail['analytic_stim_sigma'], 6)


if __name__ == '__main__':
    unittest.main()
