"""Regression and negative controls for the sampling comparison gate."""
import math
import unittest

from benchmarks.dominance.fronts.sampling_statistics import (
    binomial_reference_sigma, proportion_difference_sigma,
)


class SamplingStatisticsTests(unittest.TestCase):
    def test_independent_equal_samples_count_both_variances(self):
        # Two independent n=100 samples with 40 and 60 successes: pooled
        # p=1/2, variance=1/200, score=sqrt(8), not the one-sample score.
        self.assertAlmostEqual(proportion_difference_sigma(.4, 100, .6, 100), math.sqrt(8))

    def test_unequal_samples_are_weighted(self):
        self.assertAlmostEqual(proportion_difference_sigma(.5, 100, .25, 200), 4.330127018922193)
        self.assertAlmostEqual(proportion_difference_sigma(.25, 200, .5, 100), 4.330127018922193)

    def test_reported_issue21_frozen_detector_counts(self):
        self.assertAlmostEqual(proportion_difference_sigma(3408/200000, 200000,
                                                          3077/200000, 200000),
                               4.144024873658385)
        expected = 0.016143676730092427
        self.assertAlmostEqual(binomial_reference_sigma(3408/200000, 200000, expected),
                               3.1806259836975217)

    def test_systematic_bias_still_fails_six_sigma(self):
        self.assertGreater(proportion_difference_sigma(.025, 200000, .015, 200000), 6)
        self.assertGreater(binomial_reference_sigma(.025, 200000, .015), 6)

    def test_equal_biased_samplers_fail_analytic_reference(self):
        self.assertEqual(proportion_difference_sigma(.025, 200000, .025, 200000), 0)
        self.assertGreater(binomial_reference_sigma(.025, 200000, .015), 6)

    def test_deterministic_events_and_invalid_inputs(self):
        for probability in (0, 1):
            self.assertEqual(binomial_reference_sigma(probability, 100, probability), 0)
            self.assertEqual(proportion_difference_sigma(probability, 100, probability, 100), 0)
        self.assertTrue(math.isinf(binomial_reference_sigma(.01, 100, 0)))
        for probability in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                proportion_difference_sigma(probability, 100, .5, 100)
        with self.assertRaises(ValueError):
            proportion_difference_sigma(.5, 0, .5, 100)


if __name__ == '__main__':
    unittest.main()
