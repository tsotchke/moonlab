import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmarks.dominance.check_detector_joint_accuracy import (
    ALPHA_6SIGMA, cells_from_bits, chernoff_tail, joint_reference, pair_record, kl_bernoulli,
)


class Target:
    def __init__(self, value): self.val = value
    def is_relative_detector_id(self): return self.val is not None


class Inst:
    type = "error"
    def __init__(self, p, *detectors): self.p, self.detectors = p, detectors
    def args_copy(self): return [self.p]
    def targets_copy(self): return [Target(d) for d in self.detectors]


class Dem:
    def __init__(self, *terms): self.terms = terms
    def flattened(self): return self.terms


def test_shared_error_has_bell_like_joint_distribution():
    # One shared fault makes the two parity bits equal: p00=1-p, p11=p.
    got = joint_reference(Dem(Inst(.2, 0, 1)), 0, 1)
    np.testing.assert_allclose(got, [.8, 0, 0, .2])


def test_independent_faults_factorize():
    got = joint_reference(Dem(Inst(.1, 0), Inst(.3, 1)), 0, 1)
    np.testing.assert_allclose(got, [.63, .27, .07, .03])


def test_impossible_events_and_raw_counts_are_preserved():
    got = joint_reference(Dem(Inst(.2, 0, 1)), 0, 1)
    assert got[1] == 0 and got[2] == 0
    counts = cells_from_bits(np.array([0, 1, 1]), np.array([0, 1, 1]))
    assert counts.tolist() == [1, 0, 0, 2]
    assert chernoff_tail(1, 100, 0.0) == 0.0


def test_matching_marginals_wrong_correlation_is_rejected():
    expected = np.array([.45, .05, .05, .45])
    # Same marginals, anti-correlated joint cells: a correlation-only defect.
    wrong = np.array([.05, .45, .45, .05])
    rec = pair_record((wrong * 200_000).astype(int), expected, 200_000)
    assert rec["counts"] == [10000, 90000, 90000, 10000]
    assert not rec["passed"]
    assert rec["alpha"] == ALPHA_6SIGMA


def test_chernoff_bound_is_capped_and_not_a_p_value():
    assert 0 < chernoff_tail(501, 1000, .5) <= 1
    assert chernoff_tail(500, 1000, .5) == 1.0


def test_repeated_fault_targets_cancel():
    np.testing.assert_allclose(joint_reference(Dem(Inst(.2, 0, 0, 1)), 0, 1), [.8, .2, 0, 0])


def test_chernoff_bound_covers_exact_small_binomial_kl_tail():
    n = 20
    for p in (.01, .1, .5, .9):
        for observed in (0, 1, 5, 10, 20):
            divergence = kl_bernoulli(observed/n, p)
            exact_tail = sum(math.comb(n, k) * p**k * (1-p)**(n-k)
                             for k in range(n+1)
                             if kl_bernoulli(k/n, p) >= divergence-1e-12)
            assert chernoff_tail(observed, n, p) + 1e-12 >= exact_tail


def test_invalid_reference_is_rejected():
    for expected in ([1.1, 0, 0, -.1], [.2]*4, [float('nan'), 0, 0, 1]):
        with unittest.TestCase().assertRaises(ValueError):
            pair_record(np.array([10, 0, 0, 0]), np.array(expected), 10)


def load_tests(loader, tests, pattern):
    """Run the same function tests under either unittest or pytest."""
    return unittest.TestSuite(unittest.FunctionTestCase(fn)
                              for name, fn in sorted(globals().items())
                              if name.startswith('test_') and callable(fn))


if __name__ == '__main__':
    unittest.main()
