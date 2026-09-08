import numpy as np
import unittest
from unittest.mock import patch

from benchmarks.dominance.benchmark_estimation_accuracy import (
    aggregate_until_deadline, batch_seed, iid_expected_mse, marginal_rmse, unpack_detector_output,
)


def test_adjacent_seeds_do_not_reuse_batch_streams():
    a = {batch_seed(1, k) for k in range(20)}
    b = {batch_seed(2, k) for k in range(20)}
    assert len(a) == 20 and len(b) == 20 and a.isdisjoint(b)


def test_unpack_packed_is_detector_major_low_bit_first():
    raw = np.array([[0b00000101], [0b00000010]], dtype=np.uint8)
    got = unpack_detector_output(raw, "stim-packed", 3, 2)
    np.testing.assert_array_equal(got, [[1, 0], [0, 1], [1, 0]])


def test_late_batch_is_discarded_and_not_counted():
    now = [0.0]
    def clock(): return now[0]
    def sample(n, seed):
        now[0] += .6
        return np.ones((2, n), dtype=np.uint8)
    rec = aggregate_until_deadline(sample, np.array([.5, .5]), 1.0, 1, clock, 9)
    assert rec["shots"] == 1 and rec["batches"] == 1
    assert rec["late_batch_discarded"] is True
    assert rec["counts"] .tolist() == [1, 1]


def test_budget_and_shape_contracts_are_enforced():
    with unittest.TestCase().assertRaises(ValueError):
        aggregate_until_deadline(lambda n, s: np.zeros((1, n), dtype=np.uint8), [0.5], 1.01)
    with unittest.TestCase().assertRaises(ValueError):
        aggregate_until_deadline(lambda n, s: np.zeros((1, n), dtype=np.uint8), [0.5, 0.5], .1,
                                 batch_size=1)


def test_rmse_reports_observed_and_iid_values():
    ref = np.array([.25, .75])
    assert abs(marginal_rmse(np.array([0, 4]), 4, ref) - .25) < 1e-12
    assert abs(iid_expected_mse(ref, 4) - .046875) < 1e-12


def test_late_reduction_is_discarded_even_when_sampling_was_on_time():
    now = [0.0]
    def clock(): return now[0]
    def sample(n, seed):
        now[0] += .2                 # sampling completes before deadline
        return np.ones((2, n), dtype=np.uint8)
    def reduce(raw):
        now[0] += .9                 # reduction crosses the deadline
        return raw.sum(axis=1, dtype=np.int64)
    rec = aggregate_until_deadline(sample, np.array([.5, .5]), 1.0, 1, clock, 7, reduce)
    assert rec["shots"] == 0 and rec["discarded_shots"] == 1
    assert rec["late_batch_discarded"] is True
    assert rec["last_accepted_checkpoint_time_s"] == 0.0


def test_actual_default_sum_must_finish_before_deadline():
    now = [0.0]
    original_sum = np.sum
    def clock(): return now[0]
    def sample(n, seed):
        now[0] += .2
        return np.ones((2, n), dtype=np.uint8)
    def slow_sum(*args, **kwargs):
        now[0] += .9
        return original_sum(*args, **kwargs)
    with patch.object(np, 'sum', slow_sum):
        rec = aggregate_until_deadline(sample, [.5, .5], 1, 1, clock)
    assert rec['shots'] == 0 and rec['discarded_shots'] == 1
    np.testing.assert_array_equal(rec['counts'], [0, 0])


def test_nonbinary_results_are_rejected():
    with unittest.TestCase().assertRaises(ValueError):
        aggregate_until_deadline(lambda n, s: np.full((1, n), 2, dtype=np.uint8), [.5], .1, 1)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            suite.addTest(unittest.FunctionTestCase(fn))
    return suite


if __name__ == "__main__":
    unittest.main()
