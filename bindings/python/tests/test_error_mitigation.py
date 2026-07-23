"""Tests for moonlab.error_mitigation (ZNE, PEC, readout mitigation)."""

import math

import pytest

from moonlab.error_mitigation import (
    ZNE,
    PEC,
    MeasurementMitigation,
    ZNE_LINEAR,
    ZNE_RICHARDSON,
)


def test_zne_linear_recovers_intercept():
    # E(lambda) = 0.9 - 0.05 * lambda  ->  E(0) = 0.9 exactly.
    zne = ZNE(noise_factors=[1, 2, 3], method="linear")
    value = zne.extrapolate(lambda s: 0.9 - 0.05 * s)
    assert math.isclose(value, 0.9, abs_tol=1e-9)
    assert zne.last_stderr is not None


def test_zne_richardson_exact_on_polynomial():
    # Quadratic through the three scales; Richardson interpolates exactly.
    f = lambda s: 0.3 + 0.1 * s - 0.02 * s * s
    zne = ZNE(noise_factors=[1, 2, 3], method=ZNE_RICHARDSON)
    value = zne.extrapolate(f)
    assert math.isclose(value, f(0.0), abs_tol=1e-9)
    assert zne.last_stderr == pytest.approx(0.0, abs=1e-12)


def test_zne_validates_inputs():
    with pytest.raises(ValueError):
        ZNE(noise_factors=[1.0])
    with pytest.raises(ValueError):
        ZNE(noise_factors=[1.0, 1.0])
    with pytest.raises(ValueError):
        ZNE(noise_factors=[0.0, 1.0])
    with pytest.raises(ValueError):
        ZNE(noise_factors=[1, 2], method="bogus")


def test_pec_one_norm_and_aggregate():
    pec = PEC(etas=[0.6, -0.4])
    assert math.isclose(pec.one_norm_cost, 1.0, abs_tol=1e-12)
    # gamma * mean(sign_i * measurement_i) = 1.0 * mean(0.8, -0.2) = 0.3
    est = pec.aggregate(signs=[1.0, -1.0], measurements=[0.8, 0.2])
    assert math.isclose(est, 0.3, abs_tol=1e-9)


def test_pec_sample_returns_valid_index_and_sign():
    pec = PEC(etas=[0.5, -0.5])
    idx, sign = pec.sample(0.25)
    assert idx in (0, 1)
    assert sign in (-1.0, 1.0)


def test_measurement_mitigation_identity_readout_is_noop():
    # Ideal simulator readout -> assignment matrix is the identity, so
    # correction must return the input distribution unchanged.
    mm = MeasurementMitigation(num_qubits=2)
    mm.calibrate(shots=1)
    corrected = mm.correct({"00": 100, "11": 50})
    assert math.isclose(corrected.get("00", 0.0), 100.0, abs_tol=1e-6)
    assert math.isclose(corrected.get("11", 0.0), 50.0, abs_tol=1e-6)
    assert math.isclose(sum(corrected.values()), 150.0, abs_tol=1e-6)


def test_measurement_mitigation_requires_calibration():
    mm = MeasurementMitigation(num_qubits=2)
    with pytest.raises(RuntimeError):
        mm.correct({"00": 10})
