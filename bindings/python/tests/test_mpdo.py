"""Pytest suite for the v0.3 Python MPDO bindings.

Mirrors the assertions in tests/unit/test_mpdo_smoke.c at 1e-12.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import moonlab

if not getattr(moonlab, "_MPDO_AVAILABLE", False):
    pytest.skip(
        "MPDO bindings unavailable (libquantumsim built without "
        "noise_mpdo entry points)",
        allow_module_level=True,
    )

from moonlab.mpdo import Mpdo


TOL = 1e-12


def test_initial_state_is_all_zero():
    """|0...0><0...0| has Tr = 1 and <Z_q> = +1 for every qubit."""
    rho = Mpdo(num_qubits=3, max_bond_dim=16)
    assert rho.num_qubits == 3
    assert rho.max_bond_dim == 16
    assert rho.current_bond_dim == 1
    assert abs(rho.trace() - 1.0) < TOL
    for q in range(3):
        assert abs(rho.expect_pauli(q, "Z") - 1.0) < TOL


def test_depolarizing_z_decay():
    """<Z> -> 1 - 4p/3 on |0> under symmetric depolarising at rate p."""
    p = 0.4
    rho = Mpdo(1, 16)
    rho.apply_depolarizing(0, p)
    z = rho.expect_pauli(0, "Z")
    assert abs(z - (1 - 4 * p / 3)) < TOL
    assert abs(rho.trace() - 1.0) < TOL


def test_amplitude_damping_full_reset():
    """gamma = 1 should reset the qubit to |0>, even from |1>."""
    rho = Mpdo(1, 16)
    rho.apply_bit_flip(0, 1.0)
    assert abs(rho.expect_pauli(0, "Z") - (-1.0)) < TOL
    rho.apply_amplitude_damping(0, 1.0)
    assert abs(rho.expect_pauli(0, "Z") - 1.0) < TOL


def test_phase_damping_preserves_z_kills_x():
    """T_2 dephasing: <Z> preserved exactly, <X> contracted by sqrt(1-lambda)."""
    lam = 0.2
    rho = Mpdo(2, 16)
    H = (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    rho.apply_kraus(qubit=0, kraus=H)
    rho.apply_phase_damping(0, lam)
    x = rho.expect_pauli(0, "X")
    z = rho.expect_pauli(0, "Z")
    assert abs(x - math.sqrt(1 - lam)) < TOL
    assert abs(z - 0.0) < TOL


def test_clone_preserves_observables():
    rho = Mpdo(1, 16)
    rho.apply_depolarizing(0, 0.3)
    z_before = rho.expect_pauli(0, "Z")
    sigma = rho.clone()
    z_clone = sigma.expect_pauli(0, "Z")
    assert abs(z_before - z_clone) < TOL
    sigma.apply_amplitude_damping(0, 1.0)
    assert abs(rho.expect_pauli(0, "Z") - z_before) < TOL
    assert abs(sigma.expect_pauli(0, "Z") - 1.0) < TOL


def test_named_pauli_codes_match_integers():
    rho = Mpdo(1, 16)
    rho.apply_phase_flip(0, 0.25)
    for label, code in [("I", 0), ("X", 1), ("Y", 2), ("Z", 3)]:
        a = rho.expect_pauli(0, label)
        b = rho.expect_pauli(0, code)
        assert abs(a - b) < TOL


def test_invalid_qubit_raises():
    rho = Mpdo(2, 16)
    with pytest.raises(RuntimeError):
        rho.apply_depolarizing(qubit=99, p=0.1)


def test_invalid_pauli_raises():
    rho = Mpdo(1, 16)
    with pytest.raises(ValueError):
        rho.expect_pauli(0, "Q")
    with pytest.raises(ValueError):
        rho.expect_pauli(0, 9)


def test_custom_kraus_x_flips_z():
    rho = Mpdo(2, 16)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    rho.apply_kraus(qubit=0, kraus=X)
    assert abs(rho.expect_pauli(0, "Z") - (-1.0)) < TOL
    assert abs(rho.expect_pauli(1, "Z") - 1.0) < TOL
    assert abs(rho.trace() - 1.0) < TOL


def test_construction_validation():
    with pytest.raises(ValueError):
        Mpdo(num_qubits=0)
    with pytest.raises(ValueError):
        Mpdo(num_qubits=2, max_bond_dim=0)
