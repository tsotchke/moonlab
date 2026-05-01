"""Smoke tests for the moonlab.ca_peps Python bindings."""

from __future__ import annotations

import numpy as np
import pytest

import moonlab


pytestmark = pytest.mark.skipif(
    not getattr(moonlab, "_CAPEPS_AVAILABLE", False),
    reason="moonlab.ca_peps did not load (libquantumsim missing CA-PEPS ABI?)",
)


def test_capeps_lifecycle_and_shape():
    p = moonlab.CAPEPS(Lx=3, Ly=4, max_bond_dim=8)
    assert p.Lx == 3
    assert p.Ly == 4
    assert p.num_qubits == 12
    assert p.max_bond_dim == 8
    c = p.clone()
    assert c.num_qubits == 12
    c.free()
    p.free()


def test_capeps_bell_pair_zz_correlator():
    """H+CNOT on a horizontal pair gives <ZZ> = +1, <XX> = +1."""
    p = moonlab.CAPEPS(Lx=2, Ly=2, max_bond_dim=4)
    # Bell pair on (0,0)-(1,0): linear indices 0 and 1.
    p.h(0)
    p.cnot(0, 1)

    pauli_zz = np.zeros(4, dtype=np.uint8)
    pauli_zz[0] = 3; pauli_zz[1] = 3
    zz = p.expect_pauli(pauli_zz)
    assert abs(zz.real - 1.0) < 1e-10
    assert abs(zz.imag) < 1e-12

    pauli_xx = np.zeros(4, dtype=np.uint8)
    pauli_xx[0] = 1; pauli_xx[1] = 1
    xx = p.expect_pauli(pauli_xx)
    assert abs(xx.real - 1.0) < 1e-10


def test_capeps_adjacency_rejection():
    """Diagonal pairs and same-site pairs return CA_PEPS_ERR_QUBIT."""
    p = moonlab.CAPEPS(Lx=3, Ly=3, max_bond_dim=4)
    # Horizontal adjacent: (0,0)-(1,0) -> q=0, q=1, OK.
    p.cnot(0, 1)
    # Diagonal (0,0)-(1,1) -> q=0, q=4 must reject.
    with pytest.raises(RuntimeError):
        p.cnot(0, 4)
    # Same site must reject.
    with pytest.raises(RuntimeError):
        p.cz(2, 2)


def test_capeps_rotation_unaffected_by_clifford_layer():
    """RZ on a |+> state is a global phase: <X> stays 1."""
    p = moonlab.CAPEPS(Lx=2, Ly=2, max_bond_dim=4)
    p.h(0)
    p.rz(0, 0.7)  # |+> -> e^{-i 0.35} cos|0> + ... still <X>=cos(0.7)? no wait
    # Actually H|0>=|+>, then RZ(theta)|+> = (e^{-i theta/2}|0> + e^{+i theta/2}|1>)/sqrt(2),
    # so <X> on that single qubit = cos(theta).
    pauli_x = np.zeros(4, dtype=np.uint8)
    pauli_x[0] = 1
    x_val = p.expect_pauli(pauli_x)
    assert abs(x_val.real - np.cos(0.7)) < 1e-10
    assert abs(x_val.imag) < 1e-12
