"""Pytest suite for the Python Clifford-tableau bindings.

Validates the Aaronson-Gottesman tableau backend (`src/backends/
clifford/`) wrapped in `bindings/python/moonlab/clifford.py`.  The
tableau is exact and deterministic; every assertion below holds at
the machine-integer level (no floating-point tolerance is needed).

Reference: Scott Aaronson and Daniel Gottesman, "Improved simulation
of stabilizer circuits", Phys. Rev. A 70, 052328 (2004).
"""

from __future__ import annotations

import pytest

import moonlab
from moonlab.clifford import Clifford


def test_zero_state_z_measurement_deterministic() -> None:
    """|0...0> is a +Z eigenstate of every qubit, so measurement is
    deterministic with outcome 0 and kind = 0."""
    n = 8
    c = Clifford(n, seed=0xDEAD_BEEF)
    for q in range(n):
        outcome, kind = c.measure(q)
        assert outcome == 0
        assert kind == 0


def test_x_flips_z_eigenvalue() -> None:
    """X|0> = |1>; measurement remains deterministic with outcome 1."""
    c = Clifford(4, seed=1)
    c.x(2)
    outcome, kind = c.measure(2)
    assert outcome == 1
    assert kind == 0
    # Other qubits unaffected.
    assert c.measure(0)[0] == 0
    assert c.measure(3)[0] == 0


def test_hadamard_makes_measurement_random() -> None:
    """H|0> = |+> has equal support on |0> and |1>; the tableau
    measurement should report kind == 1 (random outcome)."""
    c = Clifford(1, seed=42)
    c.h(0)
    _, kind = c.measure(0)
    assert kind == 1


def test_ghz_state_has_perfect_correlations() -> None:
    """GHZ on n qubits: H on 0, then CNOT(0, q) for q in [1, n).  Any
    single-qubit measurement collapses every other to the same
    classical bit."""
    n = 16
    c = Clifford(n, seed=0xCAFE_F00D)
    c.h(0)
    for q in range(1, n):
        c.cnot(0, q)
    first, kind = c.measure(0)
    assert kind == 1
    for q in range(1, n):
        outcome, post_kind = c.measure(q)
        assert outcome == first
        # Once the head qubit is measured, the rest collapse to a
        # product state; subsequent measurements are deterministic.
        assert post_kind == 0


def test_bell_pair_correlations() -> None:
    """(|00> + |11>) / sqrt(2): perfect ZZ correlation."""
    c = Clifford(2, seed=7)
    c.h(0)
    c.cnot(0, 1)
    a, _ = c.measure(0)
    b, kind_b = c.measure(1)
    assert a == b
    assert kind_b == 0


def test_s_then_s_dag_is_identity() -> None:
    """S followed by S_dag returns a Z eigenstate to itself."""
    c = Clifford(1, seed=99)
    c.h(0)             # |+>
    c.s(0)             # |+i>
    c.s_dag(0)         # back to |+>
    c.h(0)             # back to |0>
    outcome, kind = c.measure(0)
    assert outcome == 0
    assert kind == 0


def test_cz_equals_h_cnot_h_on_target() -> None:
    """CZ a b == (I tensor H) CNOT a b (I tensor H).  Verify on the
    Bell-input state by checking the measured correlation pattern."""
    c1 = Clifford(2, seed=11)
    c1.h(0)
    c1.cz(0, 1)
    a1, _ = c1.measure(0)
    b1, _ = c1.measure(1)

    c2 = Clifford(2, seed=11)
    c2.h(0)
    c2.h(1)
    c2.cnot(0, 1)
    c2.h(1)
    a2, _ = c2.measure(0)
    b2, _ = c2.measure(1)

    assert (a1, b1) == (a2, b2)


def test_swap_exchanges_qubits() -> None:
    c = Clifford(2, seed=55)
    c.x(0)              # |10>
    c.swap(0, 1)        # |01>
    assert c.measure(0)[0] == 0
    assert c.measure(1)[0] == 1


def test_invalid_qubit_count_raises() -> None:
    with pytest.raises(ValueError):
        Clifford(0)
    with pytest.raises(ValueError):
        Clifford(-1)


def test_num_qubits_property() -> None:
    c = Clifford(13, seed=0)
    assert c.num_qubits == 13


def test_module_is_exported() -> None:
    """`Clifford` must be reachable as `moonlab.Clifford`."""
    assert moonlab.Clifford is Clifford
