"""Tests for ``moonlab.dmrg`` scalar-energy bindings."""

from __future__ import annotations

import math

import pytest

from moonlab.dmrg import heisenberg_ground_energy, tfim_ground_energy


def test_tfim_critical_point_energy_finite():
    """TFIM at the critical point returns a finite, negative energy."""
    e = tfim_ground_energy(num_sites=8, g=1.0, max_bond_dim=32, num_sweeps=10)
    assert math.isfinite(e), f"expected finite energy, got {e}"
    assert e < 0, f"TFIM ground energy should be negative, got {e}"


def test_tfim_larger_field_more_negative():
    """For -sum ZZ - g sum X the X-term dominates at large g, so larger
    g yields a more negative energy."""
    e_small = tfim_ground_energy(num_sites=8, g=0.1, max_bond_dim=32, num_sweeps=10)
    e_large = tfim_ground_energy(num_sites=8, g=2.0, max_bond_dim=32, num_sweeps=10)
    assert math.isfinite(e_small) and math.isfinite(e_large)
    assert e_large < e_small, (
        f"expected E(g=2.0) < E(g=0.1), got {e_large:.6f} vs {e_small:.6f}")


def test_heisenberg_isotropic_finite():
    """Isotropic Heisenberg chain returns a finite energy."""
    e = heisenberg_ground_energy(num_sites=8, J=1.0, Delta=1.0, h=0.0,
                                  max_bond_dim=32, num_sweeps=10)
    assert math.isfinite(e), f"expected finite energy, got {e}"


def test_heisenberg_ferromagnet_no_field_zero_energy_baseline():
    """XX-only (Delta = 0) gauge: the free-fermion XX chain returns
    a sensible finite energy."""
    e_xx = heisenberg_ground_energy(num_sites=8, J=1.0, Delta=0.0, h=0.0,
                                     max_bond_dim=32, num_sweeps=10)
    assert math.isfinite(e_xx)


@pytest.mark.parametrize("bad", [
    {"num_sites": 1, "g": 1.0, "max_bond_dim": 16, "num_sweeps": 5},
    {"num_sites": 4, "g": 1.0, "max_bond_dim": 0, "num_sweeps": 5},
    {"num_sites": 4, "g": 1.0, "max_bond_dim": 16, "num_sweeps": 0},
])
def test_tfim_invalid_input_returns_sentinel(bad):
    """Invalid input arguments return DBL_MAX sentinel."""
    e = tfim_ground_energy(**bad)
    assert e == pytest.approx(1.7976931348623157e308, rel=1e-6) or e > 1e300
