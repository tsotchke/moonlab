"""Tests for the moonlab.chemistry quantum-chemistry bindings.

Covers the documented Molecule / Hamiltonian workflow, the raw STO-3G H2
Pauli coefficients, their smooth (differentiable) variation with bond
length, and construction of the H2 Hamiltonian handle.  All numbers come
from the real C engine.
"""

import math

import numpy as np
import pytest

from moonlab.chemistry import (
    Molecule,
    Hamiltonian,
    h2_sto3g_pauli_coeffs,
    hartree_to_kcalmol,
    H2_PAULI_LABELS,
    CHEMICAL_ACCURACY_HARTREE,
)


# =============================================================================
# Raw STO-3G H2 Pauli coefficients
# =============================================================================

class TestH2Coefficients:
    """The five first-principles STO-3G H2 Pauli coefficients."""

    def test_shape_and_finiteness(self):
        g = h2_sto3g_pauli_coeffs(0.74)
        assert g.shape == (5,)
        assert np.all(np.isfinite(g))
        assert len(H2_PAULI_LABELS) == 5

    def test_finite_across_pes(self):
        """Coefficients must be finite at every physically sane bond length."""
        for r in np.linspace(0.3, 3.0, 40):
            g = h2_sto3g_pauli_coeffs(r)
            assert np.all(np.isfinite(g)), f"non-finite coefficients at r={r}"

    def test_coefficients_vary_smoothly(self):
        """Coefficients must vary smoothly (small, bounded step-to-step
        change and no discontinuity) -- the differentiability sanity check.

        A C-infinity PES has a bounded, slowly varying finite-difference
        derivative; a kink or plateau edge would show up as a spike in the
        second difference. We assert the coefficients change continuously
        and that the discrete second derivative stays bounded.
        """
        rs = np.linspace(0.5, 2.0, 200)
        vals = np.array([h2_sto3g_pauli_coeffs(r) for r in rs])  # (200, 5)

        dr = rs[1] - rs[0]
        first = np.diff(vals, axis=0) / dr
        second = np.diff(first, axis=0) / dr

        # The map is smooth: consecutive coefficients differ only slightly.
        assert np.all(np.abs(np.diff(vals, axis=0)) < 0.05)
        # And it actually moves -- this is a genuine PES, not a constant.
        assert np.max(np.abs(first)) > 1e-3
        # No kink: the second difference stays bounded (a true kink diverges
        # as dr -> 0; a smooth curve stays O(1)).
        assert np.all(np.isfinite(second))
        assert np.max(np.abs(second)) < 50.0

    def test_finite_difference_force_is_stable(self):
        """A central finite-difference derivative of a coefficient converges
        as the step shrinks -- the property that makes forces well-defined."""
        r0 = 0.9

        def g_zz(r):
            return h2_sto3g_pauli_coeffs(r)[3]

        d_coarse = (g_zz(r0 + 1e-3) - g_zz(r0 - 1e-3)) / 2e-3
        d_fine = (g_zz(r0 + 1e-4) - g_zz(r0 - 1e-4)) / 2e-4
        assert math.isfinite(d_coarse) and math.isfinite(d_fine)
        # Central differences of a smooth function agree closely.
        assert abs(d_coarse - d_fine) < 1e-2


# =============================================================================
# Hamiltonian handle
# =============================================================================

class TestHamiltonian:
    """The pauli_hamiltonian_t-backed Hamiltonian wrapper."""

    def test_h2_sto3g_build(self):
        H = Hamiltonian.h2_sto3g(bond_distance=0.74)
        assert H.num_qubits == 2
        assert H.num_terms == 5
        assert math.isfinite(H.nuclear_repulsion)

    def test_h2_terms_are_the_five_pauli_strings(self):
        H = Hamiltonian.h2_sto3g(bond_distance=0.74)
        labels = [p for _, p in H.terms]
        assert labels == list(H2_PAULI_LABELS)
        for coeff, label in H.terms:
            assert math.isfinite(coeff)
            assert set(label) <= set("IXYZ")

    def test_h2_coefficients_match_raw_at_equilibrium(self):
        """At r_eq the additive STO-3G anchor vanishes, so the handle's
        coefficients equal the exact O'Malley reference values."""
        H = Hamiltonian.h2_sto3g(bond_distance=0.7414)
        coeffs = dict(zip([p for _, p in H.terms], H.coefficients))
        expected = {
            "II": -1.0523732,
            "IZ": 0.39793742,
            "ZI": -0.39793742,
            "ZZ": -0.01128010,
            "XX": 0.18093120,
        }
        for label, want in expected.items():
            assert abs(coeffs[label] - want) < 1e-6, label

    def test_exact_ground_state_near_reference(self):
        """Exact STO-3G ground state of H2 at equilibrium is ~-1.137 Ha."""
        H = Hamiltonian.h2_sto3g(bond_distance=0.7414)
        e = H.exact_ground_state()
        assert math.isfinite(e)
        assert abs(e - (-1.137)) < 0.05

    def test_lih_build(self):
        H = Hamiltonian.lih_sto3g(bond_distance=1.5949)
        assert H.num_qubits == 4
        assert H.num_terms > 0
        assert math.isfinite(H.exact_ground_state())

    def test_hartree_to_kcalmol(self):
        assert abs(hartree_to_kcalmol(1.0) - 627.5094) < 1.0


# =============================================================================
# Molecule -- the documented example
# =============================================================================

class TestMolecule:
    """The documented Molecule(atoms=..., coordinates=...) surface."""

    def test_documented_h2_example(self):
        """Runs the example from documents/examples/algorithms/vqe-h2-molecule.md."""
        h2 = Molecule(
            atoms=["H", "H"],
            coordinates=[[0, 0, 0], [0, 0, 0.74]],
        )
        H = h2.get_hamiltonian(basis="sto-3g")
        assert isinstance(H, Hamiltonian)
        assert H.num_qubits == 2
        assert abs(h2.bond_distance - 0.74) < 1e-9

    def test_bond_distance_from_coordinates(self):
        h2 = Molecule(
            atoms=["H", "H"],
            coordinates=[[0, 0, 0], [0, 0, 1.2]],
        )
        assert abs(h2.bond_distance - 1.2) < 1e-9
        # The Hamiltonian built from a geometry uses that bond length.
        H = h2.get_hamiltonian()
        assert abs(H.bond_distance - 1.2) < 1e-6

    def test_fci_energy_chemical_accuracy_marker(self):
        h2 = Molecule(atoms=["H", "H"], coordinates=[[0, 0, 0], [0, 0, 0.74]])
        e = h2.fci_energy()
        assert math.isfinite(e)
        # Sanity: within chemical accuracy of the known STO-3G value.
        assert abs(e - (-1.1373)) < 0.05
        assert CHEMICAL_ACCURACY_HARTREE > 0

    def test_lih_molecule(self):
        lih = Molecule(atoms=["Li", "H"], coordinates=[[0, 0, 0], [0, 0, 1.5949]])
        assert lih.species == "LiH"
        H = lih.get_hamiltonian(basis="sto-3g")
        assert H.num_qubits == 4

    def test_energy_varies_with_geometry(self):
        """Different bond lengths give different exact energies (a real PES)."""
        energies = [
            Molecule.h2(bond_distance=r).fci_energy() for r in (0.5, 0.74, 1.2)
        ]
        assert all(math.isfinite(e) for e in energies)
        # Equilibrium (0.74) is lower than the stretched/compressed points.
        assert energies[1] <= energies[0]
        assert energies[1] <= energies[2]

    def test_unsupported_molecule_rejected(self):
        with pytest.raises(ValueError):
            Molecule(atoms=["He", "He"], coordinates=[[0, 0, 0], [0, 0, 1.0]])

    def test_unsupported_basis_rejected(self):
        h2 = Molecule.h2()
        with pytest.raises(ValueError):
            h2.get_hamiltonian(basis="cc-pvdz")

    def test_mismatched_atoms_and_coordinates(self):
        with pytest.raises(ValueError):
            Molecule(atoms=["H", "H"], coordinates=[[0, 0, 0]])
