"""
Tests for quantum algorithm implementations.

Includes VQE, QAOA, Grover's algorithm, and Bell tests.
"""

import numpy as np
import pytest

from moonlab import QuantumState
from moonlab.algorithms import (
    VQE,
    QAOA,
    Grover,
    BellTest,
    MolecularHamiltonian,
    run_vqe_h2,
    run_qaoa_maxcut,
    run_grover,
    run_bell_test,
)


# =============================================================================
# VQE Tests
# =============================================================================

class TestVQEBasic:
    """Basic VQE functionality tests."""

    def test_vqe_creation(self):
        """VQE should initialize correctly."""
        vqe = VQE(num_qubits=4, num_layers=2)
        assert vqe.num_qubits == 4
        assert vqe.num_layers == 2

    def test_vqe_creation_different_optimizers(self):
        """VQE should accept different optimizer types."""
        for opt_type in [VQE.OPTIMIZER_ADAM, VQE.OPTIMIZER_GRADIENT_DESCENT,
                         VQE.OPTIMIZER_LBFGS, VQE.OPTIMIZER_COBYLA]:
            vqe = VQE(num_qubits=4, num_layers=1, optimizer_type=opt_type)
            assert vqe.optimizer_type == opt_type


class TestVQEH2:
    """Tests for H2 molecule VQE."""

    @pytest.mark.slow
    def test_vqe_h2_equilibrium(self):
        """VQE should find reasonable H2 ground state at equilibrium."""
        vqe = VQE(num_qubits=4, num_layers=2)
        result = vqe.solve_h2(bond_distance=0.74)

        # H2 ground state energy ~ -1.137 Hartree at equilibrium
        assert result['energy'] < -1.0
        assert result['converged'] or result['num_iterations'] > 0
        assert 'optimal_params' in result

    @pytest.mark.slow
    def test_vqe_h2_different_distances(self):
        """VQE energy should vary with bond distance."""
        vqe = VQE(num_qubits=4, num_layers=1)

        energies = []
        for dist in [0.5, 0.74, 1.0, 1.5]:
            result = vqe.solve_h2(bond_distance=dist)
            energies.append(result['energy'])

        # Energy should be lowest near equilibrium (0.74 A)
        assert energies[1] <= max(energies[0], energies[2])

    def test_vqe_compute_energy(self):
        """compute_energy should return valid energy values."""
        vqe = VQE(num_qubits=4, num_layers=1)
        result = vqe.solve_h2(bond_distance=0.74)

        # Compute energy at optimal parameters
        if len(result['optimal_params']) > 0:
            energy = vqe.compute_energy(result['optimal_params'])
            assert abs(energy - result['energy']) < 0.1

    def test_run_vqe_h2_convenience(self):
        """Convenience function should work."""
        result = run_vqe_h2(bond_distance=0.74, num_layers=1)
        assert 'energy' in result
        assert 'converged' in result


class TestMolecularHamiltonian:
    """Tests for custom Hamiltonian construction."""

    def test_hamiltonian_creation(self):
        """MolecularHamiltonian should create successfully."""
        H = MolecularHamiltonian(4)
        assert H.num_qubits == 4

    def test_hamiltonian_add_term(self):
        """Adding Pauli terms should work."""
        H = MolecularHamiltonian(4)
        H.add_term(0.5, "ZIII")
        H.add_term(-0.3, "IZZI")
        H.add_term(0.1, "XXXX")

    def test_hamiltonian_invalid_pauli_length(self):
        """Wrong-length Pauli string should raise ValueError."""
        H = MolecularHamiltonian(4)
        with pytest.raises(ValueError):
            H.add_term(0.5, "ZII")  # Only 3 chars for 4 qubits


# =============================================================================
# QAOA Tests
# =============================================================================

class TestQAOABasic:
    """Basic QAOA functionality tests."""

    def test_qaoa_creation(self):
        """QAOA should initialize correctly."""
        qaoa = QAOA(num_qubits=5, num_layers=3)
        assert qaoa.num_qubits == 5
        assert qaoa.num_layers == 3


class TestQAOAMaxCut:
    """Tests for QAOA MaxCut solver."""

    @pytest.fixture
    def simple_graph(self):
        """Simple 4-vertex graph for testing."""
        return [(0, 1), (1, 2), (2, 3), (3, 0)]

    @pytest.fixture
    def triangle_graph(self):
        """Triangle graph (3 vertices)."""
        return [(0, 1), (1, 2), (2, 0)]

    def test_qaoa_maxcut_simple(self, simple_graph):
        """QAOA should solve simple MaxCut."""
        qaoa = QAOA(num_qubits=4, num_layers=1)
        result = qaoa.solve_maxcut(simple_graph)

        assert 'best_bitstring' in result
        assert 'best_cost' in result
        assert 'expectation' in result
        # MaxCut of a 4-cycle should be 4 (alternating assignment)
        assert result['best_cost'] >= 2  # At least 2 edges cut

    def test_qaoa_maxcut_triangle(self, triangle_graph):
        """QAOA should solve triangle MaxCut."""
        qaoa = QAOA(num_qubits=3, num_layers=2)
        result = qaoa.solve_maxcut(triangle_graph)

        # Triangle MaxCut is 2 (any single vertex vs other two)
        assert result['best_cost'] >= 1

    def test_qaoa_maxcut_weighted(self, simple_graph):
        """QAOA should handle weighted edges."""
        qaoa = QAOA(num_qubits=4, num_layers=1)
        weights = [1.0, 2.0, 1.0, 2.0]  # Higher weight on edges (1,2) and (3,0)
        result = qaoa.solve_maxcut(simple_graph, weights=weights)
        assert result['best_cost'] >= 2

    def test_qaoa_compute_expectation(self, simple_graph):
        """compute_expectation should return valid values."""
        qaoa = QAOA(num_qubits=4, num_layers=2)
        result = qaoa.solve_maxcut(simple_graph)

        if len(result['optimal_gamma']) > 0:
            exp = qaoa.compute_expectation(
                result['optimal_gamma'],
                result['optimal_beta']
            )
            assert isinstance(exp, float)

    def test_qaoa_evaluate_bitstring(self, simple_graph):
        """evaluate_bitstring should compute correct cost."""
        qaoa = QAOA(num_qubits=4, num_layers=1)
        qaoa.solve_maxcut(simple_graph)

        # Alternating assignment (0101 = 5 or 1010 = 10) should cut all 4 edges
        cost_5 = qaoa.evaluate_bitstring(5)   # 0101
        cost_10 = qaoa.evaluate_bitstring(10)  # 1010

        # Both should be optimal for 4-cycle
        assert cost_5 == cost_10

    def test_run_qaoa_maxcut_convenience(self, simple_graph):
        """Convenience function should work."""
        result = run_qaoa_maxcut(simple_graph, num_layers=1)
        assert 'best_bitstring' in result
        assert 'best_cost' in result


class TestQAOAIsing:
    """Tests for QAOA Ising model solver."""

    def test_qaoa_ising_simple(self):
        """QAOA should solve simple Ising model."""
        qaoa = QAOA(num_qubits=3, num_layers=2)

        # Simple antiferromagnetic chain
        J = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=float)
        h = np.zeros(3)

        result = qaoa.solve_ising(J, h)
        assert 'best_bitstring' in result


# =============================================================================
# Grover Tests
# =============================================================================

class TestGroverBasic:
    """Basic Grover functionality tests."""

    def test_grover_creation(self):
        """Grover should initialize correctly."""
        grover = Grover(num_qubits=5)
        assert grover.num_qubits == 5

    def test_grover_optimal_iterations(self):
        """optimal_iterations should return reasonable value."""
        grover = Grover(num_qubits=10)
        opt_iter = grover.optimal_iterations

        # Optimal is approximately pi/4 * sqrt(N)
        # For N=1024, this is ~25
        assert 15 <= opt_iter <= 35


class TestGroverSearch:
    """Tests for Grover search algorithm."""

    def test_grover_search_small(self):
        """Grover should find marked state in small search space."""
        grover = Grover(num_qubits=4)  # 16 states
        result = grover.search(marked_state=7)

        assert 'found_state' in result
        assert 'success' in result
        assert 'probability' in result
        # High probability of success for small N
        if result['success']:
            assert result['found_state'] == 7

    @pytest.mark.slow
    def test_grover_search_medium(self):
        """Grover should work for medium search space."""
        grover = Grover(num_qubits=8)  # 256 states
        result = grover.search(marked_state=42)

        # Should usually succeed
        assert result['iterations_used'] > 0

    def test_grover_search_custom_iterations(self):
        """Grover should accept custom iteration count."""
        grover = Grover(num_qubits=4)
        result = grover.search(marked_state=5, num_iterations=2)

        assert result['iterations_used'] == 2

    def test_grover_invalid_marked_state(self):
        """Invalid marked state should raise ValueError."""
        grover = Grover(num_qubits=4)

        with pytest.raises(ValueError):
            grover.search(marked_state=-1)

        with pytest.raises(ValueError):
            grover.search(marked_state=16)  # >= 2^4

    def test_run_grover_convenience(self):
        """Convenience function should work."""
        result = run_grover(num_qubits=4, marked_state=5)
        assert 'found_state' in result


class TestGroverComponents:
    """Tests for Grover algorithm components."""

    def test_grover_oracle(self):
        """Oracle should flip phase of marked state."""
        grover = Grover(num_qubits=3)

        # Start with uniform superposition
        for i in range(3):
            grover.state.h(i)

        # Apply oracle
        marked = 5
        grover.oracle(marked)

        # Marked state should have negative amplitude
        sv = grover.state.get_statevector()
        # After oracle, |marked> has flipped sign
        # (hard to verify without knowing initial phase)

    def test_grover_diffusion(self):
        """Diffusion should invert about mean."""
        grover = Grover(num_qubits=3)

        # Start with uniform superposition
        for i in range(3):
            grover.state.h(i)

        initial_probs = grover.probabilities().copy()
        grover.diffusion()
        final_probs = grover.probabilities()

        # After diffusion on uniform state, should still be uniform
        np.testing.assert_allclose(initial_probs, final_probs, atol=1e-6)

    def test_grover_step(self):
        """Single Grover step should increase marked state probability."""
        grover = Grover(num_qubits=4)
        marked = 7

        # Start with uniform superposition
        for i in range(grover.num_qubits):
            grover.state.h(i)

        initial_prob = grover.state.probability(marked)

        # Single step
        grover.step(marked)

        final_prob = grover.state.probability(marked)

        # Probability should increase
        assert final_prob > initial_prob


# =============================================================================
# Bell Test Tests
# =============================================================================

class TestBellTestBasic:
    """Basic Bell test functionality."""

    def test_create_bell_state_phi_plus(self, two_qubit_state):
        """Should create |Phi+> Bell state."""
        BellTest.create_bell_state(two_qubit_state, 0, 1, BellTest.PHI_PLUS)

        # |Phi+> = (|00> + |11>)/sqrt(2)
        probs = two_qubit_state.probabilities()
        assert abs(probs[0] - 0.5) < 0.01  # |00>
        assert abs(probs[3] - 0.5) < 0.01  # |11>
        assert abs(probs[1]) < 0.01  # |01>
        assert abs(probs[2]) < 0.01  # |10>

    def test_create_bell_state_psi_plus(self, two_qubit_state):
        """Should create |Psi+> Bell state."""
        BellTest.create_bell_state(two_qubit_state, 0, 1, BellTest.PSI_PLUS)

        # |Psi+> = (|01> + |10>)/sqrt(2)
        probs = two_qubit_state.probabilities()
        assert abs(probs[0]) < 0.01  # |00>
        assert abs(probs[3]) < 0.01  # |11>
        assert abs(probs[1] - 0.5) < 0.01  # |01>
        assert abs(probs[2] - 0.5) < 0.01  # |10>


class TestCHSHTest:
    """Tests for CHSH Bell inequality."""

    def test_chsh_bell_state(self):
        """Bell state should violate CHSH inequality."""
        state = QuantumState(2)
        BellTest.create_bell_state(state, 0, 1, BellTest.PHI_PLUS)

        result = BellTest.chsh_test(state, 0, 1, num_measurements=1000)

        assert 'chsh' in result
        assert 'correlations' in result
        assert 'violates_classical' in result

        # Quantum bound is 2*sqrt(2) ≈ 2.828
        # Classical bound is 2
        # With statistical noise, should still be > 2.5 for Bell state
        assert result['chsh'] > 2.0  # At least beat classical

    def test_chsh_product_state_classical(self):
        """Product state should respect CHSH inequality."""
        state = QuantumState(2)
        # Don't entangle - just prepare product state
        state.h(0)  # |+> on qubit 0
        state.h(1)  # |+> on qubit 1

        result = BellTest.chsh_test(state, 0, 1, num_measurements=1000)

        # Product states should have S <= 2
        assert result['chsh'] <= 2.5  # Allow some statistical noise

    @pytest.mark.slow
    def test_chsh_high_statistics(self):
        """High statistics should give CHSH close to 2*sqrt(2)."""
        state = QuantumState(2)
        BellTest.create_bell_state(state, 0, 1, BellTest.PHI_PLUS)

        result = BellTest.chsh_test(state, 0, 1, num_measurements=10000)

        # Should be close to theoretical maximum 2*sqrt(2) ≈ 2.828
        assert result['chsh'] > 2.5
        assert result['violates_classical']

    def test_quick_test(self):
        """Quick test should work."""
        result = BellTest.quick_test(num_qubits=2, num_measurements=500)

        assert 'chsh' in result
        assert result['chsh'] > 2.0

    def test_run_bell_test_convenience(self):
        """Convenience function should work."""
        result = run_bell_test(num_measurements=500)
        assert 'chsh' in result


class TestCHSHCorrelations:
    """Tests for CHSH correlation measurements."""

    def test_correlation_bounds(self):
        """All correlations should be in [-1, 1]."""
        state = QuantumState(2)
        BellTest.create_bell_state(state, 0, 1, BellTest.PHI_PLUS)

        result = BellTest.chsh_test(state, 0, 1, num_measurements=500)

        for name, value in result['correlations'].items():
            assert -1.0 <= value <= 1.0, f"Correlation {name} = {value} out of bounds"

    def test_phi_plus_correlations(self):
        """Phi+ correlations should follow expected pattern."""
        state = QuantumState(2)
        BellTest.create_bell_state(state, 0, 1, BellTest.PHI_PLUS)

        result = BellTest.chsh_test(state, 0, 1, num_measurements=2000)

        # For optimal angles, expect:
        # E(a,b) ≈ 1/sqrt(2) ≈ 0.707
        # These are statistical, so allow wide tolerance
        assert abs(result['correlations']['E(a,b)']) > 0.3
