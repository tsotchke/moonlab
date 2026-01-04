"""
Tests for QuantumState core functionality.
"""

import numpy as np
import pytest

from moonlab import QuantumState, QuantumError


class TestQuantumStateCreation:
    """Tests for quantum state initialization."""

    def test_create_single_qubit(self):
        """Single qubit state should initialize to |0>."""
        state = QuantumState(1)
        assert state.num_qubits == 1
        assert state.state_dim == 2
        assert abs(state.probability(0) - 1.0) < 1e-10

    def test_create_multi_qubit(self):
        """Multi-qubit state should initialize to |00...0>."""
        for n in [2, 4, 8, 10]:
            state = QuantumState(n)
            assert state.num_qubits == n
            assert state.state_dim == 2**n
            assert abs(state.probability(0) - 1.0) < 1e-10

    def test_invalid_qubit_count_zero(self):
        """Zero qubits should raise ValueError."""
        with pytest.raises(ValueError):
            QuantumState(0)

    def test_invalid_qubit_count_negative(self):
        """Negative qubits should raise ValueError."""
        with pytest.raises(ValueError):
            QuantumState(-1)

    def test_invalid_qubit_count_too_large(self):
        """More than 32 qubits should raise ValueError."""
        with pytest.raises(ValueError):
            QuantumState(33)

    def test_repr(self, two_qubit_state):
        """Repr should show qubit count and dimension."""
        r = repr(two_qubit_state)
        assert "num_qubits=2" in r
        assert "dim=4" in r


class TestQuantumStateOperations:
    """Tests for state manipulation operations."""

    def test_reset(self, two_qubit_state, assert_probability):
        """Reset should return state to |00...0>."""
        state = two_qubit_state
        state.h(0).h(1)  # Create superposition
        state.reset()
        assert_probability(state, 0, 1.0)
        assert_probability(state, 1, 0.0)
        assert_probability(state, 2, 0.0)
        assert_probability(state, 3, 0.0)

    def test_clone(self, two_qubit_state):
        """Clone should create independent copy."""
        state = two_qubit_state
        state.h(0)
        cloned = state.clone()

        # Verify clone has same state
        assert cloned.num_qubits == state.num_qubits
        np.testing.assert_allclose(
            state.probabilities(),
            cloned.probabilities(),
            atol=1e-10
        )

        # Modify original and verify clone unchanged
        original_probs = cloned.probabilities().copy()
        state.x(1)
        np.testing.assert_allclose(
            cloned.probabilities(),
            original_probs,
            atol=1e-10
        )

    def test_normalize(self, single_qubit_state, assert_normalized):
        """Normalize should ensure probabilities sum to 1."""
        state = single_qubit_state
        state.h(0)
        state.normalize()
        assert_normalized(state)

    def test_method_chaining(self, two_qubit_state):
        """Gate methods should return self for chaining."""
        state = two_qubit_state
        result = state.h(0).x(1).cnot(0, 1)
        assert result is state


class TestProbabilities:
    """Tests for probability calculations."""

    def test_ground_state_probability(self, two_qubit_state, assert_probability):
        """Ground state should have probability 1 at |0>."""
        assert_probability(two_qubit_state, 0, 1.0)
        assert_probability(two_qubit_state, 1, 0.0)
        assert_probability(two_qubit_state, 2, 0.0)
        assert_probability(two_qubit_state, 3, 0.0)

    def test_hadamard_superposition(self, single_qubit_state, assert_probability):
        """Hadamard should create equal superposition."""
        single_qubit_state.h(0)
        assert_probability(single_qubit_state, 0, 0.5)
        assert_probability(single_qubit_state, 1, 0.5)

    def test_probabilities_array(self, two_qubit_state, assert_normalized):
        """Probabilities array should be normalized."""
        two_qubit_state.h(0).h(1)
        probs = two_qubit_state.probabilities()
        assert len(probs) == 4
        assert abs(np.sum(probs) - 1.0) < 1e-10

    def test_probability_bounds(self, two_qubit_state):
        """All probabilities should be in [0, 1]."""
        two_qubit_state.h(0).cnot(0, 1)
        probs = two_qubit_state.probabilities()
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_invalid_basis_state_negative(self, two_qubit_state):
        """Negative basis state should raise ValueError."""
        with pytest.raises(ValueError):
            two_qubit_state.probability(-1)

    def test_invalid_basis_state_too_large(self, two_qubit_state):
        """Basis state >= state_dim should raise ValueError."""
        with pytest.raises(ValueError):
            two_qubit_state.probability(4)


class TestStatevector:
    """Tests for statevector extraction."""

    def test_ground_state_vector(self, two_qubit_state):
        """Ground state vector should be [1, 0, 0, 0]."""
        sv = two_qubit_state.get_statevector()
        expected = np.array([1, 0, 0, 0], dtype=complex)
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_hadamard_statevector(self, single_qubit_state):
        """Hadamard statevector should be [1, 1]/sqrt(2)."""
        single_qubit_state.h(0)
        sv = single_qubit_state.get_statevector()
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_statevector_shape(self, four_qubit_state):
        """Statevector should have shape (2^n,)."""
        sv = four_qubit_state.get_statevector()
        assert sv.shape == (16,)
        assert sv.dtype == complex

    def test_statevector_normalized(self, two_qubit_state):
        """Statevector should have unit norm."""
        two_qubit_state.h(0).cnot(0, 1)
        sv = two_qubit_state.get_statevector()
        norm = np.linalg.norm(sv)
        assert abs(norm - 1.0) < 1e-10


class TestEntangledStates:
    """Tests for entangled state properties."""

    def test_bell_state_probabilities(self, bell_state, assert_probability):
        """Bell state should have 50% for |00> and |11>."""
        assert_probability(bell_state, 0, 0.5)  # |00>
        assert_probability(bell_state, 1, 0.0)  # |01>
        assert_probability(bell_state, 2, 0.0)  # |10>
        assert_probability(bell_state, 3, 0.5)  # |11>

    def test_ghz_state_probabilities(self, ghz_state, assert_probability):
        """GHZ state should have 50% for |000> and |111>."""
        assert_probability(ghz_state, 0, 0.5)  # |000>
        assert_probability(ghz_state, 7, 0.5)  # |111>

        # All other states should have 0 probability
        for i in range(1, 7):
            assert_probability(ghz_state, i, 0.0)

    def test_bell_state_correlations(self):
        """Bell state measurements should show perfect correlations."""
        from moonlab import QuantumState

        # Run multiple trials
        correlations = []
        for _ in range(50):
            state = QuantumState(2)
            state.h(0).cnot(0, 1)
            outcome = state.measure_all_fast()
            # Check |00> or |11>
            correlations.append(outcome == 0 or outcome == 3)

        # All measurements should show correlation
        assert sum(correlations) == len(correlations)


class TestMeasurement:
    """Tests for measurement operations."""

    def test_measure_ground_state(self, two_qubit_state):
        """Measuring ground state should always give 0."""
        for _ in range(10):
            state = QuantumState(2)
            outcome = state.measure_all_fast()
            assert outcome == 0

    def test_measure_excited_state(self, two_qubit_state):
        """Measuring |11> should always give 3."""
        for _ in range(10):
            state = QuantumState(2)
            state.x(0).x(1)  # |11>
            outcome = state.measure_all_fast()
            assert outcome == 3

    def test_measure_superposition_statistics(self):
        """Superposition measurements should follow probability distribution."""
        counts = {0: 0, 1: 0}
        trials = 100

        for _ in range(trials):
            state = QuantumState(1)
            state.h(0)
            outcome = state.measure_all_fast()
            counts[outcome] += 1

        # Should be roughly 50/50 (allow statistical variance)
        assert counts[0] > 20  # Unlikely to get <20% in 100 trials
        assert counts[1] > 20

    def test_measure_collapses_state(self, two_qubit_state, assert_probability):
        """Measurement should collapse state to measured outcome."""
        state = two_qubit_state
        state.h(0).h(1)  # Equal superposition
        outcome = state.measure_all_fast()

        # State should now be collapsed
        assert_probability(state, outcome, 1.0)
