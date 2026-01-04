"""
Tests for quantum gate operations.
"""

import numpy as np
import pytest

from moonlab import QuantumState, Gates, QuantumError


class TestSingleQubitGates:
    """Tests for single-qubit gate operations."""

    def test_hadamard_on_zero(self, single_qubit_state, assert_probability):
        """H|0> = (|0> + |1>)/sqrt(2)."""
        single_qubit_state.h(0)
        assert_probability(single_qubit_state, 0, 0.5)
        assert_probability(single_qubit_state, 1, 0.5)

    def test_hadamard_on_one(self, single_qubit_state, assert_probability):
        """H|1> = (|0> - |1>)/sqrt(2)."""
        single_qubit_state.x(0).h(0)
        assert_probability(single_qubit_state, 0, 0.5)
        assert_probability(single_qubit_state, 1, 0.5)

    def test_hadamard_twice_is_identity(self, single_qubit_state):
        """H^2 = I."""
        single_qubit_state.h(0).h(0)
        sv = single_qubit_state.get_statevector()
        expected = np.array([1, 0], dtype=complex)
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_pauli_x_bit_flip(self, single_qubit_state, assert_probability):
        """X|0> = |1>, X|1> = |0>."""
        single_qubit_state.x(0)
        assert_probability(single_qubit_state, 0, 0.0)
        assert_probability(single_qubit_state, 1, 1.0)

        single_qubit_state.x(0)
        assert_probability(single_qubit_state, 0, 1.0)
        assert_probability(single_qubit_state, 1, 0.0)

    def test_pauli_x_twice_is_identity(self, single_qubit_state):
        """X^2 = I."""
        single_qubit_state.x(0).x(0)
        sv = single_qubit_state.get_statevector()
        expected = np.array([1, 0], dtype=complex)
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_pauli_y(self, single_qubit_state):
        """Y|0> = i|1>."""
        single_qubit_state.y(0)
        sv = single_qubit_state.get_statevector()
        # Y|0> = i|1> (up to global phase conventions)
        assert abs(sv[0]) < 1e-10
        assert abs(abs(sv[1]) - 1.0) < 1e-10

    def test_pauli_y_twice_is_identity(self, single_qubit_state):
        """Y^2 = I."""
        original = single_qubit_state.get_statevector().copy()
        single_qubit_state.y(0).y(0)
        sv = single_qubit_state.get_statevector()
        # Y^2 = I (up to global phase -1)
        np.testing.assert_allclose(np.abs(sv), np.abs(original), atol=1e-10)

    def test_pauli_z_phase_flip(self, single_qubit_state):
        """Z|0> = |0>, Z|1> = -|1>."""
        # Z on |0> should do nothing
        single_qubit_state.z(0)
        sv = single_qubit_state.get_statevector()
        expected = np.array([1, 0], dtype=complex)
        np.testing.assert_allclose(sv, expected, atol=1e-10)

        # Z on |1> should add phase
        single_qubit_state.reset().x(0).z(0)
        sv = single_qubit_state.get_statevector()
        expected = np.array([0, -1], dtype=complex)
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_pauli_z_twice_is_identity(self, single_qubit_state):
        """Z^2 = I."""
        single_qubit_state.h(0)  # Create superposition
        original = single_qubit_state.get_statevector().copy()
        single_qubit_state.z(0).z(0)
        sv = single_qubit_state.get_statevector()
        np.testing.assert_allclose(sv, original, atol=1e-10)

    def test_s_gate(self, single_qubit_state):
        """S is sqrt(Z): S^2 = Z."""
        single_qubit_state.x(0)  # |1>
        single_qubit_state.s(0).s(0)
        sv = single_qubit_state.get_statevector()
        # S^2|1> = Z|1> = -|1>
        expected = np.array([0, -1], dtype=complex)
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_t_gate(self, single_qubit_state):
        """T is sqrt(S): T^2 = S."""
        single_qubit_state.x(0)  # |1>
        single_qubit_state.t(0).t(0)
        # T^2|1> = S|1> = i|1>
        sv = single_qubit_state.get_statevector()
        expected = np.array([0, 1j], dtype=complex)
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_s_dagger(self, single_qubit_state):
        """S.S_dagger = I."""
        single_qubit_state.h(0)
        original = single_qubit_state.get_statevector().copy()
        single_qubit_state.s(0).sdg(0)
        sv = single_qubit_state.get_statevector()
        np.testing.assert_allclose(sv, original, atol=1e-10)

    def test_t_dagger(self, single_qubit_state):
        """T.T_dagger = I."""
        single_qubit_state.h(0)
        original = single_qubit_state.get_statevector().copy()
        single_qubit_state.t(0).tdg(0)
        sv = single_qubit_state.get_statevector()
        np.testing.assert_allclose(sv, original, atol=1e-10)


class TestRotationGates:
    """Tests for parameterized rotation gates."""

    def test_rx_zero_angle(self, single_qubit_state):
        """RX(0) = I."""
        original = single_qubit_state.get_statevector().copy()
        single_qubit_state.rx(0, 0)
        sv = single_qubit_state.get_statevector()
        np.testing.assert_allclose(sv, original, atol=1e-10)

    def test_rx_pi_is_x(self, single_qubit_state, assert_probability):
        """RX(pi) = -iX (equivalent to X up to global phase)."""
        single_qubit_state.rx(0, np.pi)
        # Should flip |0> to |1>
        assert_probability(single_qubit_state, 0, 0.0)
        assert_probability(single_qubit_state, 1, 1.0)

    def test_ry_zero_angle(self, single_qubit_state):
        """RY(0) = I."""
        original = single_qubit_state.get_statevector().copy()
        single_qubit_state.ry(0, 0)
        sv = single_qubit_state.get_statevector()
        np.testing.assert_allclose(sv, original, atol=1e-10)

    def test_ry_pi_is_y(self, single_qubit_state, assert_probability):
        """RY(pi) equivalent to Y up to global phase."""
        single_qubit_state.ry(0, np.pi)
        # Should flip |0> to |1>
        assert_probability(single_qubit_state, 0, 0.0)
        assert_probability(single_qubit_state, 1, 1.0)

    def test_ry_pi_over_2_creates_superposition(self, single_qubit_state, assert_probability):
        """RY(pi/2)|0> creates equal superposition."""
        single_qubit_state.ry(0, np.pi / 2)
        assert_probability(single_qubit_state, 0, 0.5)
        assert_probability(single_qubit_state, 1, 0.5)

    def test_rz_zero_angle(self, single_qubit_state):
        """RZ(0) = I."""
        single_qubit_state.h(0)  # Need superposition to see phase effects
        original = single_qubit_state.get_statevector().copy()
        single_qubit_state.rz(0, 0)
        sv = single_qubit_state.get_statevector()
        np.testing.assert_allclose(sv, original, atol=1e-10)

    def test_rz_pi_is_z(self, single_qubit_state):
        """RZ(pi) = -iZ (equivalent to Z up to global phase)."""
        single_qubit_state.x(0)  # |1>
        single_qubit_state.rz(0, np.pi)
        sv = single_qubit_state.get_statevector()
        # RZ(pi)|1> = -i(-|1>) = i|1> or -|1> depending on convention
        assert abs(sv[0]) < 1e-10
        assert abs(abs(sv[1]) - 1.0) < 1e-10

    def test_phase_gate(self, single_qubit_state):
        """Phase gate adds phase to |1> component."""
        single_qubit_state.x(0)  # |1>
        single_qubit_state.phase(0, np.pi / 4)
        sv = single_qubit_state.get_statevector()
        expected = np.array([0, np.exp(1j * np.pi / 4)], dtype=complex)
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_rotation_periodicity(self, single_qubit_state, random_angles):
        """Rotation by 4*pi should be identity."""
        for angle in random_angles[:3]:
            state = QuantumState(1)
            state.h(0)  # Create superposition
            original = state.get_statevector().copy()

            # Rotate by 4*pi (full rotation twice)
            state.rx(0, 4 * np.pi)
            sv = state.get_statevector()
            np.testing.assert_allclose(sv, original, atol=1e-8)


class TestTwoQubitGates:
    """Tests for two-qubit gate operations."""

    def test_cnot_control_zero(self, two_qubit_state, assert_probability):
        """CNOT with control=0 does nothing."""
        two_qubit_state.cnot(0, 1)
        assert_probability(two_qubit_state, 0, 1.0)

    def test_cnot_control_one(self, two_qubit_state, assert_probability):
        """CNOT with control=1 flips target."""
        two_qubit_state.x(0).cnot(0, 1)  # |10> -> |11>
        assert_probability(two_qubit_state, 3, 1.0)

    def test_cnot_creates_entanglement(self, two_qubit_state, assert_probability):
        """H + CNOT creates Bell state."""
        two_qubit_state.h(0).cnot(0, 1)
        assert_probability(two_qubit_state, 0, 0.5)
        assert_probability(two_qubit_state, 3, 0.5)

    def test_cx_alias(self, two_qubit_state):
        """CX should be alias for CNOT."""
        state1 = QuantumState(2)
        state2 = QuantumState(2)

        state1.h(0).cnot(0, 1)
        state2.h(0).cx(0, 1)

        np.testing.assert_allclose(
            state1.probabilities(),
            state2.probabilities(),
            atol=1e-10
        )

    def test_cz_symmetric(self, two_qubit_state):
        """CZ is symmetric: CZ(a,b) = CZ(b,a)."""
        state1 = QuantumState(2)
        state2 = QuantumState(2)

        state1.h(0).h(1).cz(0, 1)
        state2.h(0).h(1).cz(1, 0)

        np.testing.assert_allclose(
            state1.probabilities(),
            state2.probabilities(),
            atol=1e-10
        )

    def test_cz_phase(self, two_qubit_state):
        """CZ adds pi phase to |11>."""
        two_qubit_state.x(0).x(1)  # |11>
        two_qubit_state.cz(0, 1)
        sv = two_qubit_state.get_statevector()
        expected = np.array([0, 0, 0, -1], dtype=complex)
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_swap(self, two_qubit_state):
        """SWAP exchanges qubit states."""
        two_qubit_state.x(0)  # |10>
        two_qubit_state.swap(0, 1)  # -> |01>
        sv = two_qubit_state.get_statevector()
        expected = np.array([0, 1, 0, 0], dtype=complex)
        np.testing.assert_allclose(sv, expected, atol=1e-10)

    def test_swap_twice_is_identity(self, two_qubit_state):
        """SWAP^2 = I."""
        two_qubit_state.h(0).cnot(0, 1)  # Entangled state
        original = two_qubit_state.get_statevector().copy()
        two_qubit_state.swap(0, 1).swap(0, 1)
        sv = two_qubit_state.get_statevector()
        np.testing.assert_allclose(sv, original, atol=1e-10)

    def test_cphase(self, two_qubit_state):
        """Controlled phase gate adds phase to |11>."""
        two_qubit_state.x(0).x(1)  # |11>
        two_qubit_state.cphase(0, 1, np.pi / 4)
        sv = two_qubit_state.get_statevector()
        expected = np.array([0, 0, 0, np.exp(1j * np.pi / 4)], dtype=complex)
        np.testing.assert_allclose(sv, expected, atol=1e-10)


class TestThreeQubitGates:
    """Tests for three-qubit gate operations."""

    def test_toffoli_both_controls_one(self, three_qubit_state, assert_probability):
        """Toffoli flips target only when both controls are 1."""
        # |110> -> |111>
        three_qubit_state.x(0).x(1)
        three_qubit_state.toffoli(0, 1, 2)
        assert_probability(three_qubit_state, 7, 1.0)  # |111>

    def test_toffoli_one_control_zero(self, three_qubit_state, assert_probability):
        """Toffoli does nothing if any control is 0."""
        # |100> stays |100>
        three_qubit_state.x(0)
        three_qubit_state.toffoli(0, 1, 2)
        assert_probability(three_qubit_state, 1, 1.0)  # |100> = 001 in binary = 1

    def test_toffoli_reversible(self, three_qubit_state):
        """Toffoli is its own inverse."""
        three_qubit_state.h(0).h(1).h(2)
        original = three_qubit_state.get_statevector().copy()
        three_qubit_state.toffoli(0, 1, 2).toffoli(0, 1, 2)
        sv = three_qubit_state.get_statevector()
        np.testing.assert_allclose(sv, original, atol=1e-10)

    def test_ccx_alias(self, three_qubit_state):
        """CCX should be alias for Toffoli."""
        state1 = QuantumState(3)
        state2 = QuantumState(3)

        state1.x(0).x(1).toffoli(0, 1, 2)
        state2.x(0).x(1).ccx(0, 1, 2)

        np.testing.assert_allclose(
            state1.probabilities(),
            state2.probabilities(),
            atol=1e-10
        )


class TestGatesStaticInterface:
    """Tests for the Gates static interface."""

    def test_gates_h(self, two_qubit_state, assert_probability):
        """Gates.H should work like state.h."""
        Gates.H(two_qubit_state, 0)
        assert_probability(two_qubit_state, 0, 0.5)
        assert_probability(two_qubit_state, 1, 0.5)

    def test_gates_x(self, single_qubit_state, assert_probability):
        """Gates.X should work like state.x."""
        Gates.X(single_qubit_state, 0)
        assert_probability(single_qubit_state, 1, 1.0)

    def test_gates_cnot(self, two_qubit_state, assert_probability):
        """Gates.CNOT should work like state.cnot."""
        Gates.H(two_qubit_state, 0)
        Gates.CNOT(two_qubit_state, 0, 1)
        assert_probability(two_qubit_state, 0, 0.5)
        assert_probability(two_qubit_state, 3, 0.5)

    def test_gates_rotations(self, single_qubit_state, assert_probability):
        """Gates rotation methods should work."""
        Gates.RY(single_qubit_state, 0, np.pi / 2)
        assert_probability(single_qubit_state, 0, 0.5)
        assert_probability(single_qubit_state, 1, 0.5)


class TestQubitIndexValidation:
    """Tests for qubit index validation."""

    def test_negative_qubit_index(self, two_qubit_state):
        """Negative qubit index should raise error."""
        with pytest.raises((QuantumError, ValueError, IndexError)):
            two_qubit_state.h(-1)

    def test_qubit_index_out_of_range(self, two_qubit_state):
        """Qubit index >= num_qubits should raise error."""
        with pytest.raises((QuantumError, ValueError, IndexError)):
            two_qubit_state.h(2)

    def test_cnot_same_qubit(self, two_qubit_state):
        """CNOT with same control and target should raise error."""
        with pytest.raises((QuantumError, ValueError)):
            two_qubit_state.cnot(0, 0)
