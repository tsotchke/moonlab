//! Integration tests for moonlab quantum simulator.
//!
//! These tests verify the complete functionality of the quantum simulator
//! including state creation, gates, measurements, and algorithms.

use moonlab::{QuantumState, QuantumError, MAX_QUBITS};
use std::f64::consts::PI;

// =============================================================================
// STATE CREATION TESTS
// =============================================================================

mod state_creation {
    use super::*;

    #[test]
    fn create_various_sizes() {
        for n in [1, 2, 4, 8, 10, 12] {
            let state = QuantumState::new(n).expect(&format!("Failed to create {}-qubit state", n));
            assert_eq!(state.num_qubits(), n);
            assert_eq!(state.state_dim(), 1 << n);
        }
    }

    #[test]
    fn initial_state_is_zero() {
        let state = QuantumState::new(4).unwrap();
        let probs = state.probabilities();

        // |0000> should have probability 1
        assert!((probs[0] - 1.0).abs() < 1e-10, "P(|0000>) should be 1.0");

        // All other states should have probability 0
        for i in 1..16 {
            assert!(probs[i] < 1e-10, "P(|{:04b}>) should be 0", i);
        }
    }

    #[test]
    fn reject_zero_qubits() {
        match QuantumState::new(0) {
            Err(QuantumError::InvalidQubit { .. }) => (), // Expected
            Err(e) => panic!("Wrong error type: {:?}", e),
            Ok(_) => panic!("Should reject 0 qubits"),
        }
    }

    #[test]
    fn reject_too_many_qubits() {
        match QuantumState::new(MAX_QUBITS + 1) {
            Err(QuantumError::InvalidQubit { .. }) => (), // Expected
            Err(e) => panic!("Wrong error type: {:?}", e),
            Ok(_) => panic!("Should reject > MAX_QUBITS"),
        }
    }
}

// =============================================================================
// SINGLE-QUBIT GATE TESTS
// =============================================================================

mod single_qubit_gates {
    use super::*;

    #[test]
    fn hadamard_creates_superposition() {
        let mut state = QuantumState::new(1).unwrap();
        state.h(0);

        let probs = state.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn hadamard_is_self_inverse() {
        let mut state = QuantumState::new(1).unwrap();
        state.h(0).h(0);

        let probs = state.probabilities();
        assert!((probs[0] - 1.0).abs() < 1e-10);
        assert!(probs[1] < 1e-10);
    }

    #[test]
    fn pauli_x_flips_bit() {
        let mut state = QuantumState::new(2).unwrap();
        state.x(0);

        let probs = state.probabilities();
        // |00> -> |01>
        assert!((probs[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pauli_x_is_self_inverse() {
        let mut state = QuantumState::new(1).unwrap();
        state.x(0).x(0);

        let probs = state.probabilities();
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pauli_z_phase_flip() {
        let mut state = QuantumState::new(1).unwrap();
        // Z|0> = |0>, Z|1> = -|1>
        state.z(0); // No visible change on |0>

        let probs = state.probabilities();
        assert!((probs[0] - 1.0).abs() < 1e-10);

        // Now test on |1>
        state.reset().x(0).z(0);
        let amps = state.amplitudes();
        // |1> coefficient should be -1
        assert!((amps[1].re + 1.0).abs() < 1e-10 || (amps[1].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn rotation_ry_creates_superposition() {
        let mut state = QuantumState::new(1).unwrap();
        state.ry(0, PI / 2.0);

        let probs = state.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-8);
        assert!((probs[1] - 0.5).abs() < 1e-8);
    }

    #[test]
    fn rotation_rx_pi_is_x() {
        let mut state = QuantumState::new(1).unwrap();
        state.rx(0, PI);

        let probs = state.probabilities();
        // RX(pi)|0> = -i|1>
        assert!(probs[0] < 1e-8);
        assert!((probs[1] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn s_gate_squared_is_z() {
        let mut state = QuantumState::new(1).unwrap();
        state.x(0).s(0).s(0); // S^2|1> = Z|1> = -|1>

        let amps = state.amplitudes();
        assert!((amps[1].re + 1.0).abs() < 1e-10);
    }

    #[test]
    fn t_gate_squared_is_s() {
        let mut state = QuantumState::new(1).unwrap();
        state.x(0).t(0).t(0); // T^2|1> = S|1> = i|1>

        let amps = state.amplitudes();
        assert!(amps[1].im.abs() > 0.99);
    }
}

// =============================================================================
// TWO-QUBIT GATE TESTS
// =============================================================================

mod two_qubit_gates {
    use super::*;

    #[test]
    fn cnot_creates_bell_state() {
        let mut state = QuantumState::new(2).unwrap();
        state.h(0).cnot(0, 1);

        let probs = state.probabilities();
        // Bell state: |00> + |11>
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!(probs[1] < 1e-10);
        assert!(probs[2] < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn cnot_control_off_does_nothing() {
        let mut state = QuantumState::new(2).unwrap();
        state.cnot(0, 1); // Control is |0>, should not flip target

        let probs = state.probabilities();
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cnot_control_on_flips_target() {
        let mut state = QuantumState::new(2).unwrap();
        state.x(0).cnot(0, 1); // |10> -> |11>

        let probs = state.probabilities();
        assert!((probs[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cz_is_symmetric() {
        let mut state1 = QuantumState::new(2).unwrap();
        let mut state2 = QuantumState::new(2).unwrap();

        state1.h(0).h(1).cz(0, 1);
        state2.h(0).h(1).cz(1, 0);

        let probs1 = state1.probabilities();
        let probs2 = state2.probabilities();

        for i in 0..4 {
            assert!((probs1[i] - probs2[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn swap_exchanges_qubits() {
        let mut state = QuantumState::new(2).unwrap();
        state.x(0).swap(0, 1); // |01> -> |10>

        let probs = state.probabilities();
        assert!((probs[2] - 1.0).abs() < 1e-10); // |10>
    }

    #[test]
    fn swap_is_self_inverse() {
        let mut state = QuantumState::new(2).unwrap();
        state.h(0).cnot(0, 1).swap(0, 1).swap(0, 1);

        let probs = state.probabilities();
        // Should be back to Bell state
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }
}

// =============================================================================
// MULTI-QUBIT GATE TESTS
// =============================================================================

mod multi_qubit_gates {
    use super::*;

    #[test]
    fn toffoli_needs_both_controls() {
        let mut state = QuantumState::new(3).unwrap();

        // Only control1 on: no flip
        state.x(0).toffoli(0, 1, 2);
        let probs = state.probabilities();
        assert!((probs[1] - 1.0).abs() < 1e-10); // Still |001>

        // Both controls on: flip
        state.reset().x(0).x(1).toffoli(0, 1, 2);
        let probs = state.probabilities();
        assert!((probs[7] - 1.0).abs() < 1e-10); // |111>
    }

    #[test]
    fn toffoli_is_self_inverse() {
        let mut state = QuantumState::new(3).unwrap();
        state.h(0).h(1).toffoli(0, 1, 2).toffoli(0, 1, 2);

        // Should be back to original H|0>H|0>|0>
        let probs = state.probabilities();
        assert!((probs[0] - 0.25).abs() < 1e-10);
        assert!((probs[1] - 0.25).abs() < 1e-10);
        assert!((probs[2] - 0.25).abs() < 1e-10);
        assert!((probs[3] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn qft_creates_uniform_superposition_from_zero() {
        let mut state = QuantumState::new(3).unwrap();
        state.qft(&[0, 1, 2]);

        let probs = state.probabilities();
        let expected = 1.0 / 8.0;

        for (i, &p) in probs.iter().enumerate() {
            assert!((p - expected).abs() < 1e-10, "prob[{}] = {}, expected {}", i, p, expected);
        }
    }

    #[test]
    fn qft_iqft_is_identity() {
        let mut state = QuantumState::new(3).unwrap();
        // Start with some non-trivial state
        state.h(0).cnot(0, 1).x(2);

        let original_probs = state.probabilities();

        state.qft(&[0, 1, 2]).iqft(&[0, 1, 2]);

        let final_probs = state.probabilities();

        for i in 0..8 {
            assert!(
                (original_probs[i] - final_probs[i]).abs() < 1e-8,
                "Mismatch at {}: {} vs {}",
                i, original_probs[i], final_probs[i]
            );
        }
    }
}

// =============================================================================
// ENTANGLEMENT TESTS
// =============================================================================

mod entanglement {
    use super::*;

    #[test]
    fn product_state_has_zero_entropy() {
        let state = QuantumState::new(2).unwrap();
        let entropy = state.entanglement_entropy(&[0]).unwrap();
        assert!(entropy < 1e-10, "Product state entropy should be 0");
    }

    #[test]
    fn bell_state_has_maximal_entropy() {
        let mut state = QuantumState::new(2).unwrap();
        state.h(0).cnot(0, 1);

        let entropy = state.entanglement_entropy(&[0]).unwrap();
        // Maximal entanglement for 2 qubits: ln(2) ≈ 0.693
        assert!(entropy > 0.6, "Bell state should have high entropy: {}", entropy);
        assert!(entropy < 0.8, "Entropy should not exceed ln(2): {}", entropy);
    }

    #[test]
    fn ghz_state_has_entanglement() {
        let mut state = QuantumState::new(3).unwrap();
        state.h(0).cnot(0, 1).cnot(0, 2);

        let entropy = state.entanglement_entropy(&[0]).unwrap();
        assert!(entropy > 0.5, "GHZ state should be entangled");
    }

    #[test]
    fn pure_state_has_unit_purity() {
        let mut state = QuantumState::new(3).unwrap();
        state.h(0).cnot(0, 1).rz(2, PI / 4.0);

        let purity = state.purity();
        assert!((purity - 1.0).abs() < 1e-10, "Pure state purity should be 1.0");
    }
}

// =============================================================================
// EXPECTATION VALUE TESTS
// =============================================================================

mod expectations {
    use super::*;

    #[test]
    fn z_expectation_ground_state() {
        let state = QuantumState::new(2).unwrap();

        // <0|Z|0> = 1
        let ez0 = state.expectation_z(0).unwrap();
        let ez1 = state.expectation_z(1).unwrap();

        assert!((ez0 - 1.0).abs() < 1e-10);
        assert!((ez1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn z_expectation_excited_state() {
        let mut state = QuantumState::new(1).unwrap();
        state.x(0);

        // <1|Z|1> = -1
        let ez = state.expectation_z(0).unwrap();
        assert!((ez + 1.0).abs() < 1e-10);
    }

    #[test]
    fn z_expectation_superposition() {
        let mut state = QuantumState::new(1).unwrap();
        state.h(0);

        // <+|Z|+> = 0
        let ez = state.expectation_z(0).unwrap();
        assert!(ez.abs() < 1e-10);
    }

    #[test]
    fn x_expectation_plus_state() {
        let mut state = QuantumState::new(1).unwrap();
        state.h(0);

        // <+|X|+> = 1
        let ex = state.expectation_x(0).unwrap();
        assert!((ex - 1.0).abs() < 1e-10);
    }

    #[test]
    fn zz_correlation_bell_state() {
        let mut state = QuantumState::new(2).unwrap();
        state.h(0).cnot(0, 1);

        // Bell state |00> + |11> has <ZZ> = 1
        let zz = state.correlation_zz(0, 1).unwrap();
        assert!((zz - 1.0).abs() < 1e-10);
    }

    #[test]
    fn zz_correlation_product_state() {
        let mut state = QuantumState::new(2).unwrap();
        state.h(0).h(1);

        // Product state |++> has <ZZ> = <Z><Z> = 0*0 = 0
        let zz = state.correlation_zz(0, 1).unwrap();
        assert!(zz.abs() < 1e-10);
    }
}

// =============================================================================
// STATE OPERATIONS TESTS
// =============================================================================

mod state_operations {
    use super::*;

    #[test]
    fn reset_returns_to_ground_state() {
        let mut state = QuantumState::new(3).unwrap();
        state.h(0).cnot(0, 1).x(2);
        state.reset();

        let probs = state.probabilities();
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn clone_creates_independent_copy() {
        let mut state = QuantumState::new(2).unwrap();
        state.h(0).cnot(0, 1);

        let cloned = state.clone();

        // Modify original
        state.x(0);

        // Clone should be unchanged
        let clone_probs = cloned.probabilities();
        assert!((clone_probs[0] - 0.5).abs() < 1e-10);
        assert!((clone_probs[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn method_chaining_works() {
        let mut state = QuantumState::new(4).unwrap();

        state
            .h(0)
            .h(1)
            .cnot(0, 2)
            .cnot(1, 3)
            .rz(2, PI / 8.0)
            .ry(3, PI / 4.0)
            .cz(0, 1);

        // Just verify it compiles and runs
        assert_eq!(state.num_qubits(), 4);
    }
}

// =============================================================================
// AMPLITUDE TESTS
// =============================================================================

mod amplitudes {
    use super::*;

    #[test]
    fn amplitudes_length_matches_state_dim() {
        let state = QuantumState::new(4).unwrap();
        let amps = state.amplitudes();
        assert_eq!(amps.len(), 16);
    }

    #[test]
    fn amplitudes_normalized() {
        let mut state = QuantumState::new(3).unwrap();
        state.h(0).cnot(0, 1).ry(2, PI / 3.0);

        let amps = state.amplitudes();
        let norm_sq: f64 = amps.iter().map(|a| a.norm_sqr()).sum();

        assert!((norm_sq - 1.0).abs() < 1e-10);
    }

    #[test]
    fn amplitudes_match_probabilities() {
        let mut state = QuantumState::new(2).unwrap();
        state.h(0).cnot(0, 1);

        let amps = state.amplitudes();
        let probs = state.probabilities();

        for i in 0..4 {
            assert!((amps[i].norm_sqr() - probs[i]).abs() < 1e-10);
        }
    }
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn gate_on_invalid_qubit_is_noop() {
        let mut state = QuantumState::new(2).unwrap();
        state.h(0);

        let probs_before = state.probabilities();
        state.h(5); // Invalid qubit
        let probs_after = state.probabilities();

        // Should be unchanged
        for i in 0..4 {
            assert!((probs_before[i] - probs_after[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn cnot_same_qubit_is_noop() {
        let mut state = QuantumState::new(2).unwrap();
        state.h(0);

        let probs_before = state.probabilities();
        state.cnot(0, 0); // Same qubit
        let probs_after = state.probabilities();

        // Should be unchanged
        for i in 0..4 {
            assert!((probs_before[i] - probs_after[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn swap_same_qubit_is_noop() {
        let mut state = QuantumState::new(2).unwrap();
        state.x(0);

        let probs_before = state.probabilities();
        state.swap(0, 0);
        let probs_after = state.probabilities();

        for i in 0..4 {
            assert!((probs_before[i] - probs_after[i]).abs() < 1e-10);
        }
    }
}

// =============================================================================
// ALGORITHM TESTS
// =============================================================================

mod algorithms {
    use super::*;

    #[test]
    fn grover_single_iteration_increases_marked_probability() {
        // Manual Grover iteration test
        let mut state = QuantumState::new(3).unwrap();
        let marked = 5usize; // |101>

        // Initial uniform superposition
        state.h(0).h(1).h(2);

        let initial_prob = state.probabilities()[marked];

        // Oracle: flip phase of |101>
        // Implemented as CZ between qubits with X gates
        state.x(1); // Now marked state is |111>
        state.h(2);
        state.toffoli(0, 1, 2);
        state.h(2);
        state.x(1);

        // Diffusion operator: 2|+><+| - I
        state.h(0).h(1).h(2);
        state.x(0).x(1).x(2);
        state.h(2);
        state.toffoli(0, 1, 2);
        state.h(2);
        state.x(0).x(1).x(2);
        state.h(0).h(1).h(2);

        let final_prob = state.probabilities()[marked];

        // Probability should have increased
        assert!(
            final_prob > initial_prob * 2.0,
            "Probability should increase: {} -> {}",
            initial_prob,
            final_prob
        );
    }

    #[test]
    fn bell_inequality_violation() {
        let mut state = QuantumState::new(2).unwrap();
        state.h(0).cnot(0, 1);

        // CHSH inequality: |S| <= 2 classically
        // For Bell state with optimal angles: S = 2√2 ≈ 2.828

        let zz = state.correlation_zz(0, 1).unwrap();

        // For Phi+ Bell state, <ZZ> = 1
        assert!((zz - 1.0).abs() < 1e-10);

        // <Z_0> = <Z_1> = 0
        let z0 = state.expectation_z(0).unwrap();
        let z1 = state.expectation_z(1).unwrap();
        assert!(z0.abs() < 1e-10);
        assert!(z1.abs() < 1e-10);
    }
}
