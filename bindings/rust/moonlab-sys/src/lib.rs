//! Low-level FFI bindings to the Moonlab quantum simulator.
//!
//! This crate provides raw, unsafe bindings to the C library.
//! For a safe, idiomatic Rust API, use the `moonlab` crate instead.
//!
//! # Safety
//!
//! All functions in this crate are unsafe and require careful handling
//! of pointers and memory management. The safe wrappers in `moonlab`
//! handle this for you.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::all)]

// Include the generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::MaybeUninit;

    #[test]
    fn test_state_init_and_free() {
        unsafe {
            let mut state = MaybeUninit::<quantum_state_t>::uninit();
            let result = quantum_state_init(state.as_mut_ptr(), 2);
            assert_eq!(result, 0, "quantum_state_init should succeed");

            let state = state.assume_init_mut();
            assert_eq!(state.num_qubits, 2);
            assert_eq!(state.state_dim, 4); // 2^2 = 4

            quantum_state_free(state);
        }
    }

    #[test]
    fn test_hadamard_gate() {
        unsafe {
            let mut state = MaybeUninit::<quantum_state_t>::uninit();
            quantum_state_init(state.as_mut_ptr(), 1);
            let state = state.assume_init_mut();

            // Apply Hadamard to qubit 0
            let result = gate_hadamard(state, 0);
            assert_eq!(result, 0, "gate_hadamard should succeed");

            // State should now be in superposition
            // |0⟩ -> (|0⟩ + |1⟩)/√2
            let prob_0 = measurement_probability_zero(state, 0);
            let prob_1 = measurement_probability_one(state, 0);

            // Both probabilities should be ~0.5
            assert!((prob_0 - 0.5).abs() < 1e-10);
            assert!((prob_1 - 0.5).abs() < 1e-10);

            quantum_state_free(state);
        }
    }

    #[test]
    fn test_bell_state() {
        unsafe {
            let mut state = MaybeUninit::<quantum_state_t>::uninit();
            quantum_state_init(state.as_mut_ptr(), 2);
            let state = state.assume_init_mut();

            // Create Bell state: H|0⟩ then CNOT
            gate_hadamard(state, 0);
            gate_cnot(state, 0, 1);

            // Should have 50% chance of |00⟩ and 50% chance of |11⟩
            // No chance of |01⟩ or |10⟩
            // Subsystem A = qubit 0
            let subsystem_a: [i32; 1] = [0];
            let entropy = quantum_state_entanglement_entropy(
                state,
                subsystem_a.as_ptr(),
                1
            );

            // Bell state should have maximal entanglement entropy = ln(2) ≈ 0.693
            assert!(entropy > 0.6, "Bell state should have high entanglement: {}", entropy);

            quantum_state_free(state);
        }
    }
}
