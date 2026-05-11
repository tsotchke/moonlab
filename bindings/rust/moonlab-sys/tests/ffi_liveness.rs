use moonlab_sys::*;
use std::mem::MaybeUninit;

#[test]
fn state_init_and_free() {
    unsafe {
        let mut state = MaybeUninit::<quantum_state_t>::uninit();
        let result = quantum_state_init(state.as_mut_ptr(), 2);
        assert_eq!(result, 0, "quantum_state_init should succeed");

        let state = state.assume_init_mut();
        assert_eq!(state.num_qubits, 2);
        assert_eq!(state.state_dim, 4);

        quantum_state_free(state);
    }
}

#[test]
fn hadamard_gate() {
    unsafe {
        let mut state = MaybeUninit::<quantum_state_t>::uninit();
        quantum_state_init(state.as_mut_ptr(), 1);
        let state = state.assume_init_mut();

        let result = gate_hadamard(state, 0);
        assert_eq!(result, 0, "gate_hadamard should succeed");

        let prob_0 = measurement_probability_zero(state, 0);
        let prob_1 = measurement_probability_one(state, 0);

        assert!((prob_0 - 0.5).abs() < 1e-10);
        assert!((prob_1 - 0.5).abs() < 1e-10);

        quantum_state_free(state);
    }
}

#[test]
fn bell_state() {
    unsafe {
        let mut state = MaybeUninit::<quantum_state_t>::uninit();
        quantum_state_init(state.as_mut_ptr(), 2);
        let state = state.assume_init_mut();

        gate_hadamard(state, 0);
        gate_cnot(state, 0, 1);

        let subsystem_a: [i32; 1] = [0];
        let entropy = quantum_state_entanglement_entropy(state, subsystem_a.as_ptr(), 1);

        assert!(
            entropy > 0.6,
            "Bell state should have high entanglement: {}",
            entropy
        );

        quantum_state_free(state);
    }
}
