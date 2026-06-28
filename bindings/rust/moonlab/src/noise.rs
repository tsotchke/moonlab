//! Single-qubit Kraus noise channels (C-side since v0.2.1, Rust
//! wrapper since v0.4.8).
//!
//! Wraps `src/quantum/noise.{c,h}` with idiomatic Rust around the
//! per-channel sampler entry points.  Mirrors the v0.2.1 Python
//! `moonlab.noise` surface.
//!
//! Each channel takes `(state, qubit, ..., random_value)`.  The
//! caller supplies the uniform-`[0, 1)` random sample; no
//! `quantum_entropy_ctx_t` is needed because the Kraus selection is
//! decoupled from the sampling primitive.  This makes the channels
//! cheap to use in test benches and deterministic when fed a seeded
//! RNG.
//!
//! ## Quick start
//!
//! ```no_run
//! use moonlab::{QuantumState, noise};
//!
//! // 1% depolarising noise on qubit 0 of a fresh state.
//! let mut state = QuantumState::new(2).unwrap();
//! state.h(0);
//! noise::depolarizing_single(&mut state, 0, /*p=*/0.01, /*r=*/0.42);
//! ```

use crate::error::{QuantumError, Result};
use crate::state::QuantumState;
use moonlab_sys as ffi;

fn check_qubit(state: &QuantumState, q: usize) -> Result<()> {
    if q >= state.num_qubits() {
        return Err(QuantumError::InvalidQubit {
            index: q,
            max: state.num_qubits(),
        });
    }
    Ok(())
}

/// Single-qubit depolarising channel
/// `rho |-> (1 - p) rho + (p / 3) (X rho X + Y rho Y + Z rho Z)`.
/// `random_value` selects which Pauli (if any) to apply.
pub fn depolarizing_single(
    state: &mut QuantumState,
    qubit: usize,
    probability: f64,
    random_value: f64,
) -> Result<()> {
    check_qubit(state, qubit)?;
    unsafe {
        ffi::noise_depolarizing_single(
            state.as_ptr(),
            qubit as i32,
            probability,
            random_value,
        );
    }
    Ok(())
}

/// Two-qubit depolarising channel acting on the pair `(qubit1, qubit2)`.
pub fn depolarizing_two_qubit(
    state: &mut QuantumState,
    qubit1: usize,
    qubit2: usize,
    probability: f64,
    random_value: f64,
) -> Result<()> {
    check_qubit(state, qubit1)?;
    check_qubit(state, qubit2)?;
    unsafe {
        ffi::noise_depolarizing_two_qubit(
            state.as_ptr(),
            qubit1 as i32,
            qubit2 as i32,
            probability,
            random_value,
        );
    }
    Ok(())
}

/// Amplitude-damping (T1) channel with damping parameter `gamma`.
pub fn amplitude_damping(
    state: &mut QuantumState,
    qubit: usize,
    gamma: f64,
    random_value: f64,
) -> Result<()> {
    check_qubit(state, qubit)?;
    unsafe {
        ffi::noise_amplitude_damping(
            state.as_ptr(),
            qubit as i32,
            gamma,
            random_value,
        );
    }
    Ok(())
}

/// Phase-damping (pure-T2) channel with damping parameter `gamma`.
pub fn phase_damping(
    state: &mut QuantumState,
    qubit: usize,
    gamma: f64,
    random_value: f64,
) -> Result<()> {
    check_qubit(state, qubit)?;
    unsafe {
        ffi::noise_phase_damping(
            state.as_ptr(),
            qubit as i32,
            gamma,
            random_value,
        );
    }
    Ok(())
}

/// Pure-dephasing channel with phase variance `sigma`.  `random_phase`
/// is a uniform-`[0, 1)` sample used to draw the random phase angle.
pub fn pure_dephasing(
    state: &mut QuantumState,
    qubit: usize,
    sigma: f64,
    random_phase: f64,
) -> Result<()> {
    check_qubit(state, qubit)?;
    unsafe {
        ffi::noise_pure_dephasing(
            state.as_ptr(),
            qubit as i32,
            sigma,
            random_phase,
        );
    }
    Ok(())
}

/// Bit-flip channel `(1 - p) I + p X`.
pub fn bit_flip(
    state: &mut QuantumState,
    qubit: usize,
    probability: f64,
    random_value: f64,
) -> Result<()> {
    check_qubit(state, qubit)?;
    unsafe {
        ffi::noise_bit_flip(
            state.as_ptr(),
            qubit as i32,
            probability,
            random_value,
        );
    }
    Ok(())
}

/// Phase-flip channel `(1 - p) I + p Z`.
pub fn phase_flip(
    state: &mut QuantumState,
    qubit: usize,
    probability: f64,
    random_value: f64,
) -> Result<()> {
    check_qubit(state, qubit)?;
    unsafe {
        ffi::noise_phase_flip(
            state.as_ptr(),
            qubit as i32,
            probability,
            random_value,
        );
    }
    Ok(())
}

/// Bit-phase-flip channel `(1 - p) I + p Y`.
pub fn bit_phase_flip(
    state: &mut QuantumState,
    qubit: usize,
    probability: f64,
    random_value: f64,
) -> Result<()> {
    check_qubit(state, qubit)?;
    unsafe {
        ffi::noise_bit_phase_flip(
            state.as_ptr(),
            qubit as i32,
            probability,
            random_value,
        );
    }
    Ok(())
}

/// Thermal-relaxation channel combining amplitude-damping at rate `1
/// / t1` and phase-damping at rate `1 / t2` over duration `time`.
/// `random_values` must provide the two uniform-`[0, 1)` samples the
/// underlying combined-Kraus sampler consumes.
pub fn thermal_relaxation(
    state: &mut QuantumState,
    qubit: usize,
    t1: f64,
    t2: f64,
    time: f64,
    random_values: &[f64],
) -> Result<()> {
    check_qubit(state, qubit)?;
    if random_values.len() < 2 {
        return Err(QuantumError::Ffi(format!(
            "thermal_relaxation needs at least 2 random_values, got {}",
            random_values.len()
        )));
    }
    unsafe {
        ffi::noise_thermal_relaxation(
            state.as_ptr(),
            qubit as i32,
            t1,
            t2,
            time,
            random_values.as_ptr(),
        );
    }
    Ok(())
}

/// Classical readout-error model: flip the measurement outcome with
/// probability `error_0_to_1` (when the true outcome is 0) or
/// `error_1_to_0` (when 1).  Returns the noisy outcome.
pub fn readout_error(
    outcome: bool,
    error_0_to_1: f64,
    error_1_to_0: f64,
    random_value: f64,
) -> bool {
    let v = unsafe {
        ffi::noise_readout_error(
            outcome as i32,
            error_0_to_1,
            error_1_to_0,
            random_value,
        )
    };
    v != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depolarizing_single_zero_probability_is_identity() {
        let mut state = QuantumState::new(2).unwrap();
        state.h(0);
        let p_before = state.probabilities();
        depolarizing_single(&mut state, 0, 0.0, 0.5).unwrap();
        let p_after = state.probabilities();
        for (a, b) in p_before.iter().zip(p_after.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn bit_flip_full_probability_flips_qubit() {
        // |0> --bit_flip(p=1)--> |1>; only basis state 1 should
        // have probability 1 afterwards.
        let mut state = QuantumState::new(1).unwrap();
        bit_flip(&mut state, 0, 1.0, 0.0).unwrap();
        let p = state.probabilities();
        assert!(p[0].abs() < 1e-12);
        assert!((p[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn phase_flip_full_probability_preserves_basis_probabilities() {
        // Z on |+> goes to |->, but the computational-basis
        // probabilities stay (0.5, 0.5).
        let mut state = QuantumState::new(1).unwrap();
        state.h(0);
        phase_flip(&mut state, 0, 1.0, 0.0).unwrap();
        let p = state.probabilities();
        assert!((p[0] - 0.5).abs() < 1e-12);
        assert!((p[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn amplitude_damping_eventually_collapses_excited_state() {
        // |1> with full damping (gamma = 1) -- the excited state
        // decays to |0>.  The C side selects the K_1 (decay) Kraus
        // when random_value < gamma.
        let mut state = QuantumState::new(1).unwrap();
        state.x(0);
        amplitude_damping(&mut state, 0, 1.0, 0.5).unwrap();
        let p = state.probabilities();
        assert!((p[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_out_of_range_qubit() {
        let mut state = QuantumState::new(2).unwrap();
        assert!(depolarizing_single(&mut state, 5, 0.1, 0.5).is_err());
        assert!(bit_flip(&mut state, 5, 0.1, 0.5).is_err());
    }

    #[test]
    fn readout_error_threshold_pins_behaviour() {
        // P(flip | true=0) = 0.1; r=0.05 < 0.1 -> flip, r=0.5 -> keep.
        assert_eq!(readout_error(false, 0.1, 0.1, 0.05), true);
        assert_eq!(readout_error(false, 0.1, 0.1, 0.5), false);
    }

    #[test]
    fn mutual_info_bell_pair_equals_one_bit_per_subsystem() {
        // |Phi+> = (|00> + |11>) / sqrt 2.  I(A:B) = 2 * S(A) = 2 bits.
        let mut state = QuantumState::new(2).unwrap();
        state.h(0).cnot(0, 1);
        let i_ab = state.mutual_information(&[0], &[1]).unwrap();
        assert!(
            (i_ab - 2.0).abs() < 1e-6,
            "I(A:B) = {} on |Phi+>; expected 2 bits",
            i_ab
        );
    }
}
