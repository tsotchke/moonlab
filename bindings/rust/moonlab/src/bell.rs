//! Bell-inequality tests: CHSH, Mermin-GHZ, and Mermin-Klyshko
//! (since v0.2.0; safe Rust wrapper since v0.4.7).
//!
//! Wraps `src/algorithms/bell_tests.{c,h}` with idiomatic Rust
//! around the per-test entry points.  Mirrors the v0.4.2 Python
//! `moonlab.algorithms.BellTest` surface.
//!
//! Bell tests require a quantum-entropy context for measurement
//! sampling; the safe wrapper passes `NULL` so the C side falls
//! back to the default v3 QRNG.  Pass a `*mut quantum_entropy_ctx_t`
//! to the low-level FFI directly when a custom entropy source is
//! needed.
//!
//! ## Quick start
//!
//! ```no_run
//! use moonlab::{QuantumState, bell};
//!
//! // CHSH on a Phi+ Bell pair -- expect S ~= 2.8 (Tsirelson bound).
//! let mut state = QuantumState::new(2).unwrap();
//! bell::create_bell_state(&mut state, 0, 1, bell::BellState::PhiPlus).unwrap();
//! let result = bell::chsh_test(&mut state, 0, 1, /*num_measurements=*/10_000).unwrap();
//! assert!(result.chsh_value > 2.5, "S = {}", result.chsh_value);
//! ```

use crate::error::{QuantumError, Result};
use crate::state::QuantumState;
use moonlab_sys::{
    bell_get_optimal_settings, bell_measurement_settings_t, bell_test_chsh,
    bell_test_mermin_ghz, bell_test_mermin_klyshko, bell_test_result_t,
    bell_state_type_t, create_bell_state as ffi_create_bell_state,
    quantum_entropy_ctx_create_hw, quantum_entropy_ctx_destroy,
    quantum_entropy_ctx_t,
};
use std::ptr;

/// RAII guard around a hardware-entropy context.  The Bell-test
/// entry points refuse to run with `entropy == NULL`, so we lease
/// one for the duration of the call and free it on drop.
struct EntropyGuard {
    ctx: *mut quantum_entropy_ctx_t,
}

impl EntropyGuard {
    fn new() -> Result<Self> {
        let ctx = unsafe { quantum_entropy_ctx_create_hw() };
        if ctx.is_null() {
            return Err(QuantumError::Ffi(
                "quantum_entropy_ctx_create_hw returned NULL".to_string(),
            ));
        }
        Ok(Self { ctx })
    }
}

impl Drop for EntropyGuard {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { quantum_entropy_ctx_destroy(self.ctx) };
            self.ctx = ptr::null_mut();
        }
    }
}

/// Bell-state index for [`create_bell_state`].
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u32)]
pub enum BellState {
    /// `|Phi+> = (|00> + |11>) / sqrt(2)` -- the canonical Bell pair.
    PhiPlus = 0,
    /// `|Phi-> = (|00> - |11>) / sqrt(2)`.
    PhiMinus = 1,
    /// `|Psi+> = (|01> + |10>) / sqrt(2)`.
    PsiPlus = 2,
    /// `|Psi-> = (|01> - |10>) / sqrt(2)` -- the singlet.
    PsiMinus = 3,
}

/// Result of a Bell-test run.  Mirrors `bell_test_result_t`.
#[derive(Debug, Clone)]
pub struct BellTestResult {
    /// CHSH `S` (or Mermin `|M|` for the GHZ variant).
    pub chsh_value: f64,
    /// `E(a, b)`.
    pub correlation_ab: f64,
    /// `E(a, b')`.
    pub correlation_ab_prime: f64,
    /// `E(a', b)`.
    pub correlation_a_prime_b: f64,
    /// `E(a', b')`.
    pub correlation_a_prime_b_prime: f64,
    /// Classical (LHV) bound; 2.0 for CHSH, 2.0 for Mermin-GHZ.
    pub classical_bound: f64,
    /// Quantum (Tsirelson) bound; `2 * sqrt(2)` for CHSH, 4.0 for
    /// Mermin-GHZ.
    pub quantum_bound: f64,
    /// One-tail p-value against the classical-bound null.
    pub p_value: f64,
    /// Standard error of `chsh_value`.
    pub standard_error: f64,
    /// Total measurement pairs the test consumed.
    pub measurements: usize,
    /// `true` if `chsh_value > classical_bound`.
    pub violates_classical: bool,
    /// `true` if `chsh_value` is within ~0.05 of `quantum_bound`.
    pub confirms_quantum: bool,
    /// `true` if `p_value < 0.01`.
    pub statistically_significant: bool,
}

impl BellTestResult {
    fn from_c(r: bell_test_result_t) -> Self {
        Self {
            chsh_value: r.chsh_value,
            correlation_ab: r.correlation_ab,
            correlation_ab_prime: r.correlation_ab_prime,
            correlation_a_prime_b: r.correlation_a_prime_b,
            correlation_a_prime_b_prime: r.correlation_a_prime_b_prime,
            classical_bound: r.classical_bound,
            quantum_bound: r.quantum_bound,
            p_value: r.p_value,
            standard_error: r.standard_error,
            measurements: r.measurements,
            violates_classical: r.violates_classical != 0,
            confirms_quantum: r.confirms_quantum != 0,
            statistically_significant: r.statistically_significant != 0,
        }
    }
}

/// Prepare a Bell state on the named qubits in `state`.
pub fn create_bell_state(
    state: &mut QuantumState,
    qubit1: i32,
    qubit2: i32,
    bell: BellState,
) -> Result<()> {
    let rc = unsafe {
        ffi_create_bell_state(
            state.as_ptr(),
            qubit1,
            qubit2,
            bell as bell_state_type_t,
        )
    };
    if rc != 0 {
        Err(QuantumError::Ffi(format!(
            "create_bell_state rc={rc}"
        )))
    } else {
        Ok(())
    }
}

/// Run the CHSH inequality test on `state` at the optimal Tsirelson
/// angles (`a = 0`, `a' = pi/2`, `b = pi/4`, `b' = -pi/4`).
///
/// Returns the full result struct including the `S` parameter, the
/// four correlation terms, the classical / quantum bounds, and the
/// statistical-significance flags.
pub fn chsh_test(
    state: &mut QuantumState,
    qubit_a: i32,
    qubit_b: i32,
    num_measurements: usize,
) -> Result<BellTestResult> {
    let entropy = EntropyGuard::new()?;
    let mut settings = bell_measurement_settings_t {
        angle_a1: 0.0,
        angle_a2: 0.0,
        angle_b1: 0.0,
        angle_b2: 0.0,
    };
    unsafe { bell_get_optimal_settings(&mut settings) };
    let r = unsafe {
        bell_test_chsh(
            state.as_ptr(),
            qubit_a,
            qubit_b,
            num_measurements,
            &settings,
            entropy.ctx,
        )
    };
    drop(entropy);
    Ok(BellTestResult::from_c(r))
}

/// Run the Mermin inequality on a 3-qubit GHZ state at the indicated
/// qubits.  Polynomial `M = <XYY> + <YXY> + <YYX> - <XXX>`; classical
/// bound `|M| <= 2`, quantum bound `|M| = 4`.  The
/// [`BellTestResult`] returned reuses `chsh_value` for `|M|` and the
/// four correlation fields for `{XYY, YXY, YYX, XXX}` in that order.
pub fn mermin_ghz_test(
    state: &mut QuantumState,
    qubit_a: i32,
    qubit_b: i32,
    qubit_c: i32,
    num_measurements: usize,
) -> Result<BellTestResult> {
    let entropy = EntropyGuard::new()?;
    let r = unsafe {
        bell_test_mermin_ghz(
            state.as_ptr(),
            qubit_a,
            qubit_b,
            qubit_c,
            num_measurements,
            entropy.ctx,
        )
    };
    drop(entropy);
    Ok(BellTestResult::from_c(r))
}

/// Run the N-qubit Mermin-Klyshko inequality on the first
/// `num_qubits` qubits of `state`.  Returns the normalised `|M_N|`
/// value such that the classical (LHV) bound is `1.0` and the ideal
/// GHZ quantum value is `2^((N-1)/2)`; for `N = 2` this coincides
/// with `CHSH / (2 sqrt(2))`, for `N = 3` with `Mermin / 4`.
/// Returns `0.0` on argument error.
pub fn mermin_klyshko_test(
    state: &mut QuantumState,
    num_qubits: usize,
    num_measurements: usize,
) -> Result<f64> {
    let v = unsafe {
        bell_test_mermin_klyshko(
            state.as_ptr(),
            num_qubits,
            num_measurements,
            ptr::null_mut(),
        )
    };
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_bell_state_phi_plus_collapses_to_aligned_probs() {
        let mut state = QuantumState::new(2).unwrap();
        create_bell_state(&mut state, 0, 1, BellState::PhiPlus).unwrap();
        let p = state.probabilities();
        // |Phi+> = (|00> + |11>)/sqrt(2) has P(00) = P(11) = 0.5, the
        // other two basis states unreachable.
        assert!((p[0] - 0.5).abs() < 1e-10);
        assert!(p[1].abs() < 1e-10);
        assert!(p[2].abs() < 1e-10);
        assert!((p[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn chsh_test_phi_plus_violates_classical() {
        let mut state = QuantumState::new(2).unwrap();
        create_bell_state(&mut state, 0, 1, BellState::PhiPlus).unwrap();
        let result = chsh_test(&mut state, 0, 1, /*N=*/4000).unwrap();
        // Generous lower bound -- statistics can push us anywhere
        // between 2.5 and 2.83 at this measurement count.
        assert!(
            result.chsh_value > 2.4,
            "CHSH S = {} on |Phi+>; expected > 2.4 (classical = 2.0)",
            result.chsh_value
        );
        assert!(result.violates_classical);
        assert_eq!(result.classical_bound, 2.0);
        assert!((result.quantum_bound - 2.0 * 2f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn mermin_ghz_three_qubit_violates_classical() {
        // |GHZ_3> = (|000> + |111>)/sqrt(2).
        let mut state = QuantumState::new(3).unwrap();
        state.h(0).cnot(0, 1).cnot(0, 2);
        let result = mermin_ghz_test(&mut state, 0, 1, 2, 4000).unwrap();
        assert_eq!(result.classical_bound, 2.0);
        assert_eq!(result.quantum_bound, 4.0);
        // Mermin polynomial |M| reaches 4 on a clean GHZ; 2.5 is a
        // generous floor against statistical noise.
        assert!(
            result.chsh_value.abs() > 2.5,
            "|M| = {} on |GHZ_3>, expected > 2.5",
            result.chsh_value
        );
    }

    #[test]
    fn mermin_klyshko_three_qubit_clears_classical_bound() {
        let mut state = QuantumState::new(3).unwrap();
        state.h(0).cnot(0, 1).cnot(0, 2);
        let mn = mermin_klyshko_test(&mut state, 3, 4000).unwrap();
        // Classical bound after the polynomial normalisation is 1.0;
        // ideal GHZ value is 2.  Cleanly clearing 1.1 is enough to
        // pin behaviour against a wiring regression.
        assert!(
            mn > 1.1,
            "|M_N| = {} on |GHZ_3>, expected > 1.1 (classical = 1.0)",
            mn
        );
    }
}
