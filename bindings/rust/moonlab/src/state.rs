//! Quantum state representation and manipulation.
//!
//! The [`QuantumState`] type provides a safe, RAII-managed wrapper around
//! the C quantum state, with automatic memory cleanup via Drop.
//!
//! # Example
//!
//! ```no_run
//! use moonlab::QuantumState;
//!
//! // Create a 3-qubit state initialized to |000⟩
//! let mut state = QuantumState::new(3).unwrap();
//!
//! // Apply gates with method chaining
//! state.h(0)    // Hadamard on qubit 0
//!      .cnot(0, 1)  // CNOT with control=0, target=1
//!      .cnot(0, 2); // CNOT with control=0, target=2
//!
//! // Now we have the GHZ state: (|000⟩ + |111⟩)/√2
//! let probs = state.probabilities();
//! // probs[0] ≈ 0.5 (|000⟩)
//! // probs[7] ≈ 0.5 (|111⟩)
//! ```

use std::ptr::NonNull;
use std::mem::MaybeUninit;
use num_complex::Complex64;
use moonlab_sys as ffi;
use crate::error::{QuantumError, Result};

/// Maximum number of qubits supported by the simulator.
pub const MAX_QUBITS: usize = 32;

/// A quantum state of N qubits.
///
/// This type provides safe access to quantum state operations including:
/// - Single-qubit gates (X, Y, Z, H, S, T, rotations)
/// - Two-qubit gates (CNOT, CZ, SWAP, controlled rotations)
/// - Multi-qubit gates (Toffoli, Fredkin, QFT)
/// - Measurements and probability queries
///
/// # Memory Management
///
/// The state is automatically freed when dropped, ensuring no memory leaks.
/// Clone creates a deep copy of the quantum state.
#[derive(Debug)]
pub struct QuantumState {
    /// Pointer to the C quantum_state_t structure.
    inner: NonNull<ffi::quantum_state_t>,
    /// Number of qubits (cached for quick access).
    num_qubits: usize,
}

// Safety: The underlying C state is not thread-safe, but we only allow
// mutable access through &mut self, making single-threaded use safe.
// For multi-threaded use, wrap in Arc<Mutex<QuantumState>>.
unsafe impl Send for QuantumState {}

impl QuantumState {
    /// Create a new quantum state initialized to |0...0⟩.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - Number of qubits (1-32)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `num_qubits` is 0 or greater than 32
    /// - Memory allocation fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use moonlab::QuantumState;
    ///
    /// let state = QuantumState::new(4).unwrap();
    /// assert_eq!(state.num_qubits(), 4);
    /// assert_eq!(state.state_dim(), 16); // 2^4
    /// ```
    pub fn new(num_qubits: usize) -> Result<Self> {
        if num_qubits == 0 || num_qubits > MAX_QUBITS {
            return Err(QuantumError::InvalidQubit {
                index: num_qubits,
                max: MAX_QUBITS,
            });
        }

        unsafe {
            let mut state = MaybeUninit::<ffi::quantum_state_t>::uninit();
            let result = ffi::quantum_state_init(state.as_mut_ptr(), num_qubits);

            if result != 0 {
                return Err(QuantumError::AllocationFailed(num_qubits));
            }

            let state_ptr = Box::into_raw(Box::new(state.assume_init()));

            Ok(Self {
                inner: NonNull::new(state_ptr).ok_or(QuantumError::NullPointer)?,
                num_qubits,
            })
        }
    }

    /// Get the number of qubits in this state.
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the dimension of the state vector (2^n).
    #[inline]
    pub fn state_dim(&self) -> usize {
        1 << self.num_qubits
    }

    /// Get a reference to the underlying state pointer.
    #[inline]
    fn as_ptr(&self) -> *mut ffi::quantum_state_t {
        self.inner.as_ptr()
    }

    /// Check if a qubit index is valid.
    fn check_qubit(&self, qubit: usize) -> Result<()> {
        if qubit >= self.num_qubits {
            Err(QuantumError::InvalidQubit {
                index: qubit,
                max: self.num_qubits,
            })
        } else {
            Ok(())
        }
    }

    /// Get the probability of measuring each basis state.
    ///
    /// Returns a vector of length 2^n where entry i is the probability
    /// of measuring the state |i⟩ (in binary).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use moonlab::QuantumState;
    ///
    /// let mut state = QuantumState::new(2).unwrap();
    /// state.h(0);  // Create superposition on qubit 0
    ///
    /// let probs = state.probabilities();
    /// // probs[0] ≈ 0.5 (|00⟩)
    /// // probs[1] ≈ 0.5 (|01⟩)  Note: qubit 0 is LSB
    /// ```
    pub fn probabilities(&self) -> Vec<f64> {
        let dim = self.state_dim();
        let mut probs = vec![0.0; dim];

        unsafe {
            let state = self.inner.as_ref();
            for i in 0..dim {
                let amp = *state.amplitudes.add(i);
                let re = amp.re;
                let im = amp.im;
                probs[i] = re * re + im * im;
            }
        }

        probs
    }

    /// Get the complex amplitudes of the state vector.
    ///
    /// Returns a vector of length 2^n containing the amplitude for each
    /// computational basis state.
    pub fn amplitudes(&self) -> Vec<Complex64> {
        let dim = self.state_dim();
        let mut amps = Vec::with_capacity(dim);

        unsafe {
            let state = self.inner.as_ref();
            for i in 0..dim {
                let amp = *state.amplitudes.add(i);
                amps.push(Complex64::new(amp.re, amp.im));
            }
        }

        amps
    }

    /// Get the probability of measuring qubit in state |0⟩.
    pub fn prob_zero(&self, qubit: usize) -> Result<f64> {
        self.check_qubit(qubit)?;
        unsafe {
            Ok(ffi::measurement_probability_zero(self.as_ptr(), qubit as i32))
        }
    }

    /// Get the probability of measuring qubit in state |1⟩.
    pub fn prob_one(&self, qubit: usize) -> Result<f64> {
        self.check_qubit(qubit)?;
        unsafe {
            Ok(ffi::measurement_probability_one(self.as_ptr(), qubit as i32))
        }
    }

    /// Compute the entanglement entropy for a bipartition.
    ///
    /// The subsystem A is defined by the given qubit indices.
    /// Returns the von Neumann entropy S(ρ_A) in bits.
    pub fn entanglement_entropy(&self, subsystem_a: &[usize]) -> Result<f64> {
        for &q in subsystem_a {
            self.check_qubit(q)?;
        }

        let indices: Vec<i32> = subsystem_a.iter().map(|&q| q as i32).collect();

        unsafe {
            Ok(ffi::quantum_state_entanglement_entropy(
                self.as_ptr(),
                indices.as_ptr(),
                indices.len(),
            ))
        }
    }

    /// Compute the purity of the state: Tr(ρ²).
    ///
    /// Returns 1.0 for pure states, < 1.0 for mixed states.
    pub fn purity(&self) -> f64 {
        unsafe { ffi::quantum_state_purity(self.as_ptr()) }
    }

    /// Compute the von Neumann entropy of the full state.
    pub fn entropy(&self) -> f64 {
        unsafe { ffi::quantum_state_entropy(self.as_ptr()) }
    }

    /// Reset the state to |0...0⟩.
    pub fn reset(&mut self) -> &mut Self {
        unsafe {
            ffi::quantum_state_reset(self.as_ptr());
        }
        self
    }

    // ========================================================================
    // SINGLE-QUBIT GATES
    // ========================================================================

    /// Apply Pauli-X (NOT) gate to a qubit.
    pub fn x(&mut self, qubit: usize) -> &mut Self {
        if self.check_qubit(qubit).is_ok() {
            unsafe { ffi::gate_pauli_x(self.as_ptr(), qubit as i32); }
        }
        self
    }

    /// Apply Pauli-Y gate to a qubit.
    pub fn y(&mut self, qubit: usize) -> &mut Self {
        if self.check_qubit(qubit).is_ok() {
            unsafe { ffi::gate_pauli_y(self.as_ptr(), qubit as i32); }
        }
        self
    }

    /// Apply Pauli-Z gate to a qubit.
    pub fn z(&mut self, qubit: usize) -> &mut Self {
        if self.check_qubit(qubit).is_ok() {
            unsafe { ffi::gate_pauli_z(self.as_ptr(), qubit as i32); }
        }
        self
    }

    /// Apply Hadamard gate to a qubit.
    ///
    /// Creates superposition: H|0⟩ = (|0⟩ + |1⟩)/√2
    pub fn h(&mut self, qubit: usize) -> &mut Self {
        if self.check_qubit(qubit).is_ok() {
            unsafe { ffi::gate_hadamard(self.as_ptr(), qubit as i32); }
        }
        self
    }

    /// Apply S gate (phase gate, √Z) to a qubit.
    pub fn s(&mut self, qubit: usize) -> &mut Self {
        if self.check_qubit(qubit).is_ok() {
            unsafe { ffi::gate_s(self.as_ptr(), qubit as i32); }
        }
        self
    }

    /// Apply S† gate (inverse of S) to a qubit.
    pub fn sdg(&mut self, qubit: usize) -> &mut Self {
        if self.check_qubit(qubit).is_ok() {
            unsafe { ffi::gate_s_dagger(self.as_ptr(), qubit as i32); }
        }
        self
    }

    /// Apply T gate (π/8 gate, √S) to a qubit.
    pub fn t(&mut self, qubit: usize) -> &mut Self {
        if self.check_qubit(qubit).is_ok() {
            unsafe { ffi::gate_t(self.as_ptr(), qubit as i32); }
        }
        self
    }

    /// Apply T† gate (inverse of T) to a qubit.
    pub fn tdg(&mut self, qubit: usize) -> &mut Self {
        if self.check_qubit(qubit).is_ok() {
            unsafe { ffi::gate_t_dagger(self.as_ptr(), qubit as i32); }
        }
        self
    }

    /// Apply rotation around X-axis by angle theta.
    pub fn rx(&mut self, qubit: usize, theta: f64) -> &mut Self {
        if self.check_qubit(qubit).is_ok() {
            unsafe { ffi::gate_rx(self.as_ptr(), qubit as i32, theta); }
        }
        self
    }

    /// Apply rotation around Y-axis by angle theta.
    pub fn ry(&mut self, qubit: usize, theta: f64) -> &mut Self {
        if self.check_qubit(qubit).is_ok() {
            unsafe { ffi::gate_ry(self.as_ptr(), qubit as i32, theta); }
        }
        self
    }

    /// Apply rotation around Z-axis by angle theta.
    pub fn rz(&mut self, qubit: usize, theta: f64) -> &mut Self {
        if self.check_qubit(qubit).is_ok() {
            unsafe { ffi::gate_rz(self.as_ptr(), qubit as i32, theta); }
        }
        self
    }

    /// Apply phase gate with custom phase angle.
    pub fn phase(&mut self, qubit: usize, phi: f64) -> &mut Self {
        if self.check_qubit(qubit).is_ok() {
            unsafe { ffi::gate_phase(self.as_ptr(), qubit as i32, phi); }
        }
        self
    }

    /// Apply arbitrary single-qubit unitary U3(θ, φ, λ).
    pub fn u3(&mut self, qubit: usize, theta: f64, phi: f64, lambda: f64) -> &mut Self {
        if self.check_qubit(qubit).is_ok() {
            unsafe { ffi::gate_u3(self.as_ptr(), qubit as i32, theta, phi, lambda); }
        }
        self
    }

    // ========================================================================
    // TWO-QUBIT GATES
    // ========================================================================

    /// Apply CNOT (controlled-X) gate.
    ///
    /// Flips target if control is |1⟩.
    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        if self.check_qubit(control).is_ok() && self.check_qubit(target).is_ok() {
            if control != target {
                unsafe { ffi::gate_cnot(self.as_ptr(), control as i32, target as i32); }
            }
        }
        self
    }

    /// Apply CX gate (alias for CNOT).
    #[inline]
    pub fn cx(&mut self, control: usize, target: usize) -> &mut Self {
        self.cnot(control, target)
    }

    /// Apply CZ (controlled-Z) gate.
    pub fn cz(&mut self, control: usize, target: usize) -> &mut Self {
        if self.check_qubit(control).is_ok() && self.check_qubit(target).is_ok() {
            if control != target {
                unsafe { ffi::gate_cz(self.as_ptr(), control as i32, target as i32); }
            }
        }
        self
    }

    /// Apply CY (controlled-Y) gate.
    pub fn cy(&mut self, control: usize, target: usize) -> &mut Self {
        if self.check_qubit(control).is_ok() && self.check_qubit(target).is_ok() {
            if control != target {
                unsafe { ffi::gate_cy(self.as_ptr(), control as i32, target as i32); }
            }
        }
        self
    }

    /// Apply SWAP gate.
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) -> &mut Self {
        if self.check_qubit(qubit1).is_ok() && self.check_qubit(qubit2).is_ok() {
            if qubit1 != qubit2 {
                unsafe { ffi::gate_swap(self.as_ptr(), qubit1 as i32, qubit2 as i32); }
            }
        }
        self
    }

    /// Apply controlled-RX rotation.
    pub fn crx(&mut self, control: usize, target: usize, theta: f64) -> &mut Self {
        if self.check_qubit(control).is_ok() && self.check_qubit(target).is_ok() {
            if control != target {
                unsafe { ffi::gate_crx(self.as_ptr(), control as i32, target as i32, theta); }
            }
        }
        self
    }

    /// Apply controlled-RY rotation.
    pub fn cry(&mut self, control: usize, target: usize, theta: f64) -> &mut Self {
        if self.check_qubit(control).is_ok() && self.check_qubit(target).is_ok() {
            if control != target {
                unsafe { ffi::gate_cry(self.as_ptr(), control as i32, target as i32, theta); }
            }
        }
        self
    }

    /// Apply controlled-RZ rotation.
    pub fn crz(&mut self, control: usize, target: usize, theta: f64) -> &mut Self {
        if self.check_qubit(control).is_ok() && self.check_qubit(target).is_ok() {
            if control != target {
                unsafe { ffi::gate_crz(self.as_ptr(), control as i32, target as i32, theta); }
            }
        }
        self
    }

    /// Apply controlled phase gate.
    pub fn cphase(&mut self, control: usize, target: usize, phi: f64) -> &mut Self {
        if self.check_qubit(control).is_ok() && self.check_qubit(target).is_ok() {
            if control != target {
                unsafe { ffi::gate_cphase(self.as_ptr(), control as i32, target as i32, phi); }
            }
        }
        self
    }

    // ========================================================================
    // MULTI-QUBIT GATES
    // ========================================================================

    /// Apply Toffoli (CCNOT) gate.
    pub fn toffoli(&mut self, control1: usize, control2: usize, target: usize) -> &mut Self {
        if self.check_qubit(control1).is_ok()
            && self.check_qubit(control2).is_ok()
            && self.check_qubit(target).is_ok()
        {
            unsafe { ffi::gate_toffoli(self.as_ptr(), control1 as i32, control2 as i32, target as i32); }
        }
        self
    }

    /// Apply Toffoli gate (alias).
    #[inline]
    pub fn ccx(&mut self, control1: usize, control2: usize, target: usize) -> &mut Self {
        self.toffoli(control1, control2, target)
    }

    /// Apply Fredkin (CSWAP) gate.
    pub fn fredkin(&mut self, control: usize, target1: usize, target2: usize) -> &mut Self {
        if self.check_qubit(control).is_ok()
            && self.check_qubit(target1).is_ok()
            && self.check_qubit(target2).is_ok()
        {
            unsafe { ffi::gate_fredkin(self.as_ptr(), control as i32, target1 as i32, target2 as i32); }
        }
        self
    }

    /// Apply Fredkin gate (alias).
    #[inline]
    pub fn cswap(&mut self, control: usize, target1: usize, target2: usize) -> &mut Self {
        self.fredkin(control, target1, target2)
    }

    /// Apply Quantum Fourier Transform to specified qubits.
    ///
    /// # Arguments
    /// * `qubits` - Slice of qubit indices to apply QFT to
    pub fn qft(&mut self, qubits: &[usize]) -> &mut Self {
        // Validate all qubits
        for &q in qubits {
            if self.check_qubit(q).is_err() {
                return self;
            }
        }
        let indices: Vec<i32> = qubits.iter().map(|&q| q as i32).collect();
        unsafe {
            ffi::gate_qft(self.as_ptr(), indices.as_ptr(), indices.len());
        }
        self
    }

    /// Apply inverse Quantum Fourier Transform to specified qubits.
    ///
    /// # Arguments
    /// * `qubits` - Slice of qubit indices to apply inverse QFT to
    pub fn iqft(&mut self, qubits: &[usize]) -> &mut Self {
        // Validate all qubits
        for &q in qubits {
            if self.check_qubit(q).is_err() {
                return self;
            }
        }
        let indices: Vec<i32> = qubits.iter().map(|&q| q as i32).collect();
        unsafe {
            ffi::gate_iqft(self.as_ptr(), indices.as_ptr(), indices.len());
        }
        self
    }

    // ========================================================================
    // MEASUREMENT
    // ========================================================================

    /// Compute expectation value of Z operator on a qubit.
    ///
    /// Returns ⟨ψ|Z|ψ⟩ ∈ [-1, 1].
    pub fn expectation_z(&self, qubit: usize) -> Result<f64> {
        self.check_qubit(qubit)?;
        unsafe {
            Ok(ffi::measurement_expectation_z(self.as_ptr(), qubit as i32))
        }
    }

    /// Compute expectation value of X operator on a qubit.
    pub fn expectation_x(&self, qubit: usize) -> Result<f64> {
        self.check_qubit(qubit)?;
        unsafe {
            Ok(ffi::measurement_expectation_x(self.as_ptr(), qubit as i32))
        }
    }

    /// Compute expectation value of Y operator on a qubit.
    pub fn expectation_y(&self, qubit: usize) -> Result<f64> {
        self.check_qubit(qubit)?;
        unsafe {
            Ok(ffi::measurement_expectation_y(self.as_ptr(), qubit as i32))
        }
    }

    /// Compute ZZ correlation between two qubits.
    ///
    /// Returns ⟨ψ|Z_i Z_j|ψ⟩.
    pub fn correlation_zz(&self, qubit_i: usize, qubit_j: usize) -> Result<f64> {
        self.check_qubit(qubit_i)?;
        self.check_qubit(qubit_j)?;
        unsafe {
            Ok(ffi::measurement_correlation_zz(self.as_ptr(), qubit_i as i32, qubit_j as i32))
        }
    }
}

impl Clone for QuantumState {
    fn clone(&self) -> Self {
        unsafe {
            let mut new_state = MaybeUninit::<ffi::quantum_state_t>::uninit();
            ffi::quantum_state_clone(new_state.as_mut_ptr(), self.as_ptr());

            let state_ptr = Box::into_raw(Box::new(new_state.assume_init()));

            Self {
                inner: NonNull::new(state_ptr).expect("Clone returned null"),
                num_qubits: self.num_qubits,
            }
        }
    }
}

impl Drop for QuantumState {
    fn drop(&mut self) {
        unsafe {
            ffi::quantum_state_free(self.as_ptr());
            // The Box was created in new(), so we need to deallocate it
            let _ = Box::from_raw(self.inner.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_state() {
        let state = QuantumState::new(3).unwrap();
        assert_eq!(state.num_qubits(), 3);
        assert_eq!(state.state_dim(), 8);
    }

    #[test]
    fn test_invalid_qubits() {
        assert!(QuantumState::new(0).is_err());
        assert!(QuantumState::new(33).is_err());
    }

    #[test]
    fn test_initial_state() {
        let state = QuantumState::new(2).unwrap();
        let probs = state.probabilities();

        // Initial state is |00⟩
        assert!((probs[0] - 1.0).abs() < 1e-10);
        assert!(probs[1] < 1e-10);
        assert!(probs[2] < 1e-10);
        assert!(probs[3] < 1e-10);
    }

    #[test]
    fn test_hadamard() {
        let mut state = QuantumState::new(1).unwrap();
        state.h(0);

        let probs = state.probabilities();
        // Should be 50/50 superposition
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_bell_state() {
        let mut state = QuantumState::new(2).unwrap();
        state.h(0).cnot(0, 1);

        let probs = state.probabilities();
        // Bell state: 50% |00⟩, 50% |11⟩
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!(probs[1] < 1e-10);
        assert!(probs[2] < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_ghz_state() {
        let mut state = QuantumState::new(3).unwrap();
        state.h(0).cnot(0, 1).cnot(0, 2);

        let probs = state.probabilities();
        // GHZ: 50% |000⟩, 50% |111⟩
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[7] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_entanglement_entropy() {
        let mut state = QuantumState::new(2).unwrap();

        // Product state has zero entanglement
        let entropy_product = state.entanglement_entropy(&[0]).unwrap();
        assert!(entropy_product < 1e-10);

        // Bell state has maximal entanglement
        state.h(0).cnot(0, 1);
        let entropy_bell = state.entanglement_entropy(&[0]).unwrap();
        assert!(entropy_bell > 0.6); // ln(2) ≈ 0.693
    }

    #[test]
    fn test_method_chaining() {
        let mut state = QuantumState::new(4).unwrap();

        // Should compile and work with chaining
        state
            .h(0)
            .h(1)
            .cnot(0, 2)
            .cnot(1, 3)
            .rz(2, std::f64::consts::PI / 4.0)
            .reset()
            .x(0);

        let probs = state.probabilities();
        // After reset and X(0), state is |0001⟩
        assert!((probs[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clone() {
        let mut state = QuantumState::new(2).unwrap();
        state.h(0).cnot(0, 1);

        let cloned = state.clone();

        // Cloned state should have same probabilities
        let orig_probs = state.probabilities();
        let clone_probs = cloned.probabilities();

        for i in 0..4 {
            assert!((orig_probs[i] - clone_probs[i]).abs() < 1e-10);
        }
    }
}
