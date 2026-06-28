//! Matrix-product density operator (MPDO) noise simulator (v0.3).
//!
//! Polynomial-cost simulation of noisy circuits via the matrix-product
//! representation of the density matrix.  Suited to local-noise
//! circuits in quasi-1D layouts up to ~100 qubits at single-qubit
//! error rates around 1e-3.
//!
//! ## Quick start
//!
//! ```no_run
//! use moonlab::mpdo::{Mpdo, PauliCode};
//!
//! let mut m = Mpdo::new(/*qubits=*/3, /*max_bond=*/16).unwrap();
//! m.apply_depolarizing(/*qubit=*/1, 0.4).unwrap();
//! let z = m.expect_pauli(1, PauliCode::Z).unwrap();
//! // <Z> = 1 - 4*p/3 = 0.4667 on |0>
//! assert!((z - (1.0 - 4.0 * 0.4 / 3.0)).abs() < 1e-12);
//! ```
//!
//! Status: scaffold (v0.3.0).  Single-qubit Kraus channels exposed;
//! two-qubit Kraus + SVD bond truncation deferred to v0.3.x.

use crate::error::{QuantumError, Result};
use moonlab_sys::{
    moonlab_mpdo_apply_amplitude_damping_1q, moonlab_mpdo_apply_bit_flip_1q,
    moonlab_mpdo_apply_bit_phase_flip_1q, moonlab_mpdo_apply_depolarizing_1q,
    moonlab_mpdo_apply_phase_damping_1q, moonlab_mpdo_apply_phase_flip_1q,
    moonlab_mpdo_clone, moonlab_mpdo_create, moonlab_mpdo_current_bond_dim,
    moonlab_mpdo_expect_pauli_1q, moonlab_mpdo_free, moonlab_mpdo_max_bond_dim,
    moonlab_mpdo_num_qubits, moonlab_mpdo_t, moonlab_mpdo_trace,
};
use std::ptr;

/// Pauli observable selector for `Mpdo::expect_pauli`.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
pub enum PauliCode {
    /// Identity (always returns +1 for a normalised state).
    I = 0,
    X = 1,
    Y = 2,
    Z = 3,
}

/// Owned matrix-product density operator handle.
///
/// Initial state is `|0...0><0...0|` at bond dimension 1.  The bond-
/// dim cap is fixed at construction time.  Single-qubit Kraus channels
/// don't grow the bond.
pub struct Mpdo {
    handle: *mut moonlab_mpdo_t,
}

// SAFETY: Mpdo owns its handle and is not shared across threads.
unsafe impl Send for Mpdo {}

impl Mpdo {
    /// Create a fresh `|0...0><0...0|` MPDO with the given bond cap.
    ///
    /// `max_bond_dim` of 16 is plenty for ~50-qubit local-noise
    /// circuits; bump to 32 if you start to see `Tr(rho)` drift.
    pub fn new(num_qubits: u32, max_bond_dim: u32) -> Result<Self> {
        if num_qubits == 0 {
            return Err(QuantumError::InvalidQubit { index: 0, max: 1 });
        }
        let handle = unsafe { moonlab_mpdo_create(num_qubits, max_bond_dim) };
        if handle.is_null() {
            return Err(QuantumError::AllocationFailed(num_qubits as usize));
        }
        Ok(Self { handle })
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> u32 {
        unsafe { moonlab_mpdo_num_qubits(self.handle) }
    }

    /// Bond-dimension cap fixed at construction.
    pub fn max_bond_dim(&self) -> u32 {
        unsafe { moonlab_mpdo_max_bond_dim(self.handle) }
    }

    /// Maximum bond actually in use right now.  For pure single-qubit
    /// channels this stays at 1.
    pub fn current_bond_dim(&self) -> u32 {
        unsafe { moonlab_mpdo_current_bond_dim(self.handle) }
    }

    /// `Tr(rho)`.  Should always be 1 for a CPTP-evolved state; drift
    /// flags either a non-CPTP custom Kraus rep or truncation losses.
    pub fn trace(&self) -> f64 {
        unsafe { moonlab_mpdo_trace(self.handle) }
    }

    /// Depolarising channel `rho -> (1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z)`.
    pub fn apply_depolarizing(&mut self, qubit: u32, p: f64) -> Result<()> {
        check_rc(unsafe { moonlab_mpdo_apply_depolarizing_1q(self.handle, qubit, p) })
    }

    /// Amplitude damping (T_1) at strength `gamma in [0, 1]`.
    pub fn apply_amplitude_damping(&mut self, qubit: u32, gamma: f64) -> Result<()> {
        check_rc(unsafe {
            moonlab_mpdo_apply_amplitude_damping_1q(self.handle, qubit, gamma)
        })
    }

    /// Phase damping (T_2) at strength `lambda in [0, 1]`.
    pub fn apply_phase_damping(&mut self, qubit: u32, lambda: f64) -> Result<()> {
        check_rc(unsafe {
            moonlab_mpdo_apply_phase_damping_1q(self.handle, qubit, lambda)
        })
    }

    /// Bit-flip channel `rho -> (1-p) rho + p X rho X`.
    pub fn apply_bit_flip(&mut self, qubit: u32, p: f64) -> Result<()> {
        check_rc(unsafe { moonlab_mpdo_apply_bit_flip_1q(self.handle, qubit, p) })
    }

    /// Phase-flip channel `rho -> (1-p) rho + p Z rho Z`.
    pub fn apply_phase_flip(&mut self, qubit: u32, p: f64) -> Result<()> {
        check_rc(unsafe { moonlab_mpdo_apply_phase_flip_1q(self.handle, qubit, p) })
    }

    /// Bit+phase-flip channel `rho -> (1-p) rho + p Y rho Y`.
    pub fn apply_bit_phase_flip(&mut self, qubit: u32, p: f64) -> Result<()> {
        check_rc(unsafe {
            moonlab_mpdo_apply_bit_phase_flip_1q(self.handle, qubit, p)
        })
    }

    /// Single-qubit Pauli expectation value `Tr(rho * P_q)`.
    pub fn expect_pauli(&self, qubit: u32, pauli: PauliCode) -> Result<f64> {
        let mut out: f64 = 0.0;
        let rc = unsafe {
            moonlab_mpdo_expect_pauli_1q(self.handle, qubit, pauli as u8, &mut out as *mut f64)
        };
        check_rc(rc)?;
        Ok(out)
    }
}

impl Clone for Mpdo {
    fn clone(&self) -> Self {
        let handle = unsafe { moonlab_mpdo_clone(self.handle) };
        if handle.is_null() {
            // Match the documented "infallible" Clone contract: panic
            // on OOM rather than silently returning a stale state.
            panic!("moonlab_mpdo_clone failed (out of memory)");
        }
        Self { handle }
    }
}

impl Drop for Mpdo {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { moonlab_mpdo_free(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

fn check_rc(rc: moonlab_sys::mpdo_error_t) -> Result<()> {
    if rc == moonlab_sys::mpdo_error_t_MPDO_SUCCESS {
        return Ok(());
    }
    let label = match rc {
        x if x == moonlab_sys::mpdo_error_t_MPDO_ERR_INVALID => "invalid argument",
        x if x == moonlab_sys::mpdo_error_t_MPDO_ERR_QUBIT => "qubit out of range",
        x if x == moonlab_sys::mpdo_error_t_MPDO_ERR_OOM => "out of memory",
        x if x == moonlab_sys::mpdo_error_t_MPDO_ERR_BACKEND => "backend failure",
        _ => "unknown",
    };
    Err(QuantumError::Ffi(format!("MPDO: {label} (rc={rc})")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_traces_to_one() {
        let m = Mpdo::new(3, 16).unwrap();
        assert!((m.trace() - 1.0).abs() < 1e-12);
        for q in 0..3 {
            let z = m.expect_pauli(q, PauliCode::Z).unwrap();
            assert!((z - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn depolarizing_z_decay() {
        let mut m = Mpdo::new(1, 16).unwrap();
        m.apply_depolarizing(0, 0.4).unwrap();
        let z = m.expect_pauli(0, PauliCode::Z).unwrap();
        assert!((z - (1.0 - 4.0 * 0.4 / 3.0)).abs() < 1e-12);
        assert!((m.trace() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn amplitude_damping_full_reset() {
        let mut m = Mpdo::new(1, 16).unwrap();
        // First flip to |1>, then amplitude-damp at gamma=1 -> back to |0>.
        m.apply_bit_flip(0, 1.0).unwrap();
        let z_after_flip = m.expect_pauli(0, PauliCode::Z).unwrap();
        assert!((z_after_flip + 1.0).abs() < 1e-12);
        m.apply_amplitude_damping(0, 1.0).unwrap();
        let z = m.expect_pauli(0, PauliCode::Z).unwrap();
        assert!((z - 1.0).abs() < 1e-12);
    }
}
