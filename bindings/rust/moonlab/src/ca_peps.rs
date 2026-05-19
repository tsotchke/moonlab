//! Clifford-Assisted PEPS (2D) -- safe Rust wrapper since v0.4.11.
//!
//! Wraps `src/algorithms/tensor_network/ca_peps.{c,h}`, the 2D
//! generalisation of CA-MPS.  Same Clifford-tableau + physical-MPS
//! split as [`crate::ca_mps::CaMps`]: Clifford gates are free
//! (O(n) tableau updates), and non-Clifford gates conjugate through
//! to Pauli rotations on the physical factor.  Mirrors the v0.2.1
//! Python `moonlab.ca_peps.CAPEPS` surface.
//!
//! ## Quick start
//!
//! ```no_run
//! use moonlab::ca_peps::{CaPeps, PauliCode};
//!
//! // 2x3 lattice with bond-dim cap chi = 8.
//! let mut state = CaPeps::new(2, 3, 8).unwrap();
//! state.h(0).unwrap();
//! state.cnot(0, 1).unwrap();
//! // Linear index = x + Lx*y; for Lx=2, qubit 2 is (0, 1).
//! let z0 = state.expect_pauli_single(0, PauliCode::Z).unwrap();
//! ```

use crate::error::{QuantumError, Result};
use moonlab_sys::{
    ca_peps_error_t, moonlab_ca_peps_clone, moonlab_ca_peps_cnot,
    moonlab_ca_peps_create, moonlab_ca_peps_current_bond_dim,
    moonlab_ca_peps_cz, moonlab_ca_peps_expect_pauli, moonlab_ca_peps_free,
    moonlab_ca_peps_h, moonlab_ca_peps_lx, moonlab_ca_peps_ly,
    moonlab_ca_peps_max_bond_dim, moonlab_ca_peps_max_half_cut_entropy,
    moonlab_ca_peps_norm, moonlab_ca_peps_normalize, moonlab_ca_peps_num_qubits,
    moonlab_ca_peps_phase, moonlab_ca_peps_prob_z, moonlab_ca_peps_rx,
    moonlab_ca_peps_ry, moonlab_ca_peps_rz, moonlab_ca_peps_s,
    moonlab_ca_peps_sdag, moonlab_ca_peps_t as moonlab_ca_peps_handle_t,
    moonlab_ca_peps_t_dagger, moonlab_ca_peps_t_gate, moonlab_ca_peps_x,
    moonlab_ca_peps_y, moonlab_ca_peps_z, __BindgenComplex,
};
use std::ptr;

/// Pauli code per qubit for [`CaPeps::expect_pauli`] /
/// [`CaPeps::expect_pauli_single`].  Matches the C header
/// convention `0=I, 1=X, 2=Y, 3=Z`.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
pub enum PauliCode {
    I = 0,
    X = 1,
    Y = 2,
    Z = 3,
}

/// Owned CA-PEPS handle.
pub struct CaPeps {
    ptr: *mut moonlab_ca_peps_handle_t,
}

impl CaPeps {
    /// Allocate a CA-PEPS on `Lx x Ly` qubits with per-bond
    /// bond-dim cap `chi_bond`.  Initial state is `|0...0>` with
    /// `D = I`.
    pub fn new(lx: u32, ly: u32, chi_bond: u32) -> Result<Self> {
        if lx == 0 || ly == 0 {
            return Err(QuantumError::Ffi(format!(
                "Lx={lx}, Ly={ly}: both dimensions must be >= 1"
            )));
        }
        if chi_bond == 0 {
            return Err(QuantumError::Ffi(
                "chi_bond must be >= 1".to_string(),
            ));
        }
        let ptr = unsafe { moonlab_ca_peps_create(lx, ly, chi_bond) };
        if ptr.is_null() {
            return Err(QuantumError::Ffi(
                "moonlab_ca_peps_create returned NULL".to_string(),
            ));
        }
        Ok(Self { ptr })
    }

    /// Lx, the lattice's x extent.
    pub fn lx(&self) -> u32 { unsafe { moonlab_ca_peps_lx(self.ptr) } }
    /// Ly, the lattice's y extent.
    pub fn ly(&self) -> u32 { unsafe { moonlab_ca_peps_ly(self.ptr) } }
    /// Total number of qubits `Lx * Ly`.
    pub fn num_qubits(&self) -> u32 {
        unsafe { moonlab_ca_peps_num_qubits(self.ptr) }
    }
    /// Configured bond-dim cap.
    pub fn max_bond_dim(&self) -> u32 {
        unsafe { moonlab_ca_peps_max_bond_dim(self.ptr) }
    }
    /// Current maximum bond dimension across all bonds.
    pub fn current_bond_dim(&self) -> u32 {
        unsafe { moonlab_ca_peps_current_bond_dim(self.ptr) }
    }
    /// `<phi | phi>` of the physical factor; should be 1 modulo
    /// truncation noise.
    pub fn norm(&self) -> f64 { unsafe { moonlab_ca_peps_norm(self.ptr) } }
    /// Maximum half-cut von Neumann entanglement entropy of `|phi>`
    /// across all bipartitions, in nats.
    pub fn max_half_cut_entropy(&self) -> f64 {
        unsafe { moonlab_ca_peps_max_half_cut_entropy(self.ptr) }
    }

    /// Normalise the physical factor in place.
    pub fn normalize(&mut self) -> Result<()> {
        check(unsafe { moonlab_ca_peps_normalize(self.ptr) })
    }

    // ---- Clifford gates (tableau-only; O(n) bit ops) -------------------

    pub fn h(&mut self, q: u32) -> Result<&mut Self> {
        check(unsafe { moonlab_ca_peps_h(self.ptr, q) })?;
        Ok(self)
    }
    pub fn s(&mut self, q: u32) -> Result<&mut Self> {
        check(unsafe { moonlab_ca_peps_s(self.ptr, q) })?;
        Ok(self)
    }
    pub fn sdag(&mut self, q: u32) -> Result<&mut Self> {
        check(unsafe { moonlab_ca_peps_sdag(self.ptr, q) })?;
        Ok(self)
    }
    pub fn x(&mut self, q: u32) -> Result<&mut Self> {
        check(unsafe { moonlab_ca_peps_x(self.ptr, q) })?;
        Ok(self)
    }
    pub fn y(&mut self, q: u32) -> Result<&mut Self> {
        check(unsafe { moonlab_ca_peps_y(self.ptr, q) })?;
        Ok(self)
    }
    pub fn z(&mut self, q: u32) -> Result<&mut Self> {
        check(unsafe { moonlab_ca_peps_z(self.ptr, q) })?;
        Ok(self)
    }
    pub fn cnot(&mut self, control: u32, target: u32) -> Result<&mut Self> {
        check(unsafe { moonlab_ca_peps_cnot(self.ptr, control, target) })?;
        Ok(self)
    }
    pub fn cz(&mut self, a: u32, b: u32) -> Result<&mut Self> {
        check(unsafe { moonlab_ca_peps_cz(self.ptr, a, b) })?;
        Ok(self)
    }

    // ---- Single-qubit non-Clifford rotations ---------------------------

    pub fn rx(&mut self, q: u32, theta: f64) -> Result<&mut Self> {
        check(unsafe { moonlab_ca_peps_rx(self.ptr, q, theta) })?;
        Ok(self)
    }
    pub fn ry(&mut self, q: u32, theta: f64) -> Result<&mut Self> {
        check(unsafe { moonlab_ca_peps_ry(self.ptr, q, theta) })?;
        Ok(self)
    }
    pub fn rz(&mut self, q: u32, theta: f64) -> Result<&mut Self> {
        check(unsafe { moonlab_ca_peps_rz(self.ptr, q, theta) })?;
        Ok(self)
    }
    pub fn t_gate(&mut self, q: u32) -> Result<&mut Self> {
        check(unsafe { moonlab_ca_peps_t_gate(self.ptr, q) })?;
        Ok(self)
    }
    pub fn t_dagger(&mut self, q: u32) -> Result<&mut Self> {
        check(unsafe { moonlab_ca_peps_t_dagger(self.ptr, q) })?;
        Ok(self)
    }
    pub fn phase(&mut self, q: u32, theta: f64) -> Result<&mut Self> {
        check(unsafe { moonlab_ca_peps_phase(self.ptr, q, theta) })?;
        Ok(self)
    }

    // ---- Observables ---------------------------------------------------

    /// `<psi | P | psi>` for an n-qubit Pauli string `pauli` of
    /// length `num_qubits()`.  Entry per qubit is `0=I, 1=X, 2=Y, 3=Z`.
    /// Returns the complex expectation value as `(real, imag)`.
    pub fn expect_pauli(&self, pauli: &[u8]) -> Result<(f64, f64)> {
        if pauli.len() != self.num_qubits() as usize {
            return Err(QuantumError::Ffi(format!(
                "pauli length {} != num_qubits {}",
                pauli.len(),
                self.num_qubits()
            )));
        }
        let mut out = __BindgenComplex { re: 0.0, im: 0.0 };
        check(unsafe {
            moonlab_ca_peps_expect_pauli(self.ptr, pauli.as_ptr(), &mut out)
        })?;
        Ok((out.re, out.im))
    }

    /// `<psi | P_q | psi>` for a single-site Pauli on qubit `q`,
    /// identity elsewhere.  Convenience helper around
    /// [`expect_pauli`].  Returns the real part only (single-site
    /// Pauli expectations are real up to roundoff).
    pub fn expect_pauli_single(&self, q: u32, pauli: PauliCode) -> Result<f64> {
        let n = self.num_qubits() as usize;
        if (q as usize) >= n {
            return Err(QuantumError::InvalidQubit {
                index: q as usize,
                max: n,
            });
        }
        let mut p = vec![PauliCode::I as u8; n];
        p[q as usize] = pauli as u8;
        let (re, _im) = self.expect_pauli(&p)?;
        Ok(re)
    }

    /// `P(Z_q = +1)` marginal probability.
    pub fn prob_z(&self, q: u32) -> Result<f64> {
        let mut out: f64 = 0.0;
        check(unsafe { moonlab_ca_peps_prob_z(self.ptr, q, &mut out) })?;
        Ok(out)
    }
}

impl Clone for CaPeps {
    fn clone(&self) -> Self {
        let ptr = unsafe { moonlab_ca_peps_clone(self.ptr) };
        assert!(!ptr.is_null(), "moonlab_ca_peps_clone returned NULL");
        Self { ptr }
    }
}

impl Drop for CaPeps {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { moonlab_ca_peps_free(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

fn check(rc: ca_peps_error_t) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(QuantumError::Ffi(format!("ca_peps rc={rc}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_state_traces_to_unit_z() {
        // |0...0> has <Z_q> = 1 on every qubit.
        let state = CaPeps::new(2, 2, 4).unwrap();
        assert_eq!(state.num_qubits(), 4);
        assert_eq!(state.lx(), 2);
        assert_eq!(state.ly(), 2);
        for q in 0..4 {
            let z = state.expect_pauli_single(q, PauliCode::Z).unwrap();
            assert!((z - 1.0).abs() < 1e-10, "<Z_{q}> = {} on |0000>", z);
        }
    }

    #[test]
    fn hadamard_zeros_out_z_expectation() {
        // H|0> = |+>; <Z> = 0.
        let mut state = CaPeps::new(2, 2, 4).unwrap();
        state.h(0).unwrap();
        let z0 = state.expect_pauli_single(0, PauliCode::Z).unwrap();
        assert!(z0.abs() < 1e-10, "<Z_0> = {} on H|0>", z0);
        // Other qubits stay at <Z> = 1.
        let z1 = state.expect_pauli_single(1, PauliCode::Z).unwrap();
        assert!((z1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cnot_pair_yields_perfect_zz_correlation() {
        // Bell pair on (0, 1) via H(0) + CNOT(0, 1).
        let mut state = CaPeps::new(2, 1, 4).unwrap();
        state.h(0).unwrap();
        state.cnot(0, 1).unwrap();
        let zz = state.expect_pauli(&[PauliCode::Z as u8, PauliCode::Z as u8]).unwrap();
        assert!((zz.0 - 1.0).abs() < 1e-10, "<ZZ> = {} on Bell pair", zz.0);
    }

    #[test]
    fn rejects_invalid_dimensions() {
        assert!(CaPeps::new(0, 3, 4).is_err());
        assert!(CaPeps::new(3, 0, 4).is_err());
        assert!(CaPeps::new(2, 2, 0).is_err());
    }

    #[test]
    fn norm_starts_at_unit() {
        let state = CaPeps::new(2, 2, 4).unwrap();
        assert!((state.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn clone_is_independent() {
        let mut a = CaPeps::new(2, 2, 4).unwrap();
        a.h(0).unwrap();
        let b = a.clone();
        // After H(0) again on `a`, `a` returns to |0...0> but `b`
        // remains in (H|0>) (x) |0...>.
        a.h(0).unwrap();
        let z_a = a.expect_pauli_single(0, PauliCode::Z).unwrap();
        let z_b = b.expect_pauli_single(0, PauliCode::Z).unwrap();
        assert!((z_a - 1.0).abs() < 1e-10, "a back to |0>: <Z_0> = {}", z_a);
        assert!(z_b.abs() < 1e-10, "b still on |+>: <Z_0> = {}", z_b);
    }
}
