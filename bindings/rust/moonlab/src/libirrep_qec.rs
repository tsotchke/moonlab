//! libirrep QEC zoo Rust binding, since v0.6.4.
//!
//! Safe wrapper around `src/integration/libirrep_bridge.{c,h}`.
//! Exposes eight CSS-code families through one type, `QecCode`, with
//! one accessor surface:
//!
//! - rotated surface code at arbitrary distance,
//! - Kitaev 2D toric code,
//! - 2D color codes (Steane [[7, 1, 3]], Hamming [[15, 7, 3]]),
//! - IBM bivariate-bicycle qLDPC codes (Bravyi et al. 2024 Nature
//!   627, 778 -- "Gross 72/144/288"),
//! - Tillich-Zemor hypergraph product on repetition codes for
//!   `d in {3, 4, 5}`.
//!
//! The bridge is optional at C build time.  When moonlab was
//! compiled without `-DQSIM_ENABLE_LIBIRREP=ON`, every constructor
//! returns `Err(QuantumError::Ffi("...NOT_BUILT..."))`.  Callers
//! wanting "use libirrep when available, else fall back" semantics
//! can probe with `is_available()` first.
//!
//! ## Quick start
//!
//! ```no_run
//! use moonlab::libirrep_qec::{QecCode, is_available};
//!
//! if is_available() {
//!     let code = QecCode::bb_72_12_6().unwrap();
//!     assert_eq!(code.n_qubits(), 72);
//!     assert_eq!(code.logical_qubits(), 12);
//! }
//! ```

use crate::error::{QuantumError, Result};
use moonlab_sys::{
    moonlab_libirrep_available, moonlab_libirrep_bb_144_12_12_new,
    moonlab_libirrep_bb_288_12_18_new, moonlab_libirrep_bb_72_12_6_new,
    moonlab_libirrep_color_hamming_15_7_3_new, moonlab_libirrep_color_steane_new,
    moonlab_libirrep_hgp_repetition_new, moonlab_libirrep_qec_distance,
    moonlab_libirrep_qec_free, moonlab_libirrep_qec_get_x_check_row,
    moonlab_libirrep_qec_get_z_check_row, moonlab_libirrep_qec_logical_qubits,
    moonlab_libirrep_qec_n_qubits, moonlab_libirrep_qec_n_x_stabs,
    moonlab_libirrep_qec_n_z_stabs, moonlab_libirrep_qec_t,
    moonlab_libirrep_surface_code_new, moonlab_libirrep_toric_code_new,
};
use std::ptr;

/// Whether moonlab was compiled with the libirrep linkage path
/// (mirrors `moonlab_libirrep_available` in the C bridge).
pub fn is_available() -> bool {
    unsafe { moonlab_libirrep_available() == 1 }
}

/// Status codes returned by the bridge.  Matches the negative-int
/// constants in `libirrep_bridge.h` so callers dispatching on
/// `QuantumError::Ffi` can recognise the specific failure.
pub const MOONLAB_LIBIRREP_NOT_BUILT: i32 = -201;
pub const MOONLAB_LIBIRREP_BAD_ARG: i32 = -202;
pub const MOONLAB_LIBIRREP_INTERNAL: i32 = -203;
pub const MOONLAB_LIBIRREP_OOM: i32 = -204;

/// Owned handle to a CSS code built via libirrep.
///
/// Free on drop; not `Send` since the underlying C handle's
/// lifetime is tied to one thread by convention.
pub struct QecCode {
    ptr: *mut moonlab_libirrep_qec_t,
}

impl QecCode {
    fn from_rc(rc: i32, ptr: *mut moonlab_libirrep_qec_t, ctx: &str) -> Result<Self> {
        if rc == 0 && !ptr.is_null() {
            return Ok(Self { ptr });
        }
        if rc == MOONLAB_LIBIRREP_NOT_BUILT {
            return Err(QuantumError::Ffi(format!(
                "{ctx}: moonlab was compiled without libirrep \
                 (rebuild with -DQSIM_ENABLE_LIBIRREP=ON)"
            )));
        }
        Err(QuantumError::Ffi(format!("{ctx}: rc={rc}")))
    }

    /// Rotated surface code at the given (odd) distance, `>= 2`.
    pub fn surface(distance: i32) -> Result<Self> {
        let mut p: *mut moonlab_libirrep_qec_t = ptr::null_mut();
        let rc = unsafe { moonlab_libirrep_surface_code_new(distance, &mut p) };
        Self::from_rc(rc, p, &format!("surface(distance={distance})"))
    }

    /// Kitaev 2D toric code on the `Lx x Ly` torus,
    /// `[[2 Lx Ly, 2, min(Lx, Ly)]]`.
    pub fn toric(lx: i32, ly: i32) -> Result<Self> {
        let mut p: *mut moonlab_libirrep_qec_t = ptr::null_mut();
        let rc = unsafe { moonlab_libirrep_toric_code_new(lx, ly, &mut p) };
        Self::from_rc(rc, p, &format!("toric(Lx={lx}, Ly={ly})"))
    }

    /// Steane `[[7, 1, 3]]` 2D color code.
    pub fn steane() -> Result<Self> {
        let mut p: *mut moonlab_libirrep_qec_t = ptr::null_mut();
        let rc = unsafe { moonlab_libirrep_color_steane_new(&mut p) };
        Self::from_rc(rc, p, "steane()")
    }

    /// `[[15, 7, 3]]` Hamming-based CSS code.
    pub fn hamming_15_7_3() -> Result<Self> {
        let mut p: *mut moonlab_libirrep_qec_t = ptr::null_mut();
        let rc = unsafe { moonlab_libirrep_color_hamming_15_7_3_new(&mut p) };
        Self::from_rc(rc, p, "hamming_15_7_3()")
    }

    /// IBM Gross bivariate-bicycle qLDPC code, `[[72, 12, 6]]`.
    pub fn bb_72_12_6() -> Result<Self> {
        let mut p: *mut moonlab_libirrep_qec_t = ptr::null_mut();
        let rc = unsafe { moonlab_libirrep_bb_72_12_6_new(&mut p) };
        Self::from_rc(rc, p, "bb_72_12_6()")
    }

    /// IBM Gross-144 bivariate-bicycle qLDPC code, `[[144, 12, 12]]`.
    pub fn bb_144_12_12() -> Result<Self> {
        let mut p: *mut moonlab_libirrep_qec_t = ptr::null_mut();
        let rc = unsafe { moonlab_libirrep_bb_144_12_12_new(&mut p) };
        Self::from_rc(rc, p, "bb_144_12_12()")
    }

    /// IBM Gross-288 bivariate-bicycle qLDPC code, `[[288, 12, 18]]`.
    pub fn bb_288_12_18() -> Result<Self> {
        let mut p: *mut moonlab_libirrep_qec_t = ptr::null_mut();
        let rc = unsafe { moonlab_libirrep_bb_288_12_18_new(&mut p) };
        Self::from_rc(rc, p, "bb_288_12_18()")
    }

    /// Tillich-Zemor hypergraph product on two `[d, 1, d]` repetition
    /// codes.  Only `d in {3, 4, 5}` is supported in v0.6.2 (matches
    /// the published instances in libirrep's `hypergraph_product.h`).
    pub fn hgp_repetition(d: i32) -> Result<Self> {
        let mut p: *mut moonlab_libirrep_qec_t = ptr::null_mut();
        let rc = unsafe { moonlab_libirrep_hgp_repetition_new(d, &mut p) };
        Self::from_rc(rc, p, &format!("hgp_repetition(d={d})"))
    }

    /// Number of physical qubits `n`.
    pub fn n_qubits(&self) -> i32 {
        unsafe { moonlab_libirrep_qec_n_qubits(self.ptr) }
    }

    /// Number of X-stabiliser generators.
    pub fn n_x_stabs(&self) -> i32 {
        unsafe { moonlab_libirrep_qec_n_x_stabs(self.ptr) }
    }

    /// Number of Z-stabiliser generators.
    pub fn n_z_stabs(&self) -> i32 {
        unsafe { moonlab_libirrep_qec_n_z_stabs(self.ptr) }
    }

    /// Number of logical qubits `k = n - rank(H_X) - rank(H_Z)`.
    pub fn logical_qubits(&self) -> i32 {
        unsafe { moonlab_libirrep_qec_logical_qubits(self.ptr) }
    }

    /// Brute-force code distance.  Cached after the first call.
    /// Expensive on larger codes; only tractable up to `n ~ 25`,
    /// `d <= 5`.
    pub fn distance(&self) -> i32 {
        unsafe { moonlab_libirrep_qec_distance(self.ptr) }
    }

    /// Length-`n_qubits` `0 / 1` support vector for the X-stabiliser
    /// at the given row index.
    pub fn x_check_row(&self, row: i32) -> Result<Vec<u8>> {
        let n = self.n_qubits() as usize;
        let mut buf = vec![0u8; n];
        let rc = unsafe {
            moonlab_libirrep_qec_get_x_check_row(self.ptr, row, buf.as_mut_ptr())
        };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!("x_check_row({row}): rc={rc}")));
        }
        Ok(buf)
    }

    /// Length-`n_qubits` `0 / 1` support vector for the Z-stabiliser
    /// at the given row index.
    pub fn z_check_row(&self, row: i32) -> Result<Vec<u8>> {
        let n = self.n_qubits() as usize;
        let mut buf = vec![0u8; n];
        let rc = unsafe {
            moonlab_libirrep_qec_get_z_check_row(self.ptr, row, buf.as_mut_ptr())
        };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!("z_check_row({row}): rc={rc}")));
        }
        Ok(buf)
    }
}

impl Drop for QecCode {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { moonlab_libirrep_qec_free(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn require_libirrep() -> bool {
        if !is_available() {
            eprintln!("libirrep not linked -- test skipped");
            return false;
        }
        true
    }

    #[test]
    fn surface_d3() {
        if !require_libirrep() { return; }
        let code = QecCode::surface(3).unwrap();
        assert_eq!(code.n_qubits(), 9);
        assert_eq!(code.n_x_stabs(), 4);
        assert_eq!(code.n_z_stabs(), 4);
        assert_eq!(code.logical_qubits(), 1);
        assert_eq!(code.distance(), 3);
    }

    #[test]
    fn toric_l3() {
        if !require_libirrep() { return; }
        let code = QecCode::toric(3, 3).unwrap();
        assert_eq!(code.n_qubits(), 18);
        assert_eq!(code.logical_qubits(), 2);
    }

    #[test]
    fn steane() {
        if !require_libirrep() { return; }
        let code = QecCode::steane().unwrap();
        assert_eq!(code.n_qubits(), 7);
        assert_eq!(code.logical_qubits(), 1);
        assert_eq!(code.distance(), 3);
    }

    #[test]
    fn ibm_gross_72() {
        if !require_libirrep() { return; }
        let code = QecCode::bb_72_12_6().unwrap();
        assert_eq!(code.n_qubits(), 72);
        assert_eq!(code.logical_qubits(), 12);
    }

    #[test]
    fn ibm_gross_288() {
        if !require_libirrep() { return; }
        let code = QecCode::bb_288_12_18().unwrap();
        assert_eq!(code.n_qubits(), 288);
        assert_eq!(code.logical_qubits(), 12);
    }

    #[test]
    fn hgp_rep_ladder() {
        if !require_libirrep() { return; }
        for (d, n) in [(3, 13), (4, 25), (5, 41)] {
            let code = QecCode::hgp_repetition(d).unwrap();
            assert_eq!(code.n_qubits(), n);
            assert_eq!(code.logical_qubits(), 1);
        }
    }

    #[test]
    fn check_row_weight() {
        if !require_libirrep() { return; }
        let code = QecCode::steane().unwrap();
        for row in 0..code.n_x_stabs() {
            let support = code.x_check_row(row).unwrap();
            let weight: u32 = support.iter().map(|&b| b as u32).sum();
            assert_eq!(weight, 4, "Steane X-row {row} weight = {weight}");
        }
    }

    #[test]
    fn hgp_rejects_out_of_range() {
        if !require_libirrep() { return; }
        assert!(QecCode::hgp_repetition(6).is_err());
    }
}
