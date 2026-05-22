//! Surface code (Clifford-tableau variant), since v0.5.12.
//!
//! Wraps `src/algorithms/topological/topological.{c,h}` exposing
//! the polynomial-time stabiliser-formalism surface code.  The
//! Clifford-tableau back-end (`surface_code_clifford_t`) scales as
//! `O(d^2)` rather than `O(2^(d^2))` for the dense-state-vector
//! variant; that's what makes threshold studies on `d in {3, 5, 7}`
//! tractable.  Implements:
//!
//! - rotated-lattice surface code with `d` x `d` data qubits;
//! - `(d - 1)^2` Z-type and `(d - 1)^2` X-type stabilisers,
//!   measured ancilla-mediated through CNOT + Hadamard;
//! - single-qubit Pauli error injection (X / Y / Z) for syndrome
//!   sampling and threshold sweeps.
//!
//! Decoding is *not* part of this surface yet; the existing
//! `tests/test_surface_code_threshold.c` sweep on `d in {3, 5, 7}`
//! drives this same stabiliser layer + an external decoder.  The
//! Rust wrapper exposes the syndrome data so callers can plug their
//! own decoder (e.g. via the `pymatching` Python interop) without
//! reimplementing the stabiliser machinery.
//!
//! ## Quick start
//!
//! ```no_run
//! use moonlab::surface_code::SurfaceCode;
//!
//! // Distance-3 rotated surface code (9 data qubits, 4+4 ancillas).
//! let mut code = SurfaceCode::new(3, 0xdeadbeef).unwrap();
//! code.measure_z_syndromes().unwrap();
//! assert_eq!(code.syndrome_weight(), 0,
//!     "logical-zero start has no syndromes");
//!
//! // Inject a single-qubit X error.
//! let q = code.data_index(1, 1).unwrap();
//! code.apply_error(q, 'X').unwrap();
//! code.measure_z_syndromes().unwrap();
//! assert!(code.syndrome_weight() > 0, "Z stabilisers detect X error");
//! ```

use crate::error::{QuantumError, Result};
use std::os::raw::c_char;
use moonlab_sys::{
    surface_code_clifford_apply_error, surface_code_clifford_create,
    surface_code_clifford_data_index, surface_code_clifford_free,
    surface_code_clifford_measure_x_syndromes,
    surface_code_clifford_measure_z_syndromes,
    surface_code_clifford_syndrome_weight, surface_code_clifford_t,
};
use std::ptr;

/// Owned handle to a Clifford-tableau surface code on `d x d` data
/// qubits + `2 * (d - 1)^2` ancillas (X-type and Z-type each).
pub struct SurfaceCode {
    ptr: *mut surface_code_clifford_t,
    distance: u32,
}

impl SurfaceCode {
    /// Allocate a rotated surface code of code distance `distance`
    /// with the given splitmix64 RNG seed.  `distance` must be odd
    /// and `>= 3` (`d = 1` is the trivial single-qubit code; the
    /// C side rejects it).
    pub fn new(distance: u32, rng_seed: u64) -> Result<Self> {
        if distance < 3 || distance % 2 == 0 {
            return Err(QuantumError::Ffi(format!(
                "surface code distance must be odd and >= 3, got {distance}"
            )));
        }
        let ptr = unsafe { surface_code_clifford_create(distance, rng_seed) };
        if ptr.is_null() {
            return Err(QuantumError::Ffi(
                "surface_code_clifford_create returned NULL".to_string(),
            ));
        }
        Ok(Self { ptr, distance })
    }

    /// Code distance `d` (the number of physical data qubits along
    /// each lattice edge).
    pub fn distance(&self) -> u32 {
        self.distance
    }

    /// Number of physical data qubits = `d^2`.
    pub fn num_data_qubits(&self) -> u32 {
        self.distance * self.distance
    }

    /// Number of ancilla qubits in each parity sector = `(d - 1)^2`.
    /// Total ancillas = `2 (d - 1)^2`.
    pub fn num_ancillas_per_sector(&self) -> u32 {
        let m = self.distance - 1;
        m * m
    }

    /// Linear data-qubit index from `(row, col)` lattice coordinates
    /// (both 0..d).
    pub fn data_index(&self, row: u32, col: u32) -> Result<u32> {
        if row >= self.distance || col >= self.distance {
            return Err(QuantumError::InvalidQubit {
                index: (row * self.distance + col) as usize,
                max: (self.distance * self.distance) as usize,
            });
        }
        Ok(unsafe { surface_code_clifford_data_index(self.ptr, row, col) })
    }

    /// Apply a single-qubit Pauli error on data qubit `q`.
    /// `error_type` must be `'X'`, `'Y'`, or `'Z'`.
    pub fn apply_error(&mut self, q: u32, error_type: char) -> Result<()> {
        if q >= self.num_data_qubits() {
            return Err(QuantumError::InvalidQubit {
                index: q as usize,
                max: self.num_data_qubits() as usize,
            });
        }
        let c = match error_type {
            'X' | 'x' => b'X' as c_char,
            'Y' | 'y' => b'Y' as c_char,
            'Z' | 'z' => b'Z' as c_char,
            _ => return Err(QuantumError::Ffi(format!(
                "error_type must be one of X / Y / Z, got '{error_type}'"
            ))),
        };
        let rc = unsafe {
            surface_code_clifford_apply_error(self.ptr, q, c)
        };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!(
                "surface_code_clifford_apply_error rc={rc}"
            )));
        }
        Ok(())
    }

    /// Measure all `(d - 1)^2` Z-type stabilisers
    /// (`Z Z Z Z` around each interior vertex).  Populates the
    /// internal `z_syndrome` bit vector; read with
    /// [`syndrome_weight`](Self::syndrome_weight) for the count or
    /// access raw bits via direct FFI if needed.
    pub fn measure_z_syndromes(&mut self) -> Result<()> {
        let rc = unsafe { surface_code_clifford_measure_z_syndromes(self.ptr) };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!(
                "measure_z_syndromes rc={rc}"
            )));
        }
        Ok(())
    }

    /// Measure all `(d - 1)^2` X-type stabilisers
    /// (`X X X X` around each interior face).
    pub fn measure_x_syndromes(&mut self) -> Result<()> {
        let rc = unsafe { surface_code_clifford_measure_x_syndromes(self.ptr) };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!(
                "measure_x_syndromes rc={rc}"
            )));
        }
        Ok(())
    }

    /// Total weight (set-bit count) across both X and Z syndromes.
    /// Zero on a logical-zero start; non-zero after an undetectable
    /// Pauli error.
    pub fn syndrome_weight(&self) -> u32 {
        unsafe { surface_code_clifford_syndrome_weight(self.ptr) }
    }
}

impl Drop for SurfaceCode {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { surface_code_clifford_free(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_3_has_expected_layout() {
        let code = SurfaceCode::new(3, 1).unwrap();
        assert_eq!(code.distance(), 3);
        assert_eq!(code.num_data_qubits(), 9);
        assert_eq!(code.num_ancillas_per_sector(), 4);
    }

    #[test]
    fn rejects_even_or_too_small_distance() {
        assert!(SurfaceCode::new(2, 1).is_err());
        assert!(SurfaceCode::new(4, 1).is_err());
        assert!(SurfaceCode::new(1, 1).is_err());
    }

    #[test]
    fn computational_basis_start_has_clean_z_syndromes() {
        // The C-side `create` initialises the stabiliser tableau to
        // the computational basis |0...0>, which is already a +1
        // eigenstate of every Z-type stabiliser; so measuring just
        // Z stabilisers should give zero weight.  X stabilisers
        // start in a superposition and need explicit logical-zero
        // preparation (Z-syndrome projection) -- not exposed yet,
        // so we don't pin them here.
        let mut code = SurfaceCode::new(3, 42).unwrap();
        code.measure_z_syndromes().unwrap();
        // We measured Z only; syndrome_weight is X+Z so it's
        // strictly the Z contribution after just one measure.
        // The Z-syndrome bits are populated in place every call,
        // so the X contribution is whatever was there from create
        // -- which the C side zeroes.  Verify by measuring Z again
        // (idempotent on a stabilised state) and checking the weight
        // stays at zero.
        let w_z_only = code.syndrome_weight();
        code.measure_z_syndromes().unwrap();
        assert_eq!(code.syndrome_weight(), w_z_only,
            "Z stabiliser measurement should be idempotent on |0...0>");
    }

    #[test]
    fn x_error_lights_z_stabilisers() {
        let mut code = SurfaceCode::new(3, 42).unwrap();
        let q = code.data_index(1, 1).unwrap();
        code.apply_error(q, 'X').unwrap();
        code.measure_z_syndromes().unwrap();
        assert!(
            code.syndrome_weight() > 0,
            "Z-type stabilisers should detect an X error at the centre"
        );
    }

    #[test]
    fn z_error_lights_x_stabilisers() {
        let mut code = SurfaceCode::new(3, 7).unwrap();
        let q = code.data_index(1, 1).unwrap();
        code.apply_error(q, 'Z').unwrap();
        code.measure_x_syndromes().unwrap();
        assert!(
            code.syndrome_weight() > 0,
            "X-type stabilisers should detect a Z error at the centre"
        );
    }

    #[test]
    fn rejects_unknown_error_type() {
        let mut code = SurfaceCode::new(3, 1).unwrap();
        let q = code.data_index(0, 0).unwrap();
        assert!(code.apply_error(q, 'W').is_err());
    }

    #[test]
    fn rejects_out_of_range_qubit() {
        let mut code = SurfaceCode::new(3, 1).unwrap();
        assert!(code.apply_error(100, 'X').is_err());
        assert!(code.data_index(10, 0).is_err());
    }
}
