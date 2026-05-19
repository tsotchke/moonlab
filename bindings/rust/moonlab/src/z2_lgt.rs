//! 1+1D Z2 lattice gauge theory Pauli-sum primitives (since v0.2.1;
//! safe Rust wrapper since v0.4.7).
//!
//! Wraps `moonlab_z2_lgt_1d_build` and `moonlab_z2_lgt_1d_gauss_law`
//! from the stable ABI (`src/applications/moonlab_export.h`).  Used
//! by the CA-MPS variational-D + gauge-aware warmstart demonstration;
//! the C side returns a flat (num_terms, num_qubits) Pauli-byte array
//! plus matching real coefficients, and a per-site Gauss-law
//! generator (`G_x` for `x` in `[1, N-1)`).
//!
//! Mirrors the Python `moonlab.ca_mps.z2_lgt_1d_build` /
//! `z2_lgt_1d_gauss_law` and the JavaScript
//! `@moonlab/quantum-core` `z2Lgt1dBuild` / `z2Lgt1dGaussLaw`
//! surface.
//!
//! ## Quick start
//!
//! ```no_run
//! use moonlab::z2_lgt;
//!
//! // 1+1D Z2 LGT on N = 4 matter sites at the strong-coupling point.
//! let h = z2_lgt::build(/*num_matter_sites=*/4,
//!                       /*t_hop=*/1.0,
//!                       /*h_link=*/1.0,
//!                       /*mass=*/0.5,
//!                       /*gauss_penalty=*/10.0).unwrap();
//! assert_eq!(h.num_qubits, 2 * 4 - 1);  // 2N - 1 with snake encoding
//! assert!(h.num_terms > 0);
//!
//! // Pauli string for G_1 = X_1 Z_2 X_3.
//! let g1 = z2_lgt::gauss_law(/*num_matter_sites=*/4, /*site_x=*/1).unwrap();
//! assert_eq!(g1[0], 0);  // I
//! assert_eq!(g1[1], 1);  // X
//! assert_eq!(g1[2], 3);  // Z
//! assert_eq!(g1[3], 1);  // X
//! ```

use crate::error::{QuantumError, Result};
use moonlab_sys::{
    moonlab_z2_lgt_1d_build, moonlab_z2_lgt_1d_gauss_law,
};
use std::slice;

/// Owned Pauli-sum Hamiltonian for the 1+1D Z2 LGT chain.
///
/// `paulis` is the flat row-major `(num_terms, num_qubits)`
/// Pauli-byte array (0 = I, 1 = X, 2 = Y, 3 = Z).  `coeffs[k]` is
/// the real coefficient on term `k`.  `num_qubits` is `2N - 1` on
/// the standard `N`-matter-site snake encoding (matter + gauge
/// link).
pub struct Z2LgtHamiltonian {
    pub paulis: Vec<u8>,
    pub coeffs: Vec<f64>,
    pub num_terms: u32,
    pub num_qubits: u32,
}

/// Build the 1+1D Z2 lattice-gauge-theory Pauli-sum Hamiltonian on
/// `num_matter_sites` matter sites with the given couplings.  The
/// returned Hamiltonian is owned by Rust; the C-side allocations
/// are copied into `Vec`s and freed before return.
pub fn build(
    num_matter_sites: u32,
    t_hop: f64,
    h_link: f64,
    mass: f64,
    gauss_penalty: f64,
) -> Result<Z2LgtHamiltonian> {
    if num_matter_sites < 2 {
        return Err(QuantumError::InvalidQubit {
            index: num_matter_sites as usize,
            max: 2,
        });
    }
    let mut paulis_ptr: *mut u8 = std::ptr::null_mut();
    let mut coeffs_ptr: *mut f64 = std::ptr::null_mut();
    let mut num_terms: u32 = 0;
    let mut num_qubits: u32 = 0;
    let rc = unsafe {
        moonlab_z2_lgt_1d_build(
            num_matter_sites,
            t_hop,
            h_link,
            mass,
            gauss_penalty,
            &mut paulis_ptr,
            &mut coeffs_ptr,
            &mut num_terms,
            &mut num_qubits,
        )
    };
    if rc != 0 {
        return Err(QuantumError::Ffi(format!(
            "moonlab_z2_lgt_1d_build rc={rc}"
        )));
    }
    if paulis_ptr.is_null() || coeffs_ptr.is_null() {
        return Err(QuantumError::Ffi(
            "moonlab_z2_lgt_1d_build returned null buffers".to_string(),
        ));
    }
    // Copy into owned Vecs and free the C buffers.
    let paulis_len = num_terms as usize * num_qubits as usize;
    let paulis = unsafe { slice::from_raw_parts(paulis_ptr, paulis_len) }
        .to_vec();
    let coeffs = unsafe { slice::from_raw_parts(coeffs_ptr, num_terms as usize) }
        .to_vec();
    unsafe {
        libc::free(paulis_ptr as *mut libc::c_void);
        libc::free(coeffs_ptr as *mut libc::c_void);
    }
    Ok(Z2LgtHamiltonian {
        paulis,
        coeffs,
        num_terms,
        num_qubits,
    })
}

/// Pauli-byte representation of the Gauss-law generator `G_{site_x}`
/// on a chain of `num_matter_sites` matter sites.  The returned
/// vector has length `2N - 1`; byte `j` is the Pauli operator on
/// qubit `j` (0 = I, 1 = X, 2 = Y, 3 = Z).
pub fn gauss_law(num_matter_sites: u32, site_x: u32) -> Result<Vec<u8>> {
    if num_matter_sites < 2 {
        return Err(QuantumError::InvalidQubit {
            index: num_matter_sites as usize,
            max: 2,
        });
    }
    let n_qubits = (2 * num_matter_sites - 1) as usize;
    let mut out = vec![0u8; n_qubits];
    let rc = unsafe {
        moonlab_z2_lgt_1d_gauss_law(num_matter_sites, site_x, out.as_mut_ptr())
    };
    if rc != 0 {
        return Err(QuantumError::Ffi(format!(
            "moonlab_z2_lgt_1d_gauss_law rc={rc}"
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_n4_returns_nonempty_hamiltonian() {
        let h = build(4, 1.0, 1.0, 0.5, 10.0).unwrap();
        assert_eq!(h.num_qubits, 2 * 4 - 1);
        assert!(h.num_terms > 0);
        assert_eq!(
            h.paulis.len(),
            h.num_terms as usize * h.num_qubits as usize
        );
        assert_eq!(h.coeffs.len(), h.num_terms as usize);
    }

    #[test]
    fn gauss_law_n4_site1_layout() {
        // G_1 = X_1 Z_2 X_3 on the 7-qubit (2*4-1) snake encoding;
        // matches the ABI smoke test in
        // tests/abi/test_moonlab_export_abi.c.
        let g = gauss_law(4, 1).unwrap();
        assert_eq!(g.len(), 7);
        assert_eq!(g[0], 0); // I
        assert_eq!(g[1], 1); // X
        assert_eq!(g[2], 3); // Z
        assert_eq!(g[3], 1); // X
        assert_eq!(g[4], 0); // I
        assert_eq!(g[5], 0); // I
        assert_eq!(g[6], 0); // I
    }

    #[test]
    fn reject_too_few_sites() {
        assert!(build(1, 1.0, 1.0, 0.5, 10.0).is_err());
        assert!(gauss_law(1, 0).is_err());
    }
}
