//! DMRG scalar-energy bindings.
//!
//! Thin safe wrappers around the v0.10.0 stable-ABI entries
//! `moonlab_dmrg_tfim_energy` and `moonlab_dmrg_heisenberg_energy`.
//!
//! For workflows that need the MPS handle, sweep history, or per-bond
//! truncation history, use [`crate::tdvp`] (TDVP) or drop to the
//! `moonlab-sys` FFI directly.

use moonlab_sys as ffi;

/// DMRG ground-state energy of the 1D transverse-field Ising model
/// `H = -sum_i Z_i Z_{i+1} - g sum_i X_i` (J = 1).
///
/// # Arguments
///
/// * `num_sites` - Chain length (>= 2).
/// * `g` - Transverse field ratio h/J.  Critical point at g = 1.
/// * `max_bond_dim` - DMRG truncation cap.
/// * `num_sweeps` - Number of two-site DMRG sweeps.
///
/// Returns `f64::INFINITY` (mapped from `DBL_MAX`) on parameter errors;
/// otherwise the converged ground-state energy.
pub fn tfim_ground_energy(
    num_sites: u32,
    g: f64,
    max_bond_dim: u32,
    num_sweeps: u32,
) -> f64 {
    unsafe { ffi::moonlab_dmrg_tfim_energy(num_sites, g, max_bond_dim, num_sweeps) }
}

/// DMRG ground-state energy of the 1D XXZ chain with longitudinal field
/// `H = J sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Delta Z_i Z_{i+1})
///       - h sum_i Z_i`.
///
/// # Arguments
///
/// * `num_sites` - Chain length (>= 2).
/// * `j` - Exchange coupling.
/// * `delta` - XXZ anisotropy.  `delta = 1` is isotropic Heisenberg.
/// * `h` - Longitudinal field.
/// * `max_bond_dim` - DMRG truncation cap.
/// * `num_sweeps` - Number of two-site DMRG sweeps.
///
/// Returns `f64::INFINITY` (mapped from `DBL_MAX`) on parameter errors;
/// otherwise the converged ground-state energy.
pub fn heisenberg_ground_energy(
    num_sites: u32,
    j: f64,
    delta: f64,
    h: f64,
    max_bond_dim: u32,
    num_sweeps: u32,
) -> f64 {
    unsafe {
        ffi::moonlab_dmrg_heisenberg_energy(num_sites, j, delta, h, max_bond_dim, num_sweeps)
    }
}
