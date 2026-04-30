//! Clifford-Assisted MPS (CA-MPS) bindings.
//!
//! Hybrid `|psi> = D|phi>` representation that absorbs the Clifford
//! structure of a circuit into a tableau and only pushes non-Clifford
//! rotations into the MPS factor.  See `docs/research/ca_mps.md` for
//! the full theory and `MATH.md` §10–12 for the math.
//!
//! # Highlights
//!
//! - [`CaMps`] -- the state handle plus the Clifford / non-Clifford
//!   gate surface.  RAII: dropped on scope exit.
//! - [`var_d_run`] -- variational-D ground-state search (alternating
//!   greedy Clifford + imag-time loop).
//! - [`gauge_warmstart`] -- Aaronson-Gottesman symplectic-Gauss-Jordan
//!   Clifford prep on any abelian Pauli stabilizer subgroup.
//! - [`z2_lgt_1d_build`] / [`z2_lgt_1d_gauss_law`] -- 1+1D Z2 lattice
//!   gauge theory Pauli sum + Gauss-law accessor.
//! - [`status_string`] -- diagnostic stringifier for any Moonlab
//!   status code.
//!
//! All bind to the v0.2.1 stable ABI in
//! `src/applications/moonlab_export.h`.

use std::os::raw::c_char;
use std::ptr::NonNull;

use moonlab_sys as ffi;

use crate::error::{QuantumError, Result};

/// Warmstart strategies for [`var_d_run`].  The integer values match
/// the C ABI convention in `moonlab_ca_mps_var_d_run`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Warmstart {
    Identity = 0,
    HAll = 1,
    DualTfim = 2,
    FerroTfim = 3,
    /// Gauge-aware: caller supplies commuting Pauli generators in the
    /// `stab_paulis` argument of [`var_d_run`].
    StabilizerSubgroup = 4,
}

/// Module identifiers for [`status_string`], matching
/// `moonlab_status_module_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum StatusModule {
    Generic = 0,
    CaMps = 1,
    CaMpsVarD = 2,
    CaMpsStabWarmstart = 3,
    CaPeps = 4,
    TnState = 5,
    TnGate = 6,
    TnMeasure = 7,
    Tensor = 8,
    Contract = 9,
    SvdCompress = 10,
    Clifford = 11,
    Partition = 12,
    DistGate = 13,
    MpiBridge = 14,
}

/// Owned handle to a Clifford-Assisted MPS state.
pub struct CaMps {
    handle: NonNull<ffi::moonlab_ca_mps_t>,
}

// SAFETY: the underlying tableau + MPS state is single-owner; we never
// share the raw pointer across threads.  CaMps is `Send` (move across
// thread boundaries is fine) but not `Sync` (no shared mutation).
unsafe impl Send for CaMps {}

impl CaMps {
    /// Create a CA-MPS state on `num_qubits` with the given MPS
    /// truncation cap.
    pub fn new(num_qubits: u32, max_bond_dim: u32) -> Result<Self> {
        let raw = unsafe { ffi::moonlab_ca_mps_create(num_qubits, max_bond_dim) };
        let handle = NonNull::new(raw)
            .ok_or(QuantumError::AllocationFailed(num_qubits as usize))?;
        Ok(Self { handle })
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> u32 {
        unsafe { ffi::moonlab_ca_mps_num_qubits(self.handle.as_ptr()) }
    }

    /// Current MPS bond dimension (max across all bonds).
    pub fn bond_dim(&self) -> u32 {
        unsafe { ffi::moonlab_ca_mps_current_bond_dim(self.handle.as_ptr()) }
    }

    /// State norm.
    pub fn norm(&self) -> f64 {
        unsafe { ffi::moonlab_ca_mps_norm(self.handle.as_ptr()) }
    }

    /// Renormalise after non-unitary evolution (e.g. imag-time).
    pub fn normalize(&mut self) -> Result<()> {
        check(unsafe { ffi::moonlab_ca_mps_normalize(self.handle.as_ptr()) })
    }

    // --- Clifford gates: tableau-only, no MPS cost. ---

    pub fn h(&mut self, q: u32) -> Result<()> {
        check(unsafe { ffi::moonlab_ca_mps_h(self.handle.as_ptr(), q) })
    }
    pub fn s(&mut self, q: u32) -> Result<()> {
        check(unsafe { ffi::moonlab_ca_mps_s(self.handle.as_ptr(), q) })
    }
    pub fn sdag(&mut self, q: u32) -> Result<()> {
        check(unsafe { ffi::moonlab_ca_mps_sdag(self.handle.as_ptr(), q) })
    }
    pub fn x(&mut self, q: u32) -> Result<()> {
        check(unsafe { ffi::moonlab_ca_mps_x(self.handle.as_ptr(), q) })
    }
    pub fn y(&mut self, q: u32) -> Result<()> {
        check(unsafe { ffi::moonlab_ca_mps_y(self.handle.as_ptr(), q) })
    }
    pub fn z(&mut self, q: u32) -> Result<()> {
        check(unsafe { ffi::moonlab_ca_mps_z(self.handle.as_ptr(), q) })
    }
    pub fn cnot(&mut self, c: u32, t: u32) -> Result<()> {
        check(unsafe { ffi::moonlab_ca_mps_cnot(self.handle.as_ptr(), c, t) })
    }
    pub fn cz(&mut self, a: u32, b: u32) -> Result<()> {
        check(unsafe { ffi::moonlab_ca_mps_cz(self.handle.as_ptr(), a, b) })
    }
    pub fn swap(&mut self, a: u32, b: u32) -> Result<()> {
        check(unsafe { ffi::moonlab_ca_mps_swap(self.handle.as_ptr(), a, b) })
    }

    // --- Non-Clifford rotations: pushed into the MPS factor. ---

    pub fn rx(&mut self, q: u32, theta: f64) -> Result<()> {
        check(unsafe { ffi::moonlab_ca_mps_rx(self.handle.as_ptr(), q, theta) })
    }
    pub fn ry(&mut self, q: u32, theta: f64) -> Result<()> {
        check(unsafe { ffi::moonlab_ca_mps_ry(self.handle.as_ptr(), q, theta) })
    }
    pub fn rz(&mut self, q: u32, theta: f64) -> Result<()> {
        check(unsafe { ffi::moonlab_ca_mps_rz(self.handle.as_ptr(), q, theta) })
    }
    pub fn t_gate(&mut self, q: u32) -> Result<()> {
        check(unsafe { ffi::moonlab_ca_mps_t_gate(self.handle.as_ptr(), q) })
    }
    pub fn t_dagger(&mut self, q: u32) -> Result<()> {
        check(unsafe { ffi::moonlab_ca_mps_t_dagger(self.handle.as_ptr(), q) })
    }
    pub fn phase(&mut self, q: u32, theta: f64) -> Result<()> {
        check(unsafe { ffi::moonlab_ca_mps_phase(self.handle.as_ptr(), q, theta) })
    }

    /// Apply the gauge-aware stabilizer-subgroup warmstart Clifford.
    /// `paulis` is laid out row-major, shape `(num_gens, num_qubits)`,
    /// in the Moonlab Pauli-byte encoding (0=I, 1=X, 2=Y, 3=Z).
    pub fn gauge_warmstart(&mut self, paulis: &[u8], num_gens: u32) -> Result<()> {
        let n = self.num_qubits() as usize;
        if paulis.len() != n * num_gens as usize {
            return Err(QuantumError::Ffi(format!(
                "gauge_warmstart: paulis.len()={} != num_gens*num_qubits={}",
                paulis.len(),
                n * num_gens as usize,
            )));
        }
        check(unsafe {
            ffi::moonlab_ca_mps_gauge_warmstart(
                self.handle.as_ptr(),
                paulis.as_ptr(),
                num_gens,
            )
        })
    }
}

impl Drop for CaMps {
    fn drop(&mut self) {
        unsafe { ffi::moonlab_ca_mps_free(self.handle.as_ptr()) };
    }
}

fn check(rc: i32) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        let label = match rc {
            -1 => "ERR_INVALID",
            -2 => "ERR_QUBIT",
            -3 => "ERR_OOM",
            -4 => "ERR_BACKEND",
            -100 => "ERR_NOT_IMPLEMENTED",
            _ => "unknown",
        };
        Err(QuantumError::Ffi(format!(
            "CA-MPS rc = {rc} ({label})"
        )))
    }
}

/// Configuration for [`var_d_run`].
#[derive(Debug, Clone)]
pub struct VarDConfig {
    pub max_outer_iters: u32,
    pub imag_time_dtau: f64,
    pub imag_time_steps_per_outer: u32,
    pub clifford_passes_per_outer: u32,
    pub composite_2gate: bool,
    pub warmstart: Warmstart,
}

impl Default for VarDConfig {
    fn default() -> Self {
        Self {
            max_outer_iters: 25,
            imag_time_dtau: 0.10,
            imag_time_steps_per_outer: 4,
            clifford_passes_per_outer: 8,
            composite_2gate: false,
            warmstart: Warmstart::Identity,
        }
    }
}

/// Run the variational-D alternating ground-state search.
///
/// `paulis` is laid out row-major, shape `(num_terms, num_qubits)`,
/// with `coeffs.len() == num_terms`.  When
/// `cfg.warmstart == Warmstart::StabilizerSubgroup`, supply
/// `stab_paulis` of shape `(stab_num_gens, num_qubits)`; otherwise
/// pass `&[]` and `stab_num_gens = 0`.
///
/// Returns the final variational energy.
pub fn var_d_run(
    state: &mut CaMps,
    paulis: &[u8],
    coeffs: &[f64],
    num_terms: u32,
    stab_paulis: &[u8],
    stab_num_gens: u32,
    cfg: &VarDConfig,
) -> Result<f64> {
    let n = state.num_qubits() as usize;
    if paulis.len() != n * num_terms as usize || coeffs.len() != num_terms as usize {
        return Err(QuantumError::Ffi(
            "var_d_run: paulis or coeffs shape mismatch".into(),
        ));
    }
    if cfg.warmstart == Warmstart::StabilizerSubgroup
        && stab_paulis.len() != n * stab_num_gens as usize
    {
        return Err(QuantumError::Ffi(
            "var_d_run: stab_paulis shape mismatch".into(),
        ));
    }

    let mut out_e: f64 = 0.0;
    let stab_ptr = if stab_num_gens > 0 {
        stab_paulis.as_ptr()
    } else {
        std::ptr::null()
    };

    let rc = unsafe {
        ffi::moonlab_ca_mps_var_d_run(
            state.handle.as_ptr(),
            paulis.as_ptr(),
            coeffs.as_ptr(),
            num_terms,
            cfg.max_outer_iters,
            cfg.imag_time_dtau,
            cfg.imag_time_steps_per_outer,
            cfg.clifford_passes_per_outer,
            if cfg.composite_2gate { 1 } else { 0 },
            cfg.warmstart as i32,
            stab_ptr,
            stab_num_gens,
            &mut out_e as *mut f64,
        )
    };
    check(rc)?;
    Ok(out_e)
}

/// Apply the gauge-aware warmstart Clifford to a fresh CA-MPS state.
/// Standalone use case, no var-D loop.
pub fn gauge_warmstart(state: &mut CaMps, paulis: &[u8], num_gens: u32) -> Result<()> {
    state.gauge_warmstart(paulis, num_gens)
}

/// Build the 1+1D Z2 LGT Pauli sum on `num_matter_sites` matter sites.
///
/// Returns `(paulis, coeffs, num_qubits)` where `paulis` is shape
/// `(num_terms, num_qubits)` row-major, `coeffs.len() == num_terms`,
/// and `num_qubits = 2 * num_matter_sites - 1`.  Memory is allocated
/// by libquantumsim and copied into safe Rust `Vec`s here, then the
/// libquantumsim allocations are freed.
pub fn z2_lgt_1d_build(
    num_matter_sites: u32,
    t_hop: f64,
    h_link: f64,
    mass: f64,
    gauss_penalty: f64,
) -> Result<(Vec<u8>, Vec<f64>, u32)> {
    let mut out_paulis: *mut u8 = std::ptr::null_mut();
    let mut out_coeffs: *mut f64 = std::ptr::null_mut();
    let mut out_terms: u32 = 0;
    let mut out_qubits: u32 = 0;

    let rc = unsafe {
        ffi::moonlab_z2_lgt_1d_build(
            num_matter_sites,
            t_hop,
            h_link,
            mass,
            gauss_penalty,
            &mut out_paulis as *mut *mut u8,
            &mut out_coeffs as *mut *mut f64,
            &mut out_terms,
            &mut out_qubits,
        )
    };
    if rc != 0 {
        return Err(QuantumError::Ffi(format!(
            "moonlab_z2_lgt_1d_build returned {rc}"
        )));
    }

    let total = out_terms as usize * out_qubits as usize;
    let paulis = unsafe { std::slice::from_raw_parts(out_paulis, total).to_vec() };
    let coeffs =
        unsafe { std::slice::from_raw_parts(out_coeffs, out_terms as usize).to_vec() };

    // The C builder allocates with calloc; free via libc::free.
    unsafe {
        libc::free(out_paulis as *mut libc::c_void);
        libc::free(out_coeffs as *mut libc::c_void);
    }

    Ok((paulis, coeffs, out_qubits))
}

/// Return the interior Gauss-law operator at matter site `site_x`.
/// `G_x = X_{2x-1} Z_{2x} X_{2x+1}` for `1 <= site_x <= N - 2`.
pub fn z2_lgt_1d_gauss_law(num_matter_sites: u32, site_x: u32) -> Result<Vec<u8>> {
    let nq = (2 * num_matter_sites - 1) as usize;
    let mut out = vec![0u8; nq];
    let rc = unsafe {
        ffi::moonlab_z2_lgt_1d_gauss_law(num_matter_sites, site_x, out.as_mut_ptr())
    };
    if rc != 0 {
        return Err(QuantumError::Ffi(format!(
            "moonlab_z2_lgt_1d_gauss_law returned {rc}"
        )));
    }
    Ok(out)
}

/// Pretty-print a Moonlab status code for the named module.
pub fn status_string(module: StatusModule, status: i32) -> String {
    let p: *const c_char = unsafe {
        ffi::moonlab_status_string(module as i32, status)
    };
    if p.is_null() {
        return format!("<unknown module={module:?} status={status}>");
    }
    unsafe {
        std::ffi::CStr::from_ptr(p)
            .to_string_lossy()
            .into_owned()
    }
}
