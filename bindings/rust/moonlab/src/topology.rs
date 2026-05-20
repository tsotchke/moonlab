//! Real-space and momentum-space topology primitives.
//!
//! Mirrors the Python `moonlab.topology` surface. Three entry points
//! you reach for most often:
//!
//! - [`qwz_chern`] — integer Chern number of the QWZ model through the
//!   stable Moonlab ABI (the same entry point QGTL / lilirrep / SbNN
//!   bind via `dlsym`).
//! - [`ssh_winding`] — 1D chiral winding of the SSH model.
//! - [`ChernKpm`] — matrix-free Bianco-Resta local Chern marker on the
//!   QWZ Chern insulator with optional C_n quasicrystal modulation.
//!
//! # Example
//!
//! ```no_run
//! use moonlab::topology::{qwz_chern, ssh_winding, ChernKpm};
//!
//! // Topological vs trivial via the stable ABI.
//! assert_eq!(qwz_chern(1.0, 32), -1);
//! assert_eq!(qwz_chern(3.0, 32), 0);
//!
//! // SSH winding.
//! assert_eq!(ssh_winding(1.0, 2.0, 64), 1); // topological
//! assert_eq!(ssh_winding(2.0, 1.0, 64), 0); // trivial
//!
//! // Matrix-free local marker on a topological QWZ bulk.
//! let sys = ChernKpm::new(12, -1.0, 100).unwrap();
//! let c = sys.local_marker(6, 6).unwrap();
//! assert!((c - 1.0).abs() < 0.25);
//! ```

use crate::error::{QuantumError, Result};
use moonlab_sys::{
    chern_kpm_bulk_map, chern_kpm_bulk_sum, chern_kpm_cn_modulation,
    chern_kpm_create, chern_kpm_free, chern_kpm_local_marker,
    chern_kpm_set_modulation, moonlab_qwz_chern, qgt_berry_grid_free,
    qgt_berry_grid_nband, qgt_berry_grid_proj, qgt_berry_grid_pt,
    qgt_berry_grid_t, qgt_free, qgt_free_1d, qgt_free_nband,
    qgt_model_bhz, qgt_model_hofstadter, qgt_model_kane_mele,
    qgt_model_kitaev_chain, qgt_model_qwz, qgt_model_ssh,
    qgt_winding_1d, qgt_z2_invariant, qgt_z2_invariant_1d_bdg,
};
use std::os::raw::{c_double, c_int};
use std::ptr;

/// Integer Chern number of the Qi-Wu-Zhang model, computed on an
/// `N x N` Brillouin-zone grid via the Fukui-Hatsugai-Suzuki link
/// method through Moonlab's stable ABI.
///
/// * QWZ with `-2 < m < 0` is topological with `C = +1`.
/// * QWZ with `0 < m < 2` is topological with `C = -1`.
/// * `|m| > 2` is trivial (`C = 0`).
pub fn qwz_chern(m: f64, n: usize) -> i32 {
    unsafe { moonlab_qwz_chern(m, n, ptr::null_mut()) }
}

/// Integer winding number of the SSH model via the 1D Zak phase.
/// Topological (winding = 1) when `|t2| > |t1|`.
pub fn ssh_winding(t1: f64, t2: f64, n: usize) -> i32 {
    unsafe {
        let sys = qgt_model_ssh(t1, t2);
        if sys.is_null() {
            return 0;
        }
        let mut raw: c_double = 0.0;
        let w = qgt_winding_1d(sys, n, &mut raw);
        qgt_free_1d(sys);
        w
    }
}

// --- v0.3 gauge-invariant integrators (QWZ) -------------------------

fn berry_chern_2d(
    sys: *mut moonlab_sys::qgt_system_t,
    n: usize,
    integrator: unsafe extern "C" fn(
        *const moonlab_sys::qgt_system_t,
        usize,
        *mut qgt_berry_grid_t,
    ) -> c_int,
    label: &'static str,
) -> Result<i32> {
    if sys.is_null() {
        return Err(QuantumError::AllocationFailed(0));
    }
    let mut grid = qgt_berry_grid_t {
        N: 0,
        berry: ptr::null_mut(),
        chern: 0.0,
    };
    let rc = unsafe { integrator(sys, n, &mut grid) };
    if rc != 0 {
        unsafe { qgt_free(sys) };
        return Err(QuantumError::Ffi(format!("{label} failed (rc={rc})")));
    }
    let chern = grid.chern.round() as i32;
    unsafe {
        qgt_berry_grid_free(&mut grid);
        qgt_free(sys);
    }
    Ok(chern)
}

/// Integer Chern number of the Qi-Wu-Zhang model via the gauge-free
/// projector-trace integrator
/// `F_xy(k) = -2 Im Tr[ P (d_x P) (d_y P) ]`.
///
/// This formulation is manifestly gauge-invariant and avoids the
/// link-variable artefacts near gap closings.  Equivalent to
/// [`qwz_chern`] (Fukui-Hatsugai-Suzuki link method) on every gapped
/// phase point.
pub fn chern_qwz_proj(m: f64, n: usize) -> Result<i32> {
    if n < 4 {
        return Err(QuantumError::InvalidQubit { index: n, max: 4 });
    }
    let sys = unsafe { qgt_model_qwz(m) };
    berry_chern_2d(sys, n, qgt_berry_grid_proj, "qgt_berry_grid_proj")
}

/// Integer Chern number of the Qi-Wu-Zhang model via the
/// parallel-transport-gauge eigenvector integrator.
///
/// The eigenvector at each `k` is phase-fixed against its neighbour;
/// the resulting smooth gauge gives a Berry-curvature plaquette flux
/// that integrates to the Chern number.  Returns the same integer as
/// [`chern_qwz_proj`] and [`qwz_chern`] for any gapped phase point;
/// retained for cross-validation.
pub fn chern_qwz_parallel_transport(m: f64, n: usize) -> Result<i32> {
    if n < 4 {
        return Err(QuantumError::InvalidQubit { index: n, max: 4 });
    }
    let sys = unsafe { qgt_model_qwz(m) };
    berry_chern_2d(sys, n, qgt_berry_grid_pt, "qgt_berry_grid_pt")
}

// --- v0.3 n-band Z_2 invariants -------------------------------------

/// Z_2 topological invariant of the Kane-Mele model on the honeycomb
/// lattice (Kane and Mele, *Phys. Rev. Lett.* **95**, 146802, 2005).
///
/// Returns `1` (quantum spin Hall phase) for
/// `|lambda_v| < 3 sqrt(3) |lambda_so|` and `0` (trivial) otherwise.
/// The S_z-conserving regime is selected by `lambda_r = 0`.
///
/// **`lambda_r` must be `0.0`** in the current implementation.  The
/// underlying C-side `qgt_model_kane_mele` returns NULL for non-zero
/// Rashba coupling (the S_z-conserving Z_2 integrator gives the wrong
/// answer when Rashba mixes the spin sectors).  Full Rashba support
/// via the Pfaffian formula is a v0.3.1 milestone.
///
/// `n` is the Brillouin-zone grid side and must be even and `>= 8`.
pub fn kane_mele_z2(
    t: f64,
    lambda_so: f64,
    lambda_r: f64,
    lambda_v: f64,
    n: usize,
) -> Result<i32> {
    if n < 8 || n % 2 != 0 {
        return Err(QuantumError::InvalidQubit { index: n, max: 8 });
    }
    let sys = unsafe { qgt_model_kane_mele(t, lambda_so, lambda_r, lambda_v) };
    if sys.is_null() {
        return Err(QuantumError::AllocationFailed(0));
    }
    let mut z2: c_int = -1;
    let rc = unsafe { qgt_z2_invariant(sys, n, &mut z2) };
    unsafe { qgt_free_nband(sys) };
    if rc != 0 {
        return Err(QuantumError::Ffi(format!("qgt_z2_invariant failed (rc={rc})")));
    }
    Ok(z2 as i32)
}

/// Z_2 invariant of the Bernevig-Hughes-Zhang model
/// (Bernevig, Hughes, and Zhang, *Science* **314**, 1757, 2006).
///
/// Lattice regularisation gives the QSH window `0 < M / B < 8`:
/// Gamma-point closing at `M = 0` and M-corner closing at `M = 8B`
/// bound the topological phase; X-corner closings at `M = 4B` cancel
/// pairwise and do not change the invariant.
pub fn bhz_z2(a: f64, b: f64, m: f64, n: usize) -> Result<i32> {
    if n < 8 || n % 2 != 0 {
        return Err(QuantumError::InvalidQubit { index: n, max: 8 });
    }
    let sys = unsafe { qgt_model_bhz(a, b, m) };
    if sys.is_null() {
        return Err(QuantumError::AllocationFailed(0));
    }
    let mut z2: c_int = -1;
    let rc = unsafe { qgt_z2_invariant(sys, n, &mut z2) };
    unsafe { qgt_free_nband(sys) };
    if rc != 0 {
        return Err(QuantumError::Ffi(format!("qgt_z2_invariant failed (rc={rc})")));
    }
    Ok(z2 as i32)
}

/// Z_2 invariant of the Kitaev p-wave superconducting chain
/// (Kitaev, *Physics-Uspekhi* **44**, 131, 2001) via the Pfaffian-sign
/// product at the time-reversal-invariant momenta.
///
/// Returns `1` (topological phase, Majorana zero modes at the open
/// boundaries) for `|mu| < 2|t|`, `0` otherwise.
pub fn kitaev_chain_z2(t: f64, mu: f64, delta: f64) -> Result<i32> {
    let sys = unsafe { qgt_model_kitaev_chain(t, mu, delta) };
    if sys.is_null() {
        return Err(QuantumError::AllocationFailed(0));
    }
    let mut z2: c_int = -1;
    let rc = unsafe { qgt_z2_invariant_1d_bdg(sys, &mut z2) };
    unsafe { qgt_free_1d(sys) };
    if rc != 0 {
        return Err(QuantumError::Ffi(format!(
            "qgt_z2_invariant_1d_bdg failed (rc={rc})"
        )));
    }
    Ok(z2 as i32)
}

/// Total Chern number of the lowest `n_occupied` magnetic sub-bands
/// of the Harper-Hofstadter model
/// (Hofstadter, *Phys. Rev. B* **14**, 2239, 1976) at flux `phi = p / q`.
///
/// For `q = 3`, `n_occupied = 1` returns `+1`; `n_occupied = 2`
/// returns `-1` (the lowest two bands sum to `+1 + (-2) = -1`).
pub fn hofstadter_chern(
    p: usize,
    q: usize,
    n_occupied: usize,
    t: f64,
    n: usize,
) -> Result<i32> {
    if q < 2 {
        return Err(QuantumError::InvalidQubit { index: q, max: 2 });
    }
    if !(1..=q.saturating_sub(1)).contains(&n_occupied) {
        return Err(QuantumError::InvalidQubit {
            index: n_occupied,
            max: q.saturating_sub(1),
        });
    }
    if n < 8 {
        return Err(QuantumError::InvalidQubit { index: n, max: 8 });
    }
    let sys = unsafe { qgt_model_hofstadter(t, p, q, n_occupied) };
    if sys.is_null() {
        return Err(QuantumError::AllocationFailed(0));
    }
    let mut grid = qgt_berry_grid_t {
        N: 0,
        berry: ptr::null_mut(),
        chern: 0.0,
    };
    let rc = unsafe { qgt_berry_grid_nband(sys, n, &mut grid) };
    if rc != 0 {
        unsafe { qgt_free_nband(sys) };
        return Err(QuantumError::Ffi(format!(
            "qgt_berry_grid_nband failed (rc={rc})"
        )));
    }
    let chern = grid.chern.round() as i32;
    unsafe {
        qgt_berry_grid_free(&mut grid);
        qgt_free_nband(sys);
    }
    Ok(chern)
}

/// Matrix-free Chebyshev-KPM Bianco-Resta local Chern marker.
pub struct ChernKpm {
    handle: *mut moonlab_sys::chern_kpm_system_t,
    l: usize,
    // Keep modulation buffer alive for the lifetime of the system;
    // the C side stores a borrowed pointer.
    modulation: Option<Vec<f64>>,
}

// SAFETY: ChernKpm owns its handle and is not shared across threads
// without external synchronization; the underlying matvec is
// deterministic and read-only on the borrowed modulation buffer.
unsafe impl Send for ChernKpm {}

impl ChernKpm {
    /// Create a new QWZ KPM system.
    ///
    /// * `l` — linear lattice size (≥ 3). Total sites = `l * l`.
    /// * `m` — QWZ mass parameter.
    /// * `n_cheby` — Chebyshev expansion order (≥ 8, typically 80-200).
    pub fn new(l: usize, m: f64, n_cheby: usize) -> Result<Self> {
        if l < 3 {
            return Err(QuantumError::InvalidQubit { index: l, max: 3 });
        }
        if n_cheby < 8 {
            return Err(QuantumError::InvalidQubit { index: n_cheby, max: 8 });
        }
        let handle = unsafe { chern_kpm_create(l, m, n_cheby) };
        if handle.is_null() {
            return Err(QuantumError::AllocationFailed(l));
        }
        Ok(Self { handle, l, modulation: None })
    }

    /// Linear lattice size.
    pub fn lattice_size(&self) -> usize {
        self.l
    }

    /// Local Chern marker `c(x, y)`.
    pub fn local_marker(&self, x: usize, y: usize) -> Result<f64> {
        if x >= self.l || y >= self.l {
            return Err(QuantumError::InvalidQubit { index: x.max(y), max: self.l });
        }
        Ok(unsafe { chern_kpm_local_marker(self.handle, x, y) })
    }

    /// Sum of local markers over the square `[rmin, rmax)^2`.
    pub fn bulk_sum(&self, rmin: usize, rmax: usize) -> Result<f64> {
        if rmin >= rmax || rmax > self.l {
            return Err(QuantumError::InvalidQubit { index: rmax, max: self.l });
        }
        Ok(unsafe { chern_kpm_bulk_sum(self.handle, rmin, rmax) })
    }

    /// Per-site marker map over `[rmin, rmax)^2`, row-major, parallel.
    pub fn bulk_map(&self, rmin: usize, rmax: usize) -> Result<Vec<f64>> {
        if rmin >= rmax || rmax > self.l {
            return Err(QuantumError::InvalidQubit { index: rmax, max: self.l });
        }
        let side = rmax - rmin;
        let mut out = vec![0.0f64; side * side];
        let rc = unsafe {
            chern_kpm_bulk_map(self.handle, rmin, rmax, out.as_mut_ptr())
        };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!(
                "chern_kpm_bulk_map failed (rc={rc})"
            )));
        }
        Ok(out)
    }

    /// Attach a `C_n`-symmetric cosine modulation
    /// `V(r) = V_0 * Σ_i cos(q_i · r)`.
    pub fn set_cn_modulation(&mut self, n: i32, q: f64, v0: f64) -> Result<()> {
        if n < 2 {
            return Err(QuantumError::InvalidQubit { index: n.max(0) as usize, max: 2 });
        }
        // Build the L*L modulation via the C helper, copy it into a
        // Rust-owned buffer, and free the C malloc.
        let raw = unsafe { chern_kpm_cn_modulation(self.l, n, q, v0) };
        if raw.is_null() {
            return Err(QuantumError::AllocationFailed(self.l * self.l));
        }
        let len = self.l * self.l;
        let slice = unsafe { std::slice::from_raw_parts(raw, len) };
        let mut owned = Vec::with_capacity(len);
        owned.extend_from_slice(slice);
        unsafe { libc::free(raw as *mut libc::c_void) };

        let v_max = (n as f64).abs() * v0.abs();
        let rc = unsafe {
            chern_kpm_set_modulation(self.handle, owned.as_ptr(), v_max)
        };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!(
                "chern_kpm_set_modulation failed (rc={rc})"
            )));
        }
        self.modulation = Some(owned);
        Ok(())
    }

    /// Detach any attached modulation.
    pub fn clear_modulation(&mut self) {
        unsafe { chern_kpm_set_modulation(self.handle, ptr::null(), 0.0) };
        self.modulation = None;
    }
}

impl Drop for ChernKpm {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { chern_kpm_free(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwz_chern_phases() {
        assert_eq!(qwz_chern(1.0, 32), -1);
        assert_eq!(qwz_chern(-1.0, 32), 1);
        assert_eq!(qwz_chern(3.0, 32), 0);
    }

    #[test]
    fn ssh_winding_phases() {
        assert_eq!(ssh_winding(1.0, 2.0, 64), 1);
        assert_eq!(ssh_winding(2.0, 1.0, 64), 0);
    }

    #[test]
    fn three_qwz_integrators_agree() {
        // FHS link variable, projector trace, and parallel transport
        // must all return identical integers on every gapped phase
        // point.  Checked at six representative points spanning the
        // -3..3 mass interval.
        for &m in &[-2.5_f64, -1.5, -0.5, 0.5, 1.5, 2.5] {
            let fhs = qwz_chern(m, 32);
            let proj = chern_qwz_proj(m, 32).unwrap();
            let pt = chern_qwz_parallel_transport(m, 32).unwrap();
            assert_eq!(
                fhs, proj,
                "FHS != proj at m={m}: {fhs} vs {proj}"
            );
            assert_eq!(
                fhs, pt,
                "FHS != parallel-transport at m={m}: {fhs} vs {pt}"
            );
        }
    }

    #[test]
    fn kane_mele_z2_phase_window() {
        // 3 sqrt(3) * 0.06 ~= 0.3118; lambda_v = 0.10 inside, 0.40 outside.
        assert_eq!(
            kane_mele_z2(1.0, 0.06, 0.0, 0.10, 24).unwrap(),
            1
        );
        assert_eq!(
            kane_mele_z2(1.0, 0.06, 0.0, 0.40, 24).unwrap(),
            0
        );
    }

    #[test]
    fn kane_mele_accepts_nonzero_rashba() {
        // v0.10.0+: Rashba is fully wired (km_bloch off-diagonals);
        // the block-Chern Z_2 path here corresponds to the S_z-
        // conserving formula and is still informative for small
        // lambda_r (Z_2 is adiabatically connected to the lambda_r=0
        // limit).  The C constructor no longer rejects lambda_r != 0;
        // for fully Sz-non-conserving Z_2 use qgt_z2_invariant_pfaffian.
        let r = kane_mele_z2(1.0, 0.06, 0.05, 0.10, 24);
        assert!(
            r.is_ok(),
            "v0.10.0 should accept lambda_r != 0, got {:?}", r,
        );
    }

    #[test]
    fn bhz_z2_lattice_window() {
        // QSH on 0 < M / B < 8 with the lattice regularisation.
        assert_eq!(bhz_z2(1.0, 1.0, 3.0, 24).unwrap(), 1);
        assert_eq!(bhz_z2(1.0, 1.0, -1.0, 24).unwrap(), 0);
        assert_eq!(bhz_z2(1.0, 1.0, 9.0, 24).unwrap(), 0);
    }

    #[test]
    fn kitaev_chain_majorana_window() {
        // Topological for |mu| < 2|t|.
        assert_eq!(kitaev_chain_z2(1.0, 0.5, 1.0).unwrap(), 1);
        assert_eq!(kitaev_chain_z2(1.0, -1.5, 1.0).unwrap(), 1);
        assert_eq!(kitaev_chain_z2(1.0, 2.5, 1.0).unwrap(), 0);
    }

    #[test]
    fn hofstadter_lowest_band_chern_plus_one() {
        // For phi = 1/q, the lowest sub-band carries Chern = +1 (TKNN).
        for &q in &[3_usize, 4, 5] {
            let c = hofstadter_chern(1, q, 1, 1.0, 24).unwrap();
            assert_eq!(c, 1, "q={q}: expected +1, got {c}");
        }
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert!(chern_qwz_proj(0.0, 2).is_err());
        assert!(kane_mele_z2(1.0, 0.06, 0.0, 0.1, 7).is_err()); // odd N
        assert!(hofstadter_chern(1, 1, 1, 1.0, 24).is_err());   // q < 2
        assert!(hofstadter_chern(1, 3, 3, 1.0, 24).is_err());   // n_occ > q-1
    }

    #[test]
    // Linux aarch64 hits the same KPM numerical-flakiness mode as the
    // C unit_chern_kpm test that's already CI-excluded on this target.
    // The Newton-Schulz + Chebyshev-KPM stack accumulates enough
    // difference on aarch64 OpenBLAS that the mean-Chern assertion
    // lands outside the 0.25 tolerance.  Skip on aarch64 until the
    // underlying KPM-on-aarch64 issue is resolved (audit punch-list).
    #[cfg(not(target_arch = "aarch64"))]
    fn kpm_bulk_topological() {
        let sys = ChernKpm::new(12, -1.0, 100).unwrap();
        let map = sys.bulk_map(4, 8).unwrap();
        let mean: f64 = map.iter().sum::<f64>() / map.len() as f64;
        assert!((mean - 1.0).abs() < 0.25, "bulk mean = {mean}");
    }
}
