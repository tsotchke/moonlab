//! Time-dependent variational principle (TDVP) for MPS dynamics
//! (v0.4 adaptive-bond surface).
//!
//! Safe Rust wrapper around `src/algorithms/tensor_network/tdvp.{c,h}`:
//! evolves an MPS in real or imaginary time, with an optional
//! entropy-feedback PID controller that selects each bond's
//! truncation threshold individually to meet a target accuracy
//! budget (arXiv:2604.03960).  The legacy fixed-`max_bond_dim` path
//! is the default and is bit-identical to v0.3.1.
//!
//! # Quick start
//!
//! ```no_run
//! use moonlab::tdvp::{
//!     EvolutionType, Mpo, Mps, TdvpConfig, TdvpEngine,
//! };
//!
//! // Build an 8-site Heisenberg chain and a random initial MPS.
//! let mpo = Mpo::heisenberg(8, 1.0, 1.0, 0.0).unwrap();
//! let mps = Mps::random(8, /*chi_init=*/8, /*max_bond=*/32, 1e-12).unwrap();
//!
//! let mut config = TdvpConfig::adaptive(1e-3);
//! config.evolution_type = EvolutionType::ImaginaryTime;
//! config.dt = 0.05;
//!
//! let mut engine = TdvpEngine::new(mps, mpo, config).unwrap();
//! for _ in 0..30 {
//!     let result = engine.step().unwrap();
//!     println!("E = {:+.6}, chi = {:?}",
//!              result.energy, result.bond_chi_distribution);
//! }
//! ```
//!
//! References:
//! - J. Haegeman et al., *Phys. Rev. B* **94**, 165116 (2016) -- the
//!   two-site TDVP integrator.
//! - arXiv:2604.03960 -- entropy-feedback bond control implemented
//!   by the adaptive path.

use crate::error::{QuantumError, Result};
use moonlab_sys::{
    dmrg_init_random_mps, mpo_free, mpo_heisenberg_create, mpo_tfim_create,
    tdvp_adaptive_bond_config_t, tdvp_bond_chi, tdvp_config_t, tdvp_engine_create,
    tdvp_engine_free, tdvp_engine_t, tdvp_evolve_to, tdvp_evolve_to_with_observable,
    tdvp_history_create, tdvp_history_free, tdvp_history_t, tdvp_result_clear,
    tdvp_result_t, tdvp_step, tn_mps_free, tn_state_config_create,
};
use std::os::raw::c_void;
use std::ptr;
use std::slice;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Direction of TDVP evolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum EvolutionType {
    RealTime      = 0,
    ImaginaryTime = 1,
}

/// One-site (fixed chi) vs two-site (adaptive chi) TDVP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Variant {
    OneSite = 0,
    TwoSite = 1,
}

/// Time-integrator backend selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum IntegratorType {
    Lanczos    = 0,
    RungeKutta = 1,
    Expokit    = 2,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Per-bond PID controller knobs.
#[derive(Debug, Clone, Copy)]
pub struct TdvpAdaptiveBondConfig {
    pub enabled: bool,
    pub target_entropy_error: f64,
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    pub chi_floor: u32,
    pub chi_ceiling: u32,
    pub alpha: f64,
}

impl TdvpAdaptiveBondConfig {
    /// Reference-paper PID gains (arXiv:2604.03960).
    pub fn reference(target_entropy_error: f64) -> Self {
        Self {
            enabled: true,
            target_entropy_error,
            kp: 0.5,
            ki: 0.05,
            kd: 0.1,
            chi_floor: 4,
            chi_ceiling: 4096,
            alpha: 8.0,
        }
    }

    /// All-zero, disabled.  Legacy fixed-bond path uses this.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            target_entropy_error: 0.0,
            kp: 0.0,
            ki: 0.0,
            kd: 0.0,
            chi_floor: 0,
            chi_ceiling: 0,
            alpha: 0.0,
        }
    }

    fn to_c(&self) -> tdvp_adaptive_bond_config_t {
        tdvp_adaptive_bond_config_t {
            enabled: self.enabled,
            target_entropy_error: self.target_entropy_error,
            kp: self.kp,
            ki: self.ki,
            kd: self.kd,
            chi_floor: self.chi_floor,
            chi_ceiling: self.chi_ceiling,
            alpha: self.alpha,
        }
    }
}

/// TDVP configuration.  Mirrors `tdvp_config_t`.
#[derive(Debug, Clone, Copy)]
pub struct TdvpConfig {
    pub evolution_type: EvolutionType,
    pub variant: Variant,
    pub integrator: IntegratorType,
    pub dt: f64,
    pub max_bond_dim: u32,
    pub svd_cutoff: f64,
    pub lanczos_max_iter: u32,
    pub lanczos_tol: f64,
    pub normalize: bool,
    pub verbose: bool,
    pub adaptive_bond: TdvpAdaptiveBondConfig,
}

impl TdvpConfig {
    /// Legacy fixed-bond config; bit-identical to v0.3.1
    /// `tdvp_config_default()`.
    pub fn default_legacy() -> Self {
        Self {
            evolution_type: EvolutionType::RealTime,
            variant: Variant::TwoSite,
            integrator: IntegratorType::Lanczos,
            dt: 0.01,
            max_bond_dim: 128,
            svd_cutoff: 1e-10,
            lanczos_max_iter: 50,
            lanczos_tol: 1e-12,
            normalize: true,
            verbose: false,
            adaptive_bond: TdvpAdaptiveBondConfig::disabled(),
        }
    }

    /// Adaptive-bond config at the reference-paper PID gains.
    pub fn adaptive(target_entropy_error: f64) -> Self {
        let ab = TdvpAdaptiveBondConfig::reference(target_entropy_error);
        let mut cfg = Self::default_legacy();
        cfg.max_bond_dim = ab.chi_ceiling;
        cfg.adaptive_bond = ab;
        cfg
    }

    fn to_c(&self) -> tdvp_config_t {
        tdvp_config_t {
            evolution_type: self.evolution_type as u32,
            variant: self.variant as u32,
            integrator: self.integrator as u32,
            dt: self.dt,
            max_bond_dim: self.max_bond_dim,
            svd_cutoff: self.svd_cutoff,
            lanczos_max_iter: self.lanczos_max_iter,
            lanczos_tol: self.lanczos_tol,
            normalize: self.normalize,
            verbose: self.verbose,
            adaptive_bond: self.adaptive_bond.to_c(),
        }
    }
}

impl Default for TdvpConfig {
    fn default() -> Self {
        Self::default_legacy()
    }
}

// ---------------------------------------------------------------------------
// MPO + MPS handles
// ---------------------------------------------------------------------------

/// Owned ctypes handle for `mpo_t`.
pub struct Mpo {
    handle: *mut moonlab_sys::mpo_t,
    num_sites: u32,
}

// SAFETY: an Mpo handle is not shared across threads without external
// synchronization.  The underlying matrices are read-only after
// construction.
unsafe impl Send for Mpo {}

impl Mpo {
    /// XXZ Heisenberg MPO
    /// `H = J * sum_i (X X + Y Y + Delta Z Z) + h * sum_i Z`.
    pub fn heisenberg(num_sites: u32, j: f64, delta: f64, h: f64) -> Result<Self> {
        let handle = unsafe { mpo_heisenberg_create(num_sites, j, delta, h) };
        if handle.is_null() {
            return Err(QuantumError::AllocationFailed(num_sites as usize));
        }
        Ok(Self { handle, num_sites })
    }

    /// Transverse-field Ising MPO
    /// `H = -J * sum_i Z Z - h * sum_i X`.
    pub fn tfim(num_sites: u32, j: f64, h: f64) -> Result<Self> {
        let handle = unsafe { mpo_tfim_create(num_sites, j, h) };
        if handle.is_null() {
            return Err(QuantumError::AllocationFailed(num_sites as usize));
        }
        Ok(Self { handle, num_sites })
    }

    pub fn num_sites(&self) -> u32 {
        self.num_sites
    }
}

impl Drop for Mpo {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { mpo_free(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

/// Owned ctypes handle for `tn_mps_state_t`.
pub struct Mps {
    handle: *mut moonlab_sys::tn_mps_state_t,
    num_sites: u32,
}

// SAFETY: same as Mpo -- the engine borrows the MPS pointer; not
// shared across threads concurrently.
unsafe impl Send for Mps {}

impl Mps {
    /// Build a random MPS with bulk bond dimension `chi_init` and a
    /// state-level `max_bond_dim` envelope.  Useful as the starting
    /// point for TDVP evolution.
    pub fn random(
        num_sites: u32,
        chi_init: u32,
        max_bond_dim: u32,
        svd_cutoff: f64,
    ) -> Result<Self> {
        let cfg = unsafe { tn_state_config_create(max_bond_dim, svd_cutoff) };
        let handle =
            unsafe { dmrg_init_random_mps(num_sites, chi_init, &cfg) };
        if handle.is_null() {
            return Err(QuantumError::AllocationFailed(num_sites as usize));
        }
        Ok(Self { handle, num_sites })
    }

    pub fn num_sites(&self) -> u32 {
        self.num_sites
    }
}

impl Drop for Mps {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { tn_mps_free(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Snapshot of one TDVP step.  `bond_chi_distribution` is populated
/// when the adaptive controller is enabled; empty otherwise.
#[derive(Debug, Clone)]
pub struct TdvpResult {
    pub time: f64,
    pub energy: f64,
    pub norm: f64,
    pub truncation_error: f64,
    pub max_bond_dim: u32,
    pub step_time: f64,
    pub bond_chi_distribution: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Owning wrapper around `tdvp_engine_t`.  The engine borrows the
/// MPS and MPO handles; both are moved into the engine and dropped
/// with it.
pub struct TdvpEngine {
    handle: *mut tdvp_engine_t,
    _mps: Mps,
    _mpo: Mpo,
    /// Internal scratch result.  We keep one allocation per engine
    /// instance so the C side can reuse the bond_chi buffer across
    /// steps without re-allocating; `step()` copies the relevant
    /// fields into a fresh `TdvpResult` before returning.
    result: tdvp_result_t,
}

// SAFETY: TdvpEngine owns its handle and is not shared across
// threads.
unsafe impl Send for TdvpEngine {}

impl TdvpEngine {
    /// Build a new engine.  Takes ownership of `mps` and `mpo`.
    pub fn new(mps: Mps, mpo: Mpo, config: TdvpConfig) -> Result<Self> {
        let c_cfg = config.to_c();
        let handle =
            unsafe { tdvp_engine_create(mps.handle, mpo.handle, &c_cfg) };
        if handle.is_null() {
            return Err(QuantumError::AllocationFailed(0));
        }
        let result = tdvp_result_t {
            time: 0.0,
            energy: 0.0,
            norm: 0.0,
            truncation_error: 0.0,
            max_bond_dim: 0,
            step_time: 0.0,
            bond_chi_distribution: ptr::null_mut(),
            n_bonds: 0,
        };
        Ok(Self {
            handle,
            _mps: mps,
            _mpo: mpo,
            result,
        })
    }

    /// Advance the state by one `dt` and return the step result.
    pub fn step(&mut self) -> Result<TdvpResult> {
        let rc = unsafe { tdvp_step(self.handle, &mut self.result) };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!("tdvp_step rc={rc}")));
        }
        Ok(self.snapshot())
    }

    /// Per-bond chi from the adaptive controller, or 0 when
    /// disabled.
    pub fn bond_chi(&self, bond: u32) -> u32 {
        unsafe { tdvp_bond_chi(self.handle, bond) }
    }

    /// Evolve to `target_time`, optionally recording every step into
    /// `history`.  Equivalent to looping [`Self::step`] and pushing
    /// each result through [`TdvpHistory::add`].
    pub fn evolve_to(
        &mut self,
        target_time: f64,
        history: Option<&mut TdvpHistory>,
    ) -> Result<()> {
        let hist_ptr = match history {
            Some(h) => h.handle,
            None => ptr::null_mut(),
        };
        let rc = unsafe { tdvp_evolve_to(self.handle, target_time, hist_ptr) };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!(
                "tdvp_evolve_to rc={rc}"
            )));
        }
        Ok(())
    }

    /// Evolve to `target_time`, invoking `observable_fn` after every
    /// step and storing its return value in `history.observables`.
    ///
    /// `observable_fn` receives the live MPS pointer and the current
    /// time; it must return a scalar measurement.  Panics inside the
    /// closure are caught and surfaced as a Rust error (the C side
    /// has no exception channel).
    pub fn evolve_with_observable<F>(
        &mut self,
        target_time: f64,
        history: &mut TdvpHistory,
        observable_fn: F,
    ) -> Result<()>
    where
        F: FnMut(*const moonlab_sys::tn_mps_state_t, f64) -> f64,
    {
        struct Closure<F: FnMut(*const moonlab_sys::tn_mps_state_t, f64) -> f64> {
            f: F,
            panic: Option<Box<dyn std::any::Any + Send + 'static>>,
        }
        let mut state = Closure {
            f: observable_fn,
            panic: None,
        };

        unsafe extern "C" fn trampoline<F>(
            mps: *const moonlab_sys::tn_mps_state_t,
            time: f64,
            user: *mut c_void,
        ) -> f64
        where
            F: FnMut(*const moonlab_sys::tn_mps_state_t, f64) -> f64,
        {
            let state = &mut *(user as *mut Closure<F>);
            if state.panic.is_some() {
                return 0.0;
            }
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                (state.f)(mps, time)
            })) {
                Ok(v) => v,
                Err(e) => {
                    state.panic = Some(e);
                    0.0
                }
            }
        }

        let cb: unsafe extern "C" fn(
            *const moonlab_sys::tn_mps_state_t,
            f64,
            *mut c_void,
        ) -> f64 = trampoline::<F>;
        let rc = unsafe {
            tdvp_evolve_to_with_observable(
                self.handle,
                target_time,
                history.handle,
                Some(std::mem::transmute::<_, _>(cb)),
                &mut state as *mut _ as *mut c_void,
            )
        };
        if let Some(p) = state.panic.take() {
            std::panic::resume_unwind(p);
        }
        if rc != 0 {
            return Err(QuantumError::Ffi(format!(
                "tdvp_evolve_to_with_observable rc={rc}"
            )));
        }
        Ok(())
    }

    fn snapshot(&self) -> TdvpResult {
        let r = &self.result;
        let dist = if !r.bond_chi_distribution.is_null() && r.n_bonds > 0 {
            unsafe {
                slice::from_raw_parts(r.bond_chi_distribution, r.n_bonds as usize)
            }
            .to_vec()
        } else {
            Vec::new()
        };
        TdvpResult {
            time: r.time,
            energy: r.energy,
            norm: r.norm,
            truncation_error: r.truncation_error,
            max_bond_dim: r.max_bond_dim,
            step_time: r.step_time,
            bond_chi_distribution: dist,
        }
    }
}

impl Drop for TdvpEngine {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // Free the result's heap-owned bond_chi_distribution
            // before freeing the engine itself.
            unsafe { tdvp_result_clear(&mut self.result) };
            unsafe { tdvp_engine_free(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------

/// Step-by-step record of a TDVP evolution.
///
/// Records the time, energy, norm, optional scalar observable, and
/// per-bond chi snapshot for each step accepted via
/// [`TdvpEngine::evolve_to`] or
/// [`TdvpEngine::evolve_with_observable`].  Buffers grow automatically
/// as new steps land.
pub struct TdvpHistory {
    handle: *mut tdvp_history_t,
}

// SAFETY: TdvpHistory owns its handle and is not shared across
// threads without external synchronization.
unsafe impl Send for TdvpHistory {}

impl TdvpHistory {
    /// Construct an empty history with the requested initial
    /// capacity.  Passing 0 selects the C default (64).
    pub fn new(initial_capacity: u32) -> Result<Self> {
        let handle = unsafe { tdvp_history_create(initial_capacity) };
        if handle.is_null() {
            return Err(QuantumError::AllocationFailed(0));
        }
        Ok(Self { handle })
    }

    fn as_ref_struct(&self) -> &tdvp_history_t {
        unsafe { &*self.handle }
    }

    /// Number of steps recorded so far.
    pub fn num_steps(&self) -> u32 {
        self.as_ref_struct().num_steps
    }

    /// Copy of the per-step time column.
    pub fn times(&self) -> Vec<f64> {
        let s = self.as_ref_struct();
        if s.times.is_null() || s.num_steps == 0 {
            return Vec::new();
        }
        unsafe { slice::from_raw_parts(s.times, s.num_steps as usize) }.to_vec()
    }

    /// Copy of the per-step energy column.
    pub fn energies(&self) -> Vec<f64> {
        let s = self.as_ref_struct();
        if s.energies.is_null() || s.num_steps == 0 {
            return Vec::new();
        }
        unsafe { slice::from_raw_parts(s.energies, s.num_steps as usize) }
            .to_vec()
    }

    /// Copy of the per-step norm column.
    pub fn norms(&self) -> Vec<f64> {
        let s = self.as_ref_struct();
        if s.norms.is_null() || s.num_steps == 0 {
            return Vec::new();
        }
        unsafe { slice::from_raw_parts(s.norms, s.num_steps as usize) }
            .to_vec()
    }

    /// Copy of the per-step observable column.  Returns `None` if
    /// no observable has ever been recorded into this history.
    pub fn observables(&self) -> Option<Vec<f64>> {
        let s = self.as_ref_struct();
        if s.observables.is_null() {
            return None;
        }
        if s.num_steps == 0 {
            return Some(Vec::new());
        }
        Some(
            unsafe {
                slice::from_raw_parts(s.observables, s.num_steps as usize)
            }
            .to_vec(),
        )
    }

    /// Copy of the per-step per-bond chi snapshot, shape
    /// `(num_steps, n_bonds)` flattened row-major.  Returns `None`
    /// when the adaptive controller never recorded a distribution.
    pub fn bond_chi_history(&self) -> Option<(Vec<u32>, u32)> {
        let s = self.as_ref_struct();
        if s.bond_chi_history.is_null() || s.n_bonds == 0 || s.num_steps == 0 {
            return None;
        }
        let stride = s.n_bonds as usize;
        let len = (s.num_steps as usize) * stride;
        let row_major = unsafe {
            slice::from_raw_parts(s.bond_chi_history, len)
        }
        .to_vec();
        Some((row_major, s.n_bonds))
    }
}

impl Drop for TdvpHistory {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { tdvp_history_free(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_legacy() {
        let cfg = TdvpConfig::default_legacy();
        assert!(!cfg.adaptive_bond.enabled);
        assert_eq!(cfg.max_bond_dim, 128);
        assert_eq!(cfg.dt, 0.01);
    }

    #[test]
    fn adaptive_config_has_reference_gains() {
        let cfg = TdvpConfig::adaptive(1e-3);
        assert!(cfg.adaptive_bond.enabled);
        assert_eq!(cfg.adaptive_bond.target_entropy_error, 1e-3);
        assert_eq!(cfg.adaptive_bond.kp, 0.5);
        assert_eq!(cfg.adaptive_bond.ki, 0.05);
        assert_eq!(cfg.adaptive_bond.kd, 0.1);
        assert_eq!(cfg.adaptive_bond.chi_floor, 4);
        assert_eq!(cfg.adaptive_bond.chi_ceiling, 4096);
        assert_eq!(cfg.max_bond_dim, cfg.adaptive_bond.chi_ceiling);
    }

    #[test]
    fn engine_legacy_one_step() {
        let mpo = Mpo::heisenberg(6, 1.0, 1.0, 0.0).unwrap();
        let mps = Mps::random(6, 4, 16, 1e-12).unwrap();
        let mut cfg = TdvpConfig::default_legacy();
        cfg.dt = 0.02;
        cfg.max_bond_dim = 16;
        let mut engine = TdvpEngine::new(mps, mpo, cfg).unwrap();
        let r = engine.step().unwrap();
        assert!(r.energy.is_finite());
        assert!(r.norm.is_finite());
        // Legacy path: distribution stays empty.
        assert!(r.bond_chi_distribution.is_empty());
    }

    #[test]
    fn engine_adaptive_populates_bond_chi_distribution() {
        let n: u32 = 8;
        let mpo = Mpo::heisenberg(n, 1.0, 1.0, 0.0).unwrap();
        let mps = Mps::random(n, 8, 32, 1e-12).unwrap();
        let mut cfg = TdvpConfig::adaptive(1e-3);
        cfg.evolution_type = EvolutionType::RealTime;
        cfg.dt = 0.02;
        cfg.adaptive_bond.chi_ceiling = 32;
        cfg.max_bond_dim = 32;
        let mut engine = TdvpEngine::new(mps, mpo, cfg).unwrap();
        let r = engine.step().unwrap();
        assert_eq!(r.bond_chi_distribution.len(), (n - 1) as usize);
        let floor = cfg.adaptive_bond.chi_floor;
        let ceil = cfg.adaptive_bond.chi_ceiling;
        for &chi in &r.bond_chi_distribution {
            assert!(
                chi == 0 || (chi >= floor && chi <= ceil),
                "bond chi={chi} out of [{floor}, {ceil}]"
            );
        }
        // Accessor agrees with the distribution.
        for b in 0..(n - 1) {
            assert_eq!(
                engine.bond_chi(b),
                r.bond_chi_distribution[b as usize]
            );
        }
    }

    #[test]
    fn real_time_energy_conservation() {
        let n: u32 = 8;
        let mpo = Mpo::heisenberg(n, 1.0, 1.0, 0.0).unwrap();
        let mps = Mps::random(n, 8, 32, 1e-12).unwrap();
        let mut cfg = TdvpConfig::adaptive(1e-3);
        cfg.evolution_type = EvolutionType::RealTime;
        cfg.dt = 0.02;
        cfg.adaptive_bond.chi_ceiling = 32;
        cfg.max_bond_dim = 32;
        let mut engine = TdvpEngine::new(mps, mpo, cfg).unwrap();
        let mut energies = Vec::with_capacity(5);
        for _ in 0..5 {
            energies.push(engine.step().unwrap().energy);
        }
        let e0 = energies[0];
        let max_drift = energies
            .iter()
            .map(|e| (e - e0).abs() / e0.abs().max(1e-12))
            .fold(0.0_f64, f64::max);
        assert!(max_drift < 5e-3, "max relative drift {max_drift:.3e}");
    }

    #[test]
    fn evolve_to_records_history() {
        let n: u32 = 6;
        let mpo = Mpo::tfim(n, 1.0, 1.0).unwrap();
        let mps = Mps::random(n, 8, 16, 1e-12).unwrap();
        let mut cfg = TdvpConfig::adaptive(1e-3);
        cfg.evolution_type = EvolutionType::ImaginaryTime;
        cfg.dt = 0.05;
        cfg.adaptive_bond.chi_ceiling = 16;
        cfg.max_bond_dim = 16;
        let mut engine = TdvpEngine::new(mps, mpo, cfg).unwrap();
        let mut hist = TdvpHistory::new(8).unwrap();
        assert_eq!(hist.num_steps(), 0);
        assert!(hist.observables().is_none());
        assert!(hist.bond_chi_history().is_none());

        engine.evolve_to(0.2, Some(&mut hist)).unwrap();

        assert!(hist.num_steps() >= 3);
        let times = hist.times();
        assert_eq!(times.len(), hist.num_steps() as usize);
        // Times must be strictly increasing.
        for w in times.windows(2) {
            assert!(w[1] > w[0]);
        }
        // Adaptive controller is on; bond-chi history present.
        let chi_hist = hist.bond_chi_history();
        assert!(chi_hist.is_some());
        let (flat, stride) = chi_hist.unwrap();
        assert_eq!(stride, n - 1);
        assert_eq!(flat.len(), (hist.num_steps() as usize) * (stride as usize));
        // observables column still unallocated.
        assert!(hist.observables().is_none());
    }

    #[test]
    fn evolve_with_observable_records_callback_values() {
        let n: u32 = 6;
        let mpo = Mpo::tfim(n, 1.0, 1.0).unwrap();
        let mps = Mps::random(n, 8, 16, 1e-12).unwrap();
        let mut cfg = TdvpConfig::adaptive(1e-3);
        cfg.evolution_type = EvolutionType::ImaginaryTime;
        cfg.dt = 0.05;
        cfg.adaptive_bond.chi_ceiling = 16;
        cfg.max_bond_dim = 16;
        let mut engine = TdvpEngine::new(mps, mpo, cfg).unwrap();
        let mut hist = TdvpHistory::new(8).unwrap();

        // Trivial observable: the time itself.  Pins the value
        // round-trip across the FFI callback boundary.
        engine
            .evolve_with_observable(0.2, &mut hist, |_mps, time| time)
            .unwrap();

        assert!(hist.num_steps() >= 3);
        let obs = hist.observables().expect("observables column was allocated");
        let times = hist.times();
        assert_eq!(obs.len(), times.len());
        for (o, t) in obs.iter().zip(times.iter()) {
            assert!((o - t).abs() < 1e-12, "obs {o} != time {t}");
        }
    }
}
