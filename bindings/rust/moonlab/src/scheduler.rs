//! Distributed scheduler Rust binding -- since v0.7.1.
//!
//! Wraps `src/distributed/scheduler.{c,h}` (v0.7.0).  Mirrors the
//! Python and JS surfaces.
//!
//! ```no_run
//! use moonlab::scheduler::Job;
//! use moonlab::qgtl::GateType;
//!
//! let mut j = Job::new(2).unwrap();
//! j.add_gate(GateType::H, 0, -1, &[]).unwrap();
//! j.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
//! j.set_num_shots(1024).unwrap();
//! j.set_num_workers(4).unwrap();
//! j.set_rng_seed(0xdeadbeef).unwrap();
//! let r = j.execute().unwrap();
//! assert!(r.outcomes.iter().all(|&o| o == 0 || o == 3));
//! ```

use crate::error::{QuantumError, Result};
use crate::qgtl::GateType;
use moonlab_sys::{
    moonlab_job_add_gate, moonlab_job_create, moonlab_job_free,
    moonlab_job_num_gates, moonlab_job_num_qubits, moonlab_job_num_shots,
    moonlab_job_num_workers, moonlab_job_results_free, moonlab_job_results_t,
    moonlab_job_set_num_shots, moonlab_job_set_num_workers,
    moonlab_job_set_rng_seed, moonlab_job_to_json, moonlab_scheduler_run,
    moonlab_list_vendor_noise_profiles, moonlab_lookup_vendor_noise_profile,
    moonlab_num_vendor_noise_profiles, moonlab_register_vendor_noise_profile,
    moonlab_scheduler_set_completion_hook, moonlab_unregister_vendor_noise_profile,
    moonlab_vendor_noise_profile_t,
};
use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::Mutex;

#[cfg(test)]
static SCHEDULER_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Execution outputs.
#[derive(Debug, Default)]
pub struct JobResults {
    pub num_qubits: i32,
    pub total_shots: i32,
    pub outcomes: Vec<u64>,
    pub num_workers_used: i32,
    pub worker_seconds: Vec<f64>,
}

/// Owned C handle to a `moonlab_job_t`.
pub struct Job {
    ptr: *mut c_void,
}

fn check(rc: i32, ctx: &str) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(QuantumError::Ffi(format!("{ctx}: rc={rc}")))
    }
}

impl Job {
    /// Allocate a job on `num_qubits` qubits.
    pub fn new(num_qubits: i32) -> Result<Self> {
        let p = unsafe { moonlab_job_create(num_qubits) };
        if p.is_null() {
            return Err(QuantumError::Ffi(format!(
                "moonlab_job_create({num_qubits}): NULL"
            )));
        }
        Ok(Self { ptr: p as *mut c_void })
    }

    pub fn num_qubits(&self) -> i32 {
        unsafe { moonlab_job_num_qubits(self.ptr as *const _) }
    }
    pub fn num_gates(&self) -> i32 {
        unsafe { moonlab_job_num_gates(self.ptr as *const _) }
    }
    pub fn num_shots(&self) -> i32 {
        unsafe { moonlab_job_num_shots(self.ptr as *const _) }
    }
    pub fn num_workers(&self) -> i32 {
        unsafe { moonlab_job_num_workers(self.ptr as *const _) }
    }

    pub fn add_gate(&mut self,
                    gate: GateType,
                    target: i32,
                    control: i32,
                    params: &[f64]) -> Result<()> {
        let params_ptr = if params.is_empty() {
            ptr::null()
        } else {
            params.as_ptr()
        };
        let rc = unsafe {
            moonlab_job_add_gate(
                self.ptr as *mut _,
                gate as u32,
                target,
                control,
                params_ptr,
            )
        };
        check(rc, &format!("add_gate({gate:?}, target={target}, control={control})"))
    }

    pub fn set_num_shots(&mut self, n: i32) -> Result<()> {
        check(unsafe { moonlab_job_set_num_shots(self.ptr as *mut _, n) },
              &format!("set_num_shots({n})"))
    }
    pub fn set_num_workers(&mut self, n: i32) -> Result<()> {
        check(unsafe { moonlab_job_set_num_workers(self.ptr as *mut _, n) },
              &format!("set_num_workers({n})"))
    }
    pub fn set_rng_seed(&mut self, seed: u64) -> Result<()> {
        check(unsafe { moonlab_job_set_rng_seed(self.ptr as *mut _, seed) },
              "set_rng_seed")
    }

    /// Run the worker fan-out and return merged outcomes.
    pub fn execute(&mut self) -> Result<JobResults> {
        let mut res: moonlab_job_results_t = unsafe { std::mem::zeroed() };
        let rc = unsafe { moonlab_scheduler_run(self.ptr as *mut _, &mut res) };
        check(rc, "scheduler_run")?;
        let total = res.total_shots as usize;
        let nw = res.num_workers_used as usize;
        let outcomes = unsafe {
            std::slice::from_raw_parts(res.outcomes, total).to_vec()
        };
        let worker_seconds = unsafe {
            std::slice::from_raw_parts(res.worker_seconds, nw).to_vec()
        };
        let out = JobResults {
            num_qubits: res.num_qubits,
            total_shots: res.total_shots,
            outcomes,
            num_workers_used: res.num_workers_used,
            worker_seconds,
        };
        unsafe { moonlab_job_results_free(&mut res) };
        Ok(out)
    }

    /// Serialise the job as JSON (moonlab/job/v0.7.0 schema).
    pub fn to_json(&self) -> Result<String> {
        let needed = unsafe {
            moonlab_job_to_json(self.ptr as *const _, ptr::null_mut(), 0)
        };
        if needed < 0 {
            return Err(QuantumError::Ffi(format!("to_json size-probe: rc={needed}")));
        }
        let cap = needed as usize + 1;
        /* c_char so the as_mut_ptr() return type matches the FFI prototype
         * on both x86_64 (signed) and aarch64 (unsigned) platforms. */
        let mut buf = vec![0 as c_char; cap];
        let written = unsafe {
            moonlab_job_to_json(self.ptr as *const _, buf.as_mut_ptr(), cap)
        };
        if written < 0 {
            return Err(QuantumError::Ffi(format!("to_json: rc={written}")));
        }
        let cstr_bytes: Vec<u8> = buf[..written as usize]
            .iter().map(|&b| b as u8).collect();
        String::from_utf8(cstr_bytes)
            .map_err(|e| QuantumError::Ffi(format!("to_json utf8: {e}")))
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { moonlab_job_free(self.ptr as *mut _) };
            self.ptr = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_worker_bell() {
        let _scheduler_guard = SCHEDULER_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut j = Job::new(2).unwrap();
        j.add_gate(GateType::H, 0, -1, &[]).unwrap();
        j.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
        j.set_num_shots(1024).unwrap();
        j.set_num_workers(1).unwrap();
        j.set_rng_seed(0xdeadbeef).unwrap();
        let r = j.execute().unwrap();
        assert_eq!(r.total_shots, 1024);
        assert!(r.outcomes.iter().all(|&o| o == 0 || o == 3));
    }

    #[test]
    fn four_worker_bell() {
        let _scheduler_guard = SCHEDULER_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut j = Job::new(2).unwrap();
        j.add_gate(GateType::H, 0, -1, &[]).unwrap();
        j.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
        j.set_num_shots(1024).unwrap();
        j.set_num_workers(4).unwrap();
        j.set_rng_seed(0xdeadbeef).unwrap();
        let r = j.execute().unwrap();
        assert_eq!(r.num_workers_used, 4);
        let n00 = r.outcomes.iter().filter(|&&o| o == 0).count() as i32;
        let n11 = r.outcomes.iter().filter(|&&o| o == 3).count() as i32;
        let nother = r.outcomes.iter().filter(|&&o| o != 0 && o != 3).count();
        assert_eq!(nother, 0);
        assert_eq!(n00 + n11, 1024);
        assert!((n00 - 512).abs() < 80);
        assert_eq!(r.worker_seconds.len(), 4);
    }

    #[test]
    fn json_round_trip_schema() {
        let mut j = Job::new(2).unwrap();
        j.add_gate(GateType::H, 0, -1, &[]).unwrap();
        j.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
        j.set_num_shots(256).unwrap();
        j.set_num_workers(2).unwrap();
        let s = j.to_json().unwrap();
        assert!(s.contains("\"schema\": \"moonlab/job/v0.7.0\""));
        assert!(s.contains("\"num_qubits\": 2"));
        assert!(s.contains("\"num_shots\": 256"));
        assert!(s.contains("\"num_workers\": 2"));
    }

    #[test]
    fn rejects_zero_qubits() {
        assert!(Job::new(0).is_err());
    }

    #[test]
    fn rejects_zero_workers() {
        let mut j = Job::new(2).unwrap();
        assert!(j.set_num_workers(0).is_err());
    }
}

// ===================================================================
// Vendor-noise profile runtime registry (since v1.0.3)
// ===================================================================

/// Snapshot of a hardware-noise profile.  Returned by
/// [`lookup_vendor_noise_profile`]; passed to
/// [`register_vendor_noise_profile`].
#[derive(Debug, Clone)]
pub struct VendorNoiseProfile {
    pub p_gate_1q: f64,
    pub p_gate_2q: f64,
    pub p_readout: f64,
    pub description: String,
}

pub fn register_vendor_noise_profile(name: &str,
                                     profile: &VendorNoiseProfile) -> Result<()> {
    let c_name = CString::new(name)
        .map_err(|e| QuantumError::Ffi(format!("bad profile name: {e}")))?;
    let c_desc = CString::new(profile.description.as_str())
        .map_err(|e| QuantumError::Ffi(format!("bad description: {e}")))?;
    let c_prof = moonlab_vendor_noise_profile_t {
        p_gate_1q: profile.p_gate_1q,
        p_gate_2q: profile.p_gate_2q,
        p_readout: profile.p_readout,
        description: c_desc.as_ptr(),
    };
    let rc = unsafe {
        moonlab_register_vendor_noise_profile(c_name.as_ptr(), &c_prof)
    };
    if rc != 0 {
        return Err(QuantumError::Ffi(
            format!("register_vendor_noise_profile({name:?}): rc={rc}")));
    }
    Ok(())
}

pub fn unregister_vendor_noise_profile(name: &str) -> Result<()> {
    let c_name = CString::new(name)
        .map_err(|e| QuantumError::Ffi(format!("bad profile name: {e}")))?;
    let rc = unsafe { moonlab_unregister_vendor_noise_profile(c_name.as_ptr()) };
    if rc != 0 {
        return Err(QuantumError::Ffi(
            format!("unregister_vendor_noise_profile({name:?}): rc={rc}")));
    }
    Ok(())
}

pub fn lookup_vendor_noise_profile(name: &str) -> Option<VendorNoiseProfile> {
    let c_name = CString::new(name).ok()?;
    let p = unsafe { moonlab_lookup_vendor_noise_profile(c_name.as_ptr()) };
    if p.is_null() { return None; }
    unsafe {
        let raw = &*p;
        let desc = if raw.description.is_null() { String::new() } else {
            CStr::from_ptr(raw.description).to_str().unwrap_or("").to_owned()
        };
        Some(VendorNoiseProfile {
            p_gate_1q: raw.p_gate_1q,
            p_gate_2q: raw.p_gate_2q,
            p_readout: raw.p_readout,
            description: desc,
        })
    }
}

pub fn num_vendor_noise_profiles() -> i32 {
    unsafe { moonlab_num_vendor_noise_profiles() }
}

pub fn list_vendor_noise_profiles() -> Vec<String> {
    let n = num_vendor_noise_profiles();
    if n <= 0 { return Vec::new(); }
    let mut buf: Vec<*const c_char> = vec![ptr::null(); n as usize];
    let written = unsafe {
        moonlab_list_vendor_noise_profiles(buf.as_mut_ptr(), n) as usize
    };
    buf.into_iter().take(written).filter_map(|p| {
        if p.is_null() { None } else {
            unsafe { CStr::from_ptr(p).to_str().ok().map(|s| s.to_owned()) }
        }
    }).collect()
}

// ===================================================================
// Scheduler completion hook (since v1.0.3)
// ===================================================================

/// Subset of the job results that the Rust hook callback receives.
/// We don't surface the raw job pointer to keep the safe wrapper
/// independent of the FFI handle type.
#[derive(Debug, Clone)]
pub struct CompletionInfo {
    pub num_qubits: i32,
    pub total_shots: i32,
    pub backend: Option<String>,
}

/// Boxed callback fired after every successful scheduler.run().
pub type CompletionCallback = Box<dyn Fn(&CompletionInfo) + Send + Sync>;

// One slot, mutex-guarded so set/clear don't race the C side.  The
// trampoline reads the slot under lock to get a stable Arc on the
// callback before invoking it.
static COMPLETION_CALLBACK: Mutex<Option<std::sync::Arc<CompletionCallback>>> =
    Mutex::new(None);

extern "C" fn rust_completion_trampoline(
    _job: *const moonlab_sys::moonlab_job,
    results: *const moonlab_job_results_t,
    backend_name: *const c_char,
    _ctx: *mut c_void,
) {
    let arc_opt = COMPLETION_CALLBACK.lock().ok().and_then(|g| g.clone());
    let Some(arc) = arc_opt else { return; };
    if results.is_null() { return; }
    unsafe {
        let r = &*results;
        let backend = if backend_name.is_null() { None } else {
            CStr::from_ptr(backend_name).to_str().ok().map(|s| s.to_owned())
        };
        let info = CompletionInfo {
            num_qubits: r.num_qubits,
            total_shots: r.total_shots,
            backend,
        };
        // Catch panics so a panicking callback can't unwind into C.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (arc)(&info);
        }));
    }
}

/// Install a Rust callback that fires after every successful
/// scheduler.run().  Failed runs do not fire the hook.  Use cases:
/// billing meter, audit log, customer dashboard.  Since v1.0.3.
pub fn set_completion_hook<F>(f: F) -> Result<()>
where
    F: Fn(&CompletionInfo) + Send + Sync + 'static,
{
    let boxed: CompletionCallback = Box::new(f);
    let mut callback = COMPLETION_CALLBACK.lock().unwrap();
    let previous = callback.replace(std::sync::Arc::new(boxed));
    let rc = unsafe {
        moonlab_scheduler_set_completion_hook(
            Some(rust_completion_trampoline), ptr::null_mut())
    };
    if rc != 0 {
        *callback = previous;
        return Err(QuantumError::Ffi(
            format!("set_completion_hook: rc={rc}")));
    }
    Ok(())
}

/// Detach the active completion hook.
pub fn clear_completion_hook() -> Result<()> {
    let mut callback = COMPLETION_CALLBACK.lock().unwrap();
    let previous = callback.take();
    let rc = unsafe {
        moonlab_scheduler_set_completion_hook(None, ptr::null_mut())
    };
    if rc != 0 {
        *callback = previous;
        return Err(QuantumError::Ffi(
            format!("clear_completion_hook: rc={rc}")));
    }
    Ok(())
}

#[cfg(test)]
mod registry_tests {
    use super::*;

    #[test]
    fn baked_noise_profiles_present() {
        let names = list_vendor_noise_profiles();
        for required in ["ibm-falcon-emu", "rigetti-aspen-emu", "ionq-forte-emu",
                         "ibm-falcon", "rigetti-aspen", "ionq-forte"] {
            assert!(names.contains(&required.to_string()),
                    "baked profile {required:?} missing; got {names:?}");
        }
    }

    #[test]
    fn noise_profile_round_trip() {
        let p = VendorNoiseProfile {
            p_gate_1q: 0.0015, p_gate_2q: 0.012, p_readout: 0.018,
            description: "IBM Falcon (2026-05-20 snapshot)".to_string(),
        };
        register_vendor_noise_profile("rs-test-profile", &p).unwrap();
        let back = lookup_vendor_noise_profile("rs-test-profile").unwrap();
        assert!((back.p_gate_2q - 0.012).abs() < 1e-9);
        assert!(back.description.contains("snapshot"));
        unregister_vendor_noise_profile("rs-test-profile").unwrap();
        assert!(lookup_vendor_noise_profile("rs-test-profile").is_none());
    }

    #[test]
    fn completion_hook_fires_with_args() {
        use std::sync::atomic::{AtomicI32, Ordering};
        use std::sync::Arc;

        struct HookReset;
        impl Drop for HookReset {
            fn drop(&mut self) {
                let _ = clear_completion_hook();
            }
        }

        let _scheduler_guard = SCHEDULER_TEST_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _hook_reset = HookReset;
        let count = Arc::new(AtomicI32::new(0));
        let last_qubits = Arc::new(AtomicI32::new(-1));
        let last_shots = Arc::new(AtomicI32::new(-1));
        let count2 = count.clone();
        let last_q = last_qubits.clone();
        let last_s = last_shots.clone();
        set_completion_hook(move |info| {
            count2.fetch_add(1, Ordering::SeqCst);
            last_q.store(info.num_qubits, Ordering::SeqCst);
            last_s.store(info.total_shots, Ordering::SeqCst);
        }).unwrap();

        let mut j = Job::new(2).unwrap();
        j.add_gate(GateType::H, 0, -1, &[]).unwrap();
        j.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
        j.set_num_shots(128).unwrap();
        j.set_num_workers(1).unwrap();
        let _ = j.execute().unwrap();

        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert_eq!(last_qubits.load(Ordering::SeqCst), 2);
        assert_eq!(last_shots.load(Ordering::SeqCst), 128);

        clear_completion_hook().unwrap();
        let _ = j.execute().unwrap();
        // Count should NOT advance after clear.
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
}
