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
};
use std::ffi::c_void;
use std::ptr;

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
        let mut buf = vec![0i8; cap];
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
