//! QGTL-shaped circuit-ingestion Rust binding -- since v0.6.8.
//!
//! Safe wrapper around `src/applications/moonlab_qgtl_backend.{c,h}`.
//! Mirrors the Python (`moonlab.qgtl`) surface.
//!
//! ## Quick start
//!
//! ```no_run
//! use moonlab::qgtl::{QgtlCircuit, GateType};
//!
//! let mut c = QgtlCircuit::new(2).unwrap();
//! c.add_gate(GateType::H, 0, -1, &[]).unwrap();
//! c.add_gate(GateType::CNOT, 1, 0, &[]).unwrap();
//! let r = c.execute(0, 0, true).unwrap();
//! let p = r.probabilities.unwrap();
//! assert!((p[0] - 0.5).abs() < 1e-9);
//! assert!((p[3] - 0.5).abs() < 1e-9);
//! ```

use crate::error::{QuantumError, Result};
use moonlab_sys::{
    moonlab_qgtl_add_gate, moonlab_qgtl_circuit_create,
    moonlab_qgtl_circuit_deserialize, moonlab_qgtl_circuit_free,
    moonlab_qgtl_circuit_load, moonlab_qgtl_circuit_num_gates,
    moonlab_qgtl_circuit_num_qubits, moonlab_qgtl_circuit_save,
    moonlab_qgtl_circuit_serialize, moonlab_qgtl_exec_options_t,
    moonlab_qgtl_execute, moonlab_qgtl_results_free, moonlab_qgtl_results_t,
};
use std::ffi::CString;
use std::ptr;

/// Gate-type tag matching `moonlab_qgtl_gate_t` numerically.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GateType {
    I    = 0,
    X    = 1,
    Y    = 2,
    Z    = 3,
    H    = 4,
    S    = 5,
    T    = 6,
    Rx   = 7,
    Ry   = 8,
    Rz   = 9,
    Cnot = 10,
    Cy   = 11,
    Cz   = 12,
    Swap = 13,
}

/// Outputs from `QgtlCircuit::execute`.
#[derive(Debug, Default)]
pub struct QgtlResults {
    pub num_qubits: i32,
    pub num_shots: i32,
    pub outcomes: Option<Vec<u64>>,
    pub probabilities: Option<Vec<f64>>,
}

/// Owned handle to a `moonlab_qgtl_circuit_t`.
pub struct QgtlCircuit {
    ptr: *mut std::ffi::c_void,
}

impl QgtlCircuit {
    /// Allocate a circuit on `num_qubits` qubits.  Returns
    /// `Err(QuantumError::Ffi)` when `num_qubits` is out of range
    /// (current cap: 32 native).
    pub fn new(num_qubits: i32) -> Result<Self> {
        let p = unsafe { moonlab_qgtl_circuit_create(num_qubits) };
        if p.is_null() {
            return Err(QuantumError::Ffi(format!(
                "moonlab_qgtl_circuit_create({num_qubits}): NULL"
            )));
        }
        Ok(Self { ptr: p as *mut std::ffi::c_void })
    }

    /// Number of qubits the circuit was created with.
    pub fn num_qubits(&self) -> i32 {
        unsafe { moonlab_qgtl_circuit_num_qubits(self.ptr as *const _) }
    }

    /// Number of gates appended.
    pub fn num_gates(&self) -> i32 {
        unsafe { moonlab_qgtl_circuit_num_gates(self.ptr as *const _) }
    }

    /// Append a gate.  `control = -1` for single-qubit gates;
    /// `params = &[]` when the gate type does not read parameters.
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
            moonlab_qgtl_add_gate(
                self.ptr as *mut _,
                gate as u32,
                target,
                control,
                params_ptr,
            )
        };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!(
                "add_gate({:?}, target={target}, control={control}): rc={rc}",
                gate
            )));
        }
        Ok(())
    }

    /// Run the circuit through moonlab's state-vector backend.
    pub fn execute(&mut self,
                   num_shots: i32,
                   rng_seed: u64,
                   return_probabilities: bool) -> Result<QgtlResults> {
        let opts = moonlab_qgtl_exec_options_t {
            num_shots,
            rng_seed,
            return_probabilities: if return_probabilities { 1 } else { 0 },
        };
        let mut res: moonlab_qgtl_results_t = unsafe { std::mem::zeroed() };
        let rc = unsafe {
            moonlab_qgtl_execute(self.ptr as *mut _, &opts, &mut res)
        };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!("execute: rc={rc}")));
        }

        let outcomes = if res.num_shots > 0 && !res.outcomes.is_null() {
            let n = res.num_shots as usize;
            let slice = unsafe { std::slice::from_raw_parts(res.outcomes, n) };
            Some(slice.to_vec())
        } else {
            None
        };
        let probabilities = if return_probabilities && !res.probabilities.is_null() {
            let dim = 1usize << res.num_qubits;
            let slice = unsafe { std::slice::from_raw_parts(res.probabilities, dim) };
            Some(slice.to_vec())
        } else {
            None
        };

        let out = QgtlResults {
            num_qubits: res.num_qubits,
            num_shots: res.num_shots,
            outcomes,
            probabilities,
        };
        unsafe { moonlab_qgtl_results_free(&mut res) };
        Ok(out)
    }
}

impl QgtlCircuit {
    /// Serialize the circuit to a portable text string
    /// (moonlab-circuit v1 format, since v0.8.3 C surface).
    pub fn serialize(&self) -> Result<String> {
        let mut needed: usize = 0;
        let rc = unsafe {
            moonlab_qgtl_circuit_serialize(
                self.ptr as *const _,
                ptr::null_mut(),
                0,
                &mut needed,
            )
        };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!(
                "serialize size-query: rc={rc}"
            )));
        }
        let mut buf: Vec<u8> = vec![0u8; needed + 1];
        let rc = unsafe {
            moonlab_qgtl_circuit_serialize(
                self.ptr as *const _,
                buf.as_mut_ptr() as *mut i8,
                buf.len(),
                ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!("serialize: rc={rc}")));
        }
        if let Some(nul_at) = buf.iter().position(|&b| b == 0) {
            buf.truncate(nul_at);
        }
        String::from_utf8(buf).map_err(|e| {
            QuantumError::Ffi(format!("serialize: non-utf8 output: {e}"))
        })
    }

    /// Construct a circuit from a moonlab-circuit v1 text string.
    pub fn deserialize(text: &str) -> Result<Self> {
        let bytes = text.as_bytes();
        let mut status: i32 = 0;
        let p = unsafe {
            moonlab_qgtl_circuit_deserialize(
                bytes.as_ptr() as *const i8,
                bytes.len(),
                &mut status,
            )
        };
        if p.is_null() {
            return Err(QuantumError::Ffi(format!(
                "deserialize: status={status}"
            )));
        }
        Ok(Self { ptr: p as *mut std::ffi::c_void })
    }

    /// Save the circuit to a file in the portable text format.
    pub fn save(&self, path: &str) -> Result<()> {
        let cpath = CString::new(path).map_err(|e| {
            QuantumError::Ffi(format!("save: invalid path: {e}"))
        })?;
        let rc = unsafe {
            moonlab_qgtl_circuit_save(self.ptr as *const _, cpath.as_ptr())
        };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!("save({path}): rc={rc}")));
        }
        Ok(())
    }

    /// Load a circuit from a file previously written by `save`.
    pub fn load(path: &str) -> Result<Self> {
        let cpath = CString::new(path).map_err(|e| {
            QuantumError::Ffi(format!("load: invalid path: {e}"))
        })?;
        let mut status: i32 = 0;
        let p = unsafe {
            moonlab_qgtl_circuit_load(cpath.as_ptr(), &mut status)
        };
        if p.is_null() {
            return Err(QuantumError::Ffi(format!(
                "load({path}): status={status}"
            )));
        }
        Ok(Self { ptr: p as *mut std::ffi::c_void })
    }
}

impl Drop for QgtlCircuit {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { moonlab_qgtl_circuit_free(self.ptr as *mut _) };
            self.ptr = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bell_pair() {
        let mut c = QgtlCircuit::new(2).unwrap();
        c.add_gate(GateType::H, 0, -1, &[]).unwrap();
        c.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
        assert_eq!(c.num_gates(), 2);
        let r = c.execute(0, 0, true).unwrap();
        let p = r.probabilities.unwrap();
        assert!((p[0] - 0.5).abs() < 1e-9);
        assert!((p[3] - 0.5).abs() < 1e-9);
        assert!(p[1] < 1e-9 && p[2] < 1e-9);
    }

    #[test]
    fn ghz_3() {
        let mut c = QgtlCircuit::new(3).unwrap();
        c.add_gate(GateType::H, 0, -1, &[]).unwrap();
        c.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
        c.add_gate(GateType::Cnot, 2, 1, &[]).unwrap();
        let r = c.execute(0, 0, true).unwrap();
        let p = r.probabilities.unwrap();
        assert!((p[0] - 0.5).abs() < 1e-9);
        assert!((p[7] - 0.5).abs() < 1e-9);
        for b in 1..7 { assert!(p[b] < 1e-9); }
    }

    #[test]
    fn shot_sampling() {
        let mut c = QgtlCircuit::new(2).unwrap();
        c.add_gate(GateType::H, 0, -1, &[]).unwrap();
        c.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
        let r = c.execute(1024, 0xdeadbeef, false).unwrap();
        let outcomes = r.outcomes.unwrap();
        assert_eq!(outcomes.len(), 1024);
        assert!(outcomes.iter().all(|&o| o == 0 || o == 3));
        let n00 = outcomes.iter().filter(|&&o| o == 0).count() as i32;
        assert!((n00 - 512).abs() < 80);
    }

    #[test]
    fn ry_half_pi() {
        let mut c = QgtlCircuit::new(1).unwrap();
        c.add_gate(GateType::Ry, 0, -1, &[std::f64::consts::FRAC_PI_2]).unwrap();
        let r = c.execute(0, 0, true).unwrap();
        let p = r.probabilities.unwrap();
        assert!((p[0] - 0.5).abs() < 1e-9);
        assert!((p[1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn rejects_zero_qubits() {
        assert!(QgtlCircuit::new(0).is_err());
    }

    #[test]
    fn rejects_cnot_self() {
        let mut c = QgtlCircuit::new(2).unwrap();
        assert!(c.add_gate(GateType::Cnot, 0, 0, &[]).is_err());
    }

    #[test]
    fn gate_type_enum_numbers() {
        assert_eq!(GateType::I as u32, 0);
        assert_eq!(GateType::H as u32, 4);
        assert_eq!(GateType::Cnot as u32, 10);
        assert_eq!(GateType::Swap as u32, 13);
    }

    // ---- v0.8.5 circuit serialization ----

    #[test]
    fn serialize_roundtrip_byte_exact() {
        let mut c = QgtlCircuit::new(4).unwrap();
        c.add_gate(GateType::H, 0, -1, &[]).unwrap();
        c.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
        c.add_gate(GateType::Rz, 2, -1, &[std::f64::consts::FRAC_PI_3]).unwrap();
        c.add_gate(GateType::Swap, 3, 2, &[]).unwrap();

        let text = c.serialize().unwrap();
        assert!(text.contains("# moonlab-circuit v1"));
        assert!(text.contains("NUM_QUBITS 4"));
        assert!(text.contains("CNOT 1 0"));

        let c2 = QgtlCircuit::deserialize(&text).unwrap();
        assert_eq!(c2.num_qubits(), 4);
        assert_eq!(c2.num_gates(), 4);
        assert_eq!(c2.serialize().unwrap(), text);
    }

    #[test]
    fn save_load_roundtrip() {
        let mut c = QgtlCircuit::new(3).unwrap();
        c.add_gate(GateType::H, 0, -1, &[]).unwrap();
        c.add_gate(GateType::Cnot, 1, 0, &[]).unwrap();
        c.add_gate(GateType::Cnot, 2, 1, &[]).unwrap();

        let path = std::env::temp_dir().join("moonlab_qgtl_rs_test.qcir");
        let path_str = path.to_str().unwrap();

        c.save(path_str).unwrap();
        let mut c2 = QgtlCircuit::load(path_str).unwrap();
        assert_eq!(c2.num_qubits(), 3);
        assert_eq!(c2.num_gates(), 3);

        // Verify it actually runs.
        let r = c2.execute(0, 0, true).unwrap();
        let p = r.probabilities.unwrap();
        assert!((p[0] - 0.5).abs() < 1e-9);
        assert!((p[7] - 0.5).abs() < 1e-9);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn deserialize_rejects_garbage() {
        assert!(QgtlCircuit::deserialize("garbage\n").is_err());
        assert!(QgtlCircuit::deserialize("NUM_QUBITS 99\nH 0\n").is_err());
    }
}
