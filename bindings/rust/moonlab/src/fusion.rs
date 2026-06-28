//! Single-qubit gate-fusion DAG (since v0.2.0; safe Rust wrapper
//! since v0.4.6).
//!
//! Wraps `src/optimization/fusion/fusion.{c,h}`.  Builds a symbolic
//! circuit, runs the run-length single-qubit fuser, and executes
//! the fused schedule on a [`crate::QuantumState`].  Collapses
//! adjacent single-qubit gates on the same qubit into one 2x2
//! matrix, dropping repeated full-state passes that state-vector
//! simulation pays for each gate.  On a five-layer
//! hardware-efficient ansatz at `n = 16` the fused execution is
//! roughly 2.2x faster than the unfused dispatch.
//!
//! Mirrors the v0.4.4 Python `moonlab.fusion.FusedCircuit`.
//!
//! ## Quick start
//!
//! ```no_run
//! use moonlab::{QuantumState, fusion::FusedCircuit};
//!
//! let mut circuit = FusedCircuit::new(4).unwrap();
//! circuit
//!     .h(0).unwrap()
//!     .rz(0, 0.3).unwrap()
//!     .rx(0, 0.7).unwrap()
//!     .cnot(0, 1).unwrap()
//!     .rz(1, 0.4).unwrap();
//!
//! let (fused, stats) = circuit.compile().unwrap();
//! assert!(stats.merges_applied >= 1);
//!
//! let mut state = QuantumState::new(4).unwrap();
//! fused.execute(&mut state).unwrap();
//! ```

use crate::error::{QuantumError, Result};
use crate::state::QuantumState;
use moonlab_sys::{
    fuse_append_cnot, fuse_append_cphase, fuse_append_crx, fuse_append_cry,
    fuse_append_crz, fuse_append_cy, fuse_append_cz, fuse_append_h,
    fuse_append_phase, fuse_append_rx, fuse_append_ry, fuse_append_rz,
    fuse_append_s, fuse_append_sdg, fuse_append_swap, fuse_append_t,
    fuse_append_tdg, fuse_append_u3, fuse_append_x, fuse_append_y,
    fuse_append_z, fuse_circuit_create, fuse_circuit_free, fuse_circuit_len,
    fuse_circuit_num_qubits, fuse_circuit_t, fuse_compile, fuse_execute,
    fuse_stats_t,
};
use std::ptr;

/// Diagnostic counts from [`FusedCircuit::compile`].  Mirrors
/// `fuse_stats_t` in the C header.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct FuseStats {
    /// Symbol-list length of the input circuit.
    pub original_gates: usize,
    /// Symbol-list length of the fused output.
    pub fused_gates: usize,
    /// Number of pair-wise 2x2 multiplications the fuser performed;
    /// equals `(run_length - 1)` summed across every fused run.
    /// Zero when no single-qubit gate ran into a same-qubit
    /// successor before a multi-qubit barrier.
    pub merges_applied: usize,
}

/// Owned symbolic gate-fusion circuit.
pub struct FusedCircuit {
    handle: *mut fuse_circuit_t,
    num_qubits: usize,
}

// SAFETY: FusedCircuit owns its handle and is not shared across
// threads without external synchronization.
unsafe impl Send for FusedCircuit {}

impl FusedCircuit {
    /// Create an empty circuit on `num_qubits` qubits.
    pub fn new(num_qubits: usize) -> Result<Self> {
        if num_qubits == 0 {
            return Err(QuantumError::InvalidQubit { index: 0, max: 1 });
        }
        let handle = unsafe { fuse_circuit_create(num_qubits) };
        if handle.is_null() {
            return Err(QuantumError::AllocationFailed(num_qubits));
        }
        Ok(Self {
            handle,
            num_qubits,
        })
    }

    /// Number of qubits the circuit operates on.
    pub fn num_qubits(&self) -> usize {
        unsafe { fuse_circuit_num_qubits(self.handle) }
    }

    /// Number of symbolic gates currently in the circuit.
    pub fn len(&self) -> usize {
        unsafe { fuse_circuit_len(self.handle) }
    }

    /// True when no gates have been appended.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ---- Single-qubit non-parameterised gates -----------------------------
    pub fn h(&mut self, q: i32)   -> Result<&mut Self> { self.call1("fuse_append_h",   unsafe { fuse_append_h  (self.handle, q) }) }
    pub fn x(&mut self, q: i32)   -> Result<&mut Self> { self.call1("fuse_append_x",   unsafe { fuse_append_x  (self.handle, q) }) }
    pub fn y(&mut self, q: i32)   -> Result<&mut Self> { self.call1("fuse_append_y",   unsafe { fuse_append_y  (self.handle, q) }) }
    pub fn z(&mut self, q: i32)   -> Result<&mut Self> { self.call1("fuse_append_z",   unsafe { fuse_append_z  (self.handle, q) }) }
    pub fn s(&mut self, q: i32)   -> Result<&mut Self> { self.call1("fuse_append_s",   unsafe { fuse_append_s  (self.handle, q) }) }
    pub fn sdg(&mut self, q: i32) -> Result<&mut Self> { self.call1("fuse_append_sdg", unsafe { fuse_append_sdg(self.handle, q) }) }
    pub fn t(&mut self, q: i32)   -> Result<&mut Self> { self.call1("fuse_append_t",   unsafe { fuse_append_t  (self.handle, q) }) }
    pub fn tdg(&mut self, q: i32) -> Result<&mut Self> { self.call1("fuse_append_tdg", unsafe { fuse_append_tdg(self.handle, q) }) }

    // ---- Single-qubit one-parameter gates ---------------------------------
    pub fn phase(&mut self, q: i32, theta: f64) -> Result<&mut Self> {
        self.call1("fuse_append_phase", unsafe { fuse_append_phase(self.handle, q, theta) })
    }
    pub fn rx(&mut self, q: i32, theta: f64) -> Result<&mut Self> {
        self.call1("fuse_append_rx", unsafe { fuse_append_rx(self.handle, q, theta) })
    }
    pub fn ry(&mut self, q: i32, theta: f64) -> Result<&mut Self> {
        self.call1("fuse_append_ry", unsafe { fuse_append_ry(self.handle, q, theta) })
    }
    pub fn rz(&mut self, q: i32, theta: f64) -> Result<&mut Self> {
        self.call1("fuse_append_rz", unsafe { fuse_append_rz(self.handle, q, theta) })
    }

    /// Universal single-qubit U3 gate.
    pub fn u3(&mut self, q: i32, theta: f64, phi: f64, lambda: f64)
        -> Result<&mut Self>
    {
        let rc = unsafe { fuse_append_u3(self.handle, q, theta, phi, lambda) };
        self.call1("fuse_append_u3", rc)
    }

    // ---- Two-qubit non-parameterised gates --------------------------------
    pub fn cnot(&mut self, ctrl: i32, tgt: i32) -> Result<&mut Self> {
        self.call1("fuse_append_cnot", unsafe { fuse_append_cnot(self.handle, ctrl, tgt) })
    }
    pub fn cz(&mut self, ctrl: i32, tgt: i32) -> Result<&mut Self> {
        self.call1("fuse_append_cz", unsafe { fuse_append_cz(self.handle, ctrl, tgt) })
    }
    pub fn cy(&mut self, ctrl: i32, tgt: i32) -> Result<&mut Self> {
        self.call1("fuse_append_cy", unsafe { fuse_append_cy(self.handle, ctrl, tgt) })
    }
    pub fn swap(&mut self, a: i32, b: i32) -> Result<&mut Self> {
        self.call1("fuse_append_swap", unsafe { fuse_append_swap(self.handle, a, b) })
    }

    // ---- Two-qubit one-parameter gates ------------------------------------
    pub fn cphase(&mut self, ctrl: i32, tgt: i32, theta: f64) -> Result<&mut Self> {
        self.call1("fuse_append_cphase", unsafe { fuse_append_cphase(self.handle, ctrl, tgt, theta) })
    }
    pub fn crx(&mut self, ctrl: i32, tgt: i32, theta: f64) -> Result<&mut Self> {
        self.call1("fuse_append_crx", unsafe { fuse_append_crx(self.handle, ctrl, tgt, theta) })
    }
    pub fn cry(&mut self, ctrl: i32, tgt: i32, theta: f64) -> Result<&mut Self> {
        self.call1("fuse_append_cry", unsafe { fuse_append_cry(self.handle, ctrl, tgt, theta) })
    }
    pub fn crz(&mut self, ctrl: i32, tgt: i32, theta: f64) -> Result<&mut Self> {
        self.call1("fuse_append_crz", unsafe { fuse_append_crz(self.handle, ctrl, tgt, theta) })
    }

    /// Run the single-qubit fuser.  Returns the new (fused) circuit
    /// together with a [`FuseStats`] summary.
    pub fn compile(&self) -> Result<(FusedCircuit, FuseStats)> {
        let mut stats = fuse_stats_t {
            original_gates: 0,
            fused_gates: 0,
            merges_applied: 0,
        };
        let h = unsafe { fuse_compile(self.handle, &mut stats) };
        if h.is_null() {
            return Err(QuantumError::AllocationFailed(0));
        }
        let fused = FusedCircuit {
            handle: h,
            num_qubits: self.num_qubits,
        };
        Ok((
            fused,
            FuseStats {
                original_gates: stats.original_gates,
                fused_gates: stats.fused_gates,
                merges_applied: stats.merges_applied,
            },
        ))
    }

    /// Apply the (fused or unfused) circuit to `state` in place.
    pub fn execute(&self, state: &mut QuantumState) -> Result<()> {
        let rc = unsafe { fuse_execute(self.handle, state.as_ptr()) };
        if rc != 0 {
            Err(QuantumError::Ffi(format!("fuse_execute rc={rc}")))
        } else {
            Ok(())
        }
    }

    // ---- Internal helpers --------------------------------------------------

    fn call1(&mut self, name: &'static str, rc: i32) -> Result<&mut Self> {
        if rc != 0 {
            Err(QuantumError::Ffi(format!("{name} rc={rc}")))
        } else {
            Ok(self)
        }
    }
}

impl Drop for FusedCircuit {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { fuse_circuit_free(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_len() {
        let c = FusedCircuit::new(4).unwrap();
        assert_eq!(c.num_qubits(), 4);
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
    }

    #[test]
    fn reject_zero_qubits() {
        assert!(FusedCircuit::new(0).is_err());
    }

    #[test]
    fn fluent_append_increments_len() {
        let mut c = FusedCircuit::new(3).unwrap();
        c.h(0).unwrap()
            .rz(0, 0.3).unwrap()
            .rx(0, 0.4).unwrap()
            .cnot(0, 1).unwrap()
            .ry(1, 0.5).unwrap();
        assert_eq!(c.len(), 5);
    }

    #[test]
    fn compile_run_fuses_three_into_one() {
        // h, rz, rx all on qubit 0 -> 1 FUSED_1Q + 2 merges.
        let mut c = FusedCircuit::new(2).unwrap();
        c.h(0).unwrap().rz(0, 0.3).unwrap().rx(0, 0.4).unwrap().cnot(0, 1).unwrap();
        let (fused, stats) = c.compile().unwrap();
        assert_eq!(stats.original_gates, 4);
        assert_eq!(stats.fused_gates, 2);
        assert_eq!(stats.merges_applied, 2);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn compile_passthrough_when_no_runs() {
        let mut c = FusedCircuit::new(3).unwrap();
        c.h(0).unwrap().cnot(0, 1).unwrap().h(1).unwrap().cnot(1, 2).unwrap().h(2).unwrap();
        let (_fused, stats) = c.compile().unwrap();
        assert_eq!(stats.original_gates, 5);
        assert_eq!(stats.merges_applied, 0);
        assert_eq!(stats.fused_gates, 5);
    }

    #[test]
    fn execute_bell_state() {
        let mut c = FusedCircuit::new(2).unwrap();
        c.h(0).unwrap().cnot(0, 1).unwrap();
        let mut state = QuantumState::new(2).unwrap();
        c.execute(&mut state).unwrap();
        let probs = state.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!(probs[1].abs() < 1e-10);
        assert!(probs[2].abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn execute_fused_matches_unfused() {
        let build = || -> FusedCircuit {
            let mut c = FusedCircuit::new(3).unwrap();
            c.h(0).unwrap()
                .rz(0, 0.7).unwrap()
                .rx(0, 0.3).unwrap()
                .cnot(0, 1).unwrap()
                .ry(1, 0.2).unwrap()
                .rz(1, 0.9).unwrap()
                .cnot(1, 2).unwrap()
                .rx(2, 0.4).unwrap();
            c
        };

        let mut s_unfused = QuantumState::new(3).unwrap();
        build().execute(&mut s_unfused).unwrap();

        let mut s_fused = QuantumState::new(3).unwrap();
        let (fused, _) = build().compile().unwrap();
        fused.execute(&mut s_fused).unwrap();

        let pu = s_unfused.probabilities();
        let pf = s_fused.probabilities();
        for (u, f) in pu.iter().zip(pf.iter()) {
            assert!((u - f).abs() < 1e-10, "{u} != {f}");
        }
    }
}
