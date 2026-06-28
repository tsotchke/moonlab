//! Quantum Approximate Optimization Algorithm (since v0.2.0; safe
//! Rust wrapper since v0.4.9).
//!
//! Wraps `src/algorithms/qaoa.{c,h}` with idiomatic Rust around the
//! Ising-model encoding, the MaxCut graph helper, and the `qaoa_solve`
//! driver.  Mirrors the v0.2.0 Python `moonlab.algorithms.QAOA`
//! surface.
//!
//! [`QaoaSolver`] leases a hardware-entropy context for shot-noise
//! sampling via the same RAII pattern as [`crate::bell`] /
//! [`crate::grover`] / [`crate::vqe`].
//!
//! ## Quick start
//!
//! ```no_run
//! use moonlab::qaoa::{Graph, IsingModel, QaoaSolver};
//!
//! // Triangle graph -- MaxCut value = 2 (cut 2 of 3 edges).
//! let g = Graph::new(3, &[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]).unwrap();
//! let ising = IsingModel::from_maxcut(&g).unwrap();
//! let mut solver = QaoaSolver::new(ising, /*p=*/1).unwrap();
//! let result = solver.solve().unwrap();
//! println!("best cut value = {:.3}", -result.best_energy);
//! ```

use crate::error::{QuantumError, Result};
use moonlab_sys::{
    graph_add_edge, graph_create, graph_free, graph_t,
    ising_encode_maxcut, ising_model_create, ising_model_evaluate,
    ising_model_free, ising_model_set_coupling, ising_model_set_field,
    ising_model_t, qaoa_compute_expectation, qaoa_result_t, qaoa_solve,
    qaoa_solver_create, qaoa_solver_free, qaoa_solver_t,
    quantum_entropy_ctx_create_hw, quantum_entropy_ctx_destroy,
    quantum_entropy_ctx_t,
};
use std::ptr;

struct EntropyGuard {
    ctx: *mut quantum_entropy_ctx_t,
}

impl EntropyGuard {
    fn new() -> Result<Self> {
        let ctx = unsafe { quantum_entropy_ctx_create_hw() };
        if ctx.is_null() {
            return Err(QuantumError::Ffi(
                "quantum_entropy_ctx_create_hw returned NULL".to_string(),
            ));
        }
        Ok(Self { ctx })
    }
}

impl Drop for EntropyGuard {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { quantum_entropy_ctx_destroy(self.ctx) };
            self.ctx = ptr::null_mut();
        }
    }
}

/// Weighted undirected graph used to encode MaxCut.
pub struct Graph {
    ptr: *mut graph_t,
}

impl Graph {
    /// Build a graph on `num_vertices` vertices with the given
    /// weighted edges `(u, v, weight)`.
    pub fn new(num_vertices: usize, edges: &[(usize, usize, f64)]) -> Result<Self> {
        let ptr = unsafe { graph_create(num_vertices, edges.len()) };
        if ptr.is_null() {
            return Err(QuantumError::Ffi(
                "graph_create returned NULL".to_string(),
            ));
        }
        for (idx, &(u, v, w)) in edges.iter().enumerate() {
            let rc = unsafe { graph_add_edge(ptr, idx, u as i32, v as i32, w) };
            if rc != 0 {
                unsafe { graph_free(ptr) };
                return Err(QuantumError::Ffi(format!(
                    "graph_add_edge(idx={idx}, u={u}, v={v}) rc={rc}"
                )));
            }
        }
        Ok(Self { ptr })
    }

    pub(crate) fn as_ptr(&self) -> *const graph_t {
        self.ptr
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { graph_free(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

/// Ising-model Hamiltonian `H = sum_{i<j} J_{ij} s_i s_j + sum_i h_i s_i`
/// over `num_qubits` spins.
pub struct IsingModel {
    ptr: *mut ising_model_t,
}

impl IsingModel {
    /// Empty Ising model on `num_qubits` spins.  Add couplings and
    /// fields via [`set_coupling`](Self::set_coupling) /
    /// [`set_field`](Self::set_field).
    pub fn new(num_qubits: usize) -> Result<Self> {
        let ptr = unsafe { ising_model_create(num_qubits) };
        if ptr.is_null() {
            return Err(QuantumError::Ffi(
                "ising_model_create returned NULL".to_string(),
            ));
        }
        Ok(Self { ptr })
    }

    /// Encode a MaxCut problem on `graph` as the equivalent Ising
    /// model: `H = sum_{(i,j) in E} w_{ij} (1 - s_i s_j) / 2`.
    pub fn from_maxcut(graph: &Graph) -> Result<Self> {
        let ptr = unsafe { ising_encode_maxcut(graph.as_ptr()) };
        if ptr.is_null() {
            return Err(QuantumError::Ffi(
                "ising_encode_maxcut returned NULL".to_string(),
            ));
        }
        Ok(Self { ptr })
    }

    /// Set the symmetric coupling `J_{ij}`.
    pub fn set_coupling(&mut self, i: usize, j: usize, value: f64) -> Result<()> {
        let rc = unsafe { ising_model_set_coupling(self.ptr, i, j, value) };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!(
                "ising_model_set_coupling({i},{j}) rc={rc}"
            )));
        }
        Ok(())
    }

    /// Set the on-site field `h_i`.
    pub fn set_field(&mut self, i: usize, value: f64) -> Result<()> {
        let rc = unsafe { ising_model_set_field(self.ptr, i, value) };
        if rc != 0 {
            return Err(QuantumError::Ffi(format!(
                "ising_model_set_field({i}) rc={rc}"
            )));
        }
        Ok(())
    }

    /// Evaluate the classical Ising energy on the bitstring
    /// `bits[i] = (bitstring >> i) & 1`, where `s_i = 1 - 2 bits[i]`.
    pub fn evaluate(&self, bitstring: u64) -> f64 {
        unsafe { ising_model_evaluate(self.ptr, bitstring) }
    }
}

impl Drop for IsingModel {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ising_model_free(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

/// QAOA optimisation result.  Mirrors `qaoa_result_t` minus the raw
/// `energy_history` / `optimal_gamma` / `optimal_beta` pointers;
/// those are copied into owned `Vec<f64>`.
#[derive(Debug, Clone)]
pub struct QaoaResult {
    /// Bitstring of the best-energy sample seen during optimisation.
    pub best_bitstring: u64,
    /// Energy at `best_bitstring`.
    pub best_energy: f64,
    /// Number of optimiser iterations consumed.
    pub num_iterations: usize,
    /// `true` if the optimiser hit the convergence tolerance.
    pub converged: bool,
    /// Solution quality vs the global optimum (when known).
    pub approximation_ratio: f64,
    /// Optimal cost-Hamiltonian angles `gamma_1, ..., gamma_p`.
    pub optimal_gamma: Vec<f64>,
    /// Optimal mixer-Hamiltonian angles `beta_1, ..., beta_p`.
    pub optimal_beta: Vec<f64>,
    /// QAOA depth `p`.
    pub num_layers: usize,
    /// Total shot count consumed across the run.
    pub total_measurements: usize,
}

/// QAOA solver bundling `(ising, num_layers, entropy)`.
pub struct QaoaSolver {
    solver: *mut qaoa_solver_t,
    ising: IsingModel,
    _entropy: EntropyGuard,
}

impl QaoaSolver {
    /// Build a QAOA solver at depth `num_layers` over `ising`.
    /// `num_layers >= 1`.
    pub fn new(ising: IsingModel, num_layers: usize) -> Result<Self> {
        if num_layers == 0 {
            return Err(QuantumError::Ffi(
                "num_layers must be >= 1".to_string(),
            ));
        }
        let entropy = EntropyGuard::new()?;
        let solver = unsafe {
            qaoa_solver_create(ising.ptr, num_layers, entropy.ctx)
        };
        if solver.is_null() {
            return Err(QuantumError::Ffi(
                "qaoa_solver_create returned NULL".to_string(),
            ));
        }
        Ok(Self {
            solver,
            ising,
            _entropy: entropy,
        })
    }

    /// Number of spins in the underlying Ising problem.
    pub fn num_qubits(&self) -> usize {
        // qaoa_solver_t doesn't directly expose num_qubits.
        // Read it back from the ising model.
        unsafe { (*self.ising.ptr).num_qubits }
    }

    /// Run the QAOA optimisation loop to convergence.
    pub fn solve(&mut self) -> Result<QaoaResult> {
        let r: qaoa_result_t = unsafe { qaoa_solve(self.solver) };
        let n = r.num_layers;
        let gamma = if r.optimal_gamma.is_null() || n == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(r.optimal_gamma, n).to_vec() }
        };
        let beta = if r.optimal_beta.is_null() || n == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(r.optimal_beta, n).to_vec() }
        };
        Ok(QaoaResult {
            best_bitstring: r.best_bitstring,
            best_energy: r.best_energy,
            num_iterations: r.num_iterations,
            converged: r.converged != 0,
            approximation_ratio: r.approximation_ratio,
            optimal_gamma: gamma,
            optimal_beta: beta,
            num_layers: n,
            total_measurements: r.total_measurements,
        })
    }

    /// Evaluate `<H_C>(gamma, beta)` at the given angles without
    /// running the optimiser.  Useful for plotting the QAOA energy
    /// landscape.  Lengths of `gamma` and `beta` must equal the
    /// solver's `num_layers`.
    pub fn compute_expectation(&self, gamma: &[f64], beta: &[f64]) -> f64 {
        unsafe {
            qaoa_compute_expectation(
                self.solver,
                gamma.as_ptr(),
                beta.as_ptr(),
            )
        }
    }
}

impl Drop for QaoaSolver {
    fn drop(&mut self) {
        if !self.solver.is_null() {
            unsafe { qaoa_solver_free(self.solver) };
            self.solver = ptr::null_mut();
        }
        // ising + _entropy drop in field order.
    }
}

// QaoaSolver currently isn't Send/Sync because the C-side solver
// holds raw pointers into the ising model; same restriction as the
// other algorithm wrappers in this crate.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_rejects_invalid_edge_index() {
        let bad = Graph::new(3, &[(0, 5, 1.0)]); // vertex 5 out of range
        assert!(bad.is_err());
    }

    #[test]
    fn ising_evaluate_returns_finite_energy_on_triangle() {
        let g = Graph::new(3, &[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)]).unwrap();
        let ising = IsingModel::from_maxcut(&g).unwrap();
        let e_000 = ising.evaluate(0b000);
        let e_101 = ising.evaluate(0b101);
        // Triangle MaxCut: all-aligned cuts 0 edges (E maximal),
        // one-flipped cuts 2 edges (E minimal).
        assert!(
            e_000 > e_101,
            "all-zeros (no cut, E={e_000}) should have higher energy than one-flipped (cut, E={e_101})"
        );
        assert!(e_101.is_finite());
    }

    #[test]
    fn qaoa_solver_runs_on_triangle_maxcut() {
        let g = Graph::new(
            3,
            &[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)],
        )
        .unwrap();
        let ising = IsingModel::from_maxcut(&g).unwrap();
        let mut solver = QaoaSolver::new(ising, /*p=*/1).unwrap();
        let result = solver.solve().unwrap();
        assert_eq!(result.num_layers, 1);
        assert_eq!(result.optimal_gamma.len(), 1);
        assert_eq!(result.optimal_beta.len(), 1);
        assert!(result.best_energy.is_finite());
    }

    #[test]
    fn compute_expectation_returns_finite() {
        let g = Graph::new(2, &[(0, 1, 1.0)]).unwrap();
        let ising = IsingModel::from_maxcut(&g).unwrap();
        let solver = QaoaSolver::new(ising, /*p=*/1).unwrap();
        let e = solver.compute_expectation(&[0.3], &[0.5]);
        assert!(e.is_finite());
    }
}
