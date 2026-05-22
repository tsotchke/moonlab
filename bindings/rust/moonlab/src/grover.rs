//! Grover's quantum search algorithm (since v0.2.0; safe Rust
//! wrapper since v0.4.7).
//!
//! Wraps `src/algorithms/grover.{c,h}` with idiomatic Rust around
//! the full Grover search entry point plus the optimal-iteration
//! helper.  Searches a marked basis state in `O(sqrt(N))` queries
//! where `N = 2^num_qubits`.
//!
//! Mirrors the Python `moonlab.algorithms.Grover` surface.
//!
//! ## Quick start
//!
//! ```no_run
//! use moonlab::{QuantumState, grover};
//!
//! // Search the marked state |0b1010> over a 4-qubit register.
//! let mut state = QuantumState::new(4).unwrap();
//! let result = grover::search(&mut state, /*marked=*/0b1010, None).unwrap();
//! assert!(result.success_probability > 0.95);
//! assert_eq!(result.found_state, 0b1010);
//! ```

use crate::error::{QuantumError, Result};
use crate::state::QuantumState;
use moonlab_sys::{
    grover_config_t, grover_optimal_iterations, grover_result_t, grover_search,
    quantum_entropy_ctx_create_hw, quantum_entropy_ctx_destroy,
    quantum_entropy_ctx_t,
};
use std::ptr;

/// RAII guard around a hardware-entropy context.  `grover_search`
/// refuses to run with `entropy == NULL`, so we lease one for the
/// duration of the call and free it on drop.
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

/// Snapshot of one Grover-search run.  Mirrors `grover_result_t`.
#[derive(Debug, Clone, Copy)]
pub struct GroverResult {
    /// Most-probable measurement outcome after the final Grover
    /// iteration.
    pub found_state: u64,
    /// Probability of measuring the marked state on a noiseless
    /// projective measurement.
    pub success_probability: f64,
    /// Number of oracle queries the algorithm performed.
    pub oracle_calls: usize,
    /// Number of Grover iterations the algorithm performed.
    pub iterations_performed: usize,
    /// `|<target|final>|^2`.
    pub fidelity: f64,
    /// `true` when the most-probable outcome equals the marked
    /// state.
    pub found_marked_state: bool,
}

impl GroverResult {
    fn from_c(r: grover_result_t) -> Self {
        Self {
            found_state: r.found_state,
            success_probability: r.success_probability,
            oracle_calls: r.oracle_calls,
            iterations_performed: r.iterations_performed,
            fidelity: r.fidelity,
            found_marked_state: r.found_marked_state != 0,
        }
    }
}

/// Run Grover's search on `state`, looking for `marked_state`.  When
/// `num_iterations` is `None` the optimal `floor(pi sqrt(N) / 4)`
/// iteration count is used.  The state is mutated to the
/// post-iteration superposition.
pub fn search(
    state: &mut QuantumState,
    marked_state: u64,
    num_iterations: Option<usize>,
) -> Result<GroverResult> {
    let num_qubits = state.num_qubits();
    let (iters, auto) = match num_iterations {
        Some(n) => (n, 0),
        None => (0, 1),
    };
    let config = grover_config_t {
        num_qubits,
        marked_state,
        num_iterations: iters,
        use_optimal_iterations: auto,
    };
    let entropy = EntropyGuard::new()?;
    let result = unsafe {
        grover_search(
            state.as_ptr(),
            &config,
            entropy.ctx,
        )
    };
    drop(entropy);
    Ok(GroverResult::from_c(result))
}

/// Return the optimal number of Grover iterations on a
/// `num_qubits`-qubit register: `floor(pi sqrt(N) / 4)` where
/// `N = 2^num_qubits`.
pub fn optimal_iterations(num_qubits: usize) -> usize {
    unsafe { grover_optimal_iterations(num_qubits) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimal_iterations_matches_textbook_for_n4() {
        // N = 16, optimal = floor(pi * sqrt(16) / 4) = floor(pi) = 3.
        assert_eq!(optimal_iterations(4), 3);
    }

    #[test]
    fn search_finds_marked_state_at_n4() {
        // 4-qubit Grover with optimal iterations succeeds with
        // ~96% probability on noiseless simulation -- the textbook
        // single-shot success probability is sin^2((2k+1) theta)
        // with k=3, theta=arcsin(1/4) = 0.9613... = 96.13%.  The
        // `found_marked_state` flag is a Born-rule sample of that
        // distribution, so it can come up wrong on the ~4% tail.
        //
        // Verify the algorithmic invariant (probability mass is
        // concentrated on |1010>) with high confidence; retry the
        // Born sample a few times if it lands on the unlucky branch.
        // 3 retries puts the false-fail rate at 0.04^3 = 0.0064%
        // per run, which is below the noise floor of CI flake.
        for attempt in 0..3 {
            let mut state = QuantumState::new(4).unwrap();
            let r = search(&mut state, 0b1010, None).unwrap();
            // Algorithmic invariant -- always true.
            assert!(
                r.success_probability > 0.9,
                "Grover P(success) = {:.3} on 4-qubit |1010>; expected > 0.9",
                r.success_probability
            );
            assert_eq!(r.iterations_performed, optimal_iterations(4));
            // Born-rule sample -- usually but not always lands on |1010>.
            if r.found_marked_state && r.found_state == 0b1010 {
                return;
            }
            eprintln!("search_finds_marked_state_at_n4: attempt {} \
                       landed off-target (found_state={:#06b}); retrying",
                       attempt, r.found_state);
        }
        panic!("Grover Born sample missed marked state 3x in a row \
                -- chance is 0.0064%, investigate");
    }

    #[test]
    fn search_honours_explicit_iteration_count() {
        let mut state = QuantumState::new(4).unwrap();
        let r = search(&mut state, 0b0101, Some(2)).unwrap();
        assert_eq!(r.iterations_performed, 2);
    }
}
