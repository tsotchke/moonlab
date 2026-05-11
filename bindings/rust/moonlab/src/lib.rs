//! # Moonlab - Rust Quantum Simulator
//!
//! Safe, idiomatic Rust bindings for the Moonlab quantum computing simulator.
//!
//! ## Quick Start
//!
//! ```no_run
//! use moonlab::QuantumState;
//!
//! // Create a 3-qubit quantum state
//! let mut state = QuantumState::new(3).unwrap();
//!
//! // Create GHZ state with method chaining
//! state.h(0)           // Hadamard on qubit 0
//!      .cnot(0, 1)     // CNOT: control=0, target=1
//!      .cnot(0, 2);    // CNOT: control=0, target=2
//!
//! // Get measurement probabilities
//! let probs = state.probabilities();
//! println!("|000⟩: {:.3}", probs[0]); // ≈ 0.5
//! println!("|111⟩: {:.3}", probs[7]); // ≈ 0.5
//!
//! // Compute entanglement
//! let entropy = state.entanglement_entropy(&[0]).unwrap();
//! println!("Entanglement entropy: {:.3} bits", entropy);
//! ```
//!
//! ## Features
//!
//! - **Full gate set**: Pauli, Hadamard, rotation gates, CNOT, Toffoli, QFT, and more
//! - **Safe API**: RAII memory management, Rust error handling
//! - **High performance**: Powered by optimized C backend with Metal GPU acceleration
//! - **Method chaining**: Fluent API for building quantum circuits
//!
//! ## Architecture
//!
//! This crate provides safe wrappers around the `moonlab-sys` FFI bindings.
//! All memory is automatically managed through Rust's ownership system.

pub mod ca_mps;
pub mod error;
pub mod feynman;
pub mod state;
pub mod topology;

// Re-export main types
pub use ca_mps::{
    gauge_warmstart, status_string, var_d_run, z2_lgt_1d_build,
    z2_lgt_1d_gauss_law, CaMps, StatusModule, VarDConfig, Warmstart,
};
pub use error::{QuantumError, Result};
pub use feynman::{FeynmanDiagram, ParticleType};
pub use state::{QuantumState, MAX_QUBITS};
pub use topology::{qwz_chern, ssh_winding, ChernKpm};

/// Prelude module for convenient imports.
///
/// ```
/// use moonlab::prelude::*;
/// ```
pub mod prelude {
    pub use crate::ca_mps::{CaMps, VarDConfig, Warmstart};
    pub use crate::error::{QuantumError, Result};
    pub use crate::feynman::{FeynmanDiagram, ParticleType};
    pub use crate::state::{QuantumState, MAX_QUBITS};
    pub use crate::topology::{qwz_chern, ssh_winding, ChernKpm};
}
