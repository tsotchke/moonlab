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

pub mod error;
pub mod feynman;
pub mod state;

// Re-export main types
pub use error::{QuantumError, Result};
pub use feynman::{FeynmanDiagram, ParticleType};
pub use state::{QuantumState, MAX_QUBITS};

/// Prelude module for convenient imports.
///
/// ```
/// use moonlab::prelude::*;
/// ```
pub mod prelude {
    pub use crate::error::{QuantumError, Result};
    pub use crate::feynman::{FeynmanDiagram, ParticleType};
    pub use crate::state::{QuantumState, MAX_QUBITS};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bell_inequality() {
        // Create Bell state and verify entanglement
        let mut state = QuantumState::new(2).unwrap();
        state.h(0).cnot(0, 1);

        // Bell state should have perfect ZZ correlation
        let zz = state.correlation_zz(0, 1).unwrap();
        assert!((zz - 1.0).abs() < 1e-10, "ZZ correlation should be 1.0");

        // Individual Z expectations should be 0
        let z0 = state.expectation_z(0).unwrap();
        let z1 = state.expectation_z(1).unwrap();
        assert!(z0.abs() < 1e-10, "⟨Z₀⟩ should be 0");
        assert!(z1.abs() < 1e-10, "⟨Z₁⟩ should be 0");
    }

    #[test]
    fn test_qft() {
        let mut state = QuantumState::new(3).unwrap();

        // Apply QFT to all 3 qubits starting from |000⟩
        state.qft(&[0, 1, 2]);

        // QFT of |0⟩^n creates uniform superposition
        let probs = state.probabilities();
        let expected = 1.0 / 8.0; // 1/2^n

        for p in probs.iter() {
            assert!((p - expected).abs() < 1e-10);
        }
    }
}
