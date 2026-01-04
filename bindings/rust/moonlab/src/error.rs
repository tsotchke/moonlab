//! Error types for the Moonlab quantum simulator.
//!
//! This module provides a strongly-typed error hierarchy that converts
//! C error codes into Rust-idiomatic error handling.

use thiserror::Error;

/// Main error type for all quantum simulator operations.
#[derive(Debug, Error)]
pub enum QuantumError {
    /// Invalid qubit index (out of bounds).
    #[error("Invalid qubit index {index}: must be < {max}")]
    InvalidQubit { index: usize, max: usize },

    /// Attempted to measure a qubit outside the valid range.
    #[error("Invalid measurement target: qubit {0}")]
    InvalidMeasurement(usize),

    /// State vector is not properly normalized.
    #[error("Quantum state is not normalized (norm = {0:.6})")]
    NotNormalized(f64),

    /// Memory allocation failed during state creation.
    #[error("Failed to allocate quantum state with {0} qubits")]
    AllocationFailed(usize),

    /// The requested operation is not supported for this state.
    #[error("Operation not supported: {0}")]
    UnsupportedOperation(String),

    /// Control and target qubits are the same.
    #[error("Control qubit {0} cannot equal target qubit")]
    SameControlTarget(usize),

    /// Generic FFI error from the C library.
    #[error("FFI error: {0}")]
    Ffi(String),

    /// Entropy source error.
    #[error("Entropy error: {0}")]
    Entropy(String),

    /// Algorithm-specific error.
    #[error("Algorithm error: {0}")]
    Algorithm(String),

    /// Null pointer received from C.
    #[error("Null pointer returned from C library")]
    NullPointer,
}

/// Convenience type alias for Results using QuantumError.
pub type Result<T> = std::result::Result<T, QuantumError>;

/// Convert C error code to Result.
///
/// The C library uses 0 for success and negative values for errors.
#[allow(dead_code)]
pub(crate) fn check_result(code: i32, context: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(QuantumError::Ffi(format!("{}: error code {}", context, code)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = QuantumError::InvalidQubit { index: 5, max: 4 };
        assert!(err.to_string().contains("5"));
        assert!(err.to_string().contains("4"));
    }

    #[test]
    fn test_check_result() {
        assert!(check_result(0, "test").is_ok());
        assert!(check_result(-1, "test").is_err());
    }
}
