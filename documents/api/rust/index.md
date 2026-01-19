# Rust API Reference

Safe, ergonomic Rust bindings for Moonlab Quantum Simulator with full ownership semantics and zero-cost abstractions.

## Overview

| Crate | Description |
|-------|-------------|
| [moonlab](moonlab.md) | Safe high-level API |
| [moonlab-sys](moonlab-sys.md) | Raw FFI bindings |
| [moonlab-tui](moonlab-tui.md) | Terminal UI application |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
moonlab = { path = "/path/to/quantum-simulator/bindings/rust/moonlab" }
```

Or build the static library for FFI:

```bash
cd quantum-simulator
make rust-lib  # Creates libquantumsim.a
```

## Quick Start

```rust
use moonlab::{QuantumState, QuantumError};

fn main() -> Result<(), QuantumError> {
    // Create a 2-qubit state
    let mut state = QuantumState::new(2)?;

    // Create Bell state
    state.h(0)?.cnot(0, 1)?;

    // Get probabilities
    let probs = state.probabilities();
    println!("P(|00⟩) = {:.4}", probs[0]);
    println!("P(|11⟩) = {:.4}", probs[3]);

    // Measure
    let result = state.measure()?;
    println!("Measured: |{:02b}⟩", result);

    Ok(())
}
```

## QuantumState

The main type representing a quantum state vector.

### Creation

```rust
impl QuantumState {
    /// Create a new quantum state initialized to |0...0⟩
    pub fn new(num_qubits: usize) -> Result<Self, QuantumError>;

    /// Create from amplitude vector
    pub fn from_amplitudes(amplitudes: Vec<Complex64>) -> Result<Self, QuantumError>;

    /// Clone the state
    pub fn clone_state(&self) -> Result<Self, QuantumError>;
}
```

**Example**:
```rust
use moonlab::QuantumState;
use num_complex::Complex64;

// Standard initialization
let state = QuantumState::new(4)?;

// From amplitudes (Bell state)
let bell = QuantumState::from_amplitudes(vec![
    Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    Complex64::new(0.0, 0.0),
    Complex64::new(0.0, 0.0),
    Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
])?;
```

### Properties

```rust
impl QuantumState {
    /// Number of qubits
    pub fn num_qubits(&self) -> usize;

    /// State vector dimension (2^n)
    pub fn state_dim(&self) -> usize;

    /// Get amplitude for a basis state
    pub fn amplitude(&self, index: usize) -> Complex64;

    /// Get all amplitudes
    pub fn amplitudes(&self) -> &[Complex64];

    /// Get probability of a basis state
    pub fn probability(&self, index: usize) -> f64;

    /// Get all probabilities
    pub fn probabilities(&self) -> Vec<f64>;
}
```

### Gates

All gate methods return `Result<&mut Self, QuantumError>` for chaining:

```rust
impl QuantumState {
    // Single-qubit gates
    pub fn x(&mut self, qubit: usize) -> Result<&mut Self, QuantumError>;
    pub fn y(&mut self, qubit: usize) -> Result<&mut Self, QuantumError>;
    pub fn z(&mut self, qubit: usize) -> Result<&mut Self, QuantumError>;
    pub fn h(&mut self, qubit: usize) -> Result<&mut Self, QuantumError>;
    pub fn s(&mut self, qubit: usize) -> Result<&mut Self, QuantumError>;
    pub fn s_dagger(&mut self, qubit: usize) -> Result<&mut Self, QuantumError>;
    pub fn t(&mut self, qubit: usize) -> Result<&mut Self, QuantumError>;
    pub fn t_dagger(&mut self, qubit: usize) -> Result<&mut Self, QuantumError>;

    // Rotation gates
    pub fn rx(&mut self, qubit: usize, theta: f64) -> Result<&mut Self, QuantumError>;
    pub fn ry(&mut self, qubit: usize, theta: f64) -> Result<&mut Self, QuantumError>;
    pub fn rz(&mut self, qubit: usize, theta: f64) -> Result<&mut Self, QuantumError>;
    pub fn phase(&mut self, qubit: usize, theta: f64) -> Result<&mut Self, QuantumError>;

    // Two-qubit gates
    pub fn cnot(&mut self, control: usize, target: usize) -> Result<&mut Self, QuantumError>;
    pub fn cz(&mut self, control: usize, target: usize) -> Result<&mut Self, QuantumError>;
    pub fn swap(&mut self, q1: usize, q2: usize) -> Result<&mut Self, QuantumError>;

    // Three-qubit gates
    pub fn toffoli(&mut self, c1: usize, c2: usize, target: usize) -> Result<&mut Self, QuantumError>;
}
```

**Method chaining**:
```rust
let mut state = QuantumState::new(3)?;
state
    .h(0)?
    .cnot(0, 1)?
    .cnot(1, 2)?;  // GHZ state
```

### Measurement

```rust
impl QuantumState {
    /// Measure all qubits (collapses state)
    pub fn measure(&mut self) -> Result<u64, QuantumError>;

    /// Measure specific qubits
    pub fn measure_qubits(&mut self, qubits: &[usize]) -> Result<u64, QuantumError>;

    /// Sample without collapse (for statistics)
    pub fn sample(&self, shots: usize) -> Vec<u64>;
}
```

### State Operations

```rust
impl QuantumState {
    /// Reset to |0...0⟩
    pub fn reset(&mut self);

    /// Normalize the state
    pub fn normalize(&mut self) -> Result<(), QuantumError>;

    /// Calculate entanglement entropy
    pub fn entanglement_entropy(&self, subsystem: &[usize]) -> f64;

    /// Calculate purity Tr(ρ²)
    pub fn purity(&self) -> f64;

    /// Calculate fidelity with another state
    pub fn fidelity(&self, other: &QuantumState) -> f64;
}
```

## Error Handling

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum QuantumError {
    InvalidQubit { qubit: usize, max: usize },
    InvalidState,
    NotNormalized,
    OutOfMemory,
    InvalidDimension,
}

impl std::error::Error for QuantumError {}
```

**Usage**:
```rust
use moonlab::{QuantumState, QuantumError};

fn run_circuit() -> Result<(), QuantumError> {
    let mut state = QuantumState::new(10)?;

    // Error handling with match
    match state.cnot(0, 15) {
        Ok(_) => println!("Success"),
        Err(QuantumError::InvalidQubit { qubit, max }) => {
            eprintln!("Qubit {} out of range (max {})", qubit, max);
        }
        Err(e) => return Err(e),
    }

    Ok(())
}
```

## Algorithms

```rust
use moonlab::algorithms::{grover, vqe, qaoa};
```

### Grover's Search

```rust
pub fn grover_search<F>(
    num_qubits: usize,
    oracle: F,
    num_iterations: Option<usize>,
) -> Result<u64, QuantumError>
where
    F: Fn(&mut QuantumState) -> Result<(), QuantumError>;
```

**Example**:
```rust
use moonlab::algorithms::grover_search;

let result = grover_search(4, |state| {
    // Oracle marking |1010⟩
    state.cz(1, 3)?;  // Simplified oracle
    Ok(())
}, None)?;

println!("Found: {:04b}", result);
```

### VQE

```rust
pub struct VqeResult {
    pub energy: f64,
    pub parameters: Vec<f64>,
    pub iterations: usize,
}

pub fn vqe<H, A>(
    hamiltonian: H,
    ansatz: A,
    initial_params: Vec<f64>,
) -> Result<VqeResult, QuantumError>
where
    H: Fn(&QuantumState) -> f64,
    A: Fn(&mut QuantumState, &[f64]) -> Result<(), QuantumError>;
```

## Traits

### AsQuantumGate

Custom gate implementation:

```rust
pub trait AsQuantumGate {
    fn apply(&self, state: &mut QuantumState) -> Result<(), QuantumError>;
}

// Example custom gate
struct CustomRotation {
    qubit: usize,
    theta: f64,
    phi: f64,
}

impl AsQuantumGate for CustomRotation {
    fn apply(&self, state: &mut QuantumState) -> Result<(), QuantumError> {
        state.rz(self.qubit, self.phi)?;
        state.ry(self.qubit, self.theta)?;
        state.rz(self.qubit, -self.phi)?;
        Ok(())
    }
}
```

## FFI (moonlab-sys)

Raw bindings for advanced use:

```rust
use moonlab_sys::{
    quantum_state_t,
    quantum_state_init,
    quantum_state_free,
    gate_hadamard,
};

unsafe {
    let mut state = std::mem::zeroed::<quantum_state_t>();
    quantum_state_init(&mut state, 4);
    gate_hadamard(&mut state, 0);
    quantum_state_free(&mut state);
}
```

## Terminal UI (moonlab-tui)

Interactive quantum circuit builder:

```bash
cargo run --package moonlab-tui

# Or install
cargo install --path bindings/rust/moonlab-tui
moonlab-tui
```

Features:
- Visual circuit editor
- Real-time state visualization
- Bloch sphere display
- Measurement statistics

## Thread Safety

`QuantumState` is `Send` but not `Sync`:

```rust
use std::thread;

// OK: Move to another thread
let state = QuantumState::new(4)?;
thread::spawn(move || {
    // Use state in this thread
})?;

// NOT OK: Share across threads
// Use Arc<Mutex<QuantumState>> if needed
```

## Performance

### Benchmarking

```rust
use std::time::Instant;

let start = Instant::now();
let mut state = QuantumState::new(20)?;

for i in 0..20 {
    state.h(i)?;
}

let elapsed = start.elapsed();
println!("20-qubit H⊗20: {:?}", elapsed);
```

### Memory Usage

```rust
use moonlab::QuantumState;

// Query memory before allocation
let before = get_memory_usage();

let state = QuantumState::new(25)?;  // ~512 MB

let after = get_memory_usage();
println!("Memory used: {} MB", (after - before) / 1_000_000);
```

## no_std Support

The core crate supports `no_std` environments:

```toml
[dependencies]
moonlab = { path = "...", default-features = false }
```

Requires custom allocator and `libm` for math operations.

## See Also

- [moonlab crate](moonlab.md) - Full API reference
- [moonlab-sys](moonlab-sys.md) - FFI bindings
- [moonlab-tui](moonlab-tui.md) - Terminal application
- [Examples](../../examples/index.md) - Code examples
