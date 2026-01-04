# Rust Safe API

Complete reference for the safe Moonlab Rust bindings.

**Crate**: `moonlab`

## Overview

The `moonlab` crate provides safe, idiomatic Rust bindings for the Moonlab quantum simulator. Features:

- **RAII Memory Management**: Automatic cleanup via Rust ownership
- **Method Chaining**: Fluent API for building quantum circuits
- **Type Safety**: Compile-time qubit index validation
- **Error Handling**: Rust `Result` types with descriptive errors

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
moonlab = { path = "bindings/rust/moonlab" }
```

Or from crates.io (when published):

```toml
[dependencies]
moonlab = "0.1"
```

## Quick Start

```rust
use moonlab::QuantumState;

fn main() -> moonlab::Result<()> {
    // Create a 3-qubit quantum state
    let mut state = QuantumState::new(3)?;

    // Create GHZ state with method chaining
    state.h(0)           // Hadamard on qubit 0
         .cnot(0, 1)     // CNOT: control=0, target=1
         .cnot(0, 2);    // CNOT: control=0, target=2

    // Get measurement probabilities
    let probs = state.probabilities();
    println!("|000⟩: {:.3}", probs[0]); // ≈ 0.5
    println!("|111⟩: {:.3}", probs[7]); // ≈ 0.5

    // Compute entanglement
    let entropy = state.entanglement_entropy(&[0])?;
    println!("Entanglement entropy: {:.3} bits", entropy);

    Ok(())
}
```

## QuantumState

Main type for quantum state simulation.

### Constants

```rust
pub const MAX_QUBITS: usize = 32;
```

Maximum supported qubits (memory-limited).

### Constructor

```rust
impl QuantumState {
    pub fn new(num_qubits: usize) -> Result<Self>
}
```

Create a new quantum state initialized to $|0\ldots0\rangle$.

**Parameters**:
- `num_qubits`: Number of qubits (1-32)

**Returns**: `Result<QuantumState>` or error if invalid

**Errors**:
- `QuantumError::InvalidQubit` - qubit count out of range
- `QuantumError::AllocationFailed` - memory allocation failed

**Example**:
```rust
use moonlab::QuantumState;

let state = QuantumState::new(4)?;
assert_eq!(state.num_qubits(), 4);
assert_eq!(state.state_dim(), 16); // 2^4
```

### Properties

```rust
impl QuantumState {
    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize

    /// Get the dimension of the state vector (2^n)
    pub fn state_dim(&self) -> usize
}
```

### State Operations

#### reset

```rust
pub fn reset(&mut self) -> &mut Self
```

Reset state to $|0\ldots0\rangle$.

```rust
state.h(0).cnot(0, 1);
state.reset();  // Back to |00⟩
```

#### probabilities

```rust
pub fn probabilities(&self) -> Vec<f64>
```

Get probability distribution over all basis states.

**Returns**: Vector of length $2^n$ where entry $i$ is $P(|i\rangle)$

```rust
let probs = state.probabilities();
for (i, p) in probs.iter().enumerate() {
    if *p > 0.01 {
        println!("|{:0width$b}⟩: {:.4}", i, p, width = state.num_qubits());
    }
}
```

#### amplitudes

```rust
pub fn amplitudes(&self) -> Vec<Complex64>
```

Get complex amplitudes of the state vector.

**Returns**: Vector of `Complex64` values

```rust
use num_complex::Complex64;

let amps = state.amplitudes();
let norm: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
assert!((norm - 1.0).abs() < 1e-10);
```

#### prob_zero / prob_one

```rust
pub fn prob_zero(&self, qubit: usize) -> Result<f64>
pub fn prob_one(&self, qubit: usize) -> Result<f64>
```

Get probability of measuring specific qubit value.

```rust
let p0 = state.prob_zero(0)?;
let p1 = state.prob_one(0)?;
assert!((p0 + p1 - 1.0).abs() < 1e-10);
```

### Single-Qubit Gates

All single-qubit gates return `&mut Self` for method chaining.

```rust
impl QuantumState {
    /// Pauli-X (NOT) gate
    pub fn x(&mut self, qubit: usize) -> &mut Self

    /// Pauli-Y gate
    pub fn y(&mut self, qubit: usize) -> &mut Self

    /// Pauli-Z gate
    pub fn z(&mut self, qubit: usize) -> &mut Self

    /// Hadamard gate
    pub fn h(&mut self, qubit: usize) -> &mut Self

    /// S gate (√Z)
    pub fn s(&mut self, qubit: usize) -> &mut Self

    /// S† gate
    pub fn sdg(&mut self, qubit: usize) -> &mut Self

    /// T gate (π/8)
    pub fn t(&mut self, qubit: usize) -> &mut Self

    /// T† gate
    pub fn tdg(&mut self, qubit: usize) -> &mut Self

    /// X rotation
    pub fn rx(&mut self, qubit: usize, theta: f64) -> &mut Self

    /// Y rotation
    pub fn ry(&mut self, qubit: usize, theta: f64) -> &mut Self

    /// Z rotation
    pub fn rz(&mut self, qubit: usize, theta: f64) -> &mut Self

    /// Phase gate
    pub fn phase(&mut self, qubit: usize, phi: f64) -> &mut Self

    /// U3 gate (arbitrary single-qubit unitary)
    pub fn u3(&mut self, qubit: usize, theta: f64, phi: f64, lambda: f64) -> &mut Self
}
```

**Example**:
```rust
use std::f64::consts::PI;

let mut state = QuantumState::new(3)?;

// Build circuit with method chaining
state.h(0)
     .rx(1, PI / 4.0)
     .t(2)
     .phase(0, PI / 2.0);
```

### Two-Qubit Gates

```rust
impl QuantumState {
    /// CNOT (controlled-X)
    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self

    /// CX (alias for CNOT)
    pub fn cx(&mut self, control: usize, target: usize) -> &mut Self

    /// Controlled-Z
    pub fn cz(&mut self, control: usize, target: usize) -> &mut Self

    /// Controlled-Y
    pub fn cy(&mut self, control: usize, target: usize) -> &mut Self

    /// SWAP
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) -> &mut Self

    /// Controlled RX
    pub fn crx(&mut self, control: usize, target: usize, theta: f64) -> &mut Self

    /// Controlled RY
    pub fn cry(&mut self, control: usize, target: usize, theta: f64) -> &mut Self

    /// Controlled RZ
    pub fn crz(&mut self, control: usize, target: usize, theta: f64) -> &mut Self

    /// Controlled phase
    pub fn cphase(&mut self, control: usize, target: usize, phi: f64) -> &mut Self
}
```

**Example**:
```rust
// Create Bell state
state.h(0).cnot(0, 1);

// Check entanglement
let probs = state.probabilities();
assert!((probs[0] - 0.5).abs() < 1e-10); // |00⟩
assert!((probs[3] - 0.5).abs() < 1e-10); // |11⟩
```

### Multi-Qubit Gates

```rust
impl QuantumState {
    /// Toffoli (CCNOT)
    pub fn toffoli(&mut self, control1: usize, control2: usize, target: usize) -> &mut Self

    /// CCX (alias for Toffoli)
    pub fn ccx(&mut self, control1: usize, control2: usize, target: usize) -> &mut Self

    /// Fredkin (CSWAP)
    pub fn fredkin(&mut self, control: usize, target1: usize, target2: usize) -> &mut Self

    /// CSWAP (alias for Fredkin)
    pub fn cswap(&mut self, control: usize, target1: usize, target2: usize) -> &mut Self

    /// Quantum Fourier Transform
    pub fn qft(&mut self, qubits: &[usize]) -> &mut Self

    /// Inverse QFT
    pub fn iqft(&mut self, qubits: &[usize]) -> &mut Self
}
```

**Example**:
```rust
// Apply QFT to all qubits
let mut state = QuantumState::new(4)?;
state.qft(&[0, 1, 2, 3]);

// QFT of |0⟩ creates uniform superposition
let probs = state.probabilities();
let expected = 1.0 / 16.0;
for p in probs.iter() {
    assert!((p - expected).abs() < 1e-10);
}
```

### Measurement and Expectation Values

```rust
impl QuantumState {
    /// Expectation value ⟨Z⟩ on qubit
    pub fn expectation_z(&self, qubit: usize) -> Result<f64>

    /// Expectation value ⟨X⟩ on qubit
    pub fn expectation_x(&self, qubit: usize) -> Result<f64>

    /// Expectation value ⟨Y⟩ on qubit
    pub fn expectation_y(&self, qubit: usize) -> Result<f64>

    /// ZZ correlation ⟨Z_i Z_j⟩
    pub fn correlation_zz(&self, qubit_i: usize, qubit_j: usize) -> Result<f64>
}
```

**Example**:
```rust
let mut state = QuantumState::new(2)?;
state.h(0).cnot(0, 1);  // Bell state

// Bell state has perfect ZZ correlation
let zz = state.correlation_zz(0, 1)?;
assert!((zz - 1.0).abs() < 1e-10);

// Individual Z expectations are 0
let z0 = state.expectation_z(0)?;
assert!(z0.abs() < 1e-10);
```

### Entanglement Measures

```rust
impl QuantumState {
    /// Von Neumann entropy of subsystem A
    pub fn entanglement_entropy(&self, subsystem_a: &[usize]) -> Result<f64>

    /// Purity Tr(ρ²)
    pub fn purity(&self) -> f64

    /// Full state entropy
    pub fn entropy(&self) -> f64
}
```

**Example**:
```rust
let mut state = QuantumState::new(2)?;

// Product state has zero entanglement
let entropy_product = state.entanglement_entropy(&[0])?;
assert!(entropy_product < 1e-10);

// Bell state has maximal entanglement
state.h(0).cnot(0, 1);
let entropy_bell = state.entanglement_entropy(&[0])?;
assert!(entropy_bell > 0.6); // ln(2) ≈ 0.693
```

## Error Handling

### QuantumError

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum QuantumError {
    #[error("Invalid qubit index {index}: must be < {max}")]
    InvalidQubit { index: usize, max: usize },

    #[error("Invalid measurement target: qubit {0}")]
    InvalidMeasurement(usize),

    #[error("Quantum state is not normalized (norm = {0:.6})")]
    NotNormalized(f64),

    #[error("Failed to allocate quantum state with {0} qubits")]
    AllocationFailed(usize),

    #[error("Operation not supported: {0}")]
    UnsupportedOperation(String),

    #[error("Control qubit {0} cannot equal target qubit")]
    SameControlTarget(usize),

    #[error("FFI error: {0}")]
    Ffi(String),

    #[error("Entropy error: {0}")]
    Entropy(String),

    #[error("Algorithm error: {0}")]
    Algorithm(String),

    #[error("Null pointer returned from C library")]
    NullPointer,
}

pub type Result<T> = std::result::Result<T, QuantumError>;
```

**Example**:
```rust
use moonlab::{QuantumState, QuantumError};

fn run_circuit() -> moonlab::Result<()> {
    let state = QuantumState::new(50)?;  // Too many qubits
    Ok(())
}

match run_circuit() {
    Err(QuantumError::AllocationFailed(n)) => {
        eprintln!("Cannot allocate {} qubits", n);
    }
    Err(e) => eprintln!("Error: {}", e),
    Ok(_) => {}
}
```

## Traits

### Clone

`QuantumState` implements `Clone` for deep copying.

```rust
let mut original = QuantumState::new(2)?;
original.h(0).cnot(0, 1);

let copy = original.clone();

// States are independent
let orig_probs = original.probabilities();
let copy_probs = copy.probabilities();
assert_eq!(orig_probs, copy_probs);
```

### Drop

Memory is automatically freed when `QuantumState` goes out of scope.

```rust
{
    let state = QuantumState::new(20)?;
    // Use state...
} // Memory freed here
```

### Send

`QuantumState` is `Send`, allowing transfer between threads.

```rust
use std::thread;

let state = QuantumState::new(4)?;

let handle = thread::spawn(move || {
    // State moved to this thread
    state.probabilities()
});

let probs = handle.join().unwrap();
```

**Note**: `QuantumState` is NOT `Sync`. For concurrent access, use `Arc<Mutex<QuantumState>>`.

## Prelude

For convenient imports:

```rust
use moonlab::prelude::*;

// Now available: QuantumState, QuantumError, Result, MAX_QUBITS,
//                FeynmanDiagram, ParticleType
```

## Complete Example

```rust
use moonlab::prelude::*;
use std::f64::consts::PI;

fn main() -> Result<()> {
    // Create 4-qubit state
    let mut state = QuantumState::new(4)?;

    // Build a variational ansatz
    // Layer 1: Single-qubit rotations
    for q in 0..4 {
        state.ry(q, PI / 4.0);
    }

    // Layer 2: Entangling gates
    state.cnot(0, 1).cnot(1, 2).cnot(2, 3);

    // Layer 3: More rotations
    for q in 0..4 {
        state.rz(q, PI / 3.0);
    }

    // Analyze the state
    println!("Probabilities:");
    let probs = state.probabilities();
    for (i, p) in probs.iter().enumerate() {
        if *p > 0.01 {
            println!("  |{:04b}⟩: {:.4}", i, p);
        }
    }

    // Entanglement analysis
    println!("\nEntanglement:");
    for partition in 1..4 {
        let subsystem: Vec<usize> = (0..partition).collect();
        let entropy = state.entanglement_entropy(&subsystem)?;
        println!("  S({:?}|rest) = {:.4} bits", subsystem, entropy);
    }

    // Expectation values
    println!("\nExpectations:");
    for q in 0..4 {
        let z = state.expectation_z(q)?;
        println!("  ⟨Z_{}⟩ = {:.4}", q, z);
    }

    Ok(())
}
```

## Thread Safety

- `QuantumState` is `Send` but not `Sync`
- Create separate states per thread for parallel simulations
- Use `Arc<Mutex<QuantumState>>` for shared state

```rust
use std::sync::{Arc, Mutex};
use std::thread;

let state = Arc::new(Mutex::new(QuantumState::new(4)?));

let handles: Vec<_> = (0..4).map(|i| {
    let state = Arc::clone(&state);
    thread::spawn(move || {
        let mut guard = state.lock().unwrap();
        guard.h(i);
    })
}).collect();

for handle in handles {
    handle.join().unwrap();
}
```

## See Also

- [moonlab-sys](moonlab-sys.md) - Low-level FFI bindings
- [moonlab-tui](moonlab-tui.md) - Terminal user interface
- [C API: Quantum State](../c/quantum-state.md) - Underlying C API

