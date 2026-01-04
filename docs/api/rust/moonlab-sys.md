# Rust FFI Bindings

Low-level FFI bindings to the Moonlab C library.

**Crate**: `moonlab-sys`

## Overview

The `moonlab-sys` crate provides raw, unsafe FFI bindings to the Moonlab C quantum simulator library. These bindings are automatically generated using `bindgen` and provide direct access to all C functions.

**Important**: For most use cases, prefer the safe `moonlab` crate instead. This crate is intended for:
- Building higher-level abstractions
- Direct C interop requirements
- Advanced use cases requiring maximum control

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
moonlab-sys = { path = "bindings/rust/moonlab-sys" }
```

## Build Requirements

The crate requires:
- C compiler (clang or gcc)
- libclang for bindgen
- Moonlab C library source files

### macOS

```bash
brew install llvm
export LIBCLANG_PATH=$(brew --prefix llvm)/lib
```

### Linux

```bash
# Ubuntu/Debian
sudo apt install libclang-dev

# Fedora
sudo dnf install clang-devel
```

## Safety

All functions in this crate are unsafe and require careful handling of:
- Pointer validity
- Memory management
- Lifetime management
- Thread safety

The safe wrappers in the `moonlab` crate handle these concerns automatically.

## Core Types

### quantum_state_t

The main quantum state structure (opaque).

```rust
#[repr(C)]
pub struct quantum_state_t {
    pub num_qubits: usize,
    pub state_dim: u64,
    pub amplitudes: *mut complex_t,
    // ... additional internal fields
}
```

### complex_t

Complex number representation.

```rust
#[repr(C)]
pub struct complex_t {
    pub re: f64,
    pub im: f64,
}
```

## Function Reference

### State Management

```rust
/// Initialize a quantum state
///
/// # Safety
/// - `state` must be a valid pointer to uninitialized quantum_state_t
/// - `num_qubits` must be 1-32
pub unsafe fn quantum_state_init(
    state: *mut quantum_state_t,
    num_qubits: usize
) -> i32;

/// Free a quantum state
///
/// # Safety
/// - `state` must be a valid pointer to initialized quantum_state_t
/// - Must not be called twice on same state
pub unsafe fn quantum_state_free(state: *mut quantum_state_t);

/// Clone a quantum state
///
/// # Safety
/// - `dest` must be valid pointer to uninitialized quantum_state_t
/// - `src` must be valid pointer to initialized quantum_state_t
pub unsafe fn quantum_state_clone(
    dest: *mut quantum_state_t,
    src: *const quantum_state_t
);

/// Reset state to |0...0⟩
pub unsafe fn quantum_state_reset(state: *mut quantum_state_t);
```

### Quantum Gates

```rust
/// Single-qubit gates
pub unsafe fn gate_pauli_x(state: *mut quantum_state_t, qubit: i32) -> i32;
pub unsafe fn gate_pauli_y(state: *mut quantum_state_t, qubit: i32) -> i32;
pub unsafe fn gate_pauli_z(state: *mut quantum_state_t, qubit: i32) -> i32;
pub unsafe fn gate_hadamard(state: *mut quantum_state_t, qubit: i32) -> i32;
pub unsafe fn gate_s(state: *mut quantum_state_t, qubit: i32) -> i32;
pub unsafe fn gate_s_dagger(state: *mut quantum_state_t, qubit: i32) -> i32;
pub unsafe fn gate_t(state: *mut quantum_state_t, qubit: i32) -> i32;
pub unsafe fn gate_t_dagger(state: *mut quantum_state_t, qubit: i32) -> i32;

/// Rotation gates
pub unsafe fn gate_rx(state: *mut quantum_state_t, qubit: i32, theta: f64) -> i32;
pub unsafe fn gate_ry(state: *mut quantum_state_t, qubit: i32, theta: f64) -> i32;
pub unsafe fn gate_rz(state: *mut quantum_state_t, qubit: i32, theta: f64) -> i32;
pub unsafe fn gate_phase(state: *mut quantum_state_t, qubit: i32, phi: f64) -> i32;
pub unsafe fn gate_u3(
    state: *mut quantum_state_t,
    qubit: i32,
    theta: f64,
    phi: f64,
    lambda: f64
) -> i32;

/// Two-qubit gates
pub unsafe fn gate_cnot(
    state: *mut quantum_state_t,
    control: i32,
    target: i32
) -> i32;
pub unsafe fn gate_cz(
    state: *mut quantum_state_t,
    control: i32,
    target: i32
) -> i32;
pub unsafe fn gate_cy(
    state: *mut quantum_state_t,
    control: i32,
    target: i32
) -> i32;
pub unsafe fn gate_swap(
    state: *mut quantum_state_t,
    qubit1: i32,
    qubit2: i32
) -> i32;
pub unsafe fn gate_cphase(
    state: *mut quantum_state_t,
    control: i32,
    target: i32,
    phi: f64
) -> i32;
pub unsafe fn gate_crx(
    state: *mut quantum_state_t,
    control: i32,
    target: i32,
    theta: f64
) -> i32;
pub unsafe fn gate_cry(
    state: *mut quantum_state_t,
    control: i32,
    target: i32,
    theta: f64
) -> i32;
pub unsafe fn gate_crz(
    state: *mut quantum_state_t,
    control: i32,
    target: i32,
    theta: f64
) -> i32;

/// Multi-qubit gates
pub unsafe fn gate_toffoli(
    state: *mut quantum_state_t,
    control1: i32,
    control2: i32,
    target: i32
) -> i32;
pub unsafe fn gate_fredkin(
    state: *mut quantum_state_t,
    control: i32,
    target1: i32,
    target2: i32
) -> i32;
pub unsafe fn gate_qft(
    state: *mut quantum_state_t,
    qubits: *const i32,
    num_qubits: usize
) -> i32;
pub unsafe fn gate_iqft(
    state: *mut quantum_state_t,
    qubits: *const i32,
    num_qubits: usize
) -> i32;
```

### Measurement

```rust
/// Get probability of measuring |0⟩ on qubit
pub unsafe fn measurement_probability_zero(
    state: *const quantum_state_t,
    qubit: i32
) -> f64;

/// Get probability of measuring |1⟩ on qubit
pub unsafe fn measurement_probability_one(
    state: *const quantum_state_t,
    qubit: i32
) -> f64;

/// Expectation values
pub unsafe fn measurement_expectation_z(
    state: *const quantum_state_t,
    qubit: i32
) -> f64;
pub unsafe fn measurement_expectation_x(
    state: *const quantum_state_t,
    qubit: i32
) -> f64;
pub unsafe fn measurement_expectation_y(
    state: *const quantum_state_t,
    qubit: i32
) -> f64;

/// ZZ correlation
pub unsafe fn measurement_correlation_zz(
    state: *const quantum_state_t,
    qubit_i: i32,
    qubit_j: i32
) -> f64;
```

### Entanglement

```rust
/// Von Neumann entropy of subsystem
pub unsafe fn quantum_state_entanglement_entropy(
    state: *const quantum_state_t,
    subsystem: *const i32,
    subsystem_size: usize
) -> f64;

/// Purity Tr(ρ²)
pub unsafe fn quantum_state_purity(state: *const quantum_state_t) -> f64;

/// Full state entropy
pub unsafe fn quantum_state_entropy(state: *const quantum_state_t) -> f64;
```

## Usage Example

```rust
use moonlab_sys as ffi;
use std::mem::MaybeUninit;

fn main() {
    unsafe {
        // Allocate and initialize state
        let mut state = MaybeUninit::<ffi::quantum_state_t>::uninit();
        let result = ffi::quantum_state_init(state.as_mut_ptr(), 2);
        assert_eq!(result, 0, "Initialization failed");

        let state = state.assume_init_mut();

        // Create Bell state
        ffi::gate_hadamard(state, 0);
        ffi::gate_cnot(state, 0, 1);

        // Check probabilities
        let prob_00 = ffi::measurement_probability_zero(state, 0);
        let prob_11 = ffi::measurement_probability_one(state, 0);

        println!("|00⟩ + |11⟩ superposition:");
        println!("  P(0) = {:.4}", prob_00);
        println!("  P(1) = {:.4}", prob_11);

        // Compute entanglement
        let subsystem: [i32; 1] = [0];
        let entropy = ffi::quantum_state_entanglement_entropy(
            state,
            subsystem.as_ptr(),
            1
        );
        println!("  Entropy = {:.4} bits", entropy);

        // Clean up
        ffi::quantum_state_free(state);
    }
}
```

## Error Codes

Most functions return an `i32` status code:

| Code | Meaning |
|------|---------|
| 0 | Success |
| -1 | Invalid parameter |
| -2 | Allocation failed |
| -3 | Invalid qubit index |
| -4 | Internal error |

## Build Script

The crate uses a `build.rs` that:
1. Locates the Moonlab C source files
2. Runs bindgen to generate Rust bindings
3. Links against the compiled C library

```rust
// build.rs (simplified)
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");

    // Find C source directory
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let src_dir = PathBuf::from(&manifest_dir)
        .parent().unwrap()
        .parent().unwrap()
        .join("src");

    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", src_dir.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");
}
```

## Memory Management

When using `moonlab-sys` directly, you must manually manage memory:

```rust
use moonlab_sys as ffi;
use std::mem::MaybeUninit;

unsafe {
    // Allocation
    let mut state = MaybeUninit::<ffi::quantum_state_t>::uninit();
    ffi::quantum_state_init(state.as_mut_ptr(), 4);
    let state_ptr = state.as_mut_ptr();

    // Use state...
    ffi::gate_hadamard(state_ptr, 0);

    // MUST free when done
    ffi::quantum_state_free(state_ptr);

    // WARNING: Do not use state_ptr after free!
}
```

## Thread Safety

The C library is NOT thread-safe. When using `moonlab-sys`:
- Create separate states per thread
- Never share `quantum_state_t` pointers across threads without synchronization
- Use mutex if shared access is required

## See Also

- [moonlab](moonlab.md) - Safe Rust wrapper (recommended)
- [moonlab-tui](moonlab-tui.md) - Terminal user interface
- [C API](../c/index.md) - Complete C API reference

