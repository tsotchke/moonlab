# API Reference

Complete API documentation for Moonlab Quantum Simulator across all supported languages. The core simulation engine is written in C, with bindings for Python, Rust, and JavaScript.

## API Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    Moonlab Quantum Simulator                   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐   │
│  │   Python    │   │    Rust     │   │    JavaScript       │   │
│  │   Bindings  │   │   Bindings  │   │    (WebAssembly)    │   │
│  └──────┬──────┘   └──────┬──────┘   └──────────┬──────────┘   │
│         │                 │                     │              │
│         └─────────────────┼─────────────────────┘              │
│                           │                                    │
│                           ▼                                    │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                     C Core Library                        │ │
│  │                                                           │ │
│  │  quantum/    algorithms/    optimization/    distributed/ │ │
│  │  • state     • grover       • simd_ops      • mpi_bridge  │ │
│  │  • gates     • vqe          • gpu_metal     • collective  │ │
│  │  • measure   • qaoa         • parallel_ops                │ │
│  │  • entangle  • dmrg                                       │ │
│  │  • noise     • tensor                                     │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Language-Specific Documentation

### [C API](c/index.md)

The native C library providing maximum performance and direct hardware control.

**Best for**:
- Performance-critical applications
- Embedded systems
- Building custom bindings
- Integration with existing C/C++ codebases

**Features**:
- Full state-vector simulation up to 32 qubits
- Universal gate set with SIMD acceleration
- Metal GPU acceleration on Apple Silicon
- MPI-based distributed computing

### [Python API](python/index.md)

High-level Pythonic interface with NumPy integration and machine learning support.

**Best for**:
- Rapid prototyping
- Data science workflows
- Educational purposes
- Integration with PyTorch/TensorFlow

**Features**:
- NumPy-compatible array operations
- PyTorch quantum layers
- Jupyter notebook support
- Matplotlib visualization

### [Rust API](rust/index.md)

Safe, ergonomic Rust bindings with full ownership semantics.

**Best for**:
- Production systems requiring safety guarantees
- High-performance applications
- CLI tools
- WebAssembly compilation

**Features**:
- Memory-safe wrappers around C library
- Ownership-based resource management
- `moonlab-tui` terminal application
- `no_std` compatible core

### [JavaScript API](javascript/index.md)

WebAssembly-compiled core with React and Vue components.

**Best for**:
- Web applications
- Interactive visualizations
- Educational tools
- Cross-platform deployment

**Features**:
- WebAssembly performance
- React/Vue component library
- D3.js-based visualization
- Browser and Node.js support

## Core Modules

### Quantum State Management

| Module | C Header | Description |
|--------|----------|-------------|
| State | `state.h` | Quantum state initialization, manipulation, properties |
| Gates | `gates.h` | Universal gate set implementation |
| Measurement | `measurement.h` | Projective and generalized measurements |
| Entanglement | `entanglement.h` | Entropy, purity, fidelity calculations |
| Noise | `noise.h` | Decoherence and error channel simulation |

### Algorithms

| Module | C Header | Description |
|--------|----------|-------------|
| Grover | `grover.h` | Grover's search algorithm |
| VQE | `vqe.h` | Variational Quantum Eigensolver |
| QAOA | `qaoa.h` | Quantum Approximate Optimization |
| QPE | `qpe.h` | Quantum Phase Estimation |
| DMRG | `dmrg.h` | Density Matrix Renormalization Group |
| Tensor | `tensor.h` | Tensor network operations |

### Optimization

| Module | C Header | Description |
|--------|----------|-------------|
| SIMD | `simd_ops.h` | AVX-512/NEON vectorized operations |
| GPU | `gpu_metal.h` | Metal GPU acceleration |
| Parallel | `parallel_ops.h` | OpenMP parallelization |

### Distributed Computing

| Module | C Header | Description |
|--------|----------|-------------|
| MPI Bridge | `mpi_bridge.h` | MPI communication interface |
| Collective | `collective_ops.h` | Distributed operations |

## Quick Start by Language

### C

```c
#include "src/quantum/state.h"
#include "src/quantum/gates.h"

int main(void) {
    quantum_state_t state;
    quantum_state_init(&state, 2);

    gate_hadamard(&state, 0);
    gate_cnot(&state, 0, 1);

    double p00 = quantum_state_get_probability(&state, 0);
    printf("P(|00⟩) = %.4f\n", p00);

    quantum_state_free(&state);
    return 0;
}
```

### Python

```python
from moonlab import QuantumState

state = QuantumState(2)
state.h(0).cnot(0, 1)

probs = state.probabilities()
print(f"P(|00⟩) = {probs[0]:.4f}")
```

### Rust

```rust
use moonlab::QuantumState;

fn main() {
    let mut state = QuantumState::new(2);
    state.h(0).cnot(0, 1);

    let probs = state.probabilities();
    println!("P(|00⟩) = {:.4}", probs[0]);
}
```

### JavaScript

```javascript
import { QuantumState } from '@moonlab/core';

const state = new QuantumState(2);
state.h(0).cnot(0, 1);

const probs = state.probabilities();
console.log(`P(|00⟩) = ${probs[0].toFixed(4)}`);
```

## Error Handling

### C Error Codes

```c
typedef enum {
    QS_SUCCESS = 0,
    QS_ERROR_INVALID_QUBIT = -1,
    QS_ERROR_INVALID_STATE = -2,
    QS_ERROR_NOT_NORMALIZED = -3,
    QS_ERROR_OUT_OF_MEMORY = -4,
    QS_ERROR_INVALID_DIMENSION = -5
} qs_error_t;
```

### Language-Specific Error Handling

| Language | Approach |
|----------|----------|
| C | Return codes, check after each call |
| Python | Exceptions (`QuantumError`) |
| Rust | `Result<T, QuantumError>` |
| JavaScript | Promises with rejection / throw |

## Performance Considerations

### Memory Requirements

State vector simulation requires $2^n \times 16$ bytes for $n$ qubits (double-precision complex):

| Qubits | Memory |
|--------|--------|
| 20 | 16 MB |
| 25 | 512 MB |
| 28 | 4 GB |
| 30 | 16 GB |
| 32 | 64 GB |

### Optimization Levels

| Level | Description |
|-------|-------------|
| Baseline | Pure C, no acceleration |
| SIMD | AVX-512 (x86) or NEON (ARM) |
| OpenMP | Multi-core parallelization |
| GPU | Metal (macOS) / CUDA / Vulkan |
| Distributed | MPI across nodes |

### Language Performance Comparison

| Language | Relative Performance | Best Use Case |
|----------|---------------------|---------------|
| C | 1.0x (baseline) | Maximum performance |
| Rust | ~1.0x | Safe high performance |
| Python (FFI) | ~0.95x | Convenience + speed |
| JavaScript (WASM) | ~0.7x | Web applications |

## Version Compatibility

| Moonlab Version | C Standard | Python | Rust | Node.js |
|-----------------|------------|--------|------|---------|
| 1.0.x | C11 | 3.8+ | 1.70+ | 18+ |

## API Stability

| Stability Level | Description | Example |
|-----------------|-------------|---------|
| Stable | Will not change in minor versions | `quantum_state_init` |
| Experimental | May change | DMRG, QAOA |
| Internal | Not for public use | `_internal_*` |

## Documentation Conventions

### Function Signatures

C functions are documented with:
- Parameters (name, type, description)
- Return value
- Error conditions
- Example usage
- Performance notes

### Type Documentation

Structs and enums include:
- Field descriptions
- Memory layout notes
- Thread safety information

## Getting Help

- [FAQ](../faq.md) - Frequently asked questions
- [Troubleshooting](../troubleshooting.md) - Common issues
- [GitHub Issues](https://github.com/tsotchke/moonlab/issues) - Bug reports
