# C API Reference

The C library is the core of Moonlab Quantum Simulator, providing maximum performance and direct hardware access. All language bindings are built on top of this library.

## Overview

The C API is organized into modules:

| Module | Header | Description |
|--------|--------|-------------|
| [Quantum State](quantum-state.md) | `state.h` | State vector management |
| [Gates](gates.md) | `gates.h` | Universal gate set |
| [Measurement](measurement.md) | `measurement.h` | Quantum measurement |
| [Entanglement](entanglement.md) | `entanglement.h` | Entropy and fidelity |
| [Noise](noise.md) | `noise.h` | Decoherence simulation |
| [Grover](grover.md) | `grover.h` | Grover's algorithm |
| [VQE](vqe.md) | `vqe.h` | Variational eigensolver |
| [QAOA](qaoa.md) | `qaoa.h` | Quantum optimization |
| [QPE](qpe.md) | `qpe.h` | Phase estimation |
| [Tensor Network](tensor-network.md) | `tensor.h`, `dmrg.h` | MPS and DMRG |
| [Topological](topological.md) | `topological.h` | Anyon models, surface codes |
| [Skyrmion Braiding](skyrmion-braiding.md) | `skyrmion_braiding.h` | Topological qubits |
| [GPU Metal](gpu-metal.md) | `gpu_metal.h` | Metal acceleration |
| [SIMD Ops](simd-ops.md) | `simd_ops.h` | Vectorization |
| [MPI Bridge](mpi-bridge.md) | `mpi_bridge.h` | Distributed computing |
| [Entropy](entropy.md) | `entropy.h` | Random number generation |
| [Config](config.md) | `config.h` | Configuration |

## Quick Start

```c
#include "src/quantum/state.h"
#include "src/quantum/gates.h"
#include "src/quantum/measurement.h"
#include "src/quantum/entanglement.h"
#include <stdio.h>

int main(void) {
    // Create 2-qubit state |00⟩
    quantum_state_t state;
    qs_error_t err = quantum_state_init(&state, 2);
    if (err != QS_SUCCESS) {
        fprintf(stderr, "Failed to initialize state\n");
        return 1;
    }

    // Create Bell state: H on qubit 0, then CNOT(0, 1)
    gate_hadamard(&state, 0);
    gate_cnot(&state, 0, 1);

    // Check probabilities
    double p00 = quantum_state_get_probability(&state, 0);
    double p11 = quantum_state_get_probability(&state, 3);
    printf("P(|00⟩) = %.4f, P(|11⟩) = %.4f\n", p00, p11);

    // Measure entanglement entropy
    int subsystem[] = {0};
    double entropy = quantum_state_entanglement_entropy(&state, subsystem, 1);
    printf("Entanglement entropy: %.4f bits\n", entropy);

    // Cleanup
    quantum_state_free(&state);
    return 0;
}
```

## Core Types

### quantum_state_t

The primary data structure representing a quantum state:

```c
typedef struct {
    size_t num_qubits;              // Number of qubits (1-32)
    size_t state_dim;               // 2^num_qubits
    complex_t *amplitudes;          // State vector coefficients

    // Quantum properties (cached)
    double global_phase;            // Global phase factor
    double entanglement_entropy;    // Von Neumann entropy
    double purity;                  // Tr(ρ²)
    double fidelity;                // State fidelity

    // Measurement history
    uint64_t *measurement_outcomes;
    size_t num_measurements;
    size_t max_measurements;

    // Memory management
    int owns_memory;                // 1 if we allocated amplitudes
} quantum_state_t;
```

### complex_t

Complex number type (C99 `double _Complex`):

```c
typedef double _Complex complex_t;
```

### qs_error_t

Error codes returned by most functions:

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

## State Management Functions

### quantum_state_init

Initialize a quantum state in $|0\cdots0\rangle$:

```c
qs_error_t quantum_state_init(quantum_state_t *state, size_t num_qubits);
```

**Parameters**:
- `state`: Pointer to uninitialized quantum_state_t
- `num_qubits`: Number of qubits (1-32)

**Returns**: `QS_SUCCESS` or error code

**Memory**: Allocates $2^n \times 16$ bytes

**Example**:
```c
quantum_state_t state;
if (quantum_state_init(&state, 10) != QS_SUCCESS) {
    // Handle error
}
```

### quantum_state_free

Release quantum state resources:

```c
void quantum_state_free(quantum_state_t *state);
```

**Important**: Always call this to avoid memory leaks.

### quantum_state_clone

Create a copy of a quantum state:

```c
qs_error_t quantum_state_clone(quantum_state_t *dest, const quantum_state_t *src);
```

### quantum_state_reset

Reset state to $|0\cdots0\rangle$:

```c
void quantum_state_reset(quantum_state_t *state);
```

## Gate Functions

All gates return `qs_error_t` and take the state and target qubit(s):

### Single-Qubit Gates

```c
qs_error_t gate_pauli_x(quantum_state_t *state, int qubit);
qs_error_t gate_pauli_y(quantum_state_t *state, int qubit);
qs_error_t gate_pauli_z(quantum_state_t *state, int qubit);
qs_error_t gate_hadamard(quantum_state_t *state, int qubit);
qs_error_t gate_s(quantum_state_t *state, int qubit);
qs_error_t gate_s_dagger(quantum_state_t *state, int qubit);
qs_error_t gate_t(quantum_state_t *state, int qubit);
qs_error_t gate_t_dagger(quantum_state_t *state, int qubit);
```

### Rotation Gates

```c
qs_error_t gate_rx(quantum_state_t *state, int qubit, double theta);
qs_error_t gate_ry(quantum_state_t *state, int qubit, double theta);
qs_error_t gate_rz(quantum_state_t *state, int qubit, double theta);
qs_error_t gate_phase(quantum_state_t *state, int qubit, double theta);
```

**Parameters**:
- `theta`: Rotation angle in radians

### Two-Qubit Gates

```c
qs_error_t gate_cnot(quantum_state_t *state, int control, int target);
qs_error_t gate_cz(quantum_state_t *state, int control, int target);
qs_error_t gate_swap(quantum_state_t *state, int qubit1, int qubit2);
qs_error_t gate_cphase(quantum_state_t *state, int control, int target, double theta);
```

### Three-Qubit Gates

```c
qs_error_t gate_toffoli(quantum_state_t *state, int control1, int control2, int target);
qs_error_t gate_fredkin(quantum_state_t *state, int control, int target1, int target2);
```

## Measurement Functions

### quantum_state_get_probability

Get probability of measuring a specific basis state:

```c
double quantum_state_get_probability(const quantum_state_t *state, uint64_t basis_state);
```

**Example**:
```c
double p_zero = quantum_state_get_probability(&state, 0);  // P(|00...0⟩)
```

### quantum_measure_all

Measure all qubits:

```c
qs_error_t quantum_measure_all(
    quantum_state_t *state,
    measurement_result_t *result,
    quantum_entropy_ctx_t *entropy
);
```

**Note**: Collapses the state to the measured outcome.

## Entanglement Functions

### quantum_state_entanglement_entropy

Calculate von Neumann entropy of a subsystem:

```c
double quantum_state_entanglement_entropy(
    const quantum_state_t *state,
    const int *subsystem_qubits,
    size_t num_subsystem_qubits
);
```

**Example**:
```c
int qubits_a[] = {0, 1};
double S = quantum_state_entanglement_entropy(&state, qubits_a, 2);
```

## Compilation

### Basic Build

```bash
make
```

### Linking

```bash
gcc -O3 your_program.c -L. -lquantumsim -lm -o your_program
```

### Include Paths

```c
// Include from project root
#include "src/quantum/state.h"
#include "src/quantum/gates.h"

// Or set include path: -I/path/to/quantum-simulator/src
#include "quantum/state.h"
```

## Thread Safety

- **State operations**: NOT thread-safe. Each thread should have its own state.
- **Multiple states**: Different states can be manipulated by different threads.
- **Read-only operations**: Thread-safe for concurrent reads.

## Memory Alignment

Moonlab uses 64-byte aligned allocations for optimal SIMD and AMX performance:

```c
// Internal: allocates aligned memory
state->amplitudes = aligned_alloc(64, state_dim * sizeof(complex_t));
```

## Performance Tips

1. **Reuse states**: `quantum_state_reset` is faster than `init` + `free`
2. **Batch operations**: Group gates before measurement
3. **Qubit ordering**: Operations on adjacent qubits may be faster
4. **Memory locality**: Keep hot data in cache

## Error Handling Pattern

```c
qs_error_t err;

err = quantum_state_init(&state, num_qubits);
if (err != QS_SUCCESS) {
    handle_error(err);
    return err;
}

err = gate_hadamard(&state, 0);
if (err != QS_SUCCESS) {
    quantum_state_free(&state);
    return err;
}

// ... more operations ...

quantum_state_free(&state);
return QS_SUCCESS;
```

## Platform-Specific Features

### macOS / Apple Silicon

```c
#include "src/optimization/gpu_metal.h"

// Enable Metal GPU acceleration
metal_init();
metal_gate_hadamard(&state, qubit);
metal_cleanup();
```

### Linux with OpenMP

```c
// Automatically uses OpenMP if compiled with -fopenmp
// Set thread count:
export OMP_NUM_THREADS=8
```

## See Also

- [Quantum State](quantum-state.md) - Full state management reference
- [Gates](gates.md) - Complete gate documentation
- [Performance Tuning](../../guides/performance-tuning.md) - Optimization guide
