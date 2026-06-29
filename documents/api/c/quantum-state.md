# Archived Moonlab Documentation: Quantum State API

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Quantum State API

Complete reference for quantum state management in the C library.

**Header**: `src/quantum/state.h`

## Overview

The quantum state module provides functions for creating, manipulating, and querying quantum states. States are represented as complex amplitude vectors in the computational basis.

## Types

### quantum_state_t

The primary data structure representing a quantum state vector.

[archived fence delimiter: ```c]
typedef struct {
    size_t num_qubits;              // Number of qubits (1-32)
    size_t state_dim;               // State space dimension (2^num_qubits)
    complex_t *amplitudes;          // Complex amplitude array

    // Cached quantum properties
    double global_phase;            // Global phase factor
    double entanglement_entropy;    // Von Neumann entropy (cached)
    double purity;                  // Tr(ρ²) (cached)
    double fidelity;                // Fidelity with reference (cached)

    // Measurement history
    uint64_t *measurement_outcomes; // Recorded outcomes
    size_t num_measurements;        // Number of measurements
    size_t max_measurements;        // History capacity

    // Memory management
    int owns_memory;                // 1 if amplitudes allocated internally
} quantum_state_t;
[archived fence delimiter: ```]

### complex_t

Complex number type using C99 `_Complex`:

[archived fence delimiter: ```c]
typedef double _Complex complex_t;
[archived fence delimiter: ```]

### qs_error_t

Error codes returned by state functions:

[archived fence delimiter: ```c]
typedef enum {
    QS_SUCCESS = 0,               // Operation successful
    QS_ERROR_INVALID_QUBIT = -1,  // Qubit index out of range
    QS_ERROR_INVALID_STATE = -2,  // Invalid state (NULL or uninitialized)
    QS_ERROR_NOT_NORMALIZED = -3, // State not normalized
    QS_ERROR_OUT_OF_MEMORY = -4,  // Memory allocation failed
    QS_ERROR_INVALID_DIMENSION = -5 // Dimension mismatch
} qs_error_t;
[archived fence delimiter: ```]

## Constants

[archived fence delimiter: ```c]
#define MAX_QUBITS 32                      // Maximum supported qubits
#define MAX_STATE_DIM (1ULL << MAX_QUBITS) // 2^32 = 4,294,967,296
#define RECOMMENDED_MAX_QUBITS 28          // Optimal for performance/memory
[archived fence delimiter: ```]

## State Creation and Destruction

### quantum_state_init

Initialize a quantum state to $|0\cdots0\rangle$.

[archived fence delimiter: ```c]
qs_error_t quantum_state_init(quantum_state_t *state, size_t num_qubits);
[archived fence delimiter: ```]

**Parameters**:
- `state`: Pointer to uninitialized `quantum_state_t` structure
- `num_qubits`: Number of qubits (1 to 32)

**Returns**: `QS_SUCCESS` or error code

**Memory**: Allocates `2^num_qubits × 16` bytes with 64-byte alignment for SIMD/AMX optimization.

**Memory Requirements by Qubit Count**:

| Qubits | States | Memory |
|--------|--------|--------|
| 10 | 1,024 | 16 KB |
| 15 | 32,768 | 512 KB |
| 20 | 1,048,576 | 16 MB |
| 25 | 33,554,432 | 512 MB |
| 28 | 268,435,456 | 4.3 GB |
| 30 | 1,073,741,824 | 17.2 GB |
| 32 | 4,294,967,296 | 68.7 GB |

**Example**:
[archived fence delimiter: ```c]
quantum_state_t state;
qs_error_t err = quantum_state_init(&state, 4);

if (err != QS_SUCCESS) {
    if (err == QS_ERROR_OUT_OF_MEMORY) {
        fprintf(stderr, "Not enough memory for %d qubits\n", 4);
    }
    return err;
}

// State is now |0000⟩
// ... use state ...

quantum_state_free(&state);
[archived fence delimiter: ```]

### quantum_state_free

Release all resources associated with a quantum state.

[archived fence delimiter: ```c]
void quantum_state_free(quantum_state_t *state);
[archived fence delimiter: ```]

**Parameters**:
- `state`: State to free

**Notes**:
- Safe to call on already-freed or zero-initialized states
- Sets `amplitudes` to NULL and `owns_memory` to 0

**Example**:
[archived fence delimiter: ```c]
quantum_state_t state;
quantum_state_init(&state, 10);
// ... use state ...
quantum_state_free(&state);  // Always call when done
[archived fence delimiter: ```]

### quantum_state_create

Allocate and initialize a quantum state on the heap.

[archived fence delimiter: ```c]
quantum_state_t* quantum_state_init(int num_qubits);
[archived fence delimiter: ```]

**Parameters**:
- `num_qubits`: Number of qubits

**Returns**: Pointer to new state, or `NULL` on error

**Notes**: Caller must call `quantum_state_free()` when finished.

**Example**:
[archived fence delimiter: ```c]
quantum_state_t *state = quantum_state_init(8);
if (state == NULL) {
    // Handle error
}
// ... use state ...
quantum_state_free(state);
[archived fence delimiter: ```]

### quantum_state_destroy

Free a heap-allocated quantum state.

[archived fence delimiter: ```c]
void quantum_state_free(quantum_state_t *state);
[archived fence delimiter: ```]

**Parameters**:
- `state`: State to destroy (safe to pass NULL)

### quantum_state_from_amplitudes

Initialize a state from a custom amplitude vector.

[archived fence delimiter: ```c]
qs_error_t quantum_state_from_amplitudes(
    quantum_state_t *state,
    const complex_t *amplitudes,
    size_t dim
);
[archived fence delimiter: ```]

**Parameters**:
- `state`: State structure to initialize
- `amplitudes`: Array of complex amplitudes
- `dim`: Dimension (must be power of 2)

**Returns**: `QS_SUCCESS` or error code

**Notes**:
- Input amplitudes are copied
- Normalization is NOT verified; call `quantum_state_normalize()` if needed

**Example**:
[archived fence delimiter: ```c]
// Create Bell state directly
complex_t bell[4] = {
    1.0/sqrt(2.0) + 0.0*I,  // |00⟩
    0.0 + 0.0*I,             // |01⟩
    0.0 + 0.0*I,             // |10⟩
    1.0/sqrt(2.0) + 0.0*I   // |11⟩
};

quantum_state_t state;
quantum_state_from_amplitudes(&state, bell, 4);
[archived fence delimiter: ```]

### quantum_state_clone

Create a deep copy of a quantum state.

[archived fence delimiter: ```c]
qs_error_t quantum_state_clone(quantum_state_t *dest, const quantum_state_t *src);
[archived fence delimiter: ```]

**Parameters**:
- `dest`: Destination (uninitialized)
- `src`: Source state

**Returns**: `QS_SUCCESS` or error code

**Example**:
[archived fence delimiter: ```c]
quantum_state_t original, copy;
quantum_state_init(&original, 4);
gate_hadamard(&original, 0);

quantum_state_clone(&copy, &original);

// copy is independent of original
gate_pauli_x(&copy, 1);  // Doesn't affect original

quantum_state_free(&original);
quantum_state_free(&copy);
[archived fence delimiter: ```]

### quantum_state_reset

Reset a state to $|0\cdots0\rangle$ without reallocation.

[archived fence delimiter: ```c]
void quantum_state_reset(quantum_state_t *state);
[archived fence delimiter: ```]

**Parameters**:
- `state`: State to reset

**Notes**: More efficient than free+init for repeated simulations.

**Example**:
[archived fence delimiter: ```c]
quantum_state_t state;
quantum_state_init(&state, 10);

for (int trial = 0; trial < 1000; trial++) {
    quantum_state_reset(&state);  // Fast reset
    // ... run circuit ...
}

quantum_state_free(&state);
[archived fence delimiter: ```]

## State Properties

### quantum_state_is_normalized

Check if state satisfies normalization condition.

[archived fence delimiter: ```c]
int quantum_state_is_normalized(const quantum_state_t *state, double tolerance);
[archived fence delimiter: ```]

**Parameters**:
- `state`: State to check
- `tolerance`: Maximum deviation from 1.0 (e.g., 1e-10)

**Returns**: 1 if normalized, 0 otherwise

**Example**:
[archived fence delimiter: ```c]
if (!quantum_state_is_normalized(&state, 1e-10)) {
    quantum_state_normalize(&state);
}
[archived fence delimiter: ```]

### quantum_state_normalize

Normalize state vector to unit length.

[archived fence delimiter: ```c]
qs_error_t quantum_state_normalize(quantum_state_t *state);
[archived fence delimiter: ```]

**Returns**: `QS_SUCCESS`, or `QS_ERROR_NOT_NORMALIZED` if norm is zero

### quantum_state_get_amplitude

Get the complex amplitude for a basis state.

[archived fence delimiter: ```c]
complex_t quantum_state_get_amplitude(const quantum_state_t *state, uint64_t basis_index);
[archived fence delimiter: ```]

**Parameters**:
- `state`: Quantum state
- `basis_index`: Index of basis state (0 to $2^n - 1$)

**Returns**: Complex amplitude $\alpha_i$

**Example**:
[archived fence delimiter: ```c]
// Get amplitude of |0101⟩ = |5⟩
complex_t amp = quantum_state_get_amplitude(&state, 5);
printf("α₅ = %.4f + %.4fi\n", creal(amp), cimag(amp));
[archived fence delimiter: ```]

### quantum_state_get_probability

Get the probability of measuring a basis state.

[archived fence delimiter: ```c]
double quantum_state_get_probability(const quantum_state_t *state, uint64_t basis_index);
[archived fence delimiter: ```]

**Parameters**:
- `state`: Quantum state
- `basis_index`: Index of basis state

**Returns**: Probability $|\alpha_i|^2$

**Example**:
[archived fence delimiter: ```c]
// Check Bell state probabilities
quantum_state_t state;
quantum_state_init(&state, 2);
gate_hadamard(&state, 0);
gate_cnot(&state, 0, 1);

double p00 = quantum_state_get_probability(&state, 0);  // |00⟩
double p11 = quantum_state_get_probability(&state, 3);  // |11⟩

printf("P(|00⟩) = %.4f\n", p00);  // 0.5
printf("P(|11⟩) = %.4f\n", p11);  // 0.5
[archived fence delimiter: ```]

### quantum_state_entropy

Calculate von Neumann entropy of the state.

[archived fence delimiter: ```c]
double quantum_state_entropy(const quantum_state_t *state);
[archived fence delimiter: ```]

**Returns**: Entropy in bits: $S = -\sum_i |\alpha_i|^2 \log_2 |\alpha_i|^2$

**Notes**: For pure states, this is the Shannon entropy of the measurement distribution, not entanglement entropy.

### quantum_state_purity

Calculate state purity $\text{Tr}(\rho^2)$.

[archived fence delimiter: ```c]
double quantum_state_purity(const quantum_state_t *state);
[archived fence delimiter: ```]

**Returns**: Purity (1.0 for pure states)

### quantum_state_fidelity

Calculate fidelity between two states.

[archived fence delimiter: ```c]
double quantum_state_fidelity(const quantum_state_t *state1, const quantum_state_t *state2);
[archived fence delimiter: ```]

**Returns**: $F = |\langle\psi_1|\psi_2\rangle|^2$ (0 to 1)

**Example**:
[archived fence delimiter: ```c]
quantum_state_t ideal, noisy;
// ... prepare states ...

double f = quantum_state_fidelity(&ideal, &noisy);
printf("Fidelity: %.6f\n", f);

if (f > 0.99) {
    printf("States are nearly identical\n");
}
[archived fence delimiter: ```]

## Entanglement Measures

### quantum_state_entanglement_entropy

Calculate entanglement entropy between subsystems.

[archived fence delimiter: ```c]
double quantum_state_entanglement_entropy(
    const quantum_state_t *state,
    const int *qubits_subsystem_a,
    size_t num_qubits_a
);
[archived fence delimiter: ```]

**Parameters**:
- `state`: Full quantum state
- `qubits_subsystem_a`: Array of qubit indices in subsystem A
- `num_qubits_a`: Number of qubits in A

**Returns**: Von Neumann entropy $S(\rho_A)$ in bits

**Notes**:
- Computes reduced density matrix $\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|)$
- For pure bipartite states, $S(\rho_A) = S(\rho_B)$
- $S = 0$ for product states, $S = 1$ for Bell states

**Example**:
[archived fence delimiter: ```c]
// Create Bell state
quantum_state_t state;
quantum_state_init(&state, 2);
gate_hadamard(&state, 0);
gate_cnot(&state, 0, 1);

// Measure entanglement between qubit 0 and qubit 1
int subsystem_a[] = {0};
double S = quantum_state_entanglement_entropy(&state, subsystem_a, 1);

printf("Entanglement entropy: %.4f bits\n", S);  // 1.0000
[archived fence delimiter: ```]

### quantum_state_partial_trace

Compute reduced density matrix by partial trace.

[archived fence delimiter: ```c]
qs_error_t quantum_state_partial_trace(
    const quantum_state_t *state,
    const int *qubits_to_trace,
    size_t num_traced,
    complex_t *reduced_density
);
[archived fence delimiter: ```]

**Parameters**:
- `state`: Full state
- `qubits_to_trace`: Qubits to trace out
- `num_traced`: Number of qubits to trace
- `reduced_density`: Output matrix (must be pre-allocated)

**Returns**: `QS_SUCCESS` or error code

**Output size**: $(2^{n-k}) \times (2^{n-k})$ where $n$ is total qubits, $k$ is traced qubits.

## Measurement History

### quantum_state_record_measurement

Record a measurement outcome for later analysis.

[archived fence delimiter: ```c]
void quantum_state_record_measurement(quantum_state_t *state, uint64_t outcome);
[archived fence delimiter: ```]

### quantum_state_get_measurement_history

Retrieve recorded measurement outcomes.

[archived fence delimiter: ```c]
size_t quantum_state_get_measurement_history(
    const quantum_state_t *state,
    uint64_t *outcomes,
    size_t max_outcomes
);
[archived fence delimiter: ```]

**Returns**: Number of outcomes retrieved

### quantum_state_clear_measurements

Clear measurement history.

[archived fence delimiter: ```c]
void quantum_state_clear_measurements(quantum_state_t *state);
[archived fence delimiter: ```]

## Utility Functions

### quantum_state_print

Print state amplitudes for debugging.

[archived fence delimiter: ```c]
void quantum_state_print(const quantum_state_t *state, size_t max_terms);
[archived fence delimiter: ```]

**Parameters**:
- `state`: State to print
- `max_terms`: Maximum number of non-zero terms to display

**Example**:
[archived fence delimiter: ```c]
quantum_state_t state;
quantum_state_init(&state, 3);
gate_hadamard(&state, 0);
gate_hadamard(&state, 1);
gate_hadamard(&state, 2);

quantum_state_print(&state, 10);
// Output: |000⟩: 0.3536, |001⟩: 0.3536, ...
[archived fence delimiter: ```]

### quantum_basis_state_string

Convert basis state index to binary string.

[archived fence delimiter: ```c]
void quantum_basis_state_string(
    uint64_t basis_index,
    size_t num_qubits,
    char *buffer,
    size_t buffer_size
);
[archived fence delimiter: ```]

**Example**:
[archived fence delimiter: ```c]
char buf[33];
quantum_basis_state_string(5, 4, buf, sizeof(buf));
printf("|%s⟩\n", buf);  // |0101⟩
[archived fence delimiter: ```]

## Thread Safety

- **Not thread-safe**: State operations are not atomic
- **Multiple states**: Different states can be used by different threads
- **Read-only operations**: `get_probability`, `get_amplitude`, etc. are safe for concurrent reads

## Memory Layout

Amplitudes are stored in computational basis order:

[archived fence delimiter: ```]
Index 0: |00...0⟩
Index 1: |00...1⟩
Index 2: |00..10⟩
...
Index 2^n - 1: |11...1⟩
[archived fence delimiter: ```]

Qubit 0 is the least significant bit.

## Performance Tips

1. **Reuse states**: `reset()` is faster than `free()` + `init()`
2. **Aligned access**: Amplitudes are 64-byte aligned for SIMD
3. **Batch operations**: Apply multiple gates before reading properties
4. **Memory locality**: Sequential amplitude access is cache-friendly

## See Also

- [Gates API](gates.md) - Apply gates to states
- [Measurement API](measurement.md) - Measure states
- [Entanglement API](entanglement.md) - Entropy calculations
```
