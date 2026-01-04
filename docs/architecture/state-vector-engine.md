# State Vector Engine

Core simulation engine for quantum state vectors.

## Overview

The state vector engine is the heart of Moonlab's simulation capability. It manages the storage and manipulation of quantum state amplitudes, implementing all quantum operations through direct manipulation of the state vector.

## State Representation

### Mathematical Model

An $n$-qubit quantum state is represented as:

$$|\psi\rangle = \sum_{k=0}^{2^n-1} \alpha_k |k\rangle$$

where $\alpha_k \in \mathbb{C}$ are complex amplitudes satisfying $\sum_k |\alpha_k|^2 = 1$.

### Memory Layout

```c
typedef struct {
    Complex* amplitudes;      // Contiguous array of 2^n complex numbers
    uint32_t num_qubits;      // n
    uint64_t dim;             // 2^n
    uint32_t flags;           // State properties
    size_t alignment;         // Memory alignment
    void* gpu_buffer;         // Optional GPU mirror
} quantum_state_t;
```

### Complex Number Format

```c
typedef struct {
    double real;
    double imag;
} Complex;

// Total size: 16 bytes per amplitude
// For n qubits: 16 × 2^n bytes
```

### Bit Ordering Convention

Moonlab uses little-endian qubit ordering:

```
|q_{n-1} q_{n-2} ... q_1 q_0⟩

State index k encodes:
- Qubit 0: bit 0 of k
- Qubit 1: bit 1 of k
- Qubit i: bit i of k
```

Example for 3 qubits:
```
Index 0: |000⟩
Index 1: |001⟩  (qubit 0 = 1)
Index 2: |010⟩  (qubit 1 = 1)
Index 3: |011⟩  (qubits 0,1 = 1)
Index 4: |100⟩  (qubit 2 = 1)
Index 5: |101⟩
Index 6: |110⟩
Index 7: |111⟩
```

## State Creation

### Allocation

```c
quantum_state_t* quantum_state_create(uint32_t num_qubits) {
    // Validate input
    if (num_qubits > QSIM_MAX_QUBITS) {
        return NULL;
    }

    // Allocate structure
    quantum_state_t* state = malloc(sizeof(quantum_state_t));
    if (!state) return NULL;

    // Calculate dimensions
    state->num_qubits = num_qubits;
    state->dim = 1ULL << num_qubits;

    // Allocate amplitude array (aligned for SIMD)
    size_t alignment = qsim_optimal_alignment();
    size_t size = state->dim * sizeof(Complex);

    state->amplitudes = aligned_alloc(alignment, size);
    if (!state->amplitudes) {
        free(state);
        return NULL;
    }

    // Initialize to |0...0⟩
    memset(state->amplitudes, 0, size);
    state->amplitudes[0].real = 1.0;
    state->amplitudes[0].imag = 0.0;

    state->alignment = alignment;
    state->flags = 0;
    state->gpu_buffer = NULL;

    return state;
}
```

### Memory Requirements

| Qubits | Amplitudes | Memory |
|--------|------------|--------|
| 10 | 1,024 | 16 KB |
| 15 | 32,768 | 512 KB |
| 20 | 1,048,576 | 16 MB |
| 25 | 33,554,432 | 512 MB |
| 30 | 1,073,741,824 | 16 GB |
| 32 | 4,294,967,296 | 64 GB |

## Gate Application

### Single-Qubit Gate

For a single-qubit gate $U$ on qubit $q$:

$$|\psi'\rangle = (I^{\otimes (n-q-1)} \otimes U \otimes I^{\otimes q}) |\psi\rangle$$

Implementation iterates over pairs of amplitudes:

```c
void apply_single_qubit_gate(quantum_state_t* state,
                             uint32_t qubit,
                             const Complex U[2][2]) {
    uint64_t mask = 1ULL << qubit;
    uint64_t half_dim = state->dim >> 1;

    // Iterate over pairs of indices differing only in bit 'qubit'
    for (uint64_t i = 0; i < half_dim; i++) {
        // Construct indices
        uint64_t idx0 = insert_zero_bit(i, qubit);  // qubit bit = 0
        uint64_t idx1 = idx0 | mask;                 // qubit bit = 1

        // Get amplitudes
        Complex a0 = state->amplitudes[idx0];
        Complex a1 = state->amplitudes[idx1];

        // Apply 2x2 matrix
        // |ψ'⟩ = U |ψ⟩
        // a0' = U[0][0]*a0 + U[0][1]*a1
        // a1' = U[1][0]*a0 + U[1][1]*a1
        state->amplitudes[idx0] = complex_add(
            complex_mul(U[0][0], a0),
            complex_mul(U[0][1], a1)
        );
        state->amplitudes[idx1] = complex_add(
            complex_mul(U[1][0], a0),
            complex_mul(U[1][1], a1)
        );
    }
}

// Helper: insert 0 bit at position
uint64_t insert_zero_bit(uint64_t value, uint32_t position) {
    uint64_t low_mask = (1ULL << position) - 1;
    uint64_t high_mask = ~low_mask;
    return ((value & high_mask) << 1) | (value & low_mask);
}
```

### Two-Qubit Gate

For a two-qubit gate on qubits $q_1 < q_2$:

```c
void apply_two_qubit_gate(quantum_state_t* state,
                          uint32_t qubit1,
                          uint32_t qubit2,
                          const Complex U[4][4]) {
    // Ensure qubit1 < qubit2
    if (qubit1 > qubit2) {
        uint32_t tmp = qubit1;
        qubit1 = qubit2;
        qubit2 = tmp;
        // Also permute matrix...
    }

    uint64_t mask1 = 1ULL << qubit1;
    uint64_t mask2 = 1ULL << qubit2;
    uint64_t quarter_dim = state->dim >> 2;

    for (uint64_t i = 0; i < quarter_dim; i++) {
        // Construct 4 indices (all combinations of qubit1, qubit2)
        uint64_t base = insert_two_zero_bits(i, qubit1, qubit2);
        uint64_t idx00 = base;
        uint64_t idx01 = base | mask1;
        uint64_t idx10 = base | mask2;
        uint64_t idx11 = base | mask1 | mask2;

        // Get amplitudes
        Complex a[4] = {
            state->amplitudes[idx00],
            state->amplitudes[idx01],
            state->amplitudes[idx10],
            state->amplitudes[idx11]
        };

        // Apply 4x4 matrix
        for (int j = 0; j < 4; j++) {
            Complex sum = {0, 0};
            for (int k = 0; k < 4; k++) {
                sum = complex_add(sum, complex_mul(U[j][k], a[k]));
            }
            // Store in temporary, then copy back
        }
    }
}
```

### SIMD Optimization

Vectorized gate application:

```c
#include <arm_neon.h>  // For ARM NEON

void apply_hadamard_simd(quantum_state_t* state, uint32_t qubit) {
    const double inv_sqrt2 = 0.7071067811865476;
    float64x2_t scale = vdupq_n_f64(inv_sqrt2);

    uint64_t mask = 1ULL << qubit;
    uint64_t half_dim = state->dim >> 1;

    for (uint64_t i = 0; i < half_dim; i++) {
        uint64_t idx0 = insert_zero_bit(i, qubit);
        uint64_t idx1 = idx0 | mask;

        // Load amplitudes (real, imag pairs)
        float64x2_t a0 = vld1q_f64((double*)&state->amplitudes[idx0]);
        float64x2_t a1 = vld1q_f64((double*)&state->amplitudes[idx1]);

        // H: (a0 + a1)/√2, (a0 - a1)/√2
        float64x2_t sum = vaddq_f64(a0, a1);
        float64x2_t diff = vsubq_f64(a0, a1);

        float64x2_t new_a0 = vmulq_f64(sum, scale);
        float64x2_t new_a1 = vmulq_f64(diff, scale);

        // Store
        vst1q_f64((double*)&state->amplitudes[idx0], new_a0);
        vst1q_f64((double*)&state->amplitudes[idx1], new_a1);
    }
}
```

## Measurement

### Full Measurement

Collapse to computational basis state:

```c
uint64_t quantum_state_measure_all(quantum_state_t* state) {
    // Compute cumulative probabilities
    double* cumprob = malloc(state->dim * sizeof(double));
    cumprob[0] = complex_abs_squared(state->amplitudes[0]);

    for (uint64_t i = 1; i < state->dim; i++) {
        cumprob[i] = cumprob[i-1] + complex_abs_squared(state->amplitudes[i]);
    }

    // Normalize (should be ~1.0)
    double total = cumprob[state->dim - 1];

    // Sample random value
    double r = qsim_random_double() * total;

    // Binary search for outcome
    uint64_t result = binary_search_cumprob(cumprob, state->dim, r);

    // Collapse state
    memset(state->amplitudes, 0, state->dim * sizeof(Complex));
    state->amplitudes[result].real = 1.0;

    free(cumprob);
    return result;
}
```

### Partial Measurement

Measure subset of qubits:

```c
uint64_t quantum_state_measure_qubits(quantum_state_t* state,
                                       const uint32_t* qubits,
                                       uint32_t num_qubits) {
    // Calculate probabilities for each outcome
    uint64_t num_outcomes = 1ULL << num_qubits;
    double* probs = calloc(num_outcomes, sizeof(double));

    for (uint64_t i = 0; i < state->dim; i++) {
        // Extract measured qubit values
        uint64_t outcome = 0;
        for (uint32_t q = 0; q < num_qubits; q++) {
            if (i & (1ULL << qubits[q])) {
                outcome |= (1ULL << q);
            }
        }
        probs[outcome] += complex_abs_squared(state->amplitudes[i]);
    }

    // Sample outcome
    double r = qsim_random_double();
    double cumulative = 0;
    uint64_t result = 0;

    for (uint64_t o = 0; o < num_outcomes; o++) {
        cumulative += probs[o];
        if (r < cumulative) {
            result = o;
            break;
        }
    }

    // Collapse and renormalize
    double norm = sqrt(probs[result]);

    for (uint64_t i = 0; i < state->dim; i++) {
        uint64_t outcome = 0;
        for (uint32_t q = 0; q < num_qubits; q++) {
            if (i & (1ULL << qubits[q])) {
                outcome |= (1ULL << q);
            }
        }

        if (outcome == result) {
            state->amplitudes[i].real /= norm;
            state->amplitudes[i].imag /= norm;
        } else {
            state->amplitudes[i].real = 0;
            state->amplitudes[i].imag = 0;
        }
    }

    free(probs);
    return result;
}
```

## Expectation Values

### Pauli Z Expectation

```c
double quantum_state_expectation_z(quantum_state_t* state, uint32_t qubit) {
    double expectation = 0;
    uint64_t mask = 1ULL << qubit;

    for (uint64_t i = 0; i < state->dim; i++) {
        double prob = complex_abs_squared(state->amplitudes[i]);

        // Z eigenvalue: +1 if qubit is 0, -1 if qubit is 1
        if (i & mask) {
            expectation -= prob;
        } else {
            expectation += prob;
        }
    }

    return expectation;
}
```

### General Pauli String

```c
double quantum_state_expectation_pauli(quantum_state_t* state,
                                        const char* pauli_string) {
    // pauli_string: e.g., "XYZII" (applied to qubits 0,1,2,3,4)
    int n = strlen(pauli_string);

    double expectation = 0;

    for (uint64_t i = 0; i < state->dim; i++) {
        for (uint64_t j = 0; j < state->dim; j++) {
            // Compute ⟨j|P|i⟩ for Pauli string P
            Complex element = pauli_matrix_element(pauli_string, j, i);

            // Add contribution: conj(a_j) * element * a_i
            Complex contrib = complex_mul(
                complex_conj(state->amplitudes[j]),
                complex_mul(element, state->amplitudes[i])
            );

            expectation += contrib.real;
        }
    }

    return expectation;
}
```

## Normalization

### Check Normalization

```c
double quantum_state_norm(quantum_state_t* state) {
    double norm_sq = 0;

    #pragma omp parallel for reduction(+:norm_sq)
    for (uint64_t i = 0; i < state->dim; i++) {
        norm_sq += complex_abs_squared(state->amplitudes[i]);
    }

    return sqrt(norm_sq);
}
```

### Renormalize

```c
void quantum_state_normalize(quantum_state_t* state) {
    double norm = quantum_state_norm(state);

    if (norm < 1e-15) {
        // State is essentially zero - reset to |0⟩
        memset(state->amplitudes, 0, state->dim * sizeof(Complex));
        state->amplitudes[0].real = 1.0;
        return;
    }

    double inv_norm = 1.0 / norm;

    #pragma omp parallel for
    for (uint64_t i = 0; i < state->dim; i++) {
        state->amplitudes[i].real *= inv_norm;
        state->amplitudes[i].imag *= inv_norm;
    }
}
```

## State Operations

### Copy

```c
quantum_state_t* quantum_state_copy(const quantum_state_t* state) {
    quantum_state_t* copy = quantum_state_create(state->num_qubits);
    if (!copy) return NULL;

    memcpy(copy->amplitudes, state->amplitudes, state->dim * sizeof(Complex));
    copy->flags = state->flags;

    return copy;
}
```

### Inner Product

```c
Complex quantum_state_inner_product(const quantum_state_t* bra,
                                    const quantum_state_t* ket) {
    if (bra->dim != ket->dim) {
        return (Complex){0, 0};
    }

    Complex result = {0, 0};

    #pragma omp parallel for reduction(+:result.real, result.imag)
    for (uint64_t i = 0; i < bra->dim; i++) {
        Complex product = complex_mul(
            complex_conj(bra->amplitudes[i]),
            ket->amplitudes[i]
        );
        result.real += product.real;
        result.imag += product.imag;
    }

    return result;
}
```

### Fidelity

```c
double quantum_state_fidelity(const quantum_state_t* state1,
                              const quantum_state_t* state2) {
    Complex overlap = quantum_state_inner_product(state1, state2);
    return complex_abs_squared(overlap);
}
```

## Serialization

### Save to File

```c
int quantum_state_save(const quantum_state_t* state, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return -1;

    // Header
    uint32_t magic = 0x4D4F4F4E;  // "MOON"
    uint32_t version = 1;
    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);
    fwrite(&state->num_qubits, sizeof(uint32_t), 1, f);

    // Amplitudes
    fwrite(state->amplitudes, sizeof(Complex), state->dim, f);

    fclose(f);
    return 0;
}
```

### Load from File

```c
quantum_state_t* quantum_state_load(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    // Read header
    uint32_t magic, version, num_qubits;
    fread(&magic, sizeof(uint32_t), 1, f);
    fread(&version, sizeof(uint32_t), 1, f);
    fread(&num_qubits, sizeof(uint32_t), 1, f);

    // Validate
    if (magic != 0x4D4F4F4E) {
        fclose(f);
        return NULL;
    }

    // Create state
    quantum_state_t* state = quantum_state_create(num_qubits);
    if (!state) {
        fclose(f);
        return NULL;
    }

    // Read amplitudes
    fread(state->amplitudes, sizeof(Complex), state->dim, f);

    fclose(f);
    return state;
}
```

## Performance Characteristics

### Operation Complexity

| Operation | Time Complexity | Memory |
|-----------|----------------|--------|
| State creation | O(2^n) | O(2^n) |
| Single-qubit gate | O(2^n) | O(1) |
| Two-qubit gate | O(2^n) | O(1) |
| Full measurement | O(2^n) | O(2^n) |
| Partial measurement (k qubits) | O(2^n) | O(2^k) |
| Expectation value | O(2^n) or O(2^(2n)) | O(1) |
| Inner product | O(2^n) | O(1) |

### Parallelization

All O(2^n) operations parallelize over amplitude index:

```c
#pragma omp parallel for
for (uint64_t i = 0; i < state->dim; i++) {
    // Process amplitude i
}
```

Scaling with threads:
- Near-linear for memory-bound operations
- Sub-linear for compute-bound (cache effects)

## See Also

- [GPU Pipeline](gpu-pipeline.md) - GPU acceleration
- [System Overview](system-overview.md) - Architecture context
- [C API: State](../api/c/quantum-state.md) - Public API reference

