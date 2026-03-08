# Measurement API

Complete reference for quantum measurement operations in the C library.

**Header**: `src/quantum/measurement.h`

## Overview

The measurement module provides operations for extracting classical information from quantum states, including:

- Projective measurements (with collapse)
- Non-destructive probability queries
- Expectation value calculations
- Statistical sampling

## Probability Computation

### measurement_probability_one

Compute probability of measuring a qubit in $|1\rangle$ state.

```c
double measurement_probability_one(const quantum_state_t *state, int qubit);
```

**Parameters**:
- `state`: Quantum state (unchanged)
- `qubit`: Qubit index

**Returns**: $P(|1\rangle)$ in range $[0, 1]$

**Example**:
```c
quantum_state_t state;
quantum_state_init(&state, 2);
gate_hadamard(&state, 0);

double p1 = measurement_probability_one(&state, 0);
printf("P(qubit 0 = 1) = %.4f\n", p1);  // 0.5000
```

### measurement_probability_zero

Compute probability of measuring a qubit in $|0\rangle$ state.

```c
double measurement_probability_zero(const quantum_state_t *state, int qubit);
```

**Returns**: $P(|0\rangle) = 1 - P(|1\rangle)$

### measurement_all_probabilities

Compute $P(|1\rangle)$ for all qubits.

```c
void measurement_all_probabilities(const quantum_state_t *state, double *probabilities);
```

**Parameters**:
- `state`: Quantum state
- `probabilities`: Output array of size `num_qubits`

**Example**:
```c
double probs[4];
measurement_all_probabilities(&state, probs);
for (int q = 0; q < 4; q++) {
    printf("P(qubit %d = 1) = %.4f\n", q, probs[q]);
}
```

### measurement_probability_distribution

Compute full probability distribution over all basis states.

```c
void measurement_probability_distribution(const quantum_state_t *state, double *distribution);
```

**Parameters**:
- `state`: Quantum state
- `distribution`: Output array of size $2^n$

**Example**:
```c
double *dist = malloc(state.state_dim * sizeof(double));
measurement_probability_distribution(&state, dist);

for (size_t i = 0; i < state.state_dim; i++) {
    if (dist[i] > 0.001) {
        printf("P(|%llu⟩) = %.4f\n", i, dist[i]);
    }
}
free(dist);
```

## Single-Qubit Measurement

### measurement_single_qubit

Measure a single qubit with state collapse.

```c
int measurement_single_qubit(quantum_state_t *state, int qubit, double random_value);
```

**Parameters**:
- `state`: Quantum state (modified)
- `qubit`: Qubit to measure
- `random_value`: Random number in $[0, 1)$ for outcome selection

**Returns**: Measurement result (0 or 1), -1 on error

**Behavior**:
1. Computes $P(|1\rangle)$ for the qubit
2. If `random_value < P(|1\rangle)`, outcome is 1; otherwise 0
3. Collapses state to be consistent with outcome
4. Renormalizes state

**Example**:
```c
#include <stdlib.h>

quantum_state_t state;
quantum_state_init(&state, 2);
gate_hadamard(&state, 0);
gate_cnot(&state, 0, 1);  // Bell state

// Measure qubit 0
double r = (double)rand() / RAND_MAX;
int result = measurement_single_qubit(&state, 0, r);
printf("Qubit 0 measured: %d\n", result);

// After measurement, qubit 1 is perfectly correlated
double p1 = measurement_probability_one(&state, 1);
printf("P(qubit 1 = %d) = %.4f\n", result, p1);  // 1.0000
```

### measurement_single_qubit_no_collapse

Measure a qubit without modifying state (simulation only).

```c
int measurement_single_qubit_no_collapse(const quantum_state_t *state, int qubit, double random_value);
```

**Notes**: Useful for sampling statistics without disrupting the state.

## Multi-Qubit Measurement

### measurement_all_qubits

Measure all qubits simultaneously.

```c
uint64_t measurement_all_qubits(quantum_state_t *state, double random_value);
```

**Parameters**:
- `state`: Quantum state (collapses to basis state)
- `random_value`: Random number in $[0, 1)$

**Returns**: Measured basis state as bit pattern

**Behavior**:
1. Samples from full probability distribution
2. Collapses state to measured basis state
3. All amplitudes become 0 except the measured one (which becomes 1)

**Example**:
```c
quantum_state_t state;
quantum_state_init(&state, 4);
for (int q = 0; q < 4; q++) {
    gate_hadamard(&state, q);
}

double r = (double)rand() / RAND_MAX;
uint64_t result = measurement_all_qubits(&state, r);
printf("Measured: |%04llx⟩ = |", result);
for (int q = 3; q >= 0; q--) {
    printf("%d", (int)((result >> q) & 1));
}
printf("⟩\n");
```

### measurement_partial

Measure a subset of qubits.

```c
uint64_t measurement_partial(
    quantum_state_t *state,
    const int *qubits,
    int num_measure,
    double random_value
);
```

**Parameters**:
- `state`: Quantum state (partially collapses)
- `qubits`: Array of qubit indices to measure
- `num_measure`: Number of qubits to measure
- `random_value`: Random number for sampling

**Returns**: Measurement outcomes as bit pattern (LSB = first qubit in array)

**Example**:
```c
// Measure only qubits 0 and 2
int to_measure[] = {0, 2};
uint64_t result = measurement_partial(&state, to_measure, 2, r);

int q0_outcome = result & 1;
int q2_outcome = (result >> 1) & 1;
```

## Expectation Values

### measurement_expectation_z

Compute expectation value of Pauli Z operator.

```c
double measurement_expectation_z(const quantum_state_t *state, int qubit);
```

**Returns**: $\langle Z \rangle$ in range $[-1, 1]$

**Mathematical definition**:
$$\langle Z \rangle = P(|0\rangle) - P(|1\rangle)$$

**Example**:
```c
// For |0⟩: ⟨Z⟩ = 1
// For |1⟩: ⟨Z⟩ = -1
// For |+⟩: ⟨Z⟩ = 0

quantum_state_t state;
quantum_state_init(&state, 1);
printf("⟨Z⟩ for |0⟩ = %.4f\n", measurement_expectation_z(&state, 0));  // 1.0

gate_hadamard(&state, 0);
printf("⟨Z⟩ for |+⟩ = %.4f\n", measurement_expectation_z(&state, 0));  // 0.0
```

### measurement_expectation_x

Compute expectation value of Pauli X operator.

```c
double measurement_expectation_x(const quantum_state_t *state, int qubit);
```

**Returns**: $\langle X \rangle$ in range $[-1, 1]$

**Notes**: Requires computing amplitudes in X-basis.

### measurement_expectation_y

Compute expectation value of Pauli Y operator.

```c
double measurement_expectation_y(const quantum_state_t *state, int qubit);
```

**Returns**: $\langle Y \rangle$ in range $[-1, 1]$

### measurement_correlation_zz

Compute ZZ correlation between two qubits.

```c
double measurement_correlation_zz(const quantum_state_t *state, int qubit1, int qubit2);
```

**Returns**: $\langle Z_i \otimes Z_j \rangle$ in range $[-1, 1]$

**Interpretation**:
- $+1$: Qubits always have same value
- $-1$: Qubits always have opposite values
- $0$: No correlation

**Example**:
```c
// Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
quantum_state_t state;
quantum_state_init(&state, 2);
gate_hadamard(&state, 0);
gate_cnot(&state, 0, 1);

double corr = measurement_correlation_zz(&state, 0, 1);
printf("⟨Z₀Z₁⟩ = %.4f\n", corr);  // 1.0 (perfect correlation)
```

## Statistical Sampling

### measurement_sample

Sample measurement outcomes without state collapse.

```c
void measurement_sample(
    const quantum_state_t *state,
    uint64_t *outcomes,
    int num_samples,
    const double *random_values
);
```

**Parameters**:
- `state`: Quantum state (unchanged)
- `outcomes`: Output array for measured values
- `num_samples`: Number of samples to generate
- `random_values`: Array of `num_samples` random numbers in $[0, 1)$

**Example**:
```c
#define NUM_SHOTS 10000

uint64_t outcomes[NUM_SHOTS];
double randoms[NUM_SHOTS];

// Generate random values
for (int i = 0; i < NUM_SHOTS; i++) {
    randoms[i] = (double)rand() / RAND_MAX;
}

measurement_sample(&state, outcomes, NUM_SHOTS, randoms);

// Count outcomes
int counts[16] = {0};
for (int i = 0; i < NUM_SHOTS; i++) {
    counts[outcomes[i]]++;
}

for (int i = 0; i < 16; i++) {
    printf("|%d⟩: %d (%.2f%%)\n", i, counts[i], 100.0 * counts[i] / NUM_SHOTS);
}
```

### measurement_estimate_probabilities

Estimate probabilities from sample data.

```c
void measurement_estimate_probabilities(
    const uint64_t *samples,
    int num_samples,
    uint64_t state_dim,
    double *probabilities
);
```

**Parameters**:
- `samples`: Array of measurement outcomes
- `num_samples`: Number of samples
- `state_dim`: Dimension of state space ($2^n$)
- `probabilities`: Output probability estimates

## Fast Measurement

### quantum_measure_all_fast

Performance-optimized full measurement.

```c
uint64_t quantum_measure_all_fast(
    quantum_state_t *state,
    quantum_entropy_ctx_t *entropy
);
```

**Parameters**:
- `state`: Quantum state (collapses)
- `entropy`: Cryptographically secure entropy source

**Returns**: Measured basis state index

**Performance**: 8× faster than measuring qubits individually for 8-qubit systems.

**Example**:
```c
#include "src/utils/quantum_entropy.h"

quantum_entropy_ctx_t entropy;
quantum_entropy_init(&entropy, my_entropy_callback, NULL);

quantum_state_t state;
quantum_state_init(&state, 8);
// ... build circuit ...

uint64_t result = quantum_measure_all_fast(&state, &entropy);
printf("Result: %llu\n", result);
```

## Measurement Bases

Defined in `gates.h`:

```c
typedef enum {
    MEASURE_COMPUTATIONAL,  // Z-basis: |0⟩, |1⟩
    MEASURE_HADAMARD,       // X-basis: |+⟩, |-⟩
    MEASURE_CIRCULAR,       // Y-basis: |↻⟩, |↺⟩
    MEASURE_CUSTOM          // Custom basis
} measurement_basis_t;
```

### Non-computational basis measurement

To measure in a different basis, apply a basis-change gate first:

```c
// Measure in X-basis
gate_hadamard(&state, qubit);  // Rotate to computational basis
int result = measurement_single_qubit(&state, qubit, r);
gate_hadamard(&state, qubit);  // Rotate back
// result now corresponds to |+⟩ (0) or |-⟩ (1) in original basis
```

## Thread Safety

- **Probability queries**: Thread-safe for concurrent reads
- **Measurement with collapse**: NOT thread-safe; state is modified

## Entropy Sources

Secure measurement requires cryptographic randomness:

```c
// Using hardware entropy
#include "src/utils/quantum_entropy.h"

int entropy_callback(void *ctx, uint8_t *buffer, size_t size) {
    // Fill buffer with cryptographic random bytes
    return getentropy(buffer, size);  // Unix/macOS
}

quantum_entropy_ctx_t entropy;
quantum_entropy_init(&entropy, entropy_callback, NULL);
```

## See Also

- [Quantum State API](quantum-state.md) - State management
- [Gates API](gates.md) - Apply gates before measurement
- [Entanglement API](entanglement.md) - Post-measurement analysis
