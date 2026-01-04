# Entanglement API

Complete reference for quantum entanglement analysis in the C library.

**Header**: `src/quantum/entanglement.h`

## Overview

The entanglement module provides functions for analyzing quantum correlations and computing entanglement measures, including:

- Reduced density matrices via partial trace
- Von Neumann and Rényi entanglement entropy
- Concurrence for two-qubit systems
- Schmidt decomposition
- Entanglement detection

## Reduced Density Matrix

### entanglement_reduced_density_matrix

Compute reduced density matrix by tracing out specified qubits.

```c
int entanglement_reduced_density_matrix(
    const quantum_state_t *state,
    const int *trace_out_qubits,
    int num_trace_out,
    complex_t *reduced_dm,
    uint64_t *reduced_dim
);
```

**Parameters**:
- `state`: Full quantum state
- `trace_out_qubits`: Array of qubit indices to trace out
- `num_trace_out`: Number of qubits to trace out
- `reduced_dm`: Output density matrix (pre-allocated)
- `reduced_dim`: Output: dimension of reduced system

**Returns**: 0 on success, -1 on error

**Mathematical Definition**:
For a bipartite state $|\psi\rangle_{AB}$, the reduced density matrix of subsystem $A$ is:
$$\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|)$$

**Example**:
```c
quantum_state_t state;
quantum_state_init(&state, 4);  // 4 qubits

// Create entangled state
gate_hadamard(&state, 0);
gate_cnot(&state, 0, 1);
gate_cnot(&state, 1, 2);
gate_cnot(&state, 2, 3);  // GHZ state

// Trace out qubits 2 and 3
int trace_out[] = {2, 3};
uint64_t reduced_dim;
complex_t *reduced_dm = malloc(4 * 4 * sizeof(complex_t));  // 2^2 × 2^2

entanglement_reduced_density_matrix(&state, trace_out, 2, reduced_dm, &reduced_dim);
// reduced_dm is now a 4×4 mixed state density matrix
```

## Entanglement Entropy

### entanglement_von_neumann_entropy

Compute von Neumann entropy of a density matrix.

```c
double entanglement_von_neumann_entropy(const complex_t *reduced_dm, uint64_t dim);
```

**Parameters**:
- `reduced_dm`: Density matrix (dim × dim)
- `dim`: Matrix dimension

**Returns**: Von Neumann entropy $S(\rho)$ in bits

**Mathematical Definition**:
$$S(\rho) = -\text{Tr}(\rho \log_2 \rho) = -\sum_i \lambda_i \log_2 \lambda_i$$

where $\lambda_i$ are the eigenvalues of $\rho$.

**Interpretation**:
- $S = 0$: Pure state (no entanglement with traced-out subsystem)
- $S = 1$: Maximally entangled with one qubit
- $S = n$: Maximally entangled with $n$ qubits

### entanglement_renyi_entropy

Compute Rényi entropy of order $\alpha$.

```c
double entanglement_renyi_entropy(
    const complex_t *reduced_dm,
    uint64_t dim,
    double alpha
);
```

**Parameters**:
- `reduced_dm`: Density matrix
- `dim`: Matrix dimension
- `alpha`: Rényi parameter ($\alpha > 0$, $\alpha \neq 1$)

**Returns**: Rényi entropy $S_\alpha(\rho)$ in bits

**Mathematical Definition**:
$$S_\alpha(\rho) = \frac{1}{1-\alpha} \log_2 \text{Tr}(\rho^\alpha)$$

**Special Cases**:
- $\lim_{\alpha \to 1} S_\alpha = S$ (von Neumann entropy)
- $S_2 = -\log_2 \text{Tr}(\rho^2)$ (related to purity)
- $S_0 = \log_2 \text{rank}(\rho)$ (Hartley entropy)

### entanglement_entropy_bipartition

Compute entanglement entropy for a bipartition.

```c
double entanglement_entropy_bipartition(
    const quantum_state_t *state,
    const int *subsystem_b_qubits,
    int num_b_qubits
);
```

**Parameters**:
- `state`: Full quantum state
- `subsystem_b_qubits`: Qubits in subsystem B (will be traced out)
- `num_b_qubits`: Number of qubits in subsystem B

**Returns**: Entanglement entropy $S(\rho_A)$ in bits

**Example**:
```c
// Bell state: qubits 0 and 1 maximally entangled
quantum_state_t state;
quantum_state_init(&state, 2);
gate_hadamard(&state, 0);
gate_cnot(&state, 0, 1);

// Entanglement between qubit 0 and qubit 1
int subsystem_b[] = {1};
double S = entanglement_entropy_bipartition(&state, subsystem_b, 1);
printf("Entanglement entropy: %.4f bits\n", S);  // 1.0000
```

## Concurrence

### entanglement_concurrence_2qubit

Compute concurrence for a pure two-qubit state.

```c
double entanglement_concurrence_2qubit(const quantum_state_t *state);
```

**Parameters**:
- `state`: Two-qubit pure state

**Returns**: Concurrence $C \in [0, 1]$

**Mathematical Definition**:
For a pure state $|\psi\rangle$:
$$C = 2|\alpha_{00}\alpha_{11} - \alpha_{01}\alpha_{10}|$$

where $\alpha_{ij}$ are the amplitudes in the computational basis.

**Interpretation**:
- $C = 0$: Product state (separable)
- $C = 1$: Maximally entangled (Bell state)

### entanglement_concurrence_mixed

Compute concurrence from a two-qubit density matrix.

```c
double entanglement_concurrence_mixed(const complex_t *density_matrix);
```

**Parameters**:
- `density_matrix`: 4×4 density matrix

**Returns**: Concurrence $C \in [0, 1]$

**Mathematical Definition** (Wootters' formula):
$$C = \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4)$$

where $\lambda_i$ are the square roots of eigenvalues of $\rho \tilde{\rho}$ in decreasing order, and $\tilde{\rho} = (\sigma_y \otimes \sigma_y) \rho^* (\sigma_y \otimes \sigma_y)$.

## Schmidt Decomposition

### entanglement_schmidt_coefficients

Compute Schmidt coefficients for a bipartite pure state.

```c
int entanglement_schmidt_coefficients(
    const quantum_state_t *state,
    const int *partition_a_qubits,
    int num_a,
    double *coefficients,
    int *num_coefficients
);
```

**Parameters**:
- `state`: Full quantum state
- `partition_a_qubits`: Qubits in partition A
- `num_a`: Number of qubits in A
- `coefficients`: Output array for Schmidt coefficients
- `num_coefficients`: Output: number of non-zero coefficients

**Returns**: 0 on success, -1 on error

**Mathematical Definition**:
Any bipartite pure state can be written as:
$$|\psi\rangle_{AB} = \sum_{i=1}^{r} \lambda_i |u_i\rangle_A |v_i\rangle_B$$

where $\lambda_i > 0$ are the Schmidt coefficients and $r$ is the Schmidt rank.

**Example**:
```c
quantum_state_t state;
quantum_state_init(&state, 4);

// Create state with specific entanglement structure
gate_hadamard(&state, 0);
gate_cnot(&state, 0, 2);

// Get Schmidt coefficients for {0,1} vs {2,3} partition
int partition_a[] = {0, 1};
double coeffs[4];
int num_coeffs;

entanglement_schmidt_coefficients(&state, partition_a, 2, coeffs, &num_coeffs);

printf("Schmidt rank: %d\n", num_coeffs);
for (int i = 0; i < num_coeffs; i++) {
    printf("λ_%d = %.4f\n", i, coeffs[i]);
}
```

### entanglement_schmidt_rank

Compute Schmidt rank (number of non-zero Schmidt coefficients).

```c
int entanglement_schmidt_rank(
    const quantum_state_t *state,
    const int *partition_a_qubits,
    int num_a,
    double threshold
);
```

**Parameters**:
- `state`: Quantum state
- `partition_a_qubits`: Qubits in partition A
- `num_a`: Number of qubits in A
- `threshold`: Coefficients below this are considered zero

**Returns**: Schmidt rank

**Interpretation**:
- Rank 1: Product state
- Rank > 1: Entangled state
- Maximum rank $= \min(d_A, d_B)$: Maximally entangled

## Entanglement Detection

### entanglement_is_separable

Check if a state is separable (not entangled).

```c
int entanglement_is_separable(
    const quantum_state_t *state,
    const int *partition_a_qubits,
    int num_a
);
```

**Parameters**:
- `state`: Quantum state
- `partition_a_qubits`: Qubits in partition A
- `num_a`: Number of qubits in A

**Returns**: 1 if separable, 0 if entangled

**Note**: For pure states, separability is equivalent to Schmidt rank = 1.

### entanglement_purity

Compute purity of a density matrix.

```c
double entanglement_purity(const complex_t *reduced_dm, uint64_t dim);
```

**Parameters**:
- `reduced_dm`: Density matrix
- `dim`: Matrix dimension

**Returns**: Purity $\gamma = \text{Tr}(\rho^2) \in [1/d, 1]$

**Interpretation**:
- $\gamma = 1$: Pure state
- $\gamma = 1/d$: Maximally mixed state
- Lower purity indicates more entanglement with traced-out system

### entanglement_linear_entropy

Compute linear entropy.

```c
double entanglement_linear_entropy(const complex_t *reduced_dm, uint64_t dim);
```

**Parameters**:
- `reduced_dm`: Density matrix
- `dim`: Matrix dimension

**Returns**: Linear entropy $S_L = 1 - \text{Tr}(\rho^2)$

**Notes**: Linear entropy is an approximation to von Neumann entropy that avoids computing matrix logarithms.

## Area Law and Scaling

For ground states of local Hamiltonians, entanglement entropy typically follows an **area law**:

$$S(\rho_A) \propto |\partial A|$$

where $|\partial A|$ is the size of the boundary between subsystems A and B.

**Example: Verifying Area Law**:
```c
// Measure entanglement entropy for increasing subsystem sizes
quantum_state_t state;
quantum_state_init(&state, 20);  // 20-qubit chain

// Prepare ground state (e.g., via DMRG or VQE)
// ...

// Measure entropy for subsystems of size 1, 2, 3, ...
for (int size = 1; size <= 10; size++) {
    int subsystem[20];
    for (int i = 0; i < 20 - size; i++) {
        subsystem[i] = size + i;  // Trace out right part
    }
    double S = entanglement_entropy_bipartition(&state, subsystem, 20 - size);
    printf("Size %2d: S = %.4f bits\n", size, S);
}
```

## Thread Safety

- All functions are thread-safe for concurrent reads of the same state
- Functions that modify state require exclusive access

## Performance Notes

1. **Reduced density matrix**: $O(2^n)$ where $n$ is number of traced qubits
2. **Von Neumann entropy**: Requires eigenvalue decomposition, $O(d^3)$
3. **Schmidt decomposition**: Uses SVD, $O(\min(d_A, d_B)^3)$
4. **Concurrence**: $O(1)$ for pure states, $O(d^3)$ for mixed states

## See Also

- [Quantum State API](quantum-state.md) - State management
- [Measurement API](measurement.md) - Measurement operations
- [Concepts: Entanglement Measures](../../concepts/entanglement-measures.md) - Theory
