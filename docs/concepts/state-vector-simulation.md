# State Vector Simulation

How Moonlab represents and manipulates quantum states.

## Introduction

State vector simulation is the most direct approach to quantum computing simulation. It maintains the complete quantum state as a vector of complex amplitudes, enabling exact computation of all quantum operations.

This document explains the mathematical foundations and computational implementation of state vector simulation in Moonlab.

## Mathematical Foundation

### Quantum State Representation

A quantum system of $n$ qubits exists in a Hilbert space $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$ of dimension $N = 2^n$. The state is represented as:

$$|\psi\rangle = \sum_{i=0}^{N-1} \alpha_i |i\rangle$$

where:
- $\alpha_i \in \mathbb{C}$ are complex amplitudes
- $|i\rangle$ are computational basis states
- $\sum_i |\alpha_i|^2 = 1$ (normalization)

### Computational Basis States

The computational basis for $n$ qubits consists of $2^n$ orthonormal vectors:

| Qubits | Basis States |
|--------|--------------|
| 1 | $\|0\rangle, \|1\rangle$ |
| 2 | $\|00\rangle, \|01\rangle, \|10\rangle, \|11\rangle$ |
| 3 | $\|000\rangle, \|001\rangle, \ldots, \|111\rangle$ |
| $n$ | $\|0\ldots0\rangle$ through $\|1\ldots1\rangle$ |

Basis states are indexed by their binary representation:
$$|i\rangle = |b_{n-1} b_{n-2} \ldots b_1 b_0\rangle$$

where $i = \sum_{k=0}^{n-1} b_k \cdot 2^k$.

### State Vector Structure

Moonlab stores the state as an array of complex numbers:

```
Index:     0        1        2        3       ...    2^n - 1
State:   |00...0⟩  |00...1⟩  |00...10⟩ |00...11⟩ ...  |11...1⟩
Array:   [α₀,      α₁,      α₂,      α₃,      ...   α_{N-1}]
```

Each complex amplitude $\alpha_i$ is stored as a pair of double-precision floats:
- Real part: `amplitudes[2*i]`
- Imaginary part: `amplitudes[2*i + 1]`

## Memory Requirements

### Scaling

The memory required for state vector simulation grows exponentially with qubit count:

| Qubits | Amplitudes | Memory (Complex128) |
|--------|-----------|---------------------|
| 10 | 1,024 | 16 KB |
| 20 | 1,048,576 | 16 MB |
| 25 | 33,554,432 | 512 MB |
| 30 | 1,073,741,824 | 16 GB |
| 32 | 4,294,967,296 | 64 GB |
| 40 | 1,099,511,627,776 | 16 TB |

### Memory Layout

Moonlab uses aligned memory for SIMD optimization:

```c
typedef struct {
    uint32_t num_qubits;          // Number of qubits
    uint64_t num_amplitudes;      // 2^num_qubits
    double* amplitudes;           // Aligned complex array
    size_t stride;                // Memory stride
} quantum_state_t;
```

Memory is aligned to:
- 16 bytes for SSE2
- 32 bytes for AVX2
- 64 bytes for AVX-512

## Gate Application

### Single-Qubit Gates

A single-qubit gate $U$ acting on qubit $q$ transforms amplitudes in pairs. For each pair of indices $(i, i')$ that differ only in bit $q$:

$$\begin{pmatrix} \alpha_i' \\ \alpha_{i'}' \end{pmatrix} = U \begin{pmatrix} \alpha_i \\ \alpha_{i'} \end{pmatrix}$$

**Algorithm**:
```
for i in 0 to 2^n - 1:
    if bit q of i is 0:
        i' = i XOR (1 << q)
        (α[i], α[i']) = U · (α[i], α[i'])
```

**Complexity**: $O(2^n)$ - each amplitude visited once.

### Two-Qubit Gates

A two-qubit gate $U$ on qubits $(q_1, q_2)$ transforms groups of 4 amplitudes:

$$\begin{pmatrix} \alpha_{00}' \\ \alpha_{01}' \\ \alpha_{10}' \\ \alpha_{11}' \end{pmatrix} = U \begin{pmatrix} \alpha_{00} \\ \alpha_{01} \\ \alpha_{10} \\ \alpha_{11} \end{pmatrix}$$

where subscripts indicate bits $q_1$ and $q_2$.

**Complexity**: $O(2^n)$ - each amplitude visited once.

### Controlled Gates

Controlled gates only modify amplitudes where control qubits are $|1\rangle$:

```
for i in 0 to 2^n - 1:
    if control bits are all 1 AND target bit is 0:
        apply gate to (i, i XOR (1 << target))
```

CNOT example for control=0, target=1:
```
|00⟩ → |00⟩  (control=0, no change)
|01⟩ → |01⟩  (control=0, no change)
|10⟩ → |11⟩  (control=1, flip target)
|11⟩ → |10⟩  (control=1, flip target)
```

## Optimization Techniques

### SIMD Vectorization

Moonlab uses SIMD instructions to process multiple amplitudes simultaneously:

**AVX2 (256-bit)**:
- 4 complex numbers per instruction
- Fused multiply-add operations

**AVX-512 (512-bit)**:
- 8 complex numbers per instruction
- Available on recent x86 processors

**ARM NEON (128-bit)**:
- 2 complex numbers per instruction
- Apple Silicon optimization

```c
// AVX2 complex multiply-add example
__m256d a_re = _mm256_load_pd(&amplitudes[i]);
__m256d a_im = _mm256_load_pd(&amplitudes[i + 4]);
// ... matrix multiply
_mm256_store_pd(&amplitudes[i], result_re);
_mm256_store_pd(&amplitudes[i + 4], result_im);
```

### Parallelization

State vector simulation is embarrassingly parallel:

**OpenMP**:
```c
#pragma omp parallel for
for (uint64_t i = 0; i < num_amplitudes; i += stride) {
    apply_gate_block(state, gate, i);
}
```

**GPU (Metal)**:
- State vector in GPU memory
- One thread per amplitude pair
- Shared memory for gate matrices

### Cache Optimization

Gate operations exhibit poor cache locality for high qubits. Moonlab uses:

1. **Blocked Processing**: Process amplitudes in cache-sized blocks
2. **Stride Optimization**: Minimize cache line conflicts
3. **Prefetching**: Load next block while processing current

## Measurement Simulation

### Probability Computation

The probability of measuring basis state $|i\rangle$ is:

$$P(i) = |\alpha_i|^2 = \text{Re}(\alpha_i)^2 + \text{Im}(\alpha_i)^2$$

**Implementation**:
```c
double probability(const quantum_state_t* state, uint64_t index) {
    double re = state->amplitudes[2 * index];
    double im = state->amplitudes[2 * index + 1];
    return re * re + im * im;
}
```

### State Collapse

Measuring qubit $q$ collapses the state:

1. Compute $P(0)$ and $P(1)$ for qubit $q$
2. Sample outcome based on probabilities
3. Zero out amplitudes inconsistent with outcome
4. Renormalize remaining amplitudes

```c
void measure_qubit(quantum_state_t* state, int qubit) {
    double p0 = 0.0, p1 = 0.0;

    // Compute probabilities
    for (uint64_t i = 0; i < state->num_amplitudes; i++) {
        double prob = probability(state, i);
        if ((i >> qubit) & 1) {
            p1 += prob;
        } else {
            p0 += prob;
        }
    }

    // Sample outcome
    int outcome = (random_uniform() < p0) ? 0 : 1;

    // Collapse and renormalize
    double norm = sqrt(outcome ? p1 : p0);
    for (uint64_t i = 0; i < state->num_amplitudes; i++) {
        if (((i >> qubit) & 1) != outcome) {
            state->amplitudes[2*i] = 0.0;
            state->amplitudes[2*i + 1] = 0.0;
        } else {
            state->amplitudes[2*i] /= norm;
            state->amplitudes[2*i + 1] /= norm;
        }
    }
}
```

## Numerical Considerations

### Floating-Point Precision

Moonlab uses 64-bit double precision for:
- Amplitude real parts
- Amplitude imaginary parts
- Gate matrix elements
- Intermediate computations

Precision limits:
- Machine epsilon: $\approx 2.2 \times 10^{-16}$
- Denormalized minimum: $\approx 5 \times 10^{-324}$

### Normalization Drift

After many gate operations, accumulated floating-point errors can cause $\|\psi\|^2$ to drift from 1. Moonlab provides:

```c
void quantum_state_normalize(quantum_state_t* state);
```

**Automatic normalization**: Triggered when norm deviates by > $10^{-14}$.

### Unitarity Verification

Gate matrices should be unitary: $U^\dagger U = I$. Moonlab verifies:

$$\|U^\dagger U - I\|_F < \epsilon$$

where $\|\cdot\|_F$ is the Frobenius norm.

## Comparison with Other Methods

### State Vector vs. Tensor Networks

| Aspect | State Vector | Tensor Networks |
|--------|-------------|-----------------|
| Memory | $O(2^n)$ | $O(n \chi^2)$ |
| Exact | Yes | Approximate |
| Entanglement | Unlimited | Bond dimension limited |
| Best for | Small systems, high entanglement | Large 1D systems, area-law states |

### State Vector vs. Density Matrix

| Aspect | State Vector | Density Matrix |
|--------|-------------|----------------|
| Size | $2^n$ | $2^{2n}$ |
| Pure states | Natural | Redundant |
| Mixed states | Cannot represent | Natural |
| Noise simulation | Limited | Full |

## Implementation Details

### Initialization

New states are initialized to $|0\ldots0\rangle$:

```c
quantum_state_t* quantum_state_create(uint32_t num_qubits) {
    quantum_state_t* state = malloc(sizeof(quantum_state_t));
    state->num_qubits = num_qubits;
    state->num_amplitudes = 1ULL << num_qubits;

    // Aligned allocation for SIMD
    state->amplitudes = aligned_alloc(64,
        state->num_amplitudes * 2 * sizeof(double));

    // Initialize to |0...0⟩
    memset(state->amplitudes, 0,
        state->num_amplitudes * 2 * sizeof(double));
    state->amplitudes[0] = 1.0;  // |0...0⟩ amplitude

    return state;
}
```

### Qubit Ordering

Moonlab uses little-endian qubit ordering:
- Qubit 0 is the least significant bit
- $|q_{n-1} \ldots q_1 q_0\rangle$ corresponds to index $\sum_i q_i 2^i$

Example for 3 qubits:
```
Index 0: |000⟩ = |q2=0, q1=0, q0=0⟩
Index 1: |001⟩ = |q2=0, q1=0, q0=1⟩
Index 2: |010⟩ = |q2=0, q1=1, q0=0⟩
...
Index 7: |111⟩ = |q2=1, q1=1, q0=1⟩
```

## Performance Benchmarks

Typical single-qubit gate times on modern hardware:

| Qubits | CPU (single) | CPU (parallel) | GPU (Metal) |
|--------|-------------|----------------|-------------|
| 20 | 5 ms | 1 ms | 0.2 ms |
| 25 | 160 ms | 25 ms | 5 ms |
| 28 | 1.3 s | 200 ms | 40 ms |
| 30 | 5 s | 800 ms | 160 ms |

*Benchmarks on Apple M2 Max, 32GB RAM*

## References

1. Aaronson, S., & Gottesman, D. (2004). "Improved simulation of stabilizer circuits." Physical Review A, 70(5), 052328.

2. De Raedt, H., et al. (2007). "Massively parallel quantum computer simulator." Computer Physics Communications, 176(2), 121-136.

3. Pednault, E., et al. (2017). "Breaking the 49-qubit barrier in the simulation of quantum circuits." arXiv:1710.05867.

## See Also

- [Quantum Computing Basics](quantum-computing-basics.md) - Foundational concepts
- [Quantum Gates](quantum-gates.md) - Gate mathematics
- [Tensor Networks](tensor-networks.md) - Alternative simulation approach
- [GPU Acceleration Guide](../guides/gpu-acceleration.md) - Metal optimization

