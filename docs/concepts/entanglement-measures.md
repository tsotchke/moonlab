# Entanglement Measures

Quantifying quantum correlations.

## Introduction

Entanglement is a uniquely quantum phenomenon where composite systems exhibit correlations that cannot be explained classically. This document covers the mathematical measures used to detect and quantify entanglement.

## Pure State Entanglement

### Product vs. Entangled States

A bipartite pure state $|\psi\rangle_{AB}$ is a **product state** if:

$$|\psi\rangle_{AB} = |\phi\rangle_A \otimes |\chi\rangle_B$$

Otherwise, it is **entangled**.

**Examples**:
- Product: $|00\rangle = |0\rangle \otimes |0\rangle$
- Entangled: $|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$

### Schmidt Decomposition

Any bipartite pure state can be written as:

$$|\psi\rangle_{AB} = \sum_{i=1}^{r} \lambda_i |i\rangle_A \otimes |i\rangle_B$$

where:
- $\lambda_i > 0$ are **Schmidt coefficients** with $\sum_i \lambda_i^2 = 1$
- $r$ is the **Schmidt rank** (number of non-zero coefficients)
- $\{|i\rangle_A\}$ and $\{|i\rangle_B\}$ are orthonormal bases

**Entanglement criterion**: $r > 1 \Leftrightarrow$ entangled

## Von Neumann Entropy

### Definition

The **Von Neumann entropy** of a density matrix $\rho$:

$$S(\rho) = -\text{Tr}[\rho \log_2 \rho] = -\sum_i \lambda_i \log_2 \lambda_i$$

where $\lambda_i$ are eigenvalues of $\rho$.

**Properties**:
- $S(\rho) \geq 0$
- $S(\rho) = 0$ iff $\rho$ is pure
- $S(\rho) \leq \log_2(d)$ for $d$-dimensional system
- Maximum for maximally mixed state: $\rho = I/d$

### Entanglement Entropy

For pure state $|\psi\rangle_{AB}$, the **entanglement entropy** is:

$$S_E = S(\rho_A) = S(\rho_B)$$

where $\rho_A = \text{Tr}_B[|\psi\rangle\langle\psi|]$ is the reduced density matrix.

**Interpretation**: Measures quantum correlations between A and B.

```c
double entropy = quantum_state_entanglement_entropy(state, qubits_A, num_qubits_A);
```

### Examples

| State | Entanglement Entropy |
|-------|---------------------|
| $\|00\rangle$ | 0 bits |
| $\|\Phi^+\rangle = \frac{\|00\rangle + \|11\rangle}{\sqrt{2}}$ | 1 bit |
| $\|GHZ_3\rangle = \frac{\|000\rangle + \|111\rangle}{\sqrt{2}}$ (bipartition 1\|23) | 1 bit |
| Random state (typical) | $\approx n/2$ bits |

## Purity

### Definition

The **purity** of a density matrix:

$$\gamma = \text{Tr}[\rho^2] = \sum_i \lambda_i^2$$

**Properties**:
- $\frac{1}{d} \leq \gamma \leq 1$
- $\gamma = 1$ iff $\rho$ is pure
- $\gamma = 1/d$ for maximally mixed state

```c
double purity = quantum_state_purity(state);
```

### Linear Entropy

The **linear entropy** (simpler than Von Neumann):

$$S_L = 1 - \gamma = 1 - \text{Tr}[\rho^2]$$

Ranges from 0 (pure) to $1 - 1/d$ (maximally mixed).

## Fidelity

### State Fidelity

The **fidelity** between two pure states:

$$F(|\psi\rangle, |\phi\rangle) = |\langle\psi|\phi\rangle|^2$$

For general density matrices:

$$F(\rho, \sigma) = \left(\text{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\right)^2$$

**Properties**:
- $0 \leq F \leq 1$
- $F = 1$ iff $\rho = \sigma$
- $F = 0$ iff $\rho \perp \sigma$ (orthogonal support)

```c
double fidelity = quantum_state_fidelity(state1, state2);
```

### Uhlmann's Theorem

For pure states with mixed subsystems, fidelity relates to purifications:

$$F(\rho_A, \sigma_A) = \max_{|\psi\rangle, |\phi\rangle} |\langle\psi|\phi\rangle|^2$$

where max is over all purifications.

## Concurrence

### Two-Qubit Entanglement

For a two-qubit state $\rho$, the **concurrence**:

$$C(\rho) = \max(0, \lambda_1 - \lambda_2 - \lambda_3 - \lambda_4)$$

where $\lambda_i$ are eigenvalues (decreasing) of:

$$R = \sqrt{\sqrt{\rho} \tilde{\rho} \sqrt{\rho}}$$

and $\tilde{\rho} = (Y \otimes Y) \rho^* (Y \otimes Y)$.

**Properties**:
- $0 \leq C \leq 1$
- $C = 0$ for separable states
- $C = 1$ for maximally entangled Bell states

```c
double concurrence = quantum_state_concurrence(state, qubit_a, qubit_b);
```

### Entanglement of Formation

Derived from concurrence:

$$E_F = h\left(\frac{1 + \sqrt{1 - C^2}}{2}\right)$$

where $h(x) = -x\log_2 x - (1-x)\log_2(1-x)$ is the binary entropy.

## Negativity

### Definition

For bipartite state $\rho_{AB}$, the **negativity**:

$$\mathcal{N}(\rho) = \frac{\|\rho^{T_B}\|_1 - 1}{2}$$

where $\rho^{T_B}$ is partial transpose over B, and $\|M\|_1 = \text{Tr}\sqrt{M^\dagger M}$.

**Properties**:
- $\mathcal{N} = 0$ for separable states (necessary, not sufficient)
- $\mathcal{N} > 0$ implies entanglement
- Upper bounded by $(d-1)/2$ for $d$-dimensional systems

### Logarithmic Negativity

$$E_N = \log_2 \|\rho^{T_B}\|_1 = \log_2(2\mathcal{N} + 1)$$

Additive entanglement measure: $E_N(\rho \otimes \sigma) = E_N(\rho) + E_N(\sigma)$

## Mutual Information

### Quantum Mutual Information

For bipartite state $\rho_{AB}$:

$$I(A:B) = S(\rho_A) + S(\rho_B) - S(\rho_{AB})$$

**Properties**:
- $I(A:B) \geq 0$
- $I(A:B) = 0$ iff $\rho_{AB} = \rho_A \otimes \rho_B$
- For pure states: $I(A:B) = 2S(\rho_A)$

### Classical vs. Quantum Correlations

Total correlations = Classical + Quantum:

$$I(A:B) = J(A:B) + D(A:B)$$

where:
- $J(A:B)$: Classical correlations
- $D(A:B)$: **Quantum discord**

## Multipartite Entanglement

### GHZ vs. W States

Two inequivalent types of 3-qubit entanglement:

**GHZ**: $|GHZ\rangle = \frac{|000\rangle + |111\rangle}{\sqrt{2}}$
- Maximum bipartite entanglement (1 ebit)
- Loses all entanglement if one qubit is lost

**W**: $|W\rangle = \frac{|001\rangle + |010\rangle + |100\rangle}{\sqrt{3}}$
- Robust: losing one qubit leaves entangled pair
- Less bipartite entanglement than GHZ

### Entanglement Monotones

Functions $E(\rho)$ that don't increase under LOCC:

1. $E(\rho) \geq 0$
2. $E(\text{separable}) = 0$
3. $E(\Lambda[\rho]) \leq E(\rho)$ for LOCC operations $\Lambda$

Examples: Entanglement entropy, concurrence, negativity

### Geometric Measure

Distance to nearest separable state:

$$E_G(|\psi\rangle) = 1 - \max_{|\phi\rangle \in \text{SEP}} |\langle\phi|\psi\rangle|^2$$

## Area Laws

### Entanglement in Ground States

For local Hamiltonians in $d$ spatial dimensions, ground state entanglement entropy typically scales as:

$$S(A) \sim |\partial A|$$

(boundary area, not volume)

**Exceptions**: Critical 1D systems with $S \sim \log L$

### Implications for Simulation

Area law states can be efficiently represented by:
- Matrix Product States (1D)
- Tensor networks (higher D)

This enables simulation of large systems.

## Implementation

### Computing Reduced Density Matrix

```c
// Get reduced density matrix for subsystem
complex_t* rho_reduced = quantum_state_reduced_density_matrix(
    state,
    subsystem_qubits,
    num_subsystem_qubits
);

// Compute eigenvalues for entropy
double* eigenvalues = density_matrix_eigenvalues(rho_reduced, dim);

// Von Neumann entropy
double entropy = 0.0;
for (int i = 0; i < dim; i++) {
    if (eigenvalues[i] > 1e-15) {
        entropy -= eigenvalues[i] * log2(eigenvalues[i]);
    }
}
```

### Efficient Entropy Calculation

For pure states, use Schmidt decomposition instead of full density matrix:

```c
// Compute entanglement entropy via singular values
double entropy = quantum_state_entanglement_entropy_svd(
    state,
    partition_qubits,
    num_partition_qubits
);
```

Complexity: $O(2^n \cdot \min(2^{n_A}, 2^{n_B})^2)$ vs. $O(4^n)$ for full $\rho$.

## References

1. Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*. Cambridge University Press. Chapter 11.

2. Horodecki, R., Horodecki, P., Horodecki, M., & Horodecki, K. (2009). "Quantum entanglement." Reviews of Modern Physics, 81(2), 865.

3. Eisert, J., Cramer, M., & Plenio, M. B. (2010). "Colloquium: Area laws for the entanglement entropy." Reviews of Modern Physics, 82(1), 277.

## See Also

- [Quantum Computing Basics](quantum-computing-basics.md) - Foundational concepts
- [Measurement Theory](measurement-theory.md) - How measurement affects entanglement
- [Tensor Networks](tensor-networks.md) - Efficient representation of low-entanglement states
- [C API: Entanglement](../api/c/entanglement.md) - Function reference

