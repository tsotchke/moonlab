# Quantum Computing Basics

This document provides a comprehensive introduction to the fundamental concepts of quantum computing. We cover qubits, superposition, entanglement, and the basic operations that form the foundation of quantum computation.

## The Qubit

### Classical Bits vs Qubits

A classical bit exists in one of two states: 0 or 1. A **qubit** (quantum bit) can exist in a **superposition** of both states simultaneously.

Mathematically, a qubit state is a unit vector in $\mathbb{C}^2$:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

where:
- $\alpha, \beta \in \mathbb{C}$ are complex **probability amplitudes**
- $|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$ and $|1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$ are the **computational basis** states
- **Normalization**: $|\alpha|^2 + |\beta|^2 = 1$

### The Bloch Sphere

Any single-qubit pure state can be visualized as a point on the Bloch sphere:

$$|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle$$

where:
- $\theta \in [0, \pi]$ is the polar angle from the z-axis
- $\phi \in [0, 2\pi)$ is the azimuthal angle in the xy-plane

**Important points on the Bloch sphere**:

| State | Position | Coordinates |
|-------|----------|-------------|
| $\|0\rangle$ | North pole | $\theta = 0$ |
| $\|1\rangle$ | South pole | $\theta = \pi$ |
| $\|+\rangle = \frac{1}{\sqrt{2}}(\|0\rangle + \|1\rangle)$ | +x axis | $\theta = \frac{\pi}{2}, \phi = 0$ |
| $\|-\rangle = \frac{1}{\sqrt{2}}(\|0\rangle - \|1\rangle)$ | -x axis | $\theta = \frac{\pi}{2}, \phi = \pi$ |
| $\|i\rangle = \frac{1}{\sqrt{2}}(\|0\rangle + i\|1\rangle)$ | +y axis | $\theta = \frac{\pi}{2}, \phi = \frac{\pi}{2}$ |
| $\|-i\rangle = \frac{1}{\sqrt{2}}(\|0\rangle - i\|1\rangle)$ | -y axis | $\theta = \frac{\pi}{2}, \phi = \frac{3\pi}{2}$ |

### Global Phase

States differing only by a global phase factor are physically equivalent:

$$|\psi\rangle \equiv e^{i\gamma}|\psi\rangle$$

This is why the Bloch sphere represents all physically distinct single-qubit states.

## Superposition

### What Is Superposition?

Superposition is the principle that a quantum system can exist in multiple basis states simultaneously. The state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ is not in state $|0\rangle$ or $|1\rangle$—it's in both, with amplitudes $\alpha$ and $\beta$.

### Creating Superposition

The **Hadamard gate** creates equal superposition:

$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

$$H|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) = |+\rangle$$

$$H|1\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle) = |-\rangle$$

### Interference

Amplitudes can interfere constructively or destructively. Consider applying H twice:

$$H|+\rangle = H \cdot \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) = \frac{1}{2}[(|0\rangle + |1\rangle) + (|0\rangle - |1\rangle)] = |0\rangle$$

The $|1\rangle$ component cancels due to destructive interference.

**Quantum algorithms exploit interference** to amplify correct answers and suppress wrong ones.

## Multi-Qubit Systems

### Tensor Products

The state space of $n$ qubits is the tensor product of individual qubit spaces:

$$\mathcal{H} = \mathbb{C}^2 \otimes \mathbb{C}^2 \otimes \cdots \otimes \mathbb{C}^2 = \mathbb{C}^{2^n}$$

A 2-qubit system has basis states $|00\rangle, |01\rangle, |10\rangle, |11\rangle$:

$$|00\rangle = |0\rangle \otimes |0\rangle = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix}, \quad |01\rangle = \begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix}, \quad |10\rangle = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}, \quad |11\rangle = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}$$

### Qubit Ordering Convention

In Moonlab, we use the convention where qubit 0 is the **least significant bit**:

$$|q_1 q_0\rangle$$

So $|10\rangle$ means qubit 0 is in state $|0\rangle$ and qubit 1 is in state $|1\rangle$.

### General n-Qubit States

A general n-qubit state:

$$|\psi\rangle = \sum_{i=0}^{2^n-1} \alpha_i |i\rangle$$

where $|i\rangle$ is the computational basis state corresponding to the binary representation of $i$.

The exponential growth of state space dimension ($2^n$) is the source of quantum computational power—and the reason quantum simulation is classically hard.

## Entanglement

### Definition

A multi-qubit state is **entangled** if it cannot be written as a product of single-qubit states:

$$|\psi\rangle_{AB} \neq |\phi\rangle_A \otimes |\chi\rangle_B$$

### The Bell States

The four maximally entangled 2-qubit states:

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$
$$|\Phi^-\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$$
$$|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$$
$$|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$$

These form an orthonormal basis for 2-qubit states.

### Creating Bell States

The Bell state $|\Phi^+\rangle$ is created by:

```
|0⟩ ──[H]──●──
           │
|0⟩ ───────⊕──
```

1. Start with $|00\rangle$
2. Apply Hadamard: $\frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)$
3. Apply CNOT: $\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$

### Properties of Entanglement

**Non-local correlations**: Measuring one qubit of an entangled pair instantly determines the other's measurement outcome, regardless of distance.

For $|\Phi^+\rangle$:
- Measure qubit 0 → outcome $|0\rangle$ with 50% probability → qubit 1 is definitely $|0\rangle$
- Measure qubit 0 → outcome $|1\rangle$ with 50% probability → qubit 1 is definitely $|1\rangle$

**No faster-than-light communication**: The correlations don't allow signaling because individual measurement outcomes are random.

**Monogamy**: If qubits A and B are maximally entangled, neither can be entangled with a third qubit C.

### GHZ and W States

For 3+ qubits, there are inequivalent classes of entanglement:

**GHZ state** (3 qubits):
$$|GHZ\rangle = \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$$

**W state** (3 qubits):
$$|W\rangle = \frac{1}{\sqrt{3}}(|001\rangle + |010\rangle + |100\rangle)$$

These have different entanglement properties and cannot be converted to each other by local operations.

## Measurement

### Computational Basis Measurement

When measuring in the computational basis:

$$|\psi\rangle = \sum_i \alpha_i |i\rangle \xrightarrow{\text{measure}} |i\rangle \text{ with probability } |\alpha_i|^2$$

**Key properties**:
1. **Probabilistic**: Outcomes are random (weighted by $|\alpha_i|^2$)
2. **Collapse**: After measurement, state becomes the measured outcome
3. **Irreversible**: The original superposition is destroyed

### Measurement in Other Bases

Measurement can be performed in any orthonormal basis. To measure in basis $\{|u\rangle, |v\rangle\}$:

1. Apply unitary $U$ that maps $|u\rangle \to |0\rangle$ and $|v\rangle \to |1\rangle$
2. Measure in computational basis
3. (Optionally) apply $U^†$ to return to original basis

**Example**: Hadamard basis measurement
- Apply $H$, then measure in computational basis
- Distinguishes $|+\rangle$ from $|-\rangle$

### Partial Measurement

In multi-qubit systems, measuring a subset of qubits collapses only those qubits:

For $|\psi\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$, measuring qubit 0:

- Outcome $|0\rangle$ (50%): post-measurement state is $|00\rangle$
- Outcome $|1\rangle$ (50%): post-measurement state is $|11\rangle$

## Quantum Circuits

### Circuit Model

Quantum computation is typically expressed as circuits:
- **Wires** represent qubits (time flows left to right)
- **Boxes** represent gates
- **Measurement** is shown with a meter symbol

```
|0⟩ ──[H]──●──[M]───
           │
|0⟩ ───────⊕──[M]───
```

### Common Gates

**Single-qubit gates**:

| Gate | Matrix | Effect |
|------|--------|--------|
| $X$ | $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ | Bit flip |
| $Z$ | $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ | Phase flip |
| $H$ | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ | Superposition |
| $S$ | $\begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$ | $\pi/2$ phase |
| $T$ | $\begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$ | $\pi/4$ phase |

**Two-qubit gates**:

| Gate | Effect |
|------|--------|
| CNOT | Flip target if control is $\|1\rangle$ |
| CZ | Apply Z to target if control is $\|1\rangle$ |
| SWAP | Exchange two qubits |

### Gate Application in Multi-Qubit Systems

To apply gate $U$ to qubit $k$ in an $n$-qubit system:

$$U_k = I^{\otimes (n-k-1)} \otimes U \otimes I^{\otimes k}$$

In Moonlab, you simply specify the target qubit:
```c
gate_hadamard(&state, 2);  // Apply H to qubit 2
```

## Quantum Parallelism

### Superposition as Parallel Computation

A quantum computer with $n$ qubits in superposition:

$$|+\rangle^{\otimes n} = \frac{1}{\sqrt{2^n}} \sum_{x=0}^{2^n-1} |x\rangle$$

can evaluate a function $f$ on all inputs simultaneously:

$$U_f: |x\rangle|0\rangle \to |x\rangle|f(x)\rangle$$

$$U_f \frac{1}{\sqrt{2^n}} \sum_x |x\rangle|0\rangle = \frac{1}{\sqrt{2^n}} \sum_x |x\rangle|f(x)\rangle$$

### The Measurement Problem

However, measuring this state gives only **one** random $x$ and $f(x)$.

Quantum algorithms are clever about extracting global properties of $f$ without measuring individual values—using interference to amplify useful information.

## Quantum Speedups

### Where Quantum Helps

| Problem | Classical | Quantum | Speedup |
|---------|-----------|---------|---------|
| Unstructured search | $O(N)$ | $O(\sqrt{N})$ | Quadratic |
| Factoring | Sub-exponential | $O((\log N)^3)$ | Exponential |
| Simulation | Exponential | Polynomial | Exponential |
| Optimization | Problem-dependent | Potential speedup | Varies |

### Grover's Algorithm (Preview)

Grover's algorithm searches an unstructured database of $N$ items in $O(\sqrt{N})$ queries, compared to $O(N)$ classically.

Key insight: Amplitude amplification increases the amplitude of the target state by $O(1/\sqrt{N})$ per iteration, requiring $O(\sqrt{N})$ iterations.

## Summary

| Concept | Key Point |
|---------|-----------|
| Qubit | $\|\psi\rangle = \alpha\|0\rangle + \beta\|1\rangle$ with $\|\alpha\|^2 + \|\beta\|^2 = 1$ |
| Superposition | Coherent combination of basis states |
| Measurement | Probabilistic, collapses state |
| Multi-qubit | $2^n$ dimensional state space |
| Entanglement | Non-separable correlations |
| Gates | Unitary transformations |
| Parallelism | Evaluate on all inputs simultaneously |

## Next Steps

- [Quantum Gates](quantum-gates.md): Detailed gate mathematics
- [Measurement Theory](measurement-theory.md): Deeper dive into measurement
- [Entanglement Measures](entanglement-measures.md): Quantifying entanglement
