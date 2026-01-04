# Quantum Gates

Mathematical foundations of quantum logic gates.

## Introduction

Quantum gates are unitary transformations that manipulate qubit states. Unlike classical logic gates, quantum gates are reversible and can create superposition and entanglement.

## Mathematical Representation

### Unitary Operators

A quantum gate is represented by a unitary matrix $U$:

$$U^\dagger U = U U^\dagger = I$$

For an $n$-qubit gate, $U$ is a $2^n \times 2^n$ complex matrix.

**Properties**:
- Preserves norm: $\|U|\psi\rangle\| = \||\psi\rangle\|$
- Reversible: $U^{-1} = U^\dagger$
- Eigenvalues have magnitude 1: $|\lambda| = 1$

### Gate Application

Applying gate $U$ to state $|\psi\rangle$:

$$|\psi'\rangle = U |\psi\rangle$$

In component form:

$$\alpha_i' = \sum_j U_{ij} \alpha_j$$

## Pauli Gates

The Pauli matrices form a basis for single-qubit operations.

### Pauli-X (NOT Gate)

Bit flip: $|0\rangle \leftrightarrow |1\rangle$

$$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

**Properties**:
- $X^2 = I$
- Eigenvalues: $\pm 1$
- Eigenvectors: $|+\rangle, |-\rangle$

```c
quantum_state_x(state, qubit);
```

### Pauli-Y

Combined bit and phase flip:

$$Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$$

**Properties**:
- $Y^2 = I$
- $Y = iXZ$

```c
quantum_state_y(state, qubit);
```

### Pauli-Z

Phase flip: $|1\rangle \to -|1\rangle$

$$Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**Properties**:
- $Z^2 = I$
- Diagonal in computational basis

```c
quantum_state_z(state, qubit);
```

### Pauli Algebra

$$XY = iZ, \quad YZ = iX, \quad ZX = iY$$
$$XYZ = iI$$
$$\{X, Y\} = \{Y, Z\} = \{Z, X\} = 0$$

## Hadamard Gate

Creates superposition from computational basis states:

$$H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**Action**:
$$H|0\rangle = |+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$$
$$H|1\rangle = |-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$$

**Properties**:
- $H^2 = I$ (self-inverse)
- $H = \frac{X + Z}{\sqrt{2}}$
- Rotates X-basis to Z-basis

```c
quantum_state_h(state, qubit);
```

## Phase Gates

### S Gate (Phase Gate)

Quarter-turn around Z-axis:

$$S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix} = \sqrt{Z}$$

**Action**: $|1\rangle \to i|1\rangle$

```c
quantum_state_s(state, qubit);
```

### S-dagger

$$S^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}$$

```c
quantum_state_sdg(state, qubit);
```

### T Gate (pi/8 Gate)

Eighth-turn around Z-axis:

$$T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix} = \sqrt{S}$$

**Action**: $|1\rangle \to e^{i\pi/4}|1\rangle$

**Importance**: T gate + Clifford gates = universal gate set

```c
quantum_state_t(state, qubit);
```

### T-dagger

$$T^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix}$$

```c
quantum_state_tdg(state, qubit);
```

## Rotation Gates

### Rotation About X-Axis

$$R_X(\theta) = e^{-i\theta X/2} = \cos\frac{\theta}{2} I - i\sin\frac{\theta}{2} X$$

$$R_X(\theta) = \begin{pmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

```c
quantum_state_rx(state, qubit, theta);
```

### Rotation About Y-Axis

$$R_Y(\theta) = e^{-i\theta Y/2} = \cos\frac{\theta}{2} I - i\sin\frac{\theta}{2} Y$$

$$R_Y(\theta) = \begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

```c
quantum_state_ry(state, qubit, theta);
```

### Rotation About Z-Axis

$$R_Z(\theta) = e^{-i\theta Z/2} = \cos\frac{\theta}{2} I - i\sin\frac{\theta}{2} Z$$

$$R_Z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

```c
quantum_state_rz(state, qubit, theta);
```

### Phase Gate (General)

$$P(\phi) = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\phi} \end{pmatrix}$$

Relation to $R_Z$: $P(\phi) = e^{i\phi/2} R_Z(\phi)$

```c
quantum_state_phase(state, qubit, phi);
```

## Universal Single-Qubit Gates

### U3 Gate

General single-qubit unitary with 3 parameters:

$$U_3(\theta, \phi, \lambda) = \begin{pmatrix} \cos\frac{\theta}{2} & -e^{i\lambda}\sin\frac{\theta}{2} \\ e^{i\phi}\sin\frac{\theta}{2} & e^{i(\phi+\lambda)}\cos\frac{\theta}{2} \end{pmatrix}$$

Any single-qubit gate can be expressed as $U_3$ (up to global phase).

**Special cases**:
- $U_3(\theta, 0, 0) = R_Y(\theta)$
- $U_3(0, 0, \lambda) = P(\lambda)$
- $U_3(\pi, 0, \pi) = X$
- $U_3(\pi/2, 0, \pi) = H$

```c
quantum_state_u3(state, qubit, theta, phi, lambda);
```

### Euler Decomposition

Any $SU(2)$ operation can be decomposed as:

$$U = e^{i\alpha} R_Z(\beta) R_Y(\gamma) R_Z(\delta)$$

or equivalently:

$$U = e^{i\alpha} R_Z(\beta) R_X(\gamma) R_Z(\delta)$$

## Two-Qubit Gates

### CNOT (Controlled-NOT)

Flips target qubit if control is $|1\rangle$:

$$\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

**Action**:
$$|00\rangle \to |00\rangle, \quad |01\rangle \to |01\rangle$$
$$|10\rangle \to |11\rangle, \quad |11\rangle \to |10\rangle$$

**Creates entanglement**: $\text{CNOT}(H \otimes I)|00\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$

```c
quantum_state_cnot(state, control, target);
```

### CZ (Controlled-Z)

Applies phase if both qubits are $|1\rangle$:

$$\text{CZ} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}$$

**Properties**:
- Symmetric: CZ(a,b) = CZ(b,a)
- $\text{CZ} = (I \otimes H) \cdot \text{CNOT} \cdot (I \otimes H)$

```c
quantum_state_cz(state, qubit1, qubit2);
```

### SWAP

Exchanges two qubits:

$$\text{SWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

**Decomposition**: SWAP = CNOT(a,b) CNOT(b,a) CNOT(a,b)

```c
quantum_state_swap(state, qubit1, qubit2);
```

### iSWAP

SWAP with phase:

$$\text{iSWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & i & 0 \\ 0 & i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

```c
quantum_state_iswap(state, qubit1, qubit2);
```

### Controlled Rotations

$$CR_Z(\theta) = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & e^{-i\theta/2} & 0 \\ 0 & 0 & 0 & e^{i\theta/2} \end{pmatrix}$$

Similar definitions for $CR_X$ and $CR_Y$.

```c
quantum_state_crz(state, control, target, theta);
quantum_state_crx(state, control, target, theta);
quantum_state_cry(state, control, target, theta);
```

## Three-Qubit Gates

### Toffoli (CCNOT)

Flips target if both controls are $|1\rangle$:

$$\text{Toffoli} = \begin{pmatrix}
I_6 & 0 \\
0 & X
\end{pmatrix}$$

where $I_6$ is the $6 \times 6$ identity.

**Properties**:
- Universal for classical computation
- Self-inverse

```c
quantum_state_toffoli(state, control1, control2, target);
```

### Fredkin (CSWAP)

Swaps two qubits if control is $|1\rangle$:

```c
quantum_state_fredkin(state, control, target1, target2);
```

## Clifford Group

The Clifford group consists of gates that map Pauli operators to Pauli operators under conjugation.

### Generators

The single-qubit Clifford group is generated by $\{H, S\}$.

The multi-qubit Clifford group is generated by $\{H, S, \text{CNOT}\}$.

### Clifford Gates

| Gate | Clifford? |
|------|-----------|
| X, Y, Z | Yes |
| H | Yes |
| S, S† | Yes |
| CNOT | Yes |
| CZ | Yes |
| T | No |
| $R_Z(\theta)$ | Only for $\theta \in \{0, \pi/2, \pi, 3\pi/2\}$ |

### Gottesman-Knill Theorem

Clifford circuits on stabilizer states can be efficiently simulated classically in polynomial time.

## Universality

### Universal Gate Sets

A gate set is **universal** if it can approximate any unitary to arbitrary precision.

**Common universal sets**:
- $\{H, T, \text{CNOT}\}$
- $\{H, T, CZ\}$
- $\{\text{Toffoli}, H\}$
- $\{R_Y(\theta), \text{CNOT}\}$ for arbitrary $\theta$

### Solovay-Kitaev Theorem

Any single-qubit gate can be approximated to precision $\epsilon$ using $O(\log^c(1/\epsilon))$ gates from a universal set, where $c \approx 2$.

## Implementation Details

### Gate Matrix Storage

Moonlab stores gate matrices in row-major order:

```c
typedef struct {
    double matrix[8];   // 2x2 complex matrix (4 complex = 8 doubles)
} gate_1q_t;

typedef struct {
    double matrix[32];  // 4x4 complex matrix (16 complex = 32 doubles)
} gate_2q_t;
```

### Fused Gates

Multiple consecutive gates can be fused into a single operation:

```c
// Instead of:
quantum_state_h(state, 0);
quantum_state_t(state, 0);
quantum_state_h(state, 0);

// Use fused gate:
double fused[8];
gate_multiply(H, T, temp);
gate_multiply(temp, H, fused);
quantum_state_apply_1q(state, 0, fused);
```

### Gate Caching

Common gates are pre-computed at initialization:

```c
static const double PAULI_X[8] = {
    0.0, 0.0,  1.0, 0.0,   // Row 0
    1.0, 0.0,  0.0, 0.0    // Row 1
};
```

## See Also

- [Quantum Computing Basics](quantum-computing-basics.md) - Foundational concepts
- [State Vector Simulation](state-vector-simulation.md) - How gates are applied
- [Gate Reference](../reference/gate-reference.md) - Quick reference card
- [C API: Gates](../api/c/gates.md) - C function reference

