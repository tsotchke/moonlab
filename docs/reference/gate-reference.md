# Gate Reference

Quick reference for quantum gates.

## Single-Qubit Gates

### Pauli Gates

| Gate | Matrix | Action | C API |
|------|--------|--------|-------|
| **X** | $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ | Bit flip: $\|0\rangle \leftrightarrow \|1\rangle$ | `quantum_state_x(s, q)` |
| **Y** | $\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$ | Bit + phase flip | `quantum_state_y(s, q)` |
| **Z** | $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ | Phase flip: $\|1\rangle \to -\|1\rangle$ | `quantum_state_z(s, q)` |

### Hadamard

| Gate | Matrix | Action | C API |
|------|--------|--------|-------|
| **H** | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ | Creates superposition | `quantum_state_h(s, q)` |

$$H|0\rangle = |+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$$
$$H|1\rangle = |-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$$

### Phase Gates

| Gate | Matrix | Action | C API |
|------|--------|--------|-------|
| **S** | $\begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$ | $\|1\rangle \to i\|1\rangle$ | `quantum_state_s(s, q)` |
| **S†** | $\begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}$ | $\|1\rangle \to -i\|1\rangle$ | `quantum_state_sdg(s, q)` |
| **T** | $\begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$ | $\|1\rangle \to e^{i\pi/4}\|1\rangle$ | `quantum_state_t(s, q)` |
| **T†** | $\begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix}$ | $\|1\rangle \to e^{-i\pi/4}\|1\rangle$ | `quantum_state_tdg(s, q)` |

**Relations**: $S = T^2$, $Z = S^2 = T^4$

### Rotation Gates

| Gate | Matrix | C API |
|------|--------|-------|
| **Rx(θ)** | $\begin{pmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$ | `quantum_state_rx(s, q, θ)` |
| **Ry(θ)** | $\begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$ | `quantum_state_ry(s, q, θ)` |
| **Rz(θ)** | $\begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$ | `quantum_state_rz(s, q, θ)` |

**Common angles**:

| θ | Rx(θ) | Ry(θ) | Rz(θ) |
|---|-------|-------|-------|
| π | iX | iY | iZ |
| π/2 | √X | √Y | S |
| π/4 | | | T |

### General Phase

| Gate | Matrix | C API |
|------|--------|-------|
| **P(φ)** | $\begin{pmatrix} 1 & 0 \\ 0 & e^{i\phi} \end{pmatrix}$ | `quantum_state_phase(s, q, φ)` |

**Relation**: $P(\phi) = e^{i\phi/2} R_z(\phi)$

### Universal Gate

| Gate | Matrix | C API |
|------|--------|-------|
| **U3(θ,φ,λ)** | $\begin{pmatrix} \cos\frac{\theta}{2} & -e^{i\lambda}\sin\frac{\theta}{2} \\ e^{i\phi}\sin\frac{\theta}{2} & e^{i(\phi+\lambda)}\cos\frac{\theta}{2} \end{pmatrix}$ | `quantum_state_u3(s, q, θ, φ, λ)` |

Any single-qubit gate can be written as U3 (up to global phase).

## Two-Qubit Gates

### Controlled-NOT (CNOT, CX)

$$\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

| Input | Output |
|-------|--------|
| \|00⟩ | \|00⟩ |
| \|01⟩ | \|01⟩ |
| \|10⟩ | \|11⟩ |
| \|11⟩ | \|10⟩ |

```c
quantum_state_cnot(state, control, target);
quantum_state_cx(state, control, target);  // Alias
```

### Controlled-Z (CZ)

$$\text{CZ} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}$$

Symmetric: CZ(a,b) = CZ(b,a)

```c
quantum_state_cz(state, qubit1, qubit2);
```

### Controlled-Y (CY)

$$\text{CY} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & -i \\ 0 & 0 & i & 0 \end{pmatrix}$$

```c
quantum_state_cy(state, control, target);
```

### SWAP

$$\text{SWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

Exchanges two qubits: $|ab\rangle \to |ba\rangle$

```c
quantum_state_swap(state, qubit1, qubit2);
```

**Decomposition**: SWAP = CNOT(a,b) · CNOT(b,a) · CNOT(a,b)

### iSWAP

$$\text{iSWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & i & 0 \\ 0 & i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

```c
quantum_state_iswap(state, qubit1, qubit2);
```

### Controlled Rotations

| Gate | Matrix | C API |
|------|--------|-------|
| **CRx(θ)** | $\text{diag}(I, R_x(\theta))$ | `quantum_state_crx(s, c, t, θ)` |
| **CRy(θ)** | $\text{diag}(I, R_y(\theta))$ | `quantum_state_cry(s, c, t, θ)` |
| **CRz(θ)** | $\text{diag}(I, R_z(\theta))$ | `quantum_state_crz(s, c, t, θ)` |
| **CP(φ)** | $\text{diag}(1, 1, 1, e^{i\phi})$ | `quantum_state_cphase(s, c, t, φ)` |

### XX, YY, ZZ Interactions

| Gate | Definition | C API |
|------|------------|-------|
| **Rxx(θ)** | $e^{-i\theta X \otimes X / 2}$ | `quantum_state_rxx(s, q1, q2, θ)` |
| **Ryy(θ)** | $e^{-i\theta Y \otimes Y / 2}$ | `quantum_state_ryy(s, q1, q2, θ)` |
| **Rzz(θ)** | $e^{-i\theta Z \otimes Z / 2}$ | `quantum_state_rzz(s, q1, q2, θ)` |

## Three-Qubit Gates

### Toffoli (CCNOT, CCX)

Flips target if both controls are |1⟩:

| Input | Output |
|-------|--------|
| \|110⟩ | \|111⟩ |
| \|111⟩ | \|110⟩ |
| other | unchanged |

```c
quantum_state_toffoli(state, control1, control2, target);
quantum_state_ccx(state, control1, control2, target);  // Alias
```

### Fredkin (CSWAP)

Swaps two qubits if control is |1⟩:

| Input | Output |
|-------|--------|
| \|100⟩ | \|100⟩ |
| \|101⟩ | \|110⟩ |
| \|110⟩ | \|101⟩ |
| \|111⟩ | \|111⟩ |

```c
quantum_state_fredkin(state, control, target1, target2);
quantum_state_cswap(state, control, target1, target2);  // Alias
```

## Multi-Qubit Operations

### Quantum Fourier Transform

$$\text{QFT}|j\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{2\pi ijk/N} |k\rangle$$

```c
quantum_state_qft(state, qubits, num_qubits);
quantum_state_iqft(state, qubits, num_qubits);  // Inverse
```

## Gate Identities

### Conjugation

$$HXH = Z, \quad HYH = -Y, \quad HZH = X$$
$$SXS^\dagger = Y, \quad SYS^\dagger = -X$$

### Commutation

$$[X, Y] = 2iZ, \quad [Y, Z] = 2iX, \quad [Z, X] = 2iY$$

### CNOT Relations

$$\text{CNOT} = (I \otimes H) \cdot \text{CZ} \cdot (I \otimes H)$$
$$\text{CNOT}^2 = I$$

### Swap Decomposition

```
SWAP = ──●──X──●──
         │  │  │
       ──X──●──X──
```

## Gate Costs

### In Terms of Native Gates

Assuming native gate set: {Rz, √X, CNOT}

| Gate | Cost |
|------|------|
| X | 2 √X |
| Y | 2 √X + 2 Rz |
| Z | 2 Rz |
| H | 2 Rz + √X |
| S | 1 Rz |
| T | 1 Rz |
| Rx(θ) | 2 Rz + 2 √X |
| Ry(θ) | 2 Rz + 2 √X |
| Rz(θ) | 1 Rz |
| CNOT | 1 CNOT |
| CZ | 2 H + 1 CNOT |
| SWAP | 3 CNOT |
| Toffoli | 6 CNOT + single-qubit |

## Bloch Sphere Visualization

Single-qubit state: $|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle$

| Gate | Bloch Sphere Action |
|------|---------------------|
| X | 180° rotation about X-axis |
| Y | 180° rotation about Y-axis |
| Z | 180° rotation about Z-axis |
| H | 180° rotation about X+Z axis |
| S | 90° rotation about Z-axis |
| T | 45° rotation about Z-axis |
| Rx(θ) | θ rotation about X-axis |
| Ry(θ) | θ rotation about Y-axis |
| Rz(θ) | θ rotation about Z-axis |

## See Also

- [Quantum Gates (Concepts)](../concepts/quantum-gates.md) - Full mathematical details
- [C API: Gates](../api/c/gates.md) - Complete C API reference
- [Python API: Core](../api/python/core.md) - Python gate methods

