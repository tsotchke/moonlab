# Gates API

Complete reference for quantum gate operations in the C library.

**Header**: `src/quantum/gates.h`

## Overview

The gates module implements a complete universal gate set for quantum computation. All gates:

- Preserve state normalization
- Are unitary (reversible)
- Return error codes for validation
- Support arbitrary qubit indices

## Single-Qubit Gates

### Pauli Gates

#### gate_pauli_x

Pauli-X gate (NOT gate, bit flip).

```c
qs_error_t gate_pauli_x(quantum_state_t *state, int qubit);
```

**Matrix**:
$$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

**Action**: $|0\rangle \leftrightarrow |1\rangle$

**Example**:
```c
quantum_state_t state;
quantum_state_init(&state, 1);
gate_pauli_x(&state, 0);  // |0⟩ → |1⟩
```

#### gate_pauli_y

Pauli-Y gate (bit and phase flip).

```c
qs_error_t gate_pauli_y(quantum_state_t *state, int qubit);
```

**Matrix**:
$$Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$$

**Action**: $|0\rangle \to i|1\rangle$, $|1\rangle \to -i|0\rangle$

#### gate_pauli_z

Pauli-Z gate (phase flip).

```c
qs_error_t gate_pauli_z(quantum_state_t *state, int qubit);
```

**Matrix**:
$$Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

**Action**: $|0\rangle \to |0\rangle$, $|1\rangle \to -|1\rangle$

### Hadamard Gate

#### gate_hadamard

Create superposition.

```c
qs_error_t gate_hadamard(quantum_state_t *state, int qubit);
```

**Matrix**:
$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**Action**:
- $|0\rangle \to \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) = |+\rangle$
- $|1\rangle \to \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle) = |-\rangle$

**Example**:
```c
// Create uniform superposition on 4 qubits
for (int q = 0; q < 4; q++) {
    gate_hadamard(&state, q);
}
```

### Phase Gates

#### gate_s

S gate (phase gate, $\sqrt{Z}$).

```c
qs_error_t gate_s(quantum_state_t *state, int qubit);
```

**Matrix**:
$$S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$$

**Action**: $|1\rangle \to i|1\rangle$

#### gate_s_dagger

S† gate (inverse S).

```c
qs_error_t gate_s_dagger(quantum_state_t *state, int qubit);
```

**Matrix**:
$$S^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}$$

#### gate_t

T gate ($\pi/8$ gate, $\sqrt{S}$).

```c
qs_error_t gate_t(quantum_state_t *state, int qubit);
```

**Matrix**:
$$T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$$

**Notes**: Essential for universal quantum computation with {H, T, CNOT}.

#### gate_t_dagger

T† gate (inverse T).

```c
qs_error_t gate_t_dagger(quantum_state_t *state, int qubit);
```

**Matrix**:
$$T^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix}$$

#### gate_phase

Arbitrary phase gate.

```c
qs_error_t gate_phase(quantum_state_t *state, int qubit, double theta);
```

**Parameters**:
- `theta`: Phase angle in radians

**Matrix**:
$$P(\theta) = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{pmatrix}$$

**Example**:
```c
gate_phase(&state, 0, M_PI / 3);  // π/3 phase
```

### Rotation Gates

#### gate_rx

Rotation around X axis.

```c
qs_error_t gate_rx(quantum_state_t *state, int qubit, double theta);
```

**Parameters**:
- `theta`: Rotation angle in radians

**Matrix**:
$$R_X(\theta) = \begin{pmatrix} \cos(\theta/2) & -i\sin(\theta/2) \\ -i\sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

**Example**:
```c
gate_rx(&state, 0, M_PI);  // X gate (180° rotation)
gate_rx(&state, 0, M_PI / 2);  // 90° rotation
```

#### gate_ry

Rotation around Y axis.

```c
qs_error_t gate_ry(quantum_state_t *state, int qubit, double theta);
```

**Matrix**:
$$R_Y(\theta) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

#### gate_rz

Rotation around Z axis.

```c
qs_error_t gate_rz(quantum_state_t *state, int qubit, double theta);
```

**Matrix**:
$$R_Z(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

### General Single-Qubit Gate

#### gate_u3

Most general single-qubit unitary (U3 gate).

```c
qs_error_t gate_u3(quantum_state_t *state, int qubit, double theta, double phi, double lambda);
```

**Parameters**:
- `theta`: Rotation angle
- `phi`: First phase angle
- `lambda`: Second phase angle

**Matrix**:
$$U_3(\theta, \phi, \lambda) = \begin{pmatrix} \cos(\theta/2) & -e^{i\lambda}\sin(\theta/2) \\ e^{i\phi}\sin(\theta/2) & e^{i(\phi+\lambda)}\cos(\theta/2) \end{pmatrix}$$

**Notes**: Any single-qubit gate can be expressed as $U_3$ (up to global phase).

## Two-Qubit Gates

### gate_cnot

Controlled-NOT gate.

```c
qs_error_t gate_cnot(quantum_state_t *state, int control, int target);
```

**Parameters**:
- `control`: Control qubit index
- `target`: Target qubit index

**Matrix** (computational basis $|00\rangle, |01\rangle, |10\rangle, |11\rangle$):
$$\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

**Action**: Flips target qubit if control is $|1\rangle$

**Notes**: Creates entanglement when applied after Hadamard.

**Example** (Bell state):
```c
quantum_state_init(&state, 2);
gate_hadamard(&state, 0);
gate_cnot(&state, 0, 1);  // (|00⟩ + |11⟩)/√2
```

### gate_cz

Controlled-Z gate.

```c
qs_error_t gate_cz(quantum_state_t *state, int control, int target);
```

**Matrix**:
$$CZ = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}$$

**Action**: Applies Z to target if control is $|1\rangle$

**Notes**: Symmetric—control and target can be exchanged.

### gate_cy

Controlled-Y gate.

```c
qs_error_t gate_cy(quantum_state_t *state, int control, int target);
```

### gate_swap

SWAP gate.

```c
qs_error_t gate_swap(quantum_state_t *state, int qubit1, int qubit2);
```

**Matrix**:
$$\text{SWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

**Action**: $|ab\rangle \to |ba\rangle$

### gate_cphase

Controlled-Phase gate.

```c
qs_error_t gate_cphase(quantum_state_t *state, int control, int target, double theta);
```

**Parameters**:
- `theta`: Phase angle in radians

**Matrix**:
$$CP(\theta) = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & e^{i\theta} \end{pmatrix}$$

**Notes**: Used in QFT with $\theta = \pi/2^k$.

### Controlled Rotations

#### gate_crx, gate_cry, gate_crz

Controlled rotation gates.

```c
qs_error_t gate_crx(quantum_state_t *state, int control, int target, double theta);
qs_error_t gate_cry(quantum_state_t *state, int control, int target, double theta);
qs_error_t gate_crz(quantum_state_t *state, int control, int target, double theta);
```

**Action**: Apply $R_X$, $R_Y$, or $R_Z$ to target if control is $|1\rangle$.

## Three-Qubit Gates

### gate_toffoli

Toffoli gate (CCNOT, controlled-controlled-NOT).

```c
qs_error_t gate_toffoli(quantum_state_t *state, int control1, int control2, int target);
```

**Parameters**:
- `control1`, `control2`: Control qubits
- `target`: Target qubit

**Action**: Flips target if both controls are $|1\rangle$

**Notes**: Universal for classical reversible computation.

**Example** (AND gate):
```c
// Compute c = a AND b
// Initialize: |a⟩|b⟩|0⟩
gate_toffoli(&state, 0, 1, 2);  // |a⟩|b⟩|a AND b⟩
```

### gate_fredkin

Fredkin gate (CSWAP, controlled-SWAP).

```c
qs_error_t gate_fredkin(quantum_state_t *state, int control, int target1, int target2);
```

**Action**: Swaps targets if control is $|1\rangle$

## Multi-Qubit Gates

### gate_mcx

Multi-controlled X gate (generalized Toffoli).

```c
qs_error_t gate_mcx(quantum_state_t *state, const int *controls, size_t num_controls, int target);
```

**Parameters**:
- `controls`: Array of control qubit indices
- `num_controls`: Number of controls
- `target`: Target qubit

**Action**: Applies X to target if all controls are $|1\rangle$

### gate_mcz

Multi-controlled Z gate.

```c
qs_error_t gate_mcz(quantum_state_t *state, const int *controls, size_t num_controls, int target);
```

### gate_qft

Quantum Fourier Transform.

```c
qs_error_t gate_qft(quantum_state_t *state, const int *qubits, size_t num_qubits);
```

**Parameters**:
- `qubits`: Array of qubit indices
- `num_qubits`: Number of qubits

**Action**: Applies QFT to specified qubits

**Notes**: Essential for Shor's algorithm and phase estimation.

**Example**:
```c
int qubits[] = {0, 1, 2, 3};
gate_qft(&state, qubits, 4);
```

### gate_iqft

Inverse Quantum Fourier Transform.

```c
qs_error_t gate_iqft(quantum_state_t *state, const int *qubits, size_t num_qubits);
```

## Custom Gates

### apply_single_qubit_gate

Apply arbitrary 2×2 unitary.

```c
qs_error_t apply_single_qubit_gate(
    quantum_state_t *state,
    int qubit,
    const complex_t matrix[2][2]
);
```

**Example**:
```c
// Custom gate: √X
complex_t sqrt_x[2][2] = {
    {0.5 + 0.5*I, 0.5 - 0.5*I},
    {0.5 - 0.5*I, 0.5 + 0.5*I}
};
apply_single_qubit_gate(&state, 0, sqrt_x);
```

### apply_two_qubit_gate

Apply arbitrary 4×4 unitary.

```c
qs_error_t apply_two_qubit_gate(
    quantum_state_t *state,
    int qubit1,
    int qubit2,
    const complex_t matrix[4][4]
);
```

## Universal Gate Sets

These gate sets can approximate any unitary:

| Set | Gates |
|-----|-------|
| Clifford+T | H, S, CNOT, T |
| Rotation | $R_X$, $R_Y$, $R_Z$, CNOT |
| IBM | U3, CNOT |
| Rigetti | $R_X$, $R_Z$, CZ |

## Error Handling

All gate functions return `qs_error_t`:

```c
qs_error_t err = gate_cnot(&state, 0, 5);
if (err == QS_ERROR_INVALID_QUBIT) {
    fprintf(stderr, "Qubit index out of range\n");
}
```

## Performance Notes

1. **Gate order**: Gates on non-adjacent qubits may be slower due to memory access patterns
2. **Batching**: Apply multiple gates before querying state properties
3. **SIMD**: Gates are vectorized when possible

## Gate Reference Table

| Gate | Qubits | Parameters | Matrix Size |
|------|--------|------------|-------------|
| X, Y, Z, H, S, T | 1 | - | 2×2 |
| Rx, Ry, Rz, Phase | 1 | θ | 2×2 |
| U3 | 1 | θ, φ, λ | 2×2 |
| CNOT, CZ, CY | 2 | - | 4×4 |
| SWAP | 2 | - | 4×4 |
| CPhase, CRx, CRy, CRz | 2 | θ | 4×4 |
| Toffoli | 3 | - | 8×8 |
| Fredkin | 3 | - | 8×8 |
| MCX, MCZ | n | - | $2^n \times 2^n$ |
| QFT, IQFT | n | - | $2^n \times 2^n$ |

## See Also

- [Quantum State API](quantum-state.md) - State management
- [Measurement API](measurement.md) - Measure after gates
- [Gate Reference](../../reference/gate-reference.md) - Quick reference
