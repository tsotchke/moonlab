# Archived Moonlab Documentation: Gate reference

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Gate reference

Quick reference for Moonlab's state-vector gate API.  Every entry
pins the C symbol to the declaration in
[`src/quantum/gates.h`](../../src/quantum/gates.h); every signature
takes a `quantum_state_t *` and returns a `qs_error_t` (see
[error-codes.md](error-codes.md) for the return-code convention).
Mathematical conventions match
[Quantum gates (concepts)](../concepts/quantum-gates.md).

## Single-qubit gates

### Pauli gates

| Gate | Matrix | Action | C API |
|------|--------|--------|-------|
| **X** | $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ | Bit flip | `gate_pauli_x(state, qubit)` |
| **Y** | $\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$ | Bit + phase flip | `gate_pauli_y(state, qubit)` |
| **Z** | $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ | Phase flip | `gate_pauli_z(state, qubit)` |

### Hadamard

| Gate | Matrix | Action | C API |
|------|--------|--------|-------|
| **H** | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ | Creates superposition | `gate_hadamard(state, qubit)` |

$$H\lvert0\rangle = \lvert+\rangle, \qquad H\lvert1\rangle = \lvert-\rangle.$$

### Phase gates

| Gate | Matrix | Action | C API |
|------|--------|--------|-------|
| **S**       | $\begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$       | $\lvert1\rangle \to i\lvert1\rangle$       | `gate_s(state, qubit)`        |
| **S<sup>†</sup>** | $\begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}$ | $\lvert1\rangle \to -i\lvert1\rangle$     | `gate_s_dagger(state, qubit)` |
| **T**       | $\begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$  | $\lvert1\rangle \to e^{i\pi/4}\lvert1\rangle$  | `gate_t(state, qubit)`        |
| **T<sup>†</sup>** | $\begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix}$ | $\lvert1\rangle \to e^{-i\pi/4}\lvert1\rangle$ | `gate_t_dagger(state, qubit)` |

Identities: $S = T^2$, $Z = S^2 = T^4$.

### Rotation gates

| Gate | Matrix | C API |
|------|--------|-------|
| **R<sub>x</sub>(θ)** | $\begin{pmatrix} \cos\tfrac{\theta}{2} & -i\sin\tfrac{\theta}{2} \\ -i\sin\tfrac{\theta}{2} & \cos\tfrac{\theta}{2} \end{pmatrix}$ | `gate_rx(state, qubit, theta)` |
| **R<sub>y</sub>(θ)** | $\begin{pmatrix} \cos\tfrac{\theta}{2} & -\sin\tfrac{\theta}{2} \\ \sin\tfrac{\theta}{2} & \cos\tfrac{\theta}{2} \end{pmatrix}$ | `gate_ry(state, qubit, theta)` |
| **R<sub>z</sub>(θ)** | $\begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$ | `gate_rz(state, qubit, theta)` |

### General phase

| Gate | Matrix | C API |
|------|--------|-------|
| **P(φ)** | $\begin{pmatrix} 1 & 0 \\ 0 & e^{i\phi} \end{pmatrix}$ | `gate_phase(state, qubit, phi)` |

Identity: $P(\phi) = e^{i\phi/2} R_z(\phi)$.

### Universal gate

| Gate | Matrix | C API |
|------|--------|-------|
| **U3(θ,φ,λ)** | $\begin{pmatrix} \cos\tfrac{\theta}{2} & -e^{i\lambda}\sin\tfrac{\theta}{2} \\ e^{i\phi}\sin\tfrac{\theta}{2} & e^{i(\phi+\lambda)}\cos\tfrac{\theta}{2} \end{pmatrix}$ | `gate_u3(state, qubit, theta, phi, lambda)` |

Any single-qubit gate decomposes as `U3` up to a global phase.

## Two-qubit gates

### Controlled-NOT (CNOT)

$$\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

[archived fence delimiter: ```c]
gate_cnot(state, control, target);
[archived fence delimiter: ```]

### Controlled-Y (CY)

$$\text{CY} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & -i \\ 0 & 0 & i & 0 \end{pmatrix}$$

[archived fence delimiter: ```c]
gate_cy(state, control, target);
[archived fence delimiter: ```]

### Controlled-Z (CZ)

$$\text{CZ} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{pmatrix}$$

Symmetric: `CZ(a, b)` equals `CZ(b, a)`.

[archived fence delimiter: ```c]
gate_cz(state, control, target);
[archived fence delimiter: ```]

### SWAP

$$\text{SWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

[archived fence delimiter: ```c]
gate_swap(state, qubit1, qubit2);
[archived fence delimiter: ```]

Decomposition: `SWAP = CNOT(a,b) · CNOT(b,a) · CNOT(a,b)`.

### Controlled rotations

| Gate | Definition | C API |
|------|------------|-------|
| **CR<sub>x</sub>(θ)** | $\mathrm{diag}(I, R_x(\theta))$ | `gate_crx(state, control, target, theta)` |
| **CR<sub>y</sub>(θ)** | $\mathrm{diag}(I, R_y(\theta))$ | `gate_cry(state, control, target, theta)` |
| **CR<sub>z</sub>(θ)** | $\mathrm{diag}(I, R_z(\theta))$ | `gate_crz(state, control, target, theta)` |
| **CP(φ)** | $\mathrm{diag}(1, 1, 1, e^{i\phi})$ | `gate_cphase(state, control, target, phi)` |

## Three-qubit gates

### Toffoli (CCNOT)

Flips target if both controls are $\lvert1\rangle$:

[archived fence delimiter: ```c]
gate_toffoli(state, control1, control2, target);
[archived fence delimiter: ```]

### Fredkin (CSWAP)

Swaps two qubits if control is $\lvert1\rangle$:

[archived fence delimiter: ```c]
gate_fredkin(state, control, target1, target2);
[archived fence delimiter: ```]

## Multi-controlled gates

Variadic-control X and Z, with the control list passed as an array:

[archived fence delimiter: ```c]
int controls[] = {0, 1, 2, 3};
gate_mcx(state, controls, 4 /* num_controls */, 5 /* target */);
gate_mcz(state, controls, 4, 5);
[archived fence delimiter: ```]

Declared at `src/quantum/gates.h:` (search `gate_mcx`, `gate_mcz`).

## Quantum Fourier transform

`gate_qft` and `gate_iqft` operate on a contiguous or arbitrary
subset of qubits, supplied as an index array:

$$\text{QFT}\lvert j\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} e^{2\pi ijk/N} \lvert k\rangle$$

[archived fence delimiter: ```c]
int qubits[] = {0, 1, 2, 3};
gate_qft(state, qubits, 4);
gate_iqft(state, qubits, 4);
[archived fence delimiter: ```]

## Gate identities

### Conjugation

$$HXH = Z, \quad HYH = -Y, \quad HZH = X.$$
$$S X S^\dagger = Y, \quad S Y S^\dagger = -X.$$

### Commutation

$$[X, Y] = 2iZ, \quad [Y, Z] = 2iX, \quad [Z, X] = 2iY.$$

### CNOT relations

$$\text{CNOT} = (I \otimes H) \cdot \text{CZ} \cdot (I \otimes H), \qquad \text{CNOT}^2 = I.$$

## Bloch-sphere action

For $\lvert\psi\rangle = \cos\tfrac{\theta}{2}\lvert0\rangle + e^{i\phi}\sin\tfrac{\theta}{2}\lvert1\rangle$:

| Gate | Bloch-sphere action |
|------|----------------------|
| X    | 180° rotation about X-axis  |
| Y    | 180° rotation about Y-axis  |
| Z    | 180° rotation about Z-axis  |
| H    | 180° rotation about (X+Z)/√2 |
| S    | 90° rotation about Z-axis   |
| T    | 45° rotation about Z-axis   |
| R<sub>x</sub>(θ) | θ rotation about X-axis   |
| R<sub>y</sub>(θ) | θ rotation about Y-axis   |
| R<sub>z</sub>(θ) | θ rotation about Z-axis   |

## Language bindings

The Python bindings expose the gate set as methods on
`moonlab.QuantumState` (see `bindings/python/moonlab/core.py:252`
onwards): `state.h(q)`, `state.x(q)`, `state.y(q)`, `state.z(q)`,
`state.s(q)`, `state.sdg(q)`, `state.t(q)`, `state.tdg(q)`,
`state.rx(q, theta)`, `state.ry(q, theta)`, `state.rz(q, theta)`,
`state.cnot(c, t)`, etc.  Method names are short forms of the C
symbols above; the underlying ABI call is identical.

The Rust bindings live in `bindings/rust/moonlab/src/state.rs` and
mirror the C signatures one-to-one inside `impl QuantumState`.

## See also

- [Quantum gates (concepts)](../concepts/quantum-gates.md) -- full
  mathematical treatment.
- [C API: gates header](../../src/quantum/gates.h) -- canonical
  signatures.
- [Python API: core](../api/python/core.md) -- Python wrapper
  surface.
- [Error codes](error-codes.md) -- meaning of the `qs_error_t`
  return values.
```
