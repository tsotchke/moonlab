# Gate Reference

Quick reference for the universal gate set in the Moonlab C core
(`src/quantum/gates.h`). Every gate returns `qs_error_t` (`QS_SUCCESS` on success; see
[Error Codes](error-codes.md)) and mutates the `quantum_state_t` in place. Qubit indices
are 0-based, with qubit 0 the least-significant bit of the basis-state index.

The Python bindings expose the same gates as methods on `QuantumState`
(`state.h(0)`, `state.cnot(0, 1)`, `state.rx(0, theta)`, ...).

## Single-Qubit Gates

| Gate | C function | Matrix |
|------|-----------|--------|
| Pauli-X | `gate_pauli_x(state, q)` | $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ |
| Pauli-Y | `gate_pauli_y(state, q)` | $\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$ |
| Pauli-Z | `gate_pauli_z(state, q)` | $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ |
| Hadamard | `gate_hadamard(state, q)` | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ |
| S ($\sqrt{Z}$) | `gate_s(state, q)` | $\begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$ |
| S† | `gate_s_dagger(state, q)` | $\begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}$ |
| T ($\pi/8$) | `gate_t(state, q)` | $\begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$ |
| T† | `gate_t_dagger(state, q)` | $\begin{pmatrix} 1 & 0 \\ 0 & e^{-i\pi/4} \end{pmatrix}$ |
| Phase | `gate_phase(state, q, θ)` | $\begin{pmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{pmatrix}$ |
| Rx | `gate_rx(state, q, θ)` | $\cos\frac{\theta}{2} I - i\sin\frac{\theta}{2} X$ |
| Ry | `gate_ry(state, q, θ)` | $\cos\frac{\theta}{2} I - i\sin\frac{\theta}{2} Y$ |
| Rz | `gate_rz(state, q, θ)` | $\mathrm{diag}(e^{-i\theta/2}, e^{i\theta/2})$ |
| U3 | `gate_u3(state, q, θ, φ, λ)` | General single-qubit unitary $U_3(\theta, \phi, \lambda)$ |

## Two-Qubit Gates

| Gate | C function | Effect |
|------|-----------|--------|
| CNOT | `gate_cnot(state, control, target)` | Flip `target` when `control` is $\lvert 1\rangle$ |
| CZ | `gate_cz(state, control, target)` | Apply Z to `target` when `control` is $\lvert 1\rangle$ |
| CY | `gate_cy(state, control, target)` | Apply Y to `target` when `control` is $\lvert 1\rangle$ |
| SWAP | `gate_swap(state, q1, q2)` | Exchange the two qubits |
| Controlled-Phase | `gate_cphase(state, control, target, θ)` | Phase $e^{i\theta}$ on the $\lvert 11\rangle$ subspace |
| Controlled-Rx | `gate_crx(state, control, target, θ)` | Apply $R_x(\theta)$ to `target` when `control` is $\lvert 1\rangle$ |
| Controlled-Ry | `gate_cry(state, control, target, θ)` | Apply $R_y(\theta)$ to `target` when `control` is $\lvert 1\rangle$ |
| Controlled-Rz | `gate_crz(state, control, target, θ)` | Apply $R_z(\theta)$ to `target` when `control` is $\lvert 1\rangle$ |

## Three-Qubit Gates

| Gate | C function | Effect |
|------|-----------|--------|
| Toffoli (CCX) | `gate_toffoli(state, c1, c2, target)` | Flip `target` when both controls are $\lvert 1\rangle$ |
| Fredkin (CSWAP) | `gate_fredkin(state, control, t1, t2)` | Swap `t1`/`t2` when `control` is $\lvert 1\rangle$ |

## Multi-Qubit Gates

| Gate | C function | Effect |
|------|-----------|--------|
| Multi-controlled X | `gate_mcx(state, controls, num_controls, target)` | Flip `target` when all controls are $\lvert 1\rangle$ |
| Multi-controlled Z | `gate_mcz(state, controls, num_controls, target)` | Apply Z when all controls are $\lvert 1\rangle$ |
| QFT | `gate_qft(state, qubits, num_qubits)` | Quantum Fourier transform over the listed qubits |
| Inverse QFT | `gate_iqft(state, qubits, num_qubits)` | Inverse quantum Fourier transform |

## Arbitrary Unitaries

| Function | Purpose |
|----------|---------|
| `apply_single_qubit_gate(state, q, matrix[2][2])` | Apply an arbitrary 2x2 unitary |
| `apply_two_qubit_gate(state, q1, q2, matrix[4][4])` | Apply an arbitrary 4x4 unitary |
| `verify_gate_normalization(state)` | Returns 1 if the post-gate state is normalized |

## See Also

- [C API: Gates](../api/c/gates.md) - Full function reference
- [Error Codes](error-codes.md) - `qs_error_t` return values
- [Quantum Gates (concepts)](../concepts/quantum-gates.md) - Theory and intuition
