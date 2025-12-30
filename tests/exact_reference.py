#!/usr/bin/env python3
"""
Exact numerical reference for tensor network simulation.
Computes expected values using full state vector for small N.

This matches the C code's Trotter decomposition EXACTLY:
  1. ZZ gates on even bonds (0-1, 2-3, ...)
  2. ZZ gates on odd bonds (1-2, 3-4, ...)
  3. X gates on all qubits

Gate definitions:
  ZZ gate: e^{τJ ZZ} = diag(e^{τJ}, e^{-τJ}, e^{-τJ}, e^{τJ})
  X gate:  e^{τh X} = [[cosh(τh), sinh(τh)], [sinh(τh), cosh(τh)]]
"""
import numpy as np

# Parameters (must match C code)
J = 1.0
h = 0.5
tau = 0.1
tilt = 0.05

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_n(ops):
    """Kronecker product of list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

def create_initial_state(n):
    """Create tilted product state matching C code."""
    # |ψ⟩ = (cos(tilt)|0⟩ + sin(tilt)|1⟩)^⊗n
    single = np.array([np.cos(tilt), np.sin(tilt)], dtype=complex)
    state = single
    for _ in range(n - 1):
        state = np.kron(state, single)
    return state / np.linalg.norm(state)

def create_zz_gate():
    """e^{τJ ZZ} - diagonal in computational basis."""
    ep = np.exp(tau * J)
    em = np.exp(-tau * J)
    return np.diag([ep, em, em, ep]).astype(complex)

def create_x_gate():
    """e^{τh X} for single qubit."""
    c = np.cosh(tau * h)
    s = np.sinh(tau * h)
    return np.array([[c, s], [s, c]], dtype=complex)

def apply_2q_gate(state, n, i, j, gate):
    """Apply 2-qubit gate to qubits i,j in n-qubit state."""
    # Reshape state to tensor form
    state = state.reshape([2] * n)

    # For adjacent qubits i, i+1, we can reshape and apply directly
    if j == i + 1:
        # Reshape to (..., 2, 2, ...) and apply gate
        shape_before = 2**i
        shape_after = 2**(n - j - 1)
        state = state.reshape(shape_before, 4, shape_after)
        state = np.einsum('ij,ajb->aib', gate, state)
        state = state.reshape(2**n)
    else:
        raise ValueError("Only adjacent qubits supported")

    return state

def apply_1q_gate(state, n, i, gate):
    """Apply 1-qubit gate to qubit i in n-qubit state."""
    state = state.reshape([2] * n)
    # Move target qubit to front, apply gate, move back
    state = np.moveaxis(state, i, 0)
    original_shape = state.shape
    state = state.reshape(2, -1)
    state = gate @ state
    state = state.reshape(original_shape)
    state = np.moveaxis(state, 0, i)
    return state.reshape(2**n)

def apply_trotter_step(state, n):
    """
    Apply one Trotter step matching C code EXACTLY:
    1. ZZ gates on even bonds
    2. ZZ gates on odd bonds
    3. X gates on all qubits
    4. Normalize
    """
    zz_gate = create_zz_gate()
    x_gate = create_x_gate()

    # 1. ZZ on even bonds: (0,1), (2,3), (4,5), ...
    for i in range(0, n - 1, 2):
        state = apply_2q_gate(state, n, i, i + 1, zz_gate)

    # 2. ZZ on odd bonds: (1,2), (3,4), (5,6), ...
    for i in range(1, n - 1, 2):
        state = apply_2q_gate(state, n, i, i + 1, zz_gate)

    # 3. X on all qubits
    for i in range(n):
        state = apply_1q_gate(state, n, i, x_gate)

    # 4. Normalize (imaginary time evolution doesn't preserve norm)
    state = state / np.linalg.norm(state)

    return state

def Z_expectation(state, n, site):
    """Compute ⟨Z_site⟩."""
    state = state.reshape([2] * n)
    # Sum over all indices except site
    prob_0 = 0.0
    prob_1 = 0.0
    for idx in np.ndindex(*([2] * n)):
        amp = state[idx]
        prob = np.abs(amp)**2
        if idx[site] == 0:
            prob_0 += prob
        else:
            prob_1 += prob
    return prob_0 - prob_1

def ZZ_expectation(state, n, site1, site2):
    """Compute ⟨Z_site1 Z_site2⟩."""
    state = state.reshape([2] * n)
    expectation = 0.0
    for idx in np.ndindex(*([2] * n)):
        amp = state[idx]
        prob = np.abs(amp)**2
        z1 = 1 if idx[site1] == 0 else -1
        z2 = 1 if idx[site2] == 0 else -1
        expectation += prob * z1 * z2
    return expectation

def main():
    print("=" * 70)
    print("EXACT NUMERICAL REFERENCE FOR TENSOR NETWORK VALIDATION")
    print("=" * 70)
    print(f"\nParameters: J={J}, h={h}, tau={tau}, tilt={tilt}")
    print(f"Trotter order: ZZ_even -> ZZ_odd -> X_all -> normalize")
    print()

    # Test multiple system sizes
    for n in [4, 6, 8, 10, 12]:
        print(f"\n{'='*70}")
        print(f"N = {n} qubits (Hilbert space dim = {2**n})")
        print(f"{'='*70}")

        state = create_initial_state(n)

        # Initial values
        z_bnd = Z_expectation(state, n, 0)
        z_blk = Z_expectation(state, n, n // 2)
        zz_bnd = ZZ_expectation(state, n, 0, 1)
        zz_blk = ZZ_expectation(state, n, n // 2, n // 2 + 1)

        print(f"\nInitial: Z_bnd={z_bnd:.6f}  Z_blk={z_blk:.6f}  ZZ_bnd={zz_bnd:.6f}  ZZ_blk={zz_blk:.6f}")
        print()
        print(f"{'Step':<6} {'Z_bnd':<12} {'Z_blk':<12} {'ZZ_bnd':<12} {'ZZ_blk':<12}")
        print("-" * 54)

        for step in range(1, 6):
            state = apply_trotter_step(state, n)

            z_bnd = Z_expectation(state, n, 0)
            z_blk = Z_expectation(state, n, n // 2)
            zz_bnd = ZZ_expectation(state, n, 0, 1)
            zz_blk = ZZ_expectation(state, n, n // 2, n // 2 + 1)

            print(f"{step:<6} {z_bnd:<12.6f} {z_blk:<12.6f} {zz_bnd:<12.6f} {zz_blk:<12.6f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print("""
These are the EXACT values your MPS simulation should produce.
Any deviation indicates a bug in the tensor network code.

Key physics:
- Z values decrease: imaginary time evolution projects toward ground state
- ZZ values decrease: ground state has quantum fluctuations
- Boundary vs bulk: boundary sites have fewer neighbors, different behavior
- Larger N: bulk values should converge to thermodynamic limit

For N >= 14, exact simulation becomes impractical (2^14 = 16384 elements).
Use these small-N results to validate MPS, then trust MPS for large N.
""")

if __name__ == "__main__":
    main()
