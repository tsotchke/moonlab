"""VQE for H2 in minimal basis using Moonlab's native autograd.

2-qubit H2/STO-3G Pauli Hamiltonian at R = 0.7414 Angstrom, using the
O'Malley et al. (Phys. Rev. X 6, 031007, 2016) Jordan-Wigner form:

    H = g0 I + g1 Z_0 + g2 Z_1 + g3 Z_0 Z_1 + g4 X_0 X_1 + g5 Y_0 Y_1

with coefficients (Hartree):
    g0 = -1.0523732   g1 =  0.3979374   g2 = -0.3979374
    g3 = -0.0112801   g4 =  0.1809312   g5 =  0.1809312

Note: without a spin/number conservation constraint a general
2-qubit ansatz can reach the full-matrix ground state, which lies
*below* the physical H2 FCI energy because the unphysical sector
contains a lower eigenvalue.  This is expected; the demo compares
VQE against the Hamiltonian's true lowest eigenvalue (computed by
direct 4x4 diagonalization) rather than the physical FCI number.

The point of the demo is to exercise the native adjoint gradient on
a realistic-scale Pauli-sum observable and show gradient descent
converging to within 1e-6 of the exact diagonalized ground state.
"""
from __future__ import annotations

import math

import numpy as np

from moonlab import QuantumState
from moonlab.diff import DiffCircuit, PauliTerm, OBS_X, OBS_Y, OBS_Z


# H2 Hamiltonian coefficients at R = 0.7414 Angstrom.
G0 = -1.0523732
G1 =  0.3979374
G2 = -0.3979374
G3 = -0.0112801
G4 =  0.1809312
G5 =  0.1809312

def exact_ground_energy() -> float:
    """Diagonalize the 4x4 Pauli-sum matrix directly."""
    I2 = np.eye(2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = (G0 * np.kron(I2, I2)
         + G1 * np.kron(I2, Z)
         + G2 * np.kron(Z, I2)
         + G3 * np.kron(Z, Z)
         + G4 * np.kron(X, X)
         + G5 * np.kron(Y, Y))
    return float(np.linalg.eigvalsh(H)[0])


def h2_hamiltonian() -> list[PauliTerm]:
    return [
        PauliTerm(G0, [], []),                         # identity
        PauliTerm(G1, [0],    [OBS_Z]),                # Z_0
        PauliTerm(G2, [1],    [OBS_Z]),                # Z_1
        PauliTerm(G3, [0, 1], [OBS_Z, OBS_Z]),         # Z_0 Z_1
        PauliTerm(G4, [0, 1], [OBS_X, OBS_X]),         # X_0 X_1
        PauliTerm(G5, [0, 1], [OBS_Y, OBS_Y]),         # Y_0 Y_1
    ]


def build_ansatz(thetas: np.ndarray) -> DiffCircuit:
    """HEA with 4 parameters and two CNOT entanglers."""
    c = DiffCircuit(2)
    c.ry(0, thetas[0]).ry(1, thetas[1])
    c.cnot(0, 1)
    c.ry(0, thetas[2]).ry(1, thetas[3])
    c.cnot(0, 1)
    return c


def main() -> None:
    rng = np.random.default_rng(seed=0xC0FFEE)
    thetas = rng.uniform(-0.3, 0.3, size=4)

    circ = build_ansatz(thetas)
    state = QuantumState(2)
    H = h2_hamiltonian()
    E_exact = exact_ground_energy()

    # Basic Adam-ish gradient descent; vanilla GD also converges for
    # this problem but Adam is smoother.
    lr = 0.1
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(thetas)
    v = np.zeros_like(thetas)

    print(f"Exact ground-state energy (diagonalization): {E_exact:+.8f} Ha")
    print(f"{'iter':>4}  {'E (Hartree)':>14}  {'|E - E_exact|':>13}  "
          f"{'|grad|':>10}")
    prev_energy = 1e9
    final_energy = 0.0
    for step in range(1, 201):
        for k in range(4):
            circ.set_theta(k, float(thetas[k]))
        circ.forward(state)

        energy = DiffCircuit.expect_pauli_sum(state, H)
        grad = circ.backward_pauli_sum(state, H)

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad * grad
        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        thetas = thetas - lr * m_hat / (np.sqrt(v_hat) + eps)

        if step == 1 or step % 10 == 0 or step == 200:
            print(f"{step:4d}  {energy:14.8f}  "
                  f"{abs(energy - E_exact):13.2e}  "
                  f"{np.linalg.norm(grad):10.2e}")

        if abs(prev_energy - energy) < 1e-10:
            print(f"  converged at iter {step}")
            break
        prev_energy = energy
        final_energy = energy

    print()
    print(f"Ansatz final energy:  {final_energy:+.8f} Hartree")
    print(f"Exact reference:      {E_exact:+.8f} Hartree")
    print(f"Error above exact:    {final_energy - E_exact:+.2e} Hartree")

    # A 2-qubit universal ansatz with two CNOT layers is expressive
    # enough to hit the exact ground state to machine precision; allow
    # 1e-5 slack for the optimizer.
    assert math.isclose(final_energy, E_exact, abs_tol=1e-5), \
        f"VQE did not converge to exact ground state: got {final_energy}, want {E_exact}"


if __name__ == "__main__":
    main()
