#!/usr/bin/env python3
"""
Matrix-product density operator (MPDO) noise simulation: Python demo.

This example exercises the v0.3 ``moonlab.mpdo.Mpdo`` surface.  It
reproduces three canonical analytical results from the operator-sum
representation of single-qubit noise [1, 2] and demonstrates how to
compose unitary gates with noise channels using arbitrary
user-supplied Kraus operators.

References
----------
[1] M. A. Nielsen and I. L. Chuang, Quantum Computation and Quantum
    Information, 10th anniversary ed., Cambridge University Press,
    2010, ch. 8.
[2] F. Verstraete, J. J. Garcia-Ripoll, and J. I. Cirac,
    Phys. Rev. Lett. 93, 207204 (2004).

Run::

    python3 examples/applications/mpdo_noise_demo.py
"""

from __future__ import annotations

import math

import numpy as np

from moonlab.mpdo import Mpdo


def banner(title: str) -> None:
    bar = "=" * (len(title) + 4)
    print(f"\n{bar}\n  {title}\n{bar}")


def demo_initial_state() -> None:
    banner("1. Initial state |0...0><0...0| at chi = 1")
    rho = Mpdo(num_qubits=4, max_bond_dim=16)
    print(f"  qubits           = {rho.num_qubits}")
    print(f"  max_bond_dim     = {rho.max_bond_dim}")
    print(f"  current_bond_dim = {rho.current_bond_dim}")
    print(f"  Tr(rho)          = {rho.trace():.12f}  (expected 1)")
    for q in range(rho.num_qubits):
        z = rho.expect_pauli(q, "Z")
        print(f"  <Z_{q}>            = {z:+.12f}  (expected +1)")


def demo_depolarising() -> None:
    banner("2. Depolarising channel: <Z> -> 1 - 4 p / 3")
    print(f"  {'p':>5}  {'<Z>':>12}  {'1 - 4p/3':>12}  {'|err|':>10}")
    for p in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        rho = Mpdo(1, 16)
        rho.apply_depolarizing(0, p)
        z = rho.expect_pauli(0, "Z")
        ref = 1.0 - 4.0 * p / 3.0
        print(f"  {p:5.2f}  {z:+12.6f}  {ref:+12.6f}  {abs(z - ref):.2e}")


def demo_amplitude_damping() -> None:
    banner("3. Amplitude damping (T_1) from |1>")
    print("  After bit-flip(p=1) we are at |1><1|, then amplitude-damp.")
    print(f"  {'gamma':>5}  {'<Z>':>12}  {'2 gamma - 1':>12}  {'|err|':>10}")
    for gamma in (0.0, 0.25, 0.5, 0.75, 1.0):
        rho = Mpdo(1, 16)
        rho.apply_bit_flip(0, 1.0)               # |0><0| -> |1><1|
        rho.apply_amplitude_damping(0, gamma)
        z = rho.expect_pauli(0, "Z")
        # diag(1 - gamma) on |1><1| -> rho_11 = 1 - gamma, rho_00 = gamma.
        # <Z> = rho_00 - rho_11 = 2 gamma - 1.
        ref = 2.0 * gamma - 1.0
        print(f"  {gamma:5.2f}  {z:+12.6f}  {ref:+12.6f}  {abs(z - ref):.2e}")


def demo_phase_damping_with_hadamard() -> None:
    banner("4. Hadamard + phase damping (T_2): preserves <Z>, contracts <X>")
    H = (1.0 / math.sqrt(2.0)) * np.array(
        [[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128
    )
    print("  Initial: H|0> = |+>; <X> = +1, <Z> = 0.")
    print(f"  {'lambda':>6}  {'<X>':>12}  {'sqrt(1-l)':>12}  {'<Z>':>12}")
    for lam in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        rho = Mpdo(2, 16)
        rho.apply_kraus(qubit=0, kraus=H)
        rho.apply_phase_damping(0, lam)
        x = rho.expect_pauli(0, "X")
        z = rho.expect_pauli(0, "Z")
        print(f"  {lam:6.2f}  {x:+12.6f}  {math.sqrt(1 - lam):+12.6f}  {z:+12.6f}")


def demo_clone_independence() -> None:
    banner("5. Clone independence")
    rho = Mpdo(2, 16)
    rho.apply_depolarizing(0, 0.3)
    sigma = rho.clone()
    sigma.apply_amplitude_damping(0, 1.0)        # reset clone to |0>
    print(f"  rho:   <Z_0> = {rho.expect_pauli(0, 'Z'):+.6f}  (depolarised)")
    print(f"  sigma: <Z_0> = {sigma.expect_pauli(0, 'Z'):+.6f}  (reset to |0>)")


def main() -> None:
    demo_initial_state()
    demo_depolarising()
    demo_amplitude_damping()
    demo_phase_damping_with_hadamard()
    demo_clone_independence()
    print()


if __name__ == "__main__":
    main()
