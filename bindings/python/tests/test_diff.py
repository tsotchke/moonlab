"""Python-level smoke + numeric tests for the autograd bindings."""
from __future__ import annotations

import math

import numpy as np
import pytest

from moonlab import QuantumState
from moonlab.diff import DiffCircuit, PauliTerm, OBS_Z, OBS_X


def test_single_qubit_ry_analytic():
    """<Z> on RY(theta)|0> = cos(theta); d<Z>/dtheta = -sin(theta)."""
    theta = 0.7
    circ = DiffCircuit(1)
    circ.ry(0, theta)

    s = QuantumState(1)
    circ.forward(s)
    z = DiffCircuit.expect_z(s, 0)
    assert math.isclose(z, math.cos(theta), abs_tol=1e-12)

    g = circ.backward(s, OBS_Z, 0)
    assert g.shape == (1,)
    assert math.isclose(g[0], -math.sin(theta), abs_tol=1e-10)


def test_hea_two_qubit_vs_finite_diff():
    """Two-qubit HEA ansatz: adjoint <Z0> grad matches central diff."""
    thetas = [0.15, -0.42, 1.23, 0.88, -0.33]
    circ = DiffCircuit(2)
    circ.ry(0, thetas[0]).rz(1, thetas[1]).h(0).cnot(0, 1)
    circ.ry(0, thetas[2]).rz(0, thetas[3]).ry(1, thetas[4])
    assert circ.num_parameters == 5

    s = QuantumState(2)
    circ.forward(s)
    grad_adj = circ.backward(s, OBS_Z, 0)

    # Central-difference grad.
    h = 1e-4
    grad_fd = np.zeros(5)
    for k in range(5):
        circ.set_theta(k, thetas[k] + h)
        circ.forward(s)
        fp = DiffCircuit.expect_z(s, 0)
        circ.set_theta(k, thetas[k] - h)
        circ.forward(s)
        fm = DiffCircuit.expect_z(s, 0)
        circ.set_theta(k, thetas[k])
        grad_fd[k] = (fp - fm) / (2.0 * h)

    np.testing.assert_allclose(grad_adj, grad_fd, atol=1e-7)


def test_pauli_sum_h2_like():
    """5-term Hamiltonian, adjoint vs finite-diff.

    H = 0.5 I + 0.3 Z_0 + 0.4 Z_1 - 0.2 Z_0 Z_1 + 0.15 X_0 X_1
    """
    thetas = [0.35, -0.22, 0.88, 1.11]
    circ = DiffCircuit(2)
    circ.ry(0, thetas[0]).ry(1, thetas[1]).cnot(0, 1)
    circ.ry(0, thetas[2]).ry(1, thetas[3])

    terms = [
        PauliTerm(0.5,  [],     []),                     # I
        PauliTerm(0.3,  [0],    [OBS_Z]),
        PauliTerm(0.4,  [1],    [OBS_Z]),
        PauliTerm(-0.2, [0, 1], [OBS_Z, OBS_Z]),
        PauliTerm(0.15, [0, 1], [OBS_X, OBS_X]),
    ]

    s = QuantumState(2)
    circ.forward(s)
    H_val = DiffCircuit.expect_pauli_sum(s, terms)
    # sum of |c| = 1.55 is the absolute bound.
    assert abs(H_val) < 1.6

    grad_adj = circ.backward_pauli_sum(s, terms)
    assert grad_adj.shape == (4,)

    h = 1e-4
    grad_fd = np.zeros(4)
    for k in range(4):
        circ.set_theta(k, thetas[k] + h)
        circ.forward(s)
        fp = DiffCircuit.expect_pauli_sum(s, terms)
        circ.set_theta(k, thetas[k] - h)
        circ.forward(s)
        fm = DiffCircuit.expect_pauli_sum(s, terms)
        circ.set_theta(k, thetas[k])
        grad_fd[k] = (fp - fm) / (2.0 * h)

    np.testing.assert_allclose(grad_adj, grad_fd, atol=1e-7)


def test_gradient_descent_drives_cost_down():
    """Simple GD loop -- cost strictly decreases toward an analytic min."""
    thetas = np.array([0.2, -0.3])
    circ = DiffCircuit(2)
    circ.ry(0, thetas[0]).ry(1, thetas[1]).cnot(0, 1)
    # H = <Z_0> + 0.5 <Z_0 Z_1>.  Minimum is -1.5 at theta_0 = pi, any theta_1.
    terms = [
        PauliTerm(1.0, [0],    [OBS_Z]),
        PauliTerm(0.5, [0, 1], [OBS_Z, OBS_Z]),
    ]

    s = QuantumState(2)
    lr = 0.3
    cost_prev = 1e9
    for _ in range(60):
        circ.set_theta(0, thetas[0])
        circ.set_theta(1, thetas[1])
        circ.forward(s)
        cost = DiffCircuit.expect_pauli_sum(s, terms)
        assert cost <= cost_prev + 1e-6  # non-increasing up to noise
        grad = circ.backward_pauli_sum(s, terms)
        thetas -= lr * grad
        cost_prev = cost

    # Should land at -1.5 to within ~1e-3 after 60 iterations.
    assert abs(cost_prev - (-1.5)) < 1e-3
