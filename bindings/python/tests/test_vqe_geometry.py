"""VQE parameter-space geometry and pointwise band geometry.

Exercises the binding layer for the quantum geometric tensor, the Berry
curvature and the natural-gradient direction, plus the ABI 0.7.0
band-geometry one-shots. Every assertion is a property the C
implementation cannot fake: the metric is symmetric positive
semidefinite, the curvature is antisymmetric with a zero diagonal, and
the natural-gradient direction solves ``(g + eps I) x = grad``.

The shared fixture is the 2-qubit H2 Hamiltonian under a 1-layer
hardware-efficient ansatz -- 4 parameters -- driven by the quantum
natural gradient optimizer.
"""

import ctypes

import numpy as np
import pytest

from moonlab.algorithms import VQE
from moonlab.core import QuantumError, _lib
from moonlab.topology import (
    BandTouchingError,
    dsigma_metric_curvature,
    haldane_curvature_at,
    qwz_curvature_at,
)


# Parameter vector the geometry is evaluated at. Fixed, not random: the
# C evaluation is analytic and noise-free, so the tensors are functions
# of theta alone.
THETA = np.array([0.31, -0.87, 1.24, 0.55])


@pytest.fixture(scope="module")
def h2_solver():
    """2-qubit H2, 1-layer hardware-efficient ansatz, QNG optimizer.

    ``solve_h2`` is what constructs the C solver handle the geometry
    entry points need, so it runs once for the whole module.
    """
    vqe = VQE(num_qubits=2, num_layers=1, optimizer="qng")
    vqe.solve_h2(bond_distance=0.74)
    return vqe


class TestVQEGeometryShape:
    """Parameter counts and array shapes."""

    def test_hardware_efficient_ansatz_has_four_parameters(self, h2_solver):
        # 2 qubits x 1 layer x 2 rotations per qubit.
        assert h2_solver.num_parameters == 4

    def test_tensors_are_square_and_finite(self, h2_solver):
        n = h2_solver.num_parameters
        g = h2_solver.quantum_geometric_tensor(THETA)
        f = h2_solver.berry_curvature(THETA)
        assert g.shape == (n, n)
        assert f.shape == (n, n)
        assert g.dtype == np.float64
        assert f.dtype == np.float64
        assert np.all(np.isfinite(g))
        assert np.all(np.isfinite(f))

    def test_parameter_count_mismatch_raises(self, h2_solver):
        wrong = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        with pytest.raises(ValueError):
            h2_solver.quantum_geometric_tensor(wrong)
        with pytest.raises(ValueError):
            h2_solver.berry_curvature(wrong)
        with pytest.raises(ValueError):
            h2_solver.natural_gradient_direction(wrong, wrong)
        with pytest.raises(ValueError):
            h2_solver.natural_gradient_direction(THETA, np.array([0.1, 0.2]))

    def test_geometry_requires_a_solver(self):
        # A VQE that has never solved carries no C solver handle.
        vqe = VQE(num_qubits=2, num_layers=1, optimizer="qng")
        with pytest.raises(QuantumError):
            vqe.quantum_geometric_tensor(THETA)


class TestQuantumGeometricTensor:
    """The metric must be symmetric and positive semidefinite."""

    def test_metric_is_symmetric(self, h2_solver):
        g = h2_solver.quantum_geometric_tensor(THETA)
        asym = np.max(np.abs(g - g.T))
        assert asym < 1e-9, f"metric asymmetry {asym:.3e}"

    def test_metric_is_positive_semidefinite(self, h2_solver):
        g = h2_solver.quantum_geometric_tensor(THETA)
        # Symmetric eigenvalues: none may be negative beyond round-off.
        eigenvalues = np.linalg.eigvalsh(0.5 * (g + g.T))
        assert eigenvalues.min() > -1e-10, (
            f"metric has a negative eigenvalue: {eigenvalues}"
        )

    def test_metric_quadratic_form_is_nonnegative(self, h2_solver):
        # Direct check on random directions, independent of the
        # eigensolver above.
        g = h2_solver.quantum_geometric_tensor(THETA)
        rng = np.random.default_rng(20260730)
        v = rng.uniform(-1.0, 1.0, size=(256, g.shape[0]))
        q = np.einsum("ki,ij,kj->k", v, g, v)
        assert q.min() > -1e-10, f"min v^T g v = {q.min():.3e}"


class TestBerryCurvature:
    """The curvature must be antisymmetric with a zero diagonal."""

    def test_curvature_is_antisymmetric(self, h2_solver):
        f = h2_solver.berry_curvature(THETA)
        sym = np.max(np.abs(f + f.T))
        assert sym < 1e-9, f"curvature symmetry residual {sym:.3e}"

    def test_curvature_diagonal_is_zero(self, h2_solver):
        f = h2_solver.berry_curvature(THETA)
        diag = np.max(np.abs(np.diag(f)))
        assert diag < 1e-12, f"nonzero curvature diagonal {diag:.3e}"


class TestNaturalGradient:
    """The direction must solve the regularized metric system."""

    def test_direction_solves_regularized_metric_system(self, h2_solver):
        n = h2_solver.num_parameters
        g = h2_solver.quantum_geometric_tensor(THETA)
        gradient = 0.1 * np.arange(1, n + 1, dtype=np.float64)
        eps = 1e-2

        direction = h2_solver.natural_gradient_direction(
            THETA, gradient, regularization=eps
        )
        assert direction.shape == (n,)

        residual = np.max(
            np.abs((g + eps * np.eye(n)) @ direction - gradient)
        )
        assert residual < 1e-8, f"residual {residual:.3e}"

    def test_large_regularization_damps_the_step(self, h2_solver):
        # (g + eps I)^-1 grad -> grad / eps as eps grows, so a huge shift
        # must shrink the direction relative to a small one.
        gradient = np.array([0.1, 0.2, 0.3, 0.4])
        small = h2_solver.natural_gradient_direction(
            THETA, gradient, regularization=1e-3
        )
        large = h2_solver.natural_gradient_direction(
            THETA, gradient, regularization=1e3
        )
        assert np.linalg.norm(large) < np.linalg.norm(small)


class TestBandGeometryOneShots:
    """Pointwise Fubini-Study metric and Berry curvature (ABI 0.7.0)."""

    K = (0.83, -1.47)

    def test_qwz_curvature_at_returns_finite_geometry(self):
        g, omega = qwz_curvature_at(1.0, self.K)
        assert g.shape == (2, 2)
        assert g.dtype == np.float64
        assert np.all(np.isfinite(g))
        assert np.isfinite(omega)
        # Metric symmetric PSD; curvature nonzero in a Chern phase.
        assert abs(g[0, 1] - g[1, 0]) < 1e-15
        assert np.linalg.eigvalsh(g).min() > -1e-15
        assert omega != 0.0

    def test_qwz_curvature_at_raw_return_code_is_zero(self):
        # Pin the C contract the ctypes declaration marshals, not just
        # the Python translation of it.
        k = (ctypes.c_double * 2)(*self.K)
        g = (ctypes.c_double * 4)()
        omega = ctypes.c_double(0.0)
        rc = _lib.moonlab_qwz_curvature_at(
            ctypes.c_double(1.0), k, g, ctypes.byref(omega)
        )
        assert rc == 0

    def test_qwz_band_touching_returns_minus_two(self):
        # m = -2 puts a Dirac point at k = (0, 0): gap closed, geometry
        # undefined, contract says -2.
        k = (ctypes.c_double * 2)(0.0, 0.0)
        g = (ctypes.c_double * 4)()
        omega = ctypes.c_double(0.0)
        rc = _lib.moonlab_qwz_curvature_at(
            ctypes.c_double(-2.0), k, g, ctypes.byref(omega)
        )
        assert rc == -2

    def test_qwz_band_touching_raises(self):
        with pytest.raises(BandTouchingError):
            qwz_curvature_at(-2.0, (0.0, 0.0))

    def test_dsigma_matches_qwz_at_the_same_k(self):
        # Feeding the analytic QWZ d-vector to the d.sigma entry must
        # reproduce the QWZ one-shot bit for bit.
        m = 1.0
        kx, ky = self.K
        d = (np.sin(kx), np.sin(ky), m + np.cos(kx) + np.cos(ky))
        dx = (np.cos(kx), 0.0, -np.sin(kx))
        dy = (0.0, np.cos(ky), -np.sin(ky))

        g_d, omega_d = dsigma_metric_curvature(d, dx, dy)
        g_q, omega_q = qwz_curvature_at(m, self.K)

        assert np.array_equal(g_d, g_q)
        assert omega_d == omega_q

    def test_dsigma_band_touching_raises(self):
        with pytest.raises(BandTouchingError):
            dsigma_metric_curvature((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                    (0.0, 1.0, 0.0))

    def test_haldane_curvature_at_returns_finite_geometry(self):
        g, omega = haldane_curvature_at(1.0, 0.1, np.pi / 2, 0.0, (0.4, 0.9))
        assert g.shape == (2, 2)
        assert np.all(np.isfinite(g))
        assert np.isfinite(omega)
        assert abs(g[0, 1] - g[1, 0]) < 1e-15
