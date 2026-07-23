"""Ergonomic VQE API tests.

These exercise the documented, string-driven VQE constructor and prove that
the hyperparameter keyword arguments are actually threaded through to the C
optimizer rather than being accepted and silently ignored.

The C energy evaluation is analytic (noise-free), hence deterministic given
the variational parameters, and the hardware-efficient ansatz initialises its
parameters with libc ``rand()``. Seeding libc ``rand()`` before constructing a
VQE therefore fixes the initial parameters, letting us run two solves that
differ *only* in a hyperparameter and attribute any difference in outcome to
that hyperparameter.
"""

import ctypes

import numpy as np
import pytest

from moonlab.algorithms import VQE


# Process-global libc, used to seed the rand() that the ansatz initialiser and
# the moonlab shared library both draw from.
try:
    _libc = ctypes.CDLL(None)
    _libc.srand.argtypes = [ctypes.c_uint]
    _libc.srand.restype = None
    _HAVE_LIBC = True
except Exception:  # pragma: no cover - platform without a global libc handle
    _HAVE_LIBC = False


def _seed(value: int = 1234) -> None:
    if _HAVE_LIBC:
        _libc.srand(ctypes.c_uint(value))


# String optimizer name -> expected C enum value (src/algorithms/vqe.h).
_OPTIMIZER_CASES = [
    ("cobyla", VQE.OPTIMIZER_COBYLA, 0),
    ("lbfgs", VQE.OPTIMIZER_LBFGS, 1),
    ("adam", VQE.OPTIMIZER_ADAM, 2),
    ("gradient_descent", VQE.OPTIMIZER_GRADIENT_DESCENT, 3),
    ("natural_gradient", VQE.OPTIMIZER_NATURAL_GRADIENT, 4),
    ("qng", VQE.OPTIMIZER_QNG, 4),
]


class TestOptimizerSelection:
    """String / int optimizer specs must select the correct C enum."""

    @pytest.mark.parametrize("name,const,enum_value", _OPTIMIZER_CASES)
    def test_string_selects_correct_c_enum(self, name, const, enum_value):
        vqe = VQE(num_qubits=2, num_layers=2, optimizer=name)
        # The Python class constant must equal the C enum value.
        assert const == enum_value
        # The value stored on the class must match.
        assert vqe.optimizer_type == enum_value
        # And, decisively, the value written into the C optimizer struct must
        # be the right enum (this is what selects the algorithm in vqe_solve).
        assert vqe._optimizer.contents.type == enum_value

    def test_int_optimizer_still_accepted(self):
        vqe = VQE(num_qubits=2, optimizer=VQE.OPTIMIZER_LBFGS)
        assert vqe._optimizer.contents.type == VQE.OPTIMIZER_LBFGS

    def test_legacy_optimizer_type_kwarg(self):
        vqe = VQE(num_qubits=2, optimizer_type=VQE.OPTIMIZER_COBYLA)
        assert vqe._optimizer.contents.type == VQE.OPTIMIZER_COBYLA

    def test_unknown_optimizer_raises(self):
        with pytest.raises(ValueError):
            VQE(num_qubits=2, optimizer="nonesuch")


class TestOptimizerSolvesH2:
    """Every documented optimizer, addressed by string, must solve H2."""

    @pytest.mark.parametrize("name,_const,_enum", _OPTIMIZER_CASES)
    def test_string_optimizer_solves_h2(self, name, _const, _enum):
        vqe = VQE(num_qubits=2, num_layers=2, optimizer=name)
        result = vqe.solve_h2(bond_distance=0.74)
        energy = result["energy"]
        # H2 ground state ~ -1.137 Ha; every optimizer must reach a genuine
        # variational estimate well below the -1.0 Ha mark.
        assert np.isfinite(energy)
        assert energy < -1.0, f"{name} gave E={energy}"


class TestUCCSDAnsatz:
    """ansatz='uccsd' must build and solve."""

    def test_uccsd_default_electron_count_builds(self):
        # Documented form: VQE(num_qubits=4, ansatz='uccsd').
        vqe = VQE(num_qubits=4, ansatz="uccsd")
        assert vqe.ansatz_name == "uccsd"
        assert bool(vqe._ansatz)
        # num_qubits // 2 electrons -> half filling.
        assert vqe.num_electrons == 2

    def test_uccsd_solves_lih(self):
        # LiH is the built-in 4-qubit Hamiltonian, matching a 4-qubit UCCSD
        # ansatz (the solver requires equal qubit counts).
        vqe = VQE(num_qubits=4, ansatz="uccsd", num_electrons=2,
                  optimizer="adam")
        result = vqe.solve_lih(bond_distance=1.6)
        assert np.isfinite(result["energy"])
        # Sanity: a bound molecular energy, not a blow-up.
        assert result["energy"] < 0.0

    def test_unknown_ansatz_raises(self):
        with pytest.raises(ValueError):
            VQE(num_qubits=4, ansatz="banana")


class TestHyperparametersTakeEffect:
    """Prove the hyperparameter kwargs reach the C optimizer and change the
    optimization trajectory -- not merely that they are accepted."""

    def _solve_h2(self, seed, **kwargs):
        _seed(seed)
        vqe = VQE(num_qubits=2, num_layers=2, **kwargs)
        return vqe.solve_h2(bond_distance=0.74)["energy"]

    def test_learning_rate_changes_outcome(self):
        # Identical initial parameters (same seed), gradient descent, differing
        # only in learning_rate. A sane rate converges toward the ground state;
        # an absurdly large rate diverges. If learning_rate were ignored, the C
        # optimizer would use its default for both and the energies would be
        # identical.
        good = self._solve_h2(7, optimizer="gradient_descent", learning_rate=0.05)
        bad = self._solve_h2(7, optimizer="gradient_descent", learning_rate=250.0)

        assert good != bad, "learning_rate had no effect on the trajectory"
        # The sane rate must end up lower (better) than the divergent one. (The
        # C optimizer keeps the best energy seen across all iterations, so a
        # divergent run is cushioned by whatever good point it passed through;
        # the sane rate is nonetheless strictly better.)
        assert good < bad, f"good={good}, bad={bad}"
        # And the sane rate should actually find a good H2 energy.
        assert good < -1.0

    def test_regularization_changes_qng_trajectory(self):
        # Same seed, quantum natural gradient, differing only in the QNG metric
        # regularization. A huge Tikhonov shift damps the natural-gradient step
        # (direction ~ grad / reg), so it makes far less progress than the
        # small default shift. Different reg -> different energy proves the
        # kwarg reaches vqe_natural_gradient_direction.
        default_reg = self._solve_h2(11, optimizer="natural_gradient",
                                      regularization=1e-3)
        huge_reg = self._solve_h2(11, optimizer="natural_gradient",
                                  regularization=1e6)

        assert default_reg != huge_reg, "regularization had no effect"
        # Heavy damping should leave a higher (worse) energy than the small
        # default shift.
        assert default_reg < huge_reg, f"default={default_reg}, huge={huge_reg}"

    def test_beta_hyperparameters_change_adam(self):
        # Adam with default betas vs degenerate betas (beta1=beta2=0 reduces
        # Adam to a normalized sign step). Same seed -> any difference proves
        # beta1/beta2 are threaded into the C Adam update.
        default_adam = self._solve_h2(5, optimizer="adam", learning_rate=0.05)
        odd_adam = self._solve_h2(5, optimizer="adam", learning_rate=0.05,
                                  beta1=0.0, beta2=0.0)
        assert default_adam != odd_adam, "beta1/beta2 had no effect"
