"""Pytest suite for the v0.4 adaptive-bond TDVP Python bindings.

Validates the Python wrapper of
``src/algorithms/tensor_network/tdvp.{c,h}`` end-to-end: config
construction, engine lifecycle, real-time energy conservation, and
imaginary-time convergence toward the DMRG ground state.  Numerical
tolerances match the C-side acceptance tests; the Python layer is a
thin ctypes shim around the same engine.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import moonlab
from moonlab.tdvp import (
    EvolutionType,
    IntegratorType,
    Mpo,
    Mps,
    TdvpAdaptiveBondConfig,
    TdvpConfig,
    TdvpEngine,
    TdvpHistory,
    Variant,
    mpo_heisenberg,
    mpo_tfim,
    random_mps,
)


# ---- Configuration tests --------------------------------------------------


def test_default_config_leaves_adaptive_disabled():
    cfg = TdvpConfig.default()
    assert cfg.adaptive_bond.enabled is False
    assert cfg.max_bond_dim == 128
    assert cfg.dt == 0.01
    assert cfg.evolution_type == EvolutionType.REAL_TIME
    assert cfg.variant == Variant.TWO_SITE
    assert cfg.integrator == IntegratorType.LANCZOS


def test_adaptive_config_carries_reference_gains():
    cfg = TdvpConfig.adaptive(target_entropy_error=1e-3)
    assert cfg.adaptive_bond.enabled is True
    assert cfg.adaptive_bond.target_entropy_error == 1e-3
    assert cfg.adaptive_bond.kp == 0.5
    assert cfg.adaptive_bond.ki == 0.05
    assert cfg.adaptive_bond.kd == 0.1
    assert cfg.adaptive_bond.chi_floor == 4
    assert cfg.adaptive_bond.chi_ceiling == 4096
    assert cfg.max_bond_dim == cfg.adaptive_bond.chi_ceiling


def test_adaptive_bond_config_disabled_helper():
    ab = TdvpAdaptiveBondConfig()
    assert not ab.enabled
    assert ab.target_entropy_error == 0.0


# ---- Lifecycle ------------------------------------------------------------


def test_mpo_heisenberg_and_mps_lifecycle():
    mpo = mpo_heisenberg(num_sites=6, J=1.0, Delta=1.0, h=0.0)
    mps = random_mps(num_sites=6, chi_init=4, max_bond_dim=16)
    assert mpo.num_sites == 6
    assert mps.num_sites == 6
    # Drop references -- the __del__ hooks free the C handles.


def test_engine_create_and_one_step():
    mpo = mpo_heisenberg(num_sites=6, J=1.0, Delta=1.0, h=0.0)
    mps = random_mps(num_sites=6, chi_init=4, max_bond_dim=16)
    cfg = TdvpConfig.default()
    cfg.evolution_type = EvolutionType.REAL_TIME
    cfg.dt = 0.02
    cfg.max_bond_dim = 16

    engine = TdvpEngine(mps, mpo, cfg)
    result = engine.step()
    assert math.isfinite(result.energy)
    assert math.isfinite(result.norm)
    assert result.max_bond_dim >= 1
    # Legacy path -- the result's distribution should be empty.
    assert result.bond_chi_distribution.shape == (0,)


def test_adaptive_engine_populates_bond_chi_distribution():
    n = 8
    mpo = mpo_heisenberg(num_sites=n)
    mps = random_mps(num_sites=n, chi_init=8, max_bond_dim=32)
    cfg = TdvpConfig.adaptive(target_entropy_error=1e-3)
    cfg.evolution_type = EvolutionType.REAL_TIME
    cfg.dt = 0.02
    cfg.adaptive_bond.chi_ceiling = 32
    cfg.max_bond_dim = 32

    engine = TdvpEngine(mps, mpo, cfg)
    result = engine.step()
    assert result.bond_chi_distribution.shape == (n - 1,)
    assert result.bond_chi_distribution.dtype == np.uint32
    floor = cfg.adaptive_bond.chi_floor
    ceil = cfg.adaptive_bond.chi_ceiling
    for chi in result.bond_chi_distribution.tolist():
        assert chi == 0 or floor <= chi <= ceil
    # Accessor must agree with the distribution.
    for b in range(n - 1):
        assert engine.bond_chi(b) == int(result.bond_chi_distribution[b])


# ---- Physics: real-time energy conservation -------------------------------


def test_real_time_energy_conservation():
    """Mirror tests/unit/test_tdvp_adaptive_energy_conservation.c."""
    n = 8
    mpo = mpo_heisenberg(num_sites=n)
    mps = random_mps(num_sites=n, chi_init=8, max_bond_dim=32)

    cfg = TdvpConfig.adaptive(target_entropy_error=1e-3)
    cfg.evolution_type = EvolutionType.REAL_TIME
    cfg.dt = 0.02
    cfg.adaptive_bond.chi_ceiling = 32
    cfg.max_bond_dim = 32

    engine = TdvpEngine(mps, mpo, cfg)
    energies = []
    for _ in range(5):
        energies.append(engine.step().energy)

    E0 = energies[0]
    max_drift = max(abs(E - E0) / max(abs(E0), 1e-12) for E in energies)
    assert max_drift < 5e-3, f"max relative drift {max_drift:.3e}"


# ---- Physics: imag-time convergence to DMRG ground state ------------------


def test_imag_time_tfim_converges_toward_ground_state():
    """Smaller-scope sibling of
    tests/unit/test_tdvp_adaptive_tfim_ground.c."""
    n = 8
    mpo = mpo_tfim(num_sites=n, J=1.0, h=1.0)
    mps = random_mps(num_sites=n, chi_init=8, max_bond_dim=32)

    cfg = TdvpConfig.adaptive(target_entropy_error=1e-3)
    cfg.evolution_type = EvolutionType.IMAGINARY_TIME
    cfg.dt = 0.05
    cfg.adaptive_bond.chi_ceiling = 32
    cfg.max_bond_dim = 32

    engine = TdvpEngine(mps, mpo, cfg)
    E_first = engine.step().energy
    E_last = E_first
    for _ in range(29):
        E_last = engine.step().energy

    # The 8-site critical-TFIM DMRG ground-state energy is around
    # -9.84.  Don't require full convergence; just assert the
    # imag-time evolution drove the energy below the first step's
    # value by a meaningful margin.
    assert E_last < E_first - 0.1, (
        f"E_first={E_first:.4f}, E_last={E_last:.4f} "
        f"(imag-time should have dropped energy)"
    )
    # Bond distribution still inside the envelope.
    last = engine.step()
    floor = cfg.adaptive_bond.chi_floor
    ceil = cfg.adaptive_bond.chi_ceiling
    for chi in last.bond_chi_distribution.tolist():
        assert chi == 0 or floor <= chi <= ceil


# ---- Module exports -------------------------------------------------------


def test_module_exports():
    assert hasattr(moonlab, "tdvp")
    from moonlab import tdvp as _t
    for name in [
        "TdvpConfig", "TdvpEngine", "TdvpResult", "TdvpHistory",
        "TdvpAdaptiveBondConfig", "EvolutionType",
        "Variant", "IntegratorType",
        "mpo_heisenberg", "mpo_tfim", "random_mps",
        "Mpo", "Mps",
    ]:
        assert hasattr(_t, name), f"moonlab.tdvp is missing {name}"


# ---- History + evolve_to --------------------------------------------------


def test_evolve_to_records_history():
    n = 6
    mpo = mpo_tfim(num_sites=n, J=1.0, h=1.0)
    mps = random_mps(num_sites=n, chi_init=8, max_bond_dim=16)

    cfg = TdvpConfig.adaptive(target_entropy_error=1e-3)
    cfg.evolution_type = EvolutionType.IMAGINARY_TIME
    cfg.dt = 0.05
    cfg.adaptive_bond.chi_ceiling = 16
    cfg.max_bond_dim = 16

    engine = TdvpEngine(mps, mpo, cfg)
    history = TdvpHistory(initial_capacity=8)
    assert history.num_steps == 0
    assert history.observables is None
    assert history.bond_chi_history is None

    engine.evolve_to(target_time=0.2, history=history)

    assert history.num_steps >= 3
    assert history.times.shape == (history.num_steps,)
    assert history.energies.shape == (history.num_steps,)
    assert history.norms.shape == (history.num_steps,)
    # times must be strictly increasing.
    diffs = np.diff(history.times)
    assert np.all(diffs > 0), f"times not monotonic: {history.times}"
    # Adaptive controller: per-bond chi recorded.
    chi_hist = history.bond_chi_history
    assert chi_hist is not None
    assert chi_hist.shape == (history.num_steps, n - 1)
    assert chi_hist.dtype == np.uint32
    # observables column should still be unallocated.
    assert history.observables is None


def test_evolve_with_observable_records_callback_values():
    n = 6
    mpo = mpo_tfim(num_sites=n, J=1.0, h=1.0)
    mps = random_mps(num_sites=n, chi_init=8, max_bond_dim=16)

    cfg = TdvpConfig.adaptive(target_entropy_error=1e-3)
    cfg.evolution_type = EvolutionType.IMAGINARY_TIME
    cfg.dt = 0.05
    cfg.adaptive_bond.chi_ceiling = 16
    cfg.max_bond_dim = 16

    engine = TdvpEngine(mps, mpo, cfg)
    history = TdvpHistory(initial_capacity=8)

    seen_times = []

    def observable(_mps_handle: int, time: float) -> float:
        seen_times.append(time)
        # Trivial observable: the time itself.  Pins the value
        # round-trip across the ctypes callback boundary.
        return time

    engine.evolve_with_observable(
        target_time=0.2, history=history, observable_fn=observable,
    )

    assert history.num_steps >= 3
    obs = history.observables
    assert obs is not None
    assert obs.shape == (history.num_steps,)
    # Observable values must equal the recorded times to within
    # roundoff (the callback returned the time verbatim).
    assert np.allclose(obs, history.times, atol=1e-12), (
        f"observables {obs.tolist()} != times {history.times.tolist()}"
    )
    # Callback was actually invoked at every step.
    assert len(seen_times) >= history.num_steps
