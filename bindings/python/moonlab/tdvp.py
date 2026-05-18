"""Moonlab time-dependent variational principle (TDVP) bindings.

Wraps the v0.4 adaptive-bond two-site TDVP engine in
``src/algorithms/tensor_network/tdvp.{c,h}`` so Python users can
evolve MPS states in real or imaginary time with the
entropy-feedback PID bond-dimension controller.

The Python surface is deliberately minimal: a configuration class
that mirrors :c:type:`tdvp_config_t` field-for-field, factory
helpers that match the C API (``TdvpConfig.default()`` and
``TdvpConfig.adaptive(eps)``), an opaque :class:`TdvpEngine` handle
that owns the C engine + the per-bond PID state, and a
:class:`TdvpResult` dataclass that surfaces the per-step energy,
norm, truncation error, and bond-chi distribution.

Hamiltonian and initial-state construction stay in C; the Python
module ships convenience builders for the two MPOs we know are
stable in v0.3.1 -- ``mpo_heisenberg`` and ``mpo_tfim`` -- as
opaque handles, plus :func:`random_mps` for a random initial MPS.
Users who need richer MPO/MPS surfaces should drop to the C ABI
directly until those bindings land.

References:
- arXiv:2604.03960 -- entropy-feedback bond control for 2TDVP
  (the algorithm).
- Haegeman et al., Phys. Rev. B 94, 165116 (2016) -- the
  underlying two-site TDVP integrator.

Example::

    from moonlab.tdvp import (
        TdvpConfig, TdvpEngine, EvolutionType,
        mpo_heisenberg, random_mps,
    )

    mpo = mpo_heisenberg(num_sites=8, J=1.0, Delta=1.0, h=0.0)
    mps = random_mps(num_sites=8, chi_init=8, max_bond_dim=32)

    config = TdvpConfig.adaptive(target_entropy_error=1e-3)
    config.evolution_type = EvolutionType.IMAGINARY_TIME
    config.dt = 0.05

    engine = TdvpEngine(mps, mpo, config)
    for step in range(30):
        result = engine.step()
        print(f"step {step}: E = {result.energy:+.6f}, "
              f"chi = {result.bond_chi_distribution.tolist()}")
"""

from __future__ import annotations

import ctypes
import enum
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .core import _lib


__all__ = [
    "EvolutionType",
    "Variant",
    "IntegratorType",
    "TdvpAdaptiveBondConfig",
    "TdvpConfig",
    "TdvpResult",
    "TdvpEngine",
    "Mpo",
    "Mps",
    "mpo_heisenberg",
    "mpo_tfim",
    "random_mps",
]


# ---------------------------------------------------------------------------
# Enums (mirror src/algorithms/tensor_network/tdvp.h)
# ---------------------------------------------------------------------------


class EvolutionType(enum.IntEnum):
    """Direction of TDVP evolution."""

    REAL_TIME = 0
    IMAGINARY_TIME = 1


class Variant(enum.IntEnum):
    """One-site (fixed chi) vs two-site (adaptive chi) TDVP."""

    ONE_SITE = 0
    TWO_SITE = 1


class IntegratorType(enum.IntEnum):
    """Time-integrator backend selector."""

    LANCZOS = 0
    RUNGE_KUTTA = 1
    EXPOKIT = 2


# ---------------------------------------------------------------------------
# FFI structs (mirror the C layout exactly)
# ---------------------------------------------------------------------------


class _CAdaptiveBondConfig(ctypes.Structure):
    _fields_ = [
        ("enabled", ctypes.c_bool),
        ("target_entropy_error", ctypes.c_double),
        ("kp", ctypes.c_double),
        ("ki", ctypes.c_double),
        ("kd", ctypes.c_double),
        ("chi_floor", ctypes.c_uint32),
        ("chi_ceiling", ctypes.c_uint32),
        ("alpha", ctypes.c_double),
    ]


class _CTdvpConfig(ctypes.Structure):
    _fields_ = [
        ("evolution_type", ctypes.c_int),
        ("variant", ctypes.c_int),
        ("integrator", ctypes.c_int),
        ("dt", ctypes.c_double),
        ("max_bond_dim", ctypes.c_uint32),
        ("svd_cutoff", ctypes.c_double),
        ("lanczos_max_iter", ctypes.c_uint32),
        ("lanczos_tol", ctypes.c_double),
        ("normalize", ctypes.c_bool),
        ("verbose", ctypes.c_bool),
        ("adaptive_bond", _CAdaptiveBondConfig),
    ]


class _CTdvpResult(ctypes.Structure):
    _fields_ = [
        ("time", ctypes.c_double),
        ("energy", ctypes.c_double),
        ("norm", ctypes.c_double),
        ("truncation_error", ctypes.c_double),
        ("max_bond_dim", ctypes.c_uint32),
        ("step_time", ctypes.c_double),
        ("bond_chi_distribution", ctypes.POINTER(ctypes.c_uint32)),
        ("n_bonds", ctypes.c_uint32),
    ]


# ---------------------------------------------------------------------------
# FFI signatures
# ---------------------------------------------------------------------------


_lib.mpo_heisenberg_create.argtypes = [
    ctypes.c_uint32, ctypes.c_double, ctypes.c_double, ctypes.c_double,
]
_lib.mpo_heisenberg_create.restype = ctypes.c_void_p

_lib.mpo_tfim_create.argtypes = [
    ctypes.c_uint32, ctypes.c_double, ctypes.c_double,
]
_lib.mpo_tfim_create.restype = ctypes.c_void_p

_lib.mpo_free.argtypes = [ctypes.c_void_p]
_lib.mpo_free.restype = None


class _CStateConfig(ctypes.Structure):
    # `tn_state_config_t` carries more fields, but the public C helper
    # `tn_state_config_create(max_bond, cutoff)` populates a value that
    # we pass straight back into `dmrg_init_random_mps` -- ctypes only
    # needs to see the storage size, which we expose generously.
    _fields_ = [("_storage", ctypes.c_uint8 * 256)]


_lib.tn_state_config_create.argtypes = [ctypes.c_uint32, ctypes.c_double]
_lib.tn_state_config_create.restype = _CStateConfig

_lib.dmrg_init_random_mps.argtypes = [
    ctypes.c_uint32, ctypes.c_uint32,
    ctypes.POINTER(_CStateConfig),
]
_lib.dmrg_init_random_mps.restype = ctypes.c_void_p

_lib.tn_mps_free.argtypes = [ctypes.c_void_p]
_lib.tn_mps_free.restype = None

_lib.tdvp_engine_create.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(_CTdvpConfig),
]
_lib.tdvp_engine_create.restype = ctypes.c_void_p

_lib.tdvp_engine_free.argtypes = [ctypes.c_void_p]
_lib.tdvp_engine_free.restype = None

_lib.tdvp_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CTdvpResult)]
_lib.tdvp_step.restype = ctypes.c_int

_lib.tdvp_result_clear.argtypes = [ctypes.POINTER(_CTdvpResult)]
_lib.tdvp_result_clear.restype = None

_lib.tdvp_bond_chi.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
_lib.tdvp_bond_chi.restype = ctypes.c_uint32


# ---------------------------------------------------------------------------
# Python-facing configuration
# ---------------------------------------------------------------------------


@dataclass
class TdvpAdaptiveBondConfig:
    """Per-bond PID controller knobs (mirrors
    :c:type:`tdvp_adaptive_bond_config_t`).

    When ``enabled`` is ``False`` the legacy fixed-bond path is
    used and the remaining fields are ignored.
    """

    enabled: bool = False
    target_entropy_error: float = 0.0
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    chi_floor: int = 0
    chi_ceiling: int = 0
    alpha: float = 0.0

    @classmethod
    def reference(cls, target_entropy_error: float) -> "TdvpAdaptiveBondConfig":
        """Reference-paper PID gains from arXiv:2604.03960."""
        return cls(
            enabled=True,
            target_entropy_error=target_entropy_error,
            kp=0.5,
            ki=0.05,
            kd=0.1,
            chi_floor=4,
            chi_ceiling=4096,
            alpha=8.0,
        )

    def _to_c(self) -> _CAdaptiveBondConfig:
        return _CAdaptiveBondConfig(
            enabled=self.enabled,
            target_entropy_error=self.target_entropy_error,
            kp=self.kp, ki=self.ki, kd=self.kd,
            chi_floor=self.chi_floor,
            chi_ceiling=self.chi_ceiling,
            alpha=self.alpha,
        )


@dataclass
class TdvpConfig:
    """TDVP configuration (mirrors :c:type:`tdvp_config_t`).

    Use :meth:`default` for the legacy fixed-bond v0.3 surface, or
    :meth:`adaptive` to switch on the v0.4 entropy-feedback PID
    controller at the reference-paper gains.
    """

    evolution_type: EvolutionType = EvolutionType.REAL_TIME
    variant: Variant = Variant.TWO_SITE
    integrator: IntegratorType = IntegratorType.LANCZOS
    dt: float = 0.01
    max_bond_dim: int = 128
    svd_cutoff: float = 1e-10
    lanczos_max_iter: int = 50
    lanczos_tol: float = 1e-12
    normalize: bool = True
    verbose: bool = False
    adaptive_bond: TdvpAdaptiveBondConfig = field(
        default_factory=TdvpAdaptiveBondConfig)

    @classmethod
    def default(cls) -> "TdvpConfig":
        """Legacy fixed-bond config; bit-identical to v0.3.1
        ``tdvp_config_default()``."""
        return cls()

    @classmethod
    def adaptive(cls, target_entropy_error: float) -> "TdvpConfig":
        """Adaptive-bond config at the reference-paper PID gains.

        ``max_bond_dim`` is raised to the adaptive ``chi_ceiling`` so
        the outer-bound safety still applies; ``svd_cutoff`` keeps
        its default of ``1e-10`` as the singular-value noise floor.
        """
        ab = TdvpAdaptiveBondConfig.reference(target_entropy_error)
        cfg = cls(adaptive_bond=ab)
        cfg.max_bond_dim = ab.chi_ceiling
        return cfg

    def _to_c(self) -> _CTdvpConfig:
        return _CTdvpConfig(
            evolution_type=int(self.evolution_type),
            variant=int(self.variant),
            integrator=int(self.integrator),
            dt=self.dt,
            max_bond_dim=self.max_bond_dim,
            svd_cutoff=self.svd_cutoff,
            lanczos_max_iter=self.lanczos_max_iter,
            lanczos_tol=self.lanczos_tol,
            normalize=self.normalize,
            verbose=self.verbose,
            adaptive_bond=self.adaptive_bond._to_c(),
        )


@dataclass
class TdvpResult:
    """Output of one :meth:`TdvpEngine.step` call."""

    time: float
    energy: float
    norm: float
    truncation_error: float
    max_bond_dim: int
    step_time: float
    bond_chi_distribution: np.ndarray  # shape (n_bonds,), dtype uint32


# ---------------------------------------------------------------------------
# Opaque handles for MPS / MPO
# ---------------------------------------------------------------------------


class Mpo:
    """Opaque ctypes handle around a C ``mpo_t``.

    Build via :func:`mpo_heisenberg` or :func:`mpo_tfim`.
    """

    __slots__ = ("_handle", "_num_sites")

    def __init__(self, handle: int, num_sites: int):
        if not handle:
            raise MemoryError("MPO handle is NULL")
        self._handle = handle
        self._num_sites = num_sites

    def __del__(self) -> None:
        h = getattr(self, "_handle", None)
        if h:
            _lib.mpo_free(h)
            self._handle = None

    @property
    def num_sites(self) -> int:
        return self._num_sites


class Mps:
    """Opaque ctypes handle around a C ``tn_mps_state_t``.

    Build via :func:`random_mps`.  The Moonlab TDVP engine takes
    ownership of the underlying buffers; do not free the MPS while
    a :class:`TdvpEngine` is using it.
    """

    __slots__ = ("_handle", "_num_sites")

    def __init__(self, handle: int, num_sites: int):
        if not handle:
            raise MemoryError("MPS handle is NULL")
        self._handle = handle
        self._num_sites = num_sites

    def __del__(self) -> None:
        h = getattr(self, "_handle", None)
        if h:
            _lib.tn_mps_free(h)
            self._handle = None

    @property
    def num_sites(self) -> int:
        return self._num_sites


def mpo_heisenberg(num_sites: int,
                   J: float = 1.0,
                   Delta: float = 1.0,
                   h: float = 0.0) -> Mpo:
    """Build the XXZ Heisenberg MPO
    ``H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Delta Z_i Z_{i+1})
        + h * sum_i Z_i``."""
    handle = _lib.mpo_heisenberg_create(
        ctypes.c_uint32(num_sites),
        ctypes.c_double(J),
        ctypes.c_double(Delta),
        ctypes.c_double(h),
    )
    if not handle:
        raise MemoryError("mpo_heisenberg_create returned NULL")
    return Mpo(handle, num_sites)


def mpo_tfim(num_sites: int, J: float = 1.0, h: float = 1.0) -> Mpo:
    """Build the transverse-field Ising MPO
    ``H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i``."""
    handle = _lib.mpo_tfim_create(
        ctypes.c_uint32(num_sites),
        ctypes.c_double(J),
        ctypes.c_double(h),
    )
    if not handle:
        raise MemoryError("mpo_tfim_create returned NULL")
    return Mpo(handle, num_sites)


def random_mps(num_sites: int,
               chi_init: int = 8,
               max_bond_dim: int = 32,
               cutoff: float = 1e-12) -> Mps:
    """Allocate a random MPS with bulk bond dimension ``chi_init``
    and a state-level ``max_bond_dim`` envelope.

    Useful as the starting point for TDVP evolution.  The MPS is
    not normalised; the engine will call ``tn_mps_norm`` and
    renormalise on the first step if
    ``TdvpConfig.normalize == True``.
    """
    cfg = _lib.tn_state_config_create(
        ctypes.c_uint32(max_bond_dim),
        ctypes.c_double(cutoff),
    )
    handle = _lib.dmrg_init_random_mps(
        ctypes.c_uint32(num_sites),
        ctypes.c_uint32(chi_init),
        ctypes.byref(cfg),
    )
    if not handle:
        raise MemoryError("dmrg_init_random_mps returned NULL")
    return Mps(handle, num_sites)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TdvpEngine:
    """Owned wrapper around a C ``tdvp_engine_t``.

    Args:
        mps:    Initial state.  The engine borrows the MPS pointer
                for its lifetime; the caller must keep the
                :class:`Mps` alive until the engine is freed.
        mpo:    Hamiltonian.  Same borrowing rule as ``mps``.
        config: TDVP configuration.

    Example::

        config = TdvpConfig.adaptive(target_entropy_error=1e-3)
        config.evolution_type = EvolutionType.IMAGINARY_TIME
        config.dt = 0.05
        engine = TdvpEngine(mps, mpo, config)
        for _ in range(30):
            result = engine.step()
    """

    __slots__ = ("_handle", "_mps", "_mpo", "_result")

    def __init__(self, mps: Mps, mpo: Mpo, config: TdvpConfig):
        c_cfg = config._to_c()
        handle = _lib.tdvp_engine_create(
            mps._handle, mpo._handle, ctypes.byref(c_cfg),
        )
        if not handle:
            raise MemoryError("tdvp_engine_create returned NULL")
        self._handle = handle
        # Keep references so the borrowed handles stay alive.
        self._mps = mps
        self._mpo = mpo
        self._result = _CTdvpResult()

    def __del__(self) -> None:
        h = getattr(self, "_handle", None)
        if h:
            # Free the per-step result's heap-owned distribution
            # buffer (engine_free does not touch the result).
            r = getattr(self, "_result", None)
            if r is not None:
                _lib.tdvp_result_clear(ctypes.byref(r))
            _lib.tdvp_engine_free(h)
            self._handle = None

    def step(self) -> TdvpResult:
        """Advance the state by one TDVP step (`dt`).

        Returns a :class:`TdvpResult` snapshot.  The internal C
        result buffer is reused across calls, so the returned
        ``bond_chi_distribution`` is a copy.
        """
        rc = _lib.tdvp_step(self._handle, ctypes.byref(self._result))
        if rc != 0:
            raise RuntimeError(f"tdvp_step failed (rc={rc})")
        return self._snapshot()

    def bond_chi(self, bond: int) -> int:
        """Per-bond chi from the adaptive controller.

        Returns 0 when the controller is disabled.
        """
        return int(_lib.tdvp_bond_chi(self._handle, ctypes.c_uint32(bond)))

    def _snapshot(self) -> TdvpResult:
        r = self._result
        n = int(r.n_bonds)
        if r.bond_chi_distribution and n > 0:
            arr = np.ctypeslib.as_array(r.bond_chi_distribution,
                                         shape=(n,)).copy()
        else:
            arr = np.zeros(0, dtype=np.uint32)
        return TdvpResult(
            time=float(r.time),
            energy=float(r.energy),
            norm=float(r.norm),
            truncation_error=float(r.truncation_error),
            max_bond_dim=int(r.max_bond_dim),
            step_time=float(r.step_time),
            bond_chi_distribution=arr,
        )
