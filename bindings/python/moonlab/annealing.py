"""Complete transverse-field quantum annealing bindings (Moonlab v1.2.1)."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

from .core import _lib


class AnnealError(RuntimeError):
    """Native annealing input, allocation, schedule, or evolution failure."""


class AnnealSchedule(IntEnum):
    LINEAR = 0
    QUADRATIC = 1
    COSINE = 2


@dataclass(frozen=True)
class AnnealConfig:
    total_time: float = 10.0
    num_steps: int = 1000
    num_samples: int = 1024
    seed: int = 0
    schedule: AnnealSchedule = AnnealSchedule.COSINE
    driver_strength: float = 1.0
    problem_strength: float = 1.0
    second_order: bool = True


@dataclass(frozen=True)
class AnnealResult:
    num_qubits: int
    effective_seed: int
    best_bitstring: int
    most_likely_bitstring: int
    ground_bitstring: int
    ground_degeneracy: int
    best_energy: float
    ground_energy: float
    expected_energy: float
    success_probability: float
    residual_energy: float
    problem_gap: float
    final_norm: float
    samples: list[int]
    sample_energies: list[float]


class _CConfig(ctypes.Structure):
    _fields_ = [
        ("total_time", ctypes.c_double),
        ("num_steps", ctypes.c_size_t),
        ("num_samples", ctypes.c_size_t),
        ("seed", ctypes.c_uint64),
        ("schedule", ctypes.c_int),
        ("driver_strength", ctypes.c_double),
        ("problem_strength", ctypes.c_double),
        ("second_order", ctypes.c_int),
    ]


class _CResult(ctypes.Structure):
    pass


_ResultPtr = ctypes.POINTER(_CResult)
_DoublePtr = ctypes.POINTER(ctypes.c_double)

_lib.moonlab_quantum_anneal_ising.argtypes = [
    ctypes.c_size_t,
    _DoublePtr,
    _DoublePtr,
    ctypes.c_double,
    ctypes.POINTER(_CConfig),
    ctypes.POINTER(_ResultPtr),
]
_lib.moonlab_quantum_anneal_ising.restype = ctypes.c_int
_lib.moonlab_quantum_anneal_qubo.argtypes = [
    ctypes.c_size_t,
    _DoublePtr,
    ctypes.c_double,
    ctypes.POINTER(_CConfig),
    ctypes.POINTER(_ResultPtr),
]
_lib.moonlab_quantum_anneal_qubo.restype = ctypes.c_int
_lib.moonlab_qubo_to_ising.argtypes = [
    ctypes.c_size_t,
    _DoublePtr,
    ctypes.c_double,
    _DoublePtr,
    _DoublePtr,
    _DoublePtr,
]
_lib.moonlab_qubo_to_ising.restype = ctypes.c_int
_lib.moonlab_anneal_result_free.argtypes = [_ResultPtr]
_lib.moonlab_anneal_result_free.restype = None

_UINT64_GETTERS = (
    "effective_seed",
    "best_bitstring",
    "most_likely_bitstring",
    "ground_bitstring",
)
for _name in _UINT64_GETTERS:
    _fn = getattr(_lib, f"moonlab_anneal_result_{_name}")
    _fn.argtypes = [_ResultPtr]
    _fn.restype = ctypes.c_uint64
for _name in ("num_qubits", "num_samples", "ground_degeneracy"):
    _fn = getattr(_lib, f"moonlab_anneal_result_{_name}")
    _fn.argtypes = [_ResultPtr]
    _fn.restype = ctypes.c_size_t
for _name in (
    "best_energy",
    "ground_energy",
    "expected_energy",
    "success_probability",
    "residual_energy",
    "problem_gap",
    "final_norm",
):
    _fn = getattr(_lib, f"moonlab_anneal_result_{_name}")
    _fn.argtypes = [_ResultPtr]
    _fn.restype = ctypes.c_double
_lib.moonlab_anneal_result_sample.argtypes = [
    _ResultPtr,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_double),
]
_lib.moonlab_anneal_result_sample.restype = ctypes.c_int


def _matrix(values: Sequence[Sequence[float]], name: str) -> tuple[int, ctypes.Array]:
    rows = [list(row) for row in values]
    n = len(rows)
    if n == 0 or any(len(row) != n for row in rows):
        raise AnnealError(f"{name} must be a non-empty square matrix")
    flat = [float(value) for row in rows for value in row]
    return n, (ctypes.c_double * (n * n))(*flat)


def _cconfig(config: AnnealConfig | None) -> _CConfig:
    c = config or AnnealConfig()
    if not 0 <= int(c.seed) <= (1 << 64) - 1:
        raise AnnealError("seed must be in [0, 2^64-1]")
    return _CConfig(
        float(c.total_time),
        int(c.num_steps),
        int(c.num_samples),
        int(c.seed),
        int(c.schedule),
        float(c.driver_strength),
        float(c.problem_strength),
        int(bool(c.second_order)),
    )


def _collect(ptr: _ResultPtr) -> AnnealResult:
    try:
        n_samples = int(_lib.moonlab_anneal_result_num_samples(ptr))
        samples: list[int] = []
        energies: list[float] = []
        for index in range(n_samples):
            bits = ctypes.c_uint64()
            energy = ctypes.c_double()
            rc = _lib.moonlab_anneal_result_sample(
                ptr, index, ctypes.byref(bits), ctypes.byref(energy)
            )
            if rc != 0:
                raise AnnealError(f"result sample {index} failed with status {rc}")
            samples.append(int(bits.value))
            energies.append(float(energy.value))

        def get(name: str):
            return getattr(_lib, f"moonlab_anneal_result_{name}")(ptr)

        return AnnealResult(
            num_qubits=int(get("num_qubits")),
            effective_seed=int(get("effective_seed")),
            best_bitstring=int(get("best_bitstring")),
            most_likely_bitstring=int(get("most_likely_bitstring")),
            ground_bitstring=int(get("ground_bitstring")),
            ground_degeneracy=int(get("ground_degeneracy")),
            best_energy=float(get("best_energy")),
            ground_energy=float(get("ground_energy")),
            expected_energy=float(get("expected_energy")),
            success_probability=float(get("success_probability")),
            residual_energy=float(get("residual_energy")),
            problem_gap=float(get("problem_gap")),
            final_norm=float(get("final_norm")),
            samples=samples,
            sample_energies=energies,
        )
    finally:
        _lib.moonlab_anneal_result_free(ptr)


def anneal_ising(
    fields: Sequence[float],
    couplings: Sequence[Sequence[float]],
    *,
    offset: float = 0.0,
    config: AnnealConfig | None = None,
) -> AnnealResult:
    """Anneal a symmetric Ising model ``sum J_ij Z_i Z_j + sum h_i Z_i``."""
    h = [float(value) for value in fields]
    n, J = _matrix(couplings, "couplings")
    if len(h) != n:
        raise AnnealError("fields length must match the coupling matrix")
    h_array = (ctypes.c_double * n)(*h)
    c = _cconfig(config)
    out = _ResultPtr()
    rc = _lib.moonlab_quantum_anneal_ising(
        n, h_array, J, float(offset), ctypes.byref(c), ctypes.byref(out)
    )
    if rc != 0 or not out:
        raise AnnealError(f"moonlab_quantum_anneal_ising failed with status {rc}")
    return _collect(out)


def anneal_qubo(
    Q: Sequence[Sequence[float]], *, offset: float = 0.0, config: AnnealConfig | None = None
) -> AnnealResult:
    """Anneal the full-matrix objective ``x.T @ Q @ x + offset``."""
    n, q_array = _matrix(Q, "Q")
    c = _cconfig(config)
    out = _ResultPtr()
    rc = _lib.moonlab_quantum_anneal_qubo(
        n, q_array, float(offset), ctypes.byref(c), ctypes.byref(out)
    )
    if rc != 0 or not out:
        raise AnnealError(f"moonlab_quantum_anneal_qubo failed with status {rc}")
    return _collect(out)


def qubo_to_ising(
    Q: Sequence[Sequence[float]], *, offset: float = 0.0
) -> tuple[list[float], list[list[float]], float]:
    """Convert ``x.T @ Q @ x + offset`` to an exactly equivalent Ising model."""
    n, q_array = _matrix(Q, "Q")
    h = (ctypes.c_double * n)()
    J = (ctypes.c_double * (n * n))()
    converted_offset = ctypes.c_double()
    rc = _lib.moonlab_qubo_to_ising(n, q_array, float(offset), h, J, ctypes.byref(converted_offset))
    if rc != 0:
        raise AnnealError(f"moonlab_qubo_to_ising failed with status {rc}")
    return (
        [float(h[i]) for i in range(n)],
        [[float(J[i * n + j]) for j in range(n)] for i in range(n)],
        float(converted_offset.value),
    )


__all__ = [
    "AnnealError",
    "AnnealSchedule",
    "AnnealConfig",
    "AnnealResult",
    "anneal_ising",
    "anneal_qubo",
    "qubo_to_ising",
]
