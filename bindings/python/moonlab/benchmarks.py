"""Moonlab benchmark bindings: Quantum Volume."""

import ctypes
from dataclasses import dataclass
from typing import Optional

from .core import _lib


class _CQVResult(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_size_t),
        ("num_trials", ctypes.c_size_t),
        ("mean_hop", ctypes.c_double),
        ("stddev_hop", ctypes.c_double),
        ("lower_ci_97p5", ctypes.c_double),
        ("passed", ctypes.c_int),
    ]


_lib.quantum_volume_run.argtypes = [
    ctypes.c_size_t,          # width
    ctypes.c_size_t,          # num_trials
    ctypes.c_uint64,          # rng_seed
    ctypes.POINTER(_CQVResult),
]
_lib.quantum_volume_run.restype = ctypes.c_int


@dataclass
class QuantumVolumeResult:
    width: int
    num_trials: int
    mean_hop: float
    stddev_hop: float
    lower_ci_97p5: float
    passed: bool


def quantum_volume(width: int,
                   num_trials: int = 40,
                   seed: Optional[int] = None) -> QuantumVolumeResult:
    """Run the IBM Quantum Volume protocol on the noiseless statevector.

    For the ideal simulator the mean heavy-output probability converges
    to (1 + ln 2) / 2 ~ 0.847, and every width should pass (2/3
    threshold). Useful as an accuracy/performance benchmark.

    Args:
        width: qubit count and circuit depth (2 <= width <= 16).
        num_trials: independent random-circuit samples (>= 10).
        seed: 64-bit RNG seed; None uses a fixed default.

    Returns:
        QuantumVolumeResult with mean HOP, stddev, CI lower bound, and
        pass/fail boolean.
    """
    if width < 2 or width > 16:
        raise ValueError(f"width must be in [2, 16], got {width}")
    if num_trials < 10:
        raise ValueError(f"num_trials must be >= 10, got {num_trials}")

    out = _CQVResult()
    rc = _lib.quantum_volume_run(
        ctypes.c_size_t(width),
        ctypes.c_size_t(num_trials),
        ctypes.c_uint64(seed if seed is not None else 0),
        ctypes.byref(out),
    )
    if rc != 0:
        raise RuntimeError(f"quantum_volume_run failed (rc={rc})")

    return QuantumVolumeResult(
        width=out.width,
        num_trials=out.num_trials,
        mean_hop=out.mean_hop,
        stddev_hop=out.stddev_hop,
        lower_ci_97p5=out.lower_ci_97p5,
        passed=bool(out.passed),
    )


__all__ = ["QuantumVolumeResult", "quantum_volume"]
