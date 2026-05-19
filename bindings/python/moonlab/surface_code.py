"""Surface code (Clifford-tableau variant) Python binding.

Wraps ``src/algorithms/topological/topological.{c,h}`` exposing the
polynomial-time stabiliser-formalism surface code.  The Clifford-
tableau back-end (``surface_code_clifford_t``) scales as ``O(d^2)``
rather than ``O(2^(d^2))`` for the dense state-vector variant; that's
what makes threshold sweeps on ``d in {3, 5, 7}`` tractable.

Implements:

- rotated-lattice surface code with ``d x d`` data qubits;
- ``(d - 1)^2`` Z-type and ``(d - 1)^2`` X-type stabilisers,
  measured ancilla-mediated through CNOT + Hadamard;
- single-qubit Pauli error injection (X / Y / Z) for syndrome
  sampling and threshold sweeps.

Decoding is *not* part of this surface yet: callers receive the raw
syndrome data and plug their own decoder (e.g. ``pymatching``).  The
existing ``tests/test_surface_code_threshold.c`` runs the same
stabiliser layer underneath.

Example:

    >>> from moonlab.surface_code import SurfaceCode
    >>> code = SurfaceCode(distance=3, rng_seed=42)
    >>> code.apply_error(code.data_index(1, 1), 'X')
    >>> code.measure_z_syndromes()
    >>> code.syndrome_weight() > 0
    True

@since v0.5.13
"""

from __future__ import annotations

import ctypes
from typing import Literal

from .core import _lib


# ---- FFI signatures ------------------------------------------------

_lib.surface_code_clifford_create.argtypes = [ctypes.c_uint32, ctypes.c_uint64]
_lib.surface_code_clifford_create.restype = ctypes.c_void_p

_lib.surface_code_clifford_free.argtypes = [ctypes.c_void_p]
_lib.surface_code_clifford_free.restype = None

_lib.surface_code_clifford_data_index.argtypes = [
    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
]
_lib.surface_code_clifford_data_index.restype = ctypes.c_uint32

_lib.surface_code_clifford_apply_error.argtypes = [
    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_char,
]
_lib.surface_code_clifford_apply_error.restype = ctypes.c_int

_lib.surface_code_clifford_measure_z_syndromes.argtypes = [ctypes.c_void_p]
_lib.surface_code_clifford_measure_z_syndromes.restype = ctypes.c_int

_lib.surface_code_clifford_measure_x_syndromes.argtypes = [ctypes.c_void_p]
_lib.surface_code_clifford_measure_x_syndromes.restype = ctypes.c_int

_lib.surface_code_clifford_syndrome_weight.argtypes = [ctypes.c_void_p]
_lib.surface_code_clifford_syndrome_weight.restype = ctypes.c_uint32


PauliErrorType = Literal['X', 'Y', 'Z']


def _check(rc: int, where: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{where} returned rc={rc}")


class SurfaceCode:
    """Rotated surface code at distance ``d`` (data qubits = ``d^2``).

    Parameters
    ----------
    distance : int
        Code distance.  Must be odd and ``>= 3`` (the C side
        rejects even or trivial distances).
    rng_seed : int
        splitmix64 seed for the underlying Clifford-tableau RNG,
        used by ancilla-mediated stabiliser measurements.

    Notes
    -----
    Each ancilla-mediated measurement consumes one bit of RNG state
    when a stabiliser anticommutes; the state advances internally
    so consecutive calls are independent.  Pass a fixed ``rng_seed``
    for reproducibility.
    """

    __slots__ = ("_handle", "_distance")

    def __init__(self, distance: int, rng_seed: int = 0):
        if distance < 3 or distance % 2 == 0:
            raise ValueError(
                f"surface code distance must be odd and >= 3, got {distance}"
            )
        h = _lib.surface_code_clifford_create(distance, rng_seed)
        if not h:
            raise MemoryError("surface_code_clifford_create returned NULL")
        self._handle = ctypes.c_void_p(h)
        self._distance = distance

    def __del__(self):
        if getattr(self, "_handle", None) and self._handle.value:
            _lib.surface_code_clifford_free(self._handle)
            self._handle = ctypes.c_void_p(0)

    # ---- Introspection -------------------------------------------------

    @property
    def distance(self) -> int:
        """Code distance ``d``."""
        return self._distance

    @property
    def num_data_qubits(self) -> int:
        """Number of physical data qubits = ``d^2``."""
        return self._distance * self._distance

    @property
    def num_ancillas_per_sector(self) -> int:
        """Ancillas in each parity sector = ``(d - 1)^2``."""
        m = self._distance - 1
        return m * m

    # ---- Lattice geometry ----------------------------------------------

    def data_index(self, row: int, col: int) -> int:
        """Linear data-qubit index from ``(row, col)`` lattice coordinates.

        Both indices range over ``[0, d)``.  Raises :class:`IndexError`
        on out-of-range inputs.
        """
        if not (0 <= row < self._distance and 0 <= col < self._distance):
            raise IndexError(
                f"(row, col) = ({row}, {col}) out of [0, {self._distance})"
            )
        return _lib.surface_code_clifford_data_index(self._handle, row, col)

    # ---- Error injection -----------------------------------------------

    def apply_error(self, qubit: int, error_type: PauliErrorType) -> None:
        """Apply a single-qubit Pauli error ``X``, ``Y``, or ``Z``."""
        if not (0 <= qubit < self.num_data_qubits):
            raise IndexError(
                f"data qubit {qubit} out of [0, {self.num_data_qubits})"
            )
        if error_type not in ('X', 'Y', 'Z'):
            raise ValueError(
                f"error_type must be 'X', 'Y', or 'Z', got {error_type!r}"
            )
        rc = _lib.surface_code_clifford_apply_error(
            self._handle, qubit, error_type.encode('ascii'),
        )
        _check(rc, "surface_code_clifford_apply_error")

    # ---- Syndrome measurement ------------------------------------------

    def measure_z_syndromes(self) -> None:
        """Measure all ``(d - 1)^2`` Z-type stabilisers (ZZZZ on
        four data qubits around each interior vertex)."""
        rc = _lib.surface_code_clifford_measure_z_syndromes(self._handle)
        _check(rc, "surface_code_clifford_measure_z_syndromes")

    def measure_x_syndromes(self) -> None:
        """Measure all ``(d - 1)^2`` X-type stabilisers (XXXX on
        four data qubits around each interior face)."""
        rc = _lib.surface_code_clifford_measure_x_syndromes(self._handle)
        _check(rc, "surface_code_clifford_measure_x_syndromes")

    def syndrome_weight(self) -> int:
        """Set-bit count across both X and Z syndromes (diagnostic)."""
        return _lib.surface_code_clifford_syndrome_weight(self._handle)


__all__ = ["SurfaceCode"]
