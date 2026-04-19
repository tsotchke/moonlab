"""Moonlab Clifford stabilizer backend bindings.

Exposes the Aaronson-Gottesman tableau (`src/backends/clifford/`) so
Python users can simulate arbitrary-qubit Clifford circuits in
polynomial time. Useful for large-qubit GHZ/stabilizer-code
experiments that exceed the dense simulator's 32-qubit ceiling.
"""

import ctypes
from typing import Optional, Tuple

from .core import _lib

_CLIFFORD_SUCCESS = 0


class _CTableau(ctypes.Structure):
    _fields_ = [("opaque", ctypes.c_void_p)]


_lib.clifford_tableau_create.argtypes = [ctypes.c_size_t]
_lib.clifford_tableau_create.restype = ctypes.c_void_p

_lib.clifford_tableau_free.argtypes = [ctypes.c_void_p]
_lib.clifford_tableau_free.restype = None

_lib.clifford_num_qubits.argtypes = [ctypes.c_void_p]
_lib.clifford_num_qubits.restype = ctypes.c_size_t

for _name in ("h", "s", "s_dag", "x", "y", "z"):
    _fn = getattr(_lib, f"clifford_{_name}")
    _fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _fn.restype = ctypes.c_int

for _name in ("cnot", "cz", "swap"):
    _fn = getattr(_lib, f"clifford_{_name}")
    _fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    _fn.restype = ctypes.c_int

_lib.clifford_measure.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
_lib.clifford_measure.restype = ctypes.c_int


class Clifford:
    """Clifford-only quantum simulator.

    The tableau scales to thousands of qubits (O(n^2) memory, O(n^2)
    per gate, O(n^2) per measurement). Only Clifford gates are
    supported: H, S, S_dagger, X, Y, Z, CNOT, CZ, SWAP.

    Example:
        >>> c = Clifford(100, seed=0xBADDCAFE)
        >>> c.h(0)
        >>> for q in range(1, 100):
        ...     c.cnot(0, q)
        >>> first = c.measure(0)[0]
        >>> all(c.measure(q)[0] == first for q in range(1, 100))
        True
    """

    def __init__(self, num_qubits: int, seed: Optional[int] = None) -> None:
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        handle = _lib.clifford_tableau_create(ctypes.c_size_t(num_qubits))
        if not handle:
            raise MemoryError("clifford_tableau_create returned NULL")
        self._handle = handle
        self._n = num_qubits
        self._rng = ctypes.c_uint64(seed if seed is not None else 0xDEADBEEF)

    def __del__(self) -> None:
        if getattr(self, "_handle", None):
            _lib.clifford_tableau_free(self._handle)
            self._handle = None

    @property
    def num_qubits(self) -> int:
        return self._n

    def _check(self, rc: int, op: str) -> None:
        if rc != _CLIFFORD_SUCCESS:
            raise RuntimeError(f"clifford_{op} failed (rc={rc})")

    def h(self, q: int) -> None:
        self._check(_lib.clifford_h(self._handle, ctypes.c_size_t(q)), "h")

    def s(self, q: int) -> None:
        self._check(_lib.clifford_s(self._handle, ctypes.c_size_t(q)), "s")

    def s_dag(self, q: int) -> None:
        self._check(_lib.clifford_s_dag(self._handle, ctypes.c_size_t(q)), "s_dag")

    def x(self, q: int) -> None:
        self._check(_lib.clifford_x(self._handle, ctypes.c_size_t(q)), "x")

    def y(self, q: int) -> None:
        self._check(_lib.clifford_y(self._handle, ctypes.c_size_t(q)), "y")

    def z(self, q: int) -> None:
        self._check(_lib.clifford_z(self._handle, ctypes.c_size_t(q)), "z")

    def cnot(self, ctrl: int, tgt: int) -> None:
        self._check(_lib.clifford_cnot(self._handle,
                                       ctypes.c_size_t(ctrl),
                                       ctypes.c_size_t(tgt)), "cnot")

    def cz(self, a: int, b: int) -> None:
        self._check(_lib.clifford_cz(self._handle,
                                     ctypes.c_size_t(a),
                                     ctypes.c_size_t(b)), "cz")

    def swap(self, a: int, b: int) -> None:
        self._check(_lib.clifford_swap(self._handle,
                                       ctypes.c_size_t(a),
                                       ctypes.c_size_t(b)), "swap")

    def measure(self, q: int) -> Tuple[int, int]:
        """Measure qubit q in the Z basis.

        Returns:
            (outcome, kind) where outcome is 0 or 1 and kind is
            0 (deterministic, state was already a +/- Z eigenstate)
            or 1 (random, the state had support on both outcomes).
        """
        outcome = ctypes.c_int(-1)
        kind = ctypes.c_int(-1)
        rc = _lib.clifford_measure(
            self._handle,
            ctypes.c_size_t(q),
            ctypes.byref(self._rng),
            ctypes.byref(outcome),
            ctypes.byref(kind),
        )
        self._check(rc, "measure")
        return outcome.value, kind.value


__all__ = ["Clifford"]
