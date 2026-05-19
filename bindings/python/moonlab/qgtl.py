"""QGTL-shaped circuit-ingestion Python binding -- since v0.6.8.

Wraps the C contract from `src/applications/moonlab_qgtl_backend.{c,h}`
shipped in v0.6.6.  QGTL itself plugs into the C surface directly;
this Python layer is for Jupyter / scripting consumers that want to
drive moonlab as a backend without round-tripping through QGTL.

The gate-type enum numerically matches QGTL's `gate_type_t` so codes
copied from QGTL examples work unchanged.

Example:

    >>> from moonlab.qgtl import QgtlCircuit, GateType
    >>> c = QgtlCircuit(num_qubits=2)
    >>> c.add_gate(GateType.H, target=0)
    >>> c.add_gate(GateType.CNOT, target=1, control=0)
    >>> r = c.execute(return_probabilities=True)
    >>> abs(r.probabilities[0] - 0.5) < 1e-9 and abs(r.probabilities[3] - 0.5) < 1e-9
    True

@since v0.6.8
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Sequence

from .core import _lib


# Numerically matches src/applications/moonlab_qgtl_backend.h.
class GateType(IntEnum):
    I    = 0
    X    = 1
    Y    = 2
    Z    = 3
    H    = 4
    S    = 5
    T    = 6
    RX   = 7
    RY   = 8
    RZ   = 9
    CNOT = 10
    CY   = 11
    CZ   = 12
    SWAP = 13


MOONLAB_QGTL_OK = 0
MOONLAB_QGTL_BAD_ARG = -301
MOONLAB_QGTL_OOM = -302
MOONLAB_QGTL_UNSUPPORTED = -303
MOONLAB_QGTL_INTERNAL = -304


class QgtlError(RuntimeError):
    """QGTL ingestion failure (any negative status code)."""


# ---- FFI signatures -----------------------------------------------

class _ExecOptions(ctypes.Structure):
    _fields_ = [
        ("num_shots", ctypes.c_int),
        ("rng_seed",  ctypes.c_uint64),
        ("return_probabilities", ctypes.c_int),
    ]


class _Results(ctypes.Structure):
    _fields_ = [
        ("num_qubits",    ctypes.c_int),
        ("num_shots",     ctypes.c_int),
        ("outcomes",      ctypes.POINTER(ctypes.c_uint64)),
        ("probabilities", ctypes.POINTER(ctypes.c_double)),
    ]


_lib.moonlab_qgtl_circuit_create.argtypes = [ctypes.c_int]
_lib.moonlab_qgtl_circuit_create.restype = ctypes.c_void_p

_lib.moonlab_qgtl_circuit_free.argtypes = [ctypes.c_void_p]
_lib.moonlab_qgtl_circuit_free.restype = None

_lib.moonlab_qgtl_add_gate.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
]
_lib.moonlab_qgtl_add_gate.restype = ctypes.c_int

_lib.moonlab_qgtl_execute.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_ExecOptions),
    ctypes.POINTER(_Results),
]
_lib.moonlab_qgtl_execute.restype = ctypes.c_int

_lib.moonlab_qgtl_results_free.argtypes = [ctypes.POINTER(_Results)]
_lib.moonlab_qgtl_results_free.restype = None

_lib.moonlab_qgtl_circuit_num_qubits.argtypes = [ctypes.c_void_p]
_lib.moonlab_qgtl_circuit_num_qubits.restype = ctypes.c_int

_lib.moonlab_qgtl_circuit_num_gates.argtypes = [ctypes.c_void_p]
_lib.moonlab_qgtl_circuit_num_gates.restype = ctypes.c_int


def _check(rc: int, ctx: str) -> None:
    if rc != MOONLAB_QGTL_OK:
        raise QgtlError(f"{ctx}: rc={rc}")


@dataclass
class QgtlResults:
    """Execution outputs.  `outcomes` is shape `(num_shots,)` with
    integer bitstrings; `probabilities` is a length-`2^num_qubits`
    list of `float`s when `return_probabilities=True`, otherwise None."""

    num_qubits: int
    num_shots: int
    outcomes: Optional[list[int]]
    probabilities: Optional[list[float]]


class QgtlCircuit:
    """Owned C handle to a `moonlab_qgtl_circuit_t`."""

    __slots__ = ("_handle", "_n")

    def __init__(self, num_qubits: int) -> None:
        h = _lib.moonlab_qgtl_circuit_create(int(num_qubits))
        if not h:
            raise QgtlError(f"circuit_create({num_qubits}): NULL "
                            f"(num_qubits must be in [1, 32])")
        self._handle = h
        self._n = int(num_qubits)

    def __del__(self) -> None:
        if getattr(self, "_handle", None):
            _lib.moonlab_qgtl_circuit_free(self._handle)
            self._handle = None

    @property
    def num_qubits(self) -> int:
        return self._n

    @property
    def num_gates(self) -> int:
        return int(_lib.moonlab_qgtl_circuit_num_gates(self._handle))

    def add_gate(self,
                 type: GateType,
                 target: int,
                 control: int = -1,
                 params: Optional[Sequence[float]] = None) -> "QgtlCircuit":
        """Append a gate.  Returns self for fluent chaining."""
        if params is not None:
            buf = (ctypes.c_double * len(params))(*params)
            params_ptr = buf
        else:
            params_ptr = None
        rc = _lib.moonlab_qgtl_add_gate(
            self._handle, int(type), int(target), int(control), params_ptr)
        _check(rc, f"add_gate({type.name}, target={target}, control={control})")
        return self

    def execute(self,
                num_shots: int = 0,
                rng_seed: int = 0,
                return_probabilities: bool = False) -> QgtlResults:
        """Run the circuit through moonlab's state-vector backend."""
        opts = _ExecOptions(
            num_shots=int(num_shots),
            rng_seed=int(rng_seed),
            return_probabilities=1 if return_probabilities else 0,
        )
        res = _Results()
        rc = _lib.moonlab_qgtl_execute(self._handle, ctypes.byref(opts), ctypes.byref(res))
        _check(rc, "execute")

        outcomes = None
        if res.num_shots > 0 and res.outcomes:
            outcomes = [int(res.outcomes[i]) for i in range(res.num_shots)]

        probabilities = None
        if return_probabilities and res.probabilities:
            dim = 1 << res.num_qubits
            probabilities = [float(res.probabilities[i]) for i in range(dim)]

        out = QgtlResults(
            num_qubits=res.num_qubits,
            num_shots=res.num_shots,
            outcomes=outcomes,
            probabilities=probabilities,
        )
        _lib.moonlab_qgtl_results_free(ctypes.byref(res))
        return out


__all__ = [
    "GateType",
    "QgtlCircuit",
    "QgtlResults",
    "QgtlError",
]
