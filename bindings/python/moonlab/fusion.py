"""Single-qubit gate-fusion DAG bindings.

Wraps ``src/optimization/fusion/fusion.{c,h}`` so Python users can
build a symbolic circuit, run the run-length single-qubit fuser over
it, and execute the fused schedule on a :class:`moonlab.QuantumState`.

The fuser collapses adjacent single-qubit gates on the same qubit
into one 2x2 matrix, dropping repeated full-state passes that
state-vector simulation pays for each gate.  On a five-layer
hardware-efficient ansatz at n = 16 the fused execution is roughly
2.2x faster than the unfused dispatch; see the README in
``src/optimization/fusion/`` for the full performance discussion.

Example::

    from moonlab import QuantumState
    from moonlab.fusion import FusedCircuit

    circuit = FusedCircuit(num_qubits=4)
    circuit.h(0).rz(0, 0.3).rx(0, 0.7).cnot(0, 1).rz(1, 0.4)
    fused, stats = circuit.compile()
    print(stats)  # FuseStats(gates_in=5, gates_out=3, fused_gates=2, merges_applied=2)

    state = QuantumState(num_qubits=4)
    fused.execute(state)

References:
- T. Haener and D. S. Steiger, "0.5 Petabyte Simulation of a
  45-Qubit Quantum Circuit", SC17 (2017), arXiv:1704.01127.
- Y. Suzuki et al., "Qulacs: a fast and versatile quantum circuit
  simulator for research purpose", Quantum 5, 559 (2021).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Optional, Tuple

from .core import _lib, QuantumState


__all__ = ["FuseStats", "FusedCircuit"]


# ---------------------------------------------------------------------------
# FFI signatures
# ---------------------------------------------------------------------------


class _CFuseStats(ctypes.Structure):
    """Layout-mirrored from ``fuse_stats_t`` in
    ``src/optimization/fusion/fusion.h``."""

    _fields_ = [
        ("original_gates", ctypes.c_size_t),
        ("fused_gates",    ctypes.c_size_t),
        ("merges_applied", ctypes.c_size_t),
    ]


_lib.fuse_circuit_create.argtypes = [ctypes.c_size_t]
_lib.fuse_circuit_create.restype = ctypes.c_void_p

_lib.fuse_circuit_free.argtypes = [ctypes.c_void_p]
_lib.fuse_circuit_free.restype = None

_lib.fuse_circuit_len.argtypes = [ctypes.c_void_p]
_lib.fuse_circuit_len.restype = ctypes.c_size_t

_lib.fuse_circuit_num_qubits.argtypes = [ctypes.c_void_p]
_lib.fuse_circuit_num_qubits.restype = ctypes.c_size_t

# Single-qubit non-parameterised: (circuit*, q) -> int rc.
for _name in ("h", "x", "y", "z", "s", "sdg", "t", "tdg"):
    _fn = getattr(_lib, f"fuse_append_{_name}")
    _fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _fn.restype = ctypes.c_int

# Single-qubit one-parameter: (circuit*, q, theta) -> int rc.
for _name in ("phase", "rx", "ry", "rz"):
    _fn = getattr(_lib, f"fuse_append_{_name}")
    _fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
    _fn.restype = ctypes.c_int

# U3 three-parameter.
_lib.fuse_append_u3.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
]
_lib.fuse_append_u3.restype = ctypes.c_int

# Two-qubit non-parameterised.
for _name in ("cnot", "cz", "cy", "swap"):
    _fn = getattr(_lib, f"fuse_append_{_name}")
    _fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    _fn.restype = ctypes.c_int

# Two-qubit one-parameter.
for _name in ("cphase", "crx", "cry", "crz"):
    _fn = getattr(_lib, f"fuse_append_{_name}")
    _fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                    ctypes.c_double]
    _fn.restype = ctypes.c_int

_lib.fuse_compile.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CFuseStats)]
_lib.fuse_compile.restype = ctypes.c_void_p

# fuse_execute takes a CQuantumState pointer; we forward state._state.
from .core import CQuantumState as _CQuantumState
_lib.fuse_execute.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_CQuantumState),
]
_lib.fuse_execute.restype = ctypes.c_int


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FuseStats:
    """Diagnostic counts from :meth:`FusedCircuit.compile`.

    Mirrors ``fuse_stats_t``.

    Attributes:
        original_gates: Symbol-list length of the input circuit.
        fused_gates:    Symbol-list length of the fused output.
        merges_applied: Number of pair-wise 2x2 multiplications the
                        fuser performed; equals (run-length - 1)
                        summed across every fused run.  Zero when no
                        single-qubit gate ran into a same-qubit
                        successor before a multi-qubit barrier.
    """

    original_gates: int
    fused_gates: int
    merges_applied: int


class FusedCircuit:
    """Symbolic gate-fusion circuit.

    Build the circuit with the fluent gate-append methods, then call
    :meth:`compile` to produce a fused circuit and :meth:`execute` to
    apply it to a :class:`moonlab.QuantumState`.

    Args:
        num_qubits: Number of qubits the circuit operates on.

    The handle is freed automatically on garbage collection; call
    :meth:`free` explicitly to release earlier.
    """

    __slots__ = ("_handle", "_num_qubits")

    def __init__(self, num_qubits: int):
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        h = _lib.fuse_circuit_create(ctypes.c_size_t(num_qubits))
        if not h:
            raise MemoryError("fuse_circuit_create returned NULL")
        self._handle = h
        self._num_qubits = num_qubits

    # --- Lifecycle ---------------------------------------------------------

    def free(self) -> None:
        """Release the underlying C handle.  Safe to call repeatedly."""
        h = getattr(self, "_handle", None)
        if h:
            _lib.fuse_circuit_free(h)
            self._handle = None

    def __del__(self) -> None:
        self.free()

    # --- Read-only accessors ----------------------------------------------

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    def __len__(self) -> int:
        return int(_lib.fuse_circuit_len(self._handle))

    # --- Gate appenders (fluent) ------------------------------------------

    def h(self, q: int) -> "FusedCircuit":      return self._call1(f"fuse_append_h",   q)
    def x(self, q: int) -> "FusedCircuit":      return self._call1(f"fuse_append_x",   q)
    def y(self, q: int) -> "FusedCircuit":      return self._call1(f"fuse_append_y",   q)
    def z(self, q: int) -> "FusedCircuit":      return self._call1(f"fuse_append_z",   q)
    def s(self, q: int) -> "FusedCircuit":      return self._call1(f"fuse_append_s",   q)
    def sdg(self, q: int) -> "FusedCircuit":    return self._call1(f"fuse_append_sdg", q)
    def t(self, q: int) -> "FusedCircuit":      return self._call1(f"fuse_append_t",   q)
    def tdg(self, q: int) -> "FusedCircuit":    return self._call1(f"fuse_append_tdg", q)

    def phase(self, q: int, theta: float) -> "FusedCircuit":
        return self._call1p("fuse_append_phase", q, theta)
    def rx(self, q: int, theta: float) -> "FusedCircuit":
        return self._call1p("fuse_append_rx", q, theta)
    def ry(self, q: int, theta: float) -> "FusedCircuit":
        return self._call1p("fuse_append_ry", q, theta)
    def rz(self, q: int, theta: float) -> "FusedCircuit":
        return self._call1p("fuse_append_rz", q, theta)

    def u3(self, q: int, theta: float, phi: float, lam: float) -> "FusedCircuit":
        rc = _lib.fuse_append_u3(
            self._handle,
            ctypes.c_int(q),
            ctypes.c_double(theta),
            ctypes.c_double(phi),
            ctypes.c_double(lam),
        )
        if rc != 0:
            raise RuntimeError(f"fuse_append_u3 rc={rc}")
        return self

    def cnot(self, ctrl: int, tgt: int) -> "FusedCircuit":
        return self._call2("fuse_append_cnot", ctrl, tgt)
    def cz(self, ctrl: int, tgt: int) -> "FusedCircuit":
        return self._call2("fuse_append_cz", ctrl, tgt)
    def cy(self, ctrl: int, tgt: int) -> "FusedCircuit":
        return self._call2("fuse_append_cy", ctrl, tgt)
    def swap(self, a: int, b: int) -> "FusedCircuit":
        return self._call2("fuse_append_swap", a, b)

    def cphase(self, ctrl: int, tgt: int, theta: float) -> "FusedCircuit":
        return self._call2p("fuse_append_cphase", ctrl, tgt, theta)
    def crx(self, ctrl: int, tgt: int, theta: float) -> "FusedCircuit":
        return self._call2p("fuse_append_crx", ctrl, tgt, theta)
    def cry(self, ctrl: int, tgt: int, theta: float) -> "FusedCircuit":
        return self._call2p("fuse_append_cry", ctrl, tgt, theta)
    def crz(self, ctrl: int, tgt: int, theta: float) -> "FusedCircuit":
        return self._call2p("fuse_append_crz", ctrl, tgt, theta)

    # --- Compile + execute -------------------------------------------------

    def compile(self) -> Tuple["FusedCircuit", FuseStats]:
        """Run the single-qubit fuser, returning a new
        :class:`FusedCircuit` plus a :class:`FuseStats` summary."""
        stats = _CFuseStats()
        h = _lib.fuse_compile(self._handle, ctypes.byref(stats))
        if not h:
            raise MemoryError("fuse_compile returned NULL")
        wrapper = FusedCircuit.__new__(FusedCircuit)
        wrapper._handle = h
        wrapper._num_qubits = self._num_qubits
        return wrapper, FuseStats(
            original_gates=stats.original_gates,
            fused_gates=stats.fused_gates,
            merges_applied=stats.merges_applied,
        )

    def execute(self, state: QuantumState) -> None:
        """Apply the circuit to @p state in place.

        Works on both fused and unfused circuits.
        """
        rc = _lib.fuse_execute(self._handle, ctypes.byref(state._state))
        if rc != 0:
            raise RuntimeError(f"fuse_execute rc={rc}")

    # --- Internal helpers --------------------------------------------------

    def _call1(self, fn_name: str, q: int) -> "FusedCircuit":
        rc = getattr(_lib, fn_name)(self._handle, ctypes.c_int(q))
        if rc != 0:
            raise RuntimeError(f"{fn_name} rc={rc}")
        return self

    def _call1p(self, fn_name: str, q: int, theta: float) -> "FusedCircuit":
        rc = getattr(_lib, fn_name)(
            self._handle, ctypes.c_int(q), ctypes.c_double(theta))
        if rc != 0:
            raise RuntimeError(f"{fn_name} rc={rc}")
        return self

    def _call2(self, fn_name: str, a: int, b: int) -> "FusedCircuit":
        rc = getattr(_lib, fn_name)(
            self._handle, ctypes.c_int(a), ctypes.c_int(b))
        if rc != 0:
            raise RuntimeError(f"{fn_name} rc={rc}")
        return self

    def _call2p(self, fn_name: str, a: int, b: int, theta: float) -> "FusedCircuit":
        rc = getattr(_lib, fn_name)(
            self._handle, ctypes.c_int(a), ctypes.c_int(b),
            ctypes.c_double(theta))
        if rc != 0:
            raise RuntimeError(f"{fn_name} rc={rc}")
        return self
