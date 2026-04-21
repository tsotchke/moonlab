"""Reverse-mode autograd for parameterized quantum circuits.

Mirrors the C surface in ``src/algorithms/diff/differentiable.h`` so
Python users get native adjoint gradients without a PyTorch
dependency.  Typical VQE loop::

    from moonlab import QuantumState
    from moonlab.diff import DiffCircuit, PauliTerm, OBS_Z, OBS_X

    circ = DiffCircuit(num_qubits=2)
    t = [0.1, 0.2, 0.3]
    circ.ry(0, t[0])
    circ.ry(1, t[1])
    circ.cnot(0, 1)
    circ.ry(0, t[2])

    state = QuantumState(2)
    circ.forward(state)

    H = [PauliTerm(1.0, [0], [OBS_Z]),           #   Z_0
         PauliTerm(0.5, [0, 1], [OBS_Z, OBS_Z])] # + 0.5 Z_0 Z_1
    cost = circ.expect_pauli_sum(state, H)
    grad = circ.backward_pauli_sum(state, H)    # ndarray, shape (n_params,)

The circuit records parametric RX / RY / RZ rotations; for each such
call, ``circ.backward(...)`` or ``circ.backward_pauli_sum(...)`` returns
a gradient slot whose index matches the order of appends.
``circ.set_theta(k, value)`` updates the k-th parametric angle in
place so an optimiser can sweep without rebuilding the circuit.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .core import _lib, QuantumState, CQuantumState


# --------------------------------------------------------------------
# ctypes declarations
# --------------------------------------------------------------------

# Opaque circuit handle.
_lib.moonlab_diff_circuit_create.argtypes = [ctypes.c_uint32]
_lib.moonlab_diff_circuit_create.restype = ctypes.c_void_p

_lib.moonlab_diff_circuit_free.argtypes = [ctypes.c_void_p]
_lib.moonlab_diff_circuit_free.restype = None

_lib.moonlab_diff_num_qubits.argtypes = [ctypes.c_void_p]
_lib.moonlab_diff_num_qubits.restype = ctypes.c_uint32

_lib.moonlab_diff_num_parameters.argtypes = [ctypes.c_void_p]
_lib.moonlab_diff_num_parameters.restype = ctypes.c_size_t

# Non-parametric gates.
for _name in ("h", "x", "y", "z"):
    _fn = getattr(_lib, f"moonlab_diff_{_name}")
    _fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _fn.restype = ctypes.c_int

_lib.moonlab_diff_cnot.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.moonlab_diff_cnot.restype = ctypes.c_int
_lib.moonlab_diff_cz.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.moonlab_diff_cz.restype = ctypes.c_int

# Parametric rotations.
for _name in ("rx", "ry", "rz"):
    _fn = getattr(_lib, f"moonlab_diff_{_name}")
    _fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
    _fn.restype = ctypes.c_int

_lib.moonlab_diff_set_theta.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                          ctypes.c_double]
_lib.moonlab_diff_set_theta.restype = ctypes.c_int

_lib.moonlab_diff_forward.argtypes = [ctypes.c_void_p,
                                        ctypes.POINTER(CQuantumState)]
_lib.moonlab_diff_forward.restype = ctypes.c_int

_lib.moonlab_diff_expect_z.argtypes = [ctypes.POINTER(CQuantumState), ctypes.c_int]
_lib.moonlab_diff_expect_z.restype = ctypes.c_double
_lib.moonlab_diff_expect_x.argtypes = [ctypes.POINTER(CQuantumState), ctypes.c_int]
_lib.moonlab_diff_expect_x.restype = ctypes.c_double

_lib.moonlab_diff_backward.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(CQuantumState),
    ctypes.c_int,        # observable enum
    ctypes.c_int,        # qubit
    ctypes.POINTER(ctypes.c_double),  # grad_out
]
_lib.moonlab_diff_backward.restype = ctypes.c_int

# Multi-Pauli observable: mirrors moonlab_diff_pauli_term_t.  Keep the
# pointer members as c_void_p so ctypes treats them opaquely; we fill
# them by keeping the underlying arrays alive in Python.
class _CPauliTerm(ctypes.Structure):
    _fields_ = [
        ("coefficient", ctypes.c_double),
        ("num_ops",     ctypes.c_size_t),
        ("qubits",      ctypes.POINTER(ctypes.c_int)),
        ("paulis",      ctypes.POINTER(ctypes.c_int)),  # enum = c_int
    ]

_lib.moonlab_diff_expect_pauli_sum.argtypes = [
    ctypes.POINTER(CQuantumState),
    ctypes.POINTER(_CPauliTerm),
    ctypes.c_size_t,
]
_lib.moonlab_diff_expect_pauli_sum.restype = ctypes.c_double

_lib.moonlab_diff_backward_pauli_sum.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(CQuantumState),
    ctypes.POINTER(_CPauliTerm),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
]
_lib.moonlab_diff_backward_pauli_sum.restype = ctypes.c_int


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------

# Enum values mirror ``moonlab_diff_observable_t``.
OBS_Z = 0
OBS_X = 1
OBS_Y = 2


@dataclass(frozen=True)
class PauliTerm:
    """One weighted Pauli string, e.g. ``PauliTerm(0.5, [0, 1], [OBS_Z, OBS_Z])``.

    A term with ``num_ops == 0`` (empty qubit / ops lists) represents
    the identity, contributing ``coefficient`` to the expectation
    value and zero to every gradient.
    """
    coefficient: float
    qubits: Sequence[int]
    ops: Sequence[int]


class DiffCircuit:
    """Builder + differentiator for a parameterized circuit."""

    def __init__(self, num_qubits: int):
        if num_qubits < 1 or num_qubits > 63:
            raise ValueError("num_qubits must be in [1, 63]")
        ptr = _lib.moonlab_diff_circuit_create(ctypes.c_uint32(num_qubits))
        if not ptr:
            raise RuntimeError("moonlab_diff_circuit_create returned NULL")
        self._ptr = ctypes.c_void_p(ptr)
        self._num_qubits = int(num_qubits)

    def __del__(self):
        if getattr(self, "_ptr", None) is not None and self._ptr.value:
            _lib.moonlab_diff_circuit_free(self._ptr)
            self._ptr = None

    # ---- properties --------------------------------------------------

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def num_parameters(self) -> int:
        return int(_lib.moonlab_diff_num_parameters(self._ptr))

    # ---- non-parametric gates ---------------------------------------

    def h(self, qubit: int) -> "DiffCircuit":
        self._check(_lib.moonlab_diff_h(self._ptr, qubit)); return self
    def x(self, qubit: int) -> "DiffCircuit":
        self._check(_lib.moonlab_diff_x(self._ptr, qubit)); return self
    def y(self, qubit: int) -> "DiffCircuit":
        self._check(_lib.moonlab_diff_y(self._ptr, qubit)); return self
    def z(self, qubit: int) -> "DiffCircuit":
        self._check(_lib.moonlab_diff_z(self._ptr, qubit)); return self
    def cnot(self, control: int, target: int) -> "DiffCircuit":
        self._check(_lib.moonlab_diff_cnot(self._ptr, control, target))
        return self
    def cz(self, q0: int, q1: int) -> "DiffCircuit":
        self._check(_lib.moonlab_diff_cz(self._ptr, q0, q1)); return self

    # ---- parametric rotations ---------------------------------------

    def rx(self, qubit: int, theta: float) -> "DiffCircuit":
        self._check(_lib.moonlab_diff_rx(self._ptr, qubit, theta)); return self
    def ry(self, qubit: int, theta: float) -> "DiffCircuit":
        self._check(_lib.moonlab_diff_ry(self._ptr, qubit, theta)); return self
    def rz(self, qubit: int, theta: float) -> "DiffCircuit":
        self._check(_lib.moonlab_diff_rz(self._ptr, qubit, theta)); return self

    def set_theta(self, k: int, theta: float) -> None:
        """Replace the k-th parametric angle with ``theta``."""
        rc = _lib.moonlab_diff_set_theta(self._ptr, ctypes.c_size_t(k), theta)
        if rc != 0:
            raise IndexError(f"set_theta index {k} out of range ({rc})")

    # ---- forward + observables --------------------------------------

    def forward(self, state: QuantumState) -> None:
        """Evolve ``state`` from |0...0> through the circuit in place."""
        if state.num_qubits != self._num_qubits:
            raise ValueError("state num_qubits mismatch")
        rc = _lib.moonlab_diff_forward(self._ptr,
                                        ctypes.byref(state._state))
        if rc != 0:
            raise RuntimeError(f"moonlab_diff_forward failed ({rc})")

    @staticmethod
    def expect_z(state: QuantumState, qubit: int) -> float:
        return float(_lib.moonlab_diff_expect_z(ctypes.byref(state._state),
                                                 qubit))

    @staticmethod
    def expect_x(state: QuantumState, qubit: int) -> float:
        return float(_lib.moonlab_diff_expect_x(ctypes.byref(state._state),
                                                 qubit))

    # ---- gradients --------------------------------------------------

    def backward(self, state: QuantumState,
                 obs: int, qubit: int) -> np.ndarray:
        """Gradient of <obs_qubit> with respect to every parametric
        angle.  Returns a NumPy array of length ``num_parameters``.
        """
        n = self.num_parameters
        grad = (ctypes.c_double * n)()
        rc = _lib.moonlab_diff_backward(self._ptr,
                                         ctypes.byref(state._state),
                                         ctypes.c_int(obs),
                                         ctypes.c_int(qubit),
                                         grad)
        if rc != 0:
            raise RuntimeError(f"moonlab_diff_backward failed ({rc})")
        return np.array(list(grad), dtype=np.float64)

    # ---- Pauli-sum observables --------------------------------------

    @staticmethod
    def _pack_terms(terms: Sequence[PauliTerm]):
        """Build a (c_array, keepalive) for the ctypes boundary.
        The keepalive list pins all the per-term ctypes arrays so GC
        doesn't reclaim them before the C call returns."""
        n = len(terms)
        arr = (_CPauliTerm * n)()
        keepalive = []
        for i, t in enumerate(terms):
            q_list = list(t.qubits or [])
            p_list = list(t.ops or [])
            if len(q_list) != len(p_list):
                raise ValueError(f"term {i}: qubits and ops length mismatch")
            arr[i].coefficient = float(t.coefficient)
            arr[i].num_ops     = ctypes.c_size_t(len(q_list))
            if q_list:
                q_arr = (ctypes.c_int * len(q_list))(*q_list)
                p_arr = (ctypes.c_int * len(p_list))(*p_list)
                keepalive.append((q_arr, p_arr))
                arr[i].qubits = ctypes.cast(q_arr,
                                             ctypes.POINTER(ctypes.c_int))
                arr[i].paulis = ctypes.cast(p_arr,
                                             ctypes.POINTER(ctypes.c_int))
            else:
                arr[i].qubits = None
                arr[i].paulis = None
        return arr, keepalive

    @staticmethod
    def expect_pauli_sum(state: QuantumState,
                         terms: Sequence[PauliTerm]) -> float:
        arr, _ka = DiffCircuit._pack_terms(terms)
        return float(_lib.moonlab_diff_expect_pauli_sum(
            ctypes.byref(state._state), arr, ctypes.c_size_t(len(terms))))

    def backward_pauli_sum(self, state: QuantumState,
                           terms: Sequence[PauliTerm]) -> np.ndarray:
        arr, _ka = DiffCircuit._pack_terms(terms)
        n = self.num_parameters
        grad = (ctypes.c_double * n)()
        rc = _lib.moonlab_diff_backward_pauli_sum(
            self._ptr, ctypes.byref(state._state),
            arr, ctypes.c_size_t(len(terms)), grad)
        if rc != 0:
            raise RuntimeError(f"moonlab_diff_backward_pauli_sum failed ({rc})")
        return np.array(list(grad), dtype=np.float64)

    # ---- internals --------------------------------------------------

    def _check(self, rc: int) -> None:
        if rc != 0:
            raise RuntimeError(f"moonlab_diff_* call failed ({rc})")


__all__ = [
    "DiffCircuit",
    "PauliTerm",
    "OBS_Z",
    "OBS_X",
    "OBS_Y",
]
