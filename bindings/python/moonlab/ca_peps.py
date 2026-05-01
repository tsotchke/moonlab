"""Moonlab Clifford-Assisted PEPS (CA-PEPS) bindings.

Exposes the 2D generalisation of CA-MPS: a Clifford-prefactor tableau D
plus a physical factor over an Lx-by-Ly square lattice.

v0.2.1 implementation: |phi> is stored as a row-major MPS over the
Lx*Ly qubits via the existing CA-MPS engine (linear index q = x + Lx*y).
This is *not* a true PEPS factor -- a real 2D tensor with four bond
indices per site, plus split-CTMRG / boundary-MPS contraction, lands in
v0.3.  The row-major embedding nonetheless gives correct results for
any circuit at the current ABI surface (single + 2-qubit Cliffords +
RX/RY/RZ + Pauli expectation).

Quick example::

    >>> from moonlab.ca_peps import CAPEPS
    >>> import numpy as np
    >>>
    >>> p = CAPEPS(Lx=3, Ly=3, max_bond_dim=32)
    >>> # Bell pair at (0,0)-(1,0) by H + CNOT.
    >>> p.h(0); p.cnot(0, 1)
    >>> pauli = np.zeros(9, dtype=np.uint8)
    >>> pauli[0] = 3; pauli[1] = 3   # ZZ correlator on the pair
    >>> zz = p.expect_pauli(pauli)
    >>> abs(zz - 1.0) < 1e-12
    True
"""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from .core import _lib

__all__ = ["CAPEPS"]


# ---------------------------------------------------------------- #
# ABI signature setup.                                             #
# ---------------------------------------------------------------- #

_lib.moonlab_ca_peps_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
_lib.moonlab_ca_peps_create.restype = ctypes.c_void_p

_lib.moonlab_ca_peps_free.argtypes = [ctypes.c_void_p]
_lib.moonlab_ca_peps_free.restype = None

_lib.moonlab_ca_peps_clone.argtypes = [ctypes.c_void_p]
_lib.moonlab_ca_peps_clone.restype = ctypes.c_void_p

for _name in ("lx", "ly", "num_qubits", "max_bond_dim"):
    _fn = getattr(_lib, f"moonlab_ca_peps_{_name}")
    _fn.argtypes = [ctypes.c_void_p]
    _fn.restype = ctypes.c_uint32

# Single-qubit Clifford gates.
for _name in ("h", "s", "sdag", "x", "y", "z"):
    _fn = getattr(_lib, f"moonlab_ca_peps_{_name}")
    _fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _fn.restype = ctypes.c_int

# Two-qubit Clifford gates -- adjacency-validated.
for _name in ("cnot", "cz"):
    _fn = getattr(_lib, f"moonlab_ca_peps_{_name}")
    _fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    _fn.restype = ctypes.c_int

# Non-Clifford single-qubit rotations.
for _name in ("rx", "ry", "rz"):
    _fn = getattr(_lib, f"moonlab_ca_peps_{_name}")
    _fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_double]
    _fn.restype = ctypes.c_int

_lib.moonlab_ca_peps_expect_pauli.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_double * 2),  # double _Complex packed as 2-double array
]
_lib.moonlab_ca_peps_expect_pauli.restype = ctypes.c_int


# ---------------------------------------------------------------- #
# Pythonic wrapper.                                                #
# ---------------------------------------------------------------- #

class CAPEPS:
    """Clifford-Assisted PEPS state on an Lx-by-Ly square lattice.

    Linear indexing convention: ``q = x + Lx * y``, so site (x=0, y=0)
    is q=0 and site (Lx-1, Ly-1) is q=Lx*Ly-1.  Two-qubit Clifford
    gates accept any pair on a lattice edge (horizontal or vertical
    neighbour); they reject diagonals and same-site pairs with
    ``CA_PEPS_ERR_QUBIT``.

    The handle owns the underlying C state; call :meth:`free` (or rely
    on garbage collection) when done.
    """

    def __init__(self, Lx: int, Ly: int, max_bond_dim: int):
        h = _lib.moonlab_ca_peps_create(Lx, Ly, max_bond_dim)
        if not h:
            raise MemoryError("moonlab_ca_peps_create failed")
        self._handle = ctypes.c_void_p(h)

    def __del__(self):
        self.free()

    def free(self) -> None:
        if getattr(self, "_handle", None) is not None and self._handle.value:
            _lib.moonlab_ca_peps_free(self._handle)
            self._handle = ctypes.c_void_p(0)

    def clone(self) -> "CAPEPS":
        h = _lib.moonlab_ca_peps_clone(self._handle)
        if not h:
            raise MemoryError("moonlab_ca_peps_clone failed")
        out = CAPEPS.__new__(CAPEPS)
        out._handle = ctypes.c_void_p(h)
        return out

    @property
    def Lx(self) -> int:
        return int(_lib.moonlab_ca_peps_lx(self._handle))

    @property
    def Ly(self) -> int:
        return int(_lib.moonlab_ca_peps_ly(self._handle))

    @property
    def num_qubits(self) -> int:
        return int(_lib.moonlab_ca_peps_num_qubits(self._handle))

    @property
    def max_bond_dim(self) -> int:
        return int(_lib.moonlab_ca_peps_max_bond_dim(self._handle))

    # Single-qubit Cliffords.
    def h(self, q: int) -> None: self._check(_lib.moonlab_ca_peps_h(self._handle, q))
    def s(self, q: int) -> None: self._check(_lib.moonlab_ca_peps_s(self._handle, q))
    def sdag(self, q: int) -> None: self._check(_lib.moonlab_ca_peps_sdag(self._handle, q))
    def x(self, q: int) -> None: self._check(_lib.moonlab_ca_peps_x(self._handle, q))
    def y(self, q: int) -> None: self._check(_lib.moonlab_ca_peps_y(self._handle, q))
    def z(self, q: int) -> None: self._check(_lib.moonlab_ca_peps_z(self._handle, q))

    # Two-qubit Cliffords.
    def cnot(self, c: int, t: int) -> None:
        self._check(_lib.moonlab_ca_peps_cnot(self._handle, c, t))

    def cz(self, a: int, b: int) -> None:
        self._check(_lib.moonlab_ca_peps_cz(self._handle, a, b))

    # Non-Clifford rotations.
    def rx(self, q: int, theta: float) -> None:
        self._check(_lib.moonlab_ca_peps_rx(self._handle, q, theta))

    def ry(self, q: int, theta: float) -> None:
        self._check(_lib.moonlab_ca_peps_ry(self._handle, q, theta))

    def rz(self, q: int, theta: float) -> None:
        self._check(_lib.moonlab_ca_peps_rz(self._handle, q, theta))

    def expect_pauli(self, pauli: np.ndarray) -> complex:
        """Compute <psi|P|psi> for the given Pauli string.

        ``pauli`` is a length-``num_qubits`` ``np.uint8`` array using the
        encoding 0=I, 1=X, 2=Y, 3=Z.
        """
        if not isinstance(pauli, np.ndarray) or pauli.dtype != np.uint8:
            raise TypeError("pauli must be a uint8 numpy array")
        if pauli.size != self.num_qubits:
            raise ValueError(
                f"pauli length {pauli.size} != num_qubits {self.num_qubits}"
            )
        if not pauli.flags["C_CONTIGUOUS"]:
            pauli = np.ascontiguousarray(pauli)
        out = (ctypes.c_double * 2)()
        rc = _lib.moonlab_ca_peps_expect_pauli(
            self._handle,
            pauli.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(out),
        )
        self._check(rc)
        return complex(out[0], out[1])

    @staticmethod
    def _check(rc: int) -> None:
        if rc != 0:
            raise RuntimeError(f"CA-PEPS ABI call failed with rc={rc}")
