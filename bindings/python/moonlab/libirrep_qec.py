"""libirrep QEC zoo Python binding -- since v0.6.3.

Wraps the eight CSS-code factories the v0.6.2 bridge ships in
``src/integration/libirrep_bridge.{c,h}``: surface, toric, 2D color
(Steane + Hamming), IBM bivariate-bicycle (the three Bravyi-Nature 627
"Gross" qLDPC codes), and the Tillich-Zemor hypergraph product.

Every code family returns a :class:`LibirrepQecCode` -- one Python type
with one accessor surface (``n_qubits``, ``n_x_stabs``, ``n_z_stabs``,
``logical_qubits``, ``distance``, ``x_check_row``, ``z_check_row``).

The bridge is optional at build time.  When moonlab was compiled
without ``-DQSIM_ENABLE_LIBIRREP=ON``, every factory raises
:class:`LibirrepNotBuiltError` rather than returning a stub.  Callers
can probe with :func:`is_available` first if "use libirrep when
available, else fall back" semantics are needed.

Example:

    >>> from moonlab.libirrep_qec import LibirrepQecCode, is_available
    >>> if is_available():
    ...     code = LibirrepQecCode.bb_72_12_6()
    ...     print(code.n_qubits, code.logical_qubits, code.distance)
    72 12 6

@since v0.6.3
"""

from __future__ import annotations

import ctypes
from typing import Literal

from .core import _lib


# ---- status codes (mirror libirrep_bridge.h) -----------------------

MOONLAB_LIBIRREP_OK = 0
MOONLAB_LIBIRREP_NOT_BUILT = -201
MOONLAB_LIBIRREP_BAD_ARG = -202
MOONLAB_LIBIRREP_INTERNAL = -203
MOONLAB_LIBIRREP_OOM = -204


class LibirrepError(RuntimeError):
    """Generic libirrep-bridge failure (negative status code)."""


class LibirrepNotBuiltError(LibirrepError):
    """Raised when libirrep was not linked at build time."""


# ---- FFI signatures ------------------------------------------------

_lib.moonlab_libirrep_available.argtypes = []
_lib.moonlab_libirrep_available.restype = ctypes.c_int

_lib.moonlab_libirrep_surface_code_new.argtypes = [
    ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
_lib.moonlab_libirrep_surface_code_new.restype = ctypes.c_int

_lib.moonlab_libirrep_toric_code_new.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
_lib.moonlab_libirrep_toric_code_new.restype = ctypes.c_int

_lib.moonlab_libirrep_color_steane_new.argtypes = [
    ctypes.POINTER(ctypes.c_void_p)]
_lib.moonlab_libirrep_color_steane_new.restype = ctypes.c_int

_lib.moonlab_libirrep_color_hamming_15_7_3_new.argtypes = [
    ctypes.POINTER(ctypes.c_void_p)]
_lib.moonlab_libirrep_color_hamming_15_7_3_new.restype = ctypes.c_int

_lib.moonlab_libirrep_bb_72_12_6_new.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_lib.moonlab_libirrep_bb_72_12_6_new.restype = ctypes.c_int

_lib.moonlab_libirrep_bb_144_12_12_new.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_lib.moonlab_libirrep_bb_144_12_12_new.restype = ctypes.c_int

_lib.moonlab_libirrep_bb_288_12_18_new.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_lib.moonlab_libirrep_bb_288_12_18_new.restype = ctypes.c_int

_lib.moonlab_libirrep_hgp_repetition_new.argtypes = [
    ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
_lib.moonlab_libirrep_hgp_repetition_new.restype = ctypes.c_int

_lib.moonlab_libirrep_qec_free.argtypes = [ctypes.c_void_p]
_lib.moonlab_libirrep_qec_free.restype = None

for _accessor in (
    "n_qubits", "n_x_stabs", "n_z_stabs", "logical_qubits", "distance",
):
    _fn = getattr(_lib, f"moonlab_libirrep_qec_{_accessor}")
    _fn.argtypes = [ctypes.c_void_p]
    _fn.restype = ctypes.c_int

_lib.moonlab_libirrep_qec_get_x_check_row.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte)]
_lib.moonlab_libirrep_qec_get_x_check_row.restype = ctypes.c_int

_lib.moonlab_libirrep_qec_get_z_check_row.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte)]
_lib.moonlab_libirrep_qec_get_z_check_row.restype = ctypes.c_int


# ---- module-level helpers ------------------------------------------

def is_available() -> bool:
    """Whether moonlab was compiled with the libirrep linkage path."""
    return _lib.moonlab_libirrep_available() == 1


def _raise_for(rc: int, ctx: str) -> None:
    if rc == MOONLAB_LIBIRREP_OK:
        return
    if rc == MOONLAB_LIBIRREP_NOT_BUILT:
        raise LibirrepNotBuiltError(
            f"{ctx}: moonlab was compiled without libirrep "
            f"(rebuild with -DQSIM_ENABLE_LIBIRREP=ON).")
    raise LibirrepError(f"{ctx}: rc={rc}")


# ---- main class ----------------------------------------------------

class LibirrepQecCode:
    """Owned handle to a CSS code built via libirrep.

    Construct one with the class-method factories
    (:meth:`surface`, :meth:`toric`, :meth:`steane`,
    :meth:`hamming_15_7_3`, :meth:`bb_72_12_6`, :meth:`bb_144_12_12`,
    :meth:`bb_288_12_18`, :meth:`hgp_repetition`); the regular
    constructor expects a raw ``c_void_p`` and is internal.
    """

    __slots__ = ("_handle",)

    def __init__(self, handle: ctypes.c_void_p) -> None:
        if not handle:
            raise LibirrepError("LibirrepQecCode handle is NULL")
        self._handle = handle

    def __del__(self) -> None:
        if getattr(self, "_handle", None):
            _lib.moonlab_libirrep_qec_free(self._handle)
            self._handle = None

    # ---- factories ------------------------------------------------

    @classmethod
    def surface(cls, distance: int) -> "LibirrepQecCode":
        """Rotated surface code at the given (odd) distance >= 2."""
        handle = ctypes.c_void_p()
        rc = _lib.moonlab_libirrep_surface_code_new(
            int(distance), ctypes.byref(handle))
        _raise_for(rc, f"surface(distance={distance})")
        return cls(handle)

    @classmethod
    def toric(cls, Lx: int, Ly: int) -> "LibirrepQecCode":
        """Kitaev 2D toric code on the Lx x Ly torus, [[2 Lx Ly, 2, min(Lx, Ly)]]."""
        handle = ctypes.c_void_p()
        rc = _lib.moonlab_libirrep_toric_code_new(
            int(Lx), int(Ly), ctypes.byref(handle))
        _raise_for(rc, f"toric(Lx={Lx}, Ly={Ly})")
        return cls(handle)

    @classmethod
    def steane(cls) -> "LibirrepQecCode":
        """Steane [[7, 1, 3]] 2D color code."""
        handle = ctypes.c_void_p()
        rc = _lib.moonlab_libirrep_color_steane_new(ctypes.byref(handle))
        _raise_for(rc, "steane()")
        return cls(handle)

    @classmethod
    def hamming_15_7_3(cls) -> "LibirrepQecCode":
        """[[15, 7, 3]] Hamming-based CSS code."""
        handle = ctypes.c_void_p()
        rc = _lib.moonlab_libirrep_color_hamming_15_7_3_new(ctypes.byref(handle))
        _raise_for(rc, "hamming_15_7_3()")
        return cls(handle)

    @classmethod
    def bb_72_12_6(cls) -> "LibirrepQecCode":
        """IBM Gross-72 bivariate-bicycle qLDPC code, [[72, 12, 6]]."""
        handle = ctypes.c_void_p()
        rc = _lib.moonlab_libirrep_bb_72_12_6_new(ctypes.byref(handle))
        _raise_for(rc, "bb_72_12_6()")
        return cls(handle)

    @classmethod
    def bb_144_12_12(cls) -> "LibirrepQecCode":
        """IBM Gross-144 bivariate-bicycle qLDPC code, [[144, 12, 12]]."""
        handle = ctypes.c_void_p()
        rc = _lib.moonlab_libirrep_bb_144_12_12_new(ctypes.byref(handle))
        _raise_for(rc, "bb_144_12_12()")
        return cls(handle)

    @classmethod
    def bb_288_12_18(cls) -> "LibirrepQecCode":
        """IBM Gross-288 bivariate-bicycle qLDPC code, [[288, 12, 18]]."""
        handle = ctypes.c_void_p()
        rc = _lib.moonlab_libirrep_bb_288_12_18_new(ctypes.byref(handle))
        _raise_for(rc, "bb_288_12_18()")
        return cls(handle)

    @classmethod
    def hgp_repetition(cls, d: Literal[3, 4, 5]) -> "LibirrepQecCode":
        """Tillich-Zemor hypergraph product of two [d, 1, d] repetition codes.

        ``d == 3 -> [[13, 1, 3]]``,
        ``d == 4 -> [[25, 1, 4]]``,
        ``d == 5 -> [[41, 1, 5]]``."""
        handle = ctypes.c_void_p()
        rc = _lib.moonlab_libirrep_hgp_repetition_new(int(d), ctypes.byref(handle))
        _raise_for(rc, f"hgp_repetition(d={d})")
        return cls(handle)

    # ---- accessors ------------------------------------------------

    @property
    def n_qubits(self) -> int:
        return int(_lib.moonlab_libirrep_qec_n_qubits(self._handle))

    @property
    def n_x_stabs(self) -> int:
        return int(_lib.moonlab_libirrep_qec_n_x_stabs(self._handle))

    @property
    def n_z_stabs(self) -> int:
        return int(_lib.moonlab_libirrep_qec_n_z_stabs(self._handle))

    @property
    def logical_qubits(self) -> int:
        return int(_lib.moonlab_libirrep_qec_logical_qubits(self._handle))

    @property
    def distance(self) -> int:
        """Brute-force code distance.  Memoised by the C bridge; first
        call can be expensive on larger codes (Pauli-weight enumeration
        is O(C(n, w) * 3^w))."""
        return int(_lib.moonlab_libirrep_qec_distance(self._handle))

    def x_check_row(self, row: int) -> bytes:
        """Length-``n_qubits`` byte array (0 / 1) of the X-stabiliser
        support at the given row index."""
        n = self.n_qubits
        buf = (ctypes.c_ubyte * n)()
        rc = _lib.moonlab_libirrep_qec_get_x_check_row(self._handle, int(row), buf)
        _raise_for(rc, f"x_check_row({row})")
        return bytes(buf)

    def z_check_row(self, row: int) -> bytes:
        """Length-``n_qubits`` byte array (0 / 1) of the Z-stabiliser
        support at the given row index."""
        n = self.n_qubits
        buf = (ctypes.c_ubyte * n)()
        rc = _lib.moonlab_libirrep_qec_get_z_check_row(self._handle, int(row), buf)
        _raise_for(rc, f"z_check_row({row})")
        return bytes(buf)

    def __repr__(self) -> str:
        return (f"<LibirrepQecCode [[{self.n_qubits}, "
                f"{self.logical_qubits}, ?]] "
                f"(m_X={self.n_x_stabs}, m_Z={self.n_z_stabs})>")


__all__ = [
    "LibirrepQecCode",
    "LibirrepError",
    "LibirrepNotBuiltError",
    "is_available",
]
