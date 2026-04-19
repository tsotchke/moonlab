"""Moonlab real-space topology bindings.

Exposes the matrix-free Chebyshev-KPM local Chern marker on the
Qi-Wu-Zhang Chern insulator, with optional on-site modulation for
quasicrystal Chern mosaics.

Example:
    >>> from moonlab import ChernKPM
    >>> sys = ChernKPM(L=40, m=-1.0, n_cheby=140)
    >>> mosaic = sys.bulk_map(4, 36)        # 32 x 32 array of c(r)
    >>> mosaic.mean()                        # ~1.0 in topological phase
"""

import ctypes
from typing import Optional

import numpy as np

from .core import _lib


_lib.chern_kpm_create.argtypes = [
    ctypes.c_size_t, ctypes.c_double, ctypes.c_size_t,
]
_lib.chern_kpm_create.restype = ctypes.c_void_p

_lib.chern_kpm_free.argtypes = [ctypes.c_void_p]
_lib.chern_kpm_free.restype = None

_lib.chern_kpm_local_marker.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
]
_lib.chern_kpm_local_marker.restype = ctypes.c_double

_lib.chern_kpm_bulk_sum.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
]
_lib.chern_kpm_bulk_sum.restype = ctypes.c_double

_lib.chern_kpm_bulk_map.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
]
_lib.chern_kpm_bulk_map.restype = ctypes.c_int

_lib.chern_kpm_set_modulation.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_double,
]
_lib.chern_kpm_set_modulation.restype = ctypes.c_int

_lib.chern_kpm_cn_modulation.argtypes = [
    ctypes.c_size_t, ctypes.c_int, ctypes.c_double, ctypes.c_double,
]
_lib.chern_kpm_cn_modulation.restype = ctypes.POINTER(ctypes.c_double)


class ChernKPM:
    """Matrix-free Chebyshev-KPM local Chern marker on QWZ.

    Args:
        L:        linear lattice size (L x L sites, 2 orbitals each).
        m:        QWZ mass parameter. -2 < m < 0 topological (C=+1),
                  0 < m < 2 topological (C=-1), |m| > 2 trivial.
        n_cheby:  Chebyshev expansion order. 80-200 is typical.
    """

    def __init__(self, L: int, m: float, n_cheby: int = 120) -> None:
        if L < 3:
            raise ValueError(f"L must be >= 3, got {L}")
        if n_cheby < 8:
            raise ValueError(f"n_cheby must be >= 8, got {n_cheby}")
        self._handle = _lib.chern_kpm_create(
            ctypes.c_size_t(L), ctypes.c_double(m), ctypes.c_size_t(n_cheby))
        if not self._handle:
            raise MemoryError("chern_kpm_create returned NULL")
        self._L = L
        self._m = m
        self._mod_owned: Optional[np.ndarray] = None

    def __del__(self) -> None:
        if getattr(self, "_handle", None):
            _lib.chern_kpm_free(self._handle)
            self._handle = None

    @property
    def L(self) -> int:
        return self._L

    @property
    def m(self) -> float:
        return self._m

    def local_marker(self, x: int, y: int) -> float:
        """Compute the local Chern marker c(x, y)."""
        if not (0 <= x < self._L and 0 <= y < self._L):
            raise ValueError(f"(x,y) = ({x},{y}) out of range [0, {self._L})")
        return float(_lib.chern_kpm_local_marker(
            self._handle, ctypes.c_size_t(x), ctypes.c_size_t(y)))

    def bulk_sum(self, rmin: int, rmax: int) -> float:
        """Sum local markers over [rmin, rmax)^2."""
        if not (0 <= rmin < rmax <= self._L):
            raise ValueError(
                f"bad bulk range [{rmin}, {rmax}) for L={self._L}")
        return float(_lib.chern_kpm_bulk_sum(
            self._handle, ctypes.c_size_t(rmin), ctypes.c_size_t(rmax)))

    def bulk_map(self, rmin: int, rmax: int) -> np.ndarray:
        """Return a (side, side) NumPy array of c(x, y) over the bulk
        patch [rmin, rmax) x [rmin, rmax). Row-major with row index y,
        column index x, each offset by rmin.

        Parallelised over sites via OpenMP inside libquantumsim.
        """
        if not (0 <= rmin < rmax <= self._L):
            raise ValueError(
                f"bad bulk range [{rmin}, {rmax}) for L={self._L}")
        side = rmax - rmin
        out = np.zeros((side, side), dtype=np.float64)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        rc = _lib.chern_kpm_bulk_map(
            self._handle, ctypes.c_size_t(rmin),
            ctypes.c_size_t(rmax), ptr)
        if rc != 0:
            raise RuntimeError(f"chern_kpm_bulk_map failed (rc={rc})")
        return out

    def set_cn_modulation(self, n: int, Q: float, V0: float) -> None:
        """Attach a C_n-rotationally-symmetric cosine modulation
            V(r) = V0 * sum_i cos(q_i . r)
        with |q_i| = Q and angles 2*pi*i/n. Common choices:
          - n=4: square quasicrystal
          - n=8: octagonal quasicrystal
          - n=10: decagonal quasicrystal
        """
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        # Build V_per_site via the C helper to ensure layout matches.
        cptr = _lib.chern_kpm_cn_modulation(
            ctypes.c_size_t(self._L), ctypes.c_int(n),
            ctypes.c_double(Q), ctypes.c_double(V0))
        if not cptr:
            raise MemoryError("chern_kpm_cn_modulation returned NULL")
        # Copy into a NumPy array we own so GC manages the lifetime
        # with the Python side; the C side stores a borrowed pointer,
        # so we keep the buffer alive via self._mod_owned.
        arr = np.ctypeslib.as_array(cptr, shape=(self._L * self._L,)).copy()
        self._mod_owned = arr
        # free the malloc'd C buffer (we copied its contents).
        libc = ctypes.CDLL(None)
        libc.free(cptr)
        V_max = float(n) * abs(V0)
        pptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        rc = _lib.chern_kpm_set_modulation(
            self._handle, pptr, ctypes.c_double(V_max))
        if rc != 0:
            raise RuntimeError(f"chern_kpm_set_modulation failed (rc={rc})")

    def clear_modulation(self) -> None:
        """Remove any attached modulation."""
        _lib.chern_kpm_set_modulation(
            self._handle, None, ctypes.c_double(0.0))
        self._mod_owned = None


_lib.moonlab_qwz_chern.argtypes = [
    ctypes.c_double, ctypes.c_size_t, ctypes.POINTER(ctypes.c_double),
]
_lib.moonlab_qwz_chern.restype = ctypes.c_int

# ---- Internal (non-ABI) QGT entry points for phase-diagram work -----

_lib.qgt_model_qwz.argtypes = [ctypes.c_double]
_lib.qgt_model_qwz.restype = ctypes.c_void_p

_lib.qgt_model_haldane.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
]
_lib.qgt_model_haldane.restype = ctypes.c_void_p

_lib.qgt_free.argtypes = [ctypes.c_void_p]
_lib.qgt_free.restype = None


class _CBerryGrid(ctypes.Structure):
    _fields_ = [
        ("N", ctypes.c_size_t),
        ("berry", ctypes.POINTER(ctypes.c_double)),
        ("chern", ctypes.c_double),
    ]


_lib.qgt_berry_grid.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(_CBerryGrid),
]
_lib.qgt_berry_grid.restype = ctypes.c_int

_lib.qgt_berry_grid_free.argtypes = [ctypes.POINTER(_CBerryGrid)]
_lib.qgt_berry_grid_free.restype = None

_lib.qgt_model_ssh.argtypes = [ctypes.c_double, ctypes.c_double]
_lib.qgt_model_ssh.restype = ctypes.c_void_p

_lib.qgt_free_1d.argtypes = [ctypes.c_void_p]
_lib.qgt_free_1d.restype = None

_lib.qgt_winding_1d.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_double),
]
_lib.qgt_winding_1d.restype = ctypes.c_int


def berry_grid_qwz(m: float, N: int = 32) -> np.ndarray:
    """Return an (N, N) NumPy array of per-plaquette Berry curvature
    for the QWZ model at mass ``m``. Units: radians per plaquette, so
    ``sum(grid) / (2*pi) == Chern`` up to discretisation.

    Useful for plotting phase diagrams and visualising concentration
    of curvature near Dirac points.
    """
    if N < 4:
        raise ValueError(f"N must be >= 4, got {N}")
    sys = _lib.qgt_model_qwz(ctypes.c_double(m))
    if not sys:
        raise MemoryError("qgt_model_qwz returned NULL")
    cg = _CBerryGrid()
    rc = _lib.qgt_berry_grid(sys, ctypes.c_size_t(N), ctypes.byref(cg))
    if rc != 0:
        _lib.qgt_free(sys)
        raise RuntimeError(f"qgt_berry_grid failed (rc={rc})")
    arr = np.ctypeslib.as_array(cg.berry, shape=(N * N,)).reshape(N, N).copy()
    _lib.qgt_berry_grid_free(ctypes.byref(cg))
    _lib.qgt_free(sys)
    return arr


def berry_grid_haldane(t1: float, t2: float, phi: float, M: float,
                       N: int = 48) -> np.ndarray:
    """Per-plaquette Berry curvature of the Haldane model on an N x N
    Brillouin-zone grid.
    """
    if N < 4:
        raise ValueError(f"N must be >= 4, got {N}")
    sys = _lib.qgt_model_haldane(
        ctypes.c_double(t1), ctypes.c_double(t2),
        ctypes.c_double(phi), ctypes.c_double(M))
    cg = _CBerryGrid()
    _lib.qgt_berry_grid(sys, ctypes.c_size_t(N), ctypes.byref(cg))
    arr = np.ctypeslib.as_array(cg.berry, shape=(N * N,)).reshape(N, N).copy()
    _lib.qgt_berry_grid_free(ctypes.byref(cg))
    _lib.qgt_free(sys)
    return arr


def ssh_winding(t1: float, t2: float, N: int = 64) -> int:
    """Integer winding number of the SSH model via the 1D Zak phase.
    Topological (winding=+1) for ``|t2| > |t1|``; trivial otherwise.
    """
    sys = _lib.qgt_model_ssh(ctypes.c_double(t1), ctypes.c_double(t2))
    if not sys:
        raise MemoryError("qgt_model_ssh returned NULL")
    raw = ctypes.c_double(0.0)
    w = _lib.qgt_winding_1d(sys, ctypes.c_size_t(N), ctypes.byref(raw))
    _lib.qgt_free_1d(sys)
    return int(w)


def qwz_chern(m: float, N: int = 32) -> int:
    """Compute the integer Chern number of QWZ at mass ``m`` via the
    momentum-space Fukui-Hatsugai-Suzuki method on an N x N BZ grid.

    Routes through the stable Moonlab ABI entry point
    `moonlab_qwz_chern`, so the same call works for any sibling
    library (QGTL, lilirrep, SbNN) linking libquantumsim.
    """
    if N < 4:
        raise ValueError(f"N must be >= 4, got {N}")
    out = ctypes.c_double(0.0)
    c = _lib.moonlab_qwz_chern(
        ctypes.c_double(m), ctypes.c_size_t(N), ctypes.byref(out))
    if c == -2147483648:  # INT_MIN sentinel
        raise RuntimeError("moonlab_qwz_chern failed")
    return int(c)


__all__ = [
    "ChernKPM", "qwz_chern", "berry_grid_qwz",
    "berry_grid_haldane", "ssh_winding",
]
