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
        # Free the malloc'd C buffer (we copied its contents).  argtypes
        # must be set explicitly -- ctypes' default int argument
        # marshalling truncates 64-bit pointers on LLP64 targets
        # (Windows), corrupting the free() call.  Same pattern as
        # ca_mps.py's z2_lgt_1d_build.
        libc = ctypes.CDLL(None)
        libc.free.argtypes = [ctypes.c_void_p]
        libc.free.restype = None
        libc.free(ctypes.cast(cptr, ctypes.c_void_p))
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

# ---- v0.3: gauge-invariant integrators (projector trace + parallel
#       transport) and n-band primitives.

_lib.qgt_berry_grid_proj.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(_CBerryGrid),
]
_lib.qgt_berry_grid_proj.restype = ctypes.c_int

_lib.qgt_berry_grid_pt.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(_CBerryGrid),
]
_lib.qgt_berry_grid_pt.restype = ctypes.c_int

# 4-band Z_2 (Kane-Mele, BHZ) and the Pfaffian-sign 1D BdG Z_2 (Kitaev).
_lib.qgt_z2_invariant.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int),
]
_lib.qgt_z2_invariant.restype = ctypes.c_int

_lib.qgt_z2_invariant_1d_bdg.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
]
_lib.qgt_z2_invariant_1d_bdg.restype = ctypes.c_int

# n-band model factories.
_lib.qgt_free_nband.argtypes = [ctypes.c_void_p]
_lib.qgt_free_nband.restype = None

_lib.qgt_berry_grid_nband.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(_CBerryGrid),
]
_lib.qgt_berry_grid_nband.restype = ctypes.c_int

_lib.qgt_model_kane_mele.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
]
_lib.qgt_model_kane_mele.restype = ctypes.c_void_p

_lib.qgt_model_bhz.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
]
_lib.qgt_model_bhz.restype = ctypes.c_void_p

_lib.qgt_model_kitaev_chain.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
]
_lib.qgt_model_kitaev_chain.restype = ctypes.c_void_p

_lib.qgt_model_hofstadter.argtypes = [
    ctypes.c_double, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
]
_lib.qgt_model_hofstadter.restype = ctypes.c_void_p


def berry_grid_qwz(m: float, N: int = 32) -> np.ndarray:
    """Return an (N, N) NumPy array of per-plaquette Berry curvature
    for the QWZ model at mass ``m`` via the Fukui-Hatsugai-Suzuki
    link-variable integrator.  Units: radians per plaquette, so
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


def berry_grid_qwz_proj(m: float, N: int = 48) -> np.ndarray:
    """Per-plaquette Berry curvature of the QWZ model on an N x N
    Brillouin-zone grid via the gauge-free projector-trace integrator
    ``F_xy(k) = -2 Im Tr[ P (d_x P) (d_y P) ]``.

    Returns the same units as :func:`berry_grid_qwz` (radians per
    plaquette; ``arr.sum() / (2*pi)`` integrates to the Chern number).
    The projector-trace formulation is preferred for plotting near
    gap closings, where link-variable methods can carry gauge
    artefacts.

    Args:
        m: QWZ mass parameter.
        N: BZ grid size per axis; must be >= 4.

    Returns:
        ``(N, N)`` float64 NumPy array.
    """
    if N < 4:
        raise ValueError(f"N must be >= 4, got {N}")
    sys = _lib.qgt_model_qwz(ctypes.c_double(m))
    if not sys:
        raise MemoryError("qgt_model_qwz returned NULL")
    cg = _CBerryGrid()
    rc = _lib.qgt_berry_grid_proj(sys, ctypes.c_size_t(N), ctypes.byref(cg))
    if rc != 0:
        _lib.qgt_free(sys)
        raise RuntimeError(f"qgt_berry_grid_proj failed (rc={rc})")
    arr = np.ctypeslib.as_array(cg.berry, shape=(N * N,)).reshape(N, N).copy()
    _lib.qgt_berry_grid_free(ctypes.byref(cg))
    _lib.qgt_free(sys)
    return arr


def berry_grid_qwz_pt(m: float, N: int = 48) -> np.ndarray:
    """Per-plaquette Berry curvature of the QWZ model on an N x N
    Brillouin-zone grid via the parallel-transport-gauge integrator.

    Companion to :func:`berry_grid_qwz` and :func:`berry_grid_qwz_proj`:
    on every gapped phase point all three return arrays that integrate
    to the same integer Chern number, but their per-plaquette
    distributions differ near gap closings where gauge fixing matters.
    Use this routine when you need the curvature distribution under a
    smooth (parallel-transported) gauge.

    Args:
        m: QWZ mass parameter.
        N: BZ grid size per axis; must be >= 4.

    Returns:
        ``(N, N)`` float64 NumPy array.
    """
    if N < 4:
        raise ValueError(f"N must be >= 4, got {N}")
    sys = _lib.qgt_model_qwz(ctypes.c_double(m))
    if not sys:
        raise MemoryError("qgt_model_qwz returned NULL")
    cg = _CBerryGrid()
    rc = _lib.qgt_berry_grid_pt(sys, ctypes.c_size_t(N), ctypes.byref(cg))
    if rc != 0:
        _lib.qgt_free(sys)
        raise RuntimeError(f"qgt_berry_grid_pt failed (rc={rc})")
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


def chern_qwz_proj(m: float, N: int = 48) -> int:
    """Integer Chern number of the Qi-Wu-Zhang two-band Chern insulator
    via the gauge-free projector-trace integrator.

    The projector-trace formulation
    ``F_xy(k) = -2 Im Tr[ P (d_x P) (d_y P) ]`` with ``P = sum |u><u|``
    over occupied bands is manifestly gauge-invariant and avoids the
    band-degeneracy artefacts of link-variable integrators near gap
    closings.  See Provost-Vallee (1980) and Resta (2011).

    Args:
        m: QWZ mass parameter.
        N: Brillouin-zone grid size (per axis), must be >= 4.

    Returns:
        Integer Chern number (0, +-1).
    """
    if N < 4:
        raise ValueError(f"N must be >= 4, got {N}")
    sys = _lib.qgt_model_qwz(ctypes.c_double(m))
    if not sys:
        raise MemoryError("qgt_model_qwz returned NULL")
    cg = _CBerryGrid()
    rc = _lib.qgt_berry_grid_proj(sys, ctypes.c_size_t(N), ctypes.byref(cg))
    if rc != 0:
        _lib.qgt_free(sys)
        raise RuntimeError(f"qgt_berry_grid_proj failed (rc={rc})")
    chern = int(round(cg.chern))
    _lib.qgt_berry_grid_free(ctypes.byref(cg))
    _lib.qgt_free(sys)
    return chern


def chern_qwz_parallel_transport(m: float, N: int = 48) -> int:
    """Integer Chern number of QWZ via the parallel-transport-gauge
    eigenvector integrator.

    The eigenvector at each ``k`` is phase-fixed against its neighbour;
    the resulting smooth gauge gives a Berry-curvature plaquette flux
    that integrates to the Chern number.  Equivalent to
    :func:`chern_qwz_proj` and :func:`qwz_chern` for any gapped phase
    point; provided for cross-validation.
    """
    if N < 4:
        raise ValueError(f"N must be >= 4, got {N}")
    sys = _lib.qgt_model_qwz(ctypes.c_double(m))
    if not sys:
        raise MemoryError("qgt_model_qwz returned NULL")
    cg = _CBerryGrid()
    rc = _lib.qgt_berry_grid_pt(sys, ctypes.c_size_t(N), ctypes.byref(cg))
    if rc != 0:
        _lib.qgt_free(sys)
        raise RuntimeError(f"qgt_berry_grid_pt failed (rc={rc})")
    chern = int(round(cg.chern))
    _lib.qgt_berry_grid_free(ctypes.byref(cg))
    _lib.qgt_free(sys)
    return chern


def kane_mele_z2(t: float = 1.0, lambda_so: float = 0.06,
                 lambda_r: float = 0.0, lambda_v: float = 0.10,
                 N: int = 48) -> int:
    """Z_2 invariant of the Kane-Mele model on the honeycomb lattice.

    Reference: C. L. Kane and E. J. Mele, "Z_2 topological order and
    the quantum spin Hall effect", *Phys. Rev. Lett.* **95**, 146802
    (2005).  Computed via the n-field method of Fukui and Hatsugai
    (2007) on a 4-band Bloch system at half filling.

    Args:
        t: Nearest-neighbour hopping amplitude.
        lambda_so: Intrinsic spin-orbit coupling (the Z_2 driver).
        lambda_r: Rashba spin-orbit coupling (set to 0 for the
            canonical S_z-conserving regime).
        lambda_v: Sublattice mass (staggered on-site potential).
        N: BZ grid side (must be even, >= 8).

    Returns:
        ``1`` (QSH topological) when
        ``|lambda_v| < 3 sqrt(3) |lambda_so|``, ``0`` otherwise.
    """
    if N < 8 or N % 2 != 0:
        raise ValueError(f"N must be even and >= 8, got {N}")
    sys = _lib.qgt_model_kane_mele(
        ctypes.c_double(t), ctypes.c_double(lambda_so),
        ctypes.c_double(lambda_r), ctypes.c_double(lambda_v))
    if not sys:
        raise MemoryError("qgt_model_kane_mele returned NULL")
    z2 = ctypes.c_int(-1)
    rc = _lib.qgt_z2_invariant(sys, ctypes.c_size_t(N), ctypes.byref(z2))
    _lib.qgt_free_nband(sys)
    if rc != 0:
        raise RuntimeError(f"qgt_z2_invariant failed (rc={rc})")
    return int(z2.value)


def bhz_z2(A: float = 1.0, B: float = 1.0, M: float = 3.0,
           N: int = 48) -> int:
    """Z_2 invariant of the Bernevig-Hughes-Zhang model.

    Reference: B. A. Bernevig, T. L. Hughes, and S.-C. Zhang,
    "Quantum spin Hall effect and topological phase transition in
    HgTe quantum wells", *Science* **314**, 1757 (2006).

    The lattice regularisation gives the QSH window
    ``0 < M / B < 8`` (X-corner closings at ``M = 4B`` cancel; the
    M-corner closing at ``M = 8B`` re-trivialises).
    """
    if N < 8 or N % 2 != 0:
        raise ValueError(f"N must be even and >= 8, got {N}")
    sys = _lib.qgt_model_bhz(
        ctypes.c_double(A), ctypes.c_double(B), ctypes.c_double(M))
    if not sys:
        raise MemoryError("qgt_model_bhz returned NULL")
    z2 = ctypes.c_int(-1)
    rc = _lib.qgt_z2_invariant(sys, ctypes.c_size_t(N), ctypes.byref(z2))
    _lib.qgt_free_nband(sys)
    if rc != 0:
        raise RuntimeError(f"qgt_z2_invariant failed (rc={rc})")
    return int(z2.value)


def kitaev_chain_z2(t: float = 1.0, mu: float = 0.5,
                    delta: float = 1.0) -> int:
    """Z_2 invariant of the Kitaev p-wave superconducting chain via
    the Pfaffian-sign product at the time-reversal-invariant momenta.

    Reference: A. Y. Kitaev, "Unpaired Majorana fermions in quantum
    wires", *Physics-Uspekhi* **44**, 131 (2001).

    Args:
        t: Nearest-neighbour hopping amplitude.
        mu: Chemical potential.
        delta: p-wave pairing amplitude.

    Returns:
        ``1`` when ``|mu| < 2|t|`` (topological phase with Majorana
        zero modes at the boundaries), ``0`` otherwise.
    """
    sys = _lib.qgt_model_kitaev_chain(
        ctypes.c_double(t), ctypes.c_double(mu), ctypes.c_double(delta))
    if not sys:
        raise MemoryError("qgt_model_kitaev_chain returned NULL")
    z2 = ctypes.c_int(-1)
    rc = _lib.qgt_z2_invariant_1d_bdg(sys, ctypes.byref(z2))
    _lib.qgt_free_1d(sys)
    if rc != 0:
        raise RuntimeError(f"qgt_z2_invariant_1d_bdg failed (rc={rc})")
    return int(z2.value)


def hofstadter_chern(p: int = 1, q: int = 3, n_occupied: int = 1,
                     t: float = 1.0, N: int = 32) -> int:
    """Total Chern number of the lowest ``n_occupied`` magnetic
    sub-bands of the Harper-Hofstadter model at flux ``phi = p / q``.

    Reference: D. R. Hofstadter, "Energy levels and wave functions of
    Bloch electrons in rational and irrational magnetic fields",
    *Phys. Rev. B* **14**, 2239 (1976).

    For ``q = 3`` and ``n_occupied = 1`` returns ``+1`` (lowest band);
    for ``n_occupied = 2`` returns ``-1`` (lowest two bands sum to
    ``+1 + (-2) = -1``).
    """
    if q < 2:
        raise ValueError(f"q must be >= 2, got {q}")
    if not (1 <= n_occupied <= q - 1):
        raise ValueError(
            f"n_occupied must be in [1, q-1] = [1, {q - 1}], got {n_occupied}")
    if N < 8:
        raise ValueError(f"N must be >= 8, got {N}")
    sys = _lib.qgt_model_hofstadter(
        ctypes.c_double(t), ctypes.c_size_t(p), ctypes.c_size_t(q),
        ctypes.c_size_t(n_occupied))
    if not sys:
        raise MemoryError("qgt_model_hofstadter returned NULL")
    cg = _CBerryGrid()
    rc = _lib.qgt_berry_grid_nband(sys, ctypes.c_size_t(N), ctypes.byref(cg))
    if rc != 0:
        _lib.qgt_free_nband(sys)
        raise RuntimeError(f"qgt_berry_grid_nband failed (rc={rc})")
    chern = int(round(cg.chern))
    _lib.qgt_berry_grid_free(ctypes.byref(cg))
    _lib.qgt_free_nband(sys)
    return chern


__all__ = [
    "ChernKPM", "qwz_chern", "berry_grid_qwz",
    "berry_grid_haldane", "ssh_winding",
    # v0.3: gauge-invariant integrators + n-band primitives.
    "chern_qwz_proj", "chern_qwz_parallel_transport",
    "kane_mele_z2", "bhz_z2", "kitaev_chain_z2", "hofstadter_chern",
    # v0.3.2: curvature grids for the gauge-invariant integrators.
    "berry_grid_qwz_proj", "berry_grid_qwz_pt",
]
