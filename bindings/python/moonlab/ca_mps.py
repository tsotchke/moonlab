"""Moonlab Clifford-Assisted MPS (CA-MPS) bindings.

Exposes the hybrid `|psi> = D|phi>` representation that absorbs the
Clifford structure of a circuit into a tableau and only pushes the
non-Clifford rotations into the MPS factor.  See
`docs/research/ca_mps.md` for the full theory.

Highlights:
- :class:`CAMPS` -- the state object plus the Clifford / non-Clifford
  gate surface.
- :func:`var_d_run` -- variational-D ground-state search (the
  alternating greedy-Clifford + imag-time loop).
- :func:`gauge_warmstart` -- standalone gauge-aware Clifford
  preparation: takes commuting Pauli generators and emits a Clifford
  that places `|psi>` in their simultaneous +1 eigenspace.
- :func:`z2_lgt_1d_build` -- 1+1D Z2 lattice gauge theory Pauli sum
  (matter + gauge link, exactly gauge-invariant kinetic terms).
- :func:`z2_lgt_1d_gauss_law` -- the Gauss-law operator at an
  interior matter site.
- :func:`status_string` -- diagnostic stringifier for any Moonlab
  status code.

All of these bind to the v0.2.1 stable ABI in
`src/applications/moonlab_export.h`; the Python layer is a thin
ctypes wrapper that mirrors the C signatures.

Quick example -- gauge-aware ground-state prep on Z2 LGT:

    >>> from moonlab.ca_mps import (
    ...     CAMPS, var_d_run, z2_lgt_1d_build, z2_lgt_1d_gauss_law,
    ...     WARMSTART_STABILIZER_SUBGROUP)
    >>> import numpy as np
    >>>
    >>> # Build the Z2 LGT Hamiltonian on 4 matter sites (7 qubits).
    >>> paulis, coeffs = z2_lgt_1d_build(N=4, t=1.0, h=0.5, m=0.5,
    ...                                   gauss_penalty=0.0)
    >>>
    >>> # Stack the interior Gauss-law operators as a (k, n) bytes block.
    >>> gens = np.stack([z2_lgt_1d_gauss_law(N=4, site_x=x)
    ...                  for x in (1, 2)])
    >>>
    >>> # Run var-D with the gauge-aware warmstart.
    >>> state = CAMPS(num_qubits=7, max_bond_dim=32)
    >>> energy = var_d_run(state, paulis, coeffs,
    ...                     warmstart=WARMSTART_STABILIZER_SUBGROUP,
    ...                     stab_paulis=gens)
"""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from .core import _lib

__all__ = [
    "CAMPS",
    "WARMSTART_IDENTITY",
    "WARMSTART_H_ALL",
    "WARMSTART_DUAL_TFIM",
    "WARMSTART_FERRO_TFIM",
    "WARMSTART_STABILIZER_SUBGROUP",
    "var_d_run",
    "gauge_warmstart",
    "z2_lgt_1d_build",
    "z2_lgt_1d_gauss_law",
    "status_string",
]


# Warmstart codes mirror the C ABI int convention in
# moonlab_ca_mps_var_d_run; see the docstring on that function in
# `src/applications/moonlab_export.h`.
WARMSTART_IDENTITY = 0
WARMSTART_H_ALL = 1
WARMSTART_DUAL_TFIM = 2
WARMSTART_FERRO_TFIM = 3
WARMSTART_STABILIZER_SUBGROUP = 4


# ---------------------------------------------------------------- #
# ABI signature setup.                                             #
# ---------------------------------------------------------------- #

_lib.moonlab_ca_mps_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
_lib.moonlab_ca_mps_create.restype = ctypes.c_void_p

_lib.moonlab_ca_mps_free.argtypes = [ctypes.c_void_p]
_lib.moonlab_ca_mps_free.restype = None

_lib.moonlab_ca_mps_num_qubits.argtypes = [ctypes.c_void_p]
_lib.moonlab_ca_mps_num_qubits.restype = ctypes.c_uint32

_lib.moonlab_ca_mps_current_bond_dim.argtypes = [ctypes.c_void_p]
_lib.moonlab_ca_mps_current_bond_dim.restype = ctypes.c_uint32

_lib.moonlab_ca_mps_norm.argtypes = [ctypes.c_void_p]
_lib.moonlab_ca_mps_norm.restype = ctypes.c_double

# Clifford gates: int op(handle, q [, q2]).
for _name in ("h", "s", "sdag", "x", "y", "z"):
    _fn = getattr(_lib, f"moonlab_ca_mps_{_name}")
    _fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _fn.restype = ctypes.c_int

for _name in ("cnot", "cz", "swap"):
    _fn = getattr(_lib, f"moonlab_ca_mps_{_name}")
    _fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    _fn.restype = ctypes.c_int

# Non-Clifford rotations.
for _name in ("rx", "ry", "rz", "phase"):
    _fn = getattr(_lib, f"moonlab_ca_mps_{_name}")
    _fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_double]
    _fn.restype = ctypes.c_int

_lib.moonlab_ca_mps_t_gate.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
_lib.moonlab_ca_mps_t_gate.restype = ctypes.c_int
_lib.moonlab_ca_mps_t_dagger.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
_lib.moonlab_ca_mps_t_dagger.restype = ctypes.c_int

_lib.moonlab_ca_mps_normalize.argtypes = [ctypes.c_void_p]
_lib.moonlab_ca_mps_normalize.restype = ctypes.c_int

# var-D run.
_lib.moonlab_ca_mps_var_d_run.argtypes = [
    ctypes.c_void_p,                    # state
    ctypes.POINTER(ctypes.c_uint8),     # paulis
    ctypes.POINTER(ctypes.c_double),    # coeffs
    ctypes.c_uint32,                    # num_terms
    ctypes.c_uint32,                    # max_outer_iters
    ctypes.c_double,                    # imag_time_dtau
    ctypes.c_uint32,                    # imag_time_steps_per_outer
    ctypes.c_uint32,                    # clifford_passes_per_outer
    ctypes.c_int,                       # composite_2gate
    ctypes.c_int,                       # warmstart
    ctypes.POINTER(ctypes.c_uint8),     # stab_paulis (NULLable)
    ctypes.c_uint32,                    # stab_num_gens
    ctypes.POINTER(ctypes.c_double),    # out_final_energy (NULLable)
]
_lib.moonlab_ca_mps_var_d_run.restype = ctypes.c_int

# Standalone gauge warmstart.
_lib.moonlab_ca_mps_gauge_warmstart.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_uint32,
]
_lib.moonlab_ca_mps_gauge_warmstart.restype = ctypes.c_int

# Z2 LGT.
_lib.moonlab_z2_lgt_1d_build.argtypes = [
    ctypes.c_uint32, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_uint32),
]
_lib.moonlab_z2_lgt_1d_build.restype = ctypes.c_int

_lib.moonlab_z2_lgt_1d_gauss_law.argtypes = [
    ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint8)
]
_lib.moonlab_z2_lgt_1d_gauss_law.restype = ctypes.c_int

# Status stringifier.
_lib.moonlab_status_string.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.moonlab_status_string.restype = ctypes.c_char_p


# ---------------------------------------------------------------- #
# CAMPS handle wrapper.                                            #
# ---------------------------------------------------------------- #


class CAMPS:
    """Clifford-Assisted MPS state handle.

    Wraps the opaque `moonlab_ca_mps_t*` from the ABI.  The handle
    is freed automatically on garbage collection.
    """

    def __init__(self, num_qubits: int, max_bond_dim: int = 32):
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        if max_bond_dim < 1:
            raise ValueError("max_bond_dim must be >= 1")
        self._h = _lib.moonlab_ca_mps_create(num_qubits, max_bond_dim)
        if not self._h:
            raise MemoryError("moonlab_ca_mps_create returned NULL")

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            _lib.moonlab_ca_mps_free(h)
            self._h = None

    @property
    def num_qubits(self) -> int:
        return int(_lib.moonlab_ca_mps_num_qubits(self._h))

    @property
    def bond_dim(self) -> int:
        return int(_lib.moonlab_ca_mps_current_bond_dim(self._h))

    @property
    def norm(self) -> float:
        return float(_lib.moonlab_ca_mps_norm(self._h))

    # ------------------------------------------------------------ #
    # Clifford gates: tableau-only, no MPS cost.                  #
    # ------------------------------------------------------------ #

    def h(self, q: int) -> None: _check(_lib.moonlab_ca_mps_h(self._h, q))
    def s(self, q: int) -> None: _check(_lib.moonlab_ca_mps_s(self._h, q))
    def sdag(self, q: int) -> None: _check(_lib.moonlab_ca_mps_sdag(self._h, q))
    def x(self, q: int) -> None: _check(_lib.moonlab_ca_mps_x(self._h, q))
    def y(self, q: int) -> None: _check(_lib.moonlab_ca_mps_y(self._h, q))
    def z(self, q: int) -> None: _check(_lib.moonlab_ca_mps_z(self._h, q))
    def cnot(self, c: int, t: int) -> None:
        _check(_lib.moonlab_ca_mps_cnot(self._h, c, t))
    def cz(self, a: int, b: int) -> None:
        _check(_lib.moonlab_ca_mps_cz(self._h, a, b))
    def swap(self, a: int, b: int) -> None:
        _check(_lib.moonlab_ca_mps_swap(self._h, a, b))

    # ------------------------------------------------------------ #
    # Non-Clifford rotations: pushed into the MPS factor.         #
    # ------------------------------------------------------------ #

    def rx(self, q: int, theta: float) -> None:
        _check(_lib.moonlab_ca_mps_rx(self._h, q, theta))
    def ry(self, q: int, theta: float) -> None:
        _check(_lib.moonlab_ca_mps_ry(self._h, q, theta))
    def rz(self, q: int, theta: float) -> None:
        _check(_lib.moonlab_ca_mps_rz(self._h, q, theta))
    def t_gate(self, q: int) -> None:
        _check(_lib.moonlab_ca_mps_t_gate(self._h, q))
    def t_dagger(self, q: int) -> None:
        _check(_lib.moonlab_ca_mps_t_dagger(self._h, q))
    def phase(self, q: int, theta: float) -> None:
        _check(_lib.moonlab_ca_mps_phase(self._h, q, theta))

    def normalize(self) -> None:
        _check(_lib.moonlab_ca_mps_normalize(self._h))


def _check(rc: int) -> None:
    if rc != 0:
        raise RuntimeError(
            "Moonlab CA-MPS call failed: "
            + status_string(module=1, status=rc))


# ---------------------------------------------------------------- #
# Free functions.                                                  #
# ---------------------------------------------------------------- #


def _as_uint8_buf(arr) -> ctypes.POINTER(ctypes.c_uint8):
    """Return a (POINTER(c_uint8), keep_alive) tuple ensuring lifetime."""
    a = np.ascontiguousarray(arr, dtype=np.uint8)
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), a


def _as_double_buf(arr) -> ctypes.POINTER(ctypes.c_double):
    a = np.ascontiguousarray(arr, dtype=np.float64)
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), a


def gauge_warmstart(state: CAMPS, paulis) -> None:
    """Apply the gauge-aware stabilizer-subgroup warmstart Clifford.

    `paulis` is a (num_gens, num_qubits) array of pairwise-commuting
    Pauli generators in the Moonlab byte encoding (0=I, 1=X, 2=Y, 3=Z).
    The resulting state's Clifford prefactor D maps `|0^n>` into the
    simultaneous +1 eigenspace of every generator.  Raises
    `RuntimeError` if the generators don't pairwise commute or aren't
    independent.
    """
    p = np.ascontiguousarray(paulis, dtype=np.uint8)
    if p.ndim != 2:
        raise ValueError("paulis must be a 2-D (num_gens, num_qubits) array")
    num_gens, n = p.shape
    if n != state.num_qubits:
        raise ValueError(
            f"paulis num_qubits={n} != state.num_qubits={state.num_qubits}")
    rc = _lib.moonlab_ca_mps_gauge_warmstart(
        state._h, p.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_uint32(num_gens))
    _check(rc)


def var_d_run(state: CAMPS, paulis, coeffs,
              max_outer_iters: int = 25,
              imag_time_dtau: float = 0.10,
              imag_time_steps_per_outer: int = 4,
              clifford_passes_per_outer: int = 8,
              composite_2gate: bool = False,
              warmstart: int = WARMSTART_IDENTITY,
              stab_paulis=None) -> float:
    """Run the variational-D alternating ground-state search.

    Returns the final variational energy.  See
    `MATH.md` §11 for the algorithmic details and
    `docs/research/ca_mps.md` for the design.

    Parameters
    ----------
    state : CAMPS
        The state to mutate in place.
    paulis : array-like uint8, shape (num_terms, num_qubits)
        Pauli sum byte encoding.
    coeffs : array-like float64, shape (num_terms,)
        Real coefficients.
    warmstart : int
        One of WARMSTART_IDENTITY (0), WARMSTART_H_ALL (1),
        WARMSTART_DUAL_TFIM (2), WARMSTART_FERRO_TFIM (3),
        WARMSTART_STABILIZER_SUBGROUP (4).
    stab_paulis : optional array-like uint8, shape (k, num_qubits)
        Required when warmstart == WARMSTART_STABILIZER_SUBGROUP.
    """
    p = np.ascontiguousarray(paulis, dtype=np.uint8)
    c = np.ascontiguousarray(coeffs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("paulis must be (num_terms, num_qubits)")
    num_terms, n = p.shape
    if c.shape != (num_terms,):
        raise ValueError(
            f"coeffs shape {c.shape} != (num_terms,) = ({num_terms},)")
    if n != state.num_qubits:
        raise ValueError(
            f"paulis num_qubits={n} != state.num_qubits={state.num_qubits}")

    sp_ptr = ctypes.POINTER(ctypes.c_uint8)()
    sp_num = 0
    _keep = None
    if warmstart == WARMSTART_STABILIZER_SUBGROUP:
        if stab_paulis is None:
            raise ValueError(
                "warmstart == STABILIZER_SUBGROUP requires stab_paulis")
        s = np.ascontiguousarray(stab_paulis, dtype=np.uint8)
        if s.ndim != 2 or s.shape[1] != n:
            raise ValueError(
                f"stab_paulis shape {s.shape} != (k, {n})")
        sp_ptr = s.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        sp_num = s.shape[0]
        _keep = s

    out_e = ctypes.c_double(0.0)
    rc = _lib.moonlab_ca_mps_var_d_run(
        state._h,
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_uint32(num_terms),
        ctypes.c_uint32(max_outer_iters),
        ctypes.c_double(imag_time_dtau),
        ctypes.c_uint32(imag_time_steps_per_outer),
        ctypes.c_uint32(clifford_passes_per_outer),
        ctypes.c_int(1 if composite_2gate else 0),
        ctypes.c_int(int(warmstart)),
        sp_ptr,
        ctypes.c_uint32(sp_num),
        ctypes.byref(out_e),
    )
    _check(rc)
    del _keep
    return float(out_e.value)


def z2_lgt_1d_build(N: int, t: float = 1.0, h: float = 1.0,
                     m: float = 0.0, gauss_penalty: float = 0.0):
    """Build the 1+1D Z2 LGT Pauli sum on N matter sites.

    Returns ``(paulis, coeffs)`` where ``paulis`` is a
    ``(num_terms, 2*N - 1)`` uint8 array (encoding 0=I, 1=X, 2=Y, 3=Z)
    and ``coeffs`` is a ``(num_terms,)`` float64 array.  Memory is
    allocated by libquantumsim, copied into NumPy arrays here, then
    freed via the C runtime allocator (matching what the C side did).

    The kinetic terms are written in their exactly gauge-invariant
    form `K_x = -(t/2) X_{2x} Y_{2x+1} Y_{2x+2} + (t/2) Y_{2x} Y_{2x+1} X_{2x+2}`,
    which commutes with every interior Gauss-law operator term-by-
    term.  See `docs/research/var_d_lattice_gauge_theory.md`.
    """
    out_p = ctypes.POINTER(ctypes.c_uint8)()
    out_c = ctypes.POINTER(ctypes.c_double)()
    out_T = ctypes.c_uint32(0)
    out_n = ctypes.c_uint32(0)
    rc = _lib.moonlab_z2_lgt_1d_build(
        ctypes.c_uint32(N),
        ctypes.c_double(t), ctypes.c_double(h),
        ctypes.c_double(m), ctypes.c_double(gauss_penalty),
        ctypes.byref(out_p), ctypes.byref(out_c),
        ctypes.byref(out_T), ctypes.byref(out_n),
    )
    if rc != 0:
        raise RuntimeError(
            f"moonlab_z2_lgt_1d_build returned {rc}")

    T = int(out_T.value)
    n = int(out_n.value)
    paulis = np.ctypeslib.as_array(out_p, shape=(T, n)).copy()
    coeffs = np.ctypeslib.as_array(out_c, shape=(T,)).copy()

    # Free the C-side allocations.  The C builder uses calloc, so the
    # Python free() (via libc) is the matching deallocator.
    _libc = ctypes.CDLL(None)
    _libc.free.argtypes = [ctypes.c_void_p]
    _libc.free.restype = None
    _libc.free(ctypes.cast(out_p, ctypes.c_void_p))
    _libc.free(ctypes.cast(out_c, ctypes.c_void_p))
    return paulis, coeffs


def z2_lgt_1d_gauss_law(N: int, site_x: int) -> np.ndarray:
    """Return the interior Gauss-law operator at matter site `site_x`.

    `G_x = X_{2x-1} Z_{2x} X_{2x+1}`, with `1 <= site_x <= N - 2`.
    Output is a ``(2*N - 1,)`` uint8 array in Moonlab Pauli-byte
    encoding.
    """
    nq = 2 * N - 1
    out = np.zeros(nq, dtype=np.uint8)
    rc = _lib.moonlab_z2_lgt_1d_gauss_law(
        ctypes.c_uint32(N), ctypes.c_uint32(site_x),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    if rc != 0:
        raise ValueError(f"moonlab_z2_lgt_1d_gauss_law returned {rc}")
    return out


def status_string(module: int, status: int) -> str:
    """Pretty-print a Moonlab status code.

    `module` is one of the moonlab_status_module_t values:
    0=GENERIC, 1=CA_MPS, 2=CA_MPS_VAR_D, 3=CA_MPS_STAB_WARMSTART,
    4=CA_PEPS, 5=TN_STATE, 6=TN_GATE, 7=TN_MEASURE, 8=TENSOR,
    9=CONTRACT, 10=SVD_COMPRESS, 11=CLIFFORD, 12=PARTITION,
    13=DIST_GATE, 14=MPI_BRIDGE.
    """
    s = _lib.moonlab_status_string(int(module), int(status))
    if s is None:
        return f"<unknown module={module} status={status}>"
    return s.decode("utf-8", errors="replace")
