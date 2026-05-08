"""Matrix-product density operator (MPDO) noise simulator (v0.3).

Polynomial-cost simulation of noisy quantum circuits via the matrix-
product representation of the density matrix introduced by Verstraete,
Garcia-Ripoll, and Cirac (Phys. Rev. Lett. 93, 207204, 2004) and
developed for noisy circuit simulation by Werner et al. (Phys. Rev.
Lett. 116, 237201, 2016).

Each site tensor carries a four-dimensional physical leg holding the
vectorised local 2 x 2 density block, and a left/right virtual bond
of dimension at most ``max_bond_dim`` capturing site-site correlations.
A single-qubit Kraus channel acts as a 4 x 4 superoperator on the
physical leg in O(chi^2) time without growing the bond dimension.

Status: scaffold (v0.3.0).  Single-qubit Kraus channels are exposed;
two-qubit Kraus channels with SVD bond truncation land in v0.3.x.

Example:

    >>> from moonlab.mpdo import Mpdo
    >>> rho = Mpdo(num_qubits=4, max_bond_dim=16)
    >>> rho.apply_depolarizing(qubit=1, p=0.4)
    >>> z = rho.expect_pauli(qubit=1, pauli='Z')
    >>> abs(z - (1 - 4 * 0.4 / 3)) < 1e-12
    True
"""

import ctypes
from typing import Iterable, Union

import numpy as np

from .core import _lib

# ---- FFI signatures ------------------------------------------------

_lib.moonlab_mpdo_create.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
_lib.moonlab_mpdo_create.restype = ctypes.c_void_p

_lib.moonlab_mpdo_free.argtypes = [ctypes.c_void_p]
_lib.moonlab_mpdo_free.restype = None

_lib.moonlab_mpdo_clone.argtypes = [ctypes.c_void_p]
_lib.moonlab_mpdo_clone.restype = ctypes.c_void_p

_lib.moonlab_mpdo_num_qubits.argtypes = [ctypes.c_void_p]
_lib.moonlab_mpdo_num_qubits.restype = ctypes.c_uint32

_lib.moonlab_mpdo_max_bond_dim.argtypes = [ctypes.c_void_p]
_lib.moonlab_mpdo_max_bond_dim.restype = ctypes.c_uint32

_lib.moonlab_mpdo_current_bond_dim.argtypes = [ctypes.c_void_p]
_lib.moonlab_mpdo_current_bond_dim.restype = ctypes.c_uint32

_lib.moonlab_mpdo_trace.argtypes = [ctypes.c_void_p]
_lib.moonlab_mpdo_trace.restype = ctypes.c_double

_lib.moonlab_mpdo_apply_kraus_1q.argtypes = [
    ctypes.c_void_p, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_double),  # complex packed as double pairs
    ctypes.c_uint32,
]
_lib.moonlab_mpdo_apply_kraus_1q.restype = ctypes.c_int

for _name in (
    "moonlab_mpdo_apply_depolarizing_1q",
    "moonlab_mpdo_apply_amplitude_damping_1q",
    "moonlab_mpdo_apply_phase_damping_1q",
    "moonlab_mpdo_apply_bit_flip_1q",
    "moonlab_mpdo_apply_phase_flip_1q",
    "moonlab_mpdo_apply_bit_phase_flip_1q",
):
    _fn = getattr(_lib, _name)
    _fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_double]
    _fn.restype = ctypes.c_int

_lib.moonlab_mpdo_expect_pauli_1q.argtypes = [
    ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint8,
    ctypes.POINTER(ctypes.c_double),
]
_lib.moonlab_mpdo_expect_pauli_1q.restype = ctypes.c_int


# ---- Status / error codes from src/quantum/noise_mpdo.h ------------

MPDO_SUCCESS = 0
_MPDO_LABELS = {
    0: "success",
    -1: "invalid argument",
    -2: "qubit out of range",
    -3: "out of memory",
    -4: "backend failure",
}

_PAULI_CODE = {"I": 0, "X": 1, "Y": 2, "Z": 3,
               "i": 0, "x": 1, "y": 2, "z": 3}


def _check(rc: int, where: str) -> None:
    if rc != MPDO_SUCCESS:
        label = _MPDO_LABELS.get(rc, "unknown error")
        raise RuntimeError(f"{where} failed: {label} (rc={rc})")


def _kraus_array_to_double_buffer(
    kraus: Iterable[np.ndarray],
) -> "tuple[ctypes.Array, int]":
    """Pack ``num_kraus`` complex 2x2 Kraus matrices into a flat
    double buffer (real, imag interleaved per element, row-major over
    the 2x2 block)."""
    flat = []
    n = 0
    for k in kraus:
        a = np.asarray(k, dtype=np.complex128).reshape(-1)
        if a.shape != (4,):
            raise ValueError(
                f"each Kraus operator must be 2x2; got shape {k.shape}")
        for c in a:
            flat.append(c.real)
            flat.append(c.imag)
        n += 1
    buf = (ctypes.c_double * len(flat))(*flat)
    return buf, n


# ---- Public class --------------------------------------------------

class Mpdo:
    """Matrix-product density operator handle.

    The state is initialised in the product ``|0...0><0...0|`` at
    bond dimension 1.  Single-qubit Kraus channels act on the physical
    leg of the addressed site without growing the bond.

    Args:
        num_qubits: Number of qubits.  Must be >= 1.
        max_bond_dim: Bond-dimension cap.  16 is sufficient for
            ~50-qubit local-noise circuits at single-qubit error
            rates near 1e-3; 32 is recommended when ``Tr(rho)``
            drift becomes visible.
    """

    __slots__ = ("_handle",)

    def __init__(self, num_qubits: int, max_bond_dim: int = 16):
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        if max_bond_dim < 1:
            raise ValueError(
                f"max_bond_dim must be >= 1, got {max_bond_dim}")
        h = _lib.moonlab_mpdo_create(
            ctypes.c_uint32(num_qubits), ctypes.c_uint32(max_bond_dim))
        if not h:
            raise MemoryError("moonlab_mpdo_create returned NULL")
        self._handle = h

    def __del__(self) -> None:
        h = getattr(self, "_handle", None)
        if h:
            _lib.moonlab_mpdo_free(h)
            self._handle = None

    def clone(self) -> "Mpdo":
        """Return a deep copy of this MPDO."""
        out = Mpdo.__new__(Mpdo)
        h = _lib.moonlab_mpdo_clone(self._handle)
        if not h:
            raise MemoryError("moonlab_mpdo_clone returned NULL")
        out._handle = h
        return out

    @property
    def num_qubits(self) -> int:
        return int(_lib.moonlab_mpdo_num_qubits(self._handle))

    @property
    def max_bond_dim(self) -> int:
        return int(_lib.moonlab_mpdo_max_bond_dim(self._handle))

    @property
    def current_bond_dim(self) -> int:
        return int(_lib.moonlab_mpdo_current_bond_dim(self._handle))

    def trace(self) -> float:
        """Return ``Tr(rho)``.  Should equal 1 to roundoff for a
        completely-positive-trace-preserving evolution."""
        return float(_lib.moonlab_mpdo_trace(self._handle))

    # -- Named single-qubit channels --

    def apply_depolarizing(self, qubit: int, p: float) -> None:
        """Symmetric depolarising channel
        ``rho -> (1 - p) rho + (p / 3) (X rho X + Y rho Y + Z rho Z)``."""
        rc = _lib.moonlab_mpdo_apply_depolarizing_1q(
            self._handle, ctypes.c_uint32(qubit), ctypes.c_double(p))
        _check(rc, "moonlab_mpdo_apply_depolarizing_1q")

    def apply_amplitude_damping(self, qubit: int, gamma: float) -> None:
        """Amplitude damping (T_1 process) at strength
        ``gamma in [0, 1]``."""
        rc = _lib.moonlab_mpdo_apply_amplitude_damping_1q(
            self._handle, ctypes.c_uint32(qubit), ctypes.c_double(gamma))
        _check(rc, "moonlab_mpdo_apply_amplitude_damping_1q")

    def apply_phase_damping(self, qubit: int, lam: float) -> None:
        """Phase damping (T_2 process) at strength ``lam in [0, 1]``."""
        rc = _lib.moonlab_mpdo_apply_phase_damping_1q(
            self._handle, ctypes.c_uint32(qubit), ctypes.c_double(lam))
        _check(rc, "moonlab_mpdo_apply_phase_damping_1q")

    def apply_bit_flip(self, qubit: int, p: float) -> None:
        """Bit-flip channel ``rho -> (1 - p) rho + p X rho X``."""
        rc = _lib.moonlab_mpdo_apply_bit_flip_1q(
            self._handle, ctypes.c_uint32(qubit), ctypes.c_double(p))
        _check(rc, "moonlab_mpdo_apply_bit_flip_1q")

    def apply_phase_flip(self, qubit: int, p: float) -> None:
        """Phase-flip channel ``rho -> (1 - p) rho + p Z rho Z``."""
        rc = _lib.moonlab_mpdo_apply_phase_flip_1q(
            self._handle, ctypes.c_uint32(qubit), ctypes.c_double(p))
        _check(rc, "moonlab_mpdo_apply_phase_flip_1q")

    def apply_bit_phase_flip(self, qubit: int, p: float) -> None:
        """Bit + phase-flip channel ``rho -> (1 - p) rho + p Y rho Y``."""
        rc = _lib.moonlab_mpdo_apply_bit_phase_flip_1q(
            self._handle, ctypes.c_uint32(qubit), ctypes.c_double(p))
        _check(rc, "moonlab_mpdo_apply_bit_phase_flip_1q")

    # -- General Kraus channel --

    def apply_kraus(self, qubit: int,
                    kraus: Union[np.ndarray, Iterable[np.ndarray]]) -> None:
        """Apply an arbitrary single-qubit Kraus channel
        ``rho -> sum_a K_a rho K_a^dagger``.

        Args:
            qubit: Target qubit index.
            kraus: Either an iterable of 2x2 complex arrays, or a
                ``(num_kraus, 2, 2)`` complex array.  Caller is
                responsible for trace preservation
                ``sum_a K_a^dagger K_a = I``.
        """
        arr = np.asarray(kraus, dtype=np.complex128)
        if arr.ndim == 2 and arr.shape == (2, 2):
            arr = arr.reshape(1, 2, 2)
        if arr.ndim != 3 or arr.shape[1:] != (2, 2):
            raise ValueError(
                "kraus must be (n, 2, 2) complex array or list of 2x2 arrays;"
                f" got shape {arr.shape}")
        buf, n_kraus = _kraus_array_to_double_buffer(arr)
        # FFI takes mpdo_complex_t* (= double _Complex*), which has the
        # same layout as a flat double array of (real, imag) pairs.
        rc = _lib.moonlab_mpdo_apply_kraus_1q(
            self._handle, ctypes.c_uint32(qubit),
            ctypes.cast(buf, ctypes.POINTER(ctypes.c_double)),
            ctypes.c_uint32(n_kraus))
        _check(rc, "moonlab_mpdo_apply_kraus_1q")

    # -- Observables --

    def expect_pauli(self, qubit: int, pauli: Union[str, int]) -> float:
        """Single-qubit Pauli expectation ``Tr(rho * P_q)``.

        Args:
            qubit: Target qubit index.
            pauli: ``'I'``, ``'X'``, ``'Y'``, ``'Z'`` (case-insensitive)
                or an integer code in ``{0, 1, 2, 3}``.
        """
        if isinstance(pauli, str):
            try:
                code = _PAULI_CODE[pauli]
            except KeyError as exc:
                raise ValueError(
                    f"unknown Pauli '{pauli}'; expected one of I/X/Y/Z"
                ) from exc
        else:
            code = int(pauli)
            if not 0 <= code <= 3:
                raise ValueError(
                    f"Pauli code must be in [0, 3], got {code}")
        out = ctypes.c_double(0.0)
        rc = _lib.moonlab_mpdo_expect_pauli_1q(
            self._handle, ctypes.c_uint32(qubit), ctypes.c_uint8(code),
            ctypes.byref(out))
        _check(rc, "moonlab_mpdo_expect_pauli_1q")
        return float(out.value)


__all__ = ["Mpdo"]
