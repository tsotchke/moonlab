"""ctypes wiring for moonlab's Stim circuit / DEM interop surface.

Binds ``src/qec/stim_circuit.h`` and ``src/qec/stim_dem.h`` (plus the
``src/qec/uf_decoder.h`` entry points the DEM hands back) onto the
shared library ``moonlab.core`` already loaded.  Nothing here imports
stim, pymatching or sinter -- those are development/test dependencies
only, and the shipped package must run with numpy alone.

Everything in this module is private to ``moonlab.qec``; the public
names are re-exported from the subpackage.
"""

from __future__ import annotations

import ctypes
from enum import IntEnum

import numpy as np

from ..core import _lib

__all__ = [
    "StimStatus",
    "StimFormatError",
    "MoonlabStimError",
    "PFCircuitOp",
    "PF_OP_DTYPE",
    "PF_OP",
    "UF_BOUNDARY",
    "SIZE_T_DTYPE",
    "check_status",
    "take_text",
    "lib",
]

lib = _lib


# ---------------------------------------------------------------------------
# Status codes (src/qec/stim_circuit.h).
#
# The block sits at -70x, not the -60x originally drafted: -601..-606
# collide with src/applications/mwpm_exact.h.
# ---------------------------------------------------------------------------

class StimStatus(IntEnum):
    """Status codes returned by the Stim circuit and DEM readers."""

    OK = 0
    ERR_SYNTAX = -701       #: malformed token or structure
    ERR_UNSUPPORTED = -702  #: well formed but not implemented
    ERR_BAD_ARG = -703      #: bad argument from the caller
    ERR_OOM = -704          #: allocation failure
    ERR_IO = -705           #: file could not be read
    ERR_OVERFLOW = -706     #: caller buffer too small


MOONLAB_STIM_OK = int(StimStatus.OK)
MOONLAB_STIM_ERR_SYNTAX = int(StimStatus.ERR_SYNTAX)
MOONLAB_STIM_ERR_UNSUPPORTED = int(StimStatus.ERR_UNSUPPORTED)
MOONLAB_STIM_ERR_BAD_ARG = int(StimStatus.ERR_BAD_ARG)
MOONLAB_STIM_ERR_OOM = int(StimStatus.ERR_OOM)
MOONLAB_STIM_ERR_IO = int(StimStatus.ERR_IO)
MOONLAB_STIM_ERR_OVERFLOW = int(StimStatus.ERR_OVERFLOW)


class StimFormatError(ValueError):
    """A Stim circuit or DEM could not be read.

    Attributes:
        code: the raw :class:`StimStatus` value.
        line: 1-based source line, or 0 when the failure is not
            localised to one line.
        message: human-readable detail naming the offending token.
    """

    def __init__(self, code: int, line: int, message: str) -> None:
        try:
            name = StimStatus(code).name
        except ValueError:
            name = "ERR_UNKNOWN"
        if line:
            text = f"{name} at line {line}: {message}"
        else:
            text = f"{name}: {message}"
        super().__init__(text)
        self.code = int(code)
        self.line = int(line)
        self.message = str(message)


class MoonlabStimError(ctypes.Structure):
    """Mirror of ``moonlab_stim_error_t``."""

    _fields_ = [
        ("code", ctypes.c_int),
        ("line", ctypes.c_size_t),
        ("message", ctypes.c_char * 256),
    ]

    def as_exception(self, fallback: str) -> StimFormatError:
        msg = self.message.decode("utf-8", "replace") if self.message else fallback
        return StimFormatError(self.code, self.line, msg or fallback)


def check_status(rc: int, what: str) -> int:
    """Raise :class:`StimFormatError` when ``rc`` is one of our codes."""
    if rc < 0:
        try:
            name = StimStatus(rc).name
        except ValueError:
            name = "ERR_UNKNOWN"
        raise StimFormatError(rc, 0, f"{what} failed ({name}, rc={rc})")
    return rc


# ---------------------------------------------------------------------------
# Pauli-frame op list (src/backends/clifford/pauli_frame.h).
# ---------------------------------------------------------------------------

class PFCircuitOp(ctypes.Structure):
    """Mirror of ``pf_circuit_op_t``."""

    _fields_ = [
        ("kind", ctypes.c_uint8),
        ("q0", ctypes.c_uint32),
        ("q1", ctypes.c_uint32),
        ("p", ctypes.c_double),
    ]


#: numpy view of ``pf_circuit_op_t`` with the C struct's padding.
PF_OP_DTYPE = np.dtype({
    "names": ["kind", "q0", "q1", "p"],
    "formats": [np.uint8, np.uint32, np.uint32, np.float64],
    "offsets": [
        PFCircuitOp.kind.offset,
        PFCircuitOp.q0.offset,
        PFCircuitOp.q1.offset,
        PFCircuitOp.p.offset,
    ],
    "itemsize": ctypes.sizeof(PFCircuitOp),
})


class PF_OP(IntEnum):
    """``pf_op_kind_t``; values 0..18 are frozen ABI."""

    H = 0
    S = 1
    S_DAG = 2
    X = 3
    Y = 4
    Z = 5
    CNOT = 6
    CZ = 7
    SWAP = 8
    RESET = 9
    MEASURE = 10
    X_ERROR = 11
    Z_ERROR = 12
    Y_ERROR = 13
    DEPOLARIZE1 = 14
    DEPOLARIZE2 = 15
    MEASURE_NOISY = 16
    PAULI_CHANNEL_1 = 17
    PAULI_CHANNEL_2 = 18


#: ``MOONLAB_UF_BOUNDARY``: the virtual boundary node of the matching graph.
UF_BOUNDARY = 0xFFFFFFFF

#: numpy dtype matching ``size_t`` on this platform.
SIZE_T_DTYPE = np.dtype("u%d" % ctypes.sizeof(ctypes.c_size_t))


# ---------------------------------------------------------------------------
# Pointer shorthands.
# ---------------------------------------------------------------------------

_c_void_p = ctypes.c_void_p
_c_char_p = ctypes.c_char_p
_c_size_t = ctypes.c_size_t
_c_long = ctypes.c_long
_c_int = ctypes.c_int
_c_uint64 = ctypes.c_uint64
_p_err = ctypes.POINTER(MoonlabStimError)
_p_u8 = ctypes.POINTER(ctypes.c_uint8)
_p_u32 = ctypes.POINTER(ctypes.c_uint32)
_p_u64 = ctypes.POINTER(ctypes.c_uint64)
_p_f64 = ctypes.POINTER(ctypes.c_double)
_p_size = ctypes.POINTER(ctypes.c_size_t)
_p_op = ctypes.POINTER(PFCircuitOp)


def _wire(name, argtypes, restype):
    fn = getattr(lib, name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


# ---- stim_circuit.h -------------------------------------------------------

_wire("moonlab_stim_circuit_parse", [_c_char_p, _p_err], _c_void_p)
_wire("moonlab_stim_circuit_parse_file", [_c_char_p, _p_err], _c_void_p)
_wire("moonlab_stim_circuit_free", [_c_void_p], None)
_wire("moonlab_stim_circuit_to_text", [_c_void_p, _c_int], _c_void_p)
_wire("moonlab_stim_text_free", [_c_void_p], None)

for _n in ("num_qubits", "num_measurements", "num_detectors",
           "num_observables", "num_ticks"):
    _wire(f"moonlab_stim_circuit_{_n}", [_c_void_p], _c_size_t)

_wire("moonlab_stim_circuit_qubit_coords",
      [_c_void_p, _c_size_t, _p_f64, _c_size_t], _c_long)
_wire("moonlab_stim_circuit_detector_coords",
      [_c_void_p, _c_size_t, _p_f64, _c_size_t], _c_long)

_wire("moonlab_stim_circuit_num_ops", [_c_void_p], _c_long)
_wire("moonlab_stim_circuit_num_channel_args", [_c_void_p], _c_long)
_wire("moonlab_stim_circuit_lower",
      [_c_void_p, _p_op, _c_size_t, _p_f64, _c_size_t, _p_err], _c_long)

_wire("moonlab_stim_circuit_detector_csr",
      [_c_void_p, _p_size, _c_size_t, _p_u32, _c_size_t], _c_long)
_wire("moonlab_stim_circuit_observable_csr",
      [_c_void_p, _p_size, _c_size_t, _p_u32, _c_size_t], _c_long)
_wire("moonlab_stim_circuit_measurement_inversions",
      [_c_void_p, _p_u8, _c_size_t], _c_long)

_wire("moonlab_stim_circuit_sample_measurements",
      [_c_void_p, _c_size_t, _c_uint64, _c_int, _p_u8], _c_long)
_wire("moonlab_stim_circuit_sample_detectors",
      [_c_void_p, _c_size_t, _c_uint64, _c_int, _p_u8, _p_u8], _c_long)


# ---- stim_dem.h -----------------------------------------------------------

_wire("moonlab_dem_parse", [_c_char_p, _p_err], _c_void_p)
_wire("moonlab_dem_parse_file", [_c_char_p, _p_err], _c_void_p)
_wire("moonlab_dem_free", [_c_void_p], None)
_wire("moonlab_dem_to_text", [_c_void_p], _c_void_p)
_wire("moonlab_dem_text_from_edges",
      [_c_size_t, _c_size_t,
       _p_u32, _p_u32, _p_f64, _p_u64, _c_size_t,
       _p_u32, _p_u32, _p_f64, _c_size_t,
       _p_err], _c_void_p)

for _n in ("num_detectors", "num_observables", "num_edges",
           "num_correlations", "num_hyperedges", "num_detectorless"):
    _wire(f"moonlab_dem_{_n}", [_c_void_p], _c_size_t)

_wire("moonlab_dem_edges",
      [_c_void_p, _p_u32, _p_u32, _p_f64, _p_u64, _p_f64, _c_size_t], _c_long)
_wire("moonlab_dem_correlations",
      [_c_void_p, _p_u32, _p_u32, _p_f64, _c_size_t], _c_long)
_wire("moonlab_dem_detector_coords",
      [_c_void_p, _c_size_t, _p_f64, _c_size_t], _c_long)
_wire("moonlab_dem_make_uf_decoder", [_c_void_p, _c_int], _c_void_p)


# ---- uf_decoder.h ---------------------------------------------------------

_wire("moonlab_uf_decoder_free", [_c_void_p], None)
_wire("moonlab_uf_decode_batch",
      [_c_void_p, _p_u8, _c_size_t, _c_int, _p_u8], _c_long)
_wire("moonlab_uf_decoder_num_edges", [_c_void_p], _c_size_t)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def take_text(ptr) -> str:
    """Decode a malloc'd C string and release it with the moonlab allocator."""
    if not ptr:
        raise MemoryError("moonlab returned NULL text (allocation failure)")
    try:
        return ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
    finally:
        lib.moonlab_stim_text_free(ctypes.c_void_p(ptr))


def as_ptr(array, ctype):
    """View a numpy array's buffer as a C pointer, or NULL for None."""
    if array is None:
        return None
    return array.ctypes.data_as(ctypes.POINTER(ctype))
