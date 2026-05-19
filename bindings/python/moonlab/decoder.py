"""Decoder-bench Python binding -- since v0.7.3.

Wraps `src/applications/decoder_bench.{c,h}` (v0.6.7 + v0.6.9 +
v0.7.2 wiring).  Five-slot dispatcher for the QEC decoder zoo:
GREEDY, MWPM_EXACT, SBNN, LIBIRREP_SS, PYMATCHING.

Slot availability is build-conditional: GREEDY and MWPM_EXACT are
always available, LIBIRREP_SS depends on `-DQSIM_ENABLE_LIBIRREP=ON`,
SBNN and PYMATCHING return `MOONLAB_DECODER_NOT_BUILT` until v0.7+.

@since v0.7.3
"""

from __future__ import annotations

import ctypes
from enum import IntEnum
from typing import List

from .core import _lib


class DecoderSlot(IntEnum):
    GREEDY      = 0
    MWPM_EXACT  = 1
    SBNN        = 2
    LIBIRREP_SS = 3
    PYMATCHING  = 4


MOONLAB_DECODER_OK = 0
MOONLAB_DECODER_NOT_BUILT = -401
MOONLAB_DECODER_BAD_ARG = -402
MOONLAB_DECODER_INFEASIBLE = -403
MOONLAB_DECODER_OOM = -404


class DecoderError(RuntimeError):
    pass


class DecoderNotBuiltError(DecoderError):
    pass


# ---- FFI signatures -----------------------------------------------

class _Code(ctypes.Structure):
    _fields_ = [
        ("distance", ctypes.c_int),
        ("num_qubits", ctypes.c_int),
        ("is_toric", ctypes.c_int),
    ]


class _Input(ctypes.Structure):
    _fields_ = [
        ("code",            ctypes.POINTER(_Code)),
        ("syndromes",       ctypes.POINTER(ctypes.c_ubyte)),
        ("corrections",     ctypes.POINTER(ctypes.c_ubyte)),
        ("num_stabilisers", ctypes.c_int),
        ("rng_seed",        ctypes.c_uint64),
    ]


_lib.moonlab_decoder_decode.argtypes = [
    ctypes.c_int, ctypes.POINTER(_Input)]
_lib.moonlab_decoder_decode.restype = ctypes.c_int

_lib.moonlab_decoder_slot_available.argtypes = [ctypes.c_int]
_lib.moonlab_decoder_slot_available.restype = ctypes.c_int

_lib.moonlab_decoder_slot_name.argtypes = [ctypes.c_int]
_lib.moonlab_decoder_slot_name.restype = ctypes.c_char_p


def slot_available(slot: DecoderSlot) -> bool:
    """Whether the slot has its external dependency linked."""
    return _lib.moonlab_decoder_slot_available(int(slot)) == 1


def slot_name(slot: DecoderSlot) -> str:
    return _lib.moonlab_decoder_slot_name(int(slot)).decode("utf-8")


def decode(slot: DecoderSlot,
           *,
           distance: int,
           num_qubits: int,
           is_toric: bool,
           syndromes: List[int],
           rng_seed: int = 0) -> List[int]:
    """Decode a syndrome with the requested slot.

    Returns the length-`num_qubits` correction byte vector
    (0 / 1 per data qubit).  Raises `DecoderNotBuiltError` if
    the slot requires an external library not linked in this build.
    """
    n_s = len(syndromes)
    synd_buf = (ctypes.c_ubyte * n_s)(*syndromes)
    corr_buf = (ctypes.c_ubyte * num_qubits)()

    code = _Code(distance=int(distance), num_qubits=int(num_qubits),
                 is_toric=1 if is_toric else 0)
    in_ = _Input(
        code=ctypes.pointer(code),
        syndromes=synd_buf,
        corrections=corr_buf,
        num_stabilisers=n_s,
        rng_seed=int(rng_seed),
    )
    rc = _lib.moonlab_decoder_decode(int(slot), ctypes.byref(in_))
    if rc == MOONLAB_DECODER_NOT_BUILT:
        raise DecoderNotBuiltError(
            f"decoder slot {slot.name} not built (rebuild with the "
            f"matching QSIM_ENABLE_* flag)")
    if rc != MOONLAB_DECODER_OK:
        raise DecoderError(f"decode({slot.name}): rc={rc}")
    return list(corr_buf)


__all__ = [
    "DecoderSlot",
    "DecoderError",
    "DecoderNotBuiltError",
    "decode",
    "slot_available",
    "slot_name",
]
