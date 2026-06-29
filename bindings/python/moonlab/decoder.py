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
    """Raised when the native decoder dispatcher returns an error."""


class DecoderNotBuiltError(DecoderError):
    """Raised when a decoder slot requires an optional backend not linked."""


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

# Runtime decoder registry (since v1.0.3).
_HAS_DECODER_REGISTRY = hasattr(_lib, "moonlab_register_decoder")

# decoder_fn signature: (const moonlab_decoder_input_t*, void *ctx) -> int.
_DecoderCFn = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(_Input), ctypes.c_void_p)


class _DecoderEntry(ctypes.Structure):
    _fields_ = [
        ("name",        ctypes.c_char_p),
        ("fn",          ctypes.c_void_p),    # opaque, we only read name + desc
        ("ctx",         ctypes.c_void_p),
        ("description", ctypes.c_char_p),
    ]


if _HAS_DECODER_REGISTRY:
    _lib.moonlab_register_decoder.argtypes = [
        ctypes.c_char_p, _DecoderCFn, ctypes.c_void_p, ctypes.c_char_p]
    _lib.moonlab_register_decoder.restype = ctypes.c_int

    _lib.moonlab_unregister_decoder.argtypes = [ctypes.c_char_p]
    _lib.moonlab_unregister_decoder.restype = ctypes.c_int

    _lib.moonlab_lookup_decoder.argtypes = [ctypes.c_char_p]
    _lib.moonlab_lookup_decoder.restype = ctypes.POINTER(_DecoderEntry)

    _lib.moonlab_decoder_decode_by_name.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(_Input)]
    _lib.moonlab_decoder_decode_by_name.restype = ctypes.c_int

    _lib.moonlab_num_decoders.argtypes = []
    _lib.moonlab_num_decoders.restype = ctypes.c_int

    _lib.moonlab_list_decoders.argtypes = [
        ctypes.POINTER(ctypes.c_char_p), ctypes.c_int]
    _lib.moonlab_list_decoders.restype = ctypes.c_int


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


# ---- Runtime decoder registry ------------------------------------

# Keep registered trampolines alive while installed in the registry.
_active_decoders: dict = {}


def list_decoders() -> List[str]:
    """Names of all currently-registered decoders (built-in + custom)."""
    if not _HAS_DECODER_REGISTRY:
        return []
    n = int(_lib.moonlab_num_decoders())
    if n <= 0:
        return []
    arr = (ctypes.c_char_p * n)()
    written = int(_lib.moonlab_list_decoders(arr, n))
    return [arr[i].decode("utf-8") for i in range(written)]


def lookup_decoder(name: str) -> dict | None:
    """Return ``{name, description}`` for a registered decoder, or None."""
    if not _HAS_DECODER_REGISTRY:
        return None
    p = _lib.moonlab_lookup_decoder(name.encode("utf-8"))
    if not p:
        return None
    raw = p.contents
    return {
        "name": raw.name.decode("utf-8") if raw.name else "",
        "description": raw.description.decode("utf-8") if raw.description else "",
    }


def register_decoder(name: str,
                     decode_fn,
                     description: str = "") -> None:
    """Register a Python decoder under ``name``.

    ``decode_fn`` is called as ``decode_fn(distance, num_qubits,
    is_toric, syndromes) -> corrections`` where ``corrections`` is a
    ``list[int]`` of length ``num_qubits``.

    The registered name overrides any built-in with the same name; the
    enum dispatcher routes through the registry too, so re-registering
    ``"greedy"`` swaps in the Python implementation for both
    :func:`decode` and :func:`decode_by_name`."""
    if not _HAS_DECODER_REGISTRY:
        raise DecoderError(
            "register_decoder requires libquantumsim with the v1.0.3+ "
            "decoder runtime registry compiled in")

    def trampoline(in_ptr, ctx):
        try:
            raw = in_ptr.contents
            code = raw.code.contents
            n_s = raw.num_stabilisers
            n_q = code.num_qubits
            syndromes = [raw.syndromes[i] for i in range(n_s)]
            corrections = decode_fn(
                code.distance, n_q, bool(code.is_toric), syndromes)
            if not isinstance(corrections, (list, tuple)):
                return MOONLAB_DECODER_OOM
            if len(corrections) < n_q:
                return MOONLAB_DECODER_OOM
            for q in range(n_q):
                raw.corrections[q] = int(corrections[q]) & 0xff
            return MOONLAB_DECODER_OK
        except Exception:
            import traceback
            traceback.print_exc()
            return MOONLAB_DECODER_OOM

    cfn = _DecoderCFn(trampoline)
    rc = _lib.moonlab_register_decoder(
        name.encode("utf-8"), cfn, None,
        (description or "").encode("utf-8") if description else None)
    if rc != MOONLAB_DECODER_OK:
        raise DecoderError(f"register_decoder({name!r}): rc={rc}")
    # Keep the cfn + python callback alive until unregistered.
    _active_decoders[name] = (cfn, decode_fn)


def unregister_decoder(name: str) -> None:
    """Remove a decoder from the registry."""
    if not _HAS_DECODER_REGISTRY:
        return
    rc = _lib.moonlab_unregister_decoder(name.encode("utf-8"))
    _active_decoders.pop(name, None)
    if rc != MOONLAB_DECODER_OK:
        raise DecoderError(f"unregister_decoder({name!r}): rc={rc}")


def decode_by_name(name: str,
                   *,
                   distance: int,
                   num_qubits: int,
                   is_toric: bool,
                   syndromes: List[int],
                   rng_seed: int = 0) -> List[int]:
    """Decode via a name-keyed registry lookup.  Same semantics as
    :func:`decode` but accepts any registered name."""
    if not _HAS_DECODER_REGISTRY:
        raise DecoderError("decode_by_name requires libquantumsim >= v1.0.3")
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
    rc = _lib.moonlab_decoder_decode_by_name(
        name.encode("utf-8"), ctypes.byref(in_))
    if rc == MOONLAB_DECODER_NOT_BUILT:
        raise DecoderNotBuiltError(
            f"decoder {name!r} not built / not registered")
    if rc != MOONLAB_DECODER_OK:
        raise DecoderError(f"decode_by_name({name!r}): rc={rc}")
    return list(corr_buf)


__all__ = [
    "DecoderSlot",
    "DecoderError",
    "DecoderNotBuiltError",
    "decode",
    "decode_by_name",
    "slot_available",
    "slot_name",
    "register_decoder",
    "unregister_decoder",
    "lookup_decoder",
    "list_decoders",
]
