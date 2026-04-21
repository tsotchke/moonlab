"""FIPS 202 SHA3 and SHAKE bindings.

Fixed-output hashes and the two SHAKE extendable-output functions.
All inputs accept ``bytes`` or ``bytearray``; outputs are ``bytes``.

Example::

    from moonlab.crypto.sha3 import sha3_256, shake256
    digest = sha3_256(b"abc")                 # 32 bytes
    bits   = shake256(b"", outlen=64)         # 64 bytes of uniform stream
"""
from __future__ import annotations

import ctypes
from typing import Union

from ..core import _lib

_Bytes = Union[bytes, bytearray, memoryview]


def _configure():
    for name, outlen in (("sha3_224", 28), ("sha3_256", 32),
                         ("sha3_384", 48), ("sha3_512", 64)):
        fn = getattr(_lib, name)
        fn.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
        fn.restype = None
        fn._outlen = outlen   # type: ignore[attr-defined]
    _lib.shake128.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                              ctypes.c_char_p, ctypes.c_size_t]
    _lib.shake128.restype = None
    _lib.shake256.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                              ctypes.c_char_p, ctypes.c_size_t]
    _lib.shake256.restype = None


_configure()


def _hash(fn, data: _Bytes) -> bytes:
    data = bytes(data)
    out = ctypes.create_string_buffer(fn._outlen)
    fn(data, len(data), out)
    return bytes(out.raw[:fn._outlen])


def sha3_224(data: _Bytes) -> bytes:
    """FIPS 202 SHA3-224; returns 28 bytes."""
    return _hash(_lib.sha3_224, data)


def sha3_256(data: _Bytes) -> bytes:
    """FIPS 202 SHA3-256; returns 32 bytes."""
    return _hash(_lib.sha3_256, data)


def sha3_384(data: _Bytes) -> bytes:
    """FIPS 202 SHA3-384; returns 48 bytes."""
    return _hash(_lib.sha3_384, data)


def sha3_512(data: _Bytes) -> bytes:
    """FIPS 202 SHA3-512; returns 64 bytes."""
    return _hash(_lib.sha3_512, data)


def shake128(data: _Bytes, outlen: int) -> bytes:
    """FIPS 202 SHAKE128 with ``outlen`` output bytes."""
    if outlen < 0:
        raise ValueError("outlen must be non-negative")
    data = bytes(data)
    out = ctypes.create_string_buffer(outlen) if outlen > 0 else None
    _lib.shake128(data, len(data), out if out is not None else b"", outlen)
    return bytes(out.raw[:outlen]) if out is not None else b""


def shake256(data: _Bytes, outlen: int) -> bytes:
    """FIPS 202 SHAKE256 with ``outlen`` output bytes."""
    if outlen < 0:
        raise ValueError("outlen must be non-negative")
    data = bytes(data)
    out = ctypes.create_string_buffer(outlen) if outlen > 0 else None
    _lib.shake256(data, len(data), out if out is not None else b"", outlen)
    return bytes(out.raw[:outlen]) if out is not None else b""


__all__ = [
    "sha3_224", "sha3_256", "sha3_384", "sha3_512",
    "shake128", "shake256",
]
