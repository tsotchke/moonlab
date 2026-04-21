"""FIPS 203 ML-KEM-512 bindings.

Three operations: :func:`keygen`, :func:`encaps`, :func:`decaps`.
All inputs / outputs are plain ``bytes``.  Randomness may be
supplied explicitly (deterministic) or drawn from Moonlab's
Bell-verified quantum RNG via :func:`keygen_qrng` / :func:`encaps_qrng`.

Example::

    from moonlab.crypto.mlkem import keygen_qrng, encaps_qrng, decaps

    ek, dk = keygen_qrng()                  # 800, 1632 bytes
    ct, shared_alice = encaps_qrng(ek)      # 768, 32 bytes
    shared_bob       = decaps(ct, dk)       # 32 bytes
    assert shared_alice == shared_bob
"""
from __future__ import annotations

import ctypes
import os
from typing import Optional, Tuple, Union

from ..core import _lib

PUBLICKEYBYTES    = 800
SECRETKEYBYTES    = 1632
CIPHERTEXTBYTES   = 768
SHAREDSECRETBYTES = 32

_Bytes = Union[bytes, bytearray, memoryview]


def _configure():
    _lib.moonlab_mlkem512_keygen.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
    ]
    _lib.moonlab_mlkem512_keygen.restype = None
    _lib.moonlab_mlkem512_encaps.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
    ]
    _lib.moonlab_mlkem512_encaps.restype = None
    _lib.moonlab_mlkem512_decaps.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
    ]
    _lib.moonlab_mlkem512_decaps.restype = None
    _lib.moonlab_mlkem512_keygen_qrng.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    _lib.moonlab_mlkem512_keygen_qrng.restype = ctypes.c_int
    _lib.moonlab_mlkem512_encaps_qrng.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                                    ctypes.c_char_p]
    _lib.moonlab_mlkem512_encaps_qrng.restype = ctypes.c_int


_configure()


def keygen(d: Optional[_Bytes] = None,
           z: Optional[_Bytes] = None) -> Tuple[bytes, bytes]:
    """Generate an ML-KEM-512 key pair.

    If ``d`` / ``z`` are omitted, entropy is sourced from the OS CSPRNG
    (``os.urandom``).  For a Bell-verified quantum seed use
    :func:`keygen_qrng` instead.

    Returns ``(ek, dk)`` -- 800 and 1632 bytes respectively.
    """
    if d is None: d = os.urandom(32)
    if z is None: z = os.urandom(32)
    d = bytes(d); z = bytes(z)
    if len(d) != 32 or len(z) != 32:
        raise ValueError("d and z must each be exactly 32 bytes")
    ek = ctypes.create_string_buffer(PUBLICKEYBYTES)
    dk = ctypes.create_string_buffer(SECRETKEYBYTES)
    _lib.moonlab_mlkem512_keygen(ek, dk, d, z)
    return bytes(ek.raw), bytes(dk.raw)


def encaps(ek: _Bytes, m: Optional[_Bytes] = None) -> Tuple[bytes, bytes]:
    """Encapsulate a shared secret against an ML-KEM-512 public key.

    If ``m`` is omitted, a fresh 32-byte secret is drawn from
    ``os.urandom``.  Returns ``(ciphertext, shared_secret)``.
    """
    ek = bytes(ek)
    if len(ek) != PUBLICKEYBYTES:
        raise ValueError(f"ek must be {PUBLICKEYBYTES} bytes, got {len(ek)}")
    if m is None: m = os.urandom(32)
    m = bytes(m)
    if len(m) != 32:
        raise ValueError("m must be exactly 32 bytes")
    c = ctypes.create_string_buffer(CIPHERTEXTBYTES)
    K = ctypes.create_string_buffer(SHAREDSECRETBYTES)
    _lib.moonlab_mlkem512_encaps(c, K, ek, m)
    return bytes(c.raw), bytes(K.raw)


def decaps(c: _Bytes, dk: _Bytes) -> bytes:
    """Decapsulate an ML-KEM-512 ciphertext.  Returns the 32-byte
    shared secret (pseudorandom on tampered input)."""
    c = bytes(c); dk = bytes(dk)
    if len(c) != CIPHERTEXTBYTES:
        raise ValueError(f"c must be {CIPHERTEXTBYTES} bytes")
    if len(dk) != SECRETKEYBYTES:
        raise ValueError(f"dk must be {SECRETKEYBYTES} bytes")
    K = ctypes.create_string_buffer(SHAREDSECRETBYTES)
    _lib.moonlab_mlkem512_decaps(K, c, dk)
    return bytes(K.raw)


def keygen_qrng() -> Tuple[bytes, bytes]:
    """Generate a key pair with entropy drawn from Moonlab's
    Bell-verified quantum RNG (moonlab_qrng_bytes)."""
    ek = ctypes.create_string_buffer(PUBLICKEYBYTES)
    dk = ctypes.create_string_buffer(SECRETKEYBYTES)
    rc = _lib.moonlab_mlkem512_keygen_qrng(ek, dk)
    if rc != 0:
        raise RuntimeError(f"moonlab_mlkem512_keygen_qrng failed ({rc})")
    return bytes(ek.raw), bytes(dk.raw)


def encaps_qrng(ek: _Bytes) -> Tuple[bytes, bytes]:
    """Encapsulate using Bell-verified quantum entropy for the inner
    message seed."""
    ek = bytes(ek)
    if len(ek) != PUBLICKEYBYTES:
        raise ValueError(f"ek must be {PUBLICKEYBYTES} bytes")
    c = ctypes.create_string_buffer(CIPHERTEXTBYTES)
    K = ctypes.create_string_buffer(SHAREDSECRETBYTES)
    rc = _lib.moonlab_mlkem512_encaps_qrng(c, K, ek)
    if rc != 0:
        raise RuntimeError(f"moonlab_mlkem512_encaps_qrng failed ({rc})")
    return bytes(c.raw), bytes(K.raw)


__all__ = [
    "keygen", "encaps", "decaps",
    "keygen_qrng", "encaps_qrng",
    "PUBLICKEYBYTES", "SECRETKEYBYTES",
    "CIPHERTEXTBYTES", "SHAREDSECRETBYTES",
]
