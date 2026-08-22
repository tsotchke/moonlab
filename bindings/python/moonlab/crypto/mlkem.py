"""FIPS 203 ML-KEM-512 bindings.

Three operations: :func:`keygen`, :func:`encaps`, :func:`decaps`.
All inputs / outputs are plain ``bytes``.  Randomness may be
supplied explicitly (deterministic) or drawn from Moonlab's health-tested,
Bell-gated, SHAKE256-conditioned hybrid RNG via :func:`keygen_qrng` /
:func:`encaps_qrng`.

Example::

    from moonlab.crypto.mlkem import keygen_qrng, encaps_qrng, decaps

    ek, dk = keygen_qrng()                  # 800, 1632 bytes
    ct, shared_alice = encaps_qrng(ek)      # 768, 32 bytes
    shared_bob       = decaps(ct, dk)       # 32 bytes
    assert shared_alice == shared_bob
"""
from __future__ import annotations

import ctypes
import math
import os
from typing import Final, Optional, Tuple, TypedDict, Union

from ..core import _lib

PUBLICKEYBYTES    = 800
SECRETKEYBYTES    = 1632
CIPHERTEXTBYTES   = 768
SHAREDSECRETBYTES = 32

_UINT64_MAX: Final = (1 << 64) - 1
_MIN_CHSH: Final = -4.0
_MAX_CHSH: Final = 4.0

# ML-KEM-768 (NIST-recommended default)
MLKEM768_PUBLICKEYBYTES    = 1184
MLKEM768_SECRETKEYBYTES    = 2400
MLKEM768_CIPHERTEXTBYTES   = 1088

# ML-KEM-1024 (Category 5)
MLKEM1024_PUBLICKEYBYTES    = 1568
MLKEM1024_SECRETKEYBYTES    = 3168
MLKEM1024_CIPHERTEXTBYTES   = 1568

_Bytes = Union[bytes, bytearray, memoryview]

QRNG_CAP_HARDWARE_OS_ENTROPY       = 1 << 0
QRNG_CAP_CONTINUOUS_HEALTH_TESTS   = 1 << 1
QRNG_CAP_SHAKE256_CONDITIONED      = 1 << 2
QRNG_CAP_BELL_SIMULATION_GATED     = 1 << 3
QRNG_CAP_THREAD_SAFE               = 1 << 4
QRNG_CAP_BELL_EPOCH_CERTIFIED      = 1 << 5
QRNG_CAP_DEVICE_INDEPENDENT_SOURCE = 1 << 6
QRNG_CAP_FIPS140_VALIDATED         = 1 << 7


class _QrngStatus(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint64),
        ("conditioned_requests", ctypes.c_uint64),
        ("raw_bytes_generated", ctypes.c_uint64),
        ("bell_tests_performed", ctypes.c_uint64),
        ("bell_tests_passed", ctypes.c_uint64),
        ("average_chsh", ctypes.c_double),
        ("minimum_chsh", ctypes.c_double),
    ]


class _QrngStatusRecord(TypedDict):
    api_version: int
    capabilities: int
    hardware_os_entropy: bool
    continuous_health_tests: bool
    shake256_conditioned: bool
    bell_simulation_gated: bool
    thread_safe: bool
    bell_epoch_certified: bool
    device_independent_source: bool
    fips140_validated: bool
    conditioned_requests: int
    raw_bytes_generated: int
    bell_tests_performed: int
    bell_tests_passed: int
    average_chsh: float
    minimum_chsh: float


def _consume_qrng_status_record(record: _QrngStatusRecord) -> _QrngStatusRecord:
    # Enforce strict typing on the assembled public record and consume each key
    # explicitly at the public boundary before returning it to callers.
    api_version = int(record["api_version"])
    if api_version != 1:
        raise RuntimeError(
            f"Unsupported moonlab_qrng_status api_version ({api_version})"
        )

    if int(record["capabilities"]) > _UINT64_MAX:
        raise RuntimeError("moonlab_qrng_status capabilities out of uint64 bounds")

    conditioned_requests = int(record["conditioned_requests"])
    raw_bytes_generated = int(record["raw_bytes_generated"])
    bell_tests_performed = int(record["bell_tests_performed"])
    bell_tests_passed = int(record["bell_tests_passed"])

    if (
        conditioned_requests < 0
        or raw_bytes_generated < 0
        or bell_tests_performed < 0
        or bell_tests_passed < 0
    ):
        raise RuntimeError("moonlab_qrng_status counters cannot be negative")

    if raw_bytes_generated < conditioned_requests:
        raise RuntimeError(
            "moonlab_qrng_status counter ordering invalid: "
            "raw_bytes_generated < conditioned_requests"
        )
    if bell_tests_passed > bell_tests_performed:
        raise RuntimeError(
            "moonlab_qrng_status bell_tests_passed > bell_tests_performed"
        )

    if record["bell_epoch_certified"] and (
        bell_tests_performed == 0 or bell_tests_passed != bell_tests_performed
    ):
        raise RuntimeError(
            "moonlab_qrng_status bell_epoch_certified requires all Bell tests to pass"
        )

    average_chsh = float(record["average_chsh"])
    minimum_chsh = float(record["minimum_chsh"])
    if (
        not math.isfinite(average_chsh)
        or not math.isfinite(minimum_chsh)
        or average_chsh < _MIN_CHSH
        or average_chsh > _MAX_CHSH
        or minimum_chsh < _MIN_CHSH
        or minimum_chsh > _MAX_CHSH
    ):
        raise RuntimeError("moonlab_qrng_status CHSH values are invalid")
    if minimum_chsh > average_chsh:
        raise RuntimeError(
            "moonlab_qrng_status minimum_chsh cannot exceed average_chsh"
        )

    return {
        "api_version": api_version,
        "capabilities": int(record["capabilities"]),
        "hardware_os_entropy": bool(record["hardware_os_entropy"]),
        "continuous_health_tests": bool(record["continuous_health_tests"]),
        "shake256_conditioned": bool(record["shake256_conditioned"]),
        "bell_simulation_gated": bool(record["bell_simulation_gated"]),
        "thread_safe": bool(record["thread_safe"]),
        "bell_epoch_certified": bool(record["bell_epoch_certified"]),
        "device_independent_source": bool(record["device_independent_source"]),
        "fips140_validated": bool(record["fips140_validated"]),
        "conditioned_requests": conditioned_requests,
        "raw_bytes_generated": raw_bytes_generated,
        "bell_tests_performed": bell_tests_performed,
        "bell_tests_passed": bell_tests_passed,
        "average_chsh": average_chsh,
        "minimum_chsh": minimum_chsh,
    }


def _validate_qrng_status(status: _QrngStatus) -> _QrngStatusRecord:
    struct_size = int(status.struct_size)
    if struct_size != ctypes.sizeof(_QrngStatus):
        raise RuntimeError(
            f"moonlab_qrng_status struct_size mismatch ({struct_size})"
        )

    api_version = int(status.api_version)
    if api_version != 1:
        raise RuntimeError(f"Unsupported moonlab_qrng_status api_version ({api_version})")

    caps = int(status.capabilities)
    if caps < 0 or caps > _UINT64_MAX:
        raise RuntimeError("moonlab_qrng_status capabilities out of uint64 bounds")

    conditioned_requests = int(status.conditioned_requests)
    raw_bytes_generated = int(status.raw_bytes_generated)
    bell_tests_performed = int(status.bell_tests_performed)
    bell_tests_passed = int(status.bell_tests_passed)

    for name, value in (
        ("conditioned_requests", conditioned_requests),
        ("raw_bytes_generated", raw_bytes_generated),
        ("bell_tests_performed", bell_tests_performed),
        ("bell_tests_passed", bell_tests_passed),
    ):
        if value < 0 or value > _UINT64_MAX:
            raise RuntimeError(f"moonlab_qrng_status {name} out of uint64 bounds: {value}")

    if conditioned_requests < 0 or raw_bytes_generated < 0:
        raise RuntimeError("moonlab_qrng_status counters cannot be negative")
    if raw_bytes_generated < conditioned_requests:
        raise RuntimeError(
            "moonlab_qrng_status counter ordering invalid: "
            "raw_bytes_generated < conditioned_requests"
        )
    if bell_tests_passed > bell_tests_performed:
        raise RuntimeError(
            "moonlab_qrng_status bell_tests_passed > bell_tests_performed"
        )

    bell_epoch_certified = bool(caps & QRNG_CAP_BELL_EPOCH_CERTIFIED)
    if bell_epoch_certified and (
        bell_tests_performed <= 0 or bell_tests_passed != bell_tests_performed
    ):
        raise RuntimeError(
            "moonlab_qrng_status bell_epoch_certified requires all Bell tests to pass"
        )

    average_chsh = float(status.average_chsh)
    minimum_chsh = float(status.minimum_chsh)
    record: _QrngStatusRecord = {
        "api_version": api_version,
        "capabilities": caps,
        "hardware_os_entropy": bool(caps & QRNG_CAP_HARDWARE_OS_ENTROPY),
        "continuous_health_tests": bool(caps & QRNG_CAP_CONTINUOUS_HEALTH_TESTS),
        "shake256_conditioned": bool(caps & QRNG_CAP_SHAKE256_CONDITIONED),
        "bell_simulation_gated": bool(caps & QRNG_CAP_BELL_SIMULATION_GATED),
        "thread_safe": bool(caps & QRNG_CAP_THREAD_SAFE),
        "bell_epoch_certified": bell_epoch_certified,
        "device_independent_source": bool(caps & QRNG_CAP_DEVICE_INDEPENDENT_SOURCE),
        "fips140_validated": bool(caps & QRNG_CAP_FIPS140_VALIDATED),
        "conditioned_requests": conditioned_requests,
        "raw_bytes_generated": raw_bytes_generated,
        "bell_tests_performed": bell_tests_performed,
        "bell_tests_passed": bell_tests_passed,
        "average_chsh": average_chsh,
        "minimum_chsh": minimum_chsh,
    }
    return _consume_qrng_status_record(record)


def _configure():
    _lib.moonlab_qrng_get_status.argtypes = [ctypes.POINTER(_QrngStatus)]
    _lib.moonlab_qrng_get_status.restype = ctypes.c_int
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

    # 768 / 1024 wrappers use the same signatures.
    for bits in (768, 1024):
        fn = getattr(_lib, f"moonlab_mlkem{bits}_keygen")
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                       ctypes.c_char_p, ctypes.c_char_p]
        fn.restype = None
        fn = getattr(_lib, f"moonlab_mlkem{bits}_encaps")
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                       ctypes.c_char_p, ctypes.c_char_p]
        fn.restype = None
        fn = getattr(_lib, f"moonlab_mlkem{bits}_decaps")
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        fn.restype = None
        fn = getattr(_lib, f"moonlab_mlkem{bits}_keygen_qrng")
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        fn.restype = ctypes.c_int
        fn = getattr(_lib, f"moonlab_mlkem{bits}_encaps_qrng")
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        fn.restype = ctypes.c_int


_configure()


def qrng_status() -> dict:
    """Return live protection and Bell-epoch status for the hybrid RNG."""
    status = _QrngStatus()
    rc = _lib.moonlab_qrng_get_status(ctypes.byref(status))
    if rc != 0:
        raise RuntimeError(f"moonlab_qrng_get_status failed ({rc})")
    return _validate_qrng_status(status)


def keygen(d: Optional[_Bytes] = None,
           z: Optional[_Bytes] = None) -> Tuple[bytes, bytes]:
    """Generate an ML-KEM-512 key pair.

    If ``d`` / ``z`` are omitted, entropy is sourced from the OS CSPRNG
    (``os.urandom``). :func:`keygen_qrng` uses Moonlab's conditioned hybrid
    path instead.

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
    conditioned hybrid RNG (moonlab_qrng_bytes)."""
    ek = ctypes.create_string_buffer(PUBLICKEYBYTES)
    dk = ctypes.create_string_buffer(SECRETKEYBYTES)
    rc = _lib.moonlab_mlkem512_keygen_qrng(ek, dk)
    if rc != 0:
        raise RuntimeError(f"moonlab_mlkem512_keygen_qrng failed ({rc})")
    return bytes(ek.raw), bytes(dk.raw)


def encaps_qrng(ek: _Bytes) -> Tuple[bytes, bytes]:
    """Encapsulate using Moonlab's conditioned hybrid RNG for the inner
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


def _gen_wrappers(bits, pk_len, sk_len, ct_len):
    """Build the (keygen, encaps, decaps, keygen_qrng, encaps_qrng)
    quintuple for a particular ML-KEM parameter set."""
    kg  = getattr(_lib, f"moonlab_mlkem{bits}_keygen")
    en  = getattr(_lib, f"moonlab_mlkem{bits}_encaps")
    de  = getattr(_lib, f"moonlab_mlkem{bits}_decaps")
    kgq = getattr(_lib, f"moonlab_mlkem{bits}_keygen_qrng")
    enq = getattr(_lib, f"moonlab_mlkem{bits}_encaps_qrng")

    def _keygen(d=None, z=None):
        if d is None: d = os.urandom(32)
        if z is None: z = os.urandom(32)
        d = bytes(d); z = bytes(z)
        if len(d) != 32 or len(z) != 32:
            raise ValueError("d and z must each be 32 bytes")
        ek = ctypes.create_string_buffer(pk_len)
        dk = ctypes.create_string_buffer(sk_len)
        kg(ek, dk, d, z)
        return bytes(ek.raw), bytes(dk.raw)

    def _encaps(ek, m=None):
        ek = bytes(ek)
        if len(ek) != pk_len:
            raise ValueError(f"ek must be {pk_len} bytes")
        if m is None: m = os.urandom(32)
        m = bytes(m)
        if len(m) != 32: raise ValueError("m must be 32 bytes")
        c = ctypes.create_string_buffer(ct_len)
        K = ctypes.create_string_buffer(SHAREDSECRETBYTES)
        en(c, K, ek, m)
        return bytes(c.raw), bytes(K.raw)

    def _decaps(c, dk):
        c = bytes(c); dk = bytes(dk)
        if len(c) != ct_len or len(dk) != sk_len:
            raise ValueError("ciphertext / secret-key size mismatch")
        K = ctypes.create_string_buffer(SHAREDSECRETBYTES)
        de(K, c, dk)
        return bytes(K.raw)

    def _keygen_qrng():
        ek = ctypes.create_string_buffer(pk_len)
        dk = ctypes.create_string_buffer(sk_len)
        if kgq(ek, dk) != 0:
            raise RuntimeError(f"mlkem{bits}_keygen_qrng failed")
        return bytes(ek.raw), bytes(dk.raw)

    def _encaps_qrng(ek):
        ek = bytes(ek)
        if len(ek) != pk_len:
            raise ValueError(f"ek must be {pk_len} bytes")
        c = ctypes.create_string_buffer(ct_len)
        K = ctypes.create_string_buffer(SHAREDSECRETBYTES)
        if enq(c, K, ek) != 0:
            raise RuntimeError(f"mlkem{bits}_encaps_qrng failed")
        return bytes(c.raw), bytes(K.raw)

    return _keygen, _encaps, _decaps, _keygen_qrng, _encaps_qrng


(keygen768, encaps768, decaps768,
 keygen768_qrng, encaps768_qrng) = _gen_wrappers(
    768, MLKEM768_PUBLICKEYBYTES, MLKEM768_SECRETKEYBYTES,
    MLKEM768_CIPHERTEXTBYTES)
(keygen1024, encaps1024, decaps1024,
 keygen1024_qrng, encaps1024_qrng) = _gen_wrappers(
    1024, MLKEM1024_PUBLICKEYBYTES, MLKEM1024_SECRETKEYBYTES,
    MLKEM1024_CIPHERTEXTBYTES)


__all__ = [
    "qrng_status",
    "keygen", "encaps", "decaps",
    "keygen_qrng", "encaps_qrng",
    "PUBLICKEYBYTES", "SECRETKEYBYTES",
    "CIPHERTEXTBYTES", "SHAREDSECRETBYTES",
    "keygen768", "encaps768", "decaps768",
    "keygen768_qrng", "encaps768_qrng",
    "MLKEM768_PUBLICKEYBYTES", "MLKEM768_SECRETKEYBYTES",
    "MLKEM768_CIPHERTEXTBYTES",
    "keygen1024", "encaps1024", "decaps1024",
    "keygen1024_qrng", "encaps1024_qrng",
    "MLKEM1024_PUBLICKEYBYTES", "MLKEM1024_SECRETKEYBYTES",
    "MLKEM1024_CIPHERTEXTBYTES",
    "QRNG_CAP_HARDWARE_OS_ENTROPY", "QRNG_CAP_CONTINUOUS_HEALTH_TESTS",
    "QRNG_CAP_SHAKE256_CONDITIONED", "QRNG_CAP_BELL_SIMULATION_GATED",
    "QRNG_CAP_THREAD_SAFE", "QRNG_CAP_BELL_EPOCH_CERTIFIED",
    "QRNG_CAP_DEVICE_INDEPENDENT_SOURCE", "QRNG_CAP_FIPS140_VALIDATED",
]
