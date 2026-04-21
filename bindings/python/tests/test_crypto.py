"""Python-binding tests for moonlab.crypto."""
from __future__ import annotations

import os

from moonlab.crypto import sha3
from moonlab.crypto import mlkem


# --------------------------------------------------------------------
# SHA3 / SHAKE KAT vectors
# --------------------------------------------------------------------

def test_sha3_256_empty():
    assert sha3.sha3_256(b"").hex() == (
        "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
    )


def test_sha3_256_abc():
    assert sha3.sha3_256(b"abc").hex() == (
        "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"
    )


def test_sha3_512_empty():
    expected = (
        "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a6"
        "15b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26"
    )
    assert sha3.sha3_512(b"").hex() == expected


def test_shake128_empty_32():
    assert sha3.shake128(b"", outlen=32).hex() == (
        "7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26"
    )


def test_shake256_empty_32():
    assert sha3.shake256(b"", outlen=32).hex() == (
        "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f"
    )


def test_shake_length_variable():
    s = sha3.shake128(b"abc", 5)
    assert len(s) == 5
    s = sha3.shake256(b"abc", 100)
    assert len(s) == 100


# --------------------------------------------------------------------
# ML-KEM-512 end-to-end
# --------------------------------------------------------------------

def test_mlkem_sizes():
    assert mlkem.PUBLICKEYBYTES    == 800
    assert mlkem.SECRETKEYBYTES    == 1632
    assert mlkem.CIPHERTEXTBYTES   == 768
    assert mlkem.SHAREDSECRETBYTES == 32


def test_mlkem_deterministic_roundtrip():
    d = bytes(range(32))
    z = bytes((i * 3 + 1) & 0xFF for i in range(32))
    m = bytes((i * 5 + 2) & 0xFF for i in range(32))
    ek, dk = mlkem.keygen(d, z)
    assert len(ek) == mlkem.PUBLICKEYBYTES
    assert len(dk) == mlkem.SECRETKEYBYTES
    ct, K = mlkem.encaps(ek, m)
    assert len(ct) == mlkem.CIPHERTEXTBYTES
    K2 = mlkem.decaps(ct, dk)
    assert K == K2


def test_mlkem_random_roundtrip():
    for _ in range(5):
        ek, dk = mlkem.keygen()
        ct, K_a = mlkem.encaps(ek)
        K_b = mlkem.decaps(ct, dk)
        assert K_a == K_b


def test_mlkem_tampered_ciphertext():
    ek, dk = mlkem.keygen()
    ct, K = mlkem.encaps(ek)
    bad = bytearray(ct)
    bad[100] ^= 0xFF
    K_bad = mlkem.decaps(bytes(bad), dk)
    assert K_bad != K
    # Implicit rejection is deterministic
    K_bad2 = mlkem.decaps(bytes(bad), dk)
    assert K_bad == K_bad2


def test_mlkem_qrng_roundtrip():
    ek, dk = mlkem.keygen_qrng()
    ct, K_a = mlkem.encaps_qrng(ek)
    K_b = mlkem.decaps(ct, dk)
    assert K_a == K_b


def test_mlkem_keygen_qrng_nondeterministic():
    ek1, _ = mlkem.keygen_qrng()
    ek2, _ = mlkem.keygen_qrng()
    assert ek1 != ek2
