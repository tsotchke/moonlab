"""Cryptographic primitives exposed by Moonlab.

Sub-modules:
    moonlab.crypto.sha3   -- FIPS 202 SHA-3 hashes + SHAKE XOFs.
    moonlab.crypto.mlkem  -- FIPS 203 ML-KEM-512 KEM with optional
                             Bell-verified QRNG-sourced entropy.

These are reference implementations (correctness-verified against
NIST test vectors), not FIPS-certified production crypto.  Consume
Moonlab's quantum-entropy + PQC pipeline end-to-end via the *_qrng
variants of the ML-KEM functions -- those pull their randomness
from the same Bell-verified source that ``moonlab_qrng_bytes``
feeds.
"""
from . import sha3 as sha3
from . import mlkem as mlkem

__all__ = ["sha3", "mlkem"]
