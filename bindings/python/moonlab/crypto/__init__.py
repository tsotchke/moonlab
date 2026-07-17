"""Cryptographic primitives exposed by Moonlab.

Sub-modules:
    moonlab.crypto.sha3   -- FIPS 202 SHA-3 hashes + SHAKE XOFs.
    moonlab.crypto.mlkem  -- FIPS 203 ML-KEM-512 KEM with optional
                             conditioned hybrid RNG entropy.

These are reference implementations (correctness-verified against
NIST test vectors), not FIPS-certified production crypto. The *_qrng
variants consume Moonlab's continuously health-tested, Bell-gated,
SHAKE256-conditioned release path. Deployments that require a validated
module boundary can use the explicit-seed variants with that module's
approved DRBG.
"""
from . import sha3 as sha3
from . import mlkem as mlkem

__all__ = ["sha3", "mlkem"]
