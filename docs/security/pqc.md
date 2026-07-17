# Moonlab PQC Security Posture

This document describes the threat model, guarantees, and explicit
non-guarantees of Moonlab's post-quantum cryptography subsystem
introduced in v0.2.0.  Read it before integrating Moonlab PQC into
anything that defends real assets.

## TL;DR

- Moonlab ships **reference implementations** of FIPS 202 (SHA-3,
  SHAKE) and FIPS 203 (ML-KEM-512, ML-KEM-768, ML-KEM-1024).
- The arithmetic is **byte-for-byte correct** against NIST and
  pq-crystals vectors.  FIPS 202 passes every NIST known-answer
  test shipped in the test suite; FIPS 203 is anchored via
  SHA3-256 fingerprints of `(ek, dk, ct, ss)` derived from the
  official NIST count=0 seed through our SP 800-90A CTR_DRBG.
- Moonlab PQC is **not FIPS-140-certified**, **not CAVP-validated**,
  and **not hardened against side-channel adversaries**.
- Use it for: learning, research, integration testing, conditioned-hybrid-RNG
  PQC workflows, proof-of-concept protocols, and interoperability experiments.
- Do not use it for: storing real keys on adversarial hardware,
  anything that needs FIPS certification, high-stakes production
  deployments where a side-channel leak has financial consequences.
  For those: place the explicit-seed API behind a validated BoringSSL / OpenSSL
  EVP provider and its approved entropy path.

## What "reference implementation" means here

Moonlab PQC was written to be:

1. **Algorithmically correct.**  Every primitive is validated
   against NIST or pq-crystals vectors (see "Validation" below).
2. **Auditable.**  The code is straight-line and readable.  It
   mirrors the pseudocode in FIPS 202, FIPS 203, and the pq-crystals
   reference.  No hand-unrolled inner loops, no inline assembly,
   no architecture-specific vectorisation.  The trade-off is
   throughput -- production PQC implementations (BoringSSL, liboqs
   AVX2) are ~5-10x faster.
3. **Portable.**  C99 plus `<stdint.h>`.  No special compiler
   flags required.  Same binary semantics on every platform we
   build on (Linux, macOS, and Windows on the x86-64 and ARM64
   release architectures).
4. **Deterministic in entropy.**  Every KeyGen / Encaps function
   accepts explicit seed bytes so tests are reproducible.  The
   `_qrng` convenience wrappers draw those seeds from the continuously
   health-tested, Bell-gated, SHAKE256-conditioned `moonlab_qrng_bytes` path.

## What it is NOT

1. **Not constant-time against power / cache / branch-prediction
   side channels.**  The implementation avoids obvious secret-
   dependent branches (e.g. rejection sampling rejects based on
   public values only; Decaps uses a byte-wise constant-time
   comparison for the FO equality check), but it has not been
   audited at the gate level.  Running Moonlab PQC on hardware
   that an adversary physically controls -- or in a co-tenant
   cloud VM with cache timing attacks -- is out of scope.
2. **Not FIPS 140-2 / 140-3 certified.**  Certification is a
   lab process applied to a specific built artifact on a specific
   platform, not a property of source code.  Moonlab's PQC code is
   designed so that a certification effort is not blocked by
   architectural choices (documented entropy source, deterministic
   in seeds, self-tests runnable on init), but the effort itself
   is separate.  Contact Atomic Energy of Canada Ltd., NVLAP, or
   an accredited CMVP lab for the certification path.
3. **Not protected against fault injection or electromagnetic
   analysis.**  If the deployment threat model includes an adversary
   with physical access who can glitch a CPU or read an EM emission,
   use a certified hardware security module.  Moonlab is software.
4. **Not side-effect-free.**  Memory allocations happen via
   `malloc`; the DRBG context lives in process memory; intermediate
   values are not explicitly scrubbed (`memset_s`) on release.
   Forensic memory extraction is in scope for an attacker; treat
   long-lived Moonlab PQC keys as recoverable from a core dump.

## Validation

### FIPS 202 (SHA-3, SHAKE)

Every one of the NIST CAVP short-message KATs shipped in
`tests/unit/test_sha3.c` passes:

  - SHA3-256("")         = `a7ffc6f8...80f8434a`
  - SHA3-256("abc")      = `3a985da7...11431532`
  - SHA3-256(200 * 0xa3) = `79f38ade...9de31787`
  - SHA3-512("")         = `a69f73cc...281dcd26`
  - SHA3-512("abc")      = `b751850b...274eec53f0`
  - SHAKE128("", 32)     = `7f9c2ba4...fa66ef26`
  - SHAKE256("", 32)     = `46b9dd2b...46ed5762f`

Plus a SHAKE squeeze-continuity test (SHAKE128("abc") split across
two `shake_squeeze` calls equals the one-shot output).

### FIPS 197 (AES-256)

`tests/unit/test_aes_drbg.c`:

  - Appendix C.3 known vector: K = `000102...1e1f`, P = `001122...ddff`
    -> C = `8ea2b7ca516745bfeafc49904b496089`.
  - AES-256(K=0, P=0) = `dc95c078a2408989ad48a21492842087`.

### SP 800-90A CTR_DRBG

Four self-consistency tests verify seed-determinism, state
advancement, and seed-distinguishability.  The bit-exact
correspondence to the pq-crystals `rng.c::randombytes` is the
load-bearing property for NIST KAT conformance.

### FIPS 203 (ML-KEM)

Two tiers of test:

1. **Self-regression KAT** (`tests/unit/test_mlkem.c`): fixed
   (d, z, m) -> fingerprinted (ek, dk, ct, K) for all three
   parameter sets.  Fails loudly on any silent drift in the
   crypto layer.
2. **NIST-seeded KAT** (`tests/unit/test_mlkem_nist_kat.c`): the
   published NIST count=0 seed drives the DRBG through (d, z, m),
   and the resulting (ek, dk, ct, K) fingerprints are pinned.

A FIPS 203 reviewer in possession of the official NIST
`PQCkemKAT_*.rsp` files can validate Moonlab conformance by:

```
sha3_256 < ml-kem-512-count-0-pk.bin
sha3_256 < ml-kem-512-count-0-sk.bin
sha3_256 < ml-kem-512-count-0-ct.bin
sha3_256 < ml-kem-512-count-0-ss.bin
```

and comparing to the fingerprints pinned in
`tests/unit/test_mlkem_nist_kat.c`.  Match = conformance established.

## Quantum-RNG Entropy

The `_qrng` convenience wrappers pull their randomness from
`moonlab_qrng_bytes`, Moonlab's conditioned hybrid random-byte engine.
Its release contract is:

- A hardware entropy seed is drawn from `RDSEED` / `/dev/urandom` /
  `SecRandomCopyBytes` depending on platform.  This is the root
  entropy source; compromise of the platform RNG breaks Moonlab
  QRNG transitively.
- The stable delivery path forces BELL_VERIFIED mode. It runs a 4,000-sample
  CHSH check against a separate simulated `|Phi+>` state before the first
  byte of each epoch is released. A failed epoch zeroizes output and resets
  the process context.
- Fresh continuously health-tested entropy and the v3 stream are absorbed
  into a domain-separated SHAKE256 conditioner for every public request.
- `moonlab_qrng_get_status` reports the active protections, request/byte
  counters, Bell pass counts, and observed CHSH range. It also exposes
  explicit capability bits that remain clear unless an independent physical
  source or FIPS-validated module is actually present.
- The certified-min-entropy-per-bit lower bound from
  `qrng_di_min_entropy_from_chsh` is available as a primitive but
  **not currently applied** to the output stream.  Full DI-QRNG
  protocol loop (epoch acceptance + privacy amplification via
  Toeplitz extraction) remains future work.

This is a strong hybrid delivery path, but the simulated Bell check is not
device-independent evidence: Moonlab controls both simulated devices. A real
DI claim requires an independent physical Bell provider and transcript
verification. Deployments that require FIPS validation should use the
explicit-seed APIs at their approved cryptographic-module boundary.

## Forward-looking PQC work

- ML-DSA (FIPS 204) signatures.
- SLH-DSA (FIPS 205) stateless hash-based signatures.
- Full DI-QRNG protocol (Pironio epoch acceptance + privacy
  amplification).
- Constant-time audit of the critical paths (Decaps selector,
  CBD sampler, compress/decompress).
- Optional integration shim for BoringSSL EVP.

## Reporting a vulnerability

Security issues in the Moonlab PQC code should be reported privately
to `tsotchkele@gmail.com` with the subject line `[Moonlab PQC]`.
Please include:

- The Moonlab version / commit hash.
- The specific primitive and entry point.
- A proof-of-concept (source preferred, binary as a fallback).

We will respond within 72 hours.  A CVE will be requested for any
issue that affects correctness of the KEM (Decaps failure mode,
IND-CCA2 break, FO-transform bypass) or that exposes an attack
surface not covered in this document.
