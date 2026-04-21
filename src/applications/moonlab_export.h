/**
 * @file moonlab_export.h
 * @brief Stable C ABI surface for Moonlab (libquantumsim) consumers.
 *
 * This header defines the committed, versioned, downstream-facing ABI for
 * Moonlab. Downstream libraries (notably QGTL,
 * github.com:tsotchke/quantum_geometric_tensor) pin against the symbols
 * declared here and locate them at runtime via dlsym.
 *
 * ABI CONTRACT
 * ------------
 *  - Every symbol declared in this header is guaranteed to exist, with the
 *    same name and signature, across all 0.x releases. Semantic upgrades to
 *    a function's behavior will be delivered via a new function (with a
 *    version suffix such as `_v2`) rather than by breaking the existing one.
 *  - Removal of any symbol declared here requires a major-version bump
 *    (1.0 -> 2.0).
 *  - The header itself is allowed to grow: new symbols may be added, but
 *    existing declarations are locked.
 *  - Consumers should use `moonlab_abi_version()` at runtime to feature-gate
 *    newer capabilities.
 *
 * STABILITY
 * ---------
 *  - `moonlab_qrng_bytes` : stable since 0.1.2.
 *  - `moonlab_abi_version` : stable since 0.1.2.
 *
 * All future public APIs (PQC, quantum-geometry computation, Chern markers,
 * etc.) will be added here as they land in subsequent 0.x releases.
 *
 * @since 0.1.2
 * @copyright 2024-2026 tsotchke. Licensed under the Apache License, 2.0.
 */

#ifndef MOONLAB_EXPORT_H
#define MOONLAB_EXPORT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ABI version constants. The (major, minor, patch) triple is purely for
 * feature-discovery by downstream consumers; it is NOT linked to the
 * package version. Consumers should check (major, minor) and refuse to
 * bind if they require a newer minor than the installed library. */
#define MOONLAB_ABI_VERSION_MAJOR 0
#define MOONLAB_ABI_VERSION_MINOR 2
#define MOONLAB_ABI_VERSION_PATCH 0

/**
 * @brief Query the ABI version at runtime.
 *
 * Consumers dlsym this and check the major/minor values to decide whether
 * optional capabilities (added in later 0.x releases) are available.
 *
 * @param[out] major Major version component. May be NULL.
 * @param[out] minor Minor version component. May be NULL.
 * @param[out] patch Patch version component. May be NULL.
 *
 * @since 0.1.2
 */
void moonlab_abi_version(int* major, int* minor, int* patch);

/**
 * @brief Fill a buffer with cryptographically-strong quantum random bytes.
 *
 * Produces bytes from Moonlab's v3 QRNG engine, which combines a hardware
 * entropy pool (RDSEED / /dev/urandom / SecRandomCopyBytes) with a
 * Bell-verified quantum simulation layer. The function is thread-safe:
 * concurrent calls from multiple threads are serialised internally.
 *
 * The v3 context is lazily initialised on first call and released at
 * process exit via atexit; callers do not need to perform any setup.
 *
 * @param buf  Output buffer. Must be non-NULL whenever @p size > 0.
 * @param size Number of bytes to write. May be 0 (no-op, returns success).
 *
 * @return  0 on success.
 * @return -1 if @p buf is NULL and @p size > 0.
 * @return -2 if the v3 QRNG engine failed to initialise.
 * @return -3 if a subsequent byte-generation call failed (e.g. an
 *            internal Bell-verification epoch was rejected).
 *
 * @since 0.1.2
 */
int moonlab_qrng_bytes(uint8_t* buf, size_t size);

/* ---- Quantum geometric tensor (stable from 0.2.0) -------------------- */

/**
 * @brief Compute the integer Chern number of the lower band of the
 *        Qi-Wu-Zhang 2-band Chern insulator
 *
 *   H(k) = sin(kx) sigma_x + sin(ky) sigma_y
 *        + (m + cos(kx) + cos(ky)) sigma_z
 *
 * via the Fukui-Hatsugai-Suzuki link-variable method on a @p N x @p N
 * Brillouin-zone grid. The return value is the integer Chern number
 * (no gauge ambiguity) as rounded to nearest integer.
 *
 * This is the simplest primitive on the QGT ABI surface. Higher-level
 * functions (per-plaquette Berry curvature arrays, Fubini-Study
 * metric, custom models) are exposed on an opaque-handle interface via
 * the internal `src/algorithms/quantum_geometry/qgt.h` header; when
 * they stabilise they will be promoted here.
 *
 * Intended consumers: QGTL (github.com:tsotchke/quantum_geometric_tensor),
 * lilirrep, SbNN.
 *
 * @param m  QWZ mass parameter.
 * @param N  BZ grid side (N >= 4; 32 is plenty for a clean integer).
 * @param out_chern  Optional: writes the raw (pre-rounding) Chern
 *                   value; may be NULL.
 * @return integer Chern number rounded to nearest int, or INT_MIN on
 *         error (bad arguments / allocation failure).
 *
 * @since 0.2.0
 */
int moonlab_qwz_chern(double m, size_t N, double* out_chern);

/* ---- ML-KEM-512 (FIPS 203) PQC KEM (stable from 0.2.0) -------------- */

#define MOONLAB_MLKEM512_PUBLICKEYBYTES   800
#define MOONLAB_MLKEM512_SECRETKEYBYTES   1632
#define MOONLAB_MLKEM512_CIPHERTEXTBYTES  768
#define MOONLAB_MLKEM512_SHAREDSECRETBYTES 32

/**
 * @brief Generate an ML-KEM-512 key pair, with entropy sourced from
 *        Moonlab's Bell-verified quantum RNG.
 *
 * @param ek 800-byte output public key.
 * @param dk 1632-byte output secret key.
 * @return  0 on success, -1 on entropy failure.
 * @since 0.2.0
 */
int moonlab_mlkem512_keygen_qrng(uint8_t* ek, uint8_t* dk);

/**
 * @brief Encapsulate a shared secret against an ML-KEM-512 public key.
 *        Entropy for the inner message seed is drawn from
 *        @ref moonlab_qrng_bytes.
 *
 * @param c  768-byte output ciphertext.
 * @param K  32-byte output shared secret.
 * @param ek 800-byte public key.
 * @return  0 on success, -1 on entropy failure.
 * @since 0.2.0
 */
int moonlab_mlkem512_encaps_qrng(uint8_t* c, uint8_t* K, const uint8_t* ek);

/**
 * @brief Decapsulate an ML-KEM-512 ciphertext.  Constant-time;
 *        implicit-rejection on invalid ciphertexts.
 *
 * @param K  32-byte output shared secret (may be pseudorandom on
 *           tampered ciphertext).
 * @param c  768-byte ciphertext.
 * @param dk 1632-byte secret key.
 * @since 0.2.0
 */
void moonlab_mlkem512_decaps(uint8_t* K,
                              const uint8_t* c,
                              const uint8_t* dk);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MOONLAB_EXPORT_H */
