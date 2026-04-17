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
#define MOONLAB_ABI_VERSION_MINOR 1
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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MOONLAB_EXPORT_H */
