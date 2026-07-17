/**
 * @file mlkem.h
 * @brief FIPS 203 Module-Lattice-based Key-Encapsulation Mechanism.
 *
 * This header is the end-user entry point to the ML-KEM KEM.  All three
 * FIPS 203 parameter sets -- ML-KEM-512, ML-KEM-768, and ML-KEM-1024 --
 * are fully implemented and share a single generic engine driven by a
 * runtime parameter struct (see @c mlkem.c).  Each parameter set exposes
 * the same four operations, e.g. for ML-KEM-512:
 *
 *   - @ref moonlab_mlkem512_keygen  -- key generation from (d, z).
 *   - @ref moonlab_mlkem512_encaps  -- derive a shared secret and
 *                                      ciphertext given a public key.
 *   - @ref moonlab_mlkem512_decaps  -- recover the shared secret from
 *                                      the ciphertext + secret key,
 *                                      with implicit rejection on
 *                                      invalid ciphertexts (i.e. the
 *                                      Fujisaki-Okamoto transform
 *                                      returns a pseudorandom secret
 *                                      rather than aborting).
 *   - @ref moonlab_mlkem512_check_ek / @ref moonlab_mlkem512_check_dk --
 *                                      FIPS 203 Section 7.2 / 7.3 input
 *                                      validation of a public / secret key.
 *
 * Encaps validates its ek (Section 7.2 modulus check) and Decaps
 * validates its dk (Section 7.3 hash check).  Because the core encaps /
 * decaps entry points return void, an invalid key is rejected
 * fail-closed (zeroized output); callers wanting an explicit result call
 * the @c check_ek / @c check_dk predicates first, or use the @c *_qrng
 * wrappers which return -2 on an invalid ek.
 *
 * All randomness is supplied by the caller via a 32-byte seed input
 * argument to each operation that needs randomness, so the module
 * is deterministic for testing.  In production, feed high-entropy
 * bytes from hardware_entropy.h or moonlab_qrng_bytes.  The FIPS
 * 203 internal variable naming (d, z, m, rho, sigma) is followed in
 * the implementation.
 *
 * Conformance is pinned in tests/unit/test_mlkem_acvp.c against the
 * official NIST ACVP FIPS 203 vectors (keyGen, encaps, decaps, and the
 * encapsulation/decapsulation key checks) for all three parameter sets.
 *
 * @since 0.2.0
 */

#ifndef MOONLAB_MLKEM_H
#define MOONLAB_MLKEM_H

#include "params.h"
#include "../../applications/moonlab_api.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generate a fresh ML-KEM-512 (public, secret) key pair.
 *
 * Feeds @p d (32 bytes) and @p z (32 bytes) from the caller-supplied
 * entropy.  Writes 800 bytes of public key to @p ek and 1632 bytes
 * of secret key to @p dk.
 */
MOONLAB_API void moonlab_mlkem512_keygen(uint8_t ek[MLKEM512_PUBLICKEYBYTES],
                              uint8_t dk[MLKEM512_SECRETKEYBYTES],
                              const uint8_t d[32],
                              const uint8_t z[32]);

/**
 * @brief Encapsulate a shared secret against a public key.
 *
 * Uses @p m (32 bytes of fresh random) as the internal FO seed.
 * Writes a 768-byte ciphertext to @p c and a 32-byte shared secret
 * to @p K.
 */
MOONLAB_API void moonlab_mlkem512_encaps(uint8_t c[MLKEM512_CIPHERTEXTBYTES],
                              uint8_t K[32],
                              const uint8_t ek[MLKEM512_PUBLICKEYBYTES],
                              const uint8_t m[32]);

/**
 * @brief Decapsulate a ciphertext using the secret key.
 *
 * On a valid ciphertext the original @p K is recovered.  On an
 * invalid ciphertext, a *different* 32-byte value is produced
 * pseudorandomly via SHAKE256(z || c), preserving IND-CCA2 security
 * without leaking failure via timing.
 */
MOONLAB_API void moonlab_mlkem512_decaps(uint8_t K[32],
                              const uint8_t c[MLKEM512_CIPHERTEXTBYTES],
                              const uint8_t dk[MLKEM512_SECRETKEYBYTES]);

/**
 * @brief FIPS 203 Section 7.2 encapsulation-key check.
 * @return 1 if @p ek is well formed (all coefficients reduced mod q),
 *         0 otherwise.
 */
MOONLAB_API int moonlab_mlkem512_check_ek(const uint8_t ek[MLKEM512_PUBLICKEYBYTES]);

/**
 * @brief FIPS 203 Section 7.3 decapsulation-key check.
 * @return 1 if @p dk is internally consistent (embedded H(ek) matches),
 *         0 otherwise.
 */
MOONLAB_API int moonlab_mlkem512_check_dk(const uint8_t dk[MLKEM512_SECRETKEYBYTES]);

/* ---- Convenience wrappers driven by moonlab_qrng_bytes ---------- */

/**
 * @brief KeyGen with entropy sourced from @c moonlab_qrng_bytes.
 *
 * Internally draws 64 bytes from Moonlab's health-tested, Bell-gated,
 * SHAKE256-conditioned hybrid RNG to populate (d, z), then calls
 * @ref moonlab_mlkem512_keygen. Deployments requiring a validated module
 * boundary can instead supply explicit seeds from that module's approved DRBG.
 *
 * @return 0 on success; -1 on QRNG failure.
 */
MOONLAB_API int moonlab_mlkem512_keygen_qrng(uint8_t ek[MLKEM512_PUBLICKEYBYTES],
                                   uint8_t dk[MLKEM512_SECRETKEYBYTES]);

/**
 * @brief Encaps with the internal message seed drawn from
 *        @c moonlab_qrng_bytes.
 *
 * @return 0 on success; -1 on QRNG failure.
 */
MOONLAB_API int moonlab_mlkem512_encaps_qrng(uint8_t c[MLKEM512_CIPHERTEXTBYTES],
                                   uint8_t K[32],
                                   const uint8_t ek[MLKEM512_PUBLICKEYBYTES]);

/* ---- ML-KEM-768 ------------------------------------------------- */
MOONLAB_API void moonlab_mlkem768_keygen(uint8_t ek[MLKEM768_PUBLICKEYBYTES],
                              uint8_t dk[MLKEM768_SECRETKEYBYTES],
                              const uint8_t d[32], const uint8_t z[32]);
MOONLAB_API void moonlab_mlkem768_encaps(uint8_t c[MLKEM768_CIPHERTEXTBYTES],
                              uint8_t K[32],
                              const uint8_t ek[MLKEM768_PUBLICKEYBYTES],
                              const uint8_t m[32]);
MOONLAB_API void moonlab_mlkem768_decaps(uint8_t K[32],
                              const uint8_t c[MLKEM768_CIPHERTEXTBYTES],
                              const uint8_t dk[MLKEM768_SECRETKEYBYTES]);
MOONLAB_API int  moonlab_mlkem768_check_ek(const uint8_t ek[MLKEM768_PUBLICKEYBYTES]);
MOONLAB_API int  moonlab_mlkem768_check_dk(const uint8_t dk[MLKEM768_SECRETKEYBYTES]);
MOONLAB_API int  moonlab_mlkem768_keygen_qrng(uint8_t ek[MLKEM768_PUBLICKEYBYTES],
                                    uint8_t dk[MLKEM768_SECRETKEYBYTES]);
MOONLAB_API int  moonlab_mlkem768_encaps_qrng(uint8_t c[MLKEM768_CIPHERTEXTBYTES],
                                    uint8_t K[32],
                                    const uint8_t ek[MLKEM768_PUBLICKEYBYTES]);

/* ---- ML-KEM-1024 ------------------------------------------------ */
MOONLAB_API void moonlab_mlkem1024_keygen(uint8_t ek[MLKEM1024_PUBLICKEYBYTES],
                               uint8_t dk[MLKEM1024_SECRETKEYBYTES],
                               const uint8_t d[32], const uint8_t z[32]);
MOONLAB_API void moonlab_mlkem1024_encaps(uint8_t c[MLKEM1024_CIPHERTEXTBYTES],
                               uint8_t K[32],
                               const uint8_t ek[MLKEM1024_PUBLICKEYBYTES],
                               const uint8_t m[32]);
MOONLAB_API void moonlab_mlkem1024_decaps(uint8_t K[32],
                               const uint8_t c[MLKEM1024_CIPHERTEXTBYTES],
                               const uint8_t dk[MLKEM1024_SECRETKEYBYTES]);
MOONLAB_API int  moonlab_mlkem1024_check_ek(const uint8_t ek[MLKEM1024_PUBLICKEYBYTES]);
MOONLAB_API int  moonlab_mlkem1024_check_dk(const uint8_t dk[MLKEM1024_SECRETKEYBYTES]);
MOONLAB_API int  moonlab_mlkem1024_keygen_qrng(uint8_t ek[MLKEM1024_PUBLICKEYBYTES],
                                     uint8_t dk[MLKEM1024_SECRETKEYBYTES]);
MOONLAB_API int  moonlab_mlkem1024_encaps_qrng(uint8_t c[MLKEM1024_CIPHERTEXTBYTES],
                                     uint8_t K[32],
                                     const uint8_t ek[MLKEM1024_PUBLICKEYBYTES]);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_MLKEM_H */
