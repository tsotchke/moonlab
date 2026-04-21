/**
 * @file mlkem.h
 * @brief FIPS 203 Module-Lattice-based Key-Encapsulation Mechanism.
 *
 * This header is the end-user entry point to the ML-KEM-512 KEM.
 * Three operations:
 *
 *   - @ref moonlab_mlkem512_keygen  -- random key generation.
 *   - @ref moonlab_mlkem512_encaps  -- derive a shared secret and
 *                                      ciphertext given a public key.
 *   - @ref moonlab_mlkem512_decaps  -- recover the shared secret from
 *                                      the ciphertext + secret key,
 *                                      with implicit rejection on
 *                                      invalid ciphertexts (i.e. the
 *                                      Fujisaki-Okamoto transform
 *                                      returns a pseudorandom secret
 *                                      rather than aborting).
 *
 * All randomness is supplied by the caller via a 32-byte seed input
 * argument to each operation that needs randomness, so the module
 * is deterministic for testing.  In production, feed high-entropy
 * bytes from hardware_entropy.h or moonlab_qrng_bytes.  The FIPS
 * 203 internal variable naming (d, z, m, rho, sigma) is followed in
 * the implementation; see @c mlkem.c.
 *
 * ML-KEM-768 and ML-KEM-1024 are one-struct extensions on top of
 * the ML-KEM-512 code; their public interfaces will follow the
 * same pattern in the next minor release.
 *
 * @since 0.2.0
 */

#ifndef MOONLAB_MLKEM_H
#define MOONLAB_MLKEM_H

#include "params.h"

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
void moonlab_mlkem512_keygen(uint8_t ek[MLKEM512_PUBLICKEYBYTES],
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
void moonlab_mlkem512_encaps(uint8_t c[MLKEM512_CIPHERTEXTBYTES],
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
void moonlab_mlkem512_decaps(uint8_t K[32],
                              const uint8_t c[MLKEM512_CIPHERTEXTBYTES],
                              const uint8_t dk[MLKEM512_SECRETKEYBYTES]);

/* ---- Convenience wrappers driven by moonlab_qrng_bytes ---------- */

/**
 * @brief KeyGen with entropy sourced from @c moonlab_qrng_bytes.
 *
 * Internally draws 64 bytes from the Bell-verified quantum RNG to
 * populate (d, z) and then calls @ref moonlab_mlkem512_keygen.  The
 * single public entry point that ties Moonlab's quantum entropy
 * source to a FIPS-203 PQC key pair in one call.
 *
 * @return 0 on success; -1 on QRNG failure.
 */
int moonlab_mlkem512_keygen_qrng(uint8_t ek[MLKEM512_PUBLICKEYBYTES],
                                   uint8_t dk[MLKEM512_SECRETKEYBYTES]);

/**
 * @brief Encaps with the internal message seed drawn from
 *        @c moonlab_qrng_bytes.
 *
 * @return 0 on success; -1 on QRNG failure.
 */
int moonlab_mlkem512_encaps_qrng(uint8_t c[MLKEM512_CIPHERTEXTBYTES],
                                   uint8_t K[32],
                                   const uint8_t ek[MLKEM512_PUBLICKEYBYTES]);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_MLKEM_H */
