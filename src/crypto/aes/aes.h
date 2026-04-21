/**
 * @file aes.h
 * @brief AES-256 ECB block cipher (FIPS 197).
 *
 * Minimal ECB-only API used by the SP 800-90A CTR_DRBG in
 * `src/crypto/drbg/`.  Only AES-256 is exposed; the key-schedule
 * is computed once at context init.  Reference implementation --
 * not constant-time against power/cache-timing adversaries, but
 * constant-time on non-exotic CPUs and byte-for-byte identical to
 * the FIPS 197 Appendix C.3 test vector.
 *
 * For production deployment on adversarial hardware, route through
 * BoringSSL's EVP API; this module is adequate for on-host PQC
 * KAT generation and QRNG post-processing.
 *
 * @since 0.2.0
 */

#ifndef MOONLAB_AES_H
#define MOONLAB_AES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** AES-256 expanded round keys: 15 rounds * 16 bytes = 240 bytes. */
typedef struct {
    uint8_t round_keys[240];
} aes256_ctx_t;

/** @brief Expand a 32-byte key into the AES-256 round-key schedule. */
void aes256_init(aes256_ctx_t *ctx, const uint8_t key[32]);

/** @brief Encrypt one 16-byte block in place. */
void aes256_encrypt_block(const aes256_ctx_t *ctx,
                           const uint8_t in[16], uint8_t out[16]);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_AES_H */
