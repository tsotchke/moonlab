/**
 * @file ctr_drbg.h
 * @brief NIST SP 800-90A AES-256 CTR_DRBG without derivation function.
 *
 * This is the same DRBG the pq-crystals NIST KAT harness uses
 * (their `rng.c::randombytes_init` + `randombytes`), which is what
 * the FIPS 203 .rsp files are generated against.  Matching this
 * DRBG bit-for-bit is what lets a consumer validate ML-KEM output
 * against the official vectors.
 *
 * Security posture: this is a CSPRNG with ~256-bit strength assuming
 * the AES-256 block cipher is a PRP.  It is NOT a hardware RNG; it
 * should only be used with a properly-entropic seed (48 bytes here)
 * supplied by the caller.  For production KeyGen / Encaps entropy,
 * prefer `moonlab_qrng_bytes`.
 *
 * @since 0.2.0
 */

#ifndef MOONLAB_CTR_DRBG_H
#define MOONLAB_CTR_DRBG_H

#include <stddef.h>
#include <stdint.h>

#include "../aes/aes.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    aes256_ctx_t aes;      /* key schedule for current Key */
    uint8_t      key[32];  /* DRBG Key (256 bits) */
    uint8_t      V[16];    /* DRBG V (128 bits) */
    uint64_t     reseed_counter;
} ctr_drbg_ctx_t;

/**
 * @brief Instantiate the CTR_DRBG with a 48-byte seed.
 *
 * This is the simplified form used by the pq-crystals KAT harness:
 * `seed_material = entropy_input` (no personalization, no derivation
 * function), and seedlen = 48 = keylen + outlen.
 *
 * @param ctx       DRBG state.
 * @param seed      48-byte seed material.
 */
void ctr_drbg_init(ctr_drbg_ctx_t *ctx, const uint8_t seed[48]);

/**
 * @brief Generate @p len pseudorandom bytes into @p out.
 */
void ctr_drbg_generate(ctr_drbg_ctx_t *ctx, uint8_t *out, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_CTR_DRBG_H */
