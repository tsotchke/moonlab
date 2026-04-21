/**
 * @file ctr_drbg.c
 * @brief AES-256 CTR_DRBG implementation (pq-crystals NIST variant).
 *
 * Structure follows NIST SP 800-90A (2015) Section 10.2.1.2 with
 * no derivation function.  For seedlen = 48 bytes = 384 bits, the
 * Update operation re-keys on every Generate call, which is the
 * identical sequence of bytes pq-crystals's `rng.c` produces for
 * the PQC NIST KAT files.
 */

#include "ctr_drbg.h"

#include <string.h>

/* Increment the big-endian 128-bit counter V by 1 in place. */
static void inc_V(uint8_t V[16]) {
    for (int i = 15; i >= 0; i--) {
        V[i]++;
        if (V[i] != 0) break;
    }
}

/* CTR_DRBG_Update(provided_data, Key, V).
 *   temp = empty
 *   while len(temp) < 48:
 *       V = (V+1) mod 2^128
 *       temp ||= AES-256-ECB(Key, V)
 *   temp = temp[0..47]
 *   temp ^= provided_data
 *   Key = temp[0..31];  V = temp[32..47]
 * Re-keys the AES context from the new Key.
 */
static void ctr_drbg_update(ctr_drbg_ctx_t *ctx,
                             const uint8_t provided_data[48]) {
    uint8_t tmp[48];
    for (int i = 0; i < 3; i++) {
        inc_V(ctx->V);
        aes256_encrypt_block(&ctx->aes, ctx->V, tmp + 16 * i);
    }
    if (provided_data) {
        for (int i = 0; i < 48; i++) tmp[i] ^= provided_data[i];
    }
    memcpy(ctx->key, tmp,      32);
    memcpy(ctx->V,   tmp + 32, 16);
    aes256_init(&ctx->aes, ctx->key);
}

void ctr_drbg_init(ctr_drbg_ctx_t *ctx, const uint8_t seed[48]) {
    memset(ctx->key, 0, 32);
    memset(ctx->V,   0, 16);
    aes256_init(&ctx->aes, ctx->key);
    ctr_drbg_update(ctx, seed);
    ctx->reseed_counter = 1;
}

void ctr_drbg_generate(ctr_drbg_ctx_t *ctx, uint8_t *out, size_t len) {
    uint8_t block[16];
    while (len > 0) {
        inc_V(ctx->V);
        aes256_encrypt_block(&ctx->aes, ctx->V, block);
        size_t take = (len < 16) ? len : 16;
        memcpy(out, block, take);
        out += take;
        len -= take;
    }
    /* Post-generate Update with zero provided_data (pq-crystals "no
     * additional_input" path reduces to Update(NULL ...) which XORs
     * temp with zero-bytes). */
    uint8_t zero[48] = {0};
    ctr_drbg_update(ctx, zero);
    ctx->reseed_counter++;
}
