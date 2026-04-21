/**
 * @file sha3.c
 * @brief Keccak-f[1600] + FIPS 202 sponge implementation.
 *
 * Clean-room implementation following FIPS 202 (Aug 2015) and the
 * Keccak team's pseudo-code.  Passes the NIST intermediate-value
 * test vectors (0-byte, 1-byte, 4097-byte inputs for SHA3-256 and
 * SHAKE128/256) in tests/unit/test_sha3.c.
 *
 * This module is perf-adequate for PQC reference implementations
 * (~ hundreds of MB/s on modern hardware) but is not ChaCha20-style
 * fast; a table-free lane-permuted variant would roughly double
 * throughput.  The architecture makes that a drop-in upgrade.
 */

#include "sha3.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------- */
/* Keccak-f[1600] constants                                       */
/* ------------------------------------------------------------- */

/* Round constants from FIPS 202 Table 1. */
static const uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

/* Rotation offsets for rho step, indexed by (x, y) with x + 5*y. */
static const int KECCAK_ROT[25] = {
    0,  1, 62, 28, 27,
   36, 44,  6, 55, 20,
    3, 10, 43, 25, 39,
   41, 45, 15, 21,  8,
   18,  2, 61, 56, 14,
};

static inline uint64_t ROTL64(uint64_t x, int n) {
    n &= 63;
    return (x << n) | (x >> ((64 - n) & 63));
}

/* ------------------------------------------------------------- */
/* Permutation                                                    */
/* ------------------------------------------------------------- */

static void keccak_f1600(uint64_t A[25]) {
    for (int round = 0; round < 24; round++) {
        /* theta. */
        uint64_t C[5], D[5];
        for (int x = 0; x < 5; x++) {
            C[x] = A[x] ^ A[x + 5] ^ A[x + 10] ^ A[x + 15] ^ A[x + 20];
        }
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ ROTL64(C[(x + 1) % 5], 1);
        }
        for (int i = 0; i < 25; i++) {
            A[i] ^= D[i % 5];
        }

        /* rho + pi.  Build B[] = pi(rho(A)). */
        uint64_t B[25];
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                int src = x + 5 * y;
                int dst = y + 5 * ((2 * x + 3 * y) % 5);
                B[dst] = ROTL64(A[src], KECCAK_ROT[src]);
            }
        }

        /* chi. */
        for (int y = 0; y < 5; y++) {
            int row = 5 * y;
            uint64_t t0 = B[row], t1 = B[row + 1];
            for (int x = 0; x < 5; x++) {
                uint64_t nx1 = (x + 1 < 5) ? B[row + x + 1] : t0;
                uint64_t nx2 = (x + 2 < 5) ? B[row + x + 2] : ((x + 1 < 5) ? t0 : t1);
                A[row + x] = B[row + x] ^ ((~nx1) & nx2);
            }
        }

        /* iota. */
        A[0] ^= KECCAK_RC[round];
    }
}

/* ------------------------------------------------------------- */
/* Sponge driver                                                  */
/* ------------------------------------------------------------- */

static void sponge_init(sha3_ctx_t *ctx, size_t rate, uint8_t domain) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->rate   = rate;
    ctx->domain = domain;
}

void sha3_update(sha3_ctx_t *ctx, const uint8_t *in, size_t inlen) {
    if (!ctx || ctx->finalized) return;
    /* We stream byte-by-byte into the state's lane representation,
     * which avoids needing a rate-sized byte buffer. */
    while (inlen > 0) {
        size_t space = ctx->rate - ctx->offset;
        size_t take  = (inlen < space) ? inlen : space;
        for (size_t i = 0; i < take; i++) {
            size_t pos = ctx->offset + i;
            ctx->state[pos / 8] ^= ((uint64_t)in[i]) << ((pos % 8) * 8);
        }
        ctx->offset += take;
        in          += take;
        inlen       -= take;
        if (ctx->offset == ctx->rate) {
            keccak_f1600(ctx->state);
            ctx->offset = 0;
        }
    }
}

static void sponge_finalize_pad(sha3_ctx_t *ctx) {
    if (ctx->finalized) return;
    /* Append domain byte then 0x80 at end of block. */
    size_t pos = ctx->offset;
    ctx->state[pos / 8] ^= ((uint64_t)ctx->domain) << ((pos % 8) * 8);
    size_t last = ctx->rate - 1;
    ctx->state[last / 8] ^= ((uint64_t)0x80) << ((last % 8) * 8);
    keccak_f1600(ctx->state);
    ctx->offset    = 0;
    ctx->finalized = 1;
}

static void squeeze(sha3_ctx_t *ctx, uint8_t *out, size_t outlen) {
    if (!ctx->finalized) sponge_finalize_pad(ctx);
    while (outlen > 0) {
        size_t avail = ctx->rate - ctx->offset;
        size_t take  = (outlen < avail) ? outlen : avail;
        for (size_t i = 0; i < take; i++) {
            size_t pos = ctx->offset + i;
            out[i] = (uint8_t)(ctx->state[pos / 8] >> ((pos % 8) * 8));
        }
        ctx->offset += take;
        out         += take;
        outlen      -= take;
        if (ctx->offset == ctx->rate && outlen > 0) {
            keccak_f1600(ctx->state);
            ctx->offset = 0;
        }
    }
}

/* ------------------------------------------------------------- */
/* Public API                                                     */
/* ------------------------------------------------------------- */

/* SHA3-N rates: 200 - 2*(N/8) bytes.  e.g. SHA3-256 -> 136 bytes. */

void sha3_256_init(sha3_ctx_t *ctx) { sponge_init(ctx, 136, 0x06); }
void sha3_512_init(sha3_ctx_t *ctx) { sponge_init(ctx, 72,  0x06); }

static void sha3_224_init(sha3_ctx_t *ctx) { sponge_init(ctx, 144, 0x06); }
static void sha3_384_init(sha3_ctx_t *ctx) { sponge_init(ctx, 104, 0x06); }

void sha3_final(sha3_ctx_t *ctx, uint8_t *out) {
    sponge_finalize_pad(ctx);
    /* Fixed-output length: rate == 136 (256-bit), 72 (512), 144 (224), 104 (384).
     * Caller knows the length; squeeze exactly that many bytes. */
    size_t outlen;
    if      (ctx->rate == 144) outlen = 28;
    else if (ctx->rate == 136) outlen = 32;
    else if (ctx->rate == 104) outlen = 48;
    else if (ctx->rate == 72)  outlen = 64;
    else                         outlen = 0;
    /* First squeeze reads from an already-permuted state. */
    ctx->offset = 0;
    for (size_t i = 0; i < outlen; i++) {
        out[i] = (uint8_t)(ctx->state[i / 8] >> ((i % 8) * 8));
    }
}

void sha3_256(const uint8_t *in, size_t inlen, uint8_t out[32]) {
    sha3_ctx_t ctx; sha3_256_init(&ctx);
    sha3_update(&ctx, in, inlen);
    sha3_final(&ctx, out);
}
void sha3_512(const uint8_t *in, size_t inlen, uint8_t out[64]) {
    sha3_ctx_t ctx; sha3_512_init(&ctx);
    sha3_update(&ctx, in, inlen);
    sha3_final(&ctx, out);
}
void sha3_224(const uint8_t *in, size_t inlen, uint8_t out[28]) {
    sha3_ctx_t ctx; sha3_224_init(&ctx);
    sha3_update(&ctx, in, inlen);
    sha3_final(&ctx, out);
}
void sha3_384(const uint8_t *in, size_t inlen, uint8_t out[48]) {
    sha3_ctx_t ctx; sha3_384_init(&ctx);
    sha3_update(&ctx, in, inlen);
    sha3_final(&ctx, out);
}

void shake128_init(sha3_ctx_t *ctx) { sponge_init(ctx, 168, 0x1F); }
void shake256_init(sha3_ctx_t *ctx) { sponge_init(ctx, 136, 0x1F); }

void shake_squeeze(sha3_ctx_t *ctx, uint8_t *out, size_t outlen) {
    if (!ctx->finalized) {
        sponge_finalize_pad(ctx);
        /* After finalize, the state has been permuted and offset is 0:
         * immediate squeeze from byte 0 of the state. */
        ctx->offset = 0;
    }
    squeeze(ctx, out, outlen);
}

void shake128(const uint8_t *in, size_t inlen, uint8_t *out, size_t outlen) {
    sha3_ctx_t ctx; shake128_init(&ctx);
    sha3_update(&ctx, in, inlen);
    shake_squeeze(&ctx, out, outlen);
}

void shake256(const uint8_t *in, size_t inlen, uint8_t *out, size_t outlen) {
    sha3_ctx_t ctx; shake256_init(&ctx);
    sha3_update(&ctx, in, inlen);
    shake_squeeze(&ctx, out, outlen);
}
