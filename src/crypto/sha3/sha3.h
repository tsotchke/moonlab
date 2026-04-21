/**
 * @file sha3.h
 * @brief FIPS 202 (SHA-3) hashes and SHAKE XOFs.
 *
 * This header ships a reference implementation of the Keccak-f[1600]
 * permutation plus the four fixed-output SHA-3 hashes (SHA3-224,
 * SHA3-256, SHA3-384, SHA3-512) and the two extendable-output
 * functions (SHAKE128, SHAKE256) specified in NIST FIPS 202.
 *
 * This module was added in v0.2 for two reasons:
 *
 *   1. It is the foundation of every NIST-selected post-quantum
 *      primitive (ML-KEM / FIPS 203, ML-DSA / FIPS 204, SLH-DSA /
 *      FIPS 205).  None of those can be implemented without Keccak.
 *
 *   2. SHAKE128/256 is the obvious seed-stretcher for the Toeplitz
 *      extractor in @ref qrng_di_toeplitz_extract: a short
 *      genuinely-random seed plus a SHAKE call gives the expander
 *      ropes the extractor needs.
 *
 * API design:
 *   - @ref sha3_xxx_once: single-shot convenience wrapper.
 *   - @ref sha3_ctx_t + init / absorb / finalize / squeeze: streaming
 *     API for SHAKE XOFs and for hashing data that doesn't fit in
 *     memory all at once.
 *
 * This is a clean-room reference implementation (not a vendored
 * copy of tiny_sha3 or reference-xkcp); it is constant-time on
 * non-exotic CPUs, not side-channel hardened for adversarial
 * environments.  For FIPS-certified production use integrate with
 * BoringSSL / OpenSSL EVP; this module is adequate for PQC
 * reference implementations and for internal plumbing such as
 * extractor seeding.  All outputs are byte-for-byte identical to
 * the FIPS 202 reference implementation.
 *
 * @since 0.2.0
 */

#ifndef MOONLAB_SHA3_H
#define MOONLAB_SHA3_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------- */
/* Keccak-f[1600] permutation state and Sponge context           */
/* ------------------------------------------------------------- */

/**
 * @brief Sponge state for SHA-3 / SHAKE.  Treat as opaque outside
 *        this module.  The @p rate (in bytes) and @p domain byte
 *        select which specific FIPS 202 function is realised.
 */
typedef struct {
    uint64_t state[25];          /* 5 x 5 x 64-bit Keccak state */
    size_t   rate;                /* bytes absorbed per permutation */
    size_t   offset;              /* bytes already placed in the current block */
    uint8_t  domain;              /* 0x06 for SHA-3, 0x1F for SHAKE */
    uint8_t  finalized;           /* nonzero once squeezing has begun */
} sha3_ctx_t;

/* ------------------------------------------------------------- */
/* Fixed-output hashes                                            */
/* ------------------------------------------------------------- */

/** @brief One-shot SHA3-256.  @p out receives 32 bytes. */
void sha3_256(const uint8_t *in, size_t inlen, uint8_t out[32]);

/** @brief One-shot SHA3-512.  @p out receives 64 bytes. */
void sha3_512(const uint8_t *in, size_t inlen, uint8_t out[64]);

/** @brief One-shot SHA3-224.  @p out receives 28 bytes. */
void sha3_224(const uint8_t *in, size_t inlen, uint8_t out[28]);

/** @brief One-shot SHA3-384.  @p out receives 48 bytes. */
void sha3_384(const uint8_t *in, size_t inlen, uint8_t out[48]);

/* ------------------------------------------------------------- */
/* Streaming hash interface (for chunked input)                   */
/* ------------------------------------------------------------- */

/** @brief Initialise a streaming SHA3-256 context. */
void sha3_256_init(sha3_ctx_t *ctx);
/** @brief Initialise a streaming SHA3-512 context. */
void sha3_512_init(sha3_ctx_t *ctx);

/** @brief Feed bytes into a SHA-3 or SHAKE context. */
void sha3_update(sha3_ctx_t *ctx, const uint8_t *in, size_t inlen);

/**
 * @brief Finalise a fixed-output SHA-3 context and emit the digest.
 *        After this call the context is dead.  Do not call for SHAKE.
 */
void sha3_final(sha3_ctx_t *ctx, uint8_t *out);

/* ------------------------------------------------------------- */
/* SHAKE (extendable output) interface                            */
/* ------------------------------------------------------------- */

/** @brief Initialise a SHAKE128 context (rate 168, security 128). */
void shake128_init(sha3_ctx_t *ctx);
/** @brief Initialise a SHAKE256 context (rate 136, security 256). */
void shake256_init(sha3_ctx_t *ctx);

/**
 * @brief Squeeze @p outlen bytes from a SHAKE context.  May be
 *        called repeatedly; the total output stream is as defined
 *        by FIPS 202.  Once called at least once, no more
 *        @ref sha3_update calls are permitted on this context.
 */
void shake_squeeze(sha3_ctx_t *ctx, uint8_t *out, size_t outlen);

/**
 * @brief One-shot SHAKE128: absorb @p inlen bytes and squeeze
 *        @p outlen bytes.
 */
void shake128(const uint8_t *in, size_t inlen, uint8_t *out, size_t outlen);

/** @brief One-shot SHAKE256. */
void shake256(const uint8_t *in, size_t inlen, uint8_t *out, size_t outlen);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_SHA3_H */
