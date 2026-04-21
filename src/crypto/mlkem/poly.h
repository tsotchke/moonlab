/**
 * @file poly.h
 * @brief ML-KEM polynomial ring operations over R_q = Z_q[X]/(X^n + 1).
 *
 * q = 3329, n = 256.  All arithmetic happens on int16_t coefficients;
 * values are kept in the canonical range [0, q - 1] after reductions.
 *
 * The operations here are the primitives FIPS 203 needs:
 *
 *   - Polynomial add / sub / reduce.
 *   - Number-Theoretic Transform (forward + inverse) over the
 *     2^8-th root of unity 17 mod 3329.  The NTT turns Schoolbook
 *     polynomial multiplication into O(n log n) point-wise multiply
 *     of 128 degree-1 residues.
 *   - Pointwise multiplication in NTT domain (@ref poly_basemul).
 *   - Centered binomial sampler CBD_eta (FIPS 203 Algorithm 6), used
 *     to sample the noise / secret distributions.
 *   - Byte encode / decode for canonical 12-bit packed representation.
 *   - Compress / decompress used to shrink public keys and ciphertexts
 *     (FIPS 203 Section 4.2.1).
 *
 * @since 0.2.0
 */

#ifndef MOONLAB_MLKEM_POLY_H
#define MOONLAB_MLKEM_POLY_H

#include "params.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief A polynomial with N = 256 coefficients in [0, q - 1]. */
typedef struct {
    int16_t coeffs[MLKEM_N];
} mlkem_poly_t;

/* ---- Reduction helpers -------------------------------------- */

/**
 * @brief Barrett reduction: returns a mod q with a in (-q * 2^15, q * 2^15).
 *        Output in [0, q - 1].
 */
int16_t mlkem_barrett_reduce(int16_t a);

/**
 * @brief Montgomery-style reduction used by the NTT pointwise multiply
 *        (returns a * (2^-16) mod q with a in signed 32-bit).
 */
int16_t mlkem_montgomery_reduce(int32_t a);

/* ---- Polynomial operations --------------------------------- */

/** @brief dst = a + b (mod q, coefficient-wise). */
void mlkem_poly_add(mlkem_poly_t *dst, const mlkem_poly_t *a, const mlkem_poly_t *b);
/** @brief dst = a - b (mod q, coefficient-wise). */
void mlkem_poly_sub(mlkem_poly_t *dst, const mlkem_poly_t *a, const mlkem_poly_t *b);
/** @brief Reduce each coefficient of p to [0, q - 1]. */
void mlkem_poly_reduce(mlkem_poly_t *p);

/** @brief In-place forward NTT. */
void mlkem_poly_ntt(mlkem_poly_t *p);
/** @brief In-place inverse NTT + Montgomery correction. */
void mlkem_poly_invntt(mlkem_poly_t *p);
/**
 * @brief Multiply every coefficient of @p p by the Montgomery factor
 *        R = 2^16 mod q.  Used after @ref mlkem_poly_basemul to
 *        bring accumulated products back into the same form as the
 *        NTT-transformed inputs so they can be added to NTT-transformed
 *        noise/error polynomials.
 */
void mlkem_poly_tomont(mlkem_poly_t *p);

/**
 * @brief Pointwise multiplication of two polynomials in NTT domain;
 *        each pair (c[2i], c[2i+1]) = a[2i..2i+1] * b[2i..2i+1]
 *        mod (X^2 - zeta_i).  See FIPS 203 Algorithm 11.
 */
void mlkem_poly_basemul(mlkem_poly_t *dst,
                         const mlkem_poly_t *a, const mlkem_poly_t *b);

/* ---- Sampling ----------------------------------------------- */

/**
 * @brief Centered binomial sampler CBD_eta.  Expects
 *        @p buf_len >= eta * n / 4 bytes of uniformly-random input
 *        (typically SHAKE output seeded by a domain-separated seed).
 */
void mlkem_poly_cbd(mlkem_poly_t *p, const uint8_t *buf, int eta);

/* ---- Byte encoding ------------------------------------------ */

/**
 * @brief Pack a polynomial into MLKEM_POLYBYTES = 384 bytes.  Each
 *        coefficient is 12 bits (q fits in 12 bits).
 */
void mlkem_poly_tobytes(uint8_t out[MLKEM_POLYBYTES], const mlkem_poly_t *p);
/** @brief Inverse of @ref mlkem_poly_tobytes. */
void mlkem_poly_frombytes(mlkem_poly_t *p, const uint8_t in[MLKEM_POLYBYTES]);

/**
 * @brief Compress each coefficient of @p p to @p d bits, pack into
 *        ceil(n * d / 8) bytes.  FIPS 203 Algorithm 4.
 */
void mlkem_poly_compress(uint8_t *out, const mlkem_poly_t *p, int d);
/** @brief Decompress the inverse of @ref mlkem_poly_compress. */
void mlkem_poly_decompress(mlkem_poly_t *p, const uint8_t *in, int d);

/**
 * @brief Build the matrix A from a 32-byte @p seed.
 *        Writes a k*k array of polynomials; the (i, j) entry is
 *        sampled by rejection from SHAKE128(seed || j || i) (FIPS 203
 *        Algorithm 7 -- "GenMatrix").  When @p transposed is non-zero
 *        the indices are swapped (FIPS 203 uses A^T in decapsulation).
 */
void mlkem_gen_matrix(mlkem_poly_t *A, int k, const uint8_t seed[32],
                       int transposed);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_MLKEM_POLY_H */
