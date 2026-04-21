/**
 * @file qrng_di.c
 * @brief Pironio min-entropy bound + Toeplitz extractor for DI-QRNG.
 */

#include "qrng_di.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------- */
/* Min-entropy from CHSH                                          */
/* ------------------------------------------------------------- */

double qrng_di_min_entropy_from_chsh(double chsh) {
    const double TSIRELSON = 2.0 * 1.41421356237309504880;
    if (chsh <= 2.0) return 0.0;
    if (chsh >= TSIRELSON) return 1.0;
    /* Pironio et al. 2010: H_min >= 1 - log2(1 + sqrt(2 - S^2/4)).
     * At S = 2 the argument of sqrt is 1 -> H_min = 0; at
     * S = 2*sqrt(2) -> 1 - log2(1 + 0) = 1.  Monotone in S. */
    double inside = 2.0 - chsh * chsh / 4.0;
    if (inside < 0.0) inside = 0.0;
    double h = 1.0 - log2(1.0 + sqrt(inside));
    if (h < 0.0) h = 0.0;
    if (h > 1.0) h = 1.0;
    return h;
}

/* ------------------------------------------------------------- */
/* Toeplitz extractor                                             */
/* ------------------------------------------------------------- */

/* Get bit i (zero-indexed, LSB-first within each byte) from a packed
 * bitstream. */
static inline int getbit(const uint8_t *p, size_t i) {
    return (p[i >> 3] >> (i & 7)) & 1;
}
static inline void setbit(uint8_t *p, size_t i, int b) {
    uint8_t mask = (uint8_t)(1u << (i & 7));
    if (b) p[i >> 3] |= mask;
    else   p[i >> 3] &= (uint8_t)~mask;
}

int qrng_di_toeplitz_extract(const uint8_t *raw, size_t n_in,
                              const uint8_t *seed, size_t n_seed,
                              uint8_t *out, size_t n_out) {
    if (!raw || !seed || !out) return -1;
    const size_t n_in_bits  = n_in * 8;
    const size_t n_out_bits = n_out * 8;
    if (n_out_bits == 0 || n_in_bits == 0) return -1;
    /* Toeplitz matrix T is m x n with entries T_{i,j} = seed[i + j],
     * so we need n + m - 1 seed bits. */
    const size_t need_seed_bits = n_in_bits + n_out_bits - 1;
    if (n_seed * 8 < need_seed_bits) return -2;

    memset(out, 0, n_out);
    /* y[i] = XOR_j T_{i,j} * x[j] = XOR_j seed[i + j] * raw[j]. */
    for (size_t i = 0; i < n_out_bits; i++) {
        int acc = 0;
        for (size_t j = 0; j < n_in_bits; j++) {
            int tij = getbit(seed, i + j);
            if (tij) acc ^= getbit(raw, j);
        }
        setbit(out, i, acc);
    }
    return 0;
}

/* ------------------------------------------------------------- */
/* Raw-byte sizing                                                */
/* ------------------------------------------------------------- */

size_t qrng_di_raw_bytes_for_output(double chsh,
                                     size_t n_out,
                                     size_t epsilon_bits) {
    const double h = qrng_di_min_entropy_from_chsh(chsh);
    if (h <= 0.0) return 0;
    const double needed_bits = (double)(8 * n_out) + (double)epsilon_bits;
    /* n_raw_bits * h >= needed_bits. */
    double raw_bits = ceil(needed_bits / h);
    size_t raw_bytes = (size_t)((raw_bits + 7.0) / 8.0);
    /* Minimum: enough bytes to feed Toeplitz with at least n_out bits. */
    if (raw_bytes < n_out) raw_bytes = n_out;
    return raw_bytes;
}
