/**
 * @file poly.c
 * @brief ML-KEM polynomial ring and NTT implementation.
 *
 * Arithmetic over R_q = Z_q[X] / (X^256 + 1) with q = 3329.
 *
 * Conventions match the pq-crystals Kyber reference code (which
 * became FIPS 203).  The NTT uses 17 as the primitive 256-th root of
 * unity and is an "incomplete" NTT that stops at pairs of coefficients
 * (each representing a polynomial mod X^2 - zeta_i).  Pointwise
 * multiplication (@ref mlkem_poly_basemul) therefore takes two such
 * pairs and outputs another pair.
 *
 * The zetas table below is stored in Montgomery form (zeta * R mod q
 * with R = 2^16 mod q = 2285) and in bit-reversed order of exponent
 * so the NTT sweep walks memory monotonically.
 *
 * References:
 *   - FIPS 203, "Module-Lattice-based Key-Encapsulation Mechanism
 *     Standard" (2024).
 *   - pq-crystals/kyber reference code, round-3 NIST submission.
 */

#include "poly.h"
#include "../sha3/sha3.h"

#include <stdint.h>
#include <string.h>

#define Q   MLKEM_Q
#define N   MLKEM_N
#define QINV (-3327) /* q^-1 mod 2^16, stored as int16 (62209 - 65536). */

/* Zetas in Montgomery form, bit-reversed exponent order (FIPS 203 Appendix). */
static const int16_t ZETAS[128] = {
   -1044,  -758,  -359, -1517,  1493,  1422,   287,   202,
    -171,   622,  1577,   182,   962, -1202, -1474,  1468,
     573, -1325,   264,   383,  -829,  1458, -1602,  -130,
    -681,  1017,   732,   608, -1542,   411,  -205, -1571,
    1223,   652,  -552,  1015, -1293,  1491,  -282, -1544,
     516,    -8,  -320,  -666, -1618, -1162,   126,  1469,
    -853,   -90,  -271,   830,   107, -1421,  -247,  -951,
    -398,   961, -1508,  -725,   448, -1065,   677, -1275,
   -1103,   430,   555,   843, -1251,   871,  1550,   105,
     422,   587,   177,  -235,  -291,  -460,  1574,  1653,
    -246,   778,  1159,  -147,  -777,  1483,  -602,  1119,
   -1590,   644,  -872,   349,   418,   329,  -156,   -75,
     817,  1097,   603,   610,  1322, -1285, -1465,   384,
   -1215,  -136,  1218, -1335,  -874,   220, -1187, -1659,
   -1185, -1530, -1278,   794, -1510,  -854,  -870,   478,
    -108,  -308,   996,   991,   958, -1460,  1522,  1628,
};

int16_t mlkem_montgomery_reduce(int32_t a) {
    int16_t t = (int16_t)((int16_t)a * (int16_t)QINV);
    return (int16_t)((a - (int32_t)t * Q) >> 16);
}

int16_t mlkem_barrett_reduce(int16_t a) {
    /* v = floor((2^26 + Q/2) / Q).  For q = 3329 that is 20159. */
    const int16_t v = 20159;
    int16_t t = (int16_t)(((int32_t)v * a + (1 << 25)) >> 26);
    return (int16_t)(a - (int32_t)t * Q);
}

static int16_t fqmul(int16_t a, int16_t b) {
    return mlkem_montgomery_reduce((int32_t)a * (int32_t)b);
}

void mlkem_poly_add(mlkem_poly_t *dst, const mlkem_poly_t *a, const mlkem_poly_t *b) {
    for (int i = 0; i < N; i++) dst->coeffs[i] = a->coeffs[i] + b->coeffs[i];
}
void mlkem_poly_sub(mlkem_poly_t *dst, const mlkem_poly_t *a, const mlkem_poly_t *b) {
    for (int i = 0; i < N; i++) dst->coeffs[i] = a->coeffs[i] - b->coeffs[i];
}
void mlkem_poly_reduce(mlkem_poly_t *p) {
    for (int i = 0; i < N; i++) {
        int16_t r = mlkem_barrett_reduce(p->coeffs[i]);
        if (r < 0) r += Q;
        p->coeffs[i] = r;
    }
}

/* -------------------------------------------------------------- */
/* NTT                                                             */
/* -------------------------------------------------------------- */

void mlkem_poly_ntt(mlkem_poly_t *p) {
    int16_t *r = p->coeffs;
    unsigned k = 1;
    for (unsigned len = 128; len >= 2; len >>= 1) {
        for (unsigned start = 0; start < N; start += 2 * len) {
            int16_t zeta = ZETAS[k++];
            for (unsigned j = start; j < start + len; j++) {
                int16_t t = fqmul(zeta, r[j + len]);
                r[j + len] = (int16_t)(r[j] - t);
                r[j]       = (int16_t)(r[j] + t);
            }
        }
    }
    mlkem_poly_reduce(p);
}

void mlkem_poly_invntt(mlkem_poly_t *p) {
    /* f = mont^2 / 128 = 1441 (Kyber reference). */
    const int16_t f = 1441;
    int16_t *r = p->coeffs;
    unsigned k = 127;
    for (unsigned len = 2; len <= 128; len <<= 1) {
        for (unsigned start = 0; start < N; start += 2 * len) {
            int16_t zeta = ZETAS[k--];
            for (unsigned j = start; j < start + len; j++) {
                int16_t t = r[j];
                r[j]       = mlkem_barrett_reduce((int16_t)(t + r[j + len]));
                r[j + len] = (int16_t)(r[j + len] - t);
                r[j + len] = fqmul(zeta, r[j + len]);
            }
        }
    }
    for (int j = 0; j < N; j++) r[j] = fqmul(r[j], f);
    mlkem_poly_reduce(p);
}

static void basemul_pair(int16_t r[2], const int16_t a[2],
                          const int16_t b[2], int16_t zeta) {
    r[0] = fqmul(a[1], b[1]);
    r[0] = fqmul(r[0], zeta);
    r[0] = (int16_t)(r[0] + fqmul(a[0], b[0]));
    r[1] = fqmul(a[0], b[1]);
    r[1] = (int16_t)(r[1] + fqmul(a[1], b[0]));
}

void mlkem_poly_basemul(mlkem_poly_t *dst,
                         const mlkem_poly_t *a, const mlkem_poly_t *b) {
    for (int i = 0; i < N / 4; i++) {
        basemul_pair(&dst->coeffs[4 * i],
                      &a->coeffs[4 * i], &b->coeffs[4 * i],
                      ZETAS[64 + i]);
        basemul_pair(&dst->coeffs[4 * i + 2],
                      &a->coeffs[4 * i + 2], &b->coeffs[4 * i + 2],
                      (int16_t)(-ZETAS[64 + i]));
    }
    mlkem_poly_reduce(dst);
}

/* -------------------------------------------------------------- */
/* CBD sampler                                                     */
/* -------------------------------------------------------------- */

static uint32_t load24_le(const uint8_t *b) {
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16);
}
static uint32_t load32_le(const uint8_t *b) {
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static void cbd2(mlkem_poly_t *p, const uint8_t *buf) {
    /* eta = 2: 256 coeffs * 4 bits = 128 bytes. */
    for (int i = 0; i < N / 8; i++) {
        uint32_t t = load32_le(buf + 4 * i);
        uint32_t d = t & 0x55555555u;
        d += (t >> 1) & 0x55555555u;
        for (int j = 0; j < 8; j++) {
            int16_t a = (int16_t)((d >> (4 * j    )) & 0x3);
            int16_t b = (int16_t)((d >> (4 * j + 2)) & 0x3);
            p->coeffs[8 * i + j] = (int16_t)(a - b);
        }
    }
}

static void cbd3(mlkem_poly_t *p, const uint8_t *buf) {
    /* eta = 3: 256 coeffs * 6 bits = 192 bytes; walk in 3-byte chunks. */
    for (int i = 0; i < N / 4; i++) {
        uint32_t t = load24_le(buf + 3 * i);
        uint32_t d = t & 0x00249249u;
        d += (t >> 1) & 0x00249249u;
        d += (t >> 2) & 0x00249249u;
        for (int j = 0; j < 4; j++) {
            int16_t a = (int16_t)((d >> (6 * j    )) & 0x7);
            int16_t b = (int16_t)((d >> (6 * j + 3)) & 0x7);
            p->coeffs[4 * i + j] = (int16_t)(a - b);
        }
    }
}

void mlkem_poly_cbd(mlkem_poly_t *p, const uint8_t *buf, int eta) {
    if      (eta == 2) cbd2(p, buf);
    else if (eta == 3) cbd3(p, buf);
    else memset(p, 0, sizeof(*p));
}

/* -------------------------------------------------------------- */
/* Byte encoding (12-bit packing)                                   */
/* -------------------------------------------------------------- */

void mlkem_poly_tobytes(uint8_t out[MLKEM_POLYBYTES], const mlkem_poly_t *p) {
    for (int i = 0; i < N / 2; i++) {
        int16_t a0 = p->coeffs[2 * i];
        int16_t a1 = p->coeffs[2 * i + 1];
        /* Canonicalise to [0, q - 1] just in case. */
        a0 += (a0 >> 15) & Q;
        a1 += (a1 >> 15) & Q;
        out[3 * i    ] = (uint8_t)(a0 >> 0);
        out[3 * i + 1] = (uint8_t)((a0 >> 8) | (a1 << 4));
        out[3 * i + 2] = (uint8_t)(a1 >> 4);
    }
}

void mlkem_poly_frombytes(mlkem_poly_t *p, const uint8_t in[MLKEM_POLYBYTES]) {
    for (int i = 0; i < N / 2; i++) {
        p->coeffs[2 * i    ] = (int16_t)(((uint16_t)in[3 * i    ]     ) |
                                         ((uint16_t)in[3 * i + 1] << 8))
                                & 0x0FFF;
        p->coeffs[2 * i + 1] = (int16_t)((uint16_t)in[3 * i + 1] >> 4) |
                               (int16_t)((uint16_t)in[3 * i + 2] << 4);
    }
}

/* -------------------------------------------------------------- */
/* Compress / decompress                                           */
/* -------------------------------------------------------------- */
/* FIPS 203 Section 4.2.1: compress_d(x) = round(2^d / q * x) mod 2^d,
 * decompress_d(y) = round(q / 2^d * y).  The reference rounded-division
 * is implemented by adding q/2 before the multiplication.            */

static uint16_t compress_coeff(int16_t x, int d) {
    /* x must be in [0, q - 1] first. */
    uint32_t u = (uint32_t)((int32_t)x + (x >> 15 & Q));
    uint32_t t = ((u << d) + Q / 2) / Q;
    return (uint16_t)(t & ((1u << d) - 1u));
}
static int16_t decompress_coeff(uint16_t y, int d) {
    uint32_t t = ((uint32_t)y * Q + (1u << (d - 1))) >> d;
    return (int16_t)t;
}

void mlkem_poly_compress(uint8_t *out, const mlkem_poly_t *p, int d) {
    if (d == 4) {
        /* d=4: 2 coefficients per byte.  n * d / 8 = 128 bytes. */
        for (int i = 0; i < N / 2; i++) {
            uint16_t c0 = compress_coeff(p->coeffs[2 * i],     4);
            uint16_t c1 = compress_coeff(p->coeffs[2 * i + 1], 4);
            out[i] = (uint8_t)(c0 | (c1 << 4));
        }
    } else if (d == 5) {
        /* d=5: pack 8 coefficients into 5 bytes.  n * 5 / 8 = 160. */
        for (int i = 0; i < N / 8; i++) {
            uint16_t c[8];
            for (int j = 0; j < 8; j++)
                c[j] = compress_coeff(p->coeffs[8 * i + j], 5);
            out[5 * i + 0] = (uint8_t)((c[0]     ) | (c[1] << 5));
            out[5 * i + 1] = (uint8_t)((c[1] >> 3) | (c[2] << 2) | (c[3] << 7));
            out[5 * i + 2] = (uint8_t)((c[3] >> 1) | (c[4] << 4));
            out[5 * i + 3] = (uint8_t)((c[4] >> 4) | (c[5] << 1) | (c[6] << 6));
            out[5 * i + 4] = (uint8_t)((c[6] >> 2) | (c[7] << 3));
        }
    } else if (d == 10) {
        /* d=10: pack 4 coeffs into 5 bytes.  n * 10 / 8 = 320. */
        for (int i = 0; i < N / 4; i++) {
            uint16_t c[4];
            for (int j = 0; j < 4; j++)
                c[j] = compress_coeff(p->coeffs[4 * i + j], 10);
            out[5 * i + 0] = (uint8_t)(c[0]);
            out[5 * i + 1] = (uint8_t)((c[0] >> 8) | (c[1] << 2));
            out[5 * i + 2] = (uint8_t)((c[1] >> 6) | (c[2] << 4));
            out[5 * i + 3] = (uint8_t)((c[2] >> 4) | (c[3] << 6));
            out[5 * i + 4] = (uint8_t)(c[3] >> 2);
        }
    } else if (d == 11) {
        /* d=11: pack 8 coeffs into 11 bytes.  n * 11 / 8 = 352. */
        for (int i = 0; i < N / 8; i++) {
            uint16_t c[8];
            for (int j = 0; j < 8; j++)
                c[j] = compress_coeff(p->coeffs[8 * i + j], 11);
            out[11 * i + 0]  = (uint8_t)(c[0]);
            out[11 * i + 1]  = (uint8_t)((c[0] >>  8) | (c[1] << 3));
            out[11 * i + 2]  = (uint8_t)((c[1] >>  5) | (c[2] << 6));
            out[11 * i + 3]  = (uint8_t)(c[2] >>  2);
            out[11 * i + 4]  = (uint8_t)((c[2] >> 10) | (c[3] << 1));
            out[11 * i + 5]  = (uint8_t)((c[3] >>  7) | (c[4] << 4));
            out[11 * i + 6]  = (uint8_t)((c[4] >>  4) | (c[5] << 7));
            out[11 * i + 7]  = (uint8_t)(c[5] >>  1);
            out[11 * i + 8]  = (uint8_t)((c[5] >>  9) | (c[6] << 2));
            out[11 * i + 9]  = (uint8_t)((c[6] >>  6) | (c[7] << 5));
            out[11 * i + 10] = (uint8_t)(c[7] >>  3);
        }
    }
}

void mlkem_poly_decompress(mlkem_poly_t *p, const uint8_t *in, int d) {
    if (d == 4) {
        for (int i = 0; i < N / 2; i++) {
            uint16_t c0 =  in[i]       & 0xF;
            uint16_t c1 = (in[i] >> 4) & 0xF;
            p->coeffs[2 * i    ] = decompress_coeff(c0, 4);
            p->coeffs[2 * i + 1] = decompress_coeff(c1, 4);
        }
    } else if (d == 5) {
        for (int i = 0; i < N / 8; i++) {
            uint16_t c[8];
            c[0] = (uint16_t)(in[5 * i + 0]       ) & 0x1F;
            c[1] = (uint16_t)((in[5 * i + 0] >> 5) | (in[5 * i + 1] << 3)) & 0x1F;
            c[2] = (uint16_t)(in[5 * i + 1] >> 2) & 0x1F;
            c[3] = (uint16_t)((in[5 * i + 1] >> 7) | (in[5 * i + 2] << 1)) & 0x1F;
            c[4] = (uint16_t)((in[5 * i + 2] >> 4) | (in[5 * i + 3] << 4)) & 0x1F;
            c[5] = (uint16_t)(in[5 * i + 3] >> 1) & 0x1F;
            c[6] = (uint16_t)((in[5 * i + 3] >> 6) | (in[5 * i + 4] << 2)) & 0x1F;
            c[7] = (uint16_t)(in[5 * i + 4] >> 3) & 0x1F;
            for (int j = 0; j < 8; j++)
                p->coeffs[8 * i + j] = decompress_coeff(c[j], 5);
        }
    } else if (d == 10) {
        for (int i = 0; i < N / 4; i++) {
            uint16_t c[4];
            c[0] = (uint16_t)((in[5 * i + 0])      | (in[5 * i + 1] << 8)) & 0x3FF;
            c[1] = (uint16_t)((in[5 * i + 1] >> 2) | (in[5 * i + 2] << 6)) & 0x3FF;
            c[2] = (uint16_t)((in[5 * i + 2] >> 4) | (in[5 * i + 3] << 4)) & 0x3FF;
            c[3] = (uint16_t)((in[5 * i + 3] >> 6) | (in[5 * i + 4] << 2)) & 0x3FF;
            for (int j = 0; j < 4; j++)
                p->coeffs[4 * i + j] = decompress_coeff(c[j], 10);
        }
    } else if (d == 11) {
        for (int i = 0; i < N / 8; i++) {
            uint16_t c[8];
            c[0] = (uint16_t)((in[11*i+0])       | (in[11*i+1] <<  8)) & 0x7FF;
            c[1] = (uint16_t)((in[11*i+1] >> 3)  | (in[11*i+2] <<  5)) & 0x7FF;
            c[2] = (uint16_t)((in[11*i+2] >> 6)  | (in[11*i+3] <<  2) | (in[11*i+4] << 10)) & 0x7FF;
            c[3] = (uint16_t)((in[11*i+4] >> 1)  | (in[11*i+5] <<  7)) & 0x7FF;
            c[4] = (uint16_t)((in[11*i+5] >> 4)  | (in[11*i+6] <<  4)) & 0x7FF;
            c[5] = (uint16_t)((in[11*i+6] >> 7)  | (in[11*i+7] <<  1) | (in[11*i+8] << 9)) & 0x7FF;
            c[6] = (uint16_t)((in[11*i+8] >> 2)  | (in[11*i+9] <<  6)) & 0x7FF;
            c[7] = (uint16_t)((in[11*i+9] >> 5)  | (in[11*i+10] << 3)) & 0x7FF;
            for (int j = 0; j < 8; j++)
                p->coeffs[8 * i + j] = decompress_coeff(c[j], 11);
        }
    }
}

/* -------------------------------------------------------------- */
/* Matrix A from seed (rejection sampling via SHAKE128)            */
/* -------------------------------------------------------------- */

static int rej_uniform(int16_t *r, int len, const uint8_t *buf, int buflen) {
    int ctr = 0, pos = 0;
    while (ctr < len && pos + 3 <= buflen) {
        uint16_t v0 = (uint16_t)(((uint16_t)buf[pos    ] >> 0) |
                                 ((uint16_t)buf[pos + 1] << 8)) & 0x0FFF;
        uint16_t v1 = (uint16_t)(((uint16_t)buf[pos + 1] >> 4) |
                                 ((uint16_t)buf[pos + 2] << 4)) & 0x0FFF;
        pos += 3;
        if (v0 < Q)       r[ctr++] = (int16_t)v0;
        if (ctr < len && v1 < Q) r[ctr++] = (int16_t)v1;
    }
    return ctr;
}

void mlkem_gen_matrix(mlkem_poly_t *A, int k, const uint8_t seed[32],
                       int transposed) {
    /* FIPS 203 Algorithm 7: for each (i, j), run SHAKE128 over
     *     seed || j_byte || i_byte   (or transposed indices)
     * and rejection-sample n coefficients in [0, q - 1]. */
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            uint8_t input[34];
            memcpy(input, seed, 32);
            if (transposed) {
                input[32] = (uint8_t)i;
                input[33] = (uint8_t)j;
            } else {
                input[32] = (uint8_t)j;
                input[33] = (uint8_t)i;
            }
            sha3_ctx_t sh; shake128_init(&sh);
            sha3_update(&sh, input, 34);
            int16_t *dst = A[i * k + j].coeffs;
            int have = 0;
            uint8_t chunk[168 * 3];
            while (have < N) {
                /* One SHAKE block = 168 bytes; keep squeezing until full. */
                shake_squeeze(&sh, chunk, 168);
                int added = rej_uniform(dst + have, N - have, chunk, 168);
                have += added;
            }
        }
    }
}
