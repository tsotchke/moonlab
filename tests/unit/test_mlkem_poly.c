/**
 * @file test_mlkem_poly.c
 * @brief ML-KEM polynomial ring + NTT self-consistency tests.
 *
 * Anchor properties:
 *   - NTT -> inverse-NTT round-trip is the identity (up to coefficient
 *     canonicalisation).
 *   - For two polynomials a, b: inv_ntt(basemul(ntt(a), ntt(b))) equals
 *     the schoolbook polynomial product a * b  (mod X^n + 1, mod q).
 *   - poly_tobytes / poly_frombytes is identity.
 *   - CBD_2 samples coefficients in [-2, 2] with the expected mean (0)
 *     and approximate variance (1.0 for eta=2, 1.5 for eta=3).
 *   - compress + decompress on eta=2-sized noise is near-identity
 *     (rounding error bounded by ceil(q / 2^d / 2)).
 *   - mlkem_gen_matrix produces coefficients uniformly in [0, q).
 */

#include "../../src/crypto/mlkem/poly.h"
#include "../../src/crypto/sha3/sha3.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Q    MLKEM_Q
#define N    MLKEM_N

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                                  \
    if (!(cond)) {                                                  \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);        \
        failures++;                                                 \
    } else {                                                        \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);        \
    }                                                               \
} while (0)

/* Schoolbook multiplication in R_q = Z_q[X] / (X^n + 1). */
static void schoolbook_mul(mlkem_poly_t *dst,
                            const mlkem_poly_t *a, const mlkem_poly_t *b) {
    int32_t tmp[2 * N];
    memset(tmp, 0, sizeof tmp);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            tmp[i + j] += (int32_t)a->coeffs[i] * b->coeffs[j];
        }
    }
    /* Fold high coefficients via X^n = -1. */
    for (int i = 0; i < N; i++) {
        int32_t v = (tmp[i] - tmp[i + N]) % Q;
        if (v < 0) v += Q;
        dst->coeffs[i] = (int16_t)v;
    }
}

static void fill_random(mlkem_poly_t *p, int mod) {
    for (int i = 0; i < N; i++) {
        /* deterministic pseudo-random */
        p->coeffs[i] = (int16_t)(((i * 7919 + 31) % mod) - mod / 2);
    }
}

static void canonicalize(mlkem_poly_t *p) {
    for (int i = 0; i < N; i++) {
        int16_t v = p->coeffs[i] % Q;
        if (v < 0) v += Q;
        p->coeffs[i] = v;
    }
}

static void test_ntt_roundtrip(void) {
    fprintf(stdout, "\n-- NTT / inverse-NTT round-trip (pq-crystals Montgomery convention) --\n");
    /* In the pq-crystals reference convention, invntt's final multiply
     * by f = mont^2 / n leaves the output in Montgomery form (factor R
     * relative to the original), so bare ntt + invntt yields p * R mod q
     * rather than p.  The one-step correction to normal form is a final
     * Montgomery reduction with R = 2^16 mod q = 2285.  This test checks
     * the full identity: the NTT pipeline is self-inverse up to that
     * single well-known factor. */
    const int32_t R = 2285;
    mlkem_poly_t p, q;
    fill_random(&p, 8);
    canonicalize(&p);
    q = p;
    mlkem_poly_ntt(&q);
    mlkem_poly_invntt(&q);
    canonicalize(&q);
    int ok = 1;
    for (int i = 0; i < N; i++) {
        int32_t expected = ((int32_t)p.coeffs[i] * R) % Q;
        if (expected < 0) expected += Q;
        if (q.coeffs[i] != expected) {
            fprintf(stderr, "    i=%d  got %d  expected %d\n",
                    i, q.coeffs[i], expected);
            ok = 0; break;
        }
    }
    CHECK(ok, "NTT round-trip = p * R mod q (Montgomery)");
}

static void test_ntt_multiplication(void) {
    fprintf(stdout, "\n-- Schoolbook == inv(basemul(ntt(a), ntt(b))) --\n");
    mlkem_poly_t a, b, ref, via_ntt;
    fill_random(&a, 10);  canonicalize(&a);
    fill_random(&b, 8);   canonicalize(&b);
    /* b uses a different stride so they differ. */
    for (int i = 0; i < N; i++) b.coeffs[i] = (int16_t)((i * 131) % Q);

    schoolbook_mul(&ref, &a, &b);

    mlkem_poly_t an = a, bn = b;
    mlkem_poly_ntt(&an);
    mlkem_poly_ntt(&bn);
    mlkem_poly_basemul(&via_ntt, &an, &bn);
    mlkem_poly_invntt(&via_ntt);
    canonicalize(&via_ntt);

    int ok = 1;
    int first_bad = -1;
    for (int i = 0; i < N; i++) {
        if (ref.coeffs[i] != via_ntt.coeffs[i]) {
            ok = 0; first_bad = i;
            fprintf(stderr, "    diff at i=%d: ref=%d  via_ntt=%d\n",
                    i, ref.coeffs[i], via_ntt.coeffs[i]);
            break;
        }
    }
    CHECK(ok, "NTT-domain multiplication matches schoolbook (first diff: %d)",
          first_bad);
}

static void test_tobytes_frombytes(void) {
    fprintf(stdout, "\n-- poly_tobytes + poly_frombytes round-trip --\n");
    mlkem_poly_t p, q;
    for (int i = 0; i < N; i++) p.coeffs[i] = (int16_t)(i % Q);
    uint8_t buf[MLKEM_POLYBYTES];
    mlkem_poly_tobytes(buf, &p);
    mlkem_poly_frombytes(&q, buf);
    int ok = 1;
    for (int i = 0; i < N; i++) {
        if (p.coeffs[i] != q.coeffs[i]) { ok = 0; break; }
    }
    CHECK(ok, "byte round-trip preserves all 12-bit coeffs");
}

static void test_cbd_statistics(void) {
    fprintf(stdout, "\n-- CBD_2 and CBD_3 statistics --\n");
    /* Deterministic SHAKE expansion for a large sample. */
    uint8_t seed[32];
    for (int i = 0; i < 32; i++) seed[i] = (uint8_t)(i * 7 + 1);
    const size_t BIG = 4096;
    uint8_t *buf2 = malloc(BIG);
    uint8_t *buf3 = malloc(BIG);
    shake256(seed, 32, buf2, BIG);
    shake256(seed + 1, 31, buf3, BIG);  /* different stream */

    /* Accumulate many polynomials. */
    long long sum2 = 0, sum2_sq = 0; int count2 = 0;
    long long sum3 = 0, sum3_sq = 0; int count3 = 0;
    int min2 = 0, max2 = 0, min3 = 0, max3 = 0;
    for (int rep = 0; rep < 8; rep++) {
        mlkem_poly_t p;
        mlkem_poly_cbd(&p, buf2 + rep * 128, 2);
        for (int i = 0; i < N; i++) {
            int v = p.coeffs[i];
            sum2 += v;  sum2_sq += v * v;  count2++;
            if (v < min2) min2 = v;
            if (v > max2) max2 = v;
        }
        mlkem_poly_cbd(&p, buf3 + rep * 192, 3);
        for (int i = 0; i < N; i++) {
            int v = p.coeffs[i];
            sum3 += v;  sum3_sq += v * v;  count3++;
            if (v < min3) min3 = v;
            if (v > max3) max3 = v;
        }
    }
    double mean2 = (double)sum2 / count2;
    double var2  = (double)sum2_sq / count2 - mean2 * mean2;
    double mean3 = (double)sum3 / count3;
    double var3  = (double)sum3_sq / count3 - mean3 * mean3;
    fprintf(stdout, "    CBD_2: mean=%.4f var=%.4f range=[%d,%d]\n",
            mean2, var2, min2, max2);
    fprintf(stdout, "    CBD_3: mean=%.4f var=%.4f range=[%d,%d]\n",
            mean3, var3, min3, max3);
    CHECK(fabs(mean2) < 0.1, "CBD_2 mean near 0");
    CHECK(fabs(var2 - 1.0) < 0.15, "CBD_2 variance ~= 1.0");
    CHECK(min2 >= -2 && max2 <= 2, "CBD_2 in [-2, 2]");
    CHECK(fabs(mean3) < 0.1, "CBD_3 mean near 0");
    CHECK(fabs(var3 - 1.5) < 0.2, "CBD_3 variance ~= 1.5");
    CHECK(min3 >= -3 && max3 <= 3, "CBD_3 in [-3, 3]");
    free(buf2); free(buf3);
}

static void test_compress_decompress(void) {
    fprintf(stdout, "\n-- compress / decompress round-trip error bound --\n");
    mlkem_poly_t p, q;
    for (int i = 0; i < N; i++) p.coeffs[i] = (int16_t)(i % Q);

    /* d = 10: error bound = round(q / 2^11) = 2. */
    uint8_t buf[MLKEM512_POLYVEC_COMPRESSED];
    mlkem_poly_compress(buf, &p, 10);
    mlkem_poly_decompress(&q, buf, 10);
    int max_err = 0;
    for (int i = 0; i < N; i++) {
        int e = abs((int)p.coeffs[i] - (int)q.coeffs[i]);
        /* account for wrap-around: check the shorter of |d| and q - |d|. */
        if (e > Q - e) e = Q - e;
        if (e > max_err) max_err = e;
    }
    fprintf(stdout, "    d=10 max round-trip error = %d (bound 2)\n", max_err);
    CHECK(max_err <= 2, "d=10 compress round-trip within 2");

    /* d = 4: error bound = round(q / 2^5) = 104. */
    mlkem_poly_compress(buf, &p, 4);
    mlkem_poly_decompress(&q, buf, 4);
    max_err = 0;
    for (int i = 0; i < N; i++) {
        int e = abs((int)p.coeffs[i] - (int)q.coeffs[i]);
        if (e > Q - e) e = Q - e;
        if (e > max_err) max_err = e;
    }
    fprintf(stdout, "    d=4  max round-trip error = %d (bound 104)\n", max_err);
    CHECK(max_err <= 104, "d=4 compress round-trip within 104");
}

static void test_gen_matrix_coefficient_range(void) {
    fprintf(stdout, "\n-- gen_matrix coefficients in [0, q) --\n");
    uint8_t seed[32];
    for (int i = 0; i < 32; i++) seed[i] = (uint8_t)i;
    mlkem_poly_t A[MLKEM512_K * MLKEM512_K];
    mlkem_gen_matrix(A, MLKEM512_K, seed, 0);
    int all_in_range = 1;
    for (int i = 0; i < MLKEM512_K * MLKEM512_K; i++) {
        for (int j = 0; j < N; j++) {
            if (A[i].coeffs[j] < 0 || A[i].coeffs[j] >= Q) {
                all_in_range = 0; break;
            }
        }
    }
    CHECK(all_in_range, "matrix entries in [0, q)");
    /* Deterministic: the same seed gives identical output every time. */
    mlkem_poly_t A2[MLKEM512_K * MLKEM512_K];
    mlkem_gen_matrix(A2, MLKEM512_K, seed, 0);
    int deterministic = memcmp(A, A2, sizeof A) == 0;
    CHECK(deterministic, "gen_matrix is deterministic in the seed");
}

int main(void) {
    fprintf(stdout, "=== ML-KEM polynomial + NTT tests ===\n");
    test_ntt_roundtrip();
    test_ntt_multiplication();
    test_tobytes_frombytes();
    test_cbd_statistics();
    test_compress_decompress();
    test_gen_matrix_coefficient_range();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
