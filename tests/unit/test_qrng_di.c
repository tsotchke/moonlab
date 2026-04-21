/**
 * @file test_qrng_di.c
 * @brief DI-QRNG primitives: Pironio bound + Toeplitz extractor.
 */

#include "../../src/applications/qrng_di.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                                  \
    if (!(cond)) {                                                  \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);        \
        failures++;                                                 \
    } else {                                                        \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);        \
    }                                                               \
} while (0)

static void test_min_entropy_monotone(void) {
    fprintf(stdout, "\n-- H_min(S) is monotone increasing on [2, 2*sqrt(2)] --\n");
    double prev = -1.0;
    const int N = 20;
    for (int i = 0; i <= N; i++) {
        double S = 2.0 + (2.0 * sqrt(2.0) - 2.0) * (double)i / (double)N;
        double h = qrng_di_min_entropy_from_chsh(S);
        fprintf(stdout, "    S=%.4f  H_min=%.4f\n", S, h);
        CHECK(h + 1e-9 >= prev, "H_min non-decreasing");
        CHECK(h >= 0.0 && h <= 1.0, "H_min in [0,1]");
        prev = h;
    }
    CHECK(fabs(qrng_di_min_entropy_from_chsh(2.0)) < 1e-12,
          "H_min(S = 2) = 0");
    CHECK(fabs(qrng_di_min_entropy_from_chsh(2.0 * sqrt(2.0)) - 1.0) < 1e-9,
          "H_min(Tsirelson) = 1");
    /* Spot check: at S = 2.6 (comfortably above classical), Pironio
     * formula gives H_min = 1 - log2(1 + sqrt(2 - 6.76/4)) =
     * 1 - log2(1 + sqrt(0.31)) = 1 - log2(1.5568) ~= 0.36 bits. */
    double h_26 = qrng_di_min_entropy_from_chsh(2.6);
    CHECK(fabs(h_26 - 0.36) < 0.02,
          "H_min(S=2.6) ~= 0.36 (got %.4f)", h_26);
}

static void test_toeplitz_basic(void) {
    fprintf(stdout, "\n-- Toeplitz extractor: basic sanity --\n");
    const size_t n_in = 32, n_out = 8;
    uint8_t raw[32], out[8];
    uint8_t seed[(32 + 8) * 8];   /* plenty */
    for (size_t i = 0; i < n_in; i++)   raw[i]  = (uint8_t)(i * 17 + 3);
    for (size_t i = 0; i < sizeof seed; i++) seed[i] = (uint8_t)(i * 29 + 7);

    int rc = qrng_di_toeplitz_extract(raw, n_in, seed, sizeof seed,
                                       out, n_out);
    CHECK(rc == 0, "toeplitz_extract returns success");

    /* Linearity: T*(x1 xor x2) = T*x1 xor T*x2. */
    uint8_t raw2[32], out2[8], sum_raw[32], sum_out[8];
    for (size_t i = 0; i < n_in; i++)  raw2[i] = (uint8_t)(i * 59 + 11);
    for (size_t i = 0; i < n_in; i++)  sum_raw[i] = raw[i] ^ raw2[i];
    qrng_di_toeplitz_extract(raw2, n_in, seed, sizeof seed, out2, n_out);
    qrng_di_toeplitz_extract(sum_raw, n_in, seed, sizeof seed, sum_out, n_out);
    int linear = 1;
    for (size_t i = 0; i < n_out; i++) {
        if ((uint8_t)(out[i] ^ out2[i]) != sum_out[i]) { linear = 0; break; }
    }
    CHECK(linear, "Toeplitz hash is linear in the raw input");
}

static void test_toeplitz_seed_too_short(void) {
    fprintf(stdout, "\n-- Toeplitz: short seed rejected --\n");
    uint8_t raw[16] = {0}, out[8], seed[2] = {0};
    int rc = qrng_di_toeplitz_extract(raw, sizeof raw, seed, sizeof seed,
                                       out, sizeof out);
    CHECK(rc == -2, "short seed returns -2");
}

static void test_raw_bytes_sizing(void) {
    fprintf(stdout, "\n-- Raw-byte sizing tracks H_min --\n");
    /* At maximal violation H_min = 1, n_raw approx n_out + eps/8. */
    size_t n_raw_max = qrng_di_raw_bytes_for_output(2.0 * sqrt(2.0), 32, 0);
    CHECK(n_raw_max == 32, "Tsirelson -> n_raw == n_out (got %zu)", n_raw_max);
    /* At no violation we get 0 (no certified entropy). */
    size_t n_raw_zero = qrng_di_raw_bytes_for_output(2.0, 32, 0);
    CHECK(n_raw_zero == 0, "No violation -> n_raw = 0 (got %zu)", n_raw_zero);
    /* At intermediate S the required input grows.  S=2.6 -> H~0.36 ->
     * n_raw approx n_out / H = 32/0.36 ~= 88.8 -> 89 bytes. */
    size_t n_raw_26 = qrng_di_raw_bytes_for_output(2.6, 32, 0);
    CHECK(n_raw_26 >= 88 && n_raw_26 <= 100,
          "S=2.6 -> n_raw ~= 89 (got %zu)", n_raw_26);
}

int main(void) {
    fprintf(stdout, "=== QRNG DI-primitive tests ===\n");
    test_min_entropy_monotone();
    test_toeplitz_basic();
    test_toeplitz_seed_too_short();
    test_raw_bytes_sizing();
    fprintf(stdout, "\n=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
