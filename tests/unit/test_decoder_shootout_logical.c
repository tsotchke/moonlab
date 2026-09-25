/**
 * @file  test_decoder_shootout_logical.c
 * @brief Regression test for the decoder shoot-out's logical-failure
 *        check (benchmarks/decoder_shootout_toric.h).
 *
 * Covers:
 *   - Every plaquette (a pure-stabilizer residual) is NOT a failure.
 *   - Both nontrivial logical loops, their sum, and each of them times
 *     a plaquette ARE failures.
 *   - A residual that still has a syndrome is a failure.
 *   - MWPM_EXACT at p = 0.01 on the shoot-out's own error stream:
 *     d = 5 fails less often than d = 3 (below-threshold suppression).
 */

#include "../../src/applications/decoder_bench.h"
#include "../../benchmarks/decoder_shootout_toric.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

#define MAX_D 5

static void flip_plaquette(int d, unsigned char *r, int a, int b)
{
    r[h_idx(d, a, b)]           ^= 1;
    r[h_idx(d, a, (b + 1) % d)] ^= 1;
    r[v_idx(d, a, b)]           ^= 1;
    r[v_idx(d, (a + 1) % d, b)] ^= 1;
}

static void flip_loop_x1(int d, unsigned char *r, int y0)
{
    for (int x = 0; x < d; x++) r[h_idx(d, x, y0)] ^= 1;
}

static void flip_loop_x2(int d, unsigned char *r, int x0)
{
    for (int y = 0; y < d; y++) r[v_idx(d, x0, y)] ^= 1;
}

static int syndrome_weight(int d, const unsigned char *r)
{
    unsigned char syn[MAX_D * MAX_D];
    compute_syndrome(d, r, syn);
    int w = 0;
    for (int i = 0; i < d * d; i++) w += syn[i];
    return w;
}

static void test_stabilizers_are_not_failures(int d)
{
    fprintf(stdout, "\n--- d=%d: plaquette residuals ---\n", d);
    int miscounted = 0, with_syndrome = 0;
    unsigned char r[2 * MAX_D * MAX_D];
    for (int a = 0; a < d; a++) {
        for (int b = 0; b < d; b++) {
            memset(r, 0, sizeof r);
            flip_plaquette(d, r, a, b);
            if (syndrome_weight(d, r) != 0) with_syndrome++;
            if (logical_failure(d, r)) miscounted++;
        }
    }
    CHECK(with_syndrome == 0, "all %d plaquettes are syndrome-free", d * d);
    CHECK(miscounted == 0, "no plaquette counted as a logical failure (got %d of %d)",
          miscounted, d * d);

    /* A product of two plaquettes that straddles row y = 0. */
    memset(r, 0, sizeof r);
    flip_plaquette(d, r, 1, d - 1);
    flip_plaquette(d, r, 1, 0);
    CHECK(!logical_failure(d, r), "two-plaquette residual across y = 0 is not a failure");
}

static void test_logicals_are_failures(int d)
{
    fprintf(stdout, "\n--- d=%d: logical residuals ---\n", d);
    unsigned char r[2 * MAX_D * MAX_D];
    int missed = 0, total = 0;
    for (int k = 0; k < d; k++) {
        for (int which = 0; which < 3; which++) {
            memset(r, 0, sizeof r);
            if (which != 1) flip_loop_x1(d, r, k);
            if (which != 0) flip_loop_x2(d, r, (k + 2) % d);
            total++;
            if (syndrome_weight(d, r) != 0 || !logical_failure(d, r)) missed++;
            /* Same class, different representative. */
            flip_plaquette(d, r, (k + 1) % d, k);
            total++;
            if (syndrome_weight(d, r) != 0 || !logical_failure(d, r)) missed++;
        }
    }
    CHECK(missed == 0, "X_1, X_2 and X_1 X_2 loops (and loop x plaquette) all counted "
          "as failures (missed %d of %d)", missed, total);
}

static void test_detectable_residual_is_failure(int d)
{
    fprintf(stdout, "\n--- d=%d: residual with a syndrome ---\n", d);
    unsigned char r[2 * MAX_D * MAX_D];
    memset(r, 0, sizeof r);
    r[h_idx(d, 2, 2)] = 1;
    CHECK(syndrome_weight(d, r) == 2, "single-edge residual has two defects");
    CHECK(logical_failure(d, r), "single-edge residual is a failure");
}

/* Same sampler as decoder_shootout.c's estimate_rate. */
static uint64_t xorshift64(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return *s = x;
}

static int count_failures(int d, double p, int trials, uint64_t seed,
                          int *uncleared)
{
    const int n_q = 2 * d * d, n_s = d * d;
    unsigned char x_err[2 * MAX_D * MAX_D], corr[2 * MAX_D * MAX_D];
    unsigned char syn[MAX_D * MAX_D];
    const moonlab_decoder_code_t code = {
        .distance = d, .num_qubits = n_q, .is_toric = 1
    };
    int fails = 0;
    *uncleared = 0;
    for (int t = 0; t < trials; t++) {
        for (int q = 0; q < n_q; q++) {
            x_err[q] = ((double)(xorshift64(&seed) >> 11) /
                        (double)(1ULL << 53)) < p ? 1 : 0;
        }
        compute_syndrome(d, x_err, syn);
        memset(corr, 0, (size_t)n_q);
        const moonlab_decoder_input_t in = {
            .code = &code, .syndromes = syn, .corrections = corr,
            .num_stabilisers = n_s, .rng_seed = seed,
        };
        if (moonlab_decoder_decode(MOONLAB_DECODER_MWPM_EXACT, &in) != MOONLAB_DECODER_OK)
            return -1;
        for (int q = 0; q < n_q; q++) x_err[q] ^= corr[q];
        if (syndrome_weight(d, x_err) != 0) (*uncleared)++;
        if (logical_failure(d, x_err)) fails++;
    }
    return fails;
}

static void test_distance_suppression(void)
{
    fprintf(stdout, "\n--- MWPM_EXACT, p = 0.01: d=5 below d=3 ---\n");
    const int trials = 20000;
    int unc3 = 0, unc5 = 0;
    const int f3 = count_failures(3, 0.01, trials, 0xdeadbeefULL, &unc3);
    const int f5 = count_failures(5, 0.01, trials, 0xdeadbeefULL, &unc5);
    fprintf(stdout, "    d=3: %d / %d    d=5: %d / %d\n", f3, trials, f5, trials);
    CHECK(unc3 == 0 && unc5 == 0, "every MWPM_EXACT correction clears its syndrome");
    CHECK(f3 > 0 && f5 >= 0, "decode calls succeeded");
    /* Certified rates at this p are 2.3e-3 (d=3) and 2.3e-4 (d=5);
     * the old check read 2.2e-3 and 2.6e-3. */
    CHECK(3 * f5 < f3, "d=5 suppresses d=3 at least 3x (d=3 %d, d=5 %d)", f3, f5);
}

int main(void)
{
    for (int d = 3; d <= MAX_D; d += 2) {
        test_stabilizers_are_not_failures(d);
        test_logicals_are_failures(d);
        test_detectable_residual_is_failure(d);
    }
    test_distance_suppression();

    fprintf(stdout, "\n=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
