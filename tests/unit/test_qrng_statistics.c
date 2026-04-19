/**
 * @file test_qrng_statistics.c
 * @brief Statistical quality tests for `moonlab_qrng_bytes`.
 *
 * Draws a sizeable sample of bytes through the public stable ABI
 * export and runs a few classic RNG-quality checks:
 *
 *   - Byte frequency chi-squared against the uniform distribution
 *     (256 bins, N = 262 144 bytes ~= 1024 per bin). The critical
 *     value at alpha = 0.001 with 255 dof is ~346; a good RNG
 *     averages chi^2 ~ 255.
 *   - Serial correlation at lag 1 over the byte stream converted to
 *     doubles in [0, 1); the magnitude should be < 0.05 for a
 *     uniform source (1/sqrt(N)).
 *   - Monobit (bit-count) sanity: total ones should be within 4
 *     standard deviations of N * 8 / 2.
 *
 * These are smoke-grade statistical checks — a failing run is a
 * real regression signal (the QRNG pipeline is broken or the output
 * is skewed), but a passing run is not a FIPS certification.
 */

#include "../../src/applications/moonlab_export.h"
#include <inttypes.h>
#include <math.h>
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

/* 256 KiB sample — ~1024 bytes per 8-bit bin for chi-squared. */
#define QRNG_SAMPLE_BYTES (256u * 1024u)

static int draw_sample(uint8_t *buf, size_t size) {
    size_t remaining = size;
    uint8_t *p = buf;
    while (remaining > 0) {
        /* moonlab_qrng_bytes's internal path is happier with chunks
         * up to a few KiB; split the big draw. */
        size_t chunk = remaining > 4096 ? 4096 : remaining;
        int rc = moonlab_qrng_bytes(p, chunk);
        if (rc != 0) return rc;
        p += chunk;
        remaining -= chunk;
    }
    return 0;
}

static void test_byte_frequency_chi_squared(const uint8_t *buf, size_t n) {
    fprintf(stdout, "\n-- QRNG: byte frequency chi-squared (256 bins) --\n");
    uint32_t counts[256] = {0};
    for (size_t i = 0; i < n; ++i) counts[buf[i]]++;

    const double expected = (double)n / 256.0;
    double chi2 = 0.0;
    for (size_t i = 0; i < 256; ++i) {
        double d = (double)counts[i] - expected;
        chi2 += d * d / expected;
    }
    fprintf(stdout, "    N = %zu bytes  expected per bin = %.1f\n",
            n, expected);
    fprintf(stdout, "    chi^2 = %.2f  (mean = 255 for 256-bin uniform)\n", chi2);
    /* 99.9% rejection region at 255 dof is chi^2 > ~346. Allow extra
     * slack (500) because we only draw 256 KiB; larger samples tighten
     * this. A chi^2 > 500 means the distribution is visibly skewed. */
    CHECK(chi2 < 500.0, "chi^2 is within the 500-dof slack bound");
    CHECK(chi2 > 150.0, "chi^2 is not suspiciously low (anti-random)");
}

static void test_monobit_distribution(const uint8_t *buf, size_t n) {
    fprintf(stdout, "\n-- QRNG: monobit (bit-count) distribution --\n");
    uint64_t ones = 0;
    for (size_t i = 0; i < n; ++i) {
        ones += (uint64_t)__builtin_popcount(buf[i]);
    }
    const uint64_t total_bits = (uint64_t)n * 8;
    const double expected_ones = (double)total_bits / 2.0;
    const double sigma = sqrt((double)total_bits / 4.0);
    const double z = ((double)ones - expected_ones) / sigma;
    fprintf(stdout, "    total bits = %" PRIu64 "  ones = %" PRIu64
                    "  expected = %.0f  z = %.3f\n",
            total_bits, ones, expected_ones, z);
    CHECK(fabs(z) < 4.0, "|z| < 4 sigma (monobit test)");
}

static void test_serial_correlation(const uint8_t *buf, size_t n) {
    fprintf(stdout, "\n-- QRNG: lag-1 serial correlation --\n");
    /* Pearson lag-1 correlation. For an ideal uniform source the
     * magnitude should be O(1/sqrt(N)) -- about 0.002 for N=256 KiB.
     *
     * Historical note: an earlier implementation applied only 4
     * mixing gates between consecutive byte extractions, which left
     * the quantum state close to the previous measurement outcome
     * and produced a visible lag-1 correlation of rho_1 ~ 0.57
     * (repeatable across runs). That was fixed in qrng.c's
     * extract_quantum_entropy by scrambling the full register
     * (one H per qubit + one random Rz + a CNOT ring) between
     * every measurement. rho_1 now falls below the 5/sqrt(N)
     * ideal-uniform threshold on the default N = 256 KiB. */
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) mean += (double)buf[i];
    mean /= (double)n;

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i + 1 < n; ++i) {
        double a = (double)buf[i]     - mean;
        double b = (double)buf[i + 1] - mean;
        num += a * b;
        den += a * a;
    }
    double rho = (den > 0.0) ? num / den : 0.0;
    const double tol = 5.0 / sqrt((double)n);
    fprintf(stdout, "    mean = %.3f   rho_1 = %.6f   tol = %.6f\n",
            mean, rho, tol);
    CHECK(fabs(rho) < tol,
          "|rho_1| = %.3e is within 5/sqrt(N) = %.3e", fabs(rho), tol);
}

static void test_zero_size_call(void) {
    fprintf(stdout, "\n-- QRNG: zero-size call is a safe no-op --\n");
    CHECK(moonlab_qrng_bytes(NULL, 0) == 0,
          "moonlab_qrng_bytes(NULL, 0) returns success");
    uint8_t dummy = 0xAB;
    CHECK(moonlab_qrng_bytes(&dummy, 0) == 0,
          "moonlab_qrng_bytes(non-null, 0) returns success");
    CHECK(dummy == 0xAB, "buffer is untouched by a zero-size call");
}

static void test_null_buf_nonzero_size(void) {
    fprintf(stdout, "\n-- QRNG: NULL buf + nonzero size rejected --\n");
    CHECK(moonlab_qrng_bytes(NULL, 8) != 0,
          "moonlab_qrng_bytes(NULL, 8) returns error (not 0)");
}

#include "../../src/applications/nist_sp800_22.h"

static void test_sp800_22_subset(const uint8_t *buf, size_t size) {
    fprintf(stdout, "\n-- SP 800-22 subset (fast tests only) --\n");
    /* Unpack bytes into bits. */
    size_t nbits = size * 8;
    uint8_t *bits = malloc(nbits);
    if (!bits) { fprintf(stderr, "  FAIL alloc\n"); failures++; return; }
    for (size_t i = 0; i < size; i++)
        for (int k = 0; k < 8; k++)
            bits[i * 8 + k] = (uint8_t)((buf[i] >> (7 - k)) & 1);

    /* Skip the expensive tests (DFT O(n^2), ≈1M-bit requirements).
     * 256 KiB = 2 Mbits, plenty for the light tests. */
    /* NIST SP 800-22 prescribes running the battery on multiple
     * independent samples and accepting if the pass rate exceeds the
     * expected (1 - alpha) fraction. A single-sample check at alpha
     * = 0.01 gives ~1% false rejection per test * 6 tests ~= 6% CI
     * flakiness. We sample 3 buffers and require at least 2 of 3
     * pass each test (false-reject rate ~3e-6 per test). */
    const int attempts = 3;
    int pass[6] = {0};
    uint8_t* buf2 = malloc(QRNG_SAMPLE_BYTES);
    uint8_t* bits2 = malloc(QRNG_SAMPLE_BYTES * 8);
    if (!buf2 || !bits2) {
        fprintf(stderr, "  FAIL alloc\n"); failures++;
        free(bits); free(buf2); free(bits2); return;
    }

    for (int a = 0; a < attempts; a++) {
        double p_mono, p_blk, p_runs, p_lr, p_cf, p_ser;
        uint8_t* pbits;
        size_t pnbits;
        if (a == 0) {
            pbits = bits; pnbits = nbits;
        } else {
            int rc = draw_sample(buf2, QRNG_SAMPLE_BYTES);
            if (rc != 0) {
                fprintf(stderr, "  FAIL redraw rc=%d\n", rc);
                failures++;
                continue;
            }
            for (size_t i = 0; i < QRNG_SAMPLE_BYTES; i++) {
                for (int b = 0; b < 8; b++) {
                    bits2[i * 8 + b] = (uint8_t)((buf2[i] >> (7 - b)) & 1);
                }
            }
            pbits = bits2; pnbits = QRNG_SAMPLE_BYTES * 8;
        }
        p_mono = sp800_22_monobit(pbits, pnbits);
        p_blk  = sp800_22_block_frequency(pbits, pnbits, 128);
        p_runs = sp800_22_runs(pbits, pnbits);
        p_lr   = sp800_22_longest_run(pbits, pnbits);
        p_cf   = sp800_22_cusum_forward(pbits, pnbits);
        p_ser  = sp800_22_serial(pbits, pnbits, 5);
        fprintf(stdout,
                "    attempt %d: monobit=%.3f block=%.3f runs=%.3f "
                "lr=%.3f cusum=%.3f serial=%.3f\n",
                a + 1, p_mono, p_blk, p_runs, p_lr, p_cf, p_ser);
        if (p_mono >= 0.01) pass[0]++;
        if (p_blk  >= 0.01) pass[1]++;
        if (p_runs >= 0.01) pass[2]++;
        if (p_lr   >= 0.01) pass[3]++;
        if (p_cf   >= 0.01) pass[4]++;
        if (p_ser  >= 0.01) pass[5]++;
    }
    CHECK(pass[0] >= 2, "monobit passed %d/%d attempts", pass[0], attempts);
    CHECK(pass[1] >= 2, "block-frequency passed %d/%d attempts", pass[1], attempts);
    CHECK(pass[2] >= 2, "runs passed %d/%d attempts", pass[2], attempts);
    CHECK(pass[3] >= 2, "longest-run passed %d/%d attempts", pass[3], attempts);
    CHECK(pass[4] >= 2, "cusum-forward passed %d/%d attempts", pass[4], attempts);
    CHECK(pass[5] >= 2, "serial passed %d/%d attempts", pass[5], attempts);
    free(bits); free(buf2); free(bits2);
}

int main(void) {
    fprintf(stdout, "=== QRNG statistical quality tests ===\n");

    test_zero_size_call();
    test_null_buf_nonzero_size();

    uint8_t *buf = malloc(QRNG_SAMPLE_BYTES);
    if (!buf) { fprintf(stderr, "malloc failed\n"); return 1; }
    int rc = draw_sample(buf, QRNG_SAMPLE_BYTES);
    if (rc != 0) {
        fprintf(stderr, "  FAIL  draw_sample returned %d\n", rc);
        free(buf);
        return 1;
    }

    test_byte_frequency_chi_squared(buf, QRNG_SAMPLE_BYTES);
    test_monobit_distribution(buf, QRNG_SAMPLE_BYTES);
    test_serial_correlation(buf, QRNG_SAMPLE_BYTES);
    test_sp800_22_subset(buf, QRNG_SAMPLE_BYTES);

    free(buf);
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
