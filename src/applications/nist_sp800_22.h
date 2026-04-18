/**
 * @file nist_sp800_22.h
 * @brief NIST SP 800-22 rev 1a statistical test suite for RNGs.
 *
 * Every test returns a p-value in [0, 1]. Pass at the standard
 * alpha = 0.01 threshold (p >= 0.01). Output values of -1.0 mean the
 * input size is too small for the test or the implementation is a
 * minimal best-effort form — see per-test docstrings.
 *
 * @since v0.1.3
 * @stability evolving
 */

#ifndef MOONLAB_NIST_SP800_22_H
#define MOONLAB_NIST_SP800_22_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Test 1: Frequency (Monobit) */
double sp800_22_monobit(const uint8_t *bits, size_t nbits);

/* Test 2: Frequency within a Block (M bits per block) */
double sp800_22_block_frequency(const uint8_t *bits, size_t nbits, size_t M);

/* Test 3: Runs */
double sp800_22_runs(const uint8_t *bits, size_t nbits);

/* Test 4: Longest Run of Ones in a Block */
double sp800_22_longest_run(const uint8_t *bits, size_t nbits);

/* Test 5: Binary Matrix Rank (32x32 matrices per block) */
double sp800_22_rank(const uint8_t *bits, size_t nbits);

/* Test 6: Discrete Fourier Transform (Spectral) */
double sp800_22_dft(const uint8_t *bits, size_t nbits);

/* Test 7: Non-overlapping Template Matching (m=9 default pattern) */
double sp800_22_non_overlapping_template(const uint8_t *bits, size_t nbits);

/* Test 8: Overlapping Template Matching (m=9 default) */
double sp800_22_overlapping_template(const uint8_t *bits, size_t nbits);

/* Test 9: Maurer's Universal Statistical Test */
double sp800_22_universal(const uint8_t *bits, size_t nbits);

/* Test 10: Linear Complexity */
double sp800_22_linear_complexity(const uint8_t *bits, size_t nbits, size_t M);

/* Test 11: Serial (m-bit overlapping patterns) */
double sp800_22_serial(const uint8_t *bits, size_t nbits, size_t m);

/* Test 12: Approximate Entropy */
double sp800_22_approximate_entropy(const uint8_t *bits, size_t nbits, size_t m);

/* Test 13: Cumulative Sums (forward) */
double sp800_22_cusum_forward(const uint8_t *bits, size_t nbits);

/* Test 14: Cumulative Sums (reverse) */
double sp800_22_cusum_reverse(const uint8_t *bits, size_t nbits);

/* Test 15: Random Excursions (state x=+1). Returns a single representative
 *          p-value across the 8 states [-4..-1, 1..4]; the minimum over the
 *          battery is a conservative summary. */
double sp800_22_random_excursions(const uint8_t *bits, size_t nbits);

/* Test 16 (technically 15 with variant): Random Excursions Variant. */
double sp800_22_random_excursions_variant(const uint8_t *bits, size_t nbits);

/**
 * @brief Summary of the full battery.
 *
 * Each p-value in out_pvalues[15] corresponds to tests 1..15 in order
 * (counting CumulativeSums as one combined entry by taking min of
 * forward/reverse, to match the conventional "15 tests" count).
 *
 * @return Number of tests that passed at alpha = 0.01 (out of 15).
 *         -1 if nbits is below the minimum input size for the battery
 *         (recommended >= 10^6 bits; we accept >= 10^5 with degraded
 *         power).
 */
int sp800_22_run_all(const uint8_t *bits, size_t nbits, double out_pvalues[15]);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_NIST_SP800_22_H */
