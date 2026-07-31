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
#include "moonlab_api.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Test 1: Frequency (Monobit).
 *
 * Maps each bit to +-1, sums the sequence, and refers the normalised
 * absolute sum @f$|S_n|/\sqrt{n}@f$ to the complementary error
 * function.  Detects a global bias towards zeros or ones.
 *
 * @param bits  One bit per byte, each 0 or 1.
 * @param nbits Sequence length.
 * @return p-value in [0, 1]; -1.0 if @p nbits < 100.
 * @stability evolving
 */
MOONLAB_API double sp800_22_monobit(const uint8_t *bits, size_t nbits);

/**
 * @brief Test 2: Frequency within a Block (M bits per block).
 *
 * Splits the sequence into @f$N = \lfloor n/M \rfloor@f$ disjoint
 * blocks, forms the chi-square statistic
 * @f$4M\sum_i (\pi_i - 1/2)^2@f$ over the per-block one-densities
 * @f$\pi_i@f$, and refers it to @f$\chi^2@f$ with @f$N@f$ degrees of
 * freedom.  Detects local bias that cancels globally.  Trailing bits
 * beyond @f$NM@f$ are discarded.
 *
 * @param bits  One bit per byte, each 0 or 1.
 * @param nbits Sequence length.
 * @param M     Block length in bits.
 * @return p-value in [0, 1]; -1.0 if @p nbits < 100, @p M < 20, or
 *         @p M > @p nbits.
 * @stability evolving
 */
MOONLAB_API double sp800_22_block_frequency(const uint8_t *bits, size_t nbits, size_t M);

/**
 * @brief Test 3: Runs.
 *
 * Counts the total number of maximal runs of identical bits and
 * compares it with the count expected for the observed one-density
 * @f$\pi@f$.  Detects oscillation that is too fast or too slow.  The
 * monobit prerequisite is applied first: if
 * @f$|\pi - 1/2| \ge 2/\sqrt{n}@f$ the sequence has already failed the
 * frequency precondition and the test reports 0.0.
 *
 * @param bits  One bit per byte, each 0 or 1.
 * @param nbits Sequence length.
 * @return p-value in [0, 1]; 0.0 when the monobit precondition fails;
 *         -1.0 if @p nbits < 100.
 * @stability evolving
 */
MOONLAB_API double sp800_22_runs(const uint8_t *bits, size_t nbits);

/**
 * @brief Test 4: Longest Run of Ones in a Block.
 *
 * Picks the block length from the input size -- @f$M = 10000@f$ for
 * @f$n \ge 750000@f$, @f$M = 128@f$ for @f$n \ge 6272@f$, and a
 * reduced @f$M = 8@f$ binning below that -- bins each block's longest
 * run of ones into the prescribed classes, and refers the chi-square
 * against the tabulated class probabilities to @f$\chi^2@f$ with
 * @f$K@f$ degrees of freedom.  The @f$M = 8@f$ path is a best-effort
 * extension below the SP 800-22 minimum and has reduced power.
 *
 * @param bits  One bit per byte, each 0 or 1.
 * @param nbits Sequence length.
 * @return p-value in [0, 1]; -1.0 if @p nbits < 128 or on allocation
 *         failure.
 * @stability evolving
 */
MOONLAB_API double sp800_22_longest_run(const uint8_t *bits, size_t nbits);

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

/**
 * @brief Test 11: Serial (m-bit overlapping patterns).
 *
 * Counts every overlapping @f$m@f$-bit pattern with the sequence
 * treated as circular, forms @f$\psi^2_m@f$ at widths @f$m@f$,
 * @f$m-1@f$ and @f$m-2@f$, and refers the first and second differences
 * @f$\nabla\psi^2_m@f$ and @f$\nabla^2\psi^2_m@f$ to @f$\chi^2@f$ with
 * @f$2^{m-1}@f$ and @f$2^{m-2}@f$ degrees of freedom.  Detects
 * non-uniformity of fixed-width patterns.
 *
 * @param bits  One bit per byte, each 0 or 1.
 * @param nbits Sequence length.
 * @param m     Pattern width in bits; must be >= 2.
 * @return The smaller (more conservative) of the two p-values, in
 *         [0, 1]; -1.0 if @p m < 2 or @p nbits < @f$2^{m+2}@f$.
 * @stability evolving
 */
MOONLAB_API double sp800_22_serial(const uint8_t *bits, size_t nbits, size_t m);

/* Test 12: Approximate Entropy */
double sp800_22_approximate_entropy(const uint8_t *bits, size_t nbits, size_t m);

/**
 * @brief Test 13: Cumulative Sums (forward mode).
 *
 * Walks the +-1 partial sums from the start of the sequence, takes the
 * maximum excursion @f$z = \max_k |S_k|@f$, and evaluates the exact
 * random-walk tail as the alternating pair of normal sums prescribed by
 * SP 800-22 §2.13.  Detects a drift that pushes the walk too far from
 * the origin (or holds it too close).  The result is clamped into
 * [0, 1] to absorb round-off in the truncated sums.
 *
 * @param bits  One bit per byte, each 0 or 1.
 * @param nbits Sequence length.
 * @return p-value in [0, 1]; -1.0 if @p nbits < 100.
 * @stability evolving
 */
MOONLAB_API double sp800_22_cusum_forward(const uint8_t *bits, size_t nbits);

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
