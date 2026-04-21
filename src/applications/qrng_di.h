/**
 * @file qrng_di.h
 * @brief Device-independent QRNG primitives (v0.2, Plan 1F).
 *
 * Where the existing `qrng_v3` BELL_VERIFIED mode uses a periodic
 * CHSH health-check to flag outright hardware failures, DI-QRNG
 * additionally *certifies* the min-entropy rate of its output as a
 * function of the observed CHSH violation, then feeds the raw
 * measurement bits through a strong randomness extractor (Toeplitz
 * hash) to distil that certified rate into near-uniform bits.
 *
 * Two protocol pieces live in this header; the full DI-QRNG pipeline
 * (epoch acceptance, loophole-closing tests, privacy amplification
 * parameter tuning) is built on top by qrng_di.c.
 *
 *  1. @ref qrng_di_min_entropy_from_chsh -- Pironio-style certified
 *     min-entropy H_min per measurement bit, derived from a running
 *     estimate of the CHSH parameter S. Quantitative reference:
 *     Pironio et al., Nature 464, 1021 (2010) (random numbers
 *     certified by Bell's theorem), equation (1) of the main text
 *     and the Methods-section tightening thereafter.
 *
 *  2. @ref qrng_di_toeplitz_extract -- Toeplitz-hash extractor.  Given
 *     @p n raw input bits with certified min-entropy k <= n and a
 *     uniformly-random @p seed of length n + m - 1 bits, produces m
 *     output bits within statistical distance 2^((m - k)/2) of uniform
 *     (this is the leftover-hash-lemma bound for a 2-universal family,
 *     instantiated by Toeplitz matrices).
 *
 * @since 0.2.0
 */

#ifndef MOONLAB_QRNG_DI_H
#define MOONLAB_QRNG_DI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Return the Pironio-certified min-entropy H_min (bits) per
 *        measurement bit for a given CHSH parameter S.
 *
 * Defined piecewise:
 *   S <= 2       :  H_min = 0  (no certified entropy; no Bell violation)
 *   S >= 2*sqrt(2):  H_min = 1  (maximally certified)
 *   2 < S < 2sqrt(2):  H_min = 1 - log2(1 + sqrt(2 - S*S/4))
 *     (this is the main-text bound from Pironio 2010)
 *
 * Any numerically-out-of-range @p chsh clamps to the corresponding
 * endpoint.  Return value is always in [0, 1].
 */
double qrng_di_min_entropy_from_chsh(double chsh);

/**
 * @brief Toeplitz-hash strong extractor.
 *
 * @param raw        n_in * 8 raw input bits, packed LSB-first.
 * @param n_in       number of raw input *bytes*.
 * @param seed       (n_in*8 + n_out*8 - 1) / 8 + 1 seed bytes; must be
 *                   fresh and uniformly random (this is public
 *                   random, i.e. seed-extractor).  Seed bytes also
 *                   packed LSB-first.
 * @param n_seed     bytes available in @p seed.  Must be at least
 *                   (n_in*8 + n_out*8 - 1 + 7) / 8.
 * @param out        n_out bytes of extracted output; written by the
 *                   function, caller owns.
 * @param n_out      number of output bytes to produce.
 *
 * @return 0 on success, -1 on argument error, -2 if seed is too short.
 *
 * Error semantics: the caller is responsible for ensuring that the
 * min-entropy of @p raw is at least 8 * n_out.  The extractor does
 * not compute this itself.  See @ref qrng_di_min_entropy_from_chsh.
 */
int qrng_di_toeplitz_extract(const uint8_t *raw, size_t n_in,
                              const uint8_t *seed, size_t n_seed,
                              uint8_t *out, size_t n_out);

/**
 * @brief Given an observed CHSH value and a target output length,
 *        return the number of raw input bytes required so that
 *        n_raw * 8 * H_min(chsh) >= 8 * n_out + epsilon, where
 *        @p epsilon bits are the desired statistical-distance margin
 *        (32 is standard; set 0 for the minimum).
 *
 * Returns 0 if the CHSH violation is insufficient (H_min <= 0) and
 * the target length is unachievable; otherwise a positive byte count
 * suitable for sizing the input buffer.
 */
size_t qrng_di_raw_bytes_for_output(double chsh,
                                     size_t n_out,
                                     size_t epsilon_bits);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_QRNG_DI_H */
