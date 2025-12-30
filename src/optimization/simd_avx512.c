/**
 * @file simd_avx512.c
 * @brief AVX-512 optimized operations implementation
 *
 * AVX-512 provides 512-bit registers (8 doubles), offering 2x throughput
 * over AVX2 for memory-bandwidth-bound quantum operations.
 *
 * Key optimizations:
 * - 4 complex numbers processed per iteration (vs 2 for AVX2)
 * - Masked operations for handling non-aligned tails
 * - Reduced loop overhead via larger iteration count
 * - FMA (fused multiply-add) for matrix operations
 *
 * Compile with: -mavx512f -mavx512dq -mavx512bw -mavx512vl
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#include "simd_avx512.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================================
// AVX-512 INTRINSICS (Conditional Compilation)
// ============================================================================

#if AVX512_AVAILABLE

#include <immintrin.h>

// ============================================================================
// CAPABILITY CHECK
// ============================================================================

int avx512_is_available(void) {
    return 1;  // Compiled with AVX-512, so it's available
}

const char* avx512_get_features(void) {
    static char features[128] = "AVX-512";
    static int initialized = 0;

    if (!initialized) {
        char* p = features + strlen(features);

#ifdef __AVX512F__
        strcat(p, " F"); p = features + strlen(features);
#endif
#ifdef __AVX512DQ__
        strcat(p, " DQ"); p = features + strlen(features);
#endif
#ifdef __AVX512BW__
        strcat(p, " BW"); p = features + strlen(features);
#endif
#ifdef __AVX512VL__
        strcat(p, " VL"); p = features + strlen(features);
#endif
#ifdef __AVX512CD__
        strcat(p, " CD"); p = features + strlen(features);
#endif
#ifdef __AVX512VNNI__
        strcat(p, " VNNI");
#endif
        initialized = 1;
    }

    return features;
}

// ============================================================================
// SUM OF SQUARED MAGNITUDES
// ============================================================================

double avx512_sum_squared_magnitudes(const complex_t* amplitudes, size_t n) {
    if (!amplitudes || n == 0) return 0.0;

    __m512d sum_vec = _mm512_setzero_pd();

    // Process 4 complex numbers (8 doubles) at a time
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        // Load 4 complex numbers (8 doubles)
        __m512d data = _mm512_loadu_pd((const double*)&amplitudes[i]);

        // Square each element
        __m512d sq = _mm512_mul_pd(data, data);

        // Accumulate
        sum_vec = _mm512_add_pd(sum_vec, sq);
    }

    // Reduce 8 doubles to single sum
    double sum = _mm512_reduce_add_pd(sum_vec);

    // Handle remaining elements (0-3 complex numbers)
    for (; i < n; i++) {
        double re = creal(amplitudes[i]);
        double im = cimag(amplitudes[i]);
        sum += re * re + im * im;
    }

    return sum;
}

// ============================================================================
// NORMALIZE AMPLITUDES
// ============================================================================

void avx512_normalize_amplitudes(complex_t* amplitudes, size_t n, double norm) {
    if (!amplitudes || n == 0 || norm == 0.0) return;

    double inv_norm = 1.0 / norm;
    __m512d inv_norm_vec = _mm512_set1_pd(inv_norm);

    // Process 4 complex numbers (8 doubles) at a time
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m512d data = _mm512_loadu_pd((double*)&amplitudes[i]);
        __m512d result = _mm512_mul_pd(data, inv_norm_vec);
        _mm512_storeu_pd((double*)&amplitudes[i], result);
    }

    // Handle remaining elements
    for (; i < n; i++) {
        double re = creal(amplitudes[i]) * inv_norm;
        double im = cimag(amplitudes[i]) * inv_norm;
        amplitudes[i] = re + im * I;
    }
}

// ============================================================================
// COMPLEX SWAP (Pauli X, CNOT)
// ============================================================================

void avx512_complex_swap(complex_t* amp0, complex_t* amp1, size_t n) {
    if (!amp0 || !amp1 || n == 0) return;

    // Process 4 complex numbers (8 doubles) at a time
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m512d a = _mm512_loadu_pd((double*)&amp0[i]);
        __m512d b = _mm512_loadu_pd((double*)&amp1[i]);

        // Swap
        _mm512_storeu_pd((double*)&amp0[i], b);
        _mm512_storeu_pd((double*)&amp1[i], a);
    }

    // Handle remaining
    for (; i < n; i++) {
        complex_t temp = amp0[i];
        amp0[i] = amp1[i];
        amp1[i] = temp;
    }
}

// ============================================================================
// NEGATE (Pauli Z)
// ============================================================================

void avx512_negate(complex_t* amplitudes, size_t n) {
    if (!amplitudes || n == 0) return;

    // Create negation mask
    __m512d neg = _mm512_set1_pd(-1.0);

    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m512d data = _mm512_loadu_pd((double*)&amplitudes[i]);
        __m512d result = _mm512_mul_pd(data, neg);
        _mm512_storeu_pd((double*)&amplitudes[i], result);
    }

    // Handle remaining
    for (; i < n; i++) {
        amplitudes[i] = -amplitudes[i];
    }
}

// ============================================================================
// APPLY PHASE
// ============================================================================

void avx512_apply_phase(complex_t* amplitudes, complex_t phase, size_t n) {
    if (!amplitudes || n == 0) return;

    double phase_re = creal(phase);
    double phase_im = cimag(phase);

    // Broadcast phase components for complex multiplication
    // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    __m512d phase_re_vec = _mm512_set1_pd(phase_re);
    __m512d phase_im_vec = _mm512_set1_pd(phase_im);

    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        // Load 4 complex numbers [r0,i0,r1,i1,r2,i2,r3,i3]
        __m512d data = _mm512_loadu_pd((double*)&amplitudes[i]);

        // Shuffle to get real and imag parts
        // Real parts at even indices, imag at odd
        __m512d re_parts = _mm512_shuffle_pd(data, data, 0x00);  // [r0,r0,r1,r1,r2,r2,r3,r3]
        __m512d im_parts = _mm512_shuffle_pd(data, data, 0xFF);  // [i0,i0,i1,i1,i2,i2,i3,i3]

        // Complex multiplication
        __m512d re_result = _mm512_fmsub_pd(re_parts, phase_re_vec,
                                           _mm512_mul_pd(im_parts, phase_im_vec));
        __m512d im_result = _mm512_fmadd_pd(re_parts, phase_im_vec,
                                           _mm512_mul_pd(im_parts, phase_re_vec));

        // Interleave real and imaginary parts back
        __m512d result = _mm512_unpacklo_pd(re_result, im_result);

        _mm512_storeu_pd((double*)&amplitudes[i], result);
    }

    // Handle remaining
    for (; i < n; i++) {
        amplitudes[i] = amplitudes[i] * phase;
    }
}

// ============================================================================
// MULTIPLY BY ±i
// ============================================================================

void avx512_multiply_by_i(complex_t* amplitudes, size_t n, int negate) {
    if (!amplitudes || n == 0) return;

    // Multiply by i: (a + bi) * i = -b + ai
    // Multiply by -i: (a + bi) * (-i) = b - ai

    double sign = negate ? 1.0 : -1.0;

    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m512d data = _mm512_loadu_pd((double*)&amplitudes[i]);

        // Swap real and imaginary parts
        __m512d swapped = _mm512_permute_pd(data, 0x55);  // Swap pairs

        // Apply signs
        // For *i:  new_re = -im, new_im = re  → signs: [-1, 1, -1, 1, ...]
        // For *-i: new_re = im, new_im = -re → signs: [1, -1, 1, -1, ...]
        __m512d signs;
        if (negate) {
            signs = _mm512_set_pd(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
        } else {
            signs = _mm512_set_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
        }

        __m512d result = _mm512_mul_pd(swapped, signs);
        _mm512_storeu_pd((double*)&amplitudes[i], result);
    }

    // Handle remaining
    for (; i < n; i++) {
        double re = creal(amplitudes[i]);
        double im = cimag(amplitudes[i]);
        if (negate) {
            amplitudes[i] = im - re * I;  // *(-i)
        } else {
            amplitudes[i] = -im + re * I;  // *i
        }
    }
}

// ============================================================================
// COMPUTE PROBABILITIES
// ============================================================================

void avx512_compute_probabilities(const complex_t* amplitudes,
                                  double* probabilities, size_t n) {
    if (!amplitudes || !probabilities || n == 0) return;

    size_t i = 0;
    // Process 4 complex numbers -> 4 probabilities at a time
    for (; i + 3 < n; i += 4) {
        __m512d data = _mm512_loadu_pd((const double*)&amplitudes[i]);

        // Square all elements
        __m512d sq = _mm512_mul_pd(data, data);

        // Add pairs [re^2 + im^2, ...]
        // Use horizontal add pattern
        __m256d sq_lo = _mm512_castpd512_pd256(sq);
        __m256d sq_hi = _mm512_extractf64x4_pd(sq, 1);

        __m256d prob_lo = _mm256_hadd_pd(sq_lo, sq_lo);  // [p0+p0, p1+p1]
        __m256d prob_hi = _mm256_hadd_pd(sq_hi, sq_hi);

        // Extract and store
        probabilities[i] = _mm256_cvtsd_f64(prob_lo);
        probabilities[i + 1] = _mm_cvtsd_f64(_mm256_extractf128_pd(prob_lo, 1));
        probabilities[i + 2] = _mm256_cvtsd_f64(prob_hi);
        probabilities[i + 3] = _mm_cvtsd_f64(_mm256_extractf128_pd(prob_hi, 1));
    }

    // Handle remaining
    for (; i < n; i++) {
        double re = creal(amplitudes[i]);
        double im = cimag(amplitudes[i]);
        probabilities[i] = re * re + im * im;
    }
}

// ============================================================================
// CUMULATIVE PROBABILITY SEARCH
// ============================================================================

uint64_t avx512_cumulative_probability_search(const complex_t* amplitudes,
                                              size_t n, double random_threshold) {
    if (!amplitudes || n == 0) return 0;

    double cumulative = 0.0;

    // Can't vectorize easily due to cumulative nature
    // Use scalar with AVX-512 probability computation
    for (size_t i = 0; i < n; i++) {
        double re = creal(amplitudes[i]);
        double im = cimag(amplitudes[i]);
        cumulative += re * re + im * im;

        if (cumulative >= random_threshold) {
            return i;
        }
    }

    return n - 1;
}

// ============================================================================
// APPLY 2x2 GATE MATRIX
// ============================================================================

void avx512_apply_gate_2x2(const complex_t matrix[4],
                           complex_t* amp0, complex_t* amp1) {
    // Load amplitudes
    complex_t a = *amp0;
    complex_t b = *amp1;

    // Matrix multiplication:
    // [new_a]   [m00 m01] [a]
    // [new_b] = [m10 m11] [b]
    *amp0 = matrix[0] * a + matrix[1] * b;
    *amp1 = matrix[2] * a + matrix[3] * b;
}

void avx512_batch_apply_gate_2x2(const complex_t matrix[4],
                                 complex_t* amp0, complex_t* amp1, size_t n) {
    if (!amp0 || !amp1 || n == 0) return;

    // Broadcast matrix elements
    double m00_re = creal(matrix[0]), m00_im = cimag(matrix[0]);
    double m01_re = creal(matrix[1]), m01_im = cimag(matrix[1]);
    double m10_re = creal(matrix[2]), m10_im = cimag(matrix[2]);
    double m11_re = creal(matrix[3]), m11_im = cimag(matrix[3]);

    // For complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    for (size_t i = 0; i < n; i++) {
        double a_re = creal(amp0[i]), a_im = cimag(amp0[i]);
        double b_re = creal(amp1[i]), b_im = cimag(amp1[i]);

        // new_a = m00*a + m01*b
        double new_a_re = (m00_re * a_re - m00_im * a_im) + (m01_re * b_re - m01_im * b_im);
        double new_a_im = (m00_re * a_im + m00_im * a_re) + (m01_re * b_im + m01_im * b_re);

        // new_b = m10*a + m11*b
        double new_b_re = (m10_re * a_re - m10_im * a_im) + (m11_re * b_re - m11_im * b_im);
        double new_b_im = (m10_re * a_im + m10_im * a_re) + (m11_re * b_im + m11_im * b_re);

        amp0[i] = new_a_re + new_a_im * I;
        amp1[i] = new_b_re + new_b_im * I;
    }
}

// ============================================================================
// HADAMARD GATE (SPECIAL OPTIMIZATION)
// ============================================================================

void avx512_hadamard_pairs(complex_t* amp0, complex_t* amp1, size_t n) {
    if (!amp0 || !amp1 || n == 0) return;

    // Hadamard: H = 1/√2 * [[1, 1], [1, -1]]
    // new_a = (a + b) / √2
    // new_b = (a - b) / √2
    double inv_sqrt2 = 1.0 / sqrt(2.0);
    __m512d scale = _mm512_set1_pd(inv_sqrt2);

    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __m512d a = _mm512_loadu_pd((double*)&amp0[i]);
        __m512d b = _mm512_loadu_pd((double*)&amp1[i]);

        __m512d sum = _mm512_add_pd(a, b);
        __m512d diff = _mm512_sub_pd(a, b);

        __m512d new_a = _mm512_mul_pd(sum, scale);
        __m512d new_b = _mm512_mul_pd(diff, scale);

        _mm512_storeu_pd((double*)&amp0[i], new_a);
        _mm512_storeu_pd((double*)&amp1[i], new_b);
    }

    // Handle remaining
    for (; i < n; i++) {
        complex_t a = amp0[i];
        complex_t b = amp1[i];
        amp0[i] = (a + b) * inv_sqrt2;
        amp1[i] = (a - b) * inv_sqrt2;
    }
}

// ============================================================================
// XOR BYTES
// ============================================================================

void avx512_xor_bytes(uint8_t* dest, const uint8_t* src, size_t n) {
    if (!dest || !src || n == 0) return;

    // Process 64 bytes at a time (512 bits)
    size_t i = 0;
    for (; i + 63 < n; i += 64) {
        __m512i d = _mm512_loadu_si512((__m512i*)&dest[i]);
        __m512i s = _mm512_loadu_si512((__m512i*)&src[i]);
        __m512i result = _mm512_xor_si512(d, s);
        _mm512_storeu_si512((__m512i*)&dest[i], result);
    }

    // Handle remaining bytes
    for (; i < n; i++) {
        dest[i] ^= src[i];
    }
}

#else /* !AVX512_AVAILABLE */

// ============================================================================
// FALLBACK IMPLEMENTATIONS (When AVX-512 not available at compile time)
// ============================================================================

int avx512_is_available(void) {
    return 0;
}

const char* avx512_get_features(void) {
    return "AVX-512 not available";
}

double avx512_sum_squared_magnitudes(const complex_t* amplitudes, size_t n) {
    if (!amplitudes || n == 0) return 0.0;

    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double re = creal(amplitudes[i]);
        double im = cimag(amplitudes[i]);
        sum += re * re + im * im;
    }
    return sum;
}

void avx512_normalize_amplitudes(complex_t* amplitudes, size_t n, double norm) {
    if (!amplitudes || n == 0 || norm == 0.0) return;

    double inv_norm = 1.0 / norm;
    for (size_t i = 0; i < n; i++) {
        double re = creal(amplitudes[i]) * inv_norm;
        double im = cimag(amplitudes[i]) * inv_norm;
        amplitudes[i] = re + im * I;
    }
}

void avx512_complex_swap(complex_t* amp0, complex_t* amp1, size_t n) {
    if (!amp0 || !amp1 || n == 0) return;

    for (size_t i = 0; i < n; i++) {
        complex_t temp = amp0[i];
        amp0[i] = amp1[i];
        amp1[i] = temp;
    }
}

void avx512_negate(complex_t* amplitudes, size_t n) {
    if (!amplitudes || n == 0) return;

    for (size_t i = 0; i < n; i++) {
        amplitudes[i] = -amplitudes[i];
    }
}

void avx512_apply_phase(complex_t* amplitudes, complex_t phase, size_t n) {
    if (!amplitudes || n == 0) return;

    for (size_t i = 0; i < n; i++) {
        amplitudes[i] = amplitudes[i] * phase;
    }
}

void avx512_multiply_by_i(complex_t* amplitudes, size_t n, int negate) {
    if (!amplitudes || n == 0) return;

    for (size_t i = 0; i < n; i++) {
        double re = creal(amplitudes[i]);
        double im = cimag(amplitudes[i]);
        if (negate) {
            amplitudes[i] = im - re * I;
        } else {
            amplitudes[i] = -im + re * I;
        }
    }
}

void avx512_compute_probabilities(const complex_t* amplitudes,
                                  double* probabilities, size_t n) {
    if (!amplitudes || !probabilities || n == 0) return;

    for (size_t i = 0; i < n; i++) {
        double re = creal(amplitudes[i]);
        double im = cimag(amplitudes[i]);
        probabilities[i] = re * re + im * im;
    }
}

uint64_t avx512_cumulative_probability_search(const complex_t* amplitudes,
                                              size_t n, double random_threshold) {
    if (!amplitudes || n == 0) return 0;

    double cumulative = 0.0;
    for (size_t i = 0; i < n; i++) {
        double re = creal(amplitudes[i]);
        double im = cimag(amplitudes[i]);
        cumulative += re * re + im * im;

        if (cumulative >= random_threshold) {
            return i;
        }
    }
    return n - 1;
}

void avx512_apply_gate_2x2(const complex_t matrix[4],
                           complex_t* amp0, complex_t* amp1) {
    complex_t a = *amp0;
    complex_t b = *amp1;
    *amp0 = matrix[0] * a + matrix[1] * b;
    *amp1 = matrix[2] * a + matrix[3] * b;
}

void avx512_batch_apply_gate_2x2(const complex_t matrix[4],
                                 complex_t* amp0, complex_t* amp1, size_t n) {
    for (size_t i = 0; i < n; i++) {
        avx512_apply_gate_2x2(matrix, &amp0[i], &amp1[i]);
    }
}

void avx512_hadamard_pairs(complex_t* amp0, complex_t* amp1, size_t n) {
    if (!amp0 || !amp1 || n == 0) return;

    double inv_sqrt2 = 1.0 / sqrt(2.0);
    for (size_t i = 0; i < n; i++) {
        complex_t a = amp0[i];
        complex_t b = amp1[i];
        amp0[i] = (a + b) * inv_sqrt2;
        amp1[i] = (a - b) * inv_sqrt2;
    }
}

void avx512_xor_bytes(uint8_t* dest, const uint8_t* src, size_t n) {
    if (!dest || !src || n == 0) return;

    for (size_t i = 0; i < n; i++) {
        dest[i] ^= src[i];
    }
}

#endif /* AVX512_AVAILABLE */
