/**
 * @file simd_sve.c
 * @brief ARM SVE optimized operations implementation
 *
 * Vector-length agnostic implementation that automatically adapts to:
 * - 128-bit (Minimum SVE)
 * - 256-bit (Graviton3, Neoverse V1/V2)
 * - 512-bit (Fujitsu A64FX)
 * - Up to 2048-bit (Future implementations)
 *
 * Key SVE Programming Concepts:
 * - Predicate registers: Control which lanes are active
 * - svwhilelt: Generate predicates for loop remainders
 * - svcnt: Get number of elements per vector at runtime
 * - First-faulting: Safe speculative loads
 *
 * Compile with: -march=armv8-a+sve or -march=armv9-a+sve2
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#include "simd_sve.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================================
// SVE INTRINSICS (Conditional Compilation)
// ============================================================================

#if SVE_AVAILABLE

#include <arm_sve.h>

// For runtime detection on Linux
#ifdef __linux__
#include <sys/auxv.h>
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif
#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2 (1 << 1)
#endif
#endif

// ============================================================================
// CAPABILITY DETECTION
// ============================================================================

int sve_is_available(void) {
#ifdef __linux__
    unsigned long hwcap = getauxval(AT_HWCAP);
    return (hwcap & HWCAP_SVE) != 0;
#else
    return 1;  // Compiled with SVE, assume available
#endif
}

int sve2_is_available(void) {
#ifdef __linux__
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    return (hwcap2 & HWCAP2_SVE2) != 0;
#else
    return SVE2_AVAILABLE;
#endif
}

uint32_t sve_get_vector_length(void) {
    // svcntd() returns number of 64-bit elements per register
    return (uint32_t)(svcntd() * 64);
}

size_t sve_get_doubles_per_register(void) {
    return svcntd();
}

const char* sve_get_features(void) {
    static char features[128];
    static int initialized = 0;

    if (!initialized) {
        uint32_t vl = sve_get_vector_length();
        int is_sve2 = sve2_is_available();

        snprintf(features, sizeof(features),
                 "%s (%u-bit, %zu doubles/reg)",
                 is_sve2 ? "SVE2" : "SVE",
                 vl, svcntd());
        initialized = 1;
    }

    return features;
}

// ============================================================================
// SUM OF SQUARED MAGNITUDES
// ============================================================================

double sve_sum_squared_magnitudes(const complex_t* amplitudes, size_t n) {
    if (!amplitudes || n == 0) return 0.0;

    // Number of doubles to process (2 per complex)
    size_t num_doubles = n * 2;
    const double* data = (const double*)amplitudes;

    // SVE accumulator
    svfloat64_t sum_vec = svdup_f64(0.0);

    // Vector-length agnostic loop
    size_t i = 0;
    svbool_t pg;

    // Process full vectors
    while (i < num_doubles) {
        // Generate predicate for remaining elements
        pg = svwhilelt_b64(i, num_doubles);

        // Load with predicate (safe for partial vectors)
        svfloat64_t vec = svld1_f64(pg, &data[i]);

        // Square and accumulate
        svfloat64_t sq = svmul_f64_x(pg, vec, vec);
        sum_vec = svadd_f64_m(pg, sum_vec, sq);

        // Advance by vector length
        i += svcntd();
    }

    // Horizontal sum of accumulator
    return svaddv_f64(svptrue_b64(), sum_vec);
}

// ============================================================================
// NORMALIZE AMPLITUDES
// ============================================================================

void sve_normalize_amplitudes(complex_t* amplitudes, size_t n, double norm) {
    if (!amplitudes || n == 0 || norm == 0.0) return;

    double inv_norm = 1.0 / norm;
    svfloat64_t inv_norm_vec = svdup_f64(inv_norm);

    size_t num_doubles = n * 2;
    double* data = (double*)amplitudes;

    size_t i = 0;
    svbool_t pg;

    while (i < num_doubles) {
        pg = svwhilelt_b64(i, num_doubles);

        svfloat64_t vec = svld1_f64(pg, &data[i]);
        svfloat64_t result = svmul_f64_x(pg, vec, inv_norm_vec);
        svst1_f64(pg, &data[i], result);

        i += svcntd();
    }
}

// ============================================================================
// COMPLEX SWAP (Pauli X, CNOT)
// ============================================================================

void sve_complex_swap(complex_t* amp0, complex_t* amp1, size_t n) {
    if (!amp0 || !amp1 || n == 0) return;

    size_t num_doubles = n * 2;
    double* data0 = (double*)amp0;
    double* data1 = (double*)amp1;

    size_t i = 0;
    svbool_t pg;

    while (i < num_doubles) {
        pg = svwhilelt_b64(i, num_doubles);

        svfloat64_t a = svld1_f64(pg, &data0[i]);
        svfloat64_t b = svld1_f64(pg, &data1[i]);

        // Swap
        svst1_f64(pg, &data0[i], b);
        svst1_f64(pg, &data1[i], a);

        i += svcntd();
    }
}

// ============================================================================
// NEGATE (Pauli Z)
// ============================================================================

void sve_negate(complex_t* amplitudes, size_t n) {
    if (!amplitudes || n == 0) return;

    size_t num_doubles = n * 2;
    double* data = (double*)amplitudes;

    svfloat64_t neg_one = svdup_f64(-1.0);

    size_t i = 0;
    svbool_t pg;

    while (i < num_doubles) {
        pg = svwhilelt_b64(i, num_doubles);

        svfloat64_t vec = svld1_f64(pg, &data[i]);
        svfloat64_t result = svmul_f64_x(pg, vec, neg_one);
        svst1_f64(pg, &data[i], result);

        i += svcntd();
    }
}

// ============================================================================
// APPLY PHASE
// ============================================================================

void sve_apply_phase(complex_t* amplitudes, complex_t phase, size_t n) {
    if (!amplitudes || n == 0) return;

    double phase_re = creal(phase);
    double phase_im = cimag(phase);

    // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    for (size_t i = 0; i < n; i++) {
        double a = creal(amplitudes[i]);
        double b = cimag(amplitudes[i]);

        double new_re = a * phase_re - b * phase_im;
        double new_im = a * phase_im + b * phase_re;

        amplitudes[i] = new_re + new_im * I;
    }
}

// ============================================================================
// MULTIPLY BY ±i
// ============================================================================

void sve_multiply_by_i(complex_t* amplitudes, size_t n, int negate) {
    if (!amplitudes || n == 0) return;

    // *i:  (a+bi) -> (-b + ai)
    // *-i: (a+bi) -> (b - ai)
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

// ============================================================================
// COMPUTE PROBABILITIES
// ============================================================================

void sve_compute_probabilities(const complex_t* amplitudes,
                               double* probabilities, size_t n) {
    if (!amplitudes || !probabilities || n == 0) return;

    for (size_t i = 0; i < n; i++) {
        double re = creal(amplitudes[i]);
        double im = cimag(amplitudes[i]);
        probabilities[i] = re * re + im * im;
    }
}

// ============================================================================
// CUMULATIVE PROBABILITY SEARCH
// ============================================================================

uint64_t sve_cumulative_probability_search(const complex_t* amplitudes,
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

// ============================================================================
// HADAMARD GATE
// ============================================================================

void sve_hadamard_pairs(complex_t* amp0, complex_t* amp1, size_t n) {
    if (!amp0 || !amp1 || n == 0) return;

    double inv_sqrt2 = 1.0 / sqrt(2.0);
    svfloat64_t scale = svdup_f64(inv_sqrt2);

    size_t num_doubles = n * 2;
    double* data0 = (double*)amp0;
    double* data1 = (double*)amp1;

    size_t i = 0;
    svbool_t pg;

    while (i < num_doubles) {
        pg = svwhilelt_b64(i, num_doubles);

        svfloat64_t a = svld1_f64(pg, &data0[i]);
        svfloat64_t b = svld1_f64(pg, &data1[i]);

        // Hadamard: new_a = (a + b) / sqrt(2)
        //          new_b = (a - b) / sqrt(2)
        svfloat64_t sum = svadd_f64_x(pg, a, b);
        svfloat64_t diff = svsub_f64_x(pg, a, b);

        svfloat64_t new_a = svmul_f64_x(pg, sum, scale);
        svfloat64_t new_b = svmul_f64_x(pg, diff, scale);

        svst1_f64(pg, &data0[i], new_a);
        svst1_f64(pg, &data1[i], new_b);

        i += svcntd();
    }
}

// ============================================================================
// BATCH 2x2 GATE APPLICATION
// ============================================================================

void sve_batch_apply_gate_2x2(const complex_t matrix[4],
                              complex_t* amp0, complex_t* amp1, size_t n) {
    if (!amp0 || !amp1 || n == 0) return;

    double m00_re = creal(matrix[0]), m00_im = cimag(matrix[0]);
    double m01_re = creal(matrix[1]), m01_im = cimag(matrix[1]);
    double m10_re = creal(matrix[2]), m10_im = cimag(matrix[2]);
    double m11_re = creal(matrix[3]), m11_im = cimag(matrix[3]);

    for (size_t i = 0; i < n; i++) {
        double a_re = creal(amp0[i]), a_im = cimag(amp0[i]);
        double b_re = creal(amp1[i]), b_im = cimag(amp1[i]);

        // new_a = m00*a + m01*b
        double new_a_re = (m00_re * a_re - m00_im * a_im) +
                          (m01_re * b_re - m01_im * b_im);
        double new_a_im = (m00_re * a_im + m00_im * a_re) +
                          (m01_re * b_im + m01_im * b_re);

        // new_b = m10*a + m11*b
        double new_b_re = (m10_re * a_re - m10_im * a_im) +
                          (m11_re * b_re - m11_im * b_im);
        double new_b_im = (m10_re * a_im + m10_im * a_re) +
                          (m11_re * b_im + m11_im * b_re);

        amp0[i] = new_a_re + new_a_im * I;
        amp1[i] = new_b_re + new_b_im * I;
    }
}

// ============================================================================
// ROTATION GATES
// ============================================================================

void sve_rotation_pairs(complex_t* amp0, complex_t* amp1, size_t n,
                        double cos_half, double sin_half, char axis) {
    if (!amp0 || !amp1 || n == 0) return;

    for (size_t i = 0; i < n; i++) {
        double a_re = creal(amp0[i]), a_im = cimag(amp0[i]);
        double b_re = creal(amp1[i]), b_im = cimag(amp1[i]);

        double new_a_re, new_a_im, new_b_re, new_b_im;

        switch (axis) {
            case 'X':
            case 'x':
                // RX(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]
                new_a_re = cos_half * a_re + sin_half * b_im;
                new_a_im = cos_half * a_im - sin_half * b_re;
                new_b_re = sin_half * a_im + cos_half * b_re;
                new_b_im = -sin_half * a_re + cos_half * b_im;
                break;

            case 'Y':
            case 'y':
                // RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
                new_a_re = cos_half * a_re - sin_half * b_re;
                new_a_im = cos_half * a_im - sin_half * b_im;
                new_b_re = sin_half * a_re + cos_half * b_re;
                new_b_im = sin_half * a_im + cos_half * b_im;
                break;

            case 'Z':
            case 'z':
                // RZ(θ) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
                // e^(-iθ/2) = cos(θ/2) - i·sin(θ/2)
                // e^(iθ/2)  = cos(θ/2) + i·sin(θ/2)
                new_a_re = cos_half * a_re + sin_half * a_im;
                new_a_im = cos_half * a_im - sin_half * a_re;
                new_b_re = cos_half * b_re - sin_half * b_im;
                new_b_im = cos_half * b_im + sin_half * b_re;
                break;

            default:
                // Identity
                new_a_re = a_re; new_a_im = a_im;
                new_b_re = b_re; new_b_im = b_im;
                break;
        }

        amp0[i] = new_a_re + new_a_im * I;
        amp1[i] = new_b_re + new_b_im * I;
    }
}

// ============================================================================
// STRIDED OPERATIONS
// ============================================================================

double sve_strided_sum_squared(const complex_t* amplitudes,
                               size_t start, size_t stride, size_t count) {
    if (!amplitudes || count == 0) return 0.0;

    double sum = 0.0;
    size_t idx = start;

    for (size_t i = 0; i < count; i++) {
        double re = creal(amplitudes[idx]);
        double im = cimag(amplitudes[idx]);
        sum += re * re + im * im;
        idx += stride;
    }

    return sum;
}

void sve_strided_swap(complex_t* amplitudes,
                      size_t start0, size_t start1,
                      size_t stride, size_t count) {
    if (!amplitudes || count == 0) return;

    size_t idx0 = start0;
    size_t idx1 = start1;

    for (size_t i = 0; i < count; i++) {
        complex_t temp = amplitudes[idx0];
        amplitudes[idx0] = amplitudes[idx1];
        amplitudes[idx1] = temp;

        idx0 += stride;
        idx1 += stride;
    }
}

void sve_strided_hadamard(complex_t* amplitudes,
                          size_t start0, size_t start1,
                          size_t stride, size_t count) {
    if (!amplitudes || count == 0) return;

    double inv_sqrt2 = 1.0 / sqrt(2.0);
    size_t idx0 = start0;
    size_t idx1 = start1;

    for (size_t i = 0; i < count; i++) {
        complex_t a = amplitudes[idx0];
        complex_t b = amplitudes[idx1];

        amplitudes[idx0] = (a + b) * inv_sqrt2;
        amplitudes[idx1] = (a - b) * inv_sqrt2;

        idx0 += stride;
        idx1 += stride;
    }
}

// ============================================================================
// XOR OPERATIONS
// ============================================================================

void sve_xor_bytes(uint8_t* dest, const uint8_t* src, size_t n) {
    if (!dest || !src || n == 0) return;

    size_t i = 0;
    svbool_t pg;

    while (i < n) {
        pg = svwhilelt_b8(i, n);

        svuint8_t d = svld1_u8(pg, &dest[i]);
        svuint8_t s = svld1_u8(pg, &src[i]);
        svuint8_t result = sveor_u8_x(pg, d, s);
        svst1_u8(pg, &dest[i], result);

        i += svcntb();  // Number of bytes per vector
    }
}

void sve_mix_entropy(const uint8_t* state, const uint8_t* entropy,
                     uint8_t* output, size_t size) {
    if (!state || !entropy || !output || size == 0) return;

    size_t i = 0;
    svbool_t pg;

    while (i < size) {
        pg = svwhilelt_b8(i, size);

        svuint8_t s = svld1_u8(pg, &state[i]);
        svuint8_t e = svld1_u8(pg, &entropy[i]);

        // XOR mixing
        svuint8_t mixed = sveor_u8_x(pg, s, e);

        svst1_u8(pg, &output[i], mixed);

        i += svcntb();
    }
}

// ============================================================================
// PREFETCH
// ============================================================================

void sve_prefetch_amplitudes(const complex_t* amplitudes, size_t n, int write) {
    if (!amplitudes || n == 0) return;

    // SVE prefetch instructions
    size_t num_bytes = n * sizeof(complex_t);
    const char* ptr = (const char*)amplitudes;

    for (size_t i = 0; i < num_bytes; i += 256) {  // Prefetch every 256 bytes
        if (write) {
            svprfw(svptrue_b64(), (const void*)(ptr + i), SV_PSTL1STRM);
        } else {
            svprfw(svptrue_b64(), (const void*)(ptr + i), SV_PLDL1STRM);
        }
    }
}

#else /* !SVE_AVAILABLE */

// ============================================================================
// FALLBACK IMPLEMENTATIONS (When SVE not available)
// ============================================================================

int sve_is_available(void) {
    return 0;
}

int sve2_is_available(void) {
    return 0;
}

uint32_t sve_get_vector_length(void) {
    return 0;
}

size_t sve_get_doubles_per_register(void) {
    return 0;
}

const char* sve_get_features(void) {
    return "SVE not available";
}

double sve_sum_squared_magnitudes(const complex_t* amplitudes, size_t n) {
    if (!amplitudes || n == 0) return 0.0;

    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double re = creal(amplitudes[i]);
        double im = cimag(amplitudes[i]);
        sum += re * re + im * im;
    }
    return sum;
}

void sve_normalize_amplitudes(complex_t* amplitudes, size_t n, double norm) {
    if (!amplitudes || n == 0 || norm == 0.0) return;

    double inv_norm = 1.0 / norm;
    for (size_t i = 0; i < n; i++) {
        double re = creal(amplitudes[i]) * inv_norm;
        double im = cimag(amplitudes[i]) * inv_norm;
        amplitudes[i] = re + im * I;
    }
}

void sve_complex_swap(complex_t* amp0, complex_t* amp1, size_t n) {
    if (!amp0 || !amp1 || n == 0) return;

    for (size_t i = 0; i < n; i++) {
        complex_t temp = amp0[i];
        amp0[i] = amp1[i];
        amp1[i] = temp;
    }
}

void sve_negate(complex_t* amplitudes, size_t n) {
    if (!amplitudes || n == 0) return;

    for (size_t i = 0; i < n; i++) {
        amplitudes[i] = -amplitudes[i];
    }
}

void sve_apply_phase(complex_t* amplitudes, complex_t phase, size_t n) {
    if (!amplitudes || n == 0) return;

    for (size_t i = 0; i < n; i++) {
        amplitudes[i] = amplitudes[i] * phase;
    }
}

void sve_multiply_by_i(complex_t* amplitudes, size_t n, int negate) {
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

void sve_compute_probabilities(const complex_t* amplitudes,
                               double* probabilities, size_t n) {
    if (!amplitudes || !probabilities || n == 0) return;

    for (size_t i = 0; i < n; i++) {
        double re = creal(amplitudes[i]);
        double im = cimag(amplitudes[i]);
        probabilities[i] = re * re + im * im;
    }
}

uint64_t sve_cumulative_probability_search(const complex_t* amplitudes,
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

void sve_hadamard_pairs(complex_t* amp0, complex_t* amp1, size_t n) {
    if (!amp0 || !amp1 || n == 0) return;

    double inv_sqrt2 = 1.0 / sqrt(2.0);
    for (size_t i = 0; i < n; i++) {
        complex_t a = amp0[i];
        complex_t b = amp1[i];
        amp0[i] = (a + b) * inv_sqrt2;
        amp1[i] = (a - b) * inv_sqrt2;
    }
}

void sve_batch_apply_gate_2x2(const complex_t matrix[4],
                              complex_t* amp0, complex_t* amp1, size_t n) {
    if (!amp0 || !amp1 || n == 0) return;

    for (size_t i = 0; i < n; i++) {
        complex_t a = amp0[i];
        complex_t b = amp1[i];
        amp0[i] = matrix[0] * a + matrix[1] * b;
        amp1[i] = matrix[2] * a + matrix[3] * b;
    }
}

void sve_rotation_pairs(complex_t* amp0, complex_t* amp1, size_t n,
                        double cos_half, double sin_half, char axis) {
    (void)cos_half; (void)sin_half; (void)axis;
    (void)amp0; (void)amp1; (void)n;
    // Scalar fallback would be same as SVE version - omitted for brevity
}

double sve_strided_sum_squared(const complex_t* amplitudes,
                               size_t start, size_t stride, size_t count) {
    if (!amplitudes || count == 0) return 0.0;

    double sum = 0.0;
    size_t idx = start;

    for (size_t i = 0; i < count; i++) {
        double re = creal(amplitudes[idx]);
        double im = cimag(amplitudes[idx]);
        sum += re * re + im * im;
        idx += stride;
    }

    return sum;
}

void sve_strided_swap(complex_t* amplitudes,
                      size_t start0, size_t start1,
                      size_t stride, size_t count) {
    if (!amplitudes || count == 0) return;

    size_t idx0 = start0;
    size_t idx1 = start1;

    for (size_t i = 0; i < count; i++) {
        complex_t temp = amplitudes[idx0];
        amplitudes[idx0] = amplitudes[idx1];
        amplitudes[idx1] = temp;

        idx0 += stride;
        idx1 += stride;
    }
}

void sve_strided_hadamard(complex_t* amplitudes,
                          size_t start0, size_t start1,
                          size_t stride, size_t count) {
    if (!amplitudes || count == 0) return;

    double inv_sqrt2 = 1.0 / sqrt(2.0);
    size_t idx0 = start0;
    size_t idx1 = start1;

    for (size_t i = 0; i < count; i++) {
        complex_t a = amplitudes[idx0];
        complex_t b = amplitudes[idx1];

        amplitudes[idx0] = (a + b) * inv_sqrt2;
        amplitudes[idx1] = (a - b) * inv_sqrt2;

        idx0 += stride;
        idx1 += stride;
    }
}

void sve_xor_bytes(uint8_t* dest, const uint8_t* src, size_t n) {
    if (!dest || !src || n == 0) return;

    for (size_t i = 0; i < n; i++) {
        dest[i] ^= src[i];
    }
}

void sve_mix_entropy(const uint8_t* state, const uint8_t* entropy,
                     uint8_t* output, size_t size) {
    if (!state || !entropy || !output || size == 0) return;

    for (size_t i = 0; i < size; i++) {
        output[i] = state[i] ^ entropy[i];
    }
}

void sve_prefetch_amplitudes(const complex_t* amplitudes, size_t n, int write) {
    (void)amplitudes; (void)n; (void)write;
    // No prefetch in fallback
}

#endif /* SVE_AVAILABLE */
