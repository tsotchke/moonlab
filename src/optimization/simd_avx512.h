/**
 * @file simd_avx512.h
 * @brief AVX-512 optimized operations for quantum state vectors
 *
 * Provides AVX-512 implementations of performance-critical operations.
 * AVX-512 processes 8 doubles (512 bits) per instruction, giving 2x
 * throughput improvement over AVX2 for memory-bound operations.
 *
 * Supported instruction sets:
 * - AVX-512F:  Foundation (required)
 * - AVX-512BW: Byte/Word operations
 * - AVX-512DQ: Doubleword/Quadword operations
 * - AVX-512VL: Vector Length extensions (128/256 bit)
 *
 * Requirements:
 * - Intel: Skylake-X, Ice Lake, Rocket Lake, Alder Lake P-cores
 * - AMD: Zen 4 (Ryzen 7000+, EPYC Genoa)
 *
 * Compile with: -mavx512f -mavx512dq -mavx512bw -mavx512vl
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef SIMD_AVX512_H
#define SIMD_AVX512_H

#include <stddef.h>
#include <stdint.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// Check for AVX-512 support
#if defined(__AVX512F__) && defined(__x86_64__)
    #define AVX512_AVAILABLE 1
#else
    #define AVX512_AVAILABLE 0
#endif

// Complex type
typedef double _Complex complex_t;

// ============================================================================
// AVX-512 CAPABILITY CHECK
// ============================================================================

/**
 * @brief Check if AVX-512 is available at runtime
 *
 * @return 1 if AVX-512F is supported, 0 otherwise
 */
int avx512_is_available(void);

/**
 * @brief Get AVX-512 feature string
 *
 * @return String describing available AVX-512 features
 */
const char* avx512_get_features(void);

// ============================================================================
// CORE OPERATIONS (8 doubles per iteration)
// ============================================================================

/**
 * @brief Sum of squared magnitudes using AVX-512
 *
 * Computes Σ|αᵢ|² for normalization. Processes 4 complex numbers
 * (8 doubles) per iteration.
 *
 * @param amplitudes Complex amplitude array
 * @param n          Number of amplitudes
 * @return Sum of squared magnitudes
 */
double avx512_sum_squared_magnitudes(const complex_t* amplitudes, size_t n);

/**
 * @brief Normalize amplitudes using AVX-512
 *
 * Divides all amplitudes by the given norm factor.
 *
 * @param amplitudes Complex amplitude array (modified in-place)
 * @param n          Number of amplitudes
 * @param norm       Normalization factor
 */
void avx512_normalize_amplitudes(complex_t* amplitudes, size_t n, double norm);

/**
 * @brief Complex swap for Pauli X gate using AVX-512
 *
 * Swaps pairs of complex amplitudes. Used for X and CNOT gates.
 *
 * @param amp0 First amplitude array
 * @param amp1 Second amplitude array
 * @param n    Number of pairs
 */
void avx512_complex_swap(complex_t* amp0, complex_t* amp1, size_t n);

/**
 * @brief Negate amplitudes for Pauli Z gate using AVX-512
 *
 * @param amplitudes Array to negate (modified in-place)
 * @param n          Number of amplitudes
 */
void avx512_negate(complex_t* amplitudes, size_t n);

/**
 * @brief Apply phase factor using AVX-512
 *
 * Multiplies amplitudes by e^(iθ). Used for S, T, and Phase gates.
 *
 * @param amplitudes Array to modify
 * @param phase      Complex phase factor
 * @param n          Number of amplitudes
 */
void avx512_apply_phase(complex_t* amplitudes, complex_t phase, size_t n);

/**
 * @brief Multiply by ±i using AVX-512
 *
 * Efficient rotation by 90° in complex plane.
 *
 * @param amplitudes Array to modify
 * @param n          Number of amplitudes
 * @param negate     If true, multiply by -i; otherwise +i
 */
void avx512_multiply_by_i(complex_t* amplitudes, size_t n, int negate);

// ============================================================================
// MEASUREMENT OPERATIONS
// ============================================================================

/**
 * @brief Compute probabilities from amplitudes using AVX-512
 *
 * @param amplitudes   Input complex amplitudes
 * @param probabilities Output probability array (|α|²)
 * @param n             Number of elements
 */
void avx512_compute_probabilities(const complex_t* amplitudes,
                                  double* probabilities, size_t n);

/**
 * @brief Cumulative probability search using AVX-512
 *
 * Finds the index where cumulative probability exceeds threshold.
 * Critical for measurement sampling.
 *
 * @param amplitudes        Complex amplitude array
 * @param n                 Number of amplitudes
 * @param random_threshold  Random value in [0,1)
 * @return Sampled index
 */
uint64_t avx512_cumulative_probability_search(const complex_t* amplitudes,
                                              size_t n, double random_threshold);

// ============================================================================
// MATRIX OPERATIONS
// ============================================================================

/**
 * @brief Apply 2x2 gate matrix using AVX-512
 *
 * Applies a 2x2 unitary matrix to amplitude pairs.
 *
 * @param matrix 2x2 complex gate matrix (row-major)
 * @param amp0   First amplitude in pair
 * @param amp1   Second amplitude in pair
 */
void avx512_apply_gate_2x2(const complex_t matrix[4],
                           complex_t* amp0, complex_t* amp1);

/**
 * @brief Batch apply 2x2 gate using AVX-512
 *
 * Applies the same 2x2 matrix to multiple amplitude pairs.
 *
 * @param matrix  2x2 complex gate matrix
 * @param amp0    Array of first amplitudes
 * @param amp1    Array of second amplitudes
 * @param n       Number of pairs
 */
void avx512_batch_apply_gate_2x2(const complex_t matrix[4],
                                 complex_t* amp0, complex_t* amp1, size_t n);

// ============================================================================
// HADAMARD GATE (SPECIAL OPTIMIZATION)
// ============================================================================

/**
 * @brief Apply Hadamard gate to multiple pairs using AVX-512
 *
 * Optimized implementation using the structure of Hadamard:
 * H = 1/√2 * [[1, 1], [1, -1]]
 *
 * @param amp0 Array of |0⟩-side amplitudes
 * @param amp1 Array of |1⟩-side amplitudes
 * @param n    Number of pairs
 */
void avx512_hadamard_pairs(complex_t* amp0, complex_t* amp1, size_t n);

// ============================================================================
// ENTROPY OPERATIONS
// ============================================================================

/**
 * @brief XOR byte arrays using AVX-512
 *
 * @param dest Destination array (modified: dest ^= src)
 * @param src  Source array
 * @param n    Number of bytes
 */
void avx512_xor_bytes(uint8_t* dest, const uint8_t* src, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* SIMD_AVX512_H */
