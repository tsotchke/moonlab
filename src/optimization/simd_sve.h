/**
 * @file simd_sve.h
 * @brief ARM SVE (Scalable Vector Extensions) optimized operations
 *
 * Provides vector-length-agnostic SIMD operations for ARM SVE processors.
 * SVE allows the same binary to run optimally on different hardware:
 *
 * Supported Platforms:
 * - AWS Graviton3/Graviton4: 256-bit vectors (4 doubles per register)
 * - Fujitsu A64FX:           512-bit vectors (8 doubles per register)
 * - ARM Neoverse V1/V2:      256-bit vectors
 * - ARM Neoverse V3:         128/256-bit vectors
 *
 * Key Advantages:
 * - Vector-length agnostic: Same code works on all SVE implementations
 * - Predicate registers: Efficient handling of loop tails
 * - Gather/scatter: Efficient strided access patterns
 * - First-faulting loads: Safe speculative execution
 *
 * Compile with: -march=armv8-a+sve or -march=armv9-a+sve2
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef SIMD_SVE_H
#define SIMD_SVE_H

#include <stddef.h>
#include <stdint.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// Check for SVE support
#if defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    #define SVE_AVAILABLE 1
#else
    #define SVE_AVAILABLE 0
#endif

// Check for SVE2 support
#if defined(__ARM_FEATURE_SVE2) && defined(__aarch64__)
    #define SVE2_AVAILABLE 1
#else
    #define SVE2_AVAILABLE 0
#endif

// Complex type
typedef double _Complex complex_t;

// ============================================================================
// SVE CAPABILITY DETECTION
// ============================================================================

/**
 * @brief Check if SVE is available at runtime
 *
 * Uses getauxval(AT_HWCAP) to detect SVE support.
 *
 * @return 1 if SVE is supported, 0 otherwise
 */
int sve_is_available(void);

/**
 * @brief Check if SVE2 is available at runtime
 *
 * @return 1 if SVE2 is supported, 0 otherwise
 */
int sve2_is_available(void);

/**
 * @brief Get SVE vector length in bits
 *
 * Returns the hardware vector length, which varies by implementation:
 * - 128 bits (2 doubles) - minimum
 * - 256 bits (4 doubles) - Graviton3, Neoverse V1
 * - 512 bits (8 doubles) - A64FX
 * - Up to 2048 bits (32 doubles) - theoretical maximum
 *
 * @return Vector length in bits
 */
uint32_t sve_get_vector_length(void);

/**
 * @brief Get number of doubles per SVE register
 *
 * @return Number of doubles that fit in one SVE register
 */
size_t sve_get_doubles_per_register(void);

/**
 * @brief Get SVE feature string
 *
 * @return String describing SVE capabilities
 */
const char* sve_get_features(void);

// ============================================================================
// CORE OPERATIONS (Vector-Length Agnostic)
// ============================================================================

/**
 * @brief Sum of squared magnitudes using SVE
 *
 * Computes Σ|αᵢ|² using vector-length agnostic loops.
 * Automatically adapts to available vector length.
 *
 * @param amplitudes Complex amplitude array
 * @param n          Number of amplitudes
 * @return Sum of squared magnitudes
 */
double sve_sum_squared_magnitudes(const complex_t* amplitudes, size_t n);

/**
 * @brief Normalize amplitudes using SVE
 *
 * @param amplitudes Complex amplitude array (modified in-place)
 * @param n          Number of amplitudes
 * @param norm       Normalization factor (sqrt of sum squared)
 */
void sve_normalize_amplitudes(complex_t* amplitudes, size_t n, double norm);

/**
 * @brief Complex swap for Pauli X gate using SVE
 *
 * @param amp0 First amplitude array
 * @param amp1 Second amplitude array
 * @param n    Number of pairs
 */
void sve_complex_swap(complex_t* amp0, complex_t* amp1, size_t n);

/**
 * @brief Negate amplitudes for Pauli Z gate using SVE
 *
 * @param amplitudes Array to negate
 * @param n          Number of amplitudes
 */
void sve_negate(complex_t* amplitudes, size_t n);

/**
 * @brief Apply phase factor using SVE
 *
 * @param amplitudes Array to modify
 * @param phase      Complex phase factor e^(iθ)
 * @param n          Number of amplitudes
 */
void sve_apply_phase(complex_t* amplitudes, complex_t phase, size_t n);

/**
 * @brief Multiply by ±i using SVE
 *
 * @param amplitudes Array to modify
 * @param n          Number of amplitudes
 * @param negate     If true, multiply by -i; otherwise +i
 */
void sve_multiply_by_i(complex_t* amplitudes, size_t n, int negate);

// ============================================================================
// MEASUREMENT OPERATIONS
// ============================================================================

/**
 * @brief Compute probabilities from amplitudes using SVE
 *
 * @param amplitudes   Input complex amplitudes
 * @param probabilities Output probability array
 * @param n            Number of elements
 */
void sve_compute_probabilities(const complex_t* amplitudes,
                               double* probabilities, size_t n);

/**
 * @brief Cumulative probability search using SVE
 *
 * @param amplitudes       Complex amplitude array
 * @param n                Number of amplitudes
 * @param random_threshold Random value in [0,1)
 * @return Sampled index
 */
uint64_t sve_cumulative_probability_search(const complex_t* amplitudes,
                                           size_t n, double random_threshold);

// ============================================================================
// GATE OPERATIONS
// ============================================================================

/**
 * @brief Apply Hadamard to amplitude pairs using SVE
 *
 * Optimized for the Hadamard gate structure:
 * H = 1/√2 * [[1, 1], [1, -1]]
 *
 * @param amp0 Array of |0⟩-side amplitudes
 * @param amp1 Array of |1⟩-side amplitudes
 * @param n    Number of pairs
 */
void sve_hadamard_pairs(complex_t* amp0, complex_t* amp1, size_t n);

/**
 * @brief Batch apply 2x2 gate matrix using SVE
 *
 * @param matrix 2x2 complex gate matrix (row-major)
 * @param amp0   Array of first amplitudes in pairs
 * @param amp1   Array of second amplitudes in pairs
 * @param n      Number of pairs
 */
void sve_batch_apply_gate_2x2(const complex_t matrix[4],
                              complex_t* amp0, complex_t* amp1, size_t n);

/**
 * @brief Apply rotation gate (RX, RY, RZ) to pairs using SVE
 *
 * Generic rotation around axis by angle theta.
 *
 * @param amp0    First amplitudes
 * @param amp1    Second amplitudes
 * @param n       Number of pairs
 * @param cos_half cos(theta/2)
 * @param sin_half sin(theta/2)
 * @param axis    Rotation axis: 'X', 'Y', or 'Z'
 */
void sve_rotation_pairs(complex_t* amp0, complex_t* amp1, size_t n,
                        double cos_half, double sin_half, char axis);

// ============================================================================
// STRIDED ACCESS (Optimal for Quantum State Layout)
// ============================================================================

/**
 * @brief Strided sum of squared magnitudes
 *
 * Computes sum for elements at indices: start, start+stride, start+2*stride, ...
 * Uses SVE gather loads for efficient strided access.
 *
 * @param amplitudes Base array
 * @param start      Starting index
 * @param stride     Stride between elements
 * @param count      Number of elements to sum
 * @return Sum of squared magnitudes
 */
double sve_strided_sum_squared(const complex_t* amplitudes,
                               size_t start, size_t stride, size_t count);

/**
 * @brief Strided swap operation
 *
 * Swaps elements at strided positions. Critical for two-qubit gates.
 *
 * @param amplitudes Array to modify
 * @param start0     Start index for first set
 * @param start1     Start index for second set
 * @param stride     Stride between consecutive pairs
 * @param count      Number of pairs to swap
 */
void sve_strided_swap(complex_t* amplitudes,
                      size_t start0, size_t start1,
                      size_t stride, size_t count);

/**
 * @brief Strided Hadamard application
 *
 * Applies Hadamard gate to amplitude pairs at strided positions.
 *
 * @param amplitudes Array to modify
 * @param start0     Start index for |0⟩ amplitudes
 * @param start1     Start index for |1⟩ amplitudes
 * @param stride     Stride between consecutive pairs
 * @param count      Number of pairs
 */
void sve_strided_hadamard(complex_t* amplitudes,
                          size_t start0, size_t start1,
                          size_t stride, size_t count);

// ============================================================================
// ENTROPY OPERATIONS
// ============================================================================

/**
 * @brief XOR byte arrays using SVE
 *
 * @param dest Destination (modified: dest ^= src)
 * @param src  Source array
 * @param n    Number of bytes
 */
void sve_xor_bytes(uint8_t* dest, const uint8_t* src, size_t n);

/**
 * @brief Fast entropy mixing using SVE
 *
 * Mixes multiple entropy sources using XOR and rotation.
 *
 * @param state   Current state buffer
 * @param entropy New entropy to mix
 * @param output  Output buffer
 * @param size    Buffer size in bytes
 */
void sve_mix_entropy(const uint8_t* state, const uint8_t* entropy,
                     uint8_t* output, size_t size);

// ============================================================================
// PREFETCH HINTS
// ============================================================================

/**
 * @brief Prefetch amplitude data for upcoming operations
 *
 * Uses SVE prefetch instructions for optimal cache behavior.
 *
 * @param amplitudes Array to prefetch
 * @param n          Number of amplitudes
 * @param write      If true, prefetch for writing; otherwise for reading
 */
void sve_prefetch_amplitudes(const complex_t* amplitudes, size_t n, int write);

#ifdef __cplusplus
}
#endif

#endif /* SIMD_SVE_H */
