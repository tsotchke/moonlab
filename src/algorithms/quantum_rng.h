/**
 * @file quantum_rng.h
 * @brief Quantum Random Number Generator
 *
 * True quantum random number generation using:
 * - Hadamard-based superposition collapse
 * - Multi-qubit parallel generation
 * - Entropy extraction and conditioning
 * - Statistical validation
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef ALGORITHMS_QUANTUM_RNG_H
#define ALGORITHMS_QUANTUM_RNG_H

#include "../quantum/state.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// QRNG CONTEXT
// ============================================================================

/**
 * @brief QRNG context for stateful generation
 */
typedef struct {
    quantum_state_t* state;     /**< Quantum state for generation */
    int num_qubits;             /**< Number of qubits */
    uint64_t bits_generated;    /**< Total bits generated */
    uint64_t bytes_generated;   /**< Total bytes generated */

    // Buffer for partial bytes
    uint8_t buffer;             /**< Partial byte buffer */
    int buffer_bits;            /**< Bits in buffer */

    // Entropy source for measurement
    void* entropy_source;       /**< External entropy (optional) */

    // Statistics
    uint64_t zeros_count;       /**< Count of zeros generated */
    uint64_t ones_count;        /**< Count of ones generated */
} qrng_context_t;

/**
 * @brief QRNG configuration
 */
typedef struct {
    int num_qubits;             /**< Qubits to use (1-32) */
    int conditioning_rounds;    /**< Hash conditioning rounds (0=none) */
    int parallel_extractions;   /**< Parallel measurement extractions */
    void* entropy_source;       /**< External entropy source */
} qrng_config_t;

// ============================================================================
// CONTEXT MANAGEMENT
// ============================================================================

/**
 * @brief Create QRNG context with default configuration
 *
 * @param num_qubits Number of qubits (1-32)
 * @return New QRNG context or NULL on error
 */
qrng_context_t* qrng_create(int num_qubits);

/**
 * @brief Create QRNG context with configuration
 *
 * @param config Configuration options
 * @return New QRNG context or NULL on error
 */
qrng_context_t* qrng_create_configured(const qrng_config_t* config);

/**
 * @brief Destroy QRNG context
 *
 * @param ctx Context to destroy
 */
void qrng_destroy(qrng_context_t* ctx);

/**
 * @brief Reset QRNG statistics
 *
 * @param ctx QRNG context
 */
void qrng_reset_stats(qrng_context_t* ctx);

// ============================================================================
// SINGLE VALUE GENERATION
// ============================================================================

/**
 * @brief Generate single random bit
 *
 * Uses H gate + measurement on single qubit
 *
 * @param ctx QRNG context
 * @return Random bit (0 or 1)
 */
int qrng_bit(qrng_context_t* ctx);

/**
 * @brief Generate random byte (8 bits)
 *
 * @param ctx QRNG context
 * @return Random byte
 */
uint8_t qrng_byte(qrng_context_t* ctx);

/**
 * @brief Generate random 32-bit unsigned integer
 *
 * @param ctx QRNG context
 * @return Random uint32_t
 */
uint32_t qrng_uint32(qrng_context_t* ctx);

/**
 * @brief Generate random 64-bit unsigned integer
 *
 * @param ctx QRNG context
 * @return Random uint64_t
 */
uint64_t qrng_uint64(qrng_context_t* ctx);

/**
 * @brief Generate random integer in range [0, max)
 *
 * Uses rejection sampling for uniform distribution
 *
 * @param ctx QRNG context
 * @param max Exclusive upper bound
 * @return Random integer in [0, max)
 */
uint64_t qrng_range(qrng_context_t* ctx, uint64_t max);

/**
 * @brief Generate random double in [0, 1)
 *
 * Uses 53 bits for full double precision
 *
 * @param ctx QRNG context
 * @return Random double in [0, 1)
 */
double qrng_double(qrng_context_t* ctx);

/**
 * @brief Generate random double in [min, max)
 *
 * @param ctx QRNG context
 * @param min Minimum value (inclusive)
 * @param max Maximum value (exclusive)
 * @return Random double in [min, max)
 */
double qrng_double_range(qrng_context_t* ctx, double min, double max);

// ============================================================================
// BULK GENERATION
// ============================================================================

/**
 * @brief Fill buffer with random bytes
 *
 * Optimized for bulk generation using parallel measurements
 *
 * @param ctx QRNG context
 * @param buffer Output buffer
 * @param size Number of bytes to generate
 * @return Number of bytes generated
 */
size_t qrng_bytes(qrng_context_t* ctx, uint8_t* buffer, size_t size);

/**
 * @brief Fill array with random uint32 values
 *
 * @param ctx QRNG context
 * @param values Output array
 * @param count Number of values to generate
 * @return Number of values generated
 */
size_t qrng_uint32_array(qrng_context_t* ctx, uint32_t* values, size_t count);

/**
 * @brief Fill array with random doubles in [0, 1)
 *
 * @param ctx QRNG context
 * @param values Output array
 * @param count Number of values to generate
 * @return Number of values generated
 */
size_t qrng_double_array(qrng_context_t* ctx, double* values, size_t count);

// ============================================================================
// SPECIAL DISTRIBUTIONS
// ============================================================================

/**
 * @brief Generate standard normal random number (mean=0, std=1)
 *
 * Uses Box-Muller transform
 *
 * @param ctx QRNG context
 * @return Standard normal random value
 */
double qrng_normal(qrng_context_t* ctx);

/**
 * @brief Generate normal random number with parameters
 *
 * @param ctx QRNG context
 * @param mean Mean value
 * @param std Standard deviation
 * @return Normal random value
 */
double qrng_normal_params(qrng_context_t* ctx, double mean, double std);

/**
 * @brief Generate exponential random number
 *
 * @param ctx QRNG context
 * @param lambda Rate parameter
 * @return Exponential random value
 */
double qrng_exponential(qrng_context_t* ctx, double lambda);

/**
 * @brief Shuffle array using Fisher-Yates algorithm
 *
 * @param ctx QRNG context
 * @param array Array to shuffle
 * @param element_size Size of each element
 * @param count Number of elements
 */
void qrng_shuffle(qrng_context_t* ctx, void* array,
                  size_t element_size, size_t count);

// ============================================================================
// STATISTICAL VALIDATION
// ============================================================================

/**
 * @brief QRNG statistics report
 */
typedef struct {
    uint64_t total_bits;        /**< Total bits generated */
    double bias;                /**< Measured bias (0.5 = ideal) */
    double chi_squared;         /**< Chi-squared statistic */
    double p_value;             /**< P-value from chi-squared test */
    int passed_monobit;         /**< NIST monobit test result */
    int passed_runs;            /**< NIST runs test result */
    int passed_poker;           /**< Poker test result */
} qrng_stats_t;

/**
 * @brief Get QRNG statistics
 *
 * @param ctx QRNG context
 * @param stats Output statistics
 */
void qrng_get_stats(const qrng_context_t* ctx, qrng_stats_t* stats);

/**
 * @brief Run NIST statistical tests on generated data
 *
 * @param data Data to test
 * @param size Size in bytes
 * @param stats Output statistics
 * @return 1 if all tests pass, 0 otherwise
 */
int qrng_validate(const uint8_t* data, size_t size, qrng_stats_t* stats);

/**
 * @brief Run comprehensive NIST SP 800-22 tests
 *
 * @param ctx QRNG context
 * @param num_bits Number of bits to test
 * @return Number of tests passed (out of 15)
 */
int qrng_nist_tests(qrng_context_t* ctx, size_t num_bits);

// ============================================================================
// STANDALONE FUNCTIONS
// ============================================================================

/**
 * @brief Generate random bytes without context (convenience)
 *
 * Creates temporary context, generates bytes, destroys context
 *
 * @param buffer Output buffer
 * @param size Number of bytes
 * @return Number of bytes generated
 */
size_t qrng_generate_bytes(uint8_t* buffer, size_t size);

/**
 * @brief Generate single random uint64 without context
 *
 * @return Random uint64_t
 */
uint64_t qrng_generate_uint64(void);

/**
 * @brief Generate random double in [0, 1) without context
 *
 * @return Random double
 */
double qrng_generate_double(void);

// ============================================================================
// ENTROPY MIXING
// ============================================================================

/**
 * @brief Mix external entropy into QRNG
 *
 * XORs external entropy with quantum randomness for defense-in-depth
 *
 * @param ctx QRNG context
 * @param entropy External entropy bytes
 * @param size Size of external entropy
 */
void qrng_mix_entropy(qrng_context_t* ctx, const uint8_t* entropy, size_t size);

/**
 * @brief Set entropy source callback
 *
 * @param ctx QRNG context
 * @param source Entropy source (platform-specific)
 */
void qrng_set_entropy_source(qrng_context_t* ctx, void* source);

#ifdef __cplusplus
}
#endif

#endif /* ALGORITHMS_QUANTUM_RNG_H */
