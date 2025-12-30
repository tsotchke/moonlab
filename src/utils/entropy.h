/**
 * @file entropy.h
 * @brief Entropy sources and quality assessment
 *
 * Platform-independent entropy collection:
 * - Hardware RNG (RDRAND/RDSEED on x86, RNDR on ARM)
 * - OS entropy (/dev/urandom, CryptGenRandom)
 * - Time-based jitter
 * - Entropy pool mixing
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#ifndef UTILS_ENTROPY_H
#define UTILS_ENTROPY_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ENTROPY SOURCE TYPES
// ============================================================================

/**
 * @brief Entropy source identifiers
 */
typedef enum {
    ENTROPY_SOURCE_AUTO,        /**< Automatic best available */
    ENTROPY_SOURCE_HARDWARE,    /**< CPU hardware RNG (RDRAND/RNDR) */
    ENTROPY_SOURCE_OS,          /**< OS entropy (/dev/urandom, etc) */
    ENTROPY_SOURCE_JITTER,      /**< CPU timing jitter */
    ENTROPY_SOURCE_MIXED,       /**< XOR of multiple sources */
    ENTROPY_SOURCE_QUANTUM      /**< Quantum RNG (if available) */
} entropy_source_type_t;

/**
 * @brief Entropy quality assessment
 */
typedef struct {
    double estimated_entropy;   /**< Estimated bits of entropy per byte */
    double chi_squared;         /**< Chi-squared statistic */
    double compression_ratio;   /**< Compression test ratio */
    int hardware_available;     /**< Hardware RNG available */
    int os_available;           /**< OS entropy available */
    int passed_basic_tests;     /**< Passed basic randomness tests */
} entropy_quality_t;

// ============================================================================
// ENTROPY CONTEXT
// ============================================================================

/**
 * @brief Entropy context for stateful collection
 */
typedef struct entropy_ctx {
    entropy_source_type_t source;   /**< Active source type */

    // Entropy pool
    uint8_t pool[256];              /**< Entropy pool */
    size_t pool_bytes;              /**< Bytes in pool */
    size_t pool_pos;                /**< Current position */

    // Mixing state
    uint64_t mix_state[4];          /**< Mixing state */
    uint64_t reseed_counter;        /**< Reseed counter */

    // Platform-specific handles
    int fd_urandom;                 /**< File descriptor for /dev/urandom */
    void* os_handle;                /**< OS-specific handle */

    // Statistics
    uint64_t bytes_collected;       /**< Total bytes collected */
    uint64_t hardware_bytes;        /**< Bytes from hardware RNG */
    uint64_t os_bytes;              /**< Bytes from OS */
} entropy_ctx_t;

// ============================================================================
// CONTEXT MANAGEMENT
// ============================================================================

/**
 * @brief Create entropy context with automatic source selection
 *
 * @return New entropy context or NULL on error
 */
entropy_ctx_t* entropy_create(void);

/**
 * @brief Create entropy context with specific source
 *
 * @param source Preferred entropy source
 * @return New entropy context or NULL on error
 */
entropy_ctx_t* entropy_create_with_source(entropy_source_type_t source);

/**
 * @brief Destroy entropy context
 *
 * @param ctx Context to destroy
 */
void entropy_destroy(entropy_ctx_t* ctx);

/**
 * @brief Reset entropy context and reseed
 *
 * @param ctx Entropy context
 * @return 0 on success, -1 on error
 */
int entropy_reseed(entropy_ctx_t* ctx);

// ============================================================================
// ENTROPY COLLECTION
// ============================================================================

/**
 * @brief Get single random byte
 *
 * @param ctx Entropy context
 * @return Random byte
 */
uint8_t entropy_byte(entropy_ctx_t* ctx);

/**
 * @brief Get random bytes
 *
 * @param ctx Entropy context
 * @param buffer Output buffer
 * @param size Number of bytes
 * @return Number of bytes written
 */
size_t entropy_bytes(entropy_ctx_t* ctx, uint8_t* buffer, size_t size);

/**
 * @brief Get random 32-bit value
 *
 * @param ctx Entropy context
 * @return Random uint32_t
 */
uint32_t entropy_uint32(entropy_ctx_t* ctx);

/**
 * @brief Get random 64-bit value
 *
 * @param ctx Entropy context
 * @return Random uint64_t
 */
uint64_t entropy_uint64(entropy_ctx_t* ctx);

/**
 * @brief Get random double in [0, 1)
 *
 * @param ctx Entropy context
 * @return Random double
 */
double entropy_double(entropy_ctx_t* ctx);

// ============================================================================
// HARDWARE RNG
// ============================================================================

/**
 * @brief Check if hardware RNG is available
 *
 * @return 1 if available, 0 otherwise
 */
int entropy_hardware_available(void);

/**
 * @brief Get bytes from hardware RNG directly
 *
 * @param buffer Output buffer
 * @param size Number of bytes
 * @return Number of bytes obtained, -1 on error
 */
int entropy_hardware_bytes(uint8_t* buffer, size_t size);

/**
 * @brief Get hardware RNG name
 *
 * @return Name string (e.g., "RDRAND", "RNDR", "none")
 */
const char* entropy_hardware_name(void);

// ============================================================================
// OS ENTROPY
// ============================================================================

/**
 * @brief Get bytes from OS entropy source
 *
 * @param buffer Output buffer
 * @param size Number of bytes
 * @return Number of bytes obtained, -1 on error
 */
int entropy_os_bytes(uint8_t* buffer, size_t size);

/**
 * @brief Get OS entropy source name
 *
 * @return Name string (e.g., "/dev/urandom", "CryptGenRandom")
 */
const char* entropy_os_source_name(void);

// ============================================================================
// JITTER ENTROPY
// ============================================================================

/**
 * @brief Collect entropy from CPU timing jitter
 *
 * @param buffer Output buffer
 * @param size Number of bytes
 * @return Number of bytes collected
 */
size_t entropy_jitter_bytes(uint8_t* buffer, size_t size);

// ============================================================================
// ENTROPY MIXING
// ============================================================================

/**
 * @brief Mix additional entropy into context
 *
 * @param ctx Entropy context
 * @param data Additional entropy data
 * @param size Size of data
 */
void entropy_mix(entropy_ctx_t* ctx, const uint8_t* data, size_t size);

/**
 * @brief Hash-based entropy extraction
 *
 * @param input Input data
 * @param input_size Input size
 * @param output Output buffer
 * @param output_size Desired output size
 * @return 0 on success, -1 on error
 */
int entropy_extract(const uint8_t* input, size_t input_size,
                    uint8_t* output, size_t output_size);

// ============================================================================
// QUALITY ASSESSMENT
// ============================================================================

/**
 * @brief Assess entropy quality
 *
 * @param ctx Entropy context
 * @param sample_size Number of bytes to test
 * @param quality Output quality assessment
 * @return 0 on success, -1 on error
 */
int entropy_assess_quality(entropy_ctx_t* ctx, size_t sample_size,
                           entropy_quality_t* quality);

/**
 * @brief Run basic entropy tests on data
 *
 * @param data Data to test
 * @param size Size of data
 * @param quality Output quality assessment
 * @return 1 if tests pass, 0 otherwise
 */
int entropy_test_data(const uint8_t* data, size_t size,
                      entropy_quality_t* quality);

/**
 * @brief Estimate entropy per byte
 *
 * @param data Data to analyze
 * @param size Size of data
 * @return Estimated entropy in bits per byte (0-8)
 */
double entropy_estimate(const uint8_t* data, size_t size);

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/**
 * @brief Generate random bytes (one-shot, no context)
 *
 * @param buffer Output buffer
 * @param size Number of bytes
 * @return Number of bytes generated
 */
size_t entropy_generate(uint8_t* buffer, size_t size);

/**
 * @brief Get single random uint64 (one-shot)
 *
 * @return Random uint64_t
 */
uint64_t entropy_random_uint64(void);

/**
 * @brief Seed from best available entropy
 *
 * @param seed Output seed buffer
 * @param seed_size Size of seed
 * @return 0 on success, -1 on error
 */
int entropy_seed(uint8_t* seed, size_t seed_size);

// ============================================================================
// PLATFORM DETECTION
// ============================================================================

/**
 * @brief Get platform entropy capabilities
 *
 * @param has_hardware Output: hardware RNG available
 * @param has_os Output: OS entropy available
 * @param has_jitter Output: jitter entropy available
 */
void entropy_utils_get_capabilities(int* has_hardware, int* has_os, int* has_jitter);

/**
 * @brief Get entropy source description
 *
 * @param source Source type
 * @return Human-readable description
 */
const char* entropy_utils_source_name(entropy_source_type_t source);

#ifdef __cplusplus
}
#endif

#endif /* UTILS_ENTROPY_H */
