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
 * @stability evolving
 * @since v0.1.2
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
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
    ENTROPY_UTIL_SOURCE_AUTO,        /**< Automatic best available */
    ENTROPY_UTIL_SOURCE_HARDWARE,    /**< CPU hardware RNG (RDRAND/RNDR) */
    ENTROPY_UTIL_SOURCE_OS,          /**< OS entropy (/dev/urandom, etc) */
    ENTROPY_UTIL_SOURCE_JITTER,      /**< CPU timing jitter */
    ENTROPY_UTIL_SOURCE_MIXED,       /**< XOR of multiple sources */
    ENTROPY_UTIL_SOURCE_QUANTUM      /**< Quantum RNG (if available) */
} entropy_util_source_type_t;

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
} entropy_util_quality_t;

// ============================================================================
// ENTROPY CONTEXT
// ============================================================================

/**
 * @brief Entropy context for stateful collection
 */
typedef struct entropy_util_ctx {
    entropy_util_source_type_t source;   /**< Active source type */

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
} entropy_util_ctx_t;

// ============================================================================
// CONTEXT MANAGEMENT
// ============================================================================

/**
 * @brief Create entropy context with automatic source selection
 *
 * @return New entropy context or NULL on error
 */
entropy_util_ctx_t* entropy_util_create(void);

/**
 * @brief Create entropy context with specific source
 *
 * @param source Preferred entropy source
 * @return New entropy context or NULL on error
 */
entropy_util_ctx_t* entropy_util_create_with_source(entropy_util_source_type_t source);

/**
 * @brief Destroy entropy context
 *
 * @param ctx Context to destroy
 */
void entropy_util_destroy(entropy_util_ctx_t* ctx);

/**
 * @brief Reset entropy context and reseed
 *
 * @param ctx Entropy context
 * @return 0 on success, -1 on error
 */
int entropy_util_reseed(entropy_util_ctx_t* ctx);

// ============================================================================
// ENTROPY COLLECTION
// ============================================================================

/**
 * @brief Get single random byte
 *
 * @param ctx Entropy context
 * @return Random byte
 */
uint8_t entropy_util_byte(entropy_util_ctx_t* ctx);

/**
 * @brief Get random bytes
 *
 * @param ctx Entropy context
 * @param buffer Output buffer
 * @param size Number of bytes
 * @return Number of bytes written
 */
size_t entropy_util_bytes(entropy_util_ctx_t* ctx, uint8_t* buffer, size_t size);

/**
 * @brief Get random 32-bit value
 *
 * @param ctx Entropy context
 * @return Random uint32_t
 */
uint32_t entropy_util_uint32(entropy_util_ctx_t* ctx);

/**
 * @brief Get random 64-bit value
 *
 * @param ctx Entropy context
 * @return Random uint64_t
 */
uint64_t entropy_util_uint64(entropy_util_ctx_t* ctx);

/**
 * @brief Get random double in [0, 1)
 *
 * @param ctx Entropy context
 * @return Random double
 */
double entropy_util_double(entropy_util_ctx_t* ctx);

// ============================================================================
// HARDWARE RNG
// ============================================================================

/**
 * @brief Check if hardware RNG is available
 *
 * @return 1 if available, 0 otherwise
 */
int entropy_util_hardware_available(void);

/**
 * @brief Get bytes from hardware RNG directly
 *
 * @param buffer Output buffer
 * @param size Number of bytes
 * @return Number of bytes obtained, -1 on error
 */
int entropy_util_hardware_bytes(uint8_t* buffer, size_t size);

/**
 * @brief Get hardware RNG name
 *
 * @return Name string (e.g., "RDRAND", "RNDR", "none")
 */
const char* entropy_util_hardware_name(void);

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
int entropy_util_os_bytes(uint8_t* buffer, size_t size);

/**
 * @brief Get OS entropy source name
 *
 * @return Name string (e.g., "/dev/urandom", "CryptGenRandom")
 */
const char* entropy_util_os_source_name(void);

// ============================================================================
// JITTER ENTROPY
// ============================================================================

/**
 * @brief Fallback entropy source: CPU timing-jitter sampler.
 *
 * IMPORTANT: This is a *fallback* path used only when stronger
 * entropy sources (RDRAND/RDSEED/ARM RNG/`getrandom`/`/dev/urandom`)
 * are unavailable.  The reseeding paths in `entropy.c` invoke this
 * routine after first attempting the platform sources.  Callers
 * outside the entropy stack should not use it as a substitute for
 * `entropy_collect_bytes`.
 *
 * @param buffer Output buffer.
 * @param size   Number of bytes requested.
 * @return Number of bytes successfully collected (may be less than
 *         `size` if the timer is too coarse to resolve usable
 *         jitter).
 */
size_t entropy_util_jitter_bytes(uint8_t* buffer, size_t size);

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
void entropy_util_mix(entropy_util_ctx_t* ctx, const uint8_t* data, size_t size);

/**
 * @brief Non-cryptographic entropy folding (SplitMix64 + XorShift128+).
 *
 * NOT a cryptographic hash and NOT a strong randomness extractor: it folds
 * @p input into a small state with SplitMix64 and squeezes output with
 * XorShift128+. It provides mixing/whitening only, with no security
 * guarantee on the output distribution. For certified extraction use the
 * Toeplitz strong extractor in qrng_di.h; for conditioned release bytes use
 * moonlab_qrng_bytes.
 *
 * @param input Input data
 * @param input_size Input size
 * @param output Output buffer
 * @param output_size Desired output size
 * @return 0 on success, -1 on error
 */
int entropy_util_extract(const uint8_t* input, size_t input_size,
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
int entropy_util_assess_quality(entropy_util_ctx_t* ctx, size_t sample_size,
                           entropy_util_quality_t* quality);

/**
 * @brief Run basic entropy tests on data
 *
 * @param data Data to test
 * @param size Size of data
 * @param quality Output quality assessment
 * @return 1 if tests pass, 0 otherwise
 */
int entropy_util_test_data(const uint8_t* data, size_t size,
                      entropy_util_quality_t* quality);

/**
 * @brief Estimate entropy per byte
 *
 * @param data Data to analyze
 * @param size Size of data
 * @return Estimated entropy in bits per byte (0-8)
 */
double entropy_util_estimate(const uint8_t* data, size_t size);

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
size_t entropy_util_generate(uint8_t* buffer, size_t size);

/**
 * @brief Get single random uint64 (one-shot)
 *
 * @return Random uint64_t
 */
uint64_t entropy_util_random_uint64(void);

/**
 * @brief Seed from best available entropy
 *
 * @param seed Output seed buffer
 * @param seed_size Size of seed
 * @return 0 on success, -1 on error
 */
int entropy_util_seed(uint8_t* seed, size_t seed_size);

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
void entropy_util_get_capabilities(int* has_hardware, int* has_os, int* has_jitter);

/**
 * @brief Get entropy source description
 *
 * @param source Source type
 * @return Human-readable description
 */
const char* entropy_util_source_name(entropy_util_source_type_t source);

#ifdef __cplusplus
}
#endif

#endif /* UTILS_ENTROPY_H */
