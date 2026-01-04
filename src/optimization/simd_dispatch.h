/**
 * @file simd_dispatch.h
 * @brief Runtime SIMD capability detection and dispatch layer
 *
 * Provides comprehensive CPU feature detection for:
 * - x86-64: SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, FMA, AVX-512 (F, BW, DQ, VL)
 * - ARM64: NEON (baseline), SVE (Scalable Vector Extensions), SVE2
 *
 * The dispatch layer automatically selects the best available SIMD implementation
 * at runtime, with fallback chains to ensure compatibility.
 *
 * Fallback Priority:
 * - x86-64: AVX-512 -> AVX2+FMA -> AVX2 -> AVX -> SSE4.1 -> SSE2 -> Scalar
 * - ARM64: SVE2 -> SVE -> NEON -> Scalar
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef SIMD_DISPATCH_H
#define SIMD_DISPATCH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ARCHITECTURE DETECTION
// ============================================================================

// Detect target architecture at compile time
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define SIMD_ARCH_X86 1
    #define SIMD_ARCH_NAME "x86-64"
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define SIMD_ARCH_ARM64 1
    #define SIMD_ARCH_NAME "ARM64"
#elif defined(__arm__) || defined(_M_ARM)
    #define SIMD_ARCH_ARM32 1
    #define SIMD_ARCH_NAME "ARM32"
#else
    #define SIMD_ARCH_UNKNOWN 1
    #define SIMD_ARCH_NAME "Unknown"
#endif

// ============================================================================
// SIMD CAPABILITY FLAGS
// ============================================================================

/**
 * @brief Comprehensive SIMD capability flags
 *
 * Bit flags for all supported SIMD instruction sets.
 * Use simd_get_capabilities() to detect at runtime.
 */
typedef enum {
    // x86-64 SIMD
    SIMD_CAP_SSE2       = (1 << 0),    /**< SSE2 (baseline for x86-64) */
    SIMD_CAP_SSE3       = (1 << 1),    /**< SSE3 */
    SIMD_CAP_SSSE3      = (1 << 2),    /**< SSSE3 */
    SIMD_CAP_SSE4_1     = (1 << 3),    /**< SSE4.1 */
    SIMD_CAP_SSE4_2     = (1 << 4),    /**< SSE4.2 */
    SIMD_CAP_AVX        = (1 << 5),    /**< AVX */
    SIMD_CAP_AVX2       = (1 << 6),    /**< AVX2 */
    SIMD_CAP_FMA        = (1 << 7),    /**< FMA3 */
    SIMD_CAP_AVX512F    = (1 << 8),    /**< AVX-512 Foundation */
    SIMD_CAP_AVX512BW   = (1 << 9),    /**< AVX-512 Byte/Word */
    SIMD_CAP_AVX512DQ   = (1 << 10),   /**< AVX-512 Doubleword/Quadword */
    SIMD_CAP_AVX512VL   = (1 << 11),   /**< AVX-512 Vector Length */
    SIMD_CAP_AVX512CD   = (1 << 12),   /**< AVX-512 Conflict Detection */
    SIMD_CAP_AVX512VNNI = (1 << 13),   /**< AVX-512 VNNI (neural network) */

    // ARM SIMD
    SIMD_CAP_NEON       = (1 << 16),   /**< ARM NEON (baseline for ARM64) */
    SIMD_CAP_SVE        = (1 << 17),   /**< ARM SVE (Scalable Vector Extensions) */
    SIMD_CAP_SVE2       = (1 << 18),   /**< ARM SVE2 */
    SIMD_CAP_DOTPROD    = (1 << 19),   /**< ARM SDOT/UDOT instructions */

    // Platform-specific accelerators
    SIMD_CAP_AMX        = (1 << 24),   /**< Apple AMX (detected via Accelerate) */
    SIMD_CAP_INTEL_AMX  = (1 << 25),   /**< Intel AMX (matrix extensions) */

} simd_capability_t;

/**
 * @brief SIMD level for easy comparison
 *
 * Higher levels include all capabilities of lower levels.
 */
typedef enum {
    SIMD_LEVEL_SCALAR   = 0,    /**< No SIMD, scalar operations only */
    SIMD_LEVEL_SSE2     = 1,    /**< SSE2 (128-bit, 2 doubles) */
    SIMD_LEVEL_AVX      = 2,    /**< AVX (256-bit, 4 doubles) */
    SIMD_LEVEL_AVX2     = 3,    /**< AVX2 with FMA (256-bit, optimized) */
    SIMD_LEVEL_AVX512   = 4,    /**< AVX-512 (512-bit, 8 doubles) */
    SIMD_LEVEL_NEON     = 1,    /**< ARM NEON (128-bit, 2 doubles) */
    SIMD_LEVEL_SVE      = 3,    /**< ARM SVE (variable, up to 2048-bit) */
    SIMD_LEVEL_SVE2     = 4,    /**< ARM SVE2 */
} simd_level_t;

// ============================================================================
// SIMD DETECTION STRUCTURES
// ============================================================================

/**
 * @brief Complete SIMD capability information
 */
typedef struct {
    uint32_t flags;             /**< Capability bit flags (simd_capability_t) */
    simd_level_t level;         /**< Overall SIMD level */

    // x86-64 specific
    int has_sse2;
    int has_sse3;
    int has_ssse3;
    int has_sse4_1;
    int has_sse4_2;
    int has_avx;
    int has_avx2;
    int has_fma;
    int has_avx512f;
    int has_avx512bw;
    int has_avx512dq;
    int has_avx512vl;
    int has_avx512cd;
    int has_avx512vnni;

    // ARM specific
    int has_neon;
    int has_sve;
    int has_sve2;
    int has_dotprod;
    uint32_t sve_vector_length;  /**< SVE vector length in bits (128-2048) */

    // Platform accelerators
    int has_amx;                 /**< Apple AMX or Intel AMX */

    // Cache information
    uint32_t l1_cache_size;      /**< L1 cache size in bytes */
    uint32_t l2_cache_size;      /**< L2 cache size in bytes */
    uint32_t cache_line_size;    /**< Cache line size in bytes */
} simd_info_t;

// ============================================================================
// DETECTION FUNCTIONS
// ============================================================================

/**
 * @brief Detect all SIMD capabilities
 *
 * Performs comprehensive CPU feature detection. This function is called
 * once and results are cached for subsequent calls.
 *
 * @return Pointer to static SIMD info structure
 *
 * @note Thread-safe after first call
 *
 * @stability stable
 * @since v1.0.0
 */
const simd_info_t* simd_detect_capabilities_full(void);

/**
 * @brief Get capability flags
 *
 * Quick access to capability bit flags.
 *
 * @return Capability flags (simd_capability_t values OR'd together)
 *
 * @stability stable
 * @since v1.0.0
 */
uint32_t simd_get_capability_flags(void);

/**
 * @brief Check if specific capability is available
 *
 * @param cap Capability to check
 * @return 1 if available, 0 otherwise
 *
 * @stability stable
 * @since v1.0.0
 */
int simd_has_capability(simd_capability_t cap);

/**
 * @brief Get current SIMD level
 *
 * @return Current SIMD level (higher = more capable)
 *
 * @stability stable
 * @since v1.0.0
 */
simd_level_t simd_get_level(void);

/**
 * @brief Get human-readable capability string
 *
 * @return String describing detected capabilities
 *
 * @example "AVX-512 (F, BW, DQ, VL) + FMA + AVX2"
 * @example "ARM NEON + SVE (256-bit)"
 *
 * @stability stable
 * @since v1.0.0
 */
const char* simd_get_capability_string(void);

// ============================================================================
// DISPATCH FUNCTION TYPES
// ============================================================================

/**
 * @brief Operation type for dispatch
 */
typedef enum {
    SIMD_OP_SUM_SQUARED_MAG,     /**< Sum of squared magnitudes */
    SIMD_OP_NORMALIZE,           /**< Normalize amplitude array */
    SIMD_OP_COMPLEX_SWAP,        /**< Swap complex pairs (Pauli X) */
    SIMD_OP_COMPLEX_NEGATE,      /**< Negate complex array (Pauli Z) */
    SIMD_OP_APPLY_PHASE,         /**< Apply phase factor */
    SIMD_OP_MULTIPLY_I,          /**< Multiply by ±i */
    SIMD_OP_CUMULATIVE_PROB,     /**< Cumulative probability search */
    SIMD_OP_MATRIX_VEC_2X2,      /**< 2x2 complex matrix-vector multiply */
    SIMD_OP_XOR_BYTES,           /**< XOR byte arrays */
    SIMD_OP_COUNT
} simd_operation_t;

/**
 * @brief SIMD implementation backend
 */
typedef enum {
    SIMD_BACKEND_SCALAR,
    SIMD_BACKEND_SSE2,
    SIMD_BACKEND_AVX,
    SIMD_BACKEND_AVX2,
    SIMD_BACKEND_AVX512,
    SIMD_BACKEND_NEON,
    SIMD_BACKEND_SVE,
    SIMD_BACKEND_SVE2,
    SIMD_BACKEND_ACCELERATE,  /**< Apple Accelerate framework */
} simd_backend_t;

/**
 * @brief Get selected backend for an operation
 *
 * @param op Operation type
 * @return Backend that will be used
 *
 * @stability stable
 * @since v1.0.0
 */
simd_backend_t simd_get_backend(simd_operation_t op);

/**
 * @brief Get backend name string
 *
 * @param backend Backend type
 * @return Human-readable backend name
 *
 * @stability stable
 * @since v1.0.0
 */
const char* simd_backend_name(simd_backend_t backend);

// ============================================================================
// VECTOR WIDTH INFORMATION
// ============================================================================

/**
 * @brief Get native SIMD vector width in bytes
 *
 * @return Vector width: 16 (SSE/NEON), 32 (AVX), 64 (AVX-512), or variable (SVE)
 *
 * @stability stable
 * @since v1.0.0
 */
size_t simd_get_vector_width(void);

/**
 * @brief Get number of doubles per SIMD register
 *
 * @return Number of doubles: 2 (SSE/NEON), 4 (AVX), 8 (AVX-512)
 *
 * @stability stable
 * @since v1.0.0
 */
size_t simd_get_doubles_per_register(void);

/**
 * @brief Get number of complex doubles per SIMD register
 *
 * @return Number of complex: 1 (SSE/NEON), 2 (AVX), 4 (AVX-512)
 *
 * @stability stable
 * @since v1.0.0
 */
size_t simd_get_complex_per_register(void);

// ============================================================================
// OPTIMAL LOOP CONFIGURATION
// ============================================================================

/**
 * @brief Recommended loop unroll factor
 *
 * @return Unroll factor for optimal performance (typically 4-8)
 *
 * @stability stable
 * @since v1.0.0
 */
size_t simd_get_unroll_factor(void);

/**
 * @brief Minimum array size for SIMD to be beneficial
 *
 * @return Minimum element count (below this, scalar may be faster)
 *
 * @stability stable
 * @since v1.0.0
 */
size_t simd_get_min_array_size(void);

/**
 * @brief Optimal chunk size for parallel processing
 *
 * Returns a chunk size that:
 * - Fits in L1 cache
 * - Is a multiple of vector width
 * - Works well with OpenMP scheduling
 *
 * @param element_size Size of each element in bytes
 * @return Recommended chunk size in elements
 *
 * @stability stable
 * @since v1.0.0
 */
size_t simd_get_chunk_size(size_t element_size);

// ============================================================================
// DEBUG/PROFILING
// ============================================================================

/**
 * @brief Print detailed SIMD capability report
 *
 * Outputs comprehensive CPU feature information to stderr.
 * Useful for debugging and system configuration.
 *
 * @stability stable
 * @since v1.0.0
 */
void simd_print_capabilities(void);

/**
 * @brief Validate SIMD detection
 *
 * Performs quick self-tests to verify SIMD operations work correctly.
 * Returns 1 if all tests pass.
 *
 * @return 1 if validation passes, 0 otherwise
 *
 * @stability beta
 * @since v1.0.0
 */
int simd_validate(void);

#ifdef __cplusplus
}
#endif

#endif /* SIMD_DISPATCH_H */
