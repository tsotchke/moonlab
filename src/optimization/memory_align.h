/**
 * @file memory_align.h
 * @brief Cross-platform SIMD-aligned memory allocation
 *
 * Provides portable, SIMD-aligned memory management for quantum state vectors.
 * Critical for achieving maximum performance with AVX-512 (64-byte), AVX2 (32-byte),
 * SSE (16-byte), ARM NEON (16-byte), and ARM SVE (variable).
 *
 * Features:
 * - Cross-platform support: macOS, Linux, Windows, FreeBSD
 * - Configurable alignment: 16, 32, 64, 128 bytes
 * - Secure memory zeroing before deallocation
 * - Thread-safe allocation tracking (debug builds)
 * - Allocation failure diagnostics
 *
 * Memory Safety:
 * - All allocations are zeroed on initialization
 * - All deallocations securely zero memory before freeing
 * - Prevents quantum state data from persisting in freed memory
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef MEMORY_ALIGN_H
#define MEMORY_ALIGN_H

#include <stddef.h>
#include <stdint.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ALIGNMENT CONSTANTS
// ============================================================================

/**
 * @brief Standard alignment values for different SIMD architectures
 */
typedef enum {
    SIMD_ALIGN_16  = 16,   /**< SSE, ARM NEON minimum */
    SIMD_ALIGN_32  = 32,   /**< AVX, AVX2 optimal */
    SIMD_ALIGN_64  = 64,   /**< AVX-512, Apple AMX optimal, cache line */
    SIMD_ALIGN_128 = 128   /**< ARM SVE maximum, future SIMD */
} simd_alignment_t;

/**
 * @brief Default alignment for optimal cross-platform performance
 *
 * 64 bytes is optimal because:
 * - AVX-512 requires 64-byte alignment for full performance
 * - Apple AMX operates on 64-byte cache lines
 * - ARM SVE up to 512-bit vectors (64 bytes)
 * - Common cache line size across x86-64 and ARM64
 */
#define SIMD_DEFAULT_ALIGNMENT SIMD_ALIGN_64

// ============================================================================
// ALLOCATION STATISTICS (Debug/Profiling)
// ============================================================================

/**
 * @brief Memory allocation statistics
 *
 * Thread-safe counters for tracking allocation patterns.
 * Useful for detecting memory leaks and optimizing allocation strategies.
 */
typedef struct {
    uint64_t total_allocated;      /**< Total bytes currently allocated */
    uint64_t peak_allocated;       /**< Peak allocation high-water mark */
    uint64_t allocation_count;     /**< Number of active allocations */
    uint64_t total_allocations;    /**< Lifetime allocation count */
    uint64_t total_frees;          /**< Lifetime deallocation count */
    uint64_t allocation_failures;  /**< Failed allocation attempts */
} mem_alloc_stats_t;

// ============================================================================
// CORE ALLOCATION FUNCTIONS
// ============================================================================

/**
 * @brief Allocate aligned memory
 *
 * Allocates memory with specified alignment. Memory is zero-initialized.
 * Uses platform-optimal allocation:
 * - POSIX: posix_memalign() or aligned_alloc()
 * - Windows: _aligned_malloc()
 * - Fallback: Manual alignment with padding
 *
 * @param size      Number of bytes to allocate
 * @param alignment Required alignment (must be power of 2, >= sizeof(void*))
 * @return Pointer to aligned memory, or NULL on failure
 *
 * @note Caller must use simd_aligned_free() to deallocate
 * @note Memory is securely zeroed before deallocation
 *
 * @example
 * @code
 * // Allocate 1024 complex numbers with 64-byte alignment for AVX-512
 * complex_t *state = simd_aligned_alloc(1024 * sizeof(complex_t), SIMD_ALIGN_64);
 * if (!state) {
 *     // Handle allocation failure
 * }
 * // ... use state ...
 * simd_aligned_free(state, 1024 * sizeof(complex_t));
 * @endcode
 *
 * @stability stable
 * @since v1.0.0
 */
void* simd_aligned_alloc(size_t size, size_t alignment);

/**
 * @brief Allocate aligned memory with default alignment
 *
 * Convenience wrapper using SIMD_DEFAULT_ALIGNMENT (64 bytes).
 *
 * @param size Number of bytes to allocate
 * @return Pointer to aligned memory, or NULL on failure
 *
 * @stability stable
 * @since v1.0.0
 */
void* simd_aligned_alloc_default(size_t size);

/**
 * @brief Free aligned memory with secure zeroing
 *
 * Securely zeros memory before deallocation to prevent data leakage.
 * Thread-safe and compatible with all allocation functions in this module.
 *
 * @param ptr  Pointer returned by simd_aligned_alloc (may be NULL)
 * @param size Original allocation size (for secure zeroing)
 *
 * @note If size is 0, memory is freed without zeroing (use with caution)
 * @note If ptr is NULL, function returns immediately
 *
 * @stability stable
 * @since v1.0.0
 */
void simd_aligned_free(void* ptr, size_t size);

/**
 * @brief Reallocate aligned memory
 *
 * Resizes an aligned allocation, preserving contents up to min(old_size, new_size).
 * New memory is zero-initialized if new_size > old_size.
 * Old memory is securely zeroed before freeing.
 *
 * @param ptr       Pointer to current allocation (may be NULL for new allocation)
 * @param old_size  Current allocation size
 * @param new_size  Requested new size
 * @param alignment Required alignment
 * @return Pointer to reallocated memory, or NULL on failure
 *
 * @note On failure, original memory is NOT freed
 *
 * @stability stable
 * @since v1.0.0
 */
void* simd_aligned_realloc(void* ptr, size_t old_size, size_t new_size, size_t alignment);

// ============================================================================
// COMPLEX NUMBER ARRAY ALLOCATION
// ============================================================================

/**
 * @brief Complex number type (compatible with quantum state)
 */
#ifndef COMPLEX_T_DEFINED
#define COMPLEX_T_DEFINED
typedef double _Complex complex_t;
#endif

/**
 * @brief Allocate aligned array of complex numbers
 *
 * Optimized for quantum state vectors. Uses 64-byte alignment
 * for optimal AVX-512 and Apple AMX performance.
 *
 * @param num_elements Number of complex numbers to allocate
 * @return Pointer to aligned complex array, or NULL on failure
 *
 * @stability stable
 * @since v1.0.0
 */
complex_t* simd_alloc_complex_array(size_t num_elements);

/**
 * @brief Free aligned complex array with secure zeroing
 *
 * @param ptr          Pointer to complex array
 * @param num_elements Number of elements (for secure zeroing)
 *
 * @stability stable
 * @since v1.0.0
 */
void simd_free_complex_array(complex_t* ptr, size_t num_elements);

/**
 * @brief Allocate aligned array of doubles
 *
 * @param num_elements Number of doubles to allocate
 * @return Pointer to aligned double array, or NULL on failure
 *
 * @stability stable
 * @since v1.0.0
 */
double* simd_alloc_double_array(size_t num_elements);

/**
 * @brief Free aligned double array with secure zeroing
 *
 * @param ptr          Pointer to double array
 * @param num_elements Number of elements (for secure zeroing)
 *
 * @stability stable
 * @since v1.0.0
 */
void simd_free_double_array(double* ptr, size_t num_elements);

/**
 * @brief Allocate aligned array of 64-bit integers
 *
 * @param num_elements Number of uint64_t to allocate
 * @return Pointer to aligned uint64_t array, or NULL on failure
 *
 * @stability stable
 * @since v1.0.0
 */
uint64_t* simd_alloc_uint64_array(size_t num_elements);

/**
 * @brief Free aligned uint64_t array with secure zeroing
 *
 * @param ptr          Pointer to uint64_t array
 * @param num_elements Number of elements (for secure zeroing)
 *
 * @stability stable
 * @since v1.0.0
 */
void simd_free_uint64_array(uint64_t* ptr, size_t num_elements);

// ============================================================================
// MEMORY UTILITIES
// ============================================================================

/**
 * @brief Check if pointer is aligned to specified boundary
 *
 * @param ptr       Pointer to check
 * @param alignment Required alignment (must be power of 2)
 * @return 1 if aligned, 0 otherwise
 *
 * @stability stable
 * @since v1.0.0
 */
int simd_is_aligned(const void* ptr, size_t alignment);

/**
 * @brief Get optimal alignment for current platform
 *
 * Detects CPU capabilities and returns optimal alignment:
 * - AVX-512: 64 bytes
 * - AVX/AVX2: 32 bytes
 * - SSE: 16 bytes
 * - ARM SVE: 64-128 bytes (vector length dependent)
 * - ARM NEON: 16 bytes
 *
 * @return Optimal alignment in bytes
 *
 * @stability stable
 * @since v1.0.0
 */
size_t simd_get_optimal_alignment(void);

/**
 * @brief Secure zero memory
 *
 * Zeroes memory in a way that won't be optimized out by compiler.
 * Uses volatile operations and memory barriers.
 *
 * @param ptr  Pointer to memory
 * @param size Number of bytes to zero
 *
 * @stability stable
 * @since v1.0.0
 */
void simd_secure_zero(void* ptr, size_t size);

/**
 * @brief Aligned memory copy
 *
 * Optimized memcpy for aligned source and destination.
 * Uses SIMD instructions when available.
 *
 * @param dest      Destination pointer (should be aligned)
 * @param src       Source pointer (should be aligned)
 * @param size      Number of bytes to copy
 * @param alignment Alignment of both pointers
 *
 * @stability stable
 * @since v1.0.0
 */
void simd_aligned_memcpy(void* dest, const void* src, size_t size, size_t alignment);

/**
 * @brief Aligned memory set
 *
 * Optimized memset for aligned destination.
 * Uses SIMD instructions when available.
 *
 * @param ptr       Destination pointer (should be aligned)
 * @param value     Byte value to set
 * @param size      Number of bytes to set
 * @param alignment Alignment of pointer
 *
 * @stability stable
 * @since v1.0.0
 */
void simd_aligned_memset(void* ptr, int value, size_t size, size_t alignment);

// ============================================================================
// ALLOCATION STATISTICS (Optional, compile-time enabled)
// ============================================================================

#ifdef MEMORY_ALIGN_TRACK_STATS

/**
 * @brief Get current allocation statistics
 *
 * Thread-safe retrieval of allocation statistics.
 * Only available when compiled with MEMORY_ALIGN_TRACK_STATS.
 *
 * @param stats Output statistics structure
 *
 * @stability beta
 * @since v1.0.0
 */
void simd_get_alloc_stats(mem_alloc_stats_t* stats);

/**
 * @brief Reset allocation statistics
 *
 * Resets all counters to zero. Thread-safe.
 * Only available when compiled with MEMORY_ALIGN_TRACK_STATS.
 *
 * @stability beta
 * @since v1.0.0
 */
void simd_reset_alloc_stats(void);

/**
 * @brief Print allocation statistics
 *
 * Prints formatted statistics to stderr.
 * Only available when compiled with MEMORY_ALIGN_TRACK_STATS.
 *
 * @stability beta
 * @since v1.0.0
 */
void simd_print_alloc_stats(void);

#endif /* MEMORY_ALIGN_TRACK_STATS */

// ============================================================================
// PLATFORM DETECTION
// ============================================================================

/**
 * @brief Memory allocation backend information
 */
typedef struct {
    const char* platform;        /**< Platform name (macOS, Linux, Windows) */
    const char* allocator;       /**< Allocator used (posix_memalign, aligned_alloc, etc.) */
    size_t default_alignment;    /**< Default alignment being used */
    size_t page_size;            /**< System page size */
    int supports_huge_pages;     /**< Whether huge pages are supported */
} mem_platform_info_t;

/**
 * @brief Get platform-specific memory information
 *
 * @return Platform information structure
 *
 * @stability stable
 * @since v1.0.0
 */
mem_platform_info_t simd_get_platform_info(void);

#ifdef __cplusplus
}
#endif

#endif /* MEMORY_ALIGN_H */
