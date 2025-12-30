/**
 * @file memory_align.c
 * @brief Cross-platform SIMD-aligned memory allocation implementation
 *
 * Production-grade implementation with:
 * - Cross-platform support (macOS, Linux, Windows, FreeBSD)
 * - Multiple alignment options (16-128 bytes)
 * - Secure memory zeroing (prevents data leakage)
 * - Thread-safe allocation tracking
 * - Comprehensive error handling
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#include "memory_align.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Platform-specific includes
#if defined(__APPLE__)
    #include <sys/sysctl.h>
    #include <mach/mach.h>
    #include <unistd.h>
    #define PLATFORM_MACOS 1
#elif defined(__linux__)
    #include <unistd.h>
    #include <sys/auxv.h>
    #define PLATFORM_LINUX 1
#elif defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #include <malloc.h>
    #define PLATFORM_WINDOWS 1
#elif defined(__FreeBSD__)
    #include <unistd.h>
    #define PLATFORM_FREEBSD 1
#endif

// SIMD intrinsics for CPU detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define ARCH_X86 1
    #ifdef _MSC_VER
        #include <intrin.h>
    #else
        #include <cpuid.h>
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define ARCH_ARM64 1
    #ifdef __linux__
        #include <sys/auxv.h>
        #include <asm/hwcap.h>
    #endif
#endif

// ============================================================================
// SECURE MEMORY ZEROING
// ============================================================================

/**
 * @brief Compiler-safe secure zeroing
 *
 * Uses volatile to prevent compiler optimization.
 * Implements the SecureZeroMemory pattern.
 */
void simd_secure_zero(void* ptr, size_t size) {
    if (!ptr || size == 0) return;

    // Volatile pointer prevents optimization
    volatile unsigned char* volatile p = (volatile unsigned char* volatile)ptr;

    // Zero byte by byte - cannot be optimized out
    while (size--) {
        *p++ = 0;
    }

    // Memory barrier to ensure zeroing completes before any subsequent operations
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" ::: "memory");
#elif defined(_MSC_VER)
    _ReadWriteBarrier();
#endif
}

// Alternative secure zero using memset_s if available (C11 Annex K)
#if defined(__STDC_LIB_EXT1__) || (defined(_MSC_VER) && _MSC_VER >= 1900)
    #define HAVE_MEMSET_S 1
#endif

static void secure_memzero_internal(void* ptr, size_t size) {
    if (!ptr || size == 0) return;

#ifdef HAVE_MEMSET_S
    memset_s(ptr, size, 0, size);
#elif defined(_WIN32)
    SecureZeroMemory(ptr, size);
#elif defined(__APPLE__)
    // macOS has memset_s in <string.h> when __STDC_WANT_LIB_EXT1__ is defined
    memset(ptr, 0, size);
    // Barrier to prevent dead store elimination
    __asm__ __volatile__("" : : "r"(ptr) : "memory");
#else
    // Use our volatile implementation
    simd_secure_zero(ptr, size);
#endif
}

// ============================================================================
// PLATFORM-SPECIFIC ALIGNED ALLOCATION
// ============================================================================

/**
 * @brief Validate alignment value
 */
static int is_valid_alignment(size_t alignment) {
    // Must be non-zero
    if (alignment == 0) return 0;

    // Must be power of 2
    if ((alignment & (alignment - 1)) != 0) return 0;

    // Must be at least sizeof(void*) for malloc compatibility
    if (alignment < sizeof(void*)) return 0;

    return 1;
}

void* simd_aligned_alloc(size_t size, size_t alignment) {
    void* ptr = NULL;

    // Validate inputs
    if (size == 0) return NULL;
    if (!is_valid_alignment(alignment)) {
        // Default to minimum valid alignment
        alignment = sizeof(void*);
    }

    // Round up size to alignment boundary for safety
    size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

#if defined(PLATFORM_MACOS)
    // macOS: posix_memalign is the standard approach
    // Works on both Intel and Apple Silicon
    if (posix_memalign(&ptr, alignment, aligned_size) != 0) {
        ptr = NULL;
    }

#elif defined(PLATFORM_LINUX)
    // Linux: Use aligned_alloc (C11) when alignment divides size
    // Otherwise fall back to posix_memalign
    if (aligned_size % alignment == 0) {
        ptr = aligned_alloc(alignment, aligned_size);
    } else {
        if (posix_memalign(&ptr, alignment, aligned_size) != 0) {
            ptr = NULL;
        }
    }

#elif defined(PLATFORM_WINDOWS)
    // Windows: Use _aligned_malloc
    ptr = _aligned_malloc(aligned_size, alignment);

#elif defined(PLATFORM_FREEBSD)
    // FreeBSD: posix_memalign
    if (posix_memalign(&ptr, alignment, aligned_size) != 0) {
        ptr = NULL;
    }

#else
    // Fallback: Manual alignment with padding
    // Allocate extra space for alignment and original pointer storage
    size_t total_size = aligned_size + alignment + sizeof(void*);
    void* raw = malloc(total_size);
    if (raw) {
        // Calculate aligned address
        uintptr_t raw_addr = (uintptr_t)raw;
        uintptr_t aligned_addr = (raw_addr + sizeof(void*) + alignment - 1) & ~(alignment - 1);

        // Store original pointer just before aligned address
        ((void**)aligned_addr)[-1] = raw;
        ptr = (void*)aligned_addr;
    }
#endif

    // Zero-initialize on success
    if (ptr) {
        memset(ptr, 0, aligned_size);
    }

#ifdef MEMORY_ALIGN_TRACK_STATS
    if (ptr) {
        track_allocation(aligned_size);
    } else {
        track_allocation_failure();
    }
#endif

    return ptr;
}

void* simd_aligned_alloc_default(size_t size) {
    return simd_aligned_alloc(size, SIMD_DEFAULT_ALIGNMENT);
}

void simd_aligned_free(void* ptr, size_t size) {
    if (!ptr) return;

    // Secure zero before freeing
    if (size > 0) {
        secure_memzero_internal(ptr, size);
    }

#if defined(PLATFORM_WINDOWS)
    _aligned_free(ptr);
#elif defined(PLATFORM_MACOS) || defined(PLATFORM_LINUX) || defined(PLATFORM_FREEBSD)
    // posix_memalign and aligned_alloc use standard free()
    free(ptr);
#else
    // Fallback: Retrieve original pointer
    void* raw = ((void**)ptr)[-1];
    free(raw);
#endif

#ifdef MEMORY_ALIGN_TRACK_STATS
    track_deallocation(size);
#endif
}

void* simd_aligned_realloc(void* ptr, size_t old_size, size_t new_size, size_t alignment) {
    if (new_size == 0) {
        simd_aligned_free(ptr, old_size);
        return NULL;
    }

    if (!ptr) {
        return simd_aligned_alloc(new_size, alignment);
    }

    // Allocate new aligned memory
    void* new_ptr = simd_aligned_alloc(new_size, alignment);
    if (!new_ptr) {
        return NULL;  // Original memory is NOT freed on failure
    }

    // Copy existing data
    size_t copy_size = (old_size < new_size) ? old_size : new_size;
    memcpy(new_ptr, ptr, copy_size);

    // Free old memory (with secure zeroing)
    simd_aligned_free(ptr, old_size);

    return new_ptr;
}

// ============================================================================
// TYPE-SPECIFIC ALLOCATION HELPERS
// ============================================================================

complex_t* simd_alloc_complex_array(size_t num_elements) {
    if (num_elements == 0) return NULL;
    size_t size = num_elements * sizeof(complex_t);
    return (complex_t*)simd_aligned_alloc(size, SIMD_DEFAULT_ALIGNMENT);
}

void simd_free_complex_array(complex_t* ptr, size_t num_elements) {
    simd_aligned_free(ptr, num_elements * sizeof(complex_t));
}

double* simd_alloc_double_array(size_t num_elements) {
    if (num_elements == 0) return NULL;
    size_t size = num_elements * sizeof(double);
    return (double*)simd_aligned_alloc(size, SIMD_DEFAULT_ALIGNMENT);
}

void simd_free_double_array(double* ptr, size_t num_elements) {
    simd_aligned_free(ptr, num_elements * sizeof(double));
}

uint64_t* simd_alloc_uint64_array(size_t num_elements) {
    if (num_elements == 0) return NULL;
    size_t size = num_elements * sizeof(uint64_t);
    return (uint64_t*)simd_aligned_alloc(size, SIMD_DEFAULT_ALIGNMENT);
}

void simd_free_uint64_array(uint64_t* ptr, size_t num_elements) {
    simd_aligned_free(ptr, num_elements * sizeof(uint64_t));
}

// ============================================================================
// MEMORY UTILITIES
// ============================================================================

int simd_is_aligned(const void* ptr, size_t alignment) {
    if (!ptr) return 0;
    if (!is_valid_alignment(alignment)) return 0;
    return ((uintptr_t)ptr & (alignment - 1)) == 0;
}

size_t simd_get_optimal_alignment(void) {
#if defined(ARCH_X86)
    // Detect AVX-512, AVX2, or SSE
    unsigned int eax, ebx, ecx, edx;

    // Check for AVX-512 (requires CPUID leaf 7)
    #ifdef _MSC_VER
        int info[4];
        __cpuidex(info, 7, 0);
        ebx = info[1];
    #else
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            // ebx bit 16 = AVX-512F
            if (ebx & (1 << 16)) {
                return SIMD_ALIGN_64;  // AVX-512 needs 64-byte alignment
            }
        }
    #endif

    // Check for AVX2 (CPUID leaf 7, ebx bit 5)
    #ifdef _MSC_VER
        __cpuid(info, 1);
        ecx = info[2];
    #else
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
            // Check AVX (ecx bit 28)
            if (ecx & (1 << 28)) {
                return SIMD_ALIGN_32;  // AVX/AVX2 optimal at 32-byte
            }
        }
    #endif

    // SSE2 is baseline for x86-64
    return SIMD_ALIGN_16;

#elif defined(ARCH_ARM64)
    #ifdef __linux__
        // Check for SVE support
        unsigned long hwcap = getauxval(AT_HWCAP);
        if (hwcap & HWCAP_SVE) {
            // SVE vector length is variable, use max common alignment
            return SIMD_ALIGN_64;
        }
    #endif

    #ifdef __APPLE__
        // Apple Silicon always has 64-byte cache lines and AMX
        return SIMD_ALIGN_64;
    #endif

    // ARM NEON is 16-byte aligned
    return SIMD_ALIGN_16;

#else
    // Default: 64-byte for cache line alignment
    return SIMD_ALIGN_64;
#endif
}

void simd_aligned_memcpy(void* dest, const void* src, size_t size, size_t alignment) {
    if (!dest || !src || size == 0) return;

    // Verify alignment for optimized path
    if (simd_is_aligned(dest, alignment) && simd_is_aligned(src, alignment)) {
        // Both aligned - use standard memcpy which will use SIMD internally
        // Modern libc implementations (glibc, musl, Apple libSystem) are highly optimized
        memcpy(dest, src, size);
    } else {
        // Unaligned - still use memcpy, but won't be as fast
        memcpy(dest, src, size);
    }
}

void simd_aligned_memset(void* ptr, int value, size_t size, size_t alignment) {
    (void)alignment;  // Reserved for future SIMD-specific optimization
    if (!ptr || size == 0) return;

    // Standard memset is highly optimized in modern libc
    // It will use SIMD when possible
    memset(ptr, value, size);
}

// ============================================================================
// PLATFORM INFORMATION
// ============================================================================

mem_platform_info_t simd_get_platform_info(void) {
    mem_platform_info_t info = {0};

#if defined(PLATFORM_MACOS)
    info.platform = "macOS";
    info.allocator = "posix_memalign";
    info.default_alignment = SIMD_DEFAULT_ALIGNMENT;
    info.page_size = (size_t)sysconf(_SC_PAGESIZE);
    info.supports_huge_pages = 0;  // macOS doesn't expose huge pages

#elif defined(PLATFORM_LINUX)
    info.platform = "Linux";
    // Check if aligned_alloc would work (C11)
    #if __STDC_VERSION__ >= 201112L
        info.allocator = "aligned_alloc (C11)";
    #else
        info.allocator = "posix_memalign";
    #endif
    info.default_alignment = SIMD_DEFAULT_ALIGNMENT;
    info.page_size = (size_t)sysconf(_SC_PAGESIZE);
    // Check for transparent huge pages
    info.supports_huge_pages = 1;  // THP available on most Linux

#elif defined(PLATFORM_WINDOWS)
    info.platform = "Windows";
    info.allocator = "_aligned_malloc";
    info.default_alignment = SIMD_DEFAULT_ALIGNMENT;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    info.page_size = si.dwPageSize;
    info.supports_huge_pages = 1;  // Large pages available

#elif defined(PLATFORM_FREEBSD)
    info.platform = "FreeBSD";
    info.allocator = "posix_memalign";
    info.default_alignment = SIMD_DEFAULT_ALIGNMENT;
    info.page_size = (size_t)sysconf(_SC_PAGESIZE);
    info.supports_huge_pages = 1;  // Super pages available

#else
    info.platform = "Unknown";
    info.allocator = "malloc with manual alignment";
    info.default_alignment = SIMD_DEFAULT_ALIGNMENT;
    info.page_size = 4096;  // Assume 4KB pages
    info.supports_huge_pages = 0;
#endif

    return info;
}

// ============================================================================
// ALLOCATION TRACKING (Debug builds only)
// ============================================================================

#ifdef MEMORY_ALIGN_TRACK_STATS

#include <stdatomic.h>

// Thread-safe atomic counters
static atomic_uint_fast64_t g_total_allocated = 0;
static atomic_uint_fast64_t g_peak_allocated = 0;
static atomic_uint_fast64_t g_allocation_count = 0;
static atomic_uint_fast64_t g_total_allocations = 0;
static atomic_uint_fast64_t g_total_frees = 0;
static atomic_uint_fast64_t g_allocation_failures = 0;

static void track_allocation(size_t size) {
    atomic_fetch_add(&g_total_allocations, 1);
    atomic_fetch_add(&g_allocation_count, 1);

    uint64_t new_total = atomic_fetch_add(&g_total_allocated, size) + size;

    // Update peak (CAS loop for thread safety)
    uint64_t peak = atomic_load(&g_peak_allocated);
    while (new_total > peak) {
        if (atomic_compare_exchange_weak(&g_peak_allocated, &peak, new_total)) {
            break;
        }
    }
}

static void track_deallocation(size_t size) {
    atomic_fetch_add(&g_total_frees, 1);
    atomic_fetch_sub(&g_allocation_count, 1);
    atomic_fetch_sub(&g_total_allocated, size);
}

static void track_allocation_failure(void) {
    atomic_fetch_add(&g_allocation_failures, 1);
}

void simd_get_alloc_stats(mem_alloc_stats_t* stats) {
    if (!stats) return;

    stats->total_allocated = atomic_load(&g_total_allocated);
    stats->peak_allocated = atomic_load(&g_peak_allocated);
    stats->allocation_count = atomic_load(&g_allocation_count);
    stats->total_allocations = atomic_load(&g_total_allocations);
    stats->total_frees = atomic_load(&g_total_frees);
    stats->allocation_failures = atomic_load(&g_allocation_failures);
}

void simd_reset_alloc_stats(void) {
    atomic_store(&g_total_allocated, 0);
    atomic_store(&g_peak_allocated, 0);
    atomic_store(&g_allocation_count, 0);
    atomic_store(&g_total_allocations, 0);
    atomic_store(&g_total_frees, 0);
    atomic_store(&g_allocation_failures, 0);
}

void simd_print_alloc_stats(void) {
    mem_alloc_stats_t stats;
    simd_get_alloc_stats(&stats);

    fprintf(stderr, "\n=== Memory Allocation Statistics ===\n");
    fprintf(stderr, "Current allocated:   %llu bytes\n", (unsigned long long)stats.total_allocated);
    fprintf(stderr, "Peak allocated:      %llu bytes\n", (unsigned long long)stats.peak_allocated);
    fprintf(stderr, "Active allocations:  %llu\n", (unsigned long long)stats.allocation_count);
    fprintf(stderr, "Total allocations:   %llu\n", (unsigned long long)stats.total_allocations);
    fprintf(stderr, "Total frees:         %llu\n", (unsigned long long)stats.total_frees);
    fprintf(stderr, "Allocation failures: %llu\n", (unsigned long long)stats.allocation_failures);

    if (stats.allocation_count != stats.total_allocations - stats.total_frees) {
        fprintf(stderr, "WARNING: Allocation count mismatch (possible memory leak)\n");
    }
    fprintf(stderr, "====================================\n\n");
}

#else

// Empty stubs when tracking is disabled
static void track_allocation(size_t size) { (void)size; }
static void track_deallocation(size_t size) { (void)size; }
static void track_allocation_failure(void) {}

#endif /* MEMORY_ALIGN_TRACK_STATS */
