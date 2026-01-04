/**
 * @file simd_dispatch.c
 * @brief Runtime SIMD capability detection and dispatch implementation
 *
 * Comprehensive CPU feature detection for x86-64 and ARM64 architectures.
 * Supports AVX-512, AVX2, AVX, SSE, ARM NEON, and ARM SVE.
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#include "simd_dispatch.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Platform-specific includes
#ifdef SIMD_ARCH_X86
    #ifdef _MSC_VER
        #include <intrin.h>
    #else
        #include <cpuid.h>
    #endif
    // For xgetbv on non-MSVC
    #if !defined(_MSC_VER) && defined(__GNUC__)
        #include <immintrin.h>
    #endif
#endif

#ifdef SIMD_ARCH_ARM64
    #if defined(__linux__)
        #include <sys/auxv.h>
        #if __has_include(<asm/hwcap.h>)
            #include <asm/hwcap.h>
        #endif
        // Define HWCAP flags if not present
        #ifndef HWCAP_NEON
            #define HWCAP_NEON (1 << 12)
        #endif
        #ifndef HWCAP_ASIMD
            #define HWCAP_ASIMD (1 << 1)
        #endif
        #ifndef HWCAP_SVE
            #define HWCAP_SVE (1 << 22)
        #endif
        #ifndef HWCAP2_SVE2
            #define HWCAP2_SVE2 (1 << 1)
        #endif
    #elif defined(__APPLE__)
        #include <sys/sysctl.h>
    #endif
#endif

// Thread-safe initialization
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
    #include <stdatomic.h>
    static atomic_int g_simd_initialized = 0;
#else
    static volatile int g_simd_initialized = 0;
#endif

// Global cached SIMD info
static simd_info_t g_simd_info;
static char g_capability_string[256];

// ============================================================================
// x86-64 CPUID HELPERS
// ============================================================================

#ifdef SIMD_ARCH_X86

/**
 * @brief Execute CPUID instruction
 */
static void cpuid(uint32_t leaf, uint32_t subleaf,
                  uint32_t* eax, uint32_t* ebx, uint32_t* ecx, uint32_t* edx) {
#ifdef _MSC_VER
    int info[4];
    __cpuidex(info, (int)leaf, (int)subleaf);
    *eax = (uint32_t)info[0];
    *ebx = (uint32_t)info[1];
    *ecx = (uint32_t)info[2];
    *edx = (uint32_t)info[3];
#else
    __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
#endif
}

/**
 * @brief Check if OS supports AVX (via XGETBV)
 */
static int os_supports_avx(void) {
#ifdef _MSC_VER
    // MSVC has _xgetbv intrinsic
    unsigned long long xcr0 = _xgetbv(0);
    return ((xcr0 & 0x6) == 0x6);  // XMM and YMM state
#elif defined(__GNUC__) || defined(__clang__)
    // Check OSXSAVE first
    uint32_t eax, ebx, ecx, edx;
    cpuid(1, 0, &eax, &ebx, &ecx, &edx);
    if (!(ecx & (1 << 27))) return 0;  // OSXSAVE not set

    // Get XCR0 via xgetbv
    uint32_t xcr0_lo, xcr0_hi;
    __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));

    return ((xcr0_lo & 0x6) == 0x6);  // XMM and YMM state
#else
    return 0;
#endif
}

/**
 * @brief Check if OS supports AVX-512 (via XGETBV)
 */
static int os_supports_avx512(void) {
#ifdef _MSC_VER
    unsigned long long xcr0 = _xgetbv(0);
    // Check XMM, YMM, and ZMM state, plus opmask
    return ((xcr0 & 0xE6) == 0xE6);
#elif defined(__GNUC__) || defined(__clang__)
    uint32_t eax, ebx, ecx, edx;
    cpuid(1, 0, &eax, &ebx, &ecx, &edx);
    if (!(ecx & (1 << 27))) return 0;

    uint32_t xcr0_lo, xcr0_hi;
    __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));

    return ((xcr0_lo & 0xE6) == 0xE6);
#else
    return 0;
#endif
}

/**
 * @brief Detect x86-64 SIMD capabilities
 */
static void detect_x86_capabilities(simd_info_t* info) {
    uint32_t eax, ebx, ecx, edx;

    // CPUID leaf 0: Get vendor and max leaf
    uint32_t max_leaf;
    cpuid(0, 0, &max_leaf, &ebx, &ecx, &edx);

    if (max_leaf < 1) return;

    // CPUID leaf 1: Basic features
    cpuid(1, 0, &eax, &ebx, &ecx, &edx);

    // EDX flags
    info->has_sse2 = (edx >> 26) & 1;

    // ECX flags
    info->has_sse3   = (ecx >> 0) & 1;
    info->has_ssse3  = (ecx >> 9) & 1;
    info->has_sse4_1 = (ecx >> 19) & 1;
    info->has_sse4_2 = (ecx >> 20) & 1;
    info->has_fma    = (ecx >> 12) & 1;
    int has_avx_bit  = (ecx >> 28) & 1;

    // Update flags
    if (info->has_sse2)   info->flags |= SIMD_CAP_SSE2;
    if (info->has_sse3)   info->flags |= SIMD_CAP_SSE3;
    if (info->has_ssse3)  info->flags |= SIMD_CAP_SSSE3;
    if (info->has_sse4_1) info->flags |= SIMD_CAP_SSE4_1;
    if (info->has_sse4_2) info->flags |= SIMD_CAP_SSE4_2;

    // AVX requires OS support (XSAVE for YMM)
    if (has_avx_bit && os_supports_avx()) {
        info->has_avx = 1;
        info->flags |= SIMD_CAP_AVX;

        if (info->has_fma) {
            info->flags |= SIMD_CAP_FMA;
        }
    }

    // CPUID leaf 7: Extended features
    if (max_leaf >= 7) {
        cpuid(7, 0, &eax, &ebx, &ecx, &edx);

        // AVX2 (ebx bit 5)
        if ((ebx >> 5) & 1 && info->has_avx) {
            info->has_avx2 = 1;
            info->flags |= SIMD_CAP_AVX2;
        }

        // AVX-512 requires OS support
        if (os_supports_avx512()) {
            // AVX-512F (ebx bit 16)
            if ((ebx >> 16) & 1) {
                info->has_avx512f = 1;
                info->flags |= SIMD_CAP_AVX512F;
            }

            // AVX-512DQ (ebx bit 17)
            if ((ebx >> 17) & 1) {
                info->has_avx512dq = 1;
                info->flags |= SIMD_CAP_AVX512DQ;
            }

            // AVX-512BW (ebx bit 30)
            if ((ebx >> 30) & 1) {
                info->has_avx512bw = 1;
                info->flags |= SIMD_CAP_AVX512BW;
            }

            // AVX-512VL (ebx bit 31)
            if ((ebx >> 31) & 1) {
                info->has_avx512vl = 1;
                info->flags |= SIMD_CAP_AVX512VL;
            }

            // AVX-512CD (ebx bit 28)
            if ((ebx >> 28) & 1) {
                info->has_avx512cd = 1;
                info->flags |= SIMD_CAP_AVX512CD;
            }

            // AVX-512VNNI (ecx bit 11)
            if ((ecx >> 11) & 1) {
                info->has_avx512vnni = 1;
                info->flags |= SIMD_CAP_AVX512VNNI;
            }
        }
    }

    // Determine SIMD level
    if (info->has_avx512f) {
        info->level = SIMD_LEVEL_AVX512;
    } else if (info->has_avx2 && info->has_fma) {
        info->level = SIMD_LEVEL_AVX2;
    } else if (info->has_avx) {
        info->level = SIMD_LEVEL_AVX;
    } else if (info->has_sse2) {
        info->level = SIMD_LEVEL_SSE2;
    } else {
        info->level = SIMD_LEVEL_SCALAR;
    }

    // Get cache info from CPUID leaf 4
    if (max_leaf >= 4) {
        // L1 cache (type 1)
        cpuid(4, 0, &eax, &ebx, &ecx, &edx);
        if ((eax & 0x1F) != 0) {
            uint32_t ways = ((ebx >> 22) & 0x3FF) + 1;
            uint32_t partitions = ((ebx >> 12) & 0x3FF) + 1;
            uint32_t line_size = (ebx & 0xFFF) + 1;
            uint32_t sets = ecx + 1;
            info->l1_cache_size = ways * partitions * line_size * sets;
            info->cache_line_size = line_size;
        }

        // L2 cache (type 2 or 3)
        cpuid(4, 2, &eax, &ebx, &ecx, &edx);
        if ((eax & 0x1F) != 0) {
            uint32_t ways = ((ebx >> 22) & 0x3FF) + 1;
            uint32_t partitions = ((ebx >> 12) & 0x3FF) + 1;
            uint32_t line_size = (ebx & 0xFFF) + 1;
            uint32_t sets = ecx + 1;
            info->l2_cache_size = ways * partitions * line_size * sets;
        }
    }

    // Default cache line size if not detected
    if (info->cache_line_size == 0) {
        info->cache_line_size = 64;
    }
}

#endif /* SIMD_ARCH_X86 */

// ============================================================================
// ARM64 DETECTION
// ============================================================================

#ifdef SIMD_ARCH_ARM64

/**
 * @brief Get SVE vector length in bits
 */
static uint32_t get_sve_vector_length(void) {
#if defined(__linux__) && defined(__ARM_FEATURE_SVE)
    // Use inline assembly to read SVE vector length
    uint64_t vl;
    __asm__ volatile("rdvl %0, #1" : "=r"(vl));
    return (uint32_t)(vl * 8);  // Convert bytes to bits
#else
    return 0;
#endif
}

/**
 * @brief Detect ARM64 SIMD capabilities
 */
static void detect_arm64_capabilities(simd_info_t* info) {
#if defined(__linux__)
    // Get hardware capabilities from kernel
    unsigned long hwcap = getauxval(AT_HWCAP);
    unsigned long hwcap2 = 0;
#ifdef AT_HWCAP2
    hwcap2 = getauxval(AT_HWCAP2);
#endif

    // NEON is mandatory on AArch64
    info->has_neon = 1;
    info->flags |= SIMD_CAP_NEON;

    // Check for SVE
    if (hwcap & HWCAP_SVE) {
        info->has_sve = 1;
        info->flags |= SIMD_CAP_SVE;
        info->sve_vector_length = get_sve_vector_length();
    }

    // Check for SVE2
    if (hwcap2 & HWCAP2_SVE2) {
        info->has_sve2 = 1;
        info->flags |= SIMD_CAP_SVE2;
    }

    // Dot product instructions
    #ifdef HWCAP_ASIMDDP
    if (hwcap & HWCAP_ASIMDDP) {
        info->has_dotprod = 1;
        info->flags |= SIMD_CAP_DOTPROD;
    }
    #endif

#elif defined(__APPLE__)
    // Apple Silicon always has NEON and AMX
    info->has_neon = 1;
    info->flags |= SIMD_CAP_NEON;

    // Apple Silicon has AMX matrix accelerator
    info->has_amx = 1;
    info->flags |= SIMD_CAP_AMX;

    // Apple doesn't expose SVE
    info->has_sve = 0;
    info->has_sve2 = 0;
#endif

    // Determine SIMD level
    if (info->has_sve2) {
        info->level = SIMD_LEVEL_SVE2;
    } else if (info->has_sve) {
        info->level = SIMD_LEVEL_SVE;
    } else if (info->has_neon) {
        info->level = SIMD_LEVEL_NEON;
    } else {
        info->level = SIMD_LEVEL_SCALAR;
    }

    // ARM cache line is typically 64 bytes
    info->cache_line_size = 64;

#if defined(__APPLE__)
    // Get cache sizes from sysctl on macOS
    // Apple Silicon uses perflevel caches, try multiple sysctl names
    size_t size = sizeof(uint32_t);
    uint64_t val64;
    size_t size64 = sizeof(uint64_t);

    // Try performance level cache sizes first (Apple Silicon)
    if (sysctlbyname("hw.perflevel0.l1dcachesize", &val64, &size64, NULL, 0) == 0) {
        info->l1_cache_size = (uint32_t)val64;
    } else if (sysctlbyname("hw.l1dcachesize", &info->l1_cache_size, &size, NULL, 0) != 0) {
        info->l1_cache_size = 128 * 1024;  // Default 128KB for M-series
    }

    size64 = sizeof(uint64_t);
    if (sysctlbyname("hw.perflevel0.l2cachesize", &val64, &size64, NULL, 0) == 0) {
        info->l2_cache_size = (uint32_t)val64;
    } else if (sysctlbyname("hw.l2cachesize", &info->l2_cache_size, &size, NULL, 0) != 0) {
        info->l2_cache_size = 4 * 1024 * 1024;  // Default 4MB for M-series
    }

    // Get cache line size
    size = sizeof(uint32_t);
    uint32_t linesize;
    if (sysctlbyname("hw.cachelinesize", &linesize, &size, NULL, 0) == 0) {
        info->cache_line_size = linesize;
    }
#endif
}

#endif /* SIMD_ARCH_ARM64 */

// ============================================================================
// MAIN DETECTION FUNCTION
// ============================================================================

const simd_info_t* simd_detect_capabilities_full(void) {
    // Thread-safe initialization using compare-and-exchange
    // States: 0 = uninitialized, 1 = initialized, 2 = initializing
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
    int expected = 0;
    if (atomic_compare_exchange_strong(&g_simd_initialized, &expected, 2)) {
        // We won the race - do initialization
        // (fall through to initialization code below)
    } else if (expected == 1) {
        // Already initialized by another thread
        atomic_thread_fence(memory_order_acquire);
        return &g_simd_info;
    } else {
        // Another thread is initializing (expected == 2), spin until done
        while (atomic_load_explicit(&g_simd_initialized, memory_order_acquire) != 1) {
            // Spin-wait (could add sched_yield for better behavior)
        }
        return &g_simd_info;
    }
#else
    if (g_simd_initialized) {
        return &g_simd_info;
    }
    g_simd_initialized = 2;  // Mark as initializing (best effort without atomics)
#endif

    // Initialize structure
    memset(&g_simd_info, 0, sizeof(simd_info_t));

    // Detect based on architecture
#ifdef SIMD_ARCH_X86
    detect_x86_capabilities(&g_simd_info);
#elif defined(SIMD_ARCH_ARM64)
    detect_arm64_capabilities(&g_simd_info);
#else
    // Unknown architecture - scalar only
    g_simd_info.level = SIMD_LEVEL_SCALAR;
#endif

    // Build capability string
    g_capability_string[0] = '\0';
    char* p = g_capability_string;
    size_t remaining = sizeof(g_capability_string);

#ifdef SIMD_ARCH_X86
    if (g_simd_info.has_avx512f) {
        int written = snprintf(p, remaining, "AVX-512");
        p += written; remaining -= written;

        // List AVX-512 subsets
        if (g_simd_info.has_avx512bw || g_simd_info.has_avx512dq || g_simd_info.has_avx512vl) {
            written = snprintf(p, remaining, " (");
            p += written; remaining -= written;

            int first = 1;
            if (g_simd_info.has_avx512f) {
                written = snprintf(p, remaining, "F"); p += written; remaining -= written; first = 0;
            }
            if (g_simd_info.has_avx512bw) {
                written = snprintf(p, remaining, "%sBW", first ? "" : ", "); p += written; remaining -= written; first = 0;
            }
            if (g_simd_info.has_avx512dq) {
                written = snprintf(p, remaining, "%sDQ", first ? "" : ", "); p += written; remaining -= written; first = 0;
            }
            if (g_simd_info.has_avx512vl) {
                written = snprintf(p, remaining, "%sVL", first ? "" : ", "); p += written; remaining -= written; first = 0;
            }

            written = snprintf(p, remaining, ")"); p += written; remaining -= written;
        }

        if (g_simd_info.has_fma) {
            written = snprintf(p, remaining, " + FMA"); p += written; remaining -= written;
        }
    } else if (g_simd_info.has_avx2) {
        int written = snprintf(p, remaining, "AVX2");
        p += written; remaining -= written;
        if (g_simd_info.has_fma) {
            written = snprintf(p, remaining, " + FMA"); p += written; remaining -= written;
        }
    } else if (g_simd_info.has_avx) {
        snprintf(p, remaining, "AVX");
    } else if (g_simd_info.has_sse4_1) {
        snprintf(p, remaining, "SSE4.1");
    } else if (g_simd_info.has_sse2) {
        snprintf(p, remaining, "SSE2");
    } else {
        snprintf(p, remaining, "Scalar");
    }
#elif defined(SIMD_ARCH_ARM64)
    if (g_simd_info.has_sve2) {
        snprintf(p, remaining, "SVE2 (%u-bit) + NEON", g_simd_info.sve_vector_length);
    } else if (g_simd_info.has_sve) {
        snprintf(p, remaining, "SVE (%u-bit) + NEON", g_simd_info.sve_vector_length);
    } else if (g_simd_info.has_neon) {
        if (g_simd_info.has_amx) {
            snprintf(p, remaining, "NEON + AMX (Apple Silicon)");
        } else {
            snprintf(p, remaining, "NEON");
        }
    } else {
        snprintf(p, remaining, "Scalar");
    }
#else
    snprintf(p, remaining, "Scalar (unknown architecture)");
#endif

    // Mark as initialized with release semantics
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
    atomic_thread_fence(memory_order_release);
    atomic_store_explicit(&g_simd_initialized, 1, memory_order_release);
#else
    g_simd_initialized = 1;
#endif

    return &g_simd_info;
}

// ============================================================================
// ACCESSOR FUNCTIONS
// ============================================================================

uint32_t simd_get_capability_flags(void) {
    const simd_info_t* info = simd_detect_capabilities_full();
    return info->flags;
}

int simd_has_capability(simd_capability_t cap) {
    const simd_info_t* info = simd_detect_capabilities_full();
    return (info->flags & cap) != 0;
}

simd_level_t simd_get_level(void) {
    const simd_info_t* info = simd_detect_capabilities_full();
    return info->level;
}

const char* simd_get_capability_string(void) {
    simd_detect_capabilities_full();  // Ensure initialized
    return g_capability_string;
}

// ============================================================================
// DISPATCH FUNCTIONS
// ============================================================================

simd_backend_t simd_get_backend(simd_operation_t op) {
    const simd_info_t* info = simd_detect_capabilities_full();
    (void)op;  // Currently all operations use same backend selection

#ifdef SIMD_ARCH_X86
    if (info->has_avx512f) return SIMD_BACKEND_AVX512;
    if (info->has_avx2)    return SIMD_BACKEND_AVX2;
    if (info->has_avx)     return SIMD_BACKEND_AVX;
    if (info->has_sse2)    return SIMD_BACKEND_SSE2;
    return SIMD_BACKEND_SCALAR;
#elif defined(SIMD_ARCH_ARM64)
    #ifdef __APPLE__
    // Prefer Accelerate on Apple Silicon
    return SIMD_BACKEND_ACCELERATE;
    #else
    if (info->has_sve2) return SIMD_BACKEND_SVE2;
    if (info->has_sve)  return SIMD_BACKEND_SVE;
    if (info->has_neon) return SIMD_BACKEND_NEON;
    return SIMD_BACKEND_SCALAR;
    #endif
#else
    (void)info;
    return SIMD_BACKEND_SCALAR;
#endif
}

const char* simd_backend_name(simd_backend_t backend) {
    switch (backend) {
        case SIMD_BACKEND_SCALAR:     return "Scalar";
        case SIMD_BACKEND_SSE2:       return "SSE2";
        case SIMD_BACKEND_AVX:        return "AVX";
        case SIMD_BACKEND_AVX2:       return "AVX2";
        case SIMD_BACKEND_AVX512:     return "AVX-512";
        case SIMD_BACKEND_NEON:       return "NEON";
        case SIMD_BACKEND_SVE:        return "SVE";
        case SIMD_BACKEND_SVE2:       return "SVE2";
        case SIMD_BACKEND_ACCELERATE: return "Accelerate";
        default:                      return "Unknown";
    }
}

// ============================================================================
// VECTOR WIDTH FUNCTIONS
// ============================================================================

size_t simd_get_vector_width(void) {
    const simd_info_t* info = simd_detect_capabilities_full();

#ifdef SIMD_ARCH_X86
    if (info->has_avx512f) return 64;  // 512 bits = 64 bytes
    if (info->has_avx)     return 32;  // 256 bits = 32 bytes
    if (info->has_sse2)    return 16;  // 128 bits = 16 bytes
    return 8;  // 64-bit scalar
#elif defined(SIMD_ARCH_ARM64)
    if (info->sve_vector_length > 0) return info->sve_vector_length / 8;
    if (info->has_neon) return 16;  // 128 bits
    return 8;
#else
    (void)info;
    return 8;
#endif
}

size_t simd_get_doubles_per_register(void) {
    return simd_get_vector_width() / sizeof(double);
}

size_t simd_get_complex_per_register(void) {
    // complex_t is 16 bytes (two doubles)
    size_t width = simd_get_vector_width();
    size_t complex_size = 16;
    return (width >= complex_size) ? width / complex_size : 1;
}

// ============================================================================
// LOOP OPTIMIZATION
// ============================================================================

size_t simd_get_unroll_factor(void) {
    const simd_info_t* info = simd_detect_capabilities_full();

#ifdef SIMD_ARCH_X86
    if (info->has_avx512f) return 8;  // 8 doubles per register, unroll 2x
    if (info->has_avx)     return 4;
    return 4;
#elif defined(SIMD_ARCH_ARM64)
    if (info->has_sve) return 8;
    return 4;
#else
    (void)info;
    return 4;
#endif
}

size_t simd_get_min_array_size(void) {
    // Below this size, SIMD overhead may exceed benefit
    return 16;
}

size_t simd_get_chunk_size(size_t element_size) {
    const simd_info_t* info = simd_detect_capabilities_full();

    // Target: fit in L1 cache, leave room for working data
    size_t target_bytes = info->l1_cache_size > 0 ? info->l1_cache_size / 4 : 8192;

    // Ensure alignment to vector width
    size_t vector_width = simd_get_vector_width();
    size_t elements_per_vector = vector_width / element_size;
    if (elements_per_vector == 0) elements_per_vector = 1;

    size_t chunk = target_bytes / element_size;

    // Round down to multiple of vector elements
    chunk = (chunk / elements_per_vector) * elements_per_vector;

    // Minimum reasonable chunk
    if (chunk < 64) chunk = 64;

    return chunk;
}

// ============================================================================
// DEBUG OUTPUT
// ============================================================================

void simd_print_capabilities(void) {
    const simd_info_t* info = simd_detect_capabilities_full();

    fprintf(stderr, "\n");
    fprintf(stderr, "╔══════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║                 SIMD CAPABILITY REPORT                    ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════════════════════╝\n");
    fprintf(stderr, "\n");

    fprintf(stderr, "Architecture: %s\n", SIMD_ARCH_NAME);
    fprintf(stderr, "SIMD Level:   %d (%s)\n", info->level, simd_get_capability_string());
    fprintf(stderr, "\n");

#ifdef SIMD_ARCH_X86
    fprintf(stderr, "x86-64 Features:\n");
    fprintf(stderr, "  SSE2:      %s\n", info->has_sse2 ? "Yes" : "No");
    fprintf(stderr, "  SSE3:      %s\n", info->has_sse3 ? "Yes" : "No");
    fprintf(stderr, "  SSSE3:     %s\n", info->has_ssse3 ? "Yes" : "No");
    fprintf(stderr, "  SSE4.1:    %s\n", info->has_sse4_1 ? "Yes" : "No");
    fprintf(stderr, "  SSE4.2:    %s\n", info->has_sse4_2 ? "Yes" : "No");
    fprintf(stderr, "  AVX:       %s\n", info->has_avx ? "Yes" : "No");
    fprintf(stderr, "  AVX2:      %s\n", info->has_avx2 ? "Yes" : "No");
    fprintf(stderr, "  FMA:       %s\n", info->has_fma ? "Yes" : "No");
    fprintf(stderr, "  AVX-512F:  %s\n", info->has_avx512f ? "Yes" : "No");
    fprintf(stderr, "  AVX-512BW: %s\n", info->has_avx512bw ? "Yes" : "No");
    fprintf(stderr, "  AVX-512DQ: %s\n", info->has_avx512dq ? "Yes" : "No");
    fprintf(stderr, "  AVX-512VL: %s\n", info->has_avx512vl ? "Yes" : "No");
#elif defined(SIMD_ARCH_ARM64)
    fprintf(stderr, "ARM64 Features:\n");
    fprintf(stderr, "  NEON:     %s\n", info->has_neon ? "Yes" : "No");
    fprintf(stderr, "  SVE:      %s\n", info->has_sve ? "Yes" : "No");
    if (info->has_sve) {
        fprintf(stderr, "  SVE VL:   %u bits\n", info->sve_vector_length);
    }
    fprintf(stderr, "  SVE2:     %s\n", info->has_sve2 ? "Yes" : "No");
    fprintf(stderr, "  DOT:      %s\n", info->has_dotprod ? "Yes" : "No");
    fprintf(stderr, "  AMX:      %s\n", info->has_amx ? "Yes" : "No");
#endif

    fprintf(stderr, "\n");
    fprintf(stderr, "Performance Parameters:\n");
    fprintf(stderr, "  Vector width:       %zu bytes (%zu doubles)\n",
            simd_get_vector_width(), simd_get_doubles_per_register());
    fprintf(stderr, "  Complex/register:   %zu\n", simd_get_complex_per_register());
    fprintf(stderr, "  Unroll factor:      %zu\n", simd_get_unroll_factor());
    fprintf(stderr, "  Cache line size:    %u bytes\n", info->cache_line_size);
    fprintf(stderr, "  L1 cache size:      %u KB\n", info->l1_cache_size / 1024);
    fprintf(stderr, "  L2 cache size:      %u KB\n", info->l2_cache_size / 1024);
    fprintf(stderr, "\n");

    fprintf(stderr, "Selected Backend: %s\n", simd_backend_name(simd_get_backend(SIMD_OP_SUM_SQUARED_MAG)));
    fprintf(stderr, "\n");
}

int simd_validate(void) {
    // Quick validation - just ensure detection doesn't crash
    const simd_info_t* info = simd_detect_capabilities_full();
    if (!info) return 0;

    // Check consistency
    if (info->level > SIMD_LEVEL_AVX512 && info->level != SIMD_LEVEL_SVE2) {
        return 0;  // Invalid level
    }

    // Verify string generation
    const char* str = simd_get_capability_string();
    if (!str || strlen(str) == 0) {
        return 0;
    }

    return 1;
}
