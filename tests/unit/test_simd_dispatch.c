/**
 * @file test_simd_dispatch.c
 * @brief Unit tests for SIMD dispatch layer
 *
 * Tests CPU capability detection, backend selection, and configuration.
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../src/optimization/simd_dispatch.h"

// ============================================================================
// TEST UTILITIES
// ============================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(condition, message) do { \
    if (!(condition)) { \
        printf("  [FAIL] %s\n", message); \
        printf("         at %s:%d\n", __FILE__, __LINE__); \
        tests_failed++; \
        return 0; \
    } \
} while(0)

#define TEST_PASS(message) do { \
    printf("  [PASS] %s\n", message); \
    tests_passed++; \
} while(0)

#define RUN_TEST(test_func) do { \
    printf("\nRunning: %s\n", #test_func); \
    if (test_func()) { \
        printf("  Test passed.\n"); \
    } else { \
        printf("  Test FAILED.\n"); \
    } \
} while(0)

// ============================================================================
// DETECTION TESTS
// ============================================================================

static int test_capability_detection(void) {
    const simd_info_t* info = simd_detect_capabilities_full();

    TEST_ASSERT(info != NULL, "Detection should return valid pointer");

    // Level should be valid
    TEST_ASSERT(info->level >= SIMD_LEVEL_SCALAR && info->level <= 4,
                "SIMD level should be in valid range");

    // On x86-64, SSE2 is mandatory
#ifdef SIMD_ARCH_X86
    TEST_ASSERT(info->has_sse2, "SSE2 should be detected on x86-64");
    TEST_ASSERT(info->flags & SIMD_CAP_SSE2, "SSE2 flag should be set");
#endif

    // On ARM64, NEON is mandatory
#ifdef SIMD_ARCH_ARM64
    TEST_ASSERT(info->has_neon, "NEON should be detected on ARM64");
    TEST_ASSERT(info->flags & SIMD_CAP_NEON, "NEON flag should be set");
#endif

    TEST_PASS("Capability detection test passed");
    return 1;
}

static int test_capability_flags(void) {
    uint32_t flags = simd_get_capability_flags();

    // At least one capability should be detected
    TEST_ASSERT(flags != 0, "At least one SIMD capability should be detected");

    // Test individual capability checks
#ifdef SIMD_ARCH_X86
    if (simd_has_capability(SIMD_CAP_AVX2)) {
        TEST_ASSERT(simd_has_capability(SIMD_CAP_AVX), "AVX2 implies AVX");
        TEST_ASSERT(simd_has_capability(SIMD_CAP_SSE2), "AVX implies SSE2");
    }
    if (simd_has_capability(SIMD_CAP_AVX512F)) {
        TEST_ASSERT(simd_has_capability(SIMD_CAP_AVX2), "AVX-512 implies AVX2");
    }
#endif

#ifdef SIMD_ARCH_ARM64
    if (simd_has_capability(SIMD_CAP_SVE)) {
        TEST_ASSERT(simd_has_capability(SIMD_CAP_NEON), "SVE implies NEON");
    }
#endif

    TEST_PASS("Capability flags test passed");
    return 1;
}

static int test_capability_string(void) {
    const char* str = simd_get_capability_string();

    TEST_ASSERT(str != NULL, "Capability string should not be NULL");
    TEST_ASSERT(strlen(str) > 0, "Capability string should not be empty");

    printf("    Detected: %s\n", str);

    // String should contain something reasonable
    int has_valid_content =
        strstr(str, "SSE") != NULL ||
        strstr(str, "AVX") != NULL ||
        strstr(str, "NEON") != NULL ||
        strstr(str, "SVE") != NULL ||
        strstr(str, "Scalar") != NULL;

    TEST_ASSERT(has_valid_content, "Capability string should contain known SIMD names");

    TEST_PASS("Capability string test passed");
    return 1;
}

static int test_simd_level(void) {
    simd_level_t level = simd_get_level();

    printf("    SIMD Level: %d\n", level);

    // Level should be consistent with detected capabilities
    const simd_info_t* info = simd_detect_capabilities_full();

#ifdef SIMD_ARCH_X86
    if (info->has_avx512f) {
        TEST_ASSERT(level >= SIMD_LEVEL_AVX512, "AVX-512 should give level 4");
    } else if (info->has_avx2) {
        TEST_ASSERT(level >= SIMD_LEVEL_AVX2, "AVX2 should give level 3");
    } else if (info->has_avx) {
        TEST_ASSERT(level >= SIMD_LEVEL_AVX, "AVX should give level 2");
    } else if (info->has_sse2) {
        TEST_ASSERT(level >= SIMD_LEVEL_SSE2, "SSE2 should give level 1");
    }
#endif

#ifdef SIMD_ARCH_ARM64
    if (info->has_neon) {
        TEST_ASSERT(level >= SIMD_LEVEL_NEON, "NEON should give level 1");
    }
#endif

    TEST_PASS("SIMD level test passed");
    return 1;
}

// ============================================================================
// BACKEND TESTS
// ============================================================================

static int test_backend_selection(void) {
    simd_backend_t backend = simd_get_backend(SIMD_OP_SUM_SQUARED_MAG);

    const char* name = simd_backend_name(backend);
    printf("    Selected backend: %s\n", name);

    TEST_ASSERT(name != NULL, "Backend name should not be NULL");
    TEST_ASSERT(strlen(name) > 0, "Backend name should not be empty");

    // Backend should match capabilities
    const simd_info_t* info = simd_detect_capabilities_full();

#ifdef SIMD_ARCH_X86
    if (info->has_avx512f) {
        TEST_ASSERT(backend == SIMD_BACKEND_AVX512, "AVX-512 capable should select AVX-512");
    } else if (info->has_avx2) {
        TEST_ASSERT(backend == SIMD_BACKEND_AVX2, "AVX2 capable should select AVX2");
    } else if (info->has_avx) {
        TEST_ASSERT(backend == SIMD_BACKEND_AVX, "AVX capable should select AVX");
    } else if (info->has_sse2) {
        TEST_ASSERT(backend == SIMD_BACKEND_SSE2, "SSE2 capable should select SSE2");
    }
#endif

#ifdef SIMD_ARCH_ARM64
    #ifdef __APPLE__
        TEST_ASSERT(backend == SIMD_BACKEND_ACCELERATE,
                    "Apple Silicon should select Accelerate backend");
    #else
        if (info->has_sve) {
            TEST_ASSERT(backend == SIMD_BACKEND_SVE || backend == SIMD_BACKEND_SVE2,
                        "SVE capable should select SVE backend");
        } else if (info->has_neon) {
            TEST_ASSERT(backend == SIMD_BACKEND_NEON,
                        "NEON capable should select NEON backend");
        }
    #endif
#endif

    TEST_PASS("Backend selection test passed");
    return 1;
}

static int test_all_backend_names(void) {
    // Verify all backend names are valid
    const char* names[] = {
        simd_backend_name(SIMD_BACKEND_SCALAR),
        simd_backend_name(SIMD_BACKEND_SSE2),
        simd_backend_name(SIMD_BACKEND_AVX),
        simd_backend_name(SIMD_BACKEND_AVX2),
        simd_backend_name(SIMD_BACKEND_AVX512),
        simd_backend_name(SIMD_BACKEND_NEON),
        simd_backend_name(SIMD_BACKEND_SVE),
        simd_backend_name(SIMD_BACKEND_SVE2),
        simd_backend_name(SIMD_BACKEND_ACCELERATE),
    };

    for (int i = 0; i < 9; i++) {
        TEST_ASSERT(names[i] != NULL, "Backend name should not be NULL");
        TEST_ASSERT(strlen(names[i]) > 0, "Backend name should not be empty");
    }

    TEST_PASS("All backend names test passed");
    return 1;
}

// ============================================================================
// VECTOR WIDTH TESTS
// ============================================================================

static int test_vector_width(void) {
    size_t width = simd_get_vector_width();

    printf("    Vector width: %zu bytes\n", width);

    TEST_ASSERT(width >= 8, "Vector width should be at least 8 bytes");
    TEST_ASSERT(width <= 256, "Vector width should be at most 256 bytes");

    // Width should be power of 2
    TEST_ASSERT((width & (width - 1)) == 0 || width == 0,
                "Vector width should be power of 2");

    // Check consistency with capabilities
    const simd_info_t* info = simd_detect_capabilities_full();

#ifdef SIMD_ARCH_X86
    if (info->has_avx512f) {
        TEST_ASSERT(width >= 64, "AVX-512 should have 64-byte vectors");
    } else if (info->has_avx) {
        TEST_ASSERT(width >= 32, "AVX should have 32-byte vectors");
    } else if (info->has_sse2) {
        TEST_ASSERT(width >= 16, "SSE2 should have 16-byte vectors");
    }
#endif

    TEST_PASS("Vector width test passed");
    return 1;
}

static int test_doubles_per_register(void) {
    size_t count = simd_get_doubles_per_register();

    printf("    Doubles per register: %zu\n", count);

    TEST_ASSERT(count >= 1, "Should have at least 1 double per register");
    TEST_ASSERT(count <= 32, "Should have at most 32 doubles per register");

    // Should match vector width
    size_t expected = simd_get_vector_width() / sizeof(double);
    TEST_ASSERT(count == expected, "Doubles count should match vector width");

    TEST_PASS("Doubles per register test passed");
    return 1;
}

static int test_complex_per_register(void) {
    size_t count = simd_get_complex_per_register();

    printf("    Complex per register: %zu\n", count);

    TEST_ASSERT(count >= 1, "Should have at least 1 complex per register");
    TEST_ASSERT(count <= 16, "Should have at most 16 complex per register");

    // Complex is 16 bytes (2 doubles)
    size_t width = simd_get_vector_width();
    size_t expected = (width >= 16) ? width / 16 : 1;
    TEST_ASSERT(count == expected, "Complex count should match vector width");

    TEST_PASS("Complex per register test passed");
    return 1;
}

// ============================================================================
// LOOP OPTIMIZATION TESTS
// ============================================================================

static int test_unroll_factor(void) {
    size_t factor = simd_get_unroll_factor();

    printf("    Unroll factor: %zu\n", factor);

    TEST_ASSERT(factor >= 1, "Unroll factor should be at least 1");
    TEST_ASSERT(factor <= 16, "Unroll factor should be at most 16");

    TEST_PASS("Unroll factor test passed");
    return 1;
}

static int test_min_array_size(void) {
    size_t min_size = simd_get_min_array_size();

    printf("    Min array size: %zu\n", min_size);

    TEST_ASSERT(min_size >= 1, "Min array size should be at least 1");
    TEST_ASSERT(min_size <= 1024, "Min array size should be reasonable");

    TEST_PASS("Min array size test passed");
    return 1;
}

static int test_chunk_size(void) {
    // Test for complex numbers (16 bytes each)
    size_t chunk = simd_get_chunk_size(16);

    printf("    Chunk size (16-byte elements): %zu\n", chunk);

    TEST_ASSERT(chunk >= 64, "Chunk size should be at least 64 elements");

    // Should be multiple of vector width in elements
    size_t elements_per_vector = simd_get_vector_width() / 16;
    if (elements_per_vector > 0) {
        TEST_ASSERT(chunk % elements_per_vector == 0,
                    "Chunk size should be multiple of elements per vector");
    }

    TEST_PASS("Chunk size test passed");
    return 1;
}

// ============================================================================
// CACHE INFO TESTS
// ============================================================================

static int test_cache_info(void) {
    const simd_info_t* info = simd_detect_capabilities_full();

    printf("    Cache line size: %u bytes\n", info->cache_line_size);
    printf("    L1 cache: %u KB\n", info->l1_cache_size / 1024);
    printf("    L2 cache: %u KB\n", info->l2_cache_size / 1024);

    // Cache line should be reasonable (32-128 bytes)
    if (info->cache_line_size > 0) {
        TEST_ASSERT(info->cache_line_size >= 32,
                    "Cache line should be at least 32 bytes");
        TEST_ASSERT(info->cache_line_size <= 128,
                    "Cache line should be at most 128 bytes");
    }

    TEST_PASS("Cache info test passed");
    return 1;
}

// ============================================================================
// CONSISTENCY TESTS
// ============================================================================

static int test_repeated_detection(void) {
    // Call detection multiple times - should return same results
    const simd_info_t* info1 = simd_detect_capabilities_full();
    const simd_info_t* info2 = simd_detect_capabilities_full();
    const simd_info_t* info3 = simd_detect_capabilities_full();

    TEST_ASSERT(info1 == info2, "Repeated calls should return same pointer");
    TEST_ASSERT(info2 == info3, "Repeated calls should return same pointer");

    TEST_ASSERT(info1->flags == info2->flags, "Flags should be consistent");
    TEST_ASSERT(info1->level == info3->level, "Level should be consistent");

    TEST_PASS("Repeated detection consistency test passed");
    return 1;
}

static int test_validation(void) {
    int result = simd_validate();

    TEST_ASSERT(result == 1, "SIMD validation should pass");

    TEST_PASS("Validation test passed");
    return 1;
}

// ============================================================================
// PRINT TEST
// ============================================================================

static int test_print_capabilities(void) {
    printf("    Printing capability report:\n");
    simd_print_capabilities();

    TEST_PASS("Print capabilities test passed");
    return 1;
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("==============================================\n");
    printf("SIMD Dispatch Layer Unit Tests\n");
    printf("==============================================\n");

    // Detection tests
    RUN_TEST(test_capability_detection);
    RUN_TEST(test_capability_flags);
    RUN_TEST(test_capability_string);
    RUN_TEST(test_simd_level);

    // Backend tests
    RUN_TEST(test_backend_selection);
    RUN_TEST(test_all_backend_names);

    // Vector width tests
    RUN_TEST(test_vector_width);
    RUN_TEST(test_doubles_per_register);
    RUN_TEST(test_complex_per_register);

    // Loop optimization tests
    RUN_TEST(test_unroll_factor);
    RUN_TEST(test_min_array_size);
    RUN_TEST(test_chunk_size);

    // Cache info tests
    RUN_TEST(test_cache_info);

    // Consistency tests
    RUN_TEST(test_repeated_detection);
    RUN_TEST(test_validation);

    // Print test (visual verification)
    RUN_TEST(test_print_capabilities);

    // Summary
    printf("\n==============================================\n");
    printf("Test Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("==============================================\n");

    return tests_failed > 0 ? 1 : 0;
}
