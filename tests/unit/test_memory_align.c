/**
 * @file test_memory_align.c
 * @brief Unit tests for cross-platform memory alignment layer
 *
 * Comprehensive tests for:
 * - Aligned allocation across platforms
 * - Various alignment values (16, 32, 64, 128 bytes)
 * - Secure memory zeroing
 * - Edge cases and error handling
 * - Large allocations (quantum state sized)
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "../../src/optimization/memory_align.h"

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
// BASIC ALLOCATION TESTS
// ============================================================================

static int test_basic_allocation(void) {
    // Test simple allocation
    void* ptr = simd_aligned_alloc(1024, SIMD_ALIGN_64);
    TEST_ASSERT(ptr != NULL, "Basic allocation should succeed");

    // Verify alignment
    TEST_ASSERT(simd_is_aligned(ptr, SIMD_ALIGN_64), "Pointer should be 64-byte aligned");

    // Memory should be zeroed
    unsigned char* bytes = (unsigned char*)ptr;
    int all_zero = 1;
    for (int i = 0; i < 1024; i++) {
        if (bytes[i] != 0) {
            all_zero = 0;
            break;
        }
    }
    TEST_ASSERT(all_zero, "Allocated memory should be zero-initialized");

    // Free
    simd_aligned_free(ptr, 1024);
    TEST_PASS("Basic allocation test passed");
    return 1;
}

static int test_alignment_values(void) {
    size_t alignments[] = { SIMD_ALIGN_16, SIMD_ALIGN_32, SIMD_ALIGN_64, SIMD_ALIGN_128 };
    const char* names[] = { "16", "32", "64", "128" };

    for (int i = 0; i < 4; i++) {
        void* ptr = simd_aligned_alloc(1024, alignments[i]);
        char msg[64];

        snprintf(msg, sizeof(msg), "Allocation with %s-byte alignment", names[i]);
        TEST_ASSERT(ptr != NULL, msg);

        snprintf(msg, sizeof(msg), "Pointer is %s-byte aligned", names[i]);
        TEST_ASSERT(simd_is_aligned(ptr, alignments[i]), msg);

        simd_aligned_free(ptr, 1024);
    }

    TEST_PASS("All alignment values work correctly");
    return 1;
}

static int test_default_alignment(void) {
    void* ptr = simd_aligned_alloc_default(2048);
    TEST_ASSERT(ptr != NULL, "Default alignment allocation should succeed");
    TEST_ASSERT(simd_is_aligned(ptr, SIMD_DEFAULT_ALIGNMENT),
                "Should use default 64-byte alignment");
    simd_aligned_free(ptr, 2048);

    TEST_PASS("Default alignment test passed");
    return 1;
}

// ============================================================================
// COMPLEX ARRAY TESTS
// ============================================================================

static int test_complex_array_allocation(void) {
    // Allocate array for 1024 complex numbers
    size_t num_elements = 1024;
    complex_t* arr = simd_alloc_complex_array(num_elements);

    TEST_ASSERT(arr != NULL, "Complex array allocation should succeed");
    TEST_ASSERT(simd_is_aligned(arr, SIMD_DEFAULT_ALIGNMENT),
                "Complex array should be aligned");

    // Verify zero initialization
    for (size_t i = 0; i < num_elements; i++) {
        TEST_ASSERT(creal(arr[i]) == 0.0 && cimag(arr[i]) == 0.0,
                    "Complex elements should be zero-initialized");
    }

    // Write some values
    for (size_t i = 0; i < num_elements; i++) {
        arr[i] = (double)i + I * (double)(i * 2);
    }

    // Verify writes
    for (size_t i = 0; i < num_elements; i++) {
        TEST_ASSERT(creal(arr[i]) == (double)i, "Real part should match");
        TEST_ASSERT(cimag(arr[i]) == (double)(i * 2), "Imaginary part should match");
    }

    simd_free_complex_array(arr, num_elements);
    TEST_PASS("Complex array allocation and usage test passed");
    return 1;
}

static int test_double_array_allocation(void) {
    size_t num_elements = 2048;
    double* arr = simd_alloc_double_array(num_elements);

    TEST_ASSERT(arr != NULL, "Double array allocation should succeed");
    TEST_ASSERT(simd_is_aligned(arr, SIMD_DEFAULT_ALIGNMENT),
                "Double array should be aligned");

    // Write and verify
    for (size_t i = 0; i < num_elements; i++) {
        arr[i] = (double)i * 1.5;
    }
    for (size_t i = 0; i < num_elements; i++) {
        TEST_ASSERT(arr[i] == (double)i * 1.5, "Double values should match");
    }

    simd_free_double_array(arr, num_elements);
    TEST_PASS("Double array allocation test passed");
    return 1;
}

static int test_uint64_array_allocation(void) {
    size_t num_elements = 512;
    uint64_t* arr = simd_alloc_uint64_array(num_elements);

    TEST_ASSERT(arr != NULL, "Uint64 array allocation should succeed");
    TEST_ASSERT(simd_is_aligned(arr, SIMD_DEFAULT_ALIGNMENT),
                "Uint64 array should be aligned");

    // Write and verify
    for (size_t i = 0; i < num_elements; i++) {
        arr[i] = (uint64_t)i * 0xDEADBEEF;
    }
    for (size_t i = 0; i < num_elements; i++) {
        TEST_ASSERT(arr[i] == (uint64_t)i * 0xDEADBEEF, "Uint64 values should match");
    }

    simd_free_uint64_array(arr, num_elements);
    TEST_PASS("Uint64 array allocation test passed");
    return 1;
}

// ============================================================================
// QUANTUM STATE SIZE TESTS
// ============================================================================

static int test_quantum_state_sizes(void) {
    // Test various qubit counts
    int qubits[] = { 8, 12, 16, 20 };
    const char* names[] = { "8", "12", "16", "20" };

    for (int i = 0; i < 4; i++) {
        size_t state_dim = 1ULL << qubits[i];
        complex_t* state = simd_alloc_complex_array(state_dim);

        char msg[128];
        snprintf(msg, sizeof(msg), "Allocate %s-qubit state (%zu amplitudes)",
                 names[i], state_dim);
        TEST_ASSERT(state != NULL, msg);

        snprintf(msg, sizeof(msg), "%s-qubit state is aligned", names[i]);
        TEST_ASSERT(simd_is_aligned(state, SIMD_DEFAULT_ALIGNMENT), msg);

        // Initialize to |0...0>
        state[0] = 1.0 + 0.0*I;

        // Verify normalization
        double norm_sq = 0.0;
        for (size_t j = 0; j < state_dim; j++) {
            norm_sq += creal(state[j]) * creal(state[j]) +
                       cimag(state[j]) * cimag(state[j]);
        }
        TEST_ASSERT(fabs(norm_sq - 1.0) < 1e-10, "State should be normalized");

        simd_free_complex_array(state, state_dim);
    }

    TEST_PASS("Quantum state size allocation tests passed");
    return 1;
}

// ============================================================================
// REALLOCATION TESTS
// ============================================================================

static int test_reallocation(void) {
    size_t old_size = 1024;
    size_t new_size = 4096;

    void* ptr = simd_aligned_alloc(old_size, SIMD_ALIGN_64);
    TEST_ASSERT(ptr != NULL, "Initial allocation should succeed");

    // Fill with pattern
    memset(ptr, 0xAB, old_size);

    // Reallocate larger
    void* new_ptr = simd_aligned_realloc(ptr, old_size, new_size, SIMD_ALIGN_64);
    TEST_ASSERT(new_ptr != NULL, "Reallocation should succeed");
    TEST_ASSERT(simd_is_aligned(new_ptr, SIMD_ALIGN_64), "Reallocated pointer should be aligned");

    // Verify old data preserved
    unsigned char* bytes = (unsigned char*)new_ptr;
    for (size_t i = 0; i < old_size; i++) {
        TEST_ASSERT(bytes[i] == 0xAB, "Old data should be preserved after realloc");
    }

    // Verify new area is zeroed
    for (size_t i = old_size; i < new_size; i++) {
        TEST_ASSERT(bytes[i] == 0, "New area should be zero-initialized");
    }

    simd_aligned_free(new_ptr, new_size);
    TEST_PASS("Reallocation test passed");
    return 1;
}

static int test_reallocation_shrink(void) {
    size_t old_size = 4096;
    size_t new_size = 1024;

    void* ptr = simd_aligned_alloc(old_size, SIMD_ALIGN_64);
    TEST_ASSERT(ptr != NULL, "Initial allocation should succeed");

    memset(ptr, 0xCD, old_size);

    void* new_ptr = simd_aligned_realloc(ptr, old_size, new_size, SIMD_ALIGN_64);
    TEST_ASSERT(new_ptr != NULL, "Shrinking reallocation should succeed");
    TEST_ASSERT(simd_is_aligned(new_ptr, SIMD_ALIGN_64), "Shrunk pointer should be aligned");

    // Verify data preserved up to new size
    unsigned char* bytes = (unsigned char*)new_ptr;
    for (size_t i = 0; i < new_size; i++) {
        TEST_ASSERT(bytes[i] == 0xCD, "Data should be preserved in shrunk allocation");
    }

    simd_aligned_free(new_ptr, new_size);
    TEST_PASS("Shrinking reallocation test passed");
    return 1;
}

// ============================================================================
// EDGE CASES AND ERROR HANDLING
// ============================================================================

static int test_null_handling(void) {
    // Free NULL should not crash
    simd_aligned_free(NULL, 0);
    simd_aligned_free(NULL, 1024);

    // Free complex array NULL should not crash
    simd_free_complex_array(NULL, 0);
    simd_free_complex_array(NULL, 1024);

    // Zero size allocation should return NULL
    void* ptr = simd_aligned_alloc(0, SIMD_ALIGN_64);
    TEST_ASSERT(ptr == NULL, "Zero size allocation should return NULL");

    // Zero element array should return NULL
    complex_t* arr = simd_alloc_complex_array(0);
    TEST_ASSERT(arr == NULL, "Zero element array should return NULL");

    TEST_PASS("NULL handling tests passed");
    return 1;
}

static int test_alignment_check(void) {
    void* ptr = simd_aligned_alloc(1024, SIMD_ALIGN_64);
    TEST_ASSERT(ptr != NULL, "Allocation should succeed");

    // Check various alignments
    TEST_ASSERT(simd_is_aligned(ptr, 8), "64-byte aligned is also 8-byte aligned");
    TEST_ASSERT(simd_is_aligned(ptr, 16), "64-byte aligned is also 16-byte aligned");
    TEST_ASSERT(simd_is_aligned(ptr, 32), "64-byte aligned is also 32-byte aligned");
    TEST_ASSERT(simd_is_aligned(ptr, 64), "64-byte aligned is 64-byte aligned");

    // NULL should not be aligned
    TEST_ASSERT(!simd_is_aligned(NULL, 64), "NULL should not be aligned");

    simd_aligned_free(ptr, 1024);
    TEST_PASS("Alignment check tests passed");
    return 1;
}

// ============================================================================
// SECURE ZEROING TEST
// ============================================================================

static int test_secure_zero(void) {
    size_t size = 4096;
    unsigned char* buffer = (unsigned char*)simd_aligned_alloc(size, SIMD_ALIGN_64);
    TEST_ASSERT(buffer != NULL, "Allocation should succeed");

    // Fill with pattern
    memset(buffer, 0xFF, size);

    // Verify pattern
    for (size_t i = 0; i < size; i++) {
        TEST_ASSERT(buffer[i] == 0xFF, "Buffer should be filled with pattern");
    }

    // Secure zero
    simd_secure_zero(buffer, size);

    // Verify zeroed
    for (size_t i = 0; i < size; i++) {
        TEST_ASSERT(buffer[i] == 0, "Buffer should be zeroed");
    }

    simd_aligned_free(buffer, size);
    TEST_PASS("Secure zeroing test passed");
    return 1;
}

// ============================================================================
// PLATFORM INFO TEST
// ============================================================================

static int test_platform_info(void) {
    mem_platform_info_t info = simd_get_platform_info();

    printf("    Platform: %s\n", info.platform);
    printf("    Allocator: %s\n", info.allocator);
    printf("    Default alignment: %zu bytes\n", info.default_alignment);
    printf("    Page size: %zu bytes\n", info.page_size);
    printf("    Huge pages: %s\n", info.supports_huge_pages ? "yes" : "no");

    TEST_ASSERT(info.platform != NULL, "Platform should be set");
    TEST_ASSERT(info.allocator != NULL, "Allocator should be set");
    TEST_ASSERT(info.default_alignment >= 16, "Default alignment should be at least 16");
    TEST_ASSERT(info.page_size >= 4096, "Page size should be at least 4KB");

    TEST_PASS("Platform info test passed");
    return 1;
}

// ============================================================================
// OPTIMAL ALIGNMENT TEST
// ============================================================================

static int test_optimal_alignment(void) {
    size_t optimal = simd_get_optimal_alignment();

    printf("    Detected optimal alignment: %zu bytes\n", optimal);

    TEST_ASSERT(optimal >= 16, "Optimal alignment should be at least 16 bytes");
    TEST_ASSERT(optimal <= 128, "Optimal alignment should be at most 128 bytes");
    TEST_ASSERT((optimal & (optimal - 1)) == 0, "Optimal alignment should be power of 2");

    TEST_PASS("Optimal alignment detection test passed");
    return 1;
}

// ============================================================================
// LARGE ALLOCATION STRESS TEST
// ============================================================================

static int test_large_allocations(void) {
    // Test progressively larger allocations
    size_t sizes[] = {
        1ULL << 20,   // 1 MB
        1ULL << 22,   // 4 MB
        1ULL << 24,   // 16 MB
        1ULL << 26    // 64 MB
    };

    for (int i = 0; i < 4; i++) {
        void* ptr = simd_aligned_alloc(sizes[i], SIMD_ALIGN_64);
        if (ptr == NULL) {
            // Large allocation failed - not necessarily an error
            printf("    Note: %zu MB allocation failed (may be expected)\n",
                   sizes[i] / (1024 * 1024));
            continue;
        }

        TEST_ASSERT(simd_is_aligned(ptr, SIMD_ALIGN_64),
                    "Large allocation should be aligned");

        // Touch first and last bytes
        unsigned char* bytes = (unsigned char*)ptr;
        bytes[0] = 0xAA;
        bytes[sizes[i] - 1] = 0xBB;

        simd_aligned_free(ptr, sizes[i]);
        printf("    Successfully allocated and freed %zu MB\n",
               sizes[i] / (1024 * 1024));
    }

    TEST_PASS("Large allocation tests completed");
    return 1;
}

// ============================================================================
// MULTIPLE ALLOCATION STRESS TEST
// ============================================================================

static int test_multiple_allocations(void) {
    const int num_allocs = 100;
    void* ptrs[100];
    size_t sizes[100];

    // Allocate many blocks
    for (int i = 0; i < num_allocs; i++) {
        sizes[i] = (i + 1) * 64;  // Various sizes: 64, 128, 192, ...
        ptrs[i] = simd_aligned_alloc(sizes[i], SIMD_ALIGN_64);
        TEST_ASSERT(ptrs[i] != NULL, "Multiple allocations should succeed");
        TEST_ASSERT(simd_is_aligned(ptrs[i], SIMD_ALIGN_64),
                    "Each allocation should be aligned");
    }

    // Free in reverse order
    for (int i = num_allocs - 1; i >= 0; i--) {
        simd_aligned_free(ptrs[i], sizes[i]);
    }

    TEST_PASS("Multiple allocation stress test passed");
    return 1;
}

// ============================================================================
// MEMCPY/MEMSET TEST
// ============================================================================

static int test_aligned_memcpy(void) {
    size_t size = 8192;
    void* src = simd_aligned_alloc(size, SIMD_ALIGN_64);
    void* dst = simd_aligned_alloc(size, SIMD_ALIGN_64);

    TEST_ASSERT(src != NULL && dst != NULL, "Allocations should succeed");

    // Fill source with pattern
    unsigned char* src_bytes = (unsigned char*)src;
    for (size_t i = 0; i < size; i++) {
        src_bytes[i] = (unsigned char)(i & 0xFF);
    }

    // Copy
    simd_aligned_memcpy(dst, src, size, SIMD_ALIGN_64);

    // Verify
    unsigned char* dst_bytes = (unsigned char*)dst;
    for (size_t i = 0; i < size; i++) {
        TEST_ASSERT(dst_bytes[i] == (unsigned char)(i & 0xFF),
                    "Copied data should match source");
    }

    simd_aligned_free(src, size);
    simd_aligned_free(dst, size);
    TEST_PASS("Aligned memcpy test passed");
    return 1;
}

static int test_aligned_memset(void) {
    size_t size = 4096;
    void* ptr = simd_aligned_alloc(size, SIMD_ALIGN_64);
    TEST_ASSERT(ptr != NULL, "Allocation should succeed");

    simd_aligned_memset(ptr, 0x55, size, SIMD_ALIGN_64);

    unsigned char* bytes = (unsigned char*)ptr;
    for (size_t i = 0; i < size; i++) {
        TEST_ASSERT(bytes[i] == 0x55, "Memory should be set to pattern");
    }

    simd_aligned_free(ptr, size);
    TEST_PASS("Aligned memset test passed");
    return 1;
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("==============================================\n");
    printf("Memory Alignment Layer Unit Tests\n");
    printf("==============================================\n");

    // Basic allocation tests
    RUN_TEST(test_basic_allocation);
    RUN_TEST(test_alignment_values);
    RUN_TEST(test_default_alignment);

    // Type-specific allocation tests
    RUN_TEST(test_complex_array_allocation);
    RUN_TEST(test_double_array_allocation);
    RUN_TEST(test_uint64_array_allocation);

    // Quantum state size tests
    RUN_TEST(test_quantum_state_sizes);

    // Reallocation tests
    RUN_TEST(test_reallocation);
    RUN_TEST(test_reallocation_shrink);

    // Edge cases
    RUN_TEST(test_null_handling);
    RUN_TEST(test_alignment_check);

    // Security tests
    RUN_TEST(test_secure_zero);

    // Platform tests
    RUN_TEST(test_platform_info);
    RUN_TEST(test_optimal_alignment);

    // Stress tests
    RUN_TEST(test_large_allocations);
    RUN_TEST(test_multiple_allocations);

    // Memory operations
    RUN_TEST(test_aligned_memcpy);
    RUN_TEST(test_aligned_memset);

    // Summary
    printf("\n==============================================\n");
    printf("Test Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("==============================================\n");

    return tests_failed > 0 ? 1 : 0;
}
