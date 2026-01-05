/**
 * @file entropy.c
 * @brief Entropy sources and quality assessment implementation
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
 * Licensed under the MIT License
 */

#include "entropy.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Platform-specific includes
#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#endif

#ifdef __APPLE__
#include <Security/Security.h>
#endif

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#ifdef __GNUC__
#include <cpuid.h>
#endif
#define HAS_X86
#endif

#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#ifndef NT_SUCCESS
#define NT_SUCCESS(Status) (((NTSTATUS)(Status)) >= 0)
#endif
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define HAS_ARM64
#endif

// ============================================================================
// INTERNAL HELPERS
// ============================================================================

/**
 * @brief SplitMix64 for mixing
 */
static uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/**
 * @brief XorShift128+ for pool extraction
 */
static uint64_t xorshift128plus(uint64_t* s) {
    uint64_t x = s[0];
    uint64_t y = s[1];
    s[0] = y;
    x ^= x << 23;
    s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    return s[1] + y;
}

/**
 * @brief Mix data into state
 */
static void mix_into_state(uint64_t* state, const uint8_t* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        state[i % 4] ^= ((uint64_t)data[i]) << ((i * 8) % 64);
        state[(i + 1) % 4] = splitmix64(&state[i % 4]);
    }
}

// ============================================================================
// HARDWARE RNG DETECTION AND ACCESS
// ============================================================================

#ifdef HAS_X86
static int x86_has_rdrand(void) {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return (ecx >> 30) & 1;  // RDRAND bit
    }
    return 0;
}

static int x86_has_rdseed(void) {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx >> 18) & 1;  // RDSEED bit
    }
    return 0;
}

static int x86_rdrand64(uint64_t* value) {
    unsigned long long result;
    int success;
    for (int i = 0; i < 10; i++) {
        success = _rdrand64_step(&result);
        if (success) {
            *value = result;
            return 1;
        }
    }
    return 0;
}

static int x86_rdseed64(uint64_t* value) {
    unsigned long long result;
    int success;
    for (int i = 0; i < 10; i++) {
        success = _rdseed64_step(&result);
        if (success) {
            *value = result;
            return 1;
        }
    }
    return 0;
}
#endif

#ifdef HAS_ARM64
static int arm64_has_rndr(void) {
    // Check for RNDR support via ID_AA64ISAR0_EL1
    // For now, assume available on ARMv8.5+
    #if defined(__ARM_FEATURE_RNG) && __ARM_FEATURE_RNG
    return 1;
    #else
    return 0;
    #endif
}

#if defined(__ARM_FEATURE_RNG) && __ARM_FEATURE_RNG
static int arm64_rndr(uint64_t* value) {
    uint64_t result;
    int success;
    __asm__ volatile(
        "mrs %0, s3_3_c2_c4_0\n"  // RNDR
        "cset %w1, ne"
        : "=r"(result), "=r"(success)
        :
        : "cc"
    );
    if (success) {
        *value = result;
        return 1;
    }
    return 0;
}
#endif
#endif

// ============================================================================
// HARDWARE RNG PUBLIC API
// ============================================================================

int entropy_hardware_available(void) {
#ifdef HAS_X86
    return x86_has_rdrand() || x86_has_rdseed();
#elif defined(HAS_ARM64)
    return arm64_has_rndr();
#else
    return 0;
#endif
}

int entropy_hardware_bytes(uint8_t* buffer, size_t size) {
    if (!buffer || size == 0) return -1;

#ifdef HAS_X86
    if (x86_has_rdseed()) {
        size_t pos = 0;
        while (pos + 8 <= size) {
            uint64_t val;
            if (!x86_rdseed64(&val)) break;
            memcpy(buffer + pos, &val, 8);
            pos += 8;
        }
        if (pos < size) {
            uint64_t val;
            if (x86_rdseed64(&val)) {
                memcpy(buffer + pos, &val, size - pos);
                pos = size;
            }
        }
        return (int)pos;
    }

    if (x86_has_rdrand()) {
        size_t pos = 0;
        while (pos + 8 <= size) {
            uint64_t val;
            if (!x86_rdrand64(&val)) break;
            memcpy(buffer + pos, &val, 8);
            pos += 8;
        }
        if (pos < size) {
            uint64_t val;
            if (x86_rdrand64(&val)) {
                memcpy(buffer + pos, &val, size - pos);
                pos = size;
            }
        }
        return (int)pos;
    }
#endif

#ifdef HAS_ARM64
    #if defined(__ARM_FEATURE_RNG) && __ARM_FEATURE_RNG
    if (arm64_has_rndr()) {
        size_t pos = 0;
        while (pos + 8 <= size) {
            uint64_t val;
            if (!arm64_rndr(&val)) break;
            memcpy(buffer + pos, &val, 8);
            pos += 8;
        }
        if (pos < size) {
            uint64_t val;
            if (arm64_rndr(&val)) {
                memcpy(buffer + pos, &val, size - pos);
                pos = size;
            }
        }
        return (int)pos;
    }
    #endif
#endif

    return -1;
}

const char* entropy_hardware_name(void) {
#ifdef HAS_X86
    if (x86_has_rdseed()) return "RDSEED";
    if (x86_has_rdrand()) return "RDRAND";
#endif
#ifdef HAS_ARM64
    if (arm64_has_rndr()) return "RNDR";
#endif
    return "none";
}

// ============================================================================
// OS ENTROPY
// ============================================================================

int entropy_os_bytes(uint8_t* buffer, size_t size) {
    if (!buffer || size == 0) return -1;

#if defined(__APPLE__)
    // Use SecRandomCopyBytes
    if (SecRandomCopyBytes(kSecRandomDefault, size, buffer) == errSecSuccess) {
        return (int)size;
    }
#endif

#if defined(__linux__) || defined(__unix__) || defined(__APPLE__)
    // Fallback to /dev/urandom
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        ssize_t bytes_read = read(fd, buffer, size);
        close(fd);
        if (bytes_read > 0) {
            return (int)bytes_read;
        }
    }
#endif

#ifdef _WIN32
    // Use BCryptGenRandom (modern Windows API, Vista+)
    NTSTATUS status = BCryptGenRandom(NULL, (PUCHAR)buffer, (ULONG)size,
                                       BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (NT_SUCCESS(status)) {
        return (int)size;
    }
#endif

    return -1;
}

const char* entropy_os_source_name(void) {
#if defined(__APPLE__)
    return "SecRandomCopyBytes";
#elif defined(__linux__)
    return "/dev/urandom";
#elif defined(_WIN32)
    return "BCryptGenRandom";
#else
    return "unknown";
#endif
}

// ============================================================================
// JITTER ENTROPY
// ============================================================================

size_t entropy_jitter_bytes(uint8_t* buffer, size_t size) {
    if (!buffer || size == 0) return 0;

    size_t collected = 0;
    uint64_t accum = 0;
    int bits = 0;

    for (size_t i = 0; collected < size && i < size * 100; i++) {
        // Get high-resolution time
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);

        // Extract low bits of nanoseconds (jitter)
        uint64_t jitter = ts.tv_nsec & 0x1;

        // Add some CPU jitter
        volatile int dummy = 0;
        for (int j = 0; j < (ts.tv_nsec & 0xF); j++) {
            dummy += j;
        }
        (void)dummy;

        // Accumulate bits
        accum = (accum << 1) | jitter;
        bits++;

        if (bits >= 8) {
            buffer[collected++] = (uint8_t)(accum & 0xFF);
            accum = 0;
            bits = 0;
        }
    }

    return collected;
}

// ============================================================================
// CONTEXT MANAGEMENT
// ============================================================================

entropy_ctx_t* entropy_create(void) {
    return entropy_create_with_source(ENTROPY_SOURCE_AUTO);
}

entropy_ctx_t* entropy_create_with_source(entropy_source_type_t source) {
    entropy_ctx_t* ctx = calloc(1, sizeof(entropy_ctx_t));
    if (!ctx) return NULL;

    ctx->source = source;
    ctx->fd_urandom = -1;

    // Initialize mixing state with time-based seed
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ctx->mix_state[0] = ts.tv_sec;
    ctx->mix_state[1] = ts.tv_nsec;
    ctx->mix_state[2] = (uint64_t)ctx;
    ctx->mix_state[3] = (uint64_t)clock();

    // Open /dev/urandom if available
#if defined(__linux__) || defined(__unix__) || defined(__APPLE__)
    ctx->fd_urandom = open("/dev/urandom", O_RDONLY);
#endif

    // Initial pool fill
    entropy_reseed(ctx);

    return ctx;
}

void entropy_destroy(entropy_ctx_t* ctx) {
    if (!ctx) return;

    // Close file descriptor
    if (ctx->fd_urandom >= 0) {
        close(ctx->fd_urandom);
    }

    // Zero sensitive data
    memset(ctx->pool, 0, sizeof(ctx->pool));
    memset(ctx->mix_state, 0, sizeof(ctx->mix_state));

    free(ctx);
}

int entropy_reseed(entropy_ctx_t* ctx) {
    if (!ctx) return -1;

    uint8_t seed[64];
    size_t seed_bytes = 0;

    // Try hardware RNG first
    int hw_bytes = entropy_hardware_bytes(seed, sizeof(seed));
    if (hw_bytes > 0) {
        seed_bytes = hw_bytes;
        ctx->hardware_bytes += hw_bytes;
    }

    // Mix in OS entropy
    int os_bytes = entropy_os_bytes(seed + seed_bytes, sizeof(seed) - seed_bytes);
    if (os_bytes > 0) {
        seed_bytes += os_bytes;
        ctx->os_bytes += os_bytes;
    }

    // If still need more, use jitter
    if (seed_bytes < 32) {
        size_t jitter_bytes = entropy_jitter_bytes(seed + seed_bytes, 32 - seed_bytes);
        seed_bytes += jitter_bytes;
    }

    // Mix seed into state
    mix_into_state(ctx->mix_state, seed, seed_bytes);
    ctx->reseed_counter++;

    // Fill pool
    for (size_t i = 0; i < sizeof(ctx->pool); i += 8) {
        uint64_t val = xorshift128plus(ctx->mix_state);
        memcpy(ctx->pool + i, &val, 8);
    }
    ctx->pool_bytes = sizeof(ctx->pool);
    ctx->pool_pos = 0;

    // Clear seed
    memset(seed, 0, sizeof(seed));

    return 0;
}

// ============================================================================
// ENTROPY COLLECTION
// ============================================================================

uint8_t entropy_byte(entropy_ctx_t* ctx) {
    if (!ctx) {
        // Fallback
        uint8_t b;
        entropy_os_bytes(&b, 1);
        return b;
    }

    if (ctx->pool_pos >= ctx->pool_bytes) {
        entropy_reseed(ctx);
    }

    uint8_t result = ctx->pool[ctx->pool_pos++];
    ctx->bytes_collected++;

    return result;
}

size_t entropy_bytes(entropy_ctx_t* ctx, uint8_t* buffer, size_t size) {
    if (!buffer || size == 0) return 0;

    if (!ctx) {
        // Fallback to OS entropy
        int bytes = entropy_os_bytes(buffer, size);
        return (bytes > 0) ? bytes : 0;
    }

    size_t collected = 0;

    while (collected < size) {
        if (ctx->pool_pos >= ctx->pool_bytes) {
            entropy_reseed(ctx);
        }

        size_t available = ctx->pool_bytes - ctx->pool_pos;
        size_t to_copy = (size - collected < available) ? (size - collected) : available;

        memcpy(buffer + collected, ctx->pool + ctx->pool_pos, to_copy);
        ctx->pool_pos += to_copy;
        collected += to_copy;
        ctx->bytes_collected += to_copy;
    }

    return collected;
}

uint32_t entropy_uint32(entropy_ctx_t* ctx) {
    uint32_t result;
    entropy_bytes(ctx, (uint8_t*)&result, sizeof(result));
    return result;
}

uint64_t entropy_uint64(entropy_ctx_t* ctx) {
    uint64_t result;
    entropy_bytes(ctx, (uint8_t*)&result, sizeof(result));
    return result;
}

double entropy_double(entropy_ctx_t* ctx) {
    uint64_t bits = entropy_uint64(ctx);
    return (double)(bits >> 11) * (1.0 / 9007199254740992.0);
}

// ============================================================================
// ENTROPY MIXING
// ============================================================================

void entropy_mix(entropy_ctx_t* ctx, const uint8_t* data, size_t size) {
    if (!ctx || !data || size == 0) return;

    mix_into_state(ctx->mix_state, data, size);
}

int entropy_extract(const uint8_t* input, size_t input_size,
                    uint8_t* output, size_t output_size) {
    if (!input || !output || input_size == 0 || output_size == 0) {
        return -1;
    }

    // Simple hash-based extraction using xorshift
    uint64_t state[4] = {0};

    // Mix input into state
    for (size_t i = 0; i < input_size; i++) {
        state[i % 4] ^= ((uint64_t)input[i]) << ((i * 8) % 64);
        state[(i + 1) % 4] = splitmix64(&state[i % 4]);
    }

    // Extract output
    for (size_t i = 0; i < output_size; i += 8) {
        uint64_t val = xorshift128plus(state);
        size_t to_copy = (output_size - i < 8) ? (output_size - i) : 8;
        memcpy(output + i, &val, to_copy);
    }

    return 0;
}

// ============================================================================
// QUALITY ASSESSMENT
// ============================================================================

double entropy_estimate(const uint8_t* data, size_t size) {
    if (!data || size == 0) return 0.0;

    // Count byte frequencies
    uint64_t counts[256] = {0};
    for (size_t i = 0; i < size; i++) {
        counts[data[i]]++;
    }

    // Calculate Shannon entropy
    double entropy = 0.0;
    double log2_size = log2((double)size);

    for (int i = 0; i < 256; i++) {
        if (counts[i] > 0) {
            double p = (double)counts[i] / (double)size;
            entropy -= p * log2(p);
        }
    }

    return entropy;
}

int entropy_test_data(const uint8_t* data, size_t size,
                      entropy_quality_t* quality) {
    if (!data || size == 0) return 0;

    if (quality) {
        memset(quality, 0, sizeof(entropy_quality_t));

        quality->estimated_entropy = entropy_estimate(data, size);
        quality->hardware_available = entropy_hardware_available();
        quality->os_available = 1;  // Assume available

        // Chi-squared test
        uint64_t counts[256] = {0};
        for (size_t i = 0; i < size; i++) {
            counts[data[i]]++;
        }

        double expected = (double)size / 256.0;
        double chi_sq = 0.0;
        for (int i = 0; i < 256; i++) {
            double diff = (double)counts[i] - expected;
            chi_sq += (diff * diff) / expected;
        }
        quality->chi_squared = chi_sq;

        // Check if chi-squared is within acceptable range
        // For 255 degrees of freedom, 95% confidence: 210 < X < 302
        quality->passed_basic_tests = (chi_sq > 210.0 && chi_sq < 302.0);

        // Compression ratio (simplified)
        quality->compression_ratio = quality->estimated_entropy / 8.0;
    }

    // Simple pass/fail based on entropy estimate
    double entropy = entropy_estimate(data, size);
    return (entropy >= 7.0);  // At least 7 bits per byte
}

int entropy_assess_quality(entropy_ctx_t* ctx, size_t sample_size,
                           entropy_quality_t* quality) {
    if (!ctx || sample_size == 0) return -1;

    uint8_t* sample = malloc(sample_size);
    if (!sample) return -1;

    entropy_bytes(ctx, sample, sample_size);
    int result = entropy_test_data(sample, sample_size, quality);

    free(sample);
    return result ? 0 : -1;
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

size_t entropy_generate(uint8_t* buffer, size_t size) {
    if (!buffer || size == 0) return 0;

    entropy_ctx_t* ctx = entropy_create();
    if (!ctx) {
        // Fallback to OS
        int bytes = entropy_os_bytes(buffer, size);
        return (bytes > 0) ? bytes : 0;
    }

    size_t result = entropy_bytes(ctx, buffer, size);
    entropy_destroy(ctx);

    return result;
}

uint64_t entropy_random_uint64(void) {
    uint64_t result;
    entropy_generate((uint8_t*)&result, sizeof(result));
    return result;
}

int entropy_seed(uint8_t* seed, size_t seed_size) {
    if (!seed || seed_size == 0) return -1;

    // Try multiple sources for maximum entropy
    size_t collected = 0;

    // Hardware RNG
    int hw = entropy_hardware_bytes(seed, seed_size);
    if (hw > 0) collected += hw;

    // OS entropy
    if (collected < seed_size) {
        int os = entropy_os_bytes(seed + collected, seed_size - collected);
        if (os > 0) collected += os;
    }

    // Jitter if still need more
    if (collected < seed_size) {
        size_t jitter = entropy_jitter_bytes(seed + collected, seed_size - collected);
        collected += jitter;
    }

    return (collected >= seed_size) ? 0 : -1;
}

// ============================================================================
// PLATFORM DETECTION
// ============================================================================

void entropy_utils_get_capabilities(int* has_hardware, int* has_os, int* has_jitter) {
    if (has_hardware) *has_hardware = entropy_hardware_available();
    if (has_os) *has_os = 1;  // Always assume OS entropy available
    if (has_jitter) *has_jitter = 1;  // Always available
}

const char* entropy_utils_source_name(entropy_source_type_t source) {
    switch (source) {
        case ENTROPY_SOURCE_AUTO:     return "auto";
        case ENTROPY_SOURCE_HARDWARE: return "hardware";
        case ENTROPY_SOURCE_OS:       return "os";
        case ENTROPY_SOURCE_JITTER:   return "jitter";
        case ENTROPY_SOURCE_MIXED:    return "mixed";
        case ENTROPY_SOURCE_QUANTUM:  return "quantum";
        default:                      return "unknown";
    }
}
