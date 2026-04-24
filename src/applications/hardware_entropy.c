#include "hardware_entropy.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <inttypes.h>
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <bcrypt.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <signal.h>
#include <setjmp.h>
#endif
#include <time.h>

// Platform-specific includes
#ifdef __linux__
#include <sys/syscall.h>
#endif

// CPU feature detection
#if defined(__x86_64__) && !defined(_WIN32) && !defined(_WIN64)
#include <cpuid.h>
#elif defined(__aarch64__)
    // ARM feature detection - platform specific
    #ifdef __linux__
        #include <sys/auxv.h>
        #ifndef HWCAP_RNG
            #define HWCAP_RNG (1 << 16)  // ARMv8.5-A Random Number
        #endif
    #elif defined(__APPLE__)
        #include <sys/sysctl.h>
    #endif
#endif

// ============================================================================
// CPU INSTRUCTION DETECTION
// ============================================================================

int rdrand_available(void) {
#if defined(__x86_64__) && !defined(_WIN32) && !defined(_WIN64)
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        return (ecx & (1 << 30)) != 0; // RDRAND is bit 30 of ECX
    }
#endif
    return 0;
}

int rdseed_available(void) {
#if defined(__x86_64__) && !defined(_WIN32) && !defined(_WIN64)
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 18)) != 0; // RDSEED is bit 18 of EBX
    }
#endif
    return 0;
}

// ============================================================================
// ARM HARDWARE INSTRUCTIONS (ARMv8.5-A+)
// ============================================================================

#ifdef __aarch64__

#include <sys/wait.h>

// Cache the detection result (-1 = not tested, 0 = not available, 1 = available)
static int rndr_detection_result = -1;

#ifndef MOONLAB_HW_RNG_PROBE_PATH
#define MOONLAB_HW_RNG_PROBE_PATH ((const char*)0)
#endif

/**
 * @brief Path to the hardware RNG probe helper executable
 *
 * The helper is a separate binary that safely executes RNDR/RNDRRS instructions.
 * The path is compiled in as an absolute path so the library never resolves the
 * helper relative to the caller's working directory.
 */
static const char *HW_RNG_PROBE_PATH = MOONLAB_HW_RNG_PROBE_PATH;

/**
 * @brief Execute RNDR/RNDRRS via a trusted helper using fork/execv
 *
 * Spawns the helper executable directly without invoking a shell. The helper
 * path must be absolute; otherwise the probe is treated as unavailable.
 *
 * The child performs only async-signal-safe operations between fork and execv.
 *
 * @param mode Probe mode ("rndr" or "rndrrs")
 * @param out Optional parsed 64-bit output from the helper
 * @return 1 on success, 0 on failure
 */
static int hw_rng_probe_exec(const char *mode, uint64_t *out) {
    if (!mode || !HW_RNG_PROBE_PATH || HW_RNG_PROBE_PATH[0] != '/') {
        return 0;
    }

    char mode_arg[32];
    int mode_len = snprintf(mode_arg, sizeof(mode_arg), "--mode=%s", mode);
    if (mode_len <= 0 || (size_t)mode_len >= sizeof(mode_arg)) {
        return 0;
    }

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        return 0;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return 0;
    }

    if (pid == 0) {
        char *const argv[] = {
            (char *)HW_RNG_PROBE_PATH,
            mode_arg,
            NULL
        };

        close(pipefd[0]);
        if (dup2(pipefd[1], STDOUT_FILENO) < 0) {
            _exit(127);
        }
        close(pipefd[1]);
        execv(HW_RNG_PROBE_PATH, argv);
        _exit(127);
    }

    close(pipefd[1]);

    char buf[32];
    size_t total = 0;
    while (total + 1 < sizeof(buf)) {
        ssize_t n = read(pipefd[0], buf + total, sizeof(buf) - total - 1);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            close(pipefd[0]);
            (void)waitpid(pid, NULL, 0);
            return 0;
        }
        if (n == 0) {
            break;
        }
        total += (size_t)n;
        if (memchr(buf, '\n', total) != NULL) {
            break;
        }
    }
    close(pipefd[0]);
    buf[total] = '\0';

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            return 0;
        }
    }

    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 || total == 0) {
        return 0;
    }

    if (out) {
        uint64_t val = 0;
        if (sscanf(buf, "%llx", (unsigned long long *)&val) != 1) {
            return 0;
        }
        *out = val;
    }

    return 1;
}

#ifdef MOONLAB_TESTING
int moonlab_hw_rng_probe_exec(const char *mode, uint64_t *out) {
    return hw_rng_probe_exec(mode, out);
}
#endif

/**
 * @brief Check if ARM RNDR instruction is available
 *
 * Detection strategy:
 * 1. Check MOONLAB_SKIP_HW_ENTROPY env var (CI opt-out)
 * 2. Return cached result if already tested
 * 3. Check sysctl for FEAT_RNG capability
 * 4. If sysctl says yes, verify with the trusted helper
 * 5. Cache and return result
 */
int rndr_available(void) {
    // Return cached result if already tested
    if (rndr_detection_result >= 0) {
        return rndr_detection_result;
    }

    // CI opt-out: skip hardware probe entirely
    const char *skip = getenv("MOONLAB_SKIP_HW_ENTROPY");
    if (skip && (skip[0] == '1' || skip[0] == 'y' || skip[0] == 'Y')) {
        rndr_detection_result = 0;
        return rndr_detection_result;
    }

    #ifdef __linux__
        // Linux: Use getauxval (fast and reliable)
        unsigned long hwcaps = getauxval(AT_HWCAP);
        rndr_detection_result = (hwcaps & HWCAP_RNG) != 0;
        return rndr_detection_result;

    #elif defined(__APPLE__)
        // macOS: Multi-tier detection for Apple Silicon
        // VMs (including GitHub Actions runners) may report FEAT_RNG
        // but crash when the instruction is actually executed.

        // Tier 1: Check sysctl for FEAT_RNG
        int has_feature = 0;
        size_t size = sizeof(has_feature);
        int sysctl_ok = (sysctlbyname("hw.optional.arm.FEAT_RNG", &has_feature, &size, NULL, 0) == 0);

        if (!sysctl_ok) {
            // Try alternate sysctl name
            sysctl_ok = (sysctlbyname("hw.optional.armv8_5_rng", &has_feature, &size, NULL, 0) == 0);
        }

        if (!sysctl_ok || !has_feature) {
            // Sysctl says no RNDR
            rndr_detection_result = 0;
            return rndr_detection_result;
        }

        // Tier 2: Sysctl says yes - verify with the trusted helper.
        // This catches VMs that lie about FEAT_RNG without invoking a shell.
        rndr_detection_result = hw_rng_probe_exec("rndr", NULL);
        return rndr_detection_result;

    #else
        // Other ARM platforms: Conservative default
        rndr_detection_result = 0;
        return rndr_detection_result;
    #endif
}

/**
 * @brief Get entropy via ARM RNDR instruction (safe popen version)
 */
int rndr_get_uint64(uint64_t *value) {
    if (!rndr_available()) return 0;

    // Retry up to 10 times
    for (int i = 0; i < 10; i++) {
        if (hw_rng_probe_exec("rndr", value)) return 1;
    }
    return 0;
}

/**
 * @brief Get entropy via ARM RNDRRS instruction (safe popen version)
 */
int rndrrs_get_uint64(uint64_t *value) {
    if (!rndr_available()) return 0;

    // RNDRRS may take longer, retry up to 100 times
    for (int i = 0; i < 100; i++) {
        if (hw_rng_probe_exec("rndrrs", value)) return 1;
    }
    return 0;
}

#else
// Stub implementations for non-ARM platforms
int rndr_available(void) { return 0; }
int rndr_get_uint64(uint64_t *value) { (void)value; return 0; }
int rndrrs_get_uint64(uint64_t *value) { (void)value; return 0; }
#endif

// ============================================================================
// HARDWARE INSTRUCTIONS
// ============================================================================

int rdrand_get_uint64(uint64_t *value) {
    (void)value; /* unused on non-x86_64 targets */
#if defined(__x86_64__) && !defined(_WIN32) && !defined(_WIN64)
    if (!rdrand_available()) return 0;
    
    unsigned char ok;
    // Retry up to 10 times as recommended by Intel
    for (int i = 0; i < 10; i++) {
        __asm__ volatile(
            "rdrand %0; setc %1"
            : "=r" (*value), "=qm" (ok)
            :
            : "cc"
        );
        if (ok) return 1;
    }
#endif
    return 0;
}

int rdseed_get_uint64(uint64_t *value) {
    (void)value; /* unused on non-x86_64 targets */
#if defined(__x86_64__) && !defined(_WIN32) && !defined(_WIN64)
    if (!rdseed_available()) return 0;
    
    unsigned char ok;
    // RDSEED may take longer, retry up to 100 times
    for (int i = 0; i < 100; i++) {
        __asm__ volatile(
            "rdseed %0; setc %1"
            : "=r" (*value), "=qm" (ok)
            :
            : "cc"
        );
        if (ok) return 1;
    }
#endif
    return 0;
}

// ============================================================================
// SYSTEM ENTROPY (getrandom)
// ============================================================================

ssize_t entropy_getrandom(uint8_t *buffer, size_t size, unsigned int flags) {
    (void)buffer; (void)size; (void)flags; /* unused on non-Linux targets */
#if defined(_WIN32) || defined(_WIN64)
    if (!buffer) {
        errno = EINVAL;
        return -1;
    }
    if (size > ULONG_MAX) {
        errno = EINVAL;
        return -1;
    }
    NTSTATUS status = BCryptGenRandom(
        NULL,
        buffer,
        (ULONG)size,
        BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (status < 0) {
        errno = EIO;
        return -1;
    }
    return (ssize_t)size;
#elif defined(__linux__)
    #ifdef SYS_getrandom
    return syscall(SYS_getrandom, buffer, size, flags);
    #else
    errno = ENOSYS;
    return -1;
    #endif
#else
    errno = ENOSYS;
    return -1;
#endif
}

// ============================================================================
// JITTER ENTROPY
// ============================================================================

// High-resolution timer for jitter collection
static inline uint64_t get_timer_cycles(void) {
#if defined(__x86_64__) && !defined(_WIN32) && !defined(_WIN64)
    unsigned int lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#else
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
#endif
}

static inline void entropy_time_now(struct timespec *ts) {
#if defined(_WIN32) || defined(_WIN64)
    timespec_get(ts, TIME_UTC);
#else
    clock_gettime(CLOCK_MONOTONIC, ts);
#endif
}

// Cryptographic-quality mixing based on SipHash-inspired design
static inline uint64_t sipround(uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3) {
    v0 += v1; v2 += v3;
    v1 = (v1 << 13) | (v1 >> 51);
    v3 = (v3 << 16) | (v3 >> 48);
    v1 ^= v0; v3 ^= v2;
    v0 = (v0 << 32) | (v0 >> 32);
    v2 += v1; v0 += v3;
    v1 = (v1 << 17) | (v1 >> 47);
    v3 = (v3 << 21) | (v3 >> 43);
    v1 ^= v2; v3 ^= v0;
    v2 = (v2 << 32) | (v2 >> 32);
    return v0 ^ v1 ^ v2 ^ v3;
}

// Enhanced cryptographic mixing with multiple rounds
static inline uint64_t crypto_mix(uint64_t x, uint64_t key) {
    uint64_t v0 = 0x736f6d6570736575ULL ^ key;
    uint64_t v1 = 0x646f72616e646f6dULL ^ key;
    uint64_t v2 = 0x6c7967656e657261ULL ^ x;
    uint64_t v3 = 0x7465646279746573ULL ^ x;
    
    // Multiple mixing rounds for thorough diffusion
    for (int i = 0; i < 4; i++) {
        v0 = sipround(v0, v1, v2, v3);
        v1 = sipround(v1, v2, v3, v0);
        v2 = sipround(v2, v3, v0, v1);
        v3 = sipround(v3, v0, v1, v2);
    }
    
    return v0 ^ v1 ^ v2 ^ v3;
}

// Von Neumann debiasing - removes bias from binary stream
static inline int debias_bit_pair(uint8_t b1, uint8_t b2) {
    if (b1 != b2) {
        return b1; // Return first bit if different (removes bias)
    }
    return -1; // Skip if same (no information)
}

// Extract entropy from timing delta using multiple techniques
static inline uint64_t extract_timing_entropy(uint64_t delta, uint64_t prev_delta, uint64_t state) {
    // Use multiple entropy sources from timing:
    // 1. Absolute value (timing uncertainty)
    uint64_t abs_entropy = delta;
    
    // 2. Second derivative (timing jitter acceleration)
    uint64_t jitter = delta ^ prev_delta;
    
    // 3. LSBs (most uncertain bits)
    uint64_t lsb_entropy = delta & 0xFFULL;
    
    // 4. XOR fold to concentrate entropy
    uint64_t folded = delta ^ (delta >> 32);
    folded ^= (folded >> 16);
    folded ^= (folded >> 8);
    
    // Combine all sources with cryptographic mixing
    uint64_t combined = crypto_mix(abs_entropy, state) ^
                       crypto_mix(jitter, state + 1) ^
                       crypto_mix(lsb_entropy, state + 2) ^
                       crypto_mix(folded, state + 3);
    
    return combined;
}

entropy_error_t entropy_jitter(uint8_t *buffer, size_t size) {
    if (!buffer || size == 0) {
        return ENTROPY_ERROR_INVALID_PARAM;
    }
    
    // Initialize state with high-resolution time
    struct timespec ts;
    entropy_time_now(&ts);
    uint64_t state = ((uint64_t)ts.tv_sec << 32) | ts.tv_nsec;
    
    uint64_t accumulator = 0;
    uint64_t prev_delta = 0;
    size_t bytes_generated = 0;
    int bit_buffer = 0;
    int bits_in_buffer = 0;
    
    while (bytes_generated < size) {
        // Collect timing measurements with memory barrier
        uint64_t t1 = get_timer_cycles();
        
        // CPU-intensive operation with unpredictable timing
        volatile uint64_t dummy = state;
        for (int i = 0; i < 50; i++) {
            dummy = crypto_mix(dummy, i);
        }
        state ^= dummy; // Feed back to prevent optimization
        
        uint64_t t2 = get_timer_cycles();
        
        // Additional timing source: system call
        entropy_time_now(&ts);
        uint64_t sys_time = ((uint64_t)ts.tv_nsec) ^ get_timer_cycles();
        
        uint64_t delta = t2 - t1;
        
        // Extract entropy from timing delta
        uint64_t entropy = extract_timing_entropy(delta, prev_delta, state);
        entropy ^= crypto_mix(sys_time, state + delta);
        
        prev_delta = delta;
        state = crypto_mix(state, entropy);
        
        // Apply von Neumann debiasing to LSBs
        for (int i = 0; i < 8 && bytes_generated < size; i++) {
            uint8_t b1 = (entropy >> (i * 8)) & 1;
            uint8_t b2 = (entropy >> (i * 8 + 1)) & 1;
            
            int debiased = debias_bit_pair(b1, b2);
            if (debiased >= 0) {
                bit_buffer = (bit_buffer << 1) | debiased;
                bits_in_buffer++;
                
                if (bits_in_buffer >= 8) {
                    buffer[bytes_generated++] = (uint8_t)bit_buffer;
                    bit_buffer = 0;
                    bits_in_buffer = 0;
                }
            }
        }
        
        // If debiasing is too slow, fall back to direct extraction
        if (bytes_generated < size && bits_in_buffer == 0) {
            accumulator ^= entropy;
            accumulator = crypto_mix(accumulator, state++);
            
            for (int i = 0; i < 8 && bytes_generated < size; i++) {
                buffer[bytes_generated++] = (accumulator >> (i * 8)) & 0xFF;
            }
        }
    }
    
    // Secure cleanup
    state = 0;
    accumulator = 0;
    prev_delta = 0;
    
    return ENTROPY_SUCCESS;
}

// ============================================================================
// INITIALIZATION
// ============================================================================

entropy_error_t entropy_init(entropy_ctx_t *ctx) {
    if (!ctx) return ENTROPY_ERROR_INVALID_PARAM;
    
    memset(ctx, 0, sizeof(*ctx));
    ctx->dev_random_fd = -1;
    ctx->dev_urandom_fd = -1;
    
    // Detect available entropy sources
    ctx->caps.has_rdrand = rdrand_available();
    ctx->caps.has_rdseed = rdseed_available();
    
    // ARM RNDR support via sysctl detection (safe for VMs)
    // Uses sysctl to check hw.optional.arm.FEAT_RNG - no risky runtime probing
    // On VMs where sysctl fails, falls back to /dev/random (also excellent)
    #if defined(__aarch64__)
    if (rndr_available()) {
        ctx->caps.has_rdrand = 1;  // RNDR maps to RDRAND API
        ctx->caps.has_rdseed = 1;  // RNDRRS maps to RDSEED API
    }
    #endif
    
    ctx->caps.has_jitter = 1; // Always available
    
    // Test getrandom availability
    uint8_t test_byte;
    ctx->caps.has_getrandom = (entropy_getrandom(&test_byte, 1, 0) > 0);
    
#if defined(_WIN32) || defined(_WIN64)
    ctx->caps.has_dev_random = 0;
    ctx->caps.has_dev_urandom = 0;
#else
    // Try to open /dev/random
    ctx->dev_random_fd = open("/dev/random", O_RDONLY | O_NONBLOCK);
    ctx->caps.has_dev_random = (ctx->dev_random_fd >= 0);
    
    // Try to open /dev/urandom
    ctx->dev_urandom_fd = open("/dev/urandom", O_RDONLY);
    ctx->caps.has_dev_urandom = (ctx->dev_urandom_fd >= 0);
#endif
    
    // Determine preferred source (best quality first)
    if (ctx->caps.has_rdseed) {
        ctx->caps.preferred_source = ENTROPY_SOURCE_RDSEED;
    } else if (ctx->caps.has_rdrand) {
        ctx->caps.preferred_source = ENTROPY_SOURCE_RDRAND;
    } else if (ctx->caps.has_getrandom) {
        ctx->caps.preferred_source = ENTROPY_SOURCE_GETRANDOM;
    } else if (ctx->caps.has_dev_random) {
        ctx->caps.preferred_source = ENTROPY_SOURCE_DEV_RANDOM;
    } else if (ctx->caps.has_dev_urandom) {
        ctx->caps.preferred_source = ENTROPY_SOURCE_DEV_URANDOM;
    } else if (ctx->caps.has_jitter) {
        ctx->caps.preferred_source = ENTROPY_SOURCE_JITTER;
    } else {
        ctx->caps.preferred_source = ENTROPY_SOURCE_NONE;
        return ENTROPY_ERROR_NO_SOURCE;
    }
    
    return ENTROPY_SUCCESS;
}

void entropy_free(entropy_ctx_t *ctx) {
    if (!ctx) return;
    
#if !defined(_WIN32) && !defined(_WIN64)
    if (ctx->dev_random_fd >= 0) {
        close(ctx->dev_random_fd);
        ctx->dev_random_fd = -1;
    }
    
    if (ctx->dev_urandom_fd >= 0) {
        close(ctx->dev_urandom_fd);
        ctx->dev_urandom_fd = -1;
    }
#endif
    
    // Zero sensitive data
    memset(ctx, 0, sizeof(*ctx));
}

entropy_capabilities_t entropy_get_capabilities(const entropy_ctx_t *ctx) {
    if (!ctx) {
        entropy_capabilities_t empty = {0};
        return empty;
    }
    return ctx->caps;
}

// ============================================================================
// DEVICE ENTROPY (/dev/random, /dev/urandom)
// ============================================================================

ssize_t entropy_dev_random(entropy_ctx_t *ctx, uint8_t *buffer, size_t size, int blocking) {
#if defined(_WIN32) || defined(_WIN64)
    (void)ctx;
    (void)buffer;
    (void)size;
    (void)blocking;
    errno = ENOSYS;
    return -1;
#else
    if (!ctx || !buffer || size == 0) return -1;
    if (ctx->dev_random_fd < 0) return -1;
    
    // Set blocking mode if requested
    if (blocking) {
        int flags = fcntl(ctx->dev_random_fd, F_GETFL, 0);
        fcntl(ctx->dev_random_fd, F_SETFL, flags & ~O_NONBLOCK);
    }
    
    ssize_t total = 0;
    while (total < (ssize_t)size) {
        ssize_t n = read(ctx->dev_random_fd, buffer + total, size - total);
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN && !blocking) break;
            return -1;
        }
        if (n == 0) break;
        total += n;
    }
    
    return total;
#endif
}

ssize_t entropy_dev_urandom(entropy_ctx_t *ctx, uint8_t *buffer, size_t size) {
#if defined(_WIN32) || defined(_WIN64)
    (void)ctx;
    (void)buffer;
    (void)size;
    errno = ENOSYS;
    return -1;
#else
    if (!ctx || !buffer || size == 0) return -1;
    if (ctx->dev_urandom_fd < 0) return -1;
    
    ssize_t total = 0;
    while (total < (ssize_t)size) {
        ssize_t n = read(ctx->dev_urandom_fd, buffer + total, size - total);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) break;
        total += n;
    }
    
    return total;
#endif
}

// ============================================================================
// ENTROPY COLLECTION
// ============================================================================

entropy_error_t entropy_get_bytes_from_source(
    entropy_ctx_t *ctx,
    uint8_t *buffer,
    size_t size,
    entropy_source_type_t source
) {
    if (!ctx || !buffer || size == 0) {
        return ENTROPY_ERROR_INVALID_PARAM;
    }
    
    switch (source) {
        case ENTROPY_SOURCE_RDSEED: {
            if (!ctx->caps.has_rdseed) return ENTROPY_ERROR_NO_SOURCE;
            
            size_t offset = 0;
            while (offset < size) {
                uint64_t value;
                int success = 0;
                
                #ifdef __aarch64__
                // Use ARM RNDRRS (equivalent to RDSEED)
                success = rndrrs_get_uint64(&value);
                #else
                // Use x86 RDSEED
                success = rdseed_get_uint64(&value);
                #endif
                
                if (!success) {
                    ctx->rdseed_failures++;
                    return ENTROPY_ERROR_INSUFFICIENT;
                }
                
                size_t copy_size = (size - offset) < 8 ? (size - offset) : 8;
                memcpy(buffer + offset, &value, copy_size);
                offset += copy_size;
            }
            ctx->last_source = ENTROPY_SOURCE_RDSEED;
            return ENTROPY_SUCCESS;
        }
        
        case ENTROPY_SOURCE_RDRAND: {
            if (!ctx->caps.has_rdrand) return ENTROPY_ERROR_NO_SOURCE;
            
            size_t offset = 0;
            while (offset < size) {
                uint64_t value;
                int success = 0;
                
                #ifdef __aarch64__
                // Use ARM RNDR (equivalent to RDRAND)
                success = rndr_get_uint64(&value);
                #else
                // Use x86 RDRAND
                success = rdrand_get_uint64(&value);
                #endif
                
                if (!success) {
                    ctx->rdrand_failures++;
                    return ENTROPY_ERROR_INSUFFICIENT;
                }
                
                size_t copy_size = (size - offset) < 8 ? (size - offset) : 8;
                memcpy(buffer + offset, &value, copy_size);
                offset += copy_size;
            }
            ctx->last_source = ENTROPY_SOURCE_RDRAND;
            return ENTROPY_SUCCESS;
        }
        
        case ENTROPY_SOURCE_GETRANDOM: {
            if (!ctx->caps.has_getrandom) return ENTROPY_ERROR_NO_SOURCE;
            
            ssize_t n = entropy_getrandom(buffer, size, 0);
            if (n < 0 || (size_t)n < size) {
                return ENTROPY_ERROR_SYSCALL;
            }
            ctx->last_source = ENTROPY_SOURCE_GETRANDOM;
            return ENTROPY_SUCCESS;
        }
        
        case ENTROPY_SOURCE_DEV_RANDOM: {
            if (!ctx->caps.has_dev_random) return ENTROPY_ERROR_NO_SOURCE;
            
            ssize_t n = entropy_dev_random(ctx, buffer, size, 0);
            if (n < 0 || (size_t)n < size) {
                return ENTROPY_ERROR_INSUFFICIENT;
            }
            ctx->last_source = ENTROPY_SOURCE_DEV_RANDOM;
            return ENTROPY_SUCCESS;
        }
        
        case ENTROPY_SOURCE_DEV_URANDOM: {
            if (!ctx->caps.has_dev_urandom) return ENTROPY_ERROR_NO_SOURCE;
            
            ssize_t n = entropy_dev_urandom(ctx, buffer, size);
            if (n < 0 || (size_t)n < size) {
                return ENTROPY_ERROR_SYSCALL;
            }
            ctx->last_source = ENTROPY_SOURCE_DEV_URANDOM;
            return ENTROPY_SUCCESS;
        }
        
        case ENTROPY_SOURCE_JITTER: {
            entropy_error_t err = entropy_jitter(buffer, size);
            if (err != ENTROPY_SUCCESS) return err;
            ctx->last_source = ENTROPY_SOURCE_JITTER;
            return ENTROPY_SUCCESS;
        }
        
        default:
            return ENTROPY_ERROR_NO_SOURCE;
    }
}

entropy_error_t entropy_get_bytes(entropy_ctx_t *ctx, uint8_t *buffer, size_t size) {
    if (!ctx || !buffer || size == 0) {
        return ENTROPY_ERROR_INVALID_PARAM;
    }
    
    // Try sources in priority order
    entropy_source_type_t sources[] = {
        ENTROPY_SOURCE_RDSEED,
        ENTROPY_SOURCE_RDRAND,
        ENTROPY_SOURCE_GETRANDOM,
        ENTROPY_SOURCE_DEV_RANDOM,
        ENTROPY_SOURCE_DEV_URANDOM,
        ENTROPY_SOURCE_JITTER
    };
    
    for (size_t i = 0; i < sizeof(sources) / sizeof(sources[0]); i++) {
        entropy_error_t err = entropy_get_bytes_from_source(ctx, buffer, size, sources[i]);
        if (err == ENTROPY_SUCCESS) {
            ctx->total_bytes += size;
            return ENTROPY_SUCCESS;
        }
    }
    
    return ENTROPY_ERROR_NO_SOURCE;
}

entropy_error_t entropy_get_uint64(entropy_ctx_t *ctx, uint64_t *value) {
    if (!ctx || !value) return ENTROPY_ERROR_INVALID_PARAM;
    
    return entropy_get_bytes(ctx, (uint8_t*)value, sizeof(*value));
}

// ============================================================================
// QUALITY ASSESSMENT
// ============================================================================

double entropy_quality_estimate(entropy_source_type_t source) {
    switch (source) {
        case ENTROPY_SOURCE_RDSEED:
            return 8.0; // Full entropy (conditioned by hardware)
        case ENTROPY_SOURCE_RDRAND:
            return 8.0; // Full entropy (hardware TRNG)
        case ENTROPY_SOURCE_GETRANDOM:
            return 8.0; // Kernel certified
        case ENTROPY_SOURCE_DEV_RANDOM:
            return 8.0; // Kernel entropy pool
        case ENTROPY_SOURCE_DEV_URANDOM:
            return 7.8; // High quality
        case ENTROPY_SOURCE_JITTER:
            return 5.0; // Variable, conservative estimate
        default:
            return 0.0;
    }
}

const char* entropy_source_name(entropy_source_type_t source) {
    switch (source) {
        case ENTROPY_SOURCE_RDSEED: return "RDSEED";
        case ENTROPY_SOURCE_RDRAND: return "RDRAND";
        case ENTROPY_SOURCE_GETRANDOM: return "getrandom()";
        case ENTROPY_SOURCE_DEV_RANDOM: return "/dev/random";
        case ENTROPY_SOURCE_DEV_URANDOM: return "/dev/urandom";
        case ENTROPY_SOURCE_JITTER: return "CPU Jitter";
        case ENTROPY_SOURCE_FALLBACK: return "Fallback";
        case ENTROPY_SOURCE_NONE: return "None";
        default: return "Unknown";
    }
}

void entropy_print_stats(const entropy_ctx_t *ctx) {
    if (!ctx) return;
    
    printf("=== Entropy Source Statistics ===\n");
    printf("Available sources:\n");
    printf("  RDSEED:       %s\n", ctx->caps.has_rdseed ? "Yes" : "No");
    printf("  RDRAND:       %s\n", ctx->caps.has_rdrand ? "Yes" : "No");
    printf("  getrandom():  %s\n", ctx->caps.has_getrandom ? "Yes" : "No");
    printf("  /dev/random:  %s\n", ctx->caps.has_dev_random ? "Yes" : "No");
    printf("  /dev/urandom: %s\n", ctx->caps.has_dev_urandom ? "Yes" : "No");
    printf("  CPU Jitter:   %s\n", ctx->caps.has_jitter ? "Yes" : "No");
    printf("\n");
    printf("Preferred source: %s\n", entropy_source_name(ctx->caps.preferred_source));
    printf("Last used source: %s\n", entropy_source_name(ctx->last_source));
    printf("Total bytes collected: %" PRIu64 "\n", ctx->total_bytes);
    printf("RDRAND failures: %" PRIu64 "\n", ctx->rdrand_failures);
    printf("RDSEED failures: %" PRIu64 "\n", ctx->rdseed_failures);
    printf("Quality estimate: %.1f bits/byte\n", 
           entropy_quality_estimate(ctx->caps.preferred_source));
}
