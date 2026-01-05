/**
 * @file hw_rng_probe.c
 * @brief Minimal helper to probe hardware RNG instructions safely
 *
 * This is a standalone executable that attempts a single hardware RNG read.
 * It is intentionally tiny and single-threaded so it can be safely spawned
 * from the main library (which may have background threads).
 *
 * Usage: hw_rng_probe --mode=rndr|rndrrs|rdrand|rdseed
 * Output: 16-char hex value on success, exit 0
 *         exit 1 on failure or if instruction not supported
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

// ============================================================================
// x86_64 RDRAND/RDSEED
// ============================================================================

#if defined(__x86_64__) || defined(__i386__)

static int try_rdrand(uint64_t *out) {
    unsigned long long v;
    unsigned char ok;
    for (int i = 0; i < 10; ++i) {
        __asm__ volatile(
            "rdrand %0; setc %1"
            : "=r"(v), "=qm"(ok)
            :
            : "cc"
        );
        if (ok) {
            *out = (uint64_t)v;
            return 1;
        }
    }
    return 0;
}

static int try_rdseed(uint64_t *out) {
    unsigned long long v;
    unsigned char ok;
    for (int i = 0; i < 10; ++i) {
        __asm__ volatile(
            "rdseed %0; setc %1"
            : "=r"(v), "=qm"(ok)
            :
            : "cc"
        );
        if (ok) {
            *out = (uint64_t)v;
            return 1;
        }
    }
    return 0;
}

#endif

// ============================================================================
// ARM64 RNDR/RNDRRS
// ============================================================================

#if defined(__aarch64__)

static int try_rndr(uint64_t *out) {
    uint64_t result = 0;
    int ok = 0;
    __asm__ volatile(
        "mrs %0, s3_3_c2_c4_0\n"
        "cset %w1, ne"
        : "=r"(result), "=r"(ok)
        :
        : "cc"
    );
    if (ok) {
        *out = result;
        return 1;
    }
    return 0;
}

static int try_rndrrs(uint64_t *out) {
    uint64_t result = 0;
    int ok = 0;
    __asm__ volatile(
        "mrs %0, s3_3_c2_c4_1\n"
        "cset %w1, ne"
        : "=r"(result), "=r"(ok)
        :
        : "cc"
    );
    if (ok) {
        *out = result;
        return 1;
    }
    return 0;
}

#endif

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char **argv) {
    const char *mode = NULL;

    // Parse --mode=xxx argument
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--mode=", 7) == 0) {
            mode = argv[i] + 7;
        }
    }

    if (!mode) {
        fprintf(stderr, "Usage: hw_rng_probe --mode=rndr|rndrrs|rdrand|rdseed\n");
        return 1;
    }

    uint64_t value = 0;
    int ok = 0;

#if defined(__x86_64__) || defined(__i386__)
    if (strcmp(mode, "rdrand") == 0) {
        ok = try_rdrand(&value);
    } else if (strcmp(mode, "rdseed") == 0) {
        ok = try_rdseed(&value);
    }
#endif

#if defined(__aarch64__)
    if (strcmp(mode, "rndr") == 0) {
        ok = try_rndr(&value);
    } else if (strcmp(mode, "rndrrs") == 0) {
        ok = try_rndrrs(&value);
    }
#endif

    if (!ok) {
        return 1;
    }

    // Print as hex for parent to parse
    printf("%016llx\n", (unsigned long long)value);
    return 0;
}
