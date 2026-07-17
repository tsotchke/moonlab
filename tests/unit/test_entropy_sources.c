/**
 * @file test_entropy_sources.c
 * @brief Regression checks for direct entropy source collection.
 */

#include "../../src/utils/entropy.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

static int buffer_is_all_zero(const uint8_t *buffer, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (buffer[i] != 0) {
            return 0;
        }
    }
    return 1;
}

static int buffer_has_variation(const uint8_t *buffer, size_t size) {
    if (size == 0) {
        return 0;
    }
    for (size_t i = 1; i < size; i++) {
        if (buffer[i] != buffer[0]) {
            return 1;
        }
    }
    return 0;
}

static void test_direct_jitter_bytes(void) {
    fprintf(stdout, "\n-- direct jitter bytes --\n");

    uint8_t out[96];
    memset(out, 0, sizeof(out));

    size_t n = entropy_util_jitter_bytes(out, sizeof(out));
    CHECK(n == sizeof(out),
          "entropy_util_jitter_bytes returned %zu of %zu bytes", n, sizeof(out));
    CHECK(!buffer_is_all_zero(out, sizeof(out)),
          "jitter output is not all zero");
    CHECK(buffer_has_variation(out, sizeof(out)),
          "jitter output has byte variation");
}

static void test_jitter_context_reseed(void) {
    fprintf(stdout, "\n-- jitter context reseed --\n");

    entropy_util_ctx_t *ctx = entropy_util_create_with_source(ENTROPY_UTIL_SOURCE_JITTER);
    CHECK(ctx != NULL, "jitter entropy context allocated");
    if (!ctx) {
        return;
    }

    uint8_t out[128];
    memset(out, 0, sizeof(out));
    size_t n = entropy_util_bytes(ctx, out, sizeof(out));
    CHECK(n == sizeof(out),
          "entropy_util_bytes returned %zu of %zu bytes", n, sizeof(out));
    CHECK(ctx->bytes_collected >= sizeof(out),
          "context recorded collected bytes");
    CHECK(!buffer_is_all_zero(out, sizeof(out)),
          "context output is not all zero");
    CHECK(buffer_has_variation(out, sizeof(out)),
          "context output has byte variation");

    entropy_util_destroy(ctx);
}

int main(void) {
    fprintf(stdout, "=== entropy source unit tests ===\n");
    test_direct_jitter_bytes();
    test_jitter_context_reseed();
    fprintf(stdout, "\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
