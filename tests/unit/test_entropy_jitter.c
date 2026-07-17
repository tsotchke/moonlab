/**
 * @file test_entropy_jitter.c
 * @brief Regression checks for timing-jitter entropy fallbacks.
 */

#include "../../src/applications/hardware_entropy.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

size_t entropy_util_jitter_bytes(uint8_t* buffer, size_t size);

static int failures = 0;

#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

static int all_zero(const uint8_t *buffer, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (buffer[i] != 0) {
            return 0;
        }
    }
    return 1;
}

static void test_hardware_entropy_jitter_emits_bytes(void) {
    fprintf(stdout, "\n-- hardware entropy jitter emits bytes --\n");

    uint8_t out[32];
    memset(out, 0, sizeof(out));

    entropy_error_t rc = entropy_jitter(out, sizeof(out));
    CHECK(rc == ENTROPY_SUCCESS, "entropy_jitter succeeds");
    CHECK(!all_zero(out, sizeof(out)), "entropy_jitter output is not all zero");
    CHECK(entropy_jitter(NULL, sizeof(out)) == ENTROPY_ERROR_INVALID_PARAM,
          "entropy_jitter rejects NULL buffer");
}

static void test_utils_entropy_jitter_bytes_emits_bytes(void) {
    fprintf(stdout, "\n-- utils entropy jitter emits bytes --\n");

    uint8_t out[32];
    memset(out, 0, sizeof(out));

    size_t n = entropy_util_jitter_bytes(out, sizeof(out));
    CHECK(n == sizeof(out), "entropy_util_jitter_bytes filled requested buffer");
    CHECK(!all_zero(out, n), "entropy_util_jitter_bytes output is not all zero");
    CHECK(entropy_util_jitter_bytes(NULL, sizeof(out)) == 0,
          "entropy_util_jitter_bytes rejects NULL buffer");
}

int main(void) {
    fprintf(stdout, "=== entropy jitter unit tests ===\n");
    test_hardware_entropy_jitter_emits_bytes();
    test_utils_entropy_jitter_bytes_emits_bytes();
    fprintf(stdout, "\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
