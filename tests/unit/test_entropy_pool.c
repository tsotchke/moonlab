/**
 * @file test_entropy_pool.c
 * @brief Regression checks for entropy-pool initialization.
 */

#include "../../src/applications/entropy_pool.h"

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

static void test_small_pool_startup_prefill_is_capped(void) {
    fprintf(stdout, "\n-- small pool startup prefill is capped --\n");

    entropy_pool_config_t config = {
        .pool_size = 64,
        .refill_threshold = 16,
        .chunk_size = 64,
        .enable_background_thread = 0,
        .min_entropy = 4.0
    };

    entropy_pool_ctx_t *ctx = NULL;
    int rc = entropy_pool_init_with_config(&ctx, &config);
    CHECK(rc == 0, "entropy_pool_init_with_config succeeds for 64-byte pool");
    CHECK(ctx != NULL, "context allocated");
    if (!ctx) {
        return;
    }

    entropy_pool_stats_t stats;
    memset(&stats, 0, sizeof(stats));
    rc = entropy_pool_get_stats(ctx, &stats);
    CHECK(rc == 0, "entropy_pool_get_stats succeeds");
    CHECK(stats.current_fill_level <= config.pool_size,
          "startup fill level (%zu) does not exceed pool size (%zu)",
          stats.current_fill_level, config.pool_size);

    uint8_t out[64];
    memset(out, 0, sizeof(out));
    rc = entropy_pool_get_bytes(ctx, out, sizeof(out));
    CHECK(rc == 0, "64-byte read succeeds after initialization");

    entropy_pool_free(ctx);
}

int main(void) {
    fprintf(stdout, "=== entropy_pool unit tests ===\n");
    test_small_pool_startup_prefill_is_capped();
    fprintf(stdout, "\n%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
