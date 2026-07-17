/**
 * @file test_qrng_delivery.c
 * @brief Release-path assurance and fail-closed QRNG regression tests.
 */

#include "../../src/applications/moonlab_export.h"
#include "../../src/applications/qrng.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(condition, message) do {                                      \
    if (!(condition)) {                                                     \
        fprintf(stderr, "FAIL: %s\n", (message));                          \
        failures++;                                                         \
    }                                                                       \
} while (0)

static void test_release_capabilities(void) {
    moonlab_qrng_status_t before;
    memset(&before, 0, sizeof(before));
    CHECK(moonlab_qrng_get_status(&before) == 0, "initial status query");
    CHECK(before.struct_size == sizeof(before), "status size is versioned");
    CHECK(before.api_version == 1, "status API version is one");

    const uint64_t required =
        MOONLAB_QRNG_CAP_HARDWARE_OS_ENTROPY |
        MOONLAB_QRNG_CAP_CONTINUOUS_HEALTH_TESTS |
        MOONLAB_QRNG_CAP_SHAKE256_CONDITIONED |
        MOONLAB_QRNG_CAP_BELL_SIMULATION_GATED |
        MOONLAB_QRNG_CAP_THREAD_SAFE;
    CHECK((before.capabilities & required) == required,
          "release protections are active");
    CHECK((before.capabilities & MOONLAB_QRNG_CAP_DEVICE_INDEPENDENT_SOURCE) == 0,
          "simulator does not impersonate an independent physical source");
    CHECK((before.capabilities & MOONLAB_QRNG_CAP_FIPS140_VALIDATED) == 0,
          "unvalidated binary does not claim a FIPS module boundary");

    uint8_t first[64];
    uint8_t second[64];
    CHECK(moonlab_qrng_bytes(first, sizeof(first)) == 0,
          "first conditioned draw succeeds");
    CHECK(moonlab_qrng_bytes(second, sizeof(second)) == 0,
          "second conditioned draw succeeds");
    CHECK(memcmp(first, second, sizeof(first)) != 0,
          "domain-separated draws are distinct");

    moonlab_qrng_status_t after;
    memset(&after, 0, sizeof(after));
    CHECK(moonlab_qrng_get_status(&after) == 0, "post-draw status query");
    CHECK(after.conditioned_requests >= before.conditioned_requests + 2,
          "conditioned request counter advances");
    CHECK(after.raw_bytes_generated >= before.raw_bytes_generated + 128,
          "v3 raw-byte counter advances");
    CHECK(after.bell_tests_performed >= 1, "first epoch was Bell gated");
    CHECK(after.bell_tests_passed == after.bell_tests_performed,
          "every delivered epoch passed its gate");
    CHECK((after.capabilities & MOONLAB_QRNG_CAP_BELL_EPOCH_CERTIFIED) != 0,
          "live status reports the certified epoch");
    CHECK(after.minimum_chsh > 2.0, "simulated CHSH exceeded classical bound");
}

static void test_bell_failure_zeroizes_output(void) {
    qrng_v3_config_t config;
    qrng_v3_get_default_config(&config);
    config.mode = QRNG_V3_MODE_BELL_VERIFIED;
    config.enable_bell_monitoring = 1;
    config.bell_test_interval = 64;
    config.min_acceptable_chsh = 3.0; /* Above Tsirelson: deterministic reject. */
    config.enable_background_entropy = 0;

    qrng_v3_ctx_t *ctx = NULL;
    CHECK(qrng_v3_init_with_config(&ctx, &config) == QRNG_V3_SUCCESS,
          "isolated fail-closed context initializes");
    if (!ctx) return;

    uint8_t output[64];
    memset(output, 0xA5, sizeof(output));
    CHECK(qrng_v3_bytes(ctx, output, sizeof(output)) ==
              QRNG_V3_ERROR_BELL_TEST_FAILED,
          "impossible Bell threshold rejects before delivery");

    int all_zero = 1;
    for (size_t i = 0; i < sizeof(output); ++i) {
        if (output[i] != 0) all_zero = 0;
    }
    CHECK(all_zero, "rejected output is securely zeroized");
    CHECK(ctx->stats.bytes_generated == 0,
          "rejected bytes are not counted as delivered");
    CHECK(ctx->buffer_pos == ctx->output_buffer_size,
          "rejection happened before raw output generation");

    qrng_v3_free(ctx);
}

int main(void) {
    test_release_capabilities();
    test_bell_failure_zeroizes_output();
    if (failures == 0) {
        puts("conditioned QRNG delivery and fail-closed gates passed");
    }
    return failures == 0 ? 0 : 1;
}
