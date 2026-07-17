/**
 * @file test_qrng_bell_certified_output.c
 * @brief BELL_VERIFIED delivered bytes are Pironio-bounded, Toeplitz-extracted,
 *        and bell_epoch_certified is set ONLY when that path actually ran.
 *
 * The historical failure this pins: qrng_v3_verify_quantum CHSH-tested a fresh
 * scratch |Phi+> that never fed extraction, so the epoch gate passed
 * regardless of the delivered bytes and the certified capability bit was a
 * lie.  Now the CHSH statistic is measured on the same simulated measurement
 * stream that is Toeplitz-extracted into the output, the Pironio bound sets
 * the extraction ratio, and bell_epochs_certified increments only inside that
 * path.  This certifies simulator integrity, not device independence -- the
 * DEVICE_INDEPENDENT_SOURCE capability stays false.
 */

#include "../../src/applications/qrng.h"
#include "../../src/applications/moonlab_export.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                                  \
    if (!(cond)) {                                                  \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);        \
        failures++;                                                 \
    } else {                                                        \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);        \
    }                                                               \
} while (0)

static int all_zero(const uint8_t *b, size_t n) {
    uint8_t acc = 0;
    for (size_t i = 0; i < n; i++) acc |= b[i];
    return acc == 0;
}

/* A certified BELL_VERIFIED epoch runs on the delivered stream. */
static void test_certified_delivery(void) {
    fprintf(stdout, "\n-- BELL_VERIFIED delivered bytes are certified --\n");
    qrng_v3_config_t cfg;
    qrng_v3_get_default_config(&cfg);
    cfg.mode = QRNG_V3_MODE_BELL_VERIFIED;
    cfg.enable_background_entropy = 0;

    qrng_v3_ctx_t *ctx = NULL;
    CHECK(qrng_v3_init_with_config(&ctx, &cfg) == QRNG_V3_SUCCESS,
          "BELL_VERIFIED context initializes");
    if (!ctx) return;

    uint8_t out[600];
    memset(out, 0, sizeof(out));
    CHECK(qrng_v3_bytes(ctx, out, sizeof(out)) == QRNG_V3_SUCCESS,
          "certified draw succeeds");
    CHECK(!all_zero(out, sizeof(out)), "certified output is non-zero");

    qrng_v3_stats_t st;
    qrng_v3_get_stats(ctx, &st);
    CHECK(st.bell_epochs_certified >= 1,
          "at least one epoch ran the certified extraction (got %llu)",
          (unsigned long long)st.bell_epochs_certified);
    CHECK(st.bell_tests_performed >= 1, "an epoch was gated");
    CHECK(st.bell_tests_passed == st.bell_tests_performed,
          "every performed epoch passed its gate");
    CHECK(st.average_chsh > 2.0 && st.min_chsh > 2.0,
          "measured CHSH exceeded the classical bound (avg=%.4f min=%.4f)",
          st.average_chsh, st.min_chsh);
    CHECK(qrng_v3_output_is_uniform(ctx) == 1,
          "BELL_VERIFIED raw output is contractually uniform");

    qrng_v3_free(ctx);
}

/* An impossible CHSH threshold must fail closed WITHOUT marking any epoch
 * certified -- this is the "flag reflects reality" invariant. */
static void test_failclosed_leaves_uncertified(void) {
    fprintf(stdout, "\n-- Impossible gate fails closed and certifies nothing --\n");
    qrng_v3_config_t cfg;
    qrng_v3_get_default_config(&cfg);
    cfg.mode = QRNG_V3_MODE_BELL_VERIFIED;
    cfg.enable_background_entropy = 0;
    cfg.min_acceptable_chsh = 3.0;      /* above Tsirelson: always rejected */

    qrng_v3_ctx_t *ctx = NULL;
    CHECK(qrng_v3_init_with_config(&ctx, &cfg) == QRNG_V3_SUCCESS,
          "fail-closed context initializes");
    if (!ctx) return;

    uint8_t out[128];
    memset(out, 0xA5, sizeof(out));
    CHECK(qrng_v3_bytes(ctx, out, sizeof(out)) == QRNG_V3_ERROR_BELL_TEST_FAILED,
          "impossible threshold is rejected before delivery");
    CHECK(all_zero(out, sizeof(out)), "rejected output is zeroized");

    qrng_v3_stats_t st;
    qrng_v3_get_stats(ctx, &st);
    CHECK(st.bell_epochs_certified == 0,
          "no epoch is marked certified when the gate never passed");
    CHECK(st.bytes_generated == 0, "rejected bytes are not counted as delivered");

    qrng_v3_free(ctx);
}

/* GROVER raw output is non-uniform by contract. */
static void test_grover_nonuniform_contract(void) {
    fprintf(stdout, "\n-- GROVER raw output flagged non-uniform --\n");
    qrng_v3_config_t cfg;
    qrng_v3_get_default_config(&cfg);
    cfg.mode = QRNG_V3_MODE_GROVER;
    cfg.enable_background_entropy = 0;
    qrng_v3_ctx_t *ctx = NULL;
    CHECK(qrng_v3_init_with_config(&ctx, &cfg) == QRNG_V3_SUCCESS,
          "GROVER context initializes");
    if (!ctx) return;
    CHECK(qrng_v3_output_is_uniform(ctx) == 0,
          "GROVER raw output is contractually non-uniform");
    qrng_v3_free(ctx);
}

/* The stable release path exposes the certified bit only after it ran. */
static void test_release_status_bit(void) {
    fprintf(stdout, "\n-- Release path reports the certified epoch --\n");
    uint8_t buf[64];
    CHECK(moonlab_qrng_bytes(buf, sizeof(buf)) == 0, "release draw succeeds");

    moonlab_qrng_status_t status;
    memset(&status, 0, sizeof(status));
    CHECK(moonlab_qrng_get_status(&status) == 0, "status query succeeds");
    CHECK((status.capabilities & MOONLAB_QRNG_CAP_BELL_EPOCH_CERTIFIED) != 0,
          "certified capability set after a certified draw");
    CHECK((status.capabilities & MOONLAB_QRNG_CAP_DEVICE_INDEPENDENT_SOURCE) == 0,
          "device-independence is not claimed");
    CHECK(status.minimum_chsh > 2.0, "reported minimum CHSH beats classical bound");
}

int main(void) {
    fprintf(stdout, "=== QRNG Bell-certified output ===\n");
    test_certified_delivery();
    test_failclosed_leaves_uncertified();
    test_grover_nonuniform_contract();
    test_release_status_bit();
    fprintf(stdout, "\n=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
