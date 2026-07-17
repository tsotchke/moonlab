/**
 * @file test_entropy_health_adversarial.c
 * @brief Anti-theater check for the SP 800-90B RCT/APT health-test path.
 *
 * A health test that never rejects anything is decoration.  This harness feeds
 * the Repetition Count Test and Adaptive Proportion Test pathological streams
 * that a real entropy source must never emit, and asserts they are REJECTED:
 *
 *   - constant       : a stuck source (all one value)          -> RCT
 *   - period-2        : 0xAA,0x55,...                            -> APT
 *   - slowly-drifting : locally-constant runs that inch upward   -> RCT
 *   - low-entropy     : one value dominates ~90% of the stream   -> APT
 *
 * As the negative control it feeds genuine SHAKE256-conditioned entropy from
 * moonlab_qrng_bytes and asserts it PASSES.  If a positive control fails to
 * fire (a bad stream slips through), that is a real defect in the health-test
 * path owned by the QRNG/crypto lane -- it is recorded in FINDINGS.md and
 * gates.
 *
 * The default config (H_min = 4 bits/sample) gives RCT cutoff 9 and, for a
 * 512-sample window, APT cutoff ~71 -- both comfortably inside the margins the
 * pathological streams blow through, so the controls are not flaky.
 */

#include "stat_common.h"
#include "../../src/applications/health_tests.h"
#include "../../src/applications/moonlab_export.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t sm_state = 0xD1B54A32D192ED03ULL;
static uint64_t sm_next(void) {
    uint64_t z = (sm_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

#define NSAMP 2048

/* Returns the health error code for a stream (HEALTH_SUCCESS == passed). */
static health_error_t run_stream(const uint8_t *s, size_t n) {
    health_test_ctx_t ctx;
    if (health_tests_init(&ctx) != HEALTH_SUCCESS) return HEALTH_ERROR_NOT_INITIALIZED;
    health_error_t r = health_tests_run_batch(&ctx, s, n);
    health_tests_free(&ctx);
    return r;
}

static const char *err_name(health_error_t e) {
    switch (e) {
        case HEALTH_SUCCESS:            return "PASS";
        case HEALTH_ERROR_RCT_FAILURE:  return "RCT_REJECT";
        case HEALTH_ERROR_APT_FAILURE:  return "APT_REJECT";
        default:                        return "OTHER";
    }
}

int main(void) {
    printf("Entropy health-test adversarial inputs\n");

    uint8_t buf[NSAMP];

    /* --- positive controls: each MUST be rejected --- */
    memset(buf, 0x00, NSAMP);
    health_error_t e_const = run_stream(buf, NSAMP);

    for (int i = 0; i < NSAMP; i++) buf[i] = (i & 1) ? 0x55 : 0xAA;
    health_error_t e_period2 = run_stream(buf, NSAMP);

    for (int i = 0; i < NSAMP; i++) buf[i] = (uint8_t)((i / 20) & 0xFF); /* 20-long runs */
    health_error_t e_drift = run_stream(buf, NSAMP);

    for (int i = 0; i < NSAMP; i++)
        buf[i] = (sm_next() % 10 == 0) ? (uint8_t)(sm_next() & 0xFF) : 0x7E; /* ~90% 0x7E */
    health_error_t e_lowent = run_stream(buf, NSAMP);

    /* --- negative control: genuine conditioned entropy MUST pass --- */
    health_error_t e_genuine;
    if (moonlab_qrng_bytes(buf, NSAMP) != 0) {
        fprintf(stderr, "moonlab_qrng_bytes failed; cannot run genuine-entropy control\n");
        e_genuine = HEALTH_ERROR_NOT_INITIALIZED;
    } else {
        e_genuine = run_stream(buf, NSAMP);
    }

    printf("  constant        -> %s\n", err_name(e_const));
    printf("  period-2        -> %s\n", err_name(e_period2));
    printf("  slowly-drifting -> %s\n", err_name(e_drift));
    printf("  low-entropy     -> %s\n", err_name(e_lowent));
    printf("  genuine         -> %s\n", err_name(e_genuine));

    int rej_const   = (e_const   != HEALTH_SUCCESS);
    int rej_period2 = (e_period2 != HEALTH_SUCCESS);
    int rej_drift   = (e_drift   != HEALTH_SUCCESS);
    int rej_lowent  = (e_lowent  != HEALTH_SUCCESS);
    int pass_genuine = (e_genuine == HEALTH_SUCCESS);

    int all_ok = rej_const && rej_period2 && rej_drift && rej_lowent && pass_genuine;

    char stats[512];
    snprintf(stats, sizeof(stats),
        "{\"constant\":\"%s\",\"period2\":\"%s\",\"drift\":\"%s\","
        "\"low_entropy\":\"%s\",\"genuine\":\"%s\",\"rct_cutoff\":9,"
        "\"apt_cutoff_approx\":71}",
        err_name(e_const), err_name(e_period2), err_name(e_drift),
        err_name(e_lowent), err_name(e_genuine));
    stat_emit_result("entropy_health_rejects_bad", all_ok, 1, stats);

    if (!rej_const)   fprintf(stderr, "FINDING: constant stream NOT rejected by health tests\n");
    if (!rej_period2) fprintf(stderr, "FINDING: period-2 stream NOT rejected by health tests\n");
    if (!rej_drift)   fprintf(stderr, "FINDING: slowly-drifting stream NOT rejected by health tests\n");
    if (!rej_lowent)  fprintf(stderr, "FINDING: low-entropy stream NOT rejected by health tests\n");
    if (!pass_genuine) fprintf(stderr, "FAIL: genuine conditioned entropy rejected (false positive)\n");

    return all_ok ? 0 : 1;
}
