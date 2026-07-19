/**
 * @file    fuzz_target_entropy_input.c
 * @brief   Surface: caller-supplied bytes into the entropy + health path.
 *
 * Several entropy APIs accept a caller byte stream and run conditioning,
 * statistical estimation, or NIST SP 800-90B health tests over it.  This
 * target feeds adversarial streams into every such entry point:
 *
 *   - `entropy_util_extract`  -- hash-based conditioner (input -> output).
 *   - `entropy_util_estimate` -- Shannon-entropy estimator over the bytes.
 *   - `entropy_util_test_data`-- chi-squared + entropy quality assessment.
 *   - `entropy_util_mix`      -- fold caller bytes into a pool context.
 *   - `health_tests_startup`  -- SP 800-90B startup RCT/APT over a sample
 *                                block.
 *   - `health_tests_run_batch`-- continuous RCT/APT over a sample block,
 *                                which allocates an APT window buffer.
 *
 * Contract: any input length (including 0 and multi-MB) is handled
 * without OOB access, and the health context's APT window allocation is
 * always released (checked by LeakSanitizer at exit).
 */

#include "fuzz_common.h"

#include "utils/entropy.h"
#include "applications/health_tests.h"

#include <stdlib.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    const uint8_t *p   = data;
    const uint8_t *end = data + size;

    /* First byte selects an output length for the conditioner so both
     * the short-tail (< 8) and the exact-multiple cases are hit. */
    uint32_t out_len = (fuzz_u8(&p, end) % 129u); /* 0..128 */
    const uint8_t *body = p;
    size_t body_len = (size_t)(end - p);

    /* Conditioner: extract out_len bytes from the remaining stream. */
    if (out_len > 0 && body_len > 0) {
        uint8_t out[128];
        (void)entropy_util_extract(body, body_len, out, out_len);
    } else {
        /* Still call it with the degenerate zero args to exercise the
         * NULL/zero-length guard. */
        uint8_t one;
        (void)entropy_util_extract(body, body_len, &one, 0);
    }

    /* Statistical estimators over the raw stream. */
    (void)entropy_util_estimate(body, body_len);
    entropy_util_quality_t q;
    (void)entropy_util_test_data(body, body_len, &q);

    /* Pool mixing needs a live context. */
    entropy_util_ctx_t *ctx = entropy_util_create();
    if (ctx) {
        entropy_util_mix(ctx, body, body_len);
        entropy_util_destroy(ctx);
    }

    /* SP 800-90B health tests: startup + continuous batch.  Both ingest
     * the caller sample block byte-for-byte. */
    health_test_ctx_t hctx;
    if (health_tests_init(&hctx) == HEALTH_SUCCESS) {
        (void)health_tests_startup(&hctx, body, body_len);
        (void)health_tests_run_batch(&hctx, body, body_len);
        health_tests_free(&hctx);
    }

    return 0;
}
