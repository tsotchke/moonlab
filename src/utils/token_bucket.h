/**
 * @file  token_bucket.h
 * @brief Thread-safe token-bucket rate limiter (since v1.0.3).
 *
 * Mechanism only -- the public moonlab provides this primitive so
 * private overlays can build per-tenant / per-account quota
 * enforcement inside their admission hook
 * (@ref moonlab_control_server_set_admission_hook) without
 * re-implementing the lock-free token-bucket math.
 *
 * Policy decisions -- what burst / refill rate each tenant gets,
 * which tenants are rate-limited at all, how the bucket key maps
 * to the tenant table -- stay in the overlay per the open-core
 * rules (`COMMUNITY_EDITION.md`).
 *
 * Algorithm:
 *
 *   Each bucket holds a non-negative double `tokens` (current
 *   balance) and remembers the wall-clock at the last refill.
 *   `take(n)` lazily computes elapsed_seconds * refill_rate tokens
 *   to add (clamped to `burst`), then atomically subtracts `n` and
 *   reports success/failure.  No locks; concurrent takes serialise
 *   via a single atomic compare-and-swap.
 *
 *   Resolution is `1/refill_rate` seconds per token, so the lowest
 *   meaningful refill is ~1 token/sec; for sub-second precision
 *   the caller multiplies the token-count argument by an integer
 *   scale (e.g. count shots instead of jobs).
 */

#ifndef MOONLAB_TOKEN_BUCKET_H
#define MOONLAB_TOKEN_BUCKET_H

#include "../applications/moonlab_api.h"

#include <stdatomic.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque token-bucket state.  Zero-initialised buckets are
 *        valid (start empty, refilling from t=0).  Callers
 *        normally use moonlab_token_bucket_init() to set the
 *        capacity + refill explicitly.
 */
typedef struct {
    _Atomic uint64_t tokens_x1e6;     /**< current balance * 1e6 (fixed-point) */
    _Atomic uint64_t last_refill_us;  /**< wall-clock of last refill, microsec */
    uint64_t         burst_x1e6;      /**< max tokens * 1e6 */
    uint64_t         refill_per_sec;  /**< tokens added per wall-clock second */
} moonlab_token_bucket_t;

/**
 * @brief Initialise a bucket.  Starts full (tokens = burst).
 *
 * @param[out] bkt           Bucket to initialise.
 * @param[in]  burst         Maximum tokens the bucket can hold.
 * @param[in]  refill_per_sec Tokens added per wall-clock second.
 *                            Set to 0 to disable refilling (one-shot
 *                            budget that must be replenished by an
 *                            explicit refill() call).
 */
MOONLAB_API void
moonlab_token_bucket_init(moonlab_token_bucket_t *bkt,
                          uint64_t                burst,
                          uint64_t                refill_per_sec);

/**
 * @brief Attempt to remove `n` tokens.  Lazily refills based on
 *        wall-clock elapsed since the previous call before
 *        deciding.  Concurrent calls serialise via a single
 *        compare-and-swap.
 *
 * @param[in,out] bkt  Bucket.
 * @param[in]     n    Token count to remove (caller-defined unit;
 *                     can be "1 per job" or "num_shots per call").
 *
 * @return  1 on success (tokens removed), 0 on insufficient balance
 *          (no tokens removed).
 */
MOONLAB_API int
moonlab_token_bucket_take(moonlab_token_bucket_t *bkt, uint64_t n);

/**
 * @brief Add `n` tokens to the bucket, capped at the configured
 *        burst.  Used by overlays that pre-credit a tenant (e.g.
 *        on payment) or that refill on a custom schedule.
 *
 *        Thread-safe; concurrent calls compose correctly.
 */
MOONLAB_API void
moonlab_token_bucket_refill(moonlab_token_bucket_t *bkt, uint64_t n);

/**
 * @brief Read the current balance.  Lazily refills under the hood
 *        before returning so the answer reflects elapsed time.
 *        Thread-safe.
 */
MOONLAB_API uint64_t
moonlab_token_bucket_peek(moonlab_token_bucket_t *bkt);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_TOKEN_BUCKET_H */
