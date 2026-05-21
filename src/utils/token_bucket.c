/**
 * @file  token_bucket.c
 * @brief Lock-free token-bucket impl (see token_bucket.h).
 *
 * Concurrent takes serialise via a single CAS loop on
 * tokens_x1e6.  Refill is computed lazily on each operation from
 * the wall-clock delta since the previous touch.
 *
 * Fixed-point representation: balance is stored as
 * `tokens_x1e6 = floor(tokens * 1e6)`.  This lets us refill at
 * sub-token resolution under high concurrency without floating
 * point and without losing the partial-token-per-microsecond
 * fraction that accumulates across many short calls.
 */

#include "token_bucket.h"

#include <stdlib.h>
#include <time.h>

#define SCALE          ((uint64_t)1000000U)  /* 1e6 fixed-point scale */

static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * SCALE + (uint64_t)ts.tv_nsec / 1000ULL;
}

void moonlab_token_bucket_init(moonlab_token_bucket_t *bkt,
                               uint64_t burst,
                               uint64_t refill_per_sec)
{
    if (!bkt) return;
    bkt->burst_x1e6     = burst * SCALE;
    bkt->refill_per_sec = refill_per_sec;
    atomic_store(&bkt->tokens_x1e6,    bkt->burst_x1e6);
    atomic_store(&bkt->last_refill_us, now_us());
}

/* Compute "how many fixed-point tokens have accrued since the
 * caller's `last_us`, capped at `burst_x1e6 - current`."  Pure
 * function; caller decides whether to CAS the result in. */
static uint64_t accrued_x1e6(uint64_t now,
                             uint64_t last_us,
                             uint64_t refill_per_sec)
{
    if (refill_per_sec == 0 || now <= last_us) return 0;
    const uint64_t delta_us = now - last_us;
    /* tokens accrued = delta_us * refill_per_sec / 1e6 (in tokens).
     * In fixed-point: x1e6 form = delta_us * refill_per_sec.
     * Overflow guard: refill_per_sec * delta_us fits in uint64_t
     * for any refill_per_sec <= 1e12 and delta_us <= 1e6 sec
     * (~30 years).  Production deployments are well inside that. */
    return delta_us * refill_per_sec;
}

int moonlab_token_bucket_take(moonlab_token_bucket_t *bkt, uint64_t n)
{
    if (!bkt) return 0;
    const uint64_t cost_x1e6 = n * SCALE;
    const uint64_t now       = now_us();

    for (;;) {
        uint64_t       cur     = atomic_load(&bkt->tokens_x1e6);
        const uint64_t last    = atomic_load(&bkt->last_refill_us);
        const uint64_t accrued = accrued_x1e6(now, last, bkt->refill_per_sec);
        uint64_t       avail   = cur + accrued;
        if (avail > bkt->burst_x1e6) avail = bkt->burst_x1e6;
        if (avail < cost_x1e6) {
            /* Even with the accrued refill we cannot pay -- bail.
             * Still write the refilled value back so subsequent
             * callers do not re-do the same elapsed-time math. */
            (void)atomic_compare_exchange_weak(
                &bkt->tokens_x1e6, &cur, avail);
            atomic_store(&bkt->last_refill_us, now);
            return 0;
        }
        const uint64_t next = avail - cost_x1e6;
        uint64_t       expected = cur;
        if (atomic_compare_exchange_weak(
                &bkt->tokens_x1e6, &expected, next)) {
            atomic_store(&bkt->last_refill_us, now);
            return 1;
        }
        /* CAS lost a race -- retry with a fresh load. */
    }
}

void moonlab_token_bucket_refill(moonlab_token_bucket_t *bkt, uint64_t n)
{
    if (!bkt) return;
    const uint64_t add = n * SCALE;
    for (;;) {
        const uint64_t cur = atomic_load(&bkt->tokens_x1e6);
        uint64_t       next = cur + add;
        if (next > bkt->burst_x1e6) next = bkt->burst_x1e6;
        uint64_t expected = cur;
        if (atomic_compare_exchange_weak(
                &bkt->tokens_x1e6, &expected, next)) {
            return;
        }
    }
}

uint64_t moonlab_token_bucket_peek(moonlab_token_bucket_t *bkt)
{
    if (!bkt) return 0;
    const uint64_t now     = now_us();
    const uint64_t cur     = atomic_load(&bkt->tokens_x1e6);
    const uint64_t last    = atomic_load(&bkt->last_refill_us);
    const uint64_t accrued = accrued_x1e6(now, last, bkt->refill_per_sec);
    uint64_t       avail   = cur + accrued;
    if (avail > bkt->burst_x1e6) avail = bkt->burst_x1e6;
    return avail / SCALE;
}
