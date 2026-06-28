/**
 * @file  test_token_bucket.c
 * @brief Unit tests for moonlab_token_bucket_t.
 *
 * Covers:
 *   - init from zero state -> bucket starts full at burst
 *   - take(n) succeeds while balance >= n, fails after exhaustion
 *   - refill restores balance, clipped at burst
 *   - lazy time-based refill: take after a sleep accrues tokens
 *   - thread safety: 8 threads racing take() collectively remove
 *     no more than the available balance
 */

#include "../../src/utils/token_bucket.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

static void msleep(int ms)
{
    struct timespec ts = { .tv_sec = ms / 1000,
                           .tv_nsec = (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, NULL);
}

static void test_init_starts_full(void)
{
    fprintf(stdout, "\n--- init starts full ---\n");
    moonlab_token_bucket_t bkt;
    moonlab_token_bucket_init(&bkt, 100, 10);
    CHECK(moonlab_token_bucket_peek(&bkt) == 100,
          "peek = %llu (expected 100)",
          (unsigned long long)moonlab_token_bucket_peek(&bkt));
}

static void test_take_drains(void)
{
    fprintf(stdout, "\n--- take drains balance, fails after empty ---\n");
    moonlab_token_bucket_t bkt;
    moonlab_token_bucket_init(&bkt, 10, 0);  /* no refill */
    int ok = 0;
    for (int i = 0; i < 10; i++) {
        if (moonlab_token_bucket_take(&bkt, 1)) ok++;
    }
    CHECK(ok == 10, "10 successful takes (got %d)", ok);
    CHECK(moonlab_token_bucket_take(&bkt, 1) == 0,
          "11th take fails (balance empty)");
    CHECK(moonlab_token_bucket_peek(&bkt) == 0,
          "peek = 0 after drain");
}

static void test_refill_clips_at_burst(void)
{
    fprintf(stdout, "\n--- explicit refill clips at burst ---\n");
    moonlab_token_bucket_t bkt;
    moonlab_token_bucket_init(&bkt, 5, 0);
    (void)moonlab_token_bucket_take(&bkt, 5);
    CHECK(moonlab_token_bucket_peek(&bkt) == 0, "drained to 0");
    /* Refill way past the burst -- should clip. */
    moonlab_token_bucket_refill(&bkt, 1000);
    CHECK(moonlab_token_bucket_peek(&bkt) == 5,
          "after refill 1000 over burst=5, peek = %llu (expected 5)",
          (unsigned long long)moonlab_token_bucket_peek(&bkt));
}

static void test_time_refill(void)
{
    fprintf(stdout, "\n--- time-based refill accrues ---\n");
    moonlab_token_bucket_t bkt;
    moonlab_token_bucket_init(&bkt, 100, 100);  /* 100 tok/sec */
    (void)moonlab_token_bucket_take(&bkt, 100);
    CHECK(moonlab_token_bucket_peek(&bkt) == 0, "drained");
    msleep(250);  /* 250 ms -> ~25 tokens accrued */
    const uint64_t peek = moonlab_token_bucket_peek(&bkt);
    /* CI runners can overrun the 250ms sleep under load -- 250ms wall
     * can become 400-500ms, giving 40-50 tokens.  Check invariants:
     * floor (>= nominal accrual minus sleep underrun grace) and the
     * burst cap.  This is the same hardening applied to the Rust-side
     * time_refill_accrues in bindings/rust/moonlab/src/token_bucket.rs. */
    CHECK(peek >= 20,
          "after >=250ms at 100 tok/sec, peek = %llu (expected >= 20)",
          (unsigned long long)peek);
    CHECK(peek <= 100,
          "burst cap violated, peek = %llu (expected <= 100)",
          (unsigned long long)peek);
    CHECK(moonlab_token_bucket_take(&bkt, 10) == 1,
          "can take 10 from the accrued balance");
}

/* ---- threaded race ---- */

typedef struct { moonlab_token_bucket_t *bkt; int n_attempts; int n_ok; } race_arg_t;

static void *race_worker(void *arg)
{
    race_arg_t *a = (race_arg_t *)arg;
    for (int i = 0; i < a->n_attempts; i++) {
        if (moonlab_token_bucket_take(a->bkt, 1)) a->n_ok++;
    }
    return NULL;
}

static void test_thread_safety(void)
{
    fprintf(stdout, "\n--- 8 threads * 1000 attempts on burst=500 ---\n");
    moonlab_token_bucket_t bkt;
    moonlab_token_bucket_init(&bkt, 500, 0);  /* no refill */
    pthread_t  tids[8];
    race_arg_t args[8];
    for (int i = 0; i < 8; i++) {
        args[i].bkt = &bkt;
        args[i].n_attempts = 1000;
        args[i].n_ok = 0;
        pthread_create(&tids[i], NULL, race_worker, &args[i]);
    }
    int total_ok = 0;
    for (int i = 0; i < 8; i++) {
        pthread_join(tids[i], NULL);
        total_ok += args[i].n_ok;
    }
    /* No more than 500 takes should have succeeded total -- the
     * bucket starts full and never refills. */
    CHECK(total_ok == 500,
          "exactly 500 successful takes across 8000 attempts (got %d)",
          total_ok);
    CHECK(moonlab_token_bucket_peek(&bkt) == 0,
          "bucket exhausted (peek = %llu)",
          (unsigned long long)moonlab_token_bucket_peek(&bkt));
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    fprintf(stdout, "=== moonlab_token_bucket_t ===\n");
    test_init_starts_full();
    test_take_drains();
    test_refill_clips_at_burst();
    test_time_refill();
    test_thread_safety();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
