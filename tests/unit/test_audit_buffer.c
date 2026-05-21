/**
 * @file  test_audit_buffer.c
 * @brief Unit tests for moonlab_audit_buffer_t.
 *
 * Covers:
 *   - init + empty state
 *   - push -> pop FIFO ordering
 *   - len reports pending count, capped at capacity
 *   - overflow drops oldest + increments drops counter
 *   - drain after drops yields the surviving slice
 *   - reset_drops clears the counter
 *   - 4-producer / 1-consumer race: no payload corruption
 */

#include "../../src/utils/audit_buffer.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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

typedef struct { uint64_t producer; uint64_t seq; } rec_t;

static void test_init_empty(void)
{
    fprintf(stdout, "\n--- init -> empty ---\n");
    rec_t slots[4];
    moonlab_audit_buffer_t b;
    moonlab_audit_buffer_init(&b, slots, sizeof(rec_t), 4);
    CHECK(moonlab_audit_buffer_len(&b) == 0, "len = 0 after init");
    rec_t out;
    CHECK(moonlab_audit_buffer_pop(&b, &out) == 0,
          "pop on empty returns 0");
    CHECK(moonlab_audit_buffer_drops(&b) == 0, "drops = 0 after init");
}

static void test_push_pop_fifo(void)
{
    fprintf(stdout, "\n--- push * 3, pop * 3 FIFO order ---\n");
    rec_t slots[8];
    moonlab_audit_buffer_t b;
    moonlab_audit_buffer_init(&b, slots, sizeof(rec_t), 8);
    for (uint64_t i = 0; i < 3; i++) {
        rec_t r = { .producer = 1, .seq = i };
        CHECK(moonlab_audit_buffer_push(&b, &r) == 1,
              "push %llu no drop", (unsigned long long)i);
    }
    CHECK(moonlab_audit_buffer_len(&b) == 3, "len = 3 after 3 pushes");

    for (uint64_t i = 0; i < 3; i++) {
        rec_t got = {0};
        CHECK(moonlab_audit_buffer_pop(&b, &got) == 1,
              "pop #%llu succeeded", (unsigned long long)i);
        CHECK(got.seq == i, "pop #%llu order: got %llu",
              (unsigned long long)i, (unsigned long long)got.seq);
    }
    CHECK(moonlab_audit_buffer_len(&b) == 0, "len = 0 after draining");
}

static void test_overflow_drops_oldest(void)
{
    fprintf(stdout, "\n--- push 6 into capacity=4: drops oldest 2 ---\n");
    rec_t slots[4];
    moonlab_audit_buffer_t b;
    moonlab_audit_buffer_init(&b, slots, sizeof(rec_t), 4);
    int n_clean = 0;
    for (uint64_t i = 0; i < 6; i++) {
        rec_t r = { .producer = 1, .seq = i };
        if (moonlab_audit_buffer_push(&b, &r) == 1) n_clean++;
    }
    CHECK(n_clean == 4, "4 clean pushes before drop (got %d)", n_clean);
    CHECK(moonlab_audit_buffer_drops(&b) == 2,
          "drops counter = 2 (got %llu)",
          (unsigned long long)moonlab_audit_buffer_drops(&b));
    CHECK(moonlab_audit_buffer_len(&b) == 4,
          "len = 4 (capacity)");

    /* Surviving slice: 2, 3, 4, 5. */
    for (uint64_t expected = 2; expected < 6; expected++) {
        rec_t got = {0};
        CHECK(moonlab_audit_buffer_pop(&b, &got) == 1,
              "drain pop succeeded");
        CHECK(got.seq == expected,
              "drained seq = %llu (expected %llu)",
              (unsigned long long)got.seq, (unsigned long long)expected);
    }
}

static void test_reset_drops(void)
{
    fprintf(stdout, "\n--- reset_drops zeroes the counter ---\n");
    rec_t slots[2];
    moonlab_audit_buffer_t b;
    moonlab_audit_buffer_init(&b, slots, sizeof(rec_t), 2);
    for (uint64_t i = 0; i < 5; i++) {
        rec_t r = { .producer = 1, .seq = i };
        (void)moonlab_audit_buffer_push(&b, &r);
    }
    CHECK(moonlab_audit_buffer_drops(&b) == 3, "3 drops accumulated");
    moonlab_audit_buffer_reset_drops(&b);
    CHECK(moonlab_audit_buffer_drops(&b) == 0,
          "drops back to 0 after reset");
}

/* ---- 4-producer / 1-consumer race ---- */

typedef struct {
    moonlab_audit_buffer_t *buf;
    int                     producer_id;
    int                     n;
} race_arg_t;

static void *producer_thread(void *arg)
{
    race_arg_t *a = (race_arg_t *)arg;
    for (int i = 0; i < a->n; i++) {
        rec_t r = { .producer = (uint64_t)a->producer_id, .seq = (uint64_t)i };
        (void)moonlab_audit_buffer_push(a->buf, &r);
    }
    return NULL;
}

static void test_4producer_no_corruption(void)
{
    fprintf(stdout, "\n--- 4 producers * 250 pushes; payload integrity ---\n");
    rec_t slots[1024];
    moonlab_audit_buffer_t b;
    moonlab_audit_buffer_init(&b, slots, sizeof(rec_t), 1024);
    pthread_t   tids[4];
    race_arg_t  args[4];
    for (int i = 0; i < 4; i++) {
        args[i].buf         = &b;
        args[i].producer_id = i;
        args[i].n           = 250;
        pthread_create(&tids[i], NULL, producer_thread, &args[i]);
    }
    for (int i = 0; i < 4; i++) pthread_join(tids[i], NULL);

    /* All 1000 records should fit (capacity=1024 > 4*250). */
    CHECK(moonlab_audit_buffer_drops(&b) == 0,
          "no drops with capacity > total pushes (got %llu)",
          (unsigned long long)moonlab_audit_buffer_drops(&b));

    /* Drain and verify every (producer, seq) pair we saw is valid:
     * producer in [0,4), seq in [0,250).  This catches torn writes
     * where a payload splits across two CAS slots. */
    int per_producer[4] = {0,0,0,0};
    int n_drained = 0;
    rec_t got;
    while (moonlab_audit_buffer_pop(&b, &got) == 1) {
        n_drained++;
        if (got.producer >= 4) {
            fprintf(stderr, "  FAIL  garbage producer_id %llu\n",
                    (unsigned long long)got.producer);
            failures++;
            continue;
        }
        if (got.seq >= 250) {
            fprintf(stderr, "  FAIL  garbage seq %llu (producer=%llu)\n",
                    (unsigned long long)got.seq,
                    (unsigned long long)got.producer);
            failures++;
            continue;
        }
        per_producer[(int)got.producer]++;
    }
    CHECK(n_drained == 1000,
          "drained 1000 records (got %d)", n_drained);
    /* Each producer should account for exactly 250.  This proves no
     * payload was lost or duplicated by the multi-producer fanout. */
    for (int i = 0; i < 4; i++) {
        CHECK(per_producer[i] == 250,
              "producer %d contributed 250 records (got %d)",
              i, per_producer[i]);
    }
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    fprintf(stdout, "=== moonlab_audit_buffer_t ===\n");
    test_init_empty();
    test_push_pop_fifo();
    test_overflow_drops_oldest();
    test_reset_drops();
    test_4producer_no_corruption();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
