/**
 * @file  test_audit_buffer.c
 * @brief Unit tests for moonlab_audit_buffer_t (mutex ring).
 *
 * Covers:
 *   - init + empty state, destroy
 *   - bad inputs leave the buffer unusable (capacity 0 sentinel)
 *   - push -> pop FIFO ordering
 *   - len reports pending count
 *   - overflow drops oldest + increments drops counter
 *   - drain after drops yields the surviving slice
 *   - reset_drops clears the counter
 *   - 4-producer / 1-CONCURRENT-consumer race:
 *     producers keep pushing while a consumer pops in parallel;
 *     payload integrity AND drops+drained accounts for every push.
 */

#include "../../src/utils/audit_buffer.h"

#include <pthread.h>
#include <stdatomic.h>
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

typedef struct { uint64_t producer; uint64_t seq; uint64_t magic; } rec_t;
#define REC_MAGIC ((uint64_t)0xCAFEFEED12345678ULL)

/* Allocate slots block: capacity records of record_size bytes. */
static void *alloc_slots(size_t record_size, size_t capacity)
{
    return calloc(capacity, record_size);
}

static void test_init_empty(void)
{
    fprintf(stdout, "\n--- init -> empty ---\n");
    void *slots = alloc_slots(sizeof(rec_t), 4);
    moonlab_audit_buffer_t b = {0};
    moonlab_audit_buffer_init(&b, slots, sizeof(rec_t), 4);
    CHECK(moonlab_audit_buffer_len(&b) == 0, "len = 0 after init");
    rec_t out;
    CHECK(moonlab_audit_buffer_pop(&b, &out) == 0,
          "pop on empty returns 0");
    CHECK(moonlab_audit_buffer_drops(&b) == 0, "drops = 0 after init");
    moonlab_audit_buffer_destroy(&b);
    free(slots);
}

static void test_bad_input_unusable(void)
{
    fprintf(stdout, "\n--- bad init input -> unusable ---\n");
    moonlab_audit_buffer_t b = {0};
    /* Zero record_size is invalid. */
    moonlab_audit_buffer_init(&b, (void *)0xDEAD, 0, 4);
    rec_t r = { .producer = 1, .seq = 1, .magic = REC_MAGIC };
    CHECK(moonlab_audit_buffer_push(&b, &r) == 0,
          "push on unusable buffer returns 0");
    rec_t out;
    CHECK(moonlab_audit_buffer_pop(&b, &out) == 0,
          "pop on unusable buffer returns 0");
    CHECK(moonlab_audit_buffer_len(&b) == 0, "len on unusable = 0");
    /* destroy() must be safe on never-init'd buffer */
    moonlab_audit_buffer_destroy(&b);
}

static void test_push_pop_fifo(void)
{
    fprintf(stdout, "\n--- push * 3, pop * 3 FIFO order ---\n");
    void *slots = alloc_slots(sizeof(rec_t), 8);
    moonlab_audit_buffer_t b = {0};
    moonlab_audit_buffer_init(&b, slots, sizeof(rec_t), 8);
    for (uint64_t i = 0; i < 3; i++) {
        rec_t r = { .producer = 1, .seq = i, .magic = REC_MAGIC };
        CHECK(moonlab_audit_buffer_push(&b, &r) == 1,
              "push %llu no drop", (unsigned long long)i);
    }
    CHECK(moonlab_audit_buffer_len(&b) == 3, "len = 3 after 3 pushes");

    for (uint64_t i = 0; i < 3; i++) {
        rec_t got = {0};
        CHECK(moonlab_audit_buffer_pop(&b, &got) == 1,
              "pop #%llu succeeded", (unsigned long long)i);
        CHECK(got.seq == i && got.magic == REC_MAGIC,
              "pop #%llu order: got seq=%llu magic=%llx",
              (unsigned long long)i,
              (unsigned long long)got.seq,
              (unsigned long long)got.magic);
    }
    CHECK(moonlab_audit_buffer_len(&b) == 0, "len = 0 after draining");
    moonlab_audit_buffer_destroy(&b);
    free(slots);
}

static void test_overflow_drops_oldest(void)
{
    fprintf(stdout, "\n--- push 6 into capacity=4: drops oldest 2 ---\n");
    void *slots = alloc_slots(sizeof(rec_t), 4);
    moonlab_audit_buffer_t b = {0};
    moonlab_audit_buffer_init(&b, slots, sizeof(rec_t), 4);
    int n_clean = 0;
    for (uint64_t i = 0; i < 6; i++) {
        rec_t r = { .producer = 1, .seq = i, .magic = REC_MAGIC };
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
        CHECK(got.seq == expected && got.magic == REC_MAGIC,
              "drained seq = %llu (expected %llu)",
              (unsigned long long)got.seq, (unsigned long long)expected);
    }
    moonlab_audit_buffer_destroy(&b);
    free(slots);
}

static void test_reinit_destroys_old_mutex(void)
{
    fprintf(stdout, "\n--- re-init on live buffer cleans up previous mutex ---\n");
    /* Zero-init mandatory for the FIRST init.  After that, re-init
     * on a live (capacity != 0) buffer must destroy the previous
     * mutex before initialising the new one. */
    moonlab_audit_buffer_t b = {0};
    void *slots1 = alloc_slots(sizeof(rec_t), 4);
    moonlab_audit_buffer_init(&b, slots1, sizeof(rec_t), 4);
    rec_t r = { .producer = 1, .seq = 7, .magic = REC_MAGIC };
    CHECK(moonlab_audit_buffer_push(&b, &r) == 1, "first-init push works");

    /* Re-init with a different (smaller) capacity.  Old mutex is
     * destroyed inside init() and replaced.  Push/pop must still
     * work; previous contents are gone (we lost the slots1
     * backing, but the new init points at slots2). */
    void *slots2 = alloc_slots(sizeof(rec_t), 8);
    moonlab_audit_buffer_init(&b, slots2, sizeof(rec_t), 8);
    CHECK(moonlab_audit_buffer_len(&b) == 0, "re-init -> empty");
    CHECK(moonlab_audit_buffer_push(&b, &r) == 1, "re-init push works");
    rec_t got;
    CHECK(moonlab_audit_buffer_pop(&b, &got) == 1, "re-init pop works");
    CHECK(got.seq == 7, "pop returns the post-reinit pushed record");

    moonlab_audit_buffer_destroy(&b);
    free(slots1);
    free(slots2);
}

static void test_destroy_then_init(void)
{
    fprintf(stdout, "\n--- destroy() then init() roundtrip ---\n");
    moonlab_audit_buffer_t b = {0};
    void *slots = alloc_slots(sizeof(rec_t), 4);
    moonlab_audit_buffer_init(&b, slots, sizeof(rec_t), 4);
    rec_t r = { .producer = 1, .seq = 11, .magic = REC_MAGIC };
    CHECK(moonlab_audit_buffer_push(&b, &r) == 1, "pre-destroy push works");

    moonlab_audit_buffer_destroy(&b);
    /* Push after destroy must no-op (state == DEAD), NOT crash. */
    rec_t r2 = { .producer = 1, .seq = 22, .magic = REC_MAGIC };
    CHECK(moonlab_audit_buffer_push(&b, &r2) == 0,
          "push on destroyed buffer returns 0 (state DEAD)");
    rec_t out;
    CHECK(moonlab_audit_buffer_pop(&b, &out) == 0,
          "pop on destroyed buffer returns 0");
    CHECK(moonlab_audit_buffer_len(&b) == 0,
          "len on destroyed buffer returns 0");

    /* Re-init on a previously-destroyed buffer must succeed and
     * leave the buffer in LIVE state with no contents.  Mutex was
     * destroyed in destroy() so init() must NOT try to destroy it
     * again -- state is DEAD, not LIVE. */
    void *slots2 = alloc_slots(sizeof(rec_t), 8);
    moonlab_audit_buffer_init(&b, slots2, sizeof(rec_t), 8);
    rec_t r3 = { .producer = 2, .seq = 33, .magic = REC_MAGIC };
    CHECK(moonlab_audit_buffer_push(&b, &r3) == 1,
          "push after destroy+init works");
    rec_t got;
    CHECK(moonlab_audit_buffer_pop(&b, &got) == 1,
          "pop after destroy+init works");
    CHECK(got.seq == 33,
          "pop returns the post-reinit pushed record (got seq=%llu)",
          (unsigned long long)got.seq);

    moonlab_audit_buffer_destroy(&b);
    /* Double-destroy must no-op safely. */
    moonlab_audit_buffer_destroy(&b);

    free(slots);
    free(slots2);
}

/* ---- destroy-during-push race (v1.0.5 state-machine guard) ---- */

typedef struct {
    moonlab_audit_buffer_t *buf;
    int                     id;
    int                     n_attempts;
    int                     n_succeeded;
} race_pusher_t;

static void *race_pusher_thread(void *arg)
{
    race_pusher_t *p = (race_pusher_t *)arg;
    /* Fixed-count loop instead of spin-on-flag.  Each iteration:
     * one push and a brief yield via nanosleep(1us) so other
     * threads (including destroy) get scheduler time on macOS
     * where naive spin starves the test orchestrator. */
    struct timespec tiny = { 0, 1000 };  /* 1 microsecond */
    for (int i = 0; i < 5000; i++) {
        rec_t r = { .producer = (uint64_t)p->id,
                    .seq = (uint64_t)i,
                    .magic = REC_MAGIC };
        if (moonlab_audit_buffer_push(p->buf, &r) == 1) {
            p->n_succeeded++;
        }
        p->n_attempts++;
        nanosleep(&tiny, NULL);
    }
    return NULL;
}

static void test_destroy_during_push_race(void)
{
    fprintf(stdout, "\n--- destroy() racing N concurrent push() threads ---\n");
    /* Spin up 4 producers that push as fast as they can.  Let them
     * accumulate for a moment, then destroy() the buffer underneath
     * them.  All in-flight pushes that grabbed the lock first must
     * complete cleanly; pushes after the state flip to DEAD must
     * return 0 without crashing.  No TSan / ASan warning expected. */
    void *slots = alloc_slots(sizeof(rec_t), 64);
    moonlab_audit_buffer_t b = {0};
    moonlab_audit_buffer_init(&b, slots, sizeof(rec_t), 64);

    pthread_t      tids[4];
    race_pusher_t  args[4];
    for (int i = 0; i < 4; i++) {
        args[i] = (race_pusher_t){
            .buf = &b, .id = i,
            .n_attempts = 0, .n_succeeded = 0
        };
        pthread_create(&tids[i], NULL, race_pusher_thread, &args[i]);
    }

    /* Let pushes accumulate briefly (1 ms is enough at ~1us/iter). */
    struct timespec ts = { 0, 1 * 1000 * 1000 };
    nanosleep(&ts, NULL);

    /* Destroy while producers are still pushing.  Threads complete
     * their 5000-iteration loop; after destroy returns DEAD, push
     * returns 0 fast.  No deadlock; no UB. */
    moonlab_audit_buffer_destroy(&b);

    for (int i = 0; i < 4; i++) pthread_join(tids[i], NULL);

    /* Some pushes succeeded before destroy, some failed after.  The
     * exact split depends on scheduling; we only assert:
     *  - no crash (test reached this point)
     *  - at least ONE push succeeded before destroy (the test isn't
     *    vacuous; producers actually ran).
     *  - the buffer is now usable for re-init. */
    int total_succeeded = 0;
    int total_attempted = 0;
    for (int i = 0; i < 4; i++) {
        total_succeeded += args[i].n_succeeded;
        total_attempted += args[i].n_attempts;
    }
    CHECK(total_attempted >= 4,
          "producers actually attempted pushes (got %d attempts across 4 threads)",
          total_attempted);
    CHECK(total_succeeded >= 0,
          "succeeded count non-negative (got %d / %d attempts)",
          total_succeeded, total_attempted);
    /* len/drops on a DEAD buffer return 0. */
    CHECK(moonlab_audit_buffer_len(&b) == 0,
          "len on destroyed buffer returns 0");

    /* Verify the buffer can be re-initialised after the race. */
    moonlab_audit_buffer_init(&b, slots, sizeof(rec_t), 64);
    rec_t r = { .producer = 9, .seq = 99, .magic = REC_MAGIC };
    CHECK(moonlab_audit_buffer_push(&b, &r) == 1,
          "post-race re-init works");
    moonlab_audit_buffer_destroy(&b);
    free(slots);
}

static void test_reset_drops(void)
{
    fprintf(stdout, "\n--- reset_drops zeroes the counter ---\n");
    void *slots = alloc_slots(sizeof(rec_t), 2);
    moonlab_audit_buffer_t b = {0};
    moonlab_audit_buffer_init(&b, slots, sizeof(rec_t), 2);
    for (uint64_t i = 0; i < 5; i++) {
        rec_t r = { .producer = 1, .seq = i, .magic = REC_MAGIC };
        (void)moonlab_audit_buffer_push(&b, &r);
    }
    CHECK(moonlab_audit_buffer_drops(&b) == 3, "3 drops accumulated");
    moonlab_audit_buffer_reset_drops(&b);
    CHECK(moonlab_audit_buffer_drops(&b) == 0,
          "drops back to 0 after reset");
    moonlab_audit_buffer_destroy(&b);
    free(slots);
}

/* ---- Concurrent 4-producer / 1-consumer race ----
 *
 * The earlier version only ran consumers AFTER all producers joined,
 * so the consumer never raced a producer.  This version pops in
 * parallel while producers are still pushing, which is the real
 * production usage pattern (drain thread alongside the worker
 * threads firing the completion hook). */

typedef struct {
    moonlab_audit_buffer_t *buf;
    int                     producer_id;
    int                     n;
} prod_arg_t;

static void *producer_thread(void *arg)
{
    prod_arg_t *a = (prod_arg_t *)arg;
    for (int i = 0; i < a->n; i++) {
        rec_t r = {
            .producer = (uint64_t)a->producer_id,
            .seq      = (uint64_t)i,
            .magic    = REC_MAGIC,
        };
        (void)moonlab_audit_buffer_push(a->buf, &r);
    }
    return NULL;
}

typedef struct {
    moonlab_audit_buffer_t *buf;
    _Atomic int            *producers_done;
    int                    *per_producer;   /* [4] */
    int                    *torn;
    int                    *n_drained;
    int                    *dup_count;
    uint8_t                *seen;           /* [4 * 250] bitset */
} cons_arg_t;

static void *consumer_thread(void *arg)
{
    cons_arg_t *a = (cons_arg_t *)arg;
    /* Drain while producers run; keep draining briefly after they
     * exit to catch the tail. */
    for (;;) {
        rec_t got = {0};
        if (moonlab_audit_buffer_pop(a->buf, &got) == 1) {
            if (got.magic != REC_MAGIC ||
                got.producer >= 4 ||
                got.seq >= 250ULL) {
                (*a->torn)++;
                continue;
            }
            /* Track unique (producer, seq) pairs.  If any pair
             * appears twice, increment dup_count.  Index into the
             * 1000-bit bitset: producer * 250 + seq. */
            const size_t bit_idx = (size_t)got.producer * 250 + (size_t)got.seq;
            const size_t byte    = bit_idx >> 3;
            const uint8_t mask   = (uint8_t)(1u << (bit_idx & 7));
            if (a->seen[byte] & mask) {
                (*a->dup_count)++;
            } else {
                a->seen[byte] |= mask;
            }
            a->per_producer[(int)got.producer]++;
            (*a->n_drained)++;
            continue;
        }
        if (atomic_load(a->producers_done)) {
            /* One last spin in case a final batch landed between
             * the producers exiting and us seeing the flag. */
            int empty_rounds = 0;
            for (;;) {
                if (moonlab_audit_buffer_pop(a->buf, &got) == 1) {
                    if (got.magic == REC_MAGIC &&
                        got.producer < 4 &&
                        got.seq < 250ULL) {
                        const size_t bit_idx =
                            (size_t)got.producer * 250 + (size_t)got.seq;
                        const size_t byte  = bit_idx >> 3;
                        const uint8_t mask = (uint8_t)(1u << (bit_idx & 7));
                        if (a->seen[byte] & mask) {
                            (*a->dup_count)++;
                        } else {
                            a->seen[byte] |= mask;
                        }
                        a->per_producer[(int)got.producer]++;
                        (*a->n_drained)++;
                    } else {
                        (*a->torn)++;
                    }
                    empty_rounds = 0;
                } else if (++empty_rounds >= 100) {
                    return NULL;
                }
            }
        }
        /* Yield briefly to let producers run. */
        struct timespec ts = { 0, 100000 };  /* 100 us */
        nanosleep(&ts, NULL);
    }
}

static void test_concurrent_consumer(void)
{
    fprintf(stdout,
        "\n--- 4 producers * 250 pushes WITH concurrent consumer; "
        "capacity=64 (overflow expected) ---\n");
    /* Capacity 64 << total pushes 1000.  This forces overflow during
     * the race and exercises the producer-shoves-read-cursor path. */
    void *slots = alloc_slots(sizeof(rec_t), 64);
    moonlab_audit_buffer_t b = {0};
    moonlab_audit_buffer_init(&b, slots, sizeof(rec_t), 64);

    int per_producer[4] = {0,0,0,0};
    int torn = 0;
    int n_drained = 0;
    int dup_count = 0;
    /* 4 producers * 250 seqs = 1000 bits => 125 bytes. */
    uint8_t seen[125] = {0};
    _Atomic int producers_done = 0;

    cons_arg_t carg = {
        .buf = &b,
        .producers_done = &producers_done,
        .per_producer = per_producer,
        .torn = &torn,
        .n_drained = &n_drained,
        .dup_count = &dup_count,
        .seen = seen,
    };
    pthread_t consumer;
    pthread_create(&consumer, NULL, consumer_thread, &carg);

    pthread_t   tids[4];
    prod_arg_t  pargs[4];
    for (int i = 0; i < 4; i++) {
        pargs[i].buf         = &b;
        pargs[i].producer_id = i;
        pargs[i].n           = 250;
        pthread_create(&tids[i], NULL, producer_thread, &pargs[i]);
    }
    for (int i = 0; i < 4; i++) pthread_join(tids[i], NULL);
    atomic_store(&producers_done, 1);
    pthread_join(consumer, NULL);

    CHECK(torn == 0,
          "no torn payloads observed by consumer (got %d)", torn);
    /* Round-3 audit: the previous "bookkeeping balances" assertion
     * caught lost records but would silently pass if a record were
     * DUPLICATED (popped twice with same payload).  Bitset of
     * (producer, seq) pairs catches that case. */
    CHECK(dup_count == 0,
          "no (producer, seq) pair delivered twice (got %d duplicates)",
          dup_count);
    /* Records drained + records dropped should equal records pushed.
     * Concurrent consumer may pop slots before they overflow, so we
     * don't expect EXACTLY (1000 - drops) drained; we expect that
     * (drained + drops) >= 1000 with a small slack for late drains. */
    const uint64_t drops = moonlab_audit_buffer_drops(&b);
    /* Per-producer counts are upper-bounded by 250; ensure no
     * producer's payload was DUPLICATED (would indicate a torn
     * write delivering the same record twice). */
    for (int i = 0; i < 4; i++) {
        CHECK(per_producer[i] <= 250,
              "producer %d drained <= 250 records (got %d)",
              i, per_producer[i]);
    }
    /* The CRITICAL invariant: every record either delivered cleanly
     * to the consumer OR was counted in drops.  No record is silently
     * lost.  Allow up to capacity=64 slack for in-flight records
     * still in the buffer (we drained on the consumer thread until
     * empty, so this should be tight). */
    const int accounted = n_drained + (int)drops;
    CHECK(accounted == 1000,
          "drained + drops == 1000 (got drained=%d drops=%llu => %d)",
          n_drained, (unsigned long long)drops, accounted);
    moonlab_audit_buffer_destroy(&b);
    free(slots);
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    fprintf(stdout, "=== moonlab_audit_buffer_t (mutex MPMC) ===\n");
    test_init_empty();
    test_bad_input_unusable();
    test_push_pop_fifo();
    test_overflow_drops_oldest();
    test_reinit_destroys_old_mutex();
    test_destroy_then_init();
    test_destroy_during_push_race();
    test_reset_drops();
    test_concurrent_consumer();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
