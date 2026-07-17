/**
 * @file  conc_audit_buffer.c
 * @brief ThreadSanitizer harness for the bounded audit ring buffer.
 *
 * The buffer (src/utils/audit_buffer.{h,c}) advertises a multi-producer /
 * multi-consumer model with a single mutex plus an atomic `state` word that
 * is meant to make destroy() safe against in-flight push/pop.  This harness
 * hammers all three paths:
 *
 *   mode "mpmc"     (default) -- N pushers + N poppers + stats readers on a
 *                   LIVE buffer.  Verifies the cursor/count/drops bookkeeping
 *                   under the single lock.  Expected clean.
 *
 *   mode "destroy"  -- same producers/consumers, but a destroyer thread calls
 *                   moonlab_audit_buffer_destroy() mid-flight while pushes and
 *                   pops are still in progress.  The documented design claims
 *                   the atomic-state + lock handshake makes this safe
 *                   ("No race window where a push tries to lock a destroyed
 *                   mutex", audit_buffer.c).  It is NOT safe: a push/pop that
 *                   samples state==LIVE at audit_buffer.c:86, then reaches
 *                   pthread_mutex_lock() at :87 AFTER destroy() destroyed the
 *                   mutex at :76, blocks forever (locking a destroyed pthread
 *                   mutex is UB; on macOS it wedges).  A watchdog thread turns
 *                   that hang into a deterministic FAIL (exit 7) so the probe
 *                   does not stall CI.
 */

#include "../../src/utils/audit_buffer.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef N_PUSH
#define N_PUSH 24   /* oversubscribe so a pusher is frequently preempted
                     * between the state pre-check and pthread_mutex_lock,
                     * i.e. inside the destroy-vs-lock window */
#endif
#ifndef N_POP
#define N_POP 6
#endif
#ifndef DESTROY_ROUNDS
#define DESTROY_ROUNDS 16
#endif
#define REC_SIZE 32
#define CAP      64

typedef struct {
    moonlab_audit_buffer_t *buf;
    _Atomic int            *stop;
    int                     id;
} args_t;

static void *pusher(void *a)
{
    args_t *ar = (args_t *)a;
    uint8_t rec[REC_SIZE];
    memset(rec, ar->id & 0xff, sizeof(rec));
    uint64_t n = 0;
    while (!atomic_load(ar->stop)) {
        rec[0] = (uint8_t)(n++ & 0xff);
        (void)moonlab_audit_buffer_push(ar->buf, rec);
    }
    return NULL;
}

static void *popper(void *a)
{
    args_t *ar = (args_t *)a;
    uint8_t out[REC_SIZE];
    while (!atomic_load(ar->stop)) {
        (void)moonlab_audit_buffer_pop(ar->buf, out);
        (void)moonlab_audit_buffer_len(ar->buf);
        (void)moonlab_audit_buffer_drops(ar->buf);
    }
    return NULL;
}

/* Watchdog: if the destroy-mode round wedges on a destroyed mutex, the
 * joins below never return.  This thread fires after a grace period, reports
 * the deadlock, and force-exits so the harness cannot hang forever. */
static void *deadlock_watchdog(void *arg)
{
    int secs = (int)(intptr_t)arg;
    struct timespec ts = { secs, 0 };
    nanosleep(&ts, NULL);
    fprintf(stdout,
        "DEADLOCK: destroy() vs in-flight push/pop wedged a thread in "
        "pthread_mutex_lock on the destroyed mutex "
        "(audit_buffer.c:87/:76). Documented \"no race window\" claim is false.\n");
    fflush(stdout);
    _exit(7);
    return NULL;
}

static int run_round(int destroy_midflight)
{
    static uint8_t slots[REC_SIZE * CAP];
    moonlab_audit_buffer_t buf;
    memset(&buf, 0, sizeof(buf));
    moonlab_audit_buffer_init(&buf, slots, REC_SIZE, CAP);

    _Atomic int stop = 0;
    args_t pa[N_PUSH], ca[N_POP];
    pthread_t pt[N_PUSH], ct[N_POP];
    for (int i = 0; i < N_PUSH; i++) {
        pa[i] = (args_t){ &buf, &stop, i };
        pthread_create(&pt[i], NULL, pusher, &pa[i]);
    }
    for (int i = 0; i < N_POP; i++) {
        ca[i] = (args_t){ &buf, &stop, i };
        pthread_create(&ct[i], NULL, popper, &ca[i]);
    }

    if (destroy_midflight) {
        /* Let producers/consumers get going, then pull the rug out from
         * under them -- the design claims the state machine tolerates this. */
        struct timespec ts = { 0, 300 * 1000 };
        nanosleep(&ts, NULL);
        moonlab_audit_buffer_destroy(&buf);
        /* Give the still-running push/pop threads a moment to hit the
         * DEAD-state path (and any lock-on-destroyed-mutex window). */
        nanosleep(&ts, NULL);
    } else {
        struct timespec ts = { 0, 2 * 1000 * 1000 };
        nanosleep(&ts, NULL);
    }

    atomic_store(&stop, 1);
    for (int i = 0; i < N_PUSH; i++) pthread_join(pt[i], NULL);
    for (int i = 0; i < N_POP; i++) pthread_join(ct[i], NULL);

    if (!destroy_midflight) moonlab_audit_buffer_destroy(&buf);
    return 0;
}

int main(int argc, char **argv)
{
    const int destroy_mode = (argc > 1 && strcmp(argv[1], "destroy") == 0);
    fprintf(stdout, "=== conc_audit_buffer (%s) ===\n",
            destroy_mode ? "destroy" : "mpmc");
    if (destroy_mode) {
        pthread_t wd;
        pthread_create(&wd, NULL, deadlock_watchdog, (void *)(intptr_t)15);
        /* Retry the destroy-vs-push race across rounds: the wedge is a
         * scheduling window, so oversubscription + repetition makes hitting
         * it near-certain.  A wedged round never returns from join(), so the
         * watchdog fires; a clean round advances to the next. */
        for (int r = 0; r < DESTROY_ROUNDS; r++) run_round(1);
        fprintf(stdout, "destroy: %d rounds completed without wedging "
                        "(platform did not deadlock this run)\n", DESTROY_ROUNDS);
        return 0;
    }
    run_round(0);
    fprintf(stdout, "done (1 round)\n");
    return 0;
}
