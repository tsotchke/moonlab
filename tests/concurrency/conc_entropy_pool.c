/**
 * @file  conc_entropy_pool.c
 * @brief ThreadSanitizer harness for the background entropy pool.
 *
 * Exercises the background pre-generation pthread against many concurrent
 * consumers and monitors:
 *   - consumers call entropy_pool_get_bytes with a mix of small (cache-hit)
 *     and large (cache-miss, direct-generate) request sizes,
 *   - monitors call get_stats / get_fill_level / needs_refill / refill,
 *   - (toggle mode) a lifecycle thread stops+starts the background worker
 *     while monitors read stats -- probes the background_active /
 *     background_running flags that are written OUTSIDE pool_mutex.
 *
 * Targets: the stats counters (cache_hits / cache_misses / bytes_generated /
 * background_chunks), the pool ring cursors (pool_used / pool_available), the
 * health-test transaction serialised by health_mutex, and the lifecycle flags.
 *
 * argv[1] = "toggle" enables the stop/start lifecycle thread (default off).
 */

#include "../../src/applications/entropy_pool.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef N_CONSUMERS
#define N_CONSUMERS 8
#endif
#ifndef N_MONITORS
#define N_MONITORS 3
#endif
#ifndef ITERS
#define ITERS 4000
#endif

static entropy_pool_ctx_t *g_pool = NULL;
static _Atomic int         g_stop = 0;
static _Atomic int         g_errors = 0;

static void *consumer(void *arg)
{
    const int id = (int)(intptr_t)arg;
    uint8_t small_buf[64];
    uint8_t large_buf[8192]; /* > refill threshold path / cache-miss */
    for (int i = 0; i < ITERS && !atomic_load(&g_stop); i++) {
        if ((id + i) % 8 == 0) {
            /* Large request -- likely a cache miss -> direct generate under
             * health_mutex, shared with the background worker. */
            if (entropy_pool_get_bytes(g_pool, large_buf, sizeof(large_buf)) != 0)
                atomic_fetch_add(&g_errors, 1);
        } else {
            const size_t sz = 1 + ((id + i) % sizeof(small_buf));
            if (entropy_pool_get_bytes(g_pool, small_buf, sz) != 0)
                atomic_fetch_add(&g_errors, 1);
        }
    }
    return NULL;
}

static void *monitor(void *arg)
{
    (void)arg;
    entropy_pool_stats_t st;
    for (int i = 0; i < ITERS && !atomic_load(&g_stop); i++) {
        entropy_pool_get_stats(g_pool, &st);
        (void)entropy_pool_get_fill_level(g_pool);
        (void)entropy_pool_needs_refill(g_pool);
        if ((i & 63) == 0) entropy_pool_refill(g_pool);
    }
    return NULL;
}

/* Stop + restart the background worker repeatedly, concurrent with monitors
 * that snapshot stats -- probes background_active/background_running which
 * are mutated without pool_mutex. */
static void *lifecycle(void *arg)
{
    (void)arg;
    for (int i = 0; i < 200 && !atomic_load(&g_stop); i++) {
        entropy_pool_stop_background(g_pool);
        entropy_pool_start_background(g_pool);
        struct timespec ts = { 0, 500 * 1000 };
        nanosleep(&ts, NULL);
    }
    return NULL;
}

int main(int argc, char **argv)
{
    const int toggle = (argc > 1 && strcmp(argv[1], "toggle") == 0);
    fprintf(stdout, "=== conc_entropy_pool (%s) ===\n", toggle ? "toggle" : "steady");

    if (entropy_pool_init(&g_pool) != 0) {
        fprintf(stderr, "entropy_pool_init failed\n");
        return 2;
    }

    pthread_t cons[N_CONSUMERS], mons[N_MONITORS], life;
    for (int i = 0; i < N_CONSUMERS; i++)
        pthread_create(&cons[i], NULL, consumer, (void *)(intptr_t)i);
    for (int i = 0; i < N_MONITORS; i++)
        pthread_create(&mons[i], NULL, monitor, NULL);
    int have_life = 0;
    if (toggle) have_life = (pthread_create(&life, NULL, lifecycle, NULL) == 0);

    for (int i = 0; i < N_CONSUMERS; i++) pthread_join(cons[i], NULL);
    atomic_store(&g_stop, 1);
    for (int i = 0; i < N_MONITORS; i++) pthread_join(mons[i], NULL);
    if (have_life) pthread_join(life, NULL);

    entropy_pool_free(g_pool);
    fprintf(stdout, "errors=%d\n", atomic_load(&g_errors));
    return 0;
}
