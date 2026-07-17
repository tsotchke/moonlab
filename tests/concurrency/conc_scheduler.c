/**
 * @file  conc_scheduler.c
 * @brief ThreadSanitizer harness for the distributed scheduler's shared state.
 *
 * Targets the process-global backend registry + completion hook in
 * src/distributed/scheduler.c, all guarded by g_backend_lock:
 *
 *   - N runner threads each build their OWN job and call
 *     moonlab_scheduler_run concurrently.  Every run does a
 *     find_backend() (registry read) and a completion-hook snapshot
 *     under g_backend_lock, then bumps the atomic
 *     g_count_completion_hook_fires.
 *   - a churn thread concurrently register/unregister a second backend
 *     and toggles the completion hook via set_completion_hook.
 *
 * A clean pass proves the registry mutex + atomic counter discipline holds
 * under register/find/fire contention.  The shared-core lazy-init singletons
 * (config/simd -- see conc_core_init) are warmed single-threaded first so
 * they do not mask scheduler-specific races.
 */

#include "../../src/distributed/scheduler.h"
#include "../../src/applications/moonlab_qgtl_backend.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef N_RUNNERS
#define N_RUNNERS 8
#endif
#ifndef RUNS_PER_THREAD
#define RUNS_PER_THREAD 60
#endif

static _Atomic int g_stop      = 0;
static _Atomic int g_hook_fires = 0;
static _Atomic int g_run_fail   = 0;

static void completion_hook(const moonlab_job_t *job,
                            const moonlab_job_results_t *out,
                            const char *backend_name, void *ctx)
{
    (void)job; (void)out; (void)backend_name; (void)ctx;
    atomic_fetch_add(&g_hook_fires, 1);
}

/* A trivial second backend to register/unregister under contention. */
static int noise_execute(const moonlab_job_t *job,
                         moonlab_job_results_t *out, void *ctx)
{
    (void)ctx;
    const int shots = moonlab_job_num_shots(job);
    out->num_qubits       = moonlab_job_num_qubits(job);
    out->total_shots      = shots;
    out->num_workers_used = 1;
    out->outcomes       = (uint64_t *)calloc((size_t)(shots > 0 ? shots : 1), sizeof(uint64_t));
    out->worker_seconds = (double   *)calloc(1, sizeof(double));
    return (out->outcomes && out->worker_seconds) ? 0 : -1;
}

static void *runner(void *arg)
{
    (void)arg;
    for (int i = 0; i < RUNS_PER_THREAD; i++) {
        moonlab_job_t *j = moonlab_job_create(2);
        if (!j) { atomic_fetch_add(&g_run_fail, 1); continue; }
        moonlab_job_add_gate(j, MOONLAB_QGTL_GATE_H,    0, 0, NULL);
        moonlab_job_add_gate(j, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL);
        moonlab_job_set_num_shots(j, 128);
        moonlab_job_set_num_workers(j, 2);
        moonlab_job_set_rng_seed(j, 0x1234u + (unsigned)i);

        moonlab_job_results_t out;
        memset(&out, 0, sizeof(out));
        int rc = moonlab_scheduler_run(j, &out);
        if (rc != MOONLAB_SCHED_OK) atomic_fetch_add(&g_run_fail, 1);
        moonlab_job_results_free(&out);
        moonlab_job_free(j);
    }
    return NULL;
}

static void *churn(void *arg)
{
    (void)arg;
    const char *names[3] = { NULL };
    int i = 0;
    while (!atomic_load(&g_stop)) {
        moonlab_backend_t be = {
            .name = "noise", .execute = noise_execute,
            .ctx = NULL, .description = "churn"
        };
        moonlab_register_backend(&be);
        (void)moonlab_find_backend("simulator");
        (void)moonlab_num_backends();
        (void)moonlab_list_backends(names, 3);
        moonlab_scheduler_set_completion_hook((i & 1) ? completion_hook : NULL, NULL);
        moonlab_unregister_backend("noise");
        i++;
    }
    return NULL;
}

int main(void)
{
    fprintf(stdout, "=== conc_scheduler (%d runners) ===\n", N_RUNNERS);

    /* Warm shared-core lazy singletons single-threaded (see conc_core_init)
     * by running the EXACT scheduler path the runner threads use, so every
     * config/simd/measurement lazy-init is resolved before we go parallel. */
    {
        moonlab_job_t *w = moonlab_job_create(2);
        if (w) {
            moonlab_job_add_gate(w, MOONLAB_QGTL_GATE_H,    0, 0, NULL);
            moonlab_job_add_gate(w, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL);
            moonlab_job_set_num_shots(w, 128);
            moonlab_job_set_num_workers(w, 1);
            moonlab_job_set_rng_seed(w, 0x99u);
            moonlab_job_results_t r; memset(&r, 0, sizeof(r));
            moonlab_scheduler_run(w, &r);
            moonlab_job_results_free(&r);
            moonlab_job_free(w);
        }
    }

    moonlab_scheduler_set_completion_hook(completion_hook, NULL);

    pthread_t ch;
    pthread_create(&ch, NULL, churn, NULL);

    pthread_t rt[N_RUNNERS];
    for (int i = 0; i < N_RUNNERS; i++) pthread_create(&rt[i], NULL, runner, NULL);
    for (int i = 0; i < N_RUNNERS; i++) pthread_join(rt[i], NULL);

    atomic_store(&g_stop, 1);
    pthread_join(ch, NULL);

    fprintf(stdout, "hook_fires=%d run_failures=%d\n",
            atomic_load(&g_hook_fires), atomic_load(&g_run_fail));
    return atomic_load(&g_run_fail) ? 1 : 0;
}
