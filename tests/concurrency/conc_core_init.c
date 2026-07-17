/**
 * @file  conc_core_init.c
 * @brief Minimal cold-start reproducer for the shared-core lazy-init races.
 *
 * Many worker threads each build and execute their OWN independent circuit
 * from a cold process.  No state is shared between threads at the harness
 * level -- every thread has its own circuit, its own quantum_state, its own
 * results.  Any data race TSan reports here is therefore inside the library's
 * unsynchronised lazy initialisation of PROCESS-GLOBAL singletons on first
 * touch:
 *
 *   - qsim_config_global() / qsim_config_init()  (src/utils/config.c)
 *       g_initialized (plain int) + g_config (pointer) + the 248-byte
 *       config struct are read/written with no lock or atomic.
 *
 *   - simd_dispatch_init_once()                  (src/optimization/simd_ops.c)
 *       g_simd_vtable (struct of fn pointers) guarded only by a
 *       `volatile int` flag -- volatile is not synchronisation, so the
 *       vtable store can be observed torn / a NULL fn pointer read.
 *
 * These fire for ANY concurrent first use of the simulator core: the
 * scheduler's own OpenMP shot fan-out, parallel Grover, and the control
 * plane all hit them.  The standard fix is a pthread_once (or a single
 * documented single-threaded warm-up call before going parallel).
 *
 * This harness deliberately does NOT warm up, so the race window is open.
 */

#include "../../src/applications/moonlab_qgtl_backend.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef N_THREADS
#define N_THREADS 8
#endif

static _Atomic int g_exec_failures = 0;

static void *worker(void *arg)
{
    (void)arg;
    moonlab_qgtl_circuit_t *c = moonlab_qgtl_circuit_create(3);
    if (!c) { atomic_fetch_add(&g_exec_failures, 1); return NULL; }
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_H,    0, 0, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL);
    moonlab_qgtl_add_gate(c, MOONLAB_QGTL_GATE_CNOT, 2, 1, NULL);

    moonlab_qgtl_exec_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.return_probabilities = 1;

    moonlab_qgtl_results_t res;
    memset(&res, 0, sizeof(res));
    if (moonlab_qgtl_execute(c, &opts, &res) != 0) {
        atomic_fetch_add(&g_exec_failures, 1);
    }
    moonlab_qgtl_results_free(&res);
    moonlab_qgtl_circuit_free(c);
    return NULL;
}

int main(void)
{
    fprintf(stdout, "=== conc_core_init (cold-start, %d threads) ===\n", N_THREADS);
    pthread_t th[N_THREADS];
    /* All threads start as close together as possible so they collide in the
     * one-time init window. */
    for (int i = 0; i < N_THREADS; i++) pthread_create(&th[i], NULL, worker, NULL);
    for (int i = 0; i < N_THREADS; i++) pthread_join(th[i], NULL);
    fprintf(stdout, "exec_failures=%d\n", atomic_load(&g_exec_failures));
    return atomic_load(&g_exec_failures) ? 1 : 0;
}
