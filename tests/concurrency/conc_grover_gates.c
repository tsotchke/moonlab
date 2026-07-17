/**
 * @file  conc_grover_gates.c
 * @brief ThreadSanitizer harness for the OpenMP-parallel numeric core.
 *
 * Built against an OpenMP-ENABLED, TSan-instrumented libquantumsim so the
 * intra-gate and fan-out `#pragma omp parallel for` regions actually run
 * multithreaded.  Three surfaces:
 *
 *   test "gates18"  -- apply a gate sequence to an 18-qubit state
 *                      (state_dim == 2^18 == QS_BLOCK_THRESHOLD_DIM), which
 *                      trips the OpenMP block-loop in gates.c.  Also runs
 *                      several independent 18-qubit states from separate
 *                      pthreads to stress cross-state independence on top of
 *                      the intra-gate threading.
 *
 *   test "grover"   -- grover_parallel_random_batch(): the OpenMP outer
 *                      parallel-for over independent searches.  VERIFIES the
 *                      entropy-context isolation the audit flagged (each
 *                      worker must get its OWN splitmix state / user_data;
 *                      a regression to a shared/aliased master ctx would race
 *                      here).
 *
 *   test "schedule" -- moonlab_scheduler_run with num_workers=8: the OpenMP
 *                      shot fan-out, each worker writing a DISJOINT slice of
 *                      out->outcomes.
 *
 * NOTE on OpenMP + TSan: the Homebrew libomp is not Archer-instrumented, so
 * TSan cannot see the OpenMP barrier's happens-before and MAY emit false
 * positives on the runtime's own frames (frames under the __kmp / gomp
 * runtime).  The
 * harness itself shares nothing across worker iterations, so any race whose
 * BOTH stacks are in moonlab code (gates.c/state.c/grover.c/parallel_ops.c)
 * is real; run_tsan.sh applies tests/concurrency/tsan.supp to filter the
 * libomp-internal noise.
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/optimization/parallel_ops.h"
#include "../../src/distributed/scheduler.h"
#include "../../src/applications/moonlab_qgtl_backend.h"
#include "../../src/utils/quantum_entropy.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef GATE_QUBITS
#define GATE_QUBITS 18   /* 2^18 == QS_BLOCK_THRESHOLD_DIM */
#endif

static _Atomic int g_fail = 0;

/* Deterministic single-threaded master entropy source for grover batch
 * seeding (drawn sequentially before the parallel region). */
static int det_get_bytes(void *ud, uint8_t *buf, size_t n)
{
    uint64_t *s = (uint64_t *)ud;
    size_t off = 0;
    while (off < n) {
        uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        size_t c = (n - off < 8) ? (n - off) : 8;
        memcpy(buf + off, &z, c);
        off += c;
    }
    return 0;
}

static void apply_18q_sequence(void)
{
    quantum_state_t st;
    if (quantum_state_init(&st, GATE_QUBITS) != QS_SUCCESS) {
        atomic_fetch_add(&g_fail, 1);
        return;
    }
    for (int q = 0; q < GATE_QUBITS; q++) gate_hadamard(&st, q);   /* omp block loop */
    for (int q = 0; q + 1 < GATE_QUBITS; q++) gate_cnot(&st, q, q + 1);
    gate_swap(&st, 0, GATE_QUBITS - 1);
    for (int q = 0; q < GATE_QUBITS; q++) gate_pauli_x(&st, q);
    quantum_state_free(&st);
}

static void *gate_worker(void *arg) { (void)arg; apply_18q_sequence(); return NULL; }

static void test_gates18(void)
{
    fprintf(stdout, "--- gates18: %d-qubit state, intra-gate OpenMP ---\n", GATE_QUBITS);
    /* Single state first (pure intra-gate OpenMP). */
    apply_18q_sequence();
    /* Then several independent states from pthreads on top of it. */
    enum { NT = 4 };
    pthread_t th[NT];
    for (int i = 0; i < NT; i++) pthread_create(&th[i], NULL, gate_worker, NULL);
    for (int i = 0; i < NT; i++) pthread_join(th[i], NULL);
}

static void test_grover(void)
{
    fprintf(stdout, "--- grover: parallel_random_batch entropy isolation ---\n");
    uint64_t seed = 0xDEADBEEFCAFEF00DULL;
    quantum_entropy_ctx_t master;
    quantum_entropy_init(&master, det_get_bytes, &seed);
    grover_parallel_result_t r =
        grover_parallel_random_batch(/*num_searches=*/32, /*num_qubits=*/8, &master);
    if (!r.results) atomic_fetch_add(&g_fail, 1);
    grover_parallel_free_result(&r);
}

static void test_schedule(void)
{
    fprintf(stdout, "--- schedule: OpenMP shot fan-out, disjoint slices ---\n");
    moonlab_job_t *j = moonlab_job_create(3);
    if (!j) { atomic_fetch_add(&g_fail, 1); return; }
    moonlab_job_add_gate(j, MOONLAB_QGTL_GATE_H,    0, 0, NULL);
    moonlab_job_add_gate(j, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL);
    moonlab_job_add_gate(j, MOONLAB_QGTL_GATE_CNOT, 2, 1, NULL);
    moonlab_job_set_num_shots(j, 4096);
    moonlab_job_set_num_workers(j, 8);
    moonlab_job_set_rng_seed(j, 0xABCDu);
    moonlab_job_results_t out; memset(&out, 0, sizeof(out));
    if (moonlab_scheduler_run(j, &out) != MOONLAB_SCHED_OK) atomic_fetch_add(&g_fail, 1);
    moonlab_job_results_free(&out);
    moonlab_job_free(j);
}

int main(int argc, char **argv)
{
    const char *which = (argc > 1) ? argv[1] : "all";
    fprintf(stdout, "=== conc_grover_gates (%s) ===\n", which);

    /* Warm shared-core lazy singletons (config/simd/measurement) single-
     * threaded so their first-touch races (conc_core_init) do not mask the
     * OpenMP-region findings under test. */
    {
        moonlab_job_t *w = moonlab_job_create(3);
        if (w) {
            moonlab_job_add_gate(w, MOONLAB_QGTL_GATE_H,    0, 0, NULL);
            moonlab_job_add_gate(w, MOONLAB_QGTL_GATE_CNOT, 1, 0, NULL);
            moonlab_job_set_num_shots(w, 64);
            moonlab_job_set_num_workers(w, 1);
            moonlab_job_set_rng_seed(w, 0x1u);
            moonlab_job_results_t r; memset(&r, 0, sizeof(r));
            moonlab_scheduler_run(w, &r);
            moonlab_job_results_free(&r);
            moonlab_job_free(w);
        }
        /* Warm the >=2^18 intra-gate path too. */
        apply_18q_sequence();
    }

    if (!strcmp(which, "all") || !strcmp(which, "gates18"))  test_gates18();
    if (!strcmp(which, "all") || !strcmp(which, "grover"))   test_grover();
    if (!strcmp(which, "all") || !strcmp(which, "schedule")) test_schedule();

    fprintf(stdout, "failures=%d\n", atomic_load(&g_fail));
    return atomic_load(&g_fail) ? 1 : 0;
}
