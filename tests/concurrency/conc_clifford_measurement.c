/**
 * @file  conc_clifford_measurement.c
 * @brief ThreadSanitizer harness for the Clifford + measurement RNG paths.
 *
 * Both the stabilizer measurement (clifford_measure / clifford_sample_all)
 * and the state-vector measurement sampling take the RANDOMNESS as a caller-
 * owned argument -- clifford via a `uint64_t *rng_state` the caller owns,
 * measurement_* via a `double random_value` the caller supplies.  Neither
 * is supposed to touch a hidden process-global RNG.  This harness verifies
 * that under concurrency:
 *
 *   - N threads each build their OWN tableau + OWN rng_state and drive a
 *     Clifford circuit + measurements to completion,
 *   - N threads each build their OWN state vector and sample measurements
 *     with their OWN local PRNG stream.
 *
 * Nothing is shared between threads at the harness level, so a clean TSan
 * pass confirms these RNG paths carry no hidden global state.  Any race
 * reported with both stacks in clifford.c / measurement.c would be a hidden
 * global (a real bug).  The shared-core lazy singletons are warmed first.
 */

#include "../../src/backends/clifford/clifford.h"
#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/quantum/measurement.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef N_THREADS
#define N_THREADS 8
#endif
#ifndef ROUNDS
#define ROUNDS 300
#endif
#define NQ 6

static _Atomic int g_fail = 0;

static uint64_t sm64(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void *clifford_worker(void *arg)
{
    uint64_t rng = 0xA5A5u ^ (uint64_t)(intptr_t)arg;
    for (int r = 0; r < ROUNDS; r++) {
        clifford_tableau_t *t = clifford_tableau_create(NQ);
        if (!t) { atomic_fetch_add(&g_fail, 1); return NULL; }
        for (size_t q = 0; q < NQ; q++) clifford_h(t, q);
        for (size_t q = 0; q + 1 < NQ; q++) clifford_cnot(t, q, q + 1);
        clifford_s(t, 0);
        uint64_t sample = 0;
        if (clifford_sample_all(t, &rng, &sample) != CLIFFORD_SUCCESS)
            atomic_fetch_add(&g_fail, 1);
        int outcome = 0, kind = 0;
        (void)clifford_measure(t, 0, &rng, &outcome, &kind);
        clifford_tableau_free(t);
    }
    return NULL;
}

static void *measure_worker(void *arg)
{
    uint64_t rng = 0x5A5Au ^ (uint64_t)(intptr_t)arg;
    for (int r = 0; r < ROUNDS; r++) {
        quantum_state_t st;
        if (quantum_state_init(&st, NQ) != QS_SUCCESS) {
            atomic_fetch_add(&g_fail, 1);
            return NULL;
        }
        for (int q = 0; q < NQ; q++) gate_hadamard(&st, q);
        gate_cnot(&st, 0, 1);
        double u = (double)(sm64(&rng) >> 11) * 0x1.0p-53;
        (void)measurement_all_qubits(&st, u);
        quantum_state_free(&st);
    }
    return NULL;
}

int main(void)
{
    fprintf(stdout, "=== conc_clifford_measurement (%d+%d threads) ===\n",
            N_THREADS, N_THREADS);

    /* Warm shared-core lazy singletons single-threaded (config/simd). */
    {
        quantum_state_t st;
        if (quantum_state_init(&st, NQ) == QS_SUCCESS) {
            gate_hadamard(&st, 0);
            gate_cnot(&st, 0, 1);
            (void)measurement_all_qubits(&st, 0.5);
            quantum_state_free(&st);
        }
        clifford_tableau_t *t = clifford_tableau_create(NQ);
        if (t) {
            uint64_t rng = 1, s = 0;
            clifford_h(t, 0);
            clifford_sample_all(t, &rng, &s);
            clifford_tableau_free(t);
        }
    }

    pthread_t ct[N_THREADS], mt[N_THREADS];
    for (int i = 0; i < N_THREADS; i++)
        pthread_create(&ct[i], NULL, clifford_worker, (void *)(intptr_t)(i + 1));
    for (int i = 0; i < N_THREADS; i++)
        pthread_create(&mt[i], NULL, measure_worker, (void *)(intptr_t)(i + 1));
    for (int i = 0; i < N_THREADS; i++) pthread_join(ct[i], NULL);
    for (int i = 0; i < N_THREADS; i++) pthread_join(mt[i], NULL);

    fprintf(stdout, "failures=%d\n", atomic_load(&g_fail));
    return atomic_load(&g_fail) ? 1 : 0;
}
