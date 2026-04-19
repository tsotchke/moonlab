/**
 * @file bench_hadamard.c
 * @brief Single- and two-qubit gate micro-benchmarks with OpenMP.
 *
 * Covers the hot state-vector kernels:
 *   - gate_hadamard  (simd_hadamard_pair + OpenMP outer)
 *   - gate_cnot      (simd_complex_swap + OpenMP outer)
 *
 * Runs each on the middle qubit at several n, with warm-up and
 * multiple iterations. Reports wall-clock and a rough
 * memory-bandwidth estimate (two reads + two writes per pair).
 * Respects OMP_NUM_THREADS.
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static double now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

static void bench_H(size_t n) {
    quantum_state_t st;
    if (quantum_state_init(&st, n) != QS_SUCCESS) return;
    gate_hadamard(&st, 0); gate_hadamard(&st, 1);
    for (int w = 0; w < 3; w++) gate_hadamard(&st, (int)(n / 2));

    const int iters = 10;
    double t0 = now_us();
    for (int i = 0; i < iters; i++) gate_hadamard(&st, (int)(n / 2));
    double dt = (now_us() - t0) / iters;

    uint64_t dim = 1ULL << n;
    double bw = ((double)dim * 16.0 * 2.0 / (dt * 1e-6)) / 1e9;
    printf("  H       n=%2zu  %10llu  %9.2f us    %6.1f GB/s\n",
           n, (unsigned long long)dim, dt, bw);
    quantum_state_free(&st);
}

static void bench_CNOT(size_t n) {
    quantum_state_t st;
    if (quantum_state_init(&st, n) != QS_SUCCESS) return;
    gate_hadamard(&st, 0); gate_hadamard(&st, 1);

    int ctrl = (int)(n / 2);
    int tgt  = (int)(n / 2 + 1);
    for (int w = 0; w < 3; w++) gate_cnot(&st, ctrl, tgt);

    const int iters = 10;
    double t0 = now_us();
    for (int i = 0; i < iters; i++) gate_cnot(&st, ctrl, tgt);
    double dt = (now_us() - t0) / iters;

    uint64_t dim = 1ULL << n;
    /* CNOT swaps dim/4 pairs; bandwidth = dim/4 * 32 bytes R+W */
    double bw = ((double)dim * 8.0 * 2.0 / (dt * 1e-6)) / 1e9;
    printf("  CNOT    n=%2zu  %10llu  %9.2f us    %6.1f GB/s\n",
           n, (unsigned long long)dim, dt, bw);
    quantum_state_free(&st);
}

int main(void) {
#ifdef _OPENMP
    int nt = omp_get_max_threads();
    printf("=== gate micro-benchmark (OMP max=%d) ===\n\n", nt);
#else
    printf("=== gate micro-benchmark (serial; OpenMP disabled) ===\n\n");
#endif
    printf("  gate    n     dim        time         R+W bw\n");
    printf("  ----    --    ----       ----         ------\n");
    for (size_t n = 16; n <= 26; n += 2) bench_H(n);
    printf("\n");
    for (size_t n = 16; n <= 26; n += 2) bench_CNOT(n);
    return 0;
}
