/**
 * @file bench_hadamard.c
 * @brief Dedicated Hadamard-gate micro-benchmark.
 *
 * The hot inner kernel of `gate_hadamard` is `simd_hadamard_pair`
 * (NEON on AArch64; AVX-512 / SSE2 on x86_64; scalar fallback).
 * This bench measures throughput on H applied to the middle qubit at
 * several n, with warm-up and multiple iterations.  Output includes a
 * rough memory-bandwidth estimate (two reads + two writes per pair).
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

static double now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

static void bench_n(size_t n) {
    quantum_state_t st;
    if (quantum_state_init(&st, n) != QS_SUCCESS) {
        printf("n=%2zu  init failed\n", n);
        return;
    }
    /* Break symmetry so the compiler / memory system is not reading a
     * zero vector (negligible effect on dense SV, but keeps parity
     * with real workloads). */
    gate_hadamard(&st, 0);
    gate_hadamard(&st, 1);

    /* Warm-up. */
    const int warm = 3;
    for (int w = 0; w < warm; w++) gate_hadamard(&st, (int)(n / 2));

    /* Measurement. */
    const int iters = 10;
    double t0 = now_us();
    for (int i = 0; i < iters; i++) gate_hadamard(&st, (int)(n / 2));
    double dt_us = (now_us() - t0) / iters;

    uint64_t dim   = 1ULL << n;
    double   bytes = (double)dim * 16.0;       /* complex_t = 16 B */
    double   bw_gb = (bytes * 2.0 / (dt_us * 1e-6)) / 1e9; /* R + W */

    printf("  n=%2zu  dim=%10llu  H(mid): %9.2f us    %6.1f GB/s (R+W)\n",
           n, (unsigned long long)dim, dt_us, bw_gb);

    quantum_state_free(&st);
}

int main(void) {
    printf("=== gate_hadamard micro-benchmark ===\n");
    printf("Hot kernel: simd_hadamard_pair (NEON / AVX-512 / scalar).\n\n");
    for (size_t n = 16; n <= 26; n += 2) bench_n(n);
    return 0;
}
