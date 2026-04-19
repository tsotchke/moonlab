/**
 * @file bench_fusion.c
 * @brief Wall-clock comparison: fused vs unfused execution.
 *
 * Builds a VQE/QAOA-shaped circuit (alternating single-qubit rotation
 * layers and entangling CNOT ladders) at n = 16, 20, 24 qubits and
 * prints microseconds / reduction ratio for each.
 */

#include "../../src/optimization/fusion/fusion.h"
#include "../../src/quantum/state.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

qs_error_t quantum_state_init(quantum_state_t* state, size_t num_qubits);
void       quantum_state_free(quantum_state_t* state);

static double now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

static uint64_t xrng = 0xBADC0FFEE0DDF00DULL;
static double rng_angle(void) {
    xrng ^= xrng << 13; xrng ^= xrng >> 7; xrng ^= xrng << 17;
    return ((uint32_t)xrng / (double)UINT32_MAX) * 2.0 * M_PI;
}

static void build_hwea_layer(fuse_circuit_t* c, size_t n) {
    /* Hardware-efficient ansatz layer: Rz, Rx, Rz on every qubit then
     * CNOT ladder. Typical VQE/QAOA shape. */
    for (size_t q = 0; q < n; q++) fuse_append_rz(c, (int)q, rng_angle());
    for (size_t q = 0; q < n; q++) fuse_append_rx(c, (int)q, rng_angle());
    for (size_t q = 0; q < n; q++) fuse_append_rz(c, (int)q, rng_angle());
    for (size_t q = 0; q + 1 < n; q++) fuse_append_cnot(c, (int)q, (int)(q + 1));
}

static void bench_n(size_t n, int layers) {
    fuse_circuit_t* src = fuse_circuit_create(n);
    for (int l = 0; l < layers; l++) build_hwea_layer(src, n);

    fuse_stats_t stats;
    fuse_circuit_t* fc = fuse_compile(src, &stats);

    quantum_state_t a, b;
    quantum_state_init(&a, n);
    quantum_state_init(&b, n);

    /* Warm cache. */
    fuse_execute(src, &a);
    fuse_execute(fc, &b);
    quantum_state_free(&a);
    quantum_state_free(&b);

    double t_src, t_fc;
    {
        quantum_state_t s;
        quantum_state_init(&s, n);
        double t0 = now_us();
        fuse_execute(src, &s);
        t_src = now_us() - t0;
        quantum_state_free(&s);
    }
    {
        quantum_state_t s;
        quantum_state_init(&s, n);
        double t0 = now_us();
        fuse_execute(fc, &s);
        t_fc = now_us() - t0;
        quantum_state_free(&s);
    }

    printf("  n=%2zu  layers=%2d  gates %zu->%zu (%zu merges)  "
           "time: %.1f ms -> %.1f ms  (%.2fx)\n",
           n, layers,
           stats.original_gates, stats.fused_gates, stats.merges_applied,
           t_src / 1000.0, t_fc / 1000.0,
           t_src / t_fc);

    fuse_circuit_free(fc);
    fuse_circuit_free(src);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    printf("=== Gate-fusion benchmark (VQE/QAOA HWEA layers) ===\n");
    bench_n(12, 5);
    bench_n(16, 5);
    bench_n(20, 5);
    bench_n(22, 3);
    return 0;
}
