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

typedef struct {
    size_t n;
    int    layers;
    size_t original_gates;
    size_t fused_gates;
    size_t merges;
    double t_src_ms;
    double t_fc_ms;
    double speedup;
} fusion_row_t;

static void bench_n(size_t n, int layers, fusion_row_t* row) {
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

    row->n = n;
    row->layers = layers;
    row->original_gates = stats.original_gates;
    row->fused_gates = stats.fused_gates;
    row->merges = stats.merges_applied;
    row->t_src_ms = t_src / 1000.0;
    row->t_fc_ms = t_fc / 1000.0;
    row->speedup = t_src / t_fc;

    printf("  n=%2zu  layers=%2d  gates %zu->%zu (%zu merges)  "
           "time: %.1f ms -> %.1f ms  (%.2fx)\n",
           n, layers,
           stats.original_gates, stats.fused_gates, stats.merges_applied,
           row->t_src_ms, row->t_fc_ms, row->speedup);

    fuse_circuit_free(fc);
    fuse_circuit_free(src);
}

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1] : "fusion.json";
    printf("=== Gate-fusion benchmark (VQE/QAOA HWEA layers) ===\n");
    printf("  schema: moonlab/fusion_v1  out: %s\n\n", out_path);

    fusion_row_t rows[4];
    bench_n(12, 5, &rows[0]);
    bench_n(16, 5, &rows[1]);
    bench_n(20, 5, &rows[2]);
    bench_n(22, 3, &rows[3]);

    FILE* f = fopen(out_path, "w");
    if (!f) { fprintf(stderr, "fopen(%s) failed\n", out_path); return 1; }
    fprintf(f, "{\n");
    fprintf(f, "  \"schema\": \"moonlab/fusion_v1\",\n");
    fprintf(f, "  \"description\": \"Gate-fusion DAG speedup on a "
               "VQE/QAOA hardware-efficient ansatz (HWEA): per-qubit "
               "Rz-Rx-Rz rotations followed by a CNOT ladder, repeated "
               "L times.  Speedup = unfused_time / fused_time.\",\n");
    fprintf(f, "  \"rows\": [");
    for (int i = 0; i < 4; i++) {
        fprintf(f, "%s\n    {\"n\": %zu, \"layers\": %d, "
                   "\"original_gates\": %zu, \"fused_gates\": %zu, "
                   "\"merges\": %zu, \"t_unfused_ms\": %.4f, "
                   "\"t_fused_ms\": %.4f, \"speedup\": %.4f}",
                i == 0 ? "" : ",",
                rows[i].n, rows[i].layers,
                rows[i].original_gates, rows[i].fused_gates,
                rows[i].merges, rows[i].t_src_ms, rows[i].t_fc_ms,
                rows[i].speedup);
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    printf("\nwrote %s\n", out_path);
    return 0;
}
