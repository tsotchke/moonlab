/**
 * @file bench_clops.c
 * @brief CLOPS-style circuits-per-second benchmark.
 *
 * IBM CLOPS (Circuit Layer Operations Per Second) measures how many
 * parameterised circuit *executions* a simulator/QPU can complete per
 * unit wall-clock when each execution gets a fresh parameter draw.
 * This is the relevant figure of merit for variational workflows
 * (VQE, QAOA) where the circuit structure is fixed but the parameters
 * change every iteration.
 *
 * Protocol (simplified for noiseless statevector):
 *   - Build a hardware-efficient ansatz at width n with L layers.
 *   - Run K iterations.  Each iteration:
 *       . Reset state to |0...0>.
 *       . Sample 3*n*L random rotation angles.
 *       . Apply the ansatz with those angles.
 *       . Measure all qubits in the Z basis.
 *   - CLOPS = K / total_wall_clock_seconds.
 *
 * Reports CLOPS at n in {8, 12, 16, 20} for L=5 layers, and emits
 * JSON with schema "moonlab/clops_v1" for paper plots.
 *
 * Closes the moonlab paper §6.2 todo about reporting CLOPS-class
 * variational throughput numbers.
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/quantum/measurement.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static double now_s(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

static uint64_t xrng = 0xC10C5BEEF00DULL;
static double rng_angle(void) {
    xrng ^= xrng << 13; xrng ^= xrng >> 7; xrng ^= xrng << 17;
    return ((uint32_t)xrng / (double)UINT32_MAX) * 2.0 * M_PI;
}

static double rng_uniform_01(void) {
    xrng ^= xrng << 13; xrng ^= xrng >> 7; xrng ^= xrng << 17;
    return (uint32_t)xrng / (double)UINT32_MAX;
}

/* Run one HWEA circuit at width n with L layers using the supplied
 * parameter buffer.  Each layer applies Rz, Rx, Rz on every qubit
 * then a CNOT ladder. */
static void run_one_circuit(quantum_state_t* st, size_t n, int layers,
                             const double* angles) {
    /* Reset to |0...0>. */
    quantum_state_reset(st);
    size_t k = 0;
    for (int l = 0; l < layers; l++) {
        for (size_t q = 0; q < n; q++) gate_rz(st, (int)q, angles[k++]);
        for (size_t q = 0; q < n; q++) gate_rx(st, (int)q, angles[k++]);
        for (size_t q = 0; q < n; q++) gate_rz(st, (int)q, angles[k++]);
        for (size_t q = 0; q + 1 < n; q++) gate_cnot(st, (int)q, (int)(q + 1));
    }
}

typedef struct {
    size_t n;
    int    layers;
    size_t iters;
    double total_wall_s;
    double clops;
} clops_row_t;

static void bench_one(clops_row_t* row, size_t n, int layers, size_t iters) {
    quantum_state_t st;
    if (quantum_state_init(&st, n) != QS_SUCCESS) return;

    const size_t params_per_circuit = (size_t)layers * 3 * n;
    double* angles = (double*)calloc(params_per_circuit, sizeof(double));
    /* Pre-generate the full parameter stream (independent draws per
     * iteration) so the CLOPS measurement excludes RNG cost.  In
     * production these would come from the optimizer. */
    double** all_angles = (double**)calloc(iters, sizeof(double*));
    for (size_t k = 0; k < iters; k++) {
        all_angles[k] = (double*)calloc(params_per_circuit, sizeof(double));
        for (size_t i = 0; i < params_per_circuit; i++)
            all_angles[k][i] = rng_angle();
    }

    /* Warm cache. */
    run_one_circuit(&st, n, layers, all_angles[0]);

    double t0 = now_s();
    for (size_t k = 0; k < iters; k++) {
        run_one_circuit(&st, n, layers, all_angles[k]);
        /* Measure all qubits to mimic the CLOPS protocol's
         * measurement phase; result discarded. */
        size_t outcome = measurement_all_qubits(&st, rng_uniform_01());
        (void)outcome;
    }
    double dt = now_s() - t0;

    row->n = n;
    row->layers = layers;
    row->iters = iters;
    row->total_wall_s = dt;
    row->clops = (double)iters / dt;

    printf("  n=%-3zu  L=%-2d  iters=%-6zu  wall=%.3fs  CLOPS=%.0f\n",
           n, layers, iters, dt, row->clops);

    for (size_t k = 0; k < iters; k++) free(all_angles[k]);
    free(all_angles);
    free(angles);
    quantum_state_free(&st);
}

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1] : "clops.json";

    printf("=== CLOPS benchmark (HWEA, noiseless statevector) ===\n");
    printf("  schema: moonlab/clops_v1  out: %s\n", out_path);
#ifdef _OPENMP
    printf("  OpenMP threads: %d\n", omp_get_max_threads());
#endif
    printf("\n");

    /* Iter counts auto-tuned: small n -> many iters; large n -> few. */
    typedef struct { size_t n; int layers; size_t iters; } cfg_t;
    const cfg_t configs[] = {
        {  8, 5, 20000 },
        { 12, 5,  4000 },
        { 16, 5,   500 },
        { 20, 5,    40 },
    };
    const size_t n_cfg = sizeof(configs) / sizeof(configs[0]);

    clops_row_t rows[8];
    for (size_t ci = 0; ci < n_cfg; ci++) {
        bench_one(&rows[ci], configs[ci].n, configs[ci].layers, configs[ci].iters);
    }

    FILE* f = fopen(out_path, "w");
    if (!f) { fprintf(stderr, "fopen(%s) failed\n", out_path); return 1; }
    fprintf(f, "{\n");
    fprintf(f, "  \"schema\": \"moonlab/clops_v1\",\n");
    fprintf(f, "  \"description\": \"CLOPS = circuits/second on a "
               "VQE/QAOA-shaped hardware-efficient ansatz (Rz-Rx-Rz "
               "+ CNOT ladder) at L layers.  Each iteration uses a "
               "fresh parameter draw plus a final all-qubit Z-basis "
               "measurement.\",\n");
#ifdef _OPENMP
    fprintf(f, "  \"omp_threads\": %d,\n", omp_get_max_threads());
#else
    fprintf(f, "  \"omp_threads\": 0,\n");
#endif
    fprintf(f, "  \"rows\": [");
    for (size_t i = 0; i < n_cfg; i++) {
        fprintf(f, "%s\n    {\"n\": %zu, \"layers\": %d, \"iters\": %zu, "
                   "\"total_wall_s\": %.6f, \"clops\": %.3f}",
                i == 0 ? "" : ",",
                rows[i].n, rows[i].layers, rows[i].iters,
                rows[i].total_wall_s, rows[i].clops);
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    printf("\nwrote %s\n", out_path);
    return 0;
}
