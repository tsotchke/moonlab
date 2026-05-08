/**
 * @file bench_state_throughput.c
 * @brief State-vector single- and two-qubit gate throughput, JSON output.
 *
 * Paper-grade companion to bench_hadamard.c: same kernels (gate_hadamard,
 * gate_cnot, gate_ry, gate_rz) at canonical n in {16, 18, 20, 22, 24, 26},
 * but emits a JSON file the paper can pull directly.
 *
 * Closes the moonlab paper §3.2 todo (line ~231, "Per-host throughput
 * numbers (AMX on Apple Silicon, AVX-2 on x86) to be measured and
 * reported here in the next revision").
 *
 * Reported metrics per (gate, n), for k=5 independent repetitions:
 *   - mean ± stddev per-iteration wall time (microseconds)
 *   - mean read+write bandwidth (GB/s) and stddev
 *   - min wall time (noise-floor proxy)
 *
 * Output JSON schema: "moonlab/state_throughput_v2" (k-rep error bars).
 *
 * Invocation: ./bench_state_throughput [out.json [k_reps]]
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static double now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}

typedef struct {
    const char* gate_name;
    size_t n;
    uint64_t dim;
    int    k;
    double mean_us;     /* mean over k reps of "mean over 10 iters" */
    double std_us;      /* stddev over k reps */
    double min_us;      /* minimum over k reps (noise-floor proxy) */
    double mean_gbps;
    double std_gbps;
    double max_gbps;    /* corresponds to min_us */
} bench_row_t;

static double bench_one_rep(size_t n,
                             void (*apply_warmup)(quantum_state_t*),
                             void (*apply_timed)(quantum_state_t*)) {
    quantum_state_t st;
    if (quantum_state_init(&st, n) != QS_SUCCESS) return 0.0;
    gate_hadamard(&st, 0); gate_hadamard(&st, 1);
    for (int w = 0; w < 3; w++) apply_warmup(&st);
    const int iters = 10;
    double t0 = now_us();
    for (int i = 0; i < iters; i++) apply_timed(&st);
    double dt = (now_us() - t0) / (double)iters;
    quantum_state_free(&st);
    return dt;
}

static void bench_one(bench_row_t* row, const char* name, size_t n,
                       int k_reps,
                       void (*apply_warmup)(quantum_state_t*),
                       void (*apply_timed)(quantum_state_t*)) {
    double* samples = (double*)calloc((size_t)k_reps, sizeof(double));
    for (int k = 0; k < k_reps; k++) {
        samples[k] = bench_one_rep(n, apply_warmup, apply_timed);
    }
    double sum = 0, sum2 = 0, mn = samples[0];
    for (int k = 0; k < k_reps; k++) {
        sum  += samples[k];
        sum2 += samples[k] * samples[k];
        if (samples[k] < mn) mn = samples[k];
    }
    double mean = sum / (double)k_reps;
    double var  = (sum2 / (double)k_reps) - mean * mean;
    if (var < 0) var = 0;
    double std  = sqrt(var);

    uint64_t dim = 1ULL << n;
    /* Bandwidth: 2 amp passes * 16 bytes/amp.  Treat as analytic floor. */
    double bytes = (double)dim * 16.0 * 2.0;
    double mean_gbps = (bytes / (mean * 1e-6)) / 1e9;
    /* gbps stddev via delta method. */
    double std_gbps = mean_gbps * (std / mean);
    double max_gbps = (bytes / (mn * 1e-6)) / 1e9;

    row->gate_name = name;
    row->n = n;
    row->dim = dim;
    row->k = k_reps;
    row->mean_us  = mean;
    row->std_us   = std;
    row->min_us   = mn;
    row->mean_gbps = mean_gbps;
    row->std_gbps  = std_gbps;
    row->max_gbps  = max_gbps;
    free(samples);
}

static int s_qubit_mid;
static int s_qubit_mid_p1;
static double s_theta;

static void apply_h(quantum_state_t* st)    { gate_hadamard(st, s_qubit_mid); }
static void apply_cnot(quantum_state_t* st) { gate_cnot(st, s_qubit_mid, s_qubit_mid_p1); }
static void apply_ry(quantum_state_t* st)   { gate_ry(st, s_qubit_mid, s_theta); }
static void apply_rz(quantum_state_t* st)   { gate_rz(st, s_qubit_mid, s_theta); }

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1] : "state_throughput.json";
    int k_reps = (argc >= 3) ? atoi(argv[2]) : 5;
    if (k_reps < 1) k_reps = 1;

    fprintf(stdout,
            "=== State-vector gate throughput ===\n"
            "  schema: moonlab/state_throughput_v2  (k=%d reps)\n", k_reps);
#ifdef _OPENMP
    fprintf(stdout, "  OpenMP threads: %d\n", omp_get_max_threads());
#else
    fprintf(stdout, "  OpenMP: disabled\n");
#endif
    fprintf(stdout, "  gate   n     dim         mean+/-std (us)        min (us)   mean GB/s     max GB/s\n");

    const size_t ns[] = { 16, 18, 20, 22, 24, 26 };
    const size_t n_ns = sizeof(ns) / sizeof(ns[0]);

    typedef struct {
        const char* name;
        void (*warmup)(quantum_state_t*);
        void (*timed)(quantum_state_t*);
    } gate_kind_t;
    const gate_kind_t gates[] = {
        {"H",    apply_h,    apply_h},
        {"CNOT", apply_cnot, apply_cnot},
        {"RY",   apply_ry,   apply_ry},
        {"RZ",   apply_rz,   apply_rz},
    };
    const size_t n_gates = sizeof(gates) / sizeof(gates[0]);

    s_theta = 0.42;

    bench_row_t* rows = (bench_row_t*)calloc(n_gates * n_ns, sizeof(bench_row_t));
    size_t row_idx = 0;
    for (size_t gi = 0; gi < n_gates; gi++) {
        for (size_t ni = 0; ni < n_ns; ni++) {
            const size_t n = ns[ni];
            s_qubit_mid    = (int)(n / 2);
            s_qubit_mid_p1 = (int)(n / 2 + 1);
            bench_one(&rows[row_idx],
                      gates[gi].name, n, k_reps,
                      gates[gi].warmup, gates[gi].timed);
            fprintf(stdout,
                    "  %-5s  n=%-3zu %-11llu  %9.2f +/- %7.2f  %9.2f  %9.2f   %9.2f\n",
                    rows[row_idx].gate_name, rows[row_idx].n,
                    (unsigned long long)rows[row_idx].dim,
                    rows[row_idx].mean_us, rows[row_idx].std_us,
                    rows[row_idx].min_us,
                    rows[row_idx].mean_gbps, rows[row_idx].max_gbps);
            row_idx++;
        }
        fprintf(stdout, "\n");
    }

    FILE* f = fopen(out_path, "w");
    if (!f) { fprintf(stderr, "fopen(%s) failed\n", out_path); return 1; }
    fprintf(f, "{\n");
    fprintf(f, "  \"schema\": \"moonlab/state_throughput_v2\",\n");
    fprintf(f, "  \"k_reps\": %d,\n", k_reps);
#ifdef _OPENMP
    fprintf(f, "  \"omp_threads\": %d,\n", omp_get_max_threads());
#else
    fprintf(f, "  \"omp_threads\": 0,\n");
#endif
    fprintf(f, "  \"description\": \"Per-host gate throughput on the "
               "state-vector engine; for k=%d independent reps each "
               "averaging 10 iterations on the middle qubit (or middle "
               "pair for CNOT) of an n-qubit |+>-prepared state.  Reports "
               "mean +- stddev per-iter wall, min (noise floor), mean "
               "and max read+write bandwidth.  Bandwidth = 2 * dim * 16 "
               "bytes / wall is an analytic floor not a measured DRAM "
               "controller traffic; treat as an order-of-magnitude figure.\",\n",
            k_reps);
    fprintf(f, "  \"rows\": [");
    for (size_t i = 0; i < row_idx; i++) {
        fprintf(f, "%s\n    {\"gate\": \"%s\", \"n\": %zu, \"dim\": %llu, "
                   "\"k_reps\": %d, \"mean_us\": %.4f, \"std_us\": %.4f, "
                   "\"min_us\": %.4f, \"mean_gbps\": %.4f, \"std_gbps\": %.4f, "
                   "\"max_gbps\": %.4f}",
                i == 0 ? "" : ",", rows[i].gate_name, rows[i].n,
                (unsigned long long)rows[i].dim, rows[i].k,
                rows[i].mean_us, rows[i].std_us, rows[i].min_us,
                rows[i].mean_gbps, rows[i].std_gbps, rows[i].max_gbps);
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    fprintf(stdout, "wrote %s\n", out_path);

    free(rows);
    return 0;
}
