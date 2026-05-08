/**
 * @file bench_xeb.c
 * @brief Cross-entropy benchmark (XEB) on noiseless statevector.
 *
 * XEB measures how Porter-Thomas a random-circuit output distribution
 * is.  For an ideal (Haar-random) circuit:
 *   F_xeb = D * E_{s ~ p}[p(s)] - 1
 * where D = 2^n, p(s) = |<s|psi>|^2, and the expectation is over
 * bitstring samples drawn from p.  The Porter-Thomas distribution
 * gives F_xeb -> 1 in expectation (with O(1/D) fluctuations).
 *
 * Two F_xeb estimators are reported per (n, circuit_seed):
 *   - exact:     F_xeb = D * sum_s p(s)^2 - 1, computed directly
 *                from the full statevector (no sampling).  This is
 *                the asymptotic ground truth.
 *   - sampled:   draw N_shots bitstrings according to p and compute
 *                mean of p at those samples.  Includes shot noise
 *                of order ~1/sqrt(N_shots).
 *
 * Output JSON: schema "moonlab/xeb_v1".  Closes the moonlab paper
 * §6.2 todo about reporting XEB curves alongside QV / CLOPS.
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"

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

static uint64_t xrng = 0xCEBE7600F1FF00DULL;
static double rng_uniform_01(void) {
    xrng ^= xrng << 13; xrng ^= xrng >> 7; xrng ^= xrng << 17;
    return (uint32_t)xrng / (double)UINT32_MAX;
}
static double rng_angle(void) { return rng_uniform_01() * 2.0 * M_PI; }

/* Build a random hardware-efficient ansatz at width n with L layers
 * and apply it to the supplied state, which must already be |0...0>. */
static void apply_random_circuit(quantum_state_t* st, size_t n, int layers) {
    for (int l = 0; l < layers; l++) {
        for (size_t q = 0; q < n; q++) gate_rz(st, (int)q, rng_angle());
        for (size_t q = 0; q < n; q++) gate_rx(st, (int)q, rng_angle());
        for (size_t q = 0; q < n; q++) gate_rz(st, (int)q, rng_angle());
        for (size_t q = 0; q + 1 < n; q++) gate_cnot(st, (int)q, (int)(q + 1));
    }
}

static double exact_F_xeb(const quantum_state_t* st) {
    const uint64_t D = 1ULL << st->num_qubits;
    double sum_p2 = 0.0;
    /* Sum over basis states of |amp|^4. */
    for (uint64_t s = 0; s < D; s++) {
        double re = creal(st->amplitudes[s]);
        double im = cimag(st->amplitudes[s]);
        double p = re * re + im * im;
        sum_p2 += p * p;
    }
    return (double)D * sum_p2 - 1.0;
}

/* Sampled F_xeb: draw n_shots bitstrings using inverse-CDF on the
 * statevector's |amp|^2 distribution, then average p(s) at the
 * sampled bitstrings. */
static double sampled_F_xeb(const quantum_state_t* st, size_t n_shots) {
    const uint64_t D = 1ULL << st->num_qubits;
    double* p = (double*)calloc(D, sizeof(double));
    double* cdf = (double*)calloc(D, sizeof(double));
    if (!p || !cdf) { free(p); free(cdf); return 0.0; }

    double sum = 0.0;
    for (uint64_t s = 0; s < D; s++) {
        double re = creal(st->amplitudes[s]);
        double im = cimag(st->amplitudes[s]);
        p[s] = re * re + im * im;
        sum += p[s];
        cdf[s] = sum;
    }
    /* Renormalise CDF in case of tiny rounding. */
    if (sum > 0.0) for (uint64_t s = 0; s < D; s++) cdf[s] /= sum;

    double mean_p = 0.0;
    for (size_t k = 0; k < n_shots; k++) {
        double u = rng_uniform_01();
        /* Binary search for s with cdf[s-1] < u <= cdf[s]. */
        uint64_t lo = 0, hi = D - 1;
        while (lo < hi) {
            uint64_t mid = (lo + hi) / 2;
            if (cdf[mid] < u) lo = mid + 1;
            else hi = mid;
        }
        mean_p += p[lo];
    }
    mean_p /= (double)n_shots;

    free(p); free(cdf);
    return (double)D * mean_p - 1.0;
}

typedef struct {
    size_t n;
    int    layers;
    size_t n_circuits;
    size_t n_shots;
    double mean_F_exact;
    double std_F_exact;
    double mean_F_sampled;
    double std_F_sampled;
    double wall_s;
} xeb_row_t;

static void bench_one(xeb_row_t* row, size_t n, int layers,
                       size_t n_circuits, size_t n_shots) {
    double sum_e = 0, sum_e2 = 0;
    double sum_s = 0, sum_s2 = 0;

    quantum_state_t st;
    if (quantum_state_init(&st, n) != QS_SUCCESS) return;

    double t0 = now_s();
    for (size_t k = 0; k < n_circuits; k++) {
        quantum_state_reset(&st);
        apply_random_circuit(&st, n, layers);
        double Fe = exact_F_xeb(&st);
        double Fs = sampled_F_xeb(&st, n_shots);
        sum_e += Fe; sum_e2 += Fe * Fe;
        sum_s += Fs; sum_s2 += Fs * Fs;
    }
    double dt = now_s() - t0;

    double me = sum_e / (double)n_circuits;
    double ve = (sum_e2 - (double)n_circuits * me * me) /
                (double)(n_circuits > 1 ? n_circuits - 1 : 1);
    double ms = sum_s / (double)n_circuits;
    double vs = (sum_s2 - (double)n_circuits * ms * ms) /
                (double)(n_circuits > 1 ? n_circuits - 1 : 1);

    row->n = n;
    row->layers = layers;
    row->n_circuits = n_circuits;
    row->n_shots = n_shots;
    row->mean_F_exact   = me;
    row->std_F_exact    = sqrt(ve > 0 ? ve : 0.0);
    row->mean_F_sampled = ms;
    row->std_F_sampled  = sqrt(vs > 0 ? vs : 0.0);
    row->wall_s = dt;

    printf("  n=%-3zu  L=%-2d  circuits=%-4zu  shots=%-6zu  "
           "F_exact=%.4f+/-%.4f  F_sampled=%.4f+/-%.4f  wall=%.2fs\n",
           n, layers, n_circuits, n_shots,
           row->mean_F_exact, row->std_F_exact,
           row->mean_F_sampled, row->std_F_sampled, dt);

    quantum_state_free(&st);
}

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1] : "xeb.json";
    printf("=== XEB benchmark (Porter-Thomas, noiseless statevector) ===\n");
    printf("  schema: moonlab/xeb_v1  out: %s\n\n", out_path);

    typedef struct {
        size_t n; int layers; size_t n_circuits; size_t n_shots;
    } cfg_t;
    const cfg_t configs[] = {
        {  6, 10, 200,  4096 },
        {  8, 10, 200,  4096 },
        { 10, 10, 100,  4096 },
        { 12, 10,  50,  4096 },
        { 14, 10,  20,  4096 },
    };
    const size_t n_cfg = sizeof(configs) / sizeof(configs[0]);

    xeb_row_t rows[8];
    for (size_t ci = 0; ci < n_cfg; ci++) {
        bench_one(&rows[ci], configs[ci].n, configs[ci].layers,
                  configs[ci].n_circuits, configs[ci].n_shots);
    }

    FILE* f = fopen(out_path, "w");
    if (!f) { fprintf(stderr, "fopen(%s) failed\n", out_path); return 1; }
    fprintf(f, "{\n");
    fprintf(f, "  \"schema\": \"moonlab/xeb_v1\",\n");
    fprintf(f, "  \"description\": \"Cross-entropy benchmark on the "
               "noiseless state-vector engine.  Random hardware-efficient "
               "circuits at L layers; F_xeb = D * E[p(s)] - 1; "
               "Porter-Thomas predicts F_xeb -> 1 in expectation.  "
               "F_exact computes E[p(s)] directly from the statevector; "
               "F_sampled draws n_shots bitstrings and averages.  "
               "Mean and stddev over n_circuits independent circuits.\",\n");
    fprintf(f, "  \"rows\": [");
    for (size_t i = 0; i < n_cfg; i++) {
        fprintf(f, "%s\n    {\"n\": %zu, \"layers\": %d, "
                   "\"n_circuits\": %zu, \"n_shots\": %zu, "
                   "\"mean_F_exact\": %.6f, \"std_F_exact\": %.6f, "
                   "\"mean_F_sampled\": %.6f, \"std_F_sampled\": %.6f, "
                   "\"wall_s\": %.4f}",
                i == 0 ? "" : ",",
                rows[i].n, rows[i].layers,
                rows[i].n_circuits, rows[i].n_shots,
                rows[i].mean_F_exact, rows[i].std_F_exact,
                rows[i].mean_F_sampled, rows[i].std_F_sampled,
                rows[i].wall_s);
    }
    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    printf("\nwrote %s\n", out_path);
    return 0;
}
