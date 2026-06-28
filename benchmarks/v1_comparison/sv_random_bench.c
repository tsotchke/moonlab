/**
 * @file sv_random_bench.c
 * @brief v1.0 head-to-head: Moonlab state-vector vs Qiskit-Aer on random
 *        2-qubit-block Haar circuits at N = 20..30.
 *
 * Circuit family ("brick-wall random"):
 *   For each of depth = 50 layers:
 *     1. Build a random permutation of {0, ..., N-1} by Fisher-Yates.
 *     2. Pair adjacent entries: (perm[0], perm[1]), (perm[2], perm[3]), ...
 *     3. For each pair, apply a Haar-random U(4) generated via the
 *        Mezzadri 2007 QR construction (same routine as
 *        src/applications/quantum_volume.c).
 *
 * After the circuit:
 *   Draw 1024 computational-basis shots via measurement_sample (Born-rule
 *   sampling, no collapse).  Wall-clock and peak RSS are measured around
 *   the entire (circuit + sampling) phase.
 *
 * Sweep:  N in {20, 22, 24, 26, 28, 30}, depth = 50, shots = 1024.
 *
 * Output: JSON, schema "moonlab/v1_comparison/sv_random".  Records carry
 * { N, depth, shots, wall_clock_s, circuit_s, sampling_s, peak_rss_bytes,
 *   state_dim_bytes_expected }.  The competitor (Qiskit-Aer) side runs
 * separately; see docs/benchmarks/v1_comparison.md for the exact
 * `python bench/qiskit_aer_random.py` invocation.
 *
 * Memory budget (16 B per complex amplitude):
 *     N = 20  ->  16 MiB
 *     N = 24  -> 256 MiB
 *     N = 28  ->   4 GiB
 *     N = 30  ->  16 GiB
 * The harness will refuse to allocate N > MOONLAB_MAX_QUBITS, and the
 * caller is responsible for ensuring the host has enough RAM.
 *
 * @since v1.0
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/quantum/measurement.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static uint64_t peak_rss_bytes(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
    uint64_t r = (uint64_t)ru.ru_maxrss;
#if defined(__APPLE__)
    return r;
#else
    return r * 1024ULL;
#endif
}

/* xorshift64* PRNG, identical to src/applications/quantum_volume.c. */
static uint64_t rng_next(uint64_t* s) {
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 2685821657736338717ULL;
}

static double rng_uniform(uint64_t* s) {
    return (rng_next(s) >> 11) * (1.0 / 9007199254740992.0);
}

static double rng_gauss(uint64_t* s) {
    double u = rng_uniform(s);
    if (u < 1e-300) u = 1e-300;
    double v = rng_uniform(s);
    return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
}

/* Haar-random U(4) via Mezzadri 2007: Z = X + iY iid N(0, 1/2), QR
 * factorise via modified Gram-Schmidt, then divide each Q column by the
 * phase of the corresponding R diagonal entry so Q is Haar-distributed. */
static void haar_u4(uint64_t* rng, complex_t U[4][4]) {
    complex_t Z[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            double r  = rng_gauss(rng) / M_SQRT2;
            double im = rng_gauss(rng) / M_SQRT2;
            Z[i][j] = r + _Complex_I * im;
        }
    }
    complex_t Q[4][4];
    complex_t R[4][4] = {0};
    for (int j = 0; j < 4; j++) {
        complex_t v[4];
        for (int i = 0; i < 4; i++) v[i] = Z[i][j];
        for (int k = 0; k < j; k++) {
            complex_t dot = 0;
            for (int i = 0; i < 4; i++) dot += conj(Q[i][k]) * v[i];
            R[k][j] = dot;
            for (int i = 0; i < 4; i++) v[i] -= dot * Q[i][k];
        }
        double nrm = 0;
        for (int i = 0; i < 4; i++) {
            nrm += creal(v[i]) * creal(v[i]) + cimag(v[i]) * cimag(v[i]);
        }
        nrm = sqrt(nrm);
        if (nrm < 1e-300) nrm = 1e-300;
        R[j][j] = nrm;
        for (int i = 0; i < 4; i++) Q[i][j] = v[i] / nrm;
    }
    for (int j = 0; j < 4; j++) {
        double mag = cabs(R[j][j]);
        complex_t ph = (mag < 1e-300) ? 1.0 : R[j][j] / mag;
        for (int i = 0; i < 4; i++) Q[i][j] /= ph;
    }
    memcpy(U, Q, sizeof(Q));
}

/* Run one (N, depth, shots) benchmark point.  Returns 0 on success, -1
 * if state-allocation failed (out of memory or N over compile-time cap),
 * filling *out_circuit_s, *out_sampling_s, *out_peak_rss_bytes. */
static int run_point(int N, int depth, int shots, uint64_t seed,
                     double* out_circuit_s, double* out_sampling_s,
                     uint64_t* out_peak_rss_bytes) {
    if (N < 2 || N > MOONLAB_MAX_QUBITS) return -1;

    quantum_state_t* st = quantum_state_create(N);
    if (!st) return -1;

    uint64_t rng = seed ? seed : 0xC0FFEEULL;

    int* perm = (int*)malloc((size_t)N * sizeof(int));
    if (!perm) { quantum_state_destroy(st); return -1; }

    int pairs = N / 2;
    double t0 = now_s();
    for (int layer = 0; layer < depth; layer++) {
        for (int i = 0; i < N; i++) perm[i] = i;
        /* Fisher-Yates. */
        for (int i = N - 1; i > 0; i--) {
            int j = (int)(rng_next(&rng) % (uint64_t)(i + 1));
            int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
        }
        for (int p = 0; p < pairs; p++) {
            int a = perm[2 * p];
            int b = perm[2 * p + 1];
            complex_t U[4][4];
            haar_u4(&rng, U);
            if (apply_two_qubit_gate(st, a, b, U) != QS_SUCCESS) {
                free(perm); quantum_state_destroy(st);
                return -1;
            }
        }
    }
    double t1 = now_s();

    /* Born-rule sampling without collapse: caller supplies one uniform
     * per shot in [0, 1). */
    double*   rands = (double*)  malloc((size_t)shots * sizeof(double));
    uint64_t* shotv = (uint64_t*)malloc((size_t)shots * sizeof(uint64_t));
    if (!rands || !shotv) {
        free(rands); free(shotv); free(perm); quantum_state_destroy(st);
        return -1;
    }
    for (int i = 0; i < shots; i++) rands[i] = rng_uniform(&rng);
    measurement_sample(st, shotv, shots, rands);
    double t2 = now_s();

    *out_circuit_s        = t1 - t0;
    *out_sampling_s       = t2 - t1;
    *out_peak_rss_bytes   = peak_rss_bytes();

    free(rands); free(shotv); free(perm);
    quantum_state_destroy(st);
    return 0;
}

int main(int argc, char** argv) {
    const char* out_path = "sv_random.json";
    int n_qubits_list[16] = {20, 22, 24, 26, 28, 30};
    int n_n = 6;
    int depth = 50;
    int shots = 1024;
    uint64_t seed = 0xABBA1234ULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) {
            depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--shots") == 0 && i + 1 < argc) {
            shots = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 0);
        } else if (strcmp(argv[i], "--qubits") == 0 && i + 1 < argc) {
            /* comma-separated list */
            const char* p = argv[++i];
            int nn = 0;
            while (*p && nn < 16) {
                char* end = NULL;
                long v = strtol(p, &end, 10);
                if (end == p) break;
                if (v < 2 || v > MOONLAB_MAX_QUBITS) {
                    fprintf(stderr, "qubits must be in [2, %d]; got %ld\n",
                            MOONLAB_MAX_QUBITS, v);
                    return 2;
                }
                n_qubits_list[nn++] = (int)v;
                p = end;
                if (*p == ',') p++;
            }
            n_n = nn;
        } else if (argv[i][0] != '-') {
            out_path = argv[i];
        } else {
            fprintf(stderr,
                    "usage: %s [out.json] [--qubits 20,22,24,26,28,30] "
                    "[--depth 50] [--shots 1024] [--seed N]\n", argv[0]);
            return 2;
        }
    }
    if (depth < 1 || shots < 1) {
        fprintf(stderr, "depth and shots must be positive\n"); return 2;
    }

    fprintf(stdout,
            "=== v1.0 head-to-head: state-vector vs Qiskit-Aer ===\n"
            "    schema: moonlab/v1_comparison/sv_random\n"
            "    circuit: brick-wall Haar-random U(4), depth %d, %d shots\n\n",
            depth, shots);

    FILE* f = fopen(out_path, "w");
    if (!f) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }
    fprintf(f, "{\n");
    fprintf(f, "  \"schema\": \"moonlab/v1_comparison/sv_random\",\n");
    fprintf(f, "  \"circuit\": \"brick-wall: per layer, random pairing of "
               "qubits, apply Haar U(4) to each pair (Mezzadri 2007)\",\n");
    fprintf(f, "  \"depth\": %d,\n", depth);
    fprintf(f, "  \"shots\": %d,\n", shots);
    fprintf(f, "  \"seed\": %llu,\n", (unsigned long long)seed);
    fprintf(f, "  \"runs\": [");

    fprintf(stdout, "  %-4s %-10s %-12s %-12s %-12s %-14s\n",
            "N", "wall_s", "circuit_s", "sampling_s", "peak_rss_MB",
            "expected_amp_MB");

    int first = 1;
    for (int i = 0; i < n_n; i++) {
        int N = n_qubits_list[i];
        double cs = 0, ss = 0; uint64_t rss = 0;
        double t0 = now_s();
        int rc = run_point(N, depth, shots, seed + (uint64_t)N,
                            &cs, &ss, &rss);
        double dt = now_s() - t0;
        double exp_mb = (double)((size_t)1 << N) * 16.0 / (1024.0 * 1024.0);
        if (rc != 0) {
            fprintf(stdout, "  %-4d FAILED  (allocation or N>cap)\n", N);
            fprintf(f, "%s\n    {\"N\": %d, \"status\": \"alloc_failed\", "
                       "\"expected_state_dim_bytes\": %.0f}",
                    first ? "" : ",", N, exp_mb * 1024.0 * 1024.0);
            first = 0;
            continue;
        }
        fprintf(stdout, "  %-4d %-10.3f %-12.3f %-12.3f %-12.2f %-14.2f\n",
                N, dt, cs, ss, (double)rss / (1024.0 * 1024.0), exp_mb);
        fprintf(f, "%s\n    {\"N\": %d, \"depth\": %d, \"shots\": %d, "
                   "\"wall_clock_s\": %.4f, \"circuit_s\": %.4f, "
                   "\"sampling_s\": %.4f, \"peak_rss_bytes\": %llu, "
                   "\"expected_state_dim_bytes\": %.0f}",
                first ? "" : ",", N, depth, shots, dt, cs, ss,
                (unsigned long long)rss, exp_mb * 1024.0 * 1024.0);
        first = 0;
    }

    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    fprintf(stdout, "\nwrote %s\n", out_path);
    return 0;
}
