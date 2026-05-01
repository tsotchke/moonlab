/**
 * @file bell_chsh_aggregate.c
 * @brief Multi-run CHSH test with median + IQR aggregation.
 *
 * Closes the moonlab paper §4.2 todo (line ~450, "Aggregate over >= 5
 * independent runs to report a median + IQR rather than a single
 * sampled value").
 *
 * Runs bell_test_chsh on a freshly-prepared |Phi+> Bell state N times,
 * each run drawing from the same hardware-entropy-seeded measurement
 * sampler.  Reports:
 *   - per-run CHSH parameter
 *   - median, IQR (Q1..Q3), min, max
 *   - mean and standard deviation
 *   - violation rate (fraction of runs with CHSH > 2)
 *   - quantum-confirmation rate (fraction within 2-sigma of 2 sqrt(2))
 *
 * Single-host reproducibility: the per-run CHSH value depends on the
 * underlying hardware-entropy stream, so values vary across runs of
 * this example -- which is the point.  Statistical aggregates are
 * stable to the standard-error band 2/sqrt(N_shots).
 *
 * Output: JSON with schema "moonlab/bell_chsh_aggregate_v1".
 *
 * Usage: ./bell_chsh_aggregate [out.json] [n_runs] [shots_per_run]
 *   defaults: out.json -> ./bell_chsh_aggregate.json, n_runs=10, shots=10000
 */

#include "../../src/quantum/state.h"
#include "../../src/algorithms/bell_tests.h"
#include "../../src/utils/quantum_entropy.h"
#include "../../src/applications/hardware_entropy.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int dcmp(const void* a, const void* b) {
    const double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

/* Linear-interpolation quantile on a sorted array (Type 7, R / Excel). */
static double quantile(const double* sorted, size_t n, double q) {
    if (n == 0) return 0.0 / 0.0;
    if (n == 1) return sorted[0];
    const double idx = q * (double)(n - 1);
    const size_t lo = (size_t)floor(idx);
    const size_t hi = (size_t)ceil(idx);
    if (lo == hi) return sorted[lo];
    const double w = idx - (double)lo;
    return (1.0 - w) * sorted[lo] + w * sorted[hi];
}

int main(int argc, char** argv) {
    const char* out_path = (argc >= 2) ? argv[1] : "bell_chsh_aggregate.json";
    const size_t n_runs = (argc >= 3) ? (size_t)strtoul(argv[2], NULL, 10) : 10;
    const size_t shots  = (argc >= 4) ? (size_t)strtoul(argv[3], NULL, 10) : 10000;
    if (n_runs < 5) {
        fprintf(stderr, "Need at least 5 runs for IQR (got %zu); using 5.\n", n_runs);
    }
    const size_t N = n_runs < 5 ? 5 : n_runs;

    fprintf(stdout,
            "=== Bell-CHSH multi-run aggregate ===\n"
            "  schema: moonlab/bell_chsh_aggregate_v1\n"
            "  runs: %zu  shots/run: %zu  out: %s\n\n",
            N, shots, out_path);

    /* Single hardware-entropy context shared across runs.  Each
     * bell_test_chsh call consumes its own bytes from the stream, so
     * runs are independent. */
    entropy_ctx_t hw_ctx;
    if (entropy_init(&hw_ctx) != ENTROPY_SUCCESS) {
        fprintf(stderr, "entropy_init failed\n");
        return 1;
    }
    quantum_entropy_ctx_t entropy;
    quantum_entropy_init(&entropy,
        (quantum_entropy_fn)entropy_get_bytes, &hw_ctx);

    double* values = (double*)calloc(N, sizeof(double));
    int*    violates_classical = (int*)calloc(N, sizeof(int));
    int*    confirms_quantum   = (int*)calloc(N, sizeof(int));
    double* p_values           = (double*)calloc(N, sizeof(double));
    double* std_errors         = (double*)calloc(N, sizeof(double));

    for (size_t r = 0; r < N; r++) {
        quantum_state_t st;
        if (quantum_state_init(&st, 2) != QS_SUCCESS) {
            fprintf(stderr, "quantum_state_init failed\n");
            return 1;
        }
        if (create_bell_state_phi_plus(&st, 0, 1) != QS_SUCCESS) {
            fprintf(stderr, "create_bell_state_phi_plus failed\n");
            return 1;
        }
        bell_test_result_t res = bell_test_chsh(
            &st, /*A*/ 0, /*B*/ 1, shots, /*settings*/ NULL, &entropy);
        values[r]              = res.chsh_value;
        violates_classical[r]  = res.violates_classical;
        confirms_quantum[r]    = res.confirms_quantum;
        p_values[r]            = res.p_value;
        std_errors[r]          = res.standard_error;
        fprintf(stdout, "  run %2zu: CHSH = %.6f  std_err = %.4f  "
                        "violates_classical=%d  confirms_quantum=%d\n",
                r + 1, values[r], std_errors[r],
                violates_classical[r], confirms_quantum[r]);
        quantum_state_free(&st);
    }

    /* Sort a copy for quantiles. */
    double* sorted = (double*)calloc(N, sizeof(double));
    memcpy(sorted, values, N * sizeof(double));
    qsort(sorted, N, sizeof(double), dcmp);

    const double median = quantile(sorted, N, 0.5);
    const double q1     = quantile(sorted, N, 0.25);
    const double q3     = quantile(sorted, N, 0.75);
    const double iqr    = q3 - q1;
    const double minv   = sorted[0];
    const double maxv   = sorted[N - 1];

    double sum = 0.0, sum2 = 0.0;
    size_t n_violate = 0, n_confirm = 0;
    for (size_t r = 0; r < N; r++) {
        sum  += values[r];
        sum2 += values[r] * values[r];
        n_violate += violates_classical[r] != 0;
        n_confirm += confirms_quantum[r]   != 0;
    }
    const double mean = sum / (double)N;
    const double var  = (sum2 - (double)N * mean * mean) / (double)(N > 1 ? N - 1 : 1);
    const double sdev = sqrt(var > 0.0 ? var : 0.0);
    const double sem  = sdev / sqrt((double)N);

    const double tsirelson = 2.0 * sqrt(2.0);

    fprintf(stdout, "\n  --- aggregate ---\n");
    fprintf(stdout, "  median:           %.6f\n", median);
    fprintf(stdout, "  Q1 .. Q3:         %.6f .. %.6f  (IQR = %.6f)\n", q1, q3, iqr);
    fprintf(stdout, "  min .. max:       %.6f .. %.6f\n", minv, maxv);
    fprintf(stdout, "  mean +/- sdev:    %.6f +/- %.6f\n", mean, sdev);
    fprintf(stdout, "  std err of mean:  %.6f\n", sem);
    fprintf(stdout, "  Tsirelson bound:  %.6f\n", tsirelson);
    fprintf(stdout, "  classical bound:  2.000000\n");
    fprintf(stdout, "  violates classical: %zu / %zu  (%.1f%%)\n",
            n_violate, N, 100.0 * (double)n_violate / (double)N);
    fprintf(stdout, "  confirms quantum:   %zu / %zu  (%.1f%%)\n",
            n_confirm, N, 100.0 * (double)n_confirm / (double)N);

    /* JSON output. */
    FILE* f = fopen(out_path, "w");
    if (!f) {
        fprintf(stderr, "fopen(%s) failed\n", out_path);
        return 1;
    }
    fprintf(f, "{\n");
    fprintf(f, "  \"schema\": \"moonlab/bell_chsh_aggregate_v1\",\n");
    fprintf(f, "  \"params\": {\"n_runs\": %zu, \"shots_per_run\": %zu, "
                "\"bell_state\": \"phi_plus\", \"angles\": \"optimal\"},\n",
            N, shots);
    fprintf(f, "  \"per_run\": [");
    for (size_t r = 0; r < N; r++) {
        fprintf(f, "%s\n    {\"run\": %zu, \"chsh\": %.10g, \"std_err\": %.6g, "
                   "\"p_value\": %.6g, \"violates_classical\": %d, "
                   "\"confirms_quantum\": %d}",
                r == 0 ? "" : ",", r + 1, values[r], std_errors[r],
                p_values[r], violates_classical[r], confirms_quantum[r]);
    }
    fprintf(f, "\n  ],\n");
    fprintf(f, "  \"aggregate\": {\n");
    fprintf(f, "    \"median\": %.10g,\n", median);
    fprintf(f, "    \"q1\": %.10g,\n", q1);
    fprintf(f, "    \"q3\": %.10g,\n", q3);
    fprintf(f, "    \"iqr\": %.10g,\n", iqr);
    fprintf(f, "    \"min\": %.10g,\n", minv);
    fprintf(f, "    \"max\": %.10g,\n", maxv);
    fprintf(f, "    \"mean\": %.10g,\n", mean);
    fprintf(f, "    \"stddev\": %.10g,\n", sdev);
    fprintf(f, "    \"sem\": %.10g,\n", sem);
    fprintf(f, "    \"tsirelson_bound\": %.10g,\n", tsirelson);
    fprintf(f, "    \"classical_bound\": 2.0,\n");
    fprintf(f, "    \"violates_classical_rate\": %.6g,\n",
            (double)n_violate / (double)N);
    fprintf(f, "    \"confirms_quantum_rate\": %.6g\n",
            (double)n_confirm / (double)N);
    fprintf(f, "  }\n}\n");
    fclose(f);
    fprintf(stdout, "\nwrote %s\n", out_path);

    free(values); free(violates_classical); free(confirms_quantum);
    free(p_values); free(std_errors); free(sorted);
    return 0;
}
