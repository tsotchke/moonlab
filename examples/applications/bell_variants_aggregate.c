/**
 * @file bell_variants_aggregate.c
 * @brief Multi-variant Bell-inequality harness: CHSH + Mermin + Mermin-Klyshko.
 *
 * Closes the moonlab paper §4.2 follow-on ("Mermin and Mermin-Klyshko
 * N-party Bell tests are exposed under the same harness for N-qubit
 * GHZ states") -- the existing bell_chsh_aggregate.c only drives the
 * 2-qubit CHSH variant.
 *
 * For each variant, runs N independent measurement-rich passes on a
 * freshly-prepared maximally-entangled state and reports per-run
 * value plus median + IQR + min/max + violation rates.
 *
 * Variants:
 *   - CHSH        on |Phi+>           (2 qubits) — classical bound 2,
 *                                                quantum bound 2 sqrt(2)
 *                                                ≈ 2.828.
 *   - Mermin-3    on 3-qubit GHZ      (M = <XYY>+<YXY>+<YYX>-<XXX>):
 *                                                classical 2, quantum 4.
 *   - Mermin-Klyshko on 4q GHZ        (normalised to classical bound 1,
 *                                                quantum max 2^((N-1)/2)
 *                                                = 2 sqrt(2) ≈ 2.828).
 *   - Mermin-Klyshko on 5q GHZ        (quantum max 4).
 *
 * Output: JSON, schema "moonlab/bell_variants_aggregate_v1".
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
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

/* Prepare an N-qubit GHZ state |0...0> + |1...1> (unnormalised
 * intent; the H + CNOT-chain produces the normalised version). */
static qs_error_t prepare_ghz(quantum_state_t* st) {
    qs_error_t rc;
    rc = gate_hadamard(st, 0); if (rc != QS_SUCCESS) return rc;
    for (size_t q = 0; q + 1 < st->num_qubits; q++) {
        rc = gate_cnot(st, (int)q, (int)(q + 1));
        if (rc != QS_SUCCESS) return rc;
    }
    return QS_SUCCESS;
}

typedef struct {
    const char* label;
    int   n_qubits;
    double classical_bound;
    double quantum_bound;
    double* values;
    size_t n_runs;
} variant_stats_t;

static void summarise_variant(FILE* json, int* first_var,
                                const variant_stats_t* v) {
    double* sorted = (double*)calloc(v->n_runs, sizeof(double));
    memcpy(sorted, v->values, v->n_runs * sizeof(double));
    qsort(sorted, v->n_runs, sizeof(double), dcmp);
    const double med = quantile(sorted, v->n_runs, 0.5);
    const double q1  = quantile(sorted, v->n_runs, 0.25);
    const double q3  = quantile(sorted, v->n_runs, 0.75);
    double sum = 0, sum2 = 0;
    int n_violate_classical = 0, n_within_quantum = 0;
    for (size_t r = 0; r < v->n_runs; r++) {
        sum  += v->values[r];
        sum2 += v->values[r] * v->values[r];
        if (fabs(v->values[r]) > v->classical_bound) n_violate_classical++;
        /* Within the 2-sigma band of the quantum bound at 1/sqrt(N_shots).
         * For Mermin and Mermin-Klyshko we don't track per-run shot SE
         * here, so just report the absolute distance to bound. */
        if (fabs(v->values[r] - v->quantum_bound) < 0.1 * v->quantum_bound)
            n_within_quantum++;
    }
    const double mean = sum / (double)v->n_runs;
    const double var  = sum2 / (double)v->n_runs - mean * mean;
    const double sd   = sqrt(var > 0 ? var : 0);

    fprintf(stdout,
            "  %-20s: median %.6f  IQR %.4f  [min %.4f, max %.4f]  "
            "mean %.4f +/- %.4f  violate=%d/%zu\n",
            v->label, med, q3 - q1, sorted[0], sorted[v->n_runs - 1],
            mean, sd, n_violate_classical, v->n_runs);

    fprintf(json, "%s\n    {\"variant\": \"%s\", \"n_qubits\": %d, "
                  "\"classical_bound\": %.6g, \"quantum_bound\": %.6g, "
                  "\"n_runs\": %zu, "
                  "\"values\": [",
            (*first_var) ? "" : ",", v->label, v->n_qubits,
            v->classical_bound, v->quantum_bound, v->n_runs);
    for (size_t r = 0; r < v->n_runs; r++) {
        fprintf(json, "%s%.10g", r == 0 ? "" : ", ", v->values[r]);
    }
    fprintf(json, "],"
                  " \"median\": %.10g, \"q1\": %.10g, \"q3\": %.10g, "
                  "\"min\": %.10g, \"max\": %.10g, "
                  "\"mean\": %.10g, \"stddev\": %.10g, "
                  "\"violation_rate\": %.4g, \"quantum_within_10pct_rate\": %.4g}",
            med, q1, q3, sorted[0], sorted[v->n_runs - 1],
            mean, sd,
            (double)n_violate_classical / (double)v->n_runs,
            (double)n_within_quantum    / (double)v->n_runs);
    *first_var = 0;
    free(sorted);
}

int main(int argc, char** argv) {
    const char*  out_path = (argc >= 2) ? argv[1] : "bell_variants_aggregate.json";
    const size_t n_runs   = (argc >= 3) ? (size_t)strtoul(argv[2], NULL, 10) : 10;
    const size_t shots    = (argc >= 4) ? (size_t)strtoul(argv[3], NULL, 10) : 10000;
    const size_t N        = (n_runs < 5) ? 5 : n_runs;

    fprintf(stdout,
            "=== Bell-variants multi-run aggregate ===\n"
            "  schema: moonlab/bell_variants_aggregate_v1\n"
            "  runs/variant: %zu  shots/run: %zu  out: %s\n\n",
            N, shots, out_path);

    entropy_ctx_t hw_ctx;
    if (entropy_init(&hw_ctx) != ENTROPY_SUCCESS) {
        fprintf(stderr, "entropy_init failed\n"); return 1;
    }
    quantum_entropy_ctx_t entropy;
    quantum_entropy_init(&entropy,
        (quantum_entropy_fn)entropy_get_bytes, &hw_ctx);

    /* Allocate per-variant value buffers. */
    double* chsh_vals    = (double*)calloc(N, sizeof(double));
    double* mermin3_vals = (double*)calloc(N, sizeof(double));
    double* mk4_vals     = (double*)calloc(N, sizeof(double));
    double* mk5_vals     = (double*)calloc(N, sizeof(double));

    for (size_t r = 0; r < N; r++) {
        /* CHSH on |Phi+> (2 qubits). */
        {
            quantum_state_t st;
            quantum_state_init(&st, 2);
            create_bell_state_phi_plus(&st, 0, 1);
            bell_test_result_t res =
                bell_test_chsh(&st, 0, 1, shots, NULL, &entropy);
            chsh_vals[r] = res.chsh_value;
            quantum_state_free(&st);
        }
        /* Mermin-3 on 3-qubit GHZ. */
        {
            quantum_state_t st;
            quantum_state_init(&st, 3);
            prepare_ghz(&st);
            bell_test_result_t res =
                bell_test_mermin_ghz(&st, 0, 1, 2, shots, &entropy);
            mermin3_vals[r] = res.chsh_value;
            quantum_state_free(&st);
        }
        /* Mermin-Klyshko at N=4 and N=5. */
        for (size_t Nq = 4; Nq <= 5; Nq++) {
            quantum_state_t st;
            quantum_state_init(&st, Nq);
            prepare_ghz(&st);
            double mk = bell_test_mermin_klyshko(&st, Nq, shots, &entropy);
            if (Nq == 4) mk4_vals[r] = mk;
            else         mk5_vals[r] = mk;
            quantum_state_free(&st);
        }
    }

    /* Open JSON. */
    FILE* json = fopen(out_path, "w");
    if (!json) { fprintf(stderr, "fopen %s failed\n", out_path); return 1; }
    fprintf(json, "{\n");
    fprintf(json, "  \"schema\": \"moonlab/bell_variants_aggregate_v1\",\n");
    fprintf(json, "  \"description\": \"Multi-run aggregated Bell-inequality "
                  "violations: CHSH on |Phi+>, Mermin-3 on 3-qubit GHZ, "
                  "Mermin-Klyshko N-qubit on 4q and 5q GHZ.  Each variant "
                  "runs %zu independent passes; reports per-run value "
                  "plus median, IQR, min/max, mean, stddev, classical-"
                  "violation rate, and within-10%% of quantum-bound rate.\",\n",
            N);
    fprintf(json, "  \"runs_per_variant\": %zu,\n", N);
    fprintf(json, "  \"shots_per_run\": %zu,\n", shots);
    fprintf(json, "  \"variants\": [");

    int first_var = 1;
    variant_stats_t v_chsh    = { "CHSH (|Phi+>)",  2,
                                  2.0,             2.0 * sqrt(2.0),
                                  chsh_vals,       N };
    variant_stats_t v_mermin3 = { "Mermin-3 (GHZ)", 3,
                                  2.0,             4.0,
                                  mermin3_vals,    N };
    variant_stats_t v_mk4     = { "Mermin-Klyshko-4 (GHZ)", 4,
                                  1.0,             pow(2.0, (4 - 1) * 0.5),
                                  mk4_vals,        N };
    variant_stats_t v_mk5     = { "Mermin-Klyshko-5 (GHZ)", 5,
                                  1.0,             pow(2.0, (5 - 1) * 0.5),
                                  mk5_vals,        N };
    summarise_variant(json, &first_var, &v_chsh);
    summarise_variant(json, &first_var, &v_mermin3);
    summarise_variant(json, &first_var, &v_mk4);
    summarise_variant(json, &first_var, &v_mk5);

    fprintf(json, "\n  ]\n}\n");
    fclose(json);
    fprintf(stdout, "\nwrote %s\n", out_path);

    free(chsh_vals); free(mermin3_vals); free(mk4_vals); free(mk5_vals);
    return 0;
}
