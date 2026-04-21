/**
 * @file test_zne.c
 * @brief Zero-noise extrapolation estimators against known signals.
 *
 * Checks:
 *   - Linear fit recovers the intercept of an exact line to 1e-12.
 *   - Richardson interpolation is exact on any polynomial of deg <= n-1.
 *   - Exponential fit recovers E = a + b*exp(-c*x) from synthetic data.
 *   - Integration: zne_mitigate runs a user callback that simulates
 *     a depolarizing channel on <Z> and recovers the noiseless value.
 */

#include "../../src/mitigation/zne.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                                  \
    if (!(cond)) {                                                  \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);        \
        failures++;                                                 \
    } else {                                                        \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);        \
    }                                                               \
} while (0)

static void test_linear_exact(void) {
    fprintf(stdout, "\n-- ZNE linear: exact line y = 3 + 2x --\n");
    double x[4] = { 1.0, 1.5, 2.0, 3.0 };
    double y[4];
    for (int i = 0; i < 4; i++) y[i] = 3.0 + 2.0 * x[i];
    double sd = 0.0;
    double intercept = zne_extrapolate(x, y, 4, ZNE_LINEAR, &sd);
    fprintf(stdout, "    intercept = %.12f  residual sd = %.2e\n",
            intercept, sd);
    CHECK(fabs(intercept - 3.0) < 1e-12, "intercept recovered to 1e-12");
    CHECK(sd < 1e-12, "residual stddev ~ 0");
}

static void test_richardson_polynomial_exact(void) {
    fprintf(stdout, "\n-- ZNE Richardson: exact cubic --\n");
    /* y(x) = 1 - 0.5 x + 0.2 x^2 - 0.05 x^3; y(0) = 1. */
    double x[4] = { 1.0, 1.5, 2.0, 3.0 };
    double y[4];
    for (int i = 0; i < 4; i++) {
        double xi = x[i];
        y[i] = 1.0 - 0.5 * xi + 0.2 * xi * xi - 0.05 * xi * xi * xi;
    }
    double intercept = zne_extrapolate(x, y, 4, ZNE_RICHARDSON, NULL);
    fprintf(stdout, "    intercept = %.12f  (expect 1.0)\n", intercept);
    CHECK(fabs(intercept - 1.0) < 1e-10,
          "Richardson recovers cubic intercept (got %.6e)",
          fabs(intercept - 1.0));
}

static void test_exponential_fit(void) {
    fprintf(stdout, "\n-- ZNE exponential: y = 0.9 + 0.1 * exp(-0.7 x) --\n");
    const double a = 0.9, b = 0.1, c = 0.7;
    double x[5] = { 1.0, 1.5, 2.0, 2.5, 3.0 };
    double y[5];
    for (int i = 0; i < 5; i++) y[i] = a + b * exp(-c * x[i]);
    double sd = 0.0;
    double E0 = zne_extrapolate(x, y, 5, ZNE_EXPONENTIAL, &sd);
    fprintf(stdout, "    E(0) = %.8f  (expect a + b = 1.0)  residual = %.2e\n",
            E0, sd);
    CHECK(fabs(E0 - (a + b)) < 1e-6, "exponential fit recovers E(0) = 1");
}

/* --- Integration: zne_mitigate on a real depolarized <Z> signal --- */
typedef struct {
    double p;    /* physical depolarization rate per "noise unit" */
} depolar_ctx_t;

/* Depolarized <Z_0> on |1> (ideal = -1) scales as -1 * (1 - p)^lambda
 * for N-fold application.  Non-linear in lambda -> Richardson / exp
 * should do better than linear for large lambda spans. */
static double depolarized_z(double lambda, void *u) {
    depolar_ctx_t *ctx = (depolar_ctx_t*)u;
    return -pow(1.0 - ctx->p, lambda);
}

static void test_integration_depolar(void) {
    fprintf(stdout, "\n-- ZNE integration: depolarized <Z> on |1> --\n");
    depolar_ctx_t ctx = { .p = 0.08 };  /* 8% per noise unit */
    double scales[5] = { 1.0, 1.5, 2.0, 2.5, 3.0 };

    double sd_lin = 0, sd_rich = 0, sd_exp = 0;
    double lin  = zne_mitigate(depolarized_z, &ctx, scales, 5,
                                ZNE_LINEAR, &sd_lin);
    double rich = zne_mitigate(depolarized_z, &ctx, scales, 5,
                                ZNE_RICHARDSON, &sd_rich);
    double expn = zne_mitigate(depolarized_z, &ctx, scales, 5,
                                ZNE_EXPONENTIAL, &sd_exp);

    fprintf(stdout, "    linear   : E(0) = %+.6f  sd = %.2e\n", lin,  sd_lin);
    fprintf(stdout, "    Richardson: E(0) = %+.6f  sd = %.2e\n", rich, sd_rich);
    fprintf(stdout, "    exponential: E(0) = %+.6f  sd = %.2e\n", expn, sd_exp);
    fprintf(stdout, "    noiseless : E(0) = -1.000000\n");

    /* All three should improve vs the noisy E(lambda=1) = -(1-p) = -0.92
     * baseline; Richardson and exponential should be near-exact. */
    CHECK(fabs(rich - (-1.0)) < 1e-2,
          "Richardson within 1e-2 of noiseless (got %.4f)", rich);
    CHECK(fabs(expn - (-1.0)) < 1e-6,
          "exponential within 1e-6 of noiseless (got %.6f)", expn);
    /* Noisy baseline at lambda=1: |E| = 1 - p = 0.92. All mitigators
     * should be closer to -1 than that. */
    CHECK(fabs(lin + 1.0) < (1.0 - 0.92),
          "linear mitigation improves over noisy baseline");
}

/* ---------------------------------------------------------------- */
/* PEC tests                                                         */
/* ---------------------------------------------------------------- */

static void test_pec_one_norm_cost(void) {
    fprintf(stdout, "\n-- PEC one-norm cost --\n");
    double etas[3] = { 1.2, -0.5, 0.3 };
    pec_quasi_prob_t qp = { .num_terms = 3, .etas = etas };
    double gamma = pec_one_norm_cost(&qp);
    fprintf(stdout, "    gamma = %.6f (expect 2.0)\n", gamma);
    CHECK(fabs(gamma - 2.0) < 1e-12, "one-norm cost matches");
}

static void test_pec_sampling_signs(void) {
    fprintf(stdout, "\n-- PEC sampling returns correct sign --\n");
    double etas[3] = { 0.5, -0.5, 0.5 };  /* gamma = 1.5 */
    pec_quasi_prob_t qp = { .num_terms = 3, .etas = etas };
    size_t idx = 0;
    /* u < 1/3 -> pick idx 0, sign +1. */
    double s = pec_sample_index(&qp, 0.1, &idx);
    CHECK(idx == 0 && s > 0, "u=0.1 -> idx 0, sign +");
    /* 1/3 <= u < 2/3 -> pick idx 1, sign -. */
    s = pec_sample_index(&qp, 0.5, &idx);
    CHECK(idx == 1 && s < 0, "u=0.5 -> idx 1, sign -");
    /* u >= 2/3 -> pick idx 2, sign +. */
    s = pec_sample_index(&qp, 0.9, &idx);
    CHECK(idx == 2 && s > 0, "u=0.9 -> idx 2, sign +");
}

static void test_pec_aggregate_recovers_mean(void) {
    fprintf(stdout, "\n-- PEC aggregate recovers gamma * E[sgn * m] --\n");
    const size_t n = 8;
    double signs[8] = { +1, +1, -1, -1, +1, -1, +1, +1 };
    double meas[8]  = { 0.6, 0.7, 0.5, 0.6, 0.65, 0.55, 0.6, 0.7 };
    /* Raw signed sum: (+0.6 + 0.7 - 0.5 - 0.6 + 0.65 - 0.55 + 0.6 + 0.7)/8
     * = 1.6 / 8 = 0.2. */
    double sd = 0.0;
    double out = pec_aggregate(signs, meas, n, 2.0, &sd);
    fprintf(stdout, "    mitigated = %.6f (expect 0.4)  sd = %.4f\n", out, sd);
    CHECK(fabs(out - 0.4) < 1e-9, "aggregate = gamma * E[...]");
    CHECK(sd > 0.0, "standard error is positive");
}

int main(void) {
    fprintf(stdout, "=== ZNE + PEC tests ===\n");
    test_linear_exact();
    test_richardson_polynomial_exact();
    test_exponential_fit();
    test_integration_depolar();
    test_pec_one_norm_cost();
    test_pec_sampling_signs();
    test_pec_aggregate_recovers_mean();
    fprintf(stdout, "\n=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
