/**
 * @file test_povm.c
 * @brief POVM + weak-measurement validation.
 *
 * Checks:
 *   - Two-outcome "projective-as-POVM": K0 = |0><0|, K1 = |1><1|
 *     reproduces the projective Z measurement.
 *   - Trine POVM on a qubit: completeness accepted, all probabilities
 *     non-negative and sum to 1.
 *   - Weak Z at strength = 1 collapses |+> to |0> or |1> with prob 0.5.
 *   - Weak Z at strength = 0 leaves the state unchanged (up to phase).
 *   - Invalid POVM (not sum-to-I) returns QS_ERROR_NOT_NORMALIZED.
 */

#include "../../src/quantum/measurement.h"
#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, fmt, ...) do {                                  \
    if (!(cond)) {                                                  \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);        \
        failures++;                                                 \
    } else {                                                        \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);        \
    }                                                               \
} while (0)

static void test_projective_as_povm(void) {
    fprintf(stdout, "\n-- POVM reproduces projective Z --\n");
    /* K0 = diag(1, 0), K1 = diag(0, 1) on a single qubit. */
    complex_t K0[4] = { 1.0, 0.0, 0.0, 0.0 };
    complex_t K1[4] = { 0.0, 0.0, 0.0, 1.0 };
    const complex_t *ops[2] = { K0, K1 };
    povm_t pv = { .num_outcomes = 2, .state_dim = 2, .kraus_ops = ops };

    /* State: 0.6|0> + 0.8|1>.  p(0) = 0.36, p(1) = 0.64. */
    quantum_state_t s;
    quantum_state_init(&s, 1);
    s.amplitudes[0] = 0.6;
    s.amplitudes[1] = 0.8;

    double probs[2] = {0};
    CHECK(measurement_povm_probabilities(&s, &pv, probs) == QS_SUCCESS,
          "probabilities ok");
    CHECK(fabs(probs[0] - 0.36) < 1e-12 && fabs(probs[1] - 0.64) < 1e-12,
          "p(0)=%.4f p(1)=%.4f (expect 0.36, 0.64)", probs[0], probs[1]);

    /* Collapse test: uniform=0.0 picks outcome 0. */
    size_t outcome = 99;
    CHECK(measurement_povm(&s, &pv, 0.0, &outcome) == QS_SUCCESS, "povm runs");
    CHECK(outcome == 0, "outcome = 0 at uniform = 0.0");
    CHECK(fabs(creal(s.amplitudes[0]) - 1.0) < 1e-12 &&
          cabs(s.amplitudes[1]) < 1e-12,
          "state collapsed to |0>");

    quantum_state_free(&s);
}

static void test_trine_povm(void) {
    fprintf(stdout, "\n-- Trine POVM on a qubit --\n");
    /* Three POVM elements along equally-spaced directions:
     * E_k = (2/3) |v_k><v_k| with |v_k> = cos(theta_k/2)|0> + sin(theta_k/2)|1>,
     * theta_k = 0, 2pi/3, 4pi/3.
     * Kraus ops K_k = sqrt(E_k).  Each is a rank-1 matrix sqrt(2/3)
     * |v_k><v_k|.  Completeness sum_k (2/3)|v_k><v_k| = I on the
     * 2-dim space (a standard SIC-like construction). */
    double thetas[3] = { 0.0, 2.0 * M_PI / 3.0, 4.0 * M_PI / 3.0 };
    complex_t *Ks[3];
    for (int k = 0; k < 3; k++) {
        Ks[k] = (complex_t*)calloc(4, sizeof(complex_t));
        double c = cos(thetas[k] / 2.0);
        double s = sin(thetas[k] / 2.0);
        double n = sqrt(2.0 / 3.0);
        /* K_k = n * |v><v|.  |v><v| = [[c*c, c*s], [s*c, s*s]]. */
        Ks[k][0] = n * c * c;
        Ks[k][1] = n * c * s;
        Ks[k][2] = n * s * c;
        Ks[k][3] = n * s * s;
    }
    const complex_t *ops[3] = { Ks[0], Ks[1], Ks[2] };
    povm_t pv = { .num_outcomes = 3, .state_dim = 2, .kraus_ops = ops };

    quantum_state_t s;
    quantum_state_init(&s, 1);
    /* |psi> = (|0> + i|1>)/sqrt(2). */
    s.amplitudes[0] = 1.0 / sqrt(2.0);
    s.amplitudes[1] = I / sqrt(2.0);

    double p[3] = {0};
    CHECK(measurement_povm_probabilities(&s, &pv, p) == QS_SUCCESS,
          "trine probs ok");
    double sum = p[0] + p[1] + p[2];
    fprintf(stdout, "    probs: %.4f %.4f %.4f  sum=%.6f\n",
            p[0], p[1], p[2], sum);
    CHECK(fabs(sum - 1.0) < 1e-9, "trine probabilities sum to 1");
    for (int k = 0; k < 3; k++) {
        CHECK(p[k] > -1e-12, "p[%d] non-negative", k);
    }

    /* Pick an outcome, check we don't crash and state re-normalises. */
    size_t out = 0;
    CHECK(measurement_povm(&s, &pv, 0.5, &out) == QS_SUCCESS, "trine collapse ok");
    double norm = 0.0;
    for (uint64_t i = 0; i < 2; i++) {
        norm += cabs(s.amplitudes[i]) * cabs(s.amplitudes[i]);
    }
    CHECK(fabs(norm - 1.0) < 1e-9, "post-collapse norm = 1 (got %.6f)", norm);

    for (int k = 0; k < 3; k++) free(Ks[k]);
    quantum_state_free(&s);
}

static void test_weak_z_limits(void) {
    fprintf(stdout, "\n-- Weak Z: strength 0 and strength 1 limits --\n");

    /* strength = 1: projective.  On |+>, prob = 0.5 for each outcome;
     * state collapses to |0> or |1>. */
    {
        quantum_state_t s;
        quantum_state_init(&s, 1);
        gate_hadamard(&s, 0);  /* |+> */
        int outcome = 0;
        qs_error_t rc = measurement_weak_z(&s, 0, 1.0, 0.25, &outcome);
        CHECK(rc == QS_SUCCESS, "projective weak Z runs");
        /* At uniform = 0.25 and p(+) = 0.5, outcome = 0 (matched). */
        CHECK(outcome == 0, "projective strength-1 picks outcome 0");
        CHECK(fabs(creal(s.amplitudes[0]) - 1.0) < 1e-12 &&
              cabs(s.amplitudes[1]) < 1e-12,
              "projective collapses |+> to |0>");
        quantum_state_free(&s);
    }

    /* strength = 0: non-disturbing.  State should be unchanged up to
     * normalisation (both Kraus operators are (1/sqrt(2)) * I). */
    {
        quantum_state_t s;
        quantum_state_init(&s, 1);
        gate_hadamard(&s, 0);
        complex_t before_0 = s.amplitudes[0];
        complex_t before_1 = s.amplitudes[1];
        int outcome = 0;
        qs_error_t rc = measurement_weak_z(&s, 0, 0.0, 0.7, &outcome);
        CHECK(rc == QS_SUCCESS, "non-disturbing weak Z runs");
        /* State should equal the original (K_k = (1/sqrt(2)) I, so
         * p_k = 1/2 and K_k psi / sqrt(p_k) = psi). */
        CHECK(cabs(s.amplitudes[0] - before_0) < 1e-12 &&
              cabs(s.amplitudes[1] - before_1) < 1e-12,
              "non-disturbing weak Z preserves state");
        quantum_state_free(&s);
    }
}

static void test_invalid_povm_rejected(void) {
    fprintf(stdout, "\n-- Invalid (non-normalised) POVM is rejected --\n");
    complex_t K[4] = { 0.5, 0.0, 0.0, 0.5 };  /* K^dag K = 0.25 I, not I */
    const complex_t *ops[1] = { K };
    povm_t pv = { .num_outcomes = 1, .state_dim = 2, .kraus_ops = ops };
    quantum_state_t s;
    quantum_state_init(&s, 1);
    size_t out = 0;
    qs_error_t rc = measurement_povm(&s, &pv, 0.5, &out);
    CHECK(rc == QS_ERROR_NOT_NORMALIZED,
          "non-normalised POVM returns QS_ERROR_NOT_NORMALIZED");
    quantum_state_free(&s);
}

int main(void) {
    fprintf(stdout, "=== POVM / weak-measurement tests ===\n");
    test_projective_as_povm();
    test_trine_povm();
    test_weak_z_limits();
    test_invalid_povm_rejected();
    fprintf(stdout, "\n=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
