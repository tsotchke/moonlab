/**
 * @file test_bell_variants.c
 * @brief Validate Mermin and Mermin-Klyshko Bell inequalities.
 *
 * Ideal-case targets:
 *   Mermin on |GHZ_3>:           |M| = 4 exactly (analytic path).
 *   Mermin-Klyshko on |GHZ_N>:   reaches 2^((N-1)/2).
 *   Classical (product) states:  stay at or below the classical bound.
 */

#include "../../src/algorithms/bell_tests.h"
#include "../../src/utils/quantum_entropy.h"
#include "../../src/applications/hardware_entropy.h"
#include "../../src/quantum/gates.h"

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

static void prepare_ghz(quantum_state_t *s, size_t n) {
    quantum_state_init(s, (uint32_t)n);
    gate_hadamard(s, 0);
    for (size_t i = 0; i + 1 < n; i++) gate_cnot(s, (int)i, (int)(i + 1));
}

static void test_mermin_ghz3(void) {
    fprintf(stdout, "\n-- Mermin on |GHZ_3> --\n");
    quantum_state_t s;
    prepare_ghz(&s, 3);

    entropy_ctx_t hw; entropy_init(&hw);
    quantum_entropy_ctx_t e;
    quantum_entropy_init(&e, (quantum_entropy_fn)entropy_get_bytes, &hw);

    bell_test_result_t r = bell_test_mermin_ghz(&s, 0, 1, 2, 0, &e);
    fprintf(stdout, "    <XYY>=%.4f <YXY>=%.4f <YYX>=%.4f <XXX>=%.4f  |M|=%.4f\n",
            r.correlation_ab, r.correlation_ab_prime,
            r.correlation_a_prime_b, r.correlation_a_prime_b_prime,
            r.chsh_value);
    /* Analytic expectation: <XYY> = -1, <YXY> = -1, <YYX> = -1, <XXX> = +1.
     * M = -1 + -1 + -1 - 1 = -4, |M| = 4 -- maximal quantum violation. */
    CHECK(fabs(r.chsh_value - 4.0) < 1e-9,
          "Mermin M = 4 analytically (got %.6f)", r.chsh_value);
    CHECK(r.violates_classical, "Mermin violates classical bound 2");
    quantum_state_free(&s);
}

static void test_mermin_product_state_zero(void) {
    fprintf(stdout, "\n-- Mermin on separable |+++> should be bounded --\n");
    quantum_state_t s;
    quantum_state_init(&s, 3);
    gate_hadamard(&s, 0); gate_hadamard(&s, 1); gate_hadamard(&s, 2);
    /* |+++> : every single-qubit X = +1, every Y = 0.
     * M = <XYY> + <YXY> + <YYX> - <XXX> = 0 + 0 + 0 - 1 = -1, |M| = 1. */
    entropy_ctx_t hw; entropy_init(&hw);
    quantum_entropy_ctx_t e;
    quantum_entropy_init(&e, (quantum_entropy_fn)entropy_get_bytes, &hw);
    bell_test_result_t r = bell_test_mermin_ghz(&s, 0, 1, 2, 0, &e);
    fprintf(stdout, "    |M| = %.4f\n", r.chsh_value);
    CHECK(r.chsh_value < 2.0 + 1e-9, "separable state obeys Mermin");
    quantum_state_free(&s);
}

static void test_mermin_klyshko_ghz(void) {
    fprintf(stdout, "\n-- Mermin-Klyshko on |GHZ_N> --\n");
    for (size_t N = 2; N <= 5; N++) {
        quantum_state_t s;
        prepare_ghz(&s, N);
        double v = bell_test_mermin_klyshko(&s, N, 0, NULL);
        /* Normalised so classical bound is 1, ideal quantum is 2^((N-1)/2). */
        double ideal = pow(2.0, (double)(N - 1) / 2.0);
        fprintf(stdout, "    N=%zu  |M_N|/norm = %.4f  (classical <= 1, ideal %.4f)\n",
                N, v, ideal);
        CHECK(v > 1.0, "GHZ_%zu violates classical Mermin-Klyshko", N);
        CHECK(fabs(v - ideal) < 1e-9,
              "GHZ_%zu reaches ideal quantum bound %.4f (got %.6f)",
              N, ideal, v);
        quantum_state_free(&s);
    }
}

int main(void) {
    fprintf(stdout, "=== Bell inequality variant tests (Mermin, Mermin-Klyshko) ===\n");
    test_mermin_ghz3();
    test_mermin_product_state_zero();
    test_mermin_klyshko_ghz();
    fprintf(stdout, "\n=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
