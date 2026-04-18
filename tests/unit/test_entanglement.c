/**
 * @file test_entanglement.c
 * @brief Correctness tests for entanglement metrics.
 *
 * Covers:
 *  - Product states have entanglement entropy 0.
 *  - All four Bell states have maximal bipartite entanglement (1 ebit).
 *  - Fidelity is 1 between identical states and < 1 between orthogonal.
 *  - Purity is 1 for every pure state we construct (simulator is
 *    pure-state only).
 *  - Partial trace of a Bell state yields the maximally-mixed rho = I/2.
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/quantum/entanglement.h"
#include "../../src/algorithms/bell_tests.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

static int failures = 0;

#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

static int close(double a, double b, double tol) {
    return fabs(a - b) < tol;
}

static void test_product_state_zero_entropy(void) {
    fprintf(stdout, "\n-- product states have zero entanglement entropy --\n");
    quantum_state_t s; quantum_state_init(&s, 2);
    /* |00> is a product state. */
    int subsys[] = {0};
    double S = quantum_state_entanglement_entropy(&s, subsys, 1);
    CHECK(close(S, 0.0, 1e-10),
          "|00> entanglement entropy = 0 (got %.3e)", S);

    /* |+>|0> is also a product state. */
    gate_hadamard(&s, 0);
    S = quantum_state_entanglement_entropy(&s, subsys, 1);
    CHECK(close(S, 0.0, 1e-10),
          "|+>|0> entanglement entropy = 0 (got %.3e)", S);
    quantum_state_free(&s);
}

static void test_bell_states_max_entropy(void) {
    fprintf(stdout, "\n-- Bell states have maximal bipartite entropy --\n");
    const bell_state_type_t types[] = {
        BELL_PHI_PLUS, BELL_PHI_MINUS, BELL_PSI_PLUS, BELL_PSI_MINUS
    };
    const char* names[] = { "|Phi+>", "|Phi->", "|Psi+>", "|Psi->" };
    int subsys[] = {0};

    for (int i = 0; i < 4; ++i) {
        quantum_state_t s; quantum_state_init(&s, 2);
        create_bell_state(&s, 0, 1, types[i]);
        double S = quantum_state_entanglement_entropy(&s, subsys, 1);
        CHECK(close(S, 1.0, 1e-10),
              "%s entanglement entropy = 1 ebit (got %.6f)", names[i], S);
        quantum_state_free(&s);
    }
}

static void test_fidelity_identical_and_orthogonal(void) {
    fprintf(stdout, "\n-- fidelity of identical and orthogonal states --\n");
    quantum_state_t a, b;
    quantum_state_init(&a, 2);
    quantum_state_init(&b, 2);
    /* Both in |00>. */
    double F = quantum_state_fidelity(&a, &b);
    CHECK(close(F, 1.0, 1e-12), "F(|00>, |00>) = 1");

    /* Flip b to |11> -> orthogonal. */
    gate_pauli_x(&b, 0); gate_pauli_x(&b, 1);
    F = quantum_state_fidelity(&a, &b);
    CHECK(close(F, 0.0, 1e-12), "F(|00>, |11>) = 0");

    /* a = |+>|+>, b = |-\>|-\>, not quite orthogonal but < 1. */
    gate_hadamard(&a, 0); gate_hadamard(&a, 1);
    /* rebuild b as |-> on both qubits */
    quantum_state_reset(&b);
    gate_pauli_x(&b, 0); gate_hadamard(&b, 0);
    gate_pauli_x(&b, 1); gate_hadamard(&b, 1);
    F = quantum_state_fidelity(&a, &b);
    CHECK(F >= 0.0 && F <= 1.0,
          "F(|++>, |-->) is a valid probability (got %.6f)", F);

    quantum_state_free(&a); quantum_state_free(&b);
}

static void test_purity_of_pure_states(void) {
    fprintf(stdout, "\n-- purity = 1 across pure states --\n");
    const struct { const char* name; int qubits; } cases[] = {
        {"|0000>", 4},
        {"|+>|+>|+>",     3},
    };
    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i) {
        quantum_state_t s; quantum_state_init(&s, cases[i].qubits);
        if (i == 1) {
            for (int q = 0; q < cases[i].qubits; ++q) gate_hadamard(&s, q);
        }
        double p = quantum_state_purity(&s);
        CHECK(close(p, 1.0, 1e-10),
              "purity of %s is 1 (got %.6f)", cases[i].name, p);
        quantum_state_free(&s);
    }
}

static void test_partial_trace_bell_to_maximally_mixed(void) {
    fprintf(stdout, "\n-- partial trace of |Phi+> = I/2 --\n");
    /* rho_A = (1/2)|0><0| + (1/2)|1><1| = I/2 for any Bell state. */
    quantum_state_t s; quantum_state_init(&s, 2);
    create_bell_state_phi_plus(&s, 0, 1);

    complex_t reduced[4] = {0};
    int trace_out[] = {1};
    qs_error_t err = quantum_state_partial_trace(&s, trace_out, 1, reduced);
    CHECK(err == QS_SUCCESS, "partial_trace returns success");

    /* Expected rho_A = diag(0.5, 0.5). */
    CHECK(close(creal(reduced[0]), 0.5, 1e-10) &&
          close(cimag(reduced[0]), 0.0, 1e-10),
          "rho_A[0,0] = 1/2");
    CHECK(close(creal(reduced[3]), 0.5, 1e-10) &&
          close(cimag(reduced[3]), 0.0, 1e-10),
          "rho_A[1,1] = 1/2");
    CHECK(close(creal(reduced[1]), 0.0, 1e-10) &&
          close(cimag(reduced[1]), 0.0, 1e-10) &&
          close(creal(reduced[2]), 0.0, 1e-10) &&
          close(cimag(reduced[2]), 0.0, 1e-10),
          "off-diagonals rho_A[0,1] = rho_A[1,0] = 0");

    quantum_state_free(&s);
}

int main(void) {
    fprintf(stdout, "=== entanglement subsystem tests ===\n");
    test_product_state_zero_entropy();
    test_bell_states_max_entropy();
    test_fidelity_identical_and_orthogonal();
    test_purity_of_pure_states();
    test_partial_trace_bell_to_maximally_mixed();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
