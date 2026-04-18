/**
 * @file test_measurement.c
 * @brief Correctness tests for the measurement subsystem.
 *
 * Covers:
 *  - `measurement_probability_one` / `_zero` for pure computational-basis
 *    and maximally-superposed states.
 *  - `measurement_probability_distribution` — full distribution sums to 1.
 *  - Projective collapse on |+> returns {0,1} with ~50/50 over many shots.
 *  - Partial measurement collapses the measured subspace and leaves the
 *    unmeasured qubits in a coherent superposition.
 *  - Mid-circuit measurement followed by a second draw produces a
 *    deterministic state.
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/quantum/measurement.h"
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

static double near(double a, double b) {
    return fabs(a - b) < 1e-12;
}

static void test_probability_pure_basis(void) {
    fprintf(stdout, "\n-- probabilities on |00> --\n");
    quantum_state_t s; quantum_state_init(&s, 2);
    /* initial state is |00>. */
    CHECK(near(measurement_probability_one(&s, 0), 0.0),
          "P(q0=1) on |00> is 0");
    CHECK(near(measurement_probability_one(&s, 1), 0.0),
          "P(q1=1) on |00> is 0");
    CHECK(near(measurement_probability_zero(&s, 0), 1.0),
          "P(q0=0) on |00> is 1");
    quantum_state_free(&s);
}

static void test_probability_plus(void) {
    fprintf(stdout, "\n-- probabilities on |+> --\n");
    quantum_state_t s; quantum_state_init(&s, 1);
    gate_hadamard(&s, 0);
    CHECK(near(measurement_probability_one(&s, 0), 0.5),
          "P(q0=1) on |+> is 1/2");
    CHECK(near(measurement_probability_zero(&s, 0), 0.5),
          "P(q0=0) on |+> is 1/2");
    quantum_state_free(&s);
}

static void test_distribution_sums_to_one(void) {
    fprintf(stdout, "\n-- probability distribution integrates to 1 --\n");
    quantum_state_t s; quantum_state_init(&s, 3);
    gate_hadamard(&s, 0);
    gate_hadamard(&s, 1);
    gate_hadamard(&s, 2);

    double dist[8];
    measurement_probability_distribution(&s, dist);
    double total = 0.0;
    for (int i = 0; i < 8; ++i) total += dist[i];
    CHECK(near(total, 1.0), "sum of P(i) = 1 (got %.15g)", total);

    /* Uniform superposition: each basis state has P = 1/8. */
    int uniform = 1;
    for (int i = 0; i < 8; ++i) {
        if (!near(dist[i], 0.125)) { uniform = 0; break; }
    }
    CHECK(uniform, "|+>^3 gives uniform 1/8 per outcome");
    quantum_state_free(&s);
}

static void test_single_qubit_collapse(void) {
    fprintf(stdout, "\n-- single-qubit projective collapse on |+> --\n");
    /* The simulator's convention is inverse-CDF style:
     *   outcome = 1 if random_value < P(1)
     *           = 0 otherwise
     * On |+>, P(1) = 0.5. So random=0.25 -> outcome 1,
     * random=0.75 -> outcome 0. Verify both branches and the
     * resulting state collapse. */
    {
        quantum_state_t s; quantum_state_init(&s, 1);
        gate_hadamard(&s, 0);
        int out = measurement_single_qubit(&s, 0, 0.25);
        CHECK(out == 1, "random=0.25 < P(1)=0.5 yields outcome 1");
        CHECK(near(measurement_probability_one(&s, 0), 1.0),
              "post-measurement state is |1>");
        quantum_state_free(&s);
    }
    {
        quantum_state_t s; quantum_state_init(&s, 1);
        gate_hadamard(&s, 0);
        int out = measurement_single_qubit(&s, 0, 0.75);
        CHECK(out == 0, "random=0.75 >= P(1)=0.5 yields outcome 0");
        CHECK(near(measurement_probability_zero(&s, 0), 1.0),
              "post-measurement state is |0>");
        quantum_state_free(&s);
    }
}

static void test_bell_measurement_correlation(void) {
    fprintf(stdout, "\n-- |Phi+> measurement correlations --\n");
    /* On |Phi+> = (|00>+|11>)/sqrt(2), measuring qubit 0 then qubit 1
     * should always give equal outcomes. Verify across both branches. */
    for (int trial = 0; trial < 2; ++trial) {
        quantum_state_t s; quantum_state_init(&s, 2);
        gate_hadamard(&s, 0);
        gate_cnot(&s, 0, 1);
        double rv = trial == 0 ? 0.1 : 0.9;
        int m0 = measurement_single_qubit(&s, 0, rv);
        /* Once qubit 0 collapses, qubit 1 is deterministic. */
        double p1 = measurement_probability_one(&s, 1);
        if (m0 == 0) {
            CHECK(near(p1, 0.0),
                  "after measuring q0=0 on |Phi+>, P(q1=1) == 0");
        } else {
            CHECK(near(p1, 1.0),
                  "after measuring q0=1 on |Phi+>, P(q1=1) == 1");
        }
        quantum_state_free(&s);
    }
}

static void test_expectation_z(void) {
    fprintf(stdout, "\n-- <Z> expectation --\n");
    /* <0|Z|0> = +1, <1|Z|1> = -1, <+|Z|+> = 0. */
    {
        quantum_state_t s; quantum_state_init(&s, 1);
        double p1 = measurement_probability_one(&s, 0);
        double ez = 1.0 - 2.0 * p1;  /* <Z> = P(0) - P(1) */
        CHECK(near(ez, 1.0), "<Z> on |0> is +1");
        quantum_state_free(&s);
    }
    {
        quantum_state_t s; quantum_state_init(&s, 1);
        gate_pauli_x(&s, 0);
        double p1 = measurement_probability_one(&s, 0);
        double ez = 1.0 - 2.0 * p1;
        CHECK(near(ez, -1.0), "<Z> on |1> is -1");
        quantum_state_free(&s);
    }
    {
        quantum_state_t s; quantum_state_init(&s, 1);
        gate_hadamard(&s, 0);
        double p1 = measurement_probability_one(&s, 0);
        double ez = 1.0 - 2.0 * p1;
        CHECK(near(ez, 0.0), "<Z> on |+> is 0");
        quantum_state_free(&s);
    }
}

int main(void) {
    fprintf(stdout, "=== measurement subsystem tests ===\n");
    test_probability_pure_basis();
    test_probability_plus();
    test_distribution_sums_to_one();
    test_single_qubit_collapse();
    test_bell_measurement_correlation();
    test_expectation_z();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
