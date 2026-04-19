/**
 * @file test_clifford.c
 * @brief Unit tests for the Clifford stabilizer backend.
 *
 * Covers:
 *  - H² = I, X² = I, S^4 = I
 *  - Bell state correlations over 2000 shots
 *  - 100-qubit GHZ state: all measured bits are equal within each shot
 *  - Deterministic-vs-random branch of clifford_measure
 *  - |0⟩ measurement is deterministic and returns 0
 */

#include "../../src/backends/clifford/clifford.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

static void test_fresh_state_is_zero(void) {
    fprintf(stdout, "\n-- Clifford: fresh |0...0⟩ is deterministic --\n");
    clifford_tableau_t* t = clifford_tableau_create(5);
    uint64_t rng = 0xDEADBEEFu;
    for (size_t q = 0; q < 5; q++) {
        int out = -1, kind = -1;
        clifford_measure(t, q, &rng, &out, &kind);
        CHECK(out == 0, "qubit %zu measures 0", q);
        CHECK(kind == 0, "qubit %zu is deterministic", q);
    }
    clifford_tableau_free(t);
}

static void test_h_squared_is_identity(void) {
    fprintf(stdout, "\n-- Clifford: H² = I --\n");
    clifford_tableau_t* t = clifford_tableau_create(1);
    clifford_h(t, 0);
    clifford_h(t, 0);
    uint64_t rng = 0x1234u;
    int out = -1, kind = -1;
    clifford_measure(t, 0, &rng, &out, &kind);
    CHECK(kind == 0 && out == 0, "H²|0⟩ measures 0 deterministically");
    clifford_tableau_free(t);
}

static void test_x_gives_one(void) {
    fprintf(stdout, "\n-- Clifford: X|0⟩ = |1⟩ --\n");
    clifford_tableau_t* t = clifford_tableau_create(1);
    clifford_x(t, 0);
    uint64_t rng = 0x1234u;
    int out = -1, kind = -1;
    clifford_measure(t, 0, &rng, &out, &kind);
    CHECK(kind == 0 && out == 1, "X|0⟩ measures 1 deterministically");
    clifford_tableau_free(t);
}

static void test_s4_is_identity(void) {
    fprintf(stdout, "\n-- Clifford: S⁴ = I (order-4 phase gate) --\n");
    clifford_tableau_t* t = clifford_tableau_create(1);
    clifford_h(t, 0);
    clifford_s(t, 0);
    clifford_s(t, 0);
    clifford_s(t, 0);
    clifford_s(t, 0);
    clifford_h(t, 0);
    uint64_t rng = 0x1234u;
    int out = -1, kind = -1;
    clifford_measure(t, 0, &rng, &out, &kind);
    CHECK(kind == 0 && out == 0, "H·S⁴·H|0⟩ = |0⟩ deterministically");
    clifford_tableau_free(t);
}

static void test_bell_state_correlations(void) {
    fprintf(stdout, "\n-- Clifford: Bell state shot statistics --\n");
    const int shots = 2000;
    int n00 = 0, n01 = 0, n10 = 0, n11 = 0;
    uint64_t rng = 0xC0FFEEu;
    for (int s = 0; s < shots; s++) {
        clifford_tableau_t* t = clifford_tableau_create(2);
        clifford_h(t, 0);
        clifford_cnot(t, 0, 1);
        int b0 = 0, b1 = 0;
        clifford_measure(t, 0, &rng, &b0, NULL);
        clifford_measure(t, 1, &rng, &b1, NULL);
        if (!b0 && !b1) n00++;
        else if (!b0 && b1) n01++;
        else if (b0 && !b1) n10++;
        else n11++;
        clifford_tableau_free(t);
    }
    fprintf(stdout,
            "    shots=%d  (00,01,10,11) = (%d,%d,%d,%d)\n",
            shots, n00, n01, n10, n11);
    CHECK(n01 == 0 && n10 == 0, "|01⟩ and |10⟩ never observed");
    double f00 = (double)n00 / shots;
    CHECK(fabs(f00 - 0.5) < 0.05,
          "P(|00⟩) = %.3f ≈ 0.5 (|shots = %d, σ≈0.011)", f00, shots);
}

static void test_ghz_100_qubits(void) {
    fprintf(stdout, "\n-- Clifford: 100-qubit GHZ all-or-nothing --\n");
    const size_t n = 100;
    const int shots = 200;
    uint64_t rng = 0xBADDCAFEu;
    int inconsistent = 0;
    int all_zero = 0, all_one = 0;
    for (int s = 0; s < shots; s++) {
        clifford_tableau_t* t = clifford_tableau_create(n);
        clifford_h(t, 0);
        for (size_t q = 1; q < n; q++) clifford_cnot(t, 0, q);
        int first = 0;
        clifford_measure(t, 0, &rng, &first, NULL);
        int consistent = 1;
        for (size_t q = 1; q < n; q++) {
            int b = -1;
            clifford_measure(t, q, &rng, &b, NULL);
            if (b != first) { consistent = 0; break; }
        }
        if (!consistent) inconsistent++;
        else if (first == 0) all_zero++;
        else all_one++;
        clifford_tableau_free(t);
    }
    fprintf(stdout,
            "    shots=%d  all-zero=%d  all-one=%d  mixed=%d\n",
            shots, all_zero, all_one, inconsistent);
    CHECK(inconsistent == 0,
          "every shot consistent (qubit 0 outcome matches qubits 1..%zu)",
          n - 1);
    CHECK(all_zero > 0 && all_one > 0,
          "both outcomes observed (all_zero=%d, all_one=%d)",
          all_zero, all_one);
}

static void test_deterministic_after_measurement(void) {
    fprintf(stdout, "\n-- Clifford: remeasuring gives the same result --\n");
    clifford_tableau_t* t = clifford_tableau_create(3);
    clifford_h(t, 0);
    clifford_cnot(t, 0, 1);
    clifford_cnot(t, 0, 2);
    uint64_t rng = 0x42424242u;
    int b0 = -1, k0 = -1;
    clifford_measure(t, 0, &rng, &b0, &k0);
    /* After the first (random) measurement the remaining qubits' outcomes
     * should be forced to match b0 on a GHZ_3. */
    int b1 = -1, k1 = -1;
    clifford_measure(t, 1, &rng, &b1, &k1);
    int b2 = -1, k2 = -1;
    clifford_measure(t, 2, &rng, &b2, &k2);
    CHECK(k0 == 1, "qubit 0 measurement was random");
    CHECK(k1 == 0 && b1 == b0, "qubit 1 collapses to match qubit 0");
    CHECK(k2 == 0 && b2 == b0, "qubit 2 collapses to match qubit 0");
    /* Remeasure qubit 0 — must now be deterministic and equal b0. */
    int bR = -1, kR = -1;
    clifford_measure(t, 0, &rng, &bR, &kR);
    CHECK(kR == 0 && bR == b0,
          "remeasure is deterministic (got k=%d out=%d)", kR, bR);
    clifford_tableau_free(t);
}

int main(void) {
    fprintf(stdout, "=== Clifford stabilizer backend ===\n");
    test_fresh_state_is_zero();
    test_h_squared_is_identity();
    test_x_gives_one();
    test_s4_is_identity();
    test_bell_state_correlations();
    test_ghz_100_qubits();
    test_deterministic_after_measurement();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
