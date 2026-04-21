/**
 * @file test_composite_noise.c
 * @brief Composite / correlated noise channels.
 */

#include "../../src/quantum/noise.h"
#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"

#include <complex.h>
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

static void test_correlated_pauli_II_identity(void) {
    fprintf(stdout, "\n-- correlated 2q Pauli: II with prob 1 is identity --\n");
    quantum_state_t s;
    quantum_state_init(&s, 2);
    gate_hadamard(&s, 0); gate_cnot(&s, 0, 1);
    complex_t before[4];
    for (int i = 0; i < 4; i++) before[i] = s.amplitudes[i];

    double probs[16] = {0};
    probs[0] = 1.0;  /* II */
    noise_correlated_two_qubit_pauli(&s, 0, 1, probs, 0.5);
    for (int i = 0; i < 4; i++) {
        CHECK(cabs(s.amplitudes[i] - before[i]) < 1e-12,
              "amp[%d] unchanged", i);
    }
    quantum_state_free(&s);
}

static void test_correlated_pauli_XX_flips(void) {
    fprintf(stdout, "\n-- correlated 2q Pauli: XX with prob 1 flips both --\n");
    quantum_state_t s;
    quantum_state_init(&s, 2);
    /* |00> -> XX -> |11>. */
    double probs[16] = {0};
    probs[1 * 4 + 1] = 1.0;  /* index (a=X, b=X) */
    noise_correlated_two_qubit_pauli(&s, 0, 1, probs, 0.5);
    CHECK(cabs(s.amplitudes[0]) < 1e-12, "amp[00] = 0");
    CHECK(cabs(s.amplitudes[3] - 1.0) < 1e-12, "amp[11] = 1");
    quantum_state_free(&s);
}

static void test_correlated_pauli_bad_prob_rejected(void) {
    fprintf(stdout, "\n-- correlated 2q Pauli: non-normalised rejected --\n");
    quantum_state_t s;
    quantum_state_init(&s, 2);
    complex_t before0 = s.amplitudes[0];
    double probs[16] = { 0.5 };  /* sum != 1, no-op */
    noise_correlated_two_qubit_pauli(&s, 0, 1, probs, 0.5);
    CHECK(cabs(s.amplitudes[0] - before0) < 1e-12,
          "bad probability table leaves state untouched");
    quantum_state_free(&s);
}

static void test_composite_sequential(void) {
    fprintf(stdout, "\n-- composite sequential: X-flip + phase-flip = -Y --\n");
    /* Apply bit flip (prob 1) then phase flip (prob 1) sequentially to
     * |0>: first X gives |1>, then Z gives -|1>.  Since the channels
     * are parameterized by probability with random_value < prob always
     * firing, set random < prob. */
    quantum_state_t s;
    quantum_state_init(&s, 1);
    noise_composite_sequential_single(&s, 0,
        NOISE_CHANNEL_BIT_FLIP, 1.0,
        NOISE_CHANNEL_PHASE_FLIP, 1.0,
        0.0, 0.0);
    /* Expected: amp[1] = -1. */
    double p0 = cabs(s.amplitudes[0]);
    double p1 = cabs(s.amplitudes[1]);
    CHECK(p0 < 1e-12, "|0>-amplitude = 0 after X then Z");
    CHECK(fabs(p1 - 1.0) < 1e-12, "|1>-amplitude magnitude = 1");
    CHECK(creal(s.amplitudes[1]) < 0.0, "sign is negative (Z on |1>)");
    quantum_state_free(&s);
}

static void test_convex_mixture_branches(void) {
    fprintf(stdout, "\n-- convex mixture: uniform_pick selects branch --\n");
    /* Branch A: BIT_FLIP with p=1 -> X.  Branch B: PHASE_FLIP with p=1 -> Z. */
    quantum_state_t s;
    quantum_state_init(&s, 1);
    gate_hadamard(&s, 0);  /* |+> */
    noise_convex_mixture_single(&s, 0,
        NOISE_CHANNEL_BIT_FLIP, 1.0,
        NOISE_CHANNEL_PHASE_FLIP, 1.0,
        0.5,   /* mixture_prob */
        0.1,   /* uniform_pick < mixture_prob -> pick A */
        0.0);  /* random_channel */
    /* X|+> = |+>, so amp unchanged. */
    CHECK(fabs(cabs(s.amplitudes[0]) - 1.0/sqrt(2.0)) < 1e-9 &&
          fabs(cabs(s.amplitudes[1]) - 1.0/sqrt(2.0)) < 1e-9,
          "X branch on |+> preserves |+> magnitudes");
    quantum_state_free(&s);

    /* Now force branch B (Z|+> = |-> = (|0> - |1>)/sqrt(2)). */
    quantum_state_init(&s, 1);
    gate_hadamard(&s, 0);
    noise_convex_mixture_single(&s, 0,
        NOISE_CHANNEL_BIT_FLIP, 1.0,
        NOISE_CHANNEL_PHASE_FLIP, 1.0,
        0.5,
        0.9,   /* uniform_pick > mixture_prob -> pick B */
        0.0);
    CHECK(creal(s.amplitudes[1]) < 0.0,
          "Z branch on |+> yields |-> (negative phase on |1>)");
    quantum_state_free(&s);
}

int main(void) {
    fprintf(stdout, "=== composite / correlated noise tests ===\n");
    test_correlated_pauli_II_identity();
    test_correlated_pauli_XX_flips();
    test_correlated_pauli_bad_prob_rejected();
    test_composite_sequential();
    test_convex_mixture_branches();
    fprintf(stdout, "\n=== %d failure%s ===\n", failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
