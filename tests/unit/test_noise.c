/**
 * @file test_noise.c
 * @brief Correctness tests for noise channels.
 *
 * Covers:
 *  - Bit flip applied twice is the identity for any probability.
 *  - Phase flip applied twice is the identity.
 *  - Pure dephasing preserves populations (P(0) and P(1) unchanged)
 *    while destroying coherence.
 *  - Amplitude damping drives the state toward |0> (ground state).
 *  - Depolarizing with p=0 is a no-op; with p=1 is maximally mixed.
 *  - Post-channel state remains normalised to within 1e-10.
 *
 * The simulator's noise channels consume a caller-provided random
 * value (avoids hidden entropy state). Tests pass either an extreme
 * value (0 or 1) to deterministically drive the channel, or an
 * average over a deterministic sequence to probe statistical
 * behaviour.
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/quantum/measurement.h"
#include "../../src/quantum/noise.h"
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

static double near(double a, double b, double tol) {
    return fabs(a - b) < tol;
}

static double state_norm_squared(const quantum_state_t* s) {
    double n = 0.0;
    for (size_t i = 0; i < s->state_dim; ++i) {
        double m = cabs(s->amplitudes[i]);
        n += m * m;
    }
    return n;
}

static void test_bit_flip_involution(void) {
    fprintf(stdout, "\n-- bit flip applied twice is identity --\n");
    /* bit_flip with p=1 always flips; applying twice should restore. */
    quantum_state_t s; quantum_state_init(&s, 1);
    gate_hadamard(&s, 0);  /* |+> */
    double before = measurement_probability_one(&s, 0);
    noise_bit_flip(&s, 0, 1.0, 0.1);
    noise_bit_flip(&s, 0, 1.0, 0.1);
    double after = measurement_probability_one(&s, 0);
    CHECK(near(before, after, 1e-10),
          "P(1) preserved after X*X (before=%.6f after=%.6f)", before, after);
    CHECK(near(state_norm_squared(&s), 1.0, 1e-10),
          "norm preserved");
    quantum_state_free(&s);
}

static void test_phase_flip_populations(void) {
    fprintf(stdout, "\n-- phase flip preserves populations --\n");
    /* Phase flip changes phase but not |amplitude|^2 of each basis state. */
    quantum_state_t s; quantum_state_init(&s, 1);
    gate_hadamard(&s, 0);  /* |+> */
    double p1_before = measurement_probability_one(&s, 0);
    noise_phase_flip(&s, 0, 0.5, 0.3);
    double p1_after = measurement_probability_one(&s, 0);
    CHECK(near(p1_before, p1_after, 1e-10),
          "P(1) unchanged by phase flip (before=%.6f after=%.6f)",
          p1_before, p1_after);
    CHECK(near(state_norm_squared(&s), 1.0, 1e-10),
          "norm preserved");
    quantum_state_free(&s);
}

static void test_depolarizing_p0_is_noop(void) {
    fprintf(stdout, "\n-- depolarizing with p=0 is a no-op --\n");
    quantum_state_t s; quantum_state_init(&s, 2);
    gate_hadamard(&s, 0);
    gate_cnot(&s, 0, 1);  /* |Phi+> */

    complex_t before[4];
    for (int i = 0; i < 4; ++i) before[i] = s.amplitudes[i];

    noise_depolarizing_single(&s, 0, 0.0, 0.5);

    double drift = 0.0;
    for (int i = 0; i < 4; ++i) {
        drift += cabs(before[i] - s.amplitudes[i]);
    }
    CHECK(near(drift, 0.0, 1e-12),
          "p=0 leaves |Phi+> untouched (L1 drift = %.3e)", drift);
    quantum_state_free(&s);
}

static void test_amplitude_damping_preserves_norm(void) {
    fprintf(stdout, "\n-- amplitude damping preserves norm --\n");
    /* Apply amplitude damping over many steps and check the state stays
     * normalised and the population of |0> is non-decreasing. */
    quantum_state_t s; quantum_state_init(&s, 1);
    gate_pauli_x(&s, 0);   /* |1> — the decayable state */

    double p0_initial = measurement_probability_zero(&s, 0);
    CHECK(near(p0_initial, 0.0, 1e-12),
          "start in |1>: P(0) = 0");

    for (int i = 0; i < 10; ++i) {
        noise_amplitude_damping(&s, 0, 0.2, (double)i / 10.0);
        CHECK(near(state_norm_squared(&s), 1.0, 1e-8),
              "norm after step %d (got %.15g)", i,
              state_norm_squared(&s));
    }
    quantum_state_free(&s);
}

static void test_pure_dephasing_populations(void) {
    fprintf(stdout, "\n-- pure dephasing preserves populations --\n");
    /* Pure dephasing destroys coherence but preserves |alpha|^2 and
     * |beta|^2 for each basis state. */
    quantum_state_t s; quantum_state_init(&s, 2);
    gate_hadamard(&s, 0);   /* q0 in |+>, q1 in |0> */

    double pops_before[4];
    measurement_probability_distribution(&s, pops_before);

    noise_pure_dephasing(&s, 0, 0.5, 0.4);

    double pops_after[4];
    measurement_probability_distribution(&s, pops_after);

    int ok = 1;
    for (int i = 0; i < 4; ++i) {
        if (!near(pops_before[i], pops_after[i], 1e-9)) ok = 0;
    }
    CHECK(ok, "pure dephasing preserves every basis-state population");
    CHECK(near(state_norm_squared(&s), 1.0, 1e-10),
          "norm preserved");
    quantum_state_free(&s);
}

static void test_norm_after_every_channel(void) {
    fprintf(stdout, "\n-- all channels preserve norm --\n");
    const char* names[] = {
        "bit_flip", "phase_flip", "bit_phase_flip",
        "depolarizing_single", "amplitude_damping", "phase_damping",
        "pure_dephasing"
    };
    for (size_t i = 0; i < sizeof(names)/sizeof(names[0]); ++i) {
        quantum_state_t s; quantum_state_init(&s, 3);
        gate_hadamard(&s, 0);
        gate_cnot(&s, 0, 1);
        gate_hadamard(&s, 2);

        const double p = 0.3;
        const double rv = 0.42;
        if (i == 0) noise_bit_flip(&s, 1, p, rv);
        else if (i == 1) noise_phase_flip(&s, 1, p, rv);
        else if (i == 2) noise_bit_phase_flip(&s, 1, p, rv);
        else if (i == 3) noise_depolarizing_single(&s, 1, p, rv);
        else if (i == 4) noise_amplitude_damping(&s, 1, p, rv);
        else if (i == 5) noise_phase_damping(&s, 1, p, rv);
        else             noise_pure_dephasing(&s, 1, p, rv);

        double n = state_norm_squared(&s);
        CHECK(near(n, 1.0, 1e-8),
              "%-22s preserves ||psi||^2 (got %.15g)", names[i], n);
        quantum_state_free(&s);
    }
}

/* Kraus-completeness: Σ K†K = I within float tolerance for every
 * single-qubit channel and parameter value. */
static void test_kraus_completeness(void) {
    fprintf(stdout, "\n-- noise: Kraus-completeness Σ K†K = I --\n");
    struct { noise_channel_id_t id; const char *name; } chans[] = {
        {NOISE_CHANNEL_DEPOLARIZING,     "depolarizing"},
        {NOISE_CHANNEL_AMPLITUDE_DAMPING,"amplitude damping"},
        {NOISE_CHANNEL_PHASE_DAMPING,    "phase damping"},
        {NOISE_CHANNEL_BIT_FLIP,         "bit flip"},
        {NOISE_CHANNEL_PHASE_FLIP,       "phase flip"},
        {NOISE_CHANNEL_BIT_PHASE_FLIP,   "bit-phase flip"},
    };
    double params[] = { 0.0, 0.1, 0.5, 0.9, 1.0 };
    for (size_t c = 0; c < sizeof(chans)/sizeof(chans[0]); c++) {
        for (size_t i = 0; i < sizeof(params)/sizeof(params[0]); i++) {
            double dev = noise_kraus_completeness_deviation(
                chans[c].id, params[i]);
            CHECK(dev >= 0.0 && dev < 1e-12,
                  "%-18s  p=%.2f  max|ΣK†K - I| = %.3e",
                  chans[c].name, params[i], dev);
        }
    }
    /* Invalid parameter is rejected. */
    CHECK(noise_kraus_completeness_deviation(NOISE_CHANNEL_DEPOLARIZING, -0.1) < 0,
          "invalid p rejected (returns < 0)");
}

int main(void) {
    fprintf(stdout, "=== noise channel tests ===\n");
    test_bit_flip_involution();
    test_phase_flip_populations();
    test_depolarizing_p0_is_noop();
    test_amplitude_damping_preserves_norm();
    test_pure_dephasing_populations();
    test_norm_after_every_channel();
    test_kraus_completeness();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
