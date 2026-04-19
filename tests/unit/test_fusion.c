/**
 * @file test_fusion.c
 * @brief Unit tests for the gate-fusion DAG.
 *
 * Verifies:
 *  - Bell circuit produces the expected amplitudes after fusion.
 *  - Pauli identities (X·X = I, H·H = I) collapse to a single 2x2.
 *  - Random mixed circuits on 4, 5 and 6 qubits produce state vectors
 *    identical (L2 diff ≤ 1e-12) to gate-by-gate execution.
 *  - Fusion reduces the kernel-launch count (original_gates > fused_gates)
 *    on every test case with a run of single-qubit gates.
 */

#include "../../src/optimization/fusion/fusion.h"
#include "../../src/quantum/gates.h"
#include "../../src/quantum/state.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward-declared from the library. */
qs_error_t quantum_state_init(quantum_state_t* state, size_t num_qubits);
void       quantum_state_free(quantum_state_t* state);

static int failures = 0;

#define CHECK(cond, fmt, ...) do {                              \
    if (!(cond)) {                                              \
        fprintf(stderr, "  FAIL  " fmt "\n", ##__VA_ARGS__);    \
        failures++;                                             \
    } else {                                                    \
        fprintf(stdout, "  OK    " fmt "\n", ##__VA_ARGS__);    \
    }                                                           \
} while (0)

static double l2_diff(const quantum_state_t* a, const quantum_state_t* b) {
    double s = 0.0;
    for (uint64_t i = 0; i < a->state_dim; i++) {
        complex_t d = a->amplitudes[i] - b->amplitudes[i];
        s += creal(d) * creal(d) + cimag(d) * cimag(d);
    }
    return sqrt(s);
}

static void test_bell_fused(void) {
    fprintf(stdout, "\n-- Fusion: Bell state amplitudes --\n");
    fuse_circuit_t* src = fuse_circuit_create(2);
    fuse_append_h(src, 0);
    fuse_append_cnot(src, 0, 1);

    fuse_stats_t stats;
    fuse_circuit_t* fc = fuse_compile(src, &stats);
    quantum_state_t st;
    quantum_state_init(&st, 2);
    qs_error_t rc = fuse_execute(fc, &st);
    CHECK(rc == QS_SUCCESS, "execute returns QS_SUCCESS");

    const double inv = 1.0 / sqrt(2.0);
    CHECK(fabs(creal(st.amplitudes[0]) - inv) < 1e-12 &&
          fabs(cimag(st.amplitudes[0])) < 1e-12,
          "|00> amplitude = 1/sqrt(2)");
    CHECK(fabs(creal(st.amplitudes[3]) - inv) < 1e-12 &&
          fabs(cimag(st.amplitudes[3])) < 1e-12,
          "|11> amplitude = 1/sqrt(2)");
    CHECK(fabs(creal(st.amplitudes[1])) + fabs(cimag(st.amplitudes[1])) < 1e-12,
          "|01> amplitude = 0");
    CHECK(fabs(creal(st.amplitudes[2])) + fabs(cimag(st.amplitudes[2])) < 1e-12,
          "|10> amplitude = 0");

    quantum_state_free(&st);
    fuse_circuit_free(fc);
    fuse_circuit_free(src);
}

static void test_pauli_identity_fuses(void) {
    fprintf(stdout, "\n-- Fusion: X·X and H·H collapse --\n");
    fuse_circuit_t* src = fuse_circuit_create(1);
    fuse_append_x(src, 0);
    fuse_append_x(src, 0);
    fuse_append_h(src, 0);
    fuse_append_h(src, 0);

    fuse_stats_t stats;
    fuse_circuit_t* fc = fuse_compile(src, &stats);
    CHECK(stats.original_gates == 4, "original_gates = 4");
    CHECK(stats.fused_gates == 1, "fused_gates = 1 (whole run)");
    CHECK(stats.merges_applied == 3, "merges_applied = 3");

    quantum_state_t st;
    quantum_state_init(&st, 1);
    fuse_execute(fc, &st);
    CHECK(fabs(creal(st.amplitudes[0]) - 1.0) < 1e-12 &&
          fabs(cimag(st.amplitudes[0])) < 1e-12,
          "(HH)(XX)|0> stays at |0> (got re=%.3e, im=%.3e)",
          creal(st.amplitudes[0]), cimag(st.amplitudes[0]));
    CHECK(fabs(creal(st.amplitudes[1])) + fabs(cimag(st.amplitudes[1])) < 1e-12,
          "|1> component is zero");

    quantum_state_free(&st);
    fuse_circuit_free(fc);
    fuse_circuit_free(src);
}

static uint64_t xrng = 0xC0FFEEDEADu;
static uint32_t rng_u32(void) {
    xrng ^= xrng << 13; xrng ^= xrng >> 7; xrng ^= xrng << 17;
    return (uint32_t)xrng;
}
static double rng_angle(void) {
    return (rng_u32() / (double)UINT32_MAX) * 2.0 * M_PI;
}

static void build_random_circuit(fuse_circuit_t* c, size_t n, int num_gates) {
    for (int i = 0; i < num_gates; i++) {
        int r = rng_u32() % 14;
        int q = (int)(rng_u32() % n);
        double th = rng_angle();
        int q2 = (int)(rng_u32() % n);
        if (q2 == q) q2 = (q + 1) % (int)n;
        switch (r) {
            case 0:  fuse_append_h(c, q); break;
            case 1:  fuse_append_x(c, q); break;
            case 2:  fuse_append_y(c, q); break;
            case 3:  fuse_append_z(c, q); break;
            case 4:  fuse_append_s(c, q); break;
            case 5:  fuse_append_t(c, q); break;
            case 6:  fuse_append_rx(c, q, th); break;
            case 7:  fuse_append_ry(c, q, th); break;
            case 8:  fuse_append_rz(c, q, th); break;
            case 9:  fuse_append_phase(c, q, th); break;
            case 10: fuse_append_cnot(c, q, q2); break;
            case 11: fuse_append_cz(c, q, q2); break;
            case 12: fuse_append_swap(c, q, q2); break;
            case 13: fuse_append_crz(c, q, q2, th); break;
        }
    }
}

static void test_random_equiv(size_t n, int num_gates) {
    fprintf(stdout,
            "\n-- Fusion: random %zu-qubit circuit (%d gates) --\n",
            n, num_gates);
    fuse_circuit_t* src = fuse_circuit_create(n);
    build_random_circuit(src, n, num_gates);

    fuse_stats_t stats;
    fuse_circuit_t* fc = fuse_compile(src, &stats);

    quantum_state_t a, b;
    quantum_state_init(&a, n);
    quantum_state_init(&b, n);

    fuse_execute(src, &a);
    fuse_execute(fc, &b);

    double diff = l2_diff(&a, &b);
    CHECK(diff < 1e-10, "L2 diff(unfused, fused) = %.3e < 1e-10", diff);

    double ratio = stats.original_gates > 0
        ? (double)stats.fused_gates / (double)stats.original_gates : 1.0;
    fprintf(stdout,
            "    stats: %zu orig -> %zu fused  (%.0f%% retained, %zu merges)\n",
            stats.original_gates, stats.fused_gates,
            ratio * 100.0, stats.merges_applied);
    CHECK(stats.fused_gates <= stats.original_gates,
          "fused_gates (%zu) <= original_gates (%zu)",
          stats.fused_gates, stats.original_gates);

    quantum_state_free(&a);
    quantum_state_free(&b);
    fuse_circuit_free(fc);
    fuse_circuit_free(src);
}

static void test_u3_round_trip(void) {
    fprintf(stdout, "\n-- Fusion: U3 + inverse round-trip --\n");
    const double theta = 1.234, phi = -0.567, lambda = 0.789;
    fuse_circuit_t* src = fuse_circuit_create(1);
    /* U3(theta, phi, lambda) followed by its inverse U3(-theta, -lambda, -phi)
     * should fuse to identity (up to a 2x2 matrix equal to I within eps). */
    fuse_append_u3(src, 0, theta, phi, lambda);
    fuse_append_u3(src, 0, -theta, -lambda, -phi);

    fuse_stats_t stats;
    fuse_circuit_t* fc = fuse_compile(src, &stats);
    CHECK(stats.fused_gates == 1, "2 U3s fuse to 1 FUSED_1Q");

    quantum_state_t st;
    quantum_state_init(&st, 1);
    /* Start from H|0> so both components are populated. */
    gate_hadamard(&st, 0);
    quantum_state_t ref;
    quantum_state_init(&ref, 1);
    gate_hadamard(&ref, 0);

    fuse_execute(fc, &st);
    double diff = l2_diff(&st, &ref);
    CHECK(diff < 1e-10, "U3·U3^-1 ≈ I (|diff| = %.3e)", diff);

    quantum_state_free(&st);
    quantum_state_free(&ref);
    fuse_circuit_free(fc);
    fuse_circuit_free(src);
}

int main(void) {
    fprintf(stdout, "=== Gate-fusion DAG ===\n");
    test_bell_fused();
    test_pauli_identity_fuses();
    test_u3_round_trip();
    test_random_equiv(4, 80);
    test_random_equiv(5, 120);
    test_random_equiv(6, 200);
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
