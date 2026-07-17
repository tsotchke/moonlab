/**
 * @file test_measurement_edge.c
 * @brief Regression: measurement must never collapse onto a zero-amplitude
 *        outcome due to floating-point rounding.
 *
 * The old code, when the sampled random value landed at or past the cumulative
 * probability total (which rounding routinely causes), fell back to a fixed
 * |1...1> (measurement_all_qubits, quantum_measure_all_fast) or to outcome 0
 * (measurement_partial) regardless of whether that outcome carried any
 * amplitude. This corrupts the collapsed state. The fix selects the last
 * basis state / outcome that actually carries amplitude.
 */

#include "../../src/quantum/state.h"
#include "../../src/quantum/measurement.h"
#include <math.h>
#include <stdio.h>
#include <complex.h>

static int failures = 0;
#define CHECK(cond, ...) do {                                   \
    if (!(cond)) { fprintf(stderr, "  FAIL  " __VA_ARGS__);     \
                   fprintf(stderr, "\n"); failures++; }         \
    else { fprintf(stdout, "  OK    " __VA_ARGS__);             \
           fprintf(stdout, "\n"); }                             \
} while (0)

/* Prepare the computational basis state |k> in an n-qubit register. */
static void set_basis(quantum_state_t *s, uint64_t k) {
    for (uint64_t i = 0; i < s->state_dim; i++) s->amplitudes[i] = 0.0;
    s->amplitudes[k] = 1.0 + 0.0 * I;
}

int main(void) {
    quantum_state_t s;
    quantum_state_init(&s, 3);   /* state_dim = 8 */
    const uint64_t k = 2;        /* deliberately != state_dim-1 (== 7) */

    /* measurement_all_qubits: random_value == 1.0 forces the rounding-edge
     * fallback. The old code returned state_dim-1 (== 7, zero amplitude here);
     * the fix must return the only populated basis state, k. */
    set_basis(&s, k);
    uint64_t r = measurement_all_qubits(&s, 1.0);
    CHECK(r == k, "measurement_all_qubits picks the populated basis state (got %llu, want %llu)",
          (unsigned long long)r, (unsigned long long)k);
    CHECK(cabs(s.amplitudes[k]) > 0.999,
          "collapsed onto |k> (|amp[k]| = %.6f)", cabs(s.amplitudes[k]));
    CHECK(cabs(s.amplitudes[s.state_dim - 1]) < 1e-12,
          "did NOT collapse onto the zero-amplitude |1..1>");

    /* measurement_partial on all three qubits, same edge. Measuring every
     * qubit must reproduce k, not the default outcome 0. */
    set_basis(&s, k);
    int qubits[3] = {0, 1, 2};
    uint64_t po = measurement_partial(&s, qubits, 3, 1.0);
    CHECK(po == k, "measurement_partial picks the populated outcome (got %llu, want %llu)",
          (unsigned long long)po, (unsigned long long)k);

    /* Sanity: a normal (interior) random value still samples correctly. */
    set_basis(&s, k);
    uint64_t r2 = measurement_all_qubits(&s, 0.5);
    CHECK(r2 == k, "interior sample still returns k (got %llu)", (unsigned long long)r2);

    /* A genuine superposition collapses onto a populated outcome for a
     * boundary random value, never a zero-amplitude one. */
    quantum_state_reset(&s);
    s.amplitudes[0] = 0.0;
    s.amplitudes[1] = 1.0 / sqrt(2.0);   /* |001> */
    s.amplitudes[4] = 1.0 / sqrt(2.0);   /* |100> */
    uint64_t r3 = measurement_all_qubits(&s, 1.0);
    CHECK(r3 == 1 || r3 == 4,
          "boundary sample of a superposition lands on a populated state (got %llu)",
          (unsigned long long)r3);

    quantum_state_free(&s);

    if (failures == 0) { printf("test_measurement_edge: ALL PASSED\n"); return 0; }
    fprintf(stderr, "test_measurement_edge: %d FAILED\n", failures);
    return 1;
}
