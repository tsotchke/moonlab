/**
 * @file test_povm_cache.c
 * @brief POVM completeness check is verified once and cached per-POVM.
 *
 * The completeness verification is O(K*D^3); it must run on the first
 * measurement_povm() call and be cached on the struct thereafter (the header
 * documents "checks completeness on the first call"). This test proves the
 * cache is actually consulted: after a successful first call we corrupt the
 * Kraus operators to be non-complete WITHOUT clearing the cache flag, and the
 * next call must still succeed -- which is only possible if completeness is
 * not re-checked.
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

static void prep_plus(quantum_state_t *s) {
    quantum_state_reset(s);
    s->amplitudes[0] = 1.0 / sqrt(2.0);
    s->amplitudes[1] = 1.0 / sqrt(2.0);
}

int main(void) {
    quantum_state_t s;
    quantum_state_init(&s, 1);   /* D = 2 */

    /* Projective Z POVM: K0 = |0><0|, K1 = |1><1|. Complete: K0^dag K0 +
     * K1^dag K1 = I. Row-major 2x2. */
    complex_t K0[4] = { 1.0, 0.0, 0.0, 0.0 };
    complex_t K1[4] = { 0.0, 0.0, 0.0, 1.0 };
    const complex_t *ops[2] = { K0, K1 };

    povm_t pv = { .num_outcomes = 2, .state_dim = 2, .kraus_ops = ops };
    CHECK(pv.completeness_checked == 0, "fresh POVM starts unchecked");

    prep_plus(&s);
    size_t out = 99;
    qs_error_t rc = measurement_povm(&s, &pv, 0.3, &out);
    CHECK(rc == QS_SUCCESS, "first measurement_povm on a complete POVM succeeds (rc=%d)", (int)rc);
    CHECK(pv.completeness_checked == 1, "completeness verdict is cached after the first call");
    CHECK(out == 0 || out == 1, "an outcome was selected (out=%zu)", out);

    /* Corrupt the operators so they no longer sum to I, but leave the cache
     * flag set. A re-check would now reject; the cached verdict must let the
     * call through. Use uniform = 0.0 so outcome 0 (still K0, p0 = 0.5 > 0 on
     * |+>) is selected and the collapse is well-defined. */
    K1[3] = 0.0;   /* now sum K^dag K = diag(1,0) != I */
    prep_plus(&s);
    rc = measurement_povm(&s, &pv, 0.0, &out);
    CHECK(rc == QS_SUCCESS,
          "cached-complete POVM is not re-verified (still succeeds despite corruption, rc=%d)",
          (int)rc);

    /* A POVM that is incomplete from the outset is rejected on the first call
     * and the rejection is cached (still rejected on a second call). */
    complex_t P0[4] = { 1.0, 0.0, 0.0, 0.0 };   /* |0><0| only: sum != I */
    const complex_t *ops2[1] = { P0 };
    povm_t bad = { .num_outcomes = 1, .state_dim = 2, .kraus_ops = ops2 };
    prep_plus(&s);
    rc = measurement_povm(&s, &bad, 0.5, &out);
    CHECK(rc == QS_ERROR_NOT_NORMALIZED, "incomplete POVM rejected on first call (rc=%d)", (int)rc);
    CHECK(bad.completeness_checked == 1, "rejection verdict is cached");
    prep_plus(&s);
    rc = measurement_povm(&s, &bad, 0.5, &out);
    CHECK(rc == QS_ERROR_NOT_NORMALIZED, "incomplete POVM still rejected via cache (rc=%d)", (int)rc);

    quantum_state_free(&s);
    if (failures == 0) { printf("test_povm_cache: ALL PASSED\n"); return 0; }
    fprintf(stderr, "test_povm_cache: %d FAILED\n", failures);
    return 1;
}
