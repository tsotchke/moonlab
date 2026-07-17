/*
 * t_measure.c -- measurement / normalization at near-epsilon norms.
 *
 * Probes the < 1e-15 (MEASUREMENT_SMALL_NORM) collapse guards and the
 * normalize guard, forcing collapse onto (numerically) zero-weight
 * outcomes.  The contract we assert:
 *   - the guard trips cleanly (documented error return), and
 *   - NO NaN/Inf is ever written into the amplitude buffer, and
 *   - just ABOVE the guard the renormalization is correct (unit norm).
 *
 * Links the prebuilt library.
 */
#include "numerical_common.h"
#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"
#include "../../src/quantum/measurement.h"

static int any_bad(const quantum_state_t *st) {
    for (size_t i = 0; i < st->state_dim; i++)
        if (nc_is_bad_c(st->amplitudes[i])) return 1;
    return 0;
}
static double statenorm(const quantum_state_t *st) {
    double n = 0.0;
    for (size_t i = 0; i < st->state_dim; i++) {
        complex_t a = st->amplitudes[i];
        n += creal(a)*creal(a) + cimag(a)*cimag(a);
    }
    return n;
}

/* Build normalized 2-qubit state a|00> + b|01> (bit0 carries weight |b|^2). */
static int build(quantum_state_t *st, double b_re) {
    complex_t amp[4] = { 1.0, b_re, 0.0, 0.0 };
    return quantum_state_from_amplitudes(st, amp, 4) == QS_SUCCESS;
}

/* Below the guard: |b|^2 = 1e-18. Forcing outcome 1 must NOT NaN. */
static void below_guard_single(void) {
    quantum_state_t st;
    if (!build(&st, 1e-9)) { NC_MISS("below single build"); return; }
    double p1 = measurement_probability_one(&st, 0);
    NC_INFO("below-guard single: P(q0=1)=%.3e", p1);
    int rc = measurement_single_qubit(&st, 0, 1e-30); /* forces result=1 */
    nc_expect("below single: no NaN in state", !any_bad(&st));
    nc_expect("below single: guard tripped (rc<0)", rc < 0);
    quantum_state_free(&st);
}

/* Just above the guard: |b|^2 = 1e-14 > 1e-15. Forcing outcome 1 must
 * renormalize to a valid unit-norm state. */
static void above_guard_single(void) {
    quantum_state_t st;
    if (!build(&st, 1e-7)) { NC_MISS("above single build"); return; }
    int rc = measurement_single_qubit(&st, 0, 1e-20); /* forces result=1 */
    nc_expect("above single: rc==1", rc == 1);
    nc_expect("above single: no NaN", !any_bad(&st));
    nc_close("above single: renormalized", statenorm(&st), 1.0, 1e-9);
    /* all weight now on |01> */
    nc_close("above single: |amp[1]|", cabs(st.amplitudes[1]), 1.0, 1e-9);
    quantum_state_free(&st);
}

/* measurement_partial onto a near-zero outcome must not NaN. */
static void below_guard_partial(void) {
    quantum_state_t st;
    if (!build(&st, 1e-9)) { NC_MISS("partial build"); return; }
    int q[1] = {0};
    /* random_value tiny -> selects outcome bit0=1 if it carries any weight;
     * with cumulative rounding the code falls back to last-nonzero outcome. */
    uint64_t r = measurement_partial(&st, q, 1, 1e-30);
    (void)r;
    nc_expect("partial: no NaN in state", !any_bad(&st));
    quantum_state_free(&st);
}

/* weak_z forced onto a near-zero outcome. */
static void weak_z_edge(void) {
    quantum_state_t st;
    if (!build(&st, 1e-9)) { NC_MISS("weakz build"); return; }
    int outcome = -99;
    /* uniform huge -> picks outcome 1 (the "-" branch); with |01> weight
     * tiny and strength 0 the "-" prob on bit0 is small -> exercises guard. */
    qs_error_t rc = measurement_weak_z(&st, 0, 0.0, 1.0, &outcome);
    nc_expect("weakz: no NaN in state", !any_bad(&st));
    nc_expect("weakz: rc is SUCCESS or NOT_NORMALIZED",
              rc == QS_SUCCESS || rc == QS_ERROR_NOT_NORMALIZED);
    if (rc == QS_SUCCESS)
        nc_close("weakz: normalized", statenorm(&st), 1.0, 1e-9);
    quantum_state_free(&st);
}

/* normalize on a near-zero-norm state must guard, not divide by ~0. */
static void normalize_near_zero(void) {
    quantum_state_t st;
    if (quantum_state_init(&st, 2) != QS_SUCCESS) { NC_MISS("normz init"); return; }
    for (size_t i = 0; i < st.state_dim; i++) st.amplitudes[i] = 1e-200; /* norm^2 underflows */
    qs_error_t rc = quantum_state_normalize(&st);
    nc_expect("normalize near-zero: no NaN", !any_bad(&st));
    nc_expect("normalize near-zero: returns error", rc != QS_SUCCESS);
    quantum_state_free(&st);
}

/* measurement_all_qubits determinism on a peaked state. */
static void all_qubits_peaked(void) {
    quantum_state_t st;
    if (!build(&st, 1e-9)) { NC_MISS("allq build"); return; }
    uint64_t out = measurement_all_qubits(&st, 0.5);
    nc_expect("all_qubits: outcome is |00>", out == 0);
    nc_expect("all_qubits: no NaN", !any_bad(&st));
    nc_close("all_qubits: collapsed norm", statenorm(&st), 1.0, 1e-12);
    quantum_state_free(&st);
}

int main(void) {
    nc_begin("measure_edge");
    below_guard_single();
    above_guard_single();
    below_guard_partial();
    weak_z_edge();
    normalize_near_zero();
    all_qubits_peaked();
    return nc_end();
}
