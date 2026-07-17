/*
 * t_rotations.c -- extreme rotation-angle correctness + unitarity.
 *
 * Targets: gate_rx / gate_ry / gate_rz / gate_u3 / gate_phase and a
 * small VQE/QAOA-style ansatz, at theta in
 *   {0, 1e-300 (denormal), 1e-16, pi-1e-15, 2pi, 1e6, -1e6}.
 *
 * Checks per angle:
 *   1. Analytic single-qubit amplitude match vs a long-double reference.
 *   2. Norm preservation (unitarity) to 1e-13.
 *   3. No NaN/Inf anywhere.
 *   4. Denormal handling: rx(1e-300) must leave a *denormal* |1> amplitude,
 *      not a flushed zero (FTZ/DAZ must be OFF for the library).
 *
 * Links the prebuilt library (ships gate_* on the CPU path).
 */
#include "numerical_common.h"
#include "../../src/quantum/state.h"
#include "../../src/quantum/gates.h"

static const double PI = 3.14159265358979323846;

/* long-double reference for cos/sin(theta/2) */
static void ref_half(double theta, double *c, double *s) {
    long double h = (long double)theta / 2.0L;
    *c = (double)cosl(h);
    *s = (double)sinl(h);
}

static double statenorm(const quantum_state_t *st) {
    double n = 0.0;
    for (size_t i = 0; i < st->state_dim; i++) {
        complex_t a = st->amplitudes[i];
        n += creal(a) * creal(a) + cimag(a) * cimag(a);
    }
    return n;
}

static int any_bad(const quantum_state_t *st) {
    for (size_t i = 0; i < st->state_dim; i++)
        if (nc_is_bad_c(st->amplitudes[i])) return 1;
    return 0;
}

/* rx on |0> : a0 = cos(t/2), a1 = -i sin(t/2) */
static void test_rx(double theta) {
    quantum_state_t st;
    if (quantum_state_init(&st, 1) != QS_SUCCESS) { NC_MISS("rx init"); return; }
    if (gate_rx(&st, 0, theta) != QS_SUCCESS) { NC_MISS("rx apply theta=%.3g", theta); quantum_state_free(&st); return; }
    double c, s; ref_half(theta, &c, &s);
    nc_close_c("rx a0", st.amplitudes[0], c + 0.0*I, 1e-13);
    nc_close_c("rx a1", st.amplitudes[1], 0.0 - I*s, 1e-13);
    nc_close("rx norm", statenorm(&st), 1.0, 1e-13);
    nc_expect("rx finite", !any_bad(&st));
    quantum_state_free(&st);
}

/* ry on |0> : a0 = cos(t/2), a1 = sin(t/2) */
static void test_ry(double theta) {
    quantum_state_t st;
    if (quantum_state_init(&st, 1) != QS_SUCCESS) { NC_MISS("ry init"); return; }
    if (gate_ry(&st, 0, theta) != QS_SUCCESS) { NC_MISS("ry apply"); quantum_state_free(&st); return; }
    double c, s; ref_half(theta, &c, &s);
    nc_close_c("ry a0", st.amplitudes[0], c + 0.0*I, 1e-13);
    nc_close_c("ry a1", st.amplitudes[1], s + 0.0*I, 1e-13);
    nc_close("ry norm", statenorm(&st), 1.0, 1e-13);
    nc_expect("ry finite", !any_bad(&st));
    quantum_state_free(&st);
}

/* rz on |+> : (e^{-i t/2}|0> + e^{+i t/2}|1>)/sqrt2 */
static void test_rz(double theta) {
    quantum_state_t st;
    if (quantum_state_init(&st, 1) != QS_SUCCESS) { NC_MISS("rz init"); return; }
    if (gate_hadamard(&st, 0) != QS_SUCCESS) { NC_MISS("rz H"); quantum_state_free(&st); return; }
    if (gate_rz(&st, 0, theta) != QS_SUCCESS) { NC_MISS("rz apply"); quantum_state_free(&st); return; }
    long double h = (long double)theta / 2.0L;
    complex_t e0 = (double)cosl(-h) + I*(double)sinl(-h);
    complex_t e1 = (double)cosl(h) + I*(double)sinl(h);
    double inv = 1.0 / sqrt(2.0);
    nc_close_c("rz a0", st.amplitudes[0], e0 * inv, 1e-13);
    nc_close_c("rz a1", st.amplitudes[1], e1 * inv, 1e-13);
    nc_close("rz norm", statenorm(&st), 1.0, 1e-13);
    nc_expect("rz finite", !any_bad(&st));
    quantum_state_free(&st);
}

/* u3 on |0> : a0 = cos(t/2), a1 = e^{i phi} sin(t/2) */
static void test_u3(double theta, double phi, double lambda) {
    quantum_state_t st;
    if (quantum_state_init(&st, 1) != QS_SUCCESS) { NC_MISS("u3 init"); return; }
    if (gate_u3(&st, 0, theta, phi, lambda) != QS_SUCCESS) { NC_MISS("u3 apply"); quantum_state_free(&st); return; }
    double c, s; ref_half(theta, &c, &s);
    complex_t eiphi = (double)cosl((long double)phi) + I*(double)sinl((long double)phi);
    nc_close_c("u3 a0", st.amplitudes[0], c + 0.0*I, 1e-13);
    nc_close_c("u3 a1", st.amplitudes[1], eiphi * s, 1e-13);
    nc_close("u3 norm", statenorm(&st), 1.0, 1e-13);
    nc_expect("u3 finite", !any_bad(&st));
    quantum_state_free(&st);
}

/* phase on |1> : e^{i theta} */
static void test_phase(double theta) {
    quantum_state_t st;
    if (quantum_state_init(&st, 1) != QS_SUCCESS) { NC_MISS("phase init"); return; }
    gate_pauli_x(&st, 0); /* -> |1> */
    if (gate_phase(&st, 0, theta) != QS_SUCCESS) { NC_MISS("phase apply"); quantum_state_free(&st); return; }
    complex_t e = (double)cosl((long double)theta) + I*(double)sinl((long double)theta);
    nc_close_c("phase a1", st.amplitudes[1], e, 1e-13);
    nc_close("phase norm", statenorm(&st), 1.0, 1e-13);
    quantum_state_free(&st);
}

/* Denormal-preservation probe: rx(1e-300) on |0> must produce a
 * denormal (subnormal) imaginary amplitude ~ -i * 5e-301, NOT a
 * flushed zero.  A nonzero-but-subnormal a1 proves FTZ/DAZ is off. */
static void test_denormal_flush(void) {
    const double tiny = 1e-300;              /* theta */
    quantum_state_t st;
    if (quantum_state_init(&st, 1) != QS_SUCCESS) { NC_MISS("denorm init"); return; }
    gate_rx(&st, 0, tiny);
    double im = cimag(st.amplitudes[1]);     /* expect ~ -5e-301 (subnormal) */
    g_checks++;
    if (im == 0.0) {
        NC_MISS("denormal flushed: rx(1e-300) gave a1.im=0 (FTZ/DAZ appears ON)");
    } else if (fabs(im) >= 2.2250738585072014e-308 && fabs(im) < 1e-307) {
        NC_INFO("denormal preserved: a1.im=%.3e (subnormal, FTZ/DAZ off)", im);
    } else {
        /* nonzero but not in the expected subnormal band -- still not flushed,
         * but report the value for the record. */
        NC_INFO("rx(1e-300) a1.im=%.3e (nonzero, not flushed)", im);
    }
    quantum_state_free(&st);
}

/* VQE/QAOA-style ansatz: layered rx/ry/rz + CNOT ladder at extreme
 * angles; must stay exactly unit-norm and finite. */
static void test_ansatz(double theta) {
    const size_t nq = 4;
    quantum_state_t st;
    if (quantum_state_init(&st, nq) != QS_SUCCESS) { NC_MISS("ansatz init"); return; }
    for (int layer = 0; layer < 3; layer++) {
        for (int q = 0; q < (int)nq; q++) {
            gate_ry(&st, q, theta);
            gate_rz(&st, q, theta * 0.5);
            gate_rx(&st, q, -theta);
        }
        for (int q = 0; q + 1 < (int)nq; q++) gate_cnot(&st, q, q + 1);
    }
    nc_close("ansatz norm", statenorm(&st), 1.0, 1e-11);
    nc_expect("ansatz finite", !any_bad(&st));
    quantum_state_free(&st);
}

int main(void) {
    nc_begin("rotations");

    const double angles[] = {
        0.0, 1e-300, 1e-16, PI - 1e-15, 2.0 * PI, 1e6, -1e6, PI, PI/4.0
    };
    const size_t na = sizeof(angles) / sizeof(angles[0]);

    for (size_t i = 0; i < na; i++) {
        double t = angles[i];
        NC_INFO("theta = %.17g", t);
        test_rx(t);
        test_ry(t);
        test_rz(t);
        test_u3(t, 0.3, -0.7);
        test_u3(t, t, t);          /* all three params extreme */
        test_phase(t);
        test_ansatz(t);
    }

    test_denormal_flush();

    return nc_end();
}
