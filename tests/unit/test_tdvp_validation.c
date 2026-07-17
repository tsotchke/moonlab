/**
 * @file test_tdvp_validation.c
 * @brief Real-time two-site projector-splitting TDVP validated against
 *        analytically / exactly solvable references.
 *
 * This is the correctness gate for the genuine projector-splitting
 * integrator (forward two-site exp(-iH^2s dt/2), split, backward one-site
 * exp(+iH^1s dt/2)).  The previous "two forward half-sweeps, never
 * back-evolve" code fails both checks below because it over-propagates the
 * site shared between adjacent two-site blocks.
 *
 * Test 1 -- two-site transverse-field Rabi oscillation (analytic):
 *   H = -h (X_0 + X_1) (TFIM with J = 0).  From |00>, real-time evolution
 *   exp(-iHt) drives each qubit independently, giving the closed form
 *   <Z_i>(t) = cos(2 h t).  On two sites the two-site block IS the whole
 *   system, so TDVP is the exact propagator; agreement to ~1e-6 confirms
 *   the forward step and the SVD-split gauge.
 *
 * Test 2 -- three-site TFIM quench magnetization (exact dense reference):
 *   H = -J sum Z_i Z_{i+1} - h sum X_i.  From |000>, evolve real-time at
 *   full bond dimension and compare the total magnetization M(t) = sum_i
 *   <Z_i>(t) and the (conserved) energy against a dense RK4 integration of
 *   the 8-dimensional Schroedinger equation.  Three sites exercise the
 *   backward one-site sub-step and the sweep turning points; total
 *   magnetization is basis-relabeling invariant so the comparison needs no
 *   assumption about the MPS qubit-to-bit ordering.
 *
 * Reference: Haegeman, Lubich, Oseledets, Vandereycken, Verstraete,
 * "Unifying time evolution and optimization with matrix product states",
 * Phys. Rev. B 94, 165116 (2016).
 */

#include "../../src/algorithms/tensor_network/tdvp.h"
#include "../../src/algorithms/tensor_network/dmrg.h"
#include "../../src/algorithms/tensor_network/tn_state.h"
#include "../../src/algorithms/tensor_network/tn_measurement.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); \
        failures++; \
    } \
} while (0)

/* ------------------------------------------------------------------ */
/* Dense exact reference: d|psi>/dt = -i H |psi|, RK4.                 */
/* H = -J sum_{i} Z_i Z_{i+1} - h sum_i X_i on N qubits (open chain).  */
/* ------------------------------------------------------------------ */

static void apply_H(const double complex *psi, double complex *out,
                    int N, double J, double h) {
    int dim = 1 << N;
    for (int b = 0; b < dim; b++) {
        double zz = 0.0;
        for (int i = 0; i + 1 < N; i++) {
            int zi = ((b >> i) & 1) ? -1 : 1;
            int zj = ((b >> (i + 1)) & 1) ? -1 : 1;
            zz += (double)(zi * zj);
        }
        double complex val = -J * zz * psi[b];
        for (int i = 0; i < N; i++) {
            val += -h * psi[b ^ (1 << i)];
        }
        out[b] = val;
    }
}

/* f(psi) = -i H psi */
static void deriv(const double complex *psi, double complex *out,
                  int N, double J, double h) {
    int dim = 1 << N;
    apply_H(psi, out, N, J, h);
    for (int b = 0; b < dim; b++) out[b] *= -I;
}

static void rk4_step(double complex *psi, int N, double J, double h, double dt) {
    int dim = 1 << N;
    double complex *k1 = calloc(dim, sizeof(double complex));
    double complex *k2 = calloc(dim, sizeof(double complex));
    double complex *k3 = calloc(dim, sizeof(double complex));
    double complex *k4 = calloc(dim, sizeof(double complex));
    double complex *tmp = calloc(dim, sizeof(double complex));

    deriv(psi, k1, N, J, h);
    for (int b = 0; b < dim; b++) tmp[b] = psi[b] + 0.5 * dt * k1[b];
    deriv(tmp, k2, N, J, h);
    for (int b = 0; b < dim; b++) tmp[b] = psi[b] + 0.5 * dt * k2[b];
    deriv(tmp, k3, N, J, h);
    for (int b = 0; b < dim; b++) tmp[b] = psi[b] + dt * k3[b];
    deriv(tmp, k4, N, J, h);
    for (int b = 0; b < dim; b++)
        psi[b] += (dt / 6.0) * (k1[b] + 2.0 * k2[b] + 2.0 * k3[b] + k4[b]);

    free(k1); free(k2); free(k3); free(k4); free(tmp);
}

/* Total magnetization M = sum_i <Z_i> from a dense statevector. */
static double dense_magnetization(const double complex *psi, int N) {
    int dim = 1 << N;
    double M = 0.0;
    for (int i = 0; i < N; i++) {
        double zi = 0.0;
        for (int b = 0; b < dim; b++) {
            double z = ((b >> i) & 1) ? -1.0 : 1.0;
            double p = creal(psi[b]) * creal(psi[b]) + cimag(psi[b]) * cimag(psi[b]);
            zi += z * p;
        }
        M += zi;
    }
    return M;
}

static double dense_energy(const double complex *psi, int N, double J, double h) {
    int dim = 1 << N;
    double complex *Hpsi = calloc(dim, sizeof(double complex));
    apply_H(psi, Hpsi, N, J, h);
    double complex e = 0.0;
    for (int b = 0; b < dim; b++) e += conj(psi[b]) * Hpsi[b];
    free(Hpsi);
    return creal(e);
}

static double mps_magnetization(const tn_mps_state_t *mps, uint32_t n) {
    double M = 0.0;
    for (uint32_t i = 0; i < n; i++) M += tn_expectation_z(mps, i);
    return M;
}

/* ------------------------------------------------------------------ */
/* Test 1: analytic two-site Rabi oscillation.                        */
/* ------------------------------------------------------------------ */

static double test_rabi(void) {
    fprintf(stdout, "\n--- Test 1: two-site Rabi, H = -h(X0+X1), <Z>(t)=cos(2ht) ---\n");

    const uint32_t n = 2;
    const double J = 0.0;
    const double h = 0.7;

    mpo_t *mpo = mpo_tfim_create(n, J, h);
    CHECK(mpo != NULL, "mpo_tfim_create");
    if (!mpo) return 1.0;

    tn_state_config_t scfg = tn_state_config_create(16, 1e-14);
    tn_mps_state_t *mps = tn_mps_create_zero(n, &scfg);   /* |00> */
    CHECK(mps != NULL, "tn_mps_create_zero");
    if (!mps) { mpo_free(mpo); return 1.0; }

    tdvp_config_t cfg = tdvp_config_default();
    cfg.evolution_type = TDVP_REAL_TIME;
    cfg.dt = 0.02;
    cfg.max_bond_dim = 16;
    cfg.normalize = false;   /* real time is norm preserving */

    tdvp_engine_t *engine = tdvp_engine_create(mps, mpo, &cfg);
    CHECK(engine != NULL, "engine create");
    if (!engine) { tn_mps_free(mps); mpo_free(mpo); return 1.0; }

    const double T = 1.0;
    const int steps = 50;   /* dt = 0.02 */
    double max_err = 0.0;
    tdvp_result_t result = {0};

    for (int s = 1; s <= steps; s++) {
        int rc = tdvp_step(engine, &result);
        CHECK(rc == 0, "tdvp_step rc=%d step %d", rc, s);
        if (rc != 0) break;
        double t = s * cfg.dt;
        double z0 = tn_expectation_z(mps, 0);
        double exact = cos(2.0 * h * t);
        double err = fabs(z0 - exact);
        if (err > max_err) max_err = err;
    }

    double z0_final = tn_expectation_z(mps, 0);
    double exact_final = cos(2.0 * h * T);
    double norm = tn_mps_norm(mps);
    fprintf(stdout, "  t=%.2f: <Z0>_TDVP=%+.9f  cos(2hT)=%+.9f  |err|=%.3e\n",
            T, z0_final, exact_final, fabs(z0_final - exact_final));
    fprintf(stdout, "  max |<Z0> - cos(2ht)| over trajectory = %.3e\n", max_err);
    fprintf(stdout, "  norm(T) = %.12f (real-time should preserve norm)\n", norm);

    CHECK(max_err < 1e-6, "Rabi max error %.3e exceeds 1e-6", max_err);
    CHECK(fabs(norm - 1.0) < 1e-9, "norm drift %.3e exceeds 1e-9", fabs(norm - 1.0));

    tdvp_result_clear(&result);
    tdvp_engine_free(engine);
    tn_mps_free(mps);
    mpo_free(mpo);
    return max_err;
}

/* ------------------------------------------------------------------ */
/* Test 2: three-site TFIM quench vs dense RK4 reference.             */
/* ------------------------------------------------------------------ */

static double test_quench(void) {
    fprintf(stdout, "\n--- Test 2: 3-site TFIM quench, M(t)=sum<Z_i> vs dense RK4 ---\n");

    const uint32_t n = 3;
    const double J = 1.0;
    const double h = 0.8;

    mpo_t *mpo = mpo_tfim_create(n, J, h);
    CHECK(mpo != NULL, "mpo_tfim_create");
    if (!mpo) return 1.0;

    tn_state_config_t scfg = tn_state_config_create(16, 1e-14);
    tn_mps_state_t *mps = tn_mps_create_zero(n, &scfg);   /* |000> */
    CHECK(mps != NULL, "tn_mps_create_zero");
    if (!mps) { mpo_free(mpo); return 1.0; }

    /* Dense reference statevector, |000>. */
    int dim = 1 << n;
    double complex *psi = calloc(dim, sizeof(double complex));
    psi[0] = 1.0;

    double E0_ref = dense_energy(psi, n, J, h);

    tdvp_config_t cfg = tdvp_config_default();
    cfg.evolution_type = TDVP_REAL_TIME;
    cfg.dt = 0.05;
    cfg.max_bond_dim = 16;    /* full bond for 3 sites: no truncation */
    cfg.svd_cutoff = 1e-14;
    cfg.lanczos_max_iter = 60;
    cfg.lanczos_tol = 1e-13;
    cfg.normalize = false;

    tdvp_engine_t *engine = tdvp_engine_create(mps, mpo, &cfg);
    CHECK(engine != NULL, "engine create");
    if (!engine) { free(psi); tn_mps_free(mps); mpo_free(mpo); return 1.0; }

    const int steps = 12;               /* T = 0.6 */
    const int rk_sub = 100;             /* dense substeps per TDVP step */
    double max_M_err = 0.0;
    double max_E_drift = 0.0;
    tdvp_result_t result = {0};

    for (int s = 1; s <= steps; s++) {
        int rc = tdvp_step(engine, &result);
        CHECK(rc == 0, "tdvp_step rc=%d step %d", rc, s);
        if (rc != 0) break;

        /* Advance dense reference by the same dt with fine RK4 substeps. */
        for (int k = 0; k < rk_sub; k++)
            rk4_step(psi, n, J, h, cfg.dt / rk_sub);

        double M_tdvp = mps_magnetization(mps, n);
        double M_ref = dense_magnetization(psi, n);
        double M_err = fabs(M_tdvp - M_ref);
        if (M_err > max_M_err) max_M_err = M_err;

        double E_drift = fabs(result.energy - E0_ref);
        if (E_drift > max_E_drift) max_E_drift = E_drift;

        if (s % 3 == 0) {
            fprintf(stdout,
                    "  t=%.2f: M_TDVP=%+.9f  M_ref=%+.9f  |dM|=%.3e  E=%+.9f  |dE|=%.3e\n",
                    s * cfg.dt, M_tdvp, M_ref, M_err, result.energy, E_drift);
        }
    }

    fprintf(stdout, "  E0(ref) = %+.9f\n", E0_ref);
    fprintf(stdout, "  max |M_TDVP - M_ref| = %.3e\n", max_M_err);
    fprintf(stdout, "  max |E(t) - E0|      = %.3e\n", max_E_drift);

    /* Genuine projector-splitting TDVP at full bond reproduces exp(-iHt)
     * to integrator precision; the old two-forward-half-sweep code misses
     * the backward sub-step and drifts far past these thresholds. */
    CHECK(max_M_err < 1e-3, "magnetization error %.3e exceeds 1e-3", max_M_err);
    CHECK(max_E_drift < 1e-3, "energy drift %.3e exceeds 1e-3", max_E_drift);

    tdvp_result_clear(&result);
    tdvp_engine_free(engine);
    free(psi);
    tn_mps_free(mps);
    mpo_free(mpo);
    return max_M_err;
}

int main(void) {
    fprintf(stdout, "=== TDVP real-time projector-splitting validation ===\n");
    test_rabi();
    test_quench();

    if (failures == 0) {
        fprintf(stdout, "\nPASS\n");
        return 0;
    }
    fprintf(stderr, "\nFAIL: %d assertion failure(s)\n", failures);
    return 1;
}
