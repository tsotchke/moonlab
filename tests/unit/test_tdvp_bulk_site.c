/**
 * @file test_tdvp_bulk_site.c
 * @brief Real-time TDVP bulk-site correctness on chains > 2 sites.
 *
 * The old forward-only integrator (two forward two-site half-sweeps, no backward
 * one-site sub-step) over-propagated the site shared between adjacent two-site
 * blocks, so the interior/bulk dynamics were wrong for n >= 3 and the error grew
 * with the number of bulk sites (single-site magnetization drifting away from
 * the exact value, e.g. <Z0>(0.4) 0.712 -> 0.169 -> -0.42 as n grows).  The
 * genuine projector-splitting integrator (forward exp(-iH^2s dt/2), split,
 * backward exp(+iH^1s dt/2)) reproduces the exact dynamics.
 *
 * This evolves |0>^n under the TFIM H = -sum Z_i Z_{i+1} - h sum X_i to t=0.4 and
 * compares every single-site <Z_i> to an independent dense integration of
 * exp(-iHt) (fine RK4), at n=4 and n=12, tolerance 1e-6.  It fails on the
 * forward-only integrator (bulk sites off by ~0.1-1) and passes on the
 * projector-splitting one (agreement ~1e-6 or better).
 */

#include "../../src/algorithms/tensor_network/tdvp.h"
#include "../../src/algorithms/tensor_network/dmrg.h"
#include "../../src/algorithms/tensor_network/tn_state.h"
#include "../../src/algorithms/tensor_network/tn_gates.h"
#include "../../src/algorithms/tensor_network/tn_measurement.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL " fmt "\n", ##__VA_ARGS__); \
        failures++; \
    } \
} while (0)

/* dense H = -J sum Z_i Z_{i+1} - h sum X_i, qubit i occupies bit `bp[i]`. */
static void applyH(const double complex *psi, double complex *out, int N,
                   double J, double h, const int *bp) {
    int dim = 1 << N;
    for (int b = 0; b < dim; b++) {
        double zz = 0.0;
        for (int i = 0; i + 1 < N; i++) {
            int zi = ((b >> bp[i]) & 1) ? -1 : 1;
            int zj = ((b >> bp[i + 1]) & 1) ? -1 : 1;
            zz += (double)(zi * zj);
        }
        double complex v = -J * zz * psi[b];
        for (int i = 0; i < N; i++) v += -h * psi[b ^ (1 << bp[i])];
        out[b] = v;
    }
}
static void rk4(double complex *psi, int N, double J, double h, const int *bp, double dt) {
    int dim = 1 << N;
    double complex *k1 = calloc(dim, sizeof(double complex));
    double complex *k2 = calloc(dim, sizeof(double complex));
    double complex *k3 = calloc(dim, sizeof(double complex));
    double complex *k4 = calloc(dim, sizeof(double complex));
    double complex *t  = calloc(dim, sizeof(double complex));
    applyH(psi, k1, N, J, h, bp); for (int b = 0; b < dim; b++) k1[b] *= -I;
    for (int b = 0; b < dim; b++) t[b] = psi[b] + 0.5 * dt * k1[b];
    applyH(t, k2, N, J, h, bp);  for (int b = 0; b < dim; b++) k2[b] *= -I;
    for (int b = 0; b < dim; b++) t[b] = psi[b] + 0.5 * dt * k2[b];
    applyH(t, k3, N, J, h, bp);  for (int b = 0; b < dim; b++) k3[b] *= -I;
    for (int b = 0; b < dim; b++) t[b] = psi[b] + dt * k3[b];
    applyH(t, k4, N, J, h, bp);  for (int b = 0; b < dim; b++) k4[b] *= -I;
    for (int b = 0; b < dim; b++) psi[b] += (dt / 6.0) * (k1[b] + 2 * k2[b] + 2 * k3[b] + k4[b]);
    free(k1); free(k2); free(k3); free(k4); free(t);
}
static double dense_Z(const double complex *psi, int N, int bit) {
    int dim = 1 << N; double z = 0.0;
    for (int b = 0; b < dim; b++) {
        double s = ((b >> bit) & 1) ? -1.0 : 1.0;
        z += s * (creal(psi[b]) * creal(psi[b]) + cimag(psi[b]) * cimag(psi[b]));
    }
    return z;
}
static int probe_bit(uint32_t n, uint32_t q) {
    tn_state_config_t c = tn_state_config_create(8, 1e-14);
    tn_mps_state_t *s = tn_mps_create_zero(n, &c);
    tn_apply_x(s, q);
    uint64_t dim = 1ULL << n;
    double complex *sv = calloc(dim, sizeof(double complex));
    tn_mps_to_statevector(s, sv);
    int pos = -1;
    for (uint64_t b = 0; b < dim; b++) {
        double p = creal(sv[b]) * creal(sv[b]) + cimag(sv[b]) * cimag(sv[b]);
        if (p > 0.5) for (int k = 0; k < (int)n; k++) if (b == (1ULL << k)) pos = k;
    }
    free(sv); tn_mps_free(s); return pos;
}

static void run_case(uint32_t n, tdvp_variant_t variant, const char *label) {
    const double h = 1.0, T = 0.4;
    int dim = 1 << n;

    int *bp = malloc(n * sizeof(int));
    for (uint32_t q = 0; q < n; q++) bp[q] = probe_bit(n, q);

    /* dense reference: fine RK4 from |0>^n */
    double complex *psi = calloc(dim, sizeof(double complex));
    psi[0] = 1.0;
    const int sub = 4000;
    for (int k = 0; k < sub; k++) rk4(psi, (int)n, 1.0, h, bp, T / sub);

    /* TDVP */
    mpo_t *mpo = mpo_tfim_create(n, 1.0, h);
    tn_state_config_t scfg = tn_state_config_create(64, 1e-14);
    tn_mps_state_t *mps = tn_mps_create_zero(n, &scfg);
    tdvp_config_t cfg = tdvp_config_default();
    cfg.evolution_type = TDVP_REAL_TIME;
    cfg.dt = 0.02;
    cfg.max_bond_dim = 64;
    cfg.normalize = false;
    cfg.variant = variant;
    tdvp_engine_t *eng = tdvp_engine_create(mps, mpo, &cfg);
    int steps = (int)(T / cfg.dt + 0.5);
    tdvp_result_t r = {0};
    for (int s = 0; s < steps; s++) {
        int rc = tdvp_step(eng, &r);
        CHECK(rc == 0, "%s: tdvp_step rc=%d", label, rc);
        if (rc != 0) break;
    }

    double max_err = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        double zt = tn_expectation_z(mps, i);
        double zr = dense_Z(psi, (int)n, bp[i]);
        double e = fabs(zt - zr);
        if (e > max_err) max_err = e;
    }
    double norm = tn_mps_norm(mps);
    fprintf(stdout, "  %s n=%u: <Z0> TDVP=%+.6f dense=%+.6f  max_i|dZ_i|=%.3e  norm=%.6f\n",
            label, n, tn_expectation_z(mps, 0), dense_Z(psi, (int)n, bp[0]), max_err, norm);
    CHECK(max_err < 1e-6, "%s n=%u: bulk-site magnetization off dense by %.3e (>1e-6)",
          label, n, max_err);
    CHECK(fabs(norm - 1.0) < 1e-6, "%s n=%u: norm %.6f drifted from 1", label, n, norm);

    tdvp_result_clear(&r);
    tdvp_engine_free(eng);
    tn_mps_free(mps);
    mpo_free(mpo);
    free(psi);
    free(bp);
}

int main(void) {
    fprintf(stdout, "=== TDVP bulk-site <Z_i>(0.4) vs dense exp(-iHt) ===\n");
    run_case(4,  TDVP_TWO_SITE, "2-site");
    run_case(12, TDVP_TWO_SITE, "2-site");
    run_case(4,  TDVP_ONE_SITE, "1-site");

    if (failures == 0) { fprintf(stdout, "PASS\n"); return 0; }
    fprintf(stderr, "FAIL: %d assertion failure(s)\n", failures);
    return 1;
}
