/**
 * @file test_qgt_exact_curvature.c
 * @brief Exact pointwise band geometry: qgt_dsigma_metric_curvature and
 *        qgt_curvature_at.
 *
 * Verifies, for the two-band QWZ model H(k) = d(k).sigma:
 *   1. Closed-form Omega_xy from qgt_dsigma_metric_curvature equals the
 *      skyrmion density (1/2) dhat.(dx dhat x dy dhat) to ~1e-13, and
 *      the metric g equals (1/4) dmu dhat . dnu dhat.
 *   2. The pointwise Omega_xy integrated over the BZ reproduces the
 *      integer Chern number of the existing FHS integrator qgt_berry_grid
 *      across the QWZ topological phases (C = 0, +1, -1).
 *   3. g and Omega agree with the projector-form QGT assembled from an
 *      INDEPENDENT exact 2x2 diagonalisation at k (explicit eigenvectors)
 *      to ~1e-13, and g is symmetric and positive semidefinite.
 *   4. The generic finite-difference path qgt_curvature_at agrees with the
 *      closed form to finite-difference accuracy.
 */

#include "../../src/algorithms/quantum_geometry/qgt.h"

#include <complex.h>
#include <math.h>
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

typedef double _Complex cx;

/* ---- QWZ analytic d(k) and its k-gradients ------------------------ */
static void qwz_d(const double k[2], double m, double d[3]) {
    d[0] = sin(k[0]);
    d[1] = sin(k[1]);
    d[2] = m + cos(k[0]) + cos(k[1]);
}
static void qwz_dx(const double k[2], double dd[3]) {
    dd[0] = cos(k[0]); dd[1] = 0.0; dd[2] = -sin(k[0]);
}
static void qwz_dy(const double k[2], double dd[3]) {
    dd[0] = 0.0; dd[1] = cos(k[1]); dd[2] = -sin(k[1]);
}

/* Closed-form skyrmion density (1/2) dhat.(dx dhat x dy dhat), computed
 * independently of the library, as the reference for check 1. */
static double sky_density(const double d[3], const double dx[3],
                          const double dy[3]) {
    double n = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
    double n3 = n * n * n;
    double ddx = d[0]*dx[0] + d[1]*dx[1] + d[2]*dx[2];
    double ddy = d[0]*dy[0] + d[1]*dy[1] + d[2]*dy[2];
    double hd[3], ghx[3], ghy[3];
    for (int i = 0; i < 3; i++) {
        hd[i]  = d[i] / n;
        ghx[i] = dx[i]/n - d[i]*ddx/n3;
        ghy[i] = dy[i]/n - d[i]*ddy/n3;
    }
    double cx = ghx[1]*ghy[2] - ghx[2]*ghy[1];
    double cy = ghx[2]*ghy[0] - ghx[0]*ghy[2];
    double cz = ghx[0]*ghy[1] - ghx[1]*ghy[0];
    return 0.5 * (hd[0]*cx + hd[1]*cy + hd[2]*cz);
}

/* ---- Independent projector-form QGT from explicit eigenvectors ---- */
/* Normalised eigenvectors of H = d.sigma for eigenvalues -/+|d|. */
static void eigvecs_dsigma(const double d[3], cx um[2], cx up[2]) {
    double n = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
    cx off = d[0] - _Complex_I * d[1];          /* H_{01} = dx - i dy */
    /* Lower band (E = -|d|): (dz + |d|) u0 + off u1 = 0. */
    {
        cx a = -off, b = (cx)(d[2] + n);
        double nn = sqrt(creal(a)*creal(a) + cimag(a)*cimag(a)
                       + creal(b)*creal(b) + cimag(b)*cimag(b));
        if (nn < 1e-300) { a = 1.0; b = 0.0; nn = 1.0; }
        um[0] = a / nn; um[1] = b / nn;
    }
    /* Upper band (E = +|d|): (dz - |d|) u0 + off u1 = 0. */
    {
        cx a = -off, b = (cx)(d[2] - n);
        double nn = sqrt(creal(a)*creal(a) + cimag(a)*cimag(a)
                       + creal(b)*creal(b) + cimag(b)*cimag(b));
        if (nn < 1e-300) { a = 0.0; b = 1.0; nn = 1.0; }
        up[0] = a / nn; up[1] = b / nn;
    }
}
/* <a| M |b> for a 2x2 M (row-major) and 2-vectors. */
static cx sandwich(const cx a[2], const cx M[4], const cx b[2]) {
    cx Mb0 = M[0]*b[0] + M[1]*b[1];
    cx Mb1 = M[2]*b[0] + M[3]*b[1];
    return conj(a[0])*Mb0 + conj(a[1])*Mb1;
}
/* Reference g and Omega_xy via Q_mu_nu = <u-|dHmu|u+><u+|dHnu|u->/DE^2. */
static void ref_projector_qgt(const double d[3], const double dx[3],
                              const double dy[3], double g[4],
                              double* omega_xy) {
    double n = sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
    double DE2 = 4.0 * n * n;
    cx um[2], up[2];
    eigvecs_dsigma(d, um, up);
    /* dH_mu = (d_mu d) . sigma. */
    const double* dd[2] = { dx, dy };
    cx dH[2][4];
    for (int mu = 0; mu < 2; mu++) {
        dH[mu][0] = dd[mu][2];
        dH[mu][1] = dd[mu][0] - _Complex_I * dd[mu][1];
        dH[mu][2] = dd[mu][0] + _Complex_I * dd[mu][1];
        dH[mu][3] = -dd[mu][2];
    }
    cx Q[2][2];
    for (int mu = 0; mu < 2; mu++)
        for (int nu = 0; nu < 2; nu++)
            Q[mu][nu] = sandwich(um, dH[mu], up)
                      * sandwich(up, dH[nu], um) / DE2;
    g[0] = creal(Q[0][0]); g[1] = creal(Q[0][1]);
    g[2] = creal(Q[1][0]); g[3] = creal(Q[1][1]);
    *omega_xy = -2.0 * cimag(Q[0][1]);
}

/* ---- Check 1 + 3: closed form vs skyrmion density and vs explicit
 * eigenvector projector form, plus symmetry/PSD ---------------------- */
static void test_closed_form_and_projector(void) {
    fprintf(stdout, "\n-- QWZ closed form vs skyrmion density & explicit-"
                    "eigvec projector (m=+1) --\n");
    const double m = 1.0;
    const double ks[][2] = {
        {0.7, 1.3}, {2.1, -0.4}, {-1.9, 2.6}, {0.35, 0.35}, {3.0, -2.2},
    };
    double max_sky = 0.0, max_proj_g = 0.0, max_proj_om = 0.0;
    double worst_sym = 0.0, min_eig = 1e30;
    for (size_t i = 0; i < sizeof(ks)/sizeof(ks[0]); i++) {
        double d[3], dx[3], dy[3];
        qwz_d(ks[i], m, d); qwz_dx(ks[i], dx); qwz_dy(ks[i], dy);

        double g[4], om = 0.0;
        int rc = qgt_dsigma_metric_curvature(d, dx, dy, g, &om);
        CHECK(rc == 0, "qgt_dsigma_metric_curvature rc=0 at k=(%.2f,%.2f)",
              ks[i][0], ks[i][1]);

        double om_ref = sky_density(d, dx, dy);
        double e_sky = fabs(om - om_ref);
        if (e_sky > max_sky) max_sky = e_sky;

        double gref[4], omref2 = 0.0;
        ref_projector_qgt(d, dx, dy, gref, &omref2);
        for (int j = 0; j < 4; j++) {
            double e = fabs(g[j] - gref[j]);
            if (e > max_proj_g) max_proj_g = e;
        }
        double e_om2 = fabs(om - omref2);
        if (e_om2 > max_proj_om) max_proj_om = e_om2;

        double sym = fabs(g[1] - g[2]);
        if (sym > worst_sym) worst_sym = sym;
        /* PSD of the symmetric 2x2: eigenvalues via trace/det. */
        double tr = g[0] + g[3];
        double det = g[0]*g[3] - g[1]*g[2];
        double disc = sqrt(fmax(0.0, tr*tr - 4.0*det));
        double lo = 0.5 * (tr - disc);
        if (lo < min_eig) min_eig = lo;
    }
    fprintf(stdout, "    max|Omega - (1/2)dhat.(dx x dy)| = %.3e\n", max_sky);
    fprintf(stdout, "    max|g - g_proj(explicit eigvec)| = %.3e\n", max_proj_g);
    fprintf(stdout, "    max|Omega - Omega_proj|          = %.3e\n", max_proj_om);
    fprintf(stdout, "    worst |gxy - gyx| = %.3e   min metric eig = %.3e\n",
            worst_sym, min_eig);
    CHECK(max_sky   < 1e-13, "Omega matches skyrmion density (%.2e < 1e-13)", max_sky);
    CHECK(max_proj_g  < 1e-13, "g matches explicit-eigvec projector (%.2e)", max_proj_g);
    CHECK(max_proj_om < 1e-13, "Omega matches explicit-eigvec projector (%.2e)", max_proj_om);
    CHECK(worst_sym < 1e-15, "g exactly symmetric (%.2e)", worst_sym);
    CHECK(min_eig   > -1e-13, "g positive semidefinite (min eig %.2e)", min_eig);
}

/* ---- Check 2: integrate pointwise Omega vs FHS Chern -------------- */
static double integrate_curvature(double m, size_t N) {
    const double step = 2.0 * M_PI / (double)N;
    double acc = 0.0;
    for (size_t iy = 0; iy < N; iy++) {
        for (size_t ix = 0; ix < N; ix++) {
            double k[2] = { (double)ix*step + 0.5*step,
                            (double)iy*step + 0.5*step };
            double d[3], dx[3], dy[3];
            qwz_d(k, m, d); qwz_dx(k, dx); qwz_dy(k, dy);
            double g[4], om = 0.0;
            qgt_dsigma_metric_curvature(d, dx, dy, g, &om);
            acc += om * step * step;
        }
    }
    return acc / (2.0 * M_PI);
}

static void test_pointwise_vs_fhs(double m, int expected) {
    fprintf(stdout, "\n-- QWZ m=%+.1f: integrated pointwise Omega vs "
                    "qgt_berry_grid --\n", m);
    double c_point = integrate_curvature(m, 200);

    qgt_system_t* sys = qgt_model_qwz(m);
    qgt_berry_grid_t grid;
    int rc = qgt_berry_grid(sys, 64, &grid);
    CHECK(rc == 0, "qgt_berry_grid rc=0");
    double c_fhs = grid.chern;
    fprintf(stdout, "    C_pointwise = %+.6f   C_FHS = %+.6f   |diff| = %.3e"
                    "   expected %+d\n",
            c_point, c_fhs, fabs(c_point - c_fhs), expected);
    CHECK((int)lround(c_point) == expected,
          "pointwise integral rounds to %+d", expected);
    CHECK((int)lround(c_fhs) == expected,
          "FHS integral rounds to %+d", expected);
    CHECK(fabs(c_point - c_fhs) < 5e-3,
          "pointwise and FHS Chern agree (|diff| = %.3e)",
          fabs(c_point - c_fhs));
    qgt_berry_grid_free(&grid);
    qgt_free(sys);
}

/* ---- Check 4: generic finite-difference path vs closed form ------- */
static void test_generic_fd_path(void) {
    fprintf(stdout, "\n-- Generic qgt_curvature_at (H finite-difference) vs "
                    "closed form (QWZ m=-1) --\n");
    const double m = -1.0;
    qgt_system_t* sys = qgt_model_qwz(m);
    const double ks[][2] = { {1.1, 0.7}, {-0.6, 2.2}, {2.4, -1.5} };
    double max_g = 0.0, max_om = 0.0;
    for (size_t i = 0; i < sizeof(ks)/sizeof(ks[0]); i++) {
        double d[3], dx[3], dy[3];
        qwz_d(ks[i], m, d); qwz_dx(ks[i], dx); qwz_dy(ks[i], dy);
        double g_exact[4], om_exact = 0.0;
        qgt_dsigma_metric_curvature(d, dx, dy, g_exact, &om_exact);

        double g_fd[4], om_fd = 0.0;
        int rc = qgt_curvature_at(sys, ks[i], 1e-4, g_fd, &om_fd);
        CHECK(rc == 0, "qgt_curvature_at rc=0 at k=(%.2f,%.2f)",
              ks[i][0], ks[i][1]);
        for (int j = 0; j < 4; j++) {
            double e = fabs(g_fd[j] - g_exact[j]);
            if (e > max_g) max_g = e;
        }
        double e_om = fabs(om_fd - om_exact);
        if (e_om > max_om) max_om = e_om;
    }
    fprintf(stdout, "    max|g_fd - g_exact| = %.3e   max|Omega_fd - "
                    "Omega_exact| = %.3e\n", max_g, max_om);
    CHECK(max_g  < 1e-6, "FD metric matches closed form (%.2e < 1e-6)", max_g);
    CHECK(max_om < 1e-6, "FD curvature matches closed form (%.2e < 1e-6)", max_om);
    qgt_free(sys);
}

int main(void) {
    fprintf(stdout, "=== Exact pointwise band geometry (QGT) ===\n");
    test_closed_form_and_projector();
    test_pointwise_vs_fhs(-1.0, +1);   /* -2 < m < 0 -> C = +1 */
    test_pointwise_vs_fhs(+1.0, -1);   /*  0 < m < 2 -> C = -1 */
    test_pointwise_vs_fhs(+3.0,  0);   /*   |m| > 2  -> C = 0  */
    test_generic_fd_path();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
