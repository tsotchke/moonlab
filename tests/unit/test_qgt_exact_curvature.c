/**
 * @file test_qgt_exact_curvature.c
 * @brief Exact pointwise band geometry: qgt_dsigma_metric_curvature,
 *        qgt_exact_curvature_at, and qgt_curvature_at.
 *
 * Verifies, for the two-band QWZ and Haldane models:
 *   1. g and Omega agree with the projector-form QGT assembled from an
 *      INDEPENDENT exact 2x2 diagonalisation at k (explicit eigenvectors,
 *      a different route to the same tensor) to ~1e-13, and g is symmetric
 *      and positive semidefinite.  This is the load-bearing correctness
 *      check: it does not re-state the implementation's own formula.
 *   2. The pointwise Omega_xy integrated over the BZ reproduces the
 *      integer Chern number of the existing FHS integrator qgt_berry_grid,
 *      across the QWZ phases (C = 0, +1, -1) and across the Haldane
 *      transition at phi = -pi/2 (M = 0 and 0.2 topological, M = 2.0
 *      trivial).
 *   3. The exact path is reachable through the public handle API:
 *      qgt_dsigma_at / qgt_exact_curvature_at on a qgt_model_qwz or
 *      qgt_model_haldane system return the same values as the manually
 *      derived d(k) fed to qgt_dsigma_metric_curvature, and the analytic
 *      d-vector reproduces the model's own Bloch callback.
 *   4. The generic finite-difference path qgt_curvature_at agrees with the
 *      closed form to finite-difference accuracy.
 *   5. Band touchings are rejected relative to the input scale, not only at
 *      an exact zero: a near-gapless k returns -2 rather than ~1e12.
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

/* ---- Haldane analytic d(k) and its k-gradients --------------------
 *
 * Derived here from the model's published form, independently of the
 * library's own haldane_dsigma: d_x = t1 Re f, d_y = t1 Im f with
 * f = sum_j exp(i k.delta_j) over the three neighbour vectors in the
 * module's primitive coordinates, and d_z the staggering M minus the
 * next-nearest-neighbour Peierls mass.  The identity component
 * 2 t2 cos(phi) c1 is a rigid shift of both bands and does not enter
 * the projectors, hence not the geometry. */
typedef struct { double t1, t2, phi, M; } haldane_p_t;

static void haldane_d(const double k[2], const haldane_p_t* p, double d[3]) {
    double kx = k[0], ky = k[1];
    double s2 = -2.0 * p->t2 * sin(p->phi);
    d[0] = p->t1 * (cos(kx) + cos(kx - ky) + cos(ky));
    d[1] = p->t1 * (sin(kx) + sin(kx - ky) + sin(ky));
    d[2] = p->M + s2 * sin(ky) * (1.0 + 2.0 * cos(kx));
}
static void haldane_dx(const double k[2], const haldane_p_t* p, double dd[3]) {
    double kx = k[0], ky = k[1];
    double s2 = -2.0 * p->t2 * sin(p->phi);
    dd[0] = p->t1 * (-sin(kx) - sin(kx - ky));
    dd[1] = p->t1 * ( cos(kx) + cos(kx - ky));
    dd[2] = s2 * sin(ky) * (-2.0 * sin(kx));
}
static void haldane_dy(const double k[2], const haldane_p_t* p, double dd[3]) {
    double kx = k[0], ky = k[1];
    double s2 = -2.0 * p->t2 * sin(p->phi);
    dd[0] = p->t1 * ( sin(kx - ky) - sin(ky));
    dd[1] = p->t1 * (-cos(kx - ky) + cos(ky));
    dd[2] = s2 * cos(ky) * (1.0 + 2.0 * cos(kx));
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

/* ---- Check 1: closed form vs the explicit-eigenvector projector form,
 * plus symmetry / PSD.
 *
 * This is the load-bearing correctness check.  ref_projector_qgt builds the
 * same tensor by a genuinely different route -- diagonalise H = d.sigma at k,
 * take the two eigenvectors, and evaluate
 * Q_munu = <u-|dH_mu|u+><u+|dH_nu|u-> / (DeltaE)^2 -- so agreement is
 * evidence about the implementation rather than a restatement of its own
 * algebra.  (An earlier revision also compared Omega against a second
 * hand-written copy of the very formula qgt_dsigma_metric_curvature
 * evaluates; that check could only fail on a typo in the copy, and is gone.)
 *
 * Covers both built-in d.sigma models, since the Haldane d-vector exercises
 * all three components with k-dependent gradients where QWZ's are sparse. */
static void projector_agreement(const char* label, const double ks[][2],
                                size_t nk, void (*dfn)(const double[2], void*,
                                                       double[3], double[3],
                                                       double[3]),
                                void* param) {
    double max_proj_g = 0.0, max_proj_om = 0.0;
    double worst_sym = 0.0, min_eig = 1e30;
    for (size_t i = 0; i < nk; i++) {
        double d[3], dx[3], dy[3];
        dfn(ks[i], param, d, dx, dy);

        double g[4], om = 0.0;
        int rc = qgt_dsigma_metric_curvature(d, dx, dy, g, &om);
        CHECK(rc == 0, "%s: qgt_dsigma_metric_curvature rc=0 at k=(%.2f,%.2f)",
              label, ks[i][0], ks[i][1]);

        double gref[4], omref = 0.0;
        ref_projector_qgt(d, dx, dy, gref, &omref);
        for (int j = 0; j < 4; j++) {
            double e = fabs(g[j] - gref[j]);
            if (e > max_proj_g) max_proj_g = e;
        }
        double e_om = fabs(om - omref);
        if (e_om > max_proj_om) max_proj_om = e_om;

        double sym = fabs(g[1] - g[2]);
        if (sym > worst_sym) worst_sym = sym;
        /* PSD of the symmetric 2x2: eigenvalues via trace/det. */
        double tr = g[0] + g[3];
        double det = g[0]*g[3] - g[1]*g[2];
        double disc = sqrt(fmax(0.0, tr*tr - 4.0*det));
        double lo = 0.5 * (tr - disc);
        if (lo < min_eig) min_eig = lo;
    }
    fprintf(stdout, "    %s: max|g - g_proj| = %.3e   max|Omega - Omega_proj|"
                    " = %.3e\n", label, max_proj_g, max_proj_om);
    fprintf(stdout, "    %s: worst |gxy - gyx| = %.3e   min metric eig = "
                    "%.3e\n", label, worst_sym, min_eig);
    CHECK(max_proj_g  < 1e-13, "%s: g matches explicit-eigvec projector (%.2e)",
          label, max_proj_g);
    CHECK(max_proj_om < 1e-13, "%s: Omega matches explicit-eigvec projector "
          "(%.2e)", label, max_proj_om);
    CHECK(worst_sym < 1e-15, "%s: g exactly symmetric (%.2e)", label, worst_sym);
    CHECK(min_eig   > -1e-13, "%s: g positive semidefinite (min eig %.2e)",
          label, min_eig);
}

/* Adapters matching the qgt_dsigma_fn shape, so the same driver serves both
 * models and both parameter payloads. */
static void qwz_d_all(const double k[2], void* p, double d[3], double dx[3],
                      double dy[3]) {
    qwz_d(k, *(const double*)p, d); qwz_dx(k, dx); qwz_dy(k, dy);
}
static void haldane_d_all(const double k[2], void* p, double d[3],
                          double dx[3], double dy[3]) {
    const haldane_p_t* hp = (const haldane_p_t*)p;
    haldane_d(k, hp, d); haldane_dx(k, hp, dx); haldane_dy(k, hp, dy);
}

static void test_closed_form_vs_projector(void) {
    fprintf(stdout, "\n-- Closed form vs explicit-eigvec projector QGT --\n");
    const double ks[][2] = {
        {0.7, 1.3}, {2.1, -0.4}, {-1.9, 2.6}, {0.35, 0.35}, {3.0, -2.2},
    };
    const size_t nk = sizeof(ks)/sizeof(ks[0]);

    double m = 1.0;
    projector_agreement("QWZ m=+1", ks, nk, qwz_d_all, &m);

    /* Haldane at the review-verified point: t1=1, t2=0.2, phi=-pi/2 puts the
     * transition at |M| = 3 sqrt(3) t2 = 1.039, so M=0.2 is topological. */
    haldane_p_t hp = { 1.0, 0.2, -0.5 * M_PI, 0.2 };
    projector_agreement("Haldane M=0.2", ks, nk, haldane_d_all, &hp);
}

/* ---- Check 2: integrate pointwise Omega vs FHS Chern --------------
 *
 * Integrates through the public handle API (qgt_exact_curvature_at), so the
 * agreement with the FHS integrator is also an end-to-end statement about the
 * routing: the analytic d-vector the model carries has to be the one its
 * Bloch callback describes, or the two Chern numbers diverge. */
static double integrate_curvature(const qgt_system_t* sys, size_t N,
                                  int* out_rc) {
    const double step = 2.0 * M_PI / (double)N;
    double acc = 0.0;
    int worst = 0;
    for (size_t iy = 0; iy < N; iy++) {
        for (size_t ix = 0; ix < N; ix++) {
            double k[2] = { (double)ix*step + 0.5*step,
                            (double)iy*step + 0.5*step };
            double g[4], om = 0.0;
            int rc = qgt_exact_curvature_at(sys, k, g, &om);
            if (rc != 0) { worst = rc; continue; }
            acc += om * step * step;
        }
    }
    if (out_rc) *out_rc = worst;
    return acc / (2.0 * M_PI);
}

static void pointwise_vs_fhs(const char* label, qgt_system_t* sys,
                             int expected) {
    int worst_rc = 0;
    double c_point = integrate_curvature(sys, 200, &worst_rc);
    CHECK(worst_rc == 0, "%s: qgt_exact_curvature_at rc=0 across the BZ grid",
          label);

    qgt_berry_grid_t grid;
    int rc = qgt_berry_grid(sys, 64, &grid);
    CHECK(rc == 0, "%s: qgt_berry_grid rc=0", label);
    double c_fhs = grid.chern;
    fprintf(stdout, "    %s: C_pointwise = %+.6f   C_FHS = %+.6f   |diff| ="
                    " %.3e   expected %+d\n",
            label, c_point, c_fhs, fabs(c_point - c_fhs), expected);
    CHECK((int)lround(c_point) == expected,
          "%s: pointwise integral rounds to %+d", label, expected);
    CHECK((int)lround(c_fhs) == expected,
          "%s: FHS integral rounds to %+d", label, expected);
    CHECK(fabs(c_point - c_fhs) < 5e-3,
          "%s: pointwise and FHS Chern agree (|diff| = %.3e)",
          label, fabs(c_point - c_fhs));
    qgt_berry_grid_free(&grid);
}

static void test_pointwise_vs_fhs_qwz(double m, int expected) {
    fprintf(stdout, "\n-- QWZ m=%+.1f: integrated pointwise Omega vs "
                    "qgt_berry_grid --\n", m);
    char label[32];
    snprintf(label, sizeof(label), "QWZ m=%+.1f", m);
    qgt_system_t* sys = qgt_model_qwz(m);
    pointwise_vs_fhs(label, sys, expected);
    qgt_free(sys);
}

/* Haldane at t1=1, t2=0.2, phi=-pi/2: the transition sits at
 * |M| = 3 sqrt(3) t2 |sin phi| = 1.039, so M=0 and M=0.2 are topological and
 * M=2.0 is trivial.  phi=-pi/2 flips the sign of the NNN mass relative to the
 * +pi/2 convention used by test_qgt.c, hence C=+1 rather than -1. */
static void test_pointwise_vs_fhs_haldane(double M, int expected) {
    fprintf(stdout, "\n-- Haldane M=%.1f, phi=-pi/2: integrated pointwise "
                    "Omega vs qgt_berry_grid --\n", M);
    char label[40];
    snprintf(label, sizeof(label), "Haldane M=%.1f", M);
    qgt_system_t* sys = qgt_model_haldane(1.0, 0.2, -0.5 * M_PI, M);
    pointwise_vs_fhs(label, sys, expected);
    qgt_free(sys);
}

/* ---- Check 3: the exact path is reachable through the public API ----
 *
 * Before qgt_set_dsigma / qgt_dsigma_at / qgt_exact_curvature_at existed, a
 * qgt_system_t carried only its Bloch callback, so a caller holding a
 * qgt_model_qwz or qgt_model_haldane handle had no way to reach the exact
 * closed form without re-deriving d(k) and its gradients by hand.  These
 * checks pin the routing: the handle's analytic d-vector must reproduce the
 * model's own Bloch matrix, and the routed geometry must equal the geometry
 * computed from a hand-derived d(k). */
static void check_dsigma_matches_bloch(const char* label, qgt_system_t* sys,
                                       const double k[2]) {
    double d[3], dx[3], dy[3];
    int rc = qgt_dsigma_at(sys, k, d, dx, dy);
    CHECK(rc == 0, "%s: qgt_dsigma_at rc=0 on a built-in model", label);
    if (rc != 0) return;

    /* Reconstruct the traceless part of H(k) from d and compare against what
     * the model's Bloch callback produces at the same k.  qgt_metric_at is
     * the only public consumer of the callback that returns numbers directly,
     * so go through the FD curvature path, which differentiates H itself. */
    double g_fd[4], om_fd = 0.0;
    rc = qgt_curvature_at(sys, k, 1e-5, g_fd, &om_fd);
    CHECK(rc == 0, "%s: qgt_curvature_at rc=0", label);

    double g_ex[4], om_ex = 0.0;
    rc = qgt_dsigma_metric_curvature(d, dx, dy, g_ex, &om_ex);
    CHECK(rc == 0, "%s: qgt_dsigma_metric_curvature rc=0", label);

    double worst = fabs(om_fd - om_ex);
    for (int j = 0; j < 4; j++) {
        double e = fabs(g_fd[j] - g_ex[j]);
        if (e > worst) worst = e;
    }
    CHECK(worst < 1e-6,
          "%s: handle d-vector agrees with the model's Bloch callback (%.2e)",
          label, worst);
}

static void test_public_exact_path(void) {
    fprintf(stdout, "\n-- Exact path through the public handle API --\n");
    const double k[2] = { 0.83, -1.47 };

    /* QWZ: routed geometry equals geometry from a hand-derived d(k). */
    {
        const double m = 1.0;
        qgt_system_t* sys = qgt_model_qwz(m);
        double d[3], dx[3], dy[3];
        qwz_d(k, m, d); qwz_dx(k, dx); qwz_dy(k, dy);
        double g_ref[4], om_ref = 0.0;
        qgt_dsigma_metric_curvature(d, dx, dy, g_ref, &om_ref);

        double d_h[3], dx_h[3], dy_h[3];
        int rc = qgt_dsigma_at(sys, k, d_h, dx_h, dy_h);
        CHECK(rc == 0, "QWZ: qgt_dsigma_at rc=0");
        double dworst = 0.0;
        for (int j = 0; j < 3; j++) {
            double e = fmax(fabs(d_h[j] - d[j]),
                            fmax(fabs(dx_h[j] - dx[j]), fabs(dy_h[j] - dy[j])));
            if (e > dworst) dworst = e;
        }
        CHECK(dworst == 0.0,
              "QWZ: handle d-vector is bit-identical to the hand derivation");

        double g[4], om = 0.0;
        rc = qgt_exact_curvature_at(sys, k, g, &om);
        CHECK(rc == 0, "QWZ: qgt_exact_curvature_at rc=0");
        double worst = fabs(om - om_ref);
        for (int j = 0; j < 4; j++) {
            double e = fabs(g[j] - g_ref[j]);
            if (e > worst) worst = e;
        }
        CHECK(worst == 0.0,
              "QWZ: routed exact geometry is bit-identical to the direct call");

        /* omega_xy is optional. */
        double g2[4];
        CHECK(qgt_exact_curvature_at(sys, k, g2, NULL) == 0,
              "QWZ: qgt_exact_curvature_at accepts a NULL omega_xy");

        check_dsigma_matches_bloch("QWZ", sys, k);
        qgt_free(sys);
    }

    /* Haldane: same, at the review-verified parameter point. */
    {
        const haldane_p_t hp = { 1.0, 0.2, -0.5 * M_PI, 0.2 };
        qgt_system_t* sys = qgt_model_haldane(hp.t1, hp.t2, hp.phi, hp.M);
        double d[3], dx[3], dy[3];
        haldane_d(k, &hp, d); haldane_dx(k, &hp, dx); haldane_dy(k, &hp, dy);
        double g_ref[4], om_ref = 0.0;
        qgt_dsigma_metric_curvature(d, dx, dy, g_ref, &om_ref);

        double g[4], om = 0.0;
        int rc = qgt_exact_curvature_at(sys, k, g, &om);
        CHECK(rc == 0, "Haldane: qgt_exact_curvature_at rc=0");
        double worst = fabs(om - om_ref);
        for (int j = 0; j < 4; j++) {
            double e = fabs(g[j] - g_ref[j]);
            if (e > worst) worst = e;
        }
        CHECK(worst < 1e-15,
              "Haldane: routed exact geometry matches the hand derivation "
              "(%.2e)", worst);

        check_dsigma_matches_bloch("Haldane", sys, k);
        qgt_free(sys);
    }

    /* A system built from a bare Bloch callback carries no analytic d-vector
     * and says so, rather than returning junk; attaching one makes the exact
     * path work. */
    {
        double m = 1.0;
        qgt_system_t* sys = qgt_create(NULL, &m);
        CHECK(sys == NULL, "qgt_create still rejects a NULL Bloch callback");

        qgt_system_t* q = qgt_model_qwz(m);
        CHECK(qgt_set_dsigma(q, NULL) == 0, "qgt_set_dsigma detaches (rc=0)");
        double d[3], dx[3], dy[3], g[4], om = 0.0;
        CHECK(qgt_dsigma_at(q, k, d, dx, dy) == -3,
              "qgt_dsigma_at reports -3 with no analytic d-vector");
        CHECK(qgt_exact_curvature_at(q, k, g, &om) == -3,
              "qgt_exact_curvature_at reports -3 with no analytic d-vector");
        CHECK(qgt_set_dsigma(q, qwz_d_all) == 0, "qgt_set_dsigma attaches (rc=0)");
        CHECK(qgt_exact_curvature_at(q, k, g, &om) == 0,
              "qgt_exact_curvature_at works once a d-vector is attached");
        CHECK(qgt_set_dsigma(NULL, qwz_d_all) == -1,
              "qgt_set_dsigma rejects a NULL system");
        CHECK(qgt_dsigma_at(NULL, k, d, dx, dy) == -1,
              "qgt_dsigma_at rejects a NULL system");
        qgt_free(q);
    }
}

/* ---- Check 5: band-touching guard is relative, not absolute -------
 *
 * At the QWZ transition m = -2 the gap closes at k = 0.  A guard that only
 * catches an exact zero lets a k-point a hair off the touching return a
 * gigantic finite g and Omega that a caller cannot distinguish from a real
 * measurement. */
static void test_band_touching_guard(void) {
    fprintf(stdout, "\n-- Band-touching guard is scale-relative --\n");

    /* Hand-built d with a gap 1e-14 of its own construction scale. */
    double d[3]  = { 1e-14, 0.0, 0.0 };
    double dx[3] = { 0.0, 1.0, 0.0 };
    double dy[3] = { 0.0, 0.0, 1.0 };
    double g[4], om = 0.0;
    CHECK(qgt_dsigma_metric_curvature(d, dx, dy, g, &om) == -2,
          "qgt_dsigma_metric_curvature rejects a near-gapless d");

    /* Exactly at the touching. */
    d[0] = 0.0;
    CHECK(qgt_dsigma_metric_curvature(d, dx, dy, g, &om) == -2,
          "qgt_dsigma_metric_curvature rejects |d| = 0");

    /* A real gap of the same absolute size but a matching construction scale
     * is a legitimate query and must still be answered. */
    double ds[3]  = { 1e-14, 0.0, 0.0 };
    double dxs[3] = { 0.0, 1e-13, 0.0 };
    double dys[3] = { 0.0, 0.0, 1e-13 };
    CHECK(qgt_dsigma_metric_curvature(ds, dxs, dys, g, &om) == 0,
          "a small-but-consistently-scaled d is still accepted");

    /* Generic FD path at the QWZ m=-2 Dirac point, where H(0) = 0. */
    qgt_system_t* sys = qgt_model_qwz(-2.0);
    const double k0[2] = { 0.0, 0.0 };
    CHECK(qgt_curvature_at(sys, k0, 1e-4, g, &om) == -2,
          "qgt_curvature_at rejects the m=-2 band touching at k=0");
    CHECK(qgt_exact_curvature_at(sys, k0, g, &om) == -2,
          "qgt_exact_curvature_at rejects the m=-2 band touching at k=0");

    /* Just off the touching, the trace scale is ~1e-8 while |h| is ~1e-8 too,
     * so the point is genuinely resolvable and must be answered. */
    const double keps[2] = { 1e-3, 0.0 };
    CHECK(qgt_exact_curvature_at(sys, keps, g, &om) == 0,
          "qgt_exact_curvature_at answers just off the touching");
    CHECK(isfinite(om) && isfinite(g[0]),
          "off-touching values are finite (Omega=%.3e, g_xx=%.3e)", om, g[0]);
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
    test_closed_form_vs_projector();
    test_pointwise_vs_fhs_qwz(-1.0, +1);   /* -2 < m < 0 -> C = +1 */
    test_pointwise_vs_fhs_qwz(+1.0, -1);   /*  0 < m < 2 -> C = -1 */
    test_pointwise_vs_fhs_qwz(+3.0,  0);   /*   |m| > 2  -> C = 0  */
    test_pointwise_vs_fhs_haldane(0.0, +1);   /* topological */
    test_pointwise_vs_fhs_haldane(0.2, +1);   /* topological */
    test_pointwise_vs_fhs_haldane(2.0,  0);   /* trivial */
    test_public_exact_path();
    test_band_touching_guard();
    test_generic_fd_path();
    fprintf(stdout, "\n=== %d failure%s ===\n",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
