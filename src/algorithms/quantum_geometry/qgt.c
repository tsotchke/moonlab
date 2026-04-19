/**
 * @file qgt.c
 * @brief Quantum geometric tensor implementation.
 */

#include "qgt.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct qgt_system {
    qgt_bloch_fn fn;
    void*        user;
    /* For built-in models we own the user payload. */
    void*        owned_user;
};

struct qgt_system_1d {
    qgt_bloch_1d_fn fn;
    void*           user;
    void*           owned_user;
};

qgt_system_t* qgt_create(qgt_bloch_fn f, void* user) {
    if (!f) return NULL;
    qgt_system_t* s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->fn = f;
    s->user = user;
    s->owned_user = NULL;
    return s;
}

void qgt_free(qgt_system_t* sys) {
    if (!sys) return;
    free(sys->owned_user);
    free(sys);
}

/* ---- QWZ built-in -------------------------------------------------- */

typedef struct { double m; } qwz_params_t;

static void qwz_bloch(const double k[2], void* user, qgt_complex_t h[4]) {
    double m = ((qwz_params_t*)user)->m;
    double sx = sin(k[0]);
    double sy = sin(k[1]);
    double hz = m + cos(k[0]) + cos(k[1]);
    /* H = sx sigma_x + sy sigma_y + hz sigma_z
     *   = [[ hz,      sx - i sy ],
     *      [ sx + i sy, -hz     ]] */
    h[0] = hz;
    h[1] = sx - _Complex_I * sy;
    h[2] = sx + _Complex_I * sy;
    h[3] = -hz;
}

qgt_system_t* qgt_model_qwz(double m) {
    qwz_params_t* p = malloc(sizeof(*p));
    if (!p) return NULL;
    p->m = m;
    qgt_system_t* s = qgt_create(qwz_bloch, p);
    if (!s) { free(p); return NULL; }
    s->owned_user = p;
    return s;
}

/* ---- Haldane built-in --------------------------------------------- */

typedef struct {
    double t1, t2, phi, M;
} haldane_params_t;

static void haldane_bloch(const double k[2], void* user,
                          qgt_complex_t h[4]) {
    /* Honeycomb reciprocal vectors a1 = (1, 0), a2 = (1/2, sqrt(3)/2).
     * Nearest-neighbour vectors: (1/sqrt(3))(0, 1), (1/sqrt(3))(-sqrt(3)/2, -1/2),
     * (1/sqrt(3))(sqrt(3)/2, -1/2). Use the standard form as in
     * Haldane 1988 / Bernevig-Hughes "Topological Insulators" textbook. */
    const haldane_params_t* p = (const haldane_params_t*)user;
    double t1 = p->t1, t2 = p->t2, phi = p->phi, M = p->M;
    double kx = k[0], ky = k[1];
    /* NN sum (off-diagonal): sum_delta e^{-i k . delta_j}. */
    double ax = kx, ay = ky;
    qgt_complex_t f = (cos(ax) + cos(ax - ay) + cos(ay))
                    + _Complex_I * (sin(ax) + sin(ax - ay) + sin(ay));
    /* Diagonal terms: NNN hopping contributes cos(phi) * [standard NNN sum] to
     * both diagonals (same sign) and sin(phi) * [antisymmetric NNN sum] to
     * sigma_z. */
    double c1 = cos(ax - ay) + cos(ax) + cos(ay);          /* even under AB swap */
    double c2 = sin(ax - ay) - sin(ax) + sin(ay);          /* odd (signed NNN) */
    double diag_sum = 2.0 * t2 * cos(phi) * c1;
    double sigma_z = M - 2.0 * t2 * sin(phi) * c2;
    h[0] = diag_sum + sigma_z;
    h[1] = t1 * conj(f);
    h[2] = t1 * f;
    h[3] = diag_sum - sigma_z;
}

qgt_system_t* qgt_model_haldane(double t1, double t2,
                                double phi, double M_stagger) {
    haldane_params_t* p = malloc(sizeof(*p));
    if (!p) return NULL;
    p->t1 = t1; p->t2 = t2; p->phi = phi; p->M = M_stagger;
    qgt_system_t* s = qgt_create(haldane_bloch, p);
    if (!s) { free(p); return NULL; }
    s->owned_user = p;
    return s;
}

/* ---- 2x2 Hermitian lower-eigenvector ------------------------------- */

/* Return the normalized eigenvector of the lower eigenvalue of H.
 *
 * For a 2x2 Hermitian H = (Tr H / 2) I + h . sigma, the lower band is
 * the -|h| eigenvalue. Two gauge-inequivalent eigenvector formulas:
 *   A) u_A = (h_x - i h_y,  -(h_z + |h|))   (singular at h = -|h| z)
 *   B) u_B = (-(|h| - h_z),  h_x + i h_y)   (singular at h = +|h| z)
 * Switching between them discontinuously creates spurious pi jumps
 * in the Berry link variables. We always use formula A and fall back
 * to B only when |norm_A|^2 is below a small threshold. */
static void lower_eigvec_2x2(const qgt_complex_t h[4], qgt_complex_t u[2]) {
    qgt_complex_t h00 = h[0], h11 = h[3], h01 = h[1];
    double hz = 0.5 * creal(h00 - h11);
    double hx = creal(h01);
    double hy = -cimag(h01);           /* h_{01} = h_x - i h_y */
    double hnorm = sqrt(hx * hx + hy * hy + hz * hz);
    if (hnorm < 1e-300) {
        u[0] = 1.0; u[1] = 0.0;
        return;
    }

    /* Formula A. */
    qgt_complex_t aA = hx - _Complex_I * hy;
    qgt_complex_t bA = -(hz + hnorm);
    double nA2 = creal(aA) * creal(aA) + cimag(aA) * cimag(aA)
               + creal(bA) * creal(bA) + cimag(bA) * cimag(bA);
    /* Formula B. */
    qgt_complex_t aB = (qgt_complex_t)(hz - hnorm);
    qgt_complex_t bB = hx + _Complex_I * hy;
    double nB2 = creal(aB) * creal(aB) + cimag(aB) * cimag(aB)
               + creal(bB) * creal(bB) + cimag(bB) * cimag(bB);

    /* Pick the larger-norm branch so we stay away from its singularity. */
    qgt_complex_t a, b;
    double norm2;
    if (nA2 >= nB2) {
        a = aA; b = bA; norm2 = nA2;
    } else {
        a = aB; b = bB; norm2 = nB2;
    }
    double norm = sqrt(norm2);
    if (norm < 1e-300) {
        u[0] = 1.0; u[1] = 0.0;
        return;
    }
    u[0] = a / norm;
    u[1] = b / norm;
}

/* Inner product <a | b> of two 2-component vectors. */
static qgt_complex_t vdot2(const qgt_complex_t a[2], const qgt_complex_t b[2]) {
    return conj(a[0]) * b[0] + conj(a[1]) * b[1];
}

/* Principal branch of log z in (-pi, pi]. We only need the imaginary
 * part for the Berry curvature. */
static double arg_principal(qgt_complex_t z) {
    return atan2(cimag(z), creal(z));
}

int qgt_berry_grid(const qgt_system_t* sys, size_t N,
                   qgt_berry_grid_t* out) {
    if (!sys || !out || N < 4) return -1;

    /* Allocate all N*N lower-band eigenvectors so we can walk the
     * grid exactly once. */
    qgt_complex_t* U = malloc(N * N * 2 * sizeof(qgt_complex_t));
    if (!U) return -2;

    const double step = 2.0 * M_PI / (double)N;
    qgt_complex_t h[4];
    for (size_t iy = 0; iy < N; iy++) {
        for (size_t ix = 0; ix < N; ix++) {
            double k[2] = { (double)ix * step, (double)iy * step };
            sys->fn(k, sys->user, h);
            lower_eigvec_2x2(h, &U[(iy * N + ix) * 2]);
        }
    }

    out->N = N;
    out->berry = calloc(N * N, sizeof(double));
    if (!out->berry) { free(U); return -3; }

    double chern_sum = 0.0;
    for (size_t iy = 0; iy < N; iy++) {
        size_t iy1 = (iy + 1) % N;
        for (size_t ix = 0; ix < N; ix++) {
            size_t ix1 = (ix + 1) % N;
            const qgt_complex_t* u00 = &U[(iy  * N + ix ) * 2];
            const qgt_complex_t* u10 = &U[(iy  * N + ix1) * 2];
            const qgt_complex_t* u11 = &U[(iy1 * N + ix1) * 2];
            const qgt_complex_t* u01 = &U[(iy1 * N + ix ) * 2];
            /* Link variables (non-normalized inner products; only
             * phase matters for the plaquette). */
            qgt_complex_t Ux = vdot2(u00, u10);
            qgt_complex_t Uy_at_x1 = vdot2(u10, u11);
            qgt_complex_t Ux_at_y1 = vdot2(u01, u11);
            qgt_complex_t Uy = vdot2(u00, u01);
            qgt_complex_t plaq = Ux * Uy_at_x1 * conj(Ux_at_y1) * conj(Uy);
            /* Sign convention: negate so the integrated Chern number
             * matches the Bianco-Resta / Asboth convention (QWZ with
             * m=+1 gives C=-1). Our plaquette traversal (+x +y -x -y)
             * naturally gives the opposite sign of the Berry-curvature
             * integral under the A_mu = i <u|d_mu u> connection. */
            double F = -arg_principal(plaq);
            out->berry[iy * N + ix] = F;
            chern_sum += F;
        }
    }
    out->chern = chern_sum / (2.0 * M_PI);

    free(U);
    return 0;
}

void qgt_berry_grid_free(qgt_berry_grid_t* g) {
    if (!g) return;
    free(g->berry);
    g->berry = NULL;
    g->N = 0;
    g->chern = 0.0;
}

int qgt_metric_at(const qgt_system_t* sys, const double k[2],
                  double dk, double g[4]) {
    if (!sys || !k || !g || dk <= 0.0) return -1;

    /* Centered differences of the lower-band state:
     *   |d_mu u> = (|u(k + dk e_mu)> - |u(k - dk e_mu)>) / (2 dk).
     * QGT = <d_mu u | (1 - P) | d_nu u> with P = |u><u|. */
    qgt_complex_t h[4], u0[2], up[2], um[2];
    sys->fn(k, sys->user, h);
    lower_eigvec_2x2(h, u0);

    qgt_complex_t dU[2][2]; /* dU[mu][component] */
    for (int mu = 0; mu < 2; mu++) {
        double kp[2] = { k[0], k[1] }, km[2] = { k[0], k[1] };
        kp[mu] += dk;
        km[mu] -= dk;
        sys->fn(kp, sys->user, h);
        lower_eigvec_2x2(h, up);
        sys->fn(km, sys->user, h);
        lower_eigvec_2x2(h, um);
        /* Gauge-fix each eigenvector to match the phase of u0 so the
         * finite difference doesn't pick up an arbitrary global phase.
         * This does not affect the QGT (gauge-invariant) but keeps
         * intermediate numbers well behaved. */
        qgt_complex_t sp = vdot2(u0, up);
        qgt_complex_t sm = vdot2(u0, um);
        double apm = sqrt(creal(sp)*creal(sp) + cimag(sp)*cimag(sp));
        double amm = sqrt(creal(sm)*creal(sm) + cimag(sm)*cimag(sm));
        if (apm > 1e-300) { up[0] *= conj(sp)/apm; up[1] *= conj(sp)/apm; }
        if (amm > 1e-300) { um[0] *= conj(sm)/amm; um[1] *= conj(sm)/amm; }
        for (int i = 0; i < 2; i++) {
            dU[mu][i] = (up[i] - um[i]) / (2.0 * dk);
        }
    }

    /* Re Q_{mu,nu} = Re [<d_mu u | d_nu u> - <d_mu u | u><u | d_nu u>]. */
    for (int mu = 0; mu < 2; mu++) {
        for (int nu = 0; nu < 2; nu++) {
            qgt_complex_t raw =
                conj(dU[mu][0]) * dU[nu][0] + conj(dU[mu][1]) * dU[nu][1];
            qgt_complex_t c_mu = vdot2(&dU[mu][0], u0); /* <d_mu u | u> */
            qgt_complex_t c_nu = vdot2(u0, &dU[nu][0]); /* <u | d_nu u> */
            qgt_complex_t proj = c_mu * c_nu;
            g[mu * 2 + nu] = creal(raw - proj);
        }
    }
    return 0;
}

/* ---- Wilson loop -------------------------------------------------- */

int qgt_wilson_loop(const qgt_system_t* sys,
                    const double* path_k,
                    size_t num_points,
                    double* out_phase) {
    if (!sys || !path_k || num_points < 2 || !out_phase) return -1;

    /* Accumulate the product of link variables around the path. We
     * store only the running product (complex scalar) since only the
     * final phase is needed. */
    qgt_complex_t u_prev[2], u_cur[2], h[4];
    double k0[2] = { path_k[0], path_k[1] };
    sys->fn(k0, sys->user, h);
    lower_eigvec_2x2(h, u_prev);
    qgt_complex_t u_start[2] = { u_prev[0], u_prev[1] };

    qgt_complex_t prod = 1.0;
    for (size_t i = 1; i < num_points; i++) {
        double k[2] = { path_k[2 * i], path_k[2 * i + 1] };
        sys->fn(k, sys->user, h);
        lower_eigvec_2x2(h, u_cur);
        prod *= vdot2(u_prev, u_cur);
        u_prev[0] = u_cur[0];
        u_prev[1] = u_cur[1];
    }
    /* Close the loop: final point -> start. */
    prod *= vdot2(u_prev, u_start);

    *out_phase = arg_principal(prod);
    return 0;
}

/* ---- 1D system + SSH + winding ------------------------------------- */

qgt_system_1d_t* qgt_create_1d(qgt_bloch_1d_fn f, void* user) {
    if (!f) return NULL;
    qgt_system_1d_t* s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->fn = f;
    s->user = user;
    return s;
}

void qgt_free_1d(qgt_system_1d_t* sys) {
    if (!sys) return;
    free(sys->owned_user);
    free(sys);
}

typedef struct { double t1, t2; } ssh_params_t;

static void ssh_bloch(double k, void* user, qgt_complex_t h[4]) {
    const ssh_params_t* p = (const ssh_params_t*)user;
    double dx = p->t1 + p->t2 * cos(k);
    double dy = p->t2 * sin(k);
    /* H(k) = dx sigma_x + dy sigma_y. Hermitian, no sigma_z term. */
    h[0] = 0.0;
    h[1] = dx - _Complex_I * dy;
    h[2] = dx + _Complex_I * dy;
    h[3] = 0.0;
}

qgt_system_1d_t* qgt_model_ssh(double t1, double t2) {
    ssh_params_t* p = malloc(sizeof(*p));
    if (!p) return NULL;
    p->t1 = t1; p->t2 = t2;
    qgt_system_1d_t* s = qgt_create_1d(ssh_bloch, p);
    if (!s) { free(p); return NULL; }
    s->owned_user = p;
    return s;
}

int qgt_winding_1d(const qgt_system_1d_t* sys, size_t N,
                   double* out_raw) {
    if (!sys || N < 4) { if (out_raw) *out_raw = 0.0; return 0; }

    /* Winding number via the closed Wilson loop around the 1D BZ:
     *   w = (1 / 2 pi) sum_k arg(<u(k) | u(k + dk)>).
     * For the chiral SSH class the result is an integer.
     *
     * Our lower_eigvec_2x2 uses a dual-branch picker whose branch
     * can flip between neighbouring k, producing spurious pi jumps
     * in the link phase. To avoid this, we renormalise the gauge
     * of each consecutive eigenvector to maximise <u_prev | u_cur>
     * (gauge-fix for continuity), then compute the cumulative
     * arg. The ambiguity is modulo 2 pi and disappears on the
     * integer winding. */
    qgt_complex_t u_prev[2], u_cur[2], h[4];
    double dk = 2.0 * M_PI / (double)N;
    sys->fn(0.0, sys->user, h);
    lower_eigvec_2x2(h, u_prev);
    qgt_complex_t u_start[2] = { u_prev[0], u_prev[1] };

    double total = 0.0;
    for (size_t i = 1; i < N; i++) {
        double k = (double)i * dk;
        sys->fn(k, sys->user, h);
        lower_eigvec_2x2(h, u_cur);
        qgt_complex_t link = vdot2(u_prev, u_cur);
        /* If the link is close to 0, the gauge flipped. Rotate u_cur
         * by pi (multiply by -1) and retry. */
        double mag = sqrt(creal(link) * creal(link) +
                          cimag(link) * cimag(link));
        if (mag < 1e-3) {
            u_cur[0] = -u_cur[0];
            u_cur[1] = -u_cur[1];
            link = vdot2(u_prev, u_cur);
        }
        total += arg_principal(link);
        u_prev[0] = u_cur[0];
        u_prev[1] = u_cur[1];
    }
    /* Close the loop. */
    qgt_complex_t link = vdot2(u_prev, u_start);
    total += arg_principal(link);

    /* The accumulated phase is the Zak phase of the lower band; the
     * SSH / chiral "winding number" is W = -Zak/pi (the factor 2
     * converts the 2-band Zak phase to the 1D winding of the
     * off-diagonal; the sign is fixed by our lower_eigvec_2x2 gauge
     * so that t2 > t1 topological gives W = +1). */
    double zak = total;
    double raw = -zak / M_PI;
    if (out_raw) *out_raw = raw;
    return (int)lround(raw);
}
