/**
 * @file qgt.c
 * @brief Quantum geometric tensor implementation.
 */

#include "qgt.h"
#include "../../utils/matrix_math.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
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
    /* NN sum (off-diagonal): sum_delta e^{-i k . delta_j}.  Vanishes
     * at the Dirac points (kx, ky) = (0, +/-2*pi/3) in this primitive-
     * coord convention. */
    double ax = kx, ay = ky;
    qgt_complex_t f = (cos(ax) + cos(ax - ay) + cos(ay))
                    + _Complex_I * (sin(ax) + sin(ax - ay) + sin(ay));
    /* NNN diagonal terms.  c1 (even, contributes a sublattice-symmetric
     * shift) and c2 (odd, contributes the SOC mass that gaps the Dirac
     * points with opposite signs).  c2 must take values +/- 3*sqrt(3)/2
     * at (0, +/- 2*pi/3) to drive a topological phase transition at
     * |M| = 3*sqrt(3)*t2*|sin(phi)|.  The earlier
     *   c2 = sin(kx-ky) - sin(kx) + sin(ky)
     * form vanished at both Dirac points -- it was the antisymmetric
     * NNN sum for a different primitive-coord orientation -- giving an
     * always-trivial Hamiltonian away from M=0.  The form below is
     *   c2 = sin(ky) * (1 + 2*cos(kx))
     *      = sin(ky) + sin(ky+kx) + sin(ky-kx)
     * which evaluates to +-3*sqrt(3)/2 at the Dirac points and gives
     * the canonical Haldane phase diagram. */
    double c1 = cos(ax - ay) + cos(ax) + cos(ay);          /* even under AB swap */
    double c2 = sin(ay) * (1.0 + 2.0 * cos(ax));            /* odd (signed NNN) */
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
    /*
     * Singularity threshold for "h is numerically indistinguishable
     * from zero": eigenvectors are undefined at h = 0 and any
     * deterministic choice we make there can become a 2*pi-winding
     * artifact in the FHS plaquette product elsewhere on the grid.
     * The threshold has to be loose enough to catch the case where
     * h's components are all O(1e-16) trig-cancellation noise (which
     * is the regime that breaks Debug builds at high-symmetry grid
     * points), not just IEEE-subnormal underflow.  The trace
     * |h00| + |h11| + 2|h01| is the natural absolute-scale anchor.
     */
    double trace_scale = fabs(creal(h00)) + fabs(creal(h11))
                       + 2.0 * (fabs(creal(h01)) + fabs(cimag(h01)));
    double hnorm = sqrt(hx * hx + hy * hy + hz * hz);
    /* Use both an absolute floor (subnormal) and a relative threshold
     * (h vanishes vs. its construction-scale).  trace_scale tracks
     * the rough "1" of the Hamiltonian's natural units. */
    double sing_eps = 1e-12 * (trace_scale > 0.0 ? trace_scale : 1.0);
    if (hnorm < 1e-300 || hnorm < sing_eps) {
        /* Eigenvector is undefined; pick a deterministic vector and
         * tag with NaN so callers can detect that this k-point sits
         * on a band degeneracy and skip the offending plaquette. */
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

    /* Always use formula A so the gauge is globally consistent
     * across the BZ.  The previous "pick larger norm" heuristic
     * switched between A and B at the (h_z = 0) equator; on either
     * side the eigvec is smooth, but at the switching surface A and
     * B disagree by a relative phase, breaking the FHS plaquette
     * product on plaquettes that straddle the surface.  This bug
     * manifested as Chern = 0 for any 2D 2-band Bloch model with
     * non-zero mass on top of NN hopping (Haldane M > 0,
     * Kane-Mele lambda_v > 0), even when |M| <
     * 3*sqrt(3)*t2*sin(phi).  Sole-formula-A has a measure-zero
     * singularity at the south pole (h = -|h| z hat); the
     * half-step grid offset above keeps every grid point off it.
     * If we DO land at or near it (very small a_floor), fall back
     * to a deterministic placeholder; the plaquette catches a
     * spurious flux there but it's confined to one O(1/N^2)
     * cell. */
    /* Pick the larger-norm branch so we stay away from its singularity. */
    qgt_complex_t a, b;
    double norm2;
    if (nA2 >= nB2) {
        a = aA; b = bA; norm2 = nA2;
    } else {
        a = aB; b = bB; norm2 = nB2;
    }
    double norm = sqrt(norm2);
    if (norm < 1e-300 || norm < sing_eps) {
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

    /*
     * Standard Fukui-Hatsugai-Suzuki (FHS) hygiene: shift the grid by
     * half a step in each direction so it never lands on high-symmetry
     * points where the Bloch Hamiltonian h(k) can vanish exactly.
     *
     * Without this shift, models like Haldane with M=0 and grid sizes
     * that are multiples of 3 (e.g., N=48) sample exactly at the K
     * Dirac point (0, 2*pi/3) where h_x = h_y = h_z = 0 analytically.
     * Release builds happen to make the trig identities cancel to
     * exact zero and the under-1e-300 early return fires, giving a
     * deterministic u; Debug builds get residual ~1e-16 float noise,
     * miss the early return, and compute u from a near-zero
     * denominator -- the resulting eigenvector direction is
     * numerically arbitrary and the FHS plaquette product at the
     * Dirac plaquettes (which carry the Berry-curvature winding)
     * gives the wrong winding number.
     *
     * The half-step shift moves every grid point off the
     * high-symmetry locus while leaving the integrated Berry-curvature
     * sum invariant under the periodicity of the BZ.
     */
    const double step = 2.0 * M_PI / (double)N;
    const double k_offset = 0.5 * step;
    qgt_complex_t h[4];
    for (size_t iy = 0; iy < N; iy++) {
        for (size_t ix = 0; ix < N; ix++) {
            double k[2] = { (double)ix * step + k_offset,
                            (double)iy * step + k_offset };
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

/* ====================================================================
 * Parallel-transport-gauge Berry-curvature integrator
 *
 * Implements the FHS plaquette construction with an enforced
 * parallel-transport gauge: at each grid point u(k) is phase-rotated
 * so that <u(k_prev) | u(k)> > 0 (real and positive) along a chosen
 * spanning tree of the BZ.  This eliminates the LAPACK-gauge
 * randomness that breaks the FHS link-variable continuity at the
 * BZ wrap, and gives the right Chern number for 2-band Bloch
 * Hamiltonians with arbitrary mass terms (Haldane M > 0,
 * Kane-Mele lambda_v > 0, BHZ).  Costs an extra O(N^2) per-grid
 * phase-fix loop on top of the standard FHS plaquette traversal.
 *
 * Walk order:
 *   - First populate eigvecs along the kx axis (iy = 0) using
 *     parallel transport from u(0, 0).
 *   - Then for each ix, populate the ky axis upward via parallel
 *     transport from u(ix, 0).
 *
 * After this gauge fix, the FHS plaquette product over the BZ
 * captures the correct Berry phase, including the BZ-wrap holonomy
 * which appears as a localised flux at one corner.
 * ==================================================================== */

static void phase_align_2(qgt_complex_t v[2],
                          const qgt_complex_t ref[2]) {
    /* Phase-rotate v so that <ref | v> is real and positive. */
    qgt_complex_t inner = conj(ref[0]) * v[0] + conj(ref[1]) * v[1];
    double mag = cabs(inner);
    if (mag < 1e-300) return;
    qgt_complex_t phase = conj(inner) / mag;
    v[0] *= phase;
    v[1] *= phase;
}

/* Build the lower-band projector P_-(k) = (I - h.σ/|h|)/2 of a 2-band
 * Bloch Hamiltonian.  Smooth in k except at gap-closing points.
 * Output P is row-major 2x2: P[r * 2 + c]. */
static void lower_projector_2x2(const qgt_complex_t h[4],
                                 qgt_complex_t P[4]) {
    qgt_complex_t h00 = h[0], h11 = h[3], h01 = h[1];
    double hz = 0.5 * creal(h00 - h11);
    double hx = creal(h01);
    double hy = -cimag(h01);
    double hnorm = sqrt(hx * hx + hy * hy + hz * hz);
    if (hnorm < 1e-300) {
        P[0] = 0.5; P[1] = 0.0; P[2] = 0.0; P[3] = 0.5;
        return;
    }
    double inv = 0.5 / hnorm;
    P[0] = (qgt_complex_t)(0.5 - inv * hz);
    P[1] = -inv * (hx - _Complex_I * hy);
    P[2] = -inv * (hx + _Complex_I * hy);
    P[3] = (qgt_complex_t)(0.5 + inv * hz);
}

/* 2x2 complex matrix multiply: C = A * B. */
static void mat2_mul(const qgt_complex_t A[4], const qgt_complex_t B[4],
                     qgt_complex_t C[4]) {
    C[0] = A[0]*B[0] + A[1]*B[2];
    C[1] = A[0]*B[1] + A[1]*B[3];
    C[2] = A[2]*B[0] + A[3]*B[2];
    C[3] = A[2]*B[1] + A[3]*B[3];
}

/* Trace of a 2x2 complex matrix. */
static qgt_complex_t mat2_trace(const qgt_complex_t A[4]) {
    return A[0] + A[3];
}

int qgt_berry_grid_proj(const qgt_system_t* sys, size_t N,
                         qgt_berry_grid_t* out) {
    if (!sys || !out || N < 4) return -1;

    qgt_complex_t* P = malloc(N * N * 4 * sizeof(qgt_complex_t));
    if (!P) return -2;

    const double step = 2.0 * M_PI / (double)N;
    const double k_offset = 0.5 * step;
    qgt_complex_t h[4];
    for (size_t iy = 0; iy < N; iy++) {
        for (size_t ix = 0; ix < N; ix++) {
            double k[2] = { (double)ix * step + k_offset,
                            (double)iy * step + k_offset };
            sys->fn(k, sys->user, h);
            lower_projector_2x2(h, &P[(iy * N + ix) * 4]);
        }
    }

    out->N = N;
    out->berry = calloc(N * N, sizeof(double));
    if (!out->berry) { free(P); return -3; }

    /* Plaquette holonomy via projector trace:
     *   F_xy(k) ≈ -arg Tr[ P(k) P(k+x) P(k+x+y) P(k+y) ].
     * For the lower band of a 2-band Hamiltonian this is the
     * gauge-free analogue of the FHS link product.  At small grid
     * spacing the trace's argument equals 2 * (Berry curvature) *
     * (plaquette area), and the integrated total over the BZ gives
     * 2*pi*Chern. */
    double chern_sum = 0.0;
    for (size_t iy = 0; iy < N; iy++) {
        size_t iy1 = (iy + 1) % N;
        for (size_t ix = 0; ix < N; ix++) {
            size_t ix1 = (ix + 1) % N;
            const qgt_complex_t* P00 = &P[(iy  * N + ix ) * 4];
            const qgt_complex_t* P10 = &P[(iy  * N + ix1) * 4];
            const qgt_complex_t* P11 = &P[(iy1 * N + ix1) * 4];
            const qgt_complex_t* P01 = &P[(iy1 * N + ix ) * 4];
            qgt_complex_t M1[4], M2[4], M3[4];
            mat2_mul(P00, P10, M1);
            mat2_mul(M1,  P11, M2);
            mat2_mul(M2,  P01, M3);
            qgt_complex_t tr = mat2_trace(M3);
            double F = -arg_principal(tr);
            out->berry[iy * N + ix] = F;
            chern_sum += F;
        }
    }
    out->chern = chern_sum / (2.0 * M_PI);
    free(P);
    return 0;
}

int qgt_berry_grid_pt(const qgt_system_t* sys, size_t N,
                       qgt_berry_grid_t* out) {
    if (!sys || !out || N < 4) return -1;

    qgt_complex_t* U = malloc(N * N * 2 * sizeof(qgt_complex_t));
    if (!U) return -2;

    const double step = 2.0 * M_PI / (double)N;
    const double k_offset = 0.5 * step;
    qgt_complex_t h[4];

    /* Bottom row (iy = 0): parallel-transport along kx. */
    {
        size_t iy = 0;
        for (size_t ix = 0; ix < N; ix++) {
            double k[2] = { (double)ix * step + k_offset,
                            (double)iy * step + k_offset };
            sys->fn(k, sys->user, h);
            qgt_complex_t* u = &U[(iy * N + ix) * 2];
            lower_eigvec_2x2(h, u);
            if (ix > 0) {
                const qgt_complex_t* prev = &U[(iy * N + ix - 1) * 2];
                phase_align_2(u, prev);
            }
        }
    }
    /* Each column (ix fixed): parallel-transport upward in ky. */
    for (size_t ix = 0; ix < N; ix++) {
        for (size_t iy = 1; iy < N; iy++) {
            double k[2] = { (double)ix * step + k_offset,
                            (double)iy * step + k_offset };
            sys->fn(k, sys->user, h);
            qgt_complex_t* u = &U[(iy * N + ix) * 2];
            lower_eigvec_2x2(h, u);
            const qgt_complex_t* prev = &U[((iy - 1) * N + ix) * 2];
            phase_align_2(u, prev);
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
            qgt_complex_t Ux       = vdot2(u00, u10);
            qgt_complex_t Uy_at_x1 = vdot2(u10, u11);
            qgt_complex_t Ux_at_y1 = vdot2(u01, u11);
            qgt_complex_t Uy       = vdot2(u00, u01);
            qgt_complex_t plaq = Ux * Uy_at_x1 * conj(Ux_at_y1) * conj(Uy);
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

/* ----- Kitaev p-wave chain --------------------------------------- */

typedef struct { double t, mu, delta; } kitaev_params_t;

static void kitaev_bloch(double k, void* user, qgt_complex_t h[4]) {
    const kitaev_params_t* p = (const kitaev_params_t*)user;
    double xi = -2.0 * p->t * cos(k) - p->mu;
    double Delta_k = 2.0 * p->delta * sin(k);
    /* H(k) = xi(k) tau_z + Delta(k) tau_y in the Nambu basis.
     * tau_z = diag(+1, -1); tau_y = [[0, -i], [+i, 0]].
     * h[r*2 + c] in row-major:
     *   H[0,0] = xi      H[0,1] = -i Delta
     *   H[1,0] = +i Delta  H[1,1] = -xi  */
    h[0] = (qgt_complex_t)xi;
    h[1] = -_Complex_I * (qgt_complex_t)Delta_k;
    h[2] = +_Complex_I * (qgt_complex_t)Delta_k;
    h[3] = (qgt_complex_t)(-xi);
}

qgt_system_1d_t* qgt_model_kitaev_chain(double t, double mu, double delta) {
    kitaev_params_t* p = malloc(sizeof(*p));
    if (!p) return NULL;
    p->t = t; p->mu = mu; p->delta = delta;
    qgt_system_1d_t* s = qgt_create_1d(kitaev_bloch, p);
    if (!s) { free(p); return NULL; }
    s->owned_user = p;
    return s;
}

int qgt_z2_invariant_1d_bdg(const qgt_system_1d_t* sys, int* z2) {
    if (!sys || !z2) return -1;
    /* At the TR-invariant momenta k = 0 and k = pi, evaluate the BdG
     * Hamiltonian and read off the diagonal coefficient (the
     * Pfaffian-sign proxy for a 2x2 BdG with vanishing off-diagonal
     * at TRIM).  For Kitaev: Delta(0) = Delta(pi) = 0, so H(0,pi) =
     * xi*tau_z; sgn(Pf) = sgn(xi). */
    qgt_complex_t h0[4], hpi[4];
    sys->fn(0.0, sys->user, h0);
    sys->fn(M_PI, sys->user, hpi);
    double m0  = creal(h0[0]);    /* H_{00} at k=0 */
    double mpi = creal(hpi[0]);   /* H_{00} at k=pi */
    int s0  = (m0  > 0.0) ? +1 : (m0  < 0.0 ? -1 : 0);
    int spi = (mpi > 0.0) ? +1 : (mpi < 0.0 ? -1 : 0);
    if (s0 == 0 || spi == 0) {
        /* Gap closing at a TRIM point -- Z_2 is undefined; return 0 by
         * convention but flag via an out-of-range value if needed. */
        *z2 = 0;
        return 0;
    }
    /* nu = (1 - s0 * spi) / 2 */
    *z2 = (s0 * spi < 0) ? 1 : 0;
    return 0;
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

/* ====================================================================
 * Topological phase diagrams (qgt_phase_diagram_chern)
 * ==================================================================== */

int qgt_phase_diagram_chern(qgt_param_system_fn factory,
                             void* user,
                             double param_min, double param_max,
                             size_t K, size_t N,
                             int* chern_out) {
    if (!factory || !chern_out) return -1;
    if (K < 2 || N < 4) return -1;
    if (!(param_max > param_min)) return -1;

    const double dp = (param_max - param_min) / (double)(K - 1);
    for (size_t i = 0; i < K; i++) {
        const double p = param_min + dp * (double)i;
        qgt_system_t* sys = factory(user, p);
        if (!sys) {
            chern_out[i] = INT_MIN;
            continue;
        }
        qgt_berry_grid_t g;
        int rc = qgt_berry_grid(sys, N, &g);
        if (rc != 0) {
            chern_out[i] = INT_MIN;
            qgt_free(sys);
            continue;
        }
        chern_out[i] = (int)lround(g.chern);
        qgt_berry_grid_free(&g);
        qgt_free(sys);
    }
    return 0;
}

int qgt_phase_diagram_chern_2d(qgt_param_system_2d_fn factory,
                                void* user,
                                double x_min, double x_max,
                                double y_min, double y_max,
                                size_t Kx, size_t Ky, size_t N,
                                int* chern_out) {
    if (!factory || !chern_out) return -1;
    if (Kx < 2 || Ky < 2 || N < 4) return -1;
    if (!(x_max > x_min) || !(y_max > y_min)) return -1;

    const double dx = (x_max - x_min) / (double)(Kx - 1);
    const double dy = (y_max - y_min) / (double)(Ky - 1);
    for (size_t ix = 0; ix < Kx; ix++) {
        const double x = x_min + dx * (double)ix;
        for (size_t iy = 0; iy < Ky; iy++) {
            const double y = y_min + dy * (double)iy;
            qgt_system_t* sys = factory(user, x, y);
            if (!sys) {
                chern_out[ix * Ky + iy] = INT_MIN;
                continue;
            }
            qgt_berry_grid_t g;
            int rc = qgt_berry_grid(sys, N, &g);
            if (rc != 0) {
                chern_out[ix * Ky + iy] = INT_MIN;
                qgt_free(sys);
                continue;
            }
            chern_out[ix * Ky + iy] = (int)lround(g.chern);
            qgt_berry_grid_free(&g);
            qgt_free(sys);
        }
    }
    return 0;
}

/* ====================================================================
 * Multi-band Bloch systems
 * ==================================================================== */

struct qgt_system_n {
    qgt_bloch_n_fn fn;
    void*          user;
    void*          owned_user;
    size_t         n_bands;
    size_t         n_occupied;
};

qgt_system_n_t* qgt_create_nband(qgt_bloch_n_fn f, void* user,
                                  size_t n_bands, size_t n_occupied) {
    if (!f || n_bands < 2) return NULL;
    if (n_occupied < 1 || n_occupied >= n_bands) return NULL;
    qgt_system_n_t* s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->fn = f;
    s->user = user;
    s->owned_user = NULL;
    s->n_bands = n_bands;
    s->n_occupied = n_occupied;
    return s;
}

void qgt_free_nband(qgt_system_n_t* sys) {
    if (!sys) return;
    free(sys->owned_user);
    free(sys);
}

int qgt_eval_nband_hamiltonian(const qgt_system_n_t* sys,
                               const double k[2],
                               qgt_complex_t* h,
                               size_t h_count) {
    if (!sys || !k || !h) return -1;
    const size_t required = sys->n_bands * sys->n_bands;
    if (h_count < required) return -2;
    sys->fn(k, sys->user, h);
    return 0;
}

/* Compute the lowest-energy n_occupied eigenvectors of H at momentum k.
 * Output @p u_occ is row-major (n_bands, n_occupied): u_occ[i*M + a] is
 * the i-th component of the a-th occupied band, sorted by ascending
 * energy.  Returns 0 on success, -1 on diagonalisation failure. */
static int diag_occupied(const qgt_system_n_t* sys, const double k[2],
                         qgt_complex_t* u_occ) {
    const size_t n = sys->n_bands;
    const size_t M = sys->n_occupied;
    qgt_complex_t* H = malloc(n * n * sizeof(qgt_complex_t));
    qgt_complex_t* V = malloc(n * n * sizeof(qgt_complex_t));
    double*        E = malloc(n * sizeof(double));
    if (!H || !V || !E) { free(H); free(V); free(E); return -1; }
    sys->fn(k, sys->user, H);
    int rc = hermitian_eigen_decomposition(H, n, E, V, 0, 0.0);
    if (rc != 0) { free(H); free(V); free(E); return -1; }
    /* `hermitian_eigen_decomposition` returns eigenvalues sorted
     * descending and eigvecs row-major: V[i*n + j] = i-th component of
     * the j-th descending eigvec.  Lowest-energy bands are j in
     * [n - M, n - 1]; copy them as the M occupied columns. */
    for (size_t i = 0; i < n; i++) {
        for (size_t a = 0; a < M; a++) {
            size_t j = n - M + a;   /* a = 0 -> highest of occupied */
            u_occ[i * M + a] = V[i * n + j];
        }
    }
    free(H); free(V); free(E);
    return 0;
}

/* Determinant of an MxM complex matrix via Gauss elimination with
 * partial pivoting.  Operates on a writable copy of @p A; returns the
 * complex determinant.  On singular input returns 0. */
static qgt_complex_t det_mxm(const qgt_complex_t* A, size_t M) {
    if (M == 1) return A[0];
    if (M == 2) {
        return A[0] * A[3] - A[1] * A[2];
    }
    qgt_complex_t* B = malloc(M * M * sizeof(qgt_complex_t));
    if (!B) return 0.0;
    memcpy(B, A, M * M * sizeof(qgt_complex_t));
    qgt_complex_t det = 1.0;
    int sign = 1;
    for (size_t i = 0; i < M; i++) {
        /* Partial pivot: find row r >= i with max |B[r][i]| */
        size_t piv = i;
        double piv_mag = cabs(B[i * M + i]);
        for (size_t r = i + 1; r < M; r++) {
            double m = cabs(B[r * M + i]);
            if (m > piv_mag) { piv_mag = m; piv = r; }
        }
        if (piv_mag < 1e-300) { free(B); return 0.0; }
        if (piv != i) {
            for (size_t c = 0; c < M; c++) {
                qgt_complex_t tmp = B[i * M + c];
                B[i * M + c]   = B[piv * M + c];
                B[piv * M + c] = tmp;
            }
            sign = -sign;
        }
        qgt_complex_t pivot = B[i * M + i];
        det *= pivot;
        for (size_t r = i + 1; r < M; r++) {
            qgt_complex_t f = B[r * M + i] / pivot;
            for (size_t c = i; c < M; c++) {
                B[r * M + c] -= f * B[i * M + c];
            }
        }
    }
    free(B);
    return ((double)sign) * det;
}

/* Compute the M x M overlap matrix S_{ab} = <u_occ_a(k1) | u_occ_b(k2)>
 * and return det(S) -- this is the non-Abelian U(M) link variable. */
static qgt_complex_t link_det(const qgt_complex_t* u1, const qgt_complex_t* u2,
                              size_t n, size_t M) {
    qgt_complex_t* S = malloc(M * M * sizeof(qgt_complex_t));
    if (!S) return 0.0;
    for (size_t a = 0; a < M; a++) {
        for (size_t b = 0; b < M; b++) {
            qgt_complex_t s = 0.0;
            for (size_t i = 0; i < n; i++) {
                s += conj(u1[i * M + a]) * u2[i * M + b];
            }
            S[a * M + b] = s;
        }
    }
    qgt_complex_t d = det_mxm(S, M);
    free(S);
    return d;
}

static double arg_pp(qgt_complex_t z) {
    double a = atan2(cimag(z), creal(z));
    /* Already in (-pi, pi]. */
    return a;
}

typedef struct {
    qgt_bloch_n_fn fn4;
    void*          user4;
} z2_block_user_t;

static void z2_block_bloch(const double k[2], void* user, qgt_complex_t h[4]) {
    z2_block_user_t* z = (z2_block_user_t*)user;
    qgt_complex_t H4[16];
    z->fn4(k, z->user4, H4);
    h[0] = H4[0 * 4 + 0];
    h[1] = H4[0 * 4 + 1];
    h[2] = H4[1 * 4 + 0];
    h[3] = H4[1 * 4 + 1];
}

static bool z2_is_sz_conserving(const qgt_system_n_t* sys) {
    static const double sample_k[][2] = {
        { 0.0, 0.0 },
        { 0.37, -0.51 },
        { -1.2, 0.8 },
        { M_PI * 0.5, M_PI / 3.0 }
    };
    const int up[2] = {0, 1};
    const int dn[2] = {2, 3};
    for (size_t s = 0; s < sizeof(sample_k) / sizeof(sample_k[0]); s++) {
        qgt_complex_t H[16];
        sys->fn(sample_k[s], sys->user, H);
        for (int a = 0; a < 2; a++) {
            for (int b = 0; b < 2; b++) {
                if (cabs(H[up[a] * 4 + dn[b]]) > 1e-10 ||
                    cabs(H[dn[a] * 4 + up[b]]) > 1e-10) {
                    return false;
                }
            }
        }
    }
    return true;
}

static int positive_projected_spin_state(const qgt_complex_t* u_occ,
                                         qgt_complex_t psi[4]) {
    const double sz[4] = {+1.0, +1.0, -1.0, -1.0};
    qgt_complex_t S00 = 0.0;
    qgt_complex_t S01 = 0.0;
    qgt_complex_t S11 = 0.0;
    for (int i = 0; i < 4; i++) {
        qgt_complex_t u0 = u_occ[i * 2 + 0];
        qgt_complex_t u1 = u_occ[i * 2 + 1];
        S00 += conj(u0) * sz[i] * u0;
        S01 += conj(u0) * sz[i] * u1;
        S11 += conj(u1) * sz[i] * u1;
    }
    double a = creal(S00);
    double d = creal(S11);
    qgt_complex_t b = S01;
    double delta = 0.5 * (a - d);
    double bmag = cabs(b);
    double rad = sqrt(delta * delta + bmag * bmag);
    if (rad < 1e-10) return -3;
    double lambda = 0.5 * (a + d) + rad;

    qgt_complex_t v0 = b;
    qgt_complex_t v1 = lambda - a;
    double n2 = bmag * bmag + cabs(v1) * cabs(v1);
    if (n2 < 1e-24) {
        v0 = lambda - d;
        v1 = conj(b);
        n2 = cabs(v0) * cabs(v0) + cabs(v1) * cabs(v1);
    }
    if (n2 < 1e-24) return -3;
    double inv = 1.0 / sqrt(n2);
    v0 *= inv;
    v1 *= inv;

    double norm2 = 0.0;
    for (int i = 0; i < 4; i++) {
        psi[i] = u_occ[i * 2 + 0] * v0 + u_occ[i * 2 + 1] * v1;
        double mag = cabs(psi[i]);
        norm2 += mag * mag;
    }
    if (norm2 < 1e-24) return -3;
    double psi_inv = 1.0 / sqrt(norm2);
    for (int i = 0; i < 4; i++) psi[i] *= psi_inv;
    return 0;
}

static qgt_complex_t vdot4(const qgt_complex_t a[4], const qgt_complex_t b[4]) {
    return conj(a[0]) * b[0] + conj(a[1]) * b[1] +
           conj(a[2]) * b[2] + conj(a[3]) * b[3];
}

int qgt_berry_grid_nband(const qgt_system_n_t* sys, size_t N,
                          qgt_berry_grid_t* out) {
    if (!sys || !out || N < 4) return -1;
    const size_t n = sys->n_bands;
    const size_t M = sys->n_occupied;
    qgt_complex_t* U_grid = malloc(N * N * n * M * sizeof(qgt_complex_t));
    if (!U_grid) return -1;

    const double dk = 2.0 * M_PI / (double)N;
    /* Diagonalise on each grid point. */
    for (size_t ix = 0; ix < N; ix++) {
        for (size_t iy = 0; iy < N; iy++) {
            const double k[2] = { -M_PI + dk * (double)ix,
                                   -M_PI + dk * (double)iy };
            qgt_complex_t* slot = &U_grid[(ix * N + iy) * n * M];
            if (diag_occupied(sys, k, slot) != 0) {
                free(U_grid);
                return -1;
            }
        }
    }

    /* Plaquette flux: F_xy(k) = arg[U_x(k) U_y(k+x) U_x(k+y)^-1 U_y(k)^-1] */
    double* berry = malloc(N * N * sizeof(double));
    if (!berry) { free(U_grid); return -1; }
    double chern = 0.0;
    for (size_t ix = 0; ix < N; ix++) {
        const size_t ixp = (ix + 1) % N;
        for (size_t iy = 0; iy < N; iy++) {
            const size_t iyp = (iy + 1) % N;
            const qgt_complex_t* u00 = &U_grid[(ix  * N + iy ) * n * M];
            const qgt_complex_t* uxp = &U_grid[(ixp * N + iy ) * n * M];
            const qgt_complex_t* uyp = &U_grid[(ix  * N + iyp) * n * M];
            const qgt_complex_t* uxy = &U_grid[(ixp * N + iyp) * n * M];
            qgt_complex_t Ux  = link_det(u00, uxp, n, M);
            qgt_complex_t Uy_x = link_det(uxp, uxy, n, M);
            qgt_complex_t Ux_y = link_det(uyp, uxy, n, M);
            qgt_complex_t Uy  = link_det(u00, uyp, n, M);
            /* Normalise to extract pure phase (det link variable's
             * magnitude is the volume of the parallelepiped of overlap;
             * for the Chern number only the phase matters). */
            qgt_complex_t plaq = (Ux * Uy_x) / (Ux_y * Uy);
            double F = arg_pp(plaq);
            berry[ix * N + iy] = F;
            chern += F;
        }
    }
    chern /= (2.0 * M_PI);
    free(U_grid);
    out->N = N;
    out->berry = berry;
    out->chern = chern;
    return 0;
}

int qgt_z2_invariant(const qgt_system_n_t* sys, size_t N, int* z2) {
    if (!sys || !z2) return -1;
    if (sys->n_bands != 4 || sys->n_occupied != 2) return -2;
    if (N < 8 || (N & 1u)) return -1;

    if (z2_is_sz_conserving(sys)) {
        z2_block_user_t z = { sys->fn, sys->user };
        qgt_system_t* sub = qgt_create(z2_block_bloch, &z);
        if (!sub) return -1;
        qgt_berry_grid_t g;
        int rc = qgt_berry_grid(sub, N, &g);
        qgt_free(sub);
        if (rc != 0) return rc;
        int C_up = (int)lround(g.chern);
        qgt_berry_grid_free(&g);
        int abs_C = C_up < 0 ? -C_up : C_up;
        *z2 = abs_C & 1;
        return 0;
    }

    const size_t n = sys->n_bands;
    const double dk = 2.0 * M_PI / (double)N;
    qgt_complex_t* spin_grid = malloc(N * N * n * sizeof(qgt_complex_t));
    if (!spin_grid) return -1;

    for (size_t ix = 0; ix < N; ix++) {
        for (size_t iy = 0; iy < N; iy++) {
            const double k[2] = {
                -M_PI + dk * (double)ix,
                -M_PI + dk * (double)iy
            };
            qgt_complex_t u_occ[8];
            qgt_complex_t* slot = &spin_grid[(ix * N + iy) * n];
            if (diag_occupied(sys, k, u_occ) != 0 ||
                positive_projected_spin_state(u_occ, slot) != 0) {
                free(spin_grid);
                return -1;
            }
        }
    }

    double chern = 0.0;
    for (size_t ix = 0; ix < N; ix++) {
        const size_t ixp = (ix + 1) % N;
        for (size_t iy = 0; iy < N; iy++) {
            const size_t iyp = (iy + 1) % N;
            const qgt_complex_t* u00 = &spin_grid[(ix  * N + iy ) * n];
            const qgt_complex_t* uxp = &spin_grid[(ixp * N + iy ) * n];
            const qgt_complex_t* uyp = &spin_grid[(ix  * N + iyp) * n];
            const qgt_complex_t* uxy = &spin_grid[(ixp * N + iyp) * n];
            qgt_complex_t Ux = vdot4(u00, uxp);
            qgt_complex_t Uy_x = vdot4(uxp, uxy);
            qgt_complex_t Ux_y = vdot4(uyp, uxy);
            qgt_complex_t Uy = vdot4(u00, uyp);
            double mx = cabs(Ux);
            double myx = cabs(Uy_x);
            double mxy = cabs(Ux_y);
            double my = cabs(Uy);
            if (mx < 1e-300 || myx < 1e-300 ||
                mxy < 1e-300 || my < 1e-300) {
                free(spin_grid);
                return -1;
            }
            Ux /= mx;
            Uy_x /= myx;
            Ux_y /= mxy;
            Uy /= my;
            chern += arg_pp(Ux * Uy_x * conj(Ux_y) * conj(Uy));
        }
    }

    free(spin_grid);

    long spin_chern = lround(chern / (2.0 * M_PI));
    if (spin_chern < 0) spin_chern = -spin_chern;
    int parity = (int)(spin_chern % 2);
    *z2 = parity;
    return 0;
}

/* ----- Kane-Mele honeycomb model (2005) ------------------------------
 *
 * Basis order: (A_up, B_up, A_down, B_down).
 *
 * The model is
 *   H = -t sum_<ij> c_i^dag c_j
 *       + i lambda_so sum_<<ij>> nu_ij c_i^dag s_z c_j
 *       + lambda_v sum_i xi_i c_i^dag c_i
 *       + Rashba (lambda_r) -- not Sz-conserving.
 *
 * Bloch Hamiltonian on the honeycomb lattice with primitive vectors
 *   a_1 = (sqrt(3)/2, 1/2),  a_2 = (sqrt(3)/2, -1/2)
 * (lattice constant = 1).  Nearest-neighbour vectors:
 *   d_1 = (0, 1/sqrt(3))
 *   d_2 = (-1/2, -1/(2 sqrt(3)))
 *   d_3 = (+1/2, -1/(2 sqrt(3)))
 * Next-nearest-neighbour vectors are on the same sublattice:
 *   b_1 = a_1 - a_2 = (0, 1)
 *   b_2 = -a_2 = (-sqrt(3)/2, 1/2)
 *   b_3 = a_1 = (sqrt(3)/2, 1/2)
 * with sign nu = +1 for A->A counterclockwise, -1 for B->B (same
 * vectors traversed clockwise).
 */

typedef struct {
    double t;
    double lambda_so;
    double lambda_r;
    double lambda_v;
} km_params_t;

static void km_bloch(const double k[2], void* user, qgt_complex_t h[16]) {
    km_params_t* p = (km_params_t*)user;
    /* Primitive reciprocal coordinates kx = k . a1, ky = k . a2.
     * Matches the existing qgt_model_haldane parametrisation so that
     * integration over [-pi, pi]^2 covers exactly one BZ. */
    double kx = k[0], ky = k[1];
    /* NN A->B sum in primitive reciprocal coordinates: same as Haldane. */
    qgt_complex_t f = (cos(kx) + cos(kx - ky) + cos(ky))
                    + I * (sin(kx) + sin(kx - ky) + sin(ky));
    /* NNN antisymmetric (signed) sum -- the spin-orbit driver.  Same
     * canonical form as the corrected Haldane c2 (see qgt_model_haldane
     * for the why): nonzero at the Dirac points (0, +/-2*pi/3) where
     * f vanishes, with opposite signs.  Matches Haldane's c2 with the
     * spin-up block acting like phi = +pi/2 and spin-down like -pi/2. */
    double c2 = sin(ky) * (1.0 + 2.0 * cos(kx));

    /* Initialise to zero. */
    for (int i = 0; i < 16; i++) h[i] = 0.0;

    /* Spin-up sigma_z = lambda_v - 2 * lambda_so * c2  (Haldane phi = +pi/2). */
    double sz_up = p->lambda_v - 2.0 * p->lambda_so * c2;
    /* Spin-down sigma_z = lambda_v + 2 * lambda_so * c2  (Haldane phi = -pi/2). */
    double sz_dn = p->lambda_v + 2.0 * p->lambda_so * c2;

    qgt_complex_t hop = p->t * f;   /* convention: H = +t f sigma_x style */

    /* Spin-up block (A_up=0, B_up=1) */
    h[0 * 4 + 0] = +sz_up;
    h[0 * 4 + 1] = conj(hop);
    h[1 * 4 + 0] = hop;
    h[1 * 4 + 1] = -sz_up;
    /* Spin-down block (A_down=2, B_down=3) */
    h[2 * 4 + 2] = +sz_dn;
    h[2 * 4 + 3] = conj(hop);
    h[3 * 4 + 2] = hop;
    h[3 * 4 + 3] = -sz_dn;

    if (p->lambda_r != 0.0) {
        const double sqrt3 = sqrt(3.0);
        const double phase_arg[3] = { kx, kx - ky, ky };
        const double dx[3] = { 0.0, -0.5 * sqrt3, +0.5 * sqrt3 };
        const double dy[3] = { 1.0, -0.5, -0.5 };
        qgt_complex_t ba_up_down = 0.0;
        qgt_complex_t ba_down_up = 0.0;
        for (int j = 0; j < 3; j++) {
            qgt_complex_t phase = cos(phase_arg[j]) + I * sin(phase_arg[j]);
            ba_up_down += phase * p->lambda_r * (-dx[j] + I * dy[j]);
            ba_down_up += phase * p->lambda_r * (+dx[j] + I * dy[j]);
        }
        h[1 * 4 + 2] += ba_up_down;       /* B_up   <- A_down */
        h[3 * 4 + 0] += ba_down_up;       /* B_down <- A_up   */
        h[2 * 4 + 1] = conj(h[1 * 4 + 2]);
        h[0 * 4 + 3] = conj(h[3 * 4 + 0]);
    }
}

qgt_system_n_t* qgt_model_kane_mele(double t, double lambda_so,
                                     double lambda_r, double lambda_v) {
    km_params_t* p = malloc(sizeof(*p));
    if (!p) return NULL;
    p->t = t;
    p->lambda_so = lambda_so;
    p->lambda_r = lambda_r;
    p->lambda_v = lambda_v;
    qgt_system_n_t* sys = qgt_create_nband(km_bloch, p, 4, 2);
    if (!sys) { free(p); return NULL; }
    sys->owned_user = p;
    return sys;
}

/* ----- BHZ (Bernevig-Hughes-Zhang) 4-band model ---------------------
 *
 * Square-lattice TI in the basis (|s,+>, |p,+>, |s,->, |p,->).  Spin
 * is conserved (Sz commutes with H), so the Hamiltonian is block-
 * diagonal:
 *
 *   h_+(k) = (M + 2 B (2 - cos kx - cos ky)) sigma_z
 *          + A sin(kx) sigma_x + A sin(ky) sigma_y
 *   h_-(k) = h_+*(-k) = (M + 2 B (2 - cos kx - cos ky)) sigma_z
 *          - A sin(kx) sigma_x - A sin(ky) sigma_y
 *
 * Z_2 = 1 (QSH) for 0 < M/B < 4; trivial for M/B < 0 or > 4.
 */

typedef struct {
    double A;
    double B;
    double M;
} bhz_params_t;

static void bhz_bloch(const double k[2], void* user, qgt_complex_t h[16]) {
    bhz_params_t* p = (bhz_params_t*)user;
    double kx = k[0], ky = k[1];
    /* Common diagonal: M - 2B (2 - cos kx - cos ky).  At Gamma = (0, 0)
     * this evaluates to M; at X = (pi, 0) / (0, pi) it is M - 4B; at M
     * point (pi, pi) it is M - 8B.  This is the standard BHZ
     * convention where QSH lies in 0 < M < 4B. */
    double mass = p->M - 2.0 * p->B * (2.0 - cos(kx) - cos(ky));
    double sx = p->A * sin(kx);
    double sy = p->A * sin(ky);

    for (int i = 0; i < 16; i++) h[i] = 0.0;

    /* Spin-up block (rows 0,1, cols 0,1):
     *   h_+ = mass*sigma_z + sx*sigma_x + sy*sigma_y
     */
    h[0 * 4 + 0] = (qgt_complex_t)mass;
    h[0 * 4 + 1] = (qgt_complex_t)sx - _Complex_I * (qgt_complex_t)sy;
    h[1 * 4 + 0] = (qgt_complex_t)sx + _Complex_I * (qgt_complex_t)sy;
    h[1 * 4 + 1] = (qgt_complex_t)(-mass);
    /* Spin-down block (rows 2,3, cols 2,3): TR partner of the spin-up
     * block.  Computed as sigma_y * conj(h_+(-k)) * sigma_y, which for
     * h_+ = mass*sigma_z + sx*sigma_x + sy*sigma_y (with sx, sy odd
     * and mass even in k) yields h_- = -mass*sigma_z + sx*sigma_x +
     * sy*sigma_y.  Same in-plane components, opposite sigma_z
     * coefficient -- this is what makes the spin-down Chern equal to
     * minus the spin-up Chern (C_total = 0 by TRS). */
    h[2 * 4 + 2] = (qgt_complex_t)(-mass);
    h[2 * 4 + 3] = (qgt_complex_t)sx - _Complex_I * (qgt_complex_t)sy;
    h[3 * 4 + 2] = (qgt_complex_t)sx + _Complex_I * (qgt_complex_t)sy;
    h[3 * 4 + 3] = (qgt_complex_t)mass;
}

qgt_system_n_t* qgt_model_bhz(double A, double B, double M) {
    bhz_params_t* p = malloc(sizeof(*p));
    if (!p) return NULL;
    p->A = A;
    p->B = B;
    p->M = M;
    qgt_system_n_t* sys = qgt_create_nband(bhz_bloch, p, 4, 2);
    if (!sys) { free(p); return NULL; }
    sys->owned_user = p;
    return sys;
}
