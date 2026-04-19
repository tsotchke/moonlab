/**
 * @file chern_kpm.c
 * @brief Matrix-free Chebyshev-KPM Chern marker for QWZ.
 */

#include "chern_kpm.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef double _Complex cm_complex_t;

/* Hopping matrices for QWZ (same derivation as in chern_marker.c):
 *   T_x = (sigma_z + i sigma_x) / 2
 *   T_y = (sigma_z + i sigma_y) / 2
 * H connects site (x,y) to (x+1,y) via T_x on the forward bond and
 * T_x^dagger on the reverse; similarly for y. */
static const cm_complex_t Tx_[2][2] = {
    {  0.5,              0.5 * _Complex_I },
    {  0.5 * _Complex_I, -0.5             }
};
static const cm_complex_t Ty_[2][2] = {
    {  0.5,  0.5 },
    { -0.5, -0.5 }
};

static inline size_t kidx(size_t L, size_t x, size_t y, size_t s) {
    return (y * L + x) * 2 + s;
}

chern_kpm_system_t* chern_kpm_create(size_t L, double m, size_t n_cheby) {
    if (L < 3 || n_cheby < 8) return NULL;
    chern_kpm_system_t* sys = calloc(1, sizeof(*sys));
    if (!sys) return NULL;
    sys->L = L;
    sys->orbs = 2;
    sys->N = L * L * 2;
    sys->m = m;
    sys->n_cheby = n_cheby;
    /* Spectrum of QWZ is bounded by |m| + 2 (one unit each from
     * sin(kx) sigma_x etc.). Add a safety factor to keep H_hat
     * strictly inside [-1, 1] so the Chebyshev recurrence is stable. */
    sys->E_shift = 0.0;
    sys->E_scale = fabs(m) + 2.0 + 0.1;
    sys->modulation = NULL;
    sys->mod_maxabs = 0.0;
    return sys;
}

void chern_kpm_free(chern_kpm_system_t* sys) { free(sys); }

int chern_kpm_set_modulation(chern_kpm_system_t* sys,
                             const double* V_per_site,
                             double V_maxabs) {
    if (!sys) return -1;
    if (V_maxabs < 0.0) return -2;
    sys->modulation = (double*)V_per_site;   /* borrowed pointer */
    sys->mod_maxabs = V_maxabs;
    sys->E_scale = fabs(sys->m) + 2.0 + V_maxabs + 0.1;
    return 0;
}

double* chern_kpm_cn_modulation(size_t L, int n, double Q, double V0) {
    if (L < 3 || n < 2) return NULL;
    double* V = malloc(L * L * sizeof(double));
    if (!V) return NULL;
    for (size_t y = 0; y < L; y++) {
        for (size_t x = 0; x < L; x++) {
            double acc = 0.0;
            for (int i = 0; i < n; i++) {
                double theta = 2.0 * M_PI * (double)i / (double)n;
                double qx = Q * cos(theta);
                double qy = Q * sin(theta);
                acc += cos(qx * (double)x + qy * (double)y);
            }
            V[y * L + x] = V0 * acc;
        }
    }
    return V;
}

/* Apply rescaled Hamiltonian H_hat = H / E_scale to input vector.
 *   (H v)[x,y,s_out] = Sum_s (m sigma_z)[s_out,s] v[x,y,s]
 *                    + Sum_{s,dir} hop[s_out,s] v[neighbour,s]
 *
 * T_x -- forward bond: H_{x,y,s_o; x+1,y,s_i} -- what we add at (x,y) is
 * from the ket at (x+1, y). But my dense constructor uses
 * H[(x+1,y),s_o; (x,y),s_i] = T_x[s_o, s_i], so (H v)[x+1,y,s_o] gets
 * T_x[s_o,s_i] v[x,y,s_i]. Equivalently at site (x,y) the "incoming"
 * contribution from (x-1, y) is T_x[s_out, s_in] v[x-1, y, s_in].
 */
static void matvec_h_hat(const chern_kpm_system_t* sys,
                         const cm_complex_t* in, cm_complex_t* out) {
    size_t L = sys->L;
    double m = sys->m;
    double b = sys->E_scale;
    const double* V = sys->modulation;

    for (size_t y = 0; y < L; y++) {
        for (size_t x = 0; x < L; x++) {
            double Vxy = V ? V[y * L + x] : 0.0;
            for (size_t s_out = 0; s_out < 2; s_out++) {
                cm_complex_t acc = 0;
                double msz = (s_out == 0) ? m : -m;
                acc += (msz + Vxy) * in[kidx(L, x, y, s_out)];

                /* contribution from (x-1, y) via T_x forward hop */
                if (x > 0) {
                    acc += Tx_[s_out][0] * in[kidx(L, x - 1, y, 0)]
                         + Tx_[s_out][1] * in[kidx(L, x - 1, y, 1)];
                }
                /* contribution from (x+1, y) via T_x^dagger */
                if (x + 1 < L) {
                    acc += conj(Tx_[0][s_out]) * in[kidx(L, x + 1, y, 0)]
                         + conj(Tx_[1][s_out]) * in[kidx(L, x + 1, y, 1)];
                }
                /* y direction */
                if (y > 0) {
                    acc += Ty_[s_out][0] * in[kidx(L, x, y - 1, 0)]
                         + Ty_[s_out][1] * in[kidx(L, x, y - 1, 1)];
                }
                if (y + 1 < L) {
                    acc += conj(Ty_[0][s_out]) * in[kidx(L, x, y + 1, 0)]
                         + conj(Ty_[1][s_out]) * in[kidx(L, x, y + 1, 1)];
                }
                out[kidx(L, x, y, s_out)] = acc / b;
            }
        }
    }
}

static inline double jackson(size_t n, size_t N) {
    double arg = M_PI / (double)(N + 1);
    return ((double)(N - n + 1) * cos((double)n * arg)
            + sin((double)n * arg) / tan(arg)) / (double)(N + 1);
}

/* Chebyshev coefficients of sign(epsilon) on [-1, 1]:
 *   c_n = (4 / (n pi)) * sin(n pi / 2)        for n > 0
 *   c_0 = 0
 * n even -> zero. */
static inline double sign_cheby_coeff(size_t n) {
    if (n == 0) return 0.0;
    if (n % 2 == 0) return 0.0;
    double s = (((n - 1) / 2) % 2 == 0) ? 1.0 : -1.0;
    return 4.0 * s / (M_PI * (double)n);
}

/* Apply P = (I - sign(H_hat)) / 2 to v, storing the result in out.
 * Requires scratch buffers tA, tB, tC all of length sys->N. */
static void apply_projector(const chern_kpm_system_t* sys,
                            const cm_complex_t* v,
                            cm_complex_t* out,
                            cm_complex_t* tA,
                            cm_complex_t* tB,
                            cm_complex_t* tC) {
    size_t N = sys->N;
    size_t M = sys->n_cheby;

    /* tA = T_0 v = v, tB = T_1 v = H_hat v. */
    memcpy(tA, v, N * sizeof(cm_complex_t));
    matvec_h_hat(sys, v, tB);

    /* Accumulator: sign v ~= sum_{n=0..M-1} g_n c_n T_n v.
     * n=0: c_0 = 0 so skip.
     * n=1: add g_1 c_1 tB. */
    memset(out, 0, N * sizeof(cm_complex_t));
    double w1 = jackson(1, M) * sign_cheby_coeff(1);
    for (size_t i = 0; i < N; i++) out[i] += w1 * tB[i];

    for (size_t n = 2; n < M; n++) {
        /* tC = 2 H_hat tB - tA */
        matvec_h_hat(sys, tB, tC);
        for (size_t i = 0; i < N; i++) tC[i] = 2.0 * tC[i] - tA[i];
        double wn = jackson(n, M) * sign_cheby_coeff(n);
        if (wn != 0.0) {
            for (size_t i = 0; i < N; i++) out[i] += wn * tC[i];
        }
        /* advance: tA = tB, tB = tC. */
        cm_complex_t* tmp = tA; tA = tB; tB = tC; tC = tmp;
    }

    /* P v = (v - sign v) / 2. */
    for (size_t i = 0; i < N; i++) {
        out[i] = 0.5 * (v[i] - out[i]);
    }
}

static inline void apply_position(size_t L, int use_y, cm_complex_t* v) {
    for (size_t y = 0; y < L; y++) {
        for (size_t x = 0; x < L; x++) {
            double c = use_y ? (double)y : (double)x;
            v[kidx(L, x, y, 0)] *= c;
            v[kidx(L, x, y, 1)] *= c;
        }
    }
}

double chern_kpm_local_marker(const chern_kpm_system_t* sys,
                              size_t rx, size_t ry) {
    if (!sys) return 0.0;
    size_t L = sys->L;
    size_t N = sys->N;

    cm_complex_t* v   = calloc(N, sizeof(cm_complex_t));
    cm_complex_t* out = calloc(N, sizeof(cm_complex_t));
    cm_complex_t* tA  = calloc(N, sizeof(cm_complex_t));
    cm_complex_t* tB  = calloc(N, sizeof(cm_complex_t));
    cm_complex_t* tC  = calloc(N, sizeof(cm_complex_t));
    cm_complex_t* ws  = calloc(N, sizeof(cm_complex_t));

    double total_im = 0.0;
    for (size_t s = 0; s < 2; s++) {
        memset(v, 0, N * sizeof(cm_complex_t));
        v[kidx(L, rx, ry, s)] = 1.0;

        /* w = P v */
        apply_projector(sys, v, out, tA, tB, tC);
        memcpy(ws, out, N * sizeof(cm_complex_t));
        /* ws <- Y ws */
        apply_position(L, 1, ws);
        /* P ws, then subtract from ws to get Q ws */
        apply_projector(sys, ws, out, tA, tB, tC);
        for (size_t i = 0; i < N; i++) ws[i] -= out[i];
        /* ws <- X ws */
        apply_position(L, 0, ws);
        /* ws <- P ws */
        apply_projector(sys, ws, out, tA, tB, tC);

        cm_complex_t val = out[kidx(L, rx, ry, s)];
        total_im += cimag(val);
    }

    free(v); free(out); free(tA); free(tB); free(tC); free(ws);
    return -4.0 * M_PI * total_im;
}

double chern_kpm_bulk_sum(const chern_kpm_system_t* sys,
                          size_t rmin, size_t rmax) {
    if (!sys) return 0.0;
    long long side = (long long)(rmax - rmin);
    long long total_sites = side * side;
    double total = 0.0;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:total)
#endif
    for (long long k = 0; k < total_sites; k++) {
        size_t x = rmin + (size_t)(k % side);
        size_t y = rmin + (size_t)(k / side);
        total += chern_kpm_local_marker(sys, x, y);
    }
    return total;
}

int chern_kpm_bulk_map(const chern_kpm_system_t* sys,
                       size_t rmin, size_t rmax,
                       double* out) {
    if (!sys || !out || rmin >= rmax) return -1;
    long long side = (long long)(rmax - rmin);
    long long total_sites = side * side;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
#endif
    for (long long k = 0; k < total_sites; k++) {
        size_t x = rmin + (size_t)(k % side);
        size_t y = rmin + (size_t)(k / side);
        out[k] = chern_kpm_local_marker(sys, x, y);
    }
    return 0;
}
