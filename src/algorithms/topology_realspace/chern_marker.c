/**
 * @file chern_marker.c
 * @brief Dense-reference implementation of the local Chern marker.
 */

#include "chern_marker.h"

#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static inline size_t idx(size_t L, size_t orbs,
                         size_t x, size_t y, size_t s) {
    return (y * L + x) * orbs + s;
}

/* Add A[i][j] += v in a flat row-major n x n matrix. */
static inline void add(cm_complex_t* A, size_t n,
                       size_t i, size_t j, cm_complex_t v) {
    A[i * n + j] += v;
}

chern_system_t* chern_qwz_create(size_t L, double m) {
    if (L < 3) return NULL;

    chern_system_t* sys = calloc(1, sizeof(*sys));
    if (!sys) return NULL;

    sys->L = L;
    sys->orbs = 2;
    sys->m = m;
    sys->dim = L * L * 2;

    size_t N = sys->dim;
    sys->H = calloc(N * N, sizeof(cm_complex_t));
    if (!sys->H) { free(sys); return NULL; }

    /* On-site: m sigma_z. |s=0> -> +m, |s=1> -> -m. */
    for (size_t y = 0; y < L; y++) {
        for (size_t x = 0; x < L; x++) {
            add(sys->H, N, idx(L, 2, x, y, 0), idx(L, 2, x, y, 0), m);
            add(sys->H, N, idx(L, 2, x, y, 1), idx(L, 2, x, y, 1), -m);
        }
    }

    /* T(+x) = (sigma_z + i sigma_x) / 2 so that
     *   sum_R T(R) e^{-ik.R} + h.c. reproduces sin(kx) sigma_x +
     *   cos(kx) sigma_z. Matrix form:
     *     T_x = [[ 1/2 , i/2 ],
     *            [ i/2 , -1/2 ]]
     * We add H[(x+1,y),s'; (x,y),s] += T_x[s',s]; Hermitian conjugate
     * goes on the reverse direction. */
    const cm_complex_t Tx[2][2] = {
        {  0.5,              0.5 * _Complex_I },
        {  0.5 * _Complex_I, -0.5             }
    };
    /* T(+y) = (sigma_z + i sigma_y) / 2 reproducing sin(ky) sigma_y +
     * cos(ky) sigma_z. sigma_y = [[0,-i],[i,0]] so i*sigma_y = [[0,1],[-1,0]]
     * giving T_y = [[1/2, 1/2], [-1/2, -1/2]]. */
    const cm_complex_t Ty[2][2] = {
        {  0.5,  0.5 },
        { -0.5, -0.5 }
    };

    for (size_t y = 0; y < L; y++) {
        for (size_t x = 0; x < L; x++) {
            /* +x hop */
            if (x + 1 < L) {
                for (size_t s1 = 0; s1 < 2; s1++) {
                    for (size_t s0 = 0; s0 < 2; s0++) {
                        cm_complex_t v = Tx[s1][s0];
                        if (v != 0.0) {
                            add(sys->H, N,
                                idx(L, 2, x + 1, y, s1),
                                idx(L, 2, x,     y, s0), v);
                            /* Hermitian conjugate for x -> x+1 reverse. */
                            add(sys->H, N,
                                idx(L, 2, x,     y, s0),
                                idx(L, 2, x + 1, y, s1), conj(v));
                        }
                    }
                }
            }
            /* +y hop */
            if (y + 1 < L) {
                for (size_t s1 = 0; s1 < 2; s1++) {
                    for (size_t s0 = 0; s0 < 2; s0++) {
                        cm_complex_t v = Ty[s1][s0];
                        if (v != 0.0) {
                            add(sys->H, N,
                                idx(L, 2, x, y + 1, s1),
                                idx(L, 2, x, y,     s0), v);
                            add(sys->H, N,
                                idx(L, 2, x, y,     s0),
                                idx(L, 2, x, y + 1, s1), conj(v));
                        }
                    }
                }
            }
        }
    }
    return sys;
}

void chern_system_free(chern_system_t* sys) {
    if (!sys) return;
    free(sys->H);
    free(sys->P);
    free(sys);
}

/* Dense N x N matrix multiply: C = A * B (row-major). */
static void matmul(const cm_complex_t* A, const cm_complex_t* B,
                   cm_complex_t* C, size_t N) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            cm_complex_t s = 0;
            for (size_t k = 0; k < N; k++) s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
    }
}

int chern_build_projector(chern_system_t* sys) {
    if (!sys || !sys->H) return -1;
    size_t N = sys->dim;

    /* Projector onto the negative-energy band via the matrix sign
     * function: P = (I - sign(H)) / 2. Compute sign(Y) via Schulz
     * iteration Y_{k+1} = (1/2) Y_k (3 I - Y_k^2), seeded with
     * Y_0 = H / B where B > ||H||_2. This converges quadratically
     * provided spec(H) avoids 0 (the band gap guarantees this). */

    /* Spectral bound: sum of |m| + 2 bounds ||H||_2 for QWZ; use
     * Frobenius norm as a generic safe overestimate so this works
     * for other models too. Frobenius >= 2-norm always. */
    double fnorm2 = 0.0;
    for (size_t i = 0; i < N * N; i++) {
        double re = creal(sys->H[i]);
        double im = cimag(sys->H[i]);
        fnorm2 += re * re + im * im;
    }
    double B = sqrt(fnorm2) + 1e-3;

    cm_complex_t* Y  = malloc(N * N * sizeof(cm_complex_t));
    cm_complex_t* Y2 = malloc(N * N * sizeof(cm_complex_t));
    cm_complex_t* Yn = malloc(N * N * sizeof(cm_complex_t));
    if (!Y || !Y2 || !Yn) { free(Y); free(Y2); free(Yn); return -2; }

    for (size_t i = 0; i < N * N; i++) Y[i] = sys->H[i] / B;

    int max_iters = 200;
    double tol = 1e-12;
    int iter;
    for (iter = 0; iter < max_iters; iter++) {
        matmul(Y, Y, Y2, N);   /* Y2 = Y^2 */
        /* Residual: ||Y^2 - I||_F^2. When this drops below tol we're done. */
        double r2 = 0.0;
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                cm_complex_t d = Y2[i * N + j];
                if (i == j) d -= 1.0;
                double re = creal(d), im = cimag(d);
                r2 += re * re + im * im;
            }
        }
        if (r2 < tol) break;

        /* Yn = (3 I - Y2); Y <- (1/2) Y * Yn. Reuse Y2 as 3I - Y2. */
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                Y2[i * N + j] = -Y2[i * N + j];
            }
            Y2[i * N + i] += 3.0;
        }
        matmul(Y, Y2, Yn, N);
        for (size_t i = 0; i < N * N; i++) Y[i] = 0.5 * Yn[i];
    }

    if (iter >= max_iters) {
        free(Y); free(Y2); free(Yn);
        return -3; /* did not converge */
    }

    /* P = (I - sign(H)) / 2 */
    free(sys->P);
    sys->P = malloc(N * N * sizeof(cm_complex_t));
    if (!sys->P) { free(Y); free(Y2); free(Yn); return -4; }
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            sys->P[i * N + j] = -0.5 * Y[i * N + j];
        }
        sys->P[i * N + i] += 0.5;
    }

    free(Y); free(Y2); free(Yn);
    return 0;
}

/* Apply diagonal position operator (X or Y) to a column vector in place.
 * x_site for X, y_site for Y. The operator is diag(coordinate). */
static void apply_position(const chern_system_t* sys,
                           int use_y, cm_complex_t* v) {
    size_t L = sys->L;
    for (size_t y = 0; y < L; y++) {
        for (size_t x = 0; x < L; x++) {
            double coord = use_y ? (double)y : (double)x;
            for (size_t s = 0; s < 2; s++) {
                v[idx(L, 2, x, y, s)] *= coord;
            }
        }
    }
}

/* Dense matrix-vector multiply: out = M * in, M is N x N row-major. */
static void matvec(const cm_complex_t* M, const cm_complex_t* in,
                   cm_complex_t* out, size_t N) {
    for (size_t i = 0; i < N; i++) {
        cm_complex_t s = 0;
        for (size_t j = 0; j < N; j++) {
            s += M[i * N + j] * in[j];
        }
        out[i] = s;
    }
}

double chern_local_marker(const chern_system_t* sys, size_t rx, size_t ry) {
    if (!sys || !sys->P) return 0.0;
    size_t N = sys->dim;
    size_t L = sys->L;

    /* Scratch buffers. */
    cm_complex_t* v   = calloc(N, sizeof(cm_complex_t));
    cm_complex_t* tmp = calloc(N, sizeof(cm_complex_t));

    double total_im = 0.0;
    for (size_t s = 0; s < 2; s++) {
        /* Start with |r,s>. */
        memset(v, 0, N * sizeof(cm_complex_t));
        v[idx(L, 2, rx, ry, s)] = 1.0;

        /* v <- P |r,s>. */
        matvec(sys->P, v, tmp, N);
        /* apply Y */
        apply_position(sys, 1, tmp);
        /* Q = I - P, so Q tmp = tmp - P tmp. Compute P tmp into v, then
         * subtract from tmp. */
        matvec(sys->P, tmp, v, N);
        for (size_t k = 0; k < N; k++) tmp[k] -= v[k];
        /* apply X */
        apply_position(sys, 0, tmp);
        /* v <- P tmp */
        matvec(sys->P, tmp, v, N);

        /* Inner product <r,s| v >. We applied P X Q Y P from right to
         * left; that gives the Bianco-Resta matrix element. */
        cm_complex_t val = v[idx(L, 2, rx, ry, s)];
        total_im += cimag(val);
    }

    free(v); free(tmp);
    /* Bianco-Resta 2011 eq 12:
     *   C(r) = -(4 pi / A_c) Im Sum_s <r,s| P X Q Y P |r,s>
     * with unit cell area A_c = 1 on a square lattice. */
    return -4.0 * M_PI * total_im;
}

double chern_bulk_sum(const chern_system_t* sys,
                      size_t rmin, size_t rmax) {
    if (!sys || !sys->P) return 0.0;
    double total = 0.0;
    for (size_t y = rmin; y < rmax; y++) {
        for (size_t x = rmin; x < rmax; x++) {
            total += chern_local_marker(sys, x, y);
        }
    }
    return total;
}
