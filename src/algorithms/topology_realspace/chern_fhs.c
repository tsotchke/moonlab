/**
 * @file chern_fhs.c
 * @brief Fukui-Hatsugai-Suzuki momentum-space Chern integrator.
 *
 * Implementation notes
 * --------------------
 * For a 2x2 Bloch matrix H(k), the lower-band eigenvector is
 *
 *     |u(k)> = ( h_z - sqrt(h_x^2 + h_y^2 + h_z^2) ; h_x + i h_y ),
 *
 * normalised to unit length, where H = h.sigma + h_0 I (we drop h_0
 * from the eigenvector since it is proportional to identity).  This
 * closed form avoids a numerical eigendecomposition and produces a
 * smooth gauge in the gapped regions of the BZ.  At gap-closing points
 * the eigenvector is undefined; the FHS prescription handles this by
 * passing through the discretization (the link-variable phase is still
 * well-defined as long as <u_k | u_{k+dk}> is non-zero, which it is on
 * a non-singular plaquette).
 *
 * The link variable U_mu(k) = <u(k) | u(k + dk_mu)> / |...| is then
 * computed; the plaquette field strength F = -i log(U_x U_y U_x'^* U_y'^*)
 * with principal branch summed over the BZ gives 2 pi C.
 */

#include "chern_fhs.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* QWZ Bloch Hamiltonian closure data. */
typedef struct {
    double m;
} qwz_user_t;

static void qwz_bloch(double kx, double ky,
                       chern_fhs_complex_t out[4], void* user) {
    qwz_user_t* u = (qwz_user_t*)user;
    double sx = sin(kx);
    double sy = sin(ky);
    double sz = u->m + cos(kx) + cos(ky);
    /* H = sx sigma_x + sy sigma_y + sz sigma_z
     *   = [[ sz,        sx - i sy],
     *      [ sx + i sy, -sz      ]]                                    */
    out[0] = sz;
    out[1] = sx - I * sy;
    out[2] = sx + I * sy;
    out[3] = -sz;
}

/* Lower-band eigenvector of a 2x2 Hermitian h.sigma matrix at eigenvalue
 * -h, written in two patches so the link variables stay non-zero across
 * the BZ.
 *
 * Patch A (used when h_z >= 0; safe at the "north pole" h_z = +h):
 *   |u_-> propto (-(h_x - i h_y),  h + h_z)
 * Patch B (used when h_z <  0; safe at the "south pole" h_z = -h):
 *   |u_-> propto ( h - h_z,       -(h_x + i h_y))
 *
 * The two patches differ by a U(1) gauge transformation, exactly what
 * the FHS link-variable construction is designed to capture.  At
 * h_x = h_y = 0 and h_z > 0, patch A returns (0, 2 h_z) which normalises
 * to (0, 1); patch B at h_x = h_y = 0 and h_z < 0 returns (-2 h_z, 0)
 * normalising to (1, 0).  These choices keep the plaquette link
 * variables non-zero on a non-singular BZ mesh.  At a true gap closing
 * (h = 0) the eigenvector is undefined; we return (1, 0) to keep the
 * arithmetic finite (the surrounding plaquettes carry the topology
 * and the singular plaquette contributes 2 pi times an integer
 * which is what we sum). */
static void lower_band_eigenvec(const chern_fhs_complex_t H[4],
                                 chern_fhs_complex_t v[2]) {
    /* H = h_z sigma_z + h_x sigma_x + h_y sigma_y for traceless H.
     * Read off h_x = Re(H[1]), h_y = -Im(H[1]) (since H[1] = h_x - i h_y),
     * h_z = (H[0] - H[3]) / 2. */
    double hz = creal(H[0] - H[3]) * 0.5;
    double hx = creal(H[1]);
    double hy = -cimag(H[1]);
    double h  = sqrt(hx * hx + hy * hy + hz * hz);
    if (h < 1e-15) { v[0] = 1.0; v[1] = 0.0; return; }
    chern_fhs_complex_t a, b;
    if (hz >= 0.0) {
        /* Patch A: smooth at the north pole. */
        a = -((chern_fhs_complex_t)hx - I * (chern_fhs_complex_t)hy);
        b =  (chern_fhs_complex_t)(h + hz);
    } else {
        /* Patch B: smooth at the south pole. */
        a =  (chern_fhs_complex_t)(h - hz);
        b = -((chern_fhs_complex_t)hx + I * (chern_fhs_complex_t)hy);
    }
    double norm = sqrt(creal(a) * creal(a) + cimag(a) * cimag(a)
                        + creal(b) * creal(b) + cimag(b) * cimag(b));
    if (norm < 1e-15) { v[0] = 1.0; v[1] = 0.0; return; }
    v[0] = a / norm;
    v[1] = b / norm;
}

/* Inner product <u | v> for 2-vectors. */
static chern_fhs_complex_t inner2(const chern_fhs_complex_t* u,
                                   const chern_fhs_complex_t* v) {
    return conj(u[0]) * v[0] + conj(u[1]) * v[1];
}

int chern_fhs_two_band(size_t N,
                        chern_fhs_bloch_t bloch, void* user,
                        int* out_chern,
                        double* out_chern_real) {
    if (N < 6 || !bloch || !out_chern) return -1;

    /* Precompute lower-band eigenvectors at every momentum. */
    chern_fhs_complex_t* U =
        (chern_fhs_complex_t*)malloc(sizeof(chern_fhs_complex_t) * N * N * 2);
    if (!U) return -1;

    const double dk = 2.0 * M_PI / (double)N;
    for (size_t ix = 0; ix < N; ix++) {
        double kx = -M_PI + (double)ix * dk;
        for (size_t iy = 0; iy < N; iy++) {
            double ky = -M_PI + (double)iy * dk;
            chern_fhs_complex_t H[4];
            bloch(kx, ky, H, user);
            chern_fhs_complex_t* v = &U[(ix * N + iy) * 2];
            lower_band_eigenvec(H, v);
        }
    }

    /* Plaquette sum.  ix, iy index lower-left corners; periodic wrap. */
    double total = 0.0;
    for (size_t ix = 0; ix < N; ix++) {
        size_t ixp = (ix + 1) % N;
        for (size_t iy = 0; iy < N; iy++) {
            size_t iyp = (iy + 1) % N;
            const chern_fhs_complex_t* u00 = &U[(ix  * N + iy ) * 2];
            const chern_fhs_complex_t* u10 = &U[(ixp * N + iy ) * 2];
            const chern_fhs_complex_t* u01 = &U[(ix  * N + iyp) * 2];
            const chern_fhs_complex_t* u11 = &U[(ixp * N + iyp) * 2];
            chern_fhs_complex_t Ux = inner2(u00, u10);
            chern_fhs_complex_t Uy = inner2(u00, u01);
            chern_fhs_complex_t Ux_top = inner2(u01, u11);
            chern_fhs_complex_t Uy_right = inner2(u10, u11);
            /* Normalise to unimodular link variables. */
            double aUx = cabs(Ux); if (aUx > 0) Ux /= aUx;
            double aUy = cabs(Uy); if (aUy > 0) Uy /= aUy;
            double aXt = cabs(Ux_top);   if (aXt > 0) Ux_top /= aXt;
            double aYr = cabs(Uy_right); if (aYr > 0) Uy_right /= aYr;
            /* Field strength on principal branch:
             * F = -i log(U_x(k) * U_y(k+dx) * U_x(k+dy)^* * U_y(k)^*) */
            chern_fhs_complex_t loop = Ux * Uy_right * conj(Ux_top) * conj(Uy);
            double phase = atan2(cimag(loop), creal(loop));
            total += phase;
        }
    }
    free(U);

    double C_real = total / (2.0 * M_PI);
    int C_int = (int)lrint(C_real);
    *out_chern = C_int;
    if (out_chern_real) *out_chern_real = C_real;
    return 0;
}

int chern_fhs_qwz(size_t N, double m,
                   int* out_chern, double* out_chern_real) {
    qwz_user_t u = { .m = m };
    return chern_fhs_two_band(N, qwz_bloch, &u, out_chern, out_chern_real);
}
