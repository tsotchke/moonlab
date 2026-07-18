/**
 * @file chern_fhs.h
 * @brief Fukui-Hatsugai-Suzuki momentum-space Chern integrator.
 *
 * Computes the Chern integer of a two-band Bloch model on a discretized
 * Brillouin zone via link-variable quantization, following Fukui,
 * Hatsugai & Suzuki, J. Phys. Soc. Jpn. 74, 1674 (2005).  This is the
 * momentum-space companion to the real-space Bianco-Resta marker in
 * @c chern_marker.h and the matrix-free KPM marker in @c chern_kpm.h;
 * the three paths give the same Chern integer up to lattice-discretisation
 * O(1/N^2) and finite-size corrections.
 *
 * METHOD
 * ------
 * The link variable on a plaquette of the discretized BZ is
 *
 *     U_mu(k) = <u(k) | u(k + dk_mu)> / |<u(k) | u(k + dk_mu)>|
 *
 * (a unimodular complex number), and the lattice field strength on the
 * plaquette is
 *
 *     F(k) = -i log(U_x(k) U_y(k+dk_x) U_x(k+dk_y)^* U_y(k)^*)
 *
 * with the principal branch of log.  The Chern number of the lower band
 * is
 *
 *     C = (1 / 2 pi) sum_{k in BZ} F(k).
 *
 * For our two-band Bloch Hamiltonian H(k) we take |u(k)> as the
 * normalised eigenvector of the lower eigenvalue at every momentum.
 *
 * MODELS
 * ------
 * Convenience entry-point @c chern_fhs_qwz computes the lower-band Chern
 * number of the Qi-Wu-Zhang model
 *
 *     H(k) = sin(k_x) sigma_x + sin(k_y) sigma_y
 *          + (m + cos(k_x) + cos(k_y)) sigma_z
 *
 * which has C = +1 for -2 < m < 0, -1 for 0 < m < 2, and 0 for |m| > 2.
 * General two-band Bloch matrices can be passed via @c chern_fhs_two_band.
 *
 * @since v0.2.1
 */

#ifndef MOONLAB_CHERN_FHS_H
#define MOONLAB_CHERN_FHS_H
#include "applications/moonlab_api.h"

#include <stddef.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef double _Complex chern_fhs_complex_t;

/**
 * @brief Bloch-Hamiltonian callback.  Fills @p out with the 2x2 matrix
 *        H(kx, ky) in row-major order.  @p user is forwarded from the
 *        caller of @c chern_fhs_two_band.
 */
typedef void (*chern_fhs_bloch_t)(double kx, double ky,
                                    chern_fhs_complex_t out[4],
                                    void* user);

/**
 * @brief FHS Chern integer for an arbitrary two-band Bloch Hamiltonian
 *        on an N x N momentum-space mesh.
 *
 * @param N      momenta per axis (>= 6); BZ wraps at k = +- pi.
 * @param bloch  callback that fills the 2x2 Bloch matrix.
 * @param user   opaque pointer forwarded to @p bloch.
 * @param[out] out_chern  the integer Chern number (rounded to nearest int).
 * @param[out] out_chern_real (optional, may be NULL)  the unrounded
 *             plaquette sum / (2 pi); a clean computation rounds to an
 *             integer to <= 1e-3 at N >= 64 in a gapped phase.
 *
 * @return 0 on success; -1 on OOM or invalid arguments.
 */
int chern_fhs_two_band(size_t N,
                        chern_fhs_bloch_t bloch, void* user,
                        int* out_chern,
                        double* out_chern_real);

/**
 * @brief FHS Chern integer for the Qi-Wu-Zhang model.
 *
 * Convenience wrapper that constructs the QWZ Bloch Hamiltonian and
 * forwards to @c chern_fhs_two_band.
 *
 * @param N       momenta per axis.
 * @param m       QWZ mass parameter.
 * @param[out] out_chern  rounded integer Chern number of the lower band.
 * @param[out] out_chern_real  optional unrounded value.
 * @return 0 on success.
 */
MOONLAB_API int chern_fhs_qwz(size_t N, double m,
                   int* out_chern, double* out_chern_real);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_CHERN_FHS_H */
