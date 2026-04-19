/**
 * @file qgt.h
 * @brief Quantum geometric tensor, Berry curvature, and topological invariants.
 *
 * OVERVIEW
 * --------
 * For a parameterised family of pure quantum states @f$|u(\lambda)\rangle@f$
 * (typically Bloch eigenstates @f$|u_n(\mathbf k)\rangle@f$ of a periodic
 * Hamiltonian) the *quantum geometric tensor* introduced by Provost and
 * Vallee (1980) is
 * @f[
 *   Q_{\mu\nu}(\lambda) \;=\;
 *   \langle\partial_\mu u|\,(\mathbb{1} - |u\rangle\langle u|)\,|\partial_\nu u\rangle,
 * @f]
 * a @f$U(1)@f$-gauge invariant rank-two tensor on parameter space.  Its
 * symmetric part is the *Fubini-Study metric*
 * @f$g_{\mu\nu} = \operatorname{Re}\,Q_{\mu\nu}@f$, which measures how
 * quickly the state moves through projective Hilbert space as
 * @f$\lambda@f$ changes, and its antisymmetric part is half the *Berry
 * curvature*,
 * @f$\Omega_{\mu\nu} = -2\,\operatorname{Im}\,Q_{\mu\nu} =
 * \partial_\mu A_\nu - \partial_\nu A_\mu@f$
 * with Berry connection
 * @f$A_\mu = i\,\langle u|\partial_\mu u\rangle@f$ (Berry 1984).  For a
 * two-dimensional periodic system with gapped spectrum, the Chern
 * number of band @f$n@f$ is
 * @f[
 *   C_n \;=\; \frac{1}{2\pi}\,
 *       \int_{\mathrm{BZ}} \Omega_{xy}(\mathbf k)\,d^2 k \;\in\; \mathbb{Z}.
 * @f]
 *
 * DISCRETISED IMPLEMENTATION (Fukui-Hatsugai-Suzuki)
 * --------------------------------------------------
 * A naive numerical discretisation of @f$\Omega_{xy}@f$ picks up
 * arbitrary @f$2\pi@f$ branch jumps from the gauge ambiguity in the
 * numerically computed @f$|u_n(\mathbf k)\rangle@f$.  Fukui, Hatsugai
 * and Suzuki (2005) solve this by assembling the plaquette field
 * strength from *link variables*
 * @f[
 *   U_\mu(\mathbf k) \;=\;
 *   \frac{\langle u(\mathbf k)|u(\mathbf k + \delta_\mu)\rangle}
 *        {|\langle u(\mathbf k)|u(\mathbf k + \delta_\mu)\rangle|},
 * @f]
 * and writing the plaquette flux as the principal argument of the
 * product around the plaquette,
 * @f[
 *   F_{xy}(\mathbf k) \;=\; \operatorname{arg}\!\left[
 *   U_x(\mathbf k)\,U_y(\mathbf k + \delta_x)\,
 *   U_x(\mathbf k + \delta_y)^{-1}\,U_y(\mathbf k)^{-1}
 *   \right]_{(-\pi, \pi]},
 * @f]
 * yielding
 * @f$C = \tfrac{1}{2\pi}\sum_{\mathbf k} F_{xy}(\mathbf k) \in \mathbb{Z}@f$
 * *exactly* at any finite grid size (no branch ambiguity).  This is
 * what @c qgt_berry_grid implements.
 *
 * ONE-DIMENSIONAL INVARIANTS
 * --------------------------
 * For 1D chiral-symmetric two-band systems (the AIII class containing
 * the Su-Schrieffer-Heeger model), the topological invariant is the
 * Zak phase / winding number
 * @f[
 *   W \;=\; -\frac{1}{\pi}\oint_{\mathrm{BZ}} A(k)\,dk \;\in\; \mathbb{Z},
 * @f]
 * evaluated here by the 1D analogue of the FHS construction (sum of
 * discrete link-variable phases around the Brillouin zone).  The sign
 * convention matches the canonical @f$W = +1@f$ for the topological
 * SSH phase @f$|t_2| > |t_1|@f$.
 *
 * GAUGE-STABLE EIGENVECTOR SELECTION
 * ----------------------------------
 * Any formula @f$|u_-(\mathbf k)\rangle \propto f(\hat h(\mathbf k))@f$
 * for the lower-band eigenvector of the traceless 2x2 Hamiltonian
 * @f$\hat h = h_x\sigma_x + h_y\sigma_y + h_z\sigma_z@f$ has a gauge
 * singularity somewhere on the sphere @f$|\hat h| = \mathrm{const}@f$.
 * A branch-switch between two formulae introduces a @f$\pi@f$ phase
 * jump that is indistinguishable from a real plaquette flux in the
 * link-variable product.  Internally we use two complementary closed
 * forms and pick the one whose pre-normalisation 2-norm is larger at
 * the current @f$(\mathbf k, m)@f$, which guarantees we are never
 * within @f$\sqrt{\epsilon}@f$ of either formula's zero.  The FHS
 * quantisation then remains exact at all finite grid sizes.
 *
 * BUILT-IN MODELS
 * ---------------
 *  - @em QWZ (Qi-Wu-Zhang 2006): 2-band minimal Chern insulator used
 *    throughout the topology test battery.
 *  - @em Haldane (1988): honeycomb-lattice Chern insulator with
 *    Peierls phase @f$\phi@f$ on the next-nearest-neighbour hopping
 *    @f$t_2@f$ and an on-site staggering @f$M@f$; topological for
 *    @f$|M| < 3\sqrt{3}\,|t_2 \sin\phi|@f$.
 *  - @em SSH (Su-Schrieffer-Heeger 1979): 1D chiral two-band model
 *    used for the winding-number tests.
 *
 * ROLE IN THE LIBRARY
 * -------------------
 * @c qgt.h supplies momentum-space topological data consumed by
 * Moonlab's sibling libraries (QGTL, lilirrep, SbNN) via the stable
 * ABI (@c moonlab_qwz_chern) or the opaque-handle interface exported
 * here.  The accompanying @c chern_marker.h / @c chern_kpm.h modules
 * supply the real-space counterpart, which is valid in aperiodic or
 * disordered geometries where Bloch states do not exist.
 *
 * REFERENCES
 * ----------
 *  - J. P. Provost and G. Vallee, "Riemannian structure on manifolds
 *    of quantum states", Commun. Math. Phys. 76, 289 (1980),
 *    doi:10.1007/BF02193559.  Origin of the QGT.
 *  - M. V. Berry, "Quantal phase factors accompanying adiabatic
 *    changes", Proc. R. Soc. Lond. A 392, 45 (1984),
 *    doi:10.1098/rspa.1984.0023.  Berry connection and phase.
 *  - T. Fukui, Y. Hatsugai and H. Suzuki, "Chern numbers in
 *    discretized Brillouin zone", J. Phys. Soc. Jpn. 74, 1674 (2005),
 *    arXiv:cond-mat/0503172.  Link-variable quantisation we implement.
 *  - F. D. M. Haldane, "Model for a quantum Hall effect without
 *    Landau levels", Phys. Rev. Lett. 61, 2015 (1988),
 *    doi:10.1103/PhysRevLett.61.2015.
 *  - X.-L. Qi, Y.-S. Wu and S.-C. Zhang, "Topological quantization of
 *    the spin Hall effect", Phys. Rev. B 74, 085308 (2006),
 *    arXiv:cond-mat/0505308.  Source of the QWZ model.
 *  - W. P. Su, J. R. Schrieffer and A. J. Heeger, "Solitons in
 *    polyacetylene", Phys. Rev. Lett. 42, 1698 (1979),
 *    doi:10.1103/PhysRevLett.42.1698.
 *  - J. Zak, "Berry's phase for energy bands in solids",
 *    Phys. Rev. Lett. 62, 2747 (1989),
 *    doi:10.1103/PhysRevLett.62.2747.  Zak phase / 1D winding.
 *
 * @since  v0.2.0
 * @stability evolving
 */

#ifndef MOONLAB_QGT_H
#define MOONLAB_QGT_H

#include <stddef.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef double _Complex qgt_complex_t;

/**
 * @brief Callback supplying the 2x2 Bloch Hamiltonian at 2D momentum
 *        @p k.  Row-major layout: `h[0]=H_{00}`, `h[1]=H_{01}`,
 *        `h[2]=H_{10}`, `h[3]=H_{11}`.  Must be Hermitian; parameters
 *        come through the caller-owned @p user pointer.
 */
typedef void (*qgt_bloch_fn)(const double k[2], void* user,
                             qgt_complex_t h[4]);

typedef struct qgt_system qgt_system_t;

/**
 * @brief Construct a QGT system around a user-supplied Bloch callback.
 *
 * The callback is invoked once per plaquette corner in
 * @c qgt_berry_grid and a handful of times per query in
 * @c qgt_metric_at, so cheap evaluation is rewarded.  The system
 * stores @p user as a borrowed pointer; the caller retains ownership.
 */
qgt_system_t* qgt_create(qgt_bloch_fn f, void* user);

/**
 * @brief Qi-Wu-Zhang two-band Chern insulator
 *        @f$H(\mathbf k) = \sin k_x\,\sigma_x + \sin k_y\,\sigma_y +
 *        (m + \cos k_x + \cos k_y)\,\sigma_z@f$.
 *
 * Lower-band Chern:
 *   - @f$C = +1@f$ for @f$-2 < m < 0@f$,
 *   - @f$C = -1@f$ for @f$0 < m < +2@f$,
 *   - @f$C = 0@f$ for @f$|m| > 2@f$.
 */
qgt_system_t* qgt_model_qwz(double m);

/**
 * @brief Haldane (1988) honeycomb-lattice Chern insulator.
 *
 * Parameters: nearest-neighbour hopping @f$t_1@f$, next-nearest
 * @f$t_2@f$ with Peierls phase @f$\phi@f$, sublattice staggering
 * @f$M@f$.  Topological for @f$|M| < 3\sqrt{3}\,|t_2\sin\phi|@f$.
 */
qgt_system_t* qgt_model_haldane(double t1, double t2,
                                double phi, double M_stagger);

/**
 * @brief 1D Bloch callback, analogous to ::qgt_bloch_fn.
 *        Row-major 2x2 Hamiltonian at scalar momentum @p k.
 */
typedef void (*qgt_bloch_1d_fn)(double k, void* user, qgt_complex_t h[4]);

typedef struct qgt_system_1d qgt_system_1d_t;

qgt_system_1d_t* qgt_create_1d(qgt_bloch_1d_fn f, void* user);
void             qgt_free_1d(qgt_system_1d_t* sys);

/**
 * @brief Su-Schrieffer-Heeger two-band model
 *        @f$H(k) = (t_1 + t_2\cos k)\,\sigma_x + (t_2\sin k)\,\sigma_y@f$.
 *
 * Integer winding number @f$W = +1@f$ when @f$|t_2| > |t_1|@f$
 * (topological phase with zero-energy edge modes in open boundaries);
 * @f$W = 0@f$ otherwise.
 */
qgt_system_1d_t* qgt_model_ssh(double t1, double t2);

/**
 * @brief Winding number of a 1D chiral two-band system via the
 *        discrete Zak-phase formula
 *        @f$W = -(2\pi)^{-1}\sum_k \operatorname{arg}
 *        \langle u(k)|u(k+\delta k)\rangle@f$ rescaled to match the
 *        SSH convention (@f$W = +1@f$ in the topological phase).
 *
 * @param N        grid size in @f$k@f$; N >= 8 is sufficient.
 * @param out_raw  optional output: raw pre-rounding winding @f$W \in \mathbb R@f$.
 * @return integer winding.
 */
int qgt_winding_1d(const qgt_system_1d_t* sys, size_t N,
                   double* out_raw);

/**
 * @brief Lower-band Wilson loop along a caller-supplied closed path in
 *        2D momentum space.
 *
 * Returns the total @f$U(1)@f$ phase @f$\gamma \in (-\pi, \pi]@f$ as
 * the principal argument of
 * @f$\prod_{i=0}^{N-1} \langle u(\mathbf k_i) | u(\mathbf k_{i+1}) \rangle@f$
 * with @f$\mathbf k_N \equiv \mathbf k_0@f$.  Non-Abelian generalisations
 * (multi-band holonomies) will be added alongside multi-band models.
 */
int qgt_wilson_loop(const qgt_system_t* sys,
                    const double* path_k,
                    size_t num_points,
                    double* out_phase);

/**
 * @brief Release the QGT handle.
 */
void qgt_free(qgt_system_t* sys);

/**
 * @brief Output container for @c qgt_berry_grid.
 *
 * @c berry is a row-major @f$N \times N@f$ array of plaquette fluxes
 * @f$F_{xy}(\mathbf k)@f$ in the convention of Fukui-Hatsugai-Suzuki;
 * @c chern equals @f$(2\pi)^{-1}\sum_{\mathbf k} F_{xy}(\mathbf k)@f$.
 * Release the array via @c qgt_berry_grid_free.
 */
typedef struct {
    size_t  N;
    double* berry;
    double  chern;
} qgt_berry_grid_t;

/**
 * @brief Evaluate the Berry-curvature plaquette field and its
 *        integrated Chern number on an @f$N \times N@f$ BZ grid.
 *
 * The result is exactly an integer Chern number at any finite @f$N@f$
 * provided the band remains gapped throughout the grid (Fukui,
 * Hatsugai and Suzuki 2005).
 */
int qgt_berry_grid(const qgt_system_t* sys, size_t N,
                   qgt_berry_grid_t* out);

void qgt_berry_grid_free(qgt_berry_grid_t* g);

/**
 * @brief Fubini-Study metric @f$g_{\mu\nu}(\mathbf k) =
 *        \operatorname{Re}\,Q_{\mu\nu}(\mathbf k)@f$ at a single
 *        momentum via centered finite differences with step @p dk.
 *        Output @p g is row-major 2x2 (real, symmetric).
 *
 * Neither @p dk too large (truncation error) nor too small (roundoff)
 * is optimal; a reasonable default is @p dk in @f$[10^{-4}, 10^{-3}]@f$
 * for the built-in QWZ / Haldane models.
 */
int qgt_metric_at(const qgt_system_t* sys, const double k[2],
                  double dk, double g[4]);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_QGT_H */
