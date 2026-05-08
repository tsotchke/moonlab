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
 *   W \;=\; -\frac{\gamma_{\mathrm{Zak}}}{\pi} \;\in\; \mathbb{Z},
 *   \qquad
 *   \gamma_{\mathrm{Zak}} \;=\; i\oint_{\mathrm{BZ}} \langle u_-(k)|\partial_k u_-(k)\rangle\,dk,
 * @f]
 * evaluated here by the 1D analogue of the FHS construction (sum of
 * discrete link-variable phases around the Brillouin zone).  The
 * overall sign matches the canonical convention of Asboth-Oroszlany-
 * Palyi (2016) §1.5: @f$W = +1@f$ in the topological SSH phase
 * @f$|t_2| > |t_1|@f$.
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

#include "../../applications/moonlab_api.h"

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
MOONLAB_API qgt_system_t* qgt_model_qwz(double m);

/**
 * @brief Haldane (1988) honeycomb-lattice Chern insulator.
 *
 * Parameters: nearest-neighbour hopping @f$t_1@f$, next-nearest
 * @f$t_2@f$ with Peierls phase @f$\phi@f$, sublattice staggering
 * @f$M@f$.  Topological for @f$|M| < 3\sqrt{3}\,|t_2\sin\phi|@f$.
 */
MOONLAB_API qgt_system_t* qgt_model_haldane(double t1, double t2,
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
MOONLAB_API qgt_system_1d_t* qgt_model_ssh(double t1, double t2);

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
MOONLAB_API void qgt_free(qgt_system_t* sys);

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
MOONLAB_API int qgt_berry_grid(const qgt_system_t* sys, size_t N,
                   qgt_berry_grid_t* out);

/**
 * @brief Berry-grid integrator with parallel-transport gauge.
 *
 * Same FHS plaquette construction as ::qgt_berry_grid but with an
 * extra gauge-fix step: each eigenvector u(k) is phase-rotated so
 * that @f$\langle u(k_{\rm prev}) | u(k)\rangle > 0@f$ along a
 * spanning walk of the BZ (kx-axis first, then ky from each ix).
 * Removes the LAPACK / closed-form gauge mismatches that affect
 * eigvec-based methods.  Costs an extra O(N^2) per-grid phase-fix
 * on top of the standard FHS work.  Returns the same physically-
 * correct Chern number as ::qgt_berry_grid_proj.
 */
MOONLAB_API int qgt_berry_grid_pt(const qgt_system_t* sys, size_t N,
                                   qgt_berry_grid_t* out);

/**
 * @brief Berry-grid integrator using the projector-trace formula
 *        (rigorously gauge-free).
 *
 * Discrete projector formulation: the lower-band projector
 *   @f$P_-(\mathbf k) = \tfrac12(\mathbb 1 - \hat h \cdot
 *      \boldsymbol\sigma)@f$
 * is gauge-invariant by construction, so the plaquette holonomy
 *   @f$F_{xy}(\mathbf k) = -\arg
 *       \mathrm{Tr}[P_-(\mathbf k)\,P_-(\mathbf k + \mathbf{dx})
 *                  \,P_-(\mathbf k + \mathbf{dx} + \mathbf{dy})
 *                  \,P_-(\mathbf k + \mathbf{dy})]@f$
 * is gauge-free without any phase-fix scaffolding.  The integrated
 * total over the BZ gives @f$2\pi C@f$ at small grid spacing.
 *
 * Use this when you want the most directly-trustworthy Chern
 * integrator: it has no eigvec-gauge sensitivity by construction
 * and produces results identical to a correctly-implemented
 * ::qgt_berry_grid (eigvec FHS) path on the same Hamiltonian.
 */
MOONLAB_API int qgt_berry_grid_proj(const qgt_system_t* sys, size_t N,
                                     qgt_berry_grid_t* out);

MOONLAB_API void qgt_berry_grid_free(qgt_berry_grid_t* g);

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

/* ====================================================================
 * Topological phase diagrams
 *
 * Sweep a 2-band Bloch model that depends on a single tunable
 * parameter (e.g. QWZ mass m, Haldane phase phi) and classify each
 * parameter value by its integer Chern number.  Output is a length-K
 * array of integers covering K equally-spaced parameter samples in
 * [param_min, param_max] inclusive.  Phase transitions show up as
 * discrete jumps between integers.
 *
 * Cost: K * O(N^2) Bloch-Hamiltonian evaluations + K * O(N^2 log N)
 * for the FHS Chern accumulation; K = 64 with N = 32 finishes in
 * sub-second time on the QWZ model.
 * ==================================================================== */

/**
 * @brief Build a parameterised QGT system from a single double knob.
 *
 * The factory @p factory takes the user pointer and the current
 * parameter value, and must return a freshly allocated `qgt_system_t*`
 * (or NULL on failure).  ::qgt_phase_diagram_chern frees each system
 * after measuring it.
 */
typedef qgt_system_t* (*qgt_param_system_fn)(void* user, double param);

/**
 * @brief Compute integer Chern numbers along a 1D parameter sweep.
 *
 * @param factory    Builder for `qgt_system_t` parameterised by a double.
 * @param user       Opaque user data forwarded to @p factory.
 * @param param_min  Lower edge of the sweep (inclusive).
 * @param param_max  Upper edge of the sweep (inclusive).
 * @param K          Number of parameter samples (>=2).  K-1 spaces the grid.
 * @param N          BZ grid side for the FHS Chern integral (>=4; 32 ample).
 * @param[out] chern_out  Caller-allocated array of length K; populated
 *                        with integer Chern numbers.
 *
 * @return 0 on success, negative on error.
 */
int qgt_phase_diagram_chern(qgt_param_system_fn factory,
                             void* user,
                             double param_min, double param_max,
                             size_t K, size_t N,
                             int* chern_out);

/**
 * @brief Two-parameter Bloch-system factory.
 *
 * Used by ::qgt_phase_diagram_chern_2d to build a `qgt_system_t*` for
 * each (param_x, param_y) sample.  Caller of the diagram returns
 * NULL on failure; ::qgt_phase_diagram_chern_2d writes `INT_MIN`
 * into the corresponding output cell and continues.
 */
typedef qgt_system_t* (*qgt_param_system_2d_fn)(void* user,
                                                  double param_x,
                                                  double param_y);

/**
 * @brief Compute integer Chern numbers on a 2D parameter grid.
 *
 * Mirrors ::qgt_phase_diagram_chern with a second parameter axis;
 * useful for the canonical Haldane (t2, phi) diagram and similar
 * two-parameter topological-phase landscapes.
 *
 * @param factory     Builder for `qgt_system_t` parameterised by two doubles.
 * @param user        Opaque user data forwarded to @p factory.
 * @param x_min,x_max Inclusive sweep range on the first parameter.
 * @param y_min,y_max Inclusive sweep range on the second parameter.
 * @param Kx          Number of samples on the first axis (>=2).
 * @param Ky          Number of samples on the second axis (>=2).
 * @param N           BZ grid side for the FHS Chern integral (>=4).
 * @param[out] chern_out  Caller-allocated row-major array of length
 *                        Kx*Ky.  `chern_out[ix * Ky + iy]` holds the
 *                        Chern number for sample (ix, iy).
 *
 * @return 0 on success, negative on error.
 */
int qgt_phase_diagram_chern_2d(qgt_param_system_2d_fn factory,
                                void* user,
                                double x_min, double x_max,
                                double y_min, double y_max,
                                size_t Kx, size_t Ky, size_t N,
                                int* chern_out);

/* ====================================================================
 * Multi-band (n-band) Bloch systems
 *
 * The two-band primitives above are tuned for the spin-1/2 / sublattice
 * minimal models (QWZ, Haldane, SSH).  For richer Hamiltonians --
 * Kane-Mele (4 bands), BHZ (4 bands), Bernevig-Hughes-Zhang variants,
 * multi-orbital tight-binding -- the same Fukui-Hatsugai-Suzuki link-
 * variable construction generalises to a non-Abelian U(M) connection
 * over the M-dimensional occupied subspace.
 *
 * The n-band API takes a Bloch callback that emits an n*n Hermitian
 * matrix at each k, plus a fixed number of occupied bands M.  The
 * implementation diagonalises at each plaquette corner, builds the
 * MxM link variables U_mu(k) = det <u_occ(k) | u_occ(k + dk_mu)>,
 * and accumulates the FHS Chern from the principal-arg of the
 * plaquette holonomy.
 * ==================================================================== */

/**
 * @brief Bloch callback emitting an n*n Hermitian Hamiltonian at 2D
 *        momentum @p k.  @p h is row-major: h[i*n + j] = H_{ij}.
 *        Must be Hermitian; the caller's choice of @p n is fixed at
 *        ::qgt_create_nband construction time.
 */
typedef void (*qgt_bloch_n_fn)(const double k[2], void* user,
                               qgt_complex_t* h);

typedef struct qgt_system_n qgt_system_n_t;

/**
 * @brief Construct an n-band Bloch system.
 *
 * @param f             Callback emitting the Hamiltonian.
 * @param user          Opaque user pointer forwarded to @p f.
 * @param n_bands       Total number of bands (matrix dimension).  Must be >= 2.
 * @param n_occupied    Number of occupied bands counted from the bottom
 *                      (lowest energy).  Must satisfy 1 <= n_occupied <= n_bands - 1.
 *                      Pass @p n_bands / 2 for the conventional half-filling.
 *
 * @return Newly-owned handle, or NULL on bad arguments.
 */
MOONLAB_API qgt_system_n_t* qgt_create_nband(qgt_bloch_n_fn f, void* user,
                                              size_t n_bands,
                                              size_t n_occupied);

MOONLAB_API void qgt_free_nband(qgt_system_n_t* sys);

/**
 * @brief Compute the (Abelian, occupied-subspace summed) Chern number
 *        on an N x N BZ grid via the non-Abelian FHS prescription.
 *
 * The construction of T. Fukui, Y. Hatsugai and H. Suzuki (2005)
 * generalises directly to a multi-band occupied subspace by replacing
 * the Abelian U(1) link variable
 *   U_mu(k) = <u | u(k + dk_mu)> / |...|
 * with the SU(M) link
 *   U_mu(k) = det( <u_occ(k) | u_occ(k + dk_mu)> )
 * (the determinant of the M x M overlap matrix), and accumulating
 *   F_xy(k) = arg[ U_x(k) U_y(k+dk_x) U_x(k+dk_y)^{-1} U_y(k)^{-1} ]
 * around each plaquette.  The total Chern number is the sum of Chern
 * numbers of the occupied bands, exact at finite N when the gap
 * between band M and band M+1 is finite throughout the BZ.
 *
 * @param sys      Multi-band system handle.
 * @param N        Grid side; >= 8 for typical models.
 * @param[out] out Result container (caller-supplied).  Holds the
 *                 plaquette field and the integrated total Chern.
 * @return 0 on success, negative on error.
 */
MOONLAB_API int qgt_berry_grid_nband(const qgt_system_n_t* sys, size_t N,
                                      qgt_berry_grid_t* out);

/**
 * @brief Compute the Z_2 invariant (Kane-Mele 2005) of a 4-band
 *        time-reversal-symmetric Bloch Hamiltonian via the n-field
 *        method of Fukui & Hatsugai (2007).
 *
 * The Z_2 invariant nu in {0, 1} distinguishes the trivial insulator
 * (nu = 0) from the quantum spin Hall (QSH) insulator (nu = 1) of
 * 2D time-reversal-symmetric class-AII band Hamiltonians.  We require
 * @p sys to have @c n_bands == 4 with @c n_occupied == 2 (half-
 * filled), and that the Hamiltonian commutes with the time-reversal
 * operator (the caller's responsibility -- this routine does not
 * verify the symmetry, only computes the invariant assuming it
 * holds).
 *
 * Algorithm: Fukui-Hatsugai 2007.  Working on the half BZ
 * @f$0 \le k_y \le \pi@f$, build the discrete F_12 plaquette flux
 * using the determinant link variable, and the discrete A_1 along
 * the @f$k_y = 0@f$ and @f$k_y = \pi@f$ TRIM lines using the same
 * link variable.  The Z_2 invariant is
 *   nu = (1 / (2 pi)) [ sum_BZ-half F_12 - sum_TRIM A_1 ]  mod 2.
 *
 * @param sys     4-band system at half-filling.
 * @param N       Grid side; must be even, >= 8.
 * @param[out]    z2 Output: 0 (trivial) or 1 (topological).
 * @return 0 on success, negative on error.
 */
MOONLAB_API int qgt_z2_invariant(const qgt_system_n_t* sys, size_t N,
                                  int* z2);

/**
 * @brief Kane-Mele model on the honeycomb lattice (2005).
 *
 * 4-band Hamiltonian with the basis (A-up, B-up, A-down, B-down):
 *   H_KM = t * sum_<ij> c_i^dag c_j
 *        + i lambda_SO * sum_<<ij>> nu_ij c_i^dag s_z c_j
 *        + lambda_R * (Rashba SOC term)
 *        + lambda_v * sum_i xi_i c_i^dag c_i  (sublattice staggering)
 *
 * Topological phase diagram (lambda_R = 0):
 *   nu = 1 (QSH) when |lambda_v| < 3 sqrt(3) |lambda_SO|;
 *   nu = 0       otherwise.
 *
 * @param t           Nearest-neighbour hopping.
 * @param lambda_so   Intrinsic spin-orbit coupling (Z_2 driver).
 * @param lambda_r    Rashba SOC.  Set to 0 for the canonical KM.
 * @param lambda_v    Sublattice mass / staggering.
 *
 * @return Newly-owned 4-band system at half-filling (n_occupied = 2),
 *         or NULL on alloc failure.
 */
MOONLAB_API qgt_system_n_t* qgt_model_kane_mele(double t, double lambda_so,
                                                 double lambda_r,
                                                 double lambda_v);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_QGT_H */
