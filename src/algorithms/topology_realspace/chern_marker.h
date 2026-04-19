/**
 * @file chern_marker.h
 * @brief Bianco-Resta local Chern marker on the Qi-Wu-Zhang model (dense reference).
 *
 * OVERVIEW
 * --------
 * The Chern number of a two-dimensional insulator is a bulk topological
 * invariant traditionally defined as a Brillouin-zone integral of the
 * Berry curvature of the filled band(s).  Bianco and Resta showed that
 * this integer can be obtained equivalently in real space as a trace of
 * the projector onto the filled subspace sandwiched between position
 * operators, which makes the invariant well defined even for aperiodic,
 * disordered, or inhomogeneous systems where Bloch momentum is not a
 * good quantum number.  Concretely, writing @f$\hat P@f$ for the
 * projector onto the filled band(s), @f$\hat Q = \mathbb{1} - \hat P@f$,
 * and @f$\hat X, \hat Y@f$ for the on-site position operators, the
 * local Chern marker at site @f$\mathbf{r}@f$ is
 * @f[
 *   c(\mathbf{r}) \;=\; -\frac{4\pi}{A_{\mathrm{uc}}}\,
 *       \operatorname{Im}
 *       \sum_{\alpha}
 *       \langle \mathbf{r},\alpha|\,
 *       \hat P\,\hat X\,\hat Q\,\hat Y\,\hat P\,
 *       |\mathbf{r},\alpha\rangle,
 * @f]
 * where @f$\alpha@f$ runs over the internal (orbital/spin) degree of
 * freedom at each site and @f$A_{\mathrm{uc}}@f$ is the unit-cell area
 * (unity on our convention-normalised square lattice).  Summed over a
 * bulk region, @f$c(\mathbf r)@f$ converges to the integer Chern number
 * of the occupied band(s); boundary sites carry compensating surface
 * contributions so the total integral over an isolated finite sample
 * vanishes.  The precise statement is Eq. (2) of Bianco-Resta (2011).
 *
 * PROJECTOR CONSTRUCTION
 * ----------------------
 * The projector @f$\hat P@f$ is obtained here without resorting to
 * diagonalisation.  Writing the filled-band projector of a gapped
 * Hamiltonian as
 * @f[
 *   \hat P \;=\; \tfrac12\bigl(\mathbb{1} - \operatorname{sign}\hat H\bigr),
 * @f]
 * we compute @f$\operatorname{sign}\hat H@f$ by the Schulz (Newton-like)
 * iteration
 * @f[
 *   Y_{k+1} \;=\; \tfrac12\,Y_k\bigl(3\mathbb{1} - Y_k^{2}\bigr),
 *   \qquad Y_0 = \hat H / B,
 * @f]
 * with @f$B@f$ chosen above @f$\lVert\hat H\rVert_2@f$ so the spectrum
 * of @f$Y_0@f$ sits inside @f$(-1, 1)@f$.  The iteration converges
 * quadratically to @f$\operatorname{sign}Y_\infty@f$ on any matrix with
 * real eigenvalues whose absolute values avoid zero (the band gap
 * guarantees the latter), with a residual @f$\lVert Y^2 -
 * \mathbb{1}\rVert_F^{2}@f$ decaying to machine precision in
 * @f$O(\log\kappa)@f$ steps where @f$\kappa@f$ is the ratio of the
 * largest to smallest absolute eigenvalue.  A textbook treatment is
 * Higham, *Functions of Matrices: Theory and Computation* (SIAM 2008),
 * §5; the iteration is also the one used by the KPM variant in
 * @c chern_kpm.h at larger system sizes.  Going through the matrix sign
 * function side-steps the known complex-Hermitian eigenvector
 * unsoundness of @c hermitian_eigen_decomposition (see warning in
 * @c matrix_math.h).
 *
 * MODELS
 * ------
 * The built-in Hamiltonian is the Qi-Wu-Zhang (QWZ) Chern insulator, a
 * two-band minimal model with Bloch form
 * @f[
 *   \mathcal H(\mathbf k) \;=\; \sin k_x\,\sigma_x
 *     + \sin k_y\,\sigma_y
 *     + (m + \cos k_x + \cos k_y)\,\sigma_z ,
 * @f]
 * whose Chern number is @f$\mathrm{sgn}(m) - \mathrm{sgn}(m\pm 2)@f$ on
 * the lower band: @f$C = -1@f$ for @f$0 < m < 2@f$, @f$C = +1@f$ for
 * @f$-2 < m < 0@f$, and @f$C = 0@f$ for @f$|m| > 2@f$.  See Qi-Wu-Zhang
 * (2006) for the derivation.  The real-space Hamiltonian built here
 * uses the corresponding hopping matrices @f$T_{\hat x} = (\sigma_z + i
 * \sigma_x)/2@f$ and @f$T_{\hat y} = (\sigma_z + i \sigma_y)/2@f$ on an
 * open @f$L \times L@f$ lattice.
 *
 * ROLE IN MOONLAB
 * ---------------
 * This module is the ground-truth reference for the matrix-free KPM
 * implementation in @c chern_kpm.h and the tensor-network MPO/KPM path
 * planned per Antão et al. (PRL 136, 156601, 2026).  Its ceiling is
 * @f$L \lesssim 20@f$ because the projector is stored as a dense
 * @f$(L^2 \cdot \mathrm{orbs})^2@f$ complex matrix; beyond that scale
 * @c chern_kpm.h or the MPO path should be used.
 *
 * REFERENCES
 * ----------
 *  - R. Bianco and R. Resta, "Mapping topological order in coordinate
 *    space", Phys. Rev. B 84, 241106(R) (2011), arXiv:1111.5697.
 *    Derives Eq. (2) for the local Chern marker in real space.
 *  - X.-L. Qi, Y.-S. Wu and S.-C. Zhang, "Topological quantization of
 *    the spin Hall effect in two-dimensional paramagnetic
 *    semiconductors", Phys. Rev. B 74, 085308 (2006),
 *    arXiv:cond-mat/0505308.  Source of the QWZ model.
 *  - N. J. Higham, "Functions of Matrices: Theory and Computation",
 *    SIAM (2008), §5.  Schulz / Newton iteration for the matrix sign
 *    function; the route we use to avoid complex-Hermitian
 *    diagonalisation.
 *  - T. V. C. Antão, Y. Sun, A. O. Fumega and J. L. Lado,
 *    "Tensor Network Method for Real-Space Topology in Quasicrystal
 *    Chern Mosaics", Phys. Rev. Lett. 136, 156601 (2026),
 *    doi:10.1103/hhdf-xpwg.  Forward target for the MPO/KPM scale-up.
 *
 * @since  v0.2.0
 * @stability evolving
 */

#ifndef MOONLAB_CHERN_MARKER_H
#define MOONLAB_CHERN_MARKER_H

#include <stddef.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef double _Complex cm_complex_t;

typedef struct {
    size_t        L;      /**< linear lattice size; total sites = L*L  */
    size_t        orbs;   /**< internal orbitals per site (2 for QWZ)  */
    double        m;      /**< QWZ mass parameter                      */
    cm_complex_t* H;      /**< dense Hamiltonian, row-major, size dim^2 */
    cm_complex_t* P;      /**< projector onto filled band(s); size dim^2;
                               populated by chern_build_projector()    */
    size_t        dim;    /**< L * L * orbs; convenience cache         */
} chern_system_t;

/**
 * @brief Build the open-boundary QWZ Hamiltonian on an L x L lattice.
 *
 * Stores the dense Hamiltonian inside the returned struct; the
 * projector is left NULL until @c chern_build_projector is called.
 *
 * @param L  linear lattice size (L >= 3)
 * @param m  QWZ mass parameter
 * @return   allocated system, or NULL on out-of-memory.
 */
chern_system_t* chern_qwz_create(size_t L, double m);

/** @brief Release the Hamiltonian, projector, and handle. */
void chern_system_free(chern_system_t* sys);

/**
 * @brief Compute the filled-band projector @f$\hat P@f$ via the matrix
 *        sign function and store it in @p sys->P.
 *
 * @return 0 on success, non-zero on OOM or iteration failure.
 */
int chern_build_projector(chern_system_t* sys);

/**
 * @brief Evaluate the local Chern marker at site @f$(r_x, r_y)@f$.
 *
 * Requires @c chern_build_projector() to have run.  The returned value
 * converges to the band Chern number in the bulk of a clean Chern
 * insulator and to zero in a trivial insulator; boundary sites carry
 * large compensating contributions so isolated-sample sums approach
 * zero by construction.
 *
 * @return @f$c(\mathbf r)@f$ as a real number (units of @f$1/A_{\mathrm{uc}}@f$).
 */
double chern_local_marker(const chern_system_t* sys,
                          size_t rx, size_t ry);

/**
 * @brief Sum @f$c(\mathbf r)@f$ over a rectangular bulk patch
 *        @f$[r_{\min}, r_{\max})^2@f$.
 *
 * For a clean Chern insulator sampled well away from edges, this
 * converges to the integer Chern number times the patch area.
 */
double chern_bulk_sum(const chern_system_t* sys,
                      size_t rmin, size_t rmax);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_CHERN_MARKER_H */
