/**
 * @file chern_kpm.h
 * @brief Matrix-free kernel-polynomial local Chern marker.
 *
 * OVERVIEW
 * --------
 * The dense-reference module in @c chern_marker.h evaluates the
 * Bianco-Resta local Chern marker by forming the filled-band projector
 * @f$\hat P = \tfrac12(\mathbb{1} - \operatorname{sign}\hat H)@f$
 * explicitly as a dense matrix of dimension @f$N = L^2\,\mathrm{orbs}@f$.
 * That storage grows as @f$O(N^2)@f$ and caps out around
 * @f$L \approx 20@f$.  This module replaces the explicit projector with
 * a matrix-free action: the operation @f$\hat P|\psi\rangle@f$ is
 * executed by recursion on a Chebyshev polynomial of the rescaled
 * Hamiltonian, never materialising @f$\hat P@f$ as an array.  Memory is
 * @f$O(N)@f$; wall-clock per bulk site scales as
 * @f$O(N_{\mathrm{cheb}}\,\mathrm{nnz}(\hat H))@f$ with
 * @f$\mathrm{nnz}(\hat H) = 5N@f$ for the QWZ stencil, so the
 * implementation reaches @f$L = 300@f$ (90k-site lattice) on a single
 * core.  The algorithmic ingredients below all appear in the kernel
 * polynomial method (KPM) review by Weisse, Wellein, Alvermann and
 * Fehske.
 *
 * CHEBYSHEV EXPANSION OF THE PROJECTOR
 * ------------------------------------
 * Rescale the Hamiltonian so its spectrum lies in @f$(-1, 1)@f$,
 * @f[
 *   \hat{\tilde H} \;=\; \frac{\hat H - a\,\mathbb{1}}{b},
 *   \qquad b > \tfrac12(E_{\max} - E_{\min}).
 * @f]
 * The Fermi-level sign function has the classical Chebyshev expansion
 * @f[
 *   \operatorname{sign}\varepsilon \;\simeq\;
 *   \sum_{n=0}^{N_c - 1} g_n\, c_n\, T_n(\varepsilon),
 *   \qquad
 *   c_n = \begin{cases} 0 & n\ \mathrm{even} \\
 *                       \dfrac{4}{\pi n}\,\sin\!\left(\tfrac{n\pi}{2}\right)
 *                       & n\ \mathrm{odd}\end{cases}
 * @f]
 * where @f$T_n@f$ are the Chebyshev polynomials of the first kind and
 * @f$g_n@f$ is the Jackson kernel,
 * @f[
 *   g_n \;=\; \frac{(N_c - n + 1)\cos\!\bigl(\tfrac{n\pi}{N_c + 1}\bigr)
 *           + \sin\!\bigl(\tfrac{n\pi}{N_c + 1}\bigr)\cot\!\bigl(\tfrac{\pi}{N_c + 1}\bigr)}
 *           {N_c + 1},
 * @f]
 * which enforces a non-negative, polynomially-damped approximation of
 * @f$\operatorname{sign}@f$ (the "Gibbs phenomenon cure" of §II.B in
 * Weisse et al. 2006).  @f$T_n(\hat{\tilde H})|\psi\rangle@f$ is
 * evaluated by the three-term recurrence
 * @f[
 *   T_{n+1}|\psi\rangle \;=\; 2\,\hat{\tilde H}\,T_n|\psi\rangle - T_{n-1}|\psi\rangle,
 * @f]
 * requiring only a sparse matrix-vector multiplication per term.  The
 * filled-band projector action is then
 * @f[
 *   \hat P|\psi\rangle \;=\;
 *   \tfrac12\bigl(|\psi\rangle - \operatorname{sign}(\hat{\tilde H})\,|\psi\rangle\bigr).
 * @f]
 * The Bianco-Resta marker is obtained by three projector applications
 * sandwiching the diagonal position operators, per Eq. (2) of
 * Bianco-Resta.
 *
 * MATRIX-FREE HAMILTONIAN ACTION
 * ------------------------------
 * For QWZ the Hamiltonian is a five-point stencil (on-site
 * @f$m\,\sigma_z@f$, hops to the four nearest neighbours), applied in
 * @f$O(N)@f$ operations per matvec.  An optional spin-independent
 * on-site modulation @f$V(\mathbf r)\,\mathbb{1}_{2\times 2}@f$ lets
 * the caller probe quasi-crystalline Chern mosaics in the spirit of
 * Antão, Sun, Fumega, Lado (PRL 136, 156601, 2026); when a modulation
 * with maximum amplitude @f$V_{\max}@f$ is attached the spectral
 * rescale @f$b@f$ is widened by @f$V_{\max}@f$ to keep the Chebyshev
 * recurrence bounded.
 *
 * CONVERGENCE BUDGET
 * ------------------
 * The Jackson-regularised Chebyshev approximation of
 * @f$\operatorname{sign}@f$ converges uniformly on any compact subset
 * of @f$[-1, 1] \setminus \{0\}@f$, with error decaying as
 * @f$O(N_c^{-1})@f$ for a fixed distance to the gap and
 * @f$O(N_c^{-1/2})@f$ if one samples right up to it (Weisse et al. §V).
 * In practice @f$N_c \in [80, 200]@f$ gives five to six decimal digits
 * for QWZ gaps of order unity; we default to @f$N_c@f$ scaling roughly
 * as the lattice linear size so as to keep the leading KPM residual
 * below the boundary-induced finite-size correction to the bulk
 * marker.
 *
 * PARALLELISM
 * -----------
 * Bulk-marker evaluation decomposes into independent per-site tasks,
 * each of which maintains its own Chebyshev workspace.  The
 * @c chern_kpm_bulk_map entry point parallelises over sites via OpenMP
 * where available; on a 24-thread Apple Silicon host the measured
 * speedup is @f$\approx 13\times@f$ for a 256-site mosaic.
 *
 * ROLE IN MOONLAB
 * ---------------
 * This is the sparse-stencil implementation of the Bianco-Resta
 * marker, capping at @f$\sim\!10^{5}@f$ sites on a single host.  The
 * MPO/MPS generalisation needed to reach @f$10^{6} \to 10^{8}@f$ sites
 * per Antão et al. lives in @c mpo_kpm.h (P5.16 matrix-element form)
 * and is built up to the full local-marker observable in the P5.08
 * milestones.
 *
 * REFERENCES
 * ----------
 *  - A. Weisse, G. Wellein, A. Alvermann and H. Fehske,
 *    "The kernel polynomial method",
 *    Rev. Mod. Phys. 78, 275 (2006), arXiv:cond-mat/0504627.
 *    Definitive review; Chebyshev coefficients, Jackson kernel,
 *    stochastic evaluation of traces and diagonal matrix elements.
 *  - R. Bianco and R. Resta, "Mapping topological order in coordinate
 *    space", Phys. Rev. B 84, 241106(R) (2011), arXiv:1111.5697.
 *    Local-marker formula we evaluate.
 *  - T. V. C. Antão, Y. Sun, A. O. Fumega and J. L. Lado,
 *    "Tensor Network Method for Real-Space Topology in Quasicrystal
 *    Chern Mosaics", Phys. Rev. Lett. 136, 156601 (2026),
 *    doi:10.1103/hhdf-xpwg.  Blueprint for the upcoming MPO/QTCI
 *    scale-up; validates the present KPM path as a ground truth.
 *
 * @since  v0.2.0
 * @stability evolving
 */

#ifndef MOONLAB_CHERN_KPM_H
#define MOONLAB_CHERN_KPM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t  L;           /**< linear lattice size (L x L unit cells)       */
    size_t  orbs;        /**< orbitals per site (2 for QWZ)                */
    size_t  N;           /**< L * L * orbs, i.e. Hamiltonian dimension     */
    double  m;           /**< QWZ mass parameter                           */
    double  E_shift;     /**< spectrum shift a used in H_tilde = (H - a)/b */
    double  E_scale;     /**< spectrum scale b used in H_tilde = (H - a)/b */
    size_t  n_cheby;     /**< truncation order N_c of the Chebyshev series */
    double* modulation;  /**< optional L*L on-site potential V(r); NULL off*/
    double  mod_maxabs;  /**< upper bound on |V|, used to widen E_scale    */
} chern_kpm_system_t;

/**
 * @brief Initialise a matrix-free QWZ system.
 *
 * The Hamiltonian is never built explicitly; its action is executed on
 * the fly by a five-point stencil.  For canonical QWZ parameters
 * (@f$|m| \lesssim 1@f$, band gap @f$\Delta \sim |m|@f$) use
 * @f$N_c \approx 40{-}200@f$; larger gaps admit smaller @f$N_c@f$.
 *
 * @param L        linear lattice size (open BCs); L >= 3
 * @param m        QWZ mass parameter
 * @param n_cheby  Chebyshev truncation order; n_cheby >= 8
 * @return         allocated handle, or NULL on invalid arguments / OOM.
 */
chern_kpm_system_t* chern_kpm_create(size_t L, double m, size_t n_cheby);

/** @brief Release memory owned by the handle. */
void chern_kpm_free(chern_kpm_system_t* sys);

/**
 * @brief Local Bianco-Resta marker at lattice site @p (rx, ry).
 *
 * Performs three matrix-free projector applications sandwiching the
 * diagonal position operators.  Memory is @f$O(N)@f$; work is
 * @f$O(N_c \cdot \mathrm{nnz}(\hat H))@f$ per site.
 */
double chern_kpm_local_marker(const chern_kpm_system_t* sys,
                              size_t rx, size_t ry);

/**
 * @brief Integrate the marker over a square bulk patch.
 *
 * Parallelised across sites via OpenMP where available.
 */
double chern_kpm_bulk_sum(const chern_kpm_system_t* sys,
                          size_t rmin, size_t rmax);

/**
 * @brief Fill a row-major @f$(r_{\max}-r_{\min})^2@f$ array with the
 *        per-site marker, parallelised by site.
 *
 * @return 0 on success, nonzero on invalid range / OOM.
 */
int chern_kpm_bulk_map(const chern_kpm_system_t* sys,
                       size_t rmin, size_t rmax,
                       double* out);

/**
 * @brief Attach a spin-independent on-site potential @f$V(\mathbf r)
 *        \mathbb{1}_{2\times 2}@f$ to the Hamiltonian.
 *
 * The input array is a row-major @f$L \times L@f$ real field; the
 * system stores the pointer (no copy), so the caller must keep the
 * buffer alive for the lifetime of @p sys or until the next call to
 * this function.  Pass NULL to detach.
 *
 * @p V_maxabs is an upper bound on @f$|V(\mathbf r)|@f$ and is used to
 * widen the spectral rescale so the Chebyshev recurrence stays
 * numerically stable.
 *
 * @return 0 on success, non-zero on invalid arguments.
 */
int chern_kpm_set_modulation(chern_kpm_system_t* sys,
                             const double* V_per_site,
                             double V_maxabs);

/**
 * @brief Allocate an @f$L \times L@f$ real field with a
 *        @f$C_n@f$-symmetric cosine modulation
 *        @f$V(\mathbf r) = V_0\sum_{i=0}^{n-1} \cos(\mathbf q_i\cdot\mathbf r)@f$,
 *        @f$|\mathbf q_i| = Q@f$,
 *        @f$\mathrm{arg}\,\mathbf q_i = 2\pi i/n@f$.
 *
 * Canonical quasi-crystal choices: @f$n = 4@f$ (square), @f$8@f$
 * (octagonal), @f$10@f$ (decagonal).  Ownership transfers to the
 * caller; free with @c free when done.
 */
double* chern_kpm_cn_modulation(size_t L, int n, double Q, double V0);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_CHERN_KPM_H */
