/**
 * @file mpo_kpm.h
 * @brief MPO-level Chebyshev / Jackson kernel polynomial method.
 *
 * OVERVIEW
 * --------
 * The sparse-stencil KPM in @c chern_kpm.h evaluates the Bianco-Resta
 * local Chern marker by recursion on a fixed five-point stencil,
 * capping out around @f$L = 300@f$ (@f$\sim\!10^{5}@f$ sites) on a
 * single core.  The next rung of the ladder represents the
 * Hamiltonian, states, and position operators as matrix product
 * objects and runs the identical Chebyshev machinery on MPO / MPS
 * data.  The wall-clock then no longer scales with the number of
 * Hilbert-space coefficients but with the MPS bond dimension, which
 * on any gapped 2D system with @f$d \le 2@f$ is typically bounded by
 * low hundreds (Schollwoeck 2011, Orus 2014).  That is the lever that
 * lets Moonlab reach @f$10^{6} \to 10^{8}@f$ sites, per Antão, Sun,
 * Fumega and Lado (PRL 136, 156601, 2026).
 *
 * This is the first of three P5.08 milestones.  It provides:
 *
 *  1. The @b scalar form of the expansion:
 *     @f$\langle\varphi|\operatorname{sign}(\hat H)|\psi\rangle@f$
 *     computed via streaming Chebyshev moments
 *     @f$\mu_n = \langle\varphi|T_n(\hat H)|\psi\rangle@f$.
 *     Scalar matrix elements are sufficient to evaluate the
 *     Bianco-Resta trace formula once position operators are applied
 *     on @f$|\psi\rangle@f$, which is what the downstream milestones
 *     take up.
 *  2. Jackson-kernel regularised sign-function reconstruction.
 *  3. An MPS linear-combination helper implementing the
 *     three-term Chebyshev recurrence
 *     @f$|v_{n+1}\rangle = 2\hat{\tilde H}\,|v_n\rangle - |v_{n-1}\rangle@f$
 *     with block-diagonal bond construction followed by SVD
 *     truncation back to the caller's @c max_bond_dim.
 *  4. An adapter that promotes the DMRG @c mpo_t (the builder used
 *     by @c mpo_tfim_create, @c mpo_heisenberg_create, etc.) to the
 *     gate-API @c tn_mpo_t consumed by @c tn_apply_mpo.  Lets every
 *     existing Moonlab Hamiltonian feed this module without rewriting
 *     builders.
 *
 * CHEBYSHEV EXPANSION
 * -------------------
 * The rescaled Hamiltonian
 * @f$\hat{\tilde H} = (\hat H - a\mathbb{1})/b@f$ with
 * @f$b > \tfrac12(E_{\max} - E_{\min})@f$ has spectrum in
 * @f$(-1, 1)@f$.  The Fermi-level sign function admits
 * @f[
 *   \operatorname{sign}\varepsilon \;\simeq\;
 *   \sum_{n=0}^{N_c - 1} g_n\, c_n\, T_n(\varepsilon),
 *   \qquad c_n = \frac{4}{\pi n}\sin\!\left(\tfrac{n\pi}{2}\right)
 *   \text{ (}n\ \text{odd)}, \quad c_0 = c_{\text{even}} = 0,
 * @f]
 * with the Jackson kernel
 * @f$g_n = [(N_c - n + 1)\cos(n\pi/(N_c+1)) +
 *          \sin(n\pi/(N_c+1))\cot(\pi/(N_c+1))] / (N_c + 1)@f$.
 * Matrix elements follow immediately:
 * @f[
 *   \langle\varphi|\operatorname{sign}(\hat H)|\psi\rangle \;\simeq\;
 *   \sum_{n=0}^{N_c - 1} g_n\, c_n\, \mu_n,
 *   \qquad \mu_n = \langle\varphi|T_n(\hat H)|\psi\rangle.
 * @f]
 * The recurrence on the ket runs in place on two rolling MPS buffers
 * @f$|v_{n-1}\rangle@f$, @f$|v_n\rangle@f$, so peak storage is
 * @f$O(L\,d\,\chi^2)@f$ regardless of @f$N_c@f$.
 *
 * MPS LINEAR COMBINATION
 * ----------------------
 * Given two MPS @f$|A\rangle, |B\rangle@f$ on the same chain, the sum
 * @f$|C\rangle = \alpha|A\rangle + \beta|B\rangle@f$ is exactly an
 * MPS with block-diagonal interior tensors and boundary columns
 * carrying @f$\alpha@f$ and @f$\beta@f$ respectively.  The left-most
 * tensor concatenates @f$[\alpha A[0], \beta B[0]]@f$ along the right
 * bond; the right-most tensor stacks @f$A[L-1]@f$ over @f$B[L-1]@f$
 * along the left bond.  Interior sites are block-diagonal.  Bond
 * dimension at each interior cut is @f$\chi^A_i + \chi^B_i@f$;
 * following each combination we SVD-truncate back to the caller's
 * @c max_bond_dim via @c tn_mps_truncate.
 *
 * CONVERGENCE BUDGET
 * ------------------
 * For a gap @f$\Delta@f$ at the Fermi level and rescale width
 * @f$2b@f$, the Chebyshev sign approximation converges at
 * @f$|\operatorname{sign}(\varepsilon) - \text{trunc}| =
 * O(N_c^{-1})@f$ with the Jackson kernel.  Empirically
 * @f$N_c \in [80, 200]@f$ gives @f$\lesssim 10^{-4}@f$ agreement
 * against dense sign-function references for gaps of order
 * @f$\sim\!0.5 b@f$ on the TFIM test models in this package.
 *
 * @since v0.2.0
 * @stability evolving
 *
 * REFERENCES
 * ----------
 *  - A. Weisse, G. Wellein, A. Alvermann and H. Fehske, "The kernel
 *    polynomial method", Rev. Mod. Phys. 78, 275 (2006),
 *    arXiv:cond-mat/0504627.
 *  - U. Schollwoeck, "The density-matrix renormalization group in the
 *    age of matrix product states", Ann. Phys. 326, 96 (2011),
 *    arXiv:1008.3477.  Standard reference for all MPO / MPS operations
 *    used here.
 *  - T. V. C. Antão, Y. Sun, A. O. Fumega and J. L. Lado, "Tensor
 *    Network Method for Real-Space Topology in Quasicrystal Chern
 *    Mosaics", Phys. Rev. Lett. 136, 156601 (2026),
 *    doi:10.1103/hhdf-xpwg.  Blueprint for the full P5.08 pipeline.
 */

#ifndef MOONLAB_MPO_KPM_H
#define MOONLAB_MPO_KPM_H

#include <stddef.h>
#include <complex.h>

#include "../tensor_network/dmrg.h"
#include "../tensor_network/tn_gates.h"
#include "../tensor_network/tn_state.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Parameters controlling a single Chebyshev / Jackson evaluation.
 */
typedef struct {
    size_t   n_cheby;        /**< Chebyshev truncation order @f$N_c@f$ */
    double   E_shift;        /**< Spectrum shift a used in (H - a)/b    */
    double   E_scale;        /**< Spectrum scale b; must be > 0         */
    uint32_t max_bond_dim;   /**< MPS truncation cap after each step    */
    double   svd_cutoff;     /**< SVD cutoff for truncation             */
    int      use_jackson;    /**< 1 = apply Jackson kernel, 0 = raw     */
} mpo_kpm_params_t;

/**
 * @brief Default parameters: N_c = 80, unit rescale, bond dim 128,
 *        SVD cutoff 1e-12, Jackson kernel on.  Callers typically
 *        adjust n_cheby and E_shift / E_scale.
 */
mpo_kpm_params_t mpo_kpm_params_default(void);

/**
 * @brief Fill the Jackson kernel weights
 *        @f$g_n@f$ for @f$n = 0 \ldots N_c - 1@f$.
 *
 * @param n_cheby  Chebyshev truncation order; must be >= 1.
 * @param g_out    Output array of length @p n_cheby.
 */
void mpo_kpm_jackson_weights(size_t n_cheby, double* g_out);

/**
 * @brief Fill the Chebyshev coefficients of the sign function
 *        @f$c_0 = 0, c_{2k} = 0, c_{2k+1} = \tfrac{4}{\pi(2k+1)}(-1)^k@f$.
 *
 * @param n_cheby  Chebyshev truncation order.
 * @param c_out    Output array of length @p n_cheby.
 */
void mpo_kpm_sign_coefficients(size_t n_cheby, double* c_out);

/**
 * @brief Adapt a DMRG @c mpo_t (builder output) to a gate-API
 *        @c tn_mpo_t (consumed by @c tn_apply_mpo).  The returned MPO
 *        owns deep copies of the tensors and is freed with
 *        @c tn_mpo_free.  Does not mutate the input.
 *
 * @return Newly allocated @c tn_mpo_t or NULL on invalid input / OOM.
 */
tn_mpo_t* mpo_kpm_mpo_to_tn_mpo(const mpo_t* H);

/**
 * @brief Combine @f$|C\rangle = \alpha|A\rangle + \beta|B\rangle@f$ as
 *        an MPS via block-diagonal construction.  The caller is
 *        responsible for truncation; the returned MPS has bond
 *        dimension @f$\chi^A_i + \chi^B_i@f$ at each interior bond
 *        and is left in a non-canonical form.
 *
 * @param A        First MPS.
 * @param alpha    Scalar on @p A.
 * @param B        Second MPS.
 * @param beta     Scalar on @p B.
 * @return         Allocated MPS or NULL on size mismatch / OOM.
 */
tn_mps_state_t* mpo_kpm_mps_combine(
    const tn_mps_state_t* A, double complex alpha,
    const tn_mps_state_t* B, double complex beta);

/**
 * @brief Streaming Chebyshev moments
 *        @f$\mu_n = \langle\mathrm{bra}|T_n(\hat{\tilde H})|\mathrm{ket}\rangle@f$
 *        for @f$n = 0 \ldots N_c - 1@f$, where
 *        @f$\hat{\tilde H} = (\hat H - a\mathbb{1})/b@f$ with
 *        @f$a = \mathtt{params->E\_shift}@f$ and
 *        @f$b = \mathtt{params->E\_scale}@f$.
 *
 * The caller supplies the bare Hamiltonian MPO; the shift and scale
 * are baked into the three-term recurrence rather than materialised
 * as a rescaled MPO, which avoids extending the MPO bond dimension
 * and keeps the bare H tensors intact.  Peak MPS bond dimension
 * during the recurrence is bounded by
 * @f$\max(\chi^{\mathrm{ket}}, \mathtt{params->max\_bond\_dim})@f$
 * after the first truncation.
 *
 * @param H             Bare Hamiltonian MPO (not rescaled).
 * @param bra           Left state (not modified).
 * @param ket           Right state (not modified; a working copy is
 *                      made internally).
 * @param params        Parameters (@c n_cheby, bond dim, cutoff).
 * @param moments_out   Output array of length @c params->n_cheby.
 * @return 0 on success, nonzero on failure.
 */
int mpo_kpm_chebyshev_moments(
    const tn_mpo_t* H,
    const tn_mps_state_t* bra,
    const tn_mps_state_t* ket,
    const mpo_kpm_params_t* params,
    double complex* moments_out);

/**
 * @brief Convenience: compute
 *        @f$\langle\mathrm{bra}|\operatorname{sign}(\hat{\tilde H})|\mathrm{ket}\rangle@f$
 *        via moments + Jackson-kernel sign reconstruction, where
 *        @f$\hat{\tilde H} = (\hat H - \mathtt{E\_shift}\mathbb{1})/\mathtt{E\_scale}@f$.
 */
double complex mpo_kpm_sign_matrix_element(
    const tn_mpo_t* H,
    const tn_mps_state_t* bra,
    const tn_mps_state_t* ket,
    const mpo_kpm_params_t* params);

/**
 * @brief Convenience: compute the filled-band projector matrix
 *        element
 *        @f$\langle\mathrm{bra}|\hat P|\mathrm{ket}\rangle =
 *          \tfrac12(\langle\mathrm{bra}|\mathrm{ket}\rangle -
 *                   \langle\mathrm{bra}|\operatorname{sign}(\hat{\tilde H})|\mathrm{ket}\rangle)@f$,
 *        with @f$\hat{\tilde H}@f$ rescaled from @c params as above.
 *        @c params->E_shift is the Fermi energy, @c E_scale is the
 *        half-bandwidth used to place the spectrum in @f$(-1, 1)@f$.
 */
double complex mpo_kpm_projector_matrix_element(
    const tn_mpo_t* H,
    const tn_mps_state_t* bra,
    const tn_mps_state_t* ket,
    const mpo_kpm_params_t* params);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_MPO_KPM_H */
