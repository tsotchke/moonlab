/**
 * @file lattice_z2_1d.h
 * @brief 1+1D Z2 lattice gauge theory (Schwinger-style) on an OBC qubit chain.
 *
 * The simplest non-trivial lattice gauge theory: matter coupled to a Z2
 * gauge field on a 1D chain.  Hilbert space layout (interleaved):
 *
 *   matter site 0 -- link 0 -- matter site 1 -- link 1 -- ... -- matter site N-1
 *   qubit:    0          1          2              3            2N-2
 *
 * Total qubit count: 2*N - 1, where N is the number of matter sites.
 *
 * Hamiltonian (Jordan-Wigner with parallel transport, gauge-invariant
 * kinetic terms):
 *
 *   H = -(t/2) sum_x [X_{2x} Y_{2x+1} Y_{2x+2} - Y_{2x} Y_{2x+1} X_{2x+2}]
 *       - h sum_x Z_{2x+1}                       (electric field on links)
 *       + (m/2) sum_x (-1)^x Z_{2x}              (staggered mass)
 *       + lambda sum_{x=1..N-2} (I - G_x)        (Gauss-law penalty)
 *
 * where the Gauss-law operator at interior matter site x is
 *   G_x = X_{2x-1} Z_{2x} X_{2x+1}.
 *
 * The gauge-invariant (physical) subspace is the simultaneous +1 eigenspace
 * of all G_x.  Each kinetic-term Pauli string (XYY and YYX) commutes with
 * every G_x term-by-term, so H itself preserves the gauge sector exactly;
 * the lambda term is then a redundant energetic enforcement that can be
 * set to zero without changing the physical-sector spectrum.
 *
 * Why this is a clean var-D testbed:
 *   - Each G_x is a 3-qubit Pauli string -- a stabilizer operator.
 *   - The N-2 interior G_x's commute, generate a stabilizer subgroup.
 *   - A Clifford D that diagonalises this subgroup maps the gauge-invariant
 *     subspace to a "computational" subspace (some qubits frozen to |0>).
 *   - var-D should discover such a D, driving |phi>'s effective dimension
 *     down to the dynamical content (matter + electric field fluctuations).
 *
 * Connection to the broader research program: 1+1D Z2 LGT is the simplest
 * member of the gauge-theory hierarchy that connects HEP (Schwinger model,
 * lattice QCD), topological QC (toric code is the 2+1D analog), and
 * condensed matter (confinement-deconfinement transition is dual to the
 * TFIM phase transition under the Kramers-Wannier transformation).  See
 * docs/research/var_d_lattice_gauge_theory.md.
 */

#ifndef MOONLAB_LATTICE_Z2_1D_H
#define MOONLAB_LATTICE_Z2_1D_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Configuration for the 1+1D Z2 LGT Hamiltonian on N matter sites. */
typedef struct {
    uint32_t num_matter_sites;   /**< N >= 2 */
    double   t_hop;              /**< matter hopping amplitude (>= 0) */
    double   h_link;             /**< electric field strength on each link */
    double   mass;               /**< staggered fermion mass */
    double   gauss_penalty;      /**< lambda; set 0 to skip Gauss-law term */
} z2_lgt_config_t;

/**
 * @brief Build the qubit Pauli sum for the configured Hamiltonian.
 *
 * Allocates @p *out_paulis as a flat row-major (num_terms, num_qubits)
 * uint8 array (encoding 0=I, 1=X, 2=Y, 3=Z) and @p *out_coeffs as a
 * length num_terms double array of real Pauli-sum coefficients.  The
 * caller is responsible for free()-ing both arrays.
 *
 * @return 0 on success, negative on invalid configuration.
 */
int z2_lgt_1d_build_pauli_sum(const z2_lgt_config_t* cfg,
                                uint8_t** out_paulis,
                                double**  out_coeffs,
                                uint32_t* out_num_terms,
                                uint32_t* out_num_qubits);

/**
 * @brief Number of qubits the Hamiltonian acts on for a given config:
 *        2*N - 1 = N matter + (N - 1) link qubits.
 */
uint32_t z2_lgt_1d_num_qubits(const z2_lgt_config_t* cfg);

/**
 * @brief Construct the Pauli-string representation of the Gauss-law
 *        operator at interior matter site @p site_x in [1, N-2].
 *
 * Writes G_x = X_{2*site_x - 1} * X_{2*site_x + 1} * Z_{2*site_x}
 * as a length (2*N - 1) Pauli string in @p out_pauli (encoding as
 * above).  This is a Hermitian Pauli with phase 0.
 *
 * @return 0 on success, negative if site_x is out of range.
 */
int z2_lgt_1d_gauss_law_pauli(const z2_lgt_config_t* cfg,
                                 uint32_t site_x,
                                 uint8_t* out_pauli);

/**
 * @brief Construct a Wilson loop operator on consecutive links
 *        @p link_start .. @p link_end inclusive, i.e. the product
 *        Z_{link_start} ... Z_{link_end} in qubit indices.
 *
 * Writes the Pauli string into @p out_pauli of length 2*N - 1.
 *
 * @return 0 on success.
 */
int z2_lgt_1d_wilson_line_pauli(const z2_lgt_config_t* cfg,
                                  uint32_t link_start,
                                  uint32_t link_end,
                                  uint8_t* out_pauli);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_LATTICE_Z2_1D_H */
