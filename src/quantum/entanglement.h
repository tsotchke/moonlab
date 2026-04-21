/**
 * @file entanglement.h
 * @brief Bipartite entanglement measures on pure and mixed states.
 *
 * OVERVIEW
 * --------
 * For a pure state @f$|\psi\rangle@f$ on @f$\mathcal H_A \otimes
 * \mathcal H_B@f$, the reduced density matrix
 * @f$\rho_A = \operatorname{Tr}_B\,|\psi\rangle\langle\psi|@f$
 * carries the full entanglement information of the bipartition.
 * Equivalent characterisations of its "mixedness" correspond to
 * different entanglement measures:
 *
 *   - *Von Neumann entropy*
 *     @f$S(\rho_A) = -\operatorname{Tr}\,\rho_A \log_2 \rho_A@f$ is the
 *     canonical bipartite entanglement measure of a pure state, equal
 *     to the Shannon entropy of its Schmidt coefficients.  For mixed
 *     states it is replaced by the *entanglement of formation*
 *     (Bennett-DiVincenzo-Smolin-Wootters 1996).
 *   - *Renyi-@f$\alpha@f$ entropy*
 *     @f$S_\alpha(\rho_A) = (1 - \alpha)^{-1}\log_2 \operatorname{Tr}\,\rho_A^{\alpha}@f$
 *     interpolates between the min-entropy (@f$\alpha\to\infty@f$),
 *     the collision entropy (@f$\alpha = 2@f$) and the von Neumann
 *     entropy (@f$\alpha \to 1@f$).
 *   - *Linear entropy* @f$S_L = 1 - \operatorname{Tr}\,\rho_A^{2}@f$ is
 *     a numerically friendly surrogate for small bipartite systems.
 *
 * FOR TWO-QUBIT SYSTEMS
 * ---------------------
 * The Wootters *concurrence* is
 * @f[
 *   C(\rho) \;=\; \max\!\bigl(0,\; \sqrt{\lambda_1} - \sqrt{\lambda_2}
 *                - \sqrt{\lambda_3} - \sqrt{\lambda_4}\bigr),
 * @f]
 * where @f$\lambda_i@f$ are the eigenvalues of
 * @f$\rho (\sigma_y\otimes\sigma_y) \rho^{\ast} (\sigma_y\otimes\sigma_y)@f$
 * in decreasing order; @f$C(\rho)=0@f$ iff @f$\rho@f$ is separable,
 * @f$C(\rho)=1@f$ for a Bell state.  The *entanglement of formation*
 * is a monotonic function of @f$C@f$ alone.  The *logarithmic
 * negativity* uses the partial transpose:
 * @f$\mathcal N(\rho) = (\lVert\rho^{T_B}\rVert_1 - 1)/2@f$.  For pure
 * two-qubit states @f$\mathcal N = C/2@f$, reaching @f$1/2@f$ on Bell
 * states and @f$0@f$ on product states.  These measures are the
 * standard "small-system" entanglement certificates (see Horodecki
 * review for the general theory).
 *
 * REFERENCES
 * ----------
 *  - C. H. Bennett, D. P. DiVincenzo, J. A. Smolin and W. K. Wootters,
 *    "Mixed-state entanglement and quantum error correction",
 *    Phys. Rev. A 54, 3824 (1996), arXiv:quant-ph/9604024.  Origin
 *    of entanglement of formation.
 *  - W. K. Wootters, "Entanglement of Formation of an Arbitrary State
 *    of Two Qubits", Phys. Rev. Lett. 80, 2245 (1998),
 *    arXiv:quant-ph/9709029.  Closed-form concurrence.
 *  - G. Vidal and R. F. Werner, "A computable measure of entanglement",
 *    Phys. Rev. A 65, 032314 (2002), arXiv:quant-ph/0102117.
 *    Logarithmic negativity.
 *  - R. Horodecki, P. Horodecki, M. Horodecki and K. Horodecki,
 *    "Quantum entanglement", Rev. Mod. Phys. 81, 865 (2009),
 *    arXiv:quant-ph/0702225.  Canonical review; all definitions used
 *    here follow the conventions there.
 *
 * @stability evolving
 * @since v0.1.2
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef QUANTUM_ENTANGLEMENT_H
#define QUANTUM_ENTANGLEMENT_H

#include "state.h"
#include <stdint.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// REDUCED DENSITY MATRIX
// ============================================================================

/**
 * @brief Compute reduced density matrix by tracing out qubits
 */
int entanglement_reduced_density_matrix(const quantum_state_t* state,
                                        const int* trace_out_qubits,
                                        int num_trace_out,
                                        complex_t* reduced_dm,
                                        uint64_t* reduced_dim);

// ============================================================================
// ENTANGLEMENT ENTROPY
// ============================================================================

/**
 * @brief Compute von Neumann entropy S = -Tr(ρ log₂ ρ)
 */
double entanglement_von_neumann_entropy(const complex_t* reduced_dm, uint64_t dim);

/**
 * @brief Compute Renyi entropy of order α
 */
double entanglement_renyi_entropy(const complex_t* reduced_dm, uint64_t dim,
                                  double alpha);

/**
 * @brief Compute entanglement entropy for bipartition
 */
double entanglement_entropy_bipartition(const quantum_state_t* state,
                                        const int* subsystem_b_qubits,
                                        int num_b_qubits);

/**
 * @brief Quantum mutual information I(A:B) = S(A) + S(B) - S(AB).
 *
 * On a pure state of the A u B system, S(AB) = 0, so this reduces
 * to S(A) + S(B) = 2 S(A) (symmetric).  For a pure state of a
 * larger system, the caller should pass the indices of both
 * partitions A and B (disjoint); any qubits not in A u B are
 * traced out first.
 *
 * @param state         pure state over num_qubits >= |A| + |B|.
 * @param qubits_a      indices of subsystem A; distinct from B.
 * @param num_a         length of @p qubits_a.
 * @param qubits_b      indices of subsystem B.
 * @param num_b         length of @p qubits_b.
 * @return I(A:B) in bits (log base 2), >= 0; 0.0 on argument error.
 */
double entanglement_mutual_information(const quantum_state_t* state,
                                        const int* qubits_a, int num_a,
                                        const int* qubits_b, int num_b);

// ============================================================================
// CONCURRENCE
// ============================================================================

/**
 * @brief Compute concurrence for pure 2-qubit state
 * @return C in [0, 1]
 */
double entanglement_concurrence_2qubit(const quantum_state_t* state);

/**
 * @brief Compute concurrence from 2-qubit density matrix
 */
double entanglement_concurrence_mixed(const complex_t* density_matrix);

/**
 * @brief Negativity N(ρ) = (‖ρ^TB‖_1 − 1)/2 for a pure 2-qubit state.
 *        Equal to (C + 0)/2 for pure states, where C is concurrence.
 * @return N in [0, 1/2]; 0 for separable, 1/2 for a maximally-entangled
 *         Bell state.
 */
double entanglement_negativity_2qubit(const quantum_state_t* state);

// ============================================================================
// SCHMIDT DECOMPOSITION
// ============================================================================

/**
 * @brief Compute Schmidt coefficients
 */
int entanglement_schmidt_coefficients(const quantum_state_t* state,
                                      const int* partition_a_qubits,
                                      int num_a,
                                      double* coefficients,
                                      int* num_coefficients);

/**
 * @brief Compute Schmidt rank
 */
int entanglement_schmidt_rank(const quantum_state_t* state,
                              const int* partition_a_qubits,
                              int num_a,
                              double threshold);

// ============================================================================
// ENTANGLEMENT DETECTION
// ============================================================================

/**
 * @brief Check if state is separable
 * @return 1 if separable, 0 if entangled
 */
int entanglement_is_separable(const quantum_state_t* state,
                              const int* partition_a_qubits,
                              int num_a);

/**
 * @brief Compute purity Tr(ρ²)
 */
double entanglement_purity(const complex_t* reduced_dm, uint64_t dim);

/**
 * @brief Compute linear entropy
 */
double entanglement_linear_entropy(const complex_t* reduced_dm, uint64_t dim);

#ifdef __cplusplus
}
#endif

#endif /* QUANTUM_ENTANGLEMENT_H */
