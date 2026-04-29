/**
 * @file ca_mps_var_d.h
 * @brief Variational-D mode for CA-MPS (greedy local-Clifford search).
 *
 * Per docs/research/ca_mps.md §5.3.  Given a CA-MPS state |psi> = D|phi>
 * and a Hamiltonian H expressed as a Pauli sum, search over single- and
 * two-qubit Clifford gates to find a D that minimises the variational
 * energy <psi|H|psi> = <phi|D^dagger H D|phi> at fixed |phi>.
 *
 * v0.1 of this API ships only the D-update half of the alternating
 * loop -- |phi> stays fixed.  This is enough to validate the greedy
 * search machinery on the oracle-proof workloads (TFIM, where a known
 * Clifford exists that brings <phi|H'|phi> close to the variational
 * minimum given a product-state |phi>).
 *
 * The companion |phi>-update (DMRG sweep on H' = D^dagger H D) is the
 * larger chunk and lands as moonlab_ca_mps_optimize_var_d_alternating
 * in v0.2 of this header once the MPO-conjugation primitive ships.
 */

#ifndef MOONLAB_CA_MPS_VAR_D_H
#define MOONLAB_CA_MPS_VAR_D_H

#include "ca_mps.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /** Maximum number of greedy passes (each pass tries every candidate gate
     *  on every qubit / qubit pair). */
    int max_passes;

    /** Stop when no candidate gate decreases the energy by more than this
     *  (in units of the Hamiltonian's coefficient scale). */
    double improvement_eps;

    /** Include 2-qubit Cliffords (CNOT, CZ, SWAP) on nearest-neighbour
     *  qubit pairs.  Single-qubit gates (H, S, S^dag) are always
     *  searched. */
    int include_2q_gates;

    /** Print one line per accepted gate to stdout. */
    int verbose;
} ca_mps_var_d_config_t;

ca_mps_var_d_config_t ca_mps_var_d_config_default(void);

typedef struct {
    /** Energy at the start, before any gates were applied. */
    double initial_energy;
    /** Energy after the search converged or hit max_passes. */
    double final_energy;
    /** Max half-cut entanglement entropy of |phi> at the start. */
    double initial_phi_entropy;
    /** Max half-cut entanglement entropy of |phi> at the end. */
    double final_phi_entropy;
    /** Number of greedy gate-accepts the search performed. */
    int gates_added;
    /** Number of full passes executed (>= 1, <= config.max_passes). */
    int passes;
    /** True if a pass completed without accepting any gate (search reached
     *  a local minimum); false if max_passes was hit first. */
    int converged;
} ca_mps_var_d_result_t;

/**
 * @brief Greedy local-Clifford search at fixed |phi>.
 *
 * Mutates @p state->D in place.  The MPS factor @p state->phi is *not*
 * modified.  The Hamiltonian must be expressed as a Pauli sum with real
 * coefficients (Hermitian H); the convention for @p paulis matches
 * ::moonlab_ca_mps_expect_pauli_sum (row-major, [num_terms][num_qubits]
 * with bytes in {0=I, 1=X, 2=Y, 3=Z}).
 *
 * @param[in,out] state     CA-MPS handle.  state->D is mutated; state->phi
 *                          is read-only.
 * @param[in]     paulis    Pauli strings as a [num_terms][num_qubits] flat
 *                          uint8_t array.
 * @param[in]     coeffs    Real coefficients of the Pauli terms.
 * @param[in]     num_terms Number of Pauli terms.
 * @param[in]     config    Search configuration; pass NULL for defaults.
 * @param[out]    result    Optional result struct; pass NULL if not wanted.
 *
 * @return CA_MPS_SUCCESS on success.
 *
 * Note: for a generic Hermitian Hamiltonian the Pauli-sum coefficients
 * may be complex (in fact the Pauli decomposition of any Hermitian
 * operator has real coefficients up to phase, but the full Pauli sum
 * uses purely real Pauli strings with real coefficients).  This API
 * takes real coefficients to make that explicit.
 */
ca_mps_error_t moonlab_ca_mps_optimize_var_d_clifford_only(
    moonlab_ca_mps_t* state,
    const uint8_t* paulis,
    const double* coeffs,
    uint32_t num_terms,
    const ca_mps_var_d_config_t* config,
    ca_mps_var_d_result_t* result);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_CA_MPS_VAR_D_H */
