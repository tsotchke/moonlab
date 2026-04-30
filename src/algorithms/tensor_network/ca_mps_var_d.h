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

    /** Try 2-gate composite moves -- pairs (G1, G2) applied in sequence
     *  -- in addition to single-gate moves.  Helps escape 1-gate local
     *  minima where the right descent direction requires two gates that
     *  individually look bad.  Cost: O(N^2 * G^2) per pass instead of
     *  O(N * G), so set to 0 when speed matters. */
    int composite_2gate;

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

/* ================================================================== */
/*  Alternating optimization (imag-time |phi> + greedy Clifford D)     */
/* ================================================================== */

/** Warm-start initial Clifford for D.  When the alternating optimiser
 *  starts at D = I, the greedy local search can get trapped in a basin
 *  with no descent direction toward Cliffords that need many gates to
 *  reach -- notably, the dual H_all + CNOT-chain Clifford that the
 *  oracle proof showed is the right answer for transverse-field
 *  systems near criticality.  Warm-starting D to a structured Clifford
 *  before the alternating loop puts the search in a more productive
 *  basin. */
typedef enum {
    CA_MPS_WARMSTART_IDENTITY = 0,    /* D starts at I (default) */
    CA_MPS_WARMSTART_H_ALL,           /* D = product of H on every qubit */
    CA_MPS_WARMSTART_DUAL_TFIM,       /* D = H_all then CNOT-chain (TFIM-dual) */
    CA_MPS_WARMSTART_FERRO_TFIM,      /* D = CNOT-chain then H_0 (cat-state encoder) */
    /* D is built by symplectic Gauss-Jordan elimination on a
     * caller-supplied list of commuting Pauli generators (the
     * stabilizer subgroup S).  The resulting D|0^n> is in the
     * simultaneous +1 eigenspace of every g in S.  Targeted at
     * stabilizer-coded Hamiltonians: lattice gauge theories
     * (Z2 LGT Gauss-law operators), surface/toric codes, etc.  Set
     * @c warmstart_stab_paulis and @c warmstart_stab_num_gens on
     * the config struct when selecting this enum. */
    CA_MPS_WARMSTART_STABILIZER_SUBGROUP
} ca_mps_warmstart_t;

typedef struct {
    /** Outer iterations of the alternating loop. */
    int max_outer_iters;
    /** Trotter step size for the imaginary-time |phi>-update. */
    double imag_time_dtau;
    /** Number of Trotter sweeps per outer iteration.  Each sweep
     *  applies one full Pauli-term cycle of e^(-dtau * c_k * P_k). */
    int imag_time_steps_per_outer;
    /** Cap on greedy passes inside each outer iteration's D-update. */
    int clifford_passes_per_outer;
    /** Stop when the energy decreases by less than this between outer
     *  iterations. */
    double convergence_eps;
    /** Include 2-qubit Cliffords in the search (passed through to the
     *  inner Clifford-only routine). */
    int include_2q_gates;
    /** Try 2-gate composite moves in the inner Clifford search.  Passed
     *  through to ::moonlab_ca_mps_optimize_var_d_clifford_only.  Costly. */
    int composite_2gate;
    /** Initial Clifford basin for D (see ::ca_mps_warmstart_t). */
    ca_mps_warmstart_t warmstart;
    /** Generators of the stabilizer subgroup, used only when
     *  @c warmstart == ::CA_MPS_WARMSTART_STABILIZER_SUBGROUP.
     *  Flat row-major (warmstart_stab_num_gens, num_qubits) uint8
     *  array with the same Pauli-byte encoding as @c paulis on the
     *  Hamiltonian (0=I, 1=X, 2=Y, 3=Z).  The generators must
     *  pairwise commute and be linearly independent; otherwise
     *  ::moonlab_ca_mps_optimize_var_d_alternating returns
     *  ::CA_MPS_ERR_INVALID. */
    const uint8_t* warmstart_stab_paulis;
    /** Number of stabilizer generators (rows of @c warmstart_stab_paulis). */
    uint32_t       warmstart_stab_num_gens;
    /** Print one line per outer iteration. */
    int verbose;
} ca_mps_var_d_alt_config_t;

ca_mps_var_d_alt_config_t ca_mps_var_d_alt_config_default(void);

typedef struct {
    double initial_energy;
    double final_energy;
    double initial_phi_entropy;
    double final_phi_entropy;
    /** Total Clifford gates accepted across all outer iterations. */
    int total_gates_added;
    int outer_iterations;
    int converged;
} ca_mps_var_d_alt_result_t;

/**
 * @brief Alternating-optimization variational-D for CA-MPS.
 *
 * Drives @p state toward a low-energy CA-MPS approximation of the
 * ground state of @p hamiltonian by alternating two updates:
 *
 *   1. **|phi>-update** (imag-time): for each Pauli term @c P_k
 *      with coefficient @c c_k in H, applies @c exp(-dtau * c_k * P_k)
 *      to @c |psi>.  Internally each non-Clifford rotation is
 *      conjugated through @c D and pushed into @c |phi>; Clifford
 *      terms only rotate @c D (free).  Repeats for
 *      @c imag_time_steps_per_outer Trotter cycles, then renormalises.
 *
 *   2. **D-update** (greedy Clifford): runs
 *      ::moonlab_ca_mps_optimize_var_d_clifford_only at the current
 *      @c |phi> for up to @c clifford_passes_per_outer passes.  This
 *      finds a Clifford rotation that reduces @c <psi|H|psi> at
 *      fixed @c |phi>; subsequent imag-time steps then drive @c |phi>
 *      to the new D's natural target (which has lower @c S(|phi>)
 *      when @c D is well-aligned with the workload).
 *
 * Stops when an outer iteration reduces the energy by less than
 * @c convergence_eps.
 */
ca_mps_error_t moonlab_ca_mps_optimize_var_d_alternating(
    moonlab_ca_mps_t* state,
    const uint8_t* paulis,
    const double* coeffs,
    uint32_t num_terms,
    const ca_mps_var_d_alt_config_t* config,
    ca_mps_var_d_alt_result_t* result);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_CA_MPS_VAR_D_H */
