/**
 * @file ca_mps.h
 * @brief Clifford-Assisted Matrix Product States (CA-MPS).
 *
 * A CA-MPS represents a quantum state as |psi> = C |phi> where C is a
 * Clifford unitary tracked by the Aaronson-Gottesman tableau from
 * src/backends/clifford/ and |phi> is a plain MPS from tn_state.
 *
 * Clifford gates update only the tableau (O(n) bit operations, no MPS
 * cost).  Non-Clifford gates -- single-qubit rotations, T-gates, arbitrary
 * two-qubit unitaries -- apply a Pauli-string rotation to the MPS, where
 * the Pauli string is the Clifford-conjugated image of the gate's
 * generator.  This converts a circuit with t T-gates from the plain-MPS
 * requirement of chi ~ 2^{S/2} to CA-MPS's chi ~ 2^{t/2}; the gain is an
 * exponential reduction in bond dimension for Clifford-dominated circuits.
 *
 * See docs/research/ca_mps.md for the full design: the gate-application
 * rules, the expectation formula <psi|H|psi> = <phi|C^\dagger H C|phi>, and
 * the sampling algorithm.
 *
 * @since v0.3.0
 */
#ifndef MOONLAB_CA_MPS_H
#define MOONLAB_CA_MPS_H

#include <complex.h>
#include <stdint.h>

#include "../../applications/moonlab_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct moonlab_ca_mps_t moonlab_ca_mps_t;

/* Keep the ABI return type exactly `int` across compilers.  A named enum is
 * not required to have the same type as int and GCC 15 correctly diagnoses a
 * conflict when this internal header and the stable ABI header are included
 * together. */
typedef int ca_mps_error_t;
enum {
    CA_MPS_SUCCESS = 0,
    CA_MPS_ERR_INVALID = -1,
    CA_MPS_ERR_QUBIT   = -2,
    CA_MPS_ERR_OOM     = -3,
    CA_MPS_ERR_BACKEND = -4,
};

/* ================================================================== */
/*  Lifecycle                                                         */
/* ================================================================== */

/**
 * @brief Allocate a CA-MPS in the |0...0> state.
 *
 * Tableau is initialized to identity (D = I).  MPS is a bond-dim-1
 * product state.
 *
 * @param num_qubits    Number of qubits, 1..100000.
 * @param max_bond_dim  MPS bond-dim cap used during non-Clifford gate
 *                       application.  32 is a reasonable default for
 *                       VQE / QAOA workloads.
 * @return A newly owned handle, or NULL on allocation failure / bad args.
 */
MOONLAB_API moonlab_ca_mps_t* moonlab_ca_mps_create(uint32_t num_qubits, uint32_t max_bond_dim);

/** Release all memory owned by @p s.  No-op on NULL. */
MOONLAB_API void moonlab_ca_mps_free(moonlab_ca_mps_t* s);

/** Deep-copy a CA-MPS. */
MOONLAB_API moonlab_ca_mps_t* moonlab_ca_mps_clone(const moonlab_ca_mps_t* s);

/* ================================================================== */
/*  Introspection                                                     */
/* ================================================================== */

MOONLAB_API uint32_t moonlab_ca_mps_num_qubits(const moonlab_ca_mps_t* s);
MOONLAB_API uint32_t moonlab_ca_mps_max_bond_dim(const moonlab_ca_mps_t* s);
MOONLAB_API uint32_t moonlab_ca_mps_current_bond_dim(const moonlab_ca_mps_t* s);

/**
 * @brief Maximum half-cut von Neumann entanglement entropy of the MPS factor
 *        |phi> across all bipartitions.
 *
 * This is the representation-independent measure of how entangled |phi>
 * is.  Unlike ::moonlab_ca_mps_current_bond_dim, it does not depend on
 * whether DMRG / TEBD has compressed the bonds back to their actual rank
 * after each operation -- the entropy is computed directly from the
 * Schmidt spectrum.
 *
 * Use this for benchmarks comparing CA-MPS to plain MPS: the entropy
 * is the right yardstick for "how compactly does the state representation
 * have to grow" while bond_dim only reflects the working storage.
 *
 * @param s  CA-MPS handle.  Must be non-NULL.
 * @return Max bipartite entanglement entropy in nats; 0 on a NULL or
 *         single-qubit state.
 */
MOONLAB_API double moonlab_ca_mps_max_half_cut_entropy(const moonlab_ca_mps_t* s);

/* ================================================================== */
/*  Clifford gates (tableau only, O(n) per gate)                       */
/* ================================================================== */

/* All Clifford gates below update only the Aaronson-Gottesman tableau;
 * the MPS factor |phi> is left untouched (cost: O(n) bit operations
 * per gate, no SVD). */

/** Hadamard gate on qubit @p q. */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_h   (moonlab_ca_mps_t* s, uint32_t q);
/** S = sqrt(Z), phase gate diag(1, i). */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_s   (moonlab_ca_mps_t* s, uint32_t q);
/** S^dagger = diag(1, -i). */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_sdag(moonlab_ca_mps_t* s, uint32_t q);
/** Pauli X (bit flip). */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_x   (moonlab_ca_mps_t* s, uint32_t q);
/** Pauli Y. */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_y   (moonlab_ca_mps_t* s, uint32_t q);
/** Pauli Z (phase flip). */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_z   (moonlab_ca_mps_t* s, uint32_t q);
/** Controlled-NOT, control = @p ctrl, target = @p targ. */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_cnot(moonlab_ca_mps_t* s, uint32_t ctrl, uint32_t targ);
/** Controlled-Z (symmetric in @p a / @p b). */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_cz  (moonlab_ca_mps_t* s, uint32_t a, uint32_t b);
/** SWAP (symmetric in @p a / @p b). */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_swap(moonlab_ca_mps_t* s, uint32_t a, uint32_t b);

/* ================================================================== */
/*  Non-Clifford gates (push into MPS as Pauli-string rotations)       */
/* ================================================================== */

/** R_P(theta) = exp(-i theta P / 2) following the standard Qiskit/Cirq
 *  convention.  Non-Clifford in general: the MPS action is a Pauli-string
 *  rotation on the Clifford-conjugated string C^dagger P_q C. */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_rx(moonlab_ca_mps_t* s, uint32_t q, double theta);
MOONLAB_API ca_mps_error_t moonlab_ca_mps_ry(moonlab_ca_mps_t* s, uint32_t q, double theta);
MOONLAB_API ca_mps_error_t moonlab_ca_mps_rz(moonlab_ca_mps_t* s, uint32_t q, double theta);

/** T gate: equals R_Z(pi/4) up to a global phase e^{-i pi/8}. */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_t_gate(moonlab_ca_mps_t* s, uint32_t q);

/** T-dagger gate: equals R_Z(-pi/4) up to a global phase e^{+i pi/8}. */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_t_dagger(moonlab_ca_mps_t* s, uint32_t q);

/** Phase gate: P(theta) = diag(1, e^{i theta}); equals R_Z(theta) up to a
 *  global phase e^{i theta / 2}. */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_phase(moonlab_ca_mps_t* s, uint32_t q, double theta);

/** Controlled-R_Z(theta).  Decomposed as
 *    R_Z(target, theta/2) . CNOT . R_Z(target, -theta/2) . CNOT
 *  using only existing CA-MPS primitives. */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_crz(moonlab_ca_mps_t* s,
                                   uint32_t control, uint32_t target,
                                   double theta);

/** Controlled-R_X(theta).  Decomposed as H_t . CRZ . H_t. */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_crx(moonlab_ca_mps_t* s,
                                   uint32_t control, uint32_t target,
                                   double theta);

/** Controlled-R_Y(theta).  Decomposed as S_t . CRX . S^dag_t
 *  (since S X S^dag = Y). */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_cry(moonlab_ca_mps_t* s,
                                   uint32_t control, uint32_t target,
                                   double theta);

/** General single-qubit unitary U3(theta, phi, lambda) (Qiskit convention):
 *  [[cos(t/2),     -e^{i l} sin(t/2)],
 *   [e^{i p} sin(t/2),  e^{i(p+l)} cos(t/2)]]
 *  Equivalent up to a global phase to R_Z(p) . R_Y(t) . R_Z(l). */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_u3(moonlab_ca_mps_t* s, uint32_t q,
                                  double theta, double phi, double lambda);

/** Toffoli (CCX): flip target if both controls are |1>.  Decomposed via
 *  the Nielsen-Chuang 6-CNOT + 7-T construction so the operation is
 *  expressed purely in terms of CA-MPS Clifford gates plus T / T-dagger.
 *  All three qubit indices must be distinct. */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_toffoli(moonlab_ca_mps_t* s,
                                       uint32_t c1, uint32_t c2, uint32_t t);

/** Fredkin (CSWAP): swap @p t1 and @p t2 if the control is |1>.  Built
 *  from CSWAP = CNOT(t1,t2) . Toffoli(c, t2, t1) . CNOT(t1, t2).  All
 *  three qubit indices must be distinct. */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_fredkin(moonlab_ca_mps_t* s,
                                       uint32_t c, uint32_t t1, uint32_t t2);

/**
 * @brief Apply exp(i theta P) for an n-qubit Pauli string P.
 *
 * @param pauli_string Array of n bytes in {0=I, 1=X, 2=Y, 3=Z}.
 * @param theta        Rotation angle (radians).
 */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_pauli_rotation(moonlab_ca_mps_t* s,
                                             const uint8_t* pauli_string,
                                             double theta);

/**
 * @brief Apply exp(-tau P) for an n-qubit Pauli string P (non-unitary).
 *
 * This is the imaginary-time step primitive.  The operator is
 *     exp(-tau P) = cosh(tau) I - sinh(tau) P
 * which is non-unitary (for tau != 0); the caller is responsible for
 * renormalizing the state via @c moonlab_ca_mps_normalize when needed.
 *
 * @param pauli_string Array of n bytes in {0=I, 1=X, 2=Y, 3=Z}.
 * @param tau          Imaginary-time step.  Positive tau pushes the
 *                     state toward the lowest-eigenvalue sector of P.
 */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_imag_pauli_rotation(moonlab_ca_mps_t* s,
                                                  const uint8_t* pauli_string,
                                                  double tau);

/** Rescale the internal MPS to unit norm. */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_normalize(moonlab_ca_mps_t* s);

/** Return <psi|psi> (should be 1 for a normalized state). */
MOONLAB_API double moonlab_ca_mps_norm(const moonlab_ca_mps_t* s);

/* ================================================================== */
/*  Observables                                                       */
/* ================================================================== */

/**
 * @brief Compute <psi | P | psi> for a Pauli string P.
 *
 * The Clifford prefactor is absorbed via <psi|P|psi> =
 * <phi | C^dagger P C | phi>.  The conjugated Pauli string is computed in
 * O(n^2) and the MPS expectation in O(n chi^2).
 */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_expect_pauli(const moonlab_ca_mps_t* s,
                                           const uint8_t* pauli_string,
                                           double _Complex* out_expval);

/**
 * @brief Compute <psi | H | psi> for a Hamiltonian H expressed as a sum of
 *        Pauli strings: H = sum_k coeffs[k] * paulis[k].
 *
 * Each coefficient is complex (accepts Hermitian sums directly; non-
 * Hermitian H is permitted but the returned expectation may be complex).
 *
 * @param paulis   Array of Pauli strings, shape [num_terms][num_qubits].
 * @param coeffs   Array of complex coefficients, length num_terms.
 * @param num_terms Number of Pauli-string terms.
 * @param out_expval Returned expectation value.
 *
 * Cost: O(num_terms * n^2) for the n Clifford conjugations plus
 * O(num_terms * n * chi^2) for the MPS expectation on each conjugated
 * term.  Trivially parallelizable across terms.
 */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_expect_pauli_sum(const moonlab_ca_mps_t* s,
                                                const uint8_t* paulis,
                                                const double _Complex* coeffs,
                                                uint32_t num_terms,
                                                double _Complex* out_expval);

/**
 * @brief Compute @f$Q = C^\dagger P C@f$ for the current Clifford @f$C@f$ in @p s.
 *
 * Exposes the Clifford-conjugated Pauli string and accumulated phase so that
 * higher-level routines (var-D delta-caching, MPDO + CA-MPS bridging) can
 * inspect Q's support without recomputing it.  @p out_pauli must be at least
 * @c moonlab_ca_mps_num_qubits(s) bytes.  @p out_phase encodes the
 * accumulated @f$i^{phase}@f$ factor in {0,1,2,3}.
 *
 * Cost: O(n^2) Heisenberg conjugation through the tableau.
 */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_conjugate_pauli(const moonlab_ca_mps_t* s,
                                                          const uint8_t* in_pauli,
                                                          uint8_t* out_pauli,
                                                          int* out_phase);

/**
 * @brief Marginal probability of measuring Z = +1 on a single qubit.
 *
 * Returns P(Z_q = +1) = (1 + <psi|Z_q|psi>) / 2, in [0, 1] up to imag-noise
 * tolerance.  Marginal only -- ignores correlations with other qubits.  For
 * correlated multi-qubit sampling use @ref moonlab_ca_mps_sample_z (since
 * v0.10.0).
 *
 * Cost: O(n^2 + n chi^2), same as a single Pauli-string expectation.
 */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_prob_z(const moonlab_ca_mps_t* s,
                                      uint32_t qubit,
                                      double* out_prob);

/**
 * @brief Sequential Born-rule sampling of computational-basis bitstrings.
 *
 * Draws @p num_samples independent bitstrings from the distribution
 * |<x|psi>|^2 where |psi> = C|phi>.  For each sample the algorithm walks
 * qubits left to right; at qubit i it computes the conditional Pauli
 * @c g_i = C^dagger Z_i C, evaluates m_i = <phi|g_i|phi>, samples
 * @c v_i in {+1, -1} with P(v_i = +1) = (1 + m_i)/2, and projects |phi>
 * onto the v_i-eigenspace of g_i.  The Clifford layer remains unchanged.
 *
 * @param s              CA-MPS state to sample from.
 * @param num_samples    Number of bitstrings to draw.
 * @param random_values  Flat array of @p num_samples * @c n uniforms in
 *                       [0,1).  Caller supplies the RNG (cryptographic,
 *                       moonlab_qrng, deterministic seed, ...).
 * @param out_bits       Output buffer of @p num_samples * @c n bytes;
 *                       byte (s*n + i) is the i-th bit of the s-th
 *                       sample, in {0, 1}.
 *
 * Cost per sample: O(n * (n^2 + chi^2 n + chi^3)) where chi is the
 * current MPS bond dimension.  Each projection step inflates chi by 2x
 * before SVD-truncation back to @c max_bond_dim.
 *
 * @return CA_MPS_SUCCESS or a negative error code.
 */
MOONLAB_API ca_mps_error_t moonlab_ca_mps_sample_z(const moonlab_ca_mps_t* s,
                                       uint32_t num_samples,
                                       const double* random_values,
                                       uint8_t* out_bits);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_CA_MPS_H */
