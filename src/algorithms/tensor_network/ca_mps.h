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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct moonlab_ca_mps_t moonlab_ca_mps_t;

typedef enum {
    CA_MPS_SUCCESS = 0,
    CA_MPS_ERR_INVALID = -1,
    CA_MPS_ERR_QUBIT   = -2,
    CA_MPS_ERR_OOM     = -3,
    CA_MPS_ERR_BACKEND = -4,
} ca_mps_error_t;

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
moonlab_ca_mps_t* moonlab_ca_mps_create(uint32_t num_qubits, uint32_t max_bond_dim);

/** Release all memory owned by @p s.  No-op on NULL. */
void moonlab_ca_mps_free(moonlab_ca_mps_t* s);

/** Deep-copy a CA-MPS. */
moonlab_ca_mps_t* moonlab_ca_mps_clone(const moonlab_ca_mps_t* s);

/* ================================================================== */
/*  Introspection                                                     */
/* ================================================================== */

uint32_t moonlab_ca_mps_num_qubits(const moonlab_ca_mps_t* s);
uint32_t moonlab_ca_mps_max_bond_dim(const moonlab_ca_mps_t* s);
uint32_t moonlab_ca_mps_current_bond_dim(const moonlab_ca_mps_t* s);

/* ================================================================== */
/*  Clifford gates (tableau only, O(n) per gate)                       */
/* ================================================================== */

ca_mps_error_t moonlab_ca_mps_h   (moonlab_ca_mps_t* s, uint32_t q);
ca_mps_error_t moonlab_ca_mps_s   (moonlab_ca_mps_t* s, uint32_t q);
ca_mps_error_t moonlab_ca_mps_sdag(moonlab_ca_mps_t* s, uint32_t q);
ca_mps_error_t moonlab_ca_mps_x   (moonlab_ca_mps_t* s, uint32_t q);
ca_mps_error_t moonlab_ca_mps_y   (moonlab_ca_mps_t* s, uint32_t q);
ca_mps_error_t moonlab_ca_mps_z   (moonlab_ca_mps_t* s, uint32_t q);
ca_mps_error_t moonlab_ca_mps_cnot(moonlab_ca_mps_t* s, uint32_t ctrl, uint32_t targ);
ca_mps_error_t moonlab_ca_mps_cz  (moonlab_ca_mps_t* s, uint32_t a, uint32_t b);
ca_mps_error_t moonlab_ca_mps_swap(moonlab_ca_mps_t* s, uint32_t a, uint32_t b);

/* ================================================================== */
/*  Non-Clifford gates (push into MPS as Pauli-string rotations)       */
/* ================================================================== */

/** R_P(theta) = exp(-i theta P / 2) following the standard Qiskit/Cirq
 *  convention.  Non-Clifford in general: the MPS action is a Pauli-string
 *  rotation on the Clifford-conjugated string C^dagger P_q C. */
ca_mps_error_t moonlab_ca_mps_rx(moonlab_ca_mps_t* s, uint32_t q, double theta);
ca_mps_error_t moonlab_ca_mps_ry(moonlab_ca_mps_t* s, uint32_t q, double theta);
ca_mps_error_t moonlab_ca_mps_rz(moonlab_ca_mps_t* s, uint32_t q, double theta);

/** T gate: equals R_Z(pi/4) up to a global phase e^{-i pi/8}. */
ca_mps_error_t moonlab_ca_mps_t_gate(moonlab_ca_mps_t* s, uint32_t q);

/** T-dagger gate: equals R_Z(-pi/4) up to a global phase e^{+i pi/8}. */
ca_mps_error_t moonlab_ca_mps_t_dagger(moonlab_ca_mps_t* s, uint32_t q);

/** Phase gate: P(theta) = diag(1, e^{i theta}); equals R_Z(theta) up to a
 *  global phase e^{i theta / 2}. */
ca_mps_error_t moonlab_ca_mps_phase(moonlab_ca_mps_t* s, uint32_t q, double theta);

/** Controlled-R_Z(theta).  Decomposed as
 *    R_Z(target, theta/2) . CNOT . R_Z(target, -theta/2) . CNOT
 *  using only existing CA-MPS primitives. */
ca_mps_error_t moonlab_ca_mps_crz(moonlab_ca_mps_t* s,
                                   uint32_t control, uint32_t target,
                                   double theta);

/** Controlled-R_X(theta).  Decomposed as H_t . CRZ . H_t. */
ca_mps_error_t moonlab_ca_mps_crx(moonlab_ca_mps_t* s,
                                   uint32_t control, uint32_t target,
                                   double theta);

/** Controlled-R_Y(theta).  Decomposed as S_t . CRX . S^dag_t
 *  (since S X S^dag = Y). */
ca_mps_error_t moonlab_ca_mps_cry(moonlab_ca_mps_t* s,
                                   uint32_t control, uint32_t target,
                                   double theta);

/**
 * @brief Apply exp(i theta P) for an n-qubit Pauli string P.
 *
 * @param pauli_string Array of n bytes in {0=I, 1=X, 2=Y, 3=Z}.
 * @param theta        Rotation angle (radians).
 */
ca_mps_error_t moonlab_ca_mps_pauli_rotation(moonlab_ca_mps_t* s,
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
ca_mps_error_t moonlab_ca_mps_imag_pauli_rotation(moonlab_ca_mps_t* s,
                                                  const uint8_t* pauli_string,
                                                  double tau);

/** Rescale the internal MPS to unit norm. */
ca_mps_error_t moonlab_ca_mps_normalize(moonlab_ca_mps_t* s);

/** Return <psi|psi> (should be 1 for a normalized state). */
double moonlab_ca_mps_norm(const moonlab_ca_mps_t* s);

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
ca_mps_error_t moonlab_ca_mps_expect_pauli(const moonlab_ca_mps_t* s,
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
ca_mps_error_t moonlab_ca_mps_expect_pauli_sum(const moonlab_ca_mps_t* s,
                                                const uint8_t* paulis,
                                                const double _Complex* coeffs,
                                                uint32_t num_terms,
                                                double _Complex* out_expval);

/**
 * @brief Marginal probability of measuring Z = +1 on a single qubit.
 *
 * Returns P(Z_q = +1) = (1 + <psi|Z_q|psi>) / 2, in [0, 1] up to imag-noise
 * tolerance.  Marginal only -- ignores correlations with other qubits.  For
 * correlated multi-qubit sampling use the Pauli-rotation MPO + sequential
 * Born-rule sampling layer (not yet implemented).
 *
 * Cost: O(n^2 + n chi^2), same as a single Pauli-string expectation.
 */
ca_mps_error_t moonlab_ca_mps_prob_z(const moonlab_ca_mps_t* s,
                                      uint32_t qubit,
                                      double* out_prob);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_CA_MPS_H */
