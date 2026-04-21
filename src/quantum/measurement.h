/**
 * @file measurement.h
 * @brief Quantum measurement operations
 *
 * Provides measurement operations for quantum states including
 * projective measurements, partial measurements, and statistics.
 *
 * @stability evolving
 * @since v0.1.2
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef QUANTUM_MEASUREMENT_H
#define QUANTUM_MEASUREMENT_H

#include "state.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// PROBABILITY COMPUTATION
// ============================================================================

/**
 * @brief Compute probability of measuring qubit in |1⟩ state
 */
double measurement_probability_one(const quantum_state_t* state, int qubit);

/**
 * @brief Compute probability of measuring qubit in |0⟩ state
 */
double measurement_probability_zero(const quantum_state_t* state, int qubit);

/**
 * @brief Compute all single-qubit probabilities of |1⟩
 */
void measurement_all_probabilities(const quantum_state_t* state, double* probabilities);

/**
 * @brief Compute full probability distribution
 */
void measurement_probability_distribution(const quantum_state_t* state,
                                          double* distribution);

// ============================================================================
// SINGLE QUBIT MEASUREMENT
// ============================================================================

/**
 * @brief Measure single qubit with state collapse
 *
 * @param state Quantum state (modified in-place)
 * @param qubit Qubit index (0-indexed)
 * @param random_value Random value in [0, 1) for outcome
 * @return Measurement result (0 or 1), -1 on error
 */
int measurement_single_qubit(quantum_state_t* state, int qubit, double random_value);

/**
 * @brief Measure single qubit without collapse
 */
int measurement_single_qubit_no_collapse(const quantum_state_t* state,
                                         int qubit, double random_value);

// ============================================================================
// MULTI-QUBIT MEASUREMENT
// ============================================================================

/**
 * @brief Measure all qubits with state collapse
 *
 * @param state Quantum state (collapses to basis state)
 * @param random_value Random value in [0, 1)
 * @return Measurement result as bit pattern
 */
uint64_t measurement_all_qubits(quantum_state_t* state, double random_value);

/**
 * @brief Measure subset of qubits
 *
 * @param state Quantum state (partially collapses)
 * @param qubits Array of qubit indices to measure
 * @param num_measure Number of qubits to measure
 * @param random_value Random value for sampling
 * @return Measurement result as bit pattern
 */
uint64_t measurement_partial(quantum_state_t* state, const int* qubits,
                             int num_measure, double random_value);

// ============================================================================
// EXPECTATION VALUES
// ============================================================================

/**
 * @brief Compute expectation value of Z operator
 * @return <Z> in [-1, 1]
 */
double measurement_expectation_z(const quantum_state_t* state, int qubit);

/**
 * @brief Compute expectation value of X operator
 * @return <X> in [-1, 1]
 */
double measurement_expectation_x(const quantum_state_t* state, int qubit);

/**
 * @brief Compute expectation value of Y operator
 * @return <Y> in [-1, 1]
 */
double measurement_expectation_y(const quantum_state_t* state, int qubit);

/**
 * @brief Compute ZZ correlation between two qubits
 * @return <Z_i Z_j> in [-1, 1]
 */
double measurement_correlation_zz(const quantum_state_t* state,
                                  int qubit1, int qubit2);

// ============================================================================
// SAMPLING
// ============================================================================

/**
 * @brief Sample measurement outcomes without collapse
 */
void measurement_sample(const quantum_state_t* state, uint64_t* outcomes,
                        int num_samples, const double* random_values);

/**
 * @brief Estimate probabilities from samples
 */
void measurement_estimate_probabilities(const uint64_t* samples, int num_samples,
                                        uint64_t state_dim, double* probabilities);

/* =========================================================== */
/* POVM and weak-measurement API  (v0.2)                        */
/* =========================================================== */

/**
 * @brief A generalized (POVM) measurement with @p num_outcomes Kraus
 *        operators @p K_i such that sum_i K_i^dag K_i = I.
 *
 * The operators are stored as flat row-major state_dim x state_dim
 * complex matrices; the caller owns the memory and guarantees
 * completeness.  The simulator re-checks completeness on the first
 * call (violations return non-zero without mutating the state).
 */
typedef struct {
    size_t num_outcomes;    /* length of the kraus_ops[] array */
    uint64_t state_dim;     /* = 2^num_qubits of the target state */
    /* kraus_ops[i] is a state_dim * state_dim complex matrix stored
     * row-major; kraus_ops has num_outcomes entries. */
    const complex_t *const *kraus_ops;
} povm_t;

/**
 * @brief Perform a POVM measurement.  Samples an outcome k with
 *        probability p_k = <psi| K_k^dag K_k |psi>, then collapses
 *        @p state to (K_k |psi>) / sqrt(p_k).
 *
 * @param state      mutable state vector.
 * @param povm       POVM operators; completeness is checked.
 * @param uniform    a uniform [0, 1) sample used to pick an outcome.
 * @param outcome_out receives the chosen outcome index (0..n-1).
 * @return QS_SUCCESS on success; QS_ERROR_INVALID_STATE on argument
 *         error; QS_ERROR_NOT_NORMALIZED if Kraus ops don't sum to I.
 */
qs_error_t measurement_povm(quantum_state_t *state,
                             const povm_t *povm,
                             double uniform,
                             size_t *outcome_out);

/**
 * @brief Compute only the POVM outcome probabilities without
 *        mutating the state.
 *
 * @param probs_out array of length povm->num_outcomes.
 */
qs_error_t measurement_povm_probabilities(const quantum_state_t *state,
                                           const povm_t *povm,
                                           double *probs_out);

/**
 * @brief Weak Z-measurement on @p qubit with interaction strength
 *        @p strength in [0, 1].  strength = 0 is a non-disturbing
 *        probe (no information, no back-action); strength = 1 is a
 *        projective Z measurement.
 *
 * Implemented as a two-outcome POVM with Kraus operators
 *   K_+ = cos(theta) P_0 + sin(theta) P_1
 *   K_- = sin(theta) P_0 + cos(theta) P_1
 * where theta = (pi / 4) * (1 - strength), so strength = 0 gives
 * theta = pi/4 (outcomes equally likely regardless of the state,
 * P_k |psi> = |psi>/sqrt(2)) and strength = 1 gives theta = 0
 * (projective).
 */
qs_error_t measurement_weak_z(quantum_state_t *state,
                               int qubit,
                               double strength,
                               double uniform,
                               int *outcome_out);

#ifdef __cplusplus
}
#endif

#endif /* QUANTUM_MEASUREMENT_H */
