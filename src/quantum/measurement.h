/**
 * @file measurement.h
 * @brief Quantum measurement operations
 *
 * Provides measurement operations for quantum states including
 * projective measurements, partial measurements, and statistics.
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
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

#ifdef __cplusplus
}
#endif

#endif /* QUANTUM_MEASUREMENT_H */
