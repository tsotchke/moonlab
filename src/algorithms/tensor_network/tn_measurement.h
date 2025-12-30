/**
 * @file tn_measurement.h
 * @brief Measurement operations for tensor network quantum states
 *
 * Implements quantum measurement on MPS states:
 * - Single-qubit projective measurements
 * - Multi-qubit measurements
 * - Expectation values
 * - Sampling from probability distribution
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TN_MEASUREMENT_H
#define TN_MEASUREMENT_H

#include "tn_state.h"
#include "tn_gates.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ERROR CODES
// ============================================================================

typedef enum {
    TN_MEASURE_SUCCESS = 0,
    TN_MEASURE_ERROR_NULL_PTR = -1,
    TN_MEASURE_ERROR_INVALID_QUBIT = -2,
    TN_MEASURE_ERROR_ALLOC_FAILED = -3,
    TN_MEASURE_ERROR_CONTRACTION_FAILED = -4,
    TN_MEASURE_ERROR_NORMALIZATION = -5,
    TN_MEASURE_ERROR_INVALID_OPERATOR = -6,
    TN_MEASURE_ERROR_RNG_FAILED = -7
} tn_measure_error_t;

// ============================================================================
// MEASUREMENT RESULTS
// ============================================================================

/**
 * @brief Result of single measurement
 */
typedef struct {
    int outcome;                    /**< Measurement outcome (0 or 1) */
    double probability;             /**< Probability of this outcome */
} tn_measure_result_t;

/**
 * @brief Result of multi-qubit measurement
 */
typedef struct {
    uint64_t bitstring;             /**< Measurement outcome as bitstring */
    double probability;             /**< Probability of this outcome */
    uint32_t num_qubits;            /**< Number of qubits measured */
} tn_measure_multi_result_t;

/**
 * @brief Sampling statistics
 */
typedef struct {
    uint64_t num_samples;           /**< Total samples taken */
    double total_time_seconds;      /**< Total sampling time */
    double avg_sample_time;         /**< Average time per sample */
    double entropy_estimate;        /**< Estimated entropy from samples */
} tn_sample_stats_t;

// ============================================================================
// SINGLE-QUBIT MEASUREMENT
// ============================================================================

/**
 * @brief Compute probability of measuring qubit in |0> or |1>
 *
 * Does not modify state.
 *
 * @param state MPS state
 * @param qubit Qubit to measure
 * @param prob_0 Output: probability of |0>
 * @param prob_1 Output: probability of |1>
 * @return TN_MEASURE_SUCCESS or error code
 */
tn_measure_error_t tn_measure_probability(const tn_mps_state_t *state,
                                           uint32_t qubit,
                                           double *prob_0, double *prob_1);

/**
 * @brief Perform projective measurement on single qubit
 *
 * Collapses state and returns measurement outcome.
 * Uses provided random number for reproducibility.
 *
 * @param state MPS state (modified in place)
 * @param qubit Qubit to measure
 * @param random_value Random value in [0, 1) for outcome selection
 * @param result Output: measurement result
 * @return TN_MEASURE_SUCCESS or error code
 */
tn_measure_error_t tn_measure_single(tn_mps_state_t *state,
                                      uint32_t qubit,
                                      double random_value,
                                      tn_measure_result_t *result);

/**
 * @brief Perform projective measurement with specific outcome
 *
 * Projects state onto specified outcome and renormalizes.
 *
 * @param state MPS state (modified in place)
 * @param qubit Qubit to measure
 * @param outcome Desired outcome (0 or 1)
 * @param probability Output: probability of this outcome
 * @return TN_MEASURE_SUCCESS or error code
 */
tn_measure_error_t tn_measure_project(tn_mps_state_t *state,
                                       uint32_t qubit,
                                       int outcome,
                                       double *probability);

// ============================================================================
// MULTI-QUBIT MEASUREMENT
// ============================================================================

/**
 * @brief Measure all qubits
 *
 * Performs sequential measurement from qubit 0 to n-1.
 *
 * @param state MPS state (modified in place)
 * @param random_values Array of random values [num_qubits] in [0, 1)
 * @param result Output: measurement result
 * @return TN_MEASURE_SUCCESS or error code
 */
tn_measure_error_t tn_measure_all(tn_mps_state_t *state,
                                   const double *random_values,
                                   tn_measure_multi_result_t *result);

/**
 * @brief Measure subset of qubits
 *
 * @param state MPS state (modified in place)
 * @param qubits Array of qubit indices to measure
 * @param num_qubits Number of qubits to measure
 * @param random_values Array of random values [num_qubits]
 * @param result Output: measurement result
 * @return TN_MEASURE_SUCCESS or error code
 */
tn_measure_error_t tn_measure_subset(tn_mps_state_t *state,
                                      const uint32_t *qubits,
                                      uint32_t num_qubits,
                                      const double *random_values,
                                      tn_measure_multi_result_t *result);

/**
 * @brief Compute probability of specific bitstring outcome
 *
 * @param state MPS state (not modified)
 * @param bitstring Target outcome
 * @return Probability of outcome
 */
double tn_measure_bitstring_probability(const tn_mps_state_t *state,
                                         uint64_t bitstring);

// ============================================================================
// SAMPLING
// ============================================================================

/**
 * @brief Sample multiple bitstrings from state
 *
 * Efficiently samples from probability distribution without
 * fully collapsing state. Uses perfect sampling algorithm.
 *
 * @param state MPS state (not modified)
 * @param num_samples Number of samples to generate
 * @param samples Output array for samples [num_samples]
 * @param random_values Random values [num_samples * num_qubits]
 * @param stats Output: sampling statistics (can be NULL)
 * @return TN_MEASURE_SUCCESS or error code
 */
tn_measure_error_t tn_sample_bitstrings(const tn_mps_state_t *state,
                                         uint32_t num_samples,
                                         uint64_t *samples,
                                         const double *random_values,
                                         tn_sample_stats_t *stats);

/**
 * @brief Sample with automatic random number generation
 *
 * Uses system random source or provided seed.
 *
 * @param state MPS state (not modified)
 * @param num_samples Number of samples
 * @param samples Output array
 * @param seed Random seed (0 for time-based)
 * @param stats Output statistics (can be NULL)
 * @return TN_MEASURE_SUCCESS or error code
 */
tn_measure_error_t tn_sample_auto(const tn_mps_state_t *state,
                                   uint32_t num_samples,
                                   uint64_t *samples,
                                   uint64_t seed,
                                   tn_sample_stats_t *stats);

// ============================================================================
// EXPECTATION VALUES
// ============================================================================

/**
 * @brief Compute expectation value of Pauli Z on single qubit
 *
 * <Z_i> = <psi| Z_i |psi>
 *
 * @param state MPS state
 * @param qubit Qubit index
 * @return Expectation value
 */
double tn_expectation_z(const tn_mps_state_t *state, uint32_t qubit);

/**
 * @brief Compute expectation value of Pauli X on single qubit
 *
 * @param state MPS state
 * @param qubit Qubit index
 * @return Expectation value
 */
double tn_expectation_x(const tn_mps_state_t *state, uint32_t qubit);

/**
 * @brief Compute expectation value of Pauli Y on single qubit
 *
 * @param state MPS state
 * @param qubit Qubit index
 * @return Expectation value
 */
double tn_expectation_y(const tn_mps_state_t *state, uint32_t qubit);

/**
 * @brief Compute expectation value of single-qubit operator
 *
 * @param state MPS state
 * @param qubit Qubit index
 * @param op Operator (2x2 matrix)
 * @return Expectation value
 */
double complex tn_expectation_1q(const tn_mps_state_t *state,
                                  uint32_t qubit,
                                  const tn_gate_1q_t *op);

/**
 * @brief Compute expectation value of ZZ correlation
 *
 * <Z_i Z_j> = <psi| Z_i Z_j |psi>
 *
 * @param state MPS state
 * @param qubit1 First qubit
 * @param qubit2 Second qubit
 * @return Expectation value
 */
double tn_expectation_zz(const tn_mps_state_t *state,
                          uint32_t qubit1, uint32_t qubit2);

/**
 * @brief Fast ⟨Z⟩ measurement using canonical form (O(chi^2))
 *
 * Automatically mixed-canonicalizes to the target qubit and computes
 * the local expectation value. Much faster than the general method
 * for large systems.
 *
 * NOTE: This modifies the MPS state (changes canonical form).
 *
 * @param state MPS state (will be canonicalized)
 * @param qubit Qubit to measure
 * @return Expectation value ⟨Z⟩
 */
double tn_expectation_z_fast(tn_mps_state_t *state, uint32_t qubit);

/**
 * @brief Fast ⟨ZZ⟩ measurement for adjacent qubits (O(chi^3))
 *
 * Automatically mixed-canonicalizes and computes the local expectation.
 * Only works for adjacent qubits (qubit2 = qubit1 + 1).
 *
 * NOTE: This modifies the MPS state (changes canonical form).
 *
 * @param state MPS state (will be canonicalized)
 * @param qubit1 First qubit
 * @param qubit2 Second qubit (must be qubit1 + 1)
 * @return Expectation value ⟨ZZ⟩
 */
double tn_expectation_zz_fast(tn_mps_state_t *state, uint32_t qubit1, uint32_t qubit2);

/**
 * @brief Compute average ⟨Z⟩ over all qubits efficiently (O(n × chi^3))
 *
 * Uses a single sweep through the chain, moving the orthogonality center
 * and computing local expectations. Much faster than calling tn_expectation_z
 * in a loop for large systems.
 *
 * @param state MPS state (will be canonicalized)
 * @return Average magnetization Σ⟨Z⟩/n
 */
double tn_magnetization_fast(tn_mps_state_t *state);

/**
 * @brief Compute average nearest-neighbor ⟨ZZ⟩ efficiently (O(n × chi^3))
 *
 * Uses a single sweep through the chain for all adjacent pair correlations.
 *
 * @param state MPS state (will be canonicalized)
 * @return Average correlation Σ⟨ZZ⟩/(n-1)
 */
double tn_zz_correlation_fast(tn_mps_state_t *state);

/**
 * @brief Compute expectation value of two-qubit operator
 *
 * @param state MPS state
 * @param qubit1 First qubit
 * @param qubit2 Second qubit
 * @param op Operator (4x4 matrix)
 * @return Expectation value
 */
double complex tn_expectation_2q(const tn_mps_state_t *state,
                                  uint32_t qubit1, uint32_t qubit2,
                                  const tn_gate_2q_t *op);

/**
 * @brief Compute expectation value of Pauli string
 *
 * Pauli string encoded as array: 0=I, 1=X, 2=Y, 3=Z
 *
 * @param state MPS state
 * @param paulis Array of Pauli indices [num_qubits]
 * @return Expectation value
 */
double complex tn_expectation_pauli_string(const tn_mps_state_t *state,
                                            const uint8_t *paulis);

// ============================================================================
// REDUCED DENSITY MATRIX
// ============================================================================

/**
 * @brief Compute single-qubit reduced density matrix
 *
 * rho_i = Tr_{j!=i} |psi><psi|
 *
 * @param state MPS state
 * @param qubit Qubit index
 * @param rho Output: 2x2 density matrix (4 elements, row-major)
 * @return TN_MEASURE_SUCCESS or error code
 */
tn_measure_error_t tn_reduced_density_1q(const tn_mps_state_t *state,
                                          uint32_t qubit,
                                          double complex *rho);

/**
 * @brief Compute two-qubit reduced density matrix
 *
 * @param state MPS state
 * @param qubit1 First qubit
 * @param qubit2 Second qubit
 * @param rho Output: 4x4 density matrix (16 elements)
 * @return TN_MEASURE_SUCCESS or error code
 */
tn_measure_error_t tn_reduced_density_2q(const tn_mps_state_t *state,
                                          uint32_t qubit1, uint32_t qubit2,
                                          double complex *rho);

/**
 * @brief Compute von Neumann entropy of single-qubit reduced density matrix
 *
 * S = -Tr(rho log rho)
 *
 * @param state MPS state
 * @param qubit Qubit index
 * @return von Neumann entropy
 */
double tn_local_entropy(const tn_mps_state_t *state, uint32_t qubit);

// ============================================================================
// HISTOGRAM
// ============================================================================

/**
 * @brief Histogram of measurement outcomes
 */
typedef struct {
    uint64_t *outcomes;             /**< Array of distinct outcomes */
    uint64_t *counts;               /**< Count for each outcome */
    uint32_t num_outcomes;          /**< Number of distinct outcomes */
    uint32_t capacity;              /**< Allocated capacity */
    uint64_t total_samples;         /**< Total number of samples */
} tn_histogram_t;

/**
 * @brief Create histogram from samples
 *
 * @param samples Array of sample bitstrings
 * @param num_samples Number of samples
 * @return Histogram or NULL on failure
 */
tn_histogram_t *tn_histogram_create(const uint64_t *samples, uint32_t num_samples);

/**
 * @brief Get probability of outcome from histogram
 *
 * @param hist Histogram
 * @param outcome Outcome to query
 * @return Empirical probability
 */
double tn_histogram_probability(const tn_histogram_t *hist, uint64_t outcome);

/**
 * @brief Get most probable outcomes
 *
 * @param hist Histogram
 * @param outcomes Output array for outcomes
 * @param probabilities Output array for probabilities
 * @param max_outcomes Maximum outcomes to return
 * @return Number of outcomes returned
 */
uint32_t tn_histogram_top_outcomes(const tn_histogram_t *hist,
                                    uint64_t *outcomes,
                                    double *probabilities,
                                    uint32_t max_outcomes);

/**
 * @brief Print histogram summary
 *
 * @param hist Histogram
 * @param max_show Maximum outcomes to show
 */
void tn_histogram_print(const tn_histogram_t *hist, uint32_t max_show);

/**
 * @brief Free histogram
 *
 * @param hist Histogram to free
 */
void tn_histogram_free(tn_histogram_t *hist);

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * @brief Get error string
 *
 * @param error Error code
 * @return Human-readable error string
 */
const char *tn_measure_error_string(tn_measure_error_t error);

/**
 * @brief Convert bitstring to binary string representation
 *
 * @param bitstring Bitstring value
 * @param num_qubits Number of qubits
 * @param buffer Output buffer (at least num_qubits + 1 bytes)
 */
void tn_bitstring_to_str(uint64_t bitstring, uint32_t num_qubits, char *buffer);

#ifdef __cplusplus
}
#endif

#endif /* TN_MEASUREMENT_H */
