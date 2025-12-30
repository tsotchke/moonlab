/**
 * @file collective_ops.h
 * @brief Collective quantum operations for distributed simulation
 *
 * Provides MPI-coordinated operations that require global state knowledge:
 * - Measurement (sampling, projection)
 * - Expectation values
 * - State fidelity
 * - Entropy calculations
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#ifndef COLLECTIVE_OPS_H
#define COLLECTIVE_OPS_H

#include <stdint.h>
#include <complex.h>
#include "state_partition.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ERROR CODES
// ============================================================================

typedef enum {
    COLLECTIVE_SUCCESS = 0,
    COLLECTIVE_ERROR_INVALID_ARG = -1,
    COLLECTIVE_ERROR_MPI = -2,
    COLLECTIVE_ERROR_ALLOC = -3,
    COLLECTIVE_ERROR_NOT_INITIALIZED = -4,
    COLLECTIVE_ERROR_INVALID_QUBIT = -5
} collective_error_t;

// ============================================================================
// MEASUREMENT TYPES
// ============================================================================

/**
 * @brief Measurement result structure
 */
typedef struct {
    uint64_t outcome;          /**< Measured basis state */
    double probability;         /**< Probability of this outcome */
    int measured_qubit;        /**< Which qubit was measured (-1 for all) */
    int collapsed;             /**< Whether state was collapsed */
} measurement_result_t;

/**
 * @brief Sampling configuration
 */
typedef struct {
    int collapse_state;        /**< Collapse state after measurement (default: 1) */
    uint64_t seed;             /**< Random seed (0 for system entropy) */
    int use_hardware_rng;      /**< Use hardware RNG if available */
} measurement_config_t;

/**
 * @brief Probability distribution
 */
typedef struct {
    double* probabilities;     /**< Array of probabilities */
    uint64_t count;            /**< Number of entries */
    uint64_t start_index;      /**< Global start index for this segment */
    int is_partial;            /**< 1 if only partial distribution */
} probability_distribution_t;

// ============================================================================
// MEASUREMENT OPERATIONS
// ============================================================================

/**
 * @brief Measure all qubits
 *
 * Samples from the full probability distribution and optionally collapses
 * the state to the measured basis state.
 *
 * @param state Partitioned quantum state
 * @param result Output measurement result
 * @param config Optional configuration (NULL for defaults)
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_measure_all(partitioned_state_t* state,
                                          measurement_result_t* result,
                                          const measurement_config_t* config);

/**
 * @brief Measure single qubit
 *
 * Projects onto |0⟩ or |1⟩ for the specified qubit.
 *
 * @param state Partitioned state
 * @param qubit Qubit index to measure
 * @param result Output result (outcome is 0 or 1)
 * @param config Optional configuration
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_measure_qubit(partitioned_state_t* state,
                                            uint32_t qubit,
                                            measurement_result_t* result,
                                            const measurement_config_t* config);

/**
 * @brief Measure multiple qubits
 *
 * @param state Partitioned state
 * @param qubits Array of qubit indices
 * @param num_qubits Number of qubits to measure
 * @param result Output result (outcome in order of qubits array)
 * @param config Optional configuration
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_measure_qubits(partitioned_state_t* state,
                                             const uint32_t* qubits,
                                             uint32_t num_qubits,
                                             measurement_result_t* result,
                                             const measurement_config_t* config);

/**
 * @brief Sample without collapse
 *
 * Generates a sample from the probability distribution without
 * modifying the quantum state.
 *
 * @param state Partitioned state
 * @param result Output result
 * @param config Optional configuration
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_sample(const partitioned_state_t* state,
                                     measurement_result_t* result,
                                     const measurement_config_t* config);

/**
 * @brief Generate multiple samples
 *
 * Efficient batched sampling for statistics.
 *
 * @param state Partitioned state
 * @param samples Output array of samples
 * @param num_samples Number of samples to generate
 * @param config Optional configuration
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_sample_many(const partitioned_state_t* state,
                                          uint64_t* samples,
                                          uint32_t num_samples,
                                          const measurement_config_t* config);

// ============================================================================
// PROBABILITY OPERATIONS
// ============================================================================

/**
 * @brief Compute full probability distribution
 *
 * Gathers |amplitude|^2 for all basis states.
 * Warning: Requires O(2^n) memory on root.
 *
 * @param state Partitioned state
 * @param probs Output probability array (only valid at root, size 2^n)
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_get_probabilities(const partitioned_state_t* state,
                                                double* probs);

/**
 * @brief Compute local probability distribution
 *
 * Returns probabilities for locally-owned basis states.
 *
 * @param state Partitioned state
 * @param dist Output distribution structure
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_get_local_probabilities(const partitioned_state_t* state,
                                                      probability_distribution_t* dist);

/**
 * @brief Get probability of specific outcome
 *
 * @param state Partitioned state
 * @param basis_state Target basis state
 * @param prob Output probability
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_get_probability(const partitioned_state_t* state,
                                              uint64_t basis_state,
                                              double* prob);

/**
 * @brief Get probability of measuring qubit as |1⟩
 *
 * @param state Partitioned state
 * @param qubit Qubit index
 * @param prob Output probability
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_get_qubit_probability(const partitioned_state_t* state,
                                                    uint32_t qubit,
                                                    double* prob);

/**
 * @brief Free probability distribution
 *
 * @param dist Distribution to free
 */
void collective_free_distribution(probability_distribution_t* dist);

// ============================================================================
// EXPECTATION VALUES
// ============================================================================

/**
 * @brief Compute expectation value of Pauli string
 *
 * Computes ⟨ψ|P|ψ⟩ for Pauli operator P (e.g., "XYZZ").
 *
 * @param state Partitioned state
 * @param pauli_string Pauli operators (X, Y, Z, I per qubit)
 * @param expectation Output expectation value
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_expectation_pauli(const partitioned_state_t* state,
                                                const char* pauli_string,
                                                double* expectation);

/**
 * @brief Compute expectation value of Z on single qubit
 *
 * @param state Partitioned state
 * @param qubit Qubit index
 * @param expectation Output expectation value
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_expectation_z(const partitioned_state_t* state,
                                            uint32_t qubit,
                                            double* expectation);

/**
 * @brief Compute expectation value of X on single qubit
 *
 * @param state Partitioned state
 * @param qubit Qubit index
 * @param expectation Output expectation value
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_expectation_x(const partitioned_state_t* state,
                                            uint32_t qubit,
                                            double* expectation);

/**
 * @brief Compute expectation value of Y on single qubit
 *
 * @param state Partitioned state
 * @param qubit Qubit index
 * @param expectation Output expectation value
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_expectation_y(const partitioned_state_t* state,
                                            uint32_t qubit,
                                            double* expectation);

/**
 * @brief Compute ZZ correlation
 *
 * Computes ⟨Z_i Z_j⟩ for two qubits.
 *
 * @param state Partitioned state
 * @param qubit_i First qubit
 * @param qubit_j Second qubit
 * @param correlation Output correlation value
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_correlation_zz(const partitioned_state_t* state,
                                             uint32_t qubit_i,
                                             uint32_t qubit_j,
                                             double* correlation);

// ============================================================================
// STATE ANALYSIS
// ============================================================================

/**
 * @brief Compute state fidelity
 *
 * Computes |⟨ψ|φ⟩|² between two states.
 *
 * @param state1 First state
 * @param state2 Second state
 * @param fidelity Output fidelity (0 to 1)
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_fidelity(const partitioned_state_t* state1,
                                       const partitioned_state_t* state2,
                                       double* fidelity);

/**
 * @brief Compute fidelity with basis state
 *
 * @param state Quantum state
 * @param basis_state Target basis state
 * @param fidelity Output fidelity
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_fidelity_basis(const partitioned_state_t* state,
                                             uint64_t basis_state,
                                             double* fidelity);

/**
 * @brief Compute von Neumann entropy
 *
 * Warning: Requires gathering full state.
 *
 * @param state Partitioned state
 * @param entropy Output entropy (in bits)
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_von_neumann_entropy(const partitioned_state_t* state,
                                                  double* entropy);

/**
 * @brief Compute entanglement entropy for bipartition
 *
 * Computes S(A) for partition A vs rest.
 *
 * @param state Partitioned state
 * @param subsystem_qubits Qubits in subsystem A
 * @param num_subsystem Number of qubits in A
 * @param entropy Output entanglement entropy
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_entanglement_entropy(const partitioned_state_t* state,
                                                   const uint32_t* subsystem_qubits,
                                                   uint32_t num_subsystem,
                                                   double* entropy);

/**
 * @brief Find most probable basis states
 *
 * Returns top-k basis states by probability.
 *
 * @param state Partitioned state
 * @param top_states Output array of basis states
 * @param top_probs Output array of probabilities
 * @param k Number of top states to find
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_top_k_states(const partitioned_state_t* state,
                                           uint64_t* top_states,
                                           double* top_probs,
                                           uint32_t k);

// ============================================================================
// QUANTUM RANDOM NUMBER GENERATION
// ============================================================================

/**
 * @brief Generate quantum random bits
 *
 * Uses measurement of superposition states for true randomness.
 *
 * @param state Partitioned state (will be modified)
 * @param bits Output buffer
 * @param num_bits Number of random bits to generate
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_qrng_bits(partitioned_state_t* state,
                                        uint8_t* bits,
                                        uint32_t num_bits);

/**
 * @brief Generate quantum random bytes
 *
 * @param state Partitioned state
 * @param bytes Output buffer
 * @param num_bytes Number of random bytes
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_qrng_bytes(partitioned_state_t* state,
                                         uint8_t* bytes,
                                         uint32_t num_bytes);

/**
 * @brief Generate quantum random double in [0, 1)
 *
 * @param state Partitioned state
 * @param value Output random value
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_qrng_uniform(partitioned_state_t* state,
                                           double* value);

// ============================================================================
// BELL TEST OPERATIONS
// ============================================================================

/**
 * @brief CHSH inequality test
 *
 * Computes CHSH value S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|
 * Classical limit: |S| ≤ 2, Quantum max: 2√2 ≈ 2.828
 *
 * @param state Partitioned Bell state (2 qubits minimum)
 * @param qubit_a First qubit (Alice)
 * @param qubit_b Second qubit (Bob)
 * @param chsh_value Output CHSH value
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_chsh_test(partitioned_state_t* state,
                                        uint32_t qubit_a,
                                        uint32_t qubit_b,
                                        double* chsh_value);

/**
 * @brief Create Bell state
 *
 * Creates |Φ+⟩ = (|00⟩ + |11⟩)/√2 on specified qubits.
 *
 * @param state Partitioned state
 * @param qubit_a First qubit
 * @param qubit_b Second qubit
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_create_bell_state(partitioned_state_t* state,
                                                uint32_t qubit_a,
                                                uint32_t qubit_b);

/**
 * @brief Create GHZ state
 *
 * Creates (|00...0⟩ + |11...1⟩)/√2 on specified qubits.
 *
 * @param state Partitioned state
 * @param qubits Array of qubit indices
 * @param num_qubits Number of qubits in GHZ state
 * @return COLLECTIVE_SUCCESS or error code
 */
collective_error_t collective_create_ghz_state(partitioned_state_t* state,
                                               const uint32_t* qubits,
                                               uint32_t num_qubits);

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * @brief Print state summary
 *
 * Prints non-zero amplitudes and probabilities.
 *
 * @param state Partitioned state
 * @param max_entries Maximum entries to print (0 for all)
 * @param threshold Minimum probability to show
 */
void collective_print_state(const partitioned_state_t* state,
                           uint32_t max_entries,
                           double threshold);

/**
 * @brief Verify state normalization
 *
 * @param state Partitioned state
 * @param tolerance Acceptable deviation from 1.0
 * @return 1 if normalized, 0 otherwise
 */
int collective_verify_normalized(const partitioned_state_t* state,
                                 double tolerance);

/**
 * @brief Get error string
 *
 * @param error Error code
 * @return Human-readable message
 */
const char* collective_error_string(collective_error_t error);

/**
 * @brief Create default measurement config
 *
 * @return Default configuration
 */
measurement_config_t collective_default_measurement_config(void);

#ifdef __cplusplus
}
#endif

#endif /* COLLECTIVE_OPS_H */
