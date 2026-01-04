/**
 * @file entanglement.h
 * @brief Quantum entanglement analysis and utilities
 *
 * @stability stable
 * @since v1.0.0
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
