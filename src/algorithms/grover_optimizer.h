/**
 * @file grover_optimizer.h
 * @brief Optimized Grover's algorithm implementations
 *
 * Provides optimized versions of Grover's search algorithm:
 * - Parallel oracle evaluation
 * - Batched amplitude updates
 * - SIMD-accelerated diffusion
 * - Adaptive iteration count
 * - Multi-target search
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef ALGORITHMS_GROVER_OPTIMIZER_H
#define ALGORITHMS_GROVER_OPTIMIZER_H

#include "../quantum/state.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// OPTIMAL ITERATION COUNT
// ============================================================================

/**
 * @brief Compute optimal number of Grover iterations
 *
 * For N items with M targets:
 *   k_opt ≈ (π/4) * √(N/M)
 *
 * @param num_qubits Number of qubits (N = 2^n)
 * @param num_targets Number of marked items M
 * @return Optimal iteration count
 */
int grover_optimal_iterations_multi(int num_qubits, int num_targets);

/**
 * @brief Compute success probability after k iterations
 *
 * P(success) = sin²((2k+1)θ) where sin²(θ) = M/N
 *
 * @param num_qubits Number of qubits
 * @param num_targets Number of targets
 * @param iterations Number of iterations
 * @return Success probability
 */
double grover_success_probability(int num_qubits, int num_targets, int iterations);

// ============================================================================
// ORACLE OPERATIONS
// ============================================================================

/**
 * @brief Apply oracle for multiple targets
 *
 * Uses sorted target list for cache-friendly access
 *
 * @param amplitudes State amplitudes
 * @param targets Array of target indices
 * @param num_targets Number of targets
 */
void grover_oracle_multi_target(complex_t* amplitudes,
                                const uint64_t* targets,
                                int num_targets);

/**
 * @brief Apply oracle using function pointer
 *
 * @param amplitudes State amplitudes
 * @param state_dim Dimension of state
 * @param oracle_func Function that returns 1 for marked states
 * @param oracle_data User data for oracle
 */
void grover_oracle_function(complex_t* amplitudes, uint64_t state_dim,
                            int (*oracle_func)(uint64_t, void*),
                            void* oracle_data);

// ============================================================================
// DIFFUSION OPERATORS
// ============================================================================

/**
 * @brief Apply Grover diffusion operator (optimized)
 *
 * D = 2|s⟩⟨s| - I where |s⟩ is uniform superposition
 *
 * Algorithm:
 * 1. Compute mean: μ = (1/N) Σ αᵢ
 * 2. Update: αᵢ' = 2μ - αᵢ
 *
 * @param amplitudes State amplitudes (modified in-place)
 * @param state_dim Dimension (2^n)
 */
void grover_diffusion_optimized(complex_t* amplitudes, uint64_t state_dim);

/**
 * @brief SIMD-optimized diffusion operator
 *
 * Uses SIMD for sum and update operations
 *
 * @param amplitudes State amplitudes
 * @param state_dim Dimension
 */
void grover_diffusion_simd(complex_t* amplitudes, uint64_t state_dim);

// ============================================================================
// COMPLETE GROVER SEARCH
// ============================================================================

/**
 * @brief Run optimized Grover search
 *
 * @param state Quantum state (initialized to |0...0⟩)
 * @param targets Target indices
 * @param num_targets Number of targets
 * @param iterations Number of iterations (0 = auto)
 * @return Measured result
 */
uint64_t grover_search_optimized(quantum_state_t* state,
                                 const uint64_t* targets,
                                 int num_targets,
                                 int iterations);

/**
 * @brief Run Grover with custom oracle function
 *
 * @param state Quantum state
 * @param oracle_func Oracle that marks solutions
 * @param oracle_data User data for oracle
 * @param num_targets Expected number of solutions (for iteration count)
 * @param iterations Number of iterations (0 = auto)
 * @return Measured result
 */
uint64_t grover_search_with_oracle(quantum_state_t* state,
                                   int (*oracle_func)(uint64_t, void*),
                                   void* oracle_data,
                                   int num_targets,
                                   int iterations);

// ============================================================================
// AMPLITUDE AMPLIFICATION
// ============================================================================

/**
 * @brief General amplitude amplification
 *
 * Amplifies amplitude of states marked by oracle.
 * More general than Grover - works with non-uniform initial states.
 *
 * @param state Quantum state (any initial state)
 * @param oracle_func Oracle function
 * @param oracle_data Oracle user data
 * @param reflect_func Reflection operator (NULL for standard Grover)
 * @param reflect_data Reflection user data
 * @param iterations Number of iterations
 */
void amplitude_amplification(quantum_state_t* state,
                             int (*oracle_func)(uint64_t, void*),
                             void* oracle_data,
                             void (*reflect_func)(quantum_state_t*, void*),
                             void* reflect_data,
                             int iterations);

// ============================================================================
// QUANTUM COUNTING
// ============================================================================

/**
 * @brief Estimate number of solutions using quantum counting
 *
 * Uses phase estimation on Grover operator to estimate M.
 * Simplified version - returns approximation.
 *
 * @param state Quantum state
 * @param oracle_func Oracle function
 * @param oracle_data Oracle data
 * @param precision_bits Number of precision bits
 * @return Estimated number of solutions
 */
int grover_quantum_counting(quantum_state_t* state,
                            int (*oracle_func)(uint64_t, void*),
                            void* oracle_data,
                            int precision_bits);

// ============================================================================
// PARTIAL SEARCH
// ============================================================================

/**
 * @brief Grover search on subset of qubits
 *
 * Searches within a subspace defined by fixed qubits.
 *
 * @param state Full quantum state
 * @param search_qubits Qubits to search over
 * @param num_search Number of search qubits
 * @param targets Target values (in search qubit space)
 * @param num_targets Number of targets
 * @param iterations Number of iterations
 */
void grover_partial_search(quantum_state_t* state,
                           const int* search_qubits,
                           int num_search,
                           const uint64_t* targets,
                           int num_targets,
                           int iterations);

// ============================================================================
// BENCHMARK UTILITIES
// ============================================================================

/**
 * @brief Benchmark Grover iterations per second
 *
 * @param num_qubits Number of qubits
 * @param num_iterations Number of iterations to run
 * @return Iterations per second
 */
double grover_benchmark(int num_qubits, int num_iterations);

#ifdef __cplusplus
}
#endif

#endif /* ALGORITHMS_GROVER_OPTIMIZER_H */
