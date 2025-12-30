/**
 * @file distributed_gates.h
 * @brief Distributed quantum gate operations
 *
 * Provides MPI-aware implementations of quantum gates that handle
 * communication when gate operations span multiple MPI ranks.
 *
 * Gate Classification:
 * - Local gates: Both qubits within local partition (no communication)
 * - Remote gates: One or both qubits are partition qubits (requires exchange)
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#ifndef DISTRIBUTED_GATES_H
#define DISTRIBUTED_GATES_H

#include <stdint.h>
#include <complex.h>
#include "state_partition.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// GATE ERROR CODES
// ============================================================================

typedef enum {
    DIST_GATE_SUCCESS = 0,
    DIST_GATE_ERROR_INVALID_QUBIT = -1,
    DIST_GATE_ERROR_COMM = -2,
    DIST_GATE_ERROR_ALLOC = -3,
    DIST_GATE_ERROR_NOT_INITIALIZED = -4,
    DIST_GATE_ERROR_INVALID_MATRIX = -5
} dist_gate_error_t;

// ============================================================================
// GATE MATRICES
// ============================================================================

/**
 * @brief 2x2 gate matrix (single-qubit)
 *
 * Row-major: [a00, a01, a10, a11]
 */
typedef struct {
    double complex m[4];
} gate_matrix_2x2_t;

/**
 * @brief 4x4 gate matrix (two-qubit)
 *
 * Row-major: [a00, a01, ..., a33]
 */
typedef struct {
    double complex m[16];
} gate_matrix_4x4_t;

// Standard gate matrices
extern const gate_matrix_2x2_t GATE_H;    // Hadamard
extern const gate_matrix_2x2_t GATE_X;    // Pauli-X
extern const gate_matrix_2x2_t GATE_Y;    // Pauli-Y
extern const gate_matrix_2x2_t GATE_Z;    // Pauli-Z
extern const gate_matrix_2x2_t GATE_S;    // Phase (S = sqrt(Z))
extern const gate_matrix_2x2_t GATE_T;    // T gate (sqrt(S))
extern const gate_matrix_2x2_t GATE_SDAG; // S-dagger
extern const gate_matrix_2x2_t GATE_TDAG; // T-dagger

extern const gate_matrix_4x4_t GATE_CNOT;    // Controlled-NOT
extern const gate_matrix_4x4_t GATE_CZ;      // Controlled-Z
extern const gate_matrix_4x4_t GATE_SWAP;    // SWAP
extern const gate_matrix_4x4_t GATE_ISWAP;   // iSWAP

// ============================================================================
// SINGLE-QUBIT GATES
// ============================================================================

/**
 * @brief Apply arbitrary single-qubit gate
 *
 * Handles both local and remote (partition) qubits.
 *
 * @param state Partitioned quantum state
 * @param target Target qubit index
 * @param matrix 2x2 gate matrix
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_gate_1q(partitioned_state_t* state,
                               uint32_t target,
                               const gate_matrix_2x2_t* matrix);

/**
 * @brief Apply Hadamard gate
 *
 * @param state Partitioned state
 * @param target Target qubit
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_hadamard(partitioned_state_t* state, uint32_t target);

/**
 * @brief Apply Hadamard to all qubits
 *
 * Optimized batch application.
 *
 * @param state Partitioned state
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_hadamard_all(partitioned_state_t* state);

/**
 * @brief Apply Pauli-X (NOT) gate
 *
 * @param state Partitioned state
 * @param target Target qubit
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_pauli_x(partitioned_state_t* state, uint32_t target);

/**
 * @brief Apply Pauli-Y gate
 *
 * @param state Partitioned state
 * @param target Target qubit
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_pauli_y(partitioned_state_t* state, uint32_t target);

/**
 * @brief Apply Pauli-Z gate
 *
 * @param state Partitioned state
 * @param target Target qubit
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_pauli_z(partitioned_state_t* state, uint32_t target);

/**
 * @brief Apply phase rotation Rz(theta)
 *
 * Rz(θ) = diag(e^{-iθ/2}, e^{iθ/2})
 *
 * @param state Partitioned state
 * @param target Target qubit
 * @param theta Rotation angle in radians
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_rz(partitioned_state_t* state, uint32_t target, double theta);

/**
 * @brief Apply rotation around X axis
 *
 * Rx(θ) = cos(θ/2)I - i*sin(θ/2)X
 *
 * @param state Partitioned state
 * @param target Target qubit
 * @param theta Rotation angle in radians
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_rx(partitioned_state_t* state, uint32_t target, double theta);

/**
 * @brief Apply rotation around Y axis
 *
 * Ry(θ) = cos(θ/2)I - i*sin(θ/2)Y
 *
 * @param state Partitioned state
 * @param target Target qubit
 * @param theta Rotation angle in radians
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_ry(partitioned_state_t* state, uint32_t target, double theta);

/**
 * @brief Apply phase shift gate
 *
 * P(φ) = diag(1, e^{iφ})
 *
 * @param state Partitioned state
 * @param target Target qubit
 * @param phi Phase angle in radians
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_phase(partitioned_state_t* state, uint32_t target, double phi);

/**
 * @brief Apply S gate (phase gate with φ = π/2)
 *
 * @param state Partitioned state
 * @param target Target qubit
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_s_gate(partitioned_state_t* state, uint32_t target);

/**
 * @brief Apply T gate (phase gate with φ = π/4)
 *
 * @param state Partitioned state
 * @param target Target qubit
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_t_gate(partitioned_state_t* state, uint32_t target);

// ============================================================================
// TWO-QUBIT GATES
// ============================================================================

/**
 * @brief Apply arbitrary two-qubit gate
 *
 * Handles all combinations of local/remote qubits.
 *
 * @param state Partitioned state
 * @param qubit1 First qubit (control for controlled gates)
 * @param qubit2 Second qubit (target for controlled gates)
 * @param matrix 4x4 gate matrix
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_gate_2q(partitioned_state_t* state,
                               uint32_t qubit1,
                               uint32_t qubit2,
                               const gate_matrix_4x4_t* matrix);

/**
 * @brief Apply CNOT (controlled-X) gate
 *
 * @param state Partitioned state
 * @param control Control qubit
 * @param target Target qubit
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_cnot(partitioned_state_t* state,
                            uint32_t control,
                            uint32_t target);

/**
 * @brief Apply CZ (controlled-Z) gate
 *
 * @param state Partitioned state
 * @param qubit1 First qubit
 * @param qubit2 Second qubit
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_cz(partitioned_state_t* state,
                          uint32_t qubit1,
                          uint32_t qubit2);

/**
 * @brief Apply controlled phase gate
 *
 * CP(φ) applies phase e^{iφ} when both qubits are |1⟩
 *
 * @param state Partitioned state
 * @param control Control qubit
 * @param target Target qubit
 * @param phi Phase angle in radians
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_cphase(partitioned_state_t* state,
                              uint32_t control,
                              uint32_t target,
                              double phi);

/**
 * @brief Apply SWAP gate
 *
 * @param state Partitioned state
 * @param qubit1 First qubit
 * @param qubit2 Second qubit
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_swap(partitioned_state_t* state,
                            uint32_t qubit1,
                            uint32_t qubit2);

/**
 * @brief Apply iSWAP gate
 *
 * @param state Partitioned state
 * @param qubit1 First qubit
 * @param qubit2 Second qubit
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_iswap(partitioned_state_t* state,
                             uint32_t qubit1,
                             uint32_t qubit2);

/**
 * @brief Apply sqrt(SWAP) gate
 *
 * @param state Partitioned state
 * @param qubit1 First qubit
 * @param qubit2 Second qubit
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_sqrt_swap(partitioned_state_t* state,
                                 uint32_t qubit1,
                                 uint32_t qubit2);

// ============================================================================
// MULTI-QUBIT GATES
// ============================================================================

/**
 * @brief Apply Toffoli (CCNOT) gate
 *
 * @param state Partitioned state
 * @param control1 First control qubit
 * @param control2 Second control qubit
 * @param target Target qubit
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_toffoli(partitioned_state_t* state,
                               uint32_t control1,
                               uint32_t control2,
                               uint32_t target);

/**
 * @brief Apply Fredkin (CSWAP) gate
 *
 * @param state Partitioned state
 * @param control Control qubit
 * @param target1 First swap qubit
 * @param target2 Second swap qubit
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_fredkin(partitioned_state_t* state,
                               uint32_t control,
                               uint32_t target1,
                               uint32_t target2);

/**
 * @brief Apply multi-controlled Z gate
 *
 * @param state Partitioned state
 * @param controls Array of control qubit indices
 * @param num_controls Number of control qubits
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_mcz(partitioned_state_t* state,
                           const uint32_t* controls,
                           uint32_t num_controls);

/**
 * @brief Apply multi-controlled X gate
 *
 * @param state Partitioned state
 * @param controls Array of control qubit indices
 * @param num_controls Number of control qubits
 * @param target Target qubit
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_mcx(partitioned_state_t* state,
                           const uint32_t* controls,
                           uint32_t num_controls,
                           uint32_t target);

// ============================================================================
// GROVER'S ALGORITHM OPERATIONS
// ============================================================================

/**
 * @brief Apply oracle (phase flip on marked states)
 *
 * Applies -1 phase to specified target state.
 *
 * @param state Partitioned state
 * @param target_state Basis state to mark
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_oracle_single(partitioned_state_t* state,
                                     uint64_t target_state);

/**
 * @brief Apply oracle for multiple targets
 *
 * @param state Partitioned state
 * @param targets Array of target states
 * @param num_targets Number of targets
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_oracle_multi(partitioned_state_t* state,
                                    const uint64_t* targets,
                                    uint32_t num_targets);

/**
 * @brief Apply Grover diffusion operator
 *
 * D = 2|ψ⟩⟨ψ| - I where |ψ⟩ is uniform superposition.
 *
 * @param state Partitioned state
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_grover_diffusion(partitioned_state_t* state);

/**
 * @brief Complete Grover iteration (oracle + diffusion)
 *
 * @param state Partitioned state
 * @param target_state Target basis state
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_grover_iteration(partitioned_state_t* state,
                                        uint64_t target_state);

/**
 * @brief Full Grover search
 *
 * Runs optimal number of iterations for single target.
 *
 * @param state Partitioned state (should be in uniform superposition)
 * @param target_state State to find
 * @param num_iterations Number of iterations (0 for optimal)
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_grover_search(partitioned_state_t* state,
                                     uint64_t target_state,
                                     uint32_t num_iterations);

// ============================================================================
// QFT OPERATIONS
// ============================================================================

/**
 * @brief Apply Quantum Fourier Transform
 *
 * @param state Partitioned state
 * @param start_qubit First qubit index
 * @param num_qubits Number of qubits for QFT
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_qft(partitioned_state_t* state,
                           uint32_t start_qubit,
                           uint32_t num_qubits);

/**
 * @brief Apply inverse QFT
 *
 * @param state Partitioned state
 * @param start_qubit First qubit index
 * @param num_qubits Number of qubits for IQFT
 * @return DIST_GATE_SUCCESS or error code
 */
dist_gate_error_t dist_iqft(partitioned_state_t* state,
                            uint32_t start_qubit,
                            uint32_t num_qubits);

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * @brief Build Rx rotation matrix
 *
 * @param theta Rotation angle
 * @param matrix Output matrix
 */
void dist_build_rx_matrix(double theta, gate_matrix_2x2_t* matrix);

/**
 * @brief Build Ry rotation matrix
 *
 * @param theta Rotation angle
 * @param matrix Output matrix
 */
void dist_build_ry_matrix(double theta, gate_matrix_2x2_t* matrix);

/**
 * @brief Build Rz rotation matrix
 *
 * @param theta Rotation angle
 * @param matrix Output matrix
 */
void dist_build_rz_matrix(double theta, gate_matrix_2x2_t* matrix);

/**
 * @brief Build phase gate matrix
 *
 * @param phi Phase angle
 * @param matrix Output matrix
 */
void dist_build_phase_matrix(double phi, gate_matrix_2x2_t* matrix);

/**
 * @brief Build controlled-phase matrix
 *
 * @param phi Phase angle
 * @param matrix Output matrix
 */
void dist_build_cphase_matrix(double phi, gate_matrix_4x4_t* matrix);

/**
 * @brief Get error string
 *
 * @param error Error code
 * @return Human-readable error message
 */
const char* dist_gate_error_string(dist_gate_error_t error);

/**
 * @brief Calculate optimal Grover iterations
 *
 * @param num_qubits Number of qubits
 * @param num_targets Number of marked states
 * @return Optimal iteration count
 */
uint32_t dist_grover_optimal_iterations(uint32_t num_qubits, uint32_t num_targets);

#ifdef __cplusplus
}
#endif

#endif /* DISTRIBUTED_GATES_H */
