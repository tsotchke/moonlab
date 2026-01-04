/**
 * @file tn_gates.h
 * @brief Quantum gate application for tensor network states
 *
 * Implements quantum gate operations on MPS states through tensor contraction.
 * Supports single-qubit, two-qubit, and multi-qubit gates with automatic
 * bond dimension management.
 *
 * Performance characteristics:
 * - Single-qubit gate: O(chi^2 * d) where d=2
 * - Adjacent two-qubit gate: O(chi^3 * d^2)
 * - Non-adjacent two-qubit gate: O(chi^3 * d^2 * distance)
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef TN_GATES_H
#define TN_GATES_H

#include "tn_state.h"
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ERROR CODES
// ============================================================================

typedef enum {
    TN_GATE_SUCCESS = 0,
    TN_GATE_ERROR_NULL_PTR = -1,
    TN_GATE_ERROR_INVALID_QUBIT = -2,
    TN_GATE_ERROR_INVALID_GATE = -3,
    TN_GATE_ERROR_CONTRACTION_FAILED = -4,
    TN_GATE_ERROR_TRUNCATION = -5,
    TN_GATE_ERROR_BOND_TOO_LARGE = -6,
    TN_GATE_ERROR_ALLOC_FAILED = -7
} tn_gate_error_t;

// ============================================================================
// GATE MATRICES
// ============================================================================

/**
 * @brief 2x2 complex matrix for single-qubit gates
 */
typedef struct {
    double complex elements[2][2];
} tn_gate_1q_t;

/**
 * @brief 4x4 complex matrix for two-qubit gates
 */
typedef struct {
    double complex elements[4][4];
} tn_gate_2q_t;

// ============================================================================
// STANDARD GATES
// ============================================================================

/** Identity gate */
extern const tn_gate_1q_t TN_GATE_I;

/** Pauli-X gate */
extern const tn_gate_1q_t TN_GATE_X;

/** Pauli-Y gate */
extern const tn_gate_1q_t TN_GATE_Y;

/** Pauli-Z gate */
extern const tn_gate_1q_t TN_GATE_Z;

/** Hadamard gate */
extern const tn_gate_1q_t TN_GATE_H;

/** S gate (sqrt(Z)) */
extern const tn_gate_1q_t TN_GATE_S;

/** S-dagger gate */
extern const tn_gate_1q_t TN_GATE_SDG;

/** T gate (sqrt(S)) */
extern const tn_gate_1q_t TN_GATE_T;

/** T-dagger gate */
extern const tn_gate_1q_t TN_GATE_TDG;

/** CNOT gate */
extern const tn_gate_2q_t TN_GATE_CNOT;

/** CZ gate */
extern const tn_gate_2q_t TN_GATE_CZ;

/** SWAP gate */
extern const tn_gate_2q_t TN_GATE_SWAP;

/** iSWAP gate */
extern const tn_gate_2q_t TN_GATE_ISWAP;

// ============================================================================
// PARAMETERIZED GATES
// ============================================================================

/**
 * @brief Create Rx rotation gate
 *
 * Rx(theta) = exp(-i*theta/2 * X)
 *
 * @param theta Rotation angle
 * @return Gate matrix
 */
tn_gate_1q_t tn_gate_rx(double theta);

/**
 * @brief Create Ry rotation gate
 *
 * Ry(theta) = exp(-i*theta/2 * Y)
 *
 * @param theta Rotation angle
 * @return Gate matrix
 */
tn_gate_1q_t tn_gate_ry(double theta);

/**
 * @brief Create Rz rotation gate
 *
 * Rz(theta) = exp(-i*theta/2 * Z)
 *
 * @param theta Rotation angle
 * @return Gate matrix
 */
tn_gate_1q_t tn_gate_rz(double theta);

/**
 * @brief Create general U3 gate
 *
 * U3(theta, phi, lambda) = Rz(phi) * Ry(theta) * Rz(lambda)
 *
 * @param theta First rotation angle
 * @param phi Second rotation angle
 * @param lambda Third rotation angle
 * @return Gate matrix
 */
tn_gate_1q_t tn_gate_u3(double theta, double phi, double lambda);

/**
 * @brief Create phase gate
 *
 * P(phi) = diag(1, exp(i*phi))
 *
 * @param phi Phase angle
 * @return Gate matrix
 */
tn_gate_1q_t tn_gate_phase(double phi);

/**
 * @brief Create controlled-Rz gate
 *
 * @param theta Rotation angle
 * @return Gate matrix
 */
tn_gate_2q_t tn_gate_crz(double theta);

/**
 * @brief Create controlled-phase (CPhase) gate
 *
 * @param phi Phase angle
 * @return Gate matrix
 */
tn_gate_2q_t tn_gate_cphase(double phi);

/**
 * @brief Create XX interaction gate
 *
 * Rxx(theta) = exp(-i*theta/2 * X tensor X)
 *
 * @param theta Interaction strength
 * @return Gate matrix
 */
tn_gate_2q_t tn_gate_rxx(double theta);

/**
 * @brief Create YY interaction gate
 *
 * @param theta Interaction strength
 * @return Gate matrix
 */
tn_gate_2q_t tn_gate_ryy(double theta);

/**
 * @brief Create ZZ interaction gate
 *
 * @param theta Interaction strength
 * @return Gate matrix
 */
tn_gate_2q_t tn_gate_rzz(double theta);

// ============================================================================
// SINGLE-QUBIT GATE APPLICATION
// ============================================================================

/**
 * @brief Apply single-qubit gate to MPS state
 *
 * Modifies state in place. Does not increase bond dimension.
 *
 * @param state MPS state
 * @param qubit Target qubit index
 * @param gate Gate matrix
 * @return TN_GATE_SUCCESS or error code
 */
tn_gate_error_t tn_apply_gate_1q(tn_mps_state_t *state,
                                  uint32_t qubit,
                                  const tn_gate_1q_t *gate);

/**
 * @brief Apply Pauli-X gate
 */
tn_gate_error_t tn_apply_x(tn_mps_state_t *state, uint32_t qubit);

/**
 * @brief Apply Pauli-Y gate
 */
tn_gate_error_t tn_apply_y(tn_mps_state_t *state, uint32_t qubit);

/**
 * @brief Apply Pauli-Z gate
 */
tn_gate_error_t tn_apply_z(tn_mps_state_t *state, uint32_t qubit);

/**
 * @brief Apply Hadamard gate
 */
tn_gate_error_t tn_apply_h(tn_mps_state_t *state, uint32_t qubit);

/**
 * @brief Apply S gate
 */
tn_gate_error_t tn_apply_s(tn_mps_state_t *state, uint32_t qubit);

/**
 * @brief Apply T gate
 */
tn_gate_error_t tn_apply_t(tn_mps_state_t *state, uint32_t qubit);

/**
 * @brief Apply Rx rotation
 */
tn_gate_error_t tn_apply_rx(tn_mps_state_t *state, uint32_t qubit, double theta);

/**
 * @brief Apply Ry rotation
 */
tn_gate_error_t tn_apply_ry(tn_mps_state_t *state, uint32_t qubit, double theta);

/**
 * @brief Apply Rz rotation
 */
tn_gate_error_t tn_apply_rz(tn_mps_state_t *state, uint32_t qubit, double theta);

// ============================================================================
// TWO-QUBIT GATE APPLICATION
// ============================================================================

/**
 * @brief Apply two-qubit gate to MPS state
 *
 * For adjacent qubits, contracts and SVD-splits.
 * For non-adjacent qubits, uses SWAP network.
 *
 * May increase bond dimension (up to max_bond_dim from config).
 *
 * @param state MPS state
 * @param qubit1 First qubit index
 * @param qubit2 Second qubit index
 * @param gate Gate matrix
 * @param truncation_error Output: truncation error (can be NULL)
 * @return TN_GATE_SUCCESS or error code
 */
tn_gate_error_t tn_apply_gate_2q(tn_mps_state_t *state,
                                  uint32_t qubit1, uint32_t qubit2,
                                  const tn_gate_2q_t *gate,
                                  double *truncation_error);

/**
 * @brief Apply CNOT gate
 */
tn_gate_error_t tn_apply_cnot(tn_mps_state_t *state,
                               uint32_t control, uint32_t target);

/**
 * @brief Apply CZ gate
 */
tn_gate_error_t tn_apply_cz(tn_mps_state_t *state,
                             uint32_t qubit1, uint32_t qubit2);

/**
 * @brief Apply SWAP gate
 */
tn_gate_error_t tn_apply_swap(tn_mps_state_t *state,
                               uint32_t qubit1, uint32_t qubit2);

/**
 * @brief Apply ZZ interaction gate
 */
tn_gate_error_t tn_apply_rzz(tn_mps_state_t *state,
                              uint32_t qubit1, uint32_t qubit2,
                              double theta);

// ============================================================================
// MULTI-QUBIT OPERATIONS
// ============================================================================

/**
 * @brief Apply controlled gate (any number of controls)
 *
 * @param state MPS state
 * @param controls Array of control qubit indices
 * @param num_controls Number of control qubits
 * @param target Target qubit
 * @param gate Gate to apply when all controls are |1>
 * @return TN_GATE_SUCCESS or error code
 */
tn_gate_error_t tn_apply_controlled(tn_mps_state_t *state,
                                     const uint32_t *controls,
                                     uint32_t num_controls,
                                     uint32_t target,
                                     const tn_gate_1q_t *gate);

/**
 * @brief Apply Toffoli (CCX) gate
 */
tn_gate_error_t tn_apply_toffoli(tn_mps_state_t *state,
                                  uint32_t control1, uint32_t control2,
                                  uint32_t target);

/**
 * @brief Apply global phase
 *
 * @param state MPS state
 * @param phase Phase angle
 * @return TN_GATE_SUCCESS or error code
 */
tn_gate_error_t tn_apply_global_phase(tn_mps_state_t *state, double phase);

// ============================================================================
// LAYER OPERATIONS
// ============================================================================

/**
 * @brief Apply Hadamard to all qubits
 *
 * @param state MPS state
 * @return TN_GATE_SUCCESS or error code
 */
tn_gate_error_t tn_apply_h_all(tn_mps_state_t *state);

/**
 * @brief Apply layer of single-qubit gates
 *
 * @param state MPS state
 * @param gates Array of gate matrices [num_qubits]
 * @return TN_GATE_SUCCESS or error code
 */
tn_gate_error_t tn_apply_layer_1q(tn_mps_state_t *state,
                                   const tn_gate_1q_t *gates);

/**
 * @brief Apply layer of ZZ interactions (even-odd pattern)
 *
 * Common in Trotterized evolution and QAOA.
 *
 * @param state MPS state
 * @param angles Array of angles [num_qubits/2]
 * @param even If true, apply to (0,1), (2,3), ...; else (1,2), (3,4), ...
 * @return TN_GATE_SUCCESS or error code
 */
tn_gate_error_t tn_apply_layer_rzz(tn_mps_state_t *state,
                                    const double *angles,
                                    bool even);

// ============================================================================
// MATRIX PRODUCT OPERATOR (MPO)
// ============================================================================

/**
 * @brief Matrix Product Operator for Hamiltonian/gate application
 */
typedef struct {
    uint32_t num_sites;             /**< Number of sites (qubits) */
    tensor_t **tensors;             /**< MPO tensors [num_sites] */
    uint32_t *bond_dims;            /**< Bond dimensions */
} tn_mpo_t;

/**
 * @brief Create MPO for single-site operator
 *
 * @param num_sites Number of sites
 * @param site Site to apply operator
 * @param op 2x2 operator matrix
 * @return MPO or NULL on failure
 */
tn_mpo_t *tn_mpo_single_site(uint32_t num_sites, uint32_t site,
                              const tn_gate_1q_t *op);

/**
 * @brief Create MPO for two-site operator
 *
 * @param num_sites Number of sites
 * @param site1 First site
 * @param site2 Second site
 * @param op 4x4 operator matrix
 * @return MPO or NULL on failure
 */
tn_mpo_t *tn_mpo_two_site(uint32_t num_sites, uint32_t site1, uint32_t site2,
                           const tn_gate_2q_t *op);

/**
 * @brief Apply MPO to MPS state
 *
 * Result: |psi'> = MPO |psi>
 *
 * @param state MPS state (modified in place)
 * @param mpo Matrix product operator
 * @param truncation_error Output: total truncation error
 * @return TN_GATE_SUCCESS or error code
 */
tn_gate_error_t tn_apply_mpo(tn_mps_state_t *state,
                              const tn_mpo_t *mpo,
                              double *truncation_error);

/**
 * @brief Free MPO
 *
 * @param mpo MPO to free
 */
void tn_mpo_free(tn_mpo_t *mpo);

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * @brief Get error string
 *
 * @param error Error code
 * @return Human-readable error string
 */
const char *tn_gate_error_string(tn_gate_error_t error);

/**
 * @brief Print gate matrix (for debugging)
 *
 * @param gate Gate matrix
 * @param name Gate name
 */
void tn_gate_1q_print(const tn_gate_1q_t *gate, const char *name);

/**
 * @brief Print two-qubit gate matrix
 *
 * @param gate Gate matrix
 * @param name Gate name
 */
void tn_gate_2q_print(const tn_gate_2q_t *gate, const char *name);

/**
 * @brief Convert 2x2 matrix to gate structure
 *
 * @param matrix Flat 4-element complex array (row-major)
 * @return Gate structure
 */
tn_gate_1q_t tn_gate_from_matrix(const double complex *matrix);

/**
 * @brief Convert 4x4 matrix to gate structure
 *
 * @param matrix Flat 16-element complex array (row-major)
 * @return Gate structure
 */
tn_gate_2q_t tn_gate_2q_from_matrix(const double complex *matrix);

#ifdef __cplusplus
}
#endif

#endif /* TN_GATES_H */
