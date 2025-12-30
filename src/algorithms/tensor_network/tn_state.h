/**
 * @file tn_state.h
 * @brief Tensor network quantum state representation
 *
 * Implements Matrix Product State (MPS) representation for quantum states.
 * Enables simulation of 50-200+ qubits for circuits with bounded entanglement.
 *
 * Memory scaling comparison (approximate):
 * | Qubits | State Vector | MPS (chi=256) | MPS (chi=512) |
 * |--------|--------------|---------------|---------------|
 * | 50     | 18 PB        | 13 MB         | 52 MB         |
 * | 100    | 20E9 TB      | 26 MB         | 105 MB        |
 * | 200    | Impossible   | 52 MB         | 210 MB        |
 *
 * Limitations:
 * - Accuracy depends on entanglement structure
 * - High-entanglement states (Grover, QFT) require large bond dimensions
 * - Best suited for local connectivity, shallow circuits
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TN_STATE_H
#define TN_STATE_H

#include "tensor.h"
#include "svd_compress.h"
#include "contraction.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CONFIGURATION
// ============================================================================

/** Default maximum bond dimension */
#define TN_DEFAULT_BOND_DIM 256

/** Maximum bond dimension supported */
#define TN_MAX_BOND_DIM 8192

/** Maximum qubits for tensor network simulation */
#define TN_MAX_QUBITS 1024

/** Local Hilbert space dimension (2 for qubits) */
#define TN_PHYSICAL_DIM 2

// ============================================================================
// ERROR CODES
// ============================================================================

typedef enum {
    TN_STATE_SUCCESS = 0,
    TN_STATE_ERROR_NULL_PTR = -1,
    TN_STATE_ERROR_INVALID_QUBITS = -2,
    TN_STATE_ERROR_INVALID_QUBIT_INDEX = -3,
    TN_STATE_ERROR_ALLOC_FAILED = -4,
    TN_STATE_ERROR_TRUNCATION = -5,
    TN_STATE_ERROR_CONTRACTION_FAILED = -6,
    TN_STATE_ERROR_NORMALIZATION = -7,
    TN_STATE_ERROR_INVALID_CONFIG = -8,
    TN_STATE_ERROR_ENTANGLEMENT_TOO_HIGH = -9
} tn_state_error_t;

// ============================================================================
// CANONICAL FORMS
// ============================================================================

/**
 * @brief MPS canonical form
 *
 * Canonical forms provide numerical stability and efficient computation.
 */
typedef enum {
    TN_CANONICAL_NONE,      /**< No canonical form */
    TN_CANONICAL_LEFT,      /**< Left-canonical (orthogonal from left) */
    TN_CANONICAL_RIGHT,     /**< Right-canonical (orthogonal from right) */
    TN_CANONICAL_MIXED      /**< Mixed-canonical (orthogonal except center) */
} tn_canonical_form_t;

// ============================================================================
// STATE STRUCTURES
// ============================================================================

/**
 * @brief Configuration for tensor network state
 */
typedef struct {
    uint32_t max_bond_dim;          /**< Maximum bond dimension */
    double svd_cutoff;              /**< SVD truncation cutoff */
    double max_truncation_error;    /**< Max allowed truncation error per step */
    bool track_truncation;          /**< Track cumulative truncation error */
    bool auto_canonicalize;         /**< Auto-canonicalize after operations */
    tn_canonical_form_t target_form; /**< Target canonical form */
} tn_state_config_t;

/**
 * @brief Workspace for optimized MPS operations
 *
 * Pre-allocated buffers for norm/overlap calculations to avoid repeated allocation.
 */
typedef struct {
    double complex *transfer_buffer; /**< Pre-allocated transfer matrix buffer */
    double complex *local_buffer;    /**< Pre-allocated local contraction buffer */
    uint64_t buffer_capacity;        /**< Current buffer capacity in elements */
    bool valid;                      /**< Whether workspace is initialized */
} tn_mps_workspace_t;

/**
 * @brief Matrix Product State (MPS)
 *
 * Represents quantum state as chain of tensors:
 *   |psi> = sum_{s1,s2,...} A[1]^{s1} A[2]^{s2} ... A[n]^{sn} |s1,s2,...,sn>
 *
 * Each tensor A[i] has shape [left_bond, physical, right_bond]
 * - First tensor: [1, 2, bond]
 * - Middle tensors: [bond, 2, bond]
 * - Last tensor: [bond, 2, 1]
 */
typedef struct {
    uint32_t num_qubits;            /**< Number of qubits */
    tensor_t **tensors;             /**< Array of MPS tensors [num_qubits] */
    uint32_t *bond_dims;            /**< Bond dimensions [num_qubits-1] */
    tn_canonical_form_t canonical;  /**< Current canonical form */
    int32_t canonical_center;       /**< Center site for mixed canonical (-1 if none) */
    tn_state_config_t config;       /**< Configuration */
    double norm;                    /**< State norm (may be non-1 after truncation) */
    double log_norm_factor;         /**< Log of accumulated norm factor (for lazy normalization) */
    double cumulative_truncation_error; /**< Total truncation error */
    uint64_t num_truncations;       /**< Number of truncation operations */
    tn_mps_workspace_t workspace;   /**< Pre-allocated workspace for optimized operations */
} tn_mps_state_t;

/**
 * @brief Statistics for MPS state
 */
typedef struct {
    uint64_t total_elements;        /**< Total tensor elements */
    uint64_t memory_bytes;          /**< Total memory usage */
    uint32_t max_bond_dim;          /**< Maximum bond dimension */
    double avg_bond_dim;            /**< Average bond dimension */
    double entanglement_entropy;    /**< Entanglement entropy at center */
    double truncation_error;        /**< Cumulative truncation error */
    double norm;                    /**< Current norm */
} tn_mps_stats_t;

// ============================================================================
// STATE CREATION AND DESTRUCTION
// ============================================================================

/**
 * @brief Create default configuration
 *
 * @return Default MPS configuration
 */
tn_state_config_t tn_state_config_default(void);

/**
 * @brief Create configuration with specified bond dimension
 *
 * @param max_bond Maximum bond dimension
 * @param cutoff SVD truncation cutoff
 * @return Configuration
 */
tn_state_config_t tn_state_config_create(uint32_t max_bond, double cutoff);

/**
 * @brief Create MPS in |00...0> state
 *
 * @param num_qubits Number of qubits
 * @param config Configuration (NULL for defaults)
 * @return MPS state or NULL on failure
 */
tn_mps_state_t *tn_mps_create_zero(uint32_t num_qubits,
                                    const tn_state_config_t *config);

/**
 * @brief Create MPS from computational basis state
 *
 * @param num_qubits Number of qubits
 * @param basis_state Binary representation of basis state
 * @param config Configuration
 * @return MPS state or NULL on failure
 */
tn_mps_state_t *tn_mps_create_basis(uint32_t num_qubits,
                                     uint64_t basis_state,
                                     const tn_state_config_t *config);

/**
 * @brief Create MPS in product state
 *
 * Each qubit initialized to individual state.
 *
 * @param num_qubits Number of qubits
 * @param qubit_states Array of 2-component complex vectors [num_qubits][2]
 * @param config Configuration
 * @return MPS state or NULL on failure
 */
tn_mps_state_t *tn_mps_create_product(uint32_t num_qubits,
                                       const double complex (*qubit_states)[2],
                                       const tn_state_config_t *config);

/**
 * @brief Create MPS from full state vector
 *
 * Converts state vector to MPS via successive SVD.
 * May involve significant truncation for entangled states.
 *
 * @param amplitudes Full state vector (length 2^num_qubits)
 * @param num_qubits Number of qubits
 * @param config Configuration
 * @return MPS state or NULL on failure
 */
tn_mps_state_t *tn_mps_from_statevector(const double complex *amplitudes,
                                         uint32_t num_qubits,
                                         const tn_state_config_t *config);

/**
 * @brief Create copy of MPS state
 *
 * @param state State to copy
 * @return Copy or NULL on failure
 */
tn_mps_state_t *tn_mps_copy(const tn_mps_state_t *state);

/**
 * @brief Free MPS state
 *
 * @param state State to free
 */
void tn_mps_free(tn_mps_state_t *state);

// ============================================================================
// STATE PROPERTIES
// ============================================================================

/**
 * @brief Get number of qubits
 *
 * @param state MPS state
 * @return Number of qubits
 */
uint32_t tn_mps_num_qubits(const tn_mps_state_t *state);

/**
 * @brief Get bond dimension between qubits
 *
 * @param state MPS state
 * @param bond Bond index (0 to num_qubits-2)
 * @return Bond dimension
 */
uint32_t tn_mps_bond_dim(const tn_mps_state_t *state, uint32_t bond);

/**
 * @brief Get maximum bond dimension
 *
 * @param state MPS state
 * @return Maximum bond dimension across all bonds
 */
uint32_t tn_mps_max_bond_dim(const tn_mps_state_t *state);

/**
 * @brief Get tensor for specific qubit
 *
 * @param state MPS state
 * @param qubit Qubit index
 * @return Tensor (do not free)
 */
const tensor_t *tn_mps_get_tensor(const tn_mps_state_t *state, uint32_t qubit);

/**
 * @brief Get state statistics
 *
 * @param state MPS state
 * @return Statistics structure
 */
tn_mps_stats_t tn_mps_get_stats(const tn_mps_state_t *state);

/**
 * @brief Print state information
 *
 * @param state MPS state
 */
void tn_mps_print_info(const tn_mps_state_t *state);

// ============================================================================
// AMPLITUDE ACCESS
// ============================================================================

/**
 * @brief Get amplitude for basis state
 *
 * Contracts MPS to compute single amplitude.
 * Cost: O(num_qubits * bond_dim^2)
 *
 * @param state MPS state
 * @param basis_state Binary representation
 * @return Complex amplitude
 */
double complex tn_mps_amplitude(const tn_mps_state_t *state, uint64_t basis_state);

/**
 * @brief Get multiple amplitudes efficiently
 *
 * @param state MPS state
 * @param basis_states Array of basis states
 * @param num_states Number of states
 * @param amplitudes Output array for amplitudes
 * @return TN_STATE_SUCCESS or error code
 */
tn_state_error_t tn_mps_amplitudes(const tn_mps_state_t *state,
                                    const uint64_t *basis_states,
                                    uint32_t num_states,
                                    double complex *amplitudes);

/**
 * @brief Convert MPS to full state vector
 *
 * Warning: Only feasible for small qubit counts (<~30).
 *
 * @param state MPS state
 * @param amplitudes Output array (length 2^num_qubits)
 * @return TN_STATE_SUCCESS or error code
 */
tn_state_error_t tn_mps_to_statevector(const tn_mps_state_t *state,
                                        double complex *amplitudes);

// ============================================================================
// CANONICAL FORM OPERATIONS
// ============================================================================

/**
 * @brief Bring MPS to left-canonical form
 *
 * All tensors except rightmost become left-isometric.
 *
 * @param state MPS state (modified in place)
 * @return TN_STATE_SUCCESS or error code
 */
tn_state_error_t tn_mps_left_canonicalize(tn_mps_state_t *state);

/**
 * @brief Bring MPS to right-canonical form
 *
 * All tensors except leftmost become right-isometric.
 *
 * @param state MPS state (modified in place)
 * @return TN_STATE_SUCCESS or error code
 */
tn_state_error_t tn_mps_right_canonicalize(tn_mps_state_t *state);

/**
 * @brief Bring MPS to mixed-canonical form
 *
 * Tensors left of center are left-isometric, right are right-isometric.
 *
 * @param state MPS state (modified in place)
 * @param center Center site index
 * @return TN_STATE_SUCCESS or error code
 */
tn_state_error_t tn_mps_mixed_canonicalize(tn_mps_state_t *state, uint32_t center);

/**
 * @brief Move canonical center by one site
 *
 * @param state MPS state
 * @param direction Direction (-1 for left, +1 for right)
 * @return TN_STATE_SUCCESS or error code
 */
tn_state_error_t tn_mps_move_center(tn_mps_state_t *state, int direction);

/**
 * @brief Mark MPS as left-canonical after TEBD sweep (O(1) operation)
 *
 * After a complete left-to-right TEBD sweep where each two-qubit gate uses
 * SVD with singular values absorbed into the right tensor, the MPS is in
 * left-canonical form. This function marks it as such, enabling the O(chi^2)
 * fast path for norm computation instead of O(n*chi^3) transfer matrix method.
 *
 * Call this after completing a full Trotter step (even bonds then odd bonds).
 *
 * @param state MPS state
 */
void tn_mps_mark_canonical_left(tn_mps_state_t *state);

/**
 * @brief Mark MPS as right-canonical (O(1) operation)
 *
 * Similar to tn_mps_mark_canonical_left but for right-canonical form.
 *
 * @param state MPS state
 */
void tn_mps_mark_canonical_right(tn_mps_state_t *state);

// ============================================================================
// NORMALIZATION
// ============================================================================

/**
 * @brief Compute norm of MPS state
 *
 * @param state MPS state
 * @return Norm (should be 1 for normalized state)
 */
double tn_mps_norm(const tn_mps_state_t *state);

/**
 * @brief Compute norm squared using optimized algorithm with memory pooling
 *
 * Uses pre-allocated workspace buffers and avoids unnecessary transposes.
 * Approximately 3-5x faster than tn_mps_norm() for large states.
 *
 * @param state MPS state
 * @return Norm squared (|<psi|psi>|)
 */
double tn_mps_norm_squared_fast(tn_mps_state_t *state);

/**
 * @brief Normalize MPS state
 *
 * @param state MPS state (modified in place)
 * @return TN_STATE_SUCCESS or error code
 */
tn_state_error_t tn_mps_normalize(tn_mps_state_t *state);

/**
 * @brief Apply lazy normalization (accumulate norm factor)
 *
 * Instead of renormalizing tensors, accumulates log(norm) in log_norm_factor.
 * Use tn_mps_commit_normalization() to apply when needed.
 * This is much faster when norm is computed frequently but actual
 * normalized values are only needed occasionally.
 *
 * @param state MPS state
 * @return TN_STATE_SUCCESS or error code
 */
tn_state_error_t tn_mps_normalize_lazy(tn_mps_state_t *state);

/**
 * @brief Commit accumulated lazy normalization to tensors
 *
 * Applies the accumulated log_norm_factor to tensor data.
 * Call this before operations that need actual normalized amplitudes.
 *
 * @param state MPS state
 * @return TN_STATE_SUCCESS or error code
 */
tn_state_error_t tn_mps_commit_normalization(tn_mps_state_t *state);

/**
 * @brief Get true norm including lazy normalization factor
 *
 * Returns exp(log_norm_factor) * tn_mps_norm()
 *
 * @param state MPS state
 * @return True norm including lazy factor
 */
double tn_mps_true_norm(const tn_mps_state_t *state);

/**
 * @brief Initialize workspace for optimized operations
 *
 * Pre-allocates buffers sized for current max bond dimension.
 *
 * @param state MPS state
 * @return TN_STATE_SUCCESS or error code
 */
tn_state_error_t tn_mps_init_workspace(tn_mps_state_t *state);

/**
 * @brief Free workspace buffers
 *
 * @param state MPS state
 */
void tn_mps_free_workspace(tn_mps_state_t *state);

// ============================================================================
// ENTANGLEMENT
// ============================================================================

/**
 * @brief Compute entanglement entropy at bond
 *
 * Uses Schmidt decomposition (singular values at bond).
 *
 * @param state MPS state
 * @param bond Bond index
 * @return Entanglement entropy (von Neumann)
 */
double tn_mps_entanglement_entropy(const tn_mps_state_t *state, uint32_t bond);

/**
 * @brief Compute entanglement spectrum at bond
 *
 * Returns Schmidt coefficients (squared singular values).
 *
 * @param state MPS state
 * @param bond Bond index
 * @param spectrum Output array (at least bond_dim elements)
 * @param num_values Output: number of values written
 * @return TN_STATE_SUCCESS or error code
 */
tn_state_error_t tn_mps_entanglement_spectrum(const tn_mps_state_t *state,
                                               uint32_t bond,
                                               double *spectrum,
                                               uint32_t *num_values);

/**
 * @brief Check if state is approximately product state
 *
 * @param state MPS state
 * @param tolerance Tolerance for entanglement
 * @return true if all bonds have bond_dim <= 1 + tolerance
 */
bool tn_mps_is_product_state(const tn_mps_state_t *state, double tolerance);

// ============================================================================
// BOND DIMENSION MANAGEMENT
// ============================================================================

/**
 * @brief Truncate all bonds to maximum dimension
 *
 * @param state MPS state
 * @param max_bond Maximum bond dimension
 * @param truncation_error Output: total truncation error
 * @return TN_STATE_SUCCESS or error code
 */
tn_state_error_t tn_mps_truncate(tn_mps_state_t *state,
                                  uint32_t max_bond,
                                  double *truncation_error);

/**
 * @brief Truncate single bond
 *
 * @param state MPS state
 * @param bond Bond index
 * @param max_dim Maximum dimension for this bond
 * @param truncation_error Output: truncation error
 * @return TN_STATE_SUCCESS or error code
 */
tn_state_error_t tn_mps_truncate_bond(tn_mps_state_t *state,
                                       uint32_t bond,
                                       uint32_t max_dim,
                                       double *truncation_error);

/**
 * @brief Grow bond dimension to allow more entanglement
 *
 * @param state MPS state
 * @param bond Bond index
 * @param new_dim New dimension (must be >= current)
 * @return TN_STATE_SUCCESS or error code
 */
tn_state_error_t tn_mps_grow_bond(tn_mps_state_t *state,
                                   uint32_t bond,
                                   uint32_t new_dim);

// ============================================================================
// OVERLAP AND FIDELITY
// ============================================================================

/**
 * @brief Compute overlap <psi1|psi2>
 *
 * @param state1 First MPS state
 * @param state2 Second MPS state
 * @return Complex overlap
 */
double complex tn_mps_overlap(const tn_mps_state_t *state1,
                               const tn_mps_state_t *state2);

/**
 * @brief Compute fidelity |<psi1|psi2>|^2
 *
 * @param state1 First MPS state
 * @param state2 Second MPS state
 * @return Fidelity (0 to 1)
 */
double tn_mps_fidelity(const tn_mps_state_t *state1,
                        const tn_mps_state_t *state2);

// ============================================================================
// UTILITIES
// ============================================================================

/**
 * @brief Get error string
 *
 * @param error Error code
 * @return Human-readable error string
 */
const char *tn_state_error_string(tn_state_error_t error);

/**
 * @brief Estimate memory for MPS
 *
 * @param num_qubits Number of qubits
 * @param bond_dim Uniform bond dimension
 * @return Estimated memory in bytes
 */
uint64_t tn_mps_estimate_memory(uint32_t num_qubits, uint32_t bond_dim);

/**
 * @brief Check if MPS is valid
 *
 * Validates tensor shapes and bond consistency.
 *
 * @param state MPS state
 * @return TN_STATE_SUCCESS if valid
 */
tn_state_error_t tn_mps_validate(const tn_mps_state_t *state);

#ifdef __cplusplus
}
#endif

#endif /* TN_STATE_H */
