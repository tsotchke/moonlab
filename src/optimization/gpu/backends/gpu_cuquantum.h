/**
 * @file gpu_cuquantum.h
 * @brief NVIDIA cuQuantum GPU compute backend
 *
 * High-performance quantum simulation using NVIDIA cuQuantum SDK:
 * - cuStateVec for state vector simulation
 * - cuTensorNet for tensor network contraction (optional)
 *
 * Reference: https://docs.nvidia.com/cuda/cuquantum/latest/index.html
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef GPU_CUQUANTUM_H
#define GPU_CUQUANTUM_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CUQUANTUM BACKEND TYPES
// ============================================================================

/**
 * @brief Opaque cuQuantum context structure
 */
typedef struct cuquantum_compute_ctx cuquantum_compute_ctx_t;

/**
 * @brief Opaque cuQuantum buffer structure
 */
typedef struct cuquantum_buffer cuquantum_buffer_t;

/**
 * @brief cuQuantum device capabilities
 */
typedef struct {
    char device_name[256];           /**< Device name string */
    int compute_capability_major;     /**< Compute capability major version */
    int compute_capability_minor;     /**< Compute capability minor version */
    uint64_t global_memory;          /**< Global memory in bytes */
    uint32_t multiprocessor_count;   /**< Number of SMs */
    int supports_custatevec;          /**< cuStateVec available */
    int supports_cutensornet;         /**< cuTensorNet available */
    int max_qubits;                   /**< Maximum qubits for state vector */
    double memory_bandwidth_gbps;    /**< Memory bandwidth */
} cuquantum_capabilities_t;

/**
 * @brief cuQuantum workspace configuration
 */
typedef struct {
    size_t workspace_size;           /**< Workspace size in bytes */
    int use_host_memory;             /**< Use host memory for workspace */
    int use_managed_memory;          /**< Use CUDA managed memory */
} cuquantum_config_t;

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * @brief Check if cuQuantum is available on this system
 * @return 1 if available, 0 otherwise
 */
int cuquantum_is_available(void);

/**
 * @brief Initialize cuQuantum context
 *
 * @param device_id GPU device index (0 for default)
 * @param config Optional configuration (NULL for defaults)
 * @return cuQuantum context or NULL on failure
 */
cuquantum_compute_ctx_t* cuquantum_context_create(int device_id,
                                                   const cuquantum_config_t* config);

/**
 * @brief Free cuQuantum context
 * @param ctx cuQuantum context
 */
void cuquantum_context_free(cuquantum_compute_ctx_t* ctx);

/**
 * @brief Get cuQuantum device capabilities
 *
 * @param ctx cuQuantum context
 * @param caps Output capabilities structure
 * @return 0 on success, error code otherwise
 */
int cuquantum_get_capabilities(cuquantum_compute_ctx_t* ctx,
                               cuquantum_capabilities_t* caps);

/**
 * @brief Get device information
 *
 * @param ctx cuQuantum context
 * @param device_name Output device name buffer
 * @param max_qubits Output maximum qubits
 * @param multiprocessor_count Output SM count
 */
void cuquantum_get_device_info(cuquantum_compute_ctx_t* ctx, char* device_name,
                               uint32_t* max_qubits,
                               uint32_t* multiprocessor_count);

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

/**
 * @brief Allocate cuQuantum buffer for state vector
 *
 * @param ctx cuQuantum context
 * @param num_qubits Number of qubits
 * @return cuQuantum buffer or NULL on failure
 */
cuquantum_buffer_t* cuquantum_statevec_create(cuquantum_compute_ctx_t* ctx,
                                               uint32_t num_qubits);

/**
 * @brief Initialize state vector to |0⟩
 *
 * @param ctx cuQuantum context
 * @param buffer State vector buffer
 * @param num_qubits Number of qubits
 * @return 0 on success, error code otherwise
 */
int cuquantum_statevec_init_zero(cuquantum_compute_ctx_t* ctx,
                                 cuquantum_buffer_t* buffer,
                                 uint32_t num_qubits);

/**
 * @brief Initialize state vector to uniform superposition
 *
 * @param ctx cuQuantum context
 * @param buffer State vector buffer
 * @param num_qubits Number of qubits
 * @return 0 on success, error code otherwise
 */
int cuquantum_statevec_init_uniform(cuquantum_compute_ctx_t* ctx,
                                    cuquantum_buffer_t* buffer,
                                    uint32_t num_qubits);

/**
 * @brief Get GPU device pointer for buffer
 *
 * @param buffer cuQuantum buffer
 * @return Device pointer or NULL
 */
void* cuquantum_buffer_device_ptr(cuquantum_buffer_t* buffer);

/**
 * @brief Copy data to cuQuantum buffer
 *
 * @param ctx cuQuantum context
 * @param buffer cuQuantum buffer
 * @param data Source data
 * @param size Bytes to copy
 * @return 0 on success, error code otherwise
 */
int cuquantum_buffer_write(cuquantum_compute_ctx_t* ctx,
                           cuquantum_buffer_t* buffer,
                           const void* data, size_t size);

/**
 * @brief Copy data from cuQuantum buffer
 *
 * @param ctx cuQuantum context
 * @param buffer cuQuantum buffer
 * @param data Destination
 * @param size Bytes to copy
 * @return 0 on success, error code otherwise
 */
int cuquantum_buffer_read(cuquantum_compute_ctx_t* ctx,
                          cuquantum_buffer_t* buffer,
                          void* data, size_t size);

/**
 * @brief Free cuQuantum buffer
 * @param buffer cuQuantum buffer
 */
void cuquantum_buffer_free(cuquantum_buffer_t* buffer);

// ============================================================================
// QUANTUM GATE OPERATIONS (cuStateVec)
// ============================================================================

/**
 * @brief Apply single-qubit gate matrix
 *
 * Uses cuStateVec for optimal performance.
 *
 * @param ctx cuQuantum context
 * @param statevec State vector buffer
 * @param num_qubits Number of qubits
 * @param target Target qubit index
 * @param matrix 2x2 gate matrix (row-major, complex doubles)
 * @return 0 on success, error code otherwise
 */
int cuquantum_apply_gate_1q(cuquantum_compute_ctx_t* ctx,
                            cuquantum_buffer_t* statevec,
                            uint32_t num_qubits,
                            uint32_t target,
                            const void* matrix);

/**
 * @brief Apply two-qubit gate matrix
 *
 * @param ctx cuQuantum context
 * @param statevec State vector buffer
 * @param num_qubits Number of qubits
 * @param control Control qubit index
 * @param target Target qubit index
 * @param matrix 4x4 gate matrix (row-major, complex doubles)
 * @return 0 on success, error code otherwise
 */
int cuquantum_apply_gate_2q(cuquantum_compute_ctx_t* ctx,
                            cuquantum_buffer_t* statevec,
                            uint32_t num_qubits,
                            uint32_t control,
                            uint32_t target,
                            const void* matrix);

/**
 * @brief Apply Hadamard gate
 *
 * @param ctx cuQuantum context
 * @param statevec State vector buffer
 * @param num_qubits Number of qubits
 * @param target Target qubit
 * @return 0 on success, error code otherwise
 */
int cuquantum_hadamard(cuquantum_compute_ctx_t* ctx,
                       cuquantum_buffer_t* statevec,
                       uint32_t num_qubits,
                       uint32_t target);

/**
 * @brief Apply Hadamard to all qubits
 *
 * @param ctx cuQuantum context
 * @param statevec State vector buffer
 * @param num_qubits Number of qubits
 * @return 0 on success, error code otherwise
 */
int cuquantum_hadamard_all(cuquantum_compute_ctx_t* ctx,
                           cuquantum_buffer_t* statevec,
                           uint32_t num_qubits);

/**
 * @brief Apply Pauli-X gate
 *
 * @param ctx cuQuantum context
 * @param statevec State vector buffer
 * @param num_qubits Number of qubits
 * @param target Target qubit
 * @return 0 on success, error code otherwise
 */
int cuquantum_pauli_x(cuquantum_compute_ctx_t* ctx,
                      cuquantum_buffer_t* statevec,
                      uint32_t num_qubits,
                      uint32_t target);

/**
 * @brief Apply Pauli-Y gate
 *
 * @param ctx cuQuantum context
 * @param statevec State vector buffer
 * @param num_qubits Number of qubits
 * @param target Target qubit
 * @return 0 on success, error code otherwise
 */
int cuquantum_pauli_y(cuquantum_compute_ctx_t* ctx,
                      cuquantum_buffer_t* statevec,
                      uint32_t num_qubits,
                      uint32_t target);

/**
 * @brief Apply Pauli-Z gate
 *
 * @param ctx cuQuantum context
 * @param statevec State vector buffer
 * @param num_qubits Number of qubits
 * @param target Target qubit
 * @return 0 on success, error code otherwise
 */
int cuquantum_pauli_z(cuquantum_compute_ctx_t* ctx,
                      cuquantum_buffer_t* statevec,
                      uint32_t num_qubits,
                      uint32_t target);

/**
 * @brief Apply phase rotation gate (Rz)
 *
 * @param ctx cuQuantum context
 * @param statevec State vector buffer
 * @param num_qubits Number of qubits
 * @param target Target qubit
 * @param phase Phase angle in radians
 * @return 0 on success, error code otherwise
 */
int cuquantum_phase_gate(cuquantum_compute_ctx_t* ctx,
                         cuquantum_buffer_t* statevec,
                         uint32_t num_qubits,
                         uint32_t target,
                         double phase);

/**
 * @brief Apply CNOT gate
 *
 * @param ctx cuQuantum context
 * @param statevec State vector buffer
 * @param num_qubits Number of qubits
 * @param control Control qubit
 * @param target Target qubit
 * @return 0 on success, error code otherwise
 */
int cuquantum_cnot(cuquantum_compute_ctx_t* ctx,
                   cuquantum_buffer_t* statevec,
                   uint32_t num_qubits,
                   uint32_t control,
                   uint32_t target);

/**
 * @brief Apply controlled-Z gate
 *
 * @param ctx cuQuantum context
 * @param statevec State vector buffer
 * @param num_qubits Number of qubits
 * @param control Control qubit
 * @param target Target qubit
 * @return 0 on success, error code otherwise
 */
int cuquantum_cz(cuquantum_compute_ctx_t* ctx,
                 cuquantum_buffer_t* statevec,
                 uint32_t num_qubits,
                 uint32_t control,
                 uint32_t target);

// ============================================================================
// GROVER'S ALGORITHM
// ============================================================================

/**
 * @brief Apply oracle (phase flip on target state)
 *
 * @param ctx cuQuantum context
 * @param statevec State vector buffer
 * @param num_qubits Number of qubits
 * @param target Marked state
 * @return 0 on success, error code otherwise
 */
int cuquantum_oracle_single_target(cuquantum_compute_ctx_t* ctx,
                                   cuquantum_buffer_t* statevec,
                                   uint32_t num_qubits,
                                   uint64_t target);

/**
 * @brief Apply sparse oracle (multiple targets)
 *
 * @param ctx cuQuantum context
 * @param statevec State vector buffer
 * @param num_qubits Number of qubits
 * @param targets Array of marked states
 * @param num_targets Number of targets
 * @return 0 on success, error code otherwise
 */
int cuquantum_sparse_oracle(cuquantum_compute_ctx_t* ctx,
                            cuquantum_buffer_t* statevec,
                            uint32_t num_qubits,
                            const uint64_t* targets,
                            uint32_t num_targets);

/**
 * @brief Apply Grover diffusion operator
 *
 * @param ctx cuQuantum context
 * @param statevec State vector buffer
 * @param num_qubits Number of qubits
 * @return 0 on success, error code otherwise
 */
int cuquantum_grover_diffusion(cuquantum_compute_ctx_t* ctx,
                               cuquantum_buffer_t* statevec,
                               uint32_t num_qubits);

/**
 * @brief Complete Grover iteration (oracle + diffusion)
 *
 * @param ctx cuQuantum context
 * @param statevec State vector buffer
 * @param num_qubits Number of qubits
 * @param target Marked state
 * @return 0 on success, error code otherwise
 */
int cuquantum_grover_iteration(cuquantum_compute_ctx_t* ctx,
                               cuquantum_buffer_t* statevec,
                               uint32_t num_qubits,
                               uint64_t target);

/**
 * @brief Batch Grover search (multiple parallel searches)
 *
 * @param ctx cuQuantum context
 * @param batch_states Combined state buffer for all searches
 * @param targets Target states (one per search)
 * @param results Output results array
 * @param num_searches Number of parallel searches
 * @param num_qubits Qubits per search
 * @param num_iterations Iterations per search
 * @return 0 on success, error code otherwise
 */
int cuquantum_grover_batch_search(cuquantum_compute_ctx_t* ctx,
                                  cuquantum_buffer_t* batch_states,
                                  const uint64_t* targets,
                                  uint64_t* results,
                                  uint32_t num_searches,
                                  uint32_t num_qubits,
                                  uint32_t num_iterations);

// ============================================================================
// MEASUREMENT & UTILITIES
// ============================================================================

/**
 * @brief Compute probability distribution
 *
 * @param ctx cuQuantum context
 * @param statevec State vector buffer
 * @param num_qubits Number of qubits
 * @param probabilities Output probability array
 * @return 0 on success, error code otherwise
 */
int cuquantum_compute_probabilities(cuquantum_compute_ctx_t* ctx,
                                    cuquantum_buffer_t* statevec,
                                    uint32_t num_qubits,
                                    double* probabilities);

/**
 * @brief Sample measurement result
 *
 * Uses cuStateVec sampling for efficient measurement.
 *
 * @param ctx cuQuantum context
 * @param statevec State vector buffer
 * @param num_qubits Number of qubits
 * @param result Output measurement result
 * @return 0 on success, error code otherwise
 */
int cuquantum_measure(cuquantum_compute_ctx_t* ctx,
                      cuquantum_buffer_t* statevec,
                      uint32_t num_qubits,
                      uint64_t* result);

/**
 * @brief Compute expectation value
 *
 * @param ctx cuQuantum context
 * @param statevec State vector buffer
 * @param num_qubits Number of qubits
 * @param observable Pauli string (e.g., "XYZZ")
 * @param result Output expectation value
 * @return 0 on success, error code otherwise
 */
int cuquantum_expectation_value(cuquantum_compute_ctx_t* ctx,
                                cuquantum_buffer_t* statevec,
                                uint32_t num_qubits,
                                const char* observable,
                                double* result);

/**
 * @brief Normalize state vector
 *
 * @param ctx cuQuantum context
 * @param statevec State vector buffer
 * @param num_qubits Number of qubits
 * @return 0 on success, error code otherwise
 */
int cuquantum_normalize(cuquantum_compute_ctx_t* ctx,
                        cuquantum_buffer_t* statevec,
                        uint32_t num_qubits);

// ============================================================================
// SYNCHRONIZATION
// ============================================================================

/**
 * @brief Wait for all cuQuantum operations to complete
 * @param ctx cuQuantum context
 */
void cuquantum_synchronize(cuquantum_compute_ctx_t* ctx);

/**
 * @brief Get execution time of last operation
 *
 * @param ctx cuQuantum context
 * @return Execution time in seconds
 */
double cuquantum_get_last_execution_time(cuquantum_compute_ctx_t* ctx);

/**
 * @brief Get cuQuantum error string
 *
 * @param ctx cuQuantum context
 * @return Error message or NULL
 */
const char* cuquantum_get_error_string(cuquantum_compute_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif /* GPU_CUQUANTUM_H */
