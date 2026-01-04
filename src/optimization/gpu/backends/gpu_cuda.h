/**
 * @file gpu_cuda.h
 * @brief NVIDIA CUDA GPU compute backend
 *
 * Native CUDA implementation for quantum computing operations.
 * Provides efficient GPU acceleration on NVIDIA hardware.
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef GPU_CUDA_H
#define GPU_CUDA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CUDA BACKEND TYPES
// ============================================================================

/**
 * @brief Opaque CUDA context structure
 */
typedef struct cuda_compute_ctx cuda_compute_ctx_t;

/**
 * @brief Opaque CUDA buffer structure
 */
typedef struct cuda_buffer cuda_buffer_t;

/**
 * @brief CUDA device capabilities
 */
typedef struct {
    char device_name[256];           /**< Device name string */
    int compute_capability_major;     /**< Compute capability major version */
    int compute_capability_minor;     /**< Compute capability minor version */
    uint64_t global_memory;          /**< Global memory in bytes */
    uint64_t shared_memory_per_block; /**< Shared memory per block */
    uint32_t max_threads_per_block;  /**< Maximum threads per block */
    uint32_t max_block_dim[3];       /**< Maximum block dimensions */
    uint32_t max_grid_dim[3];        /**< Maximum grid dimensions */
    uint32_t warp_size;              /**< Warp size (typically 32) */
    uint32_t multiprocessor_count;   /**< Number of SMs */
    int supports_unified_memory;      /**< Unified memory support */
    double memory_bandwidth_gbps;    /**< Memory bandwidth */
} cuda_capabilities_t;

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * @brief Check if CUDA is available on this system
 * @return 1 if available, 0 otherwise
 */
int cuda_is_available(void);

/**
 * @brief Initialize CUDA context
 *
 * @param device_id GPU device index (0 for default)
 * @return CUDA context or NULL on failure
 */
cuda_compute_ctx_t* cuda_context_create(int device_id);

/**
 * @brief Free CUDA context
 * @param ctx CUDA context
 */
void cuda_context_free(cuda_compute_ctx_t* ctx);

/**
 * @brief Get CUDA device capabilities
 *
 * @param ctx CUDA context
 * @param caps Output capabilities structure
 * @return 0 on success, error code otherwise
 */
int cuda_get_capabilities(cuda_compute_ctx_t* ctx, cuda_capabilities_t* caps);

/**
 * @brief Get device information
 *
 * @param ctx CUDA context
 * @param device_name Output device name buffer
 * @param max_threads_per_block Output max threads per block
 * @param multiprocessor_count Output SM count
 */
void cuda_get_device_info(cuda_compute_ctx_t* ctx, char* device_name,
                          uint32_t* max_threads_per_block,
                          uint32_t* multiprocessor_count);

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

/**
 * @brief Allocate CUDA buffer
 *
 * @param ctx CUDA context
 * @param size Buffer size in bytes
 * @return CUDA buffer or NULL on failure
 */
cuda_buffer_t* cuda_buffer_create(cuda_compute_ctx_t* ctx, size_t size);

/**
 * @brief Create CUDA buffer from host data
 *
 * @param ctx CUDA context
 * @param data Host data pointer
 * @param size Buffer size in bytes
 * @return CUDA buffer or NULL on failure
 */
cuda_buffer_t* cuda_buffer_create_from_data(cuda_compute_ctx_t* ctx,
                                            void* data, size_t size);

/**
 * @brief Get mapped host pointer for CUDA buffer
 *
 * Uses unified memory for zero-copy access when available.
 *
 * @param buffer CUDA buffer
 * @return Host pointer or NULL if not mappable
 */
void* cuda_buffer_contents(cuda_buffer_t* buffer);

/**
 * @brief Copy data to CUDA buffer
 *
 * @param ctx CUDA context
 * @param buffer CUDA buffer
 * @param data Source data
 * @param size Bytes to copy
 * @return 0 on success, error code otherwise
 */
int cuda_buffer_write(cuda_compute_ctx_t* ctx, cuda_buffer_t* buffer,
                      const void* data, size_t size);

/**
 * @brief Copy data from CUDA buffer
 *
 * @param ctx CUDA context
 * @param buffer CUDA buffer
 * @param data Destination
 * @param size Bytes to copy
 * @return 0 on success, error code otherwise
 */
int cuda_buffer_read(cuda_compute_ctx_t* ctx, cuda_buffer_t* buffer,
                     void* data, size_t size);

/**
 * @brief Free CUDA buffer
 * @param buffer CUDA buffer
 */
void cuda_buffer_free(cuda_buffer_t* buffer);

// ============================================================================
// QUANTUM GATE OPERATIONS
// ============================================================================

/**
 * @brief Apply Hadamard gate to single qubit
 *
 * @param ctx CUDA context
 * @param amplitudes Amplitude buffer
 * @param qubit_index Qubit index
 * @param state_dim State dimension (2^n)
 * @return 0 on success, error code otherwise
 */
int cuda_hadamard(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                  uint32_t qubit_index, uint64_t state_dim);

/**
 * @brief Apply Hadamard to all qubits
 *
 * @param ctx CUDA context
 * @param amplitudes Amplitude buffer
 * @param num_qubits Number of qubits
 * @param state_dim State dimension
 * @return 0 on success, error code otherwise
 */
int cuda_hadamard_all(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                      uint32_t num_qubits, uint64_t state_dim);

/**
 * @brief Apply Pauli-X gate
 *
 * @param ctx CUDA context
 * @param amplitudes Amplitude buffer
 * @param qubit_index Qubit index
 * @param state_dim State dimension
 * @return 0 on success, error code otherwise
 */
int cuda_pauli_x(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                 uint32_t qubit_index, uint64_t state_dim);

/**
 * @brief Apply Pauli-Y gate
 *
 * @param ctx CUDA context
 * @param amplitudes Amplitude buffer
 * @param qubit_index Qubit index
 * @param state_dim State dimension
 * @return 0 on success, error code otherwise
 */
int cuda_pauli_y(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                 uint32_t qubit_index, uint64_t state_dim);

/**
 * @brief Apply Pauli-Z gate
 *
 * @param ctx CUDA context
 * @param amplitudes Amplitude buffer
 * @param qubit_index Qubit index
 * @param state_dim State dimension
 * @return 0 on success, error code otherwise
 */
int cuda_pauli_z(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                 uint32_t qubit_index, uint64_t state_dim);

/**
 * @brief Apply phase gate
 *
 * @param ctx CUDA context
 * @param amplitudes Amplitude buffer
 * @param qubit_index Qubit index
 * @param phase Phase angle in radians
 * @param state_dim State dimension
 * @return 0 on success, error code otherwise
 */
int cuda_phase_gate(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                    uint32_t qubit_index, double phase, uint64_t state_dim);

/**
 * @brief Apply CNOT gate
 *
 * @param ctx CUDA context
 * @param amplitudes Amplitude buffer
 * @param control_qubit Control qubit index
 * @param target_qubit Target qubit index
 * @param state_dim State dimension
 * @return 0 on success, error code otherwise
 */
int cuda_cnot(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
              uint32_t control_qubit, uint32_t target_qubit, uint64_t state_dim);

// ============================================================================
// GROVER'S ALGORITHM
// ============================================================================

/**
 * @brief Apply oracle (phase flip on target state)
 *
 * @param ctx CUDA context
 * @param amplitudes Amplitude buffer
 * @param target Marked state
 * @param state_dim State dimension
 * @return 0 on success, error code otherwise
 */
int cuda_oracle_single_target(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                              uint64_t target, uint64_t state_dim);

/**
 * @brief Apply sparse oracle (multiple targets)
 *
 * @param ctx CUDA context
 * @param amplitudes Amplitude buffer
 * @param targets Array of marked states
 * @param num_targets Number of targets
 * @param state_dim State dimension
 * @return 0 on success, error code otherwise
 */
int cuda_sparse_oracle(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                       const uint64_t* targets, uint32_t num_targets,
                       uint64_t state_dim);

/**
 * @brief Apply Grover diffusion operator
 *
 * @param ctx CUDA context
 * @param amplitudes Amplitude buffer
 * @param num_qubits Number of qubits
 * @param state_dim State dimension
 * @return 0 on success, error code otherwise
 */
int cuda_grover_diffusion(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                          uint32_t num_qubits, uint64_t state_dim);

/**
 * @brief Fused Grover iteration (oracle + diffusion)
 *
 * @param ctx CUDA context
 * @param amplitudes Amplitude buffer
 * @param target Marked state
 * @param num_qubits Number of qubits
 * @param state_dim State dimension
 * @return 0 on success, error code otherwise
 */
int cuda_grover_iteration(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                          uint64_t target, uint32_t num_qubits, uint64_t state_dim);

/**
 * @brief Batch Grover search (multiple parallel searches)
 *
 * @param ctx CUDA context
 * @param batch_states Combined state buffer for all searches
 * @param targets Target states (one per search)
 * @param results Output results array
 * @param num_searches Number of parallel searches
 * @param num_qubits Qubits per search
 * @param num_iterations Iterations per search
 * @return 0 on success, error code otherwise
 */
int cuda_grover_batch_search(cuda_compute_ctx_t* ctx, cuda_buffer_t* batch_states,
                             const uint64_t* targets, uint64_t* results,
                             uint32_t num_searches, uint32_t num_qubits,
                             uint32_t num_iterations);

// ============================================================================
// MEASUREMENT & UTILITIES
// ============================================================================

/**
 * @brief Compute probability distribution
 *
 * @param ctx CUDA context
 * @param amplitudes Amplitude buffer
 * @param probabilities Output probability buffer
 * @param state_dim State dimension
 * @return 0 on success, error code otherwise
 */
int cuda_compute_probabilities(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                               cuda_buffer_t* probabilities, uint64_t state_dim);

/**
 * @brief Normalize quantum state
 *
 * @param ctx CUDA context
 * @param amplitudes Amplitude buffer
 * @param norm Normalization factor
 * @param state_dim State dimension
 * @return 0 on success, error code otherwise
 */
int cuda_normalize_state(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                         double norm, uint64_t state_dim);

/**
 * @brief Compute sum of squared magnitudes
 *
 * @param ctx CUDA context
 * @param amplitudes Amplitude buffer
 * @param state_dim State dimension
 * @param result Output sum
 * @return 0 on success, error code otherwise
 */
int cuda_sum_squared_magnitudes(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                                uint64_t state_dim, double* result);

// ============================================================================
// SYNCHRONIZATION
// ============================================================================

/**
 * @brief Wait for all CUDA operations to complete
 * @param ctx CUDA context
 */
void cuda_synchronize(cuda_compute_ctx_t* ctx);

/**
 * @brief Get execution time of last operation
 *
 * @param ctx CUDA context
 * @return Execution time in seconds
 */
double cuda_get_last_execution_time(cuda_compute_ctx_t* ctx);

/**
 * @brief Get CUDA error string
 *
 * @param ctx CUDA context
 * @return Error message or NULL
 */
const char* cuda_get_error_string(cuda_compute_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif /* GPU_CUDA_H */
