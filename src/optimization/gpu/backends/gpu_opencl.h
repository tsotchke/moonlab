/**
 * @file gpu_opencl.h
 * @brief OpenCL GPU backend for quantum simulation
 *
 * Cross-platform GPU acceleration using OpenCL 1.2+
 * Supports NVIDIA, AMD, Intel, and Apple GPUs (deprecated on macOS)
 *
 * @stability beta
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef GPU_OPENCL_H
#define GPU_OPENCL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct opencl_compute_ctx opencl_compute_ctx_t;
typedef struct opencl_buffer opencl_buffer_t;

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * @brief Check if OpenCL is available on this system
 * @return 1 if available, 0 if not
 */
int opencl_is_available(void);

/**
 * @brief Initialize OpenCL compute context
 *
 * Selects the best available GPU device.
 * Priority: Discrete GPU > Integrated GPU > CPU
 *
 * @return Compute context or NULL on failure
 */
opencl_compute_ctx_t* opencl_compute_init(void);

/**
 * @brief Initialize with specific platform and device
 *
 * @param platform_index Platform index (0-based)
 * @param device_index Device index (0-based)
 * @return Compute context or NULL on failure
 */
opencl_compute_ctx_t* opencl_compute_init_device(int platform_index, int device_index);

/**
 * @brief Free OpenCL compute context
 * @param ctx Context to free
 */
void opencl_compute_free(opencl_compute_ctx_t* ctx);

/**
 * @brief Get device information
 *
 * @param ctx Compute context
 * @param name Output: Device name (256 chars max)
 * @param max_work_group_size Output: Maximum work group size
 * @param compute_units Output: Number of compute units
 */
void opencl_get_device_info(
    opencl_compute_ctx_t* ctx,
    char* name,
    uint32_t* max_work_group_size,
    uint32_t* compute_units
);

/**
 * @brief Print device information
 * @param ctx Compute context
 */
void opencl_print_device_info(opencl_compute_ctx_t* ctx);

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

/**
 * @brief Create GPU buffer
 *
 * @param ctx Compute context
 * @param size Buffer size in bytes
 * @return Buffer handle or NULL on failure
 */
opencl_buffer_t* opencl_buffer_create(opencl_compute_ctx_t* ctx, size_t size);

/**
 * @brief Create GPU buffer from existing data
 *
 * @param ctx Compute context
 * @param data Host data to copy
 * @param size Size in bytes
 * @return Buffer handle or NULL on failure
 */
opencl_buffer_t* opencl_buffer_create_from_data(
    opencl_compute_ctx_t* ctx,
    const void* data,
    size_t size
);

/**
 * @brief Read buffer contents to host
 *
 * @param ctx Compute context
 * @param buffer GPU buffer
 * @param dst Host destination
 * @param size Bytes to read
 * @return 0 on success, -1 on failure
 */
int opencl_buffer_read(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* buffer,
    void* dst,
    size_t size
);

/**
 * @brief Write host data to buffer
 *
 * @param ctx Compute context
 * @param buffer GPU buffer
 * @param src Host source
 * @param size Bytes to write
 * @return 0 on success, -1 on failure
 */
int opencl_buffer_write(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* buffer,
    const void* src,
    size_t size
);

/**
 * @brief Free GPU buffer
 * @param buffer Buffer to free
 */
void opencl_buffer_free(opencl_buffer_t* buffer);

// ============================================================================
// QUANTUM GATE OPERATIONS
// ============================================================================

/**
 * @brief Apply Hadamard gate to single qubit
 *
 * @param ctx Compute context
 * @param amplitudes State amplitudes buffer
 * @param qubit_index Target qubit
 * @param state_dim State dimension (2^n)
 * @return 0 on success, -1 on failure
 */
int opencl_hadamard(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
);

/**
 * @brief Apply Hadamard to all qubits
 *
 * Creates uniform superposition from |0...0>
 *
 * @param ctx Compute context
 * @param amplitudes State amplitudes buffer
 * @param num_qubits Number of qubits
 * @param state_dim State dimension (2^n)
 * @return 0 on success, -1 on failure
 */
int opencl_hadamard_all(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint32_t num_qubits,
    uint64_t state_dim
);

/**
 * @brief Apply Pauli X gate (bit flip)
 *
 * @param ctx Compute context
 * @param amplitudes State amplitudes buffer
 * @param qubit_index Target qubit
 * @param state_dim State dimension
 * @return 0 on success, -1 on failure
 */
int opencl_pauli_x(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
);

/**
 * @brief Apply Pauli Y gate
 *
 * @param ctx Compute context
 * @param amplitudes State amplitudes buffer
 * @param qubit_index Target qubit
 * @param state_dim State dimension
 * @return 0 on success, -1 on failure
 */
int opencl_pauli_y(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
);

/**
 * @brief Apply Pauli Z gate (phase flip)
 *
 * @param ctx Compute context
 * @param amplitudes State amplitudes buffer
 * @param qubit_index Target qubit
 * @param state_dim State dimension
 * @return 0 on success, -1 on failure
 */
int opencl_pauli_z(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
);

/**
 * @brief Apply phase rotation gate
 *
 * @param ctx Compute context
 * @param amplitudes State amplitudes buffer
 * @param qubit_index Target qubit
 * @param phase Phase angle in radians
 * @param state_dim State dimension
 * @return 0 on success, -1 on failure
 */
int opencl_phase(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint32_t qubit_index,
    float phase,
    uint64_t state_dim
);

/**
 * @brief Apply CNOT (controlled-X) gate
 *
 * @param ctx Compute context
 * @param amplitudes State amplitudes buffer
 * @param control_qubit Control qubit index
 * @param target_qubit Target qubit index
 * @param state_dim State dimension
 * @return 0 on success, -1 on failure
 */
int opencl_cnot(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint32_t control_qubit,
    uint32_t target_qubit,
    uint64_t state_dim
);

// ============================================================================
// GROVER'S ALGORITHM OPERATIONS
// ============================================================================

/**
 * @brief Apply oracle for single target state
 *
 * @param ctx Compute context
 * @param amplitudes State amplitudes buffer
 * @param target Target state index
 * @param state_dim State dimension
 * @return 0 on success, -1 on failure
 */
int opencl_oracle(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint64_t target,
    uint64_t state_dim
);

/**
 * @brief Apply oracle for multiple target states
 *
 * @param ctx Compute context
 * @param amplitudes State amplitudes buffer
 * @param targets Array of target states
 * @param num_targets Number of targets
 * @param state_dim State dimension
 * @return 0 on success, -1 on failure
 */
int opencl_oracle_multi(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    const uint64_t* targets,
    uint32_t num_targets,
    uint64_t state_dim
);

/**
 * @brief Apply Grover diffusion operator
 *
 * D = 2|s><s| - I (inversion about average)
 *
 * @param ctx Compute context
 * @param amplitudes State amplitudes buffer
 * @param num_qubits Number of qubits
 * @param state_dim State dimension
 * @return 0 on success, -1 on failure
 */
int opencl_grover_diffusion(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint32_t num_qubits,
    uint64_t state_dim
);

/**
 * @brief Execute complete Grover search
 *
 * @param ctx Compute context
 * @param amplitudes State amplitudes buffer
 * @param target Target state
 * @param num_qubits Number of qubits
 * @param num_iterations Number of Grover iterations
 * @return 0 on success, -1 on failure
 */
int opencl_grover_search(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint64_t target,
    uint32_t num_qubits,
    uint32_t num_iterations
);

/**
 * @brief Execute batch Grover searches in parallel
 *
 * Runs multiple independent searches simultaneously
 *
 * @param ctx Compute context
 * @param batch_states Batch state buffer
 * @param targets Array of targets (one per search)
 * @param results Output array for results
 * @param num_searches Number of parallel searches
 * @param num_qubits Qubits per search
 * @param num_iterations Iterations per search
 * @return 0 on success, -1 on failure
 */
int opencl_grover_batch_search(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* batch_states,
    const uint64_t* targets,
    uint64_t* results,
    uint32_t num_searches,
    uint32_t num_qubits,
    uint32_t num_iterations
);

// ============================================================================
// MEASUREMENT & NORMALIZATION
// ============================================================================

/**
 * @brief Compute measurement probabilities
 *
 * @param ctx Compute context
 * @param amplitudes State amplitudes buffer
 * @param probabilities Output probabilities buffer
 * @param state_dim State dimension
 * @return 0 on success, -1 on failure
 */
int opencl_compute_probabilities(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    opencl_buffer_t* probabilities,
    uint64_t state_dim
);

/**
 * @brief Normalize quantum state
 *
 * @param ctx Compute context
 * @param amplitudes State amplitudes buffer
 * @param norm Normalization factor
 * @param state_dim State dimension
 * @return 0 on success, -1 on failure
 */
int opencl_normalize(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    float norm,
    uint64_t state_dim
);

// ============================================================================
// SYNCHRONIZATION & PERFORMANCE
// ============================================================================

/**
 * @brief Wait for all GPU operations to complete
 * @param ctx Compute context
 */
void opencl_wait_completion(opencl_compute_ctx_t* ctx);

/**
 * @brief Get last operation execution time
 * @param ctx Compute context
 * @return Execution time in seconds
 */
double opencl_get_last_execution_time(opencl_compute_ctx_t* ctx);

/**
 * @brief Enable/disable performance monitoring
 * @param ctx Compute context
 * @param enable 1 to enable, 0 to disable
 */
void opencl_set_performance_monitoring(opencl_compute_ctx_t* ctx, int enable);

/**
 * @brief Get last error message
 * @param ctx Compute context
 * @return Error string or "No error"
 */
const char* opencl_get_error(opencl_compute_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif /* GPU_OPENCL_H */
