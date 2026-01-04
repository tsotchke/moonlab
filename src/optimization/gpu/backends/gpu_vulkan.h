/**
 * @file gpu_vulkan.h
 * @brief Vulkan GPU backend for quantum simulation
 *
 * Cross-platform GPU acceleration using Vulkan Compute
 * Supports NVIDIA, AMD, Intel, and mobile GPUs
 *
 * @stability beta
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef GPU_VULKAN_H
#define GPU_VULKAN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct vulkan_compute_ctx vulkan_compute_ctx_t;
typedef struct vulkan_buffer vulkan_buffer_t;

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * @brief Check if Vulkan is available on this system
 * @return 1 if available, 0 if not
 */
int vulkan_is_available(void);

/**
 * @brief Initialize Vulkan compute context
 *
 * Creates Vulkan instance, selects best physical device,
 * creates logical device with compute queue.
 *
 * @return Compute context or NULL on failure
 */
vulkan_compute_ctx_t* vulkan_compute_init(void);

/**
 * @brief Initialize with specific device
 *
 * @param device_index Physical device index (0-based)
 * @return Compute context or NULL on failure
 */
vulkan_compute_ctx_t* vulkan_compute_init_device(int device_index);

/**
 * @brief Free Vulkan compute context
 * @param ctx Context to free
 */
void vulkan_compute_free(vulkan_compute_ctx_t* ctx);

/**
 * @brief Get device information
 *
 * @param ctx Compute context
 * @param name Output: Device name (256 chars max)
 * @param max_work_group_size Output: Maximum work group size
 * @param compute_units Output: Number of compute units
 */
void vulkan_get_device_info(
    vulkan_compute_ctx_t* ctx,
    char* name,
    uint32_t* max_work_group_size,
    uint32_t* compute_units
);

/**
 * @brief Print device information
 * @param ctx Compute context
 */
void vulkan_print_device_info(vulkan_compute_ctx_t* ctx);

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
vulkan_buffer_t* vulkan_buffer_create(vulkan_compute_ctx_t* ctx, size_t size);

/**
 * @brief Create GPU buffer from existing data
 *
 * @param ctx Compute context
 * @param data Host data to copy
 * @param size Size in bytes
 * @return Buffer handle or NULL on failure
 */
vulkan_buffer_t* vulkan_buffer_create_from_data(
    vulkan_compute_ctx_t* ctx,
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
int vulkan_buffer_read(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* buffer,
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
int vulkan_buffer_write(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* buffer,
    const void* src,
    size_t size
);

/**
 * @brief Free GPU buffer
 * @param buffer Buffer to free
 */
void vulkan_buffer_free(vulkan_buffer_t* buffer);

// ============================================================================
// QUANTUM GATE OPERATIONS
// ============================================================================

/**
 * @brief Apply Hadamard gate to single qubit
 */
int vulkan_hadamard(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
);

/**
 * @brief Apply Hadamard to all qubits
 */
int vulkan_hadamard_all(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint32_t num_qubits,
    uint64_t state_dim
);

/**
 * @brief Apply Pauli X gate
 */
int vulkan_pauli_x(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
);

/**
 * @brief Apply Pauli Y gate
 */
int vulkan_pauli_y(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
);

/**
 * @brief Apply Pauli Z gate
 */
int vulkan_pauli_z(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
);

/**
 * @brief Apply phase rotation gate
 */
int vulkan_phase(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint32_t qubit_index,
    float phase,
    uint64_t state_dim
);

/**
 * @brief Apply CNOT gate
 */
int vulkan_cnot(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint32_t control_qubit,
    uint32_t target_qubit,
    uint64_t state_dim
);

// ============================================================================
// GROVER'S ALGORITHM OPERATIONS
// ============================================================================

/**
 * @brief Apply oracle for single target state
 */
int vulkan_oracle(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint64_t target,
    uint64_t state_dim
);

/**
 * @brief Apply oracle for multiple target states
 */
int vulkan_oracle_multi(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    const uint64_t* targets,
    uint32_t num_targets,
    uint64_t state_dim
);

/**
 * @brief Apply Grover diffusion operator
 */
int vulkan_grover_diffusion(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint32_t num_qubits,
    uint64_t state_dim
);

/**
 * @brief Execute complete Grover search
 */
int vulkan_grover_search(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    uint64_t target,
    uint32_t num_qubits,
    uint32_t num_iterations
);

/**
 * @brief Execute batch Grover searches in parallel
 */
int vulkan_grover_batch_search(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* batch_states,
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
 */
int vulkan_compute_probabilities(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    vulkan_buffer_t* probabilities,
    uint64_t state_dim
);

/**
 * @brief Normalize quantum state
 */
int vulkan_normalize(
    vulkan_compute_ctx_t* ctx,
    vulkan_buffer_t* amplitudes,
    float norm,
    uint64_t state_dim
);

// ============================================================================
// SYNCHRONIZATION & PERFORMANCE
// ============================================================================

/**
 * @brief Wait for all GPU operations to complete
 */
void vulkan_wait_completion(vulkan_compute_ctx_t* ctx);

/**
 * @brief Get last operation execution time
 */
double vulkan_get_last_execution_time(vulkan_compute_ctx_t* ctx);

/**
 * @brief Enable/disable performance monitoring
 */
void vulkan_set_performance_monitoring(vulkan_compute_ctx_t* ctx, int enable);

/**
 * @brief Get last error message
 */
const char* vulkan_get_error(vulkan_compute_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif /* GPU_VULKAN_H */
