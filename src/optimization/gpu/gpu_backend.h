/**
 * @file gpu_backend.h
 * @brief Unified GPU compute abstraction layer
 *
 * Provides a single API for GPU-accelerated quantum operations across:
 * - Metal (macOS)
 * - OpenCL (cross-platform)
 * - Vulkan (cross-platform)
 * - CUDA (NVIDIA)
 * - cuQuantum (NVIDIA - cuStateVec for state vectors, cuTensorNet for tensors)
 *
 * Backend selection priority:
 * - macOS: Metal (preferred) -> OpenCL
 * - Linux with NVIDIA: cuQuantum -> CUDA -> Vulkan -> OpenCL
 * - Linux without NVIDIA: Vulkan -> OpenCL
 * - Windows: CUDA -> Vulkan -> OpenCL
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#ifndef GPU_BACKEND_H
#define GPU_BACKEND_H

#include <stdint.h>
#include <stddef.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// Complex type
typedef double _Complex complex_t;

// ============================================================================
// GPU BACKEND TYPES
// ============================================================================

/**
 * @brief Available GPU backend types
 */
typedef enum {
    GPU_BACKEND_NONE = 0,      /**< No GPU backend (CPU fallback) */
    GPU_BACKEND_METAL,         /**< Apple Metal (macOS only) */
    GPU_BACKEND_OPENCL,        /**< OpenCL (cross-platform) */
    GPU_BACKEND_VULKAN,        /**< Vulkan compute (cross-platform) */
    GPU_BACKEND_CUDA,          /**< NVIDIA CUDA */
    GPU_BACKEND_CUQUANTUM,     /**< NVIDIA cuQuantum (cuStateVec + cuTensorNet) */
    GPU_BACKEND_AUTO           /**< Auto-select best available */
} gpu_backend_type_t;

/**
 * @brief GPU device capabilities
 */
typedef struct {
    char device_name[256];           /**< Device name string */
    char vendor_name[256];           /**< Vendor name */
    uint64_t global_memory;          /**< Global memory in bytes */
    uint64_t local_memory;           /**< Local/shared memory in bytes */
    uint32_t max_compute_units;      /**< GPU cores/CUs */
    uint32_t max_threads_per_group;  /**< Max threads per workgroup */
    uint32_t max_work_group_size[3]; /**< Max work group dimensions */
    int supports_double;             /**< Double precision support */
    int supports_unified_memory;     /**< Unified memory architecture */
    double memory_bandwidth_gbps;    /**< Estimated bandwidth */
} gpu_capabilities_t;

/**
 * @brief GPU operation error codes
 */
typedef enum {
    GPU_SUCCESS = 0,
    GPU_ERROR_NO_DEVICE = -1,
    GPU_ERROR_INIT_FAILED = -2,
    GPU_ERROR_COMPILE_FAILED = -3,
    GPU_ERROR_ALLOC_FAILED = -4,
    GPU_ERROR_KERNEL_FAILED = -5,
    GPU_ERROR_INVALID_PARAM = -6,
    GPU_ERROR_NOT_SUPPORTED = -7,
    GPU_ERROR_TIMEOUT = -8
} gpu_error_t;

// Forward declarations for opaque types
typedef struct gpu_context gpu_context_t;
typedef struct gpu_buffer gpu_buffer_t;
typedef struct gpu_kernel gpu_kernel_t;

// ============================================================================
// INITIALIZATION & DEVICE MANAGEMENT
// ============================================================================

/**
 * @brief Initialize GPU compute context
 *
 * Auto-selects the best available backend unless specified.
 *
 * @param preferred Preferred backend (GPU_BACKEND_AUTO for auto-select)
 * @return GPU context or NULL on failure
 */
gpu_context_t* gpu_compute_init(gpu_backend_type_t preferred);

/**
 * @brief Free GPU compute context
 *
 * @param ctx GPU context
 */
void gpu_compute_free(gpu_context_t* ctx);

/**
 * @brief Check if any GPU backend is available
 *
 * @return 1 if GPU available, 0 otherwise
 */
int gpu_is_available(void);

/**
 * @brief Get active backend type
 *
 * @param ctx GPU context
 * @return Active backend type
 */
gpu_backend_type_t gpu_get_backend_type(gpu_context_t* ctx);

/**
 * @brief Get backend name string
 *
 * @param type Backend type
 * @return Human-readable backend name
 */
const char* gpu_backend_name(gpu_backend_type_t type);

/**
 * @brief Get GPU device capabilities
 *
 * @param ctx GPU context
 * @param caps Output capabilities structure
 * @return GPU_SUCCESS or error code
 */
gpu_error_t gpu_get_capabilities(gpu_context_t* ctx, gpu_capabilities_t* caps);

/**
 * @brief Print GPU device information
 *
 * @param ctx GPU context
 */
void gpu_print_device_info(gpu_context_t* ctx);

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

/**
 * @brief Allocate GPU buffer
 *
 * Uses unified memory when available (Metal, some OpenCL).
 *
 * @param ctx GPU context
 * @param size Buffer size in bytes
 * @return GPU buffer or NULL on failure
 */
gpu_buffer_t* gpu_buffer_create(gpu_context_t* ctx, size_t size);

/**
 * @brief Create GPU buffer from existing CPU memory
 *
 * Zero-copy when unified memory available.
 *
 * @param ctx GPU context
 * @param data CPU memory pointer
 * @param size Buffer size in bytes
 * @return GPU buffer or NULL on failure
 */
gpu_buffer_t* gpu_buffer_create_from_data(gpu_context_t* ctx, void* data, size_t size);

/**
 * @brief Get CPU-accessible pointer to buffer
 *
 * @param buffer GPU buffer
 * @return Pointer to data (NULL if not mappable)
 */
void* gpu_buffer_contents(gpu_buffer_t* buffer);

/**
 * @brief Copy data to GPU buffer
 *
 * @param buffer GPU buffer
 * @param data Source data
 * @param size Bytes to copy
 * @param offset Offset into buffer
 * @return GPU_SUCCESS or error code
 */
gpu_error_t gpu_buffer_write(gpu_buffer_t* buffer, const void* data, size_t size, size_t offset);

/**
 * @brief Copy data from GPU buffer
 *
 * @param buffer GPU buffer
 * @param data Destination
 * @param size Bytes to copy
 * @param offset Offset into buffer
 * @return GPU_SUCCESS or error code
 */
gpu_error_t gpu_buffer_read(gpu_buffer_t* buffer, void* data, size_t size, size_t offset);

/**
 * @brief Free GPU buffer
 *
 * @param buffer GPU buffer
 */
void gpu_buffer_free(gpu_buffer_t* buffer);

// ============================================================================
// QUANTUM GATE OPERATIONS
// ============================================================================

/**
 * @brief GPU-accelerated Hadamard gate
 *
 * @param ctx GPU context
 * @param amplitudes Amplitude buffer
 * @param qubit_index Qubit index
 * @param state_dim State dimension (2^n)
 * @return GPU_SUCCESS or error code
 */
gpu_error_t gpu_hadamard(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                         uint32_t qubit_index, uint64_t state_dim);

/**
 * @brief GPU-accelerated Hadamard on all qubits
 *
 * @param ctx GPU context
 * @param amplitudes Amplitude buffer
 * @param num_qubits Number of qubits
 * @param state_dim State dimension
 * @return GPU_SUCCESS or error code
 */
gpu_error_t gpu_hadamard_all(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                             uint32_t num_qubits, uint64_t state_dim);

/**
 * @brief GPU-accelerated Pauli-X gate
 *
 * @param ctx GPU context
 * @param amplitudes Amplitude buffer
 * @param qubit_index Qubit index
 * @param state_dim State dimension
 * @return GPU_SUCCESS or error code
 */
gpu_error_t gpu_pauli_x(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                        uint32_t qubit_index, uint64_t state_dim);

/**
 * @brief GPU-accelerated Pauli-Z gate
 *
 * @param ctx GPU context
 * @param amplitudes Amplitude buffer
 * @param qubit_index Qubit index
 * @param state_dim State dimension
 * @return GPU_SUCCESS or error code
 */
gpu_error_t gpu_pauli_z(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                        uint32_t qubit_index, uint64_t state_dim);

/**
 * @brief GPU-accelerated phase rotation gate
 *
 * @param ctx GPU context
 * @param amplitudes Amplitude buffer
 * @param qubit_index Qubit index
 * @param theta Rotation angle
 * @param state_dim State dimension
 * @return GPU_SUCCESS or error code
 */
gpu_error_t gpu_phase(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                      uint32_t qubit_index, double theta, uint64_t state_dim);

/**
 * @brief GPU-accelerated CNOT gate
 *
 * @param ctx GPU context
 * @param amplitudes Amplitude buffer
 * @param control Control qubit
 * @param target Target qubit
 * @param state_dim State dimension
 * @return GPU_SUCCESS or error code
 */
gpu_error_t gpu_cnot(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                     uint32_t control, uint32_t target, uint64_t state_dim);

// ============================================================================
// GROVER'S ALGORITHM OPERATIONS
// ============================================================================

/**
 * @brief GPU-accelerated oracle (phase flip)
 *
 * @param ctx GPU context
 * @param amplitudes Amplitude buffer
 * @param target_state Marked state
 * @param state_dim State dimension
 * @return GPU_SUCCESS or error code
 */
gpu_error_t gpu_oracle_single_target(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                                     uint64_t target_state, uint64_t state_dim);

/**
 * @brief GPU-accelerated multi-target oracle
 *
 * @param ctx GPU context
 * @param amplitudes Amplitude buffer
 * @param targets Array of marked states
 * @param num_targets Number of marked states
 * @param state_dim State dimension
 * @return GPU_SUCCESS or error code
 */
gpu_error_t gpu_oracle_multi_target(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                                    const uint64_t* targets, uint32_t num_targets,
                                    uint64_t state_dim);

/**
 * @brief GPU-accelerated Grover diffusion operator
 *
 * @param ctx GPU context
 * @param amplitudes Amplitude buffer
 * @param num_qubits Number of qubits
 * @param state_dim State dimension
 * @return GPU_SUCCESS or error code
 */
gpu_error_t gpu_grover_diffusion(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                                 uint32_t num_qubits, uint64_t state_dim);

/**
 * @brief GPU-accelerated complete Grover iteration
 *
 * Fused oracle + diffusion for maximum performance.
 *
 * @param ctx GPU context
 * @param amplitudes Amplitude buffer
 * @param target_state Marked state
 * @param num_qubits Number of qubits
 * @param state_dim State dimension
 * @return GPU_SUCCESS or error code
 */
gpu_error_t gpu_grover_iteration(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                                 uint64_t target_state, uint32_t num_qubits,
                                 uint64_t state_dim);

/**
 * @brief GPU-accelerated batch Grover search
 *
 * Executes multiple independent searches in parallel.
 * Optimal for utilizing all GPU cores.
 *
 * @param ctx GPU context
 * @param batch_states Buffer for all states (num_searches * state_dim)
 * @param targets Array of targets (one per search)
 * @param results Output results array
 * @param num_searches Number of parallel searches
 * @param num_qubits Qubits per search
 * @param num_iterations Iterations per search
 * @return GPU_SUCCESS or error code
 */
gpu_error_t gpu_grover_batch_search(gpu_context_t* ctx, gpu_buffer_t* batch_states,
                                    const uint64_t* targets, uint64_t* results,
                                    uint32_t num_searches, uint32_t num_qubits,
                                    uint32_t num_iterations);

// ============================================================================
// PROBABILITY & MEASUREMENT
// ============================================================================

/**
 * @brief GPU-accelerated probability computation
 *
 * @param ctx GPU context
 * @param amplitudes Amplitude buffer
 * @param probabilities Output probability buffer
 * @param state_dim State dimension
 * @return GPU_SUCCESS or error code
 */
gpu_error_t gpu_compute_probabilities(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                                      gpu_buffer_t* probabilities, uint64_t state_dim);

/**
 * @brief GPU-accelerated state normalization
 *
 * @param ctx GPU context
 * @param amplitudes Amplitude buffer
 * @param norm Normalization factor
 * @param state_dim State dimension
 * @return GPU_SUCCESS or error code
 */
gpu_error_t gpu_normalize(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                          double norm, uint64_t state_dim);

/**
 * @brief GPU-accelerated sum of squared magnitudes
 *
 * @param ctx GPU context
 * @param amplitudes Amplitude buffer
 * @param state_dim State dimension
 * @param result Output sum
 * @return GPU_SUCCESS or error code
 */
gpu_error_t gpu_sum_squared_magnitudes(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                                       uint64_t state_dim, double* result);

// ============================================================================
// SYNCHRONIZATION & UTILITIES
// ============================================================================

/**
 * @brief Wait for all GPU operations to complete
 *
 * @param ctx GPU context
 */
void gpu_wait_completion(gpu_context_t* ctx);

/**
 * @brief Get execution time of last operation
 *
 * @param ctx GPU context
 * @return Execution time in seconds
 */
double gpu_get_last_execution_time(gpu_context_t* ctx);

/**
 * @brief Enable/disable performance monitoring
 *
 * @param ctx GPU context
 * @param enable 1 to enable, 0 to disable
 */
void gpu_set_performance_monitoring(gpu_context_t* ctx, int enable);

/**
 * @brief Get last error message
 *
 * @param ctx GPU context
 * @return Error message string
 */
const char* gpu_get_error_string(gpu_context_t* ctx);

/**
 * @brief Convert error code to string
 *
 * @param error Error code
 * @return Error description
 */
const char* gpu_error_name(gpu_error_t error);

#ifdef __cplusplus
}
#endif

#endif /* GPU_BACKEND_H */
