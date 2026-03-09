/**
 * @file gpu_webgpu.h
 * @brief WebGPU backend for WASM builds
 *
 * Initial WebGPU backend scaffold for Emscripten/WASM targets.
 * This layer owns backend detection plus buffer lifecycle so tensor-network
 * code can select a WebGPU backend in browser environments.
 *
 * @stability beta
 * @since v0.1.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef GPU_WEBGPU_H
#define GPU_WEBGPU_H

#include <stddef.h>
#include <stdint.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct webgpu_compute_ctx webgpu_compute_ctx_t;
typedef struct webgpu_buffer webgpu_buffer_t;

/**
 * @brief Check whether WebGPU is available in the current runtime.
 *
 * @return 1 if available, 0 otherwise
 */
int webgpu_is_available(void);

/**
 * @brief Initialize WebGPU backend context.
 *
 * @return Context on success, NULL on failure
 */
webgpu_compute_ctx_t* webgpu_compute_init(void);

/**
 * @brief Free WebGPU backend context.
 *
 * @param ctx Context to free
 */
void webgpu_compute_free(webgpu_compute_ctx_t* ctx);

/**
 * @brief Get basic WebGPU device info.
 *
 * @param ctx Context
 * @param name Output device/backend name
 * @param max_work_group_size Output max workgroup size hint
 * @param compute_units Output compute unit hint
 */
void webgpu_get_device_info(webgpu_compute_ctx_t* ctx,
                            char* name,
                            uint32_t* max_work_group_size,
                            uint32_t* compute_units);

/**
 * @brief Get last backend error string.
 *
 * @param ctx Context
 * @return Static error string pointer
 */
const char* webgpu_last_error(const webgpu_compute_ctx_t* ctx);

/**
 * @brief Create backend buffer.
 *
 * @param ctx Context
 * @param size Buffer size in bytes
 * @return Buffer on success, NULL on failure
 */
webgpu_buffer_t* webgpu_buffer_create(webgpu_compute_ctx_t* ctx, size_t size);

/**
 * @brief Create backend buffer from host data.
 *
 * @param ctx Context
 * @param data Source host data
 * @param size Buffer size in bytes
 * @return Buffer on success, NULL on failure
 */
webgpu_buffer_t* webgpu_buffer_create_from_data(webgpu_compute_ctx_t* ctx,
                                                const void* data,
                                                size_t size);

/**
 * @brief Get CPU-visible pointer for the buffer contents.
 *
 * @param buffer Buffer
 * @return Host pointer
 */
void* webgpu_buffer_contents(webgpu_buffer_t* buffer);

/**
 * @brief Write host data into backend buffer.
 *
 * @param ctx Context
 * @param buffer Buffer
 * @param src Source data
 * @param size Bytes to copy
 * @return 0 on success, -1 on failure
 */
int webgpu_buffer_write(webgpu_compute_ctx_t* ctx,
                        webgpu_buffer_t* buffer,
                        const void* src,
                        size_t size);

/**
 * @brief Read backend buffer data to host memory.
 *
 * @param ctx Context
 * @param buffer Buffer
 * @param dst Destination
 * @param size Bytes to copy
 * @return 0 on success, -1 on failure
 */
int webgpu_buffer_read(webgpu_compute_ctx_t* ctx,
                       webgpu_buffer_t* buffer,
                       void* dst,
                       size_t size);

/**
 * @brief Free backend buffer.
 *
 * @param buffer Buffer
 */
void webgpu_buffer_free(webgpu_buffer_t* buffer);

// ============================================================================
// QUANTUM GATE OPERATIONS
// ============================================================================

int webgpu_hadamard(webgpu_compute_ctx_t* ctx,
                    webgpu_buffer_t* amplitudes,
                    uint32_t qubit_index,
                    uint64_t state_dim);

int webgpu_hadamard_all(webgpu_compute_ctx_t* ctx,
                        webgpu_buffer_t* amplitudes,
                        uint32_t num_qubits,
                        uint64_t state_dim);

int webgpu_pauli_x(webgpu_compute_ctx_t* ctx,
                   webgpu_buffer_t* amplitudes,
                   uint32_t qubit_index,
                   uint64_t state_dim);

int webgpu_pauli_z(webgpu_compute_ctx_t* ctx,
                   webgpu_buffer_t* amplitudes,
                   uint32_t qubit_index,
                   uint64_t state_dim);

int webgpu_phase(webgpu_compute_ctx_t* ctx,
                 webgpu_buffer_t* amplitudes,
                 uint32_t qubit_index,
                 double phase,
                 uint64_t state_dim);

int webgpu_cnot(webgpu_compute_ctx_t* ctx,
                webgpu_buffer_t* amplitudes,
                uint32_t control,
                uint32_t target,
                uint64_t state_dim);

int webgpu_oracle(webgpu_compute_ctx_t* ctx,
                  webgpu_buffer_t* amplitudes,
                  uint64_t target,
                  uint64_t state_dim);

int webgpu_oracle_multi(webgpu_compute_ctx_t* ctx,
                        webgpu_buffer_t* amplitudes,
                        const uint64_t* targets,
                        uint32_t num_targets,
                        uint64_t state_dim);

int webgpu_grover_diffusion(webgpu_compute_ctx_t* ctx,
                            webgpu_buffer_t* amplitudes,
                            uint32_t num_qubits,
                            uint64_t state_dim);

int webgpu_grover_iteration(webgpu_compute_ctx_t* ctx,
                            webgpu_buffer_t* amplitudes,
                            uint64_t target,
                            uint32_t num_qubits,
                            uint64_t state_dim);

// ============================================================================
// PROBABILITY & UTILITIES
// ============================================================================

int webgpu_compute_probabilities(webgpu_compute_ctx_t* ctx,
                                 webgpu_buffer_t* amplitudes,
                                 webgpu_buffer_t* probabilities,
                                 uint64_t state_dim);

int webgpu_normalize(webgpu_compute_ctx_t* ctx,
                     webgpu_buffer_t* amplitudes,
                     double norm,
                     uint64_t state_dim);

int webgpu_sum_squared_magnitudes(webgpu_compute_ctx_t* ctx,
                                  webgpu_buffer_t* amplitudes,
                                  uint64_t state_dim,
                                  double* result);

// ============================================================================
// TENSOR-NETWORK MPS HELPERS
// ============================================================================

/**
 * @brief Apply a 2-qubit gate to pre-contracted theta tensor on WebGPU.
 *
 * Theta layout is [chi_l, 2, 2, chi_r] contiguous in row-major order as
 * complex doubles.
 *
 * @param ctx WebGPU context
 * @param theta_host_ptr Host pointer to theta data (complex double array)
 * @param gate_4x4 Host pointer to 4x4 gate matrix (complex double array, 16 entries)
 * @param chi_l Left bond dimension
 * @param chi_r Right bond dimension
 * @param used_native_out Optional output flag (1 if WGSL dispatch used, 0 if CPU fallback)
 * @return 0 on success, -1 on failure
 */
int webgpu_mps_apply_gate_theta(webgpu_compute_ctx_t* ctx,
                                void* theta_host_ptr,
                                const double complex* gate_4x4,
                                uint32_t chi_l,
                                uint32_t chi_r,
                                int* used_native_out);

/**
 * @brief Compute canonical-form <Z> from a single MPS tensor on WebGPU.
 *
 * Tensor layout is [chi_l, 2, chi_r] contiguous in row-major order as
 * complex doubles.
 *
 * @param ctx WebGPU context
 * @param tensor_host_ptr Host pointer to tensor data
 * @param chi_l Left bond dimension
 * @param chi_r Right bond dimension
 * @param expectation_out Output expectation value
 * @param used_native_out Optional output flag (1 if WGSL dispatch used, 0 if CPU fallback)
 * @return 0 on success, -1 on failure
 */
int webgpu_mps_expectation_z_canonical(webgpu_compute_ctx_t* ctx,
                                       const void* tensor_host_ptr,
                                       uint32_t chi_l,
                                       uint32_t chi_r,
                                       double* expectation_out,
                                       int* used_native_out);

void webgpu_wait_completion(webgpu_compute_ctx_t* ctx);
double webgpu_get_last_execution_time(webgpu_compute_ctx_t* ctx);
void webgpu_set_performance_monitoring(webgpu_compute_ctx_t* ctx, int enable);
int webgpu_native_compute_ready(const webgpu_compute_ctx_t* ctx);

#ifdef __cplusplus
}
#endif

#endif  // GPU_WEBGPU_H
