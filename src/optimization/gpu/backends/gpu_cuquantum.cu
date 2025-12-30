/**
 * @file gpu_cuquantum.cu
 * @brief NVIDIA cuQuantum GPU compute backend implementation
 *
 * High-performance quantum simulation using NVIDIA cuQuantum SDK.
 * Uses cuStateVec for state vector simulation.
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#if defined(HAS_CUQUANTUM) && defined(HAS_CUSTATEVEC)

#include "gpu_cuquantum.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <custatevec.h>
#ifdef HAS_CUTENSORNET
#include <cutensornet.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================================
// INTERNAL STRUCTURES
// ============================================================================

struct cuquantum_compute_ctx {
    int device_id;
    cudaDeviceProp device_props;
    cudaStream_t stream;
    custatevecHandle_t handle;
    void* workspace;
    size_t workspace_size;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    double last_execution_time;
    char error_string[512];
    int has_error;
};

struct cuquantum_buffer {
    cuquantum_compute_ctx_t* ctx;
    void* device_ptr;
    size_t size;
    uint32_t num_qubits;
};

// Gate matrices (stored on host, copied as needed)
static const cuDoubleComplex HADAMARD_MATRIX[4] = {
    {0.7071067811865476, 0.0}, {0.7071067811865476, 0.0},
    {0.7071067811865476, 0.0}, {-0.7071067811865476, 0.0}
};

static const cuDoubleComplex PAULI_X_MATRIX[4] = {
    {0.0, 0.0}, {1.0, 0.0},
    {1.0, 0.0}, {0.0, 0.0}
};

static const cuDoubleComplex PAULI_Y_MATRIX[4] = {
    {0.0, 0.0}, {0.0, -1.0},
    {0.0, 1.0}, {0.0, 0.0}
};

static const cuDoubleComplex PAULI_Z_MATRIX[4] = {
    {1.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {-1.0, 0.0}
};

static const cuDoubleComplex CNOT_MATRIX[16] = {
    {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}
};

static const cuDoubleComplex CZ_MATRIX[16] = {
    {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
    {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}
};

// Helper to check cuStateVec status
static inline int check_custatevec(custatevecStatus_t status,
                                   cuquantum_compute_ctx_t* ctx,
                                   const char* msg) {
    if (status != CUSTATEVEC_STATUS_SUCCESS) {
        if (ctx) {
            snprintf(ctx->error_string, sizeof(ctx->error_string),
                     "%s: %s", msg, custatevecGetErrorString(status));
            ctx->has_error = 1;
        }
        return -1;
    }
    return 0;
}

// ============================================================================
// INITIALIZATION
// ============================================================================

int cuquantum_is_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) return 0;

    // Check if cuStateVec can be initialized
    cudaSetDevice(0);
    custatevecHandle_t handle;
    custatevecStatus_t status = custatevecCreate(&handle);
    if (status == CUSTATEVEC_STATUS_SUCCESS) {
        custatevecDestroy(handle);
        return 1;
    }
    return 0;
}

cuquantum_compute_ctx_t* cuquantum_context_create(int device_id,
                                                   const cuquantum_config_t* config) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0 || device_id >= device_count) {
        return NULL;
    }

    cuquantum_compute_ctx_t* ctx = (cuquantum_compute_ctx_t*)calloc(1,
                                                                     sizeof(cuquantum_compute_ctx_t));
    if (!ctx) return NULL;

    ctx->device_id = device_id;

    cudaError_t cuda_err = cudaSetDevice(device_id);
    if (cuda_err != cudaSuccess) {
        snprintf(ctx->error_string, sizeof(ctx->error_string),
                 "Failed to set device %d: %s", device_id, cudaGetErrorString(cuda_err));
        free(ctx);
        return NULL;
    }

    cuda_err = cudaGetDeviceProperties(&ctx->device_props, device_id);
    if (cuda_err != cudaSuccess) {
        snprintf(ctx->error_string, sizeof(ctx->error_string),
                 "Failed to get device properties: %s", cudaGetErrorString(cuda_err));
        free(ctx);
        return NULL;
    }

    cuda_err = cudaStreamCreate(&ctx->stream);
    if (cuda_err != cudaSuccess) {
        snprintf(ctx->error_string, sizeof(ctx->error_string),
                 "Failed to create stream: %s", cudaGetErrorString(cuda_err));
        free(ctx);
        return NULL;
    }

    // Create cuStateVec handle
    custatevecStatus_t status = custatevecCreate(&ctx->handle);
    if (status != CUSTATEVEC_STATUS_SUCCESS) {
        snprintf(ctx->error_string, sizeof(ctx->error_string),
                 "Failed to create cuStateVec handle: %s",
                 custatevecGetErrorString(status));
        cudaStreamDestroy(ctx->stream);
        free(ctx);
        return NULL;
    }

    custatevecSetStream(ctx->handle, ctx->stream);

    // Allocate workspace
    ctx->workspace_size = config ? config->workspace_size : (256 * 1024 * 1024); // 256 MB default
    cuda_err = cudaMalloc(&ctx->workspace, ctx->workspace_size);
    if (cuda_err != cudaSuccess) {
        // Reduce workspace size and try again
        ctx->workspace_size = 64 * 1024 * 1024; // 64 MB fallback
        cuda_err = cudaMalloc(&ctx->workspace, ctx->workspace_size);
        if (cuda_err != cudaSuccess) {
            snprintf(ctx->error_string, sizeof(ctx->error_string),
                     "Failed to allocate workspace: %s", cudaGetErrorString(cuda_err));
            custatevecDestroy(ctx->handle);
            cudaStreamDestroy(ctx->stream);
            free(ctx);
            return NULL;
        }
    }

    cudaEventCreate(&ctx->start_event);
    cudaEventCreate(&ctx->stop_event);

    return ctx;
}

void cuquantum_context_free(cuquantum_compute_ctx_t* ctx) {
    if (!ctx) return;

    if (ctx->workspace) cudaFree(ctx->workspace);
    if (ctx->handle) custatevecDestroy(ctx->handle);
    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    if (ctx->start_event) cudaEventDestroy(ctx->start_event);
    if (ctx->stop_event) cudaEventDestroy(ctx->stop_event);

    free(ctx);
}

int cuquantum_get_capabilities(cuquantum_compute_ctx_t* ctx,
                               cuquantum_capabilities_t* caps) {
    if (!ctx || !caps) return -1;

    memset(caps, 0, sizeof(cuquantum_capabilities_t));
    strncpy(caps->device_name, ctx->device_props.name, sizeof(caps->device_name) - 1);
    caps->compute_capability_major = ctx->device_props.major;
    caps->compute_capability_minor = ctx->device_props.minor;
    caps->global_memory = ctx->device_props.totalGlobalMem;
    caps->multiprocessor_count = ctx->device_props.multiProcessorCount;
    caps->supports_custatevec = 1;
#ifdef HAS_CUTENSORNET
    caps->supports_cutensornet = 1;
#else
    caps->supports_cutensornet = 0;
#endif

    // Estimate max qubits based on memory (16 bytes per amplitude for complex double)
    uint64_t usable_memory = (uint64_t)(ctx->device_props.totalGlobalMem * 0.8);
    caps->max_qubits = 0;
    while ((1ULL << caps->max_qubits) * 16 < usable_memory && caps->max_qubits < 64) {
        caps->max_qubits++;
    }
    caps->max_qubits--; // Stay within limit

    caps->memory_bandwidth_gbps = (ctx->device_props.memoryBusWidth / 8.0) *
                                   (ctx->device_props.memoryClockRate * 2.0 / 1e6);

    return 0;
}

void cuquantum_get_device_info(cuquantum_compute_ctx_t* ctx, char* device_name,
                               uint32_t* max_qubits,
                               uint32_t* multiprocessor_count) {
    if (!ctx) return;

    if (device_name) {
        strncpy(device_name, ctx->device_props.name, 255);
    }
    if (max_qubits) {
        cuquantum_capabilities_t caps;
        cuquantum_get_capabilities(ctx, &caps);
        *max_qubits = caps.max_qubits;
    }
    if (multiprocessor_count) {
        *multiprocessor_count = ctx->device_props.multiProcessorCount;
    }
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

cuquantum_buffer_t* cuquantum_statevec_create(cuquantum_compute_ctx_t* ctx,
                                               uint32_t num_qubits) {
    if (!ctx || num_qubits == 0 || num_qubits > 50) return NULL;

    cuquantum_buffer_t* buffer = (cuquantum_buffer_t*)calloc(1, sizeof(cuquantum_buffer_t));
    if (!buffer) return NULL;

    buffer->ctx = ctx;
    buffer->num_qubits = num_qubits;
    buffer->size = (1ULL << num_qubits) * sizeof(cuDoubleComplex);

    cudaError_t err = cudaMalloc(&buffer->device_ptr, buffer->size);
    if (err != cudaSuccess) {
        snprintf(ctx->error_string, sizeof(ctx->error_string),
                 "Failed to allocate state vector: %s", cudaGetErrorString(err));
        free(buffer);
        return NULL;
    }

    return buffer;
}

int cuquantum_statevec_init_zero(cuquantum_compute_ctx_t* ctx,
                                 cuquantum_buffer_t* buffer,
                                 uint32_t num_qubits) {
    if (!ctx || !buffer) return -1;

    uint64_t state_dim = 1ULL << num_qubits;

    // Set all amplitudes to zero
    cudaMemset(buffer->device_ptr, 0, buffer->size);

    // Set |0⟩ amplitude to 1
    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    cudaMemcpy(buffer->device_ptr, &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    return 0;
}

int cuquantum_statevec_init_uniform(cuquantum_compute_ctx_t* ctx,
                                    cuquantum_buffer_t* buffer,
                                    uint32_t num_qubits) {
    if (!ctx || !buffer) return -1;

    // Initialize to |0⟩ first
    cuquantum_statevec_init_zero(ctx, buffer, num_qubits);

    // Apply Hadamard to all qubits
    return cuquantum_hadamard_all(ctx, buffer, num_qubits);
}

void* cuquantum_buffer_device_ptr(cuquantum_buffer_t* buffer) {
    return buffer ? buffer->device_ptr : NULL;
}

int cuquantum_buffer_write(cuquantum_compute_ctx_t* ctx,
                           cuquantum_buffer_t* buffer,
                           const void* data, size_t size) {
    if (!ctx || !buffer || !data || size > buffer->size) return -1;

    cudaError_t err = cudaMemcpyAsync(buffer->device_ptr, data, size,
                                       cudaMemcpyHostToDevice, ctx->stream);
    if (err != cudaSuccess) {
        snprintf(ctx->error_string, sizeof(ctx->error_string),
                 "cudaMemcpy failed: %s", cudaGetErrorString(err));
        return -1;
    }
    cudaStreamSynchronize(ctx->stream);
    return 0;
}

int cuquantum_buffer_read(cuquantum_compute_ctx_t* ctx,
                          cuquantum_buffer_t* buffer,
                          void* data, size_t size) {
    if (!ctx || !buffer || !data || size > buffer->size) return -1;

    cudaError_t err = cudaMemcpyAsync(data, buffer->device_ptr, size,
                                       cudaMemcpyDeviceToHost, ctx->stream);
    if (err != cudaSuccess) {
        snprintf(ctx->error_string, sizeof(ctx->error_string),
                 "cudaMemcpy failed: %s", cudaGetErrorString(err));
        return -1;
    }
    cudaStreamSynchronize(ctx->stream);
    return 0;
}

void cuquantum_buffer_free(cuquantum_buffer_t* buffer) {
    if (!buffer) return;
    if (buffer->device_ptr) cudaFree(buffer->device_ptr);
    free(buffer);
}

// ============================================================================
// QUANTUM GATE OPERATIONS
// ============================================================================

int cuquantum_apply_gate_1q(cuquantum_compute_ctx_t* ctx,
                            cuquantum_buffer_t* statevec,
                            uint32_t num_qubits,
                            uint32_t target,
                            const void* matrix) {
    if (!ctx || !statevec || !matrix) return -1;

    cudaEventRecord(ctx->start_event, ctx->stream);

    int32_t targets[] = {(int32_t)target};
    custatevecStatus_t status = custatevecApplyMatrix(
        ctx->handle,
        statevec->device_ptr,
        CUDA_C_64F,
        num_qubits,
        matrix,
        CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW,
        0,  // adjoint = 0
        targets,
        1,  // nTargets = 1
        NULL,  // controls
        NULL,  // controlBitValues
        0,     // nControls
        CUSTATEVEC_COMPUTE_64F,
        ctx->workspace,
        ctx->workspace_size
    );

    cudaEventRecord(ctx->stop_event, ctx->stream);
    cudaStreamSynchronize(ctx->stream);

    float ms;
    cudaEventElapsedTime(&ms, ctx->start_event, ctx->stop_event);
    ctx->last_execution_time = ms / 1000.0;

    return check_custatevec(status, ctx, "applyMatrix");
}

int cuquantum_apply_gate_2q(cuquantum_compute_ctx_t* ctx,
                            cuquantum_buffer_t* statevec,
                            uint32_t num_qubits,
                            uint32_t control,
                            uint32_t target,
                            const void* matrix) {
    if (!ctx || !statevec || !matrix) return -1;

    int32_t targets[] = {(int32_t)target, (int32_t)control};

    custatevecStatus_t status = custatevecApplyMatrix(
        ctx->handle,
        statevec->device_ptr,
        CUDA_C_64F,
        num_qubits,
        matrix,
        CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW,
        0,  // adjoint = 0
        targets,
        2,  // nTargets = 2
        NULL,  // controls
        NULL,  // controlBitValues
        0,     // nControls
        CUSTATEVEC_COMPUTE_64F,
        ctx->workspace,
        ctx->workspace_size
    );

    return check_custatevec(status, ctx, "applyMatrix 2Q");
}

int cuquantum_hadamard(cuquantum_compute_ctx_t* ctx,
                       cuquantum_buffer_t* statevec,
                       uint32_t num_qubits,
                       uint32_t target) {
    return cuquantum_apply_gate_1q(ctx, statevec, num_qubits, target, HADAMARD_MATRIX);
}

int cuquantum_hadamard_all(cuquantum_compute_ctx_t* ctx,
                           cuquantum_buffer_t* statevec,
                           uint32_t num_qubits) {
    if (!ctx || !statevec) return -1;

    for (uint32_t q = 0; q < num_qubits; q++) {
        int err = cuquantum_hadamard(ctx, statevec, num_qubits, q);
        if (err != 0) return err;
    }
    return 0;
}

int cuquantum_pauli_x(cuquantum_compute_ctx_t* ctx,
                      cuquantum_buffer_t* statevec,
                      uint32_t num_qubits,
                      uint32_t target) {
    return cuquantum_apply_gate_1q(ctx, statevec, num_qubits, target, PAULI_X_MATRIX);
}

int cuquantum_pauli_y(cuquantum_compute_ctx_t* ctx,
                      cuquantum_buffer_t* statevec,
                      uint32_t num_qubits,
                      uint32_t target) {
    return cuquantum_apply_gate_1q(ctx, statevec, num_qubits, target, PAULI_Y_MATRIX);
}

int cuquantum_pauli_z(cuquantum_compute_ctx_t* ctx,
                      cuquantum_buffer_t* statevec,
                      uint32_t num_qubits,
                      uint32_t target) {
    return cuquantum_apply_gate_1q(ctx, statevec, num_qubits, target, PAULI_Z_MATRIX);
}

int cuquantum_phase_gate(cuquantum_compute_ctx_t* ctx,
                         cuquantum_buffer_t* statevec,
                         uint32_t num_qubits,
                         uint32_t target,
                         double phase) {
    cuDoubleComplex phase_matrix[4] = {
        {1.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {cos(phase), sin(phase)}
    };
    return cuquantum_apply_gate_1q(ctx, statevec, num_qubits, target, phase_matrix);
}

int cuquantum_cnot(cuquantum_compute_ctx_t* ctx,
                   cuquantum_buffer_t* statevec,
                   uint32_t num_qubits,
                   uint32_t control,
                   uint32_t target) {
    if (!ctx || !statevec) return -1;

    // Use controlled gate API for better performance
    int32_t targets[] = {(int32_t)target};
    int32_t controls[] = {(int32_t)control};
    int32_t controlValues[] = {1};

    custatevecStatus_t status = custatevecApplyMatrix(
        ctx->handle,
        statevec->device_ptr,
        CUDA_C_64F,
        num_qubits,
        PAULI_X_MATRIX,
        CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW,
        0,
        targets,
        1,
        controls,
        controlValues,
        1,
        CUSTATEVEC_COMPUTE_64F,
        ctx->workspace,
        ctx->workspace_size
    );

    return check_custatevec(status, ctx, "CNOT");
}

int cuquantum_cz(cuquantum_compute_ctx_t* ctx,
                 cuquantum_buffer_t* statevec,
                 uint32_t num_qubits,
                 uint32_t control,
                 uint32_t target) {
    if (!ctx || !statevec) return -1;

    int32_t targets[] = {(int32_t)target};
    int32_t controls[] = {(int32_t)control};
    int32_t controlValues[] = {1};

    custatevecStatus_t status = custatevecApplyMatrix(
        ctx->handle,
        statevec->device_ptr,
        CUDA_C_64F,
        num_qubits,
        PAULI_Z_MATRIX,
        CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW,
        0,
        targets,
        1,
        controls,
        controlValues,
        1,
        CUSTATEVEC_COMPUTE_64F,
        ctx->workspace,
        ctx->workspace_size
    );

    return check_custatevec(status, ctx, "CZ");
}

// ============================================================================
// GROVER'S ALGORITHM
// ============================================================================

// Kernel to flip phase at target index
__global__ void oracle_flip_kernel(cuDoubleComplex* amplitudes, uint64_t target) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        cuDoubleComplex amp = amplitudes[target];
        amplitudes[target] = make_cuDoubleComplex(-cuCreal(amp), -cuCimag(amp));
    }
}

int cuquantum_oracle_single_target(cuquantum_compute_ctx_t* ctx,
                                   cuquantum_buffer_t* statevec,
                                   uint32_t num_qubits,
                                   uint64_t target) {
    if (!ctx || !statevec) return -1;

    oracle_flip_kernel<<<1, 1, 0, ctx->stream>>>(
        (cuDoubleComplex*)statevec->device_ptr,
        target
    );

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// Kernel for sparse oracle
__global__ void sparse_oracle_kernel(cuDoubleComplex* amplitudes,
                                     const uint64_t* targets,
                                     uint32_t num_targets,
                                     uint64_t state_dim) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= state_dim) return;

    for (uint32_t i = 0; i < num_targets; i++) {
        if (tid == targets[i]) {
            cuDoubleComplex amp = amplitudes[tid];
            amplitudes[tid] = make_cuDoubleComplex(-cuCreal(amp), -cuCimag(amp));
            break;
        }
    }
}

int cuquantum_sparse_oracle(cuquantum_compute_ctx_t* ctx,
                            cuquantum_buffer_t* statevec,
                            uint32_t num_qubits,
                            const uint64_t* targets,
                            uint32_t num_targets) {
    if (!ctx || !statevec || !targets) return -1;

    uint64_t state_dim = 1ULL << num_qubits;

    uint64_t* d_targets;
    cudaMalloc(&d_targets, num_targets * sizeof(uint64_t));
    cudaMemcpy(d_targets, targets, num_targets * sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint32_t threads = 256;
    uint32_t blocks = (state_dim + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    sparse_oracle_kernel<<<blocks, threads, 0, ctx->stream>>>(
        (cuDoubleComplex*)statevec->device_ptr,
        d_targets,
        num_targets,
        state_dim
    );

    cudaFree(d_targets);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int cuquantum_grover_diffusion(cuquantum_compute_ctx_t* ctx,
                               cuquantum_buffer_t* statevec,
                               uint32_t num_qubits) {
    if (!ctx || !statevec) return -1;

    // Diffusion operator: H⊗n (2|0⟩⟨0| - I) H⊗n
    // = 2|s⟩⟨s| - I where |s⟩ = H⊗n|0⟩

    // 1. Apply H to all qubits
    int err = cuquantum_hadamard_all(ctx, statevec, num_qubits);
    if (err != 0) return err;

    // 2. Apply 2|0⟩⟨0| - I (conditional phase flip on |0⟩)
    // This requires multi-controlled Z gate or direct manipulation

    // For now, use cuStateVec's built-in diagonal matrix
    uint64_t state_dim = 1ULL << num_qubits;

    // Create diagonal: all -1 except +1 at index 0
    cuDoubleComplex* h_diagonal = (cuDoubleComplex*)malloc(state_dim * sizeof(cuDoubleComplex));
    for (uint64_t i = 0; i < state_dim; i++) {
        h_diagonal[i] = (i == 0) ? make_cuDoubleComplex(1.0, 0.0)
                                 : make_cuDoubleComplex(-1.0, 0.0);
    }

    cuDoubleComplex* d_diagonal;
    cudaMalloc(&d_diagonal, state_dim * sizeof(cuDoubleComplex));
    cudaMemcpy(d_diagonal, h_diagonal, state_dim * sizeof(cuDoubleComplex),
               cudaMemcpyHostToDevice);

    custatevecStatus_t status = custatevecApplyGeneralizedPermutationMatrix(
        ctx->handle,
        statevec->device_ptr,
        CUDA_C_64F,
        num_qubits,
        NULL,  // No permutation, just diagonal
        d_diagonal,
        CUDA_C_64F,
        NULL,  // All qubits as basis
        num_qubits,
        NULL, NULL, 0  // No mask
    );

    free(h_diagonal);
    cudaFree(d_diagonal);

    if (status != CUSTATEVEC_STATUS_SUCCESS) {
        return check_custatevec(status, ctx, "diffusion diagonal");
    }

    // 3. Apply H to all qubits again
    return cuquantum_hadamard_all(ctx, statevec, num_qubits);
}

int cuquantum_grover_iteration(cuquantum_compute_ctx_t* ctx,
                               cuquantum_buffer_t* statevec,
                               uint32_t num_qubits,
                               uint64_t target) {
    // Oracle
    int err = cuquantum_oracle_single_target(ctx, statevec, num_qubits, target);
    if (err != 0) return err;

    // Diffusion
    return cuquantum_grover_diffusion(ctx, statevec, num_qubits);
}

int cuquantum_grover_batch_search(cuquantum_compute_ctx_t* ctx,
                                  cuquantum_buffer_t* batch_states,
                                  const uint64_t* targets,
                                  uint64_t* results,
                                  uint32_t num_searches,
                                  uint32_t num_qubits,
                                  uint32_t num_iterations) {
    if (!ctx || !batch_states || !targets || !results) return -1;

    uint64_t state_dim = 1ULL << num_qubits;

    // Process each search independently
    for (uint32_t s = 0; s < num_searches; s++) {
        // Create temporary buffer wrapper for this search
        cuquantum_buffer_t temp_buf;
        temp_buf.ctx = ctx;
        temp_buf.device_ptr = (cuDoubleComplex*)batch_states->device_ptr + (s * state_dim);
        temp_buf.size = state_dim * sizeof(cuDoubleComplex);
        temp_buf.num_qubits = num_qubits;

        for (uint32_t iter = 0; iter < num_iterations; iter++) {
            int err = cuquantum_grover_iteration(ctx, &temp_buf, num_qubits, targets[s]);
            if (err != 0) return err;
        }

        // Find maximum probability state
        cuDoubleComplex* h_amps = (cuDoubleComplex*)malloc(state_dim * sizeof(cuDoubleComplex));
        cudaMemcpy(h_amps, temp_buf.device_ptr, state_dim * sizeof(cuDoubleComplex),
                   cudaMemcpyDeviceToHost);

        double max_prob = 0.0;
        uint64_t max_state = 0;
        for (uint64_t i = 0; i < state_dim; i++) {
            double prob = cuCreal(h_amps[i]) * cuCreal(h_amps[i]) +
                         cuCimag(h_amps[i]) * cuCimag(h_amps[i]);
            if (prob > max_prob) {
                max_prob = prob;
                max_state = i;
            }
        }
        results[s] = max_state;

        free(h_amps);
    }

    return 0;
}

// ============================================================================
// MEASUREMENT & UTILITIES
// ============================================================================

int cuquantum_compute_probabilities(cuquantum_compute_ctx_t* ctx,
                                    cuquantum_buffer_t* statevec,
                                    uint32_t num_qubits,
                                    double* probabilities) {
    if (!ctx || !statevec || !probabilities) return -1;

    uint64_t state_dim = 1ULL << num_qubits;

    // Read amplitudes and compute probabilities on host
    cuDoubleComplex* h_amps = (cuDoubleComplex*)malloc(state_dim * sizeof(cuDoubleComplex));
    cudaMemcpy(h_amps, statevec->device_ptr, state_dim * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToHost);

    for (uint64_t i = 0; i < state_dim; i++) {
        probabilities[i] = cuCreal(h_amps[i]) * cuCreal(h_amps[i]) +
                          cuCimag(h_amps[i]) * cuCimag(h_amps[i]);
    }

    free(h_amps);
    return 0;
}

int cuquantum_measure(cuquantum_compute_ctx_t* ctx,
                      cuquantum_buffer_t* statevec,
                      uint32_t num_qubits,
                      uint64_t* result) {
    if (!ctx || !statevec || !result) return -1;

    // Use cuStateVec's sampling
    custatevecSamplerDescriptor_t sampler;
    custatevecStatus_t status;

    status = custatevecSamplerCreate(
        ctx->handle,
        statevec->device_ptr,
        CUDA_C_64F,
        num_qubits,
        &sampler,
        1,  // nMaxShots
        ctx->workspace,
        ctx->workspace_size
    );

    if (status != CUSTATEVEC_STATUS_SUCCESS) {
        return check_custatevec(status, ctx, "samplerCreate");
    }

    status = custatevecSamplerPreprocess(
        ctx->handle,
        sampler,
        ctx->workspace,
        ctx->workspace_size
    );

    if (status != CUSTATEVEC_STATUS_SUCCESS) {
        custatevecSamplerDestroy(sampler);
        return check_custatevec(status, ctx, "samplerPreprocess");
    }

    custatevecIndex_t sample;
    status = custatevecSamplerSample(
        ctx->handle,
        sampler,
        &sample,
        NULL,  // bitOrdering (NULL = ascending)
        num_qubits,
        NULL,  // randnums
        1,     // nShots
        CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER
    );

    custatevecSamplerDestroy(sampler);

    if (status != CUSTATEVEC_STATUS_SUCCESS) {
        return check_custatevec(status, ctx, "samplerSample");
    }

    *result = sample;
    return 0;
}

int cuquantum_expectation_value(cuquantum_compute_ctx_t* ctx,
                                cuquantum_buffer_t* statevec,
                                uint32_t num_qubits,
                                const char* observable,
                                double* result) {
    if (!ctx || !statevec || !observable || !result) return -1;

    // Parse observable string and compute expectation
    // For simplicity, support single Pauli operators
    size_t len = strlen(observable);
    if (len != num_qubits) {
        snprintf(ctx->error_string, sizeof(ctx->error_string),
                 "Observable length mismatch");
        return -1;
    }

    // Convert Pauli string to cuStateVec format
    custatevecPauli_t* paulis = (custatevecPauli_t*)malloc(num_qubits * sizeof(custatevecPauli_t));
    int32_t* targets = (int32_t*)malloc(num_qubits * sizeof(int32_t));
    uint32_t nPaulis = 0;

    for (uint32_t i = 0; i < num_qubits; i++) {
        char c = observable[num_qubits - 1 - i];  // Reverse for qubit ordering
        if (c == 'I') continue;  // Skip identity

        targets[nPaulis] = i;
        switch (c) {
            case 'X': paulis[nPaulis] = CUSTATEVEC_PAULI_X; break;
            case 'Y': paulis[nPaulis] = CUSTATEVEC_PAULI_Y; break;
            case 'Z': paulis[nPaulis] = CUSTATEVEC_PAULI_Z; break;
            default:
                free(paulis);
                free(targets);
                return -1;
        }
        nPaulis++;
    }

    if (nPaulis == 0) {
        // All identity -> expectation = 1
        *result = 1.0;
        free(paulis);
        free(targets);
        return 0;
    }

    double2 expectation;
    custatevecStatus_t status = custatevecComputeExpectation(
        ctx->handle,
        statevec->device_ptr,
        CUDA_C_64F,
        num_qubits,
        &expectation,
        CUDA_C_64F,
        NULL,  // residualNorm
        paulis,
        targets,
        nPaulis,
        CUSTATEVEC_COMPUTE_64F,
        ctx->workspace,
        ctx->workspace_size
    );

    free(paulis);
    free(targets);

    if (status != CUSTATEVEC_STATUS_SUCCESS) {
        return check_custatevec(status, ctx, "computeExpectation");
    }

    *result = expectation.x;  // Real part
    return 0;
}

// Kernel for normalization
__global__ void normalize_kernel(cuDoubleComplex* amplitudes,
                                 double inv_norm,
                                 uint64_t state_dim) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= state_dim) return;

    cuDoubleComplex amp = amplitudes[tid];
    amplitudes[tid] = make_cuDoubleComplex(cuCreal(amp) * inv_norm,
                                           cuCimag(amp) * inv_norm);
}

int cuquantum_normalize(cuquantum_compute_ctx_t* ctx,
                        cuquantum_buffer_t* statevec,
                        uint32_t num_qubits) {
    if (!ctx || !statevec) return -1;

    uint64_t state_dim = 1ULL << num_qubits;

    // Compute norm
    double2 norm;
    custatevecStatus_t status = custatevecAbs2SumOnZBasis(
        ctx->handle,
        statevec->device_ptr,
        CUDA_C_64F,
        num_qubits,
        &norm,
        NULL, NULL, 0
    );

    if (status != CUSTATEVEC_STATUS_SUCCESS) {
        return check_custatevec(status, ctx, "abs2Sum");
    }

    double total_norm = sqrt(norm.x);
    if (total_norm < 1e-15) return -1;

    double inv_norm = 1.0 / total_norm;

    uint32_t threads = 256;
    uint32_t blocks = (state_dim + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    normalize_kernel<<<blocks, threads, 0, ctx->stream>>>(
        (cuDoubleComplex*)statevec->device_ptr,
        inv_norm,
        state_dim
    );

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

// ============================================================================
// SYNCHRONIZATION
// ============================================================================

void cuquantum_synchronize(cuquantum_compute_ctx_t* ctx) {
    if (ctx && ctx->stream) {
        cudaStreamSynchronize(ctx->stream);
    }
}

double cuquantum_get_last_execution_time(cuquantum_compute_ctx_t* ctx) {
    return ctx ? ctx->last_execution_time : 0.0;
}

const char* cuquantum_get_error_string(cuquantum_compute_ctx_t* ctx) {
    if (!ctx) return "No context";
    if (ctx->has_error) return ctx->error_string;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return cudaGetErrorString(err);
    }
    return NULL;
}

#else /* !HAS_CUQUANTUM || !HAS_CUSTATEVEC */

// Stub implementations when cuQuantum is not available
#include "gpu_cuquantum.h"
#include <stddef.h>

int cuquantum_is_available(void) { return 0; }

cuquantum_compute_ctx_t* cuquantum_context_create(int device_id,
                                                   const cuquantum_config_t* config) {
    (void)device_id; (void)config;
    return NULL;
}

void cuquantum_context_free(cuquantum_compute_ctx_t* ctx) { (void)ctx; }

int cuquantum_get_capabilities(cuquantum_compute_ctx_t* ctx,
                               cuquantum_capabilities_t* caps) {
    (void)ctx; (void)caps;
    return -1;
}

void cuquantum_get_device_info(cuquantum_compute_ctx_t* ctx, char* device_name,
                               uint32_t* max_qubits, uint32_t* multiprocessor_count) {
    (void)ctx; (void)device_name; (void)max_qubits; (void)multiprocessor_count;
}

cuquantum_buffer_t* cuquantum_statevec_create(cuquantum_compute_ctx_t* ctx,
                                               uint32_t num_qubits) {
    (void)ctx; (void)num_qubits;
    return NULL;
}

int cuquantum_statevec_init_zero(cuquantum_compute_ctx_t* ctx,
                                 cuquantum_buffer_t* buffer, uint32_t num_qubits) {
    (void)ctx; (void)buffer; (void)num_qubits;
    return -1;
}

int cuquantum_statevec_init_uniform(cuquantum_compute_ctx_t* ctx,
                                    cuquantum_buffer_t* buffer, uint32_t num_qubits) {
    (void)ctx; (void)buffer; (void)num_qubits;
    return -1;
}

void* cuquantum_buffer_device_ptr(cuquantum_buffer_t* buffer) {
    (void)buffer;
    return NULL;
}

int cuquantum_buffer_write(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* buffer,
                           const void* data, size_t size) {
    (void)ctx; (void)buffer; (void)data; (void)size;
    return -1;
}

int cuquantum_buffer_read(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* buffer,
                          void* data, size_t size) {
    (void)ctx; (void)buffer; (void)data; (void)size;
    return -1;
}

void cuquantum_buffer_free(cuquantum_buffer_t* buffer) { (void)buffer; }

int cuquantum_apply_gate_1q(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* statevec,
                            uint32_t num_qubits, uint32_t target, const void* matrix) {
    (void)ctx; (void)statevec; (void)num_qubits; (void)target; (void)matrix;
    return -1;
}

int cuquantum_apply_gate_2q(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* statevec,
                            uint32_t num_qubits, uint32_t control, uint32_t target,
                            const void* matrix) {
    (void)ctx; (void)statevec; (void)num_qubits; (void)control; (void)target; (void)matrix;
    return -1;
}

int cuquantum_hadamard(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* statevec,
                       uint32_t num_qubits, uint32_t target) {
    (void)ctx; (void)statevec; (void)num_qubits; (void)target;
    return -1;
}

int cuquantum_hadamard_all(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* statevec,
                           uint32_t num_qubits) {
    (void)ctx; (void)statevec; (void)num_qubits;
    return -1;
}

int cuquantum_pauli_x(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* statevec,
                      uint32_t num_qubits, uint32_t target) {
    (void)ctx; (void)statevec; (void)num_qubits; (void)target;
    return -1;
}

int cuquantum_pauli_y(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* statevec,
                      uint32_t num_qubits, uint32_t target) {
    (void)ctx; (void)statevec; (void)num_qubits; (void)target;
    return -1;
}

int cuquantum_pauli_z(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* statevec,
                      uint32_t num_qubits, uint32_t target) {
    (void)ctx; (void)statevec; (void)num_qubits; (void)target;
    return -1;
}

int cuquantum_phase_gate(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* statevec,
                         uint32_t num_qubits, uint32_t target, double phase) {
    (void)ctx; (void)statevec; (void)num_qubits; (void)target; (void)phase;
    return -1;
}

int cuquantum_cnot(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* statevec,
                   uint32_t num_qubits, uint32_t control, uint32_t target) {
    (void)ctx; (void)statevec; (void)num_qubits; (void)control; (void)target;
    return -1;
}

int cuquantum_cz(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* statevec,
                 uint32_t num_qubits, uint32_t control, uint32_t target) {
    (void)ctx; (void)statevec; (void)num_qubits; (void)control; (void)target;
    return -1;
}

int cuquantum_oracle_single_target(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* statevec,
                                   uint32_t num_qubits, uint64_t target) {
    (void)ctx; (void)statevec; (void)num_qubits; (void)target;
    return -1;
}

int cuquantum_sparse_oracle(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* statevec,
                            uint32_t num_qubits, const uint64_t* targets, uint32_t num_targets) {
    (void)ctx; (void)statevec; (void)num_qubits; (void)targets; (void)num_targets;
    return -1;
}

int cuquantum_grover_diffusion(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* statevec,
                               uint32_t num_qubits) {
    (void)ctx; (void)statevec; (void)num_qubits;
    return -1;
}

int cuquantum_grover_iteration(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* statevec,
                               uint32_t num_qubits, uint64_t target) {
    (void)ctx; (void)statevec; (void)num_qubits; (void)target;
    return -1;
}

int cuquantum_grover_batch_search(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* batch_states,
                                  const uint64_t* targets, uint64_t* results,
                                  uint32_t num_searches, uint32_t num_qubits,
                                  uint32_t num_iterations) {
    (void)ctx; (void)batch_states; (void)targets; (void)results;
    (void)num_searches; (void)num_qubits; (void)num_iterations;
    return -1;
}

int cuquantum_compute_probabilities(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* statevec,
                                    uint32_t num_qubits, double* probabilities) {
    (void)ctx; (void)statevec; (void)num_qubits; (void)probabilities;
    return -1;
}

int cuquantum_measure(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* statevec,
                      uint32_t num_qubits, uint64_t* result) {
    (void)ctx; (void)statevec; (void)num_qubits; (void)result;
    return -1;
}

int cuquantum_expectation_value(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* statevec,
                                uint32_t num_qubits, const char* observable, double* result) {
    (void)ctx; (void)statevec; (void)num_qubits; (void)observable; (void)result;
    return -1;
}

int cuquantum_normalize(cuquantum_compute_ctx_t* ctx, cuquantum_buffer_t* statevec,
                        uint32_t num_qubits) {
    (void)ctx; (void)statevec; (void)num_qubits;
    return -1;
}

void cuquantum_synchronize(cuquantum_compute_ctx_t* ctx) { (void)ctx; }

double cuquantum_get_last_execution_time(cuquantum_compute_ctx_t* ctx) {
    (void)ctx;
    return 0.0;
}

const char* cuquantum_get_error_string(cuquantum_compute_ctx_t* ctx) {
    (void)ctx;
    return "cuQuantum not available";
}

#endif /* HAS_CUQUANTUM && HAS_CUSTATEVEC */
