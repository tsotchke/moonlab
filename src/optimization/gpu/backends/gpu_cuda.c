/**
 * @file gpu_cuda.c
 * @brief NVIDIA CUDA GPU compute backend implementation
 *
 * Native CUDA implementation for quantum computing operations.
 * Uses the CUDA Runtime API for GPU acceleration on NVIDIA hardware.
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#include "gpu_cuda.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244  // 1/sqrt(2)
#endif

// ============================================================================
// CUDA RUNTIME CONDITIONAL COMPILATION
// ============================================================================

#ifdef HAS_CUDA

#include <cuda_runtime.h>
#include <cuComplex.h>

// ============================================================================
// INTERNAL STRUCTURES
// ============================================================================

struct cuda_compute_ctx {
    int device_id;
    cudaDeviceProp device_props;
    cudaStream_t stream;
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    double last_exec_time;
    char last_error[512];
};

struct cuda_buffer {
    cuda_compute_ctx_t* ctx;
    void* device_ptr;
    void* host_ptr;     // For unified memory or staging
    size_t size;
    int is_unified;
};

// ============================================================================
// ERROR HANDLING
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            snprintf(ctx->last_error, sizeof(ctx->last_error), \
                     "CUDA error: %s", cudaGetErrorString(err)); \
            return -1; \
        } \
    } while(0)

#define CUDA_CHECK_NULL(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            return NULL; \
        } \
    } while(0)

// ============================================================================
// CUDA KERNELS
// ============================================================================

// Hadamard gate kernel: |0⟩ → (|0⟩+|1⟩)/√2, |1⟩ → (|0⟩-|1⟩)/√2
__global__ void kernel_hadamard(cuDoubleComplex* amplitudes,
                                 uint32_t qubit_idx, uint64_t state_dim) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)1 << qubit_idx;

    if (idx >= state_dim / 2) return;

    // Find the pair of indices that differ only in qubit_idx
    uint64_t i0 = (idx / stride) * (2 * stride) + (idx % stride);
    uint64_t i1 = i0 + stride;

    cuDoubleComplex a0 = amplitudes[i0];
    cuDoubleComplex a1 = amplitudes[i1];

    double inv_sqrt2 = M_SQRT1_2;
    amplitudes[i0] = make_cuDoubleComplex(
        inv_sqrt2 * (cuCreal(a0) + cuCreal(a1)),
        inv_sqrt2 * (cuCimag(a0) + cuCimag(a1))
    );
    amplitudes[i1] = make_cuDoubleComplex(
        inv_sqrt2 * (cuCreal(a0) - cuCreal(a1)),
        inv_sqrt2 * (cuCimag(a0) - cuCimag(a1))
    );
}

// Pauli-X gate kernel: swap |0⟩ ↔ |1⟩
__global__ void kernel_pauli_x(cuDoubleComplex* amplitudes,
                                uint32_t qubit_idx, uint64_t state_dim) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)1 << qubit_idx;

    if (idx >= state_dim / 2) return;

    uint64_t i0 = (idx / stride) * (2 * stride) + (idx % stride);
    uint64_t i1 = i0 + stride;

    cuDoubleComplex temp = amplitudes[i0];
    amplitudes[i0] = amplitudes[i1];
    amplitudes[i1] = temp;
}

// Pauli-Y gate kernel: |0⟩ → i|1⟩, |1⟩ → -i|0⟩
__global__ void kernel_pauli_y(cuDoubleComplex* amplitudes,
                                uint32_t qubit_idx, uint64_t state_dim) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)1 << qubit_idx;

    if (idx >= state_dim / 2) return;

    uint64_t i0 = (idx / stride) * (2 * stride) + (idx % stride);
    uint64_t i1 = i0 + stride;

    cuDoubleComplex a0 = amplitudes[i0];
    cuDoubleComplex a1 = amplitudes[i1];

    // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
    amplitudes[i0] = make_cuDoubleComplex(-cuCimag(a1), cuCreal(a1));   // -i * a1
    amplitudes[i1] = make_cuDoubleComplex(cuCimag(a0), -cuCreal(a0));   // i * a0
}

// Pauli-Z gate kernel: |0⟩ → |0⟩, |1⟩ → -|1⟩
__global__ void kernel_pauli_z(cuDoubleComplex* amplitudes,
                                uint32_t qubit_idx, uint64_t state_dim) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)1 << qubit_idx;

    if (idx >= state_dim) return;

    // Negate amplitude if qubit is |1⟩
    if (idx & stride) {
        amplitudes[idx] = make_cuDoubleComplex(
            -cuCreal(amplitudes[idx]),
            -cuCimag(amplitudes[idx])
        );
    }
}

// Phase gate kernel: |0⟩ → |0⟩, |1⟩ → e^(iθ)|1⟩
__global__ void kernel_phase_gate(cuDoubleComplex* amplitudes,
                                   uint32_t qubit_idx, double phase, uint64_t state_dim) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)1 << qubit_idx;

    if (idx >= state_dim) return;

    // Apply phase if qubit is |1⟩
    if (idx & stride) {
        double cos_p = cos(phase);
        double sin_p = sin(phase);
        cuDoubleComplex amp = amplitudes[idx];
        amplitudes[idx] = make_cuDoubleComplex(
            cuCreal(amp) * cos_p - cuCimag(amp) * sin_p,
            cuCreal(amp) * sin_p + cuCimag(amp) * cos_p
        );
    }
}

// CNOT gate kernel
__global__ void kernel_cnot(cuDoubleComplex* amplitudes,
                            uint32_t control, uint32_t target, uint64_t state_dim) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t target_stride = (uint64_t)1 << target;
    uint64_t control_mask = (uint64_t)1 << control;

    if (idx >= state_dim / 2) return;

    // Find base index where target bit is 0
    uint64_t i0 = (idx / target_stride) * (2 * target_stride) + (idx % target_stride);
    uint64_t i1 = i0 + target_stride;

    // Only swap if control qubit is |1⟩
    if (i0 & control_mask) {
        cuDoubleComplex temp = amplitudes[i0];
        amplitudes[i0] = amplitudes[i1];
        amplitudes[i1] = temp;
    }
}

// Oracle kernel: flip phase of target state
__global__ void kernel_oracle_single(cuDoubleComplex* amplitudes, uint64_t target) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == target) {
        amplitudes[idx] = make_cuDoubleComplex(
            -cuCreal(amplitudes[idx]),
            -cuCimag(amplitudes[idx])
        );
    }
}

// Sparse oracle kernel: flip phase of multiple targets
__global__ void kernel_sparse_oracle(cuDoubleComplex* amplitudes,
                                      const uint64_t* targets, uint32_t num_targets) {
    uint64_t target_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (target_idx >= num_targets) return;

    uint64_t target = targets[target_idx];
    amplitudes[target] = make_cuDoubleComplex(
        -cuCreal(amplitudes[target]),
        -cuCimag(amplitudes[target])
    );
}

// Diffusion operator: 2|s⟩⟨s| - I
// First pass: compute mean
__global__ void kernel_diffusion_mean(const cuDoubleComplex* amplitudes,
                                       double* partial_sums_re, double* partial_sums_im,
                                       uint64_t state_dim) {
    extern __shared__ double sdata[];
    double* sdata_re = sdata;
    double* sdata_im = sdata + blockDim.x;

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t tid = threadIdx.x;

    // Initialize shared memory
    sdata_re[tid] = 0.0;
    sdata_im[tid] = 0.0;

    if (idx < state_dim) {
        sdata_re[tid] = cuCreal(amplitudes[idx]);
        sdata_im[tid] = cuCimag(amplitudes[idx]);
    }
    __syncthreads();

    // Reduction in shared memory
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_re[tid] += sdata_re[tid + s];
            sdata_im[tid] += sdata_im[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        partial_sums_re[blockIdx.x] = sdata_re[0];
        partial_sums_im[blockIdx.x] = sdata_im[0];
    }
}

// Second pass: apply diffusion
__global__ void kernel_diffusion_apply(cuDoubleComplex* amplitudes,
                                        double mean_re, double mean_im, uint64_t state_dim) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_dim) return;

    cuDoubleComplex amp = amplitudes[idx];
    amplitudes[idx] = make_cuDoubleComplex(
        2.0 * mean_re - cuCreal(amp),
        2.0 * mean_im - cuCimag(amp)
    );
}

// Compute probabilities kernel
__global__ void kernel_compute_probabilities(const cuDoubleComplex* amplitudes,
                                              double* probabilities, uint64_t state_dim) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_dim) return;

    cuDoubleComplex amp = amplitudes[idx];
    probabilities[idx] = cuCreal(amp) * cuCreal(amp) + cuCimag(amp) * cuCimag(amp);
}

// Normalize state kernel
__global__ void kernel_normalize(cuDoubleComplex* amplitudes, double norm, uint64_t state_dim) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_dim) return;

    double inv_norm = 1.0 / norm;
    cuDoubleComplex amp = amplitudes[idx];
    amplitudes[idx] = make_cuDoubleComplex(
        cuCreal(amp) * inv_norm,
        cuCimag(amp) * inv_norm
    );
}

// Sum squared magnitudes kernel (reduction)
__global__ void kernel_sum_squared_magnitudes(const cuDoubleComplex* amplitudes,
                                               double* partial_sums, uint64_t state_dim) {
    extern __shared__ double sdata[];
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t tid = threadIdx.x;

    sdata[tid] = 0.0;
    if (idx < state_dim) {
        cuDoubleComplex amp = amplitudes[idx];
        sdata[tid] = cuCreal(amp) * cuCreal(amp) + cuCimag(amp) * cuCimag(amp);
    }
    __syncthreads();

    // Reduction
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

int cuda_is_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0) ? 1 : 0;
}

cuda_compute_ctx_t* cuda_context_create(int device_id) {
    if (!cuda_is_available()) return NULL;

    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_id < 0 || device_id >= device_count) {
        device_id = 0;  // Default to first device
    }

    cuda_compute_ctx_t* ctx = (cuda_compute_ctx_t*)calloc(1, sizeof(cuda_compute_ctx_t));
    if (!ctx) return NULL;

    ctx->device_id = device_id;

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        free(ctx);
        return NULL;
    }

    err = cudaGetDeviceProperties(&ctx->device_props, device_id);
    if (err != cudaSuccess) {
        free(ctx);
        return NULL;
    }

    // Create CUDA stream for async operations
    err = cudaStreamCreate(&ctx->stream);
    if (err != cudaSuccess) {
        free(ctx);
        return NULL;
    }

    // Create timing events
    cudaEventCreate(&ctx->start_event);
    cudaEventCreate(&ctx->end_event);

    return ctx;
}

void cuda_context_free(cuda_compute_ctx_t* ctx) {
    if (!ctx) return;

    cudaEventDestroy(ctx->start_event);
    cudaEventDestroy(ctx->end_event);
    cudaStreamDestroy(ctx->stream);
    free(ctx);
}

int cuda_get_capabilities(cuda_compute_ctx_t* ctx, cuda_capabilities_t* caps) {
    if (!ctx || !caps) return -1;

    strncpy(caps->device_name, ctx->device_props.name, sizeof(caps->device_name) - 1);
    caps->compute_capability_major = ctx->device_props.major;
    caps->compute_capability_minor = ctx->device_props.minor;
    caps->global_memory = ctx->device_props.totalGlobalMem;
    caps->shared_memory_per_block = ctx->device_props.sharedMemPerBlock;
    caps->max_threads_per_block = ctx->device_props.maxThreadsPerBlock;
    caps->max_block_dim[0] = ctx->device_props.maxThreadsDim[0];
    caps->max_block_dim[1] = ctx->device_props.maxThreadsDim[1];
    caps->max_block_dim[2] = ctx->device_props.maxThreadsDim[2];
    caps->max_grid_dim[0] = ctx->device_props.maxGridSize[0];
    caps->max_grid_dim[1] = ctx->device_props.maxGridSize[1];
    caps->max_grid_dim[2] = ctx->device_props.maxGridSize[2];
    caps->warp_size = ctx->device_props.warpSize;
    caps->multiprocessor_count = ctx->device_props.multiProcessorCount;
    caps->supports_unified_memory = ctx->device_props.unifiedAddressing;
    caps->memory_bandwidth_gbps = (ctx->device_props.memoryBusWidth / 8.0) *
                                  (ctx->device_props.memoryClockRate * 2.0) / 1e6;

    return 0;
}

void cuda_get_device_info(cuda_compute_ctx_t* ctx, char* device_name,
                          uint32_t* max_threads_per_block,
                          uint32_t* multiprocessor_count) {
    if (!ctx) return;

    if (device_name) {
        strncpy(device_name, ctx->device_props.name, 255);
    }
    if (max_threads_per_block) {
        *max_threads_per_block = ctx->device_props.maxThreadsPerBlock;
    }
    if (multiprocessor_count) {
        *multiprocessor_count = ctx->device_props.multiProcessorCount;
    }
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

cuda_buffer_t* cuda_buffer_create(cuda_compute_ctx_t* ctx, size_t size) {
    if (!ctx || size == 0) return NULL;

    cuda_buffer_t* buffer = (cuda_buffer_t*)calloc(1, sizeof(cuda_buffer_t));
    if (!buffer) return NULL;

    buffer->ctx = ctx;
    buffer->size = size;

    // Try unified memory first (if supported)
    if (ctx->device_props.unifiedAddressing) {
        cudaError_t err = cudaMallocManaged(&buffer->device_ptr, size, cudaMemAttachGlobal);
        if (err == cudaSuccess) {
            buffer->host_ptr = buffer->device_ptr;  // Same pointer for unified memory
            buffer->is_unified = 1;
            return buffer;
        }
    }

    // Fallback to device memory with host staging
    cudaError_t err = cudaMalloc(&buffer->device_ptr, size);
    if (err != cudaSuccess) {
        free(buffer);
        return NULL;
    }

    // Allocate pinned host memory for efficient transfers
    err = cudaMallocHost(&buffer->host_ptr, size);
    if (err != cudaSuccess) {
        cudaFree(buffer->device_ptr);
        free(buffer);
        return NULL;
    }

    buffer->is_unified = 0;
    return buffer;
}

cuda_buffer_t* cuda_buffer_create_from_data(cuda_compute_ctx_t* ctx,
                                            void* data, size_t size) {
    cuda_buffer_t* buffer = cuda_buffer_create(ctx, size);
    if (!buffer) return NULL;

    if (cuda_buffer_write(ctx, buffer, data, size) != 0) {
        cuda_buffer_free(buffer);
        return NULL;
    }

    return buffer;
}

void* cuda_buffer_contents(cuda_buffer_t* buffer) {
    if (!buffer) return NULL;
    return buffer->host_ptr;
}

int cuda_buffer_write(cuda_compute_ctx_t* ctx, cuda_buffer_t* buffer,
                      const void* data, size_t size) {
    if (!ctx || !buffer || !data) return -1;
    if (size > buffer->size) return -1;

    if (buffer->is_unified) {
        memcpy(buffer->device_ptr, data, size);
        cudaDeviceSynchronize();
    } else {
        memcpy(buffer->host_ptr, data, size);
        CUDA_CHECK(cudaMemcpyAsync(buffer->device_ptr, buffer->host_ptr, size,
                                   cudaMemcpyHostToDevice, ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    }
    return 0;
}

int cuda_buffer_read(cuda_compute_ctx_t* ctx, cuda_buffer_t* buffer,
                     void* data, size_t size) {
    if (!ctx || !buffer || !data) return -1;
    if (size > buffer->size) return -1;

    if (buffer->is_unified) {
        cudaDeviceSynchronize();
        memcpy(data, buffer->device_ptr, size);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(buffer->host_ptr, buffer->device_ptr, size,
                                   cudaMemcpyDeviceToHost, ctx->stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        memcpy(data, buffer->host_ptr, size);
    }
    return 0;
}

void cuda_buffer_free(cuda_buffer_t* buffer) {
    if (!buffer) return;

    if (buffer->is_unified) {
        cudaFree(buffer->device_ptr);
    } else {
        cudaFree(buffer->device_ptr);
        cudaFreeHost(buffer->host_ptr);
    }
    free(buffer);
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static inline uint32_t get_block_size(cuda_compute_ctx_t* ctx) {
    return 256;  // Good default for most GPUs
}

static inline uint64_t get_num_blocks(uint64_t n, uint32_t block_size) {
    return (n + block_size - 1) / block_size;
}

static void start_timing(cuda_compute_ctx_t* ctx) {
    cudaEventRecord(ctx->start_event, ctx->stream);
}

static void end_timing(cuda_compute_ctx_t* ctx) {
    cudaEventRecord(ctx->end_event, ctx->stream);
    cudaEventSynchronize(ctx->end_event);
    float ms = 0;
    cudaEventElapsedTime(&ms, ctx->start_event, ctx->end_event);
    ctx->last_exec_time = ms / 1000.0;
}

// ============================================================================
// QUANTUM GATE OPERATIONS
// ============================================================================

int cuda_hadamard(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                  uint32_t qubit_index, uint64_t state_dim) {
    if (!ctx || !amplitudes) return -1;

    uint32_t block_size = get_block_size(ctx);
    uint64_t num_blocks = get_num_blocks(state_dim / 2, block_size);

    start_timing(ctx);
    kernel_hadamard<<<num_blocks, block_size, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr, qubit_index, state_dim
    );
    end_timing(ctx);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_hadamard_all(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                      uint32_t num_qubits, uint64_t state_dim) {
    for (uint32_t q = 0; q < num_qubits; q++) {
        int err = cuda_hadamard(ctx, amplitudes, q, state_dim);
        if (err != 0) return err;
    }
    return 0;
}

int cuda_pauli_x(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                 uint32_t qubit_index, uint64_t state_dim) {
    if (!ctx || !amplitudes) return -1;

    uint32_t block_size = get_block_size(ctx);
    uint64_t num_blocks = get_num_blocks(state_dim / 2, block_size);

    start_timing(ctx);
    kernel_pauli_x<<<num_blocks, block_size, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr, qubit_index, state_dim
    );
    end_timing(ctx);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_pauli_y(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                 uint32_t qubit_index, uint64_t state_dim) {
    if (!ctx || !amplitudes) return -1;

    uint32_t block_size = get_block_size(ctx);
    uint64_t num_blocks = get_num_blocks(state_dim / 2, block_size);

    start_timing(ctx);
    kernel_pauli_y<<<num_blocks, block_size, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr, qubit_index, state_dim
    );
    end_timing(ctx);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_pauli_z(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                 uint32_t qubit_index, uint64_t state_dim) {
    if (!ctx || !amplitudes) return -1;

    uint32_t block_size = get_block_size(ctx);
    uint64_t num_blocks = get_num_blocks(state_dim, block_size);

    start_timing(ctx);
    kernel_pauli_z<<<num_blocks, block_size, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr, qubit_index, state_dim
    );
    end_timing(ctx);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_phase_gate(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                    uint32_t qubit_index, double phase, uint64_t state_dim) {
    if (!ctx || !amplitudes) return -1;

    uint32_t block_size = get_block_size(ctx);
    uint64_t num_blocks = get_num_blocks(state_dim, block_size);

    start_timing(ctx);
    kernel_phase_gate<<<num_blocks, block_size, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr, qubit_index, phase, state_dim
    );
    end_timing(ctx);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_cnot(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
              uint32_t control_qubit, uint32_t target_qubit, uint64_t state_dim) {
    if (!ctx || !amplitudes) return -1;

    uint32_t block_size = get_block_size(ctx);
    uint64_t num_blocks = get_num_blocks(state_dim / 2, block_size);

    start_timing(ctx);
    kernel_cnot<<<num_blocks, block_size, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr, control_qubit, target_qubit, state_dim
    );
    end_timing(ctx);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// ============================================================================
// GROVER'S ALGORITHM
// ============================================================================

int cuda_oracle_single_target(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                              uint64_t target, uint64_t state_dim) {
    if (!ctx || !amplitudes) return -1;
    (void)state_dim;  // Not needed for single target

    start_timing(ctx);
    kernel_oracle_single<<<1, 1, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr, target
    );
    end_timing(ctx);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_sparse_oracle(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                       const uint64_t* targets, uint32_t num_targets,
                       uint64_t state_dim) {
    if (!ctx || !amplitudes || !targets || num_targets == 0) return -1;
    (void)state_dim;

    // Copy targets to device
    uint64_t* d_targets;
    CUDA_CHECK(cudaMalloc(&d_targets, num_targets * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_targets, targets, num_targets * sizeof(uint64_t),
                               cudaMemcpyHostToDevice, ctx->stream));

    uint32_t block_size = get_block_size(ctx);
    uint64_t num_blocks = get_num_blocks(num_targets, block_size);

    start_timing(ctx);
    kernel_sparse_oracle<<<num_blocks, block_size, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr, d_targets, num_targets
    );
    end_timing(ctx);

    cudaFree(d_targets);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_grover_diffusion(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                          uint32_t num_qubits, uint64_t state_dim) {
    if (!ctx || !amplitudes) return -1;

    uint32_t block_size = get_block_size(ctx);
    uint64_t num_blocks = get_num_blocks(state_dim, block_size);

    // Allocate partial sums
    double* d_partial_re;
    double* d_partial_im;
    CUDA_CHECK(cudaMalloc(&d_partial_re, num_blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_partial_im, num_blocks * sizeof(double)));

    // First pass: compute mean
    size_t shared_mem_size = 2 * block_size * sizeof(double);
    kernel_diffusion_mean<<<num_blocks, block_size, shared_mem_size, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr, d_partial_re, d_partial_im, state_dim
    );

    // Sum partial results on host (for simplicity)
    double* h_partial_re = (double*)malloc(num_blocks * sizeof(double));
    double* h_partial_im = (double*)malloc(num_blocks * sizeof(double));
    cudaMemcpy(h_partial_re, d_partial_re, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_partial_im, d_partial_im, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);

    double sum_re = 0, sum_im = 0;
    for (uint64_t i = 0; i < num_blocks; i++) {
        sum_re += h_partial_re[i];
        sum_im += h_partial_im[i];
    }
    double mean_re = sum_re / state_dim;
    double mean_im = sum_im / state_dim;

    free(h_partial_re);
    free(h_partial_im);
    cudaFree(d_partial_re);
    cudaFree(d_partial_im);

    // Second pass: apply diffusion
    start_timing(ctx);
    kernel_diffusion_apply<<<num_blocks, block_size, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr, mean_re, mean_im, state_dim
    );
    end_timing(ctx);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_grover_iteration(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                          uint64_t target, uint32_t num_qubits, uint64_t state_dim) {
    int err;

    // Oracle
    err = cuda_oracle_single_target(ctx, amplitudes, target, state_dim);
    if (err != 0) return err;

    // Diffusion
    err = cuda_grover_diffusion(ctx, amplitudes, num_qubits, state_dim);
    return err;
}

int cuda_grover_batch_search(cuda_compute_ctx_t* ctx, cuda_buffer_t* batch_states,
                             const uint64_t* targets, uint64_t* results,
                             uint32_t num_searches, uint32_t num_qubits,
                             uint32_t num_iterations) {
    if (!ctx || !batch_states || !targets || !results) return -1;

    uint64_t state_dim = (uint64_t)1 << num_qubits;
    size_t state_size = state_dim * sizeof(cuDoubleComplex);

    // Process each search sequentially (could be parallelized with streams)
    for (uint32_t s = 0; s < num_searches; s++) {
        // Create view into batch buffer
        cuda_buffer_t state_view;
        state_view.ctx = ctx;
        state_view.device_ptr = (char*)batch_states->device_ptr + s * state_size;
        state_view.host_ptr = (char*)batch_states->host_ptr + s * state_size;
        state_view.size = state_size;
        state_view.is_unified = batch_states->is_unified;

        // Run Grover iterations
        for (uint32_t iter = 0; iter < num_iterations; iter++) {
            int err = cuda_grover_iteration(ctx, &state_view, targets[s], num_qubits, state_dim);
            if (err != 0) return err;
        }

        // Measure (find max probability)
        double* probs = (double*)malloc(state_dim * sizeof(double));
        cuda_buffer_t* prob_buf = cuda_buffer_create(ctx, state_dim * sizeof(double));
        cuda_compute_probabilities(ctx, &state_view, prob_buf, state_dim);
        cuda_buffer_read(ctx, prob_buf, probs, state_dim * sizeof(double));

        uint64_t max_idx = 0;
        double max_prob = probs[0];
        for (uint64_t i = 1; i < state_dim; i++) {
            if (probs[i] > max_prob) {
                max_prob = probs[i];
                max_idx = i;
            }
        }
        results[s] = max_idx;

        cuda_buffer_free(prob_buf);
        free(probs);
    }

    return 0;
}

// ============================================================================
// MEASUREMENT & UTILITIES
// ============================================================================

int cuda_compute_probabilities(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                               cuda_buffer_t* probabilities, uint64_t state_dim) {
    if (!ctx || !amplitudes || !probabilities) return -1;

    uint32_t block_size = get_block_size(ctx);
    uint64_t num_blocks = get_num_blocks(state_dim, block_size);

    start_timing(ctx);
    kernel_compute_probabilities<<<num_blocks, block_size, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr,
        (double*)probabilities->device_ptr,
        state_dim
    );
    end_timing(ctx);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_normalize_state(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                         double norm, uint64_t state_dim) {
    if (!ctx || !amplitudes || norm == 0.0) return -1;

    uint32_t block_size = get_block_size(ctx);
    uint64_t num_blocks = get_num_blocks(state_dim, block_size);

    start_timing(ctx);
    kernel_normalize<<<num_blocks, block_size, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr, norm, state_dim
    );
    end_timing(ctx);

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_sum_squared_magnitudes(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                                uint64_t state_dim, double* result) {
    if (!ctx || !amplitudes || !result) return -1;

    uint32_t block_size = get_block_size(ctx);
    uint64_t num_blocks = get_num_blocks(state_dim, block_size);

    double* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, num_blocks * sizeof(double)));

    size_t shared_mem_size = block_size * sizeof(double);
    kernel_sum_squared_magnitudes<<<num_blocks, block_size, shared_mem_size, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr, d_partial, state_dim
    );

    // Sum partial results on host
    double* h_partial = (double*)malloc(num_blocks * sizeof(double));
    cudaMemcpy(h_partial, d_partial, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);

    double sum = 0;
    for (uint64_t i = 0; i < num_blocks; i++) {
        sum += h_partial[i];
    }
    *result = sum;

    free(h_partial);
    cudaFree(d_partial);

    return 0;
}

// ============================================================================
// SYNCHRONIZATION
// ============================================================================

void cuda_synchronize(cuda_compute_ctx_t* ctx) {
    if (!ctx) return;
    cudaStreamSynchronize(ctx->stream);
}

double cuda_get_last_execution_time(cuda_compute_ctx_t* ctx) {
    if (!ctx) return 0.0;
    return ctx->last_exec_time;
}

const char* cuda_get_error_string(cuda_compute_ctx_t* ctx) {
    if (!ctx) return "Invalid context";
    return ctx->last_error[0] ? ctx->last_error : NULL;
}

#else  // !HAS_CUDA

// ============================================================================
// STUB IMPLEMENTATIONS (when CUDA is not available)
// ============================================================================

int cuda_is_available(void) {
    return 0;
}

cuda_compute_ctx_t* cuda_context_create(int device_id) {
    (void)device_id;
    return NULL;
}

void cuda_context_free(cuda_compute_ctx_t* ctx) {
    (void)ctx;
}

int cuda_get_capabilities(cuda_compute_ctx_t* ctx, cuda_capabilities_t* caps) {
    (void)ctx; (void)caps;
    return -1;
}

void cuda_get_device_info(cuda_compute_ctx_t* ctx, char* device_name,
                          uint32_t* max_threads_per_block,
                          uint32_t* multiprocessor_count) {
    (void)ctx; (void)device_name;
    (void)max_threads_per_block; (void)multiprocessor_count;
}

cuda_buffer_t* cuda_buffer_create(cuda_compute_ctx_t* ctx, size_t size) {
    (void)ctx; (void)size;
    return NULL;
}

cuda_buffer_t* cuda_buffer_create_from_data(cuda_compute_ctx_t* ctx,
                                            void* data, size_t size) {
    (void)ctx; (void)data; (void)size;
    return NULL;
}

void* cuda_buffer_contents(cuda_buffer_t* buffer) {
    (void)buffer;
    return NULL;
}

int cuda_buffer_write(cuda_compute_ctx_t* ctx, cuda_buffer_t* buffer,
                      const void* data, size_t size) {
    (void)ctx; (void)buffer; (void)data; (void)size;
    return -1;
}

int cuda_buffer_read(cuda_compute_ctx_t* ctx, cuda_buffer_t* buffer,
                     void* data, size_t size) {
    (void)ctx; (void)buffer; (void)data; (void)size;
    return -1;
}

void cuda_buffer_free(cuda_buffer_t* buffer) {
    (void)buffer;
}

int cuda_hadamard(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                  uint32_t qubit_index, uint64_t state_dim) {
    (void)ctx; (void)amplitudes; (void)qubit_index; (void)state_dim;
    return -1;
}

int cuda_hadamard_all(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                      uint32_t num_qubits, uint64_t state_dim) {
    (void)ctx; (void)amplitudes; (void)num_qubits; (void)state_dim;
    return -1;
}

int cuda_pauli_x(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                 uint32_t qubit_index, uint64_t state_dim) {
    (void)ctx; (void)amplitudes; (void)qubit_index; (void)state_dim;
    return -1;
}

int cuda_pauli_y(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                 uint32_t qubit_index, uint64_t state_dim) {
    (void)ctx; (void)amplitudes; (void)qubit_index; (void)state_dim;
    return -1;
}

int cuda_pauli_z(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                 uint32_t qubit_index, uint64_t state_dim) {
    (void)ctx; (void)amplitudes; (void)qubit_index; (void)state_dim;
    return -1;
}

int cuda_phase_gate(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                    uint32_t qubit_index, double phase, uint64_t state_dim) {
    (void)ctx; (void)amplitudes; (void)qubit_index; (void)phase; (void)state_dim;
    return -1;
}

int cuda_cnot(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
              uint32_t control_qubit, uint32_t target_qubit, uint64_t state_dim) {
    (void)ctx; (void)amplitudes; (void)control_qubit; (void)target_qubit; (void)state_dim;
    return -1;
}

int cuda_oracle_single_target(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                              uint64_t target, uint64_t state_dim) {
    (void)ctx; (void)amplitudes; (void)target; (void)state_dim;
    return -1;
}

int cuda_sparse_oracle(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                       const uint64_t* targets, uint32_t num_targets,
                       uint64_t state_dim) {
    (void)ctx; (void)amplitudes; (void)targets; (void)num_targets; (void)state_dim;
    return -1;
}

int cuda_grover_diffusion(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                          uint32_t num_qubits, uint64_t state_dim) {
    (void)ctx; (void)amplitudes; (void)num_qubits; (void)state_dim;
    return -1;
}

int cuda_grover_iteration(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                          uint64_t target, uint32_t num_qubits, uint64_t state_dim) {
    (void)ctx; (void)amplitudes; (void)target; (void)num_qubits; (void)state_dim;
    return -1;
}

int cuda_grover_batch_search(cuda_compute_ctx_t* ctx, cuda_buffer_t* batch_states,
                             const uint64_t* targets, uint64_t* results,
                             uint32_t num_searches, uint32_t num_qubits,
                             uint32_t num_iterations) {
    (void)ctx; (void)batch_states; (void)targets; (void)results;
    (void)num_searches; (void)num_qubits; (void)num_iterations;
    return -1;
}

int cuda_compute_probabilities(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                               cuda_buffer_t* probabilities, uint64_t state_dim) {
    (void)ctx; (void)amplitudes; (void)probabilities; (void)state_dim;
    return -1;
}

int cuda_normalize_state(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                         double norm, uint64_t state_dim) {
    (void)ctx; (void)amplitudes; (void)norm; (void)state_dim;
    return -1;
}

int cuda_sum_squared_magnitudes(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                                uint64_t state_dim, double* result) {
    (void)ctx; (void)amplitudes; (void)state_dim; (void)result;
    return -1;
}

void cuda_synchronize(cuda_compute_ctx_t* ctx) {
    (void)ctx;
}

double cuda_get_last_execution_time(cuda_compute_ctx_t* ctx) {
    (void)ctx;
    return 0.0;
}

const char* cuda_get_error_string(cuda_compute_ctx_t* ctx) {
    (void)ctx;
    return "CUDA not available";
}

#endif  // HAS_CUDA
