/**
 * @file gpu_cuda.cu
 * @brief NVIDIA CUDA GPU compute backend implementation
 *
 * Native CUDA implementation for quantum computing operations.
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#ifdef HAS_CUDA

#include "gpu_cuda.h"
#include "../kernels/cuda/quantum_kernels.cuh"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================================
// INTERNAL STRUCTURES
// ============================================================================

struct cuda_compute_ctx {
    int device_id;
    cudaDeviceProp device_props;
    cudaStream_t stream;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    double last_execution_time;
    char error_string[512];
    int has_error;
};

struct cuda_buffer {
    cuda_compute_ctx_t* ctx;
    void* device_ptr;
    void* host_ptr;
    size_t size;
    int is_unified;
};

// ============================================================================
// KERNEL IMPLEMENTATIONS
// ============================================================================

__global__ void hadamard_transform_kernel(
    cuDoubleComplex* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = 1ULL << qubit_index;
    uint64_t num_pairs = state_dim / 2;

    if (tid >= num_pairs) return;

    uint64_t mask = stride - 1;
    uint64_t base = (tid / stride) * (stride * 2);
    uint64_t offset = tid & mask;
    uint64_t idx0 = base + offset;
    uint64_t idx1 = idx0 + stride;

    cuDoubleComplex amp0 = amplitudes[idx0];
    cuDoubleComplex amp1 = amplitudes[idx1];

    const double inv_sqrt2 = 0.7071067811865476;

    amplitudes[idx0] = make_cuDoubleComplex(
        (cuCreal(amp0) + cuCreal(amp1)) * inv_sqrt2,
        (cuCimag(amp0) + cuCimag(amp1)) * inv_sqrt2
    );
    amplitudes[idx1] = make_cuDoubleComplex(
        (cuCreal(amp0) - cuCreal(amp1)) * inv_sqrt2,
        (cuCimag(amp0) - cuCimag(amp1)) * inv_sqrt2
    );
}

__global__ void hadamard_all_kernel(
    cuDoubleComplex* amplitudes,
    uint32_t num_qubits,
    uint64_t state_dim
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= state_dim) return;

    cuDoubleComplex amp = amplitudes[tid];
    uint32_t hamming = __popcll(tid);
    double sign = (hamming & 1) ? -1.0 : 1.0;
    double scale = 1.0 / sqrt((double)(1ULL << num_qubits));

    amplitudes[tid] = make_cuDoubleComplex(
        cuCreal(amp) * sign * scale,
        cuCimag(amp) * sign * scale
    );
}

__global__ void pauli_x_kernel(
    cuDoubleComplex* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = 1ULL << qubit_index;
    uint64_t num_pairs = state_dim / 2;

    if (tid >= num_pairs) return;

    uint64_t mask = stride - 1;
    uint64_t base = (tid / stride) * (stride * 2);
    uint64_t offset = tid & mask;
    uint64_t idx0 = base + offset;
    uint64_t idx1 = idx0 + stride;

    cuDoubleComplex temp = amplitudes[idx0];
    amplitudes[idx0] = amplitudes[idx1];
    amplitudes[idx1] = temp;
}

__global__ void pauli_y_kernel(
    cuDoubleComplex* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = 1ULL << qubit_index;
    uint64_t num_pairs = state_dim / 2;

    if (tid >= num_pairs) return;

    uint64_t mask = stride - 1;
    uint64_t base = (tid / stride) * (stride * 2);
    uint64_t offset = tid & mask;
    uint64_t idx0 = base + offset;
    uint64_t idx1 = idx0 + stride;

    cuDoubleComplex amp0 = amplitudes[idx0];
    cuDoubleComplex amp1 = amplitudes[idx1];

    // Y = | 0  -i |
    //     | i   0 |
    amplitudes[idx0] = make_cuDoubleComplex(cuCimag(amp1), -cuCreal(amp1));
    amplitudes[idx1] = make_cuDoubleComplex(-cuCimag(amp0), cuCreal(amp0));
}

__global__ void pauli_z_kernel(
    cuDoubleComplex* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= state_dim) return;

    uint64_t mask = 1ULL << qubit_index;
    if (tid & mask) {
        cuDoubleComplex amp = amplitudes[tid];
        amplitudes[tid] = make_cuDoubleComplex(-cuCreal(amp), -cuCimag(amp));
    }
}

__global__ void phase_gate_kernel(
    cuDoubleComplex* amplitudes,
    uint32_t qubit_index,
    double phase,
    uint64_t state_dim
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= state_dim) return;

    uint64_t mask = 1ULL << qubit_index;
    if (tid & mask) {
        cuDoubleComplex amp = amplitudes[tid];
        double c = cos(phase);
        double s = sin(phase);
        amplitudes[tid] = make_cuDoubleComplex(
            cuCreal(amp) * c - cuCimag(amp) * s,
            cuCreal(amp) * s + cuCimag(amp) * c
        );
    }
}

__global__ void cnot_kernel(
    cuDoubleComplex* amplitudes,
    uint32_t control_qubit,
    uint32_t target_qubit,
    uint64_t state_dim
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= state_dim) return;

    uint64_t control_mask = 1ULL << control_qubit;
    uint64_t target_mask = 1ULL << target_qubit;

    // Only process when control is 1 and target is 0
    if ((tid & control_mask) && !(tid & target_mask)) {
        uint64_t partner = tid | target_mask;
        cuDoubleComplex temp = amplitudes[tid];
        amplitudes[tid] = amplitudes[partner];
        amplitudes[partner] = temp;
    }
}

__global__ void oracle_single_target_kernel(
    cuDoubleComplex* amplitudes,
    uint64_t target
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == target) {
        cuDoubleComplex amp = amplitudes[tid];
        amplitudes[tid] = make_cuDoubleComplex(-cuCreal(amp), -cuCimag(amp));
    }
}

__global__ void sparse_oracle_kernel(
    cuDoubleComplex* amplitudes,
    const uint64_t* targets,
    uint32_t num_targets,
    uint64_t state_dim
) {
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

__global__ void diffusion_sum_kernel(
    const cuDoubleComplex* amplitudes,
    cuDoubleComplex* partial_sums,
    uint64_t state_dim
) {
    extern __shared__ cuDoubleComplex shared_data[];

    uint64_t tid = threadIdx.x;
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t grid_size = blockDim.x * gridDim.x;

    // Accumulate per-thread
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
    for (uint64_t i = gid; i < state_dim; i += grid_size) {
        cuDoubleComplex amp = amplitudes[i];
        sum = make_cuDoubleComplex(
            cuCreal(sum) + cuCreal(amp),
            cuCimag(sum) + cuCimag(amp)
        );
    }

    shared_data[tid] = sum;
    __syncthreads();

    // Block-level reduction
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cuDoubleComplex other = shared_data[tid + s];
            shared_data[tid] = make_cuDoubleComplex(
                cuCreal(shared_data[tid]) + cuCreal(other),
                cuCimag(shared_data[tid]) + cuCimag(other)
            );
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_data[0];
    }
}

__global__ void diffusion_apply_kernel(
    cuDoubleComplex* amplitudes,
    const cuDoubleComplex* avg_buffer,
    uint64_t state_dim
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= state_dim) return;

    cuDoubleComplex avg = avg_buffer[0];
    cuDoubleComplex amp = amplitudes[tid];

    // 2*avg - amp
    amplitudes[tid] = make_cuDoubleComplex(
        2.0 * cuCreal(avg) - cuCreal(amp),
        2.0 * cuCimag(avg) - cuCimag(amp)
    );
}

__global__ void grover_diffusion_fused_kernel(
    cuDoubleComplex* amplitudes,
    uint32_t num_qubits,
    uint64_t state_dim
) {
    extern __shared__ cuDoubleComplex shared_amps[];

    uint64_t tid = threadIdx.x;
    uint64_t local_size = blockDim.x;

    // Load amplitudes to shared memory
    cuDoubleComplex local_sum = make_cuDoubleComplex(0.0, 0.0);
    for (uint64_t i = tid; i < state_dim; i += local_size) {
        cuDoubleComplex amp = amplitudes[i];
        local_sum = make_cuDoubleComplex(
            cuCreal(local_sum) + cuCreal(amp),
            cuCimag(local_sum) + cuCimag(amp)
        );
    }

    shared_amps[tid] = local_sum;
    __syncthreads();

    // Reduction for average
    for (uint32_t s = local_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cuDoubleComplex other = shared_amps[tid + s];
            shared_amps[tid] = make_cuDoubleComplex(
                cuCreal(shared_amps[tid]) + cuCreal(other),
                cuCimag(shared_amps[tid]) + cuCimag(other)
            );
        }
        __syncthreads();
    }

    cuDoubleComplex total = shared_amps[0];
    double scale = 1.0 / (double)state_dim;
    cuDoubleComplex avg = make_cuDoubleComplex(
        cuCreal(total) * scale,
        cuCimag(total) * scale
    );

    __syncthreads();

    // Apply diffusion: 2*avg - amp
    for (uint64_t i = tid; i < state_dim; i += local_size) {
        cuDoubleComplex amp = amplitudes[i];
        amplitudes[i] = make_cuDoubleComplex(
            2.0 * cuCreal(avg) - cuCreal(amp),
            2.0 * cuCimag(avg) - cuCimag(amp)
        );
    }
}

__global__ void grover_batch_search_kernel(
    cuDoubleComplex* batch_states,
    const uint64_t* targets,
    uint64_t* results,
    uint32_t num_searches,
    uint32_t num_qubits,
    uint32_t num_iterations
) {
    extern __shared__ cuDoubleComplex shared_sum[];

    uint32_t search_id = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t local_size = blockDim.x;

    if (search_id >= num_searches) return;

    uint64_t state_dim = 1ULL << num_qubits;
    uint64_t target = targets[search_id];
    cuDoubleComplex* amplitudes = batch_states + (search_id * state_dim);

    for (uint32_t iter = 0; iter < num_iterations; iter++) {
        // Oracle: flip phase of target
        if (tid == 0) {
            cuDoubleComplex amp = amplitudes[target];
            amplitudes[target] = make_cuDoubleComplex(-cuCreal(amp), -cuCimag(amp));
        }
        __syncthreads();

        // Compute sum for diffusion
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
        for (uint64_t i = tid; i < state_dim; i += local_size) {
            cuDoubleComplex amp = amplitudes[i];
            sum = make_cuDoubleComplex(
                cuCreal(sum) + cuCreal(amp),
                cuCimag(sum) + cuCimag(amp)
            );
        }

        shared_sum[tid] = sum;
        __syncthreads();

        for (uint32_t s = local_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                cuDoubleComplex other = shared_sum[tid + s];
                shared_sum[tid] = make_cuDoubleComplex(
                    cuCreal(shared_sum[tid]) + cuCreal(other),
                    cuCimag(shared_sum[tid]) + cuCimag(other)
                );
            }
            __syncthreads();
        }

        cuDoubleComplex total = shared_sum[0];
        double scale = 1.0 / (double)state_dim;
        cuDoubleComplex avg = make_cuDoubleComplex(
            cuCreal(total) * scale,
            cuCimag(total) * scale
        );

        // Apply diffusion
        for (uint64_t i = tid; i < state_dim; i += local_size) {
            cuDoubleComplex amp = amplitudes[i];
            amplitudes[i] = make_cuDoubleComplex(
                2.0 * cuCreal(avg) - cuCreal(amp),
                2.0 * cuCimag(avg) - cuCimag(amp)
            );
        }
        __syncthreads();
    }

    // Find maximum probability state
    if (tid == 0) {
        double max_prob = 0.0;
        uint64_t max_state = 0;
        for (uint64_t i = 0; i < state_dim; i++) {
            cuDoubleComplex amp = amplitudes[i];
            double prob = cuCreal(amp) * cuCreal(amp) + cuCimag(amp) * cuCimag(amp);
            if (prob > max_prob) {
                max_prob = prob;
                max_state = i;
            }
        }
        results[search_id] = max_state;
    }
}

__global__ void compute_probabilities_kernel(
    const cuDoubleComplex* amplitudes,
    double* probabilities,
    uint64_t state_dim
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= state_dim) return;

    cuDoubleComplex amp = amplitudes[tid];
    probabilities[tid] = cuCreal(amp) * cuCreal(amp) + cuCimag(amp) * cuCimag(amp);
}

__global__ void normalize_state_kernel(
    cuDoubleComplex* amplitudes,
    double inv_norm,
    uint64_t state_dim
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= state_dim) return;

    cuDoubleComplex amp = amplitudes[tid];
    amplitudes[tid] = make_cuDoubleComplex(
        cuCreal(amp) * inv_norm,
        cuCimag(amp) * inv_norm
    );
}

__global__ void sum_squared_magnitudes_kernel(
    const cuDoubleComplex* amplitudes,
    double* partial_sums,
    uint64_t state_dim
) {
    extern __shared__ double shared_data_d[];

    uint64_t tid = threadIdx.x;
    uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t grid_size = blockDim.x * gridDim.x;

    double sum = 0.0;
    for (uint64_t i = gid; i < state_dim; i += grid_size) {
        cuDoubleComplex amp = amplitudes[i];
        sum += cuCreal(amp) * cuCreal(amp) + cuCimag(amp) * cuCimag(amp);
    }

    shared_data_d[tid] = sum;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data_d[tid] += shared_data_d[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_data_d[0];
    }
}

// ============================================================================
// HOST FUNCTIONS
// ============================================================================

int cuda_is_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0) ? 1 : 0;
}

cuda_compute_ctx_t* cuda_context_create(int device_id) {
    if (!cuda_is_available()) {
        return NULL;
    }

    cuda_compute_ctx_t* ctx = (cuda_compute_ctx_t*)calloc(1, sizeof(cuda_compute_ctx_t));
    if (!ctx) return NULL;

    ctx->device_id = device_id;

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        snprintf(ctx->error_string, sizeof(ctx->error_string),
                 "Failed to set device %d: %s", device_id, cudaGetErrorString(err));
        free(ctx);
        return NULL;
    }

    err = cudaGetDeviceProperties(&ctx->device_props, device_id);
    if (err != cudaSuccess) {
        snprintf(ctx->error_string, sizeof(ctx->error_string),
                 "Failed to get device properties: %s", cudaGetErrorString(err));
        free(ctx);
        return NULL;
    }

    err = cudaStreamCreate(&ctx->stream);
    if (err != cudaSuccess) {
        snprintf(ctx->error_string, sizeof(ctx->error_string),
                 "Failed to create stream: %s", cudaGetErrorString(err));
        free(ctx);
        return NULL;
    }

    cudaEventCreate(&ctx->start_event);
    cudaEventCreate(&ctx->stop_event);

    return ctx;
}

void cuda_context_free(cuda_compute_ctx_t* ctx) {
    if (!ctx) return;

    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    if (ctx->start_event) cudaEventDestroy(ctx->start_event);
    if (ctx->stop_event) cudaEventDestroy(ctx->stop_event);

    free(ctx);
}

int cuda_get_capabilities(cuda_compute_ctx_t* ctx, cuda_capabilities_t* caps) {
    if (!ctx || !caps) return -1;

    memset(caps, 0, sizeof(cuda_capabilities_t));
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
    caps->supports_unified_memory = (ctx->device_props.managedMemory != 0);
    caps->memory_bandwidth_gbps = (ctx->device_props.memoryBusWidth / 8.0) *
                                   (ctx->device_props.memoryClockRate * 2.0 / 1e6);

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

cuda_buffer_t* cuda_buffer_create(cuda_compute_ctx_t* ctx, size_t size) {
    if (!ctx || size == 0) return NULL;

    cuda_buffer_t* buffer = (cuda_buffer_t*)calloc(1, sizeof(cuda_buffer_t));
    if (!buffer) return NULL;

    buffer->ctx = ctx;
    buffer->size = size;

    // Try unified memory first if supported
    if (ctx->device_props.managedMemory) {
        cudaError_t err = cudaMallocManaged(&buffer->device_ptr, size, cudaMemAttachGlobal);
        if (err == cudaSuccess) {
            buffer->host_ptr = buffer->device_ptr;
            buffer->is_unified = 1;
            return buffer;
        }
    }

    // Fall back to device memory
    cudaError_t err = cudaMalloc(&buffer->device_ptr, size);
    if (err != cudaSuccess) {
        snprintf(ctx->error_string, sizeof(ctx->error_string),
                 "Failed to allocate GPU memory: %s", cudaGetErrorString(err));
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
    return buffer->is_unified ? buffer->host_ptr : NULL;
}

int cuda_buffer_write(cuda_compute_ctx_t* ctx, cuda_buffer_t* buffer,
                      const void* data, size_t size) {
    if (!ctx || !buffer || !data || size > buffer->size) return -1;

    if (buffer->is_unified) {
        memcpy(buffer->device_ptr, data, size);
        cudaDeviceSynchronize();
    } else {
        cudaError_t err = cudaMemcpyAsync(buffer->device_ptr, data, size,
                                          cudaMemcpyHostToDevice, ctx->stream);
        if (err != cudaSuccess) {
            snprintf(ctx->error_string, sizeof(ctx->error_string),
                     "cudaMemcpy failed: %s", cudaGetErrorString(err));
            return -1;
        }
        cudaStreamSynchronize(ctx->stream);
    }

    return 0;
}

int cuda_buffer_read(cuda_compute_ctx_t* ctx, cuda_buffer_t* buffer,
                     void* data, size_t size) {
    if (!ctx || !buffer || !data || size > buffer->size) return -1;

    if (buffer->is_unified) {
        cudaDeviceSynchronize();
        memcpy(data, buffer->device_ptr, size);
    } else {
        cudaError_t err = cudaMemcpyAsync(data, buffer->device_ptr, size,
                                          cudaMemcpyDeviceToHost, ctx->stream);
        if (err != cudaSuccess) {
            snprintf(ctx->error_string, sizeof(ctx->error_string),
                     "cudaMemcpy failed: %s", cudaGetErrorString(err));
            return -1;
        }
        cudaStreamSynchronize(ctx->stream);
    }

    return 0;
}

void cuda_buffer_free(cuda_buffer_t* buffer) {
    if (!buffer) return;

    if (buffer->device_ptr) {
        if (buffer->is_unified) {
            cudaFree(buffer->device_ptr);
        } else {
            cudaFree(buffer->device_ptr);
        }
    }

    free(buffer);
}

// Helper function to calculate launch configuration
static void calculate_launch_config(uint64_t n, uint32_t max_threads,
                                    dim3* grid, dim3* block) {
    uint32_t threads = (max_threads > 256) ? 256 : max_threads;
    uint64_t blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    *block = dim3(threads, 1, 1);
    *grid = dim3((uint32_t)blocks, 1, 1);
}

// ============================================================================
// QUANTUM GATE IMPLEMENTATIONS
// ============================================================================

int cuda_hadamard(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                  uint32_t qubit_index, uint64_t state_dim) {
    if (!ctx || !amplitudes) return -1;

    dim3 grid, block;
    uint64_t num_pairs = state_dim / 2;
    calculate_launch_config(num_pairs, ctx->device_props.maxThreadsPerBlock, &grid, &block);

    cudaEventRecord(ctx->start_event, ctx->stream);

    hadamard_transform_kernel<<<grid, block, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr,
        qubit_index,
        state_dim
    );

    cudaEventRecord(ctx->stop_event, ctx->stream);
    cudaStreamSynchronize(ctx->stream);

    float ms;
    cudaEventElapsedTime(&ms, ctx->start_event, ctx->stop_event);
    ctx->last_execution_time = ms / 1000.0;

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int cuda_hadamard_all(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                      uint32_t num_qubits, uint64_t state_dim) {
    if (!ctx || !amplitudes) return -1;

    dim3 grid, block;
    calculate_launch_config(state_dim, ctx->device_props.maxThreadsPerBlock, &grid, &block);

    cudaEventRecord(ctx->start_event, ctx->stream);

    hadamard_all_kernel<<<grid, block, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr,
        num_qubits,
        state_dim
    );

    cudaEventRecord(ctx->stop_event, ctx->stream);
    cudaStreamSynchronize(ctx->stream);

    float ms;
    cudaEventElapsedTime(&ms, ctx->start_event, ctx->stop_event);
    ctx->last_execution_time = ms / 1000.0;

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int cuda_pauli_x(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                 uint32_t qubit_index, uint64_t state_dim) {
    if (!ctx || !amplitudes) return -1;

    dim3 grid, block;
    uint64_t num_pairs = state_dim / 2;
    calculate_launch_config(num_pairs, ctx->device_props.maxThreadsPerBlock, &grid, &block);

    pauli_x_kernel<<<grid, block, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr,
        qubit_index,
        state_dim
    );

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int cuda_pauli_y(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                 uint32_t qubit_index, uint64_t state_dim) {
    if (!ctx || !amplitudes) return -1;

    dim3 grid, block;
    uint64_t num_pairs = state_dim / 2;
    calculate_launch_config(num_pairs, ctx->device_props.maxThreadsPerBlock, &grid, &block);

    pauli_y_kernel<<<grid, block, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr,
        qubit_index,
        state_dim
    );

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int cuda_pauli_z(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                 uint32_t qubit_index, uint64_t state_dim) {
    if (!ctx || !amplitudes) return -1;

    dim3 grid, block;
    calculate_launch_config(state_dim, ctx->device_props.maxThreadsPerBlock, &grid, &block);

    pauli_z_kernel<<<grid, block, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr,
        qubit_index,
        state_dim
    );

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int cuda_phase_gate(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                    uint32_t qubit_index, double phase, uint64_t state_dim) {
    if (!ctx || !amplitudes) return -1;

    dim3 grid, block;
    calculate_launch_config(state_dim, ctx->device_props.maxThreadsPerBlock, &grid, &block);

    phase_gate_kernel<<<grid, block, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr,
        qubit_index,
        phase,
        state_dim
    );

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int cuda_cnot(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
              uint32_t control_qubit, uint32_t target_qubit, uint64_t state_dim) {
    if (!ctx || !amplitudes) return -1;

    dim3 grid, block;
    calculate_launch_config(state_dim, ctx->device_props.maxThreadsPerBlock, &grid, &block);

    cnot_kernel<<<grid, block, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr,
        control_qubit,
        target_qubit,
        state_dim
    );

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int cuda_oracle_single_target(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                              uint64_t target, uint64_t state_dim) {
    if (!ctx || !amplitudes) return -1;

    // Only need 1 thread for single target
    oracle_single_target_kernel<<<1, 1, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr,
        target
    );

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int cuda_sparse_oracle(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                       const uint64_t* targets, uint32_t num_targets,
                       uint64_t state_dim) {
    if (!ctx || !amplitudes || !targets) return -1;

    // Copy targets to device
    uint64_t* d_targets;
    cudaMalloc(&d_targets, num_targets * sizeof(uint64_t));
    cudaMemcpy(d_targets, targets, num_targets * sizeof(uint64_t), cudaMemcpyHostToDevice);

    dim3 grid, block;
    calculate_launch_config(state_dim, ctx->device_props.maxThreadsPerBlock, &grid, &block);

    sparse_oracle_kernel<<<grid, block, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr,
        d_targets,
        num_targets,
        state_dim
    );

    cudaFree(d_targets);

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int cuda_grover_diffusion(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                          uint32_t num_qubits, uint64_t state_dim) {
    if (!ctx || !amplitudes) return -1;

    // For small states, use fused kernel
    if (state_dim <= 65536) {
        uint32_t threads = 256;
        size_t shared_size = threads * sizeof(cuDoubleComplex);

        grover_diffusion_fused_kernel<<<1, threads, shared_size, ctx->stream>>>(
            (cuDoubleComplex*)amplitudes->device_ptr,
            num_qubits,
            state_dim
        );
    } else {
        // Two-phase approach for larger states
        dim3 grid, block;
        uint32_t threads = 256;
        uint32_t blocks = (state_dim + threads - 1) / threads;
        if (blocks > 65535) blocks = 65535;

        // Allocate partial sums buffer
        cuDoubleComplex* d_partial;
        cudaMalloc(&d_partial, blocks * sizeof(cuDoubleComplex));

        size_t shared_size = threads * sizeof(cuDoubleComplex);

        // Phase 1: Compute partial sums
        diffusion_sum_kernel<<<blocks, threads, shared_size, ctx->stream>>>(
            (cuDoubleComplex*)amplitudes->device_ptr,
            d_partial,
            state_dim
        );

        // Reduce partial sums on CPU for simplicity
        cuDoubleComplex* h_partial = (cuDoubleComplex*)malloc(blocks * sizeof(cuDoubleComplex));
        cudaMemcpy(h_partial, d_partial, blocks * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

        cuDoubleComplex total = make_cuDoubleComplex(0.0, 0.0);
        for (uint32_t i = 0; i < blocks; i++) {
            total = make_cuDoubleComplex(
                cuCreal(total) + cuCreal(h_partial[i]),
                cuCimag(total) + cuCimag(h_partial[i])
            );
        }

        double scale = 1.0 / (double)state_dim;
        cuDoubleComplex avg = make_cuDoubleComplex(
            cuCreal(total) * scale,
            cuCimag(total) * scale
        );

        // Store average for kernel
        cudaMemcpy(d_partial, &avg, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

        // Phase 2: Apply diffusion
        diffusion_apply_kernel<<<blocks, threads, 0, ctx->stream>>>(
            (cuDoubleComplex*)amplitudes->device_ptr,
            d_partial,
            state_dim
        );

        free(h_partial);
        cudaFree(d_partial);
    }

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int cuda_grover_iteration(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                          uint64_t target, uint32_t num_qubits, uint64_t state_dim) {
    // Oracle
    int err = cuda_oracle_single_target(ctx, amplitudes, target, state_dim);
    if (err != 0) return err;

    // Diffusion
    return cuda_grover_diffusion(ctx, amplitudes, num_qubits, state_dim);
}

int cuda_grover_batch_search(cuda_compute_ctx_t* ctx, cuda_buffer_t* batch_states,
                             const uint64_t* targets, uint64_t* results,
                             uint32_t num_searches, uint32_t num_qubits,
                             uint32_t num_iterations) {
    if (!ctx || !batch_states || !targets || !results) return -1;

    uint64_t state_dim = 1ULL << num_qubits;

    // Copy targets to device
    uint64_t* d_targets;
    cudaMalloc(&d_targets, num_searches * sizeof(uint64_t));
    cudaMemcpy(d_targets, targets, num_searches * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Allocate results on device
    uint64_t* d_results;
    cudaMalloc(&d_results, num_searches * sizeof(uint64_t));

    uint32_t threads = 256;
    size_t shared_size = threads * sizeof(cuDoubleComplex);

    grover_batch_search_kernel<<<num_searches, threads, shared_size, ctx->stream>>>(
        (cuDoubleComplex*)batch_states->device_ptr,
        d_targets,
        d_results,
        num_searches,
        num_qubits,
        num_iterations
    );

    cudaStreamSynchronize(ctx->stream);

    // Copy results back
    cudaMemcpy(results, d_results, num_searches * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_targets);
    cudaFree(d_results);

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int cuda_compute_probabilities(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                               cuda_buffer_t* probabilities, uint64_t state_dim) {
    if (!ctx || !amplitudes || !probabilities) return -1;

    dim3 grid, block;
    calculate_launch_config(state_dim, ctx->device_props.maxThreadsPerBlock, &grid, &block);

    compute_probabilities_kernel<<<grid, block, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr,
        (double*)probabilities->device_ptr,
        state_dim
    );

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int cuda_normalize_state(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                         double norm, uint64_t state_dim) {
    if (!ctx || !amplitudes || norm <= 0.0) return -1;

    dim3 grid, block;
    calculate_launch_config(state_dim, ctx->device_props.maxThreadsPerBlock, &grid, &block);

    double inv_norm = 1.0 / norm;

    normalize_state_kernel<<<grid, block, 0, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr,
        inv_norm,
        state_dim
    );

    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int cuda_sum_squared_magnitudes(cuda_compute_ctx_t* ctx, cuda_buffer_t* amplitudes,
                                uint64_t state_dim, double* result) {
    if (!ctx || !amplitudes || !result) return -1;

    uint32_t threads = 256;
    uint32_t blocks = (state_dim + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    double* d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(double));

    size_t shared_size = threads * sizeof(double);

    sum_squared_magnitudes_kernel<<<blocks, threads, shared_size, ctx->stream>>>(
        (cuDoubleComplex*)amplitudes->device_ptr,
        d_partial,
        state_dim
    );

    // Reduce on CPU
    double* h_partial = (double*)malloc(blocks * sizeof(double));
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(double), cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for (uint32_t i = 0; i < blocks; i++) {
        sum += h_partial[i];
    }

    *result = sum;

    free(h_partial);
    cudaFree(d_partial);

    return 0;
}

void cuda_synchronize(cuda_compute_ctx_t* ctx) {
    if (ctx && ctx->stream) {
        cudaStreamSynchronize(ctx->stream);
    }
}

double cuda_get_last_execution_time(cuda_compute_ctx_t* ctx) {
    return ctx ? ctx->last_execution_time : 0.0;
}

const char* cuda_get_error_string(cuda_compute_ctx_t* ctx) {
    if (!ctx) return "No context";
    if (ctx->has_error) return ctx->error_string;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return cudaGetErrorString(err);
    }
    return NULL;
}

#else /* !HAS_CUDA */

// Stub implementations when CUDA is not available

int cuda_is_available(void) { return 0; }

cuda_compute_ctx_t* cuda_context_create(int device_id) {
    (void)device_id;
    return NULL;
}

void cuda_context_free(cuda_compute_ctx_t* ctx) { (void)ctx; }

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

void cuda_buffer_free(cuda_buffer_t* buffer) { (void)buffer; }

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

void cuda_synchronize(cuda_compute_ctx_t* ctx) { (void)ctx; }

double cuda_get_last_execution_time(cuda_compute_ctx_t* ctx) {
    (void)ctx;
    return 0.0;
}

const char* cuda_get_error_string(cuda_compute_ctx_t* ctx) {
    (void)ctx;
    return "CUDA not available";
}

#endif /* HAS_CUDA */
