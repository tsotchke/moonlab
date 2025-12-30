/**
 * @file gpu_backend.c
 * @brief Unified GPU backend implementation with runtime dispatch
 *
 * Implements backend detection, initialization, and dispatch to:
 * - Metal (macOS)
 * - OpenCL (cross-platform)
 * - Vulkan (cross-platform)
 * - CUDA (NVIDIA)
 * - cuQuantum (NVIDIA)
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#include "gpu_backend.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Backend-specific includes (conditionally compiled)
#ifdef HAS_METAL
#include "../gpu_metal.h"
#endif

#ifdef HAS_OPENCL
#include "backends/gpu_opencl.h"
#endif

#ifdef HAS_VULKAN
#include "backends/gpu_vulkan.h"
#endif

#ifdef HAS_CUDA
#include "backends/gpu_cuda.h"
#endif

#ifdef HAS_CUQUANTUM
#include "backends/gpu_cuquantum.h"
#endif

// ============================================================================
// INTERNAL STRUCTURES
// ============================================================================

/**
 * @brief GPU context internal structure
 */
struct gpu_context {
    gpu_backend_type_t backend_type;
    gpu_capabilities_t capabilities;
    char last_error[512];
    double last_exec_time;
    int perf_monitoring;

    // Backend-specific context pointers
    void* metal_ctx;
    void* opencl_ctx;
    void* vulkan_ctx;
    void* cuda_ctx;
    void* cuquantum_ctx;
};

/**
 * @brief GPU buffer internal structure
 */
struct gpu_buffer {
    gpu_context_t* ctx;
    size_t size;
    void* host_ptr;       // CPU-accessible pointer

    // Backend-specific handles
    void* metal_buffer;
    void* opencl_buffer;
    void* vulkan_buffer;
    void* cuda_buffer;
};

// ============================================================================
// BACKEND DETECTION
// ============================================================================

/**
 * @brief Check Metal availability (macOS only)
 */
static int detect_metal(void) {
#ifdef HAS_METAL
    return metal_is_available();
#else
    return 0;
#endif
}

/**
 * @brief Check OpenCL availability
 */
static int detect_opencl(void) {
#ifdef HAS_OPENCL
    return opencl_is_available();
#else
    return 0;
#endif
}

/**
 * @brief Check Vulkan availability
 */
static int detect_vulkan(void) {
#ifdef HAS_VULKAN
    return vulkan_is_available();
#else
    return 0;
#endif
}

/**
 * @brief Check CUDA availability
 */
static int detect_cuda(void) {
#ifdef HAS_CUDA
    return cuda_is_available();
#else
    return 0;
#endif
}

/**
 * @brief Check cuQuantum availability
 */
static int detect_cuquantum(void) {
#ifdef HAS_CUQUANTUM
    return cuquantum_is_available();
#else
    return 0;
#endif
}

/**
 * @brief Select best available backend
 */
static gpu_backend_type_t select_best_backend(void) {
    // Priority order based on platform
#if defined(__APPLE__)
    // macOS: Metal preferred
    if (detect_metal()) return GPU_BACKEND_METAL;
    if (detect_opencl()) return GPU_BACKEND_OPENCL;
#else
    // Linux/Windows: NVIDIA preferred, then Vulkan, then OpenCL
    if (detect_cuquantum()) return GPU_BACKEND_CUQUANTUM;
    if (detect_cuda()) return GPU_BACKEND_CUDA;
    if (detect_vulkan()) return GPU_BACKEND_VULKAN;
    if (detect_opencl()) return GPU_BACKEND_OPENCL;
#endif

    return GPU_BACKEND_NONE;
}

// ============================================================================
// INITIALIZATION
// ============================================================================

gpu_context_t* gpu_compute_init(gpu_backend_type_t preferred) {
    gpu_context_t* ctx = calloc(1, sizeof(gpu_context_t));
    if (!ctx) return NULL;

    // Select backend
    gpu_backend_type_t selected = preferred;
    if (preferred == GPU_BACKEND_AUTO) {
        selected = select_best_backend();
    }

    // Verify requested backend is available
    int available = 0;
    switch (selected) {
        case GPU_BACKEND_METAL:
            available = detect_metal();
            break;
        case GPU_BACKEND_OPENCL:
            available = detect_opencl();
            break;
        case GPU_BACKEND_VULKAN:
            available = detect_vulkan();
            break;
        case GPU_BACKEND_CUDA:
            available = detect_cuda();
            break;
        case GPU_BACKEND_CUQUANTUM:
            available = detect_cuquantum();
            break;
        case GPU_BACKEND_NONE:
            // Explicitly requested no GPU
            ctx->backend_type = GPU_BACKEND_NONE;
            return ctx;
        default:
            break;
    }

    if (!available) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "Requested backend %s not available", gpu_backend_name(selected));
        free(ctx);
        return NULL;
    }

    ctx->backend_type = selected;

    // Initialize backend-specific context
    int init_result = -1;
    switch (selected) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL:
            ctx->metal_ctx = metal_compute_init();
            init_result = ctx->metal_ctx ? 0 : -1;
            break;
#endif
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL:
            ctx->opencl_ctx = opencl_compute_init();
            init_result = ctx->opencl_ctx ? 0 : -1;
            break;
#endif
#ifdef HAS_VULKAN
        case GPU_BACKEND_VULKAN:
            ctx->vulkan_ctx = vulkan_compute_init();
            init_result = ctx->vulkan_ctx ? 0 : -1;
            break;
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            ctx->cuda_ctx = cuda_context_create(0);
            init_result = ctx->cuda_ctx ? 0 : -1;
            break;
#endif
#ifdef HAS_CUQUANTUM
        case GPU_BACKEND_CUQUANTUM:
            ctx->cuquantum_ctx = cuquantum_context_create(0, NULL);
            init_result = ctx->cuquantum_ctx ? 0 : -1;
            break;
#endif
        default:
            init_result = -1;
            break;
    }

    if (init_result != 0) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "Failed to initialize %s backend", gpu_backend_name(selected));
        free(ctx);
        return NULL;
    }

    // Populate capabilities
    gpu_get_capabilities(ctx, &ctx->capabilities);

    return ctx;
}

void gpu_compute_free(gpu_context_t* ctx) {
    if (!ctx) return;

    switch (ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL:
            if (ctx->metal_ctx) metal_compute_free(ctx->metal_ctx);
            break;
#endif
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL:
            if (ctx->opencl_ctx) opencl_compute_free(ctx->opencl_ctx);
            break;
#endif
#ifdef HAS_VULKAN
        case GPU_BACKEND_VULKAN:
            if (ctx->vulkan_ctx) vulkan_compute_free(ctx->vulkan_ctx);
            break;
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            if (ctx->cuda_ctx) cuda_context_free(ctx->cuda_ctx);
            break;
#endif
#ifdef HAS_CUQUANTUM
        case GPU_BACKEND_CUQUANTUM:
            if (ctx->cuquantum_ctx) cuquantum_context_free(ctx->cuquantum_ctx);
            break;
#endif
        default:
            break;
    }

    free(ctx);
}

int gpu_is_available(void) {
    return detect_metal() || detect_opencl() || detect_vulkan() ||
           detect_cuda() || detect_cuquantum();
}

gpu_backend_type_t gpu_get_backend_type(gpu_context_t* ctx) {
    return ctx ? ctx->backend_type : GPU_BACKEND_NONE;
}

const char* gpu_backend_name(gpu_backend_type_t type) {
    switch (type) {
        case GPU_BACKEND_NONE:      return "None (CPU)";
        case GPU_BACKEND_METAL:     return "Metal";
        case GPU_BACKEND_OPENCL:    return "OpenCL";
        case GPU_BACKEND_VULKAN:    return "Vulkan";
        case GPU_BACKEND_CUDA:      return "CUDA";
        case GPU_BACKEND_CUQUANTUM: return "cuQuantum";
        case GPU_BACKEND_AUTO:      return "Auto";
        default:                    return "Unknown";
    }
}

gpu_error_t gpu_get_capabilities(gpu_context_t* ctx, gpu_capabilities_t* caps) {
    if (!ctx || !caps) return GPU_ERROR_INVALID_PARAM;

    memset(caps, 0, sizeof(*caps));

    switch (ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL: {
            uint32_t max_threads, num_cores;
            metal_get_device_info(ctx->metal_ctx, caps->device_name,
                                  &max_threads, &num_cores);
            caps->max_threads_per_group = max_threads;
            caps->max_compute_units = num_cores;
            caps->supports_unified_memory = 1;
            caps->supports_double = 1;
            strncpy(caps->vendor_name, "Apple", sizeof(caps->vendor_name) - 1);
            break;
        }
#endif
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL: {
            uint32_t max_work_group, compute_units;
            opencl_get_device_info(ctx->opencl_ctx, caps->device_name,
                                   &max_work_group, &compute_units);
            caps->max_threads_per_group = max_work_group;
            caps->max_compute_units = compute_units;
            strncpy(caps->vendor_name, "OpenCL", sizeof(caps->vendor_name) - 1);
            break;
        }
#endif
#ifdef HAS_VULKAN
        case GPU_BACKEND_VULKAN:
            return vulkan_get_capabilities(ctx->vulkan_ctx, caps);
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA: {
            cuda_capabilities_t cuda_caps;
            int err = cuda_get_capabilities(ctx->cuda_ctx, &cuda_caps);
            if (err != 0) return GPU_ERROR_KERNEL_FAILED;
            strncpy(caps->device_name, cuda_caps.device_name, sizeof(caps->device_name) - 1);
            strncpy(caps->vendor_name, "NVIDIA", sizeof(caps->vendor_name) - 1);
            caps->global_memory = cuda_caps.global_memory;
            caps->local_memory = cuda_caps.shared_memory_per_block;
            caps->max_compute_units = cuda_caps.multiprocessor_count;
            caps->max_threads_per_group = cuda_caps.max_threads_per_block;
            caps->supports_unified_memory = cuda_caps.supports_unified_memory;
            caps->supports_double = 1;
            caps->memory_bandwidth_gbps = cuda_caps.memory_bandwidth_gbps;
            break;
        }
#endif
#ifdef HAS_CUQUANTUM
        case GPU_BACKEND_CUQUANTUM: {
            cuquantum_capabilities_t cq_caps;
            int err = cuquantum_get_capabilities(ctx->cuquantum_ctx, &cq_caps);
            if (err != 0) return GPU_ERROR_KERNEL_FAILED;
            strncpy(caps->device_name, cq_caps.device_name, sizeof(caps->device_name) - 1);
            strncpy(caps->vendor_name, "NVIDIA cuQuantum", sizeof(caps->vendor_name) - 1);
            caps->global_memory = cq_caps.global_memory;
            caps->max_compute_units = cq_caps.multiprocessor_count;
            caps->supports_unified_memory = 1;
            caps->supports_double = 1;
            caps->memory_bandwidth_gbps = cq_caps.memory_bandwidth_gbps;
            break;
        }
#endif
        default:
            strncpy(caps->device_name, "CPU (no GPU)", sizeof(caps->device_name) - 1);
            return GPU_SUCCESS;
    }

    return GPU_SUCCESS;
}

void gpu_print_device_info(gpu_context_t* ctx) {
    if (!ctx) {
        printf("GPU Context: NULL\n");
        return;
    }

    printf("\n");
    printf("========================================\n");
    printf("GPU Backend Information\n");
    printf("========================================\n");
    printf("Backend:      %s\n", gpu_backend_name(ctx->backend_type));
    printf("Device:       %s\n", ctx->capabilities.device_name);
    printf("Vendor:       %s\n", ctx->capabilities.vendor_name);
    printf("Compute Units: %u\n", ctx->capabilities.max_compute_units);
    printf("Max Threads:  %u\n", ctx->capabilities.max_threads_per_group);
    printf("Global Mem:   %.2f GB\n", ctx->capabilities.global_memory / (1024.0*1024.0*1024.0));
    printf("Unified Mem:  %s\n", ctx->capabilities.supports_unified_memory ? "Yes" : "No");
    printf("Double FP:    %s\n", ctx->capabilities.supports_double ? "Yes" : "No");
    printf("========================================\n\n");
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

gpu_buffer_t* gpu_buffer_create(gpu_context_t* ctx, size_t size) {
    if (!ctx || size == 0) return NULL;

    gpu_buffer_t* buffer = calloc(1, sizeof(gpu_buffer_t));
    if (!buffer) return NULL;

    buffer->ctx = ctx;
    buffer->size = size;

    switch (ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL:
            buffer->metal_buffer = metal_buffer_create(ctx->metal_ctx, size);
            if (!buffer->metal_buffer) {
                free(buffer);
                return NULL;
            }
            buffer->host_ptr = metal_buffer_contents(buffer->metal_buffer);
            break;
#endif
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL:
            buffer->opencl_buffer = opencl_buffer_create(ctx->opencl_ctx, size);
            if (!buffer->opencl_buffer) {
                free(buffer);
                return NULL;
            }
            break;
#endif
#ifdef HAS_VULKAN
        case GPU_BACKEND_VULKAN:
            buffer->vulkan_buffer = vulkan_buffer_create(ctx->vulkan_ctx, size);
            if (!buffer->vulkan_buffer) {
                free(buffer);
                return NULL;
            }
            break;
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            buffer->cuda_buffer = cuda_buffer_create(ctx->cuda_ctx, size);
            if (!buffer->cuda_buffer) {
                free(buffer);
                return NULL;
            }
            buffer->host_ptr = cuda_buffer_contents(buffer->cuda_buffer);
            break;
#endif
#ifdef HAS_CUQUANTUM
        case GPU_BACKEND_CUQUANTUM: {
            // For cuQuantum, calculate num_qubits from size (assuming complex double)
            // size = 2^n * sizeof(cuDoubleComplex) = 2^n * 16
            uint32_t num_qubits = 0;
            size_t temp = size / 16;
            while (temp > 1) { temp >>= 1; num_qubits++; }
            buffer->cuda_buffer = cuquantum_statevec_create(ctx->cuquantum_ctx, num_qubits);
            if (!buffer->cuda_buffer) {
                free(buffer);
                return NULL;
            }
            buffer->host_ptr = cuquantum_buffer_device_ptr(buffer->cuda_buffer);
            break;
        }
#endif
        default:
            // CPU fallback
            buffer->host_ptr = calloc(1, size);
            if (!buffer->host_ptr) {
                free(buffer);
                return NULL;
            }
            break;
    }

    return buffer;
}

gpu_buffer_t* gpu_buffer_create_from_data(gpu_context_t* ctx, void* data, size_t size) {
    if (!ctx || !data || size == 0) return NULL;

    gpu_buffer_t* buffer = calloc(1, sizeof(gpu_buffer_t));
    if (!buffer) return NULL;

    buffer->ctx = ctx;
    buffer->size = size;

    switch (ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL:
            buffer->metal_buffer = metal_buffer_create_from_data(ctx->metal_ctx, data, size);
            if (!buffer->metal_buffer) {
                free(buffer);
                return NULL;
            }
            buffer->host_ptr = metal_buffer_contents(buffer->metal_buffer);
            break;
#endif
        default:
            // For other backends, create buffer and copy data
            buffer = gpu_buffer_create(ctx, size);
            if (buffer) {
                gpu_buffer_write(buffer, data, size, 0);
            }
            return buffer;
    }

    return buffer;
}

void* gpu_buffer_contents(gpu_buffer_t* buffer) {
    if (!buffer) return NULL;
    return buffer->host_ptr;
}

gpu_error_t gpu_buffer_write(gpu_buffer_t* buffer, const void* data, size_t size, size_t offset) {
    if (!buffer || !data) return GPU_ERROR_INVALID_PARAM;
    if (offset + size > buffer->size) return GPU_ERROR_INVALID_PARAM;

    switch (buffer->ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL:
            // Metal uses unified memory - direct memcpy
            memcpy((char*)buffer->host_ptr + offset, data, size);
            break;
#endif
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL:
            // Note: OpenCL API doesn't support offset, so we ignore it for now
            // A full implementation would use clEnqueueWriteBuffer with offset
            if (offset != 0) return GPU_ERROR_NOT_SUPPORTED;
            return opencl_buffer_write(buffer->ctx->opencl_ctx, buffer->opencl_buffer, data, size)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_VULKAN
        case GPU_BACKEND_VULKAN:
            return vulkan_buffer_write(buffer->vulkan_buffer, data, size, offset);
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            if (offset != 0) return GPU_ERROR_NOT_SUPPORTED;
            return cuda_buffer_write(buffer->ctx->cuda_ctx, buffer->cuda_buffer, data, size)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_CUQUANTUM
        case GPU_BACKEND_CUQUANTUM:
            if (offset != 0) return GPU_ERROR_NOT_SUPPORTED;
            return cuquantum_buffer_write(buffer->ctx->cuquantum_ctx, buffer->cuda_buffer, data, size)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
        default:
            if (buffer->host_ptr) {
                memcpy((char*)buffer->host_ptr + offset, data, size);
            }
            break;
    }

    return GPU_SUCCESS;
}

gpu_error_t gpu_buffer_read(gpu_buffer_t* buffer, void* data, size_t size, size_t offset) {
    if (!buffer || !data) return GPU_ERROR_INVALID_PARAM;
    if (offset + size > buffer->size) return GPU_ERROR_INVALID_PARAM;

    switch (buffer->ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL:
            // Metal uses unified memory - direct memcpy
            memcpy(data, (char*)buffer->host_ptr + offset, size);
            break;
#endif
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL:
            // Note: OpenCL API doesn't support offset, so we ignore it for now
            if (offset != 0) return GPU_ERROR_NOT_SUPPORTED;
            return opencl_buffer_read(buffer->ctx->opencl_ctx, buffer->opencl_buffer, data, size)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_VULKAN
        case GPU_BACKEND_VULKAN:
            return vulkan_buffer_read(buffer->vulkan_buffer, data, size, offset);
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            if (offset != 0) return GPU_ERROR_NOT_SUPPORTED;
            return cuda_buffer_read(buffer->ctx->cuda_ctx, buffer->cuda_buffer, data, size)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_CUQUANTUM
        case GPU_BACKEND_CUQUANTUM:
            if (offset != 0) return GPU_ERROR_NOT_SUPPORTED;
            return cuquantum_buffer_read(buffer->ctx->cuquantum_ctx, buffer->cuda_buffer, data, size)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
        default:
            if (buffer->host_ptr) {
                memcpy(data, (char*)buffer->host_ptr + offset, size);
            }
            break;
    }

    return GPU_SUCCESS;
}

void gpu_buffer_free(gpu_buffer_t* buffer) {
    if (!buffer) return;

    switch (buffer->ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL:
            if (buffer->metal_buffer) metal_buffer_free(buffer->metal_buffer);
            break;
#endif
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL:
            if (buffer->opencl_buffer) opencl_buffer_free(buffer->opencl_buffer);
            break;
#endif
#ifdef HAS_VULKAN
        case GPU_BACKEND_VULKAN:
            if (buffer->vulkan_buffer) vulkan_buffer_free(buffer->vulkan_buffer);
            break;
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
        case GPU_BACKEND_CUQUANTUM:
            if (buffer->cuda_buffer) cuda_buffer_free(buffer->cuda_buffer);
            break;
#endif
        default:
            if (buffer->host_ptr) free(buffer->host_ptr);
            break;
    }

    free(buffer);
}

// ============================================================================
// QUANTUM GATE OPERATIONS
// ============================================================================

gpu_error_t gpu_hadamard(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                         uint32_t qubit_index, uint64_t state_dim) {
    if (!ctx || !amplitudes) return GPU_ERROR_INVALID_PARAM;

    switch (ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL:
            return metal_hadamard(ctx->metal_ctx, amplitudes->metal_buffer,
                                  qubit_index, (uint32_t)state_dim) == 0 ?
                   GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL:
            return opencl_hadamard(ctx->opencl_ctx, amplitudes->opencl_buffer,
                                   qubit_index, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_VULKAN
        case GPU_BACKEND_VULKAN:
            return vulkan_hadamard(ctx->vulkan_ctx, amplitudes->vulkan_buffer,
                                   qubit_index, state_dim);
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_hadamard(ctx->cuda_ctx, amplitudes->cuda_buffer,
                                 qubit_index, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_CUQUANTUM
        case GPU_BACKEND_CUQUANTUM: {
            // cuQuantum API needs num_qubits, calculate from state_dim
            uint32_t num_qubits = 0;
            uint64_t temp = state_dim;
            while (temp > 1) { temp >>= 1; num_qubits++; }
            return cuquantum_hadamard(ctx->cuquantum_ctx, amplitudes->cuda_buffer,
                                      num_qubits, qubit_index)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
        }
#endif
        default:
            return GPU_ERROR_NOT_SUPPORTED;
    }
}

gpu_error_t gpu_hadamard_all(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                             uint32_t num_qubits, uint64_t state_dim) {
    if (!ctx || !amplitudes) return GPU_ERROR_INVALID_PARAM;

    switch (ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL:
            return metal_hadamard_all(ctx->metal_ctx, amplitudes->metal_buffer,
                                      num_qubits, (uint32_t)state_dim) == 0 ?
                   GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL:
            return opencl_hadamard_all(ctx->opencl_ctx, amplitudes->opencl_buffer,
                                       num_qubits, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_VULKAN
        case GPU_BACKEND_VULKAN:
            return vulkan_hadamard_all(ctx->vulkan_ctx, amplitudes->vulkan_buffer,
                                       num_qubits, state_dim);
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_hadamard_all(ctx->cuda_ctx, amplitudes->cuda_buffer,
                                     num_qubits, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_CUQUANTUM
        case GPU_BACKEND_CUQUANTUM:
            return cuquantum_hadamard_all(ctx->cuquantum_ctx, amplitudes->cuda_buffer,
                                          num_qubits)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
        default:
            // Fallback: apply individual Hadamards
            for (uint32_t q = 0; q < num_qubits; q++) {
                gpu_error_t err = gpu_hadamard(ctx, amplitudes, q, state_dim);
                if (err != GPU_SUCCESS) return err;
            }
            return GPU_SUCCESS;
    }
}

gpu_error_t gpu_pauli_x(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                        uint32_t qubit_index, uint64_t state_dim) {
    if (!ctx || !amplitudes) return GPU_ERROR_INVALID_PARAM;

    switch (ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL:
            return metal_pauli_x(ctx->metal_ctx, amplitudes->metal_buffer,
                                 qubit_index, (uint32_t)state_dim) == 0 ?
                   GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL:
            return opencl_pauli_x(ctx->opencl_ctx, amplitudes->opencl_buffer,
                                  qubit_index, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_pauli_x(ctx->cuda_ctx, amplitudes->cuda_buffer,
                                qubit_index, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
        default:
            return GPU_ERROR_NOT_SUPPORTED;
    }
}

gpu_error_t gpu_pauli_z(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                        uint32_t qubit_index, uint64_t state_dim) {
    if (!ctx || !amplitudes) return GPU_ERROR_INVALID_PARAM;

    switch (ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL:
            return metal_pauli_z(ctx->metal_ctx, amplitudes->metal_buffer,
                                 qubit_index, (uint32_t)state_dim) == 0 ?
                   GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL:
            return opencl_pauli_z(ctx->opencl_ctx, amplitudes->opencl_buffer,
                                  qubit_index, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_pauli_z(ctx->cuda_ctx, amplitudes->cuda_buffer,
                                qubit_index, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
        default:
            return GPU_ERROR_NOT_SUPPORTED;
    }
}

gpu_error_t gpu_phase(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                      uint32_t qubit_index, double theta, uint64_t state_dim) {
    if (!ctx || !amplitudes) return GPU_ERROR_INVALID_PARAM;

    switch (ctx->backend_type) {
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL:
            return opencl_phase(ctx->opencl_ctx, amplitudes->opencl_buffer,
                                qubit_index, (float)theta, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_phase_gate(ctx->cuda_ctx, amplitudes->cuda_buffer,
                                   qubit_index, theta, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
        default:
            return GPU_ERROR_NOT_SUPPORTED;
    }
}

gpu_error_t gpu_cnot(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                     uint32_t control, uint32_t target, uint64_t state_dim) {
    if (!ctx || !amplitudes) return GPU_ERROR_INVALID_PARAM;

    switch (ctx->backend_type) {
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL:
            return opencl_cnot(ctx->opencl_ctx, amplitudes->opencl_buffer,
                               control, target, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_cnot(ctx->cuda_ctx, amplitudes->cuda_buffer,
                             control, target, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_CUQUANTUM
        case GPU_BACKEND_CUQUANTUM: {
            // cuQuantum API needs num_qubits, calculate from state_dim
            uint32_t num_qubits = 0;
            uint64_t temp = state_dim;
            while (temp > 1) { temp >>= 1; num_qubits++; }
            return cuquantum_cnot(ctx->cuquantum_ctx, amplitudes->cuda_buffer,
                                  num_qubits, control, target)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
        }
#endif
        default:
            return GPU_ERROR_NOT_SUPPORTED;
    }
}

// ============================================================================
// GROVER'S ALGORITHM OPERATIONS
// ============================================================================

gpu_error_t gpu_oracle_single_target(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                                     uint64_t target_state, uint64_t state_dim) {
    if (!ctx || !amplitudes) return GPU_ERROR_INVALID_PARAM;

    switch (ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL:
            return metal_oracle(ctx->metal_ctx, amplitudes->metal_buffer,
                                (uint32_t)target_state, (uint32_t)state_dim) == 0 ?
                   GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL:
            return opencl_oracle(ctx->opencl_ctx, amplitudes->opencl_buffer,
                                 target_state, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_oracle_single_target(ctx->cuda_ctx, amplitudes->cuda_buffer,
                                             target_state, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
        default:
            return GPU_ERROR_NOT_SUPPORTED;
    }
}

gpu_error_t gpu_oracle_multi_target(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                                    const uint64_t* targets, uint32_t num_targets,
                                    uint64_t state_dim) {
    if (!ctx || !amplitudes || !targets) return GPU_ERROR_INVALID_PARAM;

    switch (ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL: {
            // Convert uint64_t to uint32_t for Metal
            uint32_t* targets32 = malloc(num_targets * sizeof(uint32_t));
            if (!targets32) return GPU_ERROR_ALLOC_FAILED;
            for (uint32_t i = 0; i < num_targets; i++) {
                targets32[i] = (uint32_t)targets[i];
            }
            int result = metal_oracle_multi(ctx->metal_ctx, amplitudes->metal_buffer,
                                            targets32, num_targets, (uint32_t)state_dim);
            free(targets32);
            return result == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
        }
#endif
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL:
            return opencl_oracle_multi(ctx->opencl_ctx, amplitudes->opencl_buffer,
                                       targets, num_targets, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_sparse_oracle(ctx->cuda_ctx, amplitudes->cuda_buffer,
                                      targets, num_targets, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
        default:
            // Fallback: apply single oracles
            for (uint32_t i = 0; i < num_targets; i++) {
                gpu_error_t err = gpu_oracle_single_target(ctx, amplitudes, targets[i], state_dim);
                if (err != GPU_SUCCESS) return err;
            }
            return GPU_SUCCESS;
    }
}

gpu_error_t gpu_grover_diffusion(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                                 uint32_t num_qubits, uint64_t state_dim) {
    if (!ctx || !amplitudes) return GPU_ERROR_INVALID_PARAM;

    switch (ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL:
            return metal_grover_diffusion(ctx->metal_ctx, amplitudes->metal_buffer,
                                          num_qubits, (uint32_t)state_dim) == 0 ?
                   GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL:
            return opencl_grover_diffusion(ctx->opencl_ctx, amplitudes->opencl_buffer,
                                           num_qubits, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_grover_diffusion(ctx->cuda_ctx, amplitudes->cuda_buffer,
                                         num_qubits, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
        default:
            return GPU_ERROR_NOT_SUPPORTED;
    }
}

gpu_error_t gpu_grover_iteration(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                                 uint64_t target_state, uint32_t num_qubits,
                                 uint64_t state_dim) {
    if (!ctx || !amplitudes) return GPU_ERROR_INVALID_PARAM;

    switch (ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL:
            return metal_grover_iteration(ctx->metal_ctx, amplitudes->metal_buffer,
                                          (uint32_t)target_state, num_qubits,
                                          (uint32_t)state_dim) == 0 ?
                   GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
// OpenCL uses fallback path (oracle + diffusion)
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_grover_iteration(ctx->cuda_ctx, amplitudes->cuda_buffer,
                                         target_state, num_qubits, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
        default:
            // Fallback: oracle + diffusion
            gpu_error_t err = gpu_oracle_single_target(ctx, amplitudes, target_state, state_dim);
            if (err != GPU_SUCCESS) return err;
            return gpu_grover_diffusion(ctx, amplitudes, num_qubits, state_dim);
    }
}

gpu_error_t gpu_grover_batch_search(gpu_context_t* ctx, gpu_buffer_t* batch_states,
                                    const uint64_t* targets, uint64_t* results,
                                    uint32_t num_searches, uint32_t num_qubits,
                                    uint32_t num_iterations) {
    if (!ctx || !batch_states || !targets || !results) return GPU_ERROR_INVALID_PARAM;

    switch (ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL: {
            // Convert types for Metal API
            uint32_t* targets32 = malloc(num_searches * sizeof(uint32_t));
            uint32_t* results32 = malloc(num_searches * sizeof(uint32_t));
            if (!targets32 || !results32) {
                free(targets32);
                free(results32);
                return GPU_ERROR_ALLOC_FAILED;
            }
            for (uint32_t i = 0; i < num_searches; i++) {
                targets32[i] = (uint32_t)targets[i];
            }
            int result = metal_grover_batch_search(ctx->metal_ctx, batch_states->metal_buffer,
                                                   targets32, results32, num_searches,
                                                   num_qubits, num_iterations);
            for (uint32_t i = 0; i < num_searches; i++) {
                results[i] = results32[i];
            }
            free(targets32);
            free(results32);
            return result == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
        }
#endif
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL:
            return opencl_grover_batch_search(ctx->opencl_ctx, batch_states->opencl_buffer,
                                              targets, results, num_searches,
                                              num_qubits, num_iterations)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_grover_batch_search(ctx->cuda_ctx, batch_states->cuda_buffer,
                                            targets, results, num_searches,
                                            num_qubits, num_iterations)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
        default:
            return GPU_ERROR_NOT_SUPPORTED;
    }
}

// ============================================================================
// PROBABILITY & MEASUREMENT
// ============================================================================

gpu_error_t gpu_compute_probabilities(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                                      gpu_buffer_t* probabilities, uint64_t state_dim) {
    if (!ctx || !amplitudes || !probabilities) return GPU_ERROR_INVALID_PARAM;

    switch (ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL:
            return metal_compute_probabilities(ctx->metal_ctx, amplitudes->metal_buffer,
                                               probabilities->metal_buffer,
                                               (uint32_t)state_dim) == 0 ?
                   GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL:
            return opencl_compute_probabilities(ctx->opencl_ctx, amplitudes->opencl_buffer,
                                                probabilities->opencl_buffer, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_compute_probabilities(ctx->cuda_ctx, amplitudes->cuda_buffer,
                                              probabilities->cuda_buffer, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
        default:
            return GPU_ERROR_NOT_SUPPORTED;
    }
}

gpu_error_t gpu_normalize(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                          double norm, uint64_t state_dim) {
    if (!ctx || !amplitudes) return GPU_ERROR_INVALID_PARAM;

    switch (ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL:
            return metal_normalize(ctx->metal_ctx, amplitudes->metal_buffer,
                                   (float)norm, (uint32_t)state_dim) == 0 ?
                   GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL:
            return opencl_normalize(ctx->opencl_ctx, amplitudes->opencl_buffer,
                                    (float)norm, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_normalize_state(ctx->cuda_ctx, amplitudes->cuda_buffer,
                                        norm, state_dim)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
        default:
            return GPU_ERROR_NOT_SUPPORTED;
    }
}

gpu_error_t gpu_sum_squared_magnitudes(gpu_context_t* ctx, gpu_buffer_t* amplitudes,
                                       uint64_t state_dim, double* result) {
    if (!ctx || !amplitudes || !result) return GPU_ERROR_INVALID_PARAM;

    switch (ctx->backend_type) {
// OpenCL: sum_squared_magnitudes not yet implemented
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_sum_squared_magnitudes(ctx->cuda_ctx, amplitudes->cuda_buffer,
                                               state_dim, result)
                   == 0 ? GPU_SUCCESS : GPU_ERROR_KERNEL_FAILED;
#endif
        default:
            return GPU_ERROR_NOT_SUPPORTED;
    }
}

// ============================================================================
// SYNCHRONIZATION & UTILITIES
// ============================================================================

void gpu_wait_completion(gpu_context_t* ctx) {
    if (!ctx) return;

    switch (ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL:
            metal_wait_completion(ctx->metal_ctx);
            break;
#endif
#ifdef HAS_OPENCL
        case GPU_BACKEND_OPENCL:
            opencl_wait_completion(ctx->opencl_ctx);
            break;
#endif
#ifdef HAS_VULKAN
        case GPU_BACKEND_VULKAN:
            vulkan_wait_completion(ctx->vulkan_ctx);
            break;
#endif
#ifdef HAS_CUDA
        case GPU_BACKEND_CUDA:
            cuda_synchronize(ctx->cuda_ctx);
            break;
#endif
#ifdef HAS_CUQUANTUM
        case GPU_BACKEND_CUQUANTUM:
            cuquantum_synchronize(ctx->cuquantum_ctx);
            break;
#endif
        default:
            break;
    }
}

double gpu_get_last_execution_time(gpu_context_t* ctx) {
    if (!ctx) return 0.0;

    switch (ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL:
            return metal_get_last_execution_time(ctx->metal_ctx);
#endif
        default:
            return ctx->last_exec_time;
    }
}

void gpu_set_performance_monitoring(gpu_context_t* ctx, int enable) {
    if (!ctx) return;

    ctx->perf_monitoring = enable;

    switch (ctx->backend_type) {
#ifdef HAS_METAL
        case GPU_BACKEND_METAL:
            metal_set_performance_monitoring(ctx->metal_ctx, enable);
            break;
#endif
        default:
            break;
    }
}

const char* gpu_get_error_string(gpu_context_t* ctx) {
    if (!ctx) return "NULL context";
    return ctx->last_error;
}

const char* gpu_error_name(gpu_error_t error) {
    switch (error) {
        case GPU_SUCCESS:           return "Success";
        case GPU_ERROR_NO_DEVICE:   return "No GPU device found";
        case GPU_ERROR_INIT_FAILED: return "Initialization failed";
        case GPU_ERROR_COMPILE_FAILED: return "Shader compilation failed";
        case GPU_ERROR_ALLOC_FAILED:   return "Memory allocation failed";
        case GPU_ERROR_KERNEL_FAILED:  return "Kernel execution failed";
        case GPU_ERROR_INVALID_PARAM:  return "Invalid parameter";
        case GPU_ERROR_NOT_SUPPORTED:  return "Operation not supported";
        case GPU_ERROR_TIMEOUT:        return "Operation timed out";
        default:                       return "Unknown error";
    }
}
