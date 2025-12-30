/**
 * @file gpu_opencl.c
 * @brief OpenCL GPU backend implementation
 *
 * Cross-platform GPU acceleration using OpenCL 1.2+
 *
 * @stability beta
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#ifdef HAS_OPENCL

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "gpu_opencl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ============================================================================
// INTERNAL STRUCTURES
// ============================================================================

struct opencl_compute_ctx {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    // Compiled kernels
    cl_kernel hadamard_kernel;
    cl_kernel hadamard_all_kernel;
    cl_kernel oracle_kernel;
    cl_kernel oracle_multi_kernel;
    cl_kernel diffusion_sum_kernel;
    cl_kernel diffusion_apply_kernel;
    cl_kernel diffusion_fused_kernel;
    cl_kernel probabilities_kernel;
    cl_kernel normalize_kernel;
    cl_kernel pauli_x_kernel;
    cl_kernel pauli_y_kernel;
    cl_kernel pauli_z_kernel;
    cl_kernel phase_kernel;
    cl_kernel cnot_kernel;
    cl_kernel batch_search_kernel;

    // Device info
    char device_name[256];
    cl_uint compute_units;
    size_t max_work_group_size;
    cl_ulong global_mem_size;
    cl_ulong local_mem_size;

    // Performance monitoring
    int performance_monitoring;
    double last_execution_time;

    // Error tracking
    char last_error[512];
};

struct opencl_buffer {
    cl_mem mem;
    size_t size;
};

// ============================================================================
// EMBEDDED KERNEL SOURCE
// ============================================================================

// Kernel source is embedded to avoid file path issues
static const char* QUANTUM_KERNEL_SOURCE =
#include "kernel_source_opencl.inc"
;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

static double get_time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void set_error(opencl_compute_ctx_t* ctx, const char* error) {
    if (ctx && error) {
        strncpy(ctx->last_error, error, sizeof(ctx->last_error) - 1);
        ctx->last_error[sizeof(ctx->last_error) - 1] = '\0';
    }
}

static const char* get_cl_error_string(cl_int error) {
    switch (error) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        default: return "Unknown OpenCL error";
    }
}

// ============================================================================
// KERNEL LOADING
// ============================================================================

static int load_kernel_from_file(opencl_compute_ctx_t* ctx, const char* path) {
    FILE* file = fopen(path, "r");
    if (!file) {
        return -1;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* source = (char*)malloc(size + 1);
    if (!source) {
        fclose(file);
        return -1;
    }

    size_t read = fread(source, 1, size, file);
    source[read] = '\0';
    fclose(file);

    cl_int err;
    ctx->program = clCreateProgramWithSource(ctx->context, 1,
                                             (const char**)&source, NULL, &err);
    free(source);

    return (err == CL_SUCCESS) ? 0 : -1;
}

static int compile_kernels(opencl_compute_ctx_t* ctx) {
    cl_int err;

    // Try loading from file first
    const char* kernel_paths[] = {
        "src/optimization/gpu/kernels/opencl/quantum_kernels.cl",
        "../src/optimization/gpu/kernels/opencl/quantum_kernels.cl",
        "./quantum_kernels.cl"
    };

    int loaded = 0;
    for (int i = 0; i < 3 && !loaded; i++) {
        if (load_kernel_from_file(ctx, kernel_paths[i]) == 0) {
            loaded = 1;
        }
    }

    // Fall back to embedded source
    if (!loaded) {
        ctx->program = clCreateProgramWithSource(ctx->context, 1,
                                                 &QUANTUM_KERNEL_SOURCE, NULL, &err);
        if (err != CL_SUCCESS) {
            set_error(ctx, "Failed to create program from source");
            return -1;
        }
    }

    // Build program
    err = clBuildProgram(ctx->program, 1, &ctx->device,
                         "-cl-std=CL1.2 -cl-mad-enable -cl-fast-relaxed-math",
                         NULL, NULL);

    if (err != CL_SUCCESS) {
        // Get build log
        size_t log_size;
        clGetProgramBuildInfo(ctx->program, ctx->device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);

        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(ctx->program, ctx->device, CL_PROGRAM_BUILD_LOG,
                                  log_size, log, NULL);
            fprintf(stderr, "OpenCL build error:\n%s\n", log);
            free(log);
        }

        set_error(ctx, "Failed to build OpenCL program");
        return -1;
    }

    // Create kernels
    #define CREATE_KERNEL(name, var) \
        ctx->var = clCreateKernel(ctx->program, name, &err); \
        if (err != CL_SUCCESS) { \
            fprintf(stderr, "OpenCL: Failed to create kernel %s\n", name); \
        }

    CREATE_KERNEL("hadamard_transform", hadamard_kernel);
    CREATE_KERNEL("hadamard_all_qubits", hadamard_all_kernel);
    CREATE_KERNEL("oracle_single_target", oracle_kernel);
    CREATE_KERNEL("sparse_oracle", oracle_multi_kernel);
    CREATE_KERNEL("diffusion_sum", diffusion_sum_kernel);
    CREATE_KERNEL("diffusion_apply", diffusion_apply_kernel);
    CREATE_KERNEL("grover_diffusion_fused", diffusion_fused_kernel);
    CREATE_KERNEL("compute_probabilities", probabilities_kernel);
    CREATE_KERNEL("normalize_state", normalize_kernel);
    CREATE_KERNEL("pauli_x", pauli_x_kernel);
    CREATE_KERNEL("pauli_y", pauli_y_kernel);
    CREATE_KERNEL("pauli_z", pauli_z_kernel);
    CREATE_KERNEL("phase_gate", phase_kernel);
    CREATE_KERNEL("cnot_gate", cnot_kernel);
    CREATE_KERNEL("grover_batch_search", batch_search_kernel);

    #undef CREATE_KERNEL

    return 0;
}

// ============================================================================
// INITIALIZATION
// ============================================================================

int opencl_is_available(void) {
    cl_uint num_platforms;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);

    if (err != CL_SUCCESS || num_platforms == 0) {
        return 0;
    }

    // Check for at least one GPU device
    cl_platform_id platforms[8];
    err = clGetPlatformIDs(8, platforms, &num_platforms);

    for (cl_uint i = 0; i < num_platforms; i++) {
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        if (err == CL_SUCCESS && num_devices > 0) {
            return 1;
        }
    }

    return 0;
}

opencl_compute_ctx_t* opencl_compute_init(void) {
    return opencl_compute_init_device(-1, -1);  // Auto-select
}

opencl_compute_ctx_t* opencl_compute_init_device(int platform_index, int device_index) {
    cl_int err;

    opencl_compute_ctx_t* ctx = (opencl_compute_ctx_t*)calloc(1, sizeof(opencl_compute_ctx_t));
    if (!ctx) return NULL;

    // Get platforms
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        set_error(ctx, "No OpenCL platforms found");
        free(ctx);
        return NULL;
    }

    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);

    // Select platform and device
    cl_device_id best_device = NULL;
    cl_platform_id best_platform = NULL;
    cl_device_type best_type = 0;

    for (cl_uint p = 0; p < num_platforms; p++) {
        if (platform_index >= 0 && p != (cl_uint)platform_index) continue;

        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) continue;

        cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
        clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

        for (cl_uint d = 0; d < num_devices; d++) {
            if (device_index >= 0 && d != (cl_uint)device_index) continue;

            cl_device_type type;
            clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(type), &type, NULL);

            // Prefer discrete GPU over integrated
            cl_bool integrated;
            clGetDeviceInfo(devices[d], CL_DEVICE_HOST_UNIFIED_MEMORY,
                           sizeof(integrated), &integrated, NULL);

            if (!best_device || (!integrated && best_type < type)) {
                best_device = devices[d];
                best_platform = platforms[p];
                best_type = type;
            }
        }

        free(devices);
    }

    free(platforms);

    if (!best_device) {
        set_error(ctx, "No suitable OpenCL device found");
        free(ctx);
        return NULL;
    }

    ctx->platform = best_platform;
    ctx->device = best_device;

    // Get device info
    clGetDeviceInfo(ctx->device, CL_DEVICE_NAME, sizeof(ctx->device_name),
                   ctx->device_name, NULL);
    clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                   sizeof(ctx->compute_units), &ctx->compute_units, NULL);
    clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                   sizeof(ctx->max_work_group_size), &ctx->max_work_group_size, NULL);
    clGetDeviceInfo(ctx->device, CL_DEVICE_GLOBAL_MEM_SIZE,
                   sizeof(ctx->global_mem_size), &ctx->global_mem_size, NULL);
    clGetDeviceInfo(ctx->device, CL_DEVICE_LOCAL_MEM_SIZE,
                   sizeof(ctx->local_mem_size), &ctx->local_mem_size, NULL);

    printf("OpenCL: Initialized device: %s\n", ctx->device_name);
    printf("OpenCL: Compute units: %u, Max work-group: %zu\n",
           ctx->compute_units, ctx->max_work_group_size);
    printf("OpenCL: Global memory: %llu MB\n",
           (unsigned long long)(ctx->global_mem_size / (1024 * 1024)));

    // Create context
    ctx->context = clCreateContext(NULL, 1, &ctx->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        set_error(ctx, "Failed to create OpenCL context");
        free(ctx);
        return NULL;
    }

    // Create command queue
#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, 0, 0};
    ctx->queue = clCreateCommandQueueWithProperties(ctx->context, ctx->device, props, &err);
#else
    ctx->queue = clCreateCommandQueue(ctx->context, ctx->device, 0, &err);
#endif

    if (err != CL_SUCCESS) {
        set_error(ctx, "Failed to create command queue");
        clReleaseContext(ctx->context);
        free(ctx);
        return NULL;
    }

    // Compile kernels
    if (compile_kernels(ctx) != 0) {
        clReleaseCommandQueue(ctx->queue);
        clReleaseContext(ctx->context);
        free(ctx);
        return NULL;
    }

    printf("OpenCL: Kernels compiled successfully\n");
    return ctx;
}

void opencl_compute_free(opencl_compute_ctx_t* ctx) {
    if (!ctx) return;

    // Release kernels
    if (ctx->hadamard_kernel) clReleaseKernel(ctx->hadamard_kernel);
    if (ctx->hadamard_all_kernel) clReleaseKernel(ctx->hadamard_all_kernel);
    if (ctx->oracle_kernel) clReleaseKernel(ctx->oracle_kernel);
    if (ctx->oracle_multi_kernel) clReleaseKernel(ctx->oracle_multi_kernel);
    if (ctx->diffusion_sum_kernel) clReleaseKernel(ctx->diffusion_sum_kernel);
    if (ctx->diffusion_apply_kernel) clReleaseKernel(ctx->diffusion_apply_kernel);
    if (ctx->diffusion_fused_kernel) clReleaseKernel(ctx->diffusion_fused_kernel);
    if (ctx->probabilities_kernel) clReleaseKernel(ctx->probabilities_kernel);
    if (ctx->normalize_kernel) clReleaseKernel(ctx->normalize_kernel);
    if (ctx->pauli_x_kernel) clReleaseKernel(ctx->pauli_x_kernel);
    if (ctx->pauli_y_kernel) clReleaseKernel(ctx->pauli_y_kernel);
    if (ctx->pauli_z_kernel) clReleaseKernel(ctx->pauli_z_kernel);
    if (ctx->phase_kernel) clReleaseKernel(ctx->phase_kernel);
    if (ctx->cnot_kernel) clReleaseKernel(ctx->cnot_kernel);
    if (ctx->batch_search_kernel) clReleaseKernel(ctx->batch_search_kernel);

    if (ctx->program) clReleaseProgram(ctx->program);
    if (ctx->queue) clReleaseCommandQueue(ctx->queue);
    if (ctx->context) clReleaseContext(ctx->context);

    free(ctx);
}

void opencl_get_device_info(
    opencl_compute_ctx_t* ctx,
    char* name,
    uint32_t* max_work_group_size,
    uint32_t* compute_units
) {
    if (!ctx) return;

    if (name) {
        strncpy(name, ctx->device_name, 255);
        name[255] = '\0';
    }

    if (max_work_group_size) {
        *max_work_group_size = (uint32_t)ctx->max_work_group_size;
    }

    if (compute_units) {
        *compute_units = ctx->compute_units;
    }
}

void opencl_print_device_info(opencl_compute_ctx_t* ctx) {
    if (!ctx) return;

    printf("\n");
    printf("====================================================\n");
    printf("     OPENCL GPU DEVICE INFORMATION\n");
    printf("====================================================\n");
    printf("  Device: %s\n", ctx->device_name);
    printf("  Compute Units: %u\n", ctx->compute_units);
    printf("  Max Work-Group Size: %zu\n", ctx->max_work_group_size);
    printf("  Global Memory: %llu MB\n",
           (unsigned long long)(ctx->global_mem_size / (1024 * 1024)));
    printf("  Local Memory: %llu KB\n",
           (unsigned long long)(ctx->local_mem_size / 1024));
    printf("====================================================\n\n");
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

opencl_buffer_t* opencl_buffer_create(opencl_compute_ctx_t* ctx, size_t size) {
    if (!ctx || size == 0) return NULL;

    opencl_buffer_t* buffer = (opencl_buffer_t*)malloc(sizeof(opencl_buffer_t));
    if (!buffer) return NULL;

    cl_int err;
    buffer->mem = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE, size, NULL, &err);

    if (err != CL_SUCCESS) {
        free(buffer);
        return NULL;
    }

    buffer->size = size;
    return buffer;
}

opencl_buffer_t* opencl_buffer_create_from_data(
    opencl_compute_ctx_t* ctx,
    const void* data,
    size_t size
) {
    if (!ctx || !data || size == 0) return NULL;

    opencl_buffer_t* buffer = (opencl_buffer_t*)malloc(sizeof(opencl_buffer_t));
    if (!buffer) return NULL;

    cl_int err;
    buffer->mem = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 size, (void*)data, &err);

    if (err != CL_SUCCESS) {
        free(buffer);
        return NULL;
    }

    buffer->size = size;
    return buffer;
}

int opencl_buffer_read(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* buffer,
    void* dst,
    size_t size
) {
    if (!ctx || !buffer || !dst) return -1;

    cl_int err = clEnqueueReadBuffer(ctx->queue, buffer->mem, CL_TRUE,
                                     0, size, dst, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? 0 : -1;
}

int opencl_buffer_write(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* buffer,
    const void* src,
    size_t size
) {
    if (!ctx || !buffer || !src) return -1;

    cl_int err = clEnqueueWriteBuffer(ctx->queue, buffer->mem, CL_TRUE,
                                      0, size, src, 0, NULL, NULL);
    return (err == CL_SUCCESS) ? 0 : -1;
}

void opencl_buffer_free(opencl_buffer_t* buffer) {
    if (!buffer) return;

    if (buffer->mem) {
        clReleaseMemObject(buffer->mem);
    }
    free(buffer);
}

// ============================================================================
// KERNEL DISPATCH HELPER
// ============================================================================

static int dispatch_kernel_1d(
    opencl_compute_ctx_t* ctx,
    cl_kernel kernel,
    size_t global_size,
    size_t local_size
) {
    if (!ctx || !kernel) return -1;

    double start_time = ctx->performance_monitoring ? get_time_seconds() : 0.0;

    // Ensure local_size divides global_size
    if (local_size > 0 && global_size % local_size != 0) {
        global_size = ((global_size / local_size) + 1) * local_size;
    }

    cl_int err;
    if (local_size > 0) {
        err = clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL,
                                    &global_size, &local_size, 0, NULL, NULL);
    } else {
        err = clEnqueueNDRangeKernel(ctx->queue, kernel, 1, NULL,
                                    &global_size, NULL, 0, NULL, NULL);
    }

    if (err != CL_SUCCESS) {
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg), "Kernel execution failed: %s",
                 get_cl_error_string(err));
        set_error(ctx, error_msg);
        return -1;
    }

    clFinish(ctx->queue);

    if (ctx->performance_monitoring) {
        ctx->last_execution_time = get_time_seconds() - start_time;
    }

    return 0;
}

// ============================================================================
// QUANTUM GATE OPERATIONS
// ============================================================================

int opencl_hadamard(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
) {
    if (!ctx || !amplitudes || !ctx->hadamard_kernel) return -1;

    cl_uint q = qubit_index;
    cl_uint dim = (cl_uint)state_dim;

    clSetKernelArg(ctx->hadamard_kernel, 0, sizeof(cl_mem), &amplitudes->mem);
    clSetKernelArg(ctx->hadamard_kernel, 1, sizeof(cl_uint), &q);
    clSetKernelArg(ctx->hadamard_kernel, 2, sizeof(cl_uint), &dim);

    size_t global_size = state_dim / 2;
    size_t local_size = (ctx->max_work_group_size < 256) ? ctx->max_work_group_size : 256;

    return dispatch_kernel_1d(ctx, ctx->hadamard_kernel, global_size, local_size);
}

int opencl_hadamard_all(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint32_t num_qubits,
    uint64_t state_dim
) {
    if (!ctx || !amplitudes || !ctx->hadamard_all_kernel) return -1;

    cl_uint n = num_qubits;
    cl_uint dim = (cl_uint)state_dim;

    clSetKernelArg(ctx->hadamard_all_kernel, 0, sizeof(cl_mem), &amplitudes->mem);
    clSetKernelArg(ctx->hadamard_all_kernel, 1, sizeof(cl_uint), &n);
    clSetKernelArg(ctx->hadamard_all_kernel, 2, sizeof(cl_uint), &dim);

    size_t global_size = state_dim;
    size_t local_size = (ctx->max_work_group_size < 256) ? ctx->max_work_group_size : 256;

    return dispatch_kernel_1d(ctx, ctx->hadamard_all_kernel, global_size, local_size);
}

int opencl_pauli_x(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
) {
    if (!ctx || !amplitudes || !ctx->pauli_x_kernel) return -1;

    cl_uint q = qubit_index;
    cl_uint dim = (cl_uint)state_dim;

    clSetKernelArg(ctx->pauli_x_kernel, 0, sizeof(cl_mem), &amplitudes->mem);
    clSetKernelArg(ctx->pauli_x_kernel, 1, sizeof(cl_uint), &q);
    clSetKernelArg(ctx->pauli_x_kernel, 2, sizeof(cl_uint), &dim);

    size_t global_size = state_dim / 2;
    size_t local_size = (ctx->max_work_group_size < 256) ? ctx->max_work_group_size : 256;

    return dispatch_kernel_1d(ctx, ctx->pauli_x_kernel, global_size, local_size);
}

int opencl_pauli_y(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
) {
    if (!ctx || !amplitudes || !ctx->pauli_y_kernel) return -1;

    cl_uint q = qubit_index;
    cl_uint dim = (cl_uint)state_dim;

    clSetKernelArg(ctx->pauli_y_kernel, 0, sizeof(cl_mem), &amplitudes->mem);
    clSetKernelArg(ctx->pauli_y_kernel, 1, sizeof(cl_uint), &q);
    clSetKernelArg(ctx->pauli_y_kernel, 2, sizeof(cl_uint), &dim);

    size_t global_size = state_dim / 2;
    size_t local_size = (ctx->max_work_group_size < 256) ? ctx->max_work_group_size : 256;

    return dispatch_kernel_1d(ctx, ctx->pauli_y_kernel, global_size, local_size);
}

int opencl_pauli_z(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint64_t state_dim
) {
    if (!ctx || !amplitudes || !ctx->pauli_z_kernel) return -1;

    cl_uint q = qubit_index;
    cl_uint dim = (cl_uint)state_dim;

    clSetKernelArg(ctx->pauli_z_kernel, 0, sizeof(cl_mem), &amplitudes->mem);
    clSetKernelArg(ctx->pauli_z_kernel, 1, sizeof(cl_uint), &q);
    clSetKernelArg(ctx->pauli_z_kernel, 2, sizeof(cl_uint), &dim);

    size_t global_size = state_dim;
    size_t local_size = (ctx->max_work_group_size < 256) ? ctx->max_work_group_size : 256;

    return dispatch_kernel_1d(ctx, ctx->pauli_z_kernel, global_size, local_size);
}

int opencl_phase(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint32_t qubit_index,
    float phase,
    uint64_t state_dim
) {
    if (!ctx || !amplitudes || !ctx->phase_kernel) return -1;

    cl_uint q = qubit_index;
    cl_float p = phase;
    cl_uint dim = (cl_uint)state_dim;

    clSetKernelArg(ctx->phase_kernel, 0, sizeof(cl_mem), &amplitudes->mem);
    clSetKernelArg(ctx->phase_kernel, 1, sizeof(cl_uint), &q);
    clSetKernelArg(ctx->phase_kernel, 2, sizeof(cl_float), &p);
    clSetKernelArg(ctx->phase_kernel, 3, sizeof(cl_uint), &dim);

    size_t global_size = state_dim;
    size_t local_size = (ctx->max_work_group_size < 256) ? ctx->max_work_group_size : 256;

    return dispatch_kernel_1d(ctx, ctx->phase_kernel, global_size, local_size);
}

int opencl_cnot(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint32_t control_qubit,
    uint32_t target_qubit,
    uint64_t state_dim
) {
    if (!ctx || !amplitudes || !ctx->cnot_kernel) return -1;

    cl_uint ctrl = control_qubit;
    cl_uint tgt = target_qubit;
    cl_uint dim = (cl_uint)state_dim;

    clSetKernelArg(ctx->cnot_kernel, 0, sizeof(cl_mem), &amplitudes->mem);
    clSetKernelArg(ctx->cnot_kernel, 1, sizeof(cl_uint), &ctrl);
    clSetKernelArg(ctx->cnot_kernel, 2, sizeof(cl_uint), &tgt);
    clSetKernelArg(ctx->cnot_kernel, 3, sizeof(cl_uint), &dim);

    size_t global_size = state_dim;
    size_t local_size = (ctx->max_work_group_size < 256) ? ctx->max_work_group_size : 256;

    return dispatch_kernel_1d(ctx, ctx->cnot_kernel, global_size, local_size);
}

// ============================================================================
// GROVER'S ALGORITHM OPERATIONS
// ============================================================================

int opencl_oracle(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint64_t target,
    uint64_t state_dim
) {
    if (!ctx || !amplitudes || !ctx->oracle_kernel) return -1;

    cl_uint tgt = (cl_uint)target;

    clSetKernelArg(ctx->oracle_kernel, 0, sizeof(cl_mem), &amplitudes->mem);
    clSetKernelArg(ctx->oracle_kernel, 1, sizeof(cl_uint), &tgt);

    size_t global_size = state_dim;
    size_t local_size = (ctx->max_work_group_size < 256) ? ctx->max_work_group_size : 256;

    return dispatch_kernel_1d(ctx, ctx->oracle_kernel, global_size, local_size);
}

int opencl_oracle_multi(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    const uint64_t* targets,
    uint32_t num_targets,
    uint64_t state_dim
) {
    if (!ctx || !amplitudes || !targets || !ctx->oracle_multi_kernel) return -1;

    // Convert targets to uint32
    uint32_t* targets32 = (uint32_t*)malloc(num_targets * sizeof(uint32_t));
    if (!targets32) return -1;

    for (uint32_t i = 0; i < num_targets; i++) {
        targets32[i] = (uint32_t)targets[i];
    }

    opencl_buffer_t* targets_buf = opencl_buffer_create_from_data(
        ctx, targets32, num_targets * sizeof(uint32_t));
    free(targets32);

    if (!targets_buf) return -1;

    cl_uint n = num_targets;
    cl_uint dim = (cl_uint)state_dim;

    clSetKernelArg(ctx->oracle_multi_kernel, 0, sizeof(cl_mem), &amplitudes->mem);
    clSetKernelArg(ctx->oracle_multi_kernel, 1, sizeof(cl_mem), &targets_buf->mem);
    clSetKernelArg(ctx->oracle_multi_kernel, 2, sizeof(cl_uint), &n);
    clSetKernelArg(ctx->oracle_multi_kernel, 3, sizeof(cl_uint), &dim);

    size_t global_size = state_dim;
    size_t local_size = (ctx->max_work_group_size < 256) ? ctx->max_work_group_size : 256;

    int result = dispatch_kernel_1d(ctx, ctx->oracle_multi_kernel, global_size, local_size);

    opencl_buffer_free(targets_buf);
    return result;
}

int opencl_grover_diffusion(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint32_t num_qubits,
    uint64_t state_dim
) {
    if (!ctx || !amplitudes) return -1;

    // Use fused kernel for small states
    if (state_dim <= 1024 && ctx->diffusion_fused_kernel) {
        cl_uint n = num_qubits;
        cl_uint dim = (cl_uint)state_dim;

        clSetKernelArg(ctx->diffusion_fused_kernel, 0, sizeof(cl_mem), &amplitudes->mem);
        clSetKernelArg(ctx->diffusion_fused_kernel, 1, state_dim * sizeof(cl_float2), NULL);
        clSetKernelArg(ctx->diffusion_fused_kernel, 2, sizeof(cl_uint), &n);
        clSetKernelArg(ctx->diffusion_fused_kernel, 3, sizeof(cl_uint), &dim);

        size_t global_size = state_dim;
        size_t local_size = state_dim;

        return dispatch_kernel_1d(ctx, ctx->diffusion_fused_kernel, global_size, local_size);
    }

    // Two-phase diffusion for larger states
    if (!ctx->diffusion_sum_kernel || !ctx->diffusion_apply_kernel) return -1;

    // Phase 1: Compute sum
    size_t local_size = 256;
    size_t num_groups = (state_dim + local_size - 1) / local_size;
    if (num_groups > 256) num_groups = 256;
    size_t global_size = num_groups * local_size;

    opencl_buffer_t* partial_sums = opencl_buffer_create(ctx, num_groups * sizeof(cl_float2));
    opencl_buffer_t* avg_buffer = opencl_buffer_create(ctx, sizeof(cl_float2));

    if (!partial_sums || !avg_buffer) {
        if (partial_sums) opencl_buffer_free(partial_sums);
        if (avg_buffer) opencl_buffer_free(avg_buffer);
        return -1;
    }

    cl_uint dim = (cl_uint)state_dim;

    clSetKernelArg(ctx->diffusion_sum_kernel, 0, sizeof(cl_mem), &amplitudes->mem);
    clSetKernelArg(ctx->diffusion_sum_kernel, 1, sizeof(cl_mem), &partial_sums->mem);
    clSetKernelArg(ctx->diffusion_sum_kernel, 2, local_size * sizeof(cl_float2), NULL);
    clSetKernelArg(ctx->diffusion_sum_kernel, 3, sizeof(cl_uint), &dim);

    int result = dispatch_kernel_1d(ctx, ctx->diffusion_sum_kernel, global_size, local_size);
    if (result != 0) {
        opencl_buffer_free(partial_sums);
        opencl_buffer_free(avg_buffer);
        return result;
    }

    // Sum partial sums on CPU (small amount)
    cl_float2* sums = (cl_float2*)malloc(num_groups * sizeof(cl_float2));
    opencl_buffer_read(ctx, partial_sums, sums, num_groups * sizeof(cl_float2));

    cl_float2 total = {0.0f, 0.0f};
    for (size_t i = 0; i < num_groups; i++) {
        total.x += sums[i].x;
        total.y += sums[i].y;
    }
    free(sums);

    // Compute average
    cl_float2 avg = {total.x / state_dim, total.y / state_dim};
    opencl_buffer_write(ctx, avg_buffer, &avg, sizeof(cl_float2));

    // Phase 2: Apply diffusion
    clSetKernelArg(ctx->diffusion_apply_kernel, 0, sizeof(cl_mem), &amplitudes->mem);
    clSetKernelArg(ctx->diffusion_apply_kernel, 1, sizeof(cl_mem), &avg_buffer->mem);
    clSetKernelArg(ctx->diffusion_apply_kernel, 2, sizeof(cl_uint), &dim);

    result = dispatch_kernel_1d(ctx, ctx->diffusion_apply_kernel, state_dim, 256);

    opencl_buffer_free(partial_sums);
    opencl_buffer_free(avg_buffer);

    return result;
}

int opencl_grover_search(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    uint64_t target,
    uint32_t num_qubits,
    uint32_t num_iterations
) {
    if (!ctx || !amplitudes) return -1;

    uint64_t state_dim = 1ULL << num_qubits;

    // Initialize: Hadamard on all qubits
    if (opencl_hadamard_all(ctx, amplitudes, num_qubits, state_dim) != 0) {
        return -1;
    }

    // Run Grover iterations
    for (uint32_t i = 0; i < num_iterations; i++) {
        // Oracle
        if (opencl_oracle(ctx, amplitudes, target, state_dim) != 0) {
            return -1;
        }

        // Diffusion
        if (opencl_grover_diffusion(ctx, amplitudes, num_qubits, state_dim) != 0) {
            return -1;
        }
    }

    return 0;
}

int opencl_grover_batch_search(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* batch_states,
    const uint64_t* targets,
    uint64_t* results,
    uint32_t num_searches,
    uint32_t num_qubits,
    uint32_t num_iterations
) {
    if (!ctx || !batch_states || !targets || !results) return -1;
    if (!ctx->batch_search_kernel) return -1;

    // Convert targets to uint32
    uint32_t* targets32 = (uint32_t*)malloc(num_searches * sizeof(uint32_t));
    uint32_t* results32 = (uint32_t*)malloc(num_searches * sizeof(uint32_t));
    if (!targets32 || !results32) {
        free(targets32);
        free(results32);
        return -1;
    }

    for (uint32_t i = 0; i < num_searches; i++) {
        targets32[i] = (uint32_t)targets[i];
    }

    opencl_buffer_t* targets_buf = opencl_buffer_create_from_data(
        ctx, targets32, num_searches * sizeof(uint32_t));
    opencl_buffer_t* results_buf = opencl_buffer_create(
        ctx, num_searches * sizeof(uint32_t));

    free(targets32);

    if (!targets_buf || !results_buf) {
        if (targets_buf) opencl_buffer_free(targets_buf);
        if (results_buf) opencl_buffer_free(results_buf);
        free(results32);
        return -1;
    }

    cl_uint n_searches = num_searches;
    cl_uint n_qubits = num_qubits;
    cl_uint n_iters = num_iterations;

    clSetKernelArg(ctx->batch_search_kernel, 0, sizeof(cl_mem), &batch_states->mem);
    clSetKernelArg(ctx->batch_search_kernel, 1, sizeof(cl_mem), &targets_buf->mem);
    clSetKernelArg(ctx->batch_search_kernel, 2, sizeof(cl_mem), &results_buf->mem);
    clSetKernelArg(ctx->batch_search_kernel, 3, sizeof(cl_uint), &n_searches);
    clSetKernelArg(ctx->batch_search_kernel, 4, sizeof(cl_uint), &n_qubits);
    clSetKernelArg(ctx->batch_search_kernel, 5, sizeof(cl_uint), &n_iters);

    // One work-group per search
    size_t global_size = num_searches * 256;
    size_t local_size = 256;

    int result = dispatch_kernel_1d(ctx, ctx->batch_search_kernel, global_size, local_size);

    if (result == 0) {
        opencl_buffer_read(ctx, results_buf, results32, num_searches * sizeof(uint32_t));
        for (uint32_t i = 0; i < num_searches; i++) {
            results[i] = results32[i];
        }
    }

    opencl_buffer_free(targets_buf);
    opencl_buffer_free(results_buf);
    free(results32);

    return result;
}

// ============================================================================
// MEASUREMENT & NORMALIZATION
// ============================================================================

int opencl_compute_probabilities(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    opencl_buffer_t* probabilities,
    uint64_t state_dim
) {
    if (!ctx || !amplitudes || !probabilities || !ctx->probabilities_kernel) return -1;

    cl_uint dim = (cl_uint)state_dim;

    clSetKernelArg(ctx->probabilities_kernel, 0, sizeof(cl_mem), &amplitudes->mem);
    clSetKernelArg(ctx->probabilities_kernel, 1, sizeof(cl_mem), &probabilities->mem);
    clSetKernelArg(ctx->probabilities_kernel, 2, sizeof(cl_uint), &dim);

    size_t global_size = state_dim;
    size_t local_size = (ctx->max_work_group_size < 256) ? ctx->max_work_group_size : 256;

    return dispatch_kernel_1d(ctx, ctx->probabilities_kernel, global_size, local_size);
}

int opencl_normalize(
    opencl_compute_ctx_t* ctx,
    opencl_buffer_t* amplitudes,
    float norm,
    uint64_t state_dim
) {
    if (!ctx || !amplitudes || !ctx->normalize_kernel) return -1;

    cl_float n = norm;
    cl_uint dim = (cl_uint)state_dim;

    clSetKernelArg(ctx->normalize_kernel, 0, sizeof(cl_mem), &amplitudes->mem);
    clSetKernelArg(ctx->normalize_kernel, 1, sizeof(cl_float), &n);
    clSetKernelArg(ctx->normalize_kernel, 2, sizeof(cl_uint), &dim);

    size_t global_size = state_dim;
    size_t local_size = (ctx->max_work_group_size < 256) ? ctx->max_work_group_size : 256;

    return dispatch_kernel_1d(ctx, ctx->normalize_kernel, global_size, local_size);
}

// ============================================================================
// SYNCHRONIZATION & PERFORMANCE
// ============================================================================

void opencl_wait_completion(opencl_compute_ctx_t* ctx) {
    if (!ctx || !ctx->queue) return;
    clFinish(ctx->queue);
}

double opencl_get_last_execution_time(opencl_compute_ctx_t* ctx) {
    return ctx ? ctx->last_execution_time : 0.0;
}

void opencl_set_performance_monitoring(opencl_compute_ctx_t* ctx, int enable) {
    if (ctx) {
        ctx->performance_monitoring = enable;
    }
}

const char* opencl_get_error(opencl_compute_ctx_t* ctx) {
    if (!ctx || ctx->last_error[0] == '\0') return "No error";
    return ctx->last_error;
}

#endif /* HAS_OPENCL */
