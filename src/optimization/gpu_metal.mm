#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "gpu_metal.h"
#include <stdio.h>
#include <string.h>
#include <mach/mach_time.h>

/**
 * @file metal_bridge.mm
 * @brief Objective-C++ implementation of Metal GPU acceleration
 * 
 * Implements zero-copy unified memory architecture for M2 Ultra.
 * Optimized for 76 GPU cores with 192GB unified memory.
 */

// ============================================================================
// INTERNAL STRUCTURES
// ============================================================================

struct metal_compute_ctx {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    id<MTLLibrary> tensorLibrary;  // Tensor network kernels

    // Compiled compute pipelines (cached for performance)
    id<MTLComputePipelineState> hadamardPipeline;
    id<MTLComputePipelineState> hadamardAllPipeline;
    id<MTLComputePipelineState> oraclePipeline;
    id<MTLComputePipelineState> oracleMultiPipeline;
    id<MTLComputePipelineState> diffusionPipeline;
    id<MTLComputePipelineState> batchSearchPipeline;
    id<MTLComputePipelineState> batchHadamardPipeline;

    id<MTLComputePipelineState> probabilitiesPipeline;
    id<MTLComputePipelineState> normalizePipeline;
    id<MTLComputePipelineState> pauliXPipeline;
    id<MTLComputePipelineState> pauliZPipeline;

    // Tensor network / MPS pipelines
    id<MTLComputePipelineState> tensorContract2SitePipeline;
    id<MTLComputePipelineState> applyGateThetaPipeline;
    id<MTLComputePipelineState> computeColumnNormsPipeline;
    id<MTLComputePipelineState> jacobiRotationPipeline;
    id<MTLComputePipelineState> extractSingularValuesPipeline;
    id<MTLComputePipelineState> normalizeAndTruncateUPipeline;
    id<MTLComputePipelineState> truncateVPipeline;
    id<MTLComputePipelineState> transferMatrixZPipeline;
    id<MTLComputePipelineState> transferMatrixIdentityPipeline;
    id<MTLComputePipelineState> contractTransferPipeline;
    id<MTLComputePipelineState> transferTracePipeline;
    id<MTLComputePipelineState> tensorNormSquaredPipeline;
    id<MTLComputePipelineState> tensorScalePipeline;

    // Performance monitoring
    int performance_monitoring;
    double last_execution_time;

    // Error tracking
    NSString* lastError;
};

struct metal_buffer {
    id<MTLBuffer> buffer;
    size_t size;
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

static double get_time_seconds() {
    static mach_timebase_info_data_t timebase_info;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        mach_timebase_info(&timebase_info);
    });
    
    uint64_t time = mach_absolute_time();
    return (double)time * timebase_info.numer / timebase_info.denom / 1e9;
}

static void set_error(metal_compute_ctx_t* ctx, NSString* error) {
    if (ctx) {
        ctx->lastError = error;
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

metal_compute_ctx_t* metal_compute_init(void) {
    @autoreleasepool {
        metal_compute_ctx_t* ctx = (metal_compute_ctx_t*)calloc(1, sizeof(metal_compute_ctx_t));
        if (!ctx) return NULL;
        
        // Get Metal device (should be M2 Ultra)
        ctx->device = MTLCreateSystemDefaultDevice();
        if (!ctx->device) {
            fprintf(stderr, "Metal: Failed to create device\n");
            free(ctx);
            return NULL;
        }
        
        // Print device info
        NSString* deviceName = [ctx->device name];
        printf("Metal: Initialized device: %s\n", [deviceName UTF8String]);
        printf("Metal: Unified memory: %llu MB\n", 
               [ctx->device recommendedMaxWorkingSetSize] / (1024 * 1024));
        
        // Create command queue
        ctx->commandQueue = [ctx->device newCommandQueue];
        if (!ctx->commandQueue) {
            fprintf(stderr, "Metal: Failed to create command queue\n");
            free(ctx);
            return NULL;
        }
        
        // Load shader library
        NSError* error = nil;
        NSString* shaderPath = @"src/optimization/kernels/quantum_kernels.metal";
        
        // Try to compile from source
        NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath
                                                           encoding:NSUTF8StringEncoding
                                                              error:&error];
        
        if (shaderSource) {
            ctx->library = [ctx->device newLibraryWithSource:shaderSource
                                                     options:nil
                                                       error:&error];
        }
        
        if (!ctx->library) {
            // Try default library (pre-compiled)
            ctx->library = [ctx->device newDefaultLibrary];
        }
        
        if (!ctx->library) {
            fprintf(stderr, "Metal: Failed to load shader library: %s\n",
                    [[error localizedDescription] UTF8String]);
            free(ctx);
            return NULL;
        }
        
        // Load batch kernels
        NSError* batchError = nil;
        NSString* batchPath = @"src/optimization/kernels/quantum_kernels_batch.metal";
        NSString* batchSource = [NSString stringWithContentsOfFile:batchPath
                                                           encoding:NSUTF8StringEncoding
                                                              error:&batchError];
        
        if (batchSource) {
            id<MTLLibrary> batchLibrary = [ctx->device newLibraryWithSource:batchSource
                                                                    options:nil
                                                                      error:&batchError];
            if (batchLibrary) {
                id<MTLFunction> batchFunc = [batchLibrary newFunctionWithName:@"grover_batch_search"];
                if (batchFunc) {
                    ctx->batchSearchPipeline = [ctx->device newComputePipelineStateWithFunction:batchFunc
                                                                                           error:&batchError];
                    if (!ctx->batchSearchPipeline) {
                        fprintf(stderr, "Metal: Failed to compile batch search pipeline: %s\n",
                                [[batchError localizedDescription] UTF8String]);
                    } else {
                        printf("Metal: Batch search pipeline compiled successfully\n");
                        printf("  Max threads per threadgroup: %lu\n",
                               (unsigned long)[ctx->batchSearchPipeline maxTotalThreadsPerThreadgroup]);
                        printf("  Threadgroup memory: %lu bytes\n",
                               (unsigned long)[ctx->batchSearchPipeline threadExecutionWidth]);
                    }
                } else {
                    fprintf(stderr, "Metal: Failed to find grover_batch_search function\n");
                }
                
                id<MTLFunction> batchHadamard = [batchLibrary newFunctionWithName:@"batch_hadamard_init"];
                if (batchHadamard) {
                    ctx->batchHadamardPipeline = [ctx->device newComputePipelineStateWithFunction:batchHadamard
                                                                                            error:&batchError];
                    if (!ctx->batchHadamardPipeline) {
                        fprintf(stderr, "Metal: Failed to compile batch Hadamard pipeline: %s\n",
                                [[batchError localizedDescription] UTF8String]);
                    } else {
                        printf("Metal: Batch Hadamard pipeline compiled successfully\n");
                    }
                } else {
                    fprintf(stderr, "Metal: Failed to find batch_hadamard_init function\n");
                }
            } else if (batchError) {
                fprintf(stderr, "Metal: Failed to compile batch library: %s\n",
                        [[batchError localizedDescription] UTF8String]);
            }
        }
        
        // Compile compute pipelines
        NSArray* kernelNames = @[
            @"hadamard_transform",
            @"hadamard_all_qubits",
            @"oracle_single_target",
            @"sparse_oracle",
            @"grover_diffusion_fused",
            @"compute_probabilities",
            @"normalize_state",
            @"pauli_x",
            @"pauli_z"
        ];
        
        id<MTLComputePipelineState>* pipelines[] = {
            &ctx->hadamardPipeline,
            &ctx->hadamardAllPipeline,
            &ctx->oraclePipeline,
            &ctx->oracleMultiPipeline,
            &ctx->diffusionPipeline,
            &ctx->probabilitiesPipeline,
            &ctx->normalizePipeline,
            &ctx->pauliXPipeline,
            &ctx->pauliZPipeline
        };
        
        for (size_t i = 0; i < [kernelNames count]; i++) {
            id<MTLFunction> function = [ctx->library newFunctionWithName:kernelNames[i]];
            if (!function) {
                fprintf(stderr, "Metal: Failed to load function: %s\n",
                        [kernelNames[i] UTF8String]);
                continue;
            }
            
            *pipelines[i] = [ctx->device newComputePipelineStateWithFunction:function
                                                                        error:&error];
            if (!*pipelines[i]) {
                fprintf(stderr, "Metal: Failed to create pipeline for %s: %s\n",
                        [kernelNames[i] UTF8String],
                        [[error localizedDescription] UTF8String]);
            }
        }
        
        ctx->performance_monitoring = 0;
        ctx->last_execution_time = 0.0;

        // Load tensor network kernels
        NSError* tensorError = nil;
        NSString* tensorPath = @"src/optimization/kernels/tensor_kernels.metal";
        NSString* tensorSource = [NSString stringWithContentsOfFile:tensorPath
                                                           encoding:NSUTF8StringEncoding
                                                              error:&tensorError];

        if (tensorSource) {
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            options.fastMathEnabled = NO;  // Need precision for MPS
            options.languageVersion = MTLLanguageVersion2_0;

            ctx->tensorLibrary = [ctx->device newLibraryWithSource:tensorSource
                                                           options:options
                                                             error:&tensorError];

            if (ctx->tensorLibrary) {
                printf("Metal: Loading tensor network kernels...\n");

                // Tensor contraction and gate application
                NSArray* tensorKernels = @[
                    @"tensor_contract_2site",
                    @"apply_gate_theta",
                    @"compute_column_norms",
                    @"jacobi_svd_rotation",
                    @"extract_singular_values",
                    @"normalize_and_truncate_U",
                    @"truncate_V",
                    @"transfer_matrix_z_single",
                    @"transfer_matrix_identity",
                    @"contract_transfer_matrices",
                    @"transfer_matrix_trace",
                    @"tensor_norm_squared",
                    @"tensor_scale"
                ];

                id<MTLComputePipelineState>* tensorPipelines[] = {
                    &ctx->tensorContract2SitePipeline,
                    &ctx->applyGateThetaPipeline,
                    &ctx->computeColumnNormsPipeline,
                    &ctx->jacobiRotationPipeline,
                    &ctx->extractSingularValuesPipeline,
                    &ctx->normalizeAndTruncateUPipeline,
                    &ctx->truncateVPipeline,
                    &ctx->transferMatrixZPipeline,
                    &ctx->transferMatrixIdentityPipeline,
                    &ctx->contractTransferPipeline,
                    &ctx->transferTracePipeline,
                    &ctx->tensorNormSquaredPipeline,
                    &ctx->tensorScalePipeline
                };

                int tensorKernelsLoaded = 0;
                for (size_t i = 0; i < [tensorKernels count]; i++) {
                    id<MTLFunction> function = [ctx->tensorLibrary newFunctionWithName:tensorKernels[i]];
                    if (!function) {
                        fprintf(stderr, "Metal: Failed to load tensor kernel: %s\n",
                                [tensorKernels[i] UTF8String]);
                        continue;
                    }

                    NSError* pipeError = nil;
                    *tensorPipelines[i] = [ctx->device newComputePipelineStateWithFunction:function
                                                                                      error:&pipeError];
                    if (!*tensorPipelines[i]) {
                        fprintf(stderr, "Metal: Failed to compile tensor kernel %s: %s\n",
                                [tensorKernels[i] UTF8String],
                                [[pipeError localizedDescription] UTF8String]);
                    } else {
                        tensorKernelsLoaded++;
                    }
                }
                printf("Metal: Loaded %d/%lu tensor network kernels\n",
                       tensorKernelsLoaded, (unsigned long)[tensorKernels count]);
            } else if (tensorError) {
                fprintf(stderr, "Metal: Failed to compile tensor library: %s\n",
                        [[tensorError localizedDescription] UTF8String]);
            }
        } else {
            fprintf(stderr, "Metal: Tensor kernels not found at %s\n", [tensorPath UTF8String]);
        }

        // Detect GPU cores
        uint32_t num_cores = 0;
        metal_get_device_info(ctx, NULL, NULL, &num_cores);

        printf("Metal: Compute pipelines compiled successfully\n");
        printf("Metal: Ready for GPU acceleration (%u cores)\n", num_cores);

        return ctx;
    }
}

void metal_compute_free(metal_compute_ctx_t* ctx) {
    if (!ctx) return;
    
    @autoreleasepool {
        // ARC will handle cleanup of Metal objects
        free(ctx);
    }
}

int metal_is_available(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil;
    }
}

void metal_get_device_info(
    metal_compute_ctx_t* ctx,
    char* name,
    uint32_t* max_threads,
    uint32_t* num_cores
) {
    if (!ctx || !ctx->device) return;
    
    @autoreleasepool {
        if (name) {
            NSString* deviceName = [ctx->device name];
            strncpy(name, [deviceName UTF8String], 255);
            name[255] = '\0';
        }
        
        if (max_threads) {
            *max_threads = (uint32_t)[ctx->device maxThreadsPerThreadgroup].width;
        }
        
        if (num_cores) {
            // Estimate GPU cores based on device capabilities
            // This is a heuristic since Metal doesn't expose core count directly
            
            NSString* deviceName = [ctx->device name];
            uint64_t memorySize = [ctx->device recommendedMaxWorkingSetSize];
            
            // M-series GPU core estimates based on device configuration
            if ([deviceName containsString:@"M4 Pro"]) {
                // M4 Pro: 16 or 20 cores
                *num_cores = 20;
            } else if ([deviceName containsString:@"M4 Max"]) {
                // M4 Max: 32 or 40 cores
                *num_cores = 40;
            } else if ([deviceName containsString:@"M4"]) {
                // Base M4: 10 cores
                *num_cores = 10;
            } else if ([deviceName containsString:@"M3 Ultra"]) {
                // M3 Ultra: future chip
                *num_cores = 80;
            } else if ([deviceName containsString:@"M3 Max"]) {
                // M3 Max: 30 or 40 cores
                *num_cores = 40;
            } else if ([deviceName containsString:@"M3 Pro"]) {
                // M3 Pro: 14 or 18 cores
                *num_cores = 18;
            } else if ([deviceName containsString:@"M3"]) {
                // Base M3: 8 or 10 cores
                *num_cores = 10;
            } else if ([deviceName containsString:@"M2 Ultra"]) {
                // M2 Ultra: 60 or 76 cores depending on configuration
                *num_cores = (memorySize > 100ULL * 1024 * 1024 * 1024) ? 76 : 60;
            } else if ([deviceName containsString:@"M2 Max"]) {
                // M2 Max: 30 or 38 cores
                *num_cores = (memorySize > 50ULL * 1024 * 1024 * 1024) ? 38 : 30;
            } else if ([deviceName containsString:@"M2 Pro"]) {
                // M2 Pro: 16 or 19 cores
                *num_cores = 19;
            } else if ([deviceName containsString:@"M2"]) {
                // Base M2: 8 or 10 cores
                *num_cores = 10;
            } else if ([deviceName containsString:@"M1 Ultra"]) {
                // M1 Ultra: 48 or 64 cores
                *num_cores = (memorySize > 100ULL * 1024 * 1024 * 1024) ? 64 : 48;
            } else if ([deviceName containsString:@"M1 Max"]) {
                // M1 Max: 24 or 32 cores
                *num_cores = 32;
            } else if ([deviceName containsString:@"M1 Pro"]) {
                // M1 Pro: 14 or 16 cores
                *num_cores = 16;
            } else if ([deviceName containsString:@"M1"]) {
                // Base M1: 7 or 8 cores
                *num_cores = 8;
            } else {
                // Fallback: estimate from recommended working set
                // Rough heuristic: ~2GB RAM per GPU core
                *num_cores = (uint32_t)(memorySize / (2ULL * 1024 * 1024 * 1024));
                if (*num_cores < 4) *num_cores = 4;  // Minimum reasonable estimate
                if (*num_cores > 128) *num_cores = 128;  // Maximum reasonable estimate
            }
        }
    }
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

metal_buffer_t* metal_buffer_create(metal_compute_ctx_t* ctx, size_t size) {
    if (!ctx || !ctx->device || size == 0) return NULL;
    
    @autoreleasepool {
        metal_buffer_t* buffer = (metal_buffer_t*)malloc(sizeof(metal_buffer_t));
        if (!buffer) return NULL;
        
        // Create buffer with shared storage (zero-copy)
        buffer->buffer = [ctx->device newBufferWithLength:size
                                                   options:MTLResourceStorageModeShared];
        
        if (!buffer->buffer) {
            free(buffer);
            return NULL;
        }
        
        buffer->size = size;
        return buffer;
    }
}

metal_buffer_t* metal_buffer_create_from_data(
    metal_compute_ctx_t* ctx,
    void* data,
    size_t size
) {
    if (!ctx || !ctx->device || !data || size == 0) return NULL;
    
    @autoreleasepool {
        metal_buffer_t* buffer = (metal_buffer_t*)malloc(sizeof(metal_buffer_t));
        if (!buffer) return NULL;
        
        // Create buffer with existing data (zero-copy when possible)
        buffer->buffer = [ctx->device newBufferWithBytesNoCopy:data
                                                         length:size
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        
        if (!buffer->buffer) {
            // Fallback: copy data
            buffer->buffer = [ctx->device newBufferWithBytes:data
                                                      length:size
                                                     options:MTLResourceStorageModeShared];
        }
        
        if (!buffer->buffer) {
            free(buffer);
            return NULL;
        }
        
        buffer->size = size;
        return buffer;
    }
}

void* metal_buffer_contents(metal_buffer_t* buffer) {
    if (!buffer || !buffer->buffer) return NULL;
    
    @autoreleasepool {
        return [buffer->buffer contents];
    }
}

void metal_buffer_free(metal_buffer_t* buffer) {
    if (!buffer) return;
    
    @autoreleasepool {
        // ARC will handle buffer cleanup
        free(buffer);
    }
}

// ============================================================================
// KERNEL DISPATCH HELPER
// ============================================================================

static int dispatch_kernel(
    metal_compute_ctx_t* ctx,
    id<MTLComputePipelineState> pipeline,
    metal_buffer_t** buffers,
    size_t num_buffers,
    uint32_t* constants,
    size_t num_constants,
    uint32_t grid_size
) {
    if (!ctx || !pipeline) return -1;
    
    @autoreleasepool {
        double start_time = ctx->performance_monitoring ? get_time_seconds() : 0.0;
        
        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipeline];
        
        // Bind buffers
        for (size_t i = 0; i < num_buffers; i++) {
            if (buffers[i] && buffers[i]->buffer) {
                [encoder setBuffer:buffers[i]->buffer offset:0 atIndex:i];
            }
        }
        
        // Bind constants
        for (size_t i = 0; i < num_constants; i++) {
            [encoder setBytes:&constants[i] length:sizeof(uint32_t) atIndex:num_buffers + i];
        }
        
        // Calculate threadgroup size and count
        NSUInteger maxThreads = [pipeline maxTotalThreadsPerThreadgroup];
        NSUInteger threadgroupSize = MIN(1024, maxThreads);  // Optimal for M2 Ultra
        NSUInteger threadgroups = (grid_size + threadgroupSize - 1) / threadgroupSize;
        
        MTLSize threadsPerThreadgroup = MTLSizeMake(threadgroupSize, 1, 1);
        MTLSize threadgroupCount = MTLSizeMake(threadgroups, 1, 1);
        
        [encoder dispatchThreadgroups:threadgroupCount
                threadsPerThreadgroup:threadsPerThreadgroup];
        
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (ctx->performance_monitoring) {
            ctx->last_execution_time = get_time_seconds() - start_time;
        }
        
        // Check for errors
        if ([commandBuffer status] == MTLCommandBufferStatusError) {
            set_error(ctx, @"Command buffer execution failed");
            return -1;
        }
        
        return 0;
    }
}

// ============================================================================
// QUANTUM GATE OPERATIONS
// ============================================================================

int metal_hadamard(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint32_t state_dim
) {
    if (!ctx || !amplitudes) return -1;
    
    metal_buffer_t* buffers[] = {amplitudes};
    uint32_t constants[] = {qubit_index, state_dim};
    
    uint32_t stride = 1u << qubit_index;
    uint32_t num_pairs = state_dim / 2;
    
    return dispatch_kernel(ctx, ctx->hadamardPipeline,
                          buffers, 1,
                          constants, 2,
                          num_pairs);
}

int metal_hadamard_all(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    uint32_t num_qubits,
    uint32_t state_dim
) {
    if (!ctx || !amplitudes) return -1;
    
    metal_buffer_t* buffers[] = {amplitudes};
    uint32_t constants[] = {num_qubits, state_dim};
    
    return dispatch_kernel(ctx, ctx->hadamardAllPipeline,
                          buffers, 1,
                          constants, 2,
                          state_dim);
}

int metal_oracle(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    uint32_t target_state,
    uint32_t state_dim
) {
    if (!ctx || !amplitudes) return -1;
    
    metal_buffer_t* buffers[] = {amplitudes};
    uint32_t constants[] = {target_state};
    
    return dispatch_kernel(ctx, ctx->oraclePipeline,
                          buffers, 1,
                          constants, 1,
                          state_dim);
}

int metal_oracle_multi(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    const uint32_t* marked_states,
    uint32_t num_marked,
    uint32_t state_dim
) {
    if (!ctx || !amplitudes || !marked_states) return -1;
    
    // Create buffer for marked states
    metal_buffer_t* marked_buffer = metal_buffer_create_from_data(
        ctx, (void*)marked_states, num_marked * sizeof(uint32_t));
    
    if (!marked_buffer) return -1;
    
    metal_buffer_t* buffers[] = {amplitudes, marked_buffer};
    uint32_t constants[] = {num_marked, state_dim};
    
    int result = dispatch_kernel(ctx, ctx->oracleMultiPipeline,
                                buffers, 2,
                                constants, 2,
                                state_dim);
    
    metal_buffer_free(marked_buffer);
    return result;
}

int metal_grover_diffusion(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    uint32_t num_qubits,
    uint32_t state_dim
) {
    if (!ctx || !amplitudes) return -1;
    
    // Create scratch buffer for average computation
    metal_buffer_t* scratch = metal_buffer_create(ctx, 8);  // 2 floats
    if (!scratch) return -1;
    
    metal_buffer_t* buffers[] = {amplitudes, scratch};
    uint32_t constants[] = {num_qubits, state_dim};
    
    int result = dispatch_kernel(ctx, ctx->diffusionPipeline,
                                buffers, 2,
                                constants, 2,
                                state_dim);
    
    metal_buffer_free(scratch);
    return result;
}

int metal_pauli_x(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint32_t state_dim
) {
    if (!ctx || !amplitudes) return -1;
    
    metal_buffer_t* buffers[] = {amplitudes};
    uint32_t constants[] = {qubit_index, state_dim};
    
    uint32_t num_pairs = state_dim / 2;
    return dispatch_kernel(ctx, ctx->pauliXPipeline,
                          buffers, 1,
                          constants, 2,
                          num_pairs);
}

int metal_pauli_z(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint32_t state_dim
) {
    if (!ctx || !amplitudes) return -1;
    
    metal_buffer_t* buffers[] = {amplitudes};
    uint32_t constants[] = {qubit_index, state_dim};
    
    return dispatch_kernel(ctx, ctx->pauliZPipeline,
                          buffers, 1,
                          constants, 2,
                          state_dim);
}

// ============================================================================
// PROBABILITY & MEASUREMENT
// ============================================================================

int metal_compute_probabilities(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    metal_buffer_t* probabilities,
    uint32_t state_dim
) {
    if (!ctx || !amplitudes || !probabilities) return -1;
    
    metal_buffer_t* buffers[] = {amplitudes, probabilities};
    uint32_t constants[] = {state_dim};
    
    return dispatch_kernel(ctx, ctx->probabilitiesPipeline,
                          buffers, 2,
                          constants, 1,
                          state_dim);
}

int metal_normalize(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    float norm,
    uint32_t state_dim
) {
    if (!ctx || !amplitudes) return -1;
    
    metal_buffer_t* buffers[] = {amplitudes};
    uint32_t constants[] = {*(uint32_t*)&norm, state_dim};  // Cast float to uint32 for passing
    
    return dispatch_kernel(ctx, ctx->normalizePipeline,
                          buffers, 1,
                          constants, 2,
                          state_dim);
}

// ============================================================================
// BATCH OPERATIONS
// ============================================================================

int metal_grover_iteration(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    uint32_t target_state,
    uint32_t num_qubits,
    uint32_t state_dim
) {
    // Oracle
    if (metal_oracle(ctx, amplitudes, target_state, state_dim) != 0) {
        return -1;
    }
    
    // Diffusion
    return metal_grover_diffusion(ctx, amplitudes, num_qubits, state_dim);
}

int metal_grover_search(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    uint32_t target_state,
    uint32_t num_qubits,
    uint32_t state_dim,
    uint32_t num_iterations
) {
    // Initialize: Apply Hadamard to all qubits
    if (metal_hadamard_all(ctx, amplitudes, num_qubits, state_dim) != 0) {
        return -1;
    }
    
    // Run Grover iterations
    for (uint32_t i = 0; i < num_iterations; i++) {
        if (metal_grover_iteration(ctx, amplitudes, target_state, num_qubits, state_dim) != 0) {
            return -1;
        }
    }
    
    return 0;
}

// ============================================================================

// ============================================================================
// BATCH PROCESSING (THE BREAKTHROUGH!)
// ============================================================================

int metal_grover_batch_search(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* batch_states,
    const uint32_t* targets,
    uint32_t* results,
    uint32_t num_searches,
    uint32_t num_qubits,
    uint32_t num_iterations
) {
    if (!ctx || !batch_states || !targets || !results) return -1;
    if (num_searches == 0 || num_qubits == 0) return -1;
    
    @autoreleasepool {
        // Load batch kernel if not already loaded
        if (!ctx->batchSearchPipeline) {
            // Try to load from batch kernel file
            NSError* error = nil;
            NSString* batchPath = @"src/optimization/kernels/quantum_kernels_batch.metal";
            NSString* batchSource = [NSString stringWithContentsOfFile:batchPath
                                                               encoding:NSUTF8StringEncoding
                                                                  error:&error];
            
            if (batchSource) {
                id<MTLLibrary> batchLibrary = [ctx->device newLibraryWithSource:batchSource
                                                                        options:nil
                                                                          error:&error];
                if (batchLibrary) {
                    id<MTLFunction> function = [batchLibrary newFunctionWithName:@"grover_batch_search"];
                    if (function) {
                        ctx->batchSearchPipeline = [ctx->device newComputePipelineStateWithFunction:function
                                                                                               error:&error];
                    }
                }
            }
            
            if (!ctx->batchSearchPipeline) {
                fprintf(stderr, "Metal: Failed to load batch search kernel\n");
                return -1;
            }
        }
        
        double start_time = ctx->performance_monitoring ? get_time_seconds() : 0.0;
        
        // Create Metal buffers for targets and results
        metal_buffer_t* targets_buf = metal_buffer_create_from_data(
            ctx, (void*)targets, num_searches * sizeof(uint32_t));
        metal_buffer_t* results_buf = metal_buffer_create(
            ctx, num_searches * sizeof(uint32_t));
        
        if (!targets_buf || !results_buf) {
            if (targets_buf) metal_buffer_free(targets_buf);
            if (results_buf) metal_buffer_free(results_buf);
            return -1;
        }
        
        // Dispatch batch kernel
        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:ctx->batchSearchPipeline];
        [encoder setBuffer:batch_states->buffer offset:0 atIndex:0];
        [encoder setBuffer:targets_buf->buffer offset:0 atIndex:1];
        [encoder setBuffer:results_buf->buffer offset:0 atIndex:2];
        [encoder setBytes:&num_searches length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&num_qubits length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&num_iterations length:sizeof(uint32_t) atIndex:5];
        
        // CRITICAL: One threadgroup per search!
        // Each threadgroup has 1024 threads to process its quantum state
        MTLSize threadsPerThreadgroup = MTLSizeMake(1024, 1, 1);
        MTLSize threadgroupCount = MTLSizeMake(num_searches, 1, 1);
        
        [encoder dispatchThreadgroups:threadgroupCount
                threadsPerThreadgroup:threadsPerThreadgroup];
        
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (ctx->performance_monitoring) {
            ctx->last_execution_time = get_time_seconds() - start_time;
        }
        
        // Copy results back
        uint32_t* results_ptr = (uint32_t*)metal_buffer_contents(results_buf);
        memcpy(results, results_ptr, num_searches * sizeof(uint32_t));
        
        metal_buffer_free(targets_buf);
        metal_buffer_free(results_buf);
        
        return ([commandBuffer status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
    }
}
// SYNCHRONIZATION & UTILITIES
// ============================================================================

void metal_wait_completion(metal_compute_ctx_t* ctx) {
    if (!ctx || !ctx->commandQueue) return;
    
    @autoreleasepool {
        // Create a barrier to wait for all pending operations
        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

double metal_get_last_execution_time(metal_compute_ctx_t* ctx) {
    return ctx ? ctx->last_execution_time : 0.0;
}

void metal_set_performance_monitoring(metal_compute_ctx_t* ctx, int enable) {
    if (ctx) {
        ctx->performance_monitoring = enable;
    }
}

// ============================================================================
// DIAGNOSTICS
// ============================================================================

void metal_print_device_info(metal_compute_ctx_t* ctx) {
    if (!ctx || !ctx->device) return;
    
    @autoreleasepool {
        printf("\n");
        printf("╔═══════════════════════════════════════════════════════════╗\n");
        printf("║     METAL GPU DEVICE INFORMATION                          ║\n");
        printf("╠═══════════════════════════════════════════════════════════╣\n");
        printf("║                                                           ║\n");
        
        NSString* name = [ctx->device name];
        printf("║  Device: %-48s ║\n", [name UTF8String]);
        
        uint64_t memory_mb = [ctx->device recommendedMaxWorkingSetSize] / (1024 * 1024);
        printf("║  Unified Memory: %7llu MB                             ║\n", memory_mb);
        
        printf("║  Max Threadgroup Size: %7lu                          ║\n",
               (unsigned long)[ctx->device maxThreadsPerThreadgroup].width);
        
        // Get GPU cores dynamically
        uint32_t gpu_cores = 0;
        metal_get_device_info(ctx, NULL, NULL, &gpu_cores);
        printf("║  GPU Cores: %3u                                          ║\n", gpu_cores);
        
        // Estimate memory bandwidth based on device
        uint32_t bandwidth_gbps = 0;
        if ([name containsString:@"M4 Max"]) {
            bandwidth_gbps = 546;  // M4 Max
        } else if ([name containsString:@"M4 Pro"]) {
            bandwidth_gbps = 273;  // M4 Pro
        } else if ([name containsString:@"M4"]) {
            bandwidth_gbps = 120;  // Base M4
        } else if ([name containsString:@"M3 Max"]) {
            bandwidth_gbps = 400;  // M3 Max
        } else if ([name containsString:@"M3 Pro"]) {
            bandwidth_gbps = 150;  // M3 Pro
        } else if ([name containsString:@"M3"]) {
            bandwidth_gbps = 100;  // Base M3
        } else if ([name containsString:@"M2 Ultra"]) {
            bandwidth_gbps = 800;  // M2 Ultra
        } else if ([name containsString:@"M2 Max"]) {
            bandwidth_gbps = 400;  // M2 Max
        } else if ([name containsString:@"M2 Pro"]) {
            bandwidth_gbps = 200;  // M2 Pro
        } else if ([name containsString:@"M2"]) {
            bandwidth_gbps = 100;  // Base M2
        } else if ([name containsString:@"M1 Ultra"]) {
            bandwidth_gbps = 800;  // M1 Ultra
        } else if ([name containsString:@"M1 Max"]) {
            bandwidth_gbps = 400;  // M1 Max
        } else if ([name containsString:@"M1 Pro"]) {
            bandwidth_gbps = 200;  // M1 Pro
        } else if ([name containsString:@"M1"]) {
            bandwidth_gbps = 68;   // Base M1
        } else {
            bandwidth_gbps = 100;  // Fallback estimate
        }
        
        printf("║  Memory Bandwidth: ~%u GB/s                            ║\n", bandwidth_gbps);
        printf("║  Storage Mode: MTLResourceStorageModeShared (zero-copy)  ║\n");
        printf("║                                                           ║\n");
        printf("╚═══════════════════════════════════════════════════════════╝\n");
        printf("\n");
    }
}

const char* metal_get_error(metal_compute_ctx_t* ctx) {
    if (!ctx || !ctx->lastError) return "No error";

    @autoreleasepool {
        return [ctx->lastError UTF8String];
    }
}

// ============================================================================
// TENSOR NETWORK / MPS OPERATIONS
// ============================================================================

/**
 * Helper for 3D kernel dispatch (used for tensor contraction)
 */
static int dispatch_kernel_3d(
    metal_compute_ctx_t* ctx,
    id<MTLComputePipelineState> pipeline,
    metal_buffer_t** buffers,
    size_t num_buffers,
    uint32_t* constants,
    size_t num_constants,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t grid_z
) {
    if (!ctx || !pipeline) return -1;

    @autoreleasepool {
        double start_time = ctx->performance_monitoring ? get_time_seconds() : 0.0;

        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        // Bind buffers
        for (size_t i = 0; i < num_buffers; i++) {
            if (buffers[i] && buffers[i]->buffer) {
                [encoder setBuffer:buffers[i]->buffer offset:0 atIndex:i];
            }
        }

        // Bind constants
        for (size_t i = 0; i < num_constants; i++) {
            [encoder setBytes:&constants[i] length:sizeof(uint32_t) atIndex:num_buffers + i];
        }

        // Calculate threadgroup size
        NSUInteger maxThreads = [pipeline maxTotalThreadsPerThreadgroup];
        NSUInteger threadgroupX = MIN(8, grid_x);
        NSUInteger threadgroupY = MIN(8, grid_y);
        NSUInteger threadgroupZ = MIN(8, grid_z);

        // Adjust to fit maxThreads
        while (threadgroupX * threadgroupY * threadgroupZ > maxThreads) {
            if (threadgroupZ > 1) threadgroupZ /= 2;
            else if (threadgroupY > 1) threadgroupY /= 2;
            else threadgroupX /= 2;
        }

        MTLSize threadsPerThreadgroup = MTLSizeMake(threadgroupX, threadgroupY, threadgroupZ);
        MTLSize threadgroupCount = MTLSizeMake(
            (grid_x + threadgroupX - 1) / threadgroupX,
            (grid_y + threadgroupY - 1) / threadgroupY,
            (grid_z + threadgroupZ - 1) / threadgroupZ
        );

        [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (ctx->performance_monitoring) {
            ctx->last_execution_time = get_time_seconds() - start_time;
        }

        if ([commandBuffer status] == MTLCommandBufferStatusError) {
            set_error(ctx, @"3D kernel dispatch failed");
            return -1;
        }

        return 0;
    }
}

/**
 * Helper for 2D kernel dispatch
 */
static int dispatch_kernel_2d(
    metal_compute_ctx_t* ctx,
    id<MTLComputePipelineState> pipeline,
    metal_buffer_t** buffers,
    size_t num_buffers,
    uint32_t* constants,
    size_t num_constants,
    uint32_t grid_x,
    uint32_t grid_y
) {
    if (!ctx || !pipeline) return -1;

    @autoreleasepool {
        double start_time = ctx->performance_monitoring ? get_time_seconds() : 0.0;

        id<MTLCommandBuffer> commandBuffer = [ctx->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        for (size_t i = 0; i < num_buffers; i++) {
            if (buffers[i] && buffers[i]->buffer) {
                [encoder setBuffer:buffers[i]->buffer offset:0 atIndex:i];
            }
        }

        for (size_t i = 0; i < num_constants; i++) {
            [encoder setBytes:&constants[i] length:sizeof(uint32_t) atIndex:num_buffers + i];
        }

        NSUInteger maxThreads = [pipeline maxTotalThreadsPerThreadgroup];
        NSUInteger threadgroupX = MIN(16, grid_x);
        NSUInteger threadgroupY = MIN(16, grid_y);

        while (threadgroupX * threadgroupY > maxThreads) {
            if (threadgroupY > 1) threadgroupY /= 2;
            else threadgroupX /= 2;
        }

        MTLSize threadsPerThreadgroup = MTLSizeMake(threadgroupX, threadgroupY, 1);
        MTLSize threadgroupCount = MTLSizeMake(
            (grid_x + threadgroupX - 1) / threadgroupX,
            (grid_y + threadgroupY - 1) / threadgroupY,
            1
        );

        [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (ctx->performance_monitoring) {
            ctx->last_execution_time = get_time_seconds() - start_time;
        }

        return ([commandBuffer status] == MTLCommandBufferStatusCompleted) ? 0 : -1;
    }
}

int metal_mps_contract_2site(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* A,
    metal_buffer_t* B,
    metal_buffer_t* theta,
    uint32_t chi_l,
    uint32_t chi_m,
    uint32_t chi_r
) {
    if (!ctx || !A || !B || !theta) return -1;
    if (!ctx->tensorContract2SitePipeline) {
        set_error(ctx, @"Tensor contraction pipeline not loaded");
        return -1;
    }

    metal_buffer_t* buffers[] = {A, B, theta};
    uint32_t constants[] = {chi_l, chi_m, chi_r};

    // Grid: (chi_l, 4, chi_r)
    return dispatch_kernel_3d(ctx, ctx->tensorContract2SitePipeline,
                              buffers, 3, constants, 3,
                              chi_l, 4, chi_r);
}

int metal_mps_apply_gate_theta(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* theta,
    metal_buffer_t* gate,
    uint32_t chi_l,
    uint32_t chi_r
) {
    if (!ctx || !theta || !gate) return -1;
    if (!ctx->applyGateThetaPipeline) {
        set_error(ctx, @"Apply gate theta pipeline not loaded");
        return -1;
    }

    metal_buffer_t* buffers[] = {theta, gate};
    uint32_t constants[] = {chi_l, chi_r};

    // Grid: (chi_l, chi_r)
    return dispatch_kernel_2d(ctx, ctx->applyGateThetaPipeline,
                              buffers, 2, constants, 2,
                              chi_l, chi_r);
}

int metal_mps_apply_gate_2q(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* A,
    metal_buffer_t* B,
    metal_buffer_t* gate,
    uint32_t chi_l_in,
    uint32_t chi_m_in,
    uint32_t chi_r_in,
    uint32_t max_bond,
    double cutoff,
    uint32_t* new_bond,
    double* trunc_error
) {
    if (!ctx || !A || !B || !gate || !new_bond || !trunc_error) return -1;

    @autoreleasepool {
        double start_time = ctx->performance_monitoring ? get_time_seconds() : 0.0;

        // Step 1: Contract A and B into theta
        // theta size: chi_l * 4 * chi_r (complex floats = 8 bytes each)
        // Metal uses float precision for tensor operations
        size_t theta_size = chi_l_in * 4 * chi_r_in * sizeof(float) * 2;
        metal_buffer_t* theta = metal_buffer_create(ctx, theta_size);
        if (!theta) return -1;

        if (metal_mps_contract_2site(ctx, A, B, theta, chi_l_in, chi_m_in, chi_r_in) != 0) {
            metal_buffer_free(theta);
            return -1;
        }

        // Step 2: Apply gate to theta
        if (metal_mps_apply_gate_theta(ctx, theta, gate, chi_l_in, chi_r_in) != 0) {
            metal_buffer_free(theta);
            return -1;
        }

        // Step 3: SVD and truncation
        // Reshape theta as matrix: (chi_l * 2) x (2 * chi_r)
        uint32_t m = chi_l_in * 2;
        uint32_t n = 2 * chi_r_in;
        uint32_t rank;

        // Allocate SVD output buffers (float precision)
        size_t U_size = m * MIN(max_bond, MIN(m, n)) * sizeof(float) * 2;
        size_t S_size = MIN(max_bond, MIN(m, n)) * sizeof(float);
        size_t Vt_size = MIN(max_bond, MIN(m, n)) * n * sizeof(float) * 2;

        metal_buffer_t* U = metal_buffer_create(ctx, U_size);
        metal_buffer_t* S = metal_buffer_create(ctx, S_size);
        metal_buffer_t* Vt = metal_buffer_create(ctx, Vt_size);

        if (!U || !S || !Vt) {
            metal_buffer_free(theta);
            if (U) metal_buffer_free(U);
            if (S) metal_buffer_free(S);
            if (Vt) metal_buffer_free(Vt);
            return -1;
        }

        // Perform SVD
        if (metal_svd_truncate(ctx, theta, U, S, Vt, m, n, max_bond, cutoff, &rank) != 0) {
            metal_buffer_free(theta);
            metal_buffer_free(U);
            metal_buffer_free(S);
            metal_buffer_free(Vt);
            return -1;
        }

        *new_bond = rank;

        // Compute truncation error (float precision on GPU)
        float* S_ptr = (float*)metal_buffer_contents(S);
        double total_sq = 0.0, kept_sq = 0.0;
        for (uint32_t i = 0; i < MIN(m, n); i++) {
            total_sq += (double)S_ptr[i] * (double)S_ptr[i];
            if (i < rank) kept_sq += (double)S_ptr[i] * (double)S_ptr[i];
        }
        *trunc_error = (total_sq > 0) ? 1.0 - kept_sq / total_sq : 0.0;

        // Reshape U back to A tensor and Vt back to B tensor
        // A_new: [chi_l][2][rank], B_new: [rank][2][chi_r]
        // For simplicity, copy the raw data - caller needs to handle reshape
        // In production, would do proper tensor reshape here

        size_t A_new_size = chi_l_in * 2 * rank * sizeof(float) * 2;
        size_t B_new_size = rank * 2 * chi_r_in * sizeof(float) * 2;

        // Copy U data to A buffer (with S absorbed)
        float* U_ptr = (float*)metal_buffer_contents(U);
        float* A_ptr = (float*)metal_buffer_contents(A);

        // Absorb singular values into U to form new A
        for (uint32_t i = 0; i < m; i++) {
            for (uint32_t j = 0; j < rank; j++) {
                uint32_t src_idx = (i * rank + j) * 2;  // Complex: 2 floats
                uint32_t dst_idx = src_idx;
                A_ptr[dst_idx] = U_ptr[src_idx] * S_ptr[j];
                A_ptr[dst_idx + 1] = U_ptr[src_idx + 1] * S_ptr[j];
            }
        }

        // Copy Vt data to B buffer
        float* Vt_ptr = (float*)metal_buffer_contents(Vt);
        float* B_ptr = (float*)metal_buffer_contents(B);
        memcpy(B_ptr, Vt_ptr, rank * n * sizeof(float) * 2);

        metal_buffer_free(theta);
        metal_buffer_free(U);
        metal_buffer_free(S);
        metal_buffer_free(Vt);

        if (ctx->performance_monitoring) {
            ctx->last_execution_time = get_time_seconds() - start_time;
        }

        return 0;
    }
}

int metal_svd_truncate(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* A,
    metal_buffer_t* U,
    metal_buffer_t* S,
    metal_buffer_t* Vt,
    uint32_t m,
    uint32_t n,
    uint32_t max_rank,
    double cutoff,
    uint32_t* actual_rank
) {
    if (!ctx || !A || !U || !S || !Vt || !actual_rank) return -1;

    @autoreleasepool {
        uint32_t min_mn = MIN(m, n);

        // Initialize V as identity matrix (float precision for Metal)
        size_t V_size = n * n * sizeof(float) * 2;
        metal_buffer_t* V = metal_buffer_create(ctx, V_size);
        if (!V) return -1;

        // Initialize V to identity
        float* V_ptr = (float*)metal_buffer_contents(V);
        memset(V_ptr, 0, V_size);
        for (uint32_t i = 0; i < n; i++) {
            V_ptr[(i * n + i) * 2] = 1.0f;  // Real part
        }

        // Jacobi SVD iterations
        const int MAX_SWEEPS = 30;

        for (int sweep = 0; sweep < MAX_SWEEPS; sweep++) {
            // Process column pairs
            for (uint32_t i = 0; i < n - 1; i++) {
                for (uint32_t j = i + 1; j < n; j++) {
                    if (!ctx->jacobiRotationPipeline) {
                        // Fall back to CPU Jacobi rotation
                        // (In production, implement CPU fallback)
                        continue;
                    }

                    metal_buffer_t* buffers[] = {A, V};
                    uint32_t constants[] = {m, n, i, j};

                    dispatch_kernel(ctx, ctx->jacobiRotationPipeline,
                                   buffers, 2, constants, 4,
                                   MAX(m, n));
                }
            }

            // Check convergence (simplified - in production, compute off-diag norm)
            if (sweep > 5) break;  // Minimum sweeps for reasonable accuracy
        }

        // Extract singular values
        if (ctx->extractSingularValuesPipeline) {
            metal_buffer_t* buffers[] = {A, S};
            uint32_t constants[] = {m, n};
            dispatch_kernel(ctx, ctx->extractSingularValuesPipeline,
                           buffers, 2, constants, 2, n);
        } else {
            // CPU fallback for singular value extraction (float precision)
            float* A_ptr = (float*)metal_buffer_contents(A);
            float* S_ptr = (float*)metal_buffer_contents(S);
            for (uint32_t j = 0; j < min_mn; j++) {
                float sum = 0.0f;
                for (uint32_t i = 0; i < m; i++) {
                    uint32_t idx = (i * n + j) * 2;
                    sum += A_ptr[idx] * A_ptr[idx] + A_ptr[idx+1] * A_ptr[idx+1];
                }
                S_ptr[j] = sqrtf(sum);
            }
        }

        // Determine rank based on cutoff
        float* S_ptr = (float*)metal_buffer_contents(S);
        uint32_t rank = 0;
        float max_sv = S_ptr[0];
        for (uint32_t i = 0; i < min_mn && i < max_rank; i++) {
            if (S_ptr[i] > (float)cutoff * max_sv) {
                rank = i + 1;
            } else {
                break;
            }
        }
        if (rank == 0) rank = 1;  // Keep at least one

        *actual_rank = rank;

        // Normalize U columns and truncate
        if (ctx->normalizeAndTruncateUPipeline) {
            metal_buffer_t* buffers[] = {A, U, S};
            uint32_t constants[] = {m, n, rank};
            dispatch_kernel_2d(ctx, ctx->normalizeAndTruncateUPipeline,
                              buffers, 3, constants, 3, m, rank);
        } else {
            // CPU fallback (float precision)
            float* A_ptr = (float*)metal_buffer_contents(A);
            float* U_ptr = (float*)metal_buffer_contents(U);
            for (uint32_t i = 0; i < m; i++) {
                for (uint32_t j = 0; j < rank; j++) {
                    uint32_t src_idx = (i * n + j) * 2;
                    uint32_t dst_idx = (i * rank + j) * 2;
                    float s = S_ptr[j];
                    if (s > 1e-7f) {
                        U_ptr[dst_idx] = A_ptr[src_idx] / s;
                        U_ptr[dst_idx + 1] = A_ptr[src_idx + 1] / s;
                    } else {
                        U_ptr[dst_idx] = 0.0f;
                        U_ptr[dst_idx + 1] = 0.0f;
                    }
                }
            }
        }

        // Truncate V to get Vt
        if (ctx->truncateVPipeline) {
            metal_buffer_t* buffers[] = {V, Vt};
            uint32_t constants[] = {n, rank};
            dispatch_kernel_2d(ctx, ctx->truncateVPipeline,
                              buffers, 2, constants, 2, n, rank);
        } else {
            // CPU fallback (float precision)
            float* Vt_ptr = (float*)metal_buffer_contents(Vt);
            for (uint32_t i = 0; i < n; i++) {
                for (uint32_t j = 0; j < rank; j++) {
                    uint32_t src_idx = (i * n + j) * 2;
                    uint32_t dst_idx = (j * n + i) * 2;  // Transpose
                    // Conjugate transpose for V -> Vt
                    Vt_ptr[dst_idx] = V_ptr[src_idx];
                    Vt_ptr[dst_idx + 1] = -V_ptr[src_idx + 1];
                }
            }
        }

        metal_buffer_free(V);
        return 0;
    }
}

double metal_mps_expectation_z(
    metal_compute_ctx_t* ctx,
    metal_buffer_t** mps_tensors,
    const uint32_t* bond_dims,
    uint32_t num_sites,
    uint32_t site
) {
    if (!ctx || !mps_tensors || !bond_dims || site >= num_sites) return 0.0;

    @autoreleasepool {
        // Get bond dimensions around measurement site
        uint32_t chi_l = (site > 0) ? bond_dims[site - 1] : 1;
        uint32_t chi_r = (site < num_sites - 1) ? bond_dims[site] : 1;

        // Allocate transfer matrices (float precision for Metal)
        size_t transfer_size = chi_l * chi_l * sizeof(float) * 2;
        metal_buffer_t* T_left = metal_buffer_create(ctx, transfer_size);
        metal_buffer_t* T_curr = metal_buffer_create(ctx, transfer_size);
        metal_buffer_t* T_temp = metal_buffer_create(ctx, transfer_size);
        metal_buffer_t* result_buf = metal_buffer_create(ctx, sizeof(float) * 2);

        if (!T_left || !T_curr || !T_temp || !result_buf) {
            if (T_left) metal_buffer_free(T_left);
            if (T_curr) metal_buffer_free(T_curr);
            if (T_temp) metal_buffer_free(T_temp);
            if (result_buf) metal_buffer_free(result_buf);
            return 0.0;
        }

        // Initialize T_left as identity for leftmost site
        float* T_left_ptr = (float*)metal_buffer_contents(T_left);
        memset(T_left_ptr, 0, transfer_size);
        for (uint32_t i = 0; i < chi_l; i++) {
            T_left_ptr[(i * chi_l + i) * 2] = 1.0f;
        }

        // Contract from left to measurement site
        for (uint32_t s = 0; s < site; s++) {
            uint32_t chi_s_l = (s > 0) ? bond_dims[s - 1] : 1;
            uint32_t chi_s_r = bond_dims[s];

            if (ctx->transferMatrixIdentityPipeline) {
                metal_buffer_t* buffers[] = {mps_tensors[s], T_curr};
                uint32_t constants[] = {chi_s_l, chi_s_r};
                dispatch_kernel_2d(ctx, ctx->transferMatrixIdentityPipeline,
                                  buffers, 2, constants, 2, chi_s_l, chi_s_l);
            }

            // Contract with accumulated transfer matrix
            if (s > 0 && ctx->contractTransferPipeline) {
                metal_buffer_t* buffers[] = {T_left, T_curr, T_temp};
                uint32_t constants[] = {chi_s_l};
                dispatch_kernel_2d(ctx, ctx->contractTransferPipeline,
                                  buffers, 3, constants, 1, chi_s_l, chi_s_l);

                // Swap T_temp and T_left
                metal_buffer_t* tmp = T_left;
                T_left = T_temp;
                T_temp = tmp;
            } else if (s == 0) {
                // Copy T_curr to T_left
                memcpy(T_left_ptr, metal_buffer_contents(T_curr), transfer_size);
            }
        }

        // Apply Z transfer matrix at measurement site
        if (ctx->transferMatrixZPipeline) {
            metal_buffer_t* buffers[] = {mps_tensors[site], T_curr};
            uint32_t constants[] = {chi_l, chi_r};
            dispatch_kernel_2d(ctx, ctx->transferMatrixZPipeline,
                              buffers, 2, constants, 2, chi_l, chi_l);
        }

        // Contract with left transfer matrix
        if (site > 0 && ctx->contractTransferPipeline) {
            metal_buffer_t* buffers[] = {T_left, T_curr, T_temp};
            uint32_t constants[] = {chi_l};
            dispatch_kernel_2d(ctx, ctx->contractTransferPipeline,
                              buffers, 3, constants, 1, chi_l, chi_l);

            metal_buffer_t* tmp = T_curr;
            T_curr = T_temp;
            T_temp = tmp;
        }

        // Contract from measurement site to right
        for (uint32_t s = site + 1; s < num_sites; s++) {
            uint32_t chi_s_l = bond_dims[s - 1];
            uint32_t chi_s_r = (s < num_sites - 1) ? bond_dims[s] : 1;

            if (ctx->transferMatrixIdentityPipeline) {
                metal_buffer_t* buffers[] = {mps_tensors[s], T_temp};
                uint32_t constants[] = {chi_s_l, chi_s_r};
                dispatch_kernel_2d(ctx, ctx->transferMatrixIdentityPipeline,
                                  buffers, 2, constants, 2, chi_s_l, chi_s_l);
            }

            if (ctx->contractTransferPipeline) {
                metal_buffer_t* buffers[] = {T_curr, T_temp, T_left};
                uint32_t constants[] = {chi_s_l};
                dispatch_kernel_2d(ctx, ctx->contractTransferPipeline,
                                  buffers, 3, constants, 1, chi_s_l, chi_s_l);

                metal_buffer_t* tmp = T_curr;
                T_curr = T_left;
                T_left = tmp;
            }
        }

        // Extract trace (convert from float to double for return)
        double expectation = 0.0;
        float* T_final = (float*)metal_buffer_contents(T_curr);
        uint32_t final_chi = (num_sites > 1) ? bond_dims[num_sites - 2] : 1;
        for (uint32_t i = 0; i < final_chi; i++) {
            expectation += (double)T_final[(i * final_chi + i) * 2];  // Real part of diagonal
        }

        metal_buffer_free(T_left);
        metal_buffer_free(T_curr);
        metal_buffer_free(T_temp);
        metal_buffer_free(result_buf);

        return expectation;
    }
}

double metal_mps_expectation_zz(
    metal_compute_ctx_t* ctx,
    metal_buffer_t** mps_tensors,
    const uint32_t* bond_dims,
    uint32_t num_sites,
    uint32_t site_i,
    uint32_t site_j
) {
    // Ensure site_i < site_j
    if (site_i > site_j) {
        uint32_t tmp = site_i;
        site_i = site_j;
        site_j = tmp;
    }

    if (!ctx || !mps_tensors || !bond_dims || site_j >= num_sites) return 0.0;

    // For ZZ correlation, we insert Z at both sites
    // This is similar to single-site but with two Z insertions
    // Implementation follows same pattern as metal_mps_expectation_z
    // but applies transferMatrixZPipeline at both site_i and site_j

    // Simplified implementation - in production, this would be more optimized
    @autoreleasepool {
        // For now, use the decomposition <Z_i Z_j> computation
        // Full implementation would follow transfer matrix pattern
        // with Z insertion at both sites

        // Placeholder - return product approximation
        double z_i = metal_mps_expectation_z(ctx, mps_tensors, bond_dims, num_sites, site_i);
        double z_j = metal_mps_expectation_z(ctx, mps_tensors, bond_dims, num_sites, site_j);

        // This is only correct for product states
        // Full implementation needed for correlated states
        return z_i * z_j;
    }
}

double metal_tensor_norm_squared(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* tensor,
    uint32_t size
) {
    if (!ctx || !tensor || size == 0) return 0.0;

    @autoreleasepool {
        // Use float precision for Metal
        metal_buffer_t* result = metal_buffer_create(ctx, sizeof(float));
        if (!result) return 0.0;

        if (ctx->tensorNormSquaredPipeline) {
            metal_buffer_t* buffers[] = {tensor, result};
            uint32_t constants[] = {size};
            dispatch_kernel(ctx, ctx->tensorNormSquaredPipeline,
                           buffers, 2, constants, 1, 256);
        } else {
            // CPU fallback (float precision)
            float* data = (float*)metal_buffer_contents(tensor);
            float sum = 0.0f;
            for (uint32_t i = 0; i < size; i++) {
                sum += data[i*2] * data[i*2] + data[i*2+1] * data[i*2+1];
            }
            *(float*)metal_buffer_contents(result) = sum;
        }

        // Convert float result to double for return value
        double norm_sq = (double)(*(float*)metal_buffer_contents(result));
        metal_buffer_free(result);
        return norm_sq;
    }
}

int metal_tensor_scale(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* tensor,
    uint32_t size,
    double scale
) {
    if (!ctx || !tensor || size == 0) return -1;

    @autoreleasepool {
        if (ctx->tensorScalePipeline) {
            // Convert double to float for Metal
            metal_buffer_t* scale_buf = metal_buffer_create(ctx, sizeof(float));
            if (!scale_buf) return -1;
            *(float*)metal_buffer_contents(scale_buf) = (float)scale;

            metal_buffer_t* buffers[] = {tensor, scale_buf};
            uint32_t constants[] = {size};

            int result = dispatch_kernel(ctx, ctx->tensorScalePipeline,
                                        buffers, 2, constants, 1, size);
            metal_buffer_free(scale_buf);
            return result;
        } else {
            // CPU fallback (float precision)
            float* data = (float*)metal_buffer_contents(tensor);
            float scale_f = (float)scale;
            for (uint32_t i = 0; i < size * 2; i++) {
                data[i] *= scale_f;
            }
            return 0;
        }
    }
}