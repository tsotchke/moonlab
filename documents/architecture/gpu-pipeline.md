# GPU Pipeline

Metal GPU acceleration architecture for Moonlab.

## Overview

Moonlab uses Apple's Metal framework to accelerate quantum simulation on Apple Silicon. The GPU pipeline provides 10-100x speedups for large state vectors by parallelizing amplitude operations across thousands of GPU threads.

## Architecture

### Component Diagram

```
┌───────────────────────────────────────────────────────────────┐
│                        CPU Side                               │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                   C Core Library                         │ │
│  │  quantum_state_h(state, qubit)                           │ │
│  └─────────────────────────┬────────────────────────────────┘ │
│                            │                                  │
│  ┌─────────────────────────▼────────────────────────────────┐ │
│  │                 GPU Backend Interface                    │ │
│  │  gpu_metal_apply_gate(state, gate, qubits)               │ │
│  └─────────────────────────┬────────────────────────────────┘ │
│                            │                                  │
│  ┌─────────────────────────▼────────────────────────────────┐ │
│  │              Metal Command Encoder                       │ │
│  │  - Shader selection                                      │ │
│  │  - Buffer binding                                        │ │
│  │  - Thread group sizing                                   │ │
│  └─────────────────────────┬────────────────────────────────┘ │
└────────────────────────────┼──────────────────────────────────┘
                             │ Command Buffer
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                        GPU Side                               │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                  Metal Shaders                           │ │
│  │  hadamard_kernel, cnot_kernel, rotation_kernel, etc.     │ │
│  └──────────────────────────────────────────────────────────┘ │
│                            │                                  │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                 GPU Memory                               │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │           State Vector Buffer                       │ │ │
│  │  │         (2^n × 16 bytes)                            │ │ │
│  │  └─────────────────────────────────────────────────────┘ │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │          Gate Parameter Buffers                     │ │ │
│  │  └─────────────────────────────────────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

### Execution Flow

```
Gate Application Request
         │
         ▼
┌────────────────────┐
│ Check GPU enabled  │ → GPU threshold, state size
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Ensure GPU buffer  │ → Allocate or update
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Select shader      │ → Based on gate type
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Encode command     │ → Bind buffers, set params
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Submit & execute   │ → GPU executes in parallel
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Synchronize        │ → Wait for completion
└────────────────────┘
```

## Initialization

### Device Setup

```objective-c
// gpu_metal.m

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_command_queue = nil;
static id<MTLLibrary> g_library = nil;

int gpu_metal_init(void) {
    // Get default Metal device
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
        return QSIM_ERROR_GPU_UNAVAILABLE;
    }

    // Check for Apple Silicon
    if (![g_device supportsFamily:MTLGPUFamilyApple7]) {
        // Fall back to CPU
        return QSIM_ERROR_GPU_UNAVAILABLE;
    }

    // Create command queue
    g_command_queue = [g_device newCommandQueue];

    // Load shader library
    NSBundle* bundle = [NSBundle bundleForClass:[self class]];
    NSURL* libURL = [bundle URLForResource:@"quantum_shaders"
                             withExtension:@"metallib"];
    NSError* error;
    g_library = [g_device newLibraryWithURL:libURL error:&error];

    if (!g_library) {
        NSLog(@"Failed to load shaders: %@", error);
        return QSIM_ERROR_GPU_SHADER_LOAD;
    }

    return QSIM_SUCCESS;
}
```

### Buffer Management

```objective-c
typedef struct {
    id<MTLBuffer> buffer;
    uint64_t size;
    bool dirty;  // CPU has updates not yet on GPU
} gpu_buffer_t;

gpu_buffer_t* gpu_buffer_create(uint64_t size) {
    gpu_buffer_t* buf = malloc(sizeof(gpu_buffer_t));

    buf->buffer = [g_device newBufferWithLength:size
                                        options:MTLResourceStorageModeShared];
    buf->size = size;
    buf->dirty = false;

    return buf;
}

void gpu_buffer_upload(gpu_buffer_t* buf, const void* data, uint64_t size) {
    memcpy([buf->buffer contents], data, size);
    buf->dirty = false;
}

void gpu_buffer_download(gpu_buffer_t* buf, void* data, uint64_t size) {
    memcpy(data, [buf->buffer contents], size);
}
```

## Shader Implementation

### Single-Qubit Gates

```metal
// quantum_shaders.metal

#include <metal_stdlib>
using namespace metal;

struct Complex {
    float real;
    float imag;
};

Complex complex_mul(Complex a, Complex b) {
    return Complex{
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
}

Complex complex_add(Complex a, Complex b) {
    return Complex{a.real + b.real, a.imag + b.imag};
}

kernel void hadamard_kernel(
    device Complex* amplitudes [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant uint& num_qubits [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const float inv_sqrt2 = 0.7071067811865476f;

    uint64_t dim = 1ULL << num_qubits;
    uint64_t half_dim = dim >> 1;

    if (gid >= half_dim) return;

    // Calculate index pair
    uint64_t mask = 1ULL << qubit;
    uint64_t low_mask = mask - 1;
    uint64_t high_mask = ~low_mask;

    uint64_t idx0 = ((gid & high_mask) << 1) | (gid & low_mask);
    uint64_t idx1 = idx0 | mask;

    // Load amplitudes
    Complex a0 = amplitudes[idx0];
    Complex a1 = amplitudes[idx1];

    // Apply Hadamard: H = (1/√2) * [[1, 1], [1, -1]]
    amplitudes[idx0] = Complex{
        (a0.real + a1.real) * inv_sqrt2,
        (a0.imag + a1.imag) * inv_sqrt2
    };
    amplitudes[idx1] = Complex{
        (a0.real - a1.real) * inv_sqrt2,
        (a0.imag - a1.imag) * inv_sqrt2
    };
}
```

### Rotation Gates

```metal
kernel void rz_kernel(
    device Complex* amplitudes [[buffer(0)]],
    constant uint& qubit [[buffer(1)]],
    constant uint& num_qubits [[buffer(2)]],
    constant float& theta [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint64_t dim = 1ULL << num_qubits;
    if (gid >= dim) return;

    uint64_t mask = 1ULL << qubit;

    // Rz(θ) = diag(exp(-iθ/2), exp(iθ/2))
    float half_theta = theta * 0.5f;

    if (gid & mask) {
        // Qubit is 1: multiply by exp(iθ/2)
        float cos_t = cos(half_theta);
        float sin_t = sin(half_theta);

        Complex a = amplitudes[gid];
        amplitudes[gid] = Complex{
            a.real * cos_t - a.imag * sin_t,
            a.real * sin_t + a.imag * cos_t
        };
    } else {
        // Qubit is 0: multiply by exp(-iθ/2)
        float cos_t = cos(half_theta);
        float sin_t = -sin(half_theta);

        Complex a = amplitudes[gid];
        amplitudes[gid] = Complex{
            a.real * cos_t - a.imag * sin_t,
            a.real * sin_t + a.imag * cos_t
        };
    }
}
```

### Two-Qubit Gates

```metal
kernel void cnot_kernel(
    device Complex* amplitudes [[buffer(0)]],
    constant uint& control [[buffer(1)]],
    constant uint& target [[buffer(2)]],
    constant uint& num_qubits [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint64_t dim = 1ULL << num_qubits;
    uint64_t quarter_dim = dim >> 2;

    if (gid >= quarter_dim) return;

    // Ensure control < target for index calculation
    uint c = min(control, target);
    uint t = max(control, target);

    // Calculate base index (both control and target bits = 0)
    uint64_t low_mask = (1ULL << c) - 1;
    uint64_t mid_mask = ((1ULL << (t - 1)) - 1) ^ low_mask;
    uint64_t high_mask = ~((1ULL << t) - 1);

    uint64_t base = ((gid & high_mask) << 2) |
                    (((gid >> (c)) & mid_mask) << 1) |
                    (gid & low_mask);

    uint64_t control_mask = 1ULL << control;
    uint64_t target_mask = 1ULL << target;

    // Only swap when control = 1
    uint64_t idx10 = base | control_mask;               // control=1, target=0
    uint64_t idx11 = base | control_mask | target_mask; // control=1, target=1

    // Swap amplitudes
    Complex temp = amplitudes[idx10];
    amplitudes[idx10] = amplitudes[idx11];
    amplitudes[idx11] = temp;
}
```

## Command Encoding

### Gate Dispatch

```objective-c
int gpu_metal_apply_hadamard(quantum_state_t* state, uint32_t qubit) {
    // Get pipeline state for Hadamard
    id<MTLComputePipelineState> pipeline = get_pipeline(@"hadamard_kernel");

    // Create command buffer
    id<MTLCommandBuffer> cmd_buffer = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

    // Bind buffers
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:state->gpu_buffer->buffer offset:0 atIndex:0];
    [encoder setBytes:&qubit length:sizeof(uint32_t) atIndex:1];
    [encoder setBytes:&state->num_qubits length:sizeof(uint32_t) atIndex:2];

    // Calculate thread groups
    uint64_t half_dim = state->dim >> 1;
    MTLSize grid_size = MTLSizeMake(half_dim, 1, 1);
    MTLSize thread_group_size = MTLSizeMake(
        MIN(256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1
    );

    [encoder dispatchThreads:grid_size
       threadsPerThreadgroup:thread_group_size];

    [encoder endEncoding];

    // Submit and wait
    [cmd_buffer commit];
    [cmd_buffer waitUntilCompleted];

    return QSIM_SUCCESS;
}
```

### Batched Execution

```objective-c
int gpu_metal_apply_circuit(quantum_state_t* state,
                           const gate_op_t* gates,
                           size_t num_gates) {
    id<MTLCommandBuffer> cmd_buffer = [g_command_queue commandBuffer];

    for (size_t i = 0; i < num_gates; i++) {
        id<MTLComputeCommandEncoder> encoder = [cmd_buffer computeCommandEncoder];

        // Encode gate based on type
        switch (gates[i].type) {
            case GATE_H:
                encode_hadamard(encoder, state, gates[i].qubits[0]);
                break;
            case GATE_CNOT:
                encode_cnot(encoder, state, gates[i].qubits[0], gates[i].qubits[1]);
                break;
            case GATE_RZ:
                encode_rz(encoder, state, gates[i].qubits[0], gates[i].params[0]);
                break;
            // ... more gates
        }

        [encoder endEncoding];
    }

    // Submit all at once
    [cmd_buffer commit];
    [cmd_buffer waitUntilCompleted];

    return QSIM_SUCCESS;
}
```

## Memory Transfer

### Synchronization Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Transfer Strategy                        │
│                                                             │
│  State Creation → CPU only                                  │
│       │                                                     │
│       ▼                                                     │
│  First GPU Gate → Upload to GPU (lazy)                      │
│       │                                                     │
│       ▼                                                     │
│  Subsequent Gates → GPU only (no transfer)                  │
│       │                                                     │
│       ▼                                                     │
│  CPU Access Needed → Download from GPU                      │
│  (measurement, amplitude read)                              │
│       │                                                     │
│       ▼                                                     │
│  Subsequent CPU Ops → CPU only, mark dirty                  │
│       │                                                     │
│       ▼                                                     │
│  Next GPU Gate → Re-upload if dirty                         │
└─────────────────────────────────────────────────────────────┘
```

### Implementation

```objective-c
void ensure_gpu_buffer(quantum_state_t* state) {
    if (!state->gpu_buffer) {
        // First GPU access - allocate and upload
        state->gpu_buffer = gpu_buffer_create(state->dim * sizeof(Complex));
        gpu_buffer_upload(state->gpu_buffer, state->amplitudes,
                         state->dim * sizeof(Complex));
        state->flags |= QSIM_FLAG_GPU_CURRENT;
    } else if (!(state->flags & QSIM_FLAG_GPU_CURRENT)) {
        // CPU modified since last GPU use - re-upload
        gpu_buffer_upload(state->gpu_buffer, state->amplitudes,
                         state->dim * sizeof(Complex));
        state->flags |= QSIM_FLAG_GPU_CURRENT;
    }
}

void ensure_cpu_current(quantum_state_t* state) {
    if (state->gpu_buffer && !(state->flags & QSIM_FLAG_CPU_CURRENT)) {
        // GPU modified since last CPU use - download
        gpu_buffer_download(state->gpu_buffer, state->amplitudes,
                           state->dim * sizeof(Complex));
        state->flags |= QSIM_FLAG_CPU_CURRENT;
    }
}
```

## Measurement on GPU

### Probability Computation

```metal
kernel void compute_probabilities(
    device const Complex* amplitudes [[buffer(0)]],
    device float* probabilities [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    Complex a = amplitudes[gid];
    probabilities[gid] = a.real * a.real + a.imag * a.imag;
}

kernel void prefix_sum_probabilities(
    device float* probabilities [[buffer(0)]],
    device float* block_sums [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint block_idx [[threadgroup_position_in_grid]]
) {
    // Parallel prefix sum for efficient CDF computation
    threadgroup float shared[256];

    shared[tid] = (gid < n) ? probabilities[gid] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep
    for (uint stride = 1; stride < 256; stride *= 2) {
        uint idx = (tid + 1) * stride * 2 - 1;
        if (idx < 256) {
            shared[idx] += shared[idx - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store block sum and clear last element
    if (tid == 255) {
        block_sums[block_idx] = shared[255];
        shared[255] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep
    for (uint stride = 128; stride >= 1; stride /= 2) {
        uint idx = (tid + 1) * stride * 2 - 1;
        if (idx < 256) {
            float temp = shared[idx - stride];
            shared[idx - stride] = shared[idx];
            shared[idx] += temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (gid < n) {
        probabilities[gid] = shared[tid];
    }
}
```

## Performance Optimization

### Thread Group Sizing

```objective-c
MTLSize optimal_thread_group_size(id<MTLComputePipelineState> pipeline,
                                   uint64_t total_threads) {
    NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;

    // Prefer power-of-2 sizes for efficiency
    NSUInteger threads_per_group = 256;

    while (threads_per_group > max_threads) {
        threads_per_group /= 2;
    }

    // For small problems, reduce thread group size
    if (total_threads < threads_per_group * 4) {
        threads_per_group = MAX(32, total_threads / 4);
    }

    return MTLSizeMake(threads_per_group, 1, 1);
}
```

### Kernel Fusion

For sequences of single-qubit gates on different qubits:

```metal
kernel void fused_single_qubit_gates(
    device Complex* amplitudes [[buffer(0)]],
    constant uint* qubits [[buffer(1)]],
    constant uint* gate_types [[buffer(2)]],
    constant float* params [[buffer(3)]],
    constant uint& num_gates [[buffer(4)]],
    constant uint& num_qubits [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint64_t dim = 1ULL << num_qubits;
    if (gid >= dim) return;

    Complex a = amplitudes[gid];

    // Apply all gates that affect this amplitude
    for (uint g = 0; g < num_gates; g++) {
        uint qubit = qubits[g];
        bool bit_set = (gid >> qubit) & 1;

        switch (gate_types[g]) {
            case GATE_Z:
                if (bit_set) {
                    a.real = -a.real;
                    a.imag = -a.imag;
                }
                break;
            case GATE_S:
                if (bit_set) {
                    float temp = a.real;
                    a.real = -a.imag;
                    a.imag = temp;
                }
                break;
            // More diagonal gates...
        }
    }

    amplitudes[gid] = a;
}
```

## Benchmarks

### Gate Performance

| Operation | CPU (30 qubits) | GPU (30 qubits) | Speedup |
|-----------|-----------------|-----------------|---------|
| Hadamard | 52 ms | 4.2 ms | 12x |
| CNOT | 98 ms | 7.1 ms | 14x |
| Rz | 45 ms | 3.8 ms | 12x |
| 10-gate circuit | 520 ms | 42 ms | 12x |
| Full measurement | 85 ms | 12 ms | 7x |

### Crossover Point

GPU becomes faster than CPU:

| Hardware | Crossover (qubits) |
|----------|-------------------|
| M1 | 17-18 |
| M1 Pro/Max | 16-17 |
| M2 | 16-17 |
| M2 Pro/Max | 15-16 |
| M3 Pro/Max | 15-16 |

## Error Handling

```objective-c
int gpu_metal_check_error(id<MTLCommandBuffer> cmd_buffer) {
    if (cmd_buffer.status == MTLCommandBufferStatusError) {
        NSError* error = cmd_buffer.error;

        switch (error.code) {
            case MTLCommandBufferErrorOutOfMemory:
                return QSIM_ERROR_GPU_OUT_OF_MEMORY;
            case MTLCommandBufferErrorTimeout:
                return QSIM_ERROR_GPU_TIMEOUT;
            default:
                return QSIM_ERROR_GPU_UNKNOWN;
        }
    }

    return QSIM_SUCCESS;
}
```

## See Also

- [State Vector Engine](state-vector-engine.md) - CPU implementation
- [System Overview](system-overview.md) - Architecture context
- [Tutorial: GPU Acceleration](../tutorials/09-gpu-acceleration.md) - Usage guide
- [C API: GPU Metal](../api/c/gpu-metal.md) - API reference

