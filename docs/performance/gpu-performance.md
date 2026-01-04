# GPU Performance

Metal GPU acceleration performance on Apple Silicon.

## Overview

Moonlab leverages Apple's Metal API for GPU acceleration, achieving significant speedups for state vector operations. The implementation uses zero-copy unified memory architecture, eliminating data transfer overhead between CPU and GPU.

## Supported Hardware

| Chip | GPU Cores | Unified Memory | Performance Tier |
|------|-----------|----------------|------------------|
| M1 | 7-8 | 8-16 GB | Entry |
| M1 Pro | 14-16 | 16-32 GB | Professional |
| M1 Max | 24-32 | 32-64 GB | High-end |
| M1 Ultra | 48-64 | 64-128 GB | Workstation |
| M2 | 8-10 | 8-24 GB | Entry |
| M2 Pro | 16-19 | 16-32 GB | Professional |
| M2 Max | 30-38 | 32-96 GB | High-end |
| M2 Ultra | 60-76 | 64-192 GB | Workstation |
| M3 | 8-10 | 8-24 GB | Entry |
| M3 Pro | 14-18 | 18-36 GB | Professional |
| M3 Max | 30-40 | 36-128 GB | High-end |
| M4 | 10 | 16-32 GB | Entry |
| M4 Pro | 16-20 | 24-48 GB | Professional |
| M4 Max | 32-40 | 48-128 GB | High-end |

## Enabling GPU Acceleration

### Build Configuration

```bash
# Build with Metal support (macOS only)
make METAL=1

# Verify GPU support
./bin/moonlab --version
# Output should include: GPU: Metal (Apple M2 Ultra)
```

### Runtime Configuration

```c
#include "optimization/gpu_metal.h"

// Initialize Metal context
metal_compute_ctx_t* gpu = metal_compute_init();

if (gpu) {
    // Get device info
    char device_name[256];
    uint32_t max_threads, num_cores;
    metal_get_device_info(gpu, device_name, &max_threads, &num_cores);

    printf("Device: %s\n", device_name);
    printf("GPU cores: %u\n", num_cores);
    printf("Max threads: %u\n", max_threads);
}
```

### Automatic GPU Usage

```c
// Configure threshold for automatic GPU usage
quantum_config_set("gpu.enabled", "true");
quantum_config_set("gpu.threshold", "16");  // Use GPU for ≥16 qubits
```

## Performance Benchmarks

### Single-Qubit Gates

**Hadamard Transform Performance (M2 Ultra, 76 cores):**

| Qubits | CPU (SIMD) | GPU (Metal) | Speedup |
|--------|------------|-------------|---------|
| 16 | 0.32 ms | 0.05 ms | 6.4x |
| 18 | 1.2 ms | 0.08 ms | 15x |
| 20 | 4.8 ms | 0.25 ms | 19x |
| 22 | 19 ms | 0.8 ms | 24x |
| 24 | 78 ms | 2.5 ms | 31x |
| 26 | 320 ms | 9 ms | 36x |
| 28 | 1.3 s | 35 ms | 37x |
| 30 | 5.8 s | 120 ms | 48x |

### Two-Qubit Gates

**CNOT Performance:**

| Qubits | CPU (OpenMP) | GPU (Metal) | Speedup |
|--------|--------------|-------------|---------|
| 20 | 12 ms | 0.4 ms | 30x |
| 24 | 180 ms | 4 ms | 45x |
| 28 | 3.2 s | 52 ms | 62x |

### Grover's Algorithm (Complete Search)

| Qubits | Search Space | CPU Time | GPU Time | Speedup |
|--------|--------------|----------|----------|---------|
| 16 | 65,536 | 2.8 s | 60 ms | 47x |
| 18 | 262,144 | 22 s | 0.4 s | 55x |
| 20 | 1,048,576 | 3.2 min | 3.5 s | 55x |
| 22 | 4,194,304 | 28 min | 28 s | 60x |
| 24 | 16,777,216 | 4.2 hr | 3.8 min | 66x |

### Tensor Network Operations

**MPS Gate Application:**

| Sites | Bond $\chi$ | CPU Time | GPU Time | Speedup |
|-------|-------------|----------|----------|---------|
| 100 | 64 | 8.5 ms | 0.6 ms | 14x |
| 200 | 128 | 68 ms | 3.2 ms | 21x |
| 500 | 256 | 850 ms | 28 ms | 30x |

**SVD Decomposition:**

| Matrix Size | CPU (Accelerate) | GPU (Metal) | Speedup |
|-------------|------------------|-------------|---------|
| 64x64 | 0.8 ms | 0.15 ms | 5x |
| 128x128 | 4.2 ms | 0.5 ms | 8x |
| 256x256 | 28 ms | 2.1 ms | 13x |
| 512x512 | 180 ms | 9 ms | 20x |

## GPU Kernel Architecture

### Compute Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                       Metal Compute Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Quantum    │    │   Command    │    │    GPU       │      │
│  │   State      │────│   Queue      │────│   Compute    │      │
│  │   Buffer     │    │              │    │   Cores      │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         │    Zero-Copy      │    Dispatch       │               │
│         └───────────────────┴───────────────────┘               │
│                                                                 │
│  Compiled Pipelines:                                            │
│  ├── hadamard_transform        ├── grover_diffusion_fused      │
│  ├── hadamard_all_qubits       ├── compute_probabilities       │
│  ├── oracle_single_target      ├── normalize_state             │
│  ├── sparse_oracle             ├── tensor_contract_2site       │
│  └── pauli_x / pauli_z         └── jacobi_svd_rotation         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Available Kernels

| Kernel | Purpose | Threads |
|--------|---------|---------|
| `hadamard_transform` | Single-qubit Hadamard | dim/2 |
| `hadamard_all_qubits` | All-qubit Hadamard | dim |
| `oracle_single_target` | Oracle for single marked state | dim |
| `sparse_oracle` | Oracle for multiple targets | dim |
| `grover_diffusion_fused` | Fused diffusion operator | dim |
| `compute_probabilities` | Extract probability vector | dim |
| `normalize_state` | Renormalize amplitudes | dim |
| `tensor_contract_2site` | MPS two-site contraction | variable |
| `jacobi_svd_rotation` | Jacobi SVD iteration | matrix elements |

### Thread Group Optimization

```metal
// Optimal thread group size for M2 Ultra
constant uint THREADGROUP_SIZE = 256;

kernel void hadamard_transform(
    device float2* amplitudes [[buffer(0)]],
    constant uint& target [[buffer(1)]],
    constant uint& dimension [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tgsize [[threads_per_threadgroup]]
) {
    uint stride = 1u << target;
    uint num_pairs = dimension >> 1;

    if (tid >= num_pairs) return;

    // Calculate pair indices
    uint i0 = ((tid >> target) << (target + 1)) | (tid & (stride - 1));
    uint i1 = i0 | stride;

    // Load amplitudes
    float2 a0 = amplitudes[i0];
    float2 a1 = amplitudes[i1];

    // Hadamard: 1/sqrt(2) * [[1,1],[1,-1]]
    float inv_sqrt2 = 0.7071067811865476f;
    amplitudes[i0] = (a0 + a1) * inv_sqrt2;
    amplitudes[i1] = (a0 - a1) * inv_sqrt2;
}
```

## Memory Management

### Unified Memory Architecture

Apple Silicon's unified memory eliminates GPU memory copies:

```objc
// Zero-copy buffer allocation
id<MTLBuffer> stateBuffer = [device newBufferWithBytesNoCopy:amplitudes
                                                       length:size
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
```

### Memory Bandwidth

| Chip | Memory Bandwidth | Practical Throughput |
|------|------------------|---------------------|
| M1 Ultra | 800 GB/s | ~600 GB/s |
| M2 Ultra | 800 GB/s | ~650 GB/s |
| M3 Max | 400 GB/s | ~340 GB/s |
| M4 Max | 546 GB/s | ~460 GB/s |

### Buffer Sizing

```c
// Optimal buffer alignment for Metal
size_t align_to_page(size_t size) {
    size_t page_size = 16384;  // 16 KB pages on Apple Silicon
    return (size + page_size - 1) & ~(page_size - 1);
}

// Allocate aligned buffer
size_t state_size = (1ULL << num_qubits) * sizeof(complex_t);
size_t aligned_size = align_to_page(state_size);
```

## Performance Tuning

### Threshold Selection

Choose GPU usage threshold based on overhead crossover:

```c
// Measured crossover points (M2 Ultra)
// Below these, CPU is faster due to dispatch overhead
int gpu_threshold_single = 16;  // Single-qubit gates
int gpu_threshold_two = 14;     // Two-qubit gates
int gpu_threshold_measure = 18; // Measurement operations

quantum_config_set_int("gpu.threshold.single", gpu_threshold_single);
quantum_config_set_int("gpu.threshold.two", gpu_threshold_two);
```

### Batch Operations

Batching multiple operations reduces dispatch overhead:

```c
// Less efficient: individual dispatches
for (int q = 0; q < num_qubits; q++) {
    metal_hadamard(gpu, state, q);  // Separate dispatch each
}

// More efficient: batched dispatch
metal_hadamard_all(gpu, state, num_qubits);  // Single dispatch
```

### Pipeline Warmup

First kernel invocation incurs compilation overhead:

```c
// Warmup GPU pipelines at initialization
void warmup_gpu(metal_compute_ctx_t* gpu) {
    quantum_state_t warmup;
    quantum_state_init(&warmup, 4);

    metal_hadamard(gpu, &warmup, 0);  // Triggers pipeline compile
    metal_cnot(gpu, &warmup, 0, 1);

    quantum_state_free(&warmup);
}
```

## Profiling

### Enable Performance Monitoring

```c
// Enable Metal performance monitoring
metal_set_performance_monitoring(gpu, 1);

// Run computation
metal_hadamard(gpu, state, target);

// Get timing
double execution_time = metal_get_last_execution_time(gpu);
printf("Kernel time: %.3f ms\n", execution_time * 1000);
```

### Xcode GPU Profiler

1. Build with debug symbols: `make DEBUG=1 METAL=1`
2. Open in Xcode: Product → Profile → Metal System Trace
3. Capture GPU timeline and occupancy

### Command Line Profiling

```bash
# Profile with Instruments
xcrun xctrace record --template "Metal System Trace" \
    --launch ./my_quantum_app

# Analyze shader compilation
export MTL_SHADER_VALIDATION=1
./my_quantum_app
```

## Comparison: CPU vs GPU

### When to Use GPU

| Scenario | Recommendation |
|----------|----------------|
| ≥16 qubits, single gates | GPU |
| ≥14 qubits, two-qubit gates | GPU |
| Deep circuits (>100 layers) | GPU |
| Grover search | GPU |
| Tensor network contractions | GPU |
| Small circuits (<14 qubits) | CPU |
| Quick prototyping | CPU |

### Hybrid CPU/GPU

```c
// Use CPU for small operations, GPU for large
void apply_circuit(quantum_state_t* state, circuit_t* circuit,
                   metal_compute_ctx_t* gpu) {
    for (int i = 0; i < circuit->num_gates; i++) {
        gate_t* g = &circuit->gates[i];

        if (state->num_qubits >= 16 && gpu) {
            // GPU path
            metal_apply_gate(gpu, state, g);
        } else {
            // CPU path (SIMD optimized)
            apply_gate(state, g);
        }
    }
}
```

## Troubleshooting

### Common Issues

**GPU Not Detected**
```
Error: Metal device not available
```
Solution: Ensure running on macOS with Apple Silicon. Check `metal_is_available()`.

**Shader Compilation Failure**
```
Metal: Failed to compile kernel
```
Solution: Check shader source path. Verify Metal SDK version compatibility.

**Performance Regression**
- Check if buffer size exceeds unified memory
- Verify threadgroup size matches hardware
- Profile for memory bandwidth bottleneck

### Debug Mode

```bash
# Enable Metal validation
export MTL_DEBUG_LAYER=1
export MTL_SHADER_VALIDATION=1

./my_quantum_app
```

## See Also

- [Scaling Analysis](scaling-analysis.md)
- [Memory Requirements](memory-requirements.md)
- [Architecture: GPU Pipeline](../architecture/gpu-pipeline.md)
- [API: gpu_metal.h](../api/c/gpu-metal.md)
