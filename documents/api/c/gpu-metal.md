# Metal GPU API

Complete reference for Metal GPU acceleration on Apple Silicon in the C library.

**Header**: `src/optimization/gpu_metal.h`

## Overview

The Metal API provides C interface to Apple's Metal compute pipeline for quantum operations, achieving significant speedups on M-series processors:

| Processor | GPU Cores | Unified Memory | Performance |
|-----------|-----------|----------------|-------------|
| M1 | 7-8 | Up to 16GB | 20-40x CPU |
| M1 Pro/Max | 16-32 | Up to 64GB | 30-60x CPU |
| M2 | 8-10 | Up to 24GB | 25-50x CPU |
| M2 Pro/Max/Ultra | 19-76 | Up to 192GB | 40-100x CPU |
| M3 | 10 | Up to 24GB | 30-60x CPU |
| M3 Pro/Max | 14-40 | Up to 128GB | 50-80x CPU |
| M4 | 10+ | Up to 24GB+ | 40-80x CPU |

**Key Features**:
- Zero-copy unified memory architecture (MTLResourceStorageModeShared)
- Auto-detection of GPU core count for optimal threadgroup sizing
- Fused kernel operations to minimize dispatch overhead
- MPS/tensor network acceleration

## Types

### metal_compute_ctx_t

Opaque handle to Metal compute context.

```c
typedef struct metal_compute_ctx metal_compute_ctx_t;
```

### metal_buffer_t

Metal buffer handle for GPU memory.

```c
typedef struct metal_buffer metal_buffer_t;
```

## Initialization and Cleanup

### metal_compute_init

Initialize Metal compute context.

```c
metal_compute_ctx_t* metal_compute_init(void);
```

**Returns**: Metal compute context or NULL on failure

Creates Metal device, command queue, and compiles compute pipeline.

### metal_compute_free

Free Metal compute context.

```c
void metal_compute_free(metal_compute_ctx_t* ctx);
```

### metal_is_available

Check if Metal is available.

```c
int metal_is_available(void);
```

**Returns**: 1 if Metal is available, 0 otherwise

### metal_get_device_info

Get GPU device information.

```c
void metal_get_device_info(
    metal_compute_ctx_t* ctx,
    char* name,
    uint32_t* max_threads,
    uint32_t* num_cores
);
```

**Parameters**:
- `name`: Output buffer for device name (min 256 bytes)
- `max_threads`: Output: max threads per threadgroup
- `num_cores`: Output: number of GPU cores

## Memory Management

### metal_buffer_create

Allocate Metal buffer with zero-copy shared storage.

```c
metal_buffer_t* metal_buffer_create(metal_compute_ctx_t* ctx, size_t size);
```

**Parameters**:
- `ctx`: Metal compute context
- `size`: Buffer size in bytes

**Returns**: Metal buffer handle or NULL on failure

Uses `MTLResourceStorageModeShared` for unified memory access - CPU and GPU access the same memory without copying.

### metal_buffer_create_from_data

Create Metal buffer from existing CPU memory (zero-copy).

```c
metal_buffer_t* metal_buffer_create_from_data(
    metal_compute_ctx_t* ctx,
    void* data,
    size_t size
);
```

### metal_buffer_contents

Get CPU-accessible pointer to Metal buffer.

```c
void* metal_buffer_contents(metal_buffer_t* buffer);
```

### metal_buffer_free

Free Metal buffer.

```c
void metal_buffer_free(metal_buffer_t* buffer);
```

## Quantum Gate Operations

### metal_hadamard

GPU-accelerated Hadamard gate.

```c
int metal_hadamard(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint32_t state_dim
);
```

**Performance**: 20-40x faster than CPU

**Parameters**:
- `ctx`: Metal compute context
- `amplitudes`: Amplitude buffer
- `qubit_index`: Index of qubit to apply gate to
- `state_dim`: Number of amplitudes ($2^n$)

**Returns**: 0 on success, -1 on error

### metal_hadamard_all

GPU-accelerated Hadamard on all qubits in single dispatch.

```c
int metal_hadamard_all(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    uint32_t num_qubits,
    uint32_t state_dim
);
```

### metal_pauli_x

GPU-accelerated Pauli X gate.

```c
int metal_pauli_x(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint32_t state_dim
);
```

### metal_pauli_z

GPU-accelerated Pauli Z gate.

```c
int metal_pauli_z(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    uint32_t qubit_index,
    uint32_t state_dim
);
```

### metal_oracle

GPU-accelerated oracle (phase flip on target state).

```c
int metal_oracle(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    uint32_t target_state,
    uint32_t state_dim
);
```

**Performance**: 50-100x faster than CPU

### metal_oracle_multi

GPU-accelerated oracle with multiple marked states.

```c
int metal_oracle_multi(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    const uint32_t* marked_states,
    uint32_t num_marked,
    uint32_t state_dim
);
```

### metal_grover_diffusion

GPU-accelerated Grover diffusion operator.

```c
int metal_grover_diffusion(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    uint32_t num_qubits,
    uint32_t state_dim
);
```

**Performance**: 15-30x faster than CPU

Fused implementation: Hadamard → Inversion about mean → Hadamard

## Probability and Measurement

### metal_compute_probabilities

Compute probabilities from amplitudes.

```c
int metal_compute_probabilities(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    metal_buffer_t* probabilities,
    uint32_t state_dim
);
```

**Performance**: 30-50x faster than CPU

### metal_normalize

Normalize quantum state.

```c
int metal_normalize(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    float norm,
    uint32_t state_dim
);
```

## Batch Operations

### metal_grover_iteration

Execute complete Grover iteration (Oracle + Diffusion) in single dispatch.

```c
int metal_grover_iteration(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    uint32_t target_state,
    uint32_t num_qubits,
    uint32_t state_dim
);
```

Minimizes CPU↔GPU synchronization by fusing operations.

### metal_grover_search

Execute multiple Grover iterations on GPU.

```c
int metal_grover_search(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* amplitudes,
    uint32_t target_state,
    uint32_t num_qubits,
    uint32_t state_dim,
    uint32_t num_iterations
);
```

### metal_grover_batch_search

Execute MULTIPLE complete Grover searches in parallel.

```c
int metal_grover_batch_search(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* batch_states,
    const uint32_t* targets,
    uint32_t* results,
    uint32_t num_searches,
    uint32_t num_qubits,
    uint32_t num_iterations
);
```

**Parameters**:
- `batch_states`: Buffer for all quantum states (num_searches × state_dim)
- `targets`: Array of target states (one per search)
- `results`: Output array of found states
- `num_searches`: Number of parallel searches (≤76 optimal for M2 Ultra)
- `num_qubits`: Qubits per search
- `num_iterations`: Grover iterations per search

**Performance**:
- 76 searches in ~150ms (vs ~15,000ms on CPU)
- **100x+ speedup** for batch workloads!

## Tensor Network / MPS Operations

### metal_mps_contract_2site

Contract two adjacent MPS tensors into theta tensor.

```c
int metal_mps_contract_2site(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* A,
    metal_buffer_t* B,
    metal_buffer_t* theta,
    uint32_t chi_l,
    uint32_t chi_m,
    uint32_t chi_r
);
```

**Formula**: $\theta_{l,p_1,p_2,r} = \sum_m A_{l,p_1,m} \cdot B_{m,p_2,r}$

**Performance**: 25-40x speedup for $\chi > 32$

### metal_mps_apply_gate_theta

Apply 4×4 gate matrix to theta tensor in-place.

```c
int metal_mps_apply_gate_theta(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* theta,
    metal_buffer_t* gate,
    uint32_t chi_l,
    uint32_t chi_r
);
```

**Formula**: $\theta'_{l,p',r} = \sum_p G_{p',p} \cdot \theta_{l,p,r}$

**Performance**: 15-25x speedup

### metal_mps_apply_gate_2q

Complete 2-qubit gate application to MPS (TEBD step).

```c
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
);
```

Performs:
1. Contract A, B → theta
2. Apply gate to theta
3. SVD truncate: theta → A' S B'
4. Absorb S into A' or B'

**Performance**: 20-35x speedup

### metal_svd_truncate

GPU SVD with truncation using Jacobi iteration.

```c
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
);
```

**Performance**: 20-30x speedup for matrices > 64×64

### metal_mps_expectation_z

Compute $\langle Z_i \rangle$ expectation using transfer matrix method.

```c
double metal_mps_expectation_z(
    metal_compute_ctx_t* ctx,
    metal_buffer_t** mps_tensors,
    const uint32_t* bond_dims,
    uint32_t num_sites,
    uint32_t site
);
```

**Performance**: 30-40x speedup for chains > 20 sites

### metal_mps_expectation_zz

Compute $\langle Z_i Z_j \rangle$ two-point correlation.

```c
double metal_mps_expectation_zz(
    metal_compute_ctx_t* ctx,
    metal_buffer_t** mps_tensors,
    const uint32_t* bond_dims,
    uint32_t num_sites,
    uint32_t site_i,
    uint32_t site_j
);
```

### metal_tensor_norm_squared

Compute tensor squared Frobenius norm.

```c
double metal_tensor_norm_squared(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* tensor,
    uint32_t size
);
```

### metal_tensor_scale

Scale tensor by constant factor.

```c
int metal_tensor_scale(
    metal_compute_ctx_t* ctx,
    metal_buffer_t* tensor,
    uint32_t size,
    double scale
);
```

## Synchronization and Utilities

### metal_wait_completion

Wait for GPU operations to complete.

```c
void metal_wait_completion(metal_compute_ctx_t* ctx);
```

### metal_get_last_execution_time

Get GPU execution time for last operation.

```c
double metal_get_last_execution_time(metal_compute_ctx_t* ctx);
```

**Returns**: Execution time in seconds

### metal_set_performance_monitoring

Enable/disable performance monitoring.

```c
void metal_set_performance_monitoring(metal_compute_ctx_t* ctx, int enable);
```

### metal_print_device_info

Print Metal device capabilities.

```c
void metal_print_device_info(metal_compute_ctx_t* ctx);
```

### metal_get_error

Get error message for last error.

```c
const char* metal_get_error(metal_compute_ctx_t* ctx);
```

## Complete Example

```c
#include "src/optimization/gpu_metal.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    // Check Metal availability
    if (!metal_is_available()) {
        printf("Metal not available\n");
        return 1;
    }

    // Initialize Metal
    metal_compute_ctx_t* ctx = metal_compute_init();
    if (!ctx) {
        printf("Failed to initialize Metal\n");
        return 1;
    }

    // Print device info
    metal_print_device_info(ctx);

    // Setup Grover search (8 qubits = 256 states)
    uint32_t num_qubits = 8;
    uint32_t state_dim = 1 << num_qubits;
    uint32_t target = 42;
    uint32_t iterations = (uint32_t)(M_PI / 4.0 * sqrt(state_dim));

    // Allocate GPU buffer for amplitudes
    size_t buffer_size = state_dim * sizeof(double _Complex);
    metal_buffer_t* amplitudes = metal_buffer_create(ctx, buffer_size);

    // Initialize to |0⟩
    double _Complex* data = metal_buffer_contents(amplitudes);
    data[0] = 1.0;

    // Apply Hadamard to all qubits (uniform superposition)
    metal_hadamard_all(ctx, amplitudes, num_qubits, state_dim);

    // Enable performance monitoring
    metal_set_performance_monitoring(ctx, 1);

    // Run Grover iterations
    for (uint32_t i = 0; i < iterations; i++) {
        metal_grover_iteration(ctx, amplitudes, target, num_qubits, state_dim);
    }

    // Wait for completion
    metal_wait_completion(ctx);

    // Print timing
    double gpu_time = metal_get_last_execution_time(ctx);
    printf("GPU execution time: %.3f ms\n", gpu_time * 1000);

    // Check result (read probabilities)
    double* probs = malloc(state_dim * sizeof(double));
    metal_buffer_t* prob_buf = metal_buffer_create(ctx, state_dim * sizeof(double));
    metal_compute_probabilities(ctx, amplitudes, prob_buf, state_dim);

    double* prob_data = metal_buffer_contents(prob_buf);
    printf("Probability of target %u: %.4f\n", target, prob_data[target]);

    // Cleanup
    free(probs);
    metal_buffer_free(prob_buf);
    metal_buffer_free(amplitudes);
    metal_compute_free(ctx);

    return 0;
}
```

## Performance Guidelines

### Optimal Workload Sizes

| Operation | Optimal Size | Expected Speedup |
|-----------|-------------|------------------|
| Single gate | 16+ qubits | 20-40x |
| Grover search | 12+ qubits | 30-50x |
| Batch Grover | 40-76 searches | 100x+ |
| MPS contraction | $\chi > 32$ | 25-40x |
| SVD | 64×64+ matrices | 20-30x |

### Best Practices

1. **Batch operations**: Use batch APIs when possible
2. **Minimize transfers**: Keep data on GPU between operations
3. **Use fused kernels**: Prefer `metal_grover_iteration` over separate calls
4. **Optimal threadgroups**: Library auto-detects based on GPU core count

## See Also

- [Tensor Network API](tensor-network.md) - MPS operations
- [SIMD Operations](simd-ops.md) - CPU vectorization
- [Guides: GPU Acceleration](../../guides/gpu-acceleration.md) - Setup guide
