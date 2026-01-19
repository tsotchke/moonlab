# Gate Implementation Architecture

How quantum gates are applied to state vectors in Moonlab.

## Overview

Gate application is the core operation in quantum simulation. Moonlab implements gates as matrix-vector operations with extensive optimization for different hardware backends.

## Gate Application Pipeline

```
┌─────────────────┐
│  Gate Request   │
│  (type, qubits) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Gate Dispatch  │
│  Select backend │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│ CPU   │ │ GPU   │
│ Path  │ │ Path  │
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
┌───────────────────┐
│   State Update    │
│   (in-place)      │
└───────────────────┘
```

## Single-Qubit Gate Algorithm

### Mathematical Foundation

For a single-qubit gate $U$ on qubit $k$ in an $n$-qubit system:

$$|\psi'\rangle = (I^{\otimes(n-k-1)} \otimes U \otimes I^{\otimes k}) |\psi\rangle$$

The state vector update:

$$\alpha'_i = \begin{cases}
U_{00} \alpha_i + U_{01} \alpha_{i \oplus 2^k} & \text{if bit } k \text{ of } i \text{ is } 0 \\
U_{10} \alpha_{i \oplus 2^k} + U_{11} \alpha_i & \text{if bit } k \text{ of } i \text{ is } 1
\end{cases}$$

### Implementation

```c
void apply_single_qubit_gate(quantum_state_t* state,
                              int target,
                              const complex_t gate[2][2]) {
    size_t stride = 1ULL << target;
    size_t num_pairs = state->dimension >> 1;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_pairs; i++) {
        // Calculate indices for the pair
        size_t i0 = ((i >> target) << (target + 1)) | (i & (stride - 1));
        size_t i1 = i0 | stride;

        // Load amplitudes
        complex_t a0 = state->amplitudes[i0];
        complex_t a1 = state->amplitudes[i1];

        // Apply 2x2 gate
        state->amplitudes[i0] = gate[0][0] * a0 + gate[0][1] * a1;
        state->amplitudes[i1] = gate[1][0] * a0 + gate[1][1] * a1;
    }
}
```

### Memory Access Pattern

```
State vector: [α₀, α₁, α₂, α₃, α₄, α₅, α₆, α₇, ...]

Gate on qubit 0 (stride=1):
  Pairs: (0,1), (2,3), (4,5), (6,7), ...

Gate on qubit 1 (stride=2):
  Pairs: (0,2), (1,3), (4,6), (5,7), ...

Gate on qubit 2 (stride=4):
  Pairs: (0,4), (1,5), (2,6), (3,7), ...
```

## Two-Qubit Gate Algorithm

### CNOT Implementation

```c
void apply_cnot(quantum_state_t* state, int control, int target) {
    size_t ctrl_stride = 1ULL << control;
    size_t targ_stride = 1ULL << target;

    // Ensure control < target for consistent indexing
    int lo = (control < target) ? control : target;
    int hi = (control < target) ? target : control;

    size_t lo_stride = 1ULL << lo;
    size_t hi_stride = 1ULL << hi;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state->dimension >> 2; i++) {
        // Decode indices
        size_t i00 = ((i >> hi) << (hi + 1)) | (((i >> lo) & ((1 << (hi - lo - 1)) - 1)) << (lo + 1)) | (i & (lo_stride - 1));
        size_t i01 = i00 | targ_stride;
        size_t i10 = i00 | ctrl_stride;
        size_t i11 = i00 | ctrl_stride | targ_stride;

        // CNOT: swap amplitudes where control=1
        complex_t tmp = state->amplitudes[i10];
        state->amplitudes[i10] = state->amplitudes[i11];
        state->amplitudes[i11] = tmp;
    }
}
```

### General Two-Qubit Gate

```c
void apply_two_qubit_gate(quantum_state_t* state,
                           int q0, int q1,
                           const complex_t gate[4][4]) {
    size_t stride0 = 1ULL << q0;
    size_t stride1 = 1ULL << q1;

    int lo = (q0 < q1) ? q0 : q1;
    int hi = (q0 < q1) ? q1 : q0;

    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < state->dimension >> 2; k++) {
        // Calculate base index
        size_t base = ((k >> hi) << (hi + 1)) |
                      (((k >> lo) & ((1 << (hi - lo - 1)) - 1)) << (lo + 1)) |
                      (k & ((1 << lo) - 1));

        // Four indices for the 4x4 subspace
        size_t idx[4] = {
            base,
            base | stride0,
            base | stride1,
            base | stride0 | stride1
        };

        // Reorder if q1 < q0
        if (q1 < q0) {
            size_t tmp = idx[1]; idx[1] = idx[2]; idx[2] = tmp;
        }

        // Load amplitudes
        complex_t a[4];
        for (int j = 0; j < 4; j++) {
            a[j] = state->amplitudes[idx[j]];
        }

        // Apply 4x4 gate
        for (int j = 0; j < 4; j++) {
            state->amplitudes[idx[j]] =
                gate[j][0] * a[0] + gate[j][1] * a[1] +
                gate[j][2] * a[2] + gate[j][3] * a[3];
        }
    }
}
```

## SIMD Optimization

### ARM NEON (Apple Silicon)

```c
#include <arm_neon.h>

void apply_hadamard_neon(quantum_state_t* state, int target) {
    size_t stride = 1ULL << target;
    const double inv_sqrt2 = 0.7071067811865476;

    float64x2_t scale = vdupq_n_f64(inv_sqrt2);

    for (size_t i = 0; i < state->dimension >> 1; i++) {
        size_t i0 = ((i >> target) << (target + 1)) | (i & (stride - 1));
        size_t i1 = i0 | stride;

        // Load as pairs of doubles (real, imag)
        float64x2_t a0 = vld1q_f64((double*)&state->amplitudes[i0]);
        float64x2_t a1 = vld1q_f64((double*)&state->amplitudes[i1]);

        // Hadamard: (a0 + a1) / sqrt(2), (a0 - a1) / sqrt(2)
        float64x2_t sum = vaddq_f64(a0, a1);
        float64x2_t diff = vsubq_f64(a0, a1);

        // Scale and store
        vst1q_f64((double*)&state->amplitudes[i0], vmulq_f64(sum, scale));
        vst1q_f64((double*)&state->amplitudes[i1], vmulq_f64(diff, scale));
    }
}
```

### AVX2 (x86-64)

```c
#include <immintrin.h>

void apply_hadamard_avx2(quantum_state_t* state, int target) {
    size_t stride = 1ULL << target;
    const double inv_sqrt2 = 0.7071067811865476;

    __m256d scale = _mm256_set1_pd(inv_sqrt2);

    // Process 2 pairs at a time (4 complex numbers)
    for (size_t i = 0; i < state->dimension >> 2; i += 2) {
        // Calculate indices for two pairs
        size_t idx0 = ((i >> target) << (target + 1)) | (i & (stride - 1));
        size_t idx1 = idx0 | stride;
        size_t idx2 = (((i+1) >> target) << (target + 1)) | ((i+1) & (stride - 1));
        size_t idx3 = idx2 | stride;

        // Load 4 amplitudes
        __m256d a0 = _mm256_loadu_pd((double*)&state->amplitudes[idx0]);
        __m256d a1 = _mm256_loadu_pd((double*)&state->amplitudes[idx1]);

        // Hadamard transform
        __m256d sum = _mm256_add_pd(a0, a1);
        __m256d diff = _mm256_sub_pd(a0, a1);

        // Scale and store
        _mm256_storeu_pd((double*)&state->amplitudes[idx0],
                         _mm256_mul_pd(sum, scale));
        _mm256_storeu_pd((double*)&state->amplitudes[idx1],
                         _mm256_mul_pd(diff, scale));
    }
}
```

## GPU Implementation (Metal)

### Compute Shader

```metal
#include <metal_stdlib>
using namespace metal;

struct GateParams {
    uint target;
    uint dimension;
    float2 gate[4];  // 2x2 complex matrix as 4 float2
};

kernel void apply_single_gate(
    device float2* amplitudes [[buffer(0)]],
    constant GateParams& params [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    uint stride = 1u << params.target;
    uint num_pairs = params.dimension >> 1;

    if (tid >= num_pairs) return;

    // Calculate pair indices
    uint i0 = ((tid >> params.target) << (params.target + 1)) |
              (tid & (stride - 1));
    uint i1 = i0 | stride;

    // Load amplitudes
    float2 a0 = amplitudes[i0];
    float2 a1 = amplitudes[i1];

    // Complex multiply: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    float2 g00 = params.gate[0];
    float2 g01 = params.gate[1];
    float2 g10 = params.gate[2];
    float2 g11 = params.gate[3];

    float2 new_a0, new_a1;
    new_a0.x = g00.x * a0.x - g00.y * a0.y + g01.x * a1.x - g01.y * a1.y;
    new_a0.y = g00.x * a0.y + g00.y * a0.x + g01.x * a1.y + g01.y * a1.x;
    new_a1.x = g10.x * a0.x - g10.y * a0.y + g11.x * a1.x - g11.y * a1.y;
    new_a1.y = g10.x * a0.y + g10.y * a0.x + g11.x * a1.y + g11.y * a1.x;

    // Store results
    amplitudes[i0] = new_a0;
    amplitudes[i1] = new_a1;
}
```

### Host Code

```objc
- (void)applyGate:(int)target gate:(complex_t[2][2])gate {
    // Set up compute pipeline
    id<MTLComputePipelineState> pipeline = [self pipelineForGate];

    // Encode parameters
    GateParams params = {
        .target = (uint32_t)target,
        .dimension = (uint32_t)self.dimension
    };
    memcpy(params.gate, gate, sizeof(params.gate));

    id<MTLBuffer> paramsBuffer = [self.device newBufferWithBytes:&params
                                                          length:sizeof(params)
                                                         options:MTLResourceStorageModeShared];

    // Dispatch
    id<MTLCommandBuffer> cmdBuffer = [self.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:self.stateBuffer offset:0 atIndex:0];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:1];

    NSUInteger numPairs = self.dimension >> 1;
    MTLSize gridSize = MTLSizeMake(numPairs, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(MIN(256, numPairs), 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];

    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
}
```

## Optimization Strategies

### Gate Fusion

Combine consecutive single-qubit gates on the same qubit:

```c
// Instead of:
apply_gate(state, 0, H);
apply_gate(state, 0, T);
apply_gate(state, 0, H);

// Fuse into single gate:
complex_t fused[2][2];
matrix_multiply(H, T, temp);
matrix_multiply(temp, H, fused);
apply_gate(state, 0, fused);
```

### Cache Optimization

```c
// Ensure stride fits in L1 cache
void apply_gate_cache_optimized(quantum_state_t* state, int target,
                                 const complex_t gate[2][2]) {
    size_t stride = 1ULL << target;

    // Block size based on L1 cache (e.g., 32KB)
    size_t block_size = 2048;  // Pairs per block

    for (size_t block = 0; block < (state->dimension >> 1); block += block_size) {
        size_t end = MIN(block + block_size, state->dimension >> 1);

        for (size_t i = block; i < end; i++) {
            // Process pair (i0, i1)
            // ...
        }
    }
}
```

### Diagonal Gate Specialization

```c
// Z, S, T gates only multiply by phase
void apply_phase_gate(quantum_state_t* state, int target, complex_t phase) {
    size_t stride = 1ULL << target;

    #pragma omp parallel for
    for (size_t i = 0; i < state->dimension; i++) {
        if (i & stride) {  // Bit k is 1
            state->amplitudes[i] *= phase;
        }
    }
}
```

## Performance Comparison

| Gate Type | Naive | SIMD | GPU |
|-----------|-------|------|-----|
| Hadamard (20 qubits) | 15 ms | 4 ms | 0.3 ms |
| CNOT (20 qubits) | 12 ms | 5 ms | 0.4 ms |
| Random U3 (20 qubits) | 18 ms | 6 ms | 0.5 ms |
| Toffoli (20 qubits) | 45 ms | 15 ms | 1.2 ms |

*Benchmarks on Apple M2 Ultra*

## See Also

- [State Vector Engine](state-vector-engine.md)
- [GPU Pipeline](gpu-pipeline.md)
- [SIMD Operations API](../api/c/simd-ops.md)
- [Performance Tuning Guide](../guides/performance-tuning.md)
