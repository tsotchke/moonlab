# SIMD Operations API

Complete reference for SIMD-optimized operations in the C library.

**Header**: `src/optimization/simd_ops.h`

## Overview

The SIMD module provides vectorized implementations of performance-critical operations with automatic runtime CPU feature detection and fallback to scalar implementations.

**Key Optimizations**:
- Complex number arithmetic (SSE2)
- Matrix operations (AVX2 when available)
- Normalization calculations
- Probability computations
- Measurement sampling

## CPU Feature Detection

### simd_capabilities_t

CPU SIMD capabilities structure.

```c
typedef struct {
    int has_sse2;      // SSE2 support (2001+)
    int has_sse3;      // SSE3 support (2004+)
    int has_ssse3;     // SSSE3 support (2006+)
    int has_sse4_1;    // SSE4.1 support (2007+)
    int has_avx;       // AVX support (2011+)
    int has_avx2;      // AVX2 support (2013+)
    int has_fma;       // FMA3 support
    int has_avx512f;   // AVX-512 Foundation (2016+)
    int has_avx512dq;  // AVX-512 DQ extensions
    int has_arm_sve;   // ARM SVE support
} simd_capabilities_t;
```

### simd_detect_capabilities

Detect CPU SIMD capabilities at runtime.

```c
simd_capabilities_t simd_detect_capabilities(void);
```

**Returns**: Structure with detected capabilities

**Example**:
```c
simd_capabilities_t caps = simd_detect_capabilities();
if (caps.has_avx2) {
    printf("AVX2 available - using optimized path\n");
}
```

### simd_capabilities_string

Get CPU capabilities as human-readable string.

```c
const char* simd_capabilities_string(const simd_capabilities_t *caps);
```

**Returns**: String like "SSE2 SSE3 AVX AVX2"

## Vectorized Operations

### simd_sum_squared_magnitudes

Compute sum of squared magnitudes (for normalization).

```c
double simd_sum_squared_magnitudes(const complex_t *amplitudes, size_t n);
```

**Parameters**:
- `amplitudes`: Complex amplitude array
- `n`: Number of amplitudes

**Returns**: $\sum |\alpha_i|^2$

**Implementation**: Uses SSE2/AVX2 when available

### simd_normalize_amplitudes

Normalize complex amplitude array in-place.

```c
void simd_normalize_amplitudes(complex_t *amplitudes, size_t n, double norm);
```

**Parameters**:
- `amplitudes`: Array to normalize (modified in-place)
- `n`: Number of amplitudes
- `norm`: Normalization factor (sqrt of sum squared magnitudes)

**Action**: Divides all amplitudes by `norm`

### simd_matrix2x2_vec_multiply

Matrix-vector multiply for 2×2 complex matrices.

```c
void simd_matrix2x2_vec_multiply(
    const complex_t matrix[4],
    const complex_t input[2],
    complex_t output[2]
);
```

**Use Case**: Optimized for quantum gate operations

**Implementation**: Uses SSE2 for complex arithmetic

### simd_complex_multiply

Vectorized complex multiplication.

```c
complex_t simd_complex_multiply(complex_t z1, complex_t z2);
```

**Returns**: Product $z_1 \times z_2$

### simd_compute_probabilities

Batch compute probabilities from amplitudes.

```c
void simd_compute_probabilities(
    const complex_t *amplitudes,
    double *probabilities,
    size_t n
);
```

**Action**: Computes $|α|^2$ for each amplitude

## Quantum Gate Primitives

### simd_complex_swap

Vectorized complex swap for Pauli X gate.

```c
void simd_complex_swap(complex_t *amp0, complex_t *amp1, size_t n);
```

**Use Case**: Pauli X and CNOT gates

**Action**: Swaps pairs of amplitudes efficiently using SIMD

### simd_multiply_by_i

Vectorized multiply by $±i$ for Pauli Y components.

```c
void simd_multiply_by_i(complex_t *amplitudes, size_t n, int negate);
```

**Parameters**:
- `amplitudes`: Array to modify
- `n`: Number of amplitudes
- `negate`: If true, multiply by $-i$; otherwise by $+i$

### simd_negate

Vectorized negate for Pauli Z gate.

```c
void simd_negate(complex_t *amplitudes, size_t n);
```

**Action**: Negates all amplitudes in array

### simd_apply_phase

Vectorized phase multiplication.

```c
void simd_apply_phase(complex_t *amplitudes, complex_t phase, size_t n);
```

**Use Case**: S, T, and Phase gates

**Action**: Multiplies all amplitudes by phase factor $e^{i\theta}$

## Entropy Mixing Operations

### simd_xor_bytes

XOR mixing of byte arrays.

```c
void simd_xor_bytes(uint8_t *dest, const uint8_t *src, size_t n);
```

**Action**: Performs `dest[i] ^= src[i]` using SSE2/AVX2

### simd_mix_entropy

Fast entropy mixing using SIMD.

```c
void simd_mix_entropy(
    const uint8_t *state,
    const uint8_t *entropy,
    uint8_t *output,
    size_t size
);
```

**Action**: Mixes entropy buffers with cryptographic quality

## Measurement Sampling

### simd_cumulative_probability_search

SIMD-optimized cumulative probability search.

```c
uint64_t simd_cumulative_probability_search(
    const complex_t *amplitudes,
    size_t n,
    double random_threshold
);
```

**Parameters**:
- `amplitudes`: Complex amplitude array
- `n`: Number of amplitudes
- `random_threshold`: Random value in [0, 1) for sampling

**Returns**: Index where cumulative probability exceeds threshold

**Use Case**: Critical for fast quantum measurement sampling

### simd_fast_measurement_sample

Optimized batch measurement sampling.

```c
uint64_t simd_fast_measurement_sample(
    const complex_t *amplitudes,
    size_t n,
    double random_threshold
);
```

**Action**: Pre-computes all probabilities with SIMD, then does fast cumulative search

**Returns**: Sampled index

## Performance Characteristics

### Operation Speedups

| Operation | SSE2 | AVX2 | AVX-512 |
|-----------|------|------|---------|
| Sum squared magnitudes | 2x | 4x | 8x |
| Normalize | 2x | 4x | 8x |
| Compute probabilities | 2x | 4x | 8x |
| Complex swap | 2x | 4x | 8x |
| Phase application | 2x | 4x | 8x |
| Cumulative search | 1.5x | 2x | 3x |

### SIMD Width

| Instruction Set | Vector Width | Complex Elements |
|-----------------|--------------|------------------|
| SSE2 | 128 bits | 1 |
| AVX | 256 bits | 2 |
| AVX2 | 256 bits | 2 |
| AVX-512 | 512 bits | 4 |
| NEON (ARM) | 128 bits | 1 |
| SVE (ARM) | Variable | Variable |

## Complete Example

```c
#include "src/optimization/simd_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    // Detect CPU capabilities
    simd_capabilities_t caps = simd_detect_capabilities();
    printf("CPU features: %s\n", simd_capabilities_string(&caps));

    // Create test amplitudes (uniform superposition for 4 qubits)
    size_t n = 16;
    complex_t *amplitudes = aligned_alloc(64, n * sizeof(complex_t));

    // Initialize to uniform superposition
    double amp = 1.0 / sqrt(n);
    for (size_t i = 0; i < n; i++) {
        amplitudes[i] = amp;
    }

    // Apply phase gate (T gate phase) to subset
    complex_t t_phase = cexp(I * M_PI / 4.0);
    simd_apply_phase(amplitudes + 8, t_phase, 8);  // Apply to states 8-15

    // Compute normalization
    double sum_sq = simd_sum_squared_magnitudes(amplitudes, n);
    printf("Sum squared: %.6f (should be ~1.0)\n", sum_sq);

    // Normalize if needed
    if (fabs(sum_sq - 1.0) > 1e-10) {
        simd_normalize_amplitudes(amplitudes, n, sqrt(sum_sq));
    }

    // Compute probabilities
    double *probs = malloc(n * sizeof(double));
    simd_compute_probabilities(amplitudes, probs, n);

    // Sample measurement
    double random_val = 0.3;  // Simulated random value
    uint64_t result = simd_fast_measurement_sample(amplitudes, n, random_val);
    printf("Measurement result: %llu\n", result);

    // Cleanup
    free(probs);
    free(amplitudes);

    return 0;
}
```

## Usage Guidelines

### When to Use SIMD

1. **Large amplitude arrays**: SIMD overhead is amortized for n > 16
2. **Batch operations**: Processing multiple states or gates
3. **Measurement sampling**: Critical path for quantum simulation
4. **Normalization**: Frequent operation in quantum circuits

### Memory Alignment

For optimal SIMD performance, allocate memory with 64-byte alignment:

```c
// C11 aligned allocation
complex_t *data = aligned_alloc(64, n * sizeof(complex_t));

// Or use posix_memalign
complex_t *data;
posix_memalign((void**)&data, 64, n * sizeof(complex_t));
```

### Fallback Behavior

All SIMD functions automatically fall back to scalar implementations when:
- Required instruction set is not available
- Array size is too small to benefit from vectorization
- Memory is not properly aligned (may use unaligned loads)

## See Also

- [Metal GPU API](gpu-metal.md) - GPU acceleration
- [Config API](config.md) - SIMD configuration
- [Guides: Performance Tuning](../../guides/performance-tuning.md)
