# Archived Moonlab Documentation: Entropy API

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Entropy API

Complete reference for cryptographically secure random number generation in the C library.

**Header**: `src/utils/entropy.h`

## Overview

The entropy module provides high-quality random number generation for quantum measurement sampling and stochastic simulation. Features:

- Multiple entropy sources (hardware RNG, OS, timing jitter, quantum)
- Automatic source selection based on availability
- Entropy pool with continuous mixing
- Quality assessment and health monitoring
- Thread-safe operation

## Entropy Sources

| Source | Quality | Speed | Availability |
|--------|---------|-------|--------------|
| Hardware RNG (RDRAND/RDSEED) | Excellent | Fast | Intel/AMD (2012+) |
| OS Entropy (/dev/urandom, CryptGenRandom) | Excellent | Medium | All platforms |
| Timing Jitter | Good | Slow | All platforms |
| Quantum (hardware) | Excellent | Varies | Specialized hardware |
| Mixed (combined) | Excellent | Medium | Always |

## Types

### entropy_source_type_t

Entropy source selection.

[archived fence delimiter: ```c]
typedef enum {
    ENTROPY_SOURCE_AUTO,      // Automatic best-available selection
    ENTROPY_SOURCE_HARDWARE,  // CPU hardware RNG (RDRAND/RDSEED)
    ENTROPY_SOURCE_OS,        // Operating system entropy
    ENTROPY_SOURCE_JITTER,    // Timing jitter collection
    ENTROPY_SOURCE_MIXED,     // Combined sources with mixing
    ENTROPY_SOURCE_QUANTUM    // Quantum hardware (if available)
} entropy_source_type_t;
[archived fence delimiter: ```]

### entropy_config_t

Entropy source configuration.

[archived fence delimiter: ```c]
typedef struct {
    entropy_source_type_t primary_source;    // Primary entropy source
    entropy_source_type_t fallback_source;   // Fallback if primary fails
    size_t pool_size;                        // Entropy pool size (bytes)
    int reseed_interval;                     // Reseeding frequency
    int quality_threshold;                   // Minimum acceptable quality
    int enable_health_check;                 // Enable continuous testing
} entropy_config_t;
[archived fence delimiter: ```]

### entropy_ctx_t

Entropy context structure.

[archived fence delimiter: ```c]
typedef struct {
    entropy_source_type_t active_source;    // Currently active source

    // Entropy pool
    uint8_t *pool;                          // Entropy pool buffer
    size_t pool_size;                       // Pool capacity
    size_t pool_position;                   // Current position

    // PRNG state (ChaCha20-based)
    uint64_t state[16];                     // Internal PRNG state
    int state_valid;                        // State initialized flag

    // Statistics
    size_t bytes_generated;                 // Total bytes generated
    size_t reseeds;                         // Number of reseeds
    double last_quality;                    // Last quality assessment

    // Hardware detection
    int has_rdrand;                         // RDRAND available
    int has_rdseed;                         // RDSEED available
    int has_quantum;                        // Quantum source available
} entropy_ctx_t;
[archived fence delimiter: ```]

### entropy_quality_t

Entropy quality assessment result.

[archived fence delimiter: ```c]
typedef struct {
    double estimated_entropy;    // Bits of entropy per byte (0-8)
    double chi_square;           // Chi-square uniformity test
    double serial_correlation;   // Serial correlation coefficient
    int passed_monobit;          // NIST monobit test
    int passed_runs;             // NIST runs test
    int passed_poker;            // Poker test
    int overall_quality;         // 0-100 quality score
} entropy_quality_t;
[archived fence delimiter: ```]

## Initialization and Cleanup

### entropy_create

Create entropy context with configuration.

[archived fence delimiter: ```c]
entropy_ctx_t* entropy_create(const entropy_config_t *config);
[archived fence delimiter: ```]

**Parameters**:
- `config`: Configuration (NULL for defaults)

**Returns**: Entropy context or NULL on failure

**Default Configuration**:
- Primary: AUTO (best available)
- Fallback: OS entropy
- Pool size: 4096 bytes
- Health checks enabled

**Example**:
[archived fence delimiter: ```c]
// Default configuration
entropy_ctx_t *entropy = entropy_create(NULL);

// Custom configuration
entropy_config_t config = {
    .primary_source = ENTROPY_SOURCE_HARDWARE,
    .fallback_source = ENTROPY_SOURCE_OS,
    .pool_size = 8192,
    .quality_threshold = 80,
    .enable_health_check = 1
};
entropy_ctx_t *entropy = entropy_create(&config);
[archived fence delimiter: ```]

### entropy_create_default

Create entropy context with default settings.

[archived fence delimiter: ```c]
entropy_ctx_t* entropy_create_default(void);
[archived fence delimiter: ```]

Equivalent to `entropy_create(NULL)`.

### entropy_destroy

Free entropy context and securely wipe state.

[archived fence delimiter: ```c]
void entropy_destroy(entropy_ctx_t *ctx);
[archived fence delimiter: ```]

**Security**: Zeros all internal state before freeing.

### entropy_init

Initialize entropy context in-place.

[archived fence delimiter: ```c]
int entropy_init(
    entropy_ctx_t *ctx,
    const entropy_config_t *config
);
[archived fence delimiter: ```]

**Parameters**:
- `ctx`: Pre-allocated context structure
- `config`: Configuration (NULL for defaults)

**Returns**: 0 on success, -1 on error

### entropy_cleanup

Clean up in-place initialized context.

[archived fence delimiter: ```c]
void entropy_cleanup(entropy_ctx_t *ctx);
[archived fence delimiter: ```]

## Random Number Generation

### entropy_bytes

Generate random bytes.

[archived fence delimiter: ```c]
int entropy_bytes(
    entropy_ctx_t *ctx,
    uint8_t *buffer,
    size_t count
);
[archived fence delimiter: ```]

**Parameters**:
- `ctx`: Entropy context
- `buffer`: Output buffer
- `count`: Number of bytes to generate

**Returns**: 0 on success, -1 on error

**Example**:
[archived fence delimiter: ```c]
uint8_t random_bytes[32];
entropy_bytes(entropy, random_bytes, sizeof(random_bytes));
[archived fence delimiter: ```]

### entropy_uint64

Generate random 64-bit unsigned integer.

[archived fence delimiter: ```c]
uint64_t entropy_uint64(entropy_ctx_t *ctx);
[archived fence delimiter: ```]

**Returns**: Random value in [0, UINT64_MAX]

### entropy_uint32

Generate random 32-bit unsigned integer.

[archived fence delimiter: ```c]
uint32_t entropy_uint32(entropy_ctx_t *ctx);
[archived fence delimiter: ```]

### entropy_double

Generate random double in [0, 1).

[archived fence delimiter: ```c]
double entropy_double(entropy_ctx_t *ctx);
[archived fence delimiter: ```]

**Returns**: Uniformly distributed value in [0, 1)

**Precision**: 53 bits of randomness (full double precision)

**Use Case**: Quantum measurement sampling

### entropy_double_range

Generate random double in [min, max).

[archived fence delimiter: ```c]
double entropy_double_range(
    entropy_ctx_t *ctx,
    double min,
    double max
);
[archived fence delimiter: ```]

### entropy_gaussian

Generate Gaussian-distributed random value.

[archived fence delimiter: ```c]
double entropy_gaussian(
    entropy_ctx_t *ctx,
    double mean,
    double stddev
);
[archived fence delimiter: ```]

**Algorithm**: Box-Muller transform

**Use Case**: Noise simulation, dephasing

### entropy_exponential

Generate exponentially-distributed random value.

[archived fence delimiter: ```c]
double entropy_exponential(
    entropy_ctx_t *ctx,
    double lambda
);
[archived fence delimiter: ```]

**Use Case**: Decay times, waiting times

### entropy_poisson

Generate Poisson-distributed random integer.

[archived fence delimiter: ```c]
uint64_t entropy_poisson(
    entropy_ctx_t *ctx,
    double lambda
);
[archived fence delimiter: ```]

**Use Case**: Photon counting, shot noise

## Quantum-Specific Functions

### entropy_measurement_threshold

Generate threshold for quantum measurement.

[archived fence delimiter: ```c]
double entropy_measurement_threshold(entropy_ctx_t *ctx);
[archived fence delimiter: ```]

**Returns**: Value in [0, 1) for measurement sampling

**Use Case**: Born rule sampling

**Example**:
[archived fence delimiter: ```c]
double threshold = entropy_measurement_threshold(entropy);
double cumulative = 0.0;
uint64_t outcome = 0;

for (size_t i = 0; i < state_dim; i++) {
    cumulative += cabs(amplitudes[i]) * cabs(amplitudes[i]);
    if (threshold < cumulative) {
        outcome = i;
        break;
    }
}
[archived fence delimiter: ```]

### entropy_phase

Generate random phase angle.

[archived fence delimiter: ```c]
double entropy_phase(entropy_ctx_t *ctx);
[archived fence delimiter: ```]

**Returns**: Uniformly distributed angle in [0, 2π)

**Use Case**: Random unitaries, noise channels

### entropy_bloch_sphere

Generate random point on Bloch sphere.

[archived fence delimiter: ```c]
void entropy_bloch_sphere(
    entropy_ctx_t *ctx,
    double *theta,
    double *phi
);
[archived fence delimiter: ```]

**Parameters**:
- `theta`: Output: polar angle [0, π]
- `phi`: Output: azimuthal angle [0, 2π)

**Distribution**: Uniform over sphere surface (Haar measure)

### entropy_unitary_2x2

Generate random 2×2 unitary matrix (Haar-distributed).

[archived fence delimiter: ```c]
void entropy_unitary_2x2(
    entropy_ctx_t *ctx,
    complex_t matrix[4]
);
[archived fence delimiter: ```]

**Distribution**: Haar measure on U(2)

**Use Case**: Random single-qubit gates, randomized benchmarking

### entropy_shuffle

Randomly permute array (Fisher-Yates shuffle).

[archived fence delimiter: ```c]
void entropy_shuffle(
    entropy_ctx_t *ctx,
    void *array,
    size_t element_size,
    size_t count
);
[archived fence delimiter: ```]

**Use Case**: Random circuit generation, measurement order

## Entropy Pool Management

### entropy_reseed

Force reseed from entropy source.

[archived fence delimiter: ```c]
int entropy_reseed(entropy_ctx_t *ctx);
[archived fence delimiter: ```]

**Returns**: 0 on success, -1 on error

**Action**: Collects fresh entropy and mixes into state

### entropy_add_entropy

Add external entropy to pool.

[archived fence delimiter: ```c]
int entropy_add_entropy(
    entropy_ctx_t *ctx,
    const uint8_t *data,
    size_t size,
    size_t estimated_bits
);
[archived fence delimiter: ```]

**Parameters**:
- `data`: Entropy data to add
- `size`: Size in bytes
- `estimated_bits`: Estimated entropy bits (0 if unknown)

**Use Case**: Incorporating external randomness sources

### entropy_stir

Mix entropy pool without adding new entropy.

[archived fence delimiter: ```c]
void entropy_stir(entropy_ctx_t *ctx);
[archived fence delimiter: ```]

**Action**: Cryptographic mixing of pool state

### entropy_get_pool_level

Get current entropy pool level.

[archived fence delimiter: ```c]
size_t entropy_get_pool_level(const entropy_ctx_t *ctx);
[archived fence delimiter: ```]

**Returns**: Estimated bits of entropy in pool

## Quality Assessment

### entropy_assess_quality

Assess quality of entropy source.

[archived fence delimiter: ```c]
entropy_quality_t entropy_assess_quality(
    entropy_ctx_t *ctx,
    size_t sample_size
);
[archived fence delimiter: ```]

**Parameters**:
- `ctx`: Entropy context
- `sample_size`: Bytes to sample for testing (recommended: 10000+)

**Returns**: Quality assessment structure

**Example**:
[archived fence delimiter: ```c]
entropy_quality_t quality = entropy_assess_quality(entropy, 100000);

printf("Entropy: %.2f bits/byte\n", quality.estimated_entropy);
printf("Quality score: %d/100\n", quality.overall_quality);

if (quality.overall_quality < 80) {
    fprintf(stderr, "Warning: Low entropy quality\n");
}
[archived fence delimiter: ```]

### entropy_self_test

Run self-test on entropy source.

[archived fence delimiter: ```c]
int entropy_self_test(entropy_ctx_t *ctx);
[archived fence delimiter: ```]

**Returns**: 1 if passed, 0 if failed

**Tests**:
- Output produces expected distribution
- No stuck-at faults
- Minimum entropy threshold met

### entropy_get_source_info

Get information about active entropy source.

[archived fence delimiter: ```c]
const char* entropy_get_source_info(const entropy_ctx_t *ctx);
[archived fence delimiter: ```]

**Returns**: Human-readable source description

## Hardware Detection

### entropy_has_hardware_rng

Check for hardware RNG support.

[archived fence delimiter: ```c]
int entropy_has_hardware_rng(void);
[archived fence delimiter: ```]

**Returns**: 1 if RDRAND/RDSEED available, 0 otherwise

### entropy_has_rdseed

Check for RDSEED support (true hardware entropy).

[archived fence delimiter: ```c]
int entropy_has_rdseed(void);
[archived fence delimiter: ```]

**Returns**: 1 if RDSEED available, 0 otherwise

**Note**: RDSEED provides true hardware entropy; RDRAND uses a DRBG.

### entropy_detect_sources

Detect all available entropy sources.

[archived fence delimiter: ```c]
int entropy_detect_sources(
    int *sources,
    size_t max_sources
);
[archived fence delimiter: ```]

**Parameters**:
- `sources`: Output array of entropy_source_type_t values
- `max_sources`: Maximum sources to detect

**Returns**: Number of available sources

## Statistics

### entropy_get_stats

Get generation statistics.

[archived fence delimiter: ```c]
void entropy_get_stats(
    const entropy_ctx_t *ctx,
    size_t *bytes_generated,
    size_t *reseeds,
    double *last_quality
);
[archived fence delimiter: ```]

### entropy_reset_stats

Reset statistics counters.

[archived fence delimiter: ```c]
void entropy_reset_stats(entropy_ctx_t *ctx);
[archived fence delimiter: ```]

## Thread Safety

### entropy_create_thread_local

Create thread-local entropy context.

[archived fence delimiter: ```c]
entropy_ctx_t* entropy_create_thread_local(void);
[archived fence delimiter: ```]

**Returns**: Thread-local context (automatically cleaned up)

### entropy_lock

Acquire lock for thread-safe access.

[archived fence delimiter: ```c]
void entropy_lock(entropy_ctx_t *ctx);
[archived fence delimiter: ```]

### entropy_unlock

Release lock.

[archived fence delimiter: ```c]
void entropy_unlock(entropy_ctx_t *ctx);
[archived fence delimiter: ```]

## Complete Example

[archived fence delimiter: ```c]
#include "src/utils/entropy.h"
#include "src/quantum/state.h"
#include <stdio.h>

int main(void) {
    // Create entropy source with hardware RNG preference
    entropy_config_t config = {
        .primary_source = ENTROPY_SOURCE_AUTO,
        .fallback_source = ENTROPY_SOURCE_OS,
        .pool_size = 4096,
        .enable_health_check = 1
    };

    entropy_ctx_t *entropy = entropy_create(&config);
    if (!entropy) {
        fprintf(stderr, "Failed to initialize entropy\n");
        return 1;
    }

    // Check entropy quality
    entropy_quality_t quality = entropy_assess_quality(entropy, 50000);
    printf("Entropy source: %s\n", entropy_get_source_info(entropy));
    printf("Quality score: %d/100\n", quality.overall_quality);
    printf("Estimated entropy: %.2f bits/byte\n", quality.estimated_entropy);

    // Run self-test
    if (!entropy_self_test(entropy)) {
        fprintf(stderr, "Entropy self-test failed!\n");
        entropy_destroy(entropy);
        return 1;
    }

    // Generate random values for quantum simulation
    printf("\nRandom samples:\n");

    // Measurement thresholds
    for (int i = 0; i < 5; i++) {
        double threshold = entropy_measurement_threshold(entropy);
        printf("  Measurement threshold: %.6f\n", threshold);
    }

    // Random phases
    for (int i = 0; i < 5; i++) {
        double phase = entropy_phase(entropy);
        printf("  Random phase: %.4f rad\n", phase);
    }

    // Random Bloch sphere point
    double theta, phi;
    entropy_bloch_sphere(entropy, &theta, &phi);
    printf("\nRandom Bloch sphere point:\n");
    printf("  theta = %.4f, phi = %.4f\n", theta, phi);

    // Random unitary
    complex_t unitary[4];
    entropy_unitary_2x2(entropy, unitary);
    printf("\nRandom 2x2 unitary:\n");
    printf("  [[%.3f+%.3fi, %.3f+%.3fi],\n",
           creal(unitary[0]), cimag(unitary[0]),
           creal(unitary[1]), cimag(unitary[1]));
    printf("   [%.3f+%.3fi, %.3f+%.3fi]]\n",
           creal(unitary[2]), cimag(unitary[2]),
           creal(unitary[3]), cimag(unitary[3]));

    // Get statistics
    size_t bytes, reseeds;
    double last_quality;
    entropy_get_stats(entropy, &bytes, &reseeds, &last_quality);
    printf("\nStatistics:\n");
    printf("  Bytes generated: %zu\n", bytes);
    printf("  Reseeds: %zu\n", reseeds);

    entropy_destroy(entropy);
    return 0;
}
[archived fence delimiter: ```]

## Security Considerations

1. **Entropy Quality**: Always check quality for cryptographic applications
2. **Reseeding**: Pool reseeds automatically; force reseed for high-security needs
3. **State Wiping**: `entropy_destroy()` securely wipes internal state
4. **Thread Safety**: Use thread-local contexts or locking for concurrent access

## See Also

- [Quantum State API](quantum-state.md) - Uses entropy for measurement
- [Noise API](noise.md) - Uses entropy for stochastic channels
- [Grover API](grover.md) - Uses entropy for quantum sampling
```
