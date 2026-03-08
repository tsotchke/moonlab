# Configuration API

Complete reference for simulator configuration in the C library.

**Header**: `src/utils/config.h`

## Overview

The configuration module provides centralized control over all simulator settings, including:

- Compute backend selection (CPU, GPU, distributed)
- SIMD instruction set preferences
- Threading and parallelization
- Memory management
- Noise simulation parameters
- Performance tuning

## Types

### qs_backend_t

Compute backend selection.

```c
typedef enum {
    QS_BACKEND_CPU,          // Pure CPU execution
    QS_BACKEND_GPU_METAL,    // Apple Metal GPU
    QS_BACKEND_GPU_CUDA,     // NVIDIA CUDA
    QS_BACKEND_GPU_VULKAN,   // Vulkan compute
    QS_BACKEND_DISTRIBUTED,  // MPI distributed
    QS_BACKEND_HYBRID,       // GPU + CPU hybrid
    QS_BACKEND_AUTO          // Automatic selection
} qs_backend_t;
```

### qs_simd_level_t

SIMD instruction set level.

```c
typedef enum {
    QS_SIMD_NONE,           // Scalar only
    QS_SIMD_SSE2,           // SSE2 (128-bit)
    QS_SIMD_SSE4,           // SSE4.1/4.2
    QS_SIMD_AVX,            // AVX (256-bit)
    QS_SIMD_AVX2,           // AVX2 + FMA
    QS_SIMD_AVX512,         // AVX-512 (512-bit)
    QS_SIMD_NEON,           // ARM NEON
    QS_SIMD_SVE,            // ARM SVE
    QS_SIMD_AUTO            // Auto-detect best
} qs_simd_level_t;
```

### qs_threading_t

Threading model.

```c
typedef enum {
    QS_THREADING_SINGLE,     // Single-threaded
    QS_THREADING_OPENMP,     // OpenMP parallelization
    QS_THREADING_PTHREADS,   // POSIX threads
    QS_THREADING_GCD,        // Grand Central Dispatch (macOS)
    QS_THREADING_AUTO        // Automatic selection
} qs_threading_t;
```

### qs_precision_t

Numerical precision.

```c
typedef enum {
    QS_PRECISION_SINGLE,     // 32-bit float
    QS_PRECISION_DOUBLE,     // 64-bit double (default)
    QS_PRECISION_QUAD        // 128-bit quad (if available)
} qs_precision_t;
```

### qs_config_t

Main configuration structure.

```c
typedef struct {
    // Compute backend
    qs_backend_t backend;
    qs_simd_level_t simd_level;
    qs_threading_t threading;
    qs_precision_t precision;

    // Parallelization
    int num_threads;              // Thread count (0 = auto)
    int gpu_device_id;            // GPU device index
    int enable_gpu_async;         // Async GPU operations

    // Memory
    size_t max_memory;            // Maximum memory usage (bytes)
    size_t alignment;             // Memory alignment (bytes)
    int use_huge_pages;           // Enable huge pages
    int preallocate_memory;       // Preallocate state memory

    // Simulation parameters
    double tolerance;             // Numerical tolerance
    double normalization_threshold; // When to renormalize
    int auto_normalize;           // Auto-normalize after gates

    // Noise configuration
    int enable_noise;             // Enable noise simulation
    noise_model_t *noise_model;   // Noise model (if enabled)

    // Tensor network settings
    int max_bond_dimension;       // MPS bond dimension limit
    double svd_cutoff;            // SVD truncation threshold
    int use_gpu_for_svd;          // GPU-accelerate SVD

    // Distributed settings
    int mpi_enabled;              // MPI initialization flag
    int nodes_per_gpu;            // Nodes sharing each GPU
    int overlap_comm_compute;     // Overlap communication

    // Debugging
    int verbose;                  // Verbosity level (0-3)
    int validate_operations;      // Validate gate unitarity
    int track_entanglement;       // Track entanglement metrics
    FILE *log_file;               // Log output file
} qs_config_t;
```

## Configuration Management

### qs_config_create

Create configuration with defaults.

```c
qs_config_t* qs_config_create(void);
```

**Returns**: New configuration or NULL on error

**Default Values**:
- Backend: AUTO
- SIMD: AUTO
- Threading: AUTO (uses all cores)
- Precision: DOUBLE
- Tolerance: 1e-12
- Max bond dimension: 256

### qs_config_destroy

Free configuration.

```c
void qs_config_destroy(qs_config_t *config);
```

### qs_config_copy

Create copy of configuration.

```c
qs_config_t* qs_config_copy(const qs_config_t *config);
```

### qs_config_reset

Reset configuration to defaults.

```c
void qs_config_reset(qs_config_t *config);
```

## Environment Variables

### qs_config_load_env

Load configuration from environment variables.

```c
int qs_config_load_env(qs_config_t *config);
```

**Returns**: Number of variables loaded

**Environment Variables**:

| Variable | Type | Description |
|----------|------|-------------|
| `QS_BACKEND` | string | Backend: cpu, metal, cuda, vulkan, distributed, auto |
| `QS_SIMD` | string | SIMD level: none, sse2, sse4, avx, avx2, avx512, neon, auto |
| `QS_THREADS` | int | Number of threads (0 = auto) |
| `QS_GPU_DEVICE` | int | GPU device index |
| `QS_MAX_MEMORY` | size | Maximum memory (e.g., "16G", "4096M") |
| `QS_TOLERANCE` | double | Numerical tolerance |
| `QS_VERBOSE` | int | Verbosity level (0-3) |
| `QS_ENABLE_NOISE` | bool | Enable noise simulation |
| `QS_MAX_BOND_DIM` | int | MPS bond dimension limit |
| `QS_SVD_CUTOFF` | double | SVD truncation threshold |
| `QS_LOG_FILE` | string | Log file path |

**Example**:
```bash
export QS_BACKEND=metal
export QS_THREADS=8
export QS_MAX_MEMORY=32G
export QS_VERBOSE=2
./my_quantum_simulation
```

### qs_config_load_file

Load configuration from file.

```c
int qs_config_load_file(qs_config_t *config, const char *filename);
```

**File Format** (INI-style):
```ini
[backend]
type = metal
gpu_device = 0
enable_async = true

[threading]
model = openmp
num_threads = 8

[memory]
max_memory = 32G
alignment = 64
use_huge_pages = true

[simulation]
precision = double
tolerance = 1e-12
auto_normalize = true

[tensor_network]
max_bond_dimension = 512
svd_cutoff = 1e-10
use_gpu_for_svd = true

[debug]
verbose = 1
validate_operations = false
```

### qs_config_save_file

Save configuration to file.

```c
int qs_config_save_file(const qs_config_t *config, const char *filename);
```

## Presets

### qs_config_preset_performance

Configure for maximum performance.

```c
void qs_config_preset_performance(qs_config_t *config);
```

**Settings**:
- Backend: AUTO (prefers GPU)
- SIMD: AUTO (best available)
- Threading: All cores
- Validation: Disabled
- Tolerance: 1e-10

### qs_config_preset_accuracy

Configure for maximum accuracy.

```c
void qs_config_preset_accuracy(qs_config_t *config);
```

**Settings**:
- Precision: DOUBLE
- Tolerance: 1e-15
- Auto-normalize: Enabled
- Validation: Enabled

### qs_config_preset_low_memory

Configure for memory-constrained systems.

```c
void qs_config_preset_low_memory(qs_config_t *config);
```

**Settings**:
- Threading: Single (reduces overhead)
- Preallocate: Disabled
- Max bond dimension: 64

### qs_config_preset_noisy

Configure for NISQ simulation.

```c
void qs_config_preset_noisy(qs_config_t *config);
```

**Settings**:
- Noise: Enabled
- Default noise model: IBM-like parameters
- Track entanglement: Enabled

### qs_config_preset_distributed

Configure for cluster execution.

```c
void qs_config_preset_distributed(qs_config_t *config);
```

**Settings**:
- Backend: DISTRIBUTED
- MPI: Enabled
- Overlap communication: Enabled

## Backend Configuration

### qs_config_set_backend

Set compute backend.

```c
void qs_config_set_backend(qs_config_t *config, qs_backend_t backend);
```

### qs_config_set_gpu_device

Set GPU device index.

```c
void qs_config_set_gpu_device(qs_config_t *config, int device_id);
```

### qs_config_enable_gpu_async

Enable asynchronous GPU operations.

```c
void qs_config_enable_gpu_async(qs_config_t *config, int enable);
```

### qs_config_set_simd_level

Set SIMD instruction level.

```c
void qs_config_set_simd_level(qs_config_t *config, qs_simd_level_t level);
```

## Threading Configuration

### qs_config_set_threading

Set threading model.

```c
void qs_config_set_threading(qs_config_t *config, qs_threading_t model);
```

### qs_config_set_num_threads

Set number of threads.

```c
void qs_config_set_num_threads(qs_config_t *config, int num_threads);
```

**Parameters**:
- `num_threads`: Thread count (0 = automatic based on CPU cores)

## Memory Configuration

### qs_config_set_max_memory

Set maximum memory limit.

```c
void qs_config_set_max_memory(qs_config_t *config, size_t bytes);
```

### qs_config_set_alignment

Set memory alignment.

```c
void qs_config_set_alignment(qs_config_t *config, size_t alignment);
```

**Recommended**: 64 bytes for AVX-512, 32 for AVX2, 16 for SSE

### qs_config_enable_huge_pages

Enable huge pages for large allocations.

```c
void qs_config_enable_huge_pages(qs_config_t *config, int enable);
```

**Benefit**: Reduced TLB misses for large state vectors

## Simulation Parameters

### qs_config_set_precision

Set numerical precision.

```c
void qs_config_set_precision(qs_config_t *config, qs_precision_t precision);
```

### qs_config_set_tolerance

Set numerical tolerance.

```c
void qs_config_set_tolerance(qs_config_t *config, double tolerance);
```

**Use**: Gate validation, normalization threshold

### qs_config_enable_auto_normalize

Enable automatic state normalization.

```c
void qs_config_enable_auto_normalize(qs_config_t *config, int enable);
```

## Noise Configuration

### qs_config_enable_noise

Enable noise simulation.

```c
void qs_config_enable_noise(qs_config_t *config, int enable);
```

### qs_config_set_noise_model

Set noise model.

```c
void qs_config_set_noise_model(
    qs_config_t *config,
    noise_model_t *model
);
```

**Note**: Configuration takes ownership of noise model

## Tensor Network Configuration

### qs_config_set_max_bond_dimension

Set maximum MPS bond dimension.

```c
void qs_config_set_max_bond_dimension(qs_config_t *config, int chi);
```

**Impact**: Larger χ = more accurate, more memory

### qs_config_set_svd_cutoff

Set SVD truncation threshold.

```c
void qs_config_set_svd_cutoff(qs_config_t *config, double cutoff);
```

**Typical Values**: 1e-10 to 1e-14

### qs_config_enable_gpu_svd

Enable GPU-accelerated SVD.

```c
void qs_config_enable_gpu_svd(qs_config_t *config, int enable);
```

## Distributed Configuration

### qs_config_enable_mpi

Enable MPI distributed computing.

```c
void qs_config_enable_mpi(qs_config_t *config, int enable);
```

### qs_config_enable_comm_overlap

Enable computation-communication overlap.

```c
void qs_config_enable_comm_overlap(qs_config_t *config, int enable);
```

## Debugging Configuration

### qs_config_set_verbose

Set verbosity level.

```c
void qs_config_set_verbose(qs_config_t *config, int level);
```

**Levels**:
- 0: Silent (errors only)
- 1: Basic progress
- 2: Detailed progress
- 3: Debug output

### qs_config_enable_validation

Enable operation validation.

```c
void qs_config_enable_validation(qs_config_t *config, int enable);
```

**Validates**: Gate unitarity, state normalization

### qs_config_set_log_file

Set log output file.

```c
void qs_config_set_log_file(qs_config_t *config, FILE *file);
```

## Query Functions

### qs_config_get_effective_backend

Get actual backend after AUTO resolution.

```c
qs_backend_t qs_config_get_effective_backend(const qs_config_t *config);
```

### qs_config_get_effective_simd

Get actual SIMD level after AUTO resolution.

```c
qs_simd_level_t qs_config_get_effective_simd(const qs_config_t *config);
```

### qs_config_get_effective_threads

Get actual thread count after AUTO resolution.

```c
int qs_config_get_effective_threads(const qs_config_t *config);
```

### qs_config_estimate_memory

Estimate memory usage for given qubit count.

```c
size_t qs_config_estimate_memory(
    const qs_config_t *config,
    int num_qubits
);
```

**Returns**: Estimated bytes required

### qs_config_max_qubits

Calculate maximum simulatable qubits.

```c
int qs_config_max_qubits(const qs_config_t *config);
```

**Returns**: Maximum qubits given memory limit

## Validation

### qs_config_validate

Validate configuration consistency.

```c
int qs_config_validate(const qs_config_t *config, char *error_msg, size_t msg_size);
```

**Returns**: 1 if valid, 0 if invalid

**Checks**:
- Backend availability
- Memory sufficiency
- Parameter ranges
- Conflicting options

### qs_config_print

Print configuration summary.

```c
void qs_config_print(const qs_config_t *config);
```

### qs_config_to_string

Get configuration as string.

```c
char* qs_config_to_string(const qs_config_t *config);
```

**Returns**: Allocated string (caller must free)

## Global Configuration

### qs_set_global_config

Set global default configuration.

```c
void qs_set_global_config(const qs_config_t *config);
```

### qs_get_global_config

Get global default configuration.

```c
const qs_config_t* qs_get_global_config(void);
```

## Complete Example

```c
#include "src/utils/config.h"
#include "src/quantum/state.h"
#include <stdio.h>

int main(int argc, char **argv) {
    // Create configuration
    qs_config_t *config = qs_config_create();

    // Load from environment (overrides defaults)
    qs_config_load_env(config);

    // Apply performance preset
    qs_config_preset_performance(config);

    // Customize specific settings
    qs_config_set_backend(config, QS_BACKEND_GPU_METAL);
    qs_config_set_num_threads(config, 8);
    qs_config_set_max_memory(config, 16ULL * 1024 * 1024 * 1024);  // 16GB
    qs_config_set_verbose(config, 1);

    // Tensor network settings
    qs_config_set_max_bond_dimension(config, 256);
    qs_config_set_svd_cutoff(config, 1e-12);
    qs_config_enable_gpu_svd(config, 1);

    // Validate configuration
    char error_msg[256];
    if (!qs_config_validate(config, error_msg, sizeof(error_msg))) {
        fprintf(stderr, "Invalid configuration: %s\n", error_msg);
        qs_config_destroy(config);
        return 1;
    }

    // Print configuration summary
    printf("Configuration:\n");
    qs_config_print(config);

    // Check capabilities
    printf("\nCapabilities:\n");
    printf("  Effective backend: %d\n",
           qs_config_get_effective_backend(config));
    printf("  Effective SIMD: %d\n",
           qs_config_get_effective_simd(config));
    printf("  Effective threads: %d\n",
           qs_config_get_effective_threads(config));
    printf("  Max qubits: %d\n",
           qs_config_max_qubits(config));

    // Estimate memory for 25 qubits
    size_t mem_25 = qs_config_estimate_memory(config, 25);
    printf("  Memory for 25 qubits: %.2f GB\n", mem_25 / 1e9);

    // Set as global default
    qs_set_global_config(config);

    // Now create states - they'll use global config
    quantum_state_t state;
    quantum_state_init(&state, 20);

    // ... simulation code ...

    quantum_state_free(&state);
    qs_config_destroy(config);

    return 0;
}
```

## Configuration File Example

```ini
# moonlab.conf - Quantum Simulator Configuration

[backend]
# Options: cpu, metal, cuda, vulkan, distributed, hybrid, auto
type = metal
gpu_device = 0
enable_async = true

[simd]
# Options: none, sse2, sse4, avx, avx2, avx512, neon, sve, auto
level = auto

[threading]
# Options: single, openmp, pthreads, gcd, auto
model = openmp
num_threads = 0  # 0 = auto-detect

[memory]
max_memory = 32G
alignment = 64
use_huge_pages = false
preallocate = true

[precision]
# Options: single, double, quad
type = double
tolerance = 1e-12
auto_normalize = true

[noise]
enabled = false
# See noise model configuration separately

[tensor_network]
max_bond_dimension = 256
svd_cutoff = 1e-10
use_gpu_for_svd = true

[distributed]
mpi_enabled = false
overlap_comm_compute = true
nodes_per_gpu = 1

[debug]
verbose = 1
validate_operations = false
track_entanglement = false
log_file = /dev/null
```

## See Also

- [Noise API](noise.md) - Noise model configuration
- [GPU Metal API](gpu-metal.md) - GPU backend details
- [MPI Bridge API](mpi-bridge.md) - Distributed configuration
- [Guides: Performance Tuning](../../guides/performance-tuning.md)
