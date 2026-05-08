# Configuration Options Reference

Complete reference for all Moonlab configuration settings.

## Overview

Moonlab can be configured through:

1. **Runtime API** - `configure()` function
2. **Environment variables** - `MOONLAB_*` prefix
3. **Configuration file** - `~/.moonlab/config.toml`

Priority: API > Environment > Config File > Defaults

## Configuration Methods

### Python API

```python
from moonlab import configure, get_config

# Set options
configure(
    backend='auto',
    gpu_threshold=18,
    num_threads=8,
    precision='double'
)

# Get current configuration
config = get_config()
print(config)
```

### C API

```c
#include "config.h"

// Set options
qsim_config_set_string("backend", "auto");
qsim_config_set_int("gpu.threshold", 18);
qsim_config_set_int("cpu.threads", 8);

// Get options
const char* backend = qsim_config_get_string("backend");
int threshold = qsim_config_get_int("gpu.threshold");
```

### Environment Variables

```bash
export MOONLAB_BACKEND=metal
export MOONLAB_GPU_THRESHOLD=20
export MOONLAB_NUM_THREADS=8
export MOONLAB_LOG_LEVEL=DEBUG
```

### Configuration File

`~/.moonlab/config.toml`:

```toml
[backend]
default = "auto"
gpu_threshold = 18

[cpu]
num_threads = 8
simd_level = "auto"

[gpu]
enabled = true
max_memory_gb = 8.0

[precision]
default = "double"

[logging]
level = "INFO"
```

## Backend Configuration

### `backend`

Select the compute backend.

| Option | Description |
|--------|-------------|
| `"cpu"` | CPU only (SIMD optimized) |
| `"metal"` | Apple Metal GPU |
| `"auto"` | Automatic selection based on system and problem size |

**Default**: `"auto"`

```python
configure(backend='metal')  # Force GPU
configure(backend='cpu')    # Force CPU
configure(backend='auto')   # Let system decide
```

**Environment**: `MOONLAB_BACKEND`

### `gpu_threshold`

Minimum qubits for automatic GPU selection.

| Type | Range | Default |
|------|-------|---------|
| Integer | 1-32 | 18 |

GPU typically provides benefit for 18+ qubits due to kernel launch overhead.

```python
configure(gpu_threshold=16)  # Use GPU earlier
configure(gpu_threshold=22)  # Use GPU later
```

**Environment**: `MOONLAB_GPU_THRESHOLD`

## CPU Configuration

### `num_threads`

Number of CPU threads for parallel operations.

| Type | Range | Default |
|------|-------|---------|
| Integer | 1-256 | Auto (system core count) |

```python
configure(num_threads=4)   # Use 4 threads
configure(num_threads=0)   # Auto-detect
```

**Environment**: `MOONLAB_NUM_THREADS` or `OMP_NUM_THREADS`

### `simd_level`

SIMD instruction set level.

| Option | Description |
|--------|-------------|
| `"auto"` | Auto-detect best available |
| `"scalar"` | No SIMD (debugging) |
| `"neon"` | ARM NEON |
| `"sse4"` | Intel SSE4.2 |
| `"avx2"` | Intel AVX2 |
| `"avx512"` | Intel AVX-512 |

**Default**: `"auto"`

```python
configure(simd_level='avx2')
```

**Environment**: `MOONLAB_SIMD_LEVEL`

### `cache_block_size`

Memory block size for cache optimization (bytes).

| Type | Range | Default |
|------|-------|---------|
| Integer | 1024-1048576 | 262144 (256 KB) |

```python
configure(cache_block_size=128 * 1024)  # 128 KB blocks
```

## GPU Configuration

### `gpu_enabled`

Enable/disable GPU acceleration.

| Type | Default |
|------|---------|
| Boolean | True (if available) |

```python
configure(gpu_enabled=False)  # Disable GPU
```

### `gpu_max_memory_gb`

Maximum GPU memory usage in gigabytes.

| Type | Range | Default |
|------|-------|---------|
| Float | 0.1-∞ | Unlimited |

```python
configure(gpu_max_memory_gb=8.0)  # Limit to 8 GB
```

### `gpu_threadgroup_size`

Metal compute threadgroup size.

| Type | Range | Default |
|------|-------|---------|
| Integer | 32-1024 | 256 |

Must be power of 2. Optimal value depends on GPU.

```python
configure(gpu_threadgroup_size=512)
```

### `gpu_kernel_fusion`

Enable automatic kernel fusion for sequential operations.

| Type | Default |
|------|---------|
| Boolean | True |

```python
configure(gpu_kernel_fusion=True)
```

### `gpu_verify`

Enable GPU result verification against CPU.

| Type | Default |
|------|---------|
| Boolean | False |

Useful for debugging. Significant performance penalty.

```python
configure(gpu_verify=True)  # Compare results with CPU
```

## Precision Configuration

### `precision`

Floating-point precision for state vector.

| Option | Memory per amplitude | Description |
|--------|---------------------|-------------|
| `"single"` | 8 bytes | 32-bit float complex |
| `"double"` | 16 bytes | 64-bit double complex |

**Default**: `"double"`

```python
configure(precision='single')  # Half memory, ~6-7 decimal digits
configure(precision='double')  # Full precision, ~15-16 decimal digits
```

**Environment**: `MOONLAB_PRECISION`

## Memory Configuration

### `memory_efficient`

Enable memory-efficient mode (slower but less memory).

| Type | Default |
|------|---------|
| Boolean | False |

```python
configure(memory_efficient=True)
```

### `auto_normalize`

Automatically normalize state after operations.

| Type | Default |
|------|---------|
| Boolean | False |

```python
configure(auto_normalize=True)
```

### `memory_pool_size_mb`

Size of pre-allocated memory pool.

| Type | Range | Default |
|------|-------|---------|
| Integer | 0-65536 | 0 (disabled) |

```python
configure(memory_pool_size_mb=1024)  # 1 GB pool
```

## Algorithm Configuration

### `default_shots`

Default number of measurement shots.

| Type | Range | Default |
|------|-------|---------|
| Integer | 1-10000000 | 1000 |

```python
configure(default_shots=10000)
```

### `parallel_hamiltonian`

Parallelize Hamiltonian term evaluation.

| Type | Default |
|------|---------|
| Boolean | True |

```python
configure(parallel_hamiltonian=True)
```

### `dmrg_svd_method`

SVD algorithm for DMRG.

| Option | Description |
|--------|-------------|
| `"gesdd"` | Divide and conquer (fastest) |
| `"gesvd"` | Standard algorithm (more stable) |

**Default**: `"gesdd"`

```python
configure(dmrg_svd_method='gesvd')
```

### `dmrg_cutoff`

SVD truncation threshold for DMRG.

| Type | Range | Default |
|------|-------|---------|
| Float | 1e-16 to 1e-1 | 1e-10 |

```python
configure(dmrg_cutoff=1e-8)
```

## Logging Configuration

### `log_level`

Logging verbosity level.

| Option | Description |
|--------|-------------|
| `"OFF"` | No logging |
| `"ERROR"` | Errors only |
| `"WARN"` | Warnings and errors |
| `"INFO"` | General information |
| `"DEBUG"` | Detailed debug info |
| `"TRACE"` | Very detailed tracing |

**Default**: `"INFO"`

```python
configure(log_level='DEBUG')
```

**Environment**: `MOONLAB_LOG_LEVEL`

### `log_file`

File path for log output.

| Type | Default |
|------|---------|
| String | None (stderr) |

```python
configure(log_file='/path/to/moonlab.log')
```

**Environment**: `MOONLAB_LOG_FILE`

### `error_verbosity`

Error message detail level.

| Option | Description |
|--------|-------------|
| `"minimal"` | Short error message |
| `"normal"` | Standard error message |
| `"detailed"` | Include stack trace and suggestions |

**Default**: `"normal"`

```python
configure(error_verbosity='detailed')
```

## Random Number Configuration

### `seed`

Random seed for reproducibility.

| Type | Range | Default |
|------|-------|---------|
| Integer | 0-2^64 | Random |

```python
configure(seed=42)  # Reproducible results
```

### `rng_source`

Random number generator source.

| Option | Description |
|--------|-------------|
| `"hardware"` | Hardware RNG (RDRAND, etc.) |
| `"urandom"` | OS /dev/urandom |
| `"mersenne"` | Mersenne Twister PRNG |

**Default**: `"hardware"` (if available)

```python
configure(rng_source='hardware')
```

## MPI Configuration

### `mpi_enabled`

Enable MPI distributed computing.

| Type | Default |
|------|---------|
| Boolean | False |

```python
configure(mpi_enabled=True)
```

### `mpi_partition_strategy`

State vector partitioning strategy.

| Option | Description |
|--------|-------------|
| `"qubit"` | Partition by qubit index |
| `"amplitude"` | Partition by amplitude index |
| `"hybrid"` | Adaptive partitioning |

**Default**: `"qubit"`

```python
configure(mpi_partition_strategy='hybrid')
```

## Complete Configuration Example

### Python

```python
from moonlab import configure

configure(
    # Backend
    backend='auto',
    gpu_threshold=18,

    # CPU
    num_threads=8,
    simd_level='auto',
    cache_block_size=256 * 1024,

    # GPU
    gpu_enabled=True,
    gpu_max_memory_gb=16.0,
    gpu_threadgroup_size=256,
    gpu_kernel_fusion=True,

    # Precision
    precision='double',

    # Memory
    memory_efficient=False,
    auto_normalize=False,

    # Algorithms
    default_shots=1000,
    parallel_hamiltonian=True,
    dmrg_svd_method='gesdd',
    dmrg_cutoff=1e-10,

    # Logging
    log_level='INFO',

    # Random
    seed=None,
    rng_source='hardware'
)
```

### Environment Variables

```bash
# .bashrc or .zshrc
export MOONLAB_BACKEND=auto
export MOONLAB_GPU_THRESHOLD=18
export MOONLAB_NUM_THREADS=8
export MOONLAB_PRECISION=double
export MOONLAB_LOG_LEVEL=INFO
```

### Configuration File

`~/.moonlab/config.toml`:

```toml
[backend]
default = "auto"
gpu_threshold = 18

[cpu]
num_threads = 8
simd_level = "auto"
cache_block_size = 262144

[gpu]
enabled = true
max_memory_gb = 16.0
threadgroup_size = 256
kernel_fusion = true
verify = false

[precision]
default = "double"

[memory]
efficient_mode = false
auto_normalize = false
pool_size_mb = 0

[algorithms]
default_shots = 1000
parallel_hamiltonian = true
dmrg_svd_method = "gesdd"
dmrg_cutoff = 1e-10

[logging]
level = "INFO"
file = ""
error_verbosity = "normal"

[random]
seed = 0  # 0 = random
source = "hardware"

[mpi]
enabled = false
partition_strategy = "qubit"
```

## See Also

- [Performance Tuning](../guides/performance-tuning.md) - Optimization guide
- [GPU Acceleration](../guides/gpu-acceleration.md) - GPU configuration
- [Troubleshooting](../troubleshooting.md) - Common issues

