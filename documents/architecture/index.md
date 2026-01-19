# Architecture Documentation

Internal architecture and implementation details of Moonlab.

## Overview

This section provides comprehensive documentation of Moonlab's internal architecture, designed for contributors, integrators, and advanced users who need to understand the system internals.

## System Architecture

Moonlab is structured as a layered system:

```
┌────────────────────────────────────────────────────────────────┐
│                        Language Bindings                       │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐ │
│    │  Python  │    │   Rust   │    │JavaScript│    │  Swift  │ │
│    └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬────┘ │
└─────────┼───────────────┼───────────────┼───────────────┼──────┘
          │               │               │               │
          └───────────────┼───────────────┼───────────────┘
                          │               │
┌─────────────────────────┴───────────────┴───────────────────────┐
│                         C Core Library                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      High-Level API                         ││
│  │   Algorithms (VQE, QAOA, Grover) │ Tensor Networks (DMRG)   ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                     Core Quantum Layer                      ││
│  │  State Vector │ Gates │ Measurement │ Entanglement │ Noise  ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Optimization Layer                        ││
│  │      SIMD Ops    │    GPU Metal    │    MPI Bridge          ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      Utilities                              ││
│  │   Memory Management │ Config │ Entropy │ Profiling          ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Component Overview

### Core Components

| Component | Purpose | Documentation |
|-----------|---------|---------------|
| [State Vector Engine](state-vector-engine.md) | Amplitude storage and manipulation | Full details |
| [GPU Pipeline](gpu-pipeline.md) | Metal acceleration | Full details |
| [Tensor Network Engine](tensor-network-engine.md) | MPS/DMRG implementation | Full details |

### Supporting Systems

| System | Purpose |
|--------|---------|
| SIMD Operations | Vectorized complex arithmetic |
| Memory Management | Aligned allocation, pooling |
| MPI Bridge | Distributed computing |
| Profiler | Performance measurement |

## Design Principles

### 1. Performance First

- All hot paths are SIMD-vectorized
- Memory layouts optimized for cache locality
- GPU offloading for large state vectors
- Zero-copy interfaces where possible

### 2. Memory Safety

- Consistent ownership semantics
- Reference counting for shared resources
- Arena allocators for algorithm temporaries
- Bounds checking in debug builds

### 3. Extensibility

- Plugin architecture for backends
- Configurable precision (float/double)
- Custom gate definitions
- Modular algorithm implementations

### 4. Portability

- Pure C core (C11 standard)
- Platform-specific optimizations isolated
- Runtime feature detection
- Fallback implementations for all optimizations

## Data Flow

### Gate Application

```
User API Call
     │
     ▼
┌────────────────┐
│ Gate Selection │ → Dispatch based on gate type
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ Backend Select │ → Choose CPU, GPU, or distributed
└───────┬────────┘
        │
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   CPU Path    │  │   GPU Path    │  │ Distributed   │
│  (SIMD Ops)   │  │ (Metal/CUDA)  │  │   (MPI)       │
└───────────────┘  └───────────────┘  └───────────────┘
        │                  │                  │
        └──────────────────┴──────────────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │ State Updated  │
                  └────────────────┘
```

### Measurement Flow

```
Measurement Request
        │
        ▼
┌────────────────────┐
│ Probability Compute│ → |amplitude|² for all states
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Random Sampling    │ → Use entropy source
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ State Collapse     │ → Update amplitudes
└─────────┬──────────┘
          │
          ▼
     Return Result
```

## Module Dependencies

```
                    ┌─────────┐
                    │ config  │
                    └────┬────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    ▼                    ▼                    ▼
┌─────────┐        ┌──────────┐         ┌─────────┐
│ entropy │        │ simd_ops │         │  memory │
└────┬────┘        └────┬─────┘         └────┬────┘
     │                  │                    │
     │     ┌────────────┴────────────┐       │
     │     │                         │       │
     ▼     ▼                         ▼       ▼
┌──────────────┐                ┌──────────────┐
│    state     │                │  gpu_metal   │
└──────┬───────┘                └──────┬───────┘
       │                               │
       ▼                               │
┌──────────────┐                       │
│    gates     │◄──────────────────────┘
└──────┬───────┘
       │
       ├─────────────────────┬──────────────────┐
       ▼                     ▼                  ▼
┌─────────────┐       ┌────────────┐     ┌───────────┐
│ measurement │       │entanglement│     │   noise   │
└──────┬──────┘       └─────┬──────┘     └─────┬─────┘
       │                    │                  │
       └────────────────────┼──────────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
┌─────────────┐       ┌──────────┐        ┌──────────┐
│   grover    │       │   vqe    │        │   qaoa   │
└─────────────┘       └──────────┘        └──────────┘
```

## Build Configuration

### Compile-Time Options

| Option | Default | Description |
|--------|---------|-------------|
| `QSIM_DOUBLE_PRECISION` | ON | Use double precision |
| `QSIM_ENABLE_GPU` | AUTO | Enable Metal GPU support |
| `QSIM_ENABLE_MPI` | OFF | Enable distributed computing |
| `QSIM_ENABLE_OPENMP` | AUTO | Enable OpenMP threading |
| `QSIM_DEBUG` | OFF | Enable debug assertions |

### Feature Detection

Runtime feature detection:

```c
qsim_features_t features = qsim_detect_features();

printf("SIMD: %s\n", features.simd_level);     // "NEON", "AVX2", etc.
printf("GPU: %s\n", features.gpu_available ? "yes" : "no");
printf("MPI: %s\n", features.mpi_available ? "yes" : "no");
printf("Threads: %d\n", features.num_threads);
```

## Performance Characteristics

### Memory Usage

| Qubits | State Vector | With GPU Buffer |
|--------|--------------|-----------------|
| 20 | 16 MB | 32 MB |
| 25 | 512 MB | 1 GB |
| 30 | 16 GB | 32 GB |

### Operation Costs

| Operation | Complexity | Typical Time (30 qubits) |
|-----------|------------|--------------------------|
| Single-qubit gate | O(2ⁿ) | ~50 ms (CPU), ~5 ms (GPU) |
| Two-qubit gate | O(2ⁿ) | ~100 ms (CPU), ~8 ms (GPU) |
| Full measurement | O(2ⁿ) | ~80 ms |
| Partial measurement | O(2ⁿ) | ~80 ms |

## Thread Safety

### Safe Operations

- Read-only state access
- Independent state manipulation
- Configuration access

### Unsafe Operations (require synchronization)

- Concurrent gate application to same state
- GPU buffer transfers during computation
- MPI collective operations

```c
// Thread-safe: each thread has own state
#pragma omp parallel
{
    quantum_state_t* local_state = quantum_state_create(20);
    // ... operations on local_state
    quantum_state_destroy(local_state);
}

// NOT thread-safe: shared state
quantum_state_t* shared = quantum_state_create(20);
#pragma omp parallel
{
    // WRONG: concurrent modification
    quantum_state_h(shared, omp_get_thread_num());
}
```

## Error Handling

### Error Codes

All C functions return status codes:

```c
typedef enum {
    QSIM_SUCCESS = 0,
    QSIM_ERROR_NULL_POINTER = 1,
    QSIM_ERROR_INVALID_QUBIT = 2,
    QSIM_ERROR_OUT_OF_MEMORY = 3,
    QSIM_ERROR_GPU_UNAVAILABLE = 4,
    QSIM_ERROR_INVALID_STATE = 5,
    // ... more codes
} qsim_error_t;
```

### Error Handling Pattern

```c
qsim_error_t err = quantum_state_h(state, qubit);
if (err != QSIM_SUCCESS) {
    const char* msg = qsim_error_message(err);
    fprintf(stderr, "Error: %s\n", msg);
    return err;
}
```

## Profiling Integration

Built-in profiling support:

```c
// Enable profiling
qsim_profiler_start();

// ... quantum operations ...

// Get results
qsim_profile_t profile = qsim_profiler_stop();

printf("Total time: %.3f ms\n", profile.total_time_ms);
printf("Gate applications: %llu\n", profile.gate_count);
printf("GPU transfers: %llu\n", profile.gpu_transfers);
```

## Further Reading

- [State Vector Engine](state-vector-engine.md) - Core simulation engine
- [GPU Pipeline](gpu-pipeline.md) - Metal acceleration details
- [Tensor Network Engine](tensor-network-engine.md) - MPS/DMRG internals
- [Contributing Guide](../contributing/index.md) - Development setup

