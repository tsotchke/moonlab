# System Overview

High-level architecture and component interactions in Moonlab.

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              User Applications                             │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                           Language Bindings                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │    Python    │  │     Rust     │  │  JavaScript  │  │    Swift     │   │
│  │   (cffi)     │  │    (FFI)     │  │   (WASM)     │  │  (C interop) │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└───────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ C ABI
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                              C Core Library                               │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                        Algorithm Layer                               │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────────────┐  │ │
│  │  │ Grover  │ │   VQE   │ │  QAOA   │ │   QPE   │ │ Tensor Network │  │ │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ │  ┌──────────┐  │  │ │
│  │       │           │           │           │      │  │   DMRG   │  │  │ │
│  │       └───────────┴───────────┴───────────┘      │  │   TEBD   │  │  │ │
│  │                          │                       │  └──────────┘  │  │ │
│  └──────────────────────────┼───────────────────────┴────────────────┘  │ │
│                             │                                           │ │
│  ┌──────────────────────────┴───────────────────────────────────────┐   │ │
│  │                      Core Quantum Layer                          │   │ │
│  │  ┌───────────────┐  ┌──────────┐  ┌─────────────┐  ┌──────────┐  │   │ │
│  │  │ State Vector  │  │  Gates   │  │ Measurement │  │  Noise   │  │   │ │
│  │  │   Engine      │  │ Library  │  │   Engine    │  │  Models  │  │   │ │
│  │  └───────┬───────┘  └────┬─────┘  └──────┬──────┘  └────┬─────┘  │   │ │
│  │          │               │               │              │        │   │ │
│  │          └───────────────┴───────────────┴──────────────┘        │   │ │
│  └──────────────────────────┬───────────────────────────────────────┘   │ │
│                             │                                           │ │
│  ┌──────────────────────────┴───────────────────────────────────────┐   │ │
│  │                    Optimization Layer                            │   │ │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐ │   │ │
│  │  │   SIMD Ops    │  │   GPU Metal   │  │     MPI Bridge        │ │   │ │
│  │  │  (NEON/AVX)   │  │   Pipeline    │  │  (Distributed Ops)    │ │   │ │
│  │  └───────────────┘  └───────────────┘  └───────────────────────┘ │   │ │
│  └──────────────────────────────────────────────────────────────────┘   │ │
│                                                                         │ │
│  ┌──────────────────────────────────────────────────────────────────┐   │ │
│  │                      Utility Layer                               │   │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │   │ │
│  │  │ Memory   │  │ Config   │  │ Entropy  │  │    Profiler      │  │   │ │
│  │  │ Manager  │  │ System   │  │ Source   │  │                  │  │   │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │   │ │
│  └──────────────────────────────────────────────────────────────────┘   │ │
│─────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                           Hardware Layer                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   CPU/SIMD   │  │  Apple GPU   │  │  Network     │  │   Storage    │    │
│  │  (ARM/x86)   │  │   (Metal)    │  │  (MPI/TCP)   │  │   (Files)    │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└────────────────────────────────────────────────────────────────────────────┘
```

## Layer Descriptions

### Algorithm Layer

High-level quantum algorithms built on core primitives:

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **Grover** | Unstructured search | Oracle construction, amplitude amplification |
| **VQE** | Variational eigensolver | Ansatz library, optimizer integration |
| **QAOA** | Optimization problems | MaxCut, QUBO solver |
| **QPE** | Phase estimation | Iterative and standard variants |
| **Tensor Network** | Large-scale simulation | DMRG, TEBD, MPS operations |

### Core Quantum Layer

Fundamental quantum operations:

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **State Vector Engine** | Amplitude storage | 16-byte aligned complex arrays |
| **Gates Library** | Gate matrices | 40+ gates, controlled variants |
| **Measurement Engine** | Probabilistic measurement | Partial and full collapse |
| **Noise Models** | Decoherence simulation | Kraus operators, channels |

### Optimization Layer

Performance acceleration:

| Component | Purpose | Speedup |
|-----------|---------|---------|
| **SIMD Ops** | Vectorized arithmetic | 4-8x |
| **GPU Metal** | Parallel state manipulation | 10-100x |
| **MPI Bridge** | Distributed computing | Linear scaling |

### Utility Layer

Supporting infrastructure:

| Component | Purpose |
|-----------|---------|
| **Memory Manager** | Aligned allocation, pooling |
| **Config System** | Runtime configuration |
| **Entropy Source** | Cryptographic randomness |
| **Profiler** | Performance measurement |

## Data Structures

### Quantum State

```c
typedef struct {
    Complex* amplitudes;      // State vector (2^n complex numbers)
    uint32_t num_qubits;      // Number of qubits
    uint64_t dim;             // Dimension (2^num_qubits)
    uint32_t flags;           // State flags (GPU, MPI, etc.)
    void* gpu_buffer;         // GPU buffer (if applicable)
    void* mpi_data;           // MPI metadata (if applicable)
} quantum_state_t;
```

### Gate Representation

```c
typedef struct {
    Complex matrix[4][4];     // Up to 4x4 matrix (2-qubit)
    uint8_t num_qubits;       // Gate size (1, 2, or 3)
    uint8_t gate_type;        // Gate identifier
    double* parameters;       // Gate parameters (angles, etc.)
    uint8_t num_parameters;   // Number of parameters
} quantum_gate_t;
```

### Tensor Network State

```c
typedef struct {
    tensor_t** tensors;       // Array of site tensors
    uint32_t num_sites;       // Number of sites
    uint32_t* bond_dims;      // Bond dimensions
    uint32_t max_bond_dim;    // Maximum bond dimension
    uint8_t canonical_form;   // Current canonical form
    uint32_t center;          // Canonical center position
} mps_t;
```

## Execution Flow

### Gate Application

```
quantum_state_h(state, qubit)
         │
         ▼
┌────────────────────┐
│ 1. Validate input  │ → Check qubit in range
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ 2. Select backend  │ → GPU if available and state is large
└─────────┬──────────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌────────┐  ┌────────┐
│  CPU   │  │  GPU   │
└───┬────┘  └───┬────┘
    │           │
    ▼           ▼
┌────────────────────┐
│ 3. Apply gate      │ → Iterate over amplitude pairs
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ 4. Synchronize     │ → GPU sync if needed
└─────────┬──────────┘
          │
          ▼
     Return success
```

### Algorithm Execution

```
vqe_solve(hamiltonian, config)
         │
         ▼
┌────────────────────┐
│ 1. Initialize      │ → Random parameters, create state
└─────────┬──────────┘
          │
          ▼
┌────────────────────────────────────────┐
│ 2. Optimization Loop                   │
│   ┌──────────────────────────────────┐ │
│   │ a. Build ansatz circuit          │ │
│   │ b. Apply circuit to state        │ │
│   │ c. Measure Hamiltonian terms     │ │
│   │ d. Compute total energy          │ │
│   │ e. Update parameters             │ │
│   └──────────────────────────────────┘ │
│              │                         │
│              ▼                         │
│      Converged? ──No──┐                │
│         │             │                │
│        Yes            │                │
│         │             │                │
└─────────┼─────────────┘                │
          │                              │
          ▼                              │
┌────────────────────┐                   │
│ 3. Return result   │ ← Energy, state, params
└────────────────────┘
```

## Memory Management

### Allocation Strategy

```c
// Aligned allocation for SIMD
void* qsim_alloc_aligned(size_t size, size_t alignment) {
    // Use posix_memalign on POSIX, _aligned_malloc on Windows
    void* ptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

// State vector allocation (16-byte aligned for SSE, 32-byte for AVX)
Complex* allocate_state_vector(uint64_t dim) {
    size_t alignment = get_optimal_alignment();
    size_t size = dim * sizeof(Complex);
    return (Complex*)qsim_alloc_aligned(size, alignment);
}
```

### Memory Pools

For algorithm temporaries:

```c
typedef struct {
    void* base;           // Pool base address
    size_t size;          // Total pool size
    size_t offset;        // Current allocation offset
} memory_pool_t;

void* pool_alloc(memory_pool_t* pool, size_t size) {
    // Align to 16 bytes
    size_t aligned_offset = (pool->offset + 15) & ~15;

    if (aligned_offset + size > pool->size) {
        return NULL;  // Pool exhausted
    }

    void* ptr = (char*)pool->base + aligned_offset;
    pool->offset = aligned_offset + size;
    return ptr;
}
```

## Configuration System

### Hierarchy

```
┌────────────────────────────────────────┐
│          Compile-time defaults         │
└────────────────────┬───────────────────┘
                     │ overridden by
                     ▼
┌────────────────────────────────────────┐
│         Environment variables          │
│  QSIM_GPU_ENABLED, QSIM_NUM_THREADS    │
└────────────────────┬───────────────────┘
                     │ overridden by
                     ▼
┌────────────────────────────────────────┐
│           Configuration file           │
│         ~/.moonlab/config.toml         │
└────────────────────┬───────────────────┘
                     │ overridden by
                     ▼
┌────────────────────────────────────────┐
│         Runtime API calls              │
│  qsim_config_set("key", "value")       │
└────────────────────────────────────────┘
```

### Key Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `gpu.enabled` | bool | auto | Enable GPU acceleration |
| `gpu.threshold` | int | 18 | Min qubits for GPU |
| `simd.enabled` | bool | true | Enable SIMD |
| `threads.count` | int | auto | OpenMP threads |
| `memory.pool_size` | int | 256MB | Temp pool size |
| `precision` | string | "double" | Float precision |

## Error Handling

### Error Propagation

```c
// All public functions return error codes
qsim_error_t quantum_state_h(quantum_state_t* state, uint32_t qubit) {
    if (state == NULL) {
        return QSIM_ERROR_NULL_POINTER;
    }

    if (qubit >= state->num_qubits) {
        return QSIM_ERROR_INVALID_QUBIT;
    }

    // Perform operation
    return apply_hadamard(state, qubit);
}

// Error context for detailed information
typedef struct {
    qsim_error_t code;
    const char* message;
    const char* file;
    int line;
    const char* function;
} qsim_error_context_t;

// Get detailed error info
qsim_error_context_t qsim_get_last_error(void);
```

### Error Categories

| Category | Code Range | Examples |
|----------|------------|----------|
| Success | 0 | QSIM_SUCCESS |
| Input errors | 1-99 | NULL pointer, invalid qubit |
| Resource errors | 100-199 | Out of memory, GPU unavailable |
| State errors | 200-299 | Invalid state, corruption |
| Algorithm errors | 300-399 | Non-convergence, timeout |

## Thread Safety Model

### Thread-Local Storage

```c
// Thread-local error context
static __thread qsim_error_context_t tls_error_context;

// Thread-local random state
static __thread uint64_t tls_rng_state[4];
```

### Synchronization Points

| Operation | Requires Lock | Notes |
|-----------|---------------|-------|
| State creation | No | Returns new object |
| Gate application | Yes (on state) | Modifies amplitudes |
| GPU transfer | Yes (global) | Single GPU queue |
| MPI collective | Yes (global) | Barrier synchronization |
| Config read | No | Immutable after init |
| Config write | Yes (global) | Thread-safe setter |

## Performance Monitoring

### Built-in Profiler

```c
// Start profiling
qsim_profiler_enable(QSIM_PROFILE_GATES | QSIM_PROFILE_GPU);

// ... operations ...

// Get profile data
qsim_profile_t profile;
qsim_profiler_get(&profile);

printf("Gate applications: %llu\n", profile.gate_count);
printf("Gate time: %.3f ms\n", profile.gate_time_ms);
printf("GPU kernels: %llu\n", profile.gpu_kernel_count);
printf("GPU time: %.3f ms\n", profile.gpu_time_ms);
printf("Memory allocated: %zu bytes\n", profile.memory_allocated);
```

### Metrics Collected

| Metric | Description |
|--------|-------------|
| `gate_count` | Total gate applications |
| `gate_time_ms` | Time in gate operations |
| `gpu_kernel_count` | GPU kernel launches |
| `gpu_time_ms` | GPU execution time |
| `gpu_transfer_bytes` | Data transferred to/from GPU |
| `memory_allocated` | Peak memory usage |
| `mpi_messages` | MPI messages sent |

## See Also

- [State Vector Engine](state-vector-engine.md) - Core simulation details
- [GPU Pipeline](gpu-pipeline.md) - GPU acceleration architecture
- [C API Reference](../api/c/index.md) - Complete API documentation

