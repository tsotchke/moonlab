# Memory Requirements

Detailed memory analysis for quantum simulation at various scales.

## State Vector Memory

### Theoretical Memory

For an $n$-qubit system, the state vector requires:

$$\text{Memory} = 2^n \times 16 \text{ bytes}$$

Each amplitude is a `complex double` (16 bytes: 8 bytes real + 8 bytes imaginary).

### Memory Table

| Qubits | Amplitudes | Memory | Notes |
|--------|------------|--------|-------|
| 8 | 256 | 4 KB | Trivial |
| 10 | 1,024 | 16 KB | L1 cache |
| 12 | 4,096 | 64 KB | L2 cache |
| 14 | 16,384 | 256 KB | L2 cache |
| 16 | 65,536 | 1 MB | L3 cache |
| 18 | 262,144 | 4 MB | L3 cache |
| 20 | 1,048,576 | 16 MB | Main memory |
| 22 | 4,194,304 | 64 MB | Main memory |
| 24 | 16,777,216 | 256 MB | Main memory |
| 26 | 67,108,864 | 1 GB | High RAM |
| 28 | 268,435,456 | 4 GB | Typical limit |
| 30 | 1,073,741,824 | 16 GB | High-end |
| 32 | 4,294,967,296 | 64 GB | Server/M2 Ultra |
| 34 | 17,179,869,184 | 256 GB | Distributed |
| 36 | 68,719,476,736 | 1 TB | Large cluster |

### Practical Limits by Hardware

| Hardware | RAM | Max Qubits | Notes |
|----------|-----|------------|-------|
| Laptop (8 GB) | 8 GB | 26 | Leave room for OS |
| Workstation (32 GB) | 32 GB | 28-29 | 80% usable |
| M2 Ultra (192 GB) | 192 GB | 32-33 | Unified memory |
| Server (512 GB) | 512 GB | 34 | ECC recommended |
| Cluster (4 TB) | 4 TB | 37 | Distributed |

## Memory Overhead

### Beyond State Vector

Total memory includes:

```c
typedef struct quantum_state {
    size_t num_qubits;              // 8 bytes
    size_t state_dim;               // 8 bytes
    complex_t* amplitudes;          // 8 bytes (pointer)
    uint64_t* measurement_outcomes; // 8 bytes (pointer)
    size_t max_measurements;        // 8 bytes
    size_t num_measurements;        // 8 bytes
    double global_phase;            // 8 bytes
    double entanglement_entropy;    // 8 bytes
    double purity;                  // 8 bytes
    double fidelity;                // 8 bytes
    int owns_memory;                // 4 bytes
} quantum_state_t;
```

Additional allocations:
- Measurement history: `max_measurements * 8 bytes` (default: 8 KB)
- Alignment padding: Up to 64 bytes (AMX alignment)

### Temporary Buffers

Operations requiring temporary memory:

| Operation | Temporary Memory | Total Memory |
|-----------|------------------|--------------|
| Gate application | 0 | $2^n \times 16$ |
| Measurement | 0 | $2^n \times 16$ |
| Cloning | $2^n \times 16$ | $2 \times 2^n \times 16$ |
| Partial trace | $2^k \times 2^k \times 16$ | Variable |
| SVD | $3 \times 2^k \times 16$ | Variable |

### Memory Profiler Output

```
╔══════════════════════════════════════════════════════════════╗
║       QUANTUM SIMULATOR MEMORY PROFILER                      ║
╠══════════════════════════════════════════════════════════════╣
║  Analyzing memory allocation patterns and overhead           ║
╚══════════════════════════════════════════════════════════════╝

=== State Vector Memory Profile ===
qubits,theoretical_mb,actual_mb,alloc_us,free_us,overhead_pct
4,0.00,0.02,12,8,200.0
6,0.00,0.02,15,9,150.0
8,0.00,0.02,18,10,50.0
10,0.02,0.03,25,12,25.0
12,0.06,0.08,42,18,15.0
14,0.25,0.28,95,35,10.0
16,1.00,1.05,280,85,5.0
18,4.00,4.12,850,220,3.0
20,16.00,16.24,3200,650,1.5
22,64.00,64.48,12000,2400,0.75
24,256.00,257.28,48000,9500,0.5
```

## Tensor Network Memory

### MPS Memory Scaling

Matrix Product State memory is polynomial:

$$\text{Memory} = n \times \chi^2 \times d^2 \times 16 \text{ bytes}$$

where:
- $n$ = number of sites
- $\chi$ = bond dimension
- $d$ = physical dimension (2 for qubits)

| Sites | Bond $\chi$ | Memory | Equivalent Full State |
|-------|-------------|--------|----------------------|
| 20 | 8 | 20 KB | 16 MB (20 qubits) |
| 50 | 16 | 200 KB | 16 PB (50 qubits) |
| 100 | 32 | 1.6 MB | $10^{15}$ TB |
| 200 | 64 | 13 MB | Astronomical |
| 500 | 128 | 200 MB | Astronomical |
| 1000 | 256 | 1.6 GB | Astronomical |

### MPO Memory

Matrix Product Operator for Hamiltonians:

$$\text{Memory}_{\text{MPO}} = n \times D^2 \times d^4 \times 16 \text{ bytes}$$

where $D$ is the MPO bond dimension.

| Hamiltonian Type | Typical $D$ | Memory (100 sites) |
|------------------|-------------|-------------------|
| Nearest-neighbor | 3-5 | 4 MB |
| Next-nearest | 5-8 | 10 MB |
| Long-range | 10-20 | 50 MB |
| Full | $O(n)$ | $O(n^2)$ |

### DMRG Working Memory

During DMRG sweeps:

$$\text{Peak Memory} \approx 4 \times n \times \chi^2 \times d^2 \times 16$$

The factor of 4 accounts for:
- MPS state
- Left environment tensors
- Right environment tensors
- Working tensors for SVD

## Distributed Memory

### Per-Rank Memory

For $n$ qubits on $P = 2^p$ MPI ranks:

$$\text{Memory per rank} = \frac{2^n \times 16}{P} = 2^{n-p} \times 16 \text{ bytes}$$

| Qubits | Ranks | Memory/Rank | Total Memory |
|--------|-------|-------------|--------------|
| 30 | 4 | 4 GB | 16 GB |
| 32 | 16 | 4 GB | 64 GB |
| 34 | 64 | 4 GB | 256 GB |
| 36 | 256 | 4 GB | 1 TB |
| 38 | 1024 | 4 GB | 4 TB |

### Communication Buffers

Additional per-rank memory for MPI:

```c
typedef struct distributed_state {
    // ...
    complex_t* send_buffer;    // local_size / 2
    complex_t* recv_buffer;    // local_size / 2
    MPI_Request* requests;     // Small
} distributed_state_t;
```

Communication buffers: ~50% additional memory per rank.

## Memory Optimization

### Aligned Allocation

Moonlab uses 64-byte alignment for optimal SIMD/AMX performance:

```c
// AMX-aligned allocation on Apple Silicon
#if HAS_ACCELERATE
state->amplitudes = accelerate_alloc_complex_array(state->state_dim);
#else
state->amplitudes = aligned_alloc(64, state->state_dim * sizeof(complex_t));
#endif
```

### Secure Zeroing

Sensitive quantum state data is securely zeroed before freeing:

```c
void quantum_state_free(quantum_state_t *state) {
    if (state->owns_memory && state->amplitudes) {
        // Prevent memory dump attacks
        secure_memzero(state->amplitudes, state->state_dim * sizeof(complex_t));
        free(state->amplitudes);
    }
}
```

### Memory Pool

For repeated allocations (e.g., in VQE):

```c
// Pre-allocate pool for circuit evaluations
typedef struct memory_pool {
    complex_t* buffers[MAX_BUFFERS];
    size_t buffer_size;
    int next_free;
} memory_pool_t;

complex_t* pool_alloc(memory_pool_t* pool) {
    return pool->buffers[pool->next_free++];
}

void pool_free(memory_pool_t* pool) {
    pool->next_free = 0;  // Reset without deallocation
}
```

## Checking Memory Availability

### C API

```c
#include <sys/sysctl.h>

size_t get_available_memory(void) {
#ifdef __APPLE__
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    uint64_t memsize;
    size_t len = sizeof(memsize);
    sysctl(mib, 2, &memsize, &len, NULL, 0);
    return (size_t)memsize;
#else
    return sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE);
#endif
}

int can_simulate_qubits(size_t num_qubits) {
    size_t required = (1ULL << num_qubits) * sizeof(complex_t);
    size_t available = get_available_memory();

    // Use at most 80% of physical memory
    return required < available * 0.8;
}
```

### Python API

```python
import moonlab
import psutil

def max_qubits_for_system():
    """Calculate maximum qubits for available memory."""
    mem = psutil.virtual_memory()
    available = mem.available * 0.8  # Use 80%

    qubits = 0
    while (2 ** qubits) * 16 < available:
        qubits += 1

    return qubits - 1

print(f"Maximum qubits: {max_qubits_for_system()}")
```

## Memory Fragmentation

### Analysis Tool Output

```
=== Memory Fragmentation Analysis ===
  Allocating 20 quantum states of varying sizes...
  Initial: allocated=12.5 MB, resident=14.2 MB, frag=12%

  Freeing every other state...
  After partial free: allocated=6.2 MB, resident=12.8 MB, frag=52%

  Reallocating freed slots...
  After realloc: allocated=13.1 MB, resident=15.5 MB, frag=15%
```

### Mitigation Strategies

1. **Allocate largest states first**
2. **Use memory pools for same-size allocations**
3. **Prefer reuse over free/alloc cycles**

## Quick Reference

### Memory Formula

```
State vector:  2^n × 16 bytes
MPS:           n × χ² × 4 × 16 bytes
Distributed:   2^(n-p) × 16 bytes per rank
```

### Rules of Thumb

- 28 qubits = 4 GB
- 30 qubits = 16 GB
- 32 qubits = 64 GB
- Each additional qubit doubles memory

### Estimating Requirements

```c
// Quick memory estimate
double estimate_memory_gb(int qubits) {
    return (1ULL << qubits) * 16.0 / (1024.0 * 1024.0 * 1024.0);
}

printf("28 qubits: %.1f GB\n", estimate_memory_gb(28));  // 4.0 GB
printf("30 qubits: %.1f GB\n", estimate_memory_gb(30));  // 16.0 GB
printf("32 qubits: %.1f GB\n", estimate_memory_gb(32));  // 64.0 GB
```

## See Also

- [Scaling Analysis](scaling-analysis.md)
- [GPU Performance](gpu-performance.md)
- [Distributed Simulation Guide](../guides/distributed-simulation.md)
- [Tensor Networks Concept](../concepts/tensor-networks.md)
