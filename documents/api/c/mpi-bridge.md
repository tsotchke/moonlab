# MPI Bridge API

Complete reference for distributed quantum simulation via MPI in the C library.

**Header**: `src/distributed/mpi_bridge.h`

## Overview

The MPI bridge module enables distributed quantum simulation across multiple nodes, allowing simulation of larger quantum systems by partitioning the state vector across processes. Key features:

- Transparent state vector partitioning
- Collective operations for quantum gates
- Amplitude exchange for multi-qubit gates
- Load balancing and synchronization

**Scaling**: Each additional node doubles the simulatable qubits (with sufficient memory).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MPI Communicator                      │
├─────────────┬─────────────┬─────────────┬───────────────┤
│   Rank 0    │   Rank 1    │   Rank 2    │   Rank 3      │
│ Amplitudes  │ Amplitudes  │ Amplitudes  │ Amplitudes    │
│  0 to N/4   │ N/4 to N/2  │ N/2 to 3N/4 │ 3N/4 to N     │
└─────────────┴─────────────┴─────────────┴───────────────┘
```

## Types

### distributed_ctx_t

Distributed simulation context.

```c
typedef struct {
    int initialized;           // MPI initialized flag
    int world_size;            // Total number of processes
    int world_rank;            // This process's rank
    MPI_Comm comm;             // MPI communicator

    // State partitioning
    size_t total_amplitudes;   // Total state vector size
    size_t local_amplitudes;   // Amplitudes on this rank
    size_t local_offset;       // Starting index for this rank

    // Communication buffers
    complex_t *send_buffer;    // Send buffer for exchanges
    complex_t *recv_buffer;    // Receive buffer
    size_t buffer_size;        // Buffer capacity

    // Performance tracking
    double comm_time;          // Cumulative communication time
    size_t messages_sent;      // Message count
    size_t bytes_transferred;  // Total bytes transferred
} distributed_ctx_t;
```

### distributed_config_t

Configuration for distributed simulation.

```c
typedef struct {
    int num_qubits;            // Total qubits to simulate
    int use_gpu;               // Enable GPU on each node
    size_t max_buffer_size;    // Maximum exchange buffer size
    int overlap_compute_comm;  // Overlap computation with communication
    int use_shared_memory;     // Use shared memory within nodes
} distributed_config_t;
```

## Initialization and Cleanup

### distributed_init

Initialize distributed simulation context.

```c
distributed_ctx_t* distributed_init(
    int *argc,
    char ***argv,
    const distributed_config_t *config
);
```

**Parameters**:
- `argc`: Pointer to argc from main()
- `argv`: Pointer to argv from main()
- `config`: Configuration (NULL for defaults)

**Returns**: Distributed context or NULL on failure

**Example**:
```c
int main(int argc, char **argv) {
    distributed_config_t config = {
        .num_qubits = 30,
        .use_gpu = 1,
        .max_buffer_size = 1024 * 1024 * 1024  // 1GB
    };

    distributed_ctx_t *ctx = distributed_init(&argc, &argv, &config);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize MPI\n");
        return 1;
    }

    printf("Rank %d of %d initialized\n", ctx->world_rank, ctx->world_size);

    // ... simulation code ...

    distributed_finalize(ctx);
    return 0;
}
```

### distributed_finalize

Finalize distributed context and MPI.

```c
void distributed_finalize(distributed_ctx_t *ctx);
```

### distributed_is_root

Check if this is the root process.

```c
int distributed_is_root(const distributed_ctx_t *ctx);
```

**Returns**: 1 if rank 0, 0 otherwise

### distributed_get_rank

Get this process's rank.

```c
int distributed_get_rank(const distributed_ctx_t *ctx);
```

### distributed_get_size

Get total number of processes.

```c
int distributed_get_size(const distributed_ctx_t *ctx);
```

## State Partitioning

### distributed_partition_state

Partition quantum state across processes.

```c
int distributed_partition_state(
    distributed_ctx_t *ctx,
    quantum_state_t *local_state,
    size_t num_qubits
);
```

**Parameters**:
- `ctx`: Distributed context
- `local_state`: Local portion of state (allocated)
- `num_qubits`: Total number of qubits

**Returns**: 0 on success, -1 on error

**Partitioning Strategy**:
- State vector of $2^n$ amplitudes divided across $P$ processes
- Each process holds $2^n / P$ contiguous amplitudes
- Rank $r$ holds amplitudes $[r \cdot 2^n/P, (r+1) \cdot 2^n/P)$

### distributed_get_local_range

Get the index range for this process.

```c
void distributed_get_local_range(
    const distributed_ctx_t *ctx,
    size_t *start,
    size_t *end
);
```

**Parameters**:
- `start`: Output: first amplitude index (inclusive)
- `end`: Output: last amplitude index (exclusive)

### distributed_index_to_rank

Determine which rank owns a given amplitude index.

```c
int distributed_index_to_rank(
    const distributed_ctx_t *ctx,
    size_t global_index
);
```

**Returns**: Rank that owns the amplitude

## Point-to-Point Communication

### distributed_send_amplitudes

Send amplitudes to another rank.

```c
int distributed_send_amplitudes(
    distributed_ctx_t *ctx,
    const complex_t *amplitudes,
    size_t count,
    int dest_rank,
    int tag
);
```

**Parameters**:
- `amplitudes`: Array of amplitudes to send
- `count`: Number of amplitudes
- `dest_rank`: Destination rank
- `tag`: Message tag

**Returns**: 0 on success, -1 on error

### distributed_recv_amplitudes

Receive amplitudes from another rank.

```c
int distributed_recv_amplitudes(
    distributed_ctx_t *ctx,
    complex_t *amplitudes,
    size_t count,
    int src_rank,
    int tag
);
```

### distributed_sendrecv_amplitudes

Simultaneous send and receive (for exchanges).

```c
int distributed_sendrecv_amplitudes(
    distributed_ctx_t *ctx,
    const complex_t *send_buf,
    size_t send_count,
    int dest_rank,
    complex_t *recv_buf,
    size_t recv_count,
    int src_rank,
    int tag
);
```

## Collective Operations

### distributed_barrier

Synchronize all processes.

```c
void distributed_barrier(distributed_ctx_t *ctx);
```

### distributed_broadcast

Broadcast data from root to all processes.

```c
int distributed_broadcast(
    distributed_ctx_t *ctx,
    void *data,
    size_t size,
    int root
);
```

### distributed_reduce_sum

Sum values across all processes.

```c
int distributed_reduce_sum(
    distributed_ctx_t *ctx,
    const double *local_value,
    double *global_sum
);
```

**Use Case**: Computing global normalization, total probability

### distributed_reduce_sum_complex

Sum complex values across all processes.

```c
int distributed_reduce_sum_complex(
    distributed_ctx_t *ctx,
    const complex_t *local_values,
    complex_t *global_sums,
    size_t count
);
```

### distributed_allreduce_sum

All-reduce sum (result available on all ranks).

```c
int distributed_allreduce_sum(
    distributed_ctx_t *ctx,
    const double *local_value,
    double *global_sum
);
```

### distributed_gather

Gather data from all ranks to root.

```c
int distributed_gather(
    distributed_ctx_t *ctx,
    const void *send_data,
    size_t send_size,
    void *recv_data,
    int root
);
```

### distributed_allgather

Gather data from all ranks to all ranks.

```c
int distributed_allgather(
    distributed_ctx_t *ctx,
    const void *send_data,
    size_t send_size,
    void *recv_data
);
```

### distributed_scatter

Scatter data from root to all ranks.

```c
int distributed_scatter(
    distributed_ctx_t *ctx,
    const void *send_data,
    void *recv_data,
    size_t recv_size,
    int root
);
```

## Amplitude Exchange

### distributed_exchange_amplitudes

Exchange amplitudes between partner ranks for multi-qubit gates.

```c
int distributed_exchange_amplitudes(
    distributed_ctx_t *ctx,
    quantum_state_t *local_state,
    int target_qubit,
    complex_t **partner_amplitudes
);
```

**Parameters**:
- `local_state`: Local portion of state
- `target_qubit`: Qubit index requiring exchange
- `partner_amplitudes`: Output: received partner amplitudes

**Returns**: 0 on success, -1 on error

**When Exchange is Needed**:
For an n-qubit system on P processes, qubits 0 to log2(P)-1 are "global" (span multiple ranks). Gates on global qubits require amplitude exchange.

### distributed_apply_global_gate

Apply gate to a global qubit (requires communication).

```c
int distributed_apply_global_gate(
    distributed_ctx_t *ctx,
    quantum_state_t *local_state,
    int qubit,
    const complex_t gate[4]
);
```

**Parameters**:
- `local_state`: Local state partition
- `qubit`: Global qubit index
- `gate`: 2×2 gate matrix in row-major order

**Returns**: 0 on success, -1 on error

**Algorithm**:
1. Determine partner rank based on qubit index
2. Exchange relevant amplitudes with partner
3. Apply gate using local and received amplitudes
4. Update local state

### distributed_apply_global_controlled_gate

Apply controlled gate spanning ranks.

```c
int distributed_apply_global_controlled_gate(
    distributed_ctx_t *ctx,
    quantum_state_t *local_state,
    int control_qubit,
    int target_qubit,
    const complex_t gate[4]
);
```

## Distributed Gates

### distributed_hadamard

Apply Hadamard gate in distributed setting.

```c
int distributed_hadamard(
    distributed_ctx_t *ctx,
    quantum_state_t *local_state,
    int qubit
);
```

Automatically handles local vs. global qubit cases.

### distributed_cnot

Apply CNOT gate in distributed setting.

```c
int distributed_cnot(
    distributed_ctx_t *ctx,
    quantum_state_t *local_state,
    int control,
    int target
);
```

### distributed_phase

Apply phase gate in distributed setting.

```c
int distributed_phase(
    distributed_ctx_t *ctx,
    quantum_state_t *local_state,
    int qubit,
    double angle
);
```

## State Vector Operations

### distributed_normalize

Normalize state vector across all ranks.

```c
int distributed_normalize(
    distributed_ctx_t *ctx,
    quantum_state_t *local_state
);
```

**Algorithm**:
1. Compute local sum of squared magnitudes
2. Allreduce to get global sum
3. Each rank normalizes its local amplitudes

### distributed_inner_product

Compute inner product of distributed states.

```c
complex_t distributed_inner_product(
    distributed_ctx_t *ctx,
    const quantum_state_t *state1,
    const quantum_state_t *state2
);
```

### distributed_expectation_value

Compute expectation value of observable.

```c
double distributed_expectation_value(
    distributed_ctx_t *ctx,
    const quantum_state_t *local_state,
    const pauli_hamiltonian_t *hamiltonian
);
```

## Measurement

### distributed_measure

Perform measurement on distributed state.

```c
uint64_t distributed_measure(
    distributed_ctx_t *ctx,
    quantum_state_t *local_state,
    int qubit,
    double *probability,
    quantum_entropy_ctx_t *entropy
);
```

**Algorithm**:
1. Root generates random threshold
2. Broadcast threshold to all ranks
3. Compute local cumulative probabilities
4. Allreduce to determine outcome
5. Collapse local state consistently

### distributed_measure_all

Measure all qubits.

```c
uint64_t distributed_measure_all(
    distributed_ctx_t *ctx,
    quantum_state_t *local_state,
    quantum_entropy_ctx_t *entropy
);
```

**Returns**: Measured basis state index

### distributed_sample

Sample from distribution without collapse.

```c
uint64_t distributed_sample(
    distributed_ctx_t *ctx,
    const quantum_state_t *local_state,
    quantum_entropy_ctx_t *entropy
);
```

## Performance Monitoring

### distributed_get_comm_time

Get cumulative communication time.

```c
double distributed_get_comm_time(const distributed_ctx_t *ctx);
```

**Returns**: Communication time in seconds

### distributed_get_stats

Get communication statistics.

```c
void distributed_get_stats(
    const distributed_ctx_t *ctx,
    size_t *messages,
    size_t *bytes
);
```

### distributed_reset_stats

Reset performance counters.

```c
void distributed_reset_stats(distributed_ctx_t *ctx);
```

### distributed_print_stats

Print performance summary.

```c
void distributed_print_stats(const distributed_ctx_t *ctx);
```

## Utility Functions

### distributed_print_state

Print global state from root.

```c
void distributed_print_state(
    distributed_ctx_t *ctx,
    const quantum_state_t *local_state,
    int max_amplitudes
);
```

### distributed_validate_state

Check state consistency across ranks.

```c
int distributed_validate_state(
    distributed_ctx_t *ctx,
    const quantum_state_t *local_state
);
```

**Returns**: 1 if valid, 0 if inconsistent

### distributed_save_state

Save distributed state to file.

```c
int distributed_save_state(
    distributed_ctx_t *ctx,
    const quantum_state_t *local_state,
    const char *filename
);
```

### distributed_load_state

Load distributed state from file.

```c
int distributed_load_state(
    distributed_ctx_t *ctx,
    quantum_state_t *local_state,
    const char *filename
);
```

## Complete Example

```c
#include "src/distributed/mpi_bridge.h"
#include "src/quantum/state.h"
#include "src/quantum/gates.h"
#include <stdio.h>

int main(int argc, char **argv) {
    // Initialize MPI
    distributed_config_t config = {
        .num_qubits = 28,           // 28 qubits = 4GB state vector
        .use_gpu = 1,
        .overlap_compute_comm = 1
    };

    distributed_ctx_t *ctx = distributed_init(&argc, &argv, &config);
    if (!ctx) return 1;

    // Allocate local state partition
    quantum_state_t local_state;
    distributed_partition_state(ctx, &local_state, config.num_qubits);

    // Initialize |0⟩ state (only rank 0 has non-zero amplitude)
    if (distributed_is_root(ctx)) {
        local_state.amplitudes[0] = 1.0;
    }

    // Apply Hadamard to all qubits
    for (int q = 0; q < config.num_qubits; q++) {
        distributed_hadamard(ctx, &local_state, q);
    }

    // Create entanglement with CNOTs
    for (int q = 0; q < config.num_qubits - 1; q++) {
        distributed_cnot(ctx, &local_state, q, q + 1);
    }

    // Normalize
    distributed_normalize(ctx, &local_state);

    // Measure
    quantum_entropy_ctx_t entropy;
    quantum_entropy_init(&entropy, NULL, NULL);

    uint64_t result = distributed_measure_all(ctx, &local_state, &entropy);

    if (distributed_is_root(ctx)) {
        printf("Measurement result: %llu\n", result);
        distributed_print_stats(ctx);
    }

    // Cleanup
    quantum_state_free(&local_state);
    distributed_finalize(ctx);

    return 0;
}
```

**Compilation**:
```bash
mpicc -O3 distributed_example.c -o distributed_example \
    -I. -L. -lmoonlab -lm

# Run on 4 nodes
mpirun -np 4 ./distributed_example
```

## Scaling Guidelines

### Memory Requirements

| Qubits | State Size | Nodes (16GB each) |
|--------|------------|-------------------|
| 28 | 4 GB | 1 |
| 30 | 16 GB | 1-2 |
| 32 | 64 GB | 4-8 |
| 34 | 256 GB | 16-32 |
| 36 | 1 TB | 64-128 |

### Communication Overhead

| Operation | Local Qubit | Global Qubit |
|-----------|-------------|--------------|
| Single gate | O(1) | O(N/P) transfer |
| CNOT | O(1) | O(N/P) transfer |
| Measurement | O(log P) | O(log P) |
| Normalize | O(log P) | O(log P) |

### Best Practices

1. **Minimize global gates**: Place high-connectivity qubits in local range
2. **Batch communications**: Combine multiple gate exchanges
3. **Use GPU per node**: Hybrid MPI+GPU for best performance
4. **Overlap compute/comm**: Enable for large simulations

## See Also

- [GPU Metal API](gpu-metal.md) - GPU acceleration per node
- [Guides: Distributed Simulation](../../guides/distributed-simulation.md)
- [Architecture: Distributed Architecture](../../architecture/distributed-architecture.md)
