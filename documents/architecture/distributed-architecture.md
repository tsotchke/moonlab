# Distributed Architecture

Design and implementation of Moonlab's MPI-based distributed quantum simulation.

## Overview

Moonlab's distributed backend enables simulation of quantum systems larger than single-machine memory by partitioning the state vector across multiple compute nodes.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Master Node (Rank 0)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Control   │  │   State     │  │   Result    │              │
│  │   Logic     │  │   Partition │  │   Aggregator│              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         │              MPI Communication          │
         ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Rank 1    │      │   Rank 2    │      │   Rank 3    │
│ ┌─────────┐ │      │ ┌─────────┐ │      │ ┌─────────┐ │
│ │ Local   │ │      │ │ Local   │ │      │ │ Local   │ │
│ │ State   │ │      │ │ State   │ │      │ │ State   │ │
│ │ Vector  │ │      │ │ Vector  │ │      │ │ Vector  │ │
│ └─────────┘ │      │ └─────────┘ │      │ └─────────┘ │
│ ┌─────────┐ │      │ ┌─────────┐ │      │ ┌─────────┐ │
│ │ Gate    │ │      │ │ Gate    │ │      │ │ Gate    │ │
│ │ Engine  │ │      │ │ Engine  │ │      │ │ Engine  │ │
│ └─────────┘ │      │ └─────────┘ │      │ └─────────┘ │
└─────────────┘      └─────────────┘      └─────────────┘
```

## State Vector Partitioning

### Partitioning Strategy

The state vector is divided among $P = 2^p$ ranks based on the $p$ highest-order qubits:

```
n-qubit system on P ranks:
├── Global qubits: n-p to n-1 (determine rank ownership)
└── Local qubits: 0 to n-p-1 (stored locally per rank)

Rank r owns amplitudes where top p bits equal r:
  indices: [r × 2^{n-p}, (r+1) × 2^{n-p})
```

### Example: 32 Qubits on 4 Ranks

```
Total amplitudes: 2^32 = 4,294,967,296
Amplitudes per rank: 2^30 = 1,073,741,824 (16 GB each)

Rank 0: indices 0x00000000 - 0x3FFFFFFF (top bits = 00)
Rank 1: indices 0x40000000 - 0x7FFFFFFF (top bits = 01)
Rank 2: indices 0x80000000 - 0xBFFFFFFF (top bits = 10)
Rank 3: indices 0xC0000000 - 0xFFFFFFFF (top bits = 11)
```

### Data Structure

```c
typedef struct distributed_state {
    // MPI info
    int rank;
    int num_ranks;
    MPI_Comm comm;

    // State info
    size_t num_qubits;          // Total qubits (n)
    size_t local_qubits;        // Local qubits (n - log2(P))
    size_t global_qubits;       // Global qubits (log2(P))

    // Local storage
    size_t local_size;          // 2^{n-p}
    complex_t* local_amplitudes;

    // Communication buffers
    complex_t* send_buffer;
    complex_t* recv_buffer;
    size_t buffer_size;

    // Async communication
    MPI_Request* requests;
    int num_pending;
} distributed_state_t;
```

## Gate Classification

### Local Gates

Gates on qubits 0 to n-p-1 require no communication:

```c
void distributed_local_gate(distributed_state_t* state,
                            int target,
                            const complex_t gate[2][2]) {
    // Direct application within local partition
    size_t stride = 1ULL << target;

    #pragma omp parallel for
    for (size_t i = 0; i < state->local_size >> 1; i++) {
        size_t i0 = ((i >> target) << (target + 1)) | (i & (stride - 1));
        size_t i1 = i0 | stride;

        complex_t a0 = state->local_amplitudes[i0];
        complex_t a1 = state->local_amplitudes[i1];

        state->local_amplitudes[i0] = gate[0][0] * a0 + gate[0][1] * a1;
        state->local_amplitudes[i1] = gate[1][0] * a0 + gate[1][1] * a1;
    }
}
```

### Global Gates

Gates on qubits n-p to n-1 require inter-rank communication:

```c
void distributed_global_gate(distributed_state_t* state,
                             int target,
                             const complex_t gate[2][2]) {
    // Determine partner rank
    int local_target = target - state->local_qubits;
    int partner = state->rank ^ (1 << local_target);

    // Determine which half to exchange
    bool send_upper = (state->rank > partner);

    // Exchange data
    complex_t* local_half = send_upper ?
        state->local_amplitudes + state->local_size / 2 :
        state->local_amplitudes;

    MPI_Sendrecv(
        local_half, state->local_size / 2, MPI_DOUBLE_COMPLEX,
        partner, 0,
        state->recv_buffer, state->local_size / 2, MPI_DOUBLE_COMPLEX,
        partner, 0,
        state->comm, MPI_STATUS_IGNORE
    );

    // Apply gate to (local_half, recv_buffer) pairs
    apply_gate_to_exchanged(state, gate, send_upper);
}
```

### Two-Qubit Global Gates

```c
void distributed_cnot_global(distributed_state_t* state,
                             int control, int target) {
    // Classify control and target
    bool ctrl_global = control >= state->local_qubits;
    bool targ_global = target >= state->local_qubits;

    if (!ctrl_global && !targ_global) {
        // Both local
        distributed_local_cnot(state, control, target);
    } else if (ctrl_global && !targ_global) {
        // Control global, target local
        distributed_cnot_ctrl_global(state, control, target);
    } else if (!ctrl_global && targ_global) {
        // Control local, target global
        distributed_cnot_targ_global(state, control, target);
    } else {
        // Both global - most complex case
        distributed_cnot_both_global(state, control, target);
    }
}

void distributed_cnot_both_global(distributed_state_t* state,
                                  int control, int target) {
    int ctrl_bit = 1 << (control - state->local_qubits);
    int targ_bit = 1 << (target - state->local_qubits);

    // Four ranks participate
    int group_base = state->rank & ~(ctrl_bit | targ_bit);
    int ranks[4] = {
        group_base,
        group_base | ctrl_bit,
        group_base | targ_bit,
        group_base | ctrl_bit | targ_bit
    };

    // Collective all-to-all within group
    // Reorganize data, apply CNOT, redistribute
    // ...
}
```

## Communication Optimization

### Non-blocking Communication

```c
void distributed_gate_async(distributed_state_t* state,
                            int target,
                            const complex_t gate[2][2]) {
    int partner = state->rank ^ (1 << (target - state->local_qubits));
    bool send_upper = (state->rank > partner);

    complex_t* send_ptr = send_upper ?
        state->local_amplitudes + state->local_size / 2 :
        state->local_amplitudes;

    // Start non-blocking exchange
    MPI_Irecv(state->recv_buffer, state->local_size / 2,
              MPI_DOUBLE_COMPLEX, partner, 0, state->comm,
              &state->requests[0]);

    MPI_Isend(send_ptr, state->local_size / 2,
              MPI_DOUBLE_COMPLEX, partner, 0, state->comm,
              &state->requests[1]);

    // Compute on non-exchanged half while waiting
    apply_gate_local_half(state, gate, !send_upper);

    // Wait for exchange
    MPI_Waitall(2, state->requests, MPI_STATUSES_IGNORE);

    // Complete gate on exchanged data
    apply_gate_exchanged_half(state, gate, send_upper);
}
```

### Pipelining

```c
void distributed_circuit_pipelined(distributed_state_t* state,
                                   gate_sequence_t* gates) {
    for (int i = 0; i < gates->count; i++) {
        gate_t* g = &gates->gates[i];

        if (is_local_gate(state, g)) {
            // Apply local gate immediately
            apply_local_gate(state, g);
        } else {
            // Start async communication
            start_exchange(state, g);

            // Apply any pending local gates
            while (i + 1 < gates->count && is_local_gate(state, &gates->gates[i+1])) {
                i++;
                apply_local_gate(state, &gates->gates[i]);
            }

            // Complete global gate
            finish_exchange(state, g);
            apply_global_gate(state, g);
        }
    }
}
```

## Collective Operations

### Probability Calculation

```c
double distributed_probability(distributed_state_t* state, size_t index) {
    int owner_rank = index >> state->local_qubits;

    double local_prob = 0.0;
    if (state->rank == owner_rank) {
        size_t local_index = index & (state->local_size - 1);
        complex_t amp = state->local_amplitudes[local_index];
        local_prob = creal(amp) * creal(amp) + cimag(amp) * cimag(amp);
    }

    double global_prob;
    MPI_Allreduce(&local_prob, &global_prob, 1, MPI_DOUBLE,
                  MPI_SUM, state->comm);

    return global_prob;
}
```

### Measurement

```c
uint64_t distributed_measure_all(distributed_state_t* state,
                                 quantum_entropy_ctx_t* entropy) {
    // Compute local probability sums
    double local_cumsum = 0.0;
    double* cumsum = malloc(state->num_ranks * sizeof(double));

    for (size_t i = 0; i < state->local_size; i++) {
        complex_t amp = state->local_amplitudes[i];
        local_cumsum += creal(amp) * creal(amp) + cimag(amp) * cimag(amp);
    }

    // Gather cumulative sums
    MPI_Allgather(&local_cumsum, 1, MPI_DOUBLE,
                  cumsum, 1, MPI_DOUBLE, state->comm);

    // Rank 0 generates random threshold and broadcasts
    double r;
    if (state->rank == 0) {
        r = quantum_entropy_random_double(entropy);
    }
    MPI_Bcast(&r, 1, MPI_DOUBLE, 0, state->comm);

    // Find which rank contains the outcome
    double prefix = 0.0;
    int outcome_rank = -1;
    for (int i = 0; i < state->num_ranks; i++) {
        if (prefix + cumsum[i] > r) {
            outcome_rank = i;
            break;
        }
        prefix += cumsum[i];
    }

    // Rank with outcome finds specific index
    uint64_t result = 0;
    if (state->rank == outcome_rank) {
        double target = r - prefix;
        double local_sum = 0.0;

        for (size_t i = 0; i < state->local_size; i++) {
            complex_t amp = state->local_amplitudes[i];
            local_sum += creal(amp) * creal(amp) + cimag(amp) * cimag(amp);
            if (local_sum > target) {
                result = ((uint64_t)state->rank << state->local_qubits) | i;
                break;
            }
        }
    }

    // Broadcast result to all ranks
    MPI_Bcast(&result, 1, MPI_UINT64_T, outcome_rank, state->comm);

    // Collapse state
    distributed_collapse(state, result);

    free(cumsum);
    return result;
}
```

## Fault Tolerance

### Checkpointing

```c
void distributed_checkpoint(distributed_state_t* state, const char* path) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/state_rank%04d.bin",
             path, state->rank);

    FILE* f = fopen(filename, "wb");
    fwrite(&state->num_qubits, sizeof(size_t), 1, f);
    fwrite(&state->local_size, sizeof(size_t), 1, f);
    fwrite(state->local_amplitudes, sizeof(complex_t),
           state->local_size, f);
    fclose(f);

    // Barrier to ensure all ranks complete
    MPI_Barrier(state->comm);

    if (state->rank == 0) {
        // Write metadata
        snprintf(filename, sizeof(filename), "%s/metadata.json", path);
        write_checkpoint_metadata(state, filename);
    }
}

void distributed_restore(distributed_state_t* state, const char* path) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/state_rank%04d.bin",
             path, state->rank);

    FILE* f = fopen(filename, "rb");
    size_t qubits, size;
    fread(&qubits, sizeof(size_t), 1, f);
    fread(&size, sizeof(size_t), 1, f);

    assert(qubits == state->num_qubits);
    assert(size == state->local_size);

    fread(state->local_amplitudes, sizeof(complex_t), size, f);
    fclose(f);

    MPI_Barrier(state->comm);
}
```

## Scaling Characteristics

### Strong Scaling

Fixed problem size, increasing ranks:

| Qubits | 1 Rank | 4 Ranks | 16 Ranks | 64 Ranks |
|--------|--------|---------|----------|----------|
| 28 | 1.0x | 3.2x | 11.5x | 38.2x |
| 30 | OOM | 1.0x | 3.5x | 12.8x |
| 32 | OOM | OOM | 1.0x | 3.6x |

### Weak Scaling

Fixed local size, increasing total problem:

| Ranks | Total Qubits | Local Memory | Time |
|-------|--------------|--------------|------|
| 1 | 28 | 4 GB | 1.0x |
| 4 | 30 | 4 GB | 1.1x |
| 16 | 32 | 4 GB | 1.3x |
| 64 | 34 | 4 GB | 1.7x |

### Communication Overhead

| Operation | Message Size | Latency Bound |
|-----------|--------------|---------------|
| Local gate | 0 | 0 |
| Global gate | 2^{n-p-1} × 16B | Yes |
| Measurement | O(P) | Yes |
| Normalization | O(1) | Yes |

## See Also

- [MPI Bridge API](../api/c/mpi-bridge.md)
- [Distributed Simulation Guide](../guides/distributed-simulation.md)
- [Scaling Analysis](../performance/scaling-analysis.md)
