# Distributed Simulation Guide

Scale quantum simulations beyond single-machine memory limits using MPI-based distributed computing.

## Overview

A single-node dense state vector holds `2^n` complex128 amplitudes (16 bytes each), which
puts a practical ceiling around 30-32 qubits before host RAM is exhausted (32 qubits is
64 GB). Moonlab's distributed backend partitions that amplitude array across MPI ranks so
the aggregate memory of a cluster carries systems no single node can hold. Each doubling of
the rank count `P` adds one qubit of headroom: with `P = 2^k` ranks you gain `k` qubits over
the single-node ceiling (1024 ranks add ~10 qubits). Each rank's local shard can optionally
be GPU-resident (`partition_state_create_gpu`) when Moonlab is built with CUDA, so the same
sharding scheme drives multi-GPU/MPI runs.

## When to Use Distributed Simulation

| Scenario | Single Node (dense) | Distributed (MPI/CUDA sharding) |
|----------|---------------------|---------------------------------|
| ≤28 qubits | Recommended | Unnecessary communication overhead |
| 29-32 qubits | Needs 8-64 GB RAM; near the dense ceiling | Splits the shard across nodes |
| 33+ qubits | Beyond a single node's dense capacity | Required; `P` ranks add `log2(P)` qubits |

The distributed engine is compiled into `libquantumsim` only when it is built with
`-DQSIM_ENABLE_MPI=ON`. On a default (non-MPI) build the Python bindings raise
`MpiUnavailableError` and the C symbols are absent.

## Prerequisites

### Hardware Requirements

- MPI-capable cluster or multi-node setup
- High-bandwidth interconnect (InfiniBand recommended)
- Sufficient aggregate memory: 16 bytes × 2^n / num_ranks per node

### Software Requirements

```bash
# Install MPI (choose one)
# macOS
brew install open-mpi

# Ubuntu/Debian
sudo apt-get install libopenmpi-dev openmpi-bin

# RHEL/CentOS
sudo yum install openmpi openmpi-devel
```

### Building with MPI Support

The distributed engine is compiled in only when configured with `-DQSIM_ENABLE_MPI=ON`
(build the shared library too so the Python bindings can load it):

```bash
cmake -S . -B build-dist -DCMAKE_BUILD_TYPE=Release \
      -DQSIM_ENABLE_MPI=ON -DQSIM_BUILD_SHARED=ON
cmake --build build-dist

# Point the Python bindings at the MPI-enabled library.
export MOONLAB_LIB_DIR=$PWD/build-dist
```

Verify from Python that the distributed symbols are present:

```python
from moonlab.distributed import is_mpi_available
print("MPI:", "enabled" if is_mpi_available() else "disabled")
```

## Basic Usage

### C API

```c
#include "distributed/mpi_bridge.h"
#include "distributed/state_partition.h"
#include "distributed/distributed_gates.h"
#include "distributed/collective_ops.h"
#include <stdio.h>

int main(int argc, char** argv) {
    // Initialize the MPI bridge (NULL options => defaults).
    distributed_ctx_t* ctx = mpi_bridge_init(&argc, &argv, NULL);
    if (!ctx) {
        fprintf(stderr, "MPI initialization failed\n");
        return 1;
    }

    int rank = mpi_get_rank(ctx);
    int size = mpi_get_size(ctx);

    if (mpi_is_root(ctx)) {
        printf("Running on %d MPI ranks\n", size);
    }

    // Create a distributed state (32 qubits partitioned across all ranks;
    // NULL config => library defaults). Initialized to |0...0>.
    partitioned_state_t* state = partition_state_create(ctx, 32, NULL);

    // Apply gates. The engine handles inter-rank exchange for partition qubits.
    dist_hadamard(state, 0);
    dist_cnot(state, 0, 1);

    // Collective query: probability of measuring qubit 0 as |1>.
    double probability = 0.0;
    collective_get_qubit_probability(state, 0, &probability);

    if (mpi_is_root(ctx)) {
        printf("P(qubit 0 = 1): %.6f\n", probability);
    }

    // Cleanup.
    partition_state_free(state);
    mpi_bridge_free(ctx);
    mpi_bridge_finalize();

    return 0;
}
```

### Running with MPI

```bash
# Single machine, 4 processes
mpirun -np 4 ./my_quantum_app

# Cluster with hostfile
mpirun -np 64 --hostfile hosts.txt ./my_quantum_app

# SLURM cluster
srun -n 128 ./my_quantum_app
```

### Python API

```python
from moonlab.distributed import DistributedState, init_mpi, finalize_mpi

# Initialize MPI
init_mpi()

# Create distributed state
state = DistributedState(num_qubits=32)

# Operations work the same as local state
state.h(0)
state.cnot(0, 1)

# Measure (result broadcast to all ranks)
result = state.measure_all()

if state.rank == 0:
    print(f"Measurement result: {result}")

finalize_mpi()
```

## State Partitioning

### How It Works

The state vector is partitioned by the highest-order qubits:

```
32 qubits on 4 ranks:
- Rank 0: amplitudes[0 ... 2^30 - 1]           (top 2 bits = 00)
- Rank 1: amplitudes[2^30 ... 2×2^30 - 1]      (top 2 bits = 01)
- Rank 2: amplitudes[2×2^30 ... 3×2^30 - 1]    (top 2 bits = 10)
- Rank 3: amplitudes[3×2^30 ... 2^32 - 1]      (top 2 bits = 11)
```

### Local vs Global Qubits

```c
// For n qubits on P ranks (P = 2^k):
// - Qubits 0 to n-k-1: LOCAL (no communication needed)
// - Qubits n-k to n-1: PARTITION (require MPI communication)

partitioned_state_t* state = partition_state_create(ctx, 32, NULL);

// Local gate (fast, no communication)
dist_hadamard(state, 0);    // Qubit 0 is local

// Partition-qubit gate (triggers inter-rank exchange)
dist_hadamard(state, 31);   // Qubit 31 is a partition qubit

// Ask the partitioner which class a qubit falls in:
int is_partition = partition_is_partition_qubit(state, 31);  // 1 => needs exchange
```

## Communication Patterns

### Single-Qubit Global Gate

When applying a gate to a partition qubit, ranks exchange half their data. The public
`dist_gate_1q` / `dist_hadamard` / ... entry points do this for you; the sketch below
shows the mechanism using the real `partitioned_state_t` fields:

```c
// Gate on qubit k where k >= n - log2(P)
// Partner rank = my_rank XOR 2^(k - local_qubits)

void gate_on_partition_qubit(partitioned_state_t* state,
                             uint32_t target,
                             const gate_matrix_2x2_t* gate) {
    int rank = state->dist_ctx->rank;
    int partner = rank ^ (1 << (target - state->local_qubits));

    // Exchange the local shard with the partner (local_count amplitudes).
    MPI_Sendrecv(
        state->amplitudes,                         // send local shard
        state->local_count, MPI_DOUBLE_COMPLEX, partner, 0,
        state->recv_buffer,
        state->local_count, MPI_DOUBLE_COMPLEX, partner, 0,
        (MPI_Comm)state->dist_ctx->mpi_comm, MPI_STATUS_IGNORE
    );

    // Apply the 2x2 gate across the paired (local, remote) amplitudes.
    apply_gate_pairs(state, gate);
}
```

### Two-Qubit Global Gates

CNOT between global qubits may require 4-way communication:

```c
void cnot_on_partition_qubits(partitioned_state_t* state,
                              uint32_t control, uint32_t target) {
    // Determine communication pattern based on qubit positions
    int ctrl_bit = 1 << (control - state->local_qubits);
    int targ_bit = 1 << (target - state->local_qubits);

    // Ranks form groups of 4 for exchange
    int group_base = state->dist_ctx->rank & ~(ctrl_bit | targ_bit);
    int ranks[4] = {
        group_base,
        group_base | ctrl_bit,
        group_base | targ_bit,
        group_base | ctrl_bit | targ_bit
    };

    // Collective exchange within group
    MPI_Alltoall(/* ... */);

    // Apply CNOT to regrouped data
    apply_cnot_local(state);
}
```

## Configuration Options

### Runtime Configuration

Pass a `partition_config_t` to `partition_state_create` (or NULL for defaults). The fields
are declared in `distributed/state_partition.h`:

```c
partition_config_t config = {
    .use_aligned_memory   = 1,               // SIMD-aligned amplitude allocation
    .comm_buffer_size     = 64 * 1024 * 1024, // MPI exchange buffer bytes (0 = auto)
    .prefetch_remote      = 1,               // prefetch remote amplitudes
    .optimize_for_locality = 1               // choose partition to reduce exchange
};

partitioned_state_t* state = partition_state_create(ctx, 32, &config);

// Estimate the per-rank memory footprint before allocating:
size_t bytes_per_rank = partition_estimate_memory(32, mpi_get_size(ctx));
```

### GPU-Resident Shards

When `libquantumsim` is built with CUDA, each rank's local shard can live on a GPU. The
gate primitives dispatch to GPU kernels and the host buffer is used only as a staging area
for MPI exchange:

```c
// Returns NULL on a non-CUDA build or when no GPU is available.
partitioned_state_t* state = partition_state_create_gpu(ctx, 32, NULL);
```

## Performance Optimization

### Minimize Global Operations

```c
// SLOW: Many partition-qubit gates
for (int i = 0; i < 100; i++) {
    dist_hadamard(state, 31);  // Partition qubit, requires MPI exchange
}

// FAST: keep work on local qubits when possible
for (int i = 0; i < 100; i++) {
    dist_hadamard(state, i % state->local_qubits);  // Local, no communication
}
```

### Use Collective Operations

```c
// Per-qubit marginals each run a collective:
for (uint32_t q = 0; q < n; q++) {
    collective_get_qubit_probability(state, q, &probs[q]);  // one collective each
}

// The full distribution is a single gather to root (needs O(2^n) memory there):
collective_get_probabilities(state, full_probs);  // full_probs valid only on root
```

### Overlap Communication

```c
// Start async receive
MPI_Irecv(recv_buffer, count, MPI_DOUBLE_COMPLEX,
          partner, tag, MPI_COMM_WORLD, &recv_request);

// Compute on local data while waiting
apply_local_gates(state);

// Complete communication
MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

// Apply global operation
apply_global_gate(state, recv_buffer);
```

## Example: Distributed Grover Search

```c
#include "distributed/mpi_bridge.h"
#include "distributed/state_partition.h"
#include "distributed/distributed_gates.h"
#include "distributed/collective_ops.h"
#include <stdio.h>

int main(int argc, char** argv) {
    distributed_ctx_t* ctx = mpi_bridge_init(&argc, &argv, NULL);
    uint32_t n_qubits = 32;  // 4 billion states

    // Create distributed state and initialize the uniform superposition.
    partitioned_state_t* state = partition_state_create(ctx, n_qubits, NULL);
    partition_init_uniform(state);

    // Full Grover search: oracle + diffusion for the optimal iteration count
    // (num_iterations = 0 => dist_grover_optimal_iterations internally).
    uint64_t target = 123456789ULL;
    dist_grover_search(state, target, 0);

    // Measure (collective: every rank gets the same outcome).
    dist_measurement_result_t res;
    collective_measure_all(state, &res, NULL);

    if (mpi_is_root(ctx)) {
        printf("Found: %llu (target: %llu)\n",
               (unsigned long long)res.outcome, (unsigned long long)target);
    }

    partition_state_free(state);
    mpi_bridge_free(ctx);
    mpi_bridge_finalize();

    return 0;
}
```

The loop form is also available if you want to interleave progress reporting:
`dist_oracle_single(state, target)` then `dist_grover_diffusion(state)` per iteration,
or `dist_grover_iteration(state, target)` for the combined oracle+diffusion step.

## Cluster Configuration

### SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=quantum_sim
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00

module load openmpi

# Set OpenMP threads per MPI rank
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run simulation (64 total MPI ranks)
srun ./distributed_grover 34  # 34 qubits
```

### PBS/Torque Job Script

```bash
#!/bin/bash
#PBS -N quantum_sim
#PBS -l nodes=16:ppn=4
#PBS -l mem=128gb
#PBS -l walltime=04:00:00

cd $PBS_O_WORKDIR
module load openmpi

mpirun -np 64 ./distributed_grover 34
```

## Troubleshooting

### Common Issues

**MPI Initialization Failure**
```
Error: MPI_Init failed
```
Solution: Ensure MPI is properly installed and in PATH.

**Out of Memory**
```
Error: Cannot allocate distributed state
```
Solution: Increase nodes or reduce qubit count. Memory per rank = 2^n × 16 / num_ranks.

**Communication Timeout**
```
Error: MPI_Recv timed out
```
Solution: Check network connectivity, increase timeout, or reduce message size.

### Debugging

```bash
# Run with MPI debugging
mpirun -np 4 xterm -e gdb ./my_quantum_app

# Check for memory issues
mpirun -np 4 valgrind ./my_quantum_app

# Profile communication
mpirun -np 4 tau_exec ./my_quantum_app
```

## Performance Benchmarks

| Qubits | Ranks | Memory/Rank | Time (Grover) |
|--------|-------|-------------|---------------|
| 30 | 4 | 4 GB | 2.3 min |
| 32 | 16 | 4 GB | 8.5 min |
| 34 | 64 | 4 GB | 35 min |
| 36 | 256 | 4 GB | 2.5 hr |
| 38 | 1024 | 4 GB | 12 hr |

*Benchmarks on InfiniBand cluster with 100 Gbps interconnect*

## See Also

- [API Reference: MPI Bridge](../api/c/mpi-bridge.md)
- [Architecture: Distributed Design](../architecture/distributed-architecture.md)
- [Performance: Scaling Analysis](../performance/scaling-analysis.md)
