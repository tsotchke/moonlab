# Distributed Simulation Guide

Scale quantum simulations beyond single-machine memory limits using MPI-based distributed computing.

## Overview

Moonlab's distributed backend partitions the state vector across multiple nodes, enabling simulation of larger quantum systems than a single machine can handle. With 1024 MPI ranks, you can add approximately 10 extra qubits to your simulation capacity.

## When to Use Distributed Simulation

| Scenario | Single Node | Distributed |
|----------|-------------|-------------|
| ≤28 qubits | ✓ Recommended | Unnecessary overhead |
| 29-32 qubits | Requires 128GB+ RAM | ✓ Split across nodes |
| 33-38 qubits | Impossible | ✓ Required |
| >38 qubits | - | Requires 1000+ nodes |

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

```bash
# Configure with MPI
./configure --enable-mpi

# Or use make directly
make MPI=1

# Verify MPI support
./bin/moonlab --version
# Output should include: MPI: enabled
```

## Basic Usage

### C API

```c
#include "distributed/mpi_bridge.h"
#include "quantum/state.h"

int main(int argc, char** argv) {
    // Initialize MPI
    mpi_quantum_init(&argc, &argv);

    int rank = mpi_quantum_rank();
    int size = mpi_quantum_size();

    if (rank == 0) {
        printf("Running on %d MPI ranks\n", size);
    }

    // Create distributed state (32 qubits across all ranks)
    distributed_state_t* state = distributed_state_create(32);

    // Apply gates (automatically handles communication)
    distributed_hadamard(state, 0);
    distributed_cnot(state, 0, 1);

    // Global operations require synchronization
    double probability = distributed_measure_probability(state, 0);

    if (rank == 0) {
        printf("P(qubit 0 = 1): %.6f\n", probability);
    }

    // Cleanup
    distributed_state_destroy(state);
    mpi_quantum_finalize();

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
// - Qubits n-k to n-1: GLOBAL (require MPI communication)

distributed_state_t* state = distributed_state_create(32);

// Local gate (fast, no communication)
distributed_hadamard(state, 0);   // Qubit 0 is local

// Global gate (requires MPI exchange)
distributed_hadamard(state, 31);  // Qubit 31 is global
```

## Communication Patterns

### Single-Qubit Global Gate

When applying a gate to a global qubit, ranks exchange half their data:

```c
// Gate on qubit k where k >= n - log2(P)
// Partner rank = my_rank XOR 2^(k - (n - log2(P)))

void distributed_gate_global(distributed_state_t* state,
                             int target,
                             const complex_t gate[2][2]) {
    int partner = state->rank ^ (1 << (target - state->local_qubits));

    // Exchange half of local state with partner
    MPI_Sendrecv(
        state->local_amplitudes + state->local_size/2,  // send upper half
        state->local_size/2, MPI_DOUBLE_COMPLEX, partner, 0,
        state->recv_buffer,
        state->local_size/2, MPI_DOUBLE_COMPLEX, partner, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    // Apply gate using local and received data
    apply_gate_pairs(state, gate);
}
```

### Two-Qubit Global Gates

CNOT between global qubits may require 4-way communication:

```c
void distributed_cnot_global(distributed_state_t* state,
                             int control, int target) {
    // Determine communication pattern based on qubit positions
    int ctrl_bit = 1 << (control - state->local_qubits);
    int targ_bit = 1 << (target - state->local_qubits);

    // Ranks form groups of 4 for exchange
    int group_base = state->rank & ~(ctrl_bit | targ_bit);
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

### Environment Variables

```bash
# Buffer size for MPI messages (default: auto)
export MOONLAB_MPI_BUFFER_SIZE=67108864  # 64 MB

# Enable/disable non-blocking communication
export MOONLAB_MPI_ASYNC=1

# Overlap computation with communication
export MOONLAB_MPI_OVERLAP=1
```

### Runtime Configuration

```c
distributed_config_t config = {
    .buffer_size = 64 * 1024 * 1024,  // 64 MB
    .use_async = true,
    .overlap_compute = true,
    .compression = COMPRESS_NONE  // or COMPRESS_THRESHOLD
};

distributed_state_t* state = distributed_state_create_config(32, &config);
```

## Performance Optimization

### Minimize Global Operations

```c
// SLOW: Many global gates
for (int i = 0; i < 100; i++) {
    distributed_hadamard(state, 31);  // Global, requires MPI
}

// FAST: Batch operations, use local qubits when possible
for (int i = 0; i < 100; i++) {
    distributed_hadamard(state, i % state->local_qubits);  // Local
}
```

### Use Collective Operations

```c
// Instead of individual measurements
for (int q = 0; q < n; q++) {
    probs[q] = distributed_measure_probability(state, q);  // Multiple collectives
}

// Use batched measurement
distributed_measure_probabilities_all(state, probs);  // Single collective
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
#include "algorithms/grover.h"

int main(int argc, char** argv) {
    mpi_quantum_init(&argc, &argv);

    int rank = mpi_quantum_rank();
    int n_qubits = 32;  // 4 billion states

    // Create distributed state
    distributed_state_t* state = distributed_state_create(n_qubits);

    // Initialize superposition
    for (int q = 0; q < n_qubits; q++) {
        distributed_hadamard(state, q);
    }

    // Grover iterations
    uint64_t target = 123456789ULL;
    int iterations = (int)(M_PI / 4 * sqrt(1ULL << n_qubits));

    for (int iter = 0; iter < iterations; iter++) {
        // Oracle (mark target state)
        distributed_oracle_mark(state, target);

        // Diffusion operator
        distributed_grover_diffusion(state);

        if (rank == 0 && iter % 100 == 0) {
            printf("Iteration %d/%d\n", iter, iterations);
        }
    }

    // Measure
    uint64_t result = distributed_measure_all(state);

    if (rank == 0) {
        printf("Found: %llu (target: %llu)\n", result, target);
    }

    distributed_state_destroy(state);
    mpi_quantum_finalize();

    return 0;
}
```

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
