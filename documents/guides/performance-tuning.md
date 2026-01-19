# Performance Tuning

Optimize Moonlab simulation performance for your workload.

## Overview

This guide covers optimization strategies for getting the best performance from Moonlab across different use cases and hardware configurations.

## Quick Wins

### 1. Enable GPU

```python
from moonlab import set_backend
set_backend('metal')  # 10-100x speedup for 20+ qubits
```

### 2. Use Appropriate Precision

```python
from moonlab import configure
configure(precision='single')  # 2x memory, ~1.5x speed for some operations
```

### 3. Batch Operations

```python
# Instead of individual calls
for _ in range(1000):
    state.h(0)

# Use batching
state.apply_circuit(circuit, repetitions=1000)
```

## Profiling

### Built-in Profiler

```python
from moonlab import Profiler, QuantumState

with Profiler() as p:
    state = QuantumState(20)
    for _ in range(100):
        state.h(0)
        state.cnot(0, 1)

p.print_summary()
# Output:
# Total time: 45.2 ms
# Gate applications: 200
# Avg gate time: 0.23 ms
# Memory peak: 16.5 MB
```

### Detailed Timing

```python
from moonlab import Profiler

p = Profiler(detailed=True)
p.start()

# Your code here

profile = p.stop()

# Per-operation breakdown
for op, stats in profile['operations'].items():
    print(f"{op}: {stats['total_ms']:.2f}ms ({stats['count']} calls)")
```

### Memory Profiling

```python
from moonlab import MemoryProfiler

with MemoryProfiler() as mp:
    state = QuantumState(28)
    state.h(0)

print(f"Peak usage: {mp.peak_mb:.1f} MB")
print(f"Current usage: {mp.current_mb:.1f} MB")
```

## CPU Optimization

### SIMD Configuration

```python
from moonlab import configure, simd_info

# Check available SIMD
info = simd_info()
print(f"SIMD level: {info['level']}")  # NEON, AVX2, AVX512

# Force specific level (testing/debugging)
configure(simd_level='AVX2')
```

### Threading

```python
from moonlab import configure
import os

# Set thread count (default: auto-detect)
configure(num_threads=8)

# Or via environment
os.environ['OMP_NUM_THREADS'] = '8'
```

### Cache Optimization

For memory-bound operations:

```python
from moonlab import configure

# Tune for cache size
configure(cache_block_size=256 * 1024)  # 256 KB blocks
```

## GPU Optimization

### Kernel Selection

```python
from moonlab import configure

# Enable kernel fusion
configure(gpu_kernel_fusion=True)

# Set optimal thread group size
configure(gpu_threadgroup_size=256)
```

### Memory Transfers

Minimize CPU-GPU transfers:

```python
from moonlab import gpu_context

with gpu_context() as ctx:
    state = ctx.create_state(24)

    # All operations on GPU
    for _ in range(1000):
        state.h(0)
        state.cnot(0, 1)

    # Single transfer at end
    result = state.measure_all()
```

### Batched Circuits

```python
from moonlab import batch_apply, QuantumState

# Run same circuit on multiple states
states = [QuantumState(20) for _ in range(100)]

def circuit(s):
    s.h(0).cnot(0, 1)
    return s

batch_apply(circuit, states)  # Parallelized across states
```

## Algorithm-Specific Tuning

### VQE

```python
from moonlab.algorithms import VQE

vqe = VQE(
    num_qubits=10,
    optimizer='adam',
    learning_rate=0.1,
    batch_size=50,        # Batch expectation measurements
    shots=1000,           # Balance accuracy vs speed
    parallel_terms=True   # Parallelize Hamiltonian terms
)
```

### QAOA

```python
from moonlab.algorithms import QAOA

qaoa = QAOA(
    num_qubits=20,
    depth=5,
    optimizer='COBYLA',
    max_iterations=100,
    parameter_sharing=True,  # Reduce optimization dimensions
    warm_start=True          # Use previous solution
)
```

### DMRG

```python
from moonlab.tensor_network import DMRG

dmrg = DMRG(
    max_bond_dim=100,
    bond_dim_schedule=[20, 40, 60, 80, 100],  # Gradual increase
    num_threads=8,           # Parallelize contractions
    use_svd_method='gesdd',  # Fastest SVD variant
    dense_cutoff=50          # Use dense for small tensors
)
```

## Memory Optimization

### State Vector Size

| Qubits | Memory (double) | Memory (single) |
|--------|-----------------|-----------------|
| 20 | 16 MB | 8 MB |
| 25 | 512 MB | 256 MB |
| 30 | 16 GB | 8 GB |

### Reduce Memory Usage

```python
from moonlab import configure

# Use single precision (half memory)
configure(precision='single')

# Enable memory-efficient mode
configure(memory_efficient=True)
```

### Memory Pools

```python
from moonlab import MemoryPool

# Pre-allocate memory pool
pool = MemoryPool(size_mb=1024)

# Use for temporary allocations
with pool:
    state = QuantumState(25)
    # Temporary allocations use pool
```

## Specific Workloads

### Many Small Circuits

```python
from moonlab import CircuitCache

# Cache compiled circuits
cache = CircuitCache(max_size=1000)

for params in parameter_sets:
    circuit = cache.get_or_compile(circuit_template, params)
    result = circuit.run()
```

### Few Large States

```python
from moonlab import configure

# Optimize for large state operations
configure(
    gpu_enabled=True,
    cache_block_size=1024 * 1024,  # 1 MB blocks
    prefetch=True
)
```

### Parameter Sweeps

```python
from moonlab import parallel_sweep

# Parallelize parameter sweep
results = parallel_sweep(
    circuit_fn,
    parameters,
    num_workers=8,
    backend='metal'
)
```

## Benchmarking

### Built-in Benchmarks

```bash
# Run standard benchmark suite
./bin/moonlab-benchmark

# Specific benchmark
./bin/moonlab-benchmark --suite=gates --qubits=20,25,30
```

### Custom Benchmarks

```python
from moonlab.benchmark import Benchmark, GateBenchmark

bench = GateBenchmark(qubits=[16, 18, 20, 22, 24])
results = bench.run()

bench.plot_results()
bench.save_results('benchmark.json')
```

### Comparison

```python
from moonlab.benchmark import compare_backends

results = compare_backends(
    circuit_fn,
    backends=['cpu', 'metal'],
    qubits=[18, 20, 22, 24, 26]
)

for n, data in results.items():
    print(f"{n} qubits: CPU={data['cpu']:.2f}s, GPU={data['metal']:.2f}s")
```

## Configuration Reference

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MOONLAB_BACKEND` | Default backend (cpu, metal, auto) |
| `MOONLAB_GPU_THRESHOLD` | Min qubits for GPU |
| `OMP_NUM_THREADS` | OpenMP thread count |
| `MOONLAB_CACHE_SIZE` | Circuit cache size |

### Configuration Options

```python
from moonlab import configure

configure(
    # Backend
    backend='auto',
    gpu_threshold=18,

    # CPU
    num_threads=8,
    simd_level='auto',

    # GPU
    gpu_kernel_fusion=True,
    gpu_threadgroup_size=256,

    # Memory
    precision='double',
    memory_efficient=False,
    cache_size_mb=256,

    # Algorithms
    default_shots=1000,
    parallel_hamiltonian=True
)
```

## Platform-Specific Tips

### Apple Silicon (M1/M2/M3)

- GPU is almost always faster for 18+ qubits
- Unified memory reduces transfer overhead
- Use `metal` backend by default

### Intel Mac

- CPU often competitive with GPU
- Use AVX2/AVX-512 SIMD
- Profile to determine crossover point

### Linux Server

- Maximum thread count for CPU
- Consider MPI for multi-node
- Tune NUMA settings for multi-socket

## See Also

- [GPU Acceleration Guide](gpu-acceleration.md) - GPU-specific optimization
- [Benchmarks](../performance/benchmarks.md) - Performance data
- [Architecture](../architecture/index.md) - Implementation details

