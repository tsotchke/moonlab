# Performance Documentation

Benchmarks, scaling analysis, and optimization guidance for Moonlab.

## Overview

This section provides comprehensive performance data for Moonlab across different hardware configurations, qubit counts, and algorithm types. Use this information to:

- Estimate simulation times for your workload
- Choose optimal backends (CPU vs GPU)
- Configure parallelization settings
- Understand memory requirements

## Quick Reference

### Simulation Capacity

| Hardware | Max Qubits (Full State) | Memory Required |
|----------|-------------------------|-----------------|
| 16 GB RAM | 30 | 16 GB |
| 32 GB RAM | 31 | 32 GB |
| 64 GB RAM | 32 | 64 GB |
| M1/M2/M3 Max (96 GB) | 33 | 64+ GB |

### Backend Selection

| Qubits | Recommended Backend | Reason |
|--------|---------------------|--------|
| 1-16 | CPU | GPU overhead exceeds benefit |
| 17-20 | Either | Test for your workload |
| 21-30 | GPU (Metal) | 10-100x speedup |
| 31+ | Tensor Network | State vector too large |

### Gate Performance (Apple M2 Pro)

| Operation | 20 Qubits | 24 Qubits | 28 Qubits |
|-----------|-----------|-----------|-----------|
| Single-qubit gate | 0.05 ms | 0.8 ms | 12 ms |
| Two-qubit gate | 0.1 ms | 1.6 ms | 25 ms |
| Measurement | 0.02 ms | 0.3 ms | 5 ms |

## Performance Sections

### [Benchmarks](benchmarks.md)

Comprehensive benchmark results including:
- Gate operation timings
- Algorithm execution times
- Backend comparisons
- Memory usage profiles

### Scaling Analysis

Understanding how performance scales:
- Gate complexity: O(2^n) for state vector methods
- Tensor networks: Polynomial for low-entanglement states
- Memory: 16 bytes × 2^n for double precision

### GPU Performance

Metal GPU acceleration details:
- Crossover points (when GPU beats CPU)
- Memory transfer overhead
- Kernel fusion benefits
- Multi-GPU support (future)

### Memory Requirements

Memory planning for large simulations:

| Qubits | State Vector | With Scratch | Tensor (χ=100) |
|--------|--------------|--------------|----------------|
| 20 | 16 MB | ~25 MB | ~2 MB |
| 24 | 256 MB | ~400 MB | ~4 MB |
| 28 | 4 GB | ~6 GB | ~8 MB |
| 32 | 64 GB | ~100 GB | ~16 MB |
| 50 | N/A | N/A | ~100 MB |
| 100 | N/A | N/A | ~400 MB |

## Performance Tips

### 1. Use GPU for Large States

```python
from moonlab import set_backend, QuantumState

# Automatic selection (recommended)
set_backend('auto')

# Force GPU for 18+ qubits
set_backend('metal')
state = QuantumState(24)  # Uses GPU
```

### 2. Batch Operations

```python
# Slow: Individual gate calls with syncs
for i in range(1000):
    state.h(0)
    result = state.measure(0)  # Forces sync

# Fast: Batch operations
for i in range(1000):
    state.h(0)
results = state.sample(1000)  # Single sync at end
```

### 3. Use Tensor Networks for Large Systems

```python
from moonlab.tensor_network import MPS

# For 50+ qubits with limited entanglement
mps = MPS(num_qubits=50, max_bond_dim=100)
mps.apply_layer(circuit)
```

### 4. Profile Your Code

```python
from moonlab import Profiler

with Profiler() as p:
    # Your simulation
    state = QuantumState(24)
    for _ in range(100):
        state.h(0)
        state.cnot(0, 1)

p.print_summary()
```

## Hardware Comparison

### Apple Silicon Performance

| Chip | CPU Cores | GPU Cores | Memory BW | Relative Speed |
|------|-----------|-----------|-----------|----------------|
| M1 | 8 | 8 | 68 GB/s | 1.0x |
| M1 Pro | 10 | 16 | 200 GB/s | 1.8x |
| M1 Max | 10 | 32 | 400 GB/s | 2.5x |
| M2 | 8 | 10 | 100 GB/s | 1.3x |
| M2 Pro | 12 | 19 | 200 GB/s | 2.2x |
| M2 Max | 12 | 38 | 400 GB/s | 3.2x |
| M3 Max | 16 | 40 | 400 GB/s | 3.8x |

### Intel/AMD Comparison

| CPU | Cores | SIMD | 24-Qubit Gate Time |
|-----|-------|------|-------------------|
| i7-12700K | 12 | AVX2 | 2.1 ms |
| i9-13900K | 24 | AVX-512 | 1.2 ms |
| Ryzen 9 7950X | 16 | AVX2 | 1.4 ms |
| Xeon W-3375 | 38 | AVX-512 | 0.9 ms |

## Algorithm Complexity

### State Vector Methods

| Algorithm | Gate Complexity | Circuit Depth | Total Complexity |
|-----------|-----------------|---------------|------------------|
| Grover | O(√N) | O(n) | O(n·2^n·√(2^n)) |
| QFT | O(n²) | O(n²) | O(n²·2^n) |
| VQE (per iteration) | O(terms) | O(depth) | O(terms·depth·2^n) |
| QAOA (per iteration) | O(edges) | O(p) | O(edges·p·2^n) |

### Tensor Network Methods

| Method | Time Complexity | Space Complexity |
|--------|-----------------|------------------|
| MPS simulation | O(n·χ³) | O(n·χ²) |
| DMRG sweep | O(n·χ³) | O(n·χ²) |
| MPO application | O(n·χ²·d²) | O(n·χ²·d²) |

Where χ is bond dimension, n is qubits, d is physical dimension.

## Optimization Checklist

Before running large simulations:

- [ ] Choose appropriate backend (CPU/GPU/tensor)
- [ ] Verify memory availability (`estimate_memory(n)`)
- [ ] Enable SIMD optimizations (`configure(simd_level='auto')`)
- [ ] Configure thread count (`configure(num_threads=N)`)
- [ ] Consider single precision if accuracy permits
- [ ] Profile with a smaller system first
- [ ] Use batching for repeated operations

## See Also

- [Benchmarks](benchmarks.md) - Detailed benchmark data
- [GPU Acceleration Guide](../guides/gpu-acceleration.md) - GPU setup
- [Performance Tuning Guide](../guides/performance-tuning.md) - Optimization
- [Tensor Network Engine](../architecture/tensor-network-engine.md) - MPS/DMRG internals

