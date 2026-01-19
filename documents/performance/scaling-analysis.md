# Scaling Analysis

How Moonlab's performance scales with system size, circuit depth, and hardware resources.

## State Vector Scaling

### Memory Scaling

State vector memory grows exponentially with qubit count:

$$\text{Memory} = 2^n \times 16 \text{ bytes}$$

where $n$ is the number of qubits (16 bytes for complex double).

| Qubits | Dimension | Memory | Notes |
|--------|-----------|--------|-------|
| 10 | 1,024 | 16 KB | L1 cache |
| 15 | 32,768 | 512 KB | L2 cache |
| 20 | 1,048,576 | 16 MB | L3 cache |
| 25 | 33,554,432 | 512 MB | Main memory |
| 28 | 268,435,456 | 4 GB | Typical RAM |
| 30 | 1,073,741,824 | 16 GB | High-end workstation |
| 32 | 4,294,967,296 | 64 GB | Server/M2 Ultra |

### Time Complexity per Gate

Single-qubit gate application:

$$T(n) = O(2^n) \times C$$

where $C$ is a small constant depending on the gate type.

**Measured Performance (M2 Ultra, 76 GPU cores):**

| Qubits | Hadamard (CPU) | Hadamard (GPU) | Speedup |
|--------|----------------|----------------|---------|
| 16 | 0.3 ms | 0.05 ms | 6x |
| 20 | 4.0 ms | 0.3 ms | 13x |
| 24 | 65 ms | 2.5 ms | 26x |
| 28 | 1.2 s | 35 ms | 34x |
| 30 | 5.8 s | 120 ms | 48x |

## Circuit Depth Scaling

### Linear Depth Scaling

For circuits of depth $d$ (number of gate layers):

$$T_{\text{circuit}} = d \times T_{\text{layer}}$$

**Measured Performance:**

```
Benchmark: Random circuit layers (RZ, RY, CNOT pattern)

10 qubits:
  10 layers:   0.8 ms
  25 layers:   2.0 ms
  50 layers:   4.1 ms
  100 layers:  8.2 ms
  500 layers:  41 ms

14 qubits:
  10 layers:   12 ms
  25 layers:   29 ms
  50 layers:   58 ms
  100 layers:  118 ms
  500 layers:  580 ms

16 qubits:
  10 layers:   48 ms
  25 layers:   120 ms
  50 layers:   240 ms
  100 layers:  480 ms
  500 layers:  2.4 s
```

### Gate Fusion Optimization

Consecutive single-qubit gates on the same qubit can be fused:

```c
// Before: 3 memory passes
apply_gate(state, 0, H);   // O(2^n)
apply_gate(state, 0, T);   // O(2^n)
apply_gate(state, 0, H);   // O(2^n)

// After: 1 memory pass
complex_t fused[2][2];
matrix_multiply(H, T, temp);
matrix_multiply(temp, H, fused);
apply_gate(state, 0, fused);  // O(2^n)
```

**Impact:** 3x speedup for fusible sequences.

## Tensor Network Scaling

### Bond Dimension Scaling

MPS memory scales polynomially:

$$\text{Memory} = O(n \times \chi^2 \times d^2)$$

where $n$ is sites, $\chi$ is bond dimension, $d=2$ for qubits.

| Sites | Bond $\chi$ | Memory | Equivalent Qubits |
|-------|-------------|--------|-------------------|
| 50 | 16 | 0.2 MB | ~16 qubits |
| 100 | 32 | 1.6 MB | ~20 qubits |
| 200 | 64 | 13 MB | ~24 qubits |
| 500 | 128 | 200 MB | ~28 qubits |
| 1000 | 256 | 1.6 GB | ~31 qubits |

### Entanglement Entropy Limits

Maximum bond dimension needed for exact simulation:

$$\chi_{\text{max}} = 2^{S}$$

where $S$ is the entanglement entropy across the cut (in bits).

| Entropy $S$ | Required $\chi$ | Notes |
|-------------|-----------------|-------|
| 0 | 1 | Product state |
| 4 | 16 | Low entanglement |
| 8 | 256 | Moderate entanglement |
| 12 | 4096 | High entanglement |
| 16 | 65536 | Very high entanglement |

### DMRG Scaling

Ground state search complexity:

$$T_{\text{DMRG}} = O(n \times \chi^3 \times d^3 \times S_{\text{sweeps}})$$

**Measured Performance (Heisenberg Chain):**

| Sites | Bond $\chi$ | Sweeps | Time | Energy Accuracy |
|-------|-------------|--------|------|-----------------|
| 20 | 32 | 10 | 0.5 s | $10^{-8}$ |
| 50 | 64 | 20 | 8 s | $10^{-10}$ |
| 100 | 128 | 30 | 2 min | $10^{-12}$ |
| 200 | 256 | 40 | 25 min | $10^{-12}$ |

## Distributed Scaling

### Strong Scaling

Fixed problem size, increasing MPI ranks:

| Qubits | 1 Rank | 4 Ranks | 16 Ranks | 64 Ranks | Efficiency |
|--------|--------|---------|----------|----------|------------|
| 28 | 1.0x | 3.2x | 11.5x | 38.2x | 60% |
| 30 | OOM | 1.0x | 3.5x | 12.8x | 53% |
| 32 | OOM | OOM | 1.0x | 3.6x | 56% |

### Weak Scaling

Fixed local memory (4 GB per rank), increasing problem size:

| Ranks | Total Qubits | Local Qubits | Time | Efficiency |
|-------|--------------|--------------|------|------------|
| 1 | 28 | 28 | 1.0x | 100% |
| 4 | 30 | 28 | 1.1x | 91% |
| 16 | 32 | 28 | 1.3x | 77% |
| 64 | 34 | 28 | 1.7x | 59% |

### Communication Overhead

| Operation | Data Volume | Latency Bound |
|-----------|-------------|---------------|
| Local gate | 0 | No |
| Global gate | $2^{n-p-1} \times 16$ bytes | Yes |
| Measurement | $O(P)$ | Yes |
| Normalization | $O(1)$ | Yes |

For 32 qubits on 16 ranks:
- Local gate: 0 bytes transferred
- Global gate: 2 GB per rank (half of local state)

## Hardware-Specific Optimizations

### Apple Silicon (M2 Ultra)

| Feature | Impact |
|---------|--------|
| Unified memory | Zero-copy GPU access |
| 192 GB RAM | 36+ qubits possible |
| 76 GPU cores | 50-100x speedup |
| AMX acceleration | 5-10x for matrix ops |
| NEON SIMD | 4-8x for CPU path |

### Multi-core Scaling (OpenMP)

| Cores | 20 Qubits | 24 Qubits | 28 Qubits |
|-------|-----------|-----------|-----------|
| 1 | 1.0x | 1.0x | 1.0x |
| 4 | 3.8x | 3.9x | 3.9x |
| 8 | 7.2x | 7.5x | 7.6x |
| 16 | 13.5x | 14.2x | 14.8x |
| 24 | 18.5x | 20.1x | 21.3x |

Efficiency drops for small state vectors due to overhead.

## Algorithm-Specific Scaling

### Grover's Algorithm

$$\text{Iterations} = \frac{\pi}{4} \sqrt{N} = \frac{\pi}{4} \sqrt{2^n}$$

| Qubits | Search Space | Iterations | Time (GPU) |
|--------|--------------|------------|------------|
| 16 | 65,536 | 201 | 60 ms |
| 20 | 1,048,576 | 804 | 1.2 s |
| 24 | 16,777,216 | 3,217 | 25 s |
| 28 | 268,435,456 | 12,868 | 8 min |

### VQE Energy Evaluation

$$T_{\text{VQE}} = T_{\text{circuit}} + T_{\text{expectation}}$$

For $k$ Pauli terms in Hamiltonian:

$$T_{\text{expectation}} = O(k \times 2^n)$$

**Optimization:** Group commuting terms to reduce measurements.

### QAOA

$$T_{\text{QAOA}} = p \times (T_{\text{cost}} + T_{\text{mixer}})$$

where $p$ is the number of QAOA layers.

| Problem Size | Layers $p$ | Qubits | Time | Approximation |
|--------------|------------|--------|------|---------------|
| 8 vertices | 2 | 8 | 0.5 s | 0.92 |
| 10 vertices | 3 | 10 | 2 s | 0.89 |
| 12 vertices | 4 | 12 | 12 s | 0.87 |

## Benchmark Tool

Run the included benchmark to measure scaling on your hardware:

```bash
# Build benchmark tools
make benchmarks

# Run scaling benchmark
./tools/benchmarks/scaling_benchmark --qubits 4 20 --output results.csv

# Run memory profiler
./tools/benchmarks/memory_profiler --qubits 4 24

# Analyze results
python tools/analyze_benchmark.py results.csv
```

### Sample Output

```
=== State Vector Scaling ===
  4 qubits: mem=0.0 MB, init=12 μs, layer=8 μs, 100 layers=0.8 ms
  6 qubits: mem=0.0 MB, init=15 μs, layer=18 μs, 100 layers=1.8 ms
  8 qubits: mem=0.0 MB, init=22 μs, layer=45 μs, 100 layers=4.5 ms
 10 qubits: mem=0.0 MB, init=38 μs, layer=120 μs, 100 layers=12 ms
 12 qubits: mem=0.1 MB, init=85 μs, layer=380 μs, 100 layers=38 ms
 14 qubits: mem=0.3 MB, init=280 μs, layer=1.2 ms, 100 layers=120 ms
 16 qubits: mem=1.0 MB, init=950 μs, layer=4.5 ms, 100 layers=450 ms
 18 qubits: mem=4.0 MB, init=3.8 ms, layer=18 ms, 100 layers=1.8 s
 20 qubits: mem=16 MB, init=15 ms, layer=72 ms, 100 layers=7.2 s

=== Tensor Network Scaling ===
  20 sites, χ= 8: init=45 μs, layer=180 μs, 100 layers=18 ms
  50 sites, χ=16: init=120 μs, layer=1.2 ms, 100 layers=120 ms
 100 sites, χ=32: init=380 μs, layer=8.5 ms, 100 layers=850 ms
 200 sites, χ=64: init=1.5 ms, layer=68 ms, 100 layers=6.8 s
```

## Recommendations

### Choosing Simulation Method

| Scenario | Recommended Approach |
|----------|----------------------|
| ≤20 qubits | State vector (CPU) |
| 21-28 qubits | State vector (GPU) |
| 29-32 qubits | Distributed or tensor network |
| >32 qubits | Tensor network (if low entanglement) |
| Low-depth circuits | State vector |
| Deep circuits | Tensor network with truncation |

### Memory Planning

```c
// Check available memory before allocation
size_t required = (1ULL << num_qubits) * sizeof(complex_t);
if (required > available_memory * 0.8) {
    // Consider distributed or tensor network
}
```

## See Also

- [GPU Performance](gpu-performance.md)
- [Memory Requirements](memory-requirements.md)
- [Distributed Simulation Guide](../guides/distributed-simulation.md)
- [Tensor Network Concepts](../concepts/tensor-networks.md)
