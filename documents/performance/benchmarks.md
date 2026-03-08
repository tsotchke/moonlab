# Benchmark Results

Comprehensive performance measurements for Moonlab across hardware and configurations.

## Test Environment

### Primary Test System

| Component | Specification |
|-----------|--------------|
| CPU | Apple M2 Pro (12-core) |
| GPU | Apple M2 Pro (19-core GPU) |
| Memory | 32 GB Unified |
| OS | macOS 14.0 (Sonoma) |
| Compiler | Apple Clang 15.0 |
| Moonlab | v0.1.0 |

### Secondary Test Systems

- **Intel Desktop**: i9-13900K, 64 GB DDR5, Ubuntu 22.04
- **Cloud Instance**: AWS c6a.8xlarge (32 vCPU AMD EPYC), 64 GB

## Gate Benchmarks

### Single-Qubit Gates

Time to apply one Hadamard gate (averaged over 10,000 operations):

| Qubits | CPU (µs) | Metal GPU (µs) | Speedup |
|--------|----------|----------------|---------|
| 10 | 2.1 | 15.2 | 0.1x |
| 14 | 8.4 | 18.3 | 0.5x |
| 16 | 33.2 | 21.1 | 1.6x |
| 18 | 135 | 28.5 | 4.7x |
| 20 | 542 | 45.2 | 12x |
| 22 | 2,180 | 112 | 19x |
| 24 | 8,720 | 385 | 23x |
| 26 | 34,900 | 1,420 | 25x |
| 28 | 140,000 | 5,650 | 25x |

### Two-Qubit Gates

Time to apply one CNOT gate:

| Qubits | CPU (µs) | Metal GPU (µs) | Speedup |
|--------|----------|----------------|---------|
| 10 | 4.2 | 18.5 | 0.2x |
| 14 | 16.8 | 22.1 | 0.8x |
| 16 | 67.5 | 28.4 | 2.4x |
| 18 | 272 | 42.8 | 6.4x |
| 20 | 1,090 | 89.5 | 12x |
| 22 | 4,360 | 245 | 18x |
| 24 | 17,450 | 892 | 20x |
| 26 | 69,800 | 3,450 | 20x |
| 28 | 279,000 | 13,800 | 20x |

### Gate Type Comparison (24 qubits, GPU)

| Gate | Time (µs) | Relative |
|------|-----------|----------|
| X | 312 | 0.81x |
| Y | 318 | 0.83x |
| Z | 285 | 0.74x |
| H | 385 | 1.00x |
| S | 295 | 0.77x |
| T | 305 | 0.79x |
| Rx(θ) | 425 | 1.10x |
| Ry(θ) | 432 | 1.12x |
| Rz(θ) | 398 | 1.03x |
| CNOT | 892 | 2.32x |
| CZ | 875 | 2.27x |
| SWAP | 2,520 | 6.55x |
| Toffoli | 4,850 | 12.6x |

## Circuit Benchmarks

### Quantum Fourier Transform

Full n-qubit QFT circuit execution:

| Qubits | Gates | CPU (ms) | GPU (ms) | Speedup |
|--------|-------|----------|----------|---------|
| 8 | 36 | 0.12 | 0.95 | 0.1x |
| 12 | 78 | 1.8 | 2.1 | 0.9x |
| 16 | 136 | 15.2 | 4.8 | 3.2x |
| 18 | 171 | 58 | 8.5 | 6.8x |
| 20 | 210 | 245 | 18 | 14x |
| 22 | 253 | 1,020 | 52 | 20x |
| 24 | 300 | 4,150 | 185 | 22x |

### Random Circuit (Depth 10)

Random circuit with alternating single-qubit and CNOT layers:

| Qubits | Total Gates | CPU (ms) | GPU (ms) | Speedup |
|--------|-------------|----------|----------|---------|
| 16 | 160 | 18.5 | 5.2 | 3.6x |
| 18 | 180 | 72 | 9.8 | 7.3x |
| 20 | 200 | 295 | 22 | 13x |
| 22 | 220 | 1,180 | 65 | 18x |
| 24 | 240 | 4,750 | 235 | 20x |
| 26 | 260 | 19,000 | 920 | 21x |
| 28 | 280 | 76,000 | 3,650 | 21x |

### Bell State Creation

Create n/2 Bell pairs:

| Qubits | CPU (µs) | GPU (µs) |
|--------|----------|----------|
| 10 | 18 | 85 |
| 16 | 185 | 125 |
| 20 | 2,850 | 215 |
| 24 | 45,000 | 1,450 |
| 28 | 720,000 | 28,500 |

## Algorithm Benchmarks

### Grover's Search

Search for single marked item:

| Qubits | Database Size | Iterations | CPU (ms) | GPU (ms) |
|--------|---------------|------------|----------|----------|
| 8 | 256 | 12 | 2.8 | 3.2 |
| 10 | 1,024 | 25 | 15 | 5.8 |
| 12 | 4,096 | 50 | 125 | 22 |
| 14 | 16,384 | 100 | 1,050 | 95 |
| 16 | 65,536 | 201 | 8,200 | 425 |
| 18 | 262,144 | 402 | 65,000 | 2,100 |
| 20 | 1,048,576 | 804 | 520,000 | 12,500 |

### VQE Optimization

Single VQE iteration (4-qubit H₂ simulation):

| Component | Time (ms) |
|-----------|-----------|
| Ansatz preparation | 0.15 |
| Hamiltonian expectation (14 terms, 1000 shots) | 2.8 |
| Gradient computation (parameter shift) | 8.4 |
| Optimizer step | 0.05 |
| **Total per iteration** | **11.4** |

Full optimization (100 iterations): ~1.1 seconds

### QAOA for MaxCut

Random 3-regular graphs:

| Nodes | Qubits | Edges | Depth=1 (ms) | Depth=3 (ms) | Depth=5 (ms) |
|-------|--------|-------|--------------|--------------|--------------|
| 8 | 8 | 12 | 1.2 | 3.5 | 5.8 |
| 12 | 12 | 18 | 8.5 | 25 | 42 |
| 16 | 16 | 24 | 85 | 255 | 425 |
| 20 | 20 | 30 | 1,250 | 3,750 | 6,250 |
| 24 | 24 | 36 | 18,500 | 55,500 | 92,500 |

### DMRG Ground State

1D Heisenberg chain ground state:

| Sites | Bond Dim χ | Sweeps | CPU Time (s) |
|-------|------------|--------|--------------|
| 20 | 50 | 10 | 0.8 |
| 50 | 100 | 10 | 4.2 |
| 100 | 100 | 10 | 12.5 |
| 100 | 200 | 10 | 85 |
| 200 | 100 | 10 | 28 |
| 200 | 200 | 10 | 195 |

## Memory Benchmarks

### State Vector Memory Usage

| Qubits | Theoretical | Actual (with overhead) |
|--------|-------------|------------------------|
| 20 | 16 MB | 18.2 MB |
| 22 | 64 MB | 72.5 MB |
| 24 | 256 MB | 290 MB |
| 26 | 1 GB | 1.15 GB |
| 28 | 4 GB | 4.58 GB |
| 30 | 16 GB | 18.3 GB |

### MPS Memory Usage (χ = 100)

| Sites | Memory | vs State Vector |
|-------|--------|-----------------|
| 20 | 1.6 MB | 0.1x |
| 50 | 4.0 MB | N/A |
| 100 | 8.0 MB | N/A |
| 200 | 16 MB | N/A |
| 500 | 40 MB | N/A |

### GPU Memory Usage

| Qubits | State Vector | Scratch Buffers | Total |
|--------|--------------|-----------------|-------|
| 20 | 16 MB | 8 MB | 24 MB |
| 24 | 256 MB | 128 MB | 384 MB |
| 26 | 1 GB | 512 MB | 1.5 GB |
| 28 | 4 GB | 2 GB | 6 GB |
| 30 | 16 GB | 8 GB | 24 GB |

## SIMD Benchmarks

### SIMD Level Comparison (24 qubits, CPU)

| SIMD Level | H Gate (ms) | Relative |
|------------|-------------|----------|
| Scalar | 15.2 | 1.0x |
| SSE4.2 | 8.1 | 1.9x |
| AVX2 | 4.8 | 3.2x |
| AVX-512 | 2.9 | 5.2x |
| NEON (ARM) | 8.7 | 1.7x |

### Threading Scalability

24-qubit random circuit (200 gates) on 12-core M2 Pro:

| Threads | Time (ms) | Speedup | Efficiency |
|---------|-----------|---------|------------|
| 1 | 4,750 | 1.0x | 100% |
| 2 | 2,420 | 2.0x | 98% |
| 4 | 1,280 | 3.7x | 93% |
| 6 | 920 | 5.2x | 86% |
| 8 | 745 | 6.4x | 80% |
| 10 | 642 | 7.4x | 74% |
| 12 | 585 | 8.1x | 68% |

## Noise Simulation Benchmarks

### Noise Channel Application (24 qubits)

| Noise Type | Time (µs) | Relative to H Gate |
|------------|-----------|-------------------|
| Depolarizing | 425 | 1.1x |
| Amplitude Damping | 485 | 1.3x |
| Phase Damping | 395 | 1.0x |
| Custom Kraus (4 operators) | 1,250 | 3.2x |

### Noisy Circuit Overhead

Depolarizing noise (p=0.01) after each gate:

| Circuit | Clean (ms) | Noisy (ms) | Overhead |
|---------|------------|------------|----------|
| 20-qubit QFT | 18 | 28 | 1.6x |
| 24-qubit Random | 235 | 385 | 1.6x |
| 16-qubit VQE iter | 11.4 | 18.2 | 1.6x |

## Measurement Benchmarks

### Single-Shot Measurement

| Qubits | Time (µs) |
|--------|-----------|
| 10 | 5.2 |
| 16 | 28 |
| 20 | 185 |
| 24 | 2,850 |
| 28 | 45,000 |

### Sampling (1000 shots)

| Qubits | Time (ms) | Per-Shot (µs) |
|--------|-----------|---------------|
| 16 | 12.5 | 12.5 |
| 20 | 85 | 85 |
| 24 | 1,250 | 1,250 |
| 28 | 18,500 | 18,500 |

### Expectation Values

Pauli string expectation (analytical):

| Qubits | Pauli Length | Time (µs) |
|--------|--------------|-----------|
| 20 | 1 | 45 |
| 20 | 5 | 52 |
| 20 | 10 | 68 |
| 24 | 1 | 720 |
| 24 | 10 | 985 |

## Comparison with Other Simulators

### 20-Qubit Random Circuit (Depth 20)

| Simulator | Time (ms) | Notes |
|-----------|-----------|-------|
| Moonlab (GPU) | 22 | Metal backend |
| Moonlab (CPU) | 295 | SIMD optimized |
| Qiskit Aer | 185 | CPU only |
| Cirq | 425 | NumPy backend |
| QuTiP | 1,850 | General-purpose |
| cuQuantum | 8 | NVIDIA GPU |

### 24-Qubit QFT

| Simulator | Time (ms) |
|-----------|-----------|
| Moonlab (GPU) | 185 |
| Moonlab (CPU) | 4,150 |
| Qiskit Aer | 1,250 |
| PennyLane | 2,800 |

*Note: Comparisons are approximate; versions and configurations vary.*

## Reproducing Benchmarks

### Run Built-in Benchmarks

```bash
# Full benchmark suite
./bin/moonlab-benchmark

# Specific benchmarks
./bin/moonlab-benchmark --suite=gates --qubits=20,24,28
./bin/moonlab-benchmark --suite=algorithms --algorithm=grover
./bin/moonlab-benchmark --suite=comparison --backends=cpu,metal
```

### Python Benchmark Script

```python
from moonlab import Profiler, QuantumState, set_backend
import time

def benchmark_gates(n_qubits, n_gates=1000):
    """Benchmark gate operations."""
    state = QuantumState(n_qubits)

    start = time.perf_counter()
    for _ in range(n_gates):
        state.h(0)
    elapsed = time.perf_counter() - start

    return elapsed / n_gates * 1e6  # microseconds

# Run benchmarks
for backend in ['cpu', 'metal']:
    set_backend(backend)
    for n in [16, 20, 24]:
        us_per_gate = benchmark_gates(n)
        print(f"{backend:6} {n:2} qubits: {us_per_gate:.1f} µs/gate")
```

## See Also

- [Performance Index](index.md) - Overview and tips
- [GPU Acceleration](../guides/gpu-acceleration.md) - GPU configuration
- [Performance Tuning](../guides/performance-tuning.md) - Optimization guide

