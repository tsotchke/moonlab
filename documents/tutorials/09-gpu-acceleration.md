# Tutorial 09: GPU Acceleration

Speed up quantum simulation with Apple Metal.

**Duration**: 45 minutes
**Prerequisites**: Mac with Apple Silicon, [Tutorial 01](01-hello-quantum.md)
**Difficulty**: Advanced

## Learning Objectives

By the end of this tutorial, you will:

- Enable and configure GPU acceleration
- Benchmark CPU vs. GPU performance
- Understand when GPU acceleration helps
- Optimize simulations for GPU

## Why GPU?

Quantum simulation is embarrassingly parallel:
- State vector: $2^n$ independent amplitudes
- Gate application: Process amplitude pairs in parallel
- Measurement: Parallel probability accumulation

GPUs provide massive parallelism:
- Apple M2 Max: 38 GPU cores, 400+ GB/s bandwidth
- Speedups of 10-100x possible for large states

## Prerequisites

Moonlab GPU acceleration requires:
- Mac with Apple Silicon (M1, M2, M3 family)
- macOS 12.0 (Monterey) or later
- Moonlab built with Metal support

Check GPU availability:

```python
from moonlab import gpu_info

info = gpu_info()
print(f"GPU available: {info['available']}")
print(f"Device: {info['device_name']}")
print(f"Memory: {info['memory_gb']:.1f} GB")
print(f"GPU Cores: {info['gpu_cores']}")
```

## Step 1: Enable GPU Mode

### Python

```python
from moonlab import QuantumState, set_backend

# Enable GPU backend
set_backend('metal')  # Options: 'cpu', 'metal', 'auto'

# Create state (will use GPU)
state = QuantumState(20)

# Check backend
print(f"Backend: {state.backend}")
```

### C

```c
#include "quantum_sim.h"
#include "gpu_metal.h"

int main() {
    // Initialize GPU
    if (!gpu_metal_available()) {
        printf("GPU not available, using CPU\n");
        return 1;
    }

    gpu_metal_init();

    // Create GPU-accelerated state
    quantum_state_t* state = quantum_state_create_gpu(20);

    // Operations automatically use GPU
    quantum_state_h(state, 0);
    quantum_state_cnot(state, 0, 1);

    // Cleanup
    quantum_state_destroy(state);
    gpu_metal_cleanup();

    return 0;
}
```

## Step 2: Basic Benchmark

```python
import time
from moonlab import QuantumState, set_backend
import numpy as np

def benchmark_circuit(n_qubits, backend, depth=10):
    """Benchmark a random circuit."""
    set_backend(backend)
    state = QuantumState(n_qubits)

    start = time.time()

    # Apply random circuit
    for d in range(depth):
        for q in range(n_qubits):
            state.ry(q, np.random.uniform(0, 2*np.pi))
        for q in range(n_qubits - 1):
            state.cnot(q, q + 1)

    elapsed = time.time() - start
    return elapsed

# Compare CPU vs GPU
print("Benchmark: Random Circuit (depth=10)")
print("-" * 40)

for n in [16, 18, 20, 22, 24]:
    cpu_time = benchmark_circuit(n, 'cpu')
    gpu_time = benchmark_circuit(n, 'metal')
    speedup = cpu_time / gpu_time

    print(f"{n} qubits: CPU={cpu_time:.3f}s, GPU={gpu_time:.3f}s, "
          f"Speedup={speedup:.1f}x")
```

**Example Output**:
```
Benchmark: Random Circuit (depth=10)
----------------------------------------
16 qubits: CPU=0.012s, GPU=0.015s, Speedup=0.8x
18 qubits: CPU=0.048s, GPU=0.021s, Speedup=2.3x
20 qubits: CPU=0.195s, GPU=0.035s, Speedup=5.6x
22 qubits: CPU=0.782s, GPU=0.089s, Speedup=8.8x
24 qubits: CPU=3.215s, GPU=0.245s, Speedup=13.1x
```

## Step 3: Memory Management

GPU memory is separate from CPU memory:

```python
from moonlab import QuantumState, gpu_memory_info

# Check available GPU memory
mem_info = gpu_memory_info()
print(f"GPU memory: {mem_info['total_gb']:.1f} GB")
print(f"Available: {mem_info['available_gb']:.1f} GB")

# Estimate memory for state
def state_memory_gb(n_qubits):
    """Memory needed for n-qubit state (complex128)."""
    return (2 ** n_qubits) * 16 / (1024 ** 3)

# Find maximum qubit count
for n in range(20, 35):
    mem_needed = state_memory_gb(n)
    if mem_needed > mem_info['available_gb']:
        print(f"Maximum qubits: {n-1} ({state_memory_gb(n-1):.1f} GB)")
        break
```

### Explicit Memory Control

```python
from moonlab import QuantumState, gpu_sync, gpu_transfer

# Create state on CPU, transfer to GPU
state = QuantumState(24, backend='cpu')
state.h(0).cnot(0, 1)  # Prepare on CPU

# Transfer to GPU
gpu_state = gpu_transfer(state, 'to_gpu')

# Heavy computation on GPU
for layer in range(100):
    for q in range(24):
        gpu_state.ry(q, 0.1)
    for q in range(23):
        gpu_state.cnot(q, q + 1)

# Transfer back to CPU for measurement
cpu_state = gpu_transfer(gpu_state, 'to_cpu')
result = cpu_state.measure_all()
```

## Step 4: Batch Operations

Maximize GPU utilization with batch operations:

```python
from moonlab import QuantumState, batch_apply

# Prepare multiple states
states = [QuantumState(20) for _ in range(100)]

# Define circuit as a function
def circuit(state):
    state.h(0)
    for q in range(19):
        state.cnot(q, q + 1)
    return state

# Batch apply (GPU parallelizes across states)
start = time.time()
batch_apply(circuit, states)
batch_time = time.time() - start

# Compare with sequential
start = time.time()
for state in states:
    circuit(state)
seq_time = time.time() - start

print(f"Sequential: {seq_time:.3f}s")
print(f"Batch: {batch_time:.3f}s")
print(f"Speedup: {seq_time/batch_time:.1f}x")
```

## Step 5: SIMD + GPU Hybrid

For the best performance, combine SIMD and GPU:

```python
from moonlab import configure_acceleration

# Auto-select best configuration
config = configure_acceleration(
    n_qubits=26,
    circuit_depth=50,
    prefer_gpu=True
)

print(f"Selected backend: {config['backend']}")
print(f"SIMD level: {config['simd_level']}")
print(f"Thread count: {config['threads']}")

# Apply configuration
state = QuantumState(26, config=config)
```

## Step 6: Profiling GPU Performance

```python
from moonlab import QuantumState, GPUProfiler

# Enable profiling
profiler = GPUProfiler()
profiler.start()

# Run circuit
state = QuantumState(24, backend='metal')
for _ in range(10):
    for q in range(24):
        state.h(q)
    for q in range(23):
        state.cnot(q, q + 1)

# Get profile
profile = profiler.stop()

print(f"Total GPU time: {profile['total_time_ms']:.2f} ms")
print(f"Kernel launches: {profile['kernel_count']}")
print(f"Memory transfers: {profile['transfer_count']}")
print(f"Peak memory: {profile['peak_memory_mb']:.1f} MB")
print("\nKernel breakdown:")
for kernel, time_ms in profile['kernels'].items():
    print(f"  {kernel}: {time_ms:.2f} ms")
```

**Example Output**:
```
Total GPU time: 45.32 ms
Kernel launches: 470
Memory transfers: 2
Peak memory: 256.0 MB

Kernel breakdown:
  hadamard_kernel: 12.45 ms
  cnot_kernel: 28.67 ms
  reduction_kernel: 4.20 ms
```

## Step 7: Advanced GPU Features

### Fused Gate Kernels

```python
from moonlab import QuantumState, enable_fusion

# Enable kernel fusion
enable_fusion(True)

state = QuantumState(24, backend='metal')

# These gates will be fused into single kernel
state.h(0).t(0).s(0).h(0)  # Fused to single U3

# Verify fusion
from moonlab import get_kernel_stats
stats = get_kernel_stats()
print(f"Kernels without fusion: 4")
print(f"Kernels with fusion: {stats['kernels_executed']}")
```

### Persistent State

Keep state on GPU across operations:

```python
from moonlab import QuantumState, gpu_context

# Create persistent GPU context
with gpu_context() as ctx:
    state = ctx.create_state(26)

    # All operations stay on GPU
    for iteration in range(1000):
        state.h(0)
        state.cnot(0, 1)

    # Only transfer at the end
    probs = state.probabilities()  # Single transfer
```

## When to Use GPU

### GPU Recommended
- $n \geq 18$ qubits
- Many gate applications
- Batch simulations
- Circuit optimization

### CPU Preferred
- $n < 16$ qubits (GPU overhead dominates)
- Single-shot simulations
- Frequent state inspection
- Interactive development

### Auto-Select

```python
from moonlab import QuantumState, auto_backend

# Let Moonlab choose
state = QuantumState(n_qubits, backend='auto')
print(f"Auto-selected: {state.backend}")
```

## Troubleshooting

### GPU Not Detected

```python
from moonlab import gpu_diagnose

result = gpu_diagnose()
if result['issue']:
    print(f"Issue: {result['issue']}")
    print(f"Solution: {result['solution']}")
else:
    print("GPU is properly configured")
```

### Out of Memory

```python
from moonlab import estimate_memory, gpu_memory_info

n = 28
required = estimate_memory(n)
available = gpu_memory_info()['available_gb']

if required > available:
    print(f"Need {required:.1f} GB, have {available:.1f} GB")
    print("Options:")
    print("1. Reduce qubit count")
    print("2. Use CPU backend")
    print("3. Use hybrid mode (partial GPU)")
```

## Exercises

### Exercise 1: Find Crossover Point

At what qubit count does GPU become faster than CPU for your hardware?

### Exercise 2: VQE on GPU

Run the VQE tutorial using GPU acceleration. How much faster is it?

### Exercise 3: Maximum Qubits

What's the largest system you can simulate on your GPU?

### Exercise 4: Custom Kernel

Write a custom Metal kernel for a specific gate sequence and compare performance.

## Key Takeaways

1. **GPU acceleration** provides 10-100x speedup for large systems
2. **Memory bandwidth** is often the bottleneck
3. **Batch operations** maximize GPU utilization
4. **Kernel fusion** reduces overhead
5. **Auto-backend** helps choose optimal configuration

## Congratulations!

You've completed all the Moonlab tutorials! You now know how to:

- Create and manipulate quantum states
- Apply quantum gates and build circuits
- Create entanglement and Bell states
- Implement quantum algorithms (Grover, VQE, QAOA)
- Simulate large systems with tensor networks
- Accelerate with GPU

## What's Next?

- [Algorithm Deep Dives](../algorithms/index.md) - More detailed algorithm documentation
- [API Reference](../api/index.md) - Complete function reference
- [Examples](../examples/index.md) - More code samples
- [Contributing](../contributing/index.md) - Help improve Moonlab

## Further Reading

- [GPU Acceleration Guide](../guides/gpu-acceleration.md) - Complete GPU guide
- [Performance Tuning](../guides/performance-tuning.md) - Optimization tips
- [C API: GPU Metal](../api/c/gpu-metal.md) - Low-level GPU API

