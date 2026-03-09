# GPU Acceleration Guide

Configure and optimize Metal GPU acceleration for Moonlab.

## Overview

Moonlab uses Apple's Metal framework to accelerate quantum simulation on Apple Silicon. GPU acceleration can provide 10-100x speedups for large state vectors.

## Requirements

### Hardware

- Mac with Apple Silicon (M1, M2, M3 family) or AMD GPU
- Minimum 8 GB unified memory (16+ GB recommended for 25+ qubits)

### Software

- macOS 12.0 (Monterey) or later
- Moonlab built with GPU support

## Checking GPU Availability

### Python

```python
from moonlab import gpu_info, gpu_available

# Check if GPU is available
if gpu_available():
    info = gpu_info()
    print(f"Device: {info['device_name']}")
    print(f"Memory: {info['memory_gb']:.1f} GB")
    print(f"GPU Cores: {info['gpu_cores']}")
else:
    print("GPU not available")
```

### C

```c
#include "gpu_metal.h"

if (gpu_metal_available()) {
    gpu_info_t info;
    gpu_metal_get_info(&info);

    printf("Device: %s\n", info.device_name);
    printf("Memory: %.1f GB\n", info.memory_gb);
}
```

## Enabling GPU

### Python

```python
from moonlab import set_backend, QuantumState

# Option 1: Set global backend
set_backend('metal')
state = QuantumState(24)  # Uses GPU

# Option 2: Per-state backend
state = QuantumState(24, backend='metal')

# Option 3: Automatic selection
set_backend('auto')  # Uses GPU for large states
```

### C

```c
#include "quantum_sim.h"
#include "gpu_metal.h"

// Initialize GPU
gpu_metal_init();

// Create GPU-accelerated state
quantum_state_t* state = quantum_state_create_gpu(24);

// Operations automatically use GPU
quantum_state_h(state, 0);
quantum_state_cnot(state, 0, 1);

// Cleanup
quantum_state_destroy(state);
gpu_metal_cleanup();
```

## Configuration

### GPU Threshold

Set minimum qubits for GPU acceleration:

```python
from moonlab import configure

# Use GPU only for 20+ qubits (default: 18)
configure(gpu_threshold=20)
```

```c
qsim_config_set_int("gpu.threshold", 20);
```

### Memory Limits

```python
from moonlab import configure, gpu_memory_info

# Check available memory
mem = gpu_memory_info()
print(f"Available: {mem['available_gb']:.1f} GB")

# Set maximum GPU memory usage
configure(gpu_max_memory_gb=8.0)
```

### Thread Configuration

```python
from moonlab import configure

# Set GPU thread group size (advanced)
configure(gpu_threadgroup_size=256)
```

## Performance Optimization

### Batch Operations

Minimize CPU-GPU transfers by batching:

```python
from moonlab import QuantumState, batch_apply

# Inefficient: Many small transfers
state = QuantumState(24, backend='metal')
for _ in range(100):
    state.h(0)  # Each gate triggers sync

# Efficient: Batch operations
def circuit(state):
    for _ in range(100):
        state.h(0)
    return state

batch_apply(circuit, [state])  # Single transfer
```

### Keep State on GPU

```python
from moonlab import QuantumState, gpu_context

# Create persistent GPU context
with gpu_context() as ctx:
    state = ctx.create_state(26)

    # All operations stay on GPU
    for _ in range(1000):
        state.h(0)
        state.cnot(0, 1)

    # Only transfer at measurement
    result = state.measure_all()
```

### Kernel Fusion

Enable automatic gate fusion:

```python
from moonlab import configure, QuantumState

configure(gpu_kernel_fusion=True)

state = QuantumState(24, backend='metal')
# H-T-S-H sequence fused into single kernel
state.h(0).t(0).s(0).h(0)
```

## Profiling

### GPU Profiler

```python
from moonlab import QuantumState, GPUProfiler

profiler = GPUProfiler()
profiler.start()

state = QuantumState(24, backend='metal')
for _ in range(10):
    for q in range(24):
        state.h(q)
    for q in range(23):
        state.cnot(q, q + 1)

profile = profiler.stop()

print(f"Total GPU time: {profile['total_time_ms']:.2f} ms")
print(f"Kernel launches: {profile['kernel_count']}")
print(f"Memory transfers: {profile['transfer_count']}")
print(f"Peak memory: {profile['peak_memory_mb']:.1f} MB")
```

### Metal System Trace

For detailed analysis, use Xcode's Metal System Trace:

1. Open Instruments.app
2. Choose "Metal System Trace" template
3. Run your simulation
4. Analyze GPU utilization and memory

## Memory Management

### Memory Estimation

```python
from moonlab import estimate_memory, gpu_memory_info

n_qubits = 28
required = estimate_memory(n_qubits)
available = gpu_memory_info()['available_gb']

print(f"Required: {required:.1f} GB")
print(f"Available: {available:.1f} GB")

if required > available:
    print("Consider: fewer qubits, CPU backend, or hybrid mode")
```

### Memory Requirements

| Qubits | State Vector | With Overhead |
|--------|--------------|---------------|
| 20 | 16 MB | ~20 MB |
| 24 | 256 MB | ~300 MB |
| 26 | 1 GB | ~1.2 GB |
| 28 | 4 GB | ~4.5 GB |
| 30 | 16 GB | ~18 GB |

### Explicit Memory Control

```python
from moonlab import QuantumState, gpu_transfer

# Create on CPU
state = QuantumState(24, backend='cpu')
state.h(0).cnot(0, 1)

# Transfer to GPU for heavy computation
gpu_state = gpu_transfer(state, 'to_gpu')

for _ in range(1000):
    gpu_state.rx(0, 0.1)

# Transfer back for measurement
cpu_state = gpu_transfer(gpu_state, 'to_cpu')
result = cpu_state.measure_all()
```

## When to Use GPU

### GPU Recommended

- 18+ qubits
- Many gate operations
- Deep circuits
- Batch simulations
- Parameter optimization (VQE, QAOA)

### CPU Preferred

- <16 qubits (GPU overhead dominates)
- Single-shot simulations
- Frequent state inspection
- Interactive development
- Memory-constrained systems

### Auto Selection

```python
from moonlab import QuantumState

# Let Moonlab decide
state = QuantumState(n_qubits, backend='auto')
print(f"Selected: {state.backend}")  # 'metal' or 'cpu'
```

## Troubleshooting

### GPU Not Detected

```python
from moonlab import gpu_diagnose

result = gpu_diagnose()
if result['issue']:
    print(f"Issue: {result['issue']}")
    print(f"Solution: {result['solution']}")
```

Common issues:
- macOS version too old (need 12.0+)
- Running in VM without GPU passthrough
- Moonlab built without GPU support

### Out of Memory

```python
from moonlab import gpu_memory_info, estimate_memory

n = 28
needed = estimate_memory(n)
available = gpu_memory_info()['available_gb']

if needed > available:
    print("Options:")
    print("1. Reduce qubit count")
    print("2. Use CPU backend")
    print("3. Close other GPU applications")
    print("4. Use tensor network methods")
```

### Poor Performance

Check for:

1. **Too few qubits**: GPU overhead exceeds benefit
   ```python
   configure(gpu_threshold=20)  # Adjust threshold
   ```

2. **Frequent transfers**: Keep data on GPU
   ```python
   with gpu_context() as ctx:
       # All operations on GPU
   ```

3. **Small batch sizes**: Batch operations together

### Incorrect Results

```python
# Enable verification mode (slower but checks correctness)
configure(gpu_verify=True)

state = QuantumState(20, backend='metal')
state.h(0)

# Results compared with CPU reference
```

## Benchmarking

### Compare CPU vs GPU

```python
import time
from moonlab import QuantumState, set_backend

def benchmark(n_qubits, backend, depth=10):
    set_backend(backend)
    state = QuantumState(n_qubits)

    start = time.time()
    for _ in range(depth):
        for q in range(n_qubits):
            state.h(q)
        for q in range(n_qubits - 1):
            state.cnot(q, q + 1)
    return time.time() - start

for n in [16, 18, 20, 22, 24]:
    cpu_time = benchmark(n, 'cpu')
    gpu_time = benchmark(n, 'metal')
    speedup = cpu_time / gpu_time

    print(f"{n} qubits: CPU={cpu_time:.3f}s, GPU={gpu_time:.3f}s, "
          f"Speedup={speedup:.1f}x")
```

## See Also

- [Tutorial: GPU Acceleration](../tutorials/09-gpu-acceleration.md) - Step-by-step tutorial
- [GPU Pipeline Architecture](../architecture/gpu-pipeline.md) - Implementation details
- [C API: GPU Metal](../api/c/gpu-metal.md) - Low-level API reference
- [Performance Tuning](performance-tuning.md) - General optimization

