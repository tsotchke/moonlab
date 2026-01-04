# Guides

Task-oriented how-to guides for Moonlab.

## Overview

These guides provide step-by-step instructions for specific tasks. Unlike tutorials (which teach concepts), guides focus on achieving practical goals.

## Available Guides

### Setup & Configuration

| Guide | Description |
|-------|-------------|
| [Building from Source](building-from-source.md) | Compile Moonlab with custom options |
| [GPU Acceleration](gpu-acceleration.md) | Enable and configure Metal GPU support |

### Performance

| Guide | Description |
|-------|-------------|
| [Performance Tuning](performance-tuning.md) | Optimize simulation performance |

### Advanced Features

| Guide | Description |
|-------|-------------|
| [Noise Simulation](noise-simulation.md) | Add realistic noise to circuits |

## Quick Reference

### Enable GPU Acceleration

```python
from moonlab import set_backend, QuantumState

set_backend('metal')
state = QuantumState(24)  # Uses GPU automatically
```

### Add Noise to Circuits

```python
from moonlab import QuantumState
from moonlab.noise import DepolarizingChannel

state = QuantumState(4)
noise = DepolarizingChannel(error_rate=0.01)

state.h(0)
noise.apply(state, 0)  # Apply noise after gate
```

### Build with Custom Options

```bash
# Build with GPU support
cmake -DENABLE_GPU=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Build with MPI support
cmake -DENABLE_MPI=ON ..
make -j$(nproc)
```

## See Also

- [Tutorials](../tutorials/index.md) - Learn concepts step by step
- [API Reference](../api/index.md) - Complete function documentation
- [Troubleshooting](../troubleshooting.md) - Common problems and solutions

