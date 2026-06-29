# Archived Moonlab Documentation: Guides

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
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
| [Gauge-aware warmstart](gauge-aware-warmstart.md) | Project a CA-MPS state into the +1 eigenspace of any commuting Pauli stabilizer subgroup -- LGT, surface/toric/repetition codes, abelian symmetry sectors (since v0.2.1) |

## Quick Reference

### Enable GPU Acceleration

[archived fence delimiter: ```python]
from moonlab import set_backend, QuantumState

set_backend('metal')
state = QuantumState(24)  # Uses GPU automatically
[archived fence delimiter: ```]

### Add Noise to Circuits

[archived fence delimiter: ```python]
from moonlab import QuantumState
from moonlab.noise import DepolarizingChannel

state = QuantumState(4)
noise = DepolarizingChannel(error_rate=0.01)

state.h(0)
noise.apply(state, 0)  # Apply noise after gate
[archived fence delimiter: ```]

### Build with Custom Options

[archived fence delimiter: ```bash]
# Build with GPU support
cmake -DENABLE_GPU=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Build with MPI support
cmake -DENABLE_MPI=ON ..
make -j$(nproc)
[archived fence delimiter: ```]

## See Also

- [Tutorials](../tutorials/index.md) - Learn concepts step by step
- [API Reference](../api/index.md) - Complete function documentation
- [Troubleshooting](../troubleshooting.md) - Common problems and solutions

```
