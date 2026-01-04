# Tutorials

Step-by-step guides for learning quantum simulation with Moonlab.

## Learning Path

These tutorials are designed to be followed in order, each building on concepts from previous ones.

### Beginner Track

Start here if you're new to quantum computing or Moonlab.

| Tutorial | Topics | Duration |
|----------|--------|----------|
| [01. Hello Quantum](01-hello-quantum.md) | Creating your first quantum program | 15 min |
| [02. Quantum Gates Tour](02-quantum-gates-tour.md) | Single and multi-qubit gates | 30 min |
| [03. Creating Bell States](03-creating-bell-states.md) | Entanglement fundamentals | 20 min |

### Intermediate Track

Explore quantum algorithms and practical applications.

| Tutorial | Topics | Duration |
|----------|--------|----------|
| [04. Grover's Search](04-grovers-search.md) | Quantum search algorithm | 45 min |
| [05. Bell Test Verification](05-bell-test-verification.md) | CHSH inequality test | 30 min |
| [06. VQE Molecular Simulation](06-vqe-molecular-simulation.md) | Chemistry with VQE | 60 min |
| [07. QAOA Optimization](07-qaoa-optimization.md) | Combinatorial optimization | 45 min |

### Advanced Track

Scale up with tensor networks and GPU acceleration.

| Tutorial | Topics | Duration |
|----------|--------|----------|
| [08. Tensor Network Simulation](08-tensor-network-simulation.md) | MPS and DMRG basics | 60 min |
| [09. GPU Acceleration](09-gpu-acceleration.md) | Metal-accelerated simulation | 45 min |

## Prerequisites

Before starting these tutorials, you should have:

1. **Moonlab installed** - See [Installation Guide](../installation.md)
2. **Basic programming knowledge** - Python or C familiarity
3. **Linear algebra fundamentals** - Vectors, matrices, complex numbers

No prior quantum computing experience is required.

## Tutorial Format

Each tutorial includes:

- **Learning objectives** - What you'll accomplish
- **Concept introduction** - Theory behind the topic
- **Step-by-step code** - Complete working examples
- **Exercises** - Practice problems to reinforce learning
- **Next steps** - Where to go from here

## Code Samples

Tutorials provide examples in multiple languages:

### Python
```python
from moonlab import QuantumState

state = QuantumState(2)
state.h(0).cnot(0, 1)
print(state.probabilities())
```

### C
```c
#include "quantum_sim.h"

quantum_state_t* state = quantum_state_create(2);
quantum_state_h(state, 0);
quantum_state_cnot(state, 0, 1);
```

### JavaScript
```javascript
import { QuantumState } from '@moonlab/quantum-core';

const state = await QuantumState.create({ numQubits: 2 });
state.h(0).cnot(0, 1);
console.log(state.getProbabilities());
```

## Getting Help

If you get stuck:

1. Check the [FAQ](../faq.md) for common issues
2. Review the [Troubleshooting Guide](../troubleshooting.md)
3. Open an issue on [GitHub](https://github.com/tsotchke/moonlab/issues)

## Additional Resources

- [Concepts](../concepts/index.md) - Theoretical background
- [API Reference](../api/index.md) - Complete function documentation
- [Examples](../examples/index.md) - More code samples

## Quick Start

Ready to begin? Start with the first tutorial:

**[01. Hello Quantum →](01-hello-quantum.md)**

