# Code Examples

Working examples demonstrating Moonlab capabilities.

## Overview

These examples provide complete, runnable code for common quantum computing tasks. Each example includes explanations, expected output, and variations to try.

## Example Categories

### Basic Examples

Foundation examples for learning quantum simulation:

| Example | Description | Difficulty |
|---------|-------------|------------|
| [Hello Quantum](basic/hello-quantum.md) | First quantum program | Beginner |
| [Bell State](basic/bell-state.md) | Create entangled qubits | Beginner |

### Algorithm Examples

Implementations of quantum algorithms:

| Example | Description | Difficulty |
|---------|-------------|------------|
| [Grover Search](algorithms/grover-search.md) | Database search with oracle | Intermediate |
| [VQE H₂ Molecule](algorithms/vqe-h2-molecule.md) | Ground state energy | Intermediate |

### Application Examples

Real-world applications of quantum computing:

| Example | Description | Difficulty |
|---------|-------------|------------|
| [Portfolio Optimization](applications/portfolio-optimization.md) | QAOA for finance | Advanced |

## Running Examples

### Python

```bash
# Install Moonlab
pip install moonlab

# Run an example
python examples/basic/hello_quantum.py
```

### C

```bash
# Build examples (from project root)
make examples

# Run an example
./examples/quantum/hello_quantum
```

## Example Structure

Each example follows a consistent structure:

1. **Overview** - What the example demonstrates
2. **Prerequisites** - Required knowledge and setup
3. **Code** - Complete, runnable implementation
4. **Explanation** - Line-by-line walkthrough
5. **Output** - Expected results
6. **Exercises** - Variations to try

## Quick Start

### Minimal Python Example

```python
from moonlab import QuantumState

# Create a 2-qubit system
state = QuantumState(2)

# Apply Hadamard to first qubit
state.h(0)

# Entangle with CNOT
state.cnot(0, 1)

# Measure
result = state.measure_all()
print(f"Result: {result}")  # Either "00" or "11"
```

### Minimal C Example

```c
#include "quantum_sim.h"

int main() {
    // Create 2-qubit state
    quantum_state_t* state = quantum_state_create(2);

    // Create Bell state
    quantum_state_h(state, 0);
    quantum_state_cnot(state, 0, 1);

    // Measure
    uint64_t result = quantum_state_measure_all(state);
    printf("Result: %llu\n", result);

    quantum_state_destroy(state);
    return 0;
}
```

## By Concept

### Superposition

- [Hello Quantum](basic/hello-quantum.md) - Single qubit superposition
- [Bell State](basic/bell-state.md) - Two-qubit superposition

### Entanglement

- [Bell State](basic/bell-state.md) - Creating entanglement
- [VQE H₂ Molecule](algorithms/vqe-h2-molecule.md) - Entanglement in chemistry

### Measurement

- [Hello Quantum](basic/hello-quantum.md) - Basic measurement
- [Grover Search](algorithms/grover-search.md) - Oracle-based measurement

### Optimization

- [VQE H₂ Molecule](algorithms/vqe-h2-molecule.md) - Variational optimization
- [Portfolio Optimization](applications/portfolio-optimization.md) - QAOA optimization

## Contributing Examples

We welcome new examples! See [Contributing Guide](../contributing/index.md) for guidelines:

1. Examples should be self-contained
2. Include both Python and C versions when possible
3. Add comprehensive comments
4. Test on multiple platforms

## See Also

- [Tutorials](../tutorials/index.md) - Step-by-step learning
- [API Reference](../api/index.md) - Complete function documentation
- [Algorithms](../algorithms/index.md) - Algorithm theory

