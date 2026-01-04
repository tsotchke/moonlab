# Quick Start

Get running with Moonlab Quantum Simulator in under 5 minutes. This guide walks through creating and measuring a Bell state—the fundamental demonstration of quantum entanglement.

## Prerequisites

- macOS 12+ (Apple Silicon recommended) or Linux
- C compiler (Clang or GCC)
- Make

## Build the Library

```bash
git clone https://github.com/tsotchke/moonlab.git
cd quantum-simulator
make
```

On Apple Silicon, the build automatically detects your M-series chip and enables hardware acceleration.

## Your First Quantum Program

Create a file `hello_quantum.c`:

```c
#include "src/quantum/state.h"
#include "src/quantum/gates.h"
#include "src/quantum/measurement.h"
#include "src/quantum/entanglement.h"
#include <stdio.h>

int main(void) {
    // Create a 2-qubit quantum register initialized to |00⟩
    quantum_state_t state;
    quantum_state_init(&state, 2);

    // Apply Hadamard to qubit 0: |00⟩ → (|00⟩ + |10⟩)/√2
    gate_hadamard(&state, 0);

    // Apply CNOT(control=0, target=1): → (|00⟩ + |11⟩)/√2
    gate_cnot(&state, 0, 1);

    // This is the Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2

    // Verify: check probabilities
    double p00 = quantum_state_get_probability(&state, 0);  // |00⟩
    double p11 = quantum_state_get_probability(&state, 3);  // |11⟩

    printf("Bell State |Φ⁺⟩ Created\n");
    printf("P(|00⟩) = %.4f\n", p00);  // Should be 0.5
    printf("P(|11⟩) = %.4f\n", p11);  // Should be 0.5

    // Measure entanglement entropy
    int subsystem[] = {0};
    double entropy = quantum_state_entanglement_entropy(&state, subsystem, 1);
    printf("Entanglement entropy: %.4f bits\n", entropy);  // Should be 1.0

    quantum_state_free(&state);
    return 0;
}
```

Compile and run:

```bash
gcc -O3 hello_quantum.c -L. -lquantumsim -lm -o hello_quantum
LD_LIBRARY_PATH=. ./hello_quantum
```

Expected output:

```
Bell State |Φ⁺⟩ Created
P(|00⟩) = 0.5000
P(|11⟩) = 0.5000
Entanglement entropy: 1.0000 bits
```

## Python Quick Start

```python
from moonlab import QuantumState

# Create Bell state
state = QuantumState(2)
state.h(0).cnot(0, 1)

# Check probabilities
probs = state.probabilities()
print(f"P(|00⟩) = {probs[0]:.4f}")  # 0.5
print(f"P(|11⟩) = {probs[3]:.4f}")  # 0.5

# Measure (collapses state)
result = state.measure()
print(f"Measured: |{result:02b}⟩")  # Either |00⟩ or |11⟩
```

## JavaScript Quick Start

```javascript
import { QuantumState } from '@moonlab/core';

// Create Bell state
const state = new QuantumState(2);
state.h(0).cnot(0, 1);

// Check probabilities
const probs = state.probabilities();
console.log(`P(|00⟩) = ${probs[0].toFixed(4)}`);
console.log(`P(|11⟩) = ${probs[3].toFixed(4)}`);

// Measure
const result = state.measure();
console.log(`Measured: |${result.toString(2).padStart(2, '0')}⟩`);
```

## Understanding the Bell State

The circuit we just built:

```
|0⟩ ──[H]──●──
           │
|0⟩ ───────⊕──
```

1. **Initial state**: $|00\rangle$
2. **After Hadamard on qubit 0**: $\frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)$
3. **After CNOT**: $\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = |\Phi^+\rangle$

This is a **maximally entangled** state:
- Measuring qubit 0 as $|0\rangle$ guarantees qubit 1 is also $|0\rangle$
- Measuring qubit 0 as $|1\rangle$ guarantees qubit 1 is also $|1\rangle$
- The entanglement entropy $S = 1$ bit (maximum for 2 qubits)

## Next Steps

| Goal | Resource |
|------|----------|
| Learn quantum computing fundamentals | [Getting Started Guide](getting-started/index.md) |
| Explore all quantum gates | [Tutorial: Quantum Gates Tour](tutorials/02-quantum-gates-tour.md) |
| Run Grover's search algorithm | [Tutorial: Grover's Search](tutorials/04-grovers-search.md) |
| Simulate molecular ground states | [VQE Algorithm](algorithms/vqe-algorithm.md) |
| Enable GPU acceleration | [GPU Acceleration Guide](guides/gpu-acceleration.md) |
| Full C API reference | [C API Documentation](api/c/index.md) |

## Common Issues

**Library not found**
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
```

**OpenMP not available**
```bash
# macOS with Homebrew
brew install libomp
```

**Metal GPU errors on macOS**
Ensure you're running on actual hardware, not a VM. Metal requires native Apple Silicon or AMD GPU.

See [Troubleshooting](troubleshooting.md) for more solutions.
