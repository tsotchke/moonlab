# First Simulation

This walkthrough guides you through building, running, and understanding your first quantum simulation with Moonlab. We'll create a Bell state, verify its properties, and explore measurement.

## Setup

Ensure Moonlab is installed (see [Installation](../installation.md)):

```bash
cd quantum-simulator
make
```

## Part 1: Creating a Quantum State (C)

Create a file called `first_simulation.c`:

```c
#include "src/quantum/state.h"
#include "src/quantum/gates.h"
#include "src/quantum/measurement.h"
#include "src/quantum/entanglement.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    printf("=== Moonlab Quantum Simulator: First Simulation ===\n\n");

    // Step 1: Create a 2-qubit quantum register
    // Initial state: |00⟩
    quantum_state_t state;
    int result = quantum_state_init(&state, 2);

    if (result != QS_SUCCESS) {
        fprintf(stderr, "Failed to initialize quantum state\n");
        return 1;
    }

    printf("Step 1: Initialized 2-qubit state |00⟩\n");
    printf("  Number of qubits: %zu\n", state.num_qubits);
    printf("  State dimension: %zu (2^n amplitudes)\n", state.state_dim);
    printf("\n");

    // Step 2: Examine initial amplitudes
    printf("Step 2: Initial state amplitudes\n");
    for (size_t i = 0; i < state.state_dim; i++) {
        double real = state.amplitudes[i].real;
        double imag = state.amplitudes[i].imag;
        double prob = real*real + imag*imag;
        printf("  |%zu%zu⟩: (%.4f + %.4fi), P = %.4f\n",
               (i >> 1) & 1, i & 1, real, imag, prob);
    }
    printf("\n");

    // Step 3: Apply Hadamard gate to qubit 0
    printf("Step 3: Apply Hadamard to qubit 0\n");
    gate_hadamard(&state, 0);

    printf("  State is now: (|00⟩ + |10⟩)/√2\n");
    printf("  Amplitudes:\n");
    for (size_t i = 0; i < state.state_dim; i++) {
        double real = state.amplitudes[i].real;
        double imag = state.amplitudes[i].imag;
        double prob = real*real + imag*imag;
        if (prob > 0.001) {
            printf("    |%zu%zu⟩: (%.4f + %.4fi), P = %.4f\n",
                   (i >> 1) & 1, i & 1, real, imag, prob);
        }
    }
    printf("\n");

    // Step 4: Apply CNOT (control=0, target=1)
    printf("Step 4: Apply CNOT(control=0, target=1)\n");
    gate_cnot(&state, 0, 1);

    printf("  State is now: (|00⟩ + |11⟩)/√2 = |Φ⁺⟩ (Bell state)\n");
    printf("  Amplitudes:\n");
    for (size_t i = 0; i < state.state_dim; i++) {
        double real = state.amplitudes[i].real;
        double imag = state.amplitudes[i].imag;
        double prob = real*real + imag*imag;
        if (prob > 0.001) {
            printf("    |%zu%zu⟩: (%.4f + %.4fi), P = %.4f\n",
                   (i >> 1) & 1, i & 1, real, imag, prob);
        }
    }
    printf("\n");

    // Step 5: Verify probabilities
    printf("Step 5: Verify measurement probabilities\n");
    double p00 = quantum_state_get_probability(&state, 0);
    double p01 = quantum_state_get_probability(&state, 1);
    double p10 = quantum_state_get_probability(&state, 2);
    double p11 = quantum_state_get_probability(&state, 3);

    printf("  P(|00⟩) = %.6f (expected: 0.5)\n", p00);
    printf("  P(|01⟩) = %.6f (expected: 0.0)\n", p01);
    printf("  P(|10⟩) = %.6f (expected: 0.0)\n", p10);
    printf("  P(|11⟩) = %.6f (expected: 0.5)\n", p11);
    printf("  Total:    %.6f (must be 1.0)\n", p00 + p01 + p10 + p11);
    printf("\n");

    // Step 6: Calculate entanglement entropy
    printf("Step 6: Measure entanglement\n");
    int subsystem_a[] = {0};  // Trace out qubit 1
    double entropy = quantum_state_entanglement_entropy(&state, subsystem_a, 1);

    printf("  Von Neumann entropy S(A) = %.6f bits\n", entropy);
    printf("  Interpretation:\n");
    printf("    S = 0: No entanglement (product state)\n");
    printf("    S = 1: Maximum entanglement (Bell state)\n");

    if (fabs(entropy - 1.0) < 0.001) {
        printf("  ✓ Maximum entanglement confirmed!\n");
    }
    printf("\n");

    // Step 7: Clean up
    printf("Step 7: Cleanup\n");
    quantum_state_free(&state);
    printf("  Resources released.\n");

    printf("\n=== Simulation Complete ===\n");
    return 0;
}
```

### Compile and Run

```bash
gcc -O3 first_simulation.c -L. -lquantumsim -lm -o first_simulation
LD_LIBRARY_PATH=. ./first_simulation
```

### Expected Output

```
=== Moonlab Quantum Simulator: First Simulation ===

Step 1: Initialized 2-qubit state |00⟩
  Number of qubits: 2
  State dimension: 4 (2^n amplitudes)

Step 2: Initial state amplitudes
  |00⟩: (1.0000 + 0.0000i), P = 1.0000
  |01⟩: (0.0000 + 0.0000i), P = 0.0000
  |10⟩: (0.0000 + 0.0000i), P = 0.0000
  |11⟩: (0.0000 + 0.0000i), P = 0.0000

Step 3: Apply Hadamard to qubit 0
  State is now: (|00⟩ + |10⟩)/√2
  Amplitudes:
    |00⟩: (0.7071 + 0.0000i), P = 0.5000
    |10⟩: (0.7071 + 0.0000i), P = 0.5000

Step 4: Apply CNOT(control=0, target=1)
  State is now: (|00⟩ + |11⟩)/√2 = |Φ⁺⟩ (Bell state)
  Amplitudes:
    |00⟩: (0.7071 + 0.0000i), P = 0.5000
    |11⟩: (0.7071 + 0.0000i), P = 0.5000

Step 5: Verify measurement probabilities
  P(|00⟩) = 0.500000 (expected: 0.5)
  P(|01⟩) = 0.000000 (expected: 0.0)
  P(|10⟩) = 0.000000 (expected: 0.0)
  P(|11⟩) = 0.500000 (expected: 0.5)
  Total:    1.000000 (must be 1.0)

Step 6: Measure entanglement
  Von Neumann entropy S(A) = 1.000000 bits
  Interpretation:
    S = 0: No entanglement (product state)
    S = 1: Maximum entanglement (Bell state)
  ✓ Maximum entanglement confirmed!

Step 7: Cleanup
  Resources released.

=== Simulation Complete ===
```

## Part 2: Python Version

The same simulation in Python:

```python
from moonlab import QuantumState
import numpy as np

print("=== Moonlab Quantum Simulator: First Simulation (Python) ===\n")

# Create 2-qubit state
state = QuantumState(2)
print(f"Step 1: Initialized {state.num_qubits}-qubit state")
print(f"  Dimension: {state.state_dim}\n")

# Show initial amplitudes
print("Step 2: Initial amplitudes")
for i, amp in enumerate(state.amplitudes()):
    binary = format(i, '02b')
    print(f"  |{binary}⟩: {amp:.4f}, P = {abs(amp)**2:.4f}")
print()

# Apply Hadamard and CNOT
state.h(0)
print("Step 3: Applied H(0)")

state.cnot(0, 1)
print("Step 4: Applied CNOT(0, 1)")
print("  Created Bell state |Φ⁺⟩\n")

# Check probabilities
probs = state.probabilities()
print("Step 5: Probabilities")
for i, p in enumerate(probs):
    binary = format(i, '02b')
    if p > 0.001:
        print(f"  P(|{binary}⟩) = {p:.6f}")
print()

# Entanglement entropy
entropy = state.entanglement_entropy([0])
print(f"Step 6: Entanglement entropy S = {entropy:.6f} bits")
if abs(entropy - 1.0) < 0.001:
    print("  ✓ Maximum entanglement confirmed!")
print()

print("=== Simulation Complete ===")
```

## Part 3: JavaScript Version

```javascript
import { QuantumState } from '@moonlab/core';

console.log("=== Moonlab Quantum Simulator: First Simulation (JS) ===\n");

// Create 2-qubit state
const state = new QuantumState(2);
console.log(`Step 1: Initialized ${state.numQubits}-qubit state`);
console.log(`  Dimension: ${state.stateDim}\n`);

// Apply gates
state.h(0);
console.log("Step 3: Applied H(0)");

state.cnot(0, 1);
console.log("Step 4: Applied CNOT(0, 1)");
console.log("  Created Bell state |Φ⁺⟩\n");

// Check probabilities
const probs = state.probabilities();
console.log("Step 5: Probabilities");
for (let i = 0; i < probs.length; i++) {
    if (probs[i] > 0.001) {
        const binary = i.toString(2).padStart(2, '0');
        console.log(`  P(|${binary}⟩) = ${probs[i].toFixed(6)}`);
    }
}
console.log();

// Measure
console.log("Step 6: Measurement");
const result = state.measure();
const binary = result.toString(2).padStart(2, '0');
console.log(`  Measured: |${binary}⟩`);
console.log("  State has collapsed!\n");

console.log("=== Simulation Complete ===");
```

## Understanding the Simulation

### State Representation

Moonlab represents quantum states as complex amplitude vectors:

```
|ψ⟩ = α₀|00⟩ + α₁|01⟩ + α₂|10⟩ + α₃|11⟩
```

Internally stored as:
```c
state.amplitudes[0] = α₀  // |00⟩
state.amplitudes[1] = α₁  // |01⟩
state.amplitudes[2] = α₂  // |10⟩
state.amplitudes[3] = α₃  // |11⟩
```

The bit ordering convention: `|qubit₁ qubit₀⟩` where qubit 0 is the least significant bit.

### The Bell State Creation Circuit

```
|0⟩ ──[H]──●──
           │
|0⟩ ───────⊕──
```

**Step by step**:

1. **Initial**: $|00\rangle$
2. **Hadamard on qubit 0**: $\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes |0\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)$
3. **CNOT**: $\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$

The CNOT flips qubit 1 only when qubit 0 is $|1\rangle$, transforming $|10\rangle \rightarrow |11\rangle$.

### Entanglement Entropy

The von Neumann entropy $S = -\text{Tr}(\rho_A \log_2 \rho_A)$ quantifies entanglement:

For the Bell state, the reduced density matrix of qubit 0 is:
$$\rho_A = \frac{1}{2}|0\rangle\langle 0| + \frac{1}{2}|1\rangle\langle 1| = \frac{1}{2}I$$

This is the maximally mixed state, giving:
$$S = -\frac{1}{2}\log_2\frac{1}{2} - \frac{1}{2}\log_2\frac{1}{2} = 1 \text{ bit}$$

## Exercises

1. **Create other Bell states**: Modify the circuit to create $|\Phi^-\rangle$, $|\Psi^+\rangle$, and $|\Psi^-\rangle$.

2. **GHZ state**: Extend to 3 qubits and create the GHZ state $\frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$.

3. **Measurement statistics**: Run 1000 measurements and verify the 50/50 distribution between $|00\rangle$ and $|11\rangle$.

4. **Phase effects**: Add a Z gate after the Hadamard and observe how it changes the state to $|\Phi^-\rangle$.

## Troubleshooting

**"Library not found"**: Set the library path:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
```

**"Segmentation fault"**: Ensure `quantum_state_init` succeeds before using the state:
```c
if (result != QS_SUCCESS) {
    // Handle error
}
```

**Python import error**: Build the library and install the Python package:
```bash
make
cd bindings/python && pip install -e .
```

## Next Steps

- [Next Steps](next-steps.md): Continue your learning journey
- [Tutorial: Quantum Gates Tour](../tutorials/02-quantum-gates-tour.md): Explore all available gates
- [Tutorial: Grover's Search](../tutorials/04-grovers-search.md): Implement a quantum algorithm
