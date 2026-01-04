# Hello Quantum

Your first quantum program demonstrating superposition and measurement.

## Overview

This example creates a single qubit in superposition and measures it, demonstrating the fundamental probabilistic nature of quantum mechanics.

## Prerequisites

- Moonlab installed ([Installation Guide](../../installation.md))
- Basic understanding of qubits ([Quantum Computing Basics](../../concepts/quantum-computing-basics.md))

## The Quantum Coin Flip

A classical coin is either heads or tails. A quantum "coin" (qubit) can be in a superposition of both states simultaneously until measured.

### Mathematical Description

Starting state:
$$|\psi_0\rangle = |0\rangle$$

After Hadamard gate:
$$|\psi_1\rangle = H|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$$

Measurement probabilities:
- $P(0) = |\frac{1}{\sqrt{2}}|^2 = 0.5$
- $P(1) = |\frac{1}{\sqrt{2}}|^2 = 0.5$

## Python Implementation

```python
"""
Hello Quantum - First Quantum Program
Demonstrates superposition and measurement with a single qubit.
"""

from moonlab import QuantumState

def quantum_coin_flip():
    """Simulate a quantum coin flip."""

    # Create a 1-qubit quantum state (initialized to |0⟩)
    state = QuantumState(1)

    # Apply Hadamard gate to create superposition
    # |0⟩ → (|0⟩ + |1⟩)/√2
    state.h(0)

    # Measure the qubit (collapses superposition)
    result = state.measure(0)

    return result

def run_statistics(num_trials=1000):
    """Run multiple trials to demonstrate probability distribution."""

    results = {'0': 0, '1': 0}

    for _ in range(num_trials):
        outcome = quantum_coin_flip()
        results[str(outcome)] += 1

    print(f"Quantum Coin Flip Results ({num_trials} trials)")
    print("-" * 40)
    print(f"|0⟩: {results['0']:4d} ({100*results['0']/num_trials:.1f}%)")
    print(f"|1⟩: {results['1']:4d} ({100*results['1']/num_trials:.1f}%)")
    print()
    print("Expected: 50% each (within statistical fluctuation)")

def examine_amplitudes():
    """Examine quantum state amplitudes before measurement."""

    state = QuantumState(1)
    print("Initial state |0⟩:")
    print(f"  Amplitudes: {state.amplitudes}")
    print(f"  Probabilities: {state.probabilities}")
    print()

    state.h(0)
    print("After Hadamard (superposition):")
    print(f"  Amplitudes: {state.amplitudes}")
    print(f"  Probabilities: {state.probabilities}")
    print()

    result = state.measure(0)
    print(f"After measurement (collapsed to |{result}⟩):")
    print(f"  Amplitudes: {state.amplitudes}")
    print(f"  Probabilities: {state.probabilities}")

if __name__ == "__main__":
    print("=== Hello Quantum ===\n")

    # Single flip
    result = quantum_coin_flip()
    print(f"Single quantum coin flip: {result}\n")

    # Statistics
    run_statistics(1000)
    print()

    # Amplitude inspection
    print("=== Amplitude Analysis ===\n")
    examine_amplitudes()
```

## C Implementation

```c
/**
 * Hello Quantum - First Quantum Program
 * Demonstrates superposition and measurement.
 */

#include <stdio.h>
#include <stdlib.h>
#include "quantum_sim.h"

/**
 * Perform a single quantum coin flip.
 */
int quantum_coin_flip(void) {
    // Create 1-qubit state |0⟩
    quantum_state_t* state = quantum_state_create(1);

    // Apply Hadamard: |0⟩ → (|0⟩ + |1⟩)/√2
    quantum_state_h(state, 0);

    // Measure qubit 0
    int result = quantum_state_measure(state, 0);

    // Cleanup
    quantum_state_destroy(state);

    return result;
}

/**
 * Run statistics over many trials.
 */
void run_statistics(int num_trials) {
    int count_0 = 0;
    int count_1 = 0;

    for (int i = 0; i < num_trials; i++) {
        int result = quantum_coin_flip();
        if (result == 0) count_0++;
        else count_1++;
    }

    printf("Quantum Coin Flip Results (%d trials)\n", num_trials);
    printf("----------------------------------------\n");
    printf("|0⟩: %4d (%.1f%%)\n", count_0, 100.0 * count_0 / num_trials);
    printf("|1⟩: %4d (%.1f%%)\n", count_1, 100.0 * count_1 / num_trials);
    printf("\nExpected: 50%% each (within statistical fluctuation)\n");
}

/**
 * Examine state amplitudes.
 */
void examine_amplitudes(void) {
    quantum_state_t* state = quantum_state_create(1);

    printf("Initial state |0⟩:\n");
    double complex* amps = quantum_state_get_amplitudes(state);
    printf("  α₀ = %.4f + %.4fi\n", creal(amps[0]), cimag(amps[0]));
    printf("  α₁ = %.4f + %.4fi\n", creal(amps[1]), cimag(amps[1]));

    // Apply Hadamard
    quantum_state_h(state, 0);

    printf("\nAfter Hadamard (superposition):\n");
    amps = quantum_state_get_amplitudes(state);
    printf("  α₀ = %.4f + %.4fi\n", creal(amps[0]), cimag(amps[0]));
    printf("  α₁ = %.4f + %.4fi\n", creal(amps[1]), cimag(amps[1]));
    printf("  |α₀|² = %.4f, |α₁|² = %.4f\n",
           creal(amps[0]) * creal(amps[0]) + cimag(amps[0]) * cimag(amps[0]),
           creal(amps[1]) * creal(amps[1]) + cimag(amps[1]) * cimag(amps[1]));

    quantum_state_destroy(state);
}

int main(void) {
    printf("=== Hello Quantum ===\n\n");

    // Single flip
    int result = quantum_coin_flip();
    printf("Single quantum coin flip: %d\n\n", result);

    // Statistics
    run_statistics(1000);
    printf("\n");

    // Amplitude inspection
    printf("=== Amplitude Analysis ===\n\n");
    examine_amplitudes();

    return 0;
}
```

## Expected Output

```
=== Hello Quantum ===

Single quantum coin flip: 1

Quantum Coin Flip Results (1000 trials)
----------------------------------------
|0⟩:  498 (49.8%)
|1⟩:  502 (50.2%)

Expected: 50% each (within statistical fluctuation)

=== Amplitude Analysis ===

Initial state |0⟩:
  Amplitudes: [(1+0j), 0j]
  Probabilities: [1.0, 0.0]

After Hadamard (superposition):
  Amplitudes: [(0.7071+0j), (0.7071+0j)]
  Probabilities: [0.5, 0.5]

After measurement (collapsed to |0⟩):
  Amplitudes: [(1+0j), 0j]
  Probabilities: [1.0, 0.0]
```

## Line-by-Line Explanation

### Creating the State

```python
state = QuantumState(1)
```

Creates a quantum register with 1 qubit, initialized to |0⟩. The state vector is [1, 0], meaning:
- Amplitude of |0⟩ = 1 (100% probability)
- Amplitude of |1⟩ = 0 (0% probability)

### Applying Hadamard

```python
state.h(0)
```

The Hadamard gate transforms the computational basis states:
- H|0⟩ = (|0⟩ + |1⟩)/√2
- H|1⟩ = (|0⟩ - |1⟩)/√2

This creates an equal superposition with amplitudes [1/√2, 1/√2] ≈ [0.707, 0.707].

### Measurement

```python
result = state.measure(0)
```

Measurement collapses the superposition according to the Born rule:
- Probability of outcome is |amplitude|²
- After measurement, state becomes the measured basis state
- The result is inherently random (true quantum randomness)

## Key Concepts Demonstrated

1. **Initialization**: Qubits start in |0⟩
2. **Superposition**: Hadamard creates equal probability states
3. **Measurement**: Collapses superposition probabilistically
4. **Quantum Randomness**: Outcomes are fundamentally unpredictable

## Exercises

### Exercise 1: Different Initial State

Start with |1⟩ instead of |0⟩:

```python
state = QuantumState(1)
state.x(0)  # Flip to |1⟩
state.h(0)  # Apply Hadamard
# What's different about the amplitudes?
```

### Exercise 2: Multiple Qubits

Create superposition of all basis states:

```python
state = QuantumState(3)
for i in range(3):
    state.h(i)
# Each of 8 basis states has equal probability
```

### Exercise 3: Phase Matters

Compare superposition with different phases:

```python
# |+⟩ = (|0⟩ + |1⟩)/√2
state1 = QuantumState(1)
state1.h(0)

# |−⟩ = (|0⟩ - |1⟩)/√2
state2 = QuantumState(1)
state2.x(0)
state2.h(0)

# Both give 50/50 measurement, but have different phases
```

## See Also

- [Bell State](bell-state.md) - Next example: entanglement
- [Quantum Gates Tour](../../tutorials/02-quantum-gates-tour.md) - Explore more gates
- [State Vector Simulation](../../concepts/state-vector-simulation.md) - How it works

