# Tutorial 01: Hello Quantum

Create your first quantum program.

**Duration**: 15 minutes
**Prerequisites**: Moonlab installed
**Difficulty**: Beginner

## Learning Objectives

By the end of this tutorial, you will:

- Create and initialize a quantum state
- Apply the Hadamard gate to create superposition
- Measure a qubit and understand probabilistic outcomes
- Interpret simulation results

## The Goal

We'll create the quantum equivalent of "Hello World" - a qubit in superposition. Unlike a classical bit that's either 0 or 1, our qubit will be *both* 0 and 1 simultaneously until we measure it.

## Step 1: Create a Quantum State

A quantum state holds the information about our qubits. Let's start with a single qubit.

### Python

```python
from moonlab import QuantumState

# Create a 1-qubit state
state = QuantumState(1)

# Check the initial state
print("Initial amplitudes:", state.get_amplitudes())
print("Initial probabilities:", state.probabilities())
```

**Output**:
```
Initial amplitudes: [(1+0j), 0j]
Initial probabilities: [1.0, 0.0]
```

### C

```c
#include <stdio.h>
#include "quantum_sim.h"

int main() {
    // Create a 1-qubit state
    quantum_state_t* state = quantum_state_create(1);

    // Print initial probabilities
    printf("P(|0>) = %f\n", quantum_state_probability(state, 0));
    printf("P(|1>) = %f\n", quantum_state_probability(state, 1));

    quantum_state_destroy(state);
    return 0;
}
```

**Output**:
```
P(|0>) = 1.000000
P(|1>) = 0.000000
```

### Understanding the Output

The state starts in $|0\rangle$:
- Amplitude of $|0\rangle$: 1 (100% chance)
- Amplitude of $|1\rangle$: 0 (0% chance)

This is like a classical bit set to 0.

## Step 2: Apply the Hadamard Gate

The Hadamard gate creates an equal superposition:

$$H|0\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$$

### Python

```python
# Apply Hadamard gate to qubit 0
state.h(0)

# Check the new state
print("After H gate:")
print("Amplitudes:", state.get_amplitudes())
print("Probabilities:", state.probabilities())
```

**Output**:
```
After H gate:
Amplitudes: [(0.707+0j), (0.707+0j)]
Probabilities: [0.5, 0.5]
```

### C

```c
// Apply Hadamard gate
quantum_state_h(state, 0);

// Print new probabilities
printf("After H gate:\n");
printf("P(|0>) = %f\n", quantum_state_probability(state, 0));
printf("P(|1>) = %f\n", quantum_state_probability(state, 1));
```

**Output**:
```
After H gate:
P(|0>) = 0.500000
P(|1>) = 0.500000
```

### What Happened?

The qubit is now in **superposition**:
- 50% chance of measuring $|0\rangle$
- 50% chance of measuring $|1\rangle$
- It's "both" states at once

The amplitude $\frac{1}{\sqrt{2}} \approx 0.707$ for each basis state.

## Step 3: Measure the Qubit

When we measure, the superposition "collapses" to one definite outcome.

### Python

```python
# Reset and create fresh superposition
state.reset()
state.h(0)

# Measure the qubit
result = state.measure(0)
print(f"Measurement result: {result}")

# Check probabilities after measurement
print("After measurement:", state.probabilities())
```

**Output** (will vary):
```
Measurement result: 0
After measurement: [1.0, 0.0]
```

### C

```c
// Reset and create fresh superposition
quantum_state_reset(state);
quantum_state_h(state, 0);

// Measure the qubit
int result = quantum_state_measure(state, 0);
printf("Measurement result: %d\n", result);
```

### What Happened?

1. Before measurement: 50/50 superposition
2. Measurement: Randomly chose 0 or 1
3. After measurement: State collapsed to the measured value

**Run it multiple times** - you'll get different results!

## Step 4: Verify the Statistics

Let's run many measurements to see the 50/50 distribution.

### Python

```python
# Run 1000 experiments
results = {'0': 0, '1': 0}

for _ in range(1000):
    state.reset()
    state.h(0)
    outcome = state.measure(0)
    results[str(outcome)] += 1

print(f"Results: {results}")
print(f"Ratio: {results['0']/1000:.1%} zeros, {results['1']/1000:.1%} ones")
```

**Output** (approximately):
```
Results: {'0': 498, '1': 502}
Ratio: 49.8% zeros, 50.2% ones
```

### C

```c
int zeros = 0, ones = 0;

for (int i = 0; i < 1000; i++) {
    quantum_state_reset(state);
    quantum_state_h(state, 0);
    int result = quantum_state_measure(state, 0);
    if (result == 0) zeros++;
    else ones++;
}

printf("Results: %d zeros, %d ones\n", zeros, ones);
printf("Ratio: %.1f%% zeros, %.1f%% ones\n",
       100.0 * zeros / 1000, 100.0 * ones / 1000);
```

### What This Shows

The measurements are truly random with a 50/50 distribution. This is fundamental quantum randomness, not pseudo-randomness.

## Complete Example

### Python

```python
#!/usr/bin/env python3
"""Hello Quantum - First quantum program."""

from moonlab import QuantumState

def main():
    print("=== Hello Quantum ===\n")

    # Create 1-qubit state
    state = QuantumState(1)
    print("1. Created qubit in |0⟩ state")
    print(f"   Probabilities: P(0)={state.probabilities()[0]:.2f}, "
          f"P(1)={state.probabilities()[1]:.2f}\n")

    # Apply Hadamard
    state.h(0)
    print("2. Applied Hadamard gate - now in superposition")
    print(f"   Probabilities: P(0)={state.probabilities()[0]:.2f}, "
          f"P(1)={state.probabilities()[1]:.2f}\n")

    # Measure
    result = state.measure(0)
    print(f"3. Measured qubit: {result}")
    print(f"   State collapsed to |{result}⟩\n")

    # Statistics
    print("4. Running 1000 experiments...")
    counts = [0, 0]
    for _ in range(1000):
        state.reset()
        state.h(0)
        counts[state.measure(0)] += 1

    print(f"   Results: {counts[0]} zeros, {counts[1]} ones")
    print(f"   Expected: ~500 each (50/50)\n")

    print("✓ Congratulations! You've run your first quantum program.")

if __name__ == "__main__":
    main()
```

### C

```c
/* hello_quantum.c - First quantum program */

#include <stdio.h>
#include "quantum_sim.h"

int main() {
    printf("=== Hello Quantum ===\n\n");

    // Create 1-qubit state
    quantum_state_t* state = quantum_state_create(1);
    printf("1. Created qubit in |0> state\n");
    printf("   P(0)=%.2f, P(1)=%.2f\n\n",
           quantum_state_probability(state, 0),
           quantum_state_probability(state, 1));

    // Apply Hadamard
    quantum_state_h(state, 0);
    printf("2. Applied Hadamard gate - now in superposition\n");
    printf("   P(0)=%.2f, P(1)=%.2f\n\n",
           quantum_state_probability(state, 0),
           quantum_state_probability(state, 1));

    // Measure
    int result = quantum_state_measure(state, 0);
    printf("3. Measured qubit: %d\n", result);
    printf("   State collapsed to |%d>\n\n", result);

    // Statistics
    printf("4. Running 1000 experiments...\n");
    int zeros = 0, ones = 0;
    for (int i = 0; i < 1000; i++) {
        quantum_state_reset(state);
        quantum_state_h(state, 0);
        if (quantum_state_measure(state, 0) == 0) zeros++;
        else ones++;
    }
    printf("   Results: %d zeros, %d ones\n", zeros, ones);
    printf("   Expected: ~500 each (50/50)\n\n");

    printf("Congratulations! You've run your first quantum program.\n");

    quantum_state_destroy(state);
    return 0;
}
```

## Exercises

### Exercise 1: Different Superpositions

Modify the code to create other superposition states:

1. Apply $R_Y(\pi/4)$ instead of $H$ - what probabilities do you get?
2. Apply $H$ then $T$ - does this change probabilities?

### Exercise 2: Multiple Qubits

Create a 2-qubit state and put both qubits in superposition:

```python
state = QuantumState(2)
state.h(0)
state.h(1)
print(state.probabilities())  # Should be [0.25, 0.25, 0.25, 0.25]
```

### Exercise 3: Bloch Sphere Visualization

If using the Python API with visualization:

```python
from moonlab.visualization import draw_bloch

state = QuantumState(1)
state.h(0)
fig = draw_bloch(state)
fig.savefig('superposition.png')
```

## Key Takeaways

1. **Quantum states** hold qubit information
2. **Hadamard gate** creates superposition (50/50)
3. **Measurement** collapses superposition randomly
4. **Probabilities** emerge from many measurements

## Next Steps

You've created your first quantum program! Next, we'll explore all the quantum gates:

**[02. Quantum Gates Tour →](02-quantum-gates-tour.md)**

## Further Reading

- [Quantum Computing Basics](../concepts/quantum-computing-basics.md) - Theory background
- [Quantum Gates](../concepts/quantum-gates.md) - Gate mathematics
- [C API: Quantum State](../api/c/quantum-state.md) - Complete API reference

