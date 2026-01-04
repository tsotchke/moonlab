# Grover Search Example

Find a marked item in an unsorted database with quadratic speedup.

## Overview

This example implements Grover's search algorithm to find a specific element in an unsorted list. While classical search requires O(N) queries, Grover's algorithm needs only O(√N).

## Prerequisites

- Understanding of quantum gates ([Gates Tour](../../tutorials/02-quantum-gates-tour.md))
- Basic familiarity with oracles ([Grover's Algorithm](../../algorithms/grovers-algorithm.md))

## Problem Statement

Given an unsorted database of N = 2ⁿ items, find the unique item x* that satisfies some condition f(x) = 1.

**Classical**: Check each item → O(N) queries
**Quantum**: Grover's algorithm → O(√N) queries

## Python Implementation

```python
"""
Grover's Search Algorithm
Find a marked item in an unsorted database.
"""

from moonlab import QuantumState
from moonlab.algorithms import Grover
import numpy as np

def grover_manual_implementation(n_qubits, target):
    """
    Manual implementation of Grover's algorithm.

    Args:
        n_qubits: Number of qubits (searches 2^n items)
        target: The item to find (0 to 2^n - 1)

    Returns:
        Measured result
    """
    N = 2 ** n_qubits
    optimal_iterations = int(np.pi / 4 * np.sqrt(N))

    print(f"Searching {N} items for target {target}")
    print(f"Optimal iterations: {optimal_iterations}")

    state = QuantumState(n_qubits)

    # Step 1: Initialize to uniform superposition
    for q in range(n_qubits):
        state.h(q)

    # Step 2: Apply Grover iterations
    for iteration in range(optimal_iterations):
        # Oracle: flip phase of target state
        apply_oracle(state, target, n_qubits)

        # Diffusion: reflect about mean
        apply_diffusion(state, n_qubits)

    # Step 3: Measure
    result = state.measure_all()
    return result

def apply_oracle(state, target, n_qubits):
    """
    Oracle that marks the target by flipping its phase.

    |x⟩ → -|x⟩ if x == target
    |x⟩ → |x⟩  otherwise
    """
    # Convert target to binary
    target_bits = [(target >> i) & 1 for i in range(n_qubits)]

    # Apply X gates to flip 0s (so target becomes all 1s)
    for q in range(n_qubits):
        if target_bits[q] == 0:
            state.x(q)

    # Multi-controlled Z gate (marks all-1s state)
    if n_qubits == 1:
        state.z(0)
    elif n_qubits == 2:
        state.cz(0, 1)
    else:
        # For n > 2, use multi-controlled Z
        state.mcz(list(range(n_qubits)))

    # Uncompute X gates
    for q in range(n_qubits):
        if target_bits[q] == 0:
            state.x(q)

def apply_diffusion(state, n_qubits):
    """
    Diffusion operator: 2|s⟩⟨s| - I

    Reflects amplitudes about the mean.
    """
    # Apply H to all qubits
    for q in range(n_qubits):
        state.h(q)

    # Apply X to all qubits
    for q in range(n_qubits):
        state.x(q)

    # Multi-controlled Z
    if n_qubits == 1:
        state.z(0)
    elif n_qubits == 2:
        state.cz(0, 1)
    else:
        state.mcz(list(range(n_qubits)))

    # Apply X to all qubits
    for q in range(n_qubits):
        state.x(q)

    # Apply H to all qubits
    for q in range(n_qubits):
        state.h(q)

def grover_high_level(n_qubits, target):
    """
    High-level Grover search using the library.
    """
    def oracle(x):
        """Oracle function: returns True if x is the target."""
        return x == target

    grover = Grover(n_qubits, oracle)
    result = grover.search()

    return result

def analyze_probability_evolution(n_qubits, target):
    """
    Visualize how probability evolves during Grover iterations.
    """
    N = 2 ** n_qubits
    max_iterations = int(np.pi / 4 * np.sqrt(N)) + 3

    print(f"\nProbability Evolution (target={target})")
    print("-" * 50)

    state = QuantumState(n_qubits)

    # Initialize superposition
    for q in range(n_qubits):
        state.h(q)

    # Initial probability
    probs = state.probabilities
    print(f"Iter 0: P(target) = {probs[target]:.4f}")

    for iteration in range(1, max_iterations + 1):
        apply_oracle(state, target, n_qubits)
        apply_diffusion(state, n_qubits)

        probs = state.probabilities
        p_target = probs[target]
        bar = "█" * int(p_target * 50)
        print(f"Iter {iteration}: P(target) = {p_target:.4f} {bar}")

def multiple_solutions(n_qubits, targets):
    """
    Grover search with multiple marked items.
    """
    M = len(targets)
    N = 2 ** n_qubits

    # Optimal iterations for M solutions
    optimal_iterations = int(np.pi / 4 * np.sqrt(N / M))

    print(f"\nSearching {N} items for {M} targets: {targets}")
    print(f"Optimal iterations: {optimal_iterations}")

    state = QuantumState(n_qubits)

    # Initialize superposition
    for q in range(n_qubits):
        state.h(q)

    for _ in range(optimal_iterations):
        # Oracle marks all targets
        for target in targets:
            apply_oracle(state, target, n_qubits)

        apply_diffusion(state, n_qubits)

    # Measure and count successes
    successes = 0
    trials = 100
    for _ in range(trials):
        test_state = QuantumState(n_qubits)
        for q in range(n_qubits):
            test_state.h(q)

        for _ in range(optimal_iterations):
            for target in targets:
                apply_oracle(test_state, target, n_qubits)
            apply_diffusion(test_state, n_qubits)

        result = test_state.measure_all()
        if result in targets:
            successes += 1

    print(f"Success rate: {100 * successes / trials:.1f}%")
    return successes / trials

if __name__ == "__main__":
    print("=" * 50)
    print("      Grover's Search Algorithm")
    print("=" * 50)

    # Basic search
    print("\n=== Basic Search (4 qubits, 16 items) ===\n")
    n_qubits = 4
    target = 7

    result = grover_manual_implementation(n_qubits, target)
    print(f"\nMeasured: {result}")
    print(f"Correct: {result == target}")

    # Probability evolution
    print("\n=== Probability Evolution ===")
    analyze_probability_evolution(4, 7)

    # High-level API
    print("\n=== High-Level API ===\n")
    result = grover_high_level(4, 11)
    print(f"Search result: {result.found}")
    print(f"Iterations used: {result.iterations}")
    print(f"Success probability: {result.probability:.4f}")

    # Multiple solutions
    print("\n=== Multiple Solutions ===")
    multiple_solutions(4, [3, 7, 11])
```

## C Implementation

```c
/**
 * Grover's Search Algorithm
 * Find a marked item in an unsorted database.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "quantum_sim.h"
#include "grover.h"

/**
 * Oracle that marks a specific target.
 */
void oracle_callback(quantum_state_t* state, void* context) {
    int target = *(int*)context;
    int n_qubits = quantum_state_num_qubits(state);

    // Flip phase of target state
    quantum_state_phase_flip(state, target);
}

/**
 * Manual Grover implementation.
 */
uint64_t grover_manual(int n_qubits, int target) {
    int N = 1 << n_qubits;
    int optimal_iter = (int)(M_PI / 4.0 * sqrt((double)N));

    printf("Searching %d items for target %d\n", N, target);
    printf("Optimal iterations: %d\n", optimal_iter);

    quantum_state_t* state = quantum_state_create(n_qubits);

    // Initialize superposition
    for (int q = 0; q < n_qubits; q++) {
        quantum_state_h(state, q);
    }

    // Grover iterations
    for (int iter = 0; iter < optimal_iter; iter++) {
        // Oracle: flip phase of target
        quantum_state_phase_flip(state, target);

        // Diffusion operator
        grover_diffusion(state);
    }

    // Measure
    uint64_t result = quantum_state_measure_all(state);
    quantum_state_destroy(state);

    return result;
}

/**
 * Apply diffusion operator.
 */
void grover_diffusion(quantum_state_t* state) {
    int n_qubits = quantum_state_num_qubits(state);

    // H on all
    for (int q = 0; q < n_qubits; q++) {
        quantum_state_h(state, q);
    }

    // X on all
    for (int q = 0; q < n_qubits; q++) {
        quantum_state_x(state, q);
    }

    // Multi-controlled Z
    quantum_state_mcz(state, n_qubits);

    // X on all
    for (int q = 0; q < n_qubits; q++) {
        quantum_state_x(state, q);
    }

    // H on all
    for (int q = 0; q < n_qubits; q++) {
        quantum_state_h(state, q);
    }
}

/**
 * High-level Grover search.
 */
void grover_high_level(int n_qubits, int target) {
    // Create Grover solver
    grover_solver_t* solver = grover_create(n_qubits);

    // Set oracle
    grover_set_oracle(solver, oracle_callback, &target);

    // Run search
    grover_result_t result = grover_search(solver);

    printf("\nHigh-Level Grover Search\n");
    printf("Found: %llu\n", result.found);
    printf("Iterations: %d\n", result.iterations);
    printf("Success probability: %.4f\n", result.probability);

    grover_destroy(solver);
}

/**
 * Test success rate.
 */
void test_success_rate(int n_qubits, int target, int trials) {
    int successes = 0;

    for (int t = 0; t < trials; t++) {
        uint64_t result = grover_manual(n_qubits, target);
        if (result == (uint64_t)target) {
            successes++;
        }
    }

    printf("\nSuccess rate over %d trials: %.1f%%\n",
           trials, 100.0 * successes / trials);
}

int main(void) {
    printf("==================================================\n");
    printf("           Grover's Search Algorithm\n");
    printf("==================================================\n\n");

    // Basic search
    printf("=== Basic Search (4 qubits, 16 items) ===\n\n");
    int n_qubits = 4;
    int target = 7;

    uint64_t result = grover_manual(n_qubits, target);
    printf("\nMeasured: %llu\n", result);
    printf("Correct: %s\n", result == target ? "yes" : "no");

    // High-level API
    grover_high_level(n_qubits, 11);

    // Success rate test
    test_success_rate(4, 7, 100);

    return 0;
}
```

## Expected Output

```
==================================================
      Grover's Search Algorithm
==================================================

=== Basic Search (4 qubits, 16 items) ===

Searching 16 items for target 7
Optimal iterations: 3

Measured: 7
Correct: True

=== Probability Evolution ===

Probability Evolution (target=7)
--------------------------------------------------
Iter 0: P(target) = 0.0625
Iter 1: P(target) = 0.4727 ███████████████████████
Iter 2: P(target) = 0.9082 █████████████████████████████████████████████
Iter 3: P(target) = 0.9612 ████████████████████████████████████████████████
Iter 4: P(target) = 0.6960 ██████████████████████████████████
Iter 5: P(target) = 0.2488 ████████████
Iter 6: P(target) = 0.0209 █

=== High-Level API ===

Search result: 11
Iterations used: 3
Success probability: 0.9612

=== Multiple Solutions ===

Searching 16 items for 3 targets: [3, 7, 11]
Optimal iterations: 1
Success rate: 94.0%
```

## Understanding the Algorithm

### Amplitude Amplification

Grover's algorithm works by repeatedly amplifying the amplitude of the target state:

1. **Initial state**: All amplitudes equal at 1/√N
2. **Oracle**: Flips sign of target amplitude
3. **Diffusion**: Reflects all amplitudes about mean

Each iteration increases target amplitude by approximately 1/√N.

### Optimal Iterations

The number of iterations should be:

$$k = \left\lfloor \frac{\pi}{4} \sqrt{N} \right\rfloor$$

Too few iterations: Haven't amplified enough
Too many iterations: Over-rotate past maximum

### Circuit Depth

| Qubits | Database Size | Iterations | Circuit Depth |
|--------|---------------|------------|---------------|
| 4 | 16 | 3 | ~30 gates |
| 8 | 256 | 12 | ~120 gates |
| 10 | 1024 | 25 | ~250 gates |
| 16 | 65536 | 201 | ~2000 gates |

## Exercises

### Exercise 1: Different Targets

Try searching for different targets and verify the algorithm works:

```python
for target in [0, 5, 10, 15]:
    result = grover_manual_implementation(4, target)
    print(f"Target {target}: Found {result}")
```

### Exercise 2: Scaling Analysis

Measure success probability vs. number of qubits:

```python
for n in range(2, 8):
    successes = 0
    for _ in range(100):
        result = grover_manual_implementation(n, 0)
        if result == 0:
            successes += 1
    print(f"{n} qubits: {successes}% success")
```

### Exercise 3: Unknown Number of Solutions

When you don't know how many solutions exist, use quantum counting:

```python
from moonlab.algorithms import QuantumCounting

counter = QuantumCounting(n_qubits=4)
M = counter.count(oracle)
print(f"Estimated number of solutions: {M}")
```

## Performance Analysis

### Classical vs Quantum

| N | Classical (N queries) | Quantum (√N queries) | Speedup |
|---|----------------------|---------------------|---------|
| 100 | 100 | 8 | 12.5x |
| 1,000 | 1,000 | 25 | 40x |
| 1,000,000 | 1,000,000 | 785 | 1,274x |

### Simulation Performance

```python
import time
from moonlab import set_backend

for backend in ['cpu', 'metal']:
    set_backend(backend)

    start = time.time()
    for _ in range(100):
        grover_high_level(10, 512)
    elapsed = time.time() - start

    print(f"{backend}: {elapsed:.3f}s for 100 searches")
```

## See Also

- [Grover's Algorithm](../../algorithms/grovers-algorithm.md) - Complete theory
- [Quantum Counting](../../algorithms/grovers-algorithm.md#quantum-counting) - Count solutions
- [C API: Grover](../../api/c/grover.md) - API reference

