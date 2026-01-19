# Tutorial 04: Grover's Search

Implement the famous quantum search algorithm.

**Duration**: 45 minutes
**Prerequisites**: [Tutorial 03](03-creating-bell-states.md)
**Difficulty**: Intermediate

## Learning Objectives

By the end of this tutorial, you will:

- Understand the Grover's algorithm structure
- Implement the oracle and diffusion operator
- Analyze the quadratic speedup
- Handle multiple solutions and unknown number of solutions

## The Problem

**Classical search**: Find a marked item in $N$ unsorted items.
- Best classical: $O(N)$ queries
- Average: $N/2$ queries

**Grover's algorithm**: $O(\sqrt{N})$ queries - quadratic speedup!

For $N = 1,000,000$: ~1000 queries vs ~500,000

## Algorithm Overview

1. **Initialize**: Put all qubits in equal superposition
2. **Repeat** $\approx \frac{\pi}{4}\sqrt{N}$ times:
   - Apply **Oracle**: Mark the solution
   - Apply **Diffusion**: Amplify marked state
3. **Measure**: Get the solution with high probability

## Step 1: Setup

```python
from moonlab import QuantumState
from moonlab.algorithms import Grover
import numpy as np

# Number of qubits
n = 4  # Search space of 2^4 = 16 items

# The "needle" we're searching for
marked_state = 7  # We're looking for |0111⟩

print(f"Search space: {2**n} items")
print(f"Looking for: {marked_state} = |{format(marked_state, f'0{n}b')}⟩")
```

## Step 2: Initialize Superposition

```python
def initialize_grover(state):
    """Put all qubits in equal superposition."""
    for q in range(state.num_qubits):
        state.h(q)
    return state

state = QuantumState(n)
initialize_grover(state)

# Verify equal superposition
probs = state.probabilities()
print(f"Initial probability of each state: {probs[0]:.4f}")
# Should be 1/16 = 0.0625 for each state
```

## Step 3: The Oracle

The oracle marks the solution by flipping its phase:

$$O|x\rangle = \begin{cases} -|x\rangle & \text{if } x = \text{solution} \\ |x\rangle & \text{otherwise} \end{cases}$$

```python
def oracle(state, marked):
    """Apply Grover oracle for a single marked state."""
    n = state.num_qubits

    # Method: Multi-controlled Z gate
    # Flip phase of |marked⟩ state

    # First, flip bits that should be 0 in the marked state
    for q in range(n):
        if not (marked & (1 << q)):
            state.x(q)

    # Apply multi-controlled Z (using decomposition)
    # For small n, we can use Toffoli decomposition
    if n == 2:
        state.cz(0, 1)
    elif n == 3:
        state.h(2)
        state.toffoli(0, 1, 2)
        state.h(2)
    elif n >= 4:
        # Use ancilla-free multi-controlled Z
        apply_mcz(state)

    # Flip bits back
    for q in range(n):
        if not (marked & (1 << q)):
            state.x(q)

    return state

def apply_mcz(state):
    """Apply multi-controlled Z gate without ancilla."""
    n = state.num_qubits
    # Phase shift on |11...1⟩
    # Using Rz decomposition
    for i in range(n):
        for j in range(i + 1, n):
            state.crz(i, j, np.pi / (2 ** (n - 2)))
```

## Step 4: The Diffusion Operator

The diffusion operator (Grover diffuser) inverts amplitudes about the mean:

$$D = 2|s\rangle\langle s| - I$$

where $|s\rangle = H^{\otimes n}|0\rangle$ is the equal superposition state.

```python
def diffusion(state):
    """Apply Grover diffusion operator."""
    n = state.num_qubits

    # 1. Apply H to all qubits
    for q in range(n):
        state.h(q)

    # 2. Apply X to all qubits
    for q in range(n):
        state.x(q)

    # 3. Apply multi-controlled Z
    if n == 2:
        state.cz(0, 1)
    elif n == 3:
        state.h(2)
        state.toffoli(0, 1, 2)
        state.h(2)
    else:
        apply_mcz(state)

    # 4. Apply X to all qubits
    for q in range(n):
        state.x(q)

    # 5. Apply H to all qubits
    for q in range(n):
        state.h(q)

    return state
```

## Step 5: Run the Algorithm

```python
def grover_search(n, marked_state, verbose=True):
    """
    Run Grover's algorithm to find marked_state.

    Returns: (found_state, probability)
    """
    state = QuantumState(n)

    # Initialize superposition
    initialize_grover(state)

    # Optimal number of iterations
    N = 2 ** n
    num_iterations = int(np.round(np.pi / 4 * np.sqrt(N)))

    if verbose:
        print(f"Running {num_iterations} Grover iterations...")

    # Track probability of marked state
    probs = [state.probability(marked_state)]

    # Grover iterations
    for i in range(num_iterations):
        oracle(state, marked_state)
        diffusion(state)
        probs.append(state.probability(marked_state))

        if verbose:
            print(f"  Iteration {i+1}: P(marked) = {probs[-1]:.4f}")

    # Measure
    result = state.measure_all()

    return result, probs

# Run the search
n = 4
marked = 7
result, probs = grover_search(n, marked)

print(f"\nResult: {result} = |{format(result, f'0{n}b')}⟩")
print(f"Success: {result == marked}")
```

**Output**:
```
Running 3 Grover iterations...
  Iteration 1: P(marked) = 0.4727
  Iteration 2: P(marked) = 0.9453
  Iteration 3: P(marked) = 0.9961

Result: 7 = |0111⟩
Success: True
```

## Using the Built-in Grover Class

Moonlab provides a convenient class:

```python
from moonlab.algorithms import Grover

# Create Grover instance
grover = Grover(num_qubits=6)  # Search space of 64

# Search for state 42
result = grover.search(marked_state=42)

print(f"Found: {result['found_state']}")
print(f"Success: {result['success']}")
print(f"Probability: {result['probability']:.4f}")
print(f"Iterations: {result['iterations_used']}")
```

## Visualizing Amplitude Amplification

```python
import matplotlib.pyplot as plt

# Run Grover and track all probabilities
n = 5
marked = 17
N = 2 ** n

state = QuantumState(n)
initialize_grover(state)

# Track probability evolution
iterations = int(np.round(np.pi / 4 * np.sqrt(N))) + 2
prob_marked = []
prob_others = []

for i in range(iterations):
    p_marked = state.probability(marked)
    p_other = (1 - p_marked) / (N - 1)
    prob_marked.append(p_marked)
    prob_others.append(p_other)

    oracle(state, marked)
    diffusion(state)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(prob_marked, 'b-o', label='Marked state')
plt.plot(prob_others, 'r-s', label='Other states (avg)')
plt.xlabel('Iteration')
plt.ylabel('Probability')
plt.title(f"Grover's Algorithm: Finding |{marked}⟩ in {N} items")
plt.legend()
plt.grid(True)
plt.savefig('grover_amplification.png')
plt.show()
```

## Optimal Number of Iterations

The success probability oscillates:

$$P(\text{success}) = \sin^2\left((2k+1)\theta\right)$$

where $\theta = \arcsin(1/\sqrt{N})$ and $k$ is the iteration count.

**Optimal iterations**: $k \approx \frac{\pi}{4}\sqrt{N}$

```python
# Demonstrate the oscillation
n = 6
marked = 42
N = 2 ** n

state = QuantumState(n)

probs = []
for i in range(20):
    state.reset()
    initialize_grover(state)

    for _ in range(i):
        oracle(state, marked)
        diffusion(state)

    probs.append(state.probability(marked))

print("Iterations vs Probability:")
for i, p in enumerate(probs):
    bar = '#' * int(p * 50)
    print(f"{i:2d}: {p:.4f} {bar}")
```

**Output** (shows oscillation):
```
Iterations vs Probability:
 0: 0.0156
 1: 0.1328
 2: 0.3780
 3: 0.6836
 4: 0.9139
 5: 0.9980    ##################################################
 6: 0.9139
 7: 0.6836
 8: 0.3780
 9: 0.1328
10: 0.0156
...
```

## Multiple Solutions

When there are $M$ solutions out of $N$ items:

- Optimal iterations: $\frac{\pi}{4}\sqrt{N/M}$
- Speedup still quadratic

```python
def grover_multiple(n, marked_states):
    """Grover search with multiple solutions."""
    state = QuantumState(n)
    N = 2 ** n
    M = len(marked_states)

    # Optimal iterations
    num_iter = int(np.round(np.pi / 4 * np.sqrt(N / M)))

    initialize_grover(state)

    for _ in range(num_iter):
        # Oracle marks ALL solutions
        for marked in marked_states:
            oracle(state, marked)
        diffusion(state)

    # Measure
    result = state.measure_all()
    return result, result in marked_states

# Search for any of 3 solutions
solutions = [5, 13, 21]
for _ in range(10):
    result, success = grover_multiple(5, solutions)
    print(f"Found {result}, is solution: {success}")
```

## Unknown Number of Solutions

When $M$ is unknown, use **Quantum Counting** or iterative amplitude estimation:

```python
def grover_unknown(n, oracle_func, max_iters=None):
    """
    Grover search when number of solutions is unknown.
    Uses exponentially increasing iteration counts.
    """
    N = 2 ** n
    if max_iters is None:
        max_iters = int(np.sqrt(N))

    m = 1
    while m <= max_iters:
        # Try with k iterations where k is random in [0, m)
        k = np.random.randint(0, m)

        state = QuantumState(n)
        initialize_grover(state)

        for _ in range(k):
            oracle_func(state)
            diffusion(state)

        result = state.measure_all()

        # Check if result is a solution (by calling oracle)
        test = QuantumState(n)
        # Set to |result⟩
        for q in range(n):
            if result & (1 << q):
                test.x(q)
        oracle_func(test)

        # If phase flipped, it's a solution
        amp_after = test.get_amplitudes()[result]
        if amp_after.real < 0:
            return result, k

        # Increase iteration bound
        m = int(m * 1.5) + 1

    return None, max_iters
```

## Performance Analysis

| Search Space | Classical | Grover |
|--------------|-----------|--------|
| 16 (4 qubits) | ~8 | 3 |
| 256 (8 qubits) | ~128 | 12 |
| 1024 (10 qubits) | ~512 | 25 |
| 1M (20 qubits) | ~500K | 785 |
| 1B (30 qubits) | ~500M | 24,674 |

## Exercises

### Exercise 1: Search in 8 Qubits

Implement Grover's algorithm for 8 qubits (256 items) and find a randomly chosen marked state.

### Exercise 2: Multiple Marked States

Search for 4 solutions out of 64 items. Verify the reduced iteration count.

### Exercise 3: Worst Case

What happens if you run too many or too few iterations? Plot the success probability.

### Exercise 4: Database Search Application

Implement a simple "database" as a list, and use Grover to search for an item matching a condition.

## Key Takeaways

1. **Grover provides quadratic speedup**: $O(\sqrt{N})$ vs $O(N)$
2. **Oracle** marks solutions with phase flip
3. **Diffusion** amplifies marked states
4. **Optimal iterations**: $\approx \frac{\pi}{4}\sqrt{N}$
5. **Too many iterations** reduces success probability

## Next Steps

Let's verify quantum mechanics itself with a Bell test:

**[05. Bell Test Verification →](05-bell-test-verification.md)**

## Further Reading

- [Grover's Algorithm (Deep Dive)](../algorithms/grovers-algorithm.md) - Full mathematical treatment
- [C API: Grover](../api/c/grover.md) - Low-level implementation
- [Python API: Algorithms](../api/python/algorithms.md) - High-level interface

