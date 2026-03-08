# Grover's Algorithm

Complete guide to quantum search with Grover's algorithm.

## Overview

Grover's algorithm provides a quadratic speedup for searching unstructured databases. Given $N$ items, it finds a marked item in $O(\sqrt{N})$ queries, compared to $O(N)$ classically.

**Discovered**: Lov Grover, 1996

**Applications**:
- Database search
- SAT solving
- Cryptographic attacks
- Optimization (with amplitude amplification)

## Mathematical Foundation

### Problem Statement

Given:
- Search space of $N = 2^n$ items
- Oracle $f: \{0,1\}^n \to \{0,1\}$ with $f(x) = 1$ for solutions
- Number of solutions $M$

Find: An $x$ such that $f(x) = 1$

### Algorithm Structure

#### 1. Initialization

Create uniform superposition:

$$|s\rangle = H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{N}}\sum_{x=0}^{N-1}|x\rangle$$

#### 2. Grover Operator

Apply $G = D \cdot O$ repeatedly, where:

**Oracle** $O$:
$$O|x\rangle = (-1)^{f(x)}|x\rangle$$

Marks solutions with a phase flip.

**Diffusion** $D$:
$$D = 2|s\rangle\langle s| - I = H^{\otimes n}(2|0\rangle\langle 0| - I)H^{\otimes n}$$

Reflects about the mean amplitude.

#### 3. Iteration Count

Optimal number of iterations:

$$k = \left\lfloor \frac{\pi}{4}\sqrt{\frac{N}{M}} \right\rfloor$$

For single solution ($M=1$):

| Qubits | $N$ | Iterations |
|--------|-----|------------|
| 3 | 8 | 2 |
| 5 | 32 | 4 |
| 10 | 1024 | 25 |
| 20 | ~1M | 804 |

### Geometric Interpretation

The algorithm can be understood as rotation in a 2D plane:

1. Define $|\omega\rangle$ = superposition of solutions
2. Define $|s'\rangle$ = superposition of non-solutions
3. Each Grover iteration rotates by angle $2\theta$ toward $|\omega\rangle$

Where $\sin\theta = \sqrt{M/N}$.

$$G^k|s\rangle = \sin((2k+1)\theta)|\omega\rangle + \cos((2k+1)\theta)|s'\rangle$$

## Implementation

### Basic Grover Search

```c
#include "quantum_sim.h"
#include "grover.h"

// Define oracle for target state
void oracle(quantum_state_t* state, uint64_t target) {
    // Phase flip the target state
    quantum_state_phase_flip(state, target);
}

// Run Grover's algorithm
int main() {
    int n_qubits = 5;  // Search space of 32 items
    uint64_t target = 13;  // Item to find

    // Create state
    quantum_state_t* state = quantum_state_create(n_qubits);

    // Initialize to uniform superposition
    for (int i = 0; i < n_qubits; i++) {
        quantum_state_h(state, i);
    }

    // Calculate optimal iterations
    int iterations = grover_optimal_iterations(n_qubits, 1);

    // Grover iterations
    for (int i = 0; i < iterations; i++) {
        // Oracle
        oracle(state, target);

        // Diffusion operator
        grover_diffusion(state);
    }

    // Measure
    uint64_t result = quantum_state_measure_all(state);
    printf("Found: %llu (target was %llu)\n", result, target);

    quantum_state_destroy(state);
    return 0;
}
```

### Using the Built-in Solver

```c
#include "grover.h"

// Simple search
grover_result_t result = grover_search(n_qubits, target_state);

// With custom oracle
grover_result_t result = grover_search_oracle(
    n_qubits,
    my_oracle_func,
    oracle_data
);

// Multiple solutions
grover_result_t result = grover_search_multi(
    n_qubits,
    solutions,      // Array of solution states
    num_solutions
);
```

### Python Interface

```python
from moonlab.algorithms import GroverSearch

# Basic search
grover = GroverSearch(num_qubits=5)
result = grover.search(target=13)

print(f"Found: {result['measured']}")
print(f"Success probability: {result['probability']:.4f}")
print(f"Iterations used: {result['iterations']}")

# Custom oracle
def my_oracle(state):
    # Mark states where first bit equals last bit
    for i in range(2**state.num_qubits):
        if (i & 1) == ((i >> (state.num_qubits-1)) & 1):
            state.phase_flip(i)

result = grover.search_with_oracle(my_oracle)
```

## Advanced Topics

### Multiple Solutions

When there are $M$ solutions:

```python
# Known number of solutions
grover = GroverSearch(num_qubits=10)
result = grover.search(
    targets=[42, 137, 512],  # Multiple targets
    num_solutions=3
)

# Unknown number of solutions
# Use quantum counting or iterative approach
result = grover.search_unknown_solutions(oracle)
```

The success probability after $k$ iterations:

$$P_{success} = \sin^2((2k+1)\theta)$$

where $\sin\theta = \sqrt{M/N}$.

### Amplitude Amplification

Generalization when preparation isn't uniform superposition:

```python
from moonlab.algorithms import AmplitudeAmplification

def prepare(state):
    """Prepare initial state (not necessarily uniform)."""
    state.h(0)
    state.cnot(0, 1)
    state.ry(2, np.pi/3)

aa = AmplitudeAmplification(
    num_qubits=3,
    preparation=prepare,
    oracle=my_oracle
)

result = aa.amplify(iterations=2)
```

### Quantum Counting

Determine the number of solutions:

```python
from moonlab.algorithms import QuantumCounting

counter = QuantumCounting(
    num_qubits=8,
    precision_qubits=4
)

count_result = counter.count(oracle)
print(f"Estimated solutions: {count_result['count']}")
print(f"Uncertainty: ±{count_result['uncertainty']}")
```

This uses QPE on the Grover operator to estimate $\theta$, from which $M$ is computed.

## Oracle Construction

### Phase Oracle from Function

For a classical function $f(x)$:

```c
// Using phase kickback
void phase_oracle(quantum_state_t* state,
                  bool (*f)(uint64_t)) {
    // Add ancilla in |-> state
    quantum_state_t* full = quantum_state_extend(state, 1);
    int ancilla = state->num_qubits;

    quantum_state_x(full, ancilla);
    quantum_state_h(full, ancilla);

    // Apply controlled operations based on f
    // (Specific implementation depends on f)

    // Uncompute ancilla
    quantum_state_h(full, ancilla);
    quantum_state_x(full, ancilla);
}
```

### Boolean SAT Oracle

```python
def sat_oracle(state, clauses):
    """
    Oracle for SAT: marks assignments satisfying all clauses.

    clauses: list of clauses, each clause is list of literals
             positive literal i means x_i, negative means NOT x_i
    """
    n = state.num_qubits - 1  # Last qubit is ancilla

    # Evaluate each clause
    clause_ancillas = []
    for clause in clauses:
        anc = allocate_ancilla()
        evaluate_clause(state, clause, anc)
        clause_ancillas.append(anc)

    # AND all clause results
    multi_controlled_not(state, clause_ancillas, n)

    # Uncompute
    for i, clause in enumerate(clauses):
        unevaluate_clause(state, clause, clause_ancillas[i])
```

## Performance Analysis

### Success Probability

After $k$ iterations with $M$ solutions out of $N$:

$$P(k) = \sin^2\left((2k+1)\arcsin\sqrt{\frac{M}{N}}\right)$$

### Optimal Iterations

| Solutions ($M$) | Optimal $k$ | Max Probability |
|-----------------|-------------|-----------------|
| 1 | $\lfloor\frac{\pi}{4}\sqrt{N}\rfloor$ | $1 - O(1/N)$ |
| $N/4$ | 1 | 1 |
| $N/2$ | 0 | 0.5 |
| $> N/2$ | 0 | $M/N$ |

**Note**: When $M > N/4$, classical random sampling may be preferable.

### Query Complexity

| Complexity | Classical | Quantum |
|------------|-----------|---------|
| Worst case | $O(N)$ | $O(\sqrt{N})$ |
| Average case | $O(N/M)$ | $O(\sqrt{N/M})$ |
| Best case | $O(1)$ | $O(1)$ |

### Gate Complexity

Per Grover iteration:
- Oracle: Problem-dependent
- Diffusion: $O(n)$ gates

Total: $O(\sqrt{N} \cdot (\text{oracle} + n))$

## Optimizations in Moonlab

### SIMD Diffusion

The diffusion operator is vectorized:

```c
// Optimized diffusion using SIMD
void grover_diffusion_simd(quantum_state_t* state) {
    // Compute mean amplitude
    Complex mean = simd_sum(state->amplitudes, state->size) / state->size;

    // Reflect about mean: 2*mean - amplitude
    simd_reflect_mean(state->amplitudes, mean, state->size);
}
```

### GPU Acceleration

For large search spaces:

```c
// Enable GPU
gpu_metal_init();

quantum_state_t* state = quantum_state_create_gpu(20);

// Grover operations use GPU automatically
grover_search_gpu(state, oracle, iterations);
```

### Batched Oracle Evaluation

When oracle can be evaluated in parallel:

```python
grover = GroverSearch(num_qubits=20, backend='metal')
grover.enable_batched_oracle(True)

result = grover.search(target)
# Oracle evaluated on all amplitudes simultaneously
```

## Common Pitfalls

### 1. Over-rotation

Applying too many iterations decreases success probability:

```python
# Wrong: Using too many iterations
result = grover.search(target, iterations=100)  # Likely to fail

# Correct: Use optimal iterations
optimal_k = grover.optimal_iterations(num_solutions=1)
result = grover.search(target, iterations=optimal_k)
```

### 2. Unknown Solution Count

If $M$ is unknown, the fixed iteration formula fails:

```python
# Solution: Exponential search
for k in [1, 2, 4, 8, 16, ...]:
    result = grover.search(oracle, iterations=k)
    if verify(result['measured']):
        break
```

### 3. Oracle Phase Errors

Oracle must apply exactly $-1$ phase to solutions:

```python
# Wrong: Applying i instead of -1
def bad_oracle(state, target):
    state.amplitudes[target] *= 1j  # Incorrect!

# Correct: Apply -1 phase
def good_oracle(state, target):
    state.amplitudes[target] *= -1
```

## Example: Sudoku Solver

```python
from moonlab.algorithms import GroverSearch

def sudoku_oracle(state, puzzle):
    """Oracle that marks valid Sudoku solutions."""
    # Encode puzzle constraints
    for constraint in puzzle.constraints:
        encode_constraint(state, constraint)

    # Check all constraints satisfied
    multi_controlled_phase(state, constraint_ancillas)

    # Uncompute constraints
    for constraint in reversed(puzzle.constraints):
        unencode_constraint(state, constraint)

# 4x4 Sudoku: 16 cells × 2 bits = 32 qubits
grover = GroverSearch(num_qubits=32)
result = grover.search_with_oracle(
    lambda s: sudoku_oracle(s, puzzle)
)

print(f"Solution: {decode_sudoku(result['measured'])}")
```

## See Also

- [Tutorial: Grover's Search](../tutorials/04-grovers-search.md) - Step-by-step tutorial
- [C API: Grover](../api/c/grover.md) - Complete C API reference
- [Python API: Algorithms](../api/python/algorithms.md) - Python interface
- [Amplitude Amplification](https://arxiv.org/abs/quant-ph/0005055) - Generalization

## References

1. Grover, L. K. (1996). "A fast quantum mechanical algorithm for database search." Proceedings of STOC '96.
2. Boyer, M. et al. (1998). "Tight bounds on quantum searching." Fortschritte der Physik, 46(4-5), 493-505.
3. Brassard, G. et al. (2002). "Quantum Amplitude Amplification and Estimation." Contemporary Mathematics, 305, 53-74.

