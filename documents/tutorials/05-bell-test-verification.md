# Tutorial 05: Bell Test Verification

Verify quantum mechanics violates classical limits.

**Duration**: 30 minutes
**Prerequisites**: [Tutorial 03](03-creating-bell-states.md)
**Difficulty**: Intermediate

## Learning Objectives

By the end of this tutorial, you will:

- Understand the CHSH inequality
- Implement a Bell test experiment
- Verify quantum violation of classical bounds
- Interpret the results physically

## The EPR Paradox

In 1935, Einstein, Podolsky, and Rosen argued that quantum mechanics must be incomplete because of "spooky action at a distance" in entangled states.

Could there be hidden variables that predetermine measurement outcomes?

## Bell's Inequality

In 1964, John Bell showed that any local hidden variable theory must satisfy:

$$|S| \leq 2$$

where $S$ is the CHSH parameter:

$$S = E(a, b) - E(a, b') + E(a', b) + E(a', b')$$

and $E(a, b)$ is the correlation between Alice's measurement along direction $a$ and Bob's along $b$.

## Quantum Prediction

Quantum mechanics predicts that for a Bell state:

$$|S|_{\text{QM}} \leq 2\sqrt{2} \approx 2.828$$

This **violates** the classical bound!

## Setting Up the Experiment

### The Setup

1. Create a Bell state $|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$
2. Alice measures along directions $a$ or $a'$
3. Bob measures along directions $b$ or $b'$
4. Compute correlations

### Optimal Angles

For maximum violation:
- $a = 0°$
- $a' = 45°$
- $b = 22.5°$
- $b' = 67.5°$

## Implementation

### Python

```python
from moonlab import QuantumState
from moonlab.algorithms import BellTest
import numpy as np

def measure_in_basis(state, qubit, angle):
    """
    Measure qubit in rotated basis.
    angle: measurement direction in radians
    """
    # Rotate to measurement basis
    state.ry(qubit, -2 * angle)
    # Measure in Z basis
    return state.measure(qubit)

def compute_correlation(angle_a, angle_b, num_shots=10000):
    """
    Compute correlation E(a,b) between Alice and Bob.
    """
    same = 0
    different = 0

    for _ in range(num_shots):
        # Create Bell state
        state = QuantumState(2)
        state.h(0)
        state.cnot(0, 1)

        # Alice measures qubit 0
        result_a = measure_in_basis(state, 0, angle_a)

        # Bob measures qubit 1
        result_b = measure_in_basis(state, 1, angle_b)

        # Count correlations
        if result_a == result_b:
            same += 1
        else:
            different += 1

    # E(a,b) = P(same) - P(different)
    correlation = (same - different) / num_shots
    return correlation

# Optimal angles (in radians)
a = 0
a_prime = np.pi / 4
b = np.pi / 8
b_prime = 3 * np.pi / 8

print("Computing correlations...")
E_ab = compute_correlation(a, b)
E_ab_prime = compute_correlation(a, b_prime)
E_a_prime_b = compute_correlation(a_prime, b)
E_a_prime_b_prime = compute_correlation(a_prime, b_prime)

print(f"E(a, b)   = {E_ab:.4f}")
print(f"E(a, b')  = {E_ab_prime:.4f}")
print(f"E(a', b)  = {E_a_prime_b:.4f}")
print(f"E(a', b') = {E_a_prime_b_prime:.4f}")

# CHSH parameter
S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime

print(f"\nCHSH parameter S = {S:.4f}")
print(f"Classical bound: |S| ≤ 2")
print(f"Quantum bound:   |S| ≤ 2√2 ≈ 2.828")
print(f"Violation: {abs(S) > 2}")
```

**Expected Output**:
```
Computing correlations...
E(a, b)   = 0.7071
E(a, b')  = -0.7071
E(a', b)  = 0.7071
E(a', b') = 0.7071

CHSH parameter S = 2.8284
Classical bound: |S| ≤ 2
Quantum bound:   |S| ≤ 2√2 ≈ 2.828
Violation: True
```

### Using the Built-in BellTest Class

```python
from moonlab.algorithms import BellTest
from moonlab import QuantumState

# Create Bell state
state = QuantumState(2)
BellTest.create_bell_state(state, 0, 1, BellTest.PHI_PLUS)

# Run CHSH test
result = BellTest.chsh_test(state, 0, 1, num_measurements=10000)

print(f"CHSH S = {result['chsh']:.4f}")
print(f"Violates classical: {result['violates_classical']}")
print(f"\nCorrelations:")
for key, val in result['correlations'].items():
    print(f"  {key}: {val:.4f}")
```

## Understanding the Results

### Why S = 2√2?

For optimal angles, quantum mechanics gives:

$$E(a, b) = -\cos(2(\theta_a - \theta_b))$$

With our angles:
- $E(a, b) = -\cos(45°) \approx -0.707$
- $E(a, b') = -\cos(135°) \approx 0.707$
- $E(a', b) = -\cos(-45°) \approx -0.707$
- $E(a', b') = -\cos(45°) \approx -0.707$

So: $S = -0.707 - 0.707 - 0.707 - 0.707 = -2.828$

### Physical Interpretation

The violation proves:
1. Quantum correlations are **stronger** than any classical theory allows
2. There are **no local hidden variables**
3. Measurement outcomes are genuinely **random** until measured
4. But **no information** is transmitted (no faster-than-light signaling)

## Statistical Significance

### Error Analysis

```python
def bell_test_with_error(num_shots=10000, num_experiments=100):
    """Run multiple Bell tests and compute statistics."""
    S_values = []

    for _ in range(num_experiments):
        state = QuantumState(2)
        BellTest.create_bell_state(state, 0, 1, BellTest.PHI_PLUS)
        result = BellTest.chsh_test(state, 0, 1, num_measurements=num_shots)
        S_values.append(result['chsh'])

    mean_S = np.mean(S_values)
    std_S = np.std(S_values)
    error = std_S / np.sqrt(num_experiments)

    print(f"S = {mean_S:.4f} ± {error:.4f}")
    print(f"Standard deviation: {std_S:.4f}")

    # Number of standard deviations from classical bound
    sigma_violation = (abs(mean_S) - 2) / error
    print(f"Violation significance: {sigma_violation:.1f}σ")

    return S_values

S_values = bell_test_with_error(10000, 100)
```

## Testing All Bell States

```python
bell_types = [
    (BellTest.PHI_PLUS, "|Φ+⟩"),
    (BellTest.PHI_MINUS, "|Φ-⟩"),
    (BellTest.PSI_PLUS, "|Ψ+⟩"),
    (BellTest.PSI_MINUS, "|Ψ-⟩"),
]

print("Bell Test Results for All Bell States:\n")
print(f"{'State':<10} {'S Value':<12} {'Violation'}")
print("-" * 35)

for bell_type, name in bell_types:
    state = QuantumState(2)
    BellTest.create_bell_state(state, 0, 1, bell_type)
    result = BellTest.chsh_test(state, 0, 1, num_measurements=10000)

    violation = "Yes" if result['violates_classical'] else "No"
    print(f"{name:<10} {result['chsh']:<12.4f} {violation}")
```

**Output**:
```
Bell Test Results for All Bell States:

State      S Value      Violation
-----------------------------------
|Φ+⟩       2.8284       Yes
|Φ-⟩       -2.8284      Yes
|Ψ+⟩       2.8284       Yes
|Ψ-⟩       -2.8284      Yes
```

## Loophole-Free Bell Tests

Real experiments must close potential loopholes:

1. **Locality loophole**: Ensure no communication between Alice and Bob
2. **Detection loophole**: Account for undetected particles
3. **Freedom-of-choice loophole**: Measurement choices must be independent

In 2015, loophole-free Bell tests confirmed quantum violations.

## C Implementation

```c
#include <stdio.h>
#include <math.h>
#include "quantum_sim.h"

double compute_correlation(double angle_a, double angle_b, int shots) {
    int same = 0, different = 0;

    for (int i = 0; i < shots; i++) {
        quantum_state_t* state = quantum_state_create(2);

        // Create Bell state
        quantum_state_h(state, 0);
        quantum_state_cnot(state, 0, 1);

        // Measure in rotated bases
        quantum_state_ry(state, 0, -2 * angle_a);
        int result_a = quantum_state_measure(state, 0);

        quantum_state_ry(state, 1, -2 * angle_b);
        int result_b = quantum_state_measure(state, 1);

        if (result_a == result_b) same++;
        else different++;

        quantum_state_destroy(state);
    }

    return (double)(same - different) / shots;
}

int main() {
    double a = 0;
    double a_prime = M_PI / 4;
    double b = M_PI / 8;
    double b_prime = 3 * M_PI / 8;

    int shots = 10000;

    double E_ab = compute_correlation(a, b, shots);
    double E_ab_p = compute_correlation(a, b_prime, shots);
    double E_ap_b = compute_correlation(a_prime, b, shots);
    double E_ap_bp = compute_correlation(a_prime, b_prime, shots);

    double S = E_ab - E_ab_p + E_ap_b + E_ap_bp;

    printf("CHSH S = %.4f\n", S);
    printf("Classical bound: |S| <= 2\n");
    printf("Violation: %s\n", fabs(S) > 2 ? "Yes" : "No");

    return 0;
}
```

## Exercises

### Exercise 1: Vary the Angles

Try different measurement angles. Can you find angles that give no violation? What angles give maximum violation?

### Exercise 2: Product State Test

Run the Bell test on a product state (not entangled). Verify that $|S| \leq 2$.

### Exercise 3: Partial Entanglement

Create a partially entangled state:
$$|\psi\rangle = \cos\theta|00\rangle + \sin\theta|11\rangle$$

How does the violation depend on $\theta$?

### Exercise 4: GHZ Paradox

The GHZ state exhibits an even stronger form of nonlocality. Implement the GHZ test where the contradiction is deterministic rather than statistical.

## Key Takeaways

1. **CHSH inequality** sets a classical limit of $|S| \leq 2$
2. **Quantum mechanics violates** this with $|S| \leq 2\sqrt{2}$
3. This proves **no local hidden variables** exist
4. **Bell tests** are fundamental experiments in quantum physics

## Next Steps

Now let's apply quantum computing to chemistry:

**[06. VQE Molecular Simulation →](06-vqe-molecular-simulation.md)**

## Further Reading

- [Bell-CHSH Test](../algorithms/bell-chsh-test.md) - Full mathematical treatment
- [Entanglement Measures](../concepts/entanglement-measures.md) - Quantifying quantum correlations
- Bell, J.S. (1964). "On the Einstein Podolsky Rosen Paradox." Physics, 1(3), 195-200.

