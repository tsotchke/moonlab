# Bell State

Create maximally entangled qubit pairs demonstrating quantum entanglement.

## Overview

Bell states are the simplest examples of quantum entanglement - a correlation between particles that has no classical analog. This example creates all four Bell states and demonstrates their unique measurement correlations.

## Prerequisites

- [Hello Quantum](hello-quantum.md) example completed
- Basic understanding of entanglement ([Concepts](../../concepts/quantum-computing-basics.md#entanglement))

## The Four Bell States

Bell states form an orthonormal basis for two-qubit systems:

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

$$|\Phi^-\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$$

$$|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$$

$$|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$$

## Python Implementation

```python
"""
Bell States - Quantum Entanglement
Creates and analyzes all four Bell states.
"""

from moonlab import QuantumState
import numpy as np

def create_bell_phi_plus():
    """
    Create |Φ+⟩ = (|00⟩ + |11⟩)/√2

    Circuit:
        q0: ─H─●─
              │
        q1: ───X─
    """
    state = QuantumState(2)
    state.h(0)        # Create superposition on qubit 0
    state.cnot(0, 1)  # Entangle with qubit 1
    return state

def create_bell_phi_minus():
    """
    Create |Φ-⟩ = (|00⟩ - |11⟩)/√2

    Circuit:
        q0: ─H─●─Z─
              │
        q1: ───X───
    """
    state = QuantumState(2)
    state.h(0)
    state.cnot(0, 1)
    state.z(0)  # Apply phase flip
    return state

def create_bell_psi_plus():
    """
    Create |Ψ+⟩ = (|01⟩ + |10⟩)/√2

    Circuit:
        q0: ─H─●───
              │
        q1: ───X─X─
    """
    state = QuantumState(2)
    state.h(0)
    state.cnot(0, 1)
    state.x(1)  # Flip qubit 1
    return state

def create_bell_psi_minus():
    """
    Create |Ψ-⟩ = (|01⟩ - |10⟩)/√2 (singlet state)

    Circuit:
        q0: ─H─●─Z─
              │
        q1: ───X─X─
    """
    state = QuantumState(2)
    state.h(0)
    state.cnot(0, 1)
    state.x(1)
    state.z(0)
    return state

def analyze_correlations(name, state_fn, shots=10000):
    """
    Analyze measurement correlations for a Bell state.
    """
    counts = {'00': 0, '01': 0, '10': 0, '11': 0}

    for _ in range(shots):
        state = state_fn()
        result = state.measure_all()
        key = f"{result:02b}"
        counts[key] += 1

    print(f"\n{name} Measurement Statistics ({shots} shots)")
    print("-" * 45)
    for outcome, count in sorted(counts.items()):
        pct = 100 * count / shots
        bar = "█" * int(pct / 2)
        print(f"|{outcome}⟩: {count:5d} ({pct:5.1f}%) {bar}")

    # Calculate correlation
    same = counts['00'] + counts['11']
    diff = counts['01'] + counts['10']
    correlation = (same - diff) / shots
    print(f"\nCorrelation: {correlation:+.3f}")
    print(f"(+1 = perfect same, -1 = perfect opposite)")

def verify_entanglement():
    """
    Verify entanglement by checking that measuring one qubit
    immediately determines the other.
    """
    print("\n=== Entanglement Verification ===\n")

    state = create_bell_phi_plus()

    # Measure only qubit 0
    result_0 = state.measure(0)
    print(f"Measured qubit 0: {result_0}")

    # Check probabilities for qubit 1
    probs = state.probabilities
    print(f"Probabilities after measuring q0: {probs}")

    # Measure qubit 1
    result_1 = state.measure(1)
    print(f"Measured qubit 1: {result_1}")
    print(f"\nResults are {'correlated' if result_0 == result_1 else 'anti-correlated'}!")

def bell_state_tomography():
    """
    Perform simple state tomography on a Bell state.
    """
    print("\n=== Bell State Tomography ===\n")

    # Create |Φ+⟩
    state = create_bell_phi_plus()

    # Calculate entanglement entropy
    entropy = state.entanglement_entropy(qubit=0)
    print(f"Entanglement entropy: {entropy:.4f} bits")
    print(f"Maximum for 2 qubits: 1.0 bit")
    print(f"Entanglement: {'Maximal' if abs(entropy - 1.0) < 0.01 else 'Partial'}")

    # Check purity of subsystem
    purity = state.purity()
    print(f"\nGlobal purity: {purity:.4f}")
    print("(Pure state = 1.0)")

def demonstrate_no_cloning():
    """
    Demonstrate that entangled states cannot be cloned.
    """
    print("\n=== No-Cloning Demonstration ===\n")

    # Create Bell state
    state = create_bell_phi_plus()
    print("Created |Φ+⟩ Bell state")
    print(f"Initial amplitudes: {state.amplitudes}")

    # Try to "copy" to a third qubit via CNOT
    # This creates a GHZ state, not a clone
    state_3q = QuantumState(3)
    state_3q.h(0)
    state_3q.cnot(0, 1)
    state_3q.cnot(0, 2)

    print("\nAfter attempting to 'copy' with CNOT to q2:")
    print(f"Amplitudes: {state_3q.amplitudes}")
    print("\nResult: (|000⟩ + |111⟩)/√2 - a GHZ state, NOT a clone!")

if __name__ == "__main__":
    print("=" * 50)
    print("           Bell States - Quantum Entanglement")
    print("=" * 50)

    # Analyze all four Bell states
    bell_states = [
        ("|Φ+⟩ = (|00⟩ + |11⟩)/√2", create_bell_phi_plus),
        ("|Φ-⟩ = (|00⟩ - |11⟩)/√2", create_bell_phi_minus),
        ("|Ψ+⟩ = (|01⟩ + |10⟩)/√2", create_bell_psi_plus),
        ("|Ψ-⟩ = (|01⟩ - |10⟩)/√2", create_bell_psi_minus),
    ]

    for name, fn in bell_states:
        analyze_correlations(name, fn, shots=1000)

    # Verify entanglement
    verify_entanglement()

    # Tomography
    bell_state_tomography()

    # No-cloning
    demonstrate_no_cloning()
```

## C Implementation

```c
/**
 * Bell States - Quantum Entanglement
 * Creates and analyzes Bell states in C.
 */

#include <stdio.h>
#include <stdlib.h>
#include "quantum_sim.h"

/**
 * Create |Φ+⟩ = (|00⟩ + |11⟩)/√2
 */
quantum_state_t* create_bell_phi_plus(void) {
    quantum_state_t* state = quantum_state_create(2);
    quantum_state_h(state, 0);
    quantum_state_cnot(state, 0, 1);
    return state;
}

/**
 * Create |Φ-⟩ = (|00⟩ - |11⟩)/√2
 */
quantum_state_t* create_bell_phi_minus(void) {
    quantum_state_t* state = quantum_state_create(2);
    quantum_state_h(state, 0);
    quantum_state_cnot(state, 0, 1);
    quantum_state_z(state, 0);
    return state;
}

/**
 * Create |Ψ+⟩ = (|01⟩ + |10⟩)/√2
 */
quantum_state_t* create_bell_psi_plus(void) {
    quantum_state_t* state = quantum_state_create(2);
    quantum_state_h(state, 0);
    quantum_state_cnot(state, 0, 1);
    quantum_state_x(state, 1);
    return state;
}

/**
 * Create |Ψ-⟩ = (|01⟩ - |10⟩)/√2
 */
quantum_state_t* create_bell_psi_minus(void) {
    quantum_state_t* state = quantum_state_create(2);
    quantum_state_h(state, 0);
    quantum_state_cnot(state, 0, 1);
    quantum_state_x(state, 1);
    quantum_state_z(state, 0);
    return state;
}

/**
 * Analyze measurement correlations.
 */
void analyze_correlations(const char* name,
                         quantum_state_t* (*create_fn)(void),
                         int shots) {
    int counts[4] = {0, 0, 0, 0};

    for (int i = 0; i < shots; i++) {
        quantum_state_t* state = create_fn();
        uint64_t result = quantum_state_measure_all(state);
        counts[result]++;
        quantum_state_destroy(state);
    }

    printf("\n%s Measurement Statistics (%d shots)\n", name, shots);
    printf("---------------------------------------------\n");

    const char* labels[] = {"00", "01", "10", "11"};
    for (int i = 0; i < 4; i++) {
        double pct = 100.0 * counts[i] / shots;
        printf("|%s⟩: %5d (%5.1f%%) ", labels[i], counts[i], pct);

        // Print bar
        int bar_len = (int)(pct / 2);
        for (int j = 0; j < bar_len; j++) printf("█");
        printf("\n");
    }

    // Calculate correlation
    int same = counts[0] + counts[3];  // |00⟩ + |11⟩
    int diff = counts[1] + counts[2];  // |01⟩ + |10⟩
    double correlation = (double)(same - diff) / shots;
    printf("\nCorrelation: %+.3f\n", correlation);
}

/**
 * Verify entanglement through sequential measurement.
 */
void verify_entanglement(void) {
    printf("\n=== Entanglement Verification ===\n\n");

    quantum_state_t* state = create_bell_phi_plus();

    // Measure qubit 0
    int result_0 = quantum_state_measure(state, 0);
    printf("Measured qubit 0: %d\n", result_0);

    // Show state probabilities
    double p0 = quantum_state_probability(state, 0);
    double p1 = quantum_state_probability(state, 1);
    double p2 = quantum_state_probability(state, 2);
    double p3 = quantum_state_probability(state, 3);
    printf("Probabilities: |00⟩=%.2f, |01⟩=%.2f, |10⟩=%.2f, |11⟩=%.2f\n",
           p0, p1, p2, p3);

    // Measure qubit 1
    int result_1 = quantum_state_measure(state, 1);
    printf("Measured qubit 1: %d\n", result_1);

    printf("\nResults are %s!\n",
           result_0 == result_1 ? "correlated" : "anti-correlated");

    quantum_state_destroy(state);
}

/**
 * Calculate entanglement entropy.
 */
void bell_state_tomography(void) {
    printf("\n=== Bell State Tomography ===\n\n");

    quantum_state_t* state = create_bell_phi_plus();

    // Calculate entanglement entropy
    double entropy = quantum_state_entanglement_entropy(state, 0);
    printf("Entanglement entropy: %.4f bits\n", entropy);
    printf("Maximum for 2 qubits: 1.0 bit\n");
    printf("Entanglement: %s\n",
           fabs(entropy - 1.0) < 0.01 ? "Maximal" : "Partial");

    // Check purity
    double purity = quantum_state_purity(state);
    printf("\nGlobal purity: %.4f\n", purity);
    printf("(Pure state = 1.0)\n");

    quantum_state_destroy(state);
}

int main(void) {
    printf("==================================================\n");
    printf("           Bell States - Quantum Entanglement\n");
    printf("==================================================\n");

    // Analyze all four Bell states
    analyze_correlations("|Φ+⟩ = (|00⟩ + |11⟩)/√2",
                        create_bell_phi_plus, 1000);
    analyze_correlations("|Φ-⟩ = (|00⟩ - |11⟩)/√2",
                        create_bell_phi_minus, 1000);
    analyze_correlations("|Ψ+⟩ = (|01⟩ + |10⟩)/√2",
                        create_bell_psi_plus, 1000);
    analyze_correlations("|Ψ-⟩ = (|01⟩ - |10⟩)/√2",
                        create_bell_psi_minus, 1000);

    verify_entanglement();
    bell_state_tomography();

    return 0;
}
```

## Expected Output

```
==================================================
           Bell States - Quantum Entanglement
==================================================

|Φ+⟩ = (|00⟩ + |11⟩)/√2 Measurement Statistics (1000 shots)
---------------------------------------------
|00⟩:   512 ( 51.2%) █████████████████████████
|01⟩:     0 (  0.0%)
|10⟩:     0 (  0.0%)
|11⟩:   488 ( 48.8%) ████████████████████████

Correlation: +1.000
(+1 = perfect same, -1 = perfect opposite)

|Ψ+⟩ = (|01⟩ + |10⟩)/√2 Measurement Statistics (1000 shots)
---------------------------------------------
|00⟩:     0 (  0.0%)
|01⟩:   503 ( 50.3%) █████████████████████████
|10⟩:   497 ( 49.7%) ████████████████████████
|11⟩:     0 (  0.0%)

Correlation: -1.000
(+1 = perfect same, -1 = perfect opposite)

=== Entanglement Verification ===

Measured qubit 0: 1
Probabilities after measuring q0: [0.0, 0.0, 0.0, 1.0]
Measured qubit 1: 1

Results are correlated!

=== Bell State Tomography ===

Entanglement entropy: 1.0000 bits
Maximum for 2 qubits: 1.0 bit
Entanglement: Maximal

Global purity: 1.0000
(Pure state = 1.0)
```

## Understanding Entanglement

### Why Entanglement Is Special

In a Bell state like |Φ+⟩:
1. Neither qubit has a definite value before measurement
2. Measuring one instantly determines the other
3. This correlation is stronger than any classical correlation
4. It works regardless of the distance between qubits

### The Circuit

```
     ┌───┐
q0: ─┤ H ├──●──
     └───┘┌─┴─┐
q1: ──────┤ X ├
          └───┘
```

**Step by step:**

1. Initial state: |00⟩
2. After H on q0: (|0⟩ + |1⟩)|0⟩/√2 = (|00⟩ + |10⟩)/√2
3. After CNOT: (|00⟩ + |11⟩)/√2

The CNOT flips q1 when q0 is |1⟩, creating the entanglement.

### Measurement Correlations

| Bell State | Possible Outcomes | Correlation |
|------------|-------------------|-------------|
| Φ+ | 00, 11 | +1 (same) |
| Φ- | 00, 11 | +1 (same) |
| Ψ+ | 01, 10 | -1 (opposite) |
| Ψ- | 01, 10 | -1 (opposite) |

The ± phase determines the sign of interference in other bases.

## Exercises

### Exercise 1: Create Bell States from |11⟩

Start with both qubits in |1⟩:

```python
state = QuantumState(2)
state.x(0)
state.x(1)
state.h(0)
state.cnot(0, 1)
# What Bell state is this?
```

### Exercise 2: Measure in X Basis

Measure entanglement in the X basis:

```python
state = create_bell_phi_plus()
state.h(0)  # Rotate to X basis
state.h(1)
result = state.measure_all()
# What correlations do you observe?
```

### Exercise 3: Bell State Discrimination

Given an unknown Bell state, determine which one it is:

```python
def identify_bell_state(state):
    """Identify which Bell state we have."""
    # Apply reverse Bell circuit
    state.cnot(0, 1)
    state.h(0)
    # Measure to identify
    result = state.measure_all()
    # result tells us which Bell state:
    # 00 → Φ+, 01 → Ψ+, 10 → Φ-, 11 → Ψ-
    return result
```

### Exercise 4: Three-Qubit Entanglement (GHZ State)

Create a GHZ state:

```python
state = QuantumState(3)
state.h(0)
state.cnot(0, 1)
state.cnot(0, 2)
# Result: (|000⟩ + |111⟩)/√2
```

## Applications

- **Quantum Teleportation**: Bell states are the resource for teleporting quantum states
- **Superdense Coding**: Send 2 classical bits using 1 qubit + entanglement
- **Quantum Key Distribution**: Detect eavesdropping via Bell inequality violations
- **Quantum Error Correction**: Entanglement protects against decoherence

## See Also

- [Bell-CHSH Test](../../algorithms/bell-chsh-test.md) - Test quantum nonlocality
- [Creating Bell States Tutorial](../../tutorials/03-creating-bell-states.md) - Detailed tutorial
- [Entanglement Measures](../../concepts/entanglement-measures.md) - Quantifying entanglement

