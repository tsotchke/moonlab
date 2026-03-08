# Tutorial 03: Creating Bell States

Master quantum entanglement with Bell states.

**Duration**: 20 minutes
**Prerequisites**: [Tutorial 02](02-quantum-gates-tour.md)
**Difficulty**: Beginner

## Learning Objectives

By the end of this tutorial, you will:

- Understand quantum entanglement
- Create all four Bell states
- Verify entanglement through correlation
- Build a quantum teleportation circuit

## What is Entanglement?

**Entanglement** is a uniquely quantum phenomenon where two or more qubits become correlated in a way that has no classical analogue.

For entangled qubits:
- Measuring one instantly determines the other
- The correlation persists regardless of distance
- The individual qubits have no definite state

## The Four Bell States

The Bell states are maximally entangled 2-qubit states:

| Name | State | Circuit |
|------|-------|---------|
| $\|\Phi^+\rangle$ | $\frac{\|00\rangle + \|11\rangle}{\sqrt{2}}$ | H(0), CNOT(0,1) |
| $\|\Phi^-\rangle$ | $\frac{\|00\rangle - \|11\rangle}{\sqrt{2}}$ | X(0), H(0), CNOT(0,1) |
| $\|\Psi^+\rangle$ | $\frac{\|01\rangle + \|10\rangle}{\sqrt{2}}$ | H(0), CNOT(0,1), X(1) |
| $\|\Psi^-\rangle$ | $\frac{\|01\rangle - \|10\rangle}{\sqrt{2}}$ | X(0), H(0), CNOT(0,1), X(1) |

## Creating Bell States

### Python Implementation

```python
from moonlab import QuantumState

def create_bell_state(name):
    """Create one of the four Bell states."""
    state = QuantumState(2)

    if name == "Phi+":
        state.h(0).cnot(0, 1)

    elif name == "Phi-":
        state.x(0).h(0).cnot(0, 1)

    elif name == "Psi+":
        state.h(0).cnot(0, 1).x(1)

    elif name == "Psi-":
        state.x(0).h(0).cnot(0, 1).x(1)

    return state

# Create and display all Bell states
for name in ["Phi+", "Phi-", "Psi+", "Psi-"]:
    state = create_bell_state(name)
    amps = state.get_amplitudes()
    print(f"|{name}⟩:")
    print(f"  |00⟩: {amps[0]:.4f}")
    print(f"  |01⟩: {amps[1]:.4f}")
    print(f"  |10⟩: {amps[2]:.4f}")
    print(f"  |11⟩: {amps[3]:.4f}")
    print()
```

**Output**:
```
|Phi+⟩:
  |00⟩: 0.7071+0.0000j
  |01⟩: 0.0000+0.0000j
  |10⟩: 0.0000+0.0000j
  |11⟩: 0.7071+0.0000j

|Phi-⟩:
  |00⟩: 0.7071+0.0000j
  |01⟩: 0.0000+0.0000j
  |10⟩: 0.0000+0.0000j
  |11⟩: -0.7071+0.0000j
...
```

### C Implementation

```c
#include <stdio.h>
#include "quantum_sim.h"

void print_bell_state(quantum_state_t* state, const char* name) {
    printf("|%s>:\n", name);
    printf("  P(|00>) = %.4f\n", quantum_state_probability(state, 0));
    printf("  P(|01>) = %.4f\n", quantum_state_probability(state, 1));
    printf("  P(|10>) = %.4f\n", quantum_state_probability(state, 2));
    printf("  P(|11>) = %.4f\n", quantum_state_probability(state, 3));
    printf("\n");
}

int main() {
    quantum_state_t* state = quantum_state_create(2);

    // |Φ+⟩
    quantum_state_h(state, 0);
    quantum_state_cnot(state, 0, 1);
    print_bell_state(state, "Phi+");

    // |Φ-⟩
    quantum_state_reset(state);
    quantum_state_x(state, 0);
    quantum_state_h(state, 0);
    quantum_state_cnot(state, 0, 1);
    print_bell_state(state, "Phi-");

    // |Ψ+⟩
    quantum_state_reset(state);
    quantum_state_h(state, 0);
    quantum_state_cnot(state, 0, 1);
    quantum_state_x(state, 1);
    print_bell_state(state, "Psi+");

    // |Ψ-⟩
    quantum_state_reset(state);
    quantum_state_x(state, 0);
    quantum_state_h(state, 0);
    quantum_state_cnot(state, 0, 1);
    quantum_state_x(state, 1);
    print_bell_state(state, "Psi-");

    quantum_state_destroy(state);
    return 0;
}
```

## Verifying Entanglement

### Perfect Correlation

For $|\Phi^+\rangle$, measuring both qubits always gives the same result:

```python
state = create_bell_state("Phi+")

# Run 1000 measurements
same_count = 0
for _ in range(1000):
    state.reset()
    state.h(0).cnot(0, 1)  # Recreate Bell state

    q0 = state.measure(0)
    q1 = state.measure(1)

    if q0 == q1:
        same_count += 1

print(f"Same result: {same_count}/1000 ({same_count/10}%)")
# Output: Same result: 1000/1000 (100%)
```

### Anti-Correlation

For $|\Psi^+\rangle$, measurements are always opposite:

```python
state = create_bell_state("Psi+")

opposite_count = 0
for _ in range(1000):
    state.reset()
    state.h(0).cnot(0, 1).x(1)  # Recreate |Ψ+⟩

    q0 = state.measure(0)
    q1 = state.measure(1)

    if q0 != q1:
        opposite_count += 1

print(f"Opposite result: {opposite_count}/1000 ({opposite_count/10}%)")
# Output: Opposite result: 1000/1000 (100%)
```

## Entanglement Entropy

A quantitative measure of entanglement:

```python
state = create_bell_state("Phi+")

# Von Neumann entropy of reduced density matrix
entropy = state.entanglement_entropy([0])  # Trace out qubit 1

print(f"Entanglement entropy: {entropy:.4f} bits")
# Output: Entanglement entropy: 1.0000 bits
```

**Interpretation**:
- 0 bits: No entanglement (product state)
- 1 bit: Maximum entanglement (Bell state)

## GHZ States: Multi-Qubit Entanglement

The GHZ (Greenberger-Horne-Zeilinger) state generalizes Bell states to $n$ qubits:

$$|GHZ_n\rangle = \frac{|0\rangle^{\otimes n} + |1\rangle^{\otimes n}}{\sqrt{2}}$$

```python
def create_ghz(n):
    """Create n-qubit GHZ state."""
    state = QuantumState(n)
    state.h(0)
    for i in range(n - 1):
        state.cnot(i, i + 1)
    return state

# 4-qubit GHZ
ghz4 = create_ghz(4)
probs = ghz4.probabilities()

print("4-qubit GHZ state:")
print(f"  P(|0000⟩) = {probs[0]:.4f}")   # 0.5
print(f"  P(|1111⟩) = {probs[15]:.4f}")  # 0.5
# All other probabilities are 0
```

## W States

Another important multi-qubit entangled state:

$$|W_3\rangle = \frac{|001\rangle + |010\rangle + |100\rangle}{\sqrt{3}}$$

```python
def create_w3():
    """Create 3-qubit W state."""
    state = QuantumState(3)

    # Rotation to get correct amplitudes
    theta = 2 * np.arccos(1/np.sqrt(3))

    state.ry(0, theta)
    state.ch(0, 1)  # Controlled-H (if available) or decompose
    state.ccx(0, 1, 2)  # Toffoli
    state.cx(0, 1)
    state.x(0)

    return state
```

## Quantum Teleportation

Use entanglement to "teleport" quantum information:

```python
def quantum_teleport(state_to_send):
    """
    Teleport a quantum state using Bell pair.

    Returns the teleported state.
    """
    # Create 3-qubit system: [message, alice, bob]
    full_state = QuantumState(3)

    # Prepare the state to teleport on qubit 0
    # (In this example, we'll teleport |+⟩)
    full_state.h(0)

    # Create Bell pair between Alice (1) and Bob (2)
    full_state.h(1)
    full_state.cnot(1, 2)

    # Alice's Bell measurement
    full_state.cnot(0, 1)
    full_state.h(0)

    # Measure Alice's qubits
    m0 = full_state.measure(0)
    m1 = full_state.measure(1)

    # Bob applies corrections based on measurements
    if m1 == 1:
        full_state.x(2)
    if m0 == 1:
        full_state.z(2)

    # Bob's qubit now has the teleported state
    print(f"Measurements: m0={m0}, m1={m1}")
    print(f"Bob's qubit probabilities: P(0)={full_state.probability_zero(2):.4f}")

    return full_state

# Run teleportation
result = quantum_teleport(None)
```

### Teleportation Circuit Diagram

```
|ψ⟩ ──────●───H───M─────────────────
          │       │
|0⟩ ──H───●───────M───────────┐
          │                   │
|0⟩ ──────X───────────────X───Z───|ψ⟩
                          │   │
              Classical ──┴───┘
```

## Superdense Coding

Send 2 classical bits using 1 qubit (plus shared entanglement):

```python
def superdense_encode(bits, state):
    """
    Alice encodes 2 classical bits into her qubit.
    bits: string "00", "01", "10", or "11"
    state: Bell state shared with Bob
    """
    if bits[1] == '1':
        state.x(0)
    if bits[0] == '1':
        state.z(0)
    return state

def superdense_decode(state):
    """Bob decodes the 2 bits."""
    # Reverse Bell circuit
    state.cnot(0, 1)
    state.h(0)

    # Measure both qubits
    b0 = state.measure(0)
    b1 = state.measure(1)
    return f"{b0}{b1}"

# Test superdense coding
for message in ["00", "01", "10", "11"]:
    # Create Bell pair
    state = QuantumState(2)
    state.h(0).cnot(0, 1)

    # Alice encodes
    superdense_encode(message, state)

    # Bob decodes
    received = superdense_decode(state)

    print(f"Sent: {message}, Received: {received}, Match: {message == received}")
```

## Exercises

### Exercise 1: Verify All Bell States

Write code to verify that each Bell state has the expected correlations:
- $|\Phi^+\rangle$, $|\Phi^-\rangle$: Same measurement outcomes
- $|\Psi^+\rangle$, $|\Psi^-\rangle$: Opposite measurement outcomes

### Exercise 2: Create a 5-Qubit GHZ State

Create and verify the 5-qubit GHZ state.

### Exercise 3: Bell State Distinguisher

Given an unknown Bell state, determine which one it is using measurements.

### Exercise 4: Entanglement Swapping

Start with two separate Bell pairs: (0,1) and (2,3). By performing a Bell measurement on qubits 1 and 2, create entanglement between qubits 0 and 3 (which never directly interacted).

## Key Takeaways

1. **Bell states** are maximally entangled 2-qubit states
2. **Entanglement** creates perfect correlations
3. **GHZ states** extend entanglement to $n$ qubits
4. **Teleportation** and **superdense coding** are practical applications

## Next Steps

Now let's use entanglement in an algorithm:

**[04. Grover's Search →](04-grovers-search.md)**

## Further Reading

- [Entanglement Measures](../concepts/entanglement-measures.md) - Quantifying entanglement
- [Bell-CHSH Test](../algorithms/bell-chsh-test.md) - Verifying quantum correlations
- [Quantum Teleportation Example](../examples/basic/quantum-teleportation.md) - Full implementation

