# Tutorial 02: Quantum Gates Tour

Explore the complete quantum gate library.

**Duration**: 30 minutes
**Prerequisites**: [Tutorial 01](01-hello-quantum.md)
**Difficulty**: Beginner

## Learning Objectives

By the end of this tutorial, you will:

- Understand the different categories of quantum gates
- Apply single-qubit gates and observe their effects
- Use two-qubit gates for entanglement
- Build quantum circuits with method chaining

## Gate Categories

Quantum gates fall into several categories:

| Category | Gates | Purpose |
|----------|-------|---------|
| Pauli | X, Y, Z | Bit/phase flips |
| Hadamard | H | Create superposition |
| Phase | S, T, P(θ) | Add phases |
| Rotation | Rx, Ry, Rz | Continuous rotations |
| Controlled | CNOT, CZ, Toffoli | Conditional operations |
| SWAP | SWAP, iSWAP | Exchange qubits |

## Single-Qubit Gates

### Setup

```python
from moonlab import QuantumState
import numpy as np

def show_state(state, name=""):
    """Display state probabilities nicely."""
    probs = state.probabilities()
    n = state.num_qubits
    print(f"{name}:")
    for i, p in enumerate(probs):
        if p > 0.001:  # Only show non-zero
            bits = format(i, f'0{n}b')
            print(f"  |{bits}⟩: {p:.4f}")
    print()
```

### Pauli-X (NOT Gate)

The X gate flips $|0\rangle \leftrightarrow |1\rangle$:

```python
state = QuantumState(1)
print("Initial:", state.probabilities())  # [1, 0]

state.x(0)
print("After X:", state.probabilities())  # [0, 1]

state.x(0)
print("After X again:", state.probabilities())  # [1, 0]
```

**Key insight**: X² = I (applying twice returns to original).

### Pauli-Y and Pauli-Z

```python
# Y gate: combined bit and phase flip
state = QuantumState(1)
state.h(0)  # Create |+⟩
amps_before = state.get_amplitudes()
state.y(0)
amps_after = state.get_amplitudes()
print(f"Y gate changed amplitudes: {amps_before} → {amps_after}")

# Z gate: phase flip (only visible in superposition)
state.reset()
state.h(0)  # |+⟩ state
state.z(0)  # Now |−⟩ state
print("After Z on |+⟩:", state.get_amplitudes())
```

### Hadamard Gate

Creates equal superposition:

```python
state = QuantumState(1)
state.h(0)
show_state(state, "After H")

# H is self-inverse
state.h(0)
show_state(state, "After H again (back to |0⟩)")
```

### Phase Gates: S and T

```python
state = QuantumState(1)
state.h(0)  # Create superposition

# S gate adds π/2 phase to |1⟩
print("Before S:", state.get_amplitudes())
state.s(0)
print("After S:", state.get_amplitudes())

# T gate adds π/4 phase to |1⟩
state.reset()
state.h(0)
state.t(0)
print("After T:", state.get_amplitudes())
```

**Relation**: S = T², Z = S²

### Rotation Gates

Continuous rotation around Bloch sphere axes:

```python
state = QuantumState(1)

# Rx: rotation around X axis
state.rx(0, np.pi/4)  # 45° rotation
show_state(state, "Rx(π/4)")

# Ry: rotation around Y axis
state.reset()
state.ry(0, np.pi/2)  # 90° rotation
show_state(state, "Ry(π/2)")

# Rz: rotation around Z axis
state.reset()
state.h(0)
state.rz(0, np.pi/3)  # 60° rotation
show_state(state, "Rz(π/3)")
```

**Special cases**:
- Rx(π) = iX
- Ry(π) = iY
- Rz(π) = iZ

### General Phase Gate

```python
# P(φ) adds phase e^(iφ) to |1⟩
state = QuantumState(1)
state.h(0)
state.phase(0, np.pi/6)  # 30° phase
print("After P(π/6):", state.get_amplitudes())
```

## Two-Qubit Gates

### CNOT (Controlled-NOT)

Flips target if control is |1⟩:

```python
state = QuantumState(2)

# Test all input combinations
for initial in [0, 1, 2, 3]:
    state.reset()
    # Set to |initial⟩
    if initial & 1:
        state.x(0)
    if initial & 2:
        state.x(1)

    state.cnot(1, 0)  # control=1, target=0

    result = state.measure_all()
    bits_in = format(initial, '02b')
    bits_out = format(result, '02b')
    print(f"|{bits_in}⟩ → |{bits_out}⟩")
```

**Output**:
```
|00⟩ → |00⟩
|01⟩ → |01⟩
|10⟩ → |11⟩
|11⟩ → |10⟩
```

### Creating Entanglement

```python
state = QuantumState(2)

# Bell state: (|00⟩ + |11⟩)/√2
state.h(0)
state.cnot(0, 1)

show_state(state, "Bell State")
# Output: |00⟩: 0.5, |11⟩: 0.5
```

### Controlled-Z

```python
state = QuantumState(2)
state.h(0)
state.h(1)  # |++⟩ state

state.cz(0, 1)

show_state(state, "After CZ on |++⟩")
# Creates entanglement!
```

**Property**: CZ is symmetric - CZ(0,1) = CZ(1,0)

### SWAP Gate

Exchanges two qubits:

```python
state = QuantumState(2)
state.x(0)  # |01⟩ (qubit 0 is |1⟩)
show_state(state, "Before SWAP")

state.swap(0, 1)
show_state(state, "After SWAP")  # Now |10⟩
```

### Controlled Rotations

```python
state = QuantumState(2)
state.x(0)  # Control is |1⟩
state.h(1)  # Target in superposition

# CRz: controlled rotation
state.crz(0, 1, np.pi/2)

show_state(state, "After CRz(π/2)")
```

## Three-Qubit Gates

### Toffoli (CCNOT)

Flips target if both controls are |1⟩:

```python
state = QuantumState(3)

# Test: |110⟩ → |111⟩
state.x(1)
state.x(2)
state.toffoli(1, 2, 0)  # controls=1,2, target=0
show_state(state, "Toffoli on |110⟩")

# Reset and try |100⟩ → |100⟩ (no flip)
state.reset()
state.x(2)
state.toffoli(1, 2, 0)
show_state(state, "Toffoli on |100⟩")
```

### Fredkin (CSWAP)

Swaps two qubits if control is |1⟩:

```python
state = QuantumState(3)
state.x(0)  # Set qubit 0 to |1⟩
state.x(2)  # Control is |1⟩

state.fredkin(2, 0, 1)  # control=2, swap 0 and 1

show_state(state, "After Fredkin")
```

## Method Chaining

Moonlab supports fluent API for building circuits:

```python
state = QuantumState(4)

# Build a GHZ state with method chaining
state.h(0).cnot(0, 1).cnot(1, 2).cnot(2, 3)

show_state(state, "GHZ State")
# Output: |0000⟩: 0.5, |1111⟩: 0.5
```

### Complex Circuit

```python
state = QuantumState(3)

# Quantum circuit with multiple layers
circuit_result = (
    state
    .h(0).h(1).h(2)           # Layer 1: Hadamards
    .cz(0, 1).cz(1, 2)        # Layer 2: CZ gates
    .rx(0, np.pi/4)           # Layer 3: Rotations
    .ry(1, np.pi/3)
    .rz(2, np.pi/6)
    .cnot(0, 2)               # Layer 4: Entanglement
)

show_state(circuit_result, "Complex Circuit")
```

## Gate Effects Summary

### Single-Qubit Gate Effects

| Gate | On $\|0\rangle$ | On $\|1\rangle$ | On $\|+\rangle$ |
|------|-----------------|-----------------|-----------------|
| X | $\|1\rangle$ | $\|0\rangle$ | $\|+\rangle$ |
| Y | $i\|1\rangle$ | $-i\|0\rangle$ | $-\|-\rangle$ |
| Z | $\|0\rangle$ | $-\|1\rangle$ | $\|-\rangle$ |
| H | $\|+\rangle$ | $\|-\rangle$ | $\|0\rangle$ |
| S | $\|0\rangle$ | $i\|1\rangle$ | $\|R\rangle$ |
| T | $\|0\rangle$ | $e^{i\pi/4}\|1\rangle$ | rotated |

### Useful Identities

```python
state = QuantumState(1)

# HZH = X
state.h(0).z(0).h(0)
# Same as: state.x(0)

# HXH = Z
state.reset()
state.h(0).x(0).h(0)
# Same as: state.z(0)

# SXS† = Y
state.reset()
state.s(0).x(0).sdg(0)
# Same as: state.y(0) (up to phase)
```

## Exercises

### Exercise 1: Verify Gate Properties

Write code to verify:
1. X² = I
2. H² = I
3. S² = Z
4. T⁴ = Z

### Exercise 2: Build a 4-Qubit GHZ State

Create the state:
$$|GHZ_4\rangle = \frac{|0000\rangle + |1111\rangle}{\sqrt{2}}$$

### Exercise 3: Quantum NOT Using Only H and CNOT

Can you create an X gate effect using only H and CNOT gates?

Hint: CNOT with target first...

### Exercise 4: Swap Without SWAP

Implement SWAP using only CNOT gates:

```python
# This should work:
state.cnot(0, 1).cnot(1, 0).cnot(0, 1)
# Equivalent to: state.swap(0, 1)
```

Verify this works!

## Key Takeaways

1. **Pauli gates** (X, Y, Z) are fundamental building blocks
2. **Hadamard** creates superposition
3. **Phase gates** (S, T) add quantum phases
4. **Rotation gates** (Rx, Ry, Rz) allow continuous control
5. **CNOT** is the workhorse for entanglement
6. **Method chaining** makes circuit building elegant

## Next Steps

Now that you know the gates, let's use them to create entanglement:

**[03. Creating Bell States →](03-creating-bell-states.md)**

## Quick Reference

See the [Gate Reference](../reference/gate-reference.md) for complete gate matrices and identities.

