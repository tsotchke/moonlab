# Python Core API

Complete reference for the Python quantum simulation API.

**Module**: `moonlab.core`

## Overview

The core module provides the primary Python interface for quantum simulation, wrapping the high-performance C library with a Pythonic API. Features:

- Full quantum state manipulation
- Complete universal gate set
- Measurement with wavefunction collapse
- NumPy integration for state vectors
- Method chaining for fluent circuits

## Installation

```bash
pip install moonlab
```

Or from source:

```bash
cd bindings/python
pip install -e .
```

## Quick Start

```python
from moonlab import QuantumState, Gates, Measurement

# Create Bell state
state = QuantumState(2)
state.h(0).cnot(0, 1)

# Check probabilities
probs = state.probabilities()
print(probs)  # [0.5, 0.0, 0.0, 0.5]

# Measure
result = Measurement.measure_all(state)
print(f"Measured: |{result:02b}⟩")
```

## QuantumState

Main class for quantum state simulation.

### Constructor

```python
QuantumState(num_qubits: int)
```

**Parameters**:
- `num_qubits`: Number of qubits (1-32)

**Raises**: `ValueError` if num_qubits out of range

**Example**:
```python
state = QuantumState(4)  # 4-qubit state
print(state)  # QuantumState(num_qubits=4, dim=16)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_qubits` | int | Number of qubits |
| `state_dim` | int | Dimension of state vector ($2^n$) |

### State Operations

#### clone

```python
clone() -> QuantumState
```

Create deep copy of quantum state.

```python
original = QuantumState(2).h(0).cnot(0, 1)
copy = original.clone()
```

#### reset

```python
reset() -> QuantumState
```

Reset to $|0\ldots0\rangle$ state.

```python
state.reset()  # Back to |00...0⟩
```

#### normalize

```python
normalize() -> QuantumState
```

Normalize state vector to unit length.

```python
state.normalize()
```

### Single-Qubit Gates

All single-qubit gates return `self` for method chaining.

#### h (Hadamard)

```python
h(qubit: int) -> QuantumState
```

Creates superposition: $H|0\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$

```python
state.h(0)  # Hadamard on qubit 0
```

#### x, y, z (Pauli Gates)

```python
x(qubit: int) -> QuantumState  # Bit flip
y(qubit: int) -> QuantumState  # Bit and phase flip
z(qubit: int) -> QuantumState  # Phase flip
```

```python
state.x(0)  # Pauli-X on qubit 0
state.y(1)  # Pauli-Y on qubit 1
state.z(2)  # Pauli-Z on qubit 2
```

#### s, t (Phase Gates)

```python
s(qubit: int) -> QuantumState    # S gate (√Z)
sdg(qubit: int) -> QuantumState  # S† gate
t(qubit: int) -> QuantumState    # T gate (π/8)
tdg(qubit: int) -> QuantumState  # T† gate
```

```python
state.s(0).t(1)  # S on q0, T on q1
```

### Rotation Gates

#### rx, ry, rz

```python
rx(qubit: int, angle: float) -> QuantumState  # X-axis rotation
ry(qubit: int, angle: float) -> QuantumState  # Y-axis rotation
rz(qubit: int, angle: float) -> QuantumState  # Z-axis rotation
```

**Parameters**:
- `angle`: Rotation angle in radians

```python
import numpy as np

state.rx(0, np.pi/4)   # RX(π/4) on qubit 0
state.ry(1, np.pi/2)   # RY(π/2) on qubit 1
state.rz(2, np.pi)     # RZ(π) on qubit 2
```

#### phase

```python
phase(qubit: int, angle: float) -> QuantumState
```

Phase gate: $|0\rangle \to |0\rangle$, $|1\rangle \to e^{i\theta}|1\rangle$

```python
state.phase(0, np.pi/4)  # Phase gate with θ=π/4
```

### Two-Qubit Gates

#### cnot, cx

```python
cnot(control: int, target: int) -> QuantumState
cx(control: int, target: int) -> QuantumState  # Alias
```

Controlled-NOT (CNOT) gate.

```python
state.cnot(0, 1)  # CNOT with q0 as control, q1 as target
```

#### cz

```python
cz(control: int, target: int) -> QuantumState
```

Controlled-Z gate.

```python
state.cz(0, 1)  # CZ gate
```

#### swap

```python
swap(qubit1: int, qubit2: int) -> QuantumState
```

SWAP gate.

```python
state.swap(0, 1)  # Swap qubits 0 and 1
```

#### cphase

```python
cphase(control: int, target: int, angle: float) -> QuantumState
```

Controlled phase gate.

```python
state.cphase(0, 1, np.pi/2)  # Controlled-S gate
```

### Three-Qubit Gates

#### toffoli, ccx

```python
toffoli(control1: int, control2: int, target: int) -> QuantumState
ccx(control1: int, control2: int, target: int) -> QuantumState  # Alias
```

Toffoli (CCNOT) gate.

```python
state.toffoli(0, 1, 2)  # Toffoli with controls 0,1 and target 2
```

### State Queries

#### probability

```python
probability(basis_state: int) -> float
```

Get probability of measuring specific basis state.

```python
p = state.probability(0)  # P(|00...0⟩)
```

#### probabilities

```python
probabilities() -> np.ndarray
```

Get full probability distribution as NumPy array.

```python
probs = state.probabilities()
print(probs)  # Array of length 2^n
```

#### get_statevector

```python
get_statevector() -> np.ndarray
```

Get state vector as complex NumPy array.

```python
sv = state.get_statevector()
print(sv.shape)  # (2^n,)
print(sv.dtype)  # complex128
```

### Measurement

#### measure_all_fast

```python
measure_all_fast() -> int
```

Measure all qubits simultaneously with wavefunction collapse.

**Returns**: Measured basis state index (0 to $2^n - 1$)

```python
result = state.measure_all_fast()
print(f"Measured: {result} = |{result:b}⟩")
```

## Gates Class

Static interface for quantum gates (Qiskit-like syntax).

```python
from moonlab import Gates

state = QuantumState(2)
Gates.H(state, 0)
Gates.CNOT(state, 0, 1)
```

### Available Gates

| Method | Description |
|--------|-------------|
| `Gates.H(state, qubit)` | Hadamard |
| `Gates.X(state, qubit)` | Pauli-X |
| `Gates.Y(state, qubit)` | Pauli-Y |
| `Gates.Z(state, qubit)` | Pauli-Z |
| `Gates.S(state, qubit)` | S gate |
| `Gates.T(state, qubit)` | T gate |
| `Gates.CNOT(state, ctrl, tgt)` | CNOT |
| `Gates.CX(state, ctrl, tgt)` | CNOT alias |
| `Gates.CZ(state, ctrl, tgt)` | CZ |
| `Gates.SWAP(state, q1, q2)` | SWAP |
| `Gates.Toffoli(state, c1, c2, tgt)` | Toffoli |
| `Gates.RX(state, qubit, angle)` | X rotation |
| `Gates.RY(state, qubit, angle)` | Y rotation |
| `Gates.RZ(state, qubit, angle)` | Z rotation |

## Measurement Class

Quantum measurement operations.

### measure

```python
Measurement.measure(state: QuantumState, qubit: int) -> int
```

Measure single qubit in computational basis with collapse.

**Returns**: 0 or 1

```python
from moonlab import Measurement

result = Measurement.measure(state, 0)
print(f"Qubit 0: {result}")
```

### measure_all

```python
Measurement.measure_all(state: QuantumState) -> int
```

Measure all qubits simultaneously.

**Returns**: Basis state index

### measure_x

```python
Measurement.measure_x(state: QuantumState, qubit: int) -> int
```

Measure in X basis ($|+\rangle$, $|-\rangle$).

### measure_y

```python
Measurement.measure_y(state: QuantumState, qubit: int) -> int
```

Measure in Y basis.

## Utility Functions

### create_bell_state

```python
create_bell_state(qubit_a: int = 0, qubit_b: int = 1) -> QuantumState
```

Create Bell state $|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}$.

```python
from moonlab import create_bell_state

bell = create_bell_state(0, 1)
```

### create_ghz_state

```python
create_ghz_state(num_qubits: int) -> QuantumState
```

Create GHZ state $|GHZ\rangle = (|00\ldots0\rangle + |11\ldots1\rangle)/\sqrt{2}$.

```python
from moonlab import create_ghz_state

ghz = create_ghz_state(5)  # 5-qubit GHZ state
```

### statevector_to_numpy

```python
statevector_to_numpy(state: QuantumState) -> np.ndarray
```

Convert quantum state to NumPy array.

### numpy_to_statevector

```python
numpy_to_statevector(amplitudes: np.ndarray, normalize: bool = True) -> QuantumState
```

Create quantum state from NumPy array.

```python
from moonlab import numpy_to_statevector
import numpy as np

# Create custom superposition
amps = np.array([1, 0, 0, 1]) / np.sqrt(2)
state = numpy_to_statevector(amps)
```

## QuantumError

Exception raised for quantum computing errors.

```python
from moonlab import QuantumError

try:
    state = QuantumState(50)  # Too many qubits
except QuantumError as e:
    print(f"Error: {e}")
```

## Complete Example

```python
from moonlab import QuantumState, Measurement, create_bell_state
import numpy as np

# Method 1: Fluent interface
state = QuantumState(3)
state.h(0).cnot(0, 1).cnot(1, 2)  # GHZ state

# Check probabilities
probs = state.probabilities()
print("GHZ state probabilities:")
for i, p in enumerate(probs):
    if p > 0.01:
        print(f"  |{i:03b}⟩: {p:.4f}")

# Measure
result = Measurement.measure_all(state)
print(f"\nMeasured: |{result:03b}⟩")

# Method 2: Custom state
amplitudes = np.zeros(8, dtype=complex)
amplitudes[0] = 1/np.sqrt(2)
amplitudes[7] = 1/np.sqrt(2)
custom_ghz = numpy_to_statevector(amplitudes)

# Verify fidelity
sv1 = state.get_statevector()
sv2 = custom_ghz.get_statevector()
fidelity = abs(np.vdot(sv1, sv2))**2
print(f"Fidelity: {fidelity:.6f}")
```

## Thread Safety

- `QuantumState` instances are NOT thread-safe
- Create separate states per thread for parallel simulations

## Memory Management

States are automatically freed when Python objects are garbage collected. For explicit cleanup in memory-constrained scenarios:

```python
del state  # Explicit deletion
```

## See Also

- [Algorithms API](algorithms.md) - VQE, QAOA, Grover
- [ML API](ml.md) - Quantum machine learning
- [PyTorch Integration](torch-layer.md) - Neural network layers
