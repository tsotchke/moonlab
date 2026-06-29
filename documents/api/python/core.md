# Archived Moonlab Documentation: Python Core API

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
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

[archived fence delimiter: ```bash]
pip install moonlab
[archived fence delimiter: ```]

Or from source:

[archived fence delimiter: ```bash]
cd bindings/python
pip install -e .
[archived fence delimiter: ```]

## Quick Start

[archived fence delimiter: ```python]
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
[archived fence delimiter: ```]

## QuantumState

Main class for quantum state simulation.

### Constructor

[archived fence delimiter: ```python]
QuantumState(num_qubits: int)
[archived fence delimiter: ```]

**Parameters**:
- `num_qubits`: Number of qubits (1-32)

**Raises**: `ValueError` if num_qubits out of range

**Example**:
[archived fence delimiter: ```python]
state = QuantumState(4)  # 4-qubit state
print(state)  # QuantumState(num_qubits=4, dim=16)
[archived fence delimiter: ```]

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_qubits` | int | Number of qubits |
| `state_dim` | int | Dimension of state vector ($2^n$) |

### State Operations

#### clone

[archived fence delimiter: ```python]
clone() -> QuantumState
[archived fence delimiter: ```]

Create deep copy of quantum state.

[archived fence delimiter: ```python]
original = QuantumState(2).h(0).cnot(0, 1)
copy = original.clone()
[archived fence delimiter: ```]

#### reset

[archived fence delimiter: ```python]
reset() -> QuantumState
[archived fence delimiter: ```]

Reset to $|0\ldots0\rangle$ state.

[archived fence delimiter: ```python]
state.reset()  # Back to |00...0⟩
[archived fence delimiter: ```]

#### normalize

[archived fence delimiter: ```python]
normalize() -> QuantumState
[archived fence delimiter: ```]

Normalize state vector to unit length.

[archived fence delimiter: ```python]
state.normalize()
[archived fence delimiter: ```]

### Single-Qubit Gates

All single-qubit gates return `self` for method chaining.

#### h (Hadamard)

[archived fence delimiter: ```python]
h(qubit: int) -> QuantumState
[archived fence delimiter: ```]

Creates superposition: $H|0\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$

[archived fence delimiter: ```python]
state.h(0)  # Hadamard on qubit 0
[archived fence delimiter: ```]

#### x, y, z (Pauli Gates)

[archived fence delimiter: ```python]
x(qubit: int) -> QuantumState  # Bit flip
y(qubit: int) -> QuantumState  # Bit and phase flip
z(qubit: int) -> QuantumState  # Phase flip
[archived fence delimiter: ```]

[archived fence delimiter: ```python]
state.x(0)  # Pauli-X on qubit 0
state.y(1)  # Pauli-Y on qubit 1
state.z(2)  # Pauli-Z on qubit 2
[archived fence delimiter: ```]

#### s, t (Phase Gates)

[archived fence delimiter: ```python]
s(qubit: int) -> QuantumState    # S gate (√Z)
sdg(qubit: int) -> QuantumState  # S† gate
t(qubit: int) -> QuantumState    # T gate (π/8)
tdg(qubit: int) -> QuantumState  # T† gate
[archived fence delimiter: ```]

[archived fence delimiter: ```python]
state.s(0).t(1)  # S on q0, T on q1
[archived fence delimiter: ```]

### Rotation Gates

#### rx, ry, rz

[archived fence delimiter: ```python]
rx(qubit: int, angle: float) -> QuantumState  # X-axis rotation
ry(qubit: int, angle: float) -> QuantumState  # Y-axis rotation
rz(qubit: int, angle: float) -> QuantumState  # Z-axis rotation
[archived fence delimiter: ```]

**Parameters**:
- `angle`: Rotation angle in radians

[archived fence delimiter: ```python]
import numpy as np

state.rx(0, np.pi/4)   # RX(π/4) on qubit 0
state.ry(1, np.pi/2)   # RY(π/2) on qubit 1
state.rz(2, np.pi)     # RZ(π) on qubit 2
[archived fence delimiter: ```]

#### phase

[archived fence delimiter: ```python]
phase(qubit: int, angle: float) -> QuantumState
[archived fence delimiter: ```]

Phase gate: $|0\rangle \to |0\rangle$, $|1\rangle \to e^{i\theta}|1\rangle$

[archived fence delimiter: ```python]
state.phase(0, np.pi/4)  # Phase gate with θ=π/4
[archived fence delimiter: ```]

### Two-Qubit Gates

#### cnot, cx

[archived fence delimiter: ```python]
cnot(control: int, target: int) -> QuantumState
cx(control: int, target: int) -> QuantumState  # Alias
[archived fence delimiter: ```]

Controlled-NOT (CNOT) gate.

[archived fence delimiter: ```python]
state.cnot(0, 1)  # CNOT with q0 as control, q1 as target
[archived fence delimiter: ```]

#### cz

[archived fence delimiter: ```python]
cz(control: int, target: int) -> QuantumState
[archived fence delimiter: ```]

Controlled-Z gate.

[archived fence delimiter: ```python]
state.cz(0, 1)  # CZ gate
[archived fence delimiter: ```]

#### swap

[archived fence delimiter: ```python]
swap(qubit1: int, qubit2: int) -> QuantumState
[archived fence delimiter: ```]

SWAP gate.

[archived fence delimiter: ```python]
state.swap(0, 1)  # Swap qubits 0 and 1
[archived fence delimiter: ```]

#### cphase

[archived fence delimiter: ```python]
cphase(control: int, target: int, angle: float) -> QuantumState
[archived fence delimiter: ```]

Controlled phase gate.

[archived fence delimiter: ```python]
state.cphase(0, 1, np.pi/2)  # Controlled-S gate
[archived fence delimiter: ```]

### Three-Qubit Gates

#### toffoli, ccx

[archived fence delimiter: ```python]
toffoli(control1: int, control2: int, target: int) -> QuantumState
ccx(control1: int, control2: int, target: int) -> QuantumState  # Alias
[archived fence delimiter: ```]

Toffoli (CCNOT) gate.

[archived fence delimiter: ```python]
state.toffoli(0, 1, 2)  # Toffoli with controls 0,1 and target 2
[archived fence delimiter: ```]

### State Queries

#### probability

[archived fence delimiter: ```python]
probability(basis_state: int) -> float
[archived fence delimiter: ```]

Get probability of measuring specific basis state.

[archived fence delimiter: ```python]
p = state.probability(0)  # P(|00...0⟩)
[archived fence delimiter: ```]

#### probabilities

[archived fence delimiter: ```python]
probabilities() -> np.ndarray
[archived fence delimiter: ```]

Get full probability distribution as NumPy array.

[archived fence delimiter: ```python]
probs = state.probabilities()
print(probs)  # Array of length 2^n
[archived fence delimiter: ```]

#### get_statevector

[archived fence delimiter: ```python]
get_statevector() -> np.ndarray
[archived fence delimiter: ```]

Get state vector as complex NumPy array.

[archived fence delimiter: ```python]
sv = state.get_statevector()
print(sv.shape)  # (2^n,)
print(sv.dtype)  # complex128
[archived fence delimiter: ```]

### Measurement

#### measure_all_fast

[archived fence delimiter: ```python]
measure_all_fast() -> int
[archived fence delimiter: ```]

Measure all qubits simultaneously with wavefunction collapse.

**Returns**: Measured basis state index (0 to $2^n - 1$)

[archived fence delimiter: ```python]
result = state.measure_all_fast()
print(f"Measured: {result} = |{result:b}⟩")
[archived fence delimiter: ```]

## Gates Class

Static interface for quantum gates (Qiskit-like syntax).

[archived fence delimiter: ```python]
from moonlab import Gates

state = QuantumState(2)
Gates.H(state, 0)
Gates.CNOT(state, 0, 1)
[archived fence delimiter: ```]

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

[archived fence delimiter: ```python]
Measurement.measure(state: QuantumState, qubit: int) -> int
[archived fence delimiter: ```]

Measure single qubit in computational basis with collapse.

**Returns**: 0 or 1

[archived fence delimiter: ```python]
from moonlab import Measurement

result = Measurement.measure(state, 0)
print(f"Qubit 0: {result}")
[archived fence delimiter: ```]

### measure_all

[archived fence delimiter: ```python]
Measurement.measure_all(state: QuantumState) -> int
[archived fence delimiter: ```]

Measure all qubits simultaneously.

**Returns**: Basis state index

### measure_x

[archived fence delimiter: ```python]
Measurement.measure_x(state: QuantumState, qubit: int) -> int
[archived fence delimiter: ```]

Measure in X basis ($|+\rangle$, $|-\rangle$).

### measure_y

[archived fence delimiter: ```python]
Measurement.measure_y(state: QuantumState, qubit: int) -> int
[archived fence delimiter: ```]

Measure in Y basis.

## Utility Functions

### create_bell_state

[archived fence delimiter: ```python]
create_bell_state(qubit_a: int = 0, qubit_b: int = 1) -> QuantumState
[archived fence delimiter: ```]

Create Bell state $|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}$.

[archived fence delimiter: ```python]
from moonlab import create_bell_state

bell = create_bell_state(0, 1)
[archived fence delimiter: ```]

### create_ghz_state

[archived fence delimiter: ```python]
create_ghz_state(num_qubits: int) -> QuantumState
[archived fence delimiter: ```]

Create GHZ state $|GHZ\rangle = (|00\ldots0\rangle + |11\ldots1\rangle)/\sqrt{2}$.

[archived fence delimiter: ```python]
from moonlab import create_ghz_state

ghz = create_ghz_state(5)  # 5-qubit GHZ state
[archived fence delimiter: ```]

### statevector_to_numpy

[archived fence delimiter: ```python]
statevector_to_numpy(state: QuantumState) -> np.ndarray
[archived fence delimiter: ```]

Convert quantum state to NumPy array.

### numpy_to_statevector

[archived fence delimiter: ```python]
numpy_to_statevector(amplitudes: np.ndarray, normalize: bool = True) -> QuantumState
[archived fence delimiter: ```]

Create quantum state from NumPy array.

[archived fence delimiter: ```python]
from moonlab import numpy_to_statevector
import numpy as np

# Create custom superposition
amps = np.array([1, 0, 0, 1]) / np.sqrt(2)
state = numpy_to_statevector(amps)
[archived fence delimiter: ```]

## QuantumError

Exception raised for quantum computing errors.

[archived fence delimiter: ```python]
from moonlab import QuantumError

try:
    state = QuantumState(50)  # Too many qubits
except QuantumError as e:
    print(f"Error: {e}")
[archived fence delimiter: ```]

## Complete Example

[archived fence delimiter: ```python]
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
[archived fence delimiter: ```]

## Thread Safety

- `QuantumState` instances are NOT thread-safe
- Create separate states per thread for parallel simulations

## Memory Management

States are automatically freed when Python objects are garbage collected. For explicit cleanup in memory-constrained scenarios:

[archived fence delimiter: ```python]
del state  # Explicit deletion
[archived fence delimiter: ```]

## See Also

- [Algorithms API](algorithms.md) - VQE, QAOA, Grover
- [ML API](ml.md) - Quantum machine learning
- [PyTorch Integration](torch-layer.md) - Neural network layers
```
