# Archived Moonlab Documentation: Python API Reference

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Python API Reference

The Python bindings provide a high-level, Pythonic interface to Moonlab Quantum Simulator with seamless NumPy integration and machine learning support.

## Overview

| Module | Description |
|--------|-------------|
| [Core](core.md) | `QuantumState`, gates, measurement |
| [Algorithms](algorithms.md) | VQE, QAOA, Grover implementations |
| [CA-MPS + var-D](ca_mps.md) | `CAMPS` class plus `var_d_run`, `gauge_warmstart`, `z2_lgt_1d_build`, `z2_lgt_1d_gauss_law`, `status_string`. Since v0.2.1. |
| [ML](ml.md) | Quantum machine learning (QSVM, kernels) |
| [Torch Layer](torch-layer.md) | PyTorch integration |
| [Visualization](visualization.md) | Circuit diagrams, state plots |

## Installation

[archived fence delimiter: ```bash]
# Build C library first
cd quantum-simulator
make

# Install Python package
cd bindings/python
pip install -e .
[archived fence delimiter: ```]

**Requirements**:
- Python 3.8+
- NumPy
- (Optional) PyTorch for ML features
- (Optional) Matplotlib for visualization

## Quick Start

[archived fence delimiter: ```python]
from moonlab import QuantumState

# Create a 2-qubit state
state = QuantumState(2)

# Create Bell state with method chaining
state.h(0).cnot(0, 1)

# Get probabilities as NumPy array
probs = state.probabilities()
print(f"Probabilities: {probs}")

# Measure (collapses state)
result = state.measure()
print(f"Measured: |{result:02b}⟩")
[archived fence delimiter: ```]

## QuantumState Class

### Constructor

[archived fence delimiter: ```python]
class QuantumState:
    def __init__(self, num_qubits: int)
[archived fence delimiter: ```]

**Parameters**:
- `num_qubits`: Number of qubits (1-32)

**Raises**:
- `ValueError`: If num_qubits not in valid range
- `MemoryError`: If insufficient memory

**Example**:
[archived fence delimiter: ```python]
state = QuantumState(10)  # 10-qubit system
print(f"Dimension: {state.state_dim}")  # 1024
[archived fence delimiter: ```]

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_qubits` | `int` | Number of qubits |
| `state_dim` | `int` | State vector dimension ($2^n$) |

### Gate Methods

All gate methods return `self` for method chaining.

#### Single-Qubit Gates

[archived fence delimiter: ```python]
def x(self, qubit: int) -> QuantumState: ...
def y(self, qubit: int) -> QuantumState: ...
def z(self, qubit: int) -> QuantumState: ...
def h(self, qubit: int) -> QuantumState: ...
def s(self, qubit: int) -> QuantumState: ...
def s_dagger(self, qubit: int) -> QuantumState: ...
def t(self, qubit: int) -> QuantumState: ...
def t_dagger(self, qubit: int) -> QuantumState: ...
[archived fence delimiter: ```]

**Example**:
[archived fence delimiter: ```python]
state = QuantumState(3)
state.h(0).x(1).z(2)  # Chain multiple gates
[archived fence delimiter: ```]

#### Rotation Gates

[archived fence delimiter: ```python]
def rx(self, qubit: int, theta: float) -> QuantumState: ...
def ry(self, qubit: int, theta: float) -> QuantumState: ...
def rz(self, qubit: int, theta: float) -> QuantumState: ...
def phase(self, qubit: int, theta: float) -> QuantumState: ...
[archived fence delimiter: ```]

**Example**:
[archived fence delimiter: ```python]
import numpy as np
state = QuantumState(1)
state.rx(0, np.pi / 4)  # Rotate by π/4 around X
[archived fence delimiter: ```]

#### Two-Qubit Gates

[archived fence delimiter: ```python]
def cnot(self, control: int, target: int) -> QuantumState: ...
def cz(self, control: int, target: int) -> QuantumState: ...
def swap(self, qubit1: int, qubit2: int) -> QuantumState: ...
def cphase(self, control: int, target: int, theta: float) -> QuantumState: ...
[archived fence delimiter: ```]

#### Three-Qubit Gates

[archived fence delimiter: ```python]
def toffoli(self, control1: int, control2: int, target: int) -> QuantumState: ...
[archived fence delimiter: ```]

### State Access

#### amplitudes()

[archived fence delimiter: ```python]
def amplitudes(self) -> np.ndarray[complex]
[archived fence delimiter: ```]

Returns the state vector as a NumPy array of complex numbers.

**Example**:
[archived fence delimiter: ```python]
state = QuantumState(2)
state.h(0)
amps = state.amplitudes()
print(amps)  # [0.707+0j, 0+0j, 0.707+0j, 0+0j]
[archived fence delimiter: ```]

#### probabilities()

[archived fence delimiter: ```python]
def probabilities(self) -> np.ndarray[float]
[archived fence delimiter: ```]

Returns probability of each basis state.

**Example**:
[archived fence delimiter: ```python]
probs = state.probabilities()
for i, p in enumerate(probs):
    if p > 0.01:
        print(f"|{i:02b}⟩: {p:.4f}")
[archived fence delimiter: ```]

### Measurement

#### measure()

[archived fence delimiter: ```python]
def measure(self, qubits: Optional[List[int]] = None) -> int
[archived fence delimiter: ```]

Measure qubits (collapses state).

**Parameters**:
- `qubits`: Optional list of qubits to measure. If None, measures all.

**Returns**: Measured outcome as integer

**Example**:
[archived fence delimiter: ```python]
state = QuantumState(3)
state.h(0).h(1).h(2)

# Measure all qubits
result = state.measure()
print(f"Outcome: {result}")

# Measure specific qubits
state.reset()
state.h(0).h(1).h(2)
partial = state.measure([0, 1])  # Measure qubits 0 and 1 only
[archived fence delimiter: ```]

### State Manipulation

#### reset()

[archived fence delimiter: ```python]
def reset(self) -> QuantumState
[archived fence delimiter: ```]

Reset state to $|0\cdots0\rangle$.

#### clone()

[archived fence delimiter: ```python]
def clone(self) -> QuantumState
[archived fence delimiter: ```]

Create an independent copy of the state.

**Example**:
[archived fence delimiter: ```python]
state = QuantumState(2)
state.h(0).cnot(0, 1)

copy = state.clone()
copy.x(0)  # Doesn't affect original
[archived fence delimiter: ```]

### Entanglement

#### entanglement_entropy()

[archived fence delimiter: ```python]
def entanglement_entropy(self, subsystem: List[int]) -> float
[archived fence delimiter: ```]

Calculate von Neumann entropy of a subsystem.

**Example**:
[archived fence delimiter: ```python]
state = QuantumState(2)
state.h(0).cnot(0, 1)

entropy = state.entanglement_entropy([0])
print(f"Entropy: {entropy:.4f} bits")  # 1.0000 for Bell state
[archived fence delimiter: ```]

## Algorithms Module

[archived fence delimiter: ```python]
from moonlab.algorithms import grover_search, vqe, qaoa
[archived fence delimiter: ```]

### grover_search

[archived fence delimiter: ```python]
def grover_search(
    num_qubits: int,
    oracle: Callable[[QuantumState], None],
    num_iterations: Optional[int] = None
) -> int
[archived fence delimiter: ```]

Run Grover's search algorithm.

**Example**:
[archived fence delimiter: ```python]
from moonlab.algorithms import grover_search

def oracle(state):
    # Mark state |101⟩
    state.cz(0, 1)  # Custom oracle implementation

result = grover_search(3, oracle)
print(f"Found: {result}")
[archived fence delimiter: ```]

### vqe

[archived fence delimiter: ```python]
def vqe(
    hamiltonian: np.ndarray,
    ansatz: Callable[[QuantumState, np.ndarray], None],
    initial_params: np.ndarray,
    optimizer: str = "COBYLA"
) -> Tuple[float, np.ndarray]
[archived fence delimiter: ```]

Variational Quantum Eigensolver.

**Example**:
[archived fence delimiter: ```python]
from moonlab.algorithms import vqe
import numpy as np

# H2 Hamiltonian (simplified)
H = np.array([[1, 0, 0, 0],
              [0, -1, 2, 0],
              [0, 2, -1, 0],
              [0, 0, 0, 1]])

def ansatz(state, params):
    state.ry(0, params[0])
    state.ry(1, params[1])
    state.cnot(0, 1)
    state.rz(1, params[2])

energy, optimal_params = vqe(H, ansatz, np.random.randn(3))
print(f"Ground state energy: {energy:.6f}")
[archived fence delimiter: ```]

## PyTorch Integration

[archived fence delimiter: ```python]
from moonlab.torch_layer import QuantumLayer
[archived fence delimiter: ```]

### QuantumLayer

A PyTorch module for hybrid quantum-classical models:

[archived fence delimiter: ```python]
class QuantumLayer(torch.nn.Module):
    def __init__(
        self,
        num_qubits: int,
        num_layers: int,
        encoding: str = "angle"
    )
[archived fence delimiter: ```]

**Example**:
[archived fence delimiter: ```python]
import torch
from moonlab.torch_layer import QuantumLayer

# Hybrid quantum-classical model
class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classical = torch.nn.Linear(4, 4)
        self.quantum = QuantumLayer(4, num_layers=2)
        self.output = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = self.classical(x)
        x = self.quantum(x)
        return self.output(x)

model = HybridModel()
x = torch.randn(32, 4)  # Batch of 32
y = model(x)
[archived fence delimiter: ```]

## NumPy Integration

### Custom State Initialization

[archived fence delimiter: ```python]
import numpy as np
from moonlab import QuantumState

# Create state from custom amplitudes
amplitudes = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
state = QuantumState.from_amplitudes(amplitudes)
[archived fence delimiter: ```]

### Working with Amplitudes

[archived fence delimiter: ```python]
state = QuantumState(3)
state.h(0).h(1).h(2)

# Get amplitudes as NumPy array
amps = state.amplitudes()

# Verify normalization
norm = np.sum(np.abs(amps)**2)
assert np.isclose(norm, 1.0)

# Find maximum amplitude
max_idx = np.argmax(np.abs(amps))
print(f"Max amplitude at |{max_idx:03b}⟩")
[archived fence delimiter: ```]

## Exception Handling

[archived fence delimiter: ```python]
from moonlab import QuantumState, QuantumError

try:
    state = QuantumState(50)  # Too many qubits
except QuantumError as e:
    print(f"Quantum error: {e}")
except MemoryError:
    print("Not enough memory")

try:
    state = QuantumState(4)
    state.cnot(0, 10)  # Invalid qubit
except QuantumError as e:
    print(f"Invalid qubit: {e}")
[archived fence delimiter: ```]

## Context Managers

[archived fence delimiter: ```python]
from moonlab import QuantumState

# Automatic cleanup
with QuantumState(10) as state:
    state.h(0)
    result = state.measure()
# Resources automatically released
[archived fence delimiter: ```]

## Performance Tips

1. **Use method chaining**: Reduces Python overhead
[archived fence delimiter:    ```python]
   state.h(0).cnot(0, 1).h(1)  # Better
[archived fence delimiter:    ```]

2. **Avoid repeated amplitudes() calls**: Cache the result
[archived fence delimiter:    ```python]
   amps = state.amplitudes()  # Call once
   # Use amps multiple times
[archived fence delimiter:    ```]

3. **Batch measurements**: Use `measure_shots()` for statistics
[archived fence delimiter:    ```python]
   counts = state.measure_shots(1000)
[archived fence delimiter:    ```]

4. **Preallocate for loops**: Reuse state with `reset()`
[archived fence delimiter:    ```python]
   state = QuantumState(10)
   for params in parameter_list:
       state.reset()
       # ... apply gates ...
[archived fence delimiter:    ```]

## See Also

- [Core API](core.md) - Complete QuantumState reference
- [Algorithms](algorithms.md) - VQE, QAOA, Grover details
- [Torch Layer](torch-layer.md) - PyTorch integration
- [Examples](../../examples/index.md) - Code examples
```
