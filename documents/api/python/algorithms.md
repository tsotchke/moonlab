# Python Algorithms API

Complete reference for quantum algorithms in the Python API.

**Module**: `moonlab.algorithms`

## Overview

The algorithms module provides production-ready implementations of major quantum algorithms:

- **VQE**: Variational Quantum Eigensolver for molecular simulation
- **QAOA**: Quantum Approximate Optimization Algorithm
- **Grover**: Quantum search with quadratic speedup
- **Bell Test**: CHSH inequality verification

## VQE (Variational Quantum Eigensolver)

Find ground state energies of molecular Hamiltonians using variational optimization.

### VQE Class

```python
VQE(
    num_qubits: int,
    num_layers: int = 2,
    optimizer_type: int = VQE.OPTIMIZER_ADAM
)
```

**Parameters**:
- `num_qubits`: Number of qubits for the ansatz
- `num_layers`: Ansatz circuit depth
- `optimizer_type`: Optimization algorithm

**Optimizer Types**:
- `VQE.OPTIMIZER_ADAM` - Adam optimizer (default)
- `VQE.OPTIMIZER_GRADIENT_DESCENT` - Vanilla gradient descent
- `VQE.OPTIMIZER_LBFGS` - L-BFGS quasi-Newton
- `VQE.OPTIMIZER_COBYLA` - COBYLA (derivative-free)

### Methods

#### solve_h2

```python
solve_h2(bond_distance: float = 0.74) -> dict
```

Find ground state of H$_2$ molecule.

**Parameters**:
- `bond_distance`: H-H bond distance in Angstroms (equilibrium: 0.74 Å)

**Returns**: Dictionary with results

**Example**:
```python
from moonlab.algorithms import VQE

vqe = VQE(num_qubits=4, num_layers=2)
result = vqe.solve_h2(bond_distance=0.74)

print(f"Energy: {result['energy']:.6f} Hartree")
print(f"Energy: {result['energy_kcal_mol']:.2f} kcal/mol")
print(f"Converged: {result['converged']}")
print(f"Iterations: {result['num_iterations']}")
```

#### solve_lih

```python
solve_lih(bond_distance: float = 1.6) -> dict
```

Find ground state of LiH molecule.

**Parameters**:
- `bond_distance`: Li-H bond distance in Angstroms (equilibrium: 1.6 Å)

#### solve_h2o

```python
solve_h2o() -> dict
```

Find ground state of H$_2$O molecule.

#### solve

```python
solve(hamiltonian: MolecularHamiltonian) -> dict
```

Solve for custom Hamiltonian.

#### compute_energy

```python
compute_energy(params: np.ndarray) -> float
```

Compute energy for given variational parameters.

### Result Dictionary

| Key | Type | Description |
|-----|------|-------------|
| `energy` | float | Ground state energy (Hartree) |
| `energy_kcal_mol` | float | Energy in kcal/mol |
| `optimal_params` | np.ndarray | Optimized circuit parameters |
| `converged` | bool | Convergence status |
| `num_iterations` | int | Iterations performed |
| `gradient_norm` | float | Final gradient norm |

### MolecularHamiltonian

Build custom Hamiltonians as sums of Pauli strings.

```python
from moonlab.algorithms import MolecularHamiltonian, VQE

# Create 4-qubit Hamiltonian
H = MolecularHamiltonian(4)
H.add_term(0.5, "ZIZI")   # ZZ on qubits 0,2
H.add_term(-0.3, "XIXI")  # XX on qubits 0,2
H.add_term(0.2, "IIZI")   # Z on qubit 1

# Solve
vqe = VQE(num_qubits=4, num_layers=3)
result = vqe.solve(H)
```

## QAOA (Quantum Approximate Optimization Algorithm)

Solve combinatorial optimization problems.

### QAOA Class

```python
QAOA(num_qubits: int, num_layers: int = 3)
```

**Parameters**:
- `num_qubits`: Number of qubits (vertices in MaxCut)
- `num_layers`: QAOA depth (p parameter)

### Methods

#### solve_maxcut

```python
solve_maxcut(
    edges: List[Tuple[int, int]],
    weights: Optional[List[float]] = None
) -> dict
```

Solve MaxCut problem on a graph.

**Parameters**:
- `edges`: List of (u, v) edge tuples
- `weights`: Optional edge weights (default: all 1.0)

**Example**:
```python
from moonlab.algorithms import QAOA

# Pentagon graph
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]

qaoa = QAOA(num_qubits=5, num_layers=3)
result = qaoa.solve_maxcut(edges)

print(f"Best cut: {result['best_bitstring']:05b}")
print(f"Cut value: {result['best_cost']}")
print(f"Expectation: {result['expectation']:.4f}")
```

#### solve_ising

```python
solve_ising(J: np.ndarray, h: np.ndarray) -> dict
```

Solve general Ising model: $H = \sum_{ij} J_{ij} \sigma_i \sigma_j + \sum_i h_i \sigma_i$

**Parameters**:
- `J`: Coupling matrix (n × n)
- `h`: Field vector (n,)

**Example**:
```python
import numpy as np

# 4-spin Ising model
J = np.array([
    [0, -1, 0, 0.5],
    [-1, 0, -1, 0],
    [0, -1, 0, -1],
    [0.5, 0, -1, 0]
])
h = np.array([0.1, 0, -0.1, 0])

qaoa = QAOA(num_qubits=4, num_layers=4)
result = qaoa.solve_ising(J, h)
```

#### compute_expectation

```python
compute_expectation(gamma: np.ndarray, beta: np.ndarray) -> float
```

Compute cost expectation for given QAOA parameters.

#### evaluate_bitstring

```python
evaluate_bitstring(bitstring: int) -> float
```

Evaluate cost of a specific solution.

### Result Dictionary

| Key | Type | Description |
|-----|------|-------------|
| `expectation` | float | Cost function expectation |
| `optimal_gamma` | np.ndarray | Optimal phase separation angles |
| `optimal_beta` | np.ndarray | Optimal mixer angles |
| `best_bitstring` | int | Best solution found |
| `best_cost` | float | Cost of best solution |
| `converged` | bool | Convergence status |

## Grover (Quantum Search)

Quadratic speedup for unstructured search.

### Grover Class

```python
Grover(num_qubits: int)
```

**Parameters**:
- `num_qubits`: Search space size = $2^{\text{num\_qubits}}$

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `optimal_iterations` | int | Optimal Grover iterations ($\approx \frac{\pi}{4}\sqrt{N}$) |
| `state` | QuantumState | Underlying quantum state |

### Methods

#### search

```python
search(
    marked_state: int,
    num_iterations: Optional[int] = None
) -> dict
```

Search for marked state.

**Parameters**:
- `marked_state`: Target state index (0 to $2^n - 1$)
- `num_iterations`: Iterations (default: optimal)

**Example**:
```python
from moonlab.algorithms import Grover

grover = Grover(num_qubits=10)  # Search space of 1024
result = grover.search(marked_state=42)

print(f"Target: 42")
print(f"Found: {result['found_state']}")
print(f"Success: {result['success']}")
print(f"Probability: {result['probability']:.4f}")
print(f"Iterations: {result['iterations_used']}")
```

#### step

```python
step(marked_state: int) -> None
```

Perform single Grover iteration.

#### oracle

```python
oracle(marked_state: int) -> None
```

Apply Grover oracle (phase flip on marked state).

#### diffusion

```python
diffusion() -> None
```

Apply diffusion operator (inversion about mean).

#### probabilities

```python
probabilities() -> np.ndarray
```

Get measurement probabilities.

### Result Dictionary

| Key | Type | Description |
|-----|------|-------------|
| `found_state` | int | Measured state |
| `success` | bool | Whether marked state was found |
| `probability` | float | Success probability |
| `iterations_used` | int | Iterations performed |

### Step-by-Step Example

```python
from moonlab.algorithms import Grover
import matplotlib.pyplot as plt

grover = Grover(num_qubits=6)
marked = 42

# Initialize superposition
grover.state.reset()
for q in range(6):
    grover.state.h(q)

# Track probability over iterations
probs_marked = []
for i in range(grover.optimal_iterations):
    grover.step(marked)
    probs_marked.append(grover.state.probability(marked))

plt.plot(probs_marked)
plt.xlabel("Iteration")
plt.ylabel("P(marked)")
plt.title("Grover's Algorithm Amplitude Amplification")
plt.show()
```

## BellTest (CHSH Inequality)

Verify quantum correlations violate classical bounds.

### Bell State Types

```python
BellTest.PHI_PLUS   # |Φ+⟩ = (|00⟩ + |11⟩)/√2
BellTest.PHI_MINUS  # |Φ-⟩ = (|00⟩ - |11⟩)/√2
BellTest.PSI_PLUS   # |Ψ+⟩ = (|01⟩ + |10⟩)/√2
BellTest.PSI_MINUS  # |Ψ-⟩ = (|01⟩ - |10⟩)/√2
```

### Methods

#### create_bell_state

```python
BellTest.create_bell_state(
    state: QuantumState,
    qubit_a: int,
    qubit_b: int,
    bell_type: int = BellTest.PHI_PLUS
) -> None
```

Create Bell state on two qubits.

#### chsh_test

```python
BellTest.chsh_test(
    state: QuantumState,
    qubit_a: int,
    qubit_b: int,
    num_measurements: int = 10000
) -> dict
```

Perform CHSH Bell test.

**Classical bound**: $S \leq 2$
**Quantum bound**: $S \leq 2\sqrt{2} \approx 2.828$

**Example**:
```python
from moonlab import QuantumState
from moonlab.algorithms import BellTest

state = QuantumState(2)
BellTest.create_bell_state(state, 0, 1, BellTest.PHI_PLUS)

result = BellTest.chsh_test(state, 0, 1, num_measurements=10000)

print(f"CHSH S = {result['chsh']:.4f}")
print(f"Violates classical bound: {result['violates_classical']}")
print(f"Correlations:")
for key, val in result['correlations'].items():
    print(f"  {key}: {val:.4f}")
```

#### quick_test

```python
BellTest.quick_test(
    num_qubits: int = 2,
    num_measurements: int = 1000
) -> dict
```

Quick Bell test with automatic state preparation.

### Result Dictionary

| Key | Type | Description |
|-----|------|-------------|
| `chsh` | float | CHSH parameter S |
| `correlations` | dict | E(a,b), E(a,b'), E(a',b), E(a',b') |
| `violates_classical` | bool | Whether $S > 2$ |
| `num_measurements` | int | Measurements performed |
| `variance` | float | Statistical variance |

## Convenience Functions

### run_vqe_h2

```python
run_vqe_h2(bond_distance: float = 0.74, num_layers: int = 2) -> dict
```

Quick VQE for H$_2$ molecule.

### run_qaoa_maxcut

```python
run_qaoa_maxcut(edges: List[Tuple[int, int]], num_layers: int = 3) -> dict
```

Quick QAOA for MaxCut.

### run_grover

```python
run_grover(num_qubits: int, marked_state: int) -> dict
```

Quick Grover search.

### run_bell_test

```python
run_bell_test(num_measurements: int = 10000) -> dict
```

Quick Bell test.

## Complete Example: VQE Energy Surface

```python
from moonlab.algorithms import VQE
import numpy as np
import matplotlib.pyplot as plt

# Compute H2 potential energy curve
distances = np.linspace(0.3, 3.0, 20)
energies = []

vqe = VQE(num_qubits=4, num_layers=3)

for d in distances:
    result = vqe.solve_h2(bond_distance=d)
    energies.append(result['energy'])
    print(f"d = {d:.2f} Å: E = {result['energy']:.6f} Ha")

plt.plot(distances, energies, 'o-')
plt.xlabel("Bond Distance (Å)")
plt.ylabel("Energy (Hartree)")
plt.title("H₂ Potential Energy Curve (VQE)")
plt.axhline(y=-1.137, color='r', linestyle='--', label='Exact')
plt.legend()
plt.show()
```

## See Also

- [Core API](core.md) - QuantumState, Gates
- [ML API](ml.md) - Quantum machine learning
- [C API: VQE](../c/vqe.md) - Low-level VQE reference
- [C API: QAOA](../c/qaoa.md) - Low-level QAOA reference
- [C API: Grover](../c/grover.md) - Low-level Grover reference
