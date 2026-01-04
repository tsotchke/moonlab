# Custom Hamiltonians Guide

Construct and simulate custom Hamiltonians for VQE, time evolution, and ground state calculations.

## Overview

Hamiltonians describe the energy of quantum systems. Moonlab provides flexible APIs for building Hamiltonians from Pauli strings, molecular integrals, or custom operators.

## Hamiltonian Representations

### Pauli String Decomposition

Any Hamiltonian can be written as a sum of Pauli strings:

$$H = \sum_i c_i P_i$$

where $P_i \in \{I, X, Y, Z\}^{\otimes n}$ and $c_i \in \mathbb{R}$.

### C API

```c
#include "algorithms/hamiltonian.h"

// Create empty Hamiltonian
hamiltonian_t* H = hamiltonian_create(4);  // 4 qubits

// Add Pauli terms: coefficient * Pauli string
// Format: "IXYZ" means I⊗X⊗Y⊗Z

hamiltonian_add_term(H, 0.5, "XXII");   // 0.5 * X₀X₁
hamiltonian_add_term(H, 0.5, "YYII");   // 0.5 * Y₀Y₁
hamiltonian_add_term(H, 0.3, "ZZII");   // 0.3 * Z₀Z₁
hamiltonian_add_term(H, -1.0, "ZIII");  // -1.0 * Z₀
hamiltonian_add_term(H, -1.0, "IZII");  // -1.0 * Z₁

// Compute expectation value ⟨ψ|H|ψ⟩
double energy = hamiltonian_expectation(H, state);

printf("Energy: %.6f\n", energy);

hamiltonian_destroy(H);
```

### Python API

```python
from moonlab import Hamiltonian, QuantumState

# Create Hamiltonian from terms
H = Hamiltonian(num_qubits=4)

H.add_term(0.5, "XXII")
H.add_term(0.5, "YYII")
H.add_term(0.3, "ZZII")
H.add_term(-1.0, "ZIII")
H.add_term(-1.0, "IZII")

# Or create from dictionary
H = Hamiltonian.from_dict({
    "XXII": 0.5,
    "YYII": 0.5,
    "ZZII": 0.3,
    "ZIII": -1.0,
    "IZII": -1.0
})

# Evaluate
state = QuantumState(4)
state.h(0)
state.cnot(0, 1)

energy = H.expectation(state)
print(f"Energy: {energy:.6f}")
```

## Common Hamiltonians

### Transverse-Field Ising Model

$$H = -J \sum_{\langle i,j \rangle} Z_i Z_j - h \sum_i X_i$$

```python
def transverse_ising(n_qubits: int, J: float = 1.0, h: float = 0.5) -> Hamiltonian:
    """
    1D Transverse-field Ising model with periodic boundary conditions.

    Args:
        n_qubits: Number of spins
        J: Coupling strength
        h: Transverse field strength
    """
    H = Hamiltonian(n_qubits)

    # ZZ interactions
    for i in range(n_qubits):
        j = (i + 1) % n_qubits  # Periodic BC
        pauli = ['I'] * n_qubits
        pauli[i] = 'Z'
        pauli[j] = 'Z'
        H.add_term(-J, ''.join(pauli))

    # Transverse field
    for i in range(n_qubits):
        pauli = ['I'] * n_qubits
        pauli[i] = 'X'
        H.add_term(-h, ''.join(pauli))

    return H

# Example
H_ising = transverse_ising(8, J=1.0, h=0.5)
print(f"Number of terms: {len(H_ising)}")
```

### Heisenberg Model

$$H = J \sum_{\langle i,j \rangle} (X_i X_j + Y_i Y_j + Z_i Z_j)$$

```python
def heisenberg_chain(n_qubits: int, J: float = 1.0) -> Hamiltonian:
    """Heisenberg XXX chain."""
    H = Hamiltonian(n_qubits)

    for i in range(n_qubits - 1):
        for pauli in ['X', 'Y', 'Z']:
            term = ['I'] * n_qubits
            term[i] = pauli
            term[i + 1] = pauli
            H.add_term(J, ''.join(term))

    return H
```

### Molecular Hamiltonians

For chemistry applications, use Jordan-Wigner transformation:

```python
from moonlab.chemistry import MolecularHamiltonian, jordan_wigner

# Define molecular system
mol = MolecularHamiltonian(
    geometry=[
        ('H', (0.0, 0.0, 0.0)),
        ('H', (0.0, 0.0, 0.74))  # Bond length in Angstroms
    ],
    basis='sto-3g'
)

# Get one- and two-electron integrals
h1, h2 = mol.get_integrals()

# Transform to qubit Hamiltonian
H_qubit = jordan_wigner(h1, h2)

print(f"H₂ Hamiltonian: {len(H_qubit)} Pauli terms")
print(f"Qubits required: {H_qubit.num_qubits}")
```

## Building from Operators

### Ladder Operators

```python
from moonlab import PauliOperator

# Creation and annihilation operators (Jordan-Wigner)
def creation(n_qubits: int, site: int) -> Hamiltonian:
    """Fermionic creation operator a†ᵢ."""
    # a† = (X - iY) / 2 with Jordan-Wigner string
    H = Hamiltonian(n_qubits)

    # Jordan-Wigner string: Z on all sites before i
    prefix = 'Z' * site + 'I' * (n_qubits - site)

    # (X - iY) / 2
    term_x = list(prefix)
    term_x[site] = 'X'
    H.add_term(0.5, ''.join(term_x))

    term_y = list(prefix)
    term_y[site] = 'Y'
    H.add_term(-0.5j, ''.join(term_y))

    return H

def annihilation(n_qubits: int, site: int) -> Hamiltonian:
    """Fermionic annihilation operator aᵢ."""
    # a = (X + iY) / 2 with Jordan-Wigner string
    H = Hamiltonian(n_qubits)

    prefix = 'Z' * site + 'I' * (n_qubits - site)

    term_x = list(prefix)
    term_x[site] = 'X'
    H.add_term(0.5, ''.join(term_x))

    term_y = list(prefix)
    term_y[site] = 'Y'
    H.add_term(0.5j, ''.join(term_y))

    return H
```

### Number Operators

```python
def number_operator(n_qubits: int, site: int) -> Hamiltonian:
    """Number operator nᵢ = a†ᵢaᵢ = (1 - Zᵢ) / 2."""
    H = Hamiltonian(n_qubits)

    H.add_term(0.5, 'I' * n_qubits)  # Identity term

    z_term = ['I'] * n_qubits
    z_term[site] = 'Z'
    H.add_term(-0.5, ''.join(z_term))

    return H
```

## Hamiltonian Operations

### Addition and Scaling

```python
H1 = transverse_ising(4, J=1.0, h=0.0)
H2 = transverse_ising(4, J=0.0, h=1.0)

# Linear combination
H_combined = 0.5 * H1 + 0.5 * H2

# In-place operations
H1 += H2
H1 *= 2.0
```

### Commutators

```python
def commutator(A: Hamiltonian, B: Hamiltonian) -> Hamiltonian:
    """Compute [A, B] = AB - BA."""
    return A @ B - B @ A

# Example: check if operators commute
H_x = Hamiltonian.from_dict({"XIII": 1.0})
H_z = Hamiltonian.from_dict({"ZIII": 1.0})

comm = commutator(H_x, H_z)
print(f"[X, Z] has {len(comm)} terms")  # Should be 1 term: 2iY
```

### Matrix Representation

```c
// Get dense matrix (for small systems)
complex_t* matrix = hamiltonian_to_matrix(H);

// Get sparse representation
sparse_matrix_t* sparse = hamiltonian_to_sparse(H);

// Diagonalize (exact ground state)
double* eigenvalues;
complex_t* eigenvectors;
hamiltonian_diagonalize(H, &eigenvalues, &eigenvectors);

printf("Ground state energy: %.6f\n", eigenvalues[0]);
```

## Time Evolution

### Trotterization

```python
from moonlab import QuantumState
from moonlab.algorithms import trotter_evolution

# Evolve |ψ(t)⟩ = e^{-iHt}|ψ(0)⟩
state = QuantumState(4)
state.h(0)
state.cnot(0, 1)

H = transverse_ising(4)

# First-order Trotter
evolved = trotter_evolution(
    state,
    H,
    time=1.0,
    steps=100,
    order=1
)

# Second-order (Suzuki-Trotter)
evolved = trotter_evolution(
    state,
    H,
    time=1.0,
    steps=50,
    order=2
)
```

### Gate Decomposition

```python
from moonlab.algorithms import hamiltonian_to_circuit

# Convert Hamiltonian evolution to circuit
circuit = hamiltonian_to_circuit(
    H,
    time=0.1,
    trotter_steps=10
)

# Apply to state
state = QuantumState(4)
circuit.apply(state)
```

## VQE with Custom Hamiltonians

```python
from moonlab.algorithms import VQE
from moonlab.ansatz import EfficientSU2

# Custom Hamiltonian
H = transverse_ising(6, J=1.0, h=0.5)

# Ansatz
ansatz = EfficientSU2(num_qubits=6, num_layers=3)

# VQE solver
vqe = VQE(
    hamiltonian=H,
    ansatz=ansatz,
    optimizer='COBYLA',
    shots=1000
)

result = vqe.run()

print(f"Ground state energy: {result.optimal_value:.6f}")
print(f"Optimal parameters: {result.optimal_params}")
```

## MPO Representation

For tensor network methods, convert to Matrix Product Operator:

```python
from moonlab.tensor_network import Hamiltonian_to_MPO

# Convert to MPO with bond dimension D
mpo = Hamiltonian_to_MPO(H, max_bond=16)

# Use with DMRG
from moonlab.tensor_network import DMRG

dmrg = DMRG(mpo, bond_dimension=64)
ground_state_mps, energy = dmrg.run()

print(f"DMRG energy: {energy:.8f}")
```

## Performance Tips

### Grouping Commuting Terms

```python
from moonlab.algorithms import group_commuting_terms

# Group terms that can be measured simultaneously
groups = group_commuting_terms(H)

print(f"Original terms: {len(H)}")
print(f"Measurement groups: {len(groups)}")

# Reduces measurement overhead in VQE
```

### Sparse Evaluation

```python
# For Hamiltonians with many terms, use sparse mode
H.set_sparse_threshold(1000)  # Use sparse if > 1000 terms

# Parallel evaluation
energy = H.expectation(state, parallel=True, num_threads=8)
```

## Reference Hamiltonians

| System | Formula | Typical Terms | Qubits |
|--------|---------|---------------|--------|
| Ising 1D | $-J\sum ZZ - h\sum X$ | 2n | n |
| Heisenberg | $J\sum (XX + YY + ZZ)$ | 3(n-1) | n |
| Hubbard | $-t\sum(a†a + h.c.) + U\sum n↑n↓$ | O(n²) | 2n |
| H₂ (STO-3G) | JW transformed | 15 | 4 |
| LiH (STO-3G) | JW transformed | 631 | 12 |

## See Also

- [VQE Algorithm](../algorithms/vqe-algorithm.md)
- [DMRG Algorithm](../algorithms/dmrg-algorithm.md)
- [Tutorial: VQE Molecular Simulation](../tutorials/06-vqe-molecular-simulation.md)
- [API: Hamiltonian](../api/python/algorithms.md#hamiltonian)
