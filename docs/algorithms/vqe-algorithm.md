# VQE Algorithm

Complete guide to the Variational Quantum Eigensolver.

## Overview

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm for finding ground state energies of quantum systems. It's particularly suited for near-term quantum devices due to its shallow circuit depth.

**Discovered**: Peruzzo et al., 2014

**Applications**:
- Molecular ground state energy
- Materials simulation
- Quantum chemistry
- Condensed matter physics

## Mathematical Foundation

### The Variational Principle

For any normalized trial state $|\psi\rangle$ and Hamiltonian $H$:

$$E_0 \leq \langle\psi|H|\psi\rangle$$

where $E_0$ is the ground state energy. Equality holds when $|\psi\rangle$ is the ground state.

### VQE Objective

Minimize the energy expectation:

$$E(\boldsymbol{\theta}) = \langle\psi(\boldsymbol{\theta})|H|\psi(\boldsymbol{\theta})\rangle$$

over parameters $\boldsymbol{\theta}$ of a parameterized quantum circuit.

### Hamiltonian Decomposition

Express the Hamiltonian as a sum of Pauli strings:

$$H = \sum_i c_i P_i$$

where each $P_i \in \{I, X, Y, Z\}^{\otimes n}$.

The expectation value becomes:

$$\langle H \rangle = \sum_i c_i \langle P_i \rangle$$

Each $\langle P_i \rangle$ is measured separately.

## Algorithm Structure

### 1. Ansatz Preparation

Prepare parameterized state $|\psi(\boldsymbol{\theta})\rangle$:

```
|0⟩──[Ry(θ₁)]──[Rz(θ₂)]──●────
                          │
|0⟩──[Ry(θ₃)]──[Rz(θ₄)]──X────
```

### 2. Energy Measurement

For each Pauli string $P_i$:

1. Change basis (H for X, S†H for Y)
2. Measure in computational basis
3. Compute parity to get $\pm 1$
4. Average over shots

### 3. Classical Optimization

Use classical optimizer (COBYLA, L-BFGS, Adam) to minimize $E(\boldsymbol{\theta})$.

### 4. Convergence

Iterate until:
- Energy change $< \epsilon$
- Gradient norm $< \epsilon$
- Maximum iterations reached

## Implementation

### Basic VQE

```c
#include "vqe.h"

int main() {
    // H2 Hamiltonian (STO-3G, d=0.74 Å)
    pauli_term_t terms[] = {
        {.coefficient = -0.8105, .pauli = "II"},
        {.coefficient =  0.1721, .pauli = "IZ"},
        {.coefficient = -0.2257, .pauli = "ZI"},
        {.coefficient =  0.1689, .pauli = "ZZ"},
        {.coefficient =  0.0454, .pauli = "XX"},
        {.coefficient =  0.0454, .pauli = "YY"}
    };

    hamiltonian_t* H = hamiltonian_create(terms, 6);

    // Configure VQE
    vqe_config_t config = {
        .num_qubits = 2,
        .num_layers = 2,
        .optimizer = VQE_OPTIMIZER_COBYLA,
        .max_iterations = 500,
        .tolerance = 1e-6
    };

    // Run VQE
    vqe_result_t result = vqe_solve(H, &config);

    printf("Ground state energy: %.6f Hartree\n", result.energy);
    printf("Converged: %s\n", result.converged ? "yes" : "no");
    printf("Iterations: %d\n", result.iterations);

    hamiltonian_destroy(H);
    return 0;
}
```

### Python Interface

```python
from moonlab.algorithms import VQE, Hamiltonian

# Create H2 Hamiltonian
H2 = Hamiltonian.h2_sto3g(bond_distance=0.74)

# Configure VQE
vqe = VQE(
    num_qubits=4,
    ansatz='hardware_efficient',
    num_layers=2,
    optimizer='adam',
    learning_rate=0.1
)

# Run optimization
result = vqe.compute_ground_state(H2)

print(f"Energy: {result.energy:.6f} Hartree")
print(f"Exact:  {H2.exact_ground_state():.6f} Hartree")
print(f"Error:  {abs(result.energy - H2.exact_ground_state()):.6f} Hartree")
```

## Ansatz Design

### Hardware-Efficient Ansatz

Alternating rotation and entangling layers:

```python
def hardware_efficient_ansatz(state, params, layers):
    """
    Hardware-efficient ansatz.

    params shape: (layers, n_qubits, 3)
    """
    n = state.num_qubits

    for l in range(layers):
        # Rotation layer
        for q in range(n):
            state.rx(q, params[l, q, 0])
            state.ry(q, params[l, q, 1])
            state.rz(q, params[l, q, 2])

        # Entangling layer (linear connectivity)
        for q in range(n - 1):
            state.cnot(q, q + 1)
```

### UCCSD Ansatz

Unitary Coupled Cluster Singles and Doubles:

```python
def uccsd_ansatz(state, params, n_electrons, n_orbitals):
    """
    UCCSD ansatz for chemistry.

    Preserves particle number and respects fermionic antisymmetry.
    """
    # Hartree-Fock initial state
    for i in range(n_electrons):
        state.x(i)

    # Singles excitations
    idx = 0
    for i in range(n_electrons):
        for a in range(n_electrons, n_orbitals):
            apply_single_excitation(state, i, a, params[idx])
            idx += 1

    # Doubles excitations
    for i in range(n_electrons):
        for j in range(i+1, n_electrons):
            for a in range(n_electrons, n_orbitals):
                for b in range(a+1, n_orbitals):
                    apply_double_excitation(state, i, j, a, b, params[idx])
                    idx += 1
```

### Symmetry-Preserving Ansatz

For systems with conserved quantities:

```python
def symmetry_preserving_ansatz(state, params, symmetry):
    """
    Ansatz that preserves specified symmetry.

    symmetry: 'particle_number', 'spin', 'point_group'
    """
    if symmetry == 'particle_number':
        # Only use particle-conserving gates
        for l in range(num_layers):
            for i in range(n_qubits - 1):
                # Givens rotation preserves particle number
                apply_givens_rotation(state, i, i+1, params[l, i])
```

## Gradient Computation

### Parameter Shift Rule

For gates of the form $e^{-i\theta G/2}$ where $G^2 = I$:

$$\frac{\partial E}{\partial \theta} = \frac{1}{2}\left[E\left(\theta + \frac{\pi}{2}\right) - E\left(\theta - \frac{\pi}{2}\right)\right]$$

```python
def parameter_shift_gradient(vqe, params, hamiltonian):
    """Compute gradient using parameter shift rule."""
    gradients = np.zeros_like(params)

    for i in range(len(params)):
        # Forward shift
        params_plus = params.copy()
        params_plus[i] += np.pi / 2
        E_plus = vqe.evaluate(params_plus, hamiltonian)

        # Backward shift
        params_minus = params.copy()
        params_minus[i] -= np.pi / 2
        E_minus = vqe.evaluate(params_minus, hamiltonian)

        gradients[i] = (E_plus - E_minus) / 2

    return gradients
```

### Simultaneous Perturbation

For high-dimensional parameter spaces:

```python
def spsa_gradient(vqe, params, hamiltonian, epsilon=0.1):
    """SPSA gradient estimate (2 evaluations regardless of dimension)."""
    delta = np.random.choice([-1, 1], size=len(params))

    E_plus = vqe.evaluate(params + epsilon * delta, hamiltonian)
    E_minus = vqe.evaluate(params - epsilon * delta, hamiltonian)

    return (E_plus - E_minus) / (2 * epsilon) * delta
```

## Optimizers

### Gradient-Free

| Optimizer | Best For | Notes |
|-----------|----------|-------|
| COBYLA | Small parameter count | Constraint-friendly |
| Nelder-Mead | Noisy landscapes | Robust to noise |
| Powell | Moderate dimensions | Good local search |

### Gradient-Based

| Optimizer | Best For | Notes |
|-----------|----------|-------|
| Adam | Large parameter count | Adaptive learning rate |
| L-BFGS | Smooth landscapes | Fast convergence |
| Natural Gradient | Variational circuits | Geometry-aware |

```python
# Configure optimizer
vqe = VQE(
    num_qubits=8,
    optimizer='adam',
    learning_rate=0.1,
    beta1=0.9,
    beta2=0.999
)

# Or use natural gradient
vqe = VQE(
    num_qubits=8,
    optimizer='natural_gradient',
    regularization=0.01  # For metric singularities
)
```

## Measurement Strategies

### Grouping Commuting Terms

Pauli strings that commute can be measured simultaneously:

```python
def group_paulis(hamiltonian):
    """Group commuting Pauli strings for efficient measurement."""
    groups = []

    for term in hamiltonian.terms:
        placed = False
        for group in groups:
            if all(commutes(term, other) for other in group):
                group.append(term)
                placed = True
                break

        if not placed:
            groups.append([term])

    return groups

# Example: H2 reduces from 6 to 2 measurement groups
groups = group_paulis(H2)
print(f"Reduced from {len(H2.terms)} to {len(groups)} measurements")
```

### Variance Reduction

Allocate shots based on term variance:

```python
def optimal_shot_allocation(hamiltonian, total_shots):
    """Allocate shots to minimize energy variance."""
    variances = [abs(term.coefficient)**2 for term in hamiltonian.terms]
    total_var = sum(np.sqrt(variances))

    shots = {}
    for term, var in zip(hamiltonian.terms, variances):
        shots[term] = int(total_shots * np.sqrt(var) / total_var)

    return shots
```

## Error Mitigation

### Zero-Noise Extrapolation

```python
def zne_energy(vqe, params, hamiltonian, noise_factors=[1, 2, 3]):
    """Zero-noise extrapolation."""
    energies = []

    for factor in noise_factors:
        vqe.set_noise_factor(factor)
        E = vqe.evaluate(params, hamiltonian)
        energies.append(E)

    # Extrapolate to zero noise
    return richardson_extrapolation(noise_factors, energies)
```

### Symmetry Verification

```python
def verify_symmetry(state, symmetry):
    """Verify state satisfies expected symmetry."""
    if symmetry == 'particle_number':
        N = sum(state.expectation_z(i) for i in range(state.num_qubits))
        return abs(N - expected_particles) < 0.1
```

## Convergence Analysis

### Energy Landscape

```python
import matplotlib.pyplot as plt

# Scan energy landscape (for 2 parameters)
theta1 = np.linspace(-np.pi, np.pi, 50)
theta2 = np.linspace(-np.pi, np.pi, 50)
T1, T2 = np.meshgrid(theta1, theta2)

E = np.zeros_like(T1)
for i in range(50):
    for j in range(50):
        E[i, j] = vqe.evaluate([T1[i,j], T2[i,j]], hamiltonian)

plt.contourf(T1, T2, E, levels=50)
plt.colorbar(label='Energy')
plt.xlabel('θ₁')
plt.ylabel('θ₂')
plt.title('VQE Energy Landscape')
```

### Barren Plateaus

Deep random circuits exhibit exponentially vanishing gradients:

$$\text{Var}[\partial_\theta E] \sim 2^{-n}$$

Mitigations:
1. Shallow circuits with local connectivity
2. Layer-by-layer training
3. Identity initialization
4. Symmetry-preserving ansätze

```python
# Layer-by-layer training to avoid barren plateaus
def layerwise_training(vqe, hamiltonian, num_layers):
    params = np.zeros(num_layers * params_per_layer)

    for l in range(num_layers):
        # Train only layer l, keeping previous fixed
        active_params = slice(l * params_per_layer, (l+1) * params_per_layer)

        result = optimize(
            lambda p: vqe.evaluate(set_params(params, active_params, p), H),
            params[active_params]
        )

        params[active_params] = result.x

    return params
```

## Molecular Systems

### H₂ Molecule

```python
from moonlab.algorithms import VQE
from moonlab.chemistry import Molecule

# Create H2 molecule
h2 = Molecule(
    atoms=['H', 'H'],
    coordinates=[[0, 0, 0], [0, 0, 0.74]]  # Angstroms
)

# Get Hamiltonian (Jordan-Wigner encoding)
H = h2.get_hamiltonian(basis='sto-3g')

# Run VQE
vqe = VQE(num_qubits=4, ansatz='uccsd')
result = vqe.compute_ground_state(H)

print(f"VQE Energy: {result.energy:.6f} Hartree")
print(f"Chemical accuracy: {abs(result.energy - h2.fci_energy()) < 0.0016}")
```

### Potential Energy Surface

```python
distances = np.linspace(0.3, 3.0, 30)
energies = []

for d in distances:
    h2 = Molecule(atoms=['H', 'H'], coordinates=[[0,0,0], [0,0,d]])
    H = h2.get_hamiltonian()

    result = vqe.compute_ground_state(H)
    energies.append(result.energy)

plt.plot(distances, energies)
plt.xlabel('Bond distance (Å)')
plt.ylabel('Energy (Hartree)')
plt.title('H₂ Potential Energy Surface')
```

## Complexity Analysis

| Component | Complexity |
|-----------|------------|
| Ansatz depth | $O(p \cdot n)$ gates |
| Pauli terms | $O(n^4)$ for chemistry |
| Measurements | $O(n^4 / \epsilon^2)$ |
| Optimization | Problem-dependent |

Total circuit evaluations scale as:
$$N_{eval} = N_{terms} \times N_{shots} \times N_{iterations}$$

## See Also

- [Tutorial: VQE](../tutorials/06-vqe-molecular-simulation.md) - Step-by-step tutorial
- [C API: VQE](../api/c/vqe.md) - Complete C API reference
- [Variational Algorithms](../concepts/variational-algorithms.md) - Theory background
- [QAOA Algorithm](qaoa-algorithm.md) - Related variational algorithm

## References

1. Peruzzo, A. et al. (2014). "A variational eigenvalue solver on a photonic quantum processor." Nature Communications, 5, 4213.
2. McClean, J. R. et al. (2016). "The theory of variational hybrid quantum-classical algorithms." New Journal of Physics, 18, 023023.
3. Kandala, A. et al. (2017). "Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets." Nature, 549, 242-246.

