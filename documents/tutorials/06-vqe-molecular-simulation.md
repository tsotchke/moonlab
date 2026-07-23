# Tutorial 06: VQE Molecular Simulation

Simulate molecular ground states with the Variational Quantum Eigensolver.

**Duration**: 60 minutes
**Prerequisites**: [Tutorial 04](04-grovers-search.md)
**Difficulty**: Intermediate

## Learning Objectives

By the end of this tutorial, you will:

- Understand the VQE algorithm structure
- Map molecular Hamiltonians to qubit operators
- Implement and optimize variational ansätze
- Compute the ground state energy of H₂

## The Problem

Finding the ground state energy of molecules is crucial for:
- Drug discovery
- Materials science
- Catalysis design
- Understanding chemical reactions

Classically, this scales exponentially with system size. Quantum computers offer a potential advantage.

## VQE Overview

**Variational Quantum Eigensolver** is a hybrid quantum-classical algorithm:

1. Prepare a parameterized quantum state $|\psi(\theta)\rangle$
2. Measure the energy $E(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle$
3. Use classical optimizer to minimize $E(\theta)$
4. Repeat until convergence

## Step 1: The Molecular Hamiltonian

For the H₂ molecule, the electronic Hamiltonian in second quantization:

$$H = \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} g_{pqrs} a_p^\dagger a_q^\dagger a_r a_s$$

Using the Jordan-Wigner transformation, this maps to qubit operators (Pauli strings).

### Simplified H₂ Hamiltonian

For H₂ at the equilibrium bond distance (0.74 Å), the `moonlab.chemistry` engine builds
the two-qubit Hamiltonian from real STO-3G integrals -- do not hardcode the coefficients:

```python
from moonlab.chemistry import Hamiltonian

H = Hamiltonian.h2_sto3g(bond_distance=0.74)

# Five electronic Pauli terms, computed by the C engine.
H2_HAMILTONIAN = {p: c for c, p in H.terms}
print(H2_HAMILTONIAN)
# {'II': -1.0521, 'IZ': 0.3988, 'ZI': -0.3988, 'ZZ': -0.0113, 'XX': 0.1809}

# The constant nuclear-repulsion energy is tracked separately and added to the
# total energy (it is NOT folded into the Pauli terms).
NUCLEAR_REPULSION = H.nuclear_repulsion
print(NUCLEAR_REPULSION)   # 0.7165 Hartree
```

The full molecular energy is the Pauli-sum expectation plus the nuclear-repulsion constant:
$$E = \underbrace{-1.0521\,\langle I\rangle + 0.3988\,\langle Z_0\rangle - 0.3988\,\langle Z_1\rangle - 0.0113\,\langle Z_0Z_1\rangle + 0.1809\,\langle X_0X_1\rangle}_{\text{electronic}} + \underbrace{0.7165}_{\text{nuclear}}$$

## Step 2: The Variational Ansatz

We'll use a hardware-efficient ansatz:

```python
from moonlab import QuantumState
import numpy as np

def apply_ansatz(state, params, num_layers=2):
    """
    Hardware-efficient ansatz for 2 qubits.

    params: array of shape (num_layers, 2, 3)
            [layer][qubit][Rx, Ry, Rz angles]
    """
    n_qubits = 2

    for layer in range(num_layers):
        # Single-qubit rotations
        for q in range(n_qubits):
            state.rx(q, params[layer, q, 0])
            state.ry(q, params[layer, q, 1])
            state.rz(q, params[layer, q, 2])

        # Entangling layer (except last layer)
        if layer < num_layers - 1:
            state.cnot(0, 1)

    return state

# Example: Create ansatz with random parameters
num_layers = 2
params = np.random.uniform(-np.pi, np.pi, (num_layers, 2, 3))

state = QuantumState(2)
apply_ansatz(state, params)
```

## Step 3: Measuring the Energy

We need the expectation value of each Pauli term in the Hamiltonian. In simulation we can
read the exact state vector (`QuantumState.get_statevector`) and contract it against each
Pauli operator with NumPy -- no sampling noise:

```python
import numpy as np

# Single-qubit Pauli matrices.
_PAULI = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}

def pauli_matrix(pauli_string):
    """Full operator for a Pauli string, e.g. 'XX' -> X (x) X.

    Qubit 0 is the least-significant bit of the state index, so we build the
    Kronecker product from the last character to the first.
    """
    op = np.array([[1.0 + 0j]])
    for p in reversed(pauli_string):
        op = np.kron(op, _PAULI[p])
    return op

def expectation_pauli(state, pauli_string):
    """Exact <psi|P|psi> from the state vector."""
    psi = state.get_statevector()
    return complex(np.vdot(psi, pauli_matrix(pauli_string) @ psi)).real

def compute_energy_exact(state, hamiltonian, nuclear_repulsion=0.0):
    """Total energy E = nuclear_repulsion + sum_k coeff_k <P_k>."""
    return nuclear_repulsion + sum(coeff * expectation_pauli(state, pauli)
                                   for pauli, coeff in hamiltonian.items())
```

On real hardware you would instead rotate each qubit into the measurement basis
(`H` for X, `Sdg` then `H` for Y), sample bitstrings, and average the parity -- trading the
exact contraction above for shot statistics.

## Step 4: Classical Optimization

```python
from scipy.optimize import minimize

def vqe_cost(params_flat, hamiltonian, num_layers):
    """Cost function for VQE optimization."""
    params = params_flat.reshape(num_layers, 2, 3)

    state = QuantumState(2)
    apply_ansatz(state, params, num_layers)

    energy = compute_energy_exact(state, hamiltonian, NUCLEAR_REPULSION)
    return energy

# Initial parameters
num_layers = 2
initial_params = np.random.uniform(-np.pi, np.pi, (num_layers, 2, 3))

# Optimize
result = minimize(
    vqe_cost,
    initial_params.flatten(),
    args=(H2_HAMILTONIAN, num_layers),
    method='COBYLA',
    options={'maxiter': 500, 'rhobeg': 0.5}
)

from moonlab.chemistry import Hamiltonian
exact = Hamiltonian.h2_sto3g(bond_distance=0.74).exact_ground_state()

print(f"Optimized energy: {result.fun:.6f} Hartree")
print(f"Exact H2 energy:  {exact:.6f} Hartree")
print(f"Error: {abs(result.fun - exact):.6f} Hartree")
```

## Using the Built-in VQE

Moonlab provides a convenient VQE class:

```python
from moonlab.algorithms import VQE

# H2 maps to a 2-qubit Hamiltonian. Select the optimizer with the `optimizer`
# string: 'cobyla', 'lbfgs', 'adam', 'gradient_descent', or
# 'natural_gradient' (quantum natural gradient / QNG).
vqe = VQE(num_qubits=2, num_layers=2, optimizer='adam', learning_rate=0.1)

# Solve for H2 at equilibrium distance.
result = vqe.solve_h2(bond_distance=0.74)

print(f"Ground state energy: {result['energy']:.6f} Hartree")
print(f"Energy in kcal/mol: {result['energy_kcal_mol']:.2f}")
print(f"Converged: {result['converged']}")
print(f"Iterations: {result['num_iterations']}")
```

The quantum natural gradient optimizer preconditions the gradient by the Fubini-Study
metric of the ansatz state and reaches the exact STO-3G ground state on H₂:

```python
vqe = VQE(num_qubits=2, num_layers=2, optimizer='natural_gradient', regularization=1e-3)
result = vqe.solve_h2(bond_distance=0.74)
print(f"QNG ground state energy: {result['energy']:.6f} Hartree")
```

## Step 5: Potential Energy Surface

Compute energy vs. bond distance:

The bond distance passed to `solve_h2` selects the point on the surface; the chemistry
engine recomputes the STO-3G coefficients for each geometry (`h2_sto3g_pauli_coeffs`), so
you never hand-build a Hamiltonian:

```python
import numpy as np
import matplotlib.pyplot as plt
from moonlab.algorithms import VQE
from moonlab.chemistry import Hamiltonian

# Compute the potential energy curve.
distances = np.linspace(0.3, 3.0, 20)
vqe_energies = []
exact_energies = []

vqe = VQE(num_qubits=2, num_layers=2, optimizer='natural_gradient')
for d in distances:
    vqe_energies.append(vqe.solve_h2(bond_distance=d)['energy'])
    exact_energies.append(Hamiltonian.h2_sto3g(bond_distance=d).exact_ground_state())
    print(f"d = {d:.2f} A: E = {vqe_energies[-1]:.6f} Hartree")

# The equilibrium is the minimum of the curve (~0.74 A, ~-1.142 Hartree).
i = int(np.argmin(exact_energies))
print(f"Equilibrium near d = {distances[i]:.2f} A, E = {exact_energies[i]:.6f} Hartree")

# Plot.
plt.figure(figsize=(10, 6))
plt.plot(distances, exact_energies, 'r-', label='Exact (STO-3G)')
plt.plot(distances, vqe_energies, 'bo', label='VQE')
plt.xlabel('Bond Distance (Å)')
plt.ylabel('Energy (Hartree)')
plt.title('H₂ Potential Energy Curve from VQE')
plt.legend()
plt.grid(True)
plt.savefig('h2_pes.png')
plt.show()
```

## Gradient-Based Optimization

The **parameter shift rule** gives exact gradients:

```python
def parameter_shift_gradient(params, hamiltonian, num_layers):
    """Compute gradient using parameter shift rule."""
    gradients = np.zeros_like(params)
    shift = np.pi / 2

    for i in range(len(params)):
        # f(θ + π/2)
        params_plus = params.copy()
        params_plus[i] += shift
        e_plus = vqe_cost(params_plus, hamiltonian, num_layers)

        # f(θ - π/2)
        params_minus = params.copy()
        params_minus[i] -= shift
        e_minus = vqe_cost(params_minus, hamiltonian, num_layers)

        # Gradient
        gradients[i] = (e_plus - e_minus) / 2

    return gradients

# Gradient descent
params = np.random.uniform(-np.pi, np.pi, (num_layers, 2, 3)).flatten()
learning_rate = 0.1
energies = []

for i in range(100):
    energy = vqe_cost(params, H2_HAMILTONIAN, num_layers)
    energies.append(energy)

    grad = parameter_shift_gradient(params, H2_HAMILTONIAN, num_layers)
    params -= learning_rate * grad

    if i % 10 == 0:
        print(f"Iteration {i}: E = {energy:.6f}")

print(f"Final energy: {energies[-1]:.6f}")
```

## UCCSD Ansatz for Chemistry

For a chemistry-motivated, particle-number-preserving circuit, select the built-in UCCSD
ansatz on the VQE solver. It initializes the Hartree-Fock reference and applies the singles
and doubles excitation operators for you; pass the electron count with `num_electrons`:

```python
from moonlab.algorithms import VQE

# UCCSD on a 4-qubit active space with 2 electrons.
vqe = VQE(num_qubits=4, ansatz='uccsd', num_electrons=2,
          optimizer='natural_gradient')
result = vqe.solve_lih(bond_distance=1.6)
print(f"UCCSD variational energy: {result['energy']:.6f} Hartree")
```

The returned energy is a variational upper bound on the true ground state (the Rayleigh-Ritz
principle guarantees `E(θ) ≥ E₀`); how tightly it approaches the exact
`Hamiltonian.lih_sto3g(1.6).exact_ground_state()` depends on the ansatz depth, electron
count, and optimizer. Increase `num_layers` or compare optimizers to tighten the gap.

## Larger Molecules

For larger molecules (LiH, H₂O, etc.):

```python
# LiH: frozen-core active space maps to 4 qubits.
vqe = VQE(num_qubits=4, num_layers=4, optimizer='natural_gradient')
result = vqe.solve_lih(bond_distance=1.6)

print(f"LiH ground state: {result['energy']:.6f} Hartree")

# H2O: the built-in Hamiltonian is an 8-qubit active space.
vqe_h2o = VQE(num_qubits=8, num_layers=4, optimizer='adam')
result_h2o = vqe_h2o.solve_h2o()

print(f"H2O ground state: {result_h2o['energy']:.6f} Hartree")
```

## Exercises

### Exercise 1: Vary Ansatz Depth

How does the accuracy change with the number of layers? Plot final energy vs. num_layers.

### Exercise 2: Different Optimizers

Compare COBYLA, Adam, and L-BFGS. Which converges fastest? Which gives best accuracy?

### Exercise 3: He Atom

Implement VQE for the Helium atom (simpler than H₂).

### Exercise 4: Noise Effects

Add depolarizing noise to the circuit. How does it affect the energy accuracy?

## Key Takeaways

1. **VQE** is a hybrid quantum-classical algorithm
2. **Ansatz design** is crucial for accuracy
3. **Parameter shift rule** gives exact gradients
4. **Hamiltonian measurement** requires many Pauli string measurements
5. **Classical optimization** finds the ground state

## Next Steps

Apply optimization to combinatorial problems:

**[07. QAOA Optimization →](07-qaoa-optimization.md)**

## Further Reading

- [VQE Algorithm](../algorithms/vqe-algorithm.md) - Full mathematical treatment
- [Variational Algorithms](../concepts/variational-algorithms.md) - Theory background
- [C API: VQE](../api/c/vqe.md) - Low-level implementation
- Peruzzo et al. (2014). "A variational eigenvalue solver on a photonic quantum processor." Nature Communications, 5, 4213.

