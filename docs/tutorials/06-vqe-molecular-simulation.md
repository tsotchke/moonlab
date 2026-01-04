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

For H₂ at the equilibrium bond distance (0.74 Å):

```python
# H2 Hamiltonian coefficients (STO-3G basis)
H2_HAMILTONIAN = {
    'II': -0.8105,
    'IZ': 0.1721,
    'ZI': -0.2257,
    'ZZ': 0.1689,
    'XX': 0.0454,
    'YY': 0.0454,
}
```

The full Hamiltonian:
$$H = -0.8105 \cdot I + 0.1721 \cdot Z_0 - 0.2257 \cdot Z_1 + 0.1689 \cdot Z_0Z_1 + 0.0454 \cdot X_0X_1 + 0.0454 \cdot Y_0Y_1$$

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

We need to measure each Pauli term in the Hamiltonian:

```python
def measure_pauli_string(state, pauli_string, shots=1000):
    """
    Measure expectation value of a Pauli string.

    pauli_string: e.g., "XY" means X on qubit 0, Y on qubit 1
    """
    n = len(pauli_string)

    # Change to appropriate basis for measurement
    for q, p in enumerate(pauli_string):
        if p == 'X':
            state.h(q)
        elif p == 'Y':
            state.sdg(q)
            state.h(q)
        # Z and I don't need basis change

    # Count parity
    parity_sum = 0
    for _ in range(shots):
        result = state.measure_all()

        # Compute parity (XOR of all measured bits)
        parity = 0
        for q, p in enumerate(pauli_string):
            if p != 'I':
                parity ^= (result >> q) & 1

        parity_sum += 1 if parity == 0 else -1

        # Reset for next measurement
        state.reset()
        apply_ansatz(state, current_params)

    return parity_sum / shots

def compute_energy(state, hamiltonian):
    """Compute total energy from Hamiltonian."""
    energy = 0.0

    for pauli_string, coefficient in hamiltonian.items():
        expectation = measure_pauli_string(state, pauli_string)
        energy += coefficient * expectation

    return energy
```

### More Efficient Energy Computation

For simulation, we can compute exact expectation values:

```python
def compute_energy_exact(state, hamiltonian):
    """
    Compute energy using exact expectation values.
    More efficient for simulation.
    """
    energy = 0.0

    for pauli_string, coefficient in hamiltonian.items():
        # Compute ⟨ψ|P|ψ⟩ for Pauli string P
        exp_val = state.expectation_pauli(pauli_string)
        energy += coefficient * exp_val

    return energy
```

## Step 4: Classical Optimization

```python
from scipy.optimize import minimize

def vqe_cost(params_flat, hamiltonian, num_layers):
    """Cost function for VQE optimization."""
    params = params_flat.reshape(num_layers, 2, 3)

    state = QuantumState(2)
    apply_ansatz(state, params, num_layers)

    energy = compute_energy_exact(state, hamiltonian)
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

print(f"Optimized energy: {result.fun:.6f} Hartree")
print(f"Exact H2 energy:  -1.137 Hartree")
print(f"Error: {abs(result.fun - (-1.137)):.6f} Hartree")
```

## Using the Built-in VQE

Moonlab provides a convenient VQE class:

```python
from moonlab.algorithms import VQE

# Create VQE solver
vqe = VQE(num_qubits=4, num_layers=2, optimizer_type=VQE.OPTIMIZER_ADAM)

# Solve for H2 at equilibrium distance
result = vqe.solve_h2(bond_distance=0.74)

print(f"Ground state energy: {result['energy']:.6f} Hartree")
print(f"Energy in kcal/mol: {result['energy_kcal_mol']:.2f}")
print(f"Converged: {result['converged']}")
print(f"Iterations: {result['num_iterations']}")
```

## Step 5: Potential Energy Surface

Compute energy vs. bond distance:

```python
import matplotlib.pyplot as plt

def h2_hamiltonian(distance):
    """Generate H2 Hamiltonian for given bond distance."""
    # Simplified: coefficients vary with distance
    # In reality, you'd use a quantum chemistry package
    d = distance

    return {
        'II': -0.8105 + 0.1 * (d - 0.74),
        'IZ': 0.1721 - 0.05 * (d - 0.74),
        'ZI': -0.2257 + 0.03 * (d - 0.74),
        'ZZ': 0.1689,
        'XX': 0.0454,
        'YY': 0.0454,
    }

# Compute potential energy curve
distances = np.linspace(0.3, 3.0, 20)
energies = []

vqe = VQE(num_qubits=4, num_layers=2)

for d in distances:
    result = vqe.solve_h2(bond_distance=d)
    energies.append(result['energy'])
    print(f"d = {d:.2f} Å: E = {result['energy']:.6f} Hartree")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(distances, energies, 'b-o', label='VQE')
plt.axhline(y=-1.137, color='r', linestyle='--', label='Exact (equilibrium)')
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

For more accurate results, use the UCCSD (Unitary Coupled Cluster Singles Doubles) ansatz:

```python
def uccsd_ansatz(state, params):
    """
    UCCSD ansatz for 4-qubit H2 simulation.
    """
    # Reference state: |0011⟩ (2 electrons in lowest orbitals)
    state.x(0)
    state.x(1)

    # Single excitation terms
    theta_1 = params[0]
    state.rx(0, theta_1)
    state.cnot(0, 2)

    # Double excitation terms (simplified)
    theta_2 = params[1]
    state.crx(0, 2, theta_2)
    state.crx(1, 3, theta_2)

    return state
```

## Larger Molecules

For larger molecules (LiH, H₂O, etc.):

```python
# LiH simulation
vqe = VQE(num_qubits=10, num_layers=4)
result = vqe.solve_lih(bond_distance=1.6)

print(f"LiH ground state: {result['energy']:.6f} Hartree")

# H2O simulation (requires more qubits)
vqe_h2o = VQE(num_qubits=14, num_layers=6)
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

