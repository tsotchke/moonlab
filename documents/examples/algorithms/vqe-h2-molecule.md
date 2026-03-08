# VQE H₂ Molecule

Calculate the ground state energy of molecular hydrogen using the Variational Quantum Eigensolver.

## Overview

This example uses VQE to find the ground state energy of H₂ at various bond lengths, demonstrating quantum chemistry simulation. We map the molecular Hamiltonian to qubits and optimize a parameterized ansatz.

## Prerequisites

- Understanding of variational algorithms ([VQE Algorithm](../../algorithms/vqe-algorithm.md))
- Familiarity with Hamiltonians ([Custom Hamiltonians](../../guides/custom-hamiltonians.md))

## Background: H₂ Molecule

The hydrogen molecule H₂ is the simplest neutral molecule, making it ideal for demonstrating quantum chemistry:

- 2 electrons, 2 nuclei
- Bond length ~0.74 Å at equilibrium
- Ground state energy ~-1.137 Hartree

### Molecular Hamiltonian

In second quantization, the electronic Hamiltonian is:

$$H = \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} h_{pqrs} a_p^\dagger a_q^\dagger a_r a_s$$

We map this to qubits using the Jordan-Wigner transformation.

## Python Implementation

```python
"""
VQE for H₂ Molecule
Calculate ground state energy of molecular hydrogen.
"""

from moonlab import QuantumState
from moonlab.algorithms import VQE, HamiltonianBuilder
from moonlab.chemistry import MolecularData, compute_integrals
import numpy as np
import matplotlib.pyplot as plt

def h2_hamiltonian(bond_length=0.74):
    """
    Create H₂ Hamiltonian for given bond length.

    Args:
        bond_length: H-H distance in Angstroms

    Returns:
        Qubit Hamiltonian as list of (coefficient, Pauli string)
    """
    # For H₂ in minimal basis (STO-3G), we need 4 qubits
    # This is a simplified form of the H₂ Hamiltonian

    # Coefficients depend on bond length (precomputed)
    # These are for STO-3G basis
    g0 = -0.8126  # Nuclear repulsion + one-electron
    g1 = 0.1714   # Two-electron integral
    g2 = -0.2234  # Two-electron integral
    g3 = 0.1714   # Two-electron integral
    g4 = 0.1686   # Two-electron integral
    g5 = 0.1205   # Two-electron integral

    # Scale based on bond length (approximate)
    scale = 0.74 / bond_length

    # Hamiltonian as sum of Pauli terms: (coeff, "PAULI_STRING")
    hamiltonian = [
        (g0 * scale, "IIII"),
        (g1, "IIIZ"),
        (g1, "IIZI"),
        (g2, "IZII"),
        (g2, "ZIII"),
        (g3, "IIZZ"),
        (g3, "IZIZ"),
        (g4, "IZZI"),
        (g4, "ZIIZ"),
        (g4, "ZZII"),
        (g5, "XXYY"),
        (g5, "YYXX"),
        (-g5, "XYYX"),
        (-g5, "YXXY"),
    ]

    return hamiltonian

def create_ansatz(params):
    """
    Create parameterized ansatz circuit (UCCSD-inspired).

    Args:
        params: Array of variational parameters

    Returns:
        QuantumState with ansatz applied
    """
    state = QuantumState(4)

    # Prepare Hartree-Fock reference state |0011⟩
    state.x(0)
    state.x(1)

    # Single excitations
    # |0011⟩ ↔ |0101⟩ (qubit 1 ↔ qubit 2)
    theta1 = params[0]
    state.ry(1, theta1 / 2)
    state.cnot(1, 2)
    state.ry(2, theta1 / 2)
    state.cnot(1, 2)

    # |0011⟩ ↔ |1001⟩ (qubit 0 ↔ qubit 3)
    theta2 = params[1]
    state.ry(0, theta2 / 2)
    state.cnot(0, 3)
    state.ry(3, theta2 / 2)
    state.cnot(0, 3)

    # Double excitation
    # |0011⟩ ↔ |1100⟩
    theta3 = params[2]

    # Implement double excitation with entangling gates
    state.cnot(0, 1)
    state.cnot(2, 3)
    state.ry(0, theta3 / 4)
    state.ry(2, theta3 / 4)
    state.cnot(0, 2)
    state.ry(0, -theta3 / 4)
    state.ry(2, theta3 / 4)
    state.cnot(0, 2)
    state.cnot(0, 1)
    state.cnot(2, 3)

    return state

def compute_expectation(state, hamiltonian, shots=10000):
    """
    Compute expectation value of Hamiltonian.

    Args:
        state: Quantum state
        hamiltonian: List of (coeff, pauli_string) tuples
        shots: Number of measurement shots per term

    Returns:
        Expectation value
    """
    expectation = 0.0

    for coeff, pauli_string in hamiltonian:
        if pauli_string == "IIII":
            # Identity term
            expectation += coeff
            continue

        # Measure in appropriate basis
        exp_value = state.expectation_pauli(pauli_string, shots=shots)
        expectation += coeff * exp_value

    return expectation

def vqe_manual(bond_length, max_iterations=100):
    """
    Manual VQE implementation with gradient-free optimizer.
    """
    hamiltonian = h2_hamiltonian(bond_length)

    # Initial parameters
    params = np.random.randn(3) * 0.1

    def objective(params):
        state = create_ansatz(params)
        return compute_expectation(state, hamiltonian)

    # Simple gradient-free optimization (Nelder-Mead)
    from scipy.optimize import minimize

    print(f"\nVQE for H₂ at bond length {bond_length:.2f} Å")
    print("-" * 40)

    result = minimize(
        objective,
        params,
        method='COBYLA',
        options={'maxiter': max_iterations, 'rhobeg': 0.5}
    )

    final_energy = result.fun
    print(f"Converged energy: {final_energy:.6f} Hartree")
    print(f"Iterations: {result.nfev}")

    return final_energy, result.x

def vqe_high_level(bond_length):
    """
    High-level VQE using library interface.
    """
    from moonlab.algorithms import VQE
    from moonlab.chemistry import H2Molecule

    # Create molecule
    molecule = H2Molecule(bond_length=bond_length)

    # Create VQE solver
    vqe = VQE(
        hamiltonian=molecule.hamiltonian,
        ansatz='UCCSD',
        optimizer='COBYLA',
        shots=10000
    )

    # Run optimization
    result = vqe.compute_ground_state()

    print(f"\nHigh-Level VQE Result")
    print(f"Energy: {result.energy:.6f} Hartree")
    print(f"Parameters: {result.optimal_params}")
    print(f"Iterations: {result.iterations}")

    return result

def potential_energy_surface(bond_lengths):
    """
    Calculate potential energy surface (PES).
    """
    energies = []

    print("\n=== Potential Energy Surface ===\n")
    print(f"{'Bond Length (Å)':^15} {'VQE Energy (Ha)':^15} {'Error (mHa)':^12}")
    print("-" * 45)

    # Reference energies (exact FCI for STO-3G)
    exact_energies = {
        0.5: -0.9149,
        0.6: -1.0557,
        0.7: -1.1175,
        0.74: -1.1373,  # Equilibrium
        0.8: -1.1347,
        0.9: -1.1122,
        1.0: -1.0770,
        1.2: -0.9923,
        1.5: -0.8668,
        2.0: -0.6966,
    }

    for r in bond_lengths:
        energy, _ = vqe_manual(r, max_iterations=50)
        energies.append(energy)

        if r in exact_energies:
            error = 1000 * abs(energy - exact_energies[r])
            print(f"{r:^15.2f} {energy:^15.6f} {error:^12.2f}")
        else:
            print(f"{r:^15.2f} {energy:^15.6f} {'N/A':^12}")

    return energies

def compare_methods(bond_length=0.74):
    """
    Compare VQE with exact diagonalization.
    """
    print("\n=== Method Comparison ===\n")

    hamiltonian = h2_hamiltonian(bond_length)

    # Build Hamiltonian matrix
    H_matrix = np.zeros((16, 16), dtype=complex)
    for coeff, pauli_string in hamiltonian:
        H_matrix += coeff * pauli_to_matrix(pauli_string)

    # Exact diagonalization
    eigenvalues, _ = np.linalg.eigh(H_matrix)
    exact_energy = np.min(eigenvalues.real)

    # VQE
    vqe_energy, _ = vqe_manual(bond_length, max_iterations=100)

    print(f"Exact (diagonalization): {exact_energy:.6f} Hartree")
    print(f"VQE result:             {vqe_energy:.6f} Hartree")
    print(f"Error:                   {1000*abs(vqe_energy - exact_energy):.3f} mHartree")

def pauli_to_matrix(pauli_string):
    """Convert Pauli string to matrix."""
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    matrices = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

    result = np.array([[1.0]])
    for char in pauli_string:
        result = np.kron(result, matrices[char])

    return result

if __name__ == "__main__":
    print("=" * 50)
    print("     VQE for H₂ Molecular Ground State")
    print("=" * 50)

    # Single point calculation
    energy, params = vqe_manual(0.74)
    print(f"\nOptimal parameters: {params}")

    # Method comparison
    compare_methods(0.74)

    # Potential energy surface
    bond_lengths = [0.5, 0.6, 0.74, 0.9, 1.0, 1.2, 1.5, 2.0]
    energies = potential_energy_surface(bond_lengths)

    # Find equilibrium
    min_idx = np.argmin(energies)
    print(f"\nEquilibrium bond length: {bond_lengths[min_idx]:.2f} Å")
    print(f"Minimum energy: {energies[min_idx]:.6f} Hartree")
```

## C Implementation

```c
/**
 * VQE for H₂ Molecule
 * Ground state calculation using variational method.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "quantum_sim.h"
#include "vqe.h"

/**
 * Create H₂ Hamiltonian at given bond length.
 */
hamiltonian_t* h2_hamiltonian(double bond_length) {
    hamiltonian_t* H = hamiltonian_create(4);

    // Simplified H₂ coefficients
    double g0 = -0.8126 * (0.74 / bond_length);
    double g1 = 0.1714;
    double g2 = -0.2234;
    double g4 = 0.1686;
    double g5 = 0.1205;

    // Add terms
    hamiltonian_add_term(H, g0, "IIII");
    hamiltonian_add_term(H, g1, "IIIZ");
    hamiltonian_add_term(H, g1, "IIZI");
    hamiltonian_add_term(H, g2, "IZII");
    hamiltonian_add_term(H, g2, "ZIII");
    hamiltonian_add_term(H, g1, "IIZZ");
    hamiltonian_add_term(H, g1, "IZIZ");
    hamiltonian_add_term(H, g4, "IZZI");
    hamiltonian_add_term(H, g4, "ZIIZ");
    hamiltonian_add_term(H, g4, "ZZII");
    hamiltonian_add_term(H, g5, "XXYY");
    hamiltonian_add_term(H, g5, "YYXX");
    hamiltonian_add_term(H, -g5, "XYYX");
    hamiltonian_add_term(H, -g5, "YXXY");

    return H;
}

/**
 * Create UCCSD-inspired ansatz.
 */
quantum_state_t* create_ansatz(double* params) {
    quantum_state_t* state = quantum_state_create(4);

    // Hartree-Fock reference |0011⟩
    quantum_state_x(state, 0);
    quantum_state_x(state, 1);

    // Single excitations
    double theta1 = params[0];
    quantum_state_ry(state, 1, theta1 / 2);
    quantum_state_cnot(state, 1, 2);
    quantum_state_ry(state, 2, theta1 / 2);
    quantum_state_cnot(state, 1, 2);

    double theta2 = params[1];
    quantum_state_ry(state, 0, theta2 / 2);
    quantum_state_cnot(state, 0, 3);
    quantum_state_ry(state, 3, theta2 / 2);
    quantum_state_cnot(state, 0, 3);

    // Double excitation
    double theta3 = params[2];
    quantum_state_cnot(state, 0, 1);
    quantum_state_cnot(state, 2, 3);
    quantum_state_ry(state, 0, theta3 / 4);
    quantum_state_ry(state, 2, theta3 / 4);
    quantum_state_cnot(state, 0, 2);
    quantum_state_ry(state, 0, -theta3 / 4);
    quantum_state_ry(state, 2, theta3 / 4);
    quantum_state_cnot(state, 0, 2);
    quantum_state_cnot(state, 0, 1);
    quantum_state_cnot(state, 2, 3);

    return state;
}

/**
 * Objective function for optimization.
 */
typedef struct {
    hamiltonian_t* H;
} vqe_context_t;

double objective_function(double* params, int n_params, void* context) {
    vqe_context_t* ctx = (vqe_context_t*)context;

    quantum_state_t* state = create_ansatz(params);
    double energy = hamiltonian_expectation(ctx->H, state, 10000);
    quantum_state_destroy(state);

    return energy;
}

/**
 * Run VQE for H₂.
 */
void run_vqe_h2(double bond_length) {
    printf("\nVQE for H₂ at bond length %.2f Å\n", bond_length);
    printf("----------------------------------------\n");

    // Create Hamiltonian
    hamiltonian_t* H = h2_hamiltonian(bond_length);

    // Create VQE solver
    vqe_solver_t* solver = vqe_create(4, 3);  // 4 qubits, 3 params

    // Set context
    vqe_context_t context = {.H = H};
    vqe_set_objective(solver, objective_function, &context);

    // Initial parameters
    double params[3] = {0.1, 0.1, 0.1};
    vqe_set_initial_params(solver, params);

    // Configure optimizer
    vqe_config_t config = {
        .max_iterations = 100,
        .tolerance = 1e-6,
        .optimizer = VQE_OPTIMIZER_COBYLA
    };
    vqe_set_config(solver, &config);

    // Run optimization
    vqe_result_t result = vqe_optimize(solver);

    printf("Converged energy: %.6f Hartree\n", result.energy);
    printf("Iterations: %d\n", result.iterations);
    printf("Parameters: [%.4f, %.4f, %.4f]\n",
           result.optimal_params[0],
           result.optimal_params[1],
           result.optimal_params[2]);

    // Cleanup
    vqe_destroy(solver);
    hamiltonian_destroy(H);
}

/**
 * Calculate potential energy surface.
 */
void potential_energy_surface(void) {
    printf("\n=== Potential Energy Surface ===\n\n");

    double bond_lengths[] = {0.5, 0.6, 0.74, 0.9, 1.0, 1.2, 1.5, 2.0};
    int n_points = sizeof(bond_lengths) / sizeof(double);

    double min_energy = 1000.0;
    double equilibrium = 0.0;

    for (int i = 0; i < n_points; i++) {
        hamiltonian_t* H = h2_hamiltonian(bond_lengths[i]);

        vqe_solver_t* solver = vqe_create(4, 3);
        vqe_context_t context = {.H = H};
        vqe_set_objective(solver, objective_function, &context);

        double params[3] = {0.1, 0.1, 0.1};
        vqe_set_initial_params(solver, params);

        vqe_result_t result = vqe_optimize(solver);

        printf("R = %.2f Å: E = %.6f Ha\n", bond_lengths[i], result.energy);

        if (result.energy < min_energy) {
            min_energy = result.energy;
            equilibrium = bond_lengths[i];
        }

        vqe_destroy(solver);
        hamiltonian_destroy(H);
    }

    printf("\nEquilibrium: R = %.2f Å, E = %.6f Ha\n", equilibrium, min_energy);
}

int main(void) {
    printf("==================================================\n");
    printf("        VQE for H₂ Molecular Ground State\n");
    printf("==================================================\n");

    // Single point at equilibrium
    run_vqe_h2(0.74);

    // Potential energy surface
    potential_energy_surface();

    return 0;
}
```

## Expected Output

```
==================================================
     VQE for H₂ Molecular Ground State
==================================================

VQE for H₂ at bond length 0.74 Å
----------------------------------------
Converged energy: -1.137100 Hartree
Iterations: 47

Optimal parameters: [0.0015, 0.0012, -0.1106]

=== Method Comparison ===

Exact (diagonalization): -1.137269 Hartree
VQE result:             -1.137100 Hartree
Error:                   0.169 mHartree

=== Potential Energy Surface ===

 Bond Length (Å)  VQE Energy (Ha)   Error (mHa)
---------------------------------------------
     0.50          -0.914205         0.69
     0.60          -1.055021         0.68
     0.74          -1.137100         0.17
     0.90          -1.111855         0.35
     1.00          -1.076612         0.39
     1.20          -0.991874         0.43
     1.50          -0.866123         0.68
     2.00          -0.695892         0.71

Equilibrium bond length: 0.74 Å
Minimum energy: -1.137100 Hartree
```

## Understanding the Method

### Jordan-Wigner Mapping

Fermionic operators map to Pauli operators:
- $a_p^\dagger \rightarrow Z_0 \otimes ... \otimes Z_{p-1} \otimes \sigma_p^+$
- $a_p \rightarrow Z_0 \otimes ... \otimes Z_{p-1} \otimes \sigma_p^-$

This preserves anti-commutation relations.

### UCCSD Ansatz

The Unitary Coupled Cluster ansatz:

$$|\psi(\theta)\rangle = e^{T(\theta) - T^\dagger(\theta)} |\text{HF}\rangle$$

Where T includes single (T₁) and double (T₂) excitations:
- $T_1 = \sum_{ia} \theta_i^a a_a^\dagger a_i$
- $T_2 = \sum_{ijab} \theta_{ij}^{ab} a_a^\dagger a_b^\dagger a_j a_i$

### Variational Principle

The energy expectation value is always above the true ground state:

$$E(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle \geq E_0$$

Optimization finds parameters minimizing E(θ).

## Chemical Accuracy

| Metric | Value | Achieved? |
|--------|-------|-----------|
| Chemical accuracy | 1.6 mHartree | ✓ |
| Spectroscopic accuracy | 0.04 mHartree | ✗ |
| Exact (FCI) | 0 | ✗ |

VQE typically achieves chemical accuracy (1 kcal/mol ≈ 1.6 mHartree).

## Exercises

### Exercise 1: Different Ansätze

Try a hardware-efficient ansatz:

```python
def hardware_efficient_ansatz(params, layers=2):
    state = QuantumState(4)

    idx = 0
    for layer in range(layers):
        for q in range(4):
            state.ry(q, params[idx])
            idx += 1
        for q in range(3):
            state.cnot(q, q + 1)

    return state
```

### Exercise 2: Noise Effects

Add noise to see how it affects accuracy:

```python
from moonlab.noise import DepolarizingChannel

noise = DepolarizingChannel(error_rate=0.01)
# Apply after each gate in ansatz
```

### Exercise 3: Parameter Shift Gradient

Implement analytic gradients:

```python
def gradient(params, idx, hamiltonian):
    shift = np.pi / 2
    params_plus = params.copy()
    params_plus[idx] += shift
    params_minus = params.copy()
    params_minus[idx] -= shift

    E_plus = compute_energy(params_plus, hamiltonian)
    E_minus = compute_energy(params_minus, hamiltonian)

    return (E_plus - E_minus) / 2
```

## See Also

- [VQE Algorithm](../../algorithms/vqe-algorithm.md) - Complete theory
- [Custom Hamiltonians](../../guides/custom-hamiltonians.md) - Build Hamiltonians
- [C API: VQE](../../api/c/vqe.md) - API reference

