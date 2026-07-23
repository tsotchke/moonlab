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

The `moonlab.chemistry` engine builds the H₂ qubit Hamiltonian from genuine STO-3G
Gaussian integrals (Jordan-Wigner mapped to two qubits), and `moonlab.algorithms.VQE`
runs the variational loop. No coefficients are hardcoded.

```python
"""
VQE for H2 Molecule
Ground-state energy of molecular hydrogen across the potential energy surface.
"""

import numpy as np
from moonlab.algorithms import VQE
from moonlab.chemistry import Hamiltonian


def h2_pauli_terms(bond_length=0.74):
    """First-principles STO-3G H2 Pauli terms at a bond length.

    Returns the two-qubit Hamiltonian as (coefficient, pauli_string) pairs,
    computed by the C electronic-structure engine -- not hardcoded.
    """
    return Hamiltonian.h2_sto3g(bond_distance=bond_length).terms
    # e.g. [(c_II, 'II'), (c_IZ, 'IZ'), (c_ZI, 'ZI'), (c_ZZ, 'ZZ'), (c_XX, 'XX')]


def vqe_h2(bond_length=0.74):
    """Run VQE on H2 at one bond length and compare to the exact reference.

    H2 maps to a 2-qubit Hamiltonian. The hardware-efficient ansatz with the
    quantum natural gradient optimizer reaches the exact ground state.
    """
    exact = Hamiltonian.h2_sto3g(bond_distance=bond_length).exact_ground_state()

    vqe = VQE(num_qubits=2, num_layers=2, optimizer='natural_gradient')
    result = vqe.solve_h2(bond_distance=bond_length)  # builds the Hamiltonian internally

    energy = result['energy']
    print(f"R = {bond_length:.2f} A:  VQE = {energy:.6f} Ha, "
          f"exact = {exact:.6f} Ha, error = {1e3 * abs(energy - exact):.3f} mHa")
    return energy, exact


def potential_energy_surface(bond_lengths):
    """Compute the H2 PES and compare each point to the exact diagonalization."""
    print(f"\n{'Bond Length (A)':^16} {'VQE (Ha)':^12} {'Exact (Ha)':^12} {'Error (mHa)':^12}")
    print("-" * 54)

    vqe = VQE(num_qubits=2, num_layers=2, optimizer='natural_gradient')
    energies = []
    for r in bond_lengths:
        energy = vqe.solve_h2(bond_distance=r)['energy']
        exact = Hamiltonian.h2_sto3g(bond_distance=r).exact_ground_state()
        energies.append(energy)
        print(f"{r:^16.2f} {energy:^12.6f} {exact:^12.6f} {1e3 * abs(energy - exact):^12.3f}")
    return energies


if __name__ == "__main__":
    print("VQE for H2 Molecular Ground State")

    # Inspect the (real) Hamiltonian terms at equilibrium.
    print("Hamiltonian terms at R = 0.74 A:")
    for coeff, pauli in h2_pauli_terms(0.74):
        print(f"  {coeff:+.4f}  {pauli}")

    # Potential energy surface.
    bond_lengths = [0.5, 0.6, 0.74, 0.9, 1.0, 1.2, 1.5, 2.0]
    energies = potential_energy_surface(bond_lengths)

    # Find equilibrium.
    i = int(np.argmin(energies))
    print(f"\nEquilibrium near R = {bond_lengths[i]:.2f} A, E = {energies[i]:.6f} Ha")
```

The VQE solver prints a per-run optimization banner to stdout; the summary lines above
are what the script itself reports.

## C Implementation

```c
/**
 * VQE for H2 Molecule
 * Ground-state calculation across the potential energy surface.
 * Uses the pre-built STO-3G H2 Hamiltonian (vqe_create_h2_hamiltonian) --
 * no hardcoded coefficients.
 */

#include <stdio.h>
#include "algorithms/vqe.h"
#include "utils/quantum_entropy.h"

/* Run VQE on the STO-3G H2 Hamiltonian at one bond length; return the energy. */
static double run_vqe_h2(double bond_length, quantum_entropy_ctx_t* entropy) {
    pauli_hamiltonian_t* H = vqe_create_h2_hamiltonian(bond_length);   // 2 qubits
    vqe_ansatz_t* ansatz = vqe_create_hardware_efficient_ansatz(2, 2);
    vqe_optimizer_t* opt = vqe_optimizer_create(VQE_OPTIMIZER_QNG);    // natural gradient
    vqe_solver_t* solver = vqe_solver_create(H, ansatz, opt, entropy);

    vqe_result_t result = vqe_solve(solver);
    double energy = result.ground_state_energy;
    double exact = vqe_exact_ground_state_energy(H);

    printf("R = %.2f A: VQE = %.6f Ha, exact = %.6f Ha, error = %.3f mHa\n",
           bond_length, energy, exact, 1e3 * (energy - exact));

    vqe_result_free(&result);
    vqe_solver_free(solver);
    vqe_optimizer_free(opt);
    vqe_ansatz_free(ansatz);
    pauli_hamiltonian_free(H);
    return energy;
}

int main(void) {
    printf("VQE for H2 Molecular Ground State\n\n");
    quantum_entropy_ctx_t* entropy = quantum_entropy_ctx_create_hw();

    double bond_lengths[] = {0.5, 0.6, 0.74, 0.9, 1.0, 1.2, 1.5, 2.0};
    int n_points = (int)(sizeof(bond_lengths) / sizeof(double));

    double min_energy = 1e9, equilibrium = 0.0;
    for (int i = 0; i < n_points; i++) {
        double e = run_vqe_h2(bond_lengths[i], entropy);
        if (e < min_energy) { min_energy = e; equilibrium = bond_lengths[i]; }
    }

    printf("\nEquilibrium: R = %.2f A, E = %.6f Ha\n", equilibrium, min_energy);

    quantum_entropy_ctx_destroy(entropy);
    return 0;
}
```

## Expected Output

```
VQE for H2 Molecular Ground State
Hamiltonian terms at R = 0.74 A:
  -1.0521  II
  +0.3988  IZ
  -0.3988  ZI
  -0.0113  ZZ
  +0.1809  XX

Bond Length (A)    VQE (Ha)    Exact (Ha)  Error (mHa)
------------------------------------------------------
      0.50        -1.059555    -1.059555      0.000
      0.60        -1.120967    -1.120967      0.000
      0.74        -1.142183    -1.142183      0.000
      0.90        -1.125528    -1.125528      0.000
      1.00        -1.106077    -1.106077      0.000
      1.20        -1.061384    -1.061384      0.000
      1.50        -1.001820    -1.001820      0.000
      2.00        -0.949883    -0.949883      0.000

Equilibrium near R = 0.74 A, E = -1.142183 Ha
```

(The `natural_gradient` optimizer drives the hardware-efficient ansatz to the exact
STO-3G ground state at every bond length, so the VQE and exact columns coincide. The VQE
solver also prints a per-run optimization banner, elided here.)

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
| Chemical accuracy | 1.6 mHartree | Yes |
| Spectroscopic accuracy | 0.04 mHartree | Yes |
| Exact (STO-3G FCI) | 0 | Yes (sub-µHartree) |

On this 2-qubit H₂ Hamiltonian the hardware-efficient ansatz is expressive enough that the
`natural_gradient` optimizer reaches the exact STO-3G ground state to numerical precision.
For larger molecules the ansatz expressibility, not the optimizer, sets the accuracy gap;
chemical accuracy (1 kcal/mol ≈ 1.6 mHartree) is the usual target.

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

Attach a NISQ noise model to the VQE solver and watch the accuracy degrade. Noise-model
injection lives on the C VQE surface (`vqe_create_depolarizing_noise` /
`vqe_create_nisq_noise` + `vqe_solver_set_noise`, see `algorithms/vqe.h`):

```c
noise_model_t* noise = vqe_create_depolarizing_noise(0.001, 0.01, 0.02);
vqe_solver_set_noise(solver, noise);  // solver takes ownership
vqe_result_t result = vqe_solve(solver);
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

