# VQE API

Complete reference for the Variational Quantum Eigensolver (VQE) in the C library.

**Header**: `src/algorithms/vqe.h`

## Overview

VQE is a hybrid quantum-classical algorithm for finding ground state energies of molecular systems. It is the primary algorithm for quantum chemistry applications, enabling:

- **Drug discovery**: Molecular binding energies
- **Materials science**: Battery chemistry, catalysts
- **Chemical reactions**: Activation energies

The 32-qubit capability enables simulation of:
- H₂ (2 qubits), LiH (4 qubits), H₂O (8 qubits)
- NH₃ (10 qubits), CH₄ (12 qubits), C₂H₄ (16 qubits)
- Benzene C₆H₆ (24 qubits), small proteins (28-30 qubits)

## Algorithm

The VQE algorithm proceeds as follows:

1. **Prepare trial state** with variational circuit: $|\psi(\theta)\rangle$
2. **Measure energy expectation**: $E(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle$
3. **Classical optimization**: minimize $E(\theta)$ over parameters $\theta$
4. **Iterate** until convergence

The variational principle guarantees $E(\theta) \geq E_0$ where $E_0$ is the true ground state energy.

## Pauli Hamiltonian

### pauli_term_t

A single term in the Pauli Hamiltonian.

```c
typedef struct {
    double coefficient;       // Coefficient for this term
    char *pauli_string;      // String of Pauli operators (X,Y,Z,I)
    size_t num_qubits;       // Length of Pauli string
} pauli_term_t;
```

**Example**: The term $0.5 Z_0 Z_1 X_2 I_3$ is represented as:
- `coefficient = 0.5`
- `pauli_string = "ZZXI"`
- `num_qubits = 4`

### pauli_hamiltonian_t

Pauli Hamiltonian as a sum of Pauli string terms.

```c
typedef struct {
    size_t num_qubits;           // Number of qubits needed
    size_t num_terms;            // Number of Pauli terms
    pauli_term_t *terms;         // Array of Pauli terms
    double nuclear_repulsion;    // Nuclear-nuclear repulsion energy
    char *molecule_name;         // Name (e.g., "H2", "LiH")
    double bond_distance;        // Internuclear distance (Angstroms)
} pauli_hamiltonian_t;
```

**Mathematical Form**:
$$H = \sum_i c_i P_i$$

where $P_i$ are tensor products of Pauli operators.

**Example for H₂ molecule**:
$$H = -1.05\, II + 0.40\, ZI + 0.40\, IZ - 0.22\, ZZ + 0.18\, XX$$

### pauli_hamiltonian_create

Create a Pauli Hamiltonian.

```c
pauli_hamiltonian_t* pauli_hamiltonian_create(
    size_t num_qubits,
    size_t num_terms
);
```

**Parameters**:
- `num_qubits`: Number of qubits in the system
- `num_terms`: Number of Pauli terms to allocate

**Returns**: Pointer to initialized Hamiltonian, or NULL on error

### pauli_hamiltonian_free

Free a Pauli Hamiltonian.

```c
void pauli_hamiltonian_free(pauli_hamiltonian_t *hamiltonian);
```

### pauli_hamiltonian_add_term

Add a Pauli term to the Hamiltonian.

```c
int pauli_hamiltonian_add_term(
    pauli_hamiltonian_t *hamiltonian,
    double coefficient,
    const char *pauli_string,
    size_t term_index
);
```

**Parameters**:
- `hamiltonian`: Pauli Hamiltonian
- `coefficient`: Term coefficient
- `pauli_string`: Pauli string (e.g., "ZZXI")
- `term_index`: Index where to add term

**Returns**: 0 on success, -1 on error

## Pre-built Molecular Hamiltonians

### vqe_create_h2_hamiltonian

Create H₂ (Hydrogen) molecule Hamiltonian.

```c
pauli_hamiltonian_t* vqe_create_h2_hamiltonian(double bond_distance);
```

**Parameters**:
- `bond_distance`: Internuclear distance in Angstroms (typical: 0.5-2.0)

**Returns**: 2-qubit Hamiltonian using STO-3G basis, Jordan-Wigner transformation

**Example**:
```c
// H₂ at equilibrium bond distance
pauli_hamiltonian_t *h2 = vqe_create_h2_hamiltonian(0.735);
printf("H₂ Hamiltonian: %zu qubits, %zu terms\n",
       h2->num_qubits, h2->num_terms);
```

### vqe_create_lih_hamiltonian

Create LiH (Lithium Hydride) molecule Hamiltonian.

```c
pauli_hamiltonian_t* vqe_create_lih_hamiltonian(double bond_distance);
```

**Parameters**:
- `bond_distance`: Internuclear distance in Angstroms

**Returns**: 4-qubit Hamiltonian

**Use Case**: Important for battery research (lithium-based materials)

### vqe_create_h2o_hamiltonian

Create H₂O (Water) molecule Hamiltonian.

```c
pauli_hamiltonian_t* vqe_create_h2o_hamiltonian(void);
```

**Returns**: 8-qubit Hamiltonian with fixed geometry (O-H bond = 0.958 Å, H-O-H angle = 104.5°)

## Variational Ansatz

### vqe_ansatz_type_t

Ansatz types for trial state preparation.

```c
typedef enum {
    VQE_ANSATZ_HARDWARE_EFFICIENT,  // Hardware-efficient ansatz
    VQE_ANSATZ_UCCSD,               // Unitary Coupled Cluster (chemistry)
    VQE_ANSATZ_CUSTOM               // User-defined ansatz
} vqe_ansatz_type_t;
```

### vqe_ansatz_t

Variational circuit ansatz.

```c
typedef struct {
    vqe_ansatz_type_t type;      // Ansatz type
    size_t num_qubits;           // Number of qubits
    size_t num_layers;           // Circuit depth
    size_t num_parameters;       // Total parameters to optimize
    double *parameters;          // Current parameter values
    void *circuit_data;          // Opaque circuit description
} vqe_ansatz_t;
```

### vqe_create_hardware_efficient_ansatz

Create hardware-efficient ansatz.

```c
vqe_ansatz_t* vqe_create_hardware_efficient_ansatz(
    size_t num_qubits,
    size_t num_layers
);
```

**Parameters**:
- `num_qubits`: Number of qubits
- `num_layers`: Circuit depth

**Returns**: Ansatz with alternating rotation and entanglement layers

**Circuit Structure**:
```
Layer k:
  RY(θ₁) RY(θ₂) RY(θ₃) ...
  RZ(θ₄) RZ(θ₅) RZ(θ₆) ...
  CNOT(0,1) CNOT(1,2) CNOT(2,3) ...
```

**Parameter Count**: $3 \times n \times L$ where $n$ is qubits, $L$ is layers

### vqe_create_uccsd_ansatz

Create UCCSD ansatz (chemistry-inspired).

```c
vqe_ansatz_t* vqe_create_uccsd_ansatz(
    size_t num_qubits,
    size_t num_electrons
);
```

**Parameters**:
- `num_qubits`: Number of qubits (spin orbitals)
- `num_electrons`: Number of electrons in molecule

**Returns**: Unitary Coupled Cluster Singles and Doubles ansatz

**Mathematical Form**:
$$|\psi(\theta)\rangle = e^{T(\theta) - T^\dagger(\theta)} |HF\rangle$$

where $T = T_1 + T_2$ includes single and double excitations.

**Advantage**: Provides chemical accuracy for molecular systems

### vqe_ansatz_free

Free ansatz structure.

```c
void vqe_ansatz_free(vqe_ansatz_t *ansatz);
```

### vqe_apply_ansatz

Apply ansatz circuit to quantum state.

```c
qs_error_t vqe_apply_ansatz(
    quantum_state_t *state,
    const vqe_ansatz_t *ansatz
);
```

**Parameters**:
- `state`: Quantum state to prepare
- `ansatz`: Variational ansatz with current parameters

**Returns**: `QS_SUCCESS` or error code

**Action**: Prepares trial state $|\psi(\theta)\rangle$ using current parameters

### vqe_apply_ansatz_noisy

Apply ansatz circuit with noise simulation.

```c
qs_error_t vqe_apply_ansatz_noisy(
    quantum_state_t *state,
    const vqe_ansatz_t *ansatz,
    const noise_model_t *noise,
    quantum_entropy_ctx_t *entropy
);
```

**Parameters**:
- `state`: Quantum state
- `ansatz`: Variational ansatz
- `noise`: Noise model (NULL for ideal simulation)
- `entropy`: Random number source for noise

**Returns**: `QS_SUCCESS` or error code

## Classical Optimizers

### vqe_optimizer_type_t

Optimizer types.

```c
typedef enum {
    VQE_OPTIMIZER_COBYLA,          // Constrained optimization
    VQE_OPTIMIZER_LBFGS,           // Limited-memory BFGS
    VQE_OPTIMIZER_ADAM,            // Adaptive moment estimation
    VQE_OPTIMIZER_GRADIENT_DESCENT // Simple gradient descent
} vqe_optimizer_type_t;
```

| Optimizer | Gradient-Free | Memory | Best For |
|-----------|--------------|--------|----------|
| COBYLA | Yes | O(n) | Noisy objectives |
| L-BFGS | No | O(nm) | Smooth objectives |
| ADAM | No | O(n) | Large parameter space |
| Gradient Descent | No | O(n) | Simple cases |

### vqe_optimizer_t

Classical optimizer configuration.

```c
typedef struct {
    vqe_optimizer_type_t type;   // Optimizer type
    size_t max_iterations;       // Maximum iterations
    double tolerance;            // Convergence tolerance
    double learning_rate;        // Learning rate (for gradient methods)
    int verbose;                 // Print progress
} vqe_optimizer_t;
```

### vqe_optimizer_create

Create optimizer.

```c
vqe_optimizer_t* vqe_optimizer_create(vqe_optimizer_type_t type);
```

**Parameters**:
- `type`: Optimizer type

**Returns**: Optimizer with default parameters

### vqe_optimizer_free

Free optimizer.

```c
void vqe_optimizer_free(vqe_optimizer_t *optimizer);
```

## VQE Solver

### vqe_solver_t

VQE solver context.

```c
typedef struct {
    pauli_hamiltonian_t *hamiltonian;  // Pauli Hamiltonian
    vqe_ansatz_t *ansatz;              // Variational ansatz
    vqe_optimizer_t *optimizer;        // Classical optimizer
    quantum_entropy_ctx_t *entropy;    // Entropy for measurements
    noise_model_t *noise_model;        // Hardware noise model (NULL = ideal)
    size_t iteration;                  // Current iteration
    double *energy_history;            // Energy at each iteration
    size_t max_history;                // History buffer size
    size_t total_measurements;         // Total measurements performed
    double total_time;                 // Total optimization time
} vqe_solver_t;
```

### vqe_result_t

VQE result.

```c
typedef struct {
    double ground_state_energy;       // Computed ground state energy (Hartree)
    double *optimal_parameters;       // Optimal variational parameters
    size_t num_parameters;           // Number of parameters
    size_t iterations;               // Iterations to convergence
    double convergence_tolerance;    // Final gradient norm
    int converged;                   // 1 if converged, 0 if max iterations
    double fci_energy;              // Full CI energy (if available)
    double hf_energy;               // Hartree-Fock energy (if available)
    double chemical_accuracy;       // Error in kcal/mol
} vqe_result_t;
```

**Note**: Chemical accuracy is 1 kcal/mol = 0.0016 Hartree

### vqe_solver_create

Create VQE solver.

```c
vqe_solver_t* vqe_solver_create(
    pauli_hamiltonian_t *hamiltonian,
    vqe_ansatz_t *ansatz,
    vqe_optimizer_t *optimizer,
    quantum_entropy_ctx_t *entropy
);
```

**Parameters**:
- `hamiltonian`: Molecular Hamiltonian
- `ansatz`: Variational ansatz
- `optimizer`: Classical optimizer
- `entropy`: Entropy context for measurements

**Returns**: VQE solver context

### vqe_solver_free

Free VQE solver.

```c
void vqe_solver_free(vqe_solver_t *solver);
```

### vqe_solver_set_noise

Set noise model for VQE solver.

```c
void vqe_solver_set_noise(vqe_solver_t *solver, noise_model_t *noise_model);
```

**Parameters**:
- `solver`: VQE solver context
- `noise_model`: Noise model (solver takes ownership)

### vqe_compute_energy

Compute energy expectation for given parameters.

```c
double vqe_compute_energy(
    vqe_solver_t *solver,
    const double *parameters
);
```

**Parameters**:
- `solver`: VQE solver context
- `parameters`: Current variational parameters

**Returns**: Energy expectation value $E(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle$

### vqe_solve

Run VQE optimization to find ground state.

```c
vqe_result_t vqe_solve(vqe_solver_t *solver);
```

**Parameters**:
- `solver`: VQE solver context

**Returns**: VQE result with ground state energy and optimal parameters

**Algorithm**:
1. Initialize variational parameters (random or Hartree-Fock)
2. Loop:
   - Prepare state with current parameters
   - Measure energy expectation
   - Classical optimization step
   - Check convergence
3. Return optimal energy and parameters

### vqe_compute_gradient

Compute gradient of energy with respect to parameters.

```c
int vqe_compute_gradient(
    vqe_solver_t *solver,
    const double *parameters,
    double *gradient
);
```

**Parameters**:
- `solver`: VQE solver context
- `parameters`: Current parameters
- `gradient`: Output gradient vector

**Returns**: 0 on success, -1 on error

**Method**: Uses parameter shift rule:
$$\frac{\partial E}{\partial \theta_i} = \frac{E(\theta + \frac{\pi}{2}e_i) - E(\theta - \frac{\pi}{2}e_i)}{2}$$

## Noise Models for VQE

### vqe_create_depolarizing_noise

Create depolarizing noise model.

```c
noise_model_t* vqe_create_depolarizing_noise(
    double single_qubit_error,
    double two_qubit_error,
    double readout_error
);
```

**Parameters**:
- `single_qubit_error`: Single-qubit gate error probability
- `two_qubit_error`: Two-qubit gate error probability (default: 10× single)
- `readout_error`: Measurement error probability

**Returns**: Configured noise model

### vqe_create_nisq_noise

Create realistic NISQ noise model.

```c
noise_model_t* vqe_create_nisq_noise(
    double t1_us,
    double t2_us,
    double gate_error,
    double readout_error
);
```

**Parameters**:
- `t1_us`: T1 relaxation time in microseconds
- `t2_us`: T2 dephasing time in microseconds
- `gate_error`: Single-qubit gate error rate
- `readout_error`: Readout error rate

**Returns**: Noise model based on superconducting qubit parameters

## Utility Functions

### vqe_measure_pauli_expectation

Measure expectation value of a Pauli term.

```c
double vqe_measure_pauli_expectation(
    quantum_state_t *state,
    const pauli_term_t *pauli_term,
    quantum_entropy_ctx_t *entropy,
    size_t num_samples
);
```

**Parameters**:
- `state`: Quantum state
- `pauli_term`: Pauli term to measure
- `entropy`: Entropy for measurements
- `num_samples`: Number of measurement samples

**Returns**: Expectation value $\langle P \rangle = \langle\psi|P|\psi\rangle$

### vqe_apply_pauli_rotation

Apply Pauli rotation to state.

```c
qs_error_t vqe_apply_pauli_rotation(
    quantum_state_t *state,
    const char *pauli_string,
    double angle
);
```

**Parameters**:
- `state`: Quantum state
- `pauli_string`: Pauli operator string
- `angle`: Rotation angle

**Returns**: `QS_SUCCESS` or error

**Action**: Applies $\exp(-i\theta P)$ where $P$ is the Pauli string

### vqe_hartree_to_kcalmol

Convert energy units.

```c
double vqe_hartree_to_kcalmol(double energy);
```

**Conversion**: 1 Hartree = 627.5 kcal/mol

### vqe_print_result

Print VQE result.

```c
void vqe_print_result(const vqe_result_t *result);
```

### vqe_print_hamiltonian

Print Hamiltonian.

```c
void vqe_print_hamiltonian(const pauli_hamiltonian_t *hamiltonian);
```

## Complete Example

```c
#include "src/algorithms/vqe.h"
#include "src/utils/quantum_entropy.h"

int main(void) {
    // Initialize entropy
    quantum_entropy_ctx_t entropy;
    quantum_entropy_init(&entropy, NULL, NULL);

    // Create H₂ Hamiltonian at equilibrium
    pauli_hamiltonian_t *h2 = vqe_create_h2_hamiltonian(0.735);
    vqe_print_hamiltonian(h2);

    // Create hardware-efficient ansatz
    vqe_ansatz_t *ansatz = vqe_create_hardware_efficient_ansatz(
        h2->num_qubits,
        2  // 2 layers
    );

    // Create optimizer
    vqe_optimizer_t *optimizer = vqe_optimizer_create(VQE_OPTIMIZER_COBYLA);
    optimizer->max_iterations = 200;
    optimizer->tolerance = 1e-6;

    // Create solver
    vqe_solver_t *solver = vqe_solver_create(h2, ansatz, optimizer, &entropy);

    // Optional: add realistic noise
    noise_model_t *noise = vqe_create_nisq_noise(100.0, 50.0, 0.001, 0.02);
    vqe_solver_set_noise(solver, noise);

    // Run VQE
    vqe_result_t result = vqe_solve(solver);
    vqe_print_result(&result);

    printf("\n=== Results ===\n");
    printf("Ground state energy: %.6f Hartree\n", result.ground_state_energy);
    printf("Chemical accuracy: %.2f kcal/mol\n", result.chemical_accuracy);
    printf("Converged: %s\n", result.converged ? "Yes" : "No");

    // Cleanup
    free(result.optimal_parameters);
    vqe_solver_free(solver);

    return 0;
}
```

## Energy Units and Chemical Accuracy

| Unit | Conversion |
|------|------------|
| 1 Hartree | 627.5 kcal/mol |
| 1 Hartree | 27.2 eV |
| Chemical accuracy | 1 kcal/mol = 0.0016 Ha |

Chemical accuracy (1 kcal/mol) is the threshold for useful chemical predictions.

## See Also

- [QAOA API](qaoa.md) - Combinatorial optimization
- [Noise API](noise.md) - Noise models
- [Algorithms: VQE](../../algorithms/vqe-algorithm.md) - Full theory
- [Tutorial: VQE Molecular Simulation](../../tutorials/06-vqe-molecular-simulation.md)
