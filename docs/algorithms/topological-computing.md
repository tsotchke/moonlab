# Topological Quantum Computing

Fault-tolerant quantum computation using topologically protected degrees of freedom.

## Overview

Topological quantum computing (TQC) encodes quantum information in non-local, topologically protected states that are inherently robust against local perturbations. Instead of using individual qubit states, TQC uses exotic quasiparticles called **anyons** whose quantum properties depend only on the topology of their worldlines.

Moonlab provides comprehensive TQC simulation including:
- **Anyon models**: Fibonacci, Ising, and SU(2)_k
- **Braiding operations**: F-matrices, R-matrices, fusion trees
- **Error-correcting codes**: Surface codes and toric codes
- **Topological invariants**: Entanglement entropy, modular matrices

## Theoretical Background

### Why Topology?

In standard quantum computing, qubits are vulnerable to decoherence from environmental noise. A single stray photon can flip a qubit's state. Topological protection arises because:

1. **Non-local encoding**: Information is stored in global properties of the system
2. **Energy gap**: Local perturbations cannot change the topological sector
3. **Discrete quantum numbers**: Topological charges can only take discrete values

The key insight is that braiding operations—physically exchanging anyons—produce quantum gates that depend only on the topology of the exchange, not on the precise trajectory.

### Anyons

Anyons are quasiparticles that exist only in 2D systems and exhibit exchange statistics that are neither bosonic nor fermionic. When two anyons are exchanged, the wavefunction acquires a phase that can be any value (hence "anyons").

**Non-Abelian anyons** are even more exotic: exchanging them produces a unitary transformation in a degenerate ground state manifold, not just a phase. This degeneracy is topologically protected and can encode qubits.

### Fusion Rules

Anyons combine according to fusion rules:

$$a \times b = \sum_c N^c_{ab} \, c$$

where $N^c_{ab}$ are non-negative integers (fusion multiplicities).

**Fibonacci anyons**:
$$\tau \times \tau = 1 + \tau$$

**Ising anyons**:
$$\sigma \times \sigma = 1 + \psi$$
$$\sigma \times \psi = \sigma$$
$$\psi \times \psi = 1$$

### Quantum Dimensions

Each anyon type has a quantum dimension $d_a$ that characterizes the growth of the Hilbert space:

| Model | Anyon | Quantum Dimension |
|-------|-------|-------------------|
| Fibonacci | 1 (vacuum) | 1 |
| Fibonacci | τ (tau) | φ ≈ 1.618 |
| Ising | 1 (vacuum) | 1 |
| Ising | σ (sigma) | √2 |
| Ising | ψ (psi) | 1 |

The **total quantum dimension** is:
$$D = \sqrt{\sum_a d_a^2}$$

For Fibonacci anyons: $D = \sqrt{1 + \phi^2} = \sqrt{2 + \phi}$

## Anyon Systems

### Creating Anyon Models

```c
#include "algorithms/topological/topological.h"

// Fibonacci anyons (universal for quantum computing)
anyon_system_t *fib = anyon_system_fibonacci();

// Ising anyons (Majorana fermions)
anyon_system_t *ising = anyon_system_ising();

// General SU(2)_k (k=3 gives Fibonacci, k=2 gives Ising)
anyon_system_t *su2_4 = anyon_system_su2k(4);

// Query properties
double d_tau = anyon_quantum_dimension(fib, FIB_TAU);  // φ
double D = anyon_total_dimension(fib);  // √(2+φ)

anyon_system_free(fib);
```

### Fusion Trees

A **fusion tree** represents how multiple anyons fuse to a definite total charge:

```c
// Create 4 tau anyons with total charge 1 (vacuum)
anyon_charge_t charges[] = {FIB_TAU, FIB_TAU, FIB_TAU, FIB_TAU};
fusion_tree_t *tree = fusion_tree_create(fib, charges, 4, FIB_VACUUM);

// Count valid fusion paths (Hilbert space dimension)
uint32_t dim = fusion_count_paths(fib, charges, 4, FIB_VACUUM);
printf("Fusion space dimension: %u\n", dim);  // 2 (one qubit)

fusion_tree_free(tree);
```

The fusion tree amplitudes encode the quantum state in the topologically protected subspace.

### F-Matrices and R-Matrices

**F-matrices** (6j-symbols) relate different fusion orderings:

$$(a \times b) \times c \xleftrightarrow{F} a \times (b \times c)$$

```c
// Get F-symbol F^{τττ}_τ[1, τ]
double complex F = get_F_symbol(fib,
    FIB_TAU, FIB_TAU, FIB_TAU, FIB_TAU,  // a,b,c,d
    FIB_VACUUM, FIB_TAU);                 // e,f
```

**R-matrices** encode the phase acquired during braiding:

$$R^{ab}_c = \text{phase when exchanging } a \text{ and } b \text{ that fuse to } c$$

```c
// Get braiding phase R^{ττ}_τ
double complex R = get_R_symbol(fib, FIB_TAU, FIB_TAU, FIB_TAU);
// R = e^{4πi/5} for Fibonacci anyons
```

## Braiding Operations

### Elementary Braids

Braiding is the fundamental operation in TQC. When anyon $i$ is exchanged with anyon $i+1$:

```c
// Braid anyons at positions 1 and 2 (clockwise)
qs_error_t err = braid_anyons(tree, 1, true);

// Counter-clockwise (inverse braid)
braid_anyons(tree, 1, false);
```

The braiding operation:
1. Applies R-matrix for the direct phase
2. Uses F-moves to change basis when needed
3. Preserves the total charge

### Basis Changes

F-moves change the fusion order without braiding:

```c
// Apply F-move at vertex 2
apply_F_move(tree, 2);
```

This is essential for computing composite braids involving non-adjacent anyons.

## Anyonic Qubits

### Qubit Encoding

For Fibonacci anyons, a single qubit is encoded in 4 anyons with total charge 1:

$$|0\rangle \sim (\tau \times \tau \to 1) \times (\tau \times \tau \to 1) \to 1$$
$$|1\rangle \sim (\tau \times \tau \to \tau) \times (\tau \times \tau \to \tau) \to 1$$

```c
// Create a 2-qubit anyonic register
anyonic_register_t *reg = anyonic_register_create(fib, 2);

// Qubits are initialized in |0⟩ state
```

### Anyonic Gates

Gates are implemented via braiding sequences:

```c
// NOT gate (via middle anyon braids)
anyonic_not(reg, 0);

// Approximate Hadamard (Fibonacci anyons are universal)
anyonic_hadamard(reg, 0);

// T gate with specified precision
anyonic_T_gate(reg, 0, 1e-6);

// Two-qubit entangling gate
anyonic_entangle(reg, 0, 1);

anyonic_register_free(reg);
```

### Universality

**Fibonacci anyons** are universal for quantum computation—any quantum gate can be approximated to arbitrary precision using braiding alone. This is remarkable because:
- No additional operations (like magic state injection) are needed
- The approximation converges efficiently (Solovay-Kitaev theorem applies)

**Ising anyons** are not universal alone but become universal with the addition of a non-topological "magic" gate.

## Surface Codes

Surface codes are a practical realization of topological error correction using only 2D nearest-neighbor interactions.

### Code Structure

A distance-$d$ surface code has:
- $d^2$ data qubits on a square lattice
- $(d-1)^2$ X-type (plaquette) stabilizers
- $(d-1)^2$ Z-type (vertex) stabilizers
- 1 logical qubit

```c
// Create distance-5 surface code
surface_code_t *code = surface_code_create(5);

// Initialize in logical |0⟩
surface_code_init_logical_zero(code);

// Or logical |+⟩
surface_code_init_logical_plus(code);
```

### Stabilizer Measurements

```c
// Measure all X-type stabilizers (plaquettes)
surface_code_measure_X_stabilizers(code);

// Measure all Z-type stabilizers (vertices)
surface_code_measure_Z_stabilizers(code);

// Access syndrome
for (int i = 0; i < code->num_ancilla_qubits; i++) {
    printf("X-syndrome[%d] = %d\n", i, code->x_syndrome[i]);
    printf("Z-syndrome[%d] = %d\n", i, code->z_syndrome[i]);
}
```

### Logical Operations

Logical gates are implemented via strings of physical operations:

```c
// Logical X: string from left to right edge
surface_code_logical_X(code);

// Logical Z: string from top to bottom edge
surface_code_logical_Z(code);
```

### Error Correction

```c
// Introduce error on qubit 7
surface_code_apply_error(code, 7, 'X');

// Measure syndromes (will show non-trivial pattern)
surface_code_measure_X_stabilizers(code);
surface_code_measure_Z_stabilizers(code);

// Decode and correct using MWPM decoder
surface_code_decode_correct(code);

surface_code_free(code);
```

### Error Threshold

The surface code has an error threshold of approximately 1% for depolarizing noise—below this physical error rate, logical errors can be suppressed exponentially by increasing the code distance.

## Toric Codes

The toric code is defined on a torus (periodic boundary conditions) and encodes 2 logical qubits.

### Code Structure

```c
// Create L×L toric code (2L² physical qubits, 2 logical qubits)
toric_code_t *toric = toric_code_create(6);

// Initialize to ground state (+1 eigenstate of all stabilizers)
toric_code_init_ground_state(toric);
```

### Anyon Excitations

The toric code has two types of anyons:
- **e-anyons** (electric): Created by Z-string endpoints at vertices
- **m-anyons** (magnetic): Created by X-string endpoints at plaquettes

```c
// Create e-anyon pair at (1,1) and (3,3)
toric_code_create_anyon_pair(toric, 'e', 1, 1, 3, 3);

// Create m-anyon pair
toric_code_create_anyon_pair(toric, 'm', 2, 0, 2, 4);

// Move an anyon
toric_code_move_anyon(toric, 'e', 3, 3, 4, 3);

// Braid an e-anyon around an m-anyon
toric_code_braid(toric, 1, 1, 2, 2);
```

### Mutual Statistics

When an e-anyon encircles an m-anyon, the state acquires a phase of -1. This mutual statistics is the defining feature of the toric code's topological order.

```c
toric_code_free(toric);
```

## Topological Entanglement Entropy

Topological order leaves a universal signature in the entanglement entropy:

$$S_A = \alpha |\partial A| - \gamma + O(1/|\partial A|)$$

where $\gamma = \log D$ is the **topological entanglement entropy**.

### Levin-Wen Formula

```c
// Define three regions forming an annular partition
uint32_t region_A[] = {0, 1, 2, 3};
uint32_t region_B[] = {4, 5, 6, 7};
uint32_t region_C[] = {8, 9, 10, 11};

double S_topo = topological_entanglement_entropy(
    state,
    region_A, 4,
    region_B, 4,
    region_C, 4
);

printf("Topological entropy: %.4f\n", S_topo);
printf("Expected log(D) = %.4f\n", log(anyon_total_dimension(sys)));
```

### Kitaev-Preskill Formula

```c
// Alternative using disk and ring regions
uint32_t center[] = {0, 1, 2, 3, 4};
uint32_t ring[] = {5, 6, 7, 8, 9, 10, 11, 12};

double gamma = kitaev_preskill_entropy(state, center, 5, ring, 8);
```

## Modular Matrices

The modular S and T matrices characterize the topological order and determine anyon fusion and braiding.

### S-Matrix

```c
size_t n = sys->num_charges;
double complex *S = malloc(n * n * sizeof(double complex));

compute_modular_S_matrix(sys, S);

// S-matrix is symmetric and unitary
// S_{ab} relates to braiding statistics
```

### T-Matrix

```c
double complex *T = malloc(n * n * sizeof(double complex));

compute_modular_T_matrix(sys, T);

// T is diagonal: T_{aa} = e^{2πi θ_a}
```

### Topological Spin

```c
// Get topological spin of tau anyon
double complex theta = topological_spin(fib, FIB_TAU);
// θ_τ = 4/5 for Fibonacci anyons
```

## Physical Realizations

| Platform | Anyon Type | Status |
|----------|------------|--------|
| Fractional quantum Hall (ν=5/2) | Ising | Experimental |
| Semiconductor/superconductor | Majorana (Ising) | Demonstrated |
| Kitaev honeycomb materials | Ising | Active research |
| Topological superconductors | Ising | Active research |
| Ultracold atoms | Various | Proposed |

## Example: Fibonacci Qubit

Complete example of a topologically protected qubit:

```c
#include "algorithms/topological/topological.h"
#include <stdio.h>

int main(void) {
    printf("=== Fibonacci Qubit Demo ===\n\n");

    // Create Fibonacci anyon system
    anyon_system_t *fib = anyon_system_fibonacci();
    printf("Fibonacci anyons:\n");
    printf("  d_1 = %.4f\n", anyon_quantum_dimension(fib, FIB_VACUUM));
    printf("  d_τ = %.4f (golden ratio)\n", anyon_quantum_dimension(fib, FIB_TAU));
    printf("  D = %.4f\n\n", anyon_total_dimension(fib));

    // Create single logical qubit (4 tau anyons)
    anyonic_register_t *reg = anyonic_register_create(fib, 1);
    printf("Created 1 logical qubit from 4 tau anyons\n\n");

    // Apply gates via braiding
    printf("Applying NOT gate via braiding...\n");
    anyonic_not(reg, 0);

    printf("Applying approximate Hadamard...\n");
    anyonic_hadamard(reg, 0);

    printf("Applying T gate (precision 1e-6)...\n");
    anyonic_T_gate(reg, 0, 1e-6);

    printf("\nGates are topologically protected!\n");
    printf("Small perturbations cannot affect the computation.\n");

    // Cleanup
    anyonic_register_free(reg);
    anyon_system_free(fib);

    return 0;
}
```

## References

**Foundational Papers**:
- Kitaev, A. (2003). Fault-tolerant quantum computation by anyons. *Ann. Phys.* 303, 2-30.
- Freedman, M., Kitaev, A., Larsen, M., & Wang, Z. (2003). Topological quantum computation. *Bull. Amer. Math. Soc.* 40, 31-38.
- Nayak, C., Simon, S.H., Stern, A., Freedman, M., & Das Sarma, S. (2008). Non-Abelian anyons and topological quantum computation. *Rev. Mod. Phys.* 80, 1083.

**Surface Codes**:
- Dennis, E., Kitaev, A., Landahl, A., & Preskill, J. (2002). Topological quantum memory. *J. Math. Phys.* 43, 4452.
- Fowler, A.G., Mariantoni, M., Martinis, J.M., & Cleland, A.N. (2012). Surface codes: Towards practical large-scale quantum computation. *Phys. Rev. A* 86, 032324.

**Topological Entropy**:
- Kitaev, A. & Preskill, J. (2006). Topological entanglement entropy. *Phys. Rev. Lett.* 96, 110404.
- Levin, M. & Wen, X.G. (2006). Detecting topological order in a ground state wave function. *Phys. Rev. Lett.* 96, 110405.

## See Also

- [Tensor Networks](../concepts/tensor-networks.md) - MPS methods used in anyon simulations
- [Skyrmion Braiding](skyrmion-braiding.md) - Another topological qubit platform
- [API: topological.h](../api/c/topological.md) - Full API reference
