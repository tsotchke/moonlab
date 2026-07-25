# Topological Quantum Computing

Fault-tolerant quantum computation using topologically protected degrees of freedom.

## Overview

Topological quantum computing (TQC) encodes quantum information in non-local, topologically protected states that are inherently robust against local perturbations. Instead of using individual qubit states, TQC uses exotic quasiparticles called **anyons** whose quantum properties depend only on the topology of their worldlines.

Moonlab provides comprehensive TQC simulation including:
- **Anyon models**: Fibonacci, Ising, and SU(2)_k
- **Braiding operations**: F-matrices, R-matrices, fusion trees
- **Topological charge measurement** and measurement-only braiding
- **Braid-word compilation**: exact for Ising Cliffords, Solovay-Kitaev to a
  caller-chosen ε for Fibonacci
- **Error-correcting codes**: Surface codes and toric codes
- **Topological invariants**: Entanglement entropy, modular matrices

> **What is verified.** The F/R symbol tables satisfy the fusion-category
> coherence conditions (pentagon, both hexagons, F-matrix unitarity) to 2.4e-15
> for every built-in model, and the braiding built on them is a faithful unitary
> representation of the Artin braid group: Yang-Baxter to 9.0e-16, far
> commutation exactly 0, σᵢσᵢ⁻¹ = I to 1.3e-15, across Fibonacci, Ising and
> SU(2)₃₋₅ and several anyon counts and charge sectors. Ising braiding realises
> the single-qubit Clifford group exactly (order 24) and the two-qubit Clifford
> group exactly in the dense 6-anyon encoding (order 11520, CNOT included).
> Every number in this document is asserted by `unit_topological`.

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

// General SU(2)_k on k+1 charges labelled 2j = 0..k (k=2 gives Ising).
// Note: k=3 is *not* Fibonacci — SU(2)_3 has four charges, and Fibonacci is
// its even-integer-spin subcategory (use anyon_system_fibonacci()).
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
// Get braiding phase R^{ττ}_c for each fusion channel c
double complex R1 = get_R_symbol(fib, FIB_TAU, FIB_TAU, FIB_VACUUM);  // e^{4πi/5}
double complex Rt = get_R_symbol(fib, FIB_TAU, FIB_TAU, FIB_TAU);     // e^{-3πi/5}
```

(Moonlab uses the conjugate of the convention in which $R^{\tau\tau}_1 = e^{-4\pi i/5}$; the two differ by braiding orientation and are equally consistent.)

## Braiding Operations

### The fusion-path basis

A fusion tree over external charges $a_0 \ldots a_{n-1}$ with total charge $Q$
stores its state in the standard left-linear basis: vertex $v$ fuses
$e_{v-1} \times a_v \to e_v$, with $e_0 = a_0$ and $e_{n-1} = Q$. A basis vector
is one admissible tuple $(e_1, \ldots, e_{n-1})$; `tree->labels` holds one row
of edge charges per path and `tree->amplitudes[p]` is that path's amplitude.
Braiding acts on those labels, which is what makes it positional.

### Elementary Braids

Braiding is the fundamental operation in TQC. When anyon $i$ is exchanged with anyon $i+1$:

```c
// Braid anyons at positions 1 and 2 (clockwise)
qs_error_t err = braid_anyons(tree, 1, true);

// Counter-clockwise (inverse braid)
braid_anyons(tree, 1, false);
```

The braiding operation:
1. Applies the R-matrix phase of each fusion path's own intermediate charge
2. Uses F-moves to change basis when the braided pair is not adjacent in the tree
3. Preserves the total charge

Concretely, $\sigma_0$ is diagonal — the pair meets at vertex 1 — while for
$i \ge 1$ the pair does not meet at a vertex of the standard tree and the
operator is the F-conjugated R-matrix

$$[\sigma_i]_{e'_i e_i} = \sum_f \overline{[F^{e_{i-1} a_{i+1} a_i}_{e_{i+1}}]_{e'_i f}}\; R^{a_i a_{i+1}}_f\; [F^{e_{i-1} a_i a_{i+1}}_{e_{i+1}}]_{e_i f}$$

which is what makes $\sigma_1, \sigma_2, \sigma_3, \ldots$ genuinely different
operators. `unit_topological` measures, over Fibonacci, Ising and SU(2)₃₋₅ and
several anyon counts and charge sectors:

| Relation | Worst residual |
|----------|----------------|
| Yang-Baxter $\sigma_i\sigma_{i+1}\sigma_i = \sigma_{i+1}\sigma_i\sigma_{i+1}$ | 9.0e-16 |
| Far commutation $\sigma_i\sigma_j = \sigma_j\sigma_i$, $\lvert i-j\rvert \ge 2$ | 0.0 |
| Generator unitarity | 1.3e-15 |
| $\sigma_i \sigma_i^{-1} = I$ | 1.3e-15 |

On the four-τ vacuum tree $\sigma_1$ and $\sigma_2$ differ by 1.263 in
amplitude; $\sigma_1$ and $\sigma_3$ coincide there, and that is the physics
rather than a bug — with total charge 1 the pairs $(a_0,a_1)$ and $(a_2,a_3)$
carry conjugate charges, so both braids read the same R-symbol on every path.
On a five-τ tree they separate by 1.618.

### Basis Changes

F-moves change the fusion order without braiding:

```c
// Apply F-move at vertex 2
apply_F_move(tree, 2);
```

This is essential for computing composite braids involving non-adjacent anyons.
The subtree $((e_{v-1} a_v)_{e_v} a_{v+1})_{e_{v+1}}$ becomes
$(e_{v-1} (a_v a_{v+1})_f)_{e_{v+1}}$, the amplitudes are transformed by
$[F^{e_{v-1} a_v a_{v+1}}_{e_{v+1}}]_{e_v f}$, and the label column at that
vertex now carries $f$. Applying it again on the same vertex applies
$F^\dagger$ and restores the standard basis: measured round-trip error 1.1e-16
with the norm preserved to 1.000000000000000.

### Topological charge measurement

A pair's total charge can be measured directly, and braiding can be realised by
measurement alone:

```c
double p = anyon_measure_pair_charge(tree, 1, FIB_VACUUM);  // projects and renormalises
anyon_pair_charge_distribution(tree, 1, probs);             // non-destructive
anyon_forced_measurement_braid(tree, 1, true);              // no anyon transported
```

The projector is the exact one built from the F-symbols. The forced-measurement
braid is the construction of Bonderson, Freedman and Nayak (PRL 101, 010501
(2008)): $\sigma = \sum_c R^{ab}_c \Pi_c$, decomposed in the eigenbasis of the
pair charge that measurement resolves. It reproduces `braid_anyons()` to **0.0**
on all six generators of a four-τ tree — which also shows that measurement adds
no unitary that braiding cannot already produce.

### Verifying the symbol tables

`anyon_verify_coherence()` returns the largest violation of the pentagon
equation, both hexagon equations, and F-matrix unitarity across all charge
configurations:

```c
double residual = anyon_verify_coherence(fib);   // ~3e-16
```

Measured for every built-in model: Fibonacci 3.1e-16, Ising 1.8e-15,
SU(2)_2 1.8e-15, SU(2)_3 1.6e-15, SU(2)_4 2.4e-15, SU(2)_5 1.7e-15.

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
// NOT gate (via braiding inside the qubit's block)
anyonic_not(reg, 0);

// Hadamard (Fibonacci anyons are universal)
anyonic_hadamard(reg, 0);

// T gate with specified precision
anyonic_T_gate(reg, 0, 1e-6);

// Arbitrary single-qubit unitary at a chosen epsilon
double achieved;
anyonic_apply_unitary(reg, 0, target, 1e-10, &achieved);

// Two-qubit entangling gate
anyonic_entangle(reg, 0, 1);

anyonic_register_free(reg);
```

Measured, starting from $|0\rangle_L$:

| Gate | Ising | Fibonacci |
|------|-------|-----------|
| NOT  | exact: $\lvert a_0\rvert$ = 0.0, $\lvert a_1\rvert$ = 1.000000000000000 | ε = 1e-8, achieved 1.3e-12 |
| H    | exact: (0.707106781186548, 0.707106781186548) | (0.707106784, 0.707106778) |
| T    | `QS_ERROR_NOT_SUPPORTED` — not a Clifford | compiled to the requested ε |

### Compiling braid words

```c
// Ising: every single-qubit Clifford has an exact braid word.
double err;
braid_word_t *w = ising_compile_clifford(ising, target, &err);   // err ~ 1e-16

// Ising, dense 6-anyon encoding: exact two-qubit Cliffords, CNOT included,
// and no leakage subspace exists at all.
braid_word_t *cnot = ising_compile_clifford2(ising, cnot_4x4, &err);

// Fibonacci: Solovay-Kitaev to any epsilon the caller asks for. The distance
// to the target is measured on the returned word before it is handed back.
braid_word_t *h = fibonacci_compile_su2(fib, hadamard, 1e-10, &err);

// Fibonacci exact gates: R_z(m pi/5), m = 0..9, including the logical Z.
braid_word_t *z = fibonacci_exact_phase_gate(5);
```

Solovay-Kitaev scaling, target H, base net 52959 elements with covering radius
0.101:

| ε | crossings | achieved |
|---|-----------|----------|
| 1e-2 | 297 | 1.315e-03 |
| 1e-4 | 1413 | 7.941e-05 |
| 1e-6 | 33099 | 9.133e-09 |
| 1e-8 | 33099 | 9.133e-09 |
| 1e-10 | 160469 | 1.663e-12 |

Word length grows as $\log^{4.1}(1/\epsilon)$ measured, against
$\log_{3/2} 5 = 3.97$ from the recursion's five-fold branching and 3/2 error
exponent. The floor is about 1.7e-12, where rounding in a length-$5^n$ matrix
product overtakes the recursion's own residual, so ε ≥ 1e-11 is the guaranteed
range.

### Universality

**Fibonacci anyons** are universal for quantum computation—any quantum gate can be approximated to arbitrary precision using braiding alone. This is remarkable because:
- No additional operations (like magic state injection) are needed
- The approximation converges efficiently (Solovay-Kitaev theorem applies)

**Ising anyons** are not universal alone but become universal with the addition of a non-topological "magic" gate. Their braid image is finite — order 24 on one
qubit, 11520 on two — so Moonlab compiles Cliffords exactly and reports T as
unreachable instead of approximating it.

### Which Fibonacci gates are exactly realisable

"Universal by approximation" is precise, and the boundary is decidable. Every
Fibonacci braid word has the form

$$B = \begin{pmatrix} p & \varphi^{-1/2} r \\ \varphi^{-1/2} s & t\end{pmatrix},\qquad p,r,s,t \in K = \mathbb{Q}(\zeta_5)$$

which follows from $\sigma_1 = \mathrm{diag}(e^{4\pi i/5}, e^{-3\pi i/5})$, the
real $F^{\tau\tau\tau}_\tau$, and closure of that shape under multiplication.
Hence:

- **Exact**: $R_z(m\pi/5)$, $m = 0..9$. In particular
  $\sigma_1^5 = \mathrm{diag}(1,-1) = Z$, measured error 5.6e-16, and all ten
  phase gates to 1.2e-15.
- **Impossible**: H, X and T. H would force $\varphi^{-1/2} = p/r \in K$; X would
  force $r^2 = \varphi\mu$ for a 10th root of unity $\mu$; T would force
  $\mathrm{tr}^2/\det = 2+\sqrt2 \in K$. Each contradicts a property of
  $\mathbb{Q}(\zeta_5)$ — see
  [MATH.md](../../MATH.md#exact-realisability-of-fibonacci-braid-gates) for the
  three proofs. Exhaustive enumeration of every braid word up to length 12
  agrees: the closest approaches are 0.066 (H), 0.113 (X) and 0.075 (T).

Measurement-only protocols reproduce braid transformations exactly — which
`anyon_forced_measurement_braid()` shows directly by matching `braid_anyons()`
to 0.0 — so charge measurement is at least as powerful as braiding. The proofs
above concern finite braid words; no exact adaptive construction for the
Fibonacci H, X or T is known, and none is implemented here. For those three the
answer is the ε-guaranteed compiler, which measures what it returns.

### Two-qubit gates and leakage

`anyonic_entangle()` is a unitary 32-crossing inter-qubit weave in the geometry
of Bonesteel, Hormozi, Zikos and Simon (PRL 95, 140503 (2005)), found by
exhaustive search to length 8 followed by randomised search with hill climbing
at length 32. It entangles: concurrence **0.414242** from a product input.

It also leaks **8.3043e-2**, and that is a property of the 4-anyons-per-qubit
encoding, not of the search. Qubit $q$'s logical bit is the charge of the pair
$(4q, 4q{+}1)$, and the block's vacuum constraint forces $(4q{+}2, 4q{+}3)$ to
carry the same charge; a braid that entangles two blocks must change one of
those pair charges without changing its partner, taking the block out of the
vacuum channel. Exhaustive search confirms it directly: no braid word of length
≤ 8 on the two-block register is both entangling and leakage-free, and none
makes the logical block proportional to a unitary while entangling.

For an exact, leakage-free two-qubit gate use `ising_compile_clifford2()`: its
dense 6-anyon encoding carries two qubits in a 4-dimensional fusion space with
no non-computational subspace at all, and CNOT compiles to a 7-crossing braid
with maximum element error 1.2e-15.

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
