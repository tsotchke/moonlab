# Topological Computing API

C API for topological quantum computing simulation.

## Overview

The topological computing API (`topological.h`) provides functions for simulating topological quantum computing primitives including anyon models, braiding operations, fusion trees, surface codes, and toric codes.

### Maturity

| Area | Status |
|------|--------|
| Anyon models, fusion rules, F/R symbol tables | **Verified.** Pentagon, both hexagons and F-matrix unitarity hold to 2.4e-15 for every built-in model. |
| Fusion trees, per-path charge labelling, quantum dimensions | Stable |
| Braiding (`braid_anyons`, `apply_F_move`) | **Verified braid-group representation.** Yang-Baxter 9.0e-16, far commutation exactly 0, unitarity 1.3e-15. |
| Topological charge measurement, measurement-only braiding | Verified exact against `braid_anyons` (0.0) |
| Ising Clifford compilation (`ising_compile_clifford`, `ising_compile_clifford2`) | **Exact**, error ≤ 1.2e-15 |
| Fibonacci Solovay-Kitaev (`fibonacci_compile_su2`) | Meets any caller ε ≥ 1e-11, measured on the returned word |
| Anyonic gates (`anyonic_*`) | Real logical rotations; exactness per model, see below |
| Surface code, toric code | Stable |
| Topological entanglement entropy | Stable |

Every number above is asserted by `unit_topological` on this host.

### Fusion-tree basis

A `fusion_tree_t` over external charges $a_0 \ldots a_{n-1}$ with total charge
$Q$ stores its state in the standard left-linear fusion basis: vertex $v$ fuses
$e_{v-1} \times a_v \to e_v$, with $e_0 = a_0$ and $e_{n-1} = Q$. A basis
vector is one admissible tuple $(e_1, \ldots, e_{n-1})$, held in `tree->labels`
(one row of `tree->num_vertices` charges per path, lexicographically ordered)
with the amplitude in `tree->amplitudes[p]`. All built-in models are
multiplicity free, so a path is fully determined by its edge labels.

Braiding $\sigma_i$ is diagonal for $i = 0$ — the pair meets at vertex 1 — and
F-conjugated otherwise:

$$[\sigma_i]_{e'_i e_i} = \sum_f \overline{[F^{e_{i-1} a_{i+1} a_i}_{e_{i+1}}]_{e'_i f}}\; R^{a_i a_{i+1}}_f\; [F^{e_{i-1} a_i a_{i+1}}_{e_{i+1}}]_{e_i f}$$

which is what makes $\sigma_1, \sigma_2, \sigma_3, \ldots$ different operators.

### Exact realisability

Ising braiding generates a **finite** group, so exact compilation is a lookup:
the projective image on 4 σ anyons has order 24 (the single-qubit Clifford
group) and on 6 σ anyons order 11520 (the two-qubit Clifford group, CNOT
included). `ising_compile_clifford()` returns exact words, and returns `NULL`
for a non-Clifford target such as T rather than approximating it.

Fibonacci braiding generates a **countable dense** subgroup of PSU(2). Which
targets are exactly realisable is decided by a field argument: every braid word
has the form

$$B = \begin{pmatrix} p & \varphi^{-1/2} r \\ \varphi^{-1/2} s & t\end{pmatrix},\qquad p,r,s,t \in K = \mathbb{Q}(\zeta_5)$$

which follows from $\sigma_1 = \mathrm{diag}(e^{4\pi i/5}, e^{-3\pi i/5})$, the
real $F^{\tau\tau\tau}_\tau$, and closure of that shape under multiplication.
Consequently:

- **Exact:** $R_z(m\pi/5)$ for $m = 0..9$, including the logical Pauli
  $Z = \sigma_1^5$ — see `fibonacci_exact_phase_gate()`.
- **Impossible:** H, X and T. The three proofs are in
  [MATH.md](../../../MATH.md#exact-realisability-of-fibonacci-braid-gates).
  `fibonacci_compile_su2()` is the answer for those, and it meets whatever ε the
  caller asks for.

Measurement-only protocols (Bonderson, Freedman and Nayak, PRL 101, 010501
(2008)) reproduce braid transformations exactly, which
`anyon_forced_measurement_braid()` demonstrates by agreeing with
`braid_anyons()` to 0.0. The proofs concern finite braid words; no exact
adaptive construction for the Fibonacci H, X or T is known, and none is
implemented here. For those three the answer is `fibonacci_compile_su2()`'s
checked ε guarantee.

## Header

```c
#include "algorithms/topological/topological.h"
```

## Anyon Models

### Types

#### `anyon_model_t`

Anyon model enumeration.

```c
typedef enum {
    ANYON_MODEL_FIBONACCI,  // Fibonacci anyons (τ×τ = 1+τ)
    ANYON_MODEL_ISING,      // Ising anyons (σ×σ = 1+ψ)
    ANYON_MODEL_SU2_K       // SU(2)_k anyons
} anyon_model_t;
```

#### `anyon_charge_t`

Anyon charge type (alias for `uint32_t`).

**Predefined charges**:

```c
// Fibonacci anyons
#define FIB_VACUUM 0
#define FIB_TAU    1

// Ising anyons
#define ISING_VACUUM 0
#define ISING_SIGMA  1
#define ISING_PSI    2
```

#### `anyon_system_t`

Complete anyon model specification.

```c
typedef struct {
    anyon_model_t type;
    uint32_t num_charges;           // Number of distinct charges
    uint32_t level;                 // Level k for SU(2)_k
    double complex **F_matrices;    // F-symbols (6j-symbols)
    double complex **R_matrices;    // R-symbols (braiding phases)
    uint32_t ***fusion_rules;       // N^c_{ab} fusion multiplicities
} anyon_system_t;
```

### Functions

#### `anyon_system_fibonacci`

Create Fibonacci anyon system.

```c
anyon_system_t *anyon_system_fibonacci(void);
```

Fibonacci anyons have fusion rule $\tau \times \tau = 1 + \tau$ and are universal for quantum computation via braiding alone.

**Returns**: Fibonacci anyon system.

#### `anyon_system_ising`

Create Ising anyon system.

```c
anyon_system_t *anyon_system_ising(void);
```

Ising anyons have fusion rules:
- $\sigma \times \sigma = 1 + \psi$
- $\sigma \times \psi = \sigma$
- $\psi \times \psi = 1$

**Returns**: Ising anyon system.

#### `anyon_system_su2k`

Create SU(2)_k anyon system.

```c
anyon_system_t *anyon_system_su2k(uint32_t k);
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `k` | `uint32_t` | Level parameter (k ≥ 1); k=2 gives Ising |

**Returns**: SU(2)_k anyon system on $k+1$ charges labelled by $2j = 0 \ldots k$, with F/R symbols generated from the quantum 6j-symbols.

**Note**: `k = 3` does *not* return Fibonacci. SU(2)_3 has four charges ($2j = 0,1,2,3$); Fibonacci is the even-integer-spin subcategory of SU(2)_3 and is a separate hand-coded model. Use `anyon_system_fibonacci()` for Fibonacci anyons.

#### `anyon_system_free`

Free anyon system.

```c
void anyon_system_free(anyon_system_t *sys);
```

#### `anyon_quantum_dimension`

Get quantum dimension of an anyon charge.

```c
double anyon_quantum_dimension(const anyon_system_t *sys, anyon_charge_t charge);
```

**Quantum dimensions**:
- Fibonacci: $d_1 = 1$, $d_\tau = \phi$ (golden ratio ≈ 1.618)
- Ising: $d_1 = 1$, $d_\sigma = \sqrt{2}$, $d_\psi = 1$

**Returns**: Quantum dimension $d_a$.

#### `anyon_total_dimension`

Get total quantum dimension.

```c
double anyon_total_dimension(const anyon_system_t *sys);
```

**Returns**: $D = \sqrt{\sum_a d_a^2}$

## Fusion Trees

### Types

#### `fusion_node_t`

Fusion tree node.

```c
typedef struct fusion_node {
    anyon_charge_t left;         // Left incoming charge
    anyon_charge_t right;        // Right incoming charge
    anyon_charge_t result;       // Outgoing fused charge
    struct fusion_node *parent;
    struct fusion_node *left_child;
    struct fusion_node *right_child;
} fusion_node_t;
```

#### `fusion_tree_t`

Fusion tree state.

```c
typedef struct {
    anyon_system_t *anyon_sys;     // Anyon model
    anyon_charge_t *external;       // External (physical) anyon charges
    uint32_t num_anyons;            // Number of external anyons
    anyon_charge_t total_charge;    // Total fused charge
    fusion_node_t *root;            // Root of fusion tree
    double complex *amplitudes;     // Amplitudes for each fusion path
    uint32_t num_paths;             // Number of valid fusion paths
} fusion_tree_t;
```

### Functions

#### `fusion_tree_create`

Create fusion tree from external charges.

```c
fusion_tree_t *fusion_tree_create(anyon_system_t *sys,
                                  const anyon_charge_t *charges,
                                  uint32_t num_anyons,
                                  anyon_charge_t total_charge);
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `sys` | `anyon_system_t*` | Anyon system |
| `charges` | `const anyon_charge_t*` | External anyon charges |
| `num_anyons` | `uint32_t` | Number of anyons |
| `total_charge` | `anyon_charge_t` | Required total charge |

**Returns**: Fusion tree state.

#### `fusion_tree_free`

Free fusion tree.

```c
void fusion_tree_free(fusion_tree_t *tree);
```

#### `fusion_count_paths`

Count valid fusion paths.

```c
uint32_t fusion_count_paths(const anyon_system_t *sys,
                            const anyon_charge_t *charges,
                            uint32_t num_anyons,
                            anyon_charge_t total_charge);
```

**Returns**: Dimension of the fusion space.

## Braiding Operations

#### `braid_anyons`

Braid two adjacent anyons: the braid generator σᵢ.

```c
qs_error_t braid_anyons(fusion_tree_t *tree, uint32_t position, bool clockwise);
```

Exchanges the anyons at `position` and `position + 1`, applying the R-matrix
phase of each fusion path's own intermediate charge together with the F-matrix
basis change needed when the exchanged pair does not meet at a vertex of the
standard tree. See [Fusion-tree basis](#fusion-tree-basis) for the matrix.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `tree` | `fusion_tree_t*` | Fusion tree (modified in place); must be in the standard basis |
| `position` | `uint32_t` | Position of left anyon to braid |
| `clockwise` | `bool` | Direction (true = σ, false = σ⁻¹) |

**Returns**: `QS_SUCCESS`, `QS_ERROR_INVALID_QUBIT` if `position` is out of
range, `QS_ERROR_INVALID_STATE` if an F-move is outstanding on the tree.

The map i ↦ σᵢ is a unitary representation of the Artin braid group Bₙ.
Measured on this host:

| Relation | Fibonacci | Ising | SU(2)₃ | SU(2)₄ | SU(2)₅ |
|----------|-----------|-------|--------|--------|--------|
| Yang-Baxter σᵢσᵢ₊₁σᵢ = σᵢ₊₁σᵢσᵢ₊₁ | 3.9e-16 | 7.7e-16 | 9.0e-16 | 8.3e-16 | 4.5e-16 |
| Far commutation, \|i−j\| ≥ 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Unitarity | 2.2e-16 | 1.3e-15 | 1.0e-15 | 8.9e-16 | 6.7e-16 |
| σᵢσᵢ⁻¹ = I | 4.5e-16 | 1.3e-15 | 1.0e-15 | 8.9e-16 | 6.7e-16 |

#### `apply_F_move`

Apply an F-move (change of fusion basis) at a vertex.

```c
qs_error_t apply_F_move(fusion_tree_t *tree, uint32_t vertex);
```

Changes fusion order: $(a \times b) \times c \leftrightarrow a \times (b \times c)$.
The subtree $((e_{v-1} a_v)_{e_v} a_{v+1})_{e_{v+1}}$ becomes
$(e_{v-1} (a_v a_{v+1})_f)_{e_{v+1}}$ and the amplitudes are transformed by
$[F^{e_{v-1} a_v a_{v+1}}_{e_{v+1}}]_{e_v f}$; the column of `tree->labels` at
`vertex` then holds $f$ and `tree->recoupled_vertex` is set. Calling it again on
the same vertex applies $F^\dagger$ and restores the standard basis, so it is an
involution (measured round-trip error 1.1e-16, norm 1.000000000000000). This is
the basis change that lets braids of anyons that are not adjacent in the tree be
composed from F- and R-moves.

Valid `vertex` range is `1 <= vertex <= num_anyons - 2`; anything else returns
`QS_ERROR_INVALID_PARAM`. Only one vertex may be recoupled at a time.

#### `anyon_measure_pair_charge`

Measure the total topological charge of an adjacent anyon pair.

```c
double anyon_measure_pair_charge(fusion_tree_t *tree, uint32_t position,
                                 anyon_charge_t outcome);
```

Projects onto the sector where the anyons at `position` and `position + 1` fuse
to `outcome`, renormalising, and returns the probability. The projector is the
exact one built from the F-symbols: the tree is recoupled so the pair meets at a
vertex, the other channels are annihilated, and the tree is recoupled back.

`anyon_pair_charge_distribution()` returns the same probabilities
non-destructively.

#### `anyon_forced_measurement_braid`

Realise a braid generator by topological charge measurement alone.

```c
qs_error_t anyon_forced_measurement_braid(fusion_tree_t *tree, uint32_t position,
                                          bool clockwise);
```

The forced-measurement construction of Bonderson, Freedman and Nayak, PRL 101,
010501 (2008): the braid transformation is $\sigma = \sum_c R^{ab}_c \Pi_c$ with
$\Pi_c$ the charge-$c$ projector that measurement implements, so no anyon is
transported. Agrees with `braid_anyons()` to **0.0** on all six generators of a
four-τ tree.

#### `get_F_symbol`

Get F-matrix element.

```c
double complex get_F_symbol(const anyon_system_t *sys,
                            anyon_charge_t a, anyon_charge_t b,
                            anyon_charge_t c, anyon_charge_t d,
                            anyon_charge_t e, anyon_charge_t f);
```

**Returns**: $F^{abc}_d[e,f]$

#### `get_R_symbol`

Get R-matrix element.

```c
double complex get_R_symbol(const anyon_system_t *sys,
                            anyon_charge_t a, anyon_charge_t b,
                            anyon_charge_t c);
```

**Returns**: $R^{ab}_c$ (braiding phase)

#### `anyon_verify_coherence`

Maximum residual of the fusion-category coherence conditions.

```c
double anyon_verify_coherence(const anyon_system_t *sys);
```

Returns the largest absolute violation, over all charge configurations, of

- the MacLane pentagon equation for the F-symbols,
- both hexagon equations relating F- and R-symbols, and
- unitarity of every F-matrix, over the row set fixed by the fusion rules.

A consistent braided fusion category returns ~0 (machine precision); a nonzero
value certifies that the tabulated F/R symbols are not mutually consistent.

**Returns**: max coherence residual (≥ 0), or −1 on error.

**Measured** (all built-in models, macOS/arm64, Release):

| Model | Residual |
|-------|----------|
| Fibonacci | 3.1e-16 |
| Ising | 1.8e-15 |
| SU(2)_2 | 1.8e-15 |
| SU(2)_3 | 1.6e-15 |
| SU(2)_4 | 2.4e-15 |
| SU(2)_5 | 1.7e-15 |

## Braid Words and Compilation

#### `braid_word_t`

A word in the braid generators, applied left to right.

```c
typedef struct { uint32_t position; uint8_t clockwise; } braid_gen_t;
typedef struct { braid_gen_t *gens; uint32_t length; uint32_t capacity; } braid_word_t;
```

`braid_word_create/free/append/append_word/append_inverse/clone/length/reduce`
build and manipulate them; `braid_word_apply()` runs one on a fusion tree and
`braid_word_matrix()` returns its matrix on the fusion space (column `j` is the
image of basis path `j`).

#### `su2_projective_distance`

```c
double su2_projective_distance(const double complex a[4], const double complex b[4]);
```

$d(U,V) = \min_\phi \lVert U - e^{i\phi} V\rVert_{op}$ after normalising both to
SU(2); 0 iff they agree up to a phase. This is the metric every ε in this API is
measured in. It is evaluated as a phase-aligned Frobenius difference rather than
from $\sqrt{2 - |\mathrm{tr}|}$, which would cancel to $\sqrt{\epsilon}$ once the
two agree to machine precision.

#### `ising_compile_clifford`

```c
braid_word_t *ising_compile_clifford(anyon_system_t *sys,
                                     const double complex target[4],
                                     double *achieved_error);
```

Exact single-qubit Clifford braid word on the 4-σ qubit (dimension 2, no
leakage). The projective braid image is enumerated by breadth-first search of
its Cayley graph and has order **24** — the full single-qubit Clifford group —
so a Clifford target gets its shortest exact word and a non-Clifford target gets
`NULL`. Measured: X 2 crossings / 3.1e-16, Y 4 / 4.0e-16, Z 2 / 5.6e-17,
H 3 / 3.4e-16, S 1 / 1.4e-16; T returns `NULL`.

Conjugation by each generator maps Paulis to Paulis to ≤ 2.5e-16, i.e. the
braids act as Cliffords in the Heisenberg picture as well.

#### `ising_compile_clifford2`

```c
braid_word_t *ising_compile_clifford2(anyon_system_t *sys,
                                      const double complex target[16],
                                      double *achieved_error);
```

Exact **two-qubit** Clifford braid word in the dense encoding: 6 σ anyons of
total charge 1 carry exactly two qubits (dimension 4), so there is no
non-computational subspace and hence no leakage at all. Qubit 0 is the charge of
the pair (0,1), qubit 1 the charge of the pair (2,3), with 1 → |0⟩ and ψ → |1⟩.

The projective image of ⟨σ₁…σ₅⟩ has order **11520** — the full two-qubit
Clifford group modulo phase. Measured: CNOT 7 crossings / max element error
1.2e-15, CZ 3 crossings / 8.9e-16.

#### `fibonacci_compile_su2`

```c
braid_word_t *fibonacci_compile_su2(anyon_system_t *sys,
                                    const double complex target[4],
                                    double epsilon, double *achieved_error);
```

Solovay-Kitaev compilation on the 3-τ qubit (3 τ anyons of total charge τ,
dimension 2, generators σ₁ and σ₂ at positions 0 and 1). The returned word's
distance to the target is **measured before it is returned**, so `epsilon` is a
checked guarantee, not an asymptotic statement; `NULL` is returned if it cannot
be met. The base net (52959 elements, covering radius 0.101) is built on first
use in about 1.8 s and cached.

Measured, target H:

| ε | crossings | achieved |
|---|-----------|----------|
| 1e-2 | 297 | 1.315e-03 |
| 1e-4 | 1413 | 7.941e-05 |
| 1e-6 | 33099 | 9.133e-09 |
| 1e-8 | 33099 | 9.133e-09 |
| 1e-10 | 160469 | 1.663e-12 |

Word length grows as $\log^{4.1}(1/\epsilon)$ measured, against $\log_{3/2} 5 =
3.97$ from the recursion's 5-fold branching and 3/2 error exponent. The floor is
about 1.7e-12: below that the rounding in a length-5ⁿ matrix product dominates
the recursion's own residual, so ε ≥ 1e-11 is the guaranteed range.

#### `fibonacci_exact_phase_gate`

```c
braid_word_t *fibonacci_exact_phase_gate(uint32_t m);
```

Exact $R_z(m\pi/5)$, m = 0..9. Since $\sigma_1$ is projectively
$\mathrm{diag}(1, e^{3\pi i/5})$ and 3 is invertible mod 10, $\langle\sigma_1\rangle$
is exactly the order-10 group of those phase gates; m = 5 is the logical Pauli
Z. Worst measured error over all ten: 1.2e-15.

#### `fibonacci_braid_net_size`

```c
uint32_t fibonacci_braid_net_size(anyon_system_t *sys, double *covering_radius);
```

Size and measured covering radius of the Solovay-Kitaev base net — the ε₀ the
recursion starts from.

## Anyonic Quantum Gates

Every gate below is a braid word applied to the register's fusion tree, so each
is a genuine unitary on the encoded qubit. Qubit `q` occupies anyons 4q..4q+3
and its logical bit is the charge of the pair (4q, 4q+1); braids inside a block
preserve every other block's charge, so a single-qubit gate acts as U ⊗ I.

How exact a gate is depends on the model, and the difference is not glossed:

- **Ising**: X, Z, H and every other single-qubit Clifford are exact (~1e-16).
  T is not a Clifford, so `anyonic_T_gate()` returns `QS_ERROR_NOT_SUPPORTED`
  on an Ising register rather than pretending otherwise.
- **Fibonacci**: Z is exact; X, H and T are compiled by `fibonacci_compile_su2`
  to `ANYONIC_GATE_DEFAULT_EPSILON` (1e-6) or to whatever ε
  `anyonic_apply_unitary()` is given, with the achieved error measured.

### Types

#### `anyonic_register_t`

Anyonic qubit register.

```c
typedef struct {
    fusion_tree_t *tree;
    anyon_system_t *sys;
    uint32_t num_logical_qubits;
} anyonic_register_t;
```

### Functions

#### `anyonic_register_create`

Create anyonic qubit register.

```c
anyonic_register_t *anyonic_register_create(anyon_system_t *sys,
                                            uint32_t num_qubits);
```

#### `anyonic_register_free`

Free anyonic register.

```c
void anyonic_register_free(anyonic_register_t *reg);
```

#### `anyonic_not`

Apply a logical NOT (Pauli X) via braiding.

```c
qs_error_t anyonic_not(anyonic_register_t *reg, uint32_t qubit);
```

Ising: exact Clifford braid word. Measured — |0⟩_L → |1⟩_L with |a₀| = 0.0 and
|a₁| = 1.000000000000000. Fibonacci: compiled to
`ANYONIC_GATE_DEFAULT_EPSILON`; at ε = 1e-8 the measured error is 1.3e-12 and
the state reaches |a₁| = 1.000000000.

#### `anyonic_hadamard`

Apply a Hadamard via braiding.

```c
qs_error_t anyonic_hadamard(anyonic_register_t *reg, uint32_t qubit);
```

Ising: exact (H is Clifford) — measured (0.707106781186548, 0.707106781186548)
from |0⟩_L. Fibonacci: compiled, measured (0.707106784, 0.707106778).

#### `anyonic_T_gate`

Apply a T gate (π/8 phase gate) via braiding.

```c
qs_error_t anyonic_T_gate(anyonic_register_t *reg, uint32_t qubit,
                          double precision);
```

Fibonacci only: T is compiled by Solovay-Kitaev to within `precision`, and the
returned state is that of a braid word whose measured distance to T is below it.
Ising braiding generates only the Clifford group, which does not contain T, so
an Ising register returns `QS_ERROR_NOT_SUPPORTED`.

#### `anyonic_apply_unitary`

Apply an arbitrary single-qubit unitary via braiding.

```c
qs_error_t anyonic_apply_unitary(anyonic_register_t *reg, uint32_t qubit,
                                 const double complex target[4],
                                 double epsilon, double *achieved);
```

Compiles `target` with `ising_compile_clifford()` or `fibonacci_compile_su2()`
as appropriate and applies the resulting braid word, reporting the accuracy
actually attained.

#### `anyonic_entangle`

Apply a two-qubit entangling gate by weaving between blocks.

```c
qs_error_t anyonic_entangle(anyonic_register_t *reg,
                            uint32_t qubit1, uint32_t qubit2);
```

A unitary 32-crossing inter-qubit weave in the geometry of Bonesteel, Hormozi,
Zikos and Simon (PRL 95, 140503 (2005)), obtained by exhaustive search to length
8 followed by randomised search with hill climbing at length 32. It entangles: a
product input reaches concurrence **0.414242**.

It also leaks **8.3043e-2** of the amplitude out of the logical subspace, and
that is a property of the encoding rather than of the search. Qubit q's logical
bit is the charge of the pair (4q, 4q+1), and the block's vacuum constraint
forces the pair (4q+2, 4q+3) to carry the same charge; a braid that entangles
two blocks must change one of those pair charges without changing its partner,
taking the block out of the vacuum channel. Exhaustive search confirms it: no
braid word of length ≤ 8 on the two-block register is both entangling and
leakage-free, and none makes the logical block proportional to a unitary while
entangling. For an exact, leakage-free two-qubit gate use
[`ising_compile_clifford2`](#ising_compile_clifford2).

#### `anyonic_register_logical_state`

```c
double anyonic_register_logical_state(const anyonic_register_t *reg,
                                      double complex *out);
```

Writes the amplitudes of the register's logical basis states and returns the
probability remaining inside the logical subspace (1 − leakage).

## Surface Code

### Types

#### `surface_code_t`

Surface code lattice.

```c
typedef struct {
    uint32_t distance;           // Code distance
    uint32_t num_data_qubits;    // d²
    uint32_t num_ancilla_qubits; // (d-1)² for each type
    quantum_state_t *state;      // Full quantum state
    uint8_t *x_syndrome;         // X-type syndrome measurements
    uint8_t *z_syndrome;         // Z-type syndrome measurements
} surface_code_t;
```

### Functions

#### `surface_code_create`

Create surface code.

```c
surface_code_t *surface_code_create(uint32_t distance);
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `distance` | `uint32_t` | Code distance (odd, ≥3) |

#### `surface_code_free`

Free surface code.

```c
void surface_code_free(surface_code_t *code);
```

#### `surface_code_init_logical_zero`

Initialize surface code in logical |0⟩.

```c
qs_error_t surface_code_init_logical_zero(surface_code_t *code);
```

#### `surface_code_init_logical_plus`

Initialize surface code in logical |+⟩.

```c
qs_error_t surface_code_init_logical_plus(surface_code_t *code);
```

#### `surface_code_logical_X`

Apply logical X gate.

```c
qs_error_t surface_code_logical_X(surface_code_t *code);
```

#### `surface_code_logical_Z`

Apply logical Z gate.

```c
qs_error_t surface_code_logical_Z(surface_code_t *code);
```

#### `surface_code_measure_X_stabilizers`

Measure X-type (plaquette) stabilizers.

```c
qs_error_t surface_code_measure_X_stabilizers(surface_code_t *code);
```

#### `surface_code_measure_Z_stabilizers`

Measure Z-type (vertex) stabilizers.

```c
qs_error_t surface_code_measure_Z_stabilizers(surface_code_t *code);
```

#### `surface_code_apply_error`

Apply single-qubit error.

```c
qs_error_t surface_code_apply_error(surface_code_t *code,
                                    uint32_t qubit, char error_type);
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `qubit` | `uint32_t` | Data qubit index |
| `error_type` | `char` | 'X', 'Y', or 'Z' |

#### `surface_code_decode_correct`

Decode syndrome and apply correction using MWPM decoder.

```c
qs_error_t surface_code_decode_correct(surface_code_t *code);
```

## Toric Code

### Types

#### `toric_code_t`

Toric code on a torus.

```c
typedef struct {
    uint32_t L;                  // Linear size (L×L torus)
    uint32_t num_qubits;         // 2L² edge qubits
    quantum_state_t *state;      // Full quantum state
    uint8_t *vertex_syndrome;    // A_v eigenvalues
    uint8_t *plaquette_syndrome; // B_p eigenvalues
} toric_code_t;
```

### Functions

#### `toric_code_create`

Create toric code.

```c
toric_code_t *toric_code_create(uint32_t L);
```

#### `toric_code_free`

Free toric code.

```c
void toric_code_free(toric_code_t *code);
```

#### `toric_code_init_ground_state`

Initialize toric code ground state.

```c
qs_error_t toric_code_init_ground_state(toric_code_t *code);
```

#### `toric_code_create_anyon_pair`

Create an anyon pair.

```c
qs_error_t toric_code_create_anyon_pair(toric_code_t *code,
                                        char type,
                                        uint32_t x1, uint32_t y1,
                                        uint32_t x2, uint32_t y2);
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | `char` | 'e' for electric (Z-string), 'm' for magnetic (X-string) |
| `x1, y1` | `uint32_t` | Start position |
| `x2, y2` | `uint32_t` | End position |

#### `toric_code_move_anyon`

Move an anyon.

```c
qs_error_t toric_code_move_anyon(toric_code_t *code, char type,
                                 uint32_t from_x, uint32_t from_y,
                                 uint32_t to_x, uint32_t to_y);
```

#### `toric_code_braid`

Braid anyons in toric code.

```c
qs_error_t toric_code_braid(toric_code_t *code,
                            uint32_t anyon1_x, uint32_t anyon1_y,
                            uint32_t anyon2_x, uint32_t anyon2_y);
```

## Topological Entanglement Entropy

#### `topological_entanglement_entropy`

Compute topological entanglement entropy using Levin-Wen formula.

```c
double topological_entanglement_entropy(const quantum_state_t *state,
                                        const uint32_t *region_A, uint32_t num_A,
                                        const uint32_t *region_B, uint32_t num_B,
                                        const uint32_t *region_C, uint32_t num_C);
```

Computes $S_{\text{topo}} = S_A + S_B + S_C - S_{AB} - S_{BC} - S_{AC} + S_{ABC}$

For topologically ordered states, $S_{\text{topo}} = \log D$ where $D$ is the total quantum dimension.

#### `kitaev_preskill_entropy`

Compute Kitaev-Preskill topological entropy.

```c
double kitaev_preskill_entropy(const quantum_state_t *state,
                               const uint32_t *center_qubits, uint32_t num_center,
                               const uint32_t *ring_qubits, uint32_t num_ring);
```

**Returns**: Topological entropy $\gamma = \log D$

## Modular Matrices

#### `compute_modular_S_matrix`

Compute modular S-matrix.

```c
void compute_modular_S_matrix(const anyon_system_t *sys,
                              double complex *S_matrix);
```

#### `compute_modular_T_matrix`

Compute modular T-matrix.

```c
void compute_modular_T_matrix(const anyon_system_t *sys,
                              double complex *T_matrix);
```

#### `topological_spin`

Compute topological spin.

```c
double complex topological_spin(const anyon_system_t *sys,
                                anyon_charge_t charge);
```

**Returns**: $e^{2\pi i \theta_a}$ where $\theta_a$ is the topological spin.

## Example

```c
#include "algorithms/topological/topological.h"
#include <stdio.h>

int main(void) {
    // Create Fibonacci anyon system
    anyon_system_t *fib = anyon_system_fibonacci();

    printf("Fibonacci anyons:\n");
    printf("  d_1 = %.4f\n", anyon_quantum_dimension(fib, FIB_VACUUM));
    printf("  d_τ = %.4f (golden ratio)\n", anyon_quantum_dimension(fib, FIB_TAU));
    printf("  D = %.4f\n", anyon_total_dimension(fib));

    // Create a qubit (4 tau anyons with total charge 1)
    anyonic_register_t *reg = anyonic_register_create(fib, 1);

    // Apply gates via braiding
    anyonic_not(reg, 0);
    anyonic_hadamard(reg, 0);
    anyonic_T_gate(reg, 0, 1e-6);

    // Cleanup
    anyonic_register_free(reg);
    anyon_system_free(fib);

    return 0;
}
```

## See Also

- [Topological Computing Algorithm](../../algorithms/topological-computing.md) - Theory and usage guide
- [Skyrmion Braiding API](skyrmion-braiding.md) - Skyrmion-based topological qubits
- [Tensor Network API](tensor-network.md) - MPS and MPO operations
