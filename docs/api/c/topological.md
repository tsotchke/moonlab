# Topological Computing API

C API for topological quantum computing simulation.

## Overview

The topological computing API (`topological.h`) provides functions for simulating topological quantum computing primitives including anyon models, braiding operations, fusion trees, surface codes, and toric codes.

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
| `k` | `uint32_t` | Level parameter (k=2 gives Ising, k=3 gives Fibonacci) |

**Returns**: SU(2)_k anyon system.

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

Braid two adjacent anyons.

```c
qs_error_t braid_anyons(fusion_tree_t *tree, uint32_t position, bool clockwise);
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `tree` | `fusion_tree_t*` | Fusion tree (modified in place) |
| `position` | `uint32_t` | Position of left anyon to braid |
| `clockwise` | `bool` | Direction (true = σ, false = σ⁻¹) |

**Returns**: `QS_SUCCESS` or error code.

#### `apply_F_move`

Apply F-move (basis change).

```c
qs_error_t apply_F_move(fusion_tree_t *tree, uint32_t vertex);
```

Changes fusion order: $(a \times b) \times c \leftrightarrow a \times (b \times c)$

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

## Anyonic Quantum Gates

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

Apply NOT gate via braiding.

```c
qs_error_t anyonic_not(anyonic_register_t *reg, uint32_t qubit);
```

#### `anyonic_hadamard`

Apply Hadamard-like gate via braiding.

```c
qs_error_t anyonic_hadamard(anyonic_register_t *reg, uint32_t qubit);
```

#### `anyonic_T_gate`

Apply T gate approximation via braiding.

```c
qs_error_t anyonic_T_gate(anyonic_register_t *reg, uint32_t qubit,
                          double precision);
```

#### `anyonic_entangle`

Apply two-qubit entangling gate.

```c
qs_error_t anyonic_entangle(anyonic_register_t *reg,
                            uint32_t qubit1, uint32_t qubit2);
```

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
