# Clifford-Assisted MPS (CA-MPS) C API

Complete reference for the CA-MPS hybrid `|psi> = D|phi>` representation,
variational-D ground-state search, and gauge-aware stabilizer-subgroup
warmstart.  All entry points listed here are committed on the stable
ABI surface (`src/applications/moonlab_export.h`) **since v0.2.1**.

**Headers**:
- `src/algorithms/tensor_network/ca_mps.h` -- state handle + gates.
- `src/algorithms/tensor_network/ca_mps_var_d.h` -- variational-D
  ground-state search.
- `src/algorithms/tensor_network/ca_mps_var_d_stab_warmstart.h` --
  gauge-aware Clifford preparation.
- `src/applications/hep/lattice_z2_1d.h` -- 1+1D Z2 LGT Pauli-sum
  builder.
- `src/utils/moonlab_status.h` -- status code registry.

## Overview

CA-MPS splits an n-qubit state into:

- A **Clifford prefactor `D`** stored as an Aaronson-Gottesman tableau.
  Clifford gates update only the tableau (`O(n)` bit ops) and
  contribute zero MPS cost.
- An **MPS factor `|phi>`** with bond-dimension cap `chi`.  Non-Clifford
  rotations of the form `exp(-i theta P/2)` for a Pauli string `P` are
  conjugated through `D` (yielding `D^dagger P D`) and applied to
  `|phi>` as a Pauli-rotation MPO.

For stabilizer-rich circuits (high Clifford fraction), CA-MPS keeps
`|phi>` at very low bond dimension regardless of the qubit count.
Headline benchmark at n=12: 64x bond-dim advantage and 13884x speedup
vs plain MPS on a random Clifford circuit (`bench_ca_mps`).

## State handle

```c
typedef struct moonlab_ca_mps_t moonlab_ca_mps_t;

moonlab_ca_mps_t* moonlab_ca_mps_create(uint32_t num_qubits,
                                          uint32_t max_bond_dim);
void              moonlab_ca_mps_free  (moonlab_ca_mps_t* s);
moonlab_ca_mps_t* moonlab_ca_mps_clone (const moonlab_ca_mps_t* s);

uint32_t moonlab_ca_mps_num_qubits        (const moonlab_ca_mps_t* s);
uint32_t moonlab_ca_mps_max_bond_dim      (const moonlab_ca_mps_t* s);
uint32_t moonlab_ca_mps_current_bond_dim  (const moonlab_ca_mps_t* s);
```

`moonlab_ca_mps_create` returns NULL on bad arguments or allocation
failure.  `moonlab_ca_mps_free` is no-op on NULL.

## Clifford gates (tableau-only, no MPS cost)

```c
int moonlab_ca_mps_h    (moonlab_ca_mps_t* s, uint32_t q);
int moonlab_ca_mps_s    (moonlab_ca_mps_t* s, uint32_t q);
int moonlab_ca_mps_sdag (moonlab_ca_mps_t* s, uint32_t q);
int moonlab_ca_mps_x    (moonlab_ca_mps_t* s, uint32_t q);
int moonlab_ca_mps_y    (moonlab_ca_mps_t* s, uint32_t q);
int moonlab_ca_mps_z    (moonlab_ca_mps_t* s, uint32_t q);

int moonlab_ca_mps_cnot (moonlab_ca_mps_t* s, uint32_t ctrl, uint32_t targ);
int moonlab_ca_mps_cz   (moonlab_ca_mps_t* s, uint32_t a,    uint32_t b);
int moonlab_ca_mps_swap (moonlab_ca_mps_t* s, uint32_t a,    uint32_t b);
```

## Non-Clifford rotations (Pauli-string MPO into `|phi>`)

```c
int moonlab_ca_mps_rx        (moonlab_ca_mps_t* s, uint32_t q, double theta);
int moonlab_ca_mps_ry        (moonlab_ca_mps_t* s, uint32_t q, double theta);
int moonlab_ca_mps_rz        (moonlab_ca_mps_t* s, uint32_t q, double theta);
int moonlab_ca_mps_t_gate    (moonlab_ca_mps_t* s, uint32_t q);
int moonlab_ca_mps_t_dagger  (moonlab_ca_mps_t* s, uint32_t q);
int moonlab_ca_mps_phase     (moonlab_ca_mps_t* s, uint32_t q, double theta);
int moonlab_ca_mps_pauli_rotation     (moonlab_ca_mps_t* s,
                                         const uint8_t* pauli,
                                         double theta);
int moonlab_ca_mps_imag_pauli_rotation(moonlab_ca_mps_t* s,
                                         const uint8_t* pauli,
                                         double tau);
```

Pauli strings are flat `uint8_t` arrays of length `num_qubits` with
the byte encoding `0=I, 1=X, 2=Y, 3=Z`.

`moonlab_ca_mps_imag_pauli_rotation` is non-unitary; pair with
`moonlab_ca_mps_normalize` between Trotter cycles.

## Norm + observables

```c
int    moonlab_ca_mps_normalize (moonlab_ca_mps_t* s);
double moonlab_ca_mps_norm      (const moonlab_ca_mps_t* s);

int moonlab_ca_mps_expect_pauli     (const moonlab_ca_mps_t* s,
                                       const uint8_t* pauli,
                                       double _Complex* out);
int moonlab_ca_mps_expect_pauli_sum (const moonlab_ca_mps_t* s,
                                       const uint8_t* paulis,
                                       const double _Complex* coeffs,
                                       uint32_t num_terms,
                                       double _Complex* out);
int moonlab_ca_mps_prob_z           (const moonlab_ca_mps_t* s,
                                       uint32_t qubit,
                                       double* out_prob);
```

## Variational-D ground-state search

```c
int moonlab_ca_mps_var_d_run(moonlab_ca_mps_t* state,
                              const uint8_t*  paulis,         /* (T, n)   */
                              const double*   coeffs,         /* (T,)     */
                              uint32_t        num_terms,
                              uint32_t        max_outer_iters,
                              double          imag_time_dtau,
                              uint32_t        imag_time_steps_per_outer,
                              uint32_t        clifford_passes_per_outer,
                              int             composite_2gate,
                              int             warmstart,
                              const uint8_t*  stab_paulis,    /* (k, n) or NULL */
                              uint32_t        stab_num_gens,
                              double*         out_final_energy);
```

`warmstart` is one of:

| Code | Name |
|------|------|
| `0` | `IDENTITY` |
| `1` | `H_ALL` |
| `2` | `DUAL_TFIM` (`H_ALL` + CNOT chain) |
| `3` | `FERRO_TFIM` (cat-state encoder) |
| `4` | `STABILIZER_SUBGROUP` (gauge-aware; provide `stab_paulis`) |

When `warmstart == 4`, `stab_paulis` is a row-major `(stab_num_gens,
num_qubits)` array of pairwise-commuting Pauli generators; otherwise
pass `NULL` and `stab_num_gens = 0`.

## Gauge-aware stabilizer-subgroup warmstart

```c
int moonlab_ca_mps_gauge_warmstart(moonlab_ca_mps_t* state,
                                     const uint8_t*  paulis,
                                     uint32_t        num_gens);
```

Standalone use (no var-D loop): builds and applies the
Aaronson-Gottesman symplectic-Gauss-Jordan Clifford that places
`state->D|0^n>` in the simultaneous +1 eigenspace of every supplied
generator.  Returns `CA_MPS_ERR_INVALID = -1` if the generators don't
pairwise commute or aren't independent.

## 1+1D Z2 lattice gauge theory

```c
int moonlab_z2_lgt_1d_build(uint32_t num_matter_sites,
                              double t_hop, double h_link,
                              double mass, double gauss_penalty,
                              uint8_t**  out_paulis,
                              double**   out_coeffs,
                              uint32_t*  out_num_terms,
                              uint32_t*  out_num_qubits);

int moonlab_z2_lgt_1d_gauss_law(uint32_t num_matter_sites,
                                  uint32_t site_x,
                                  uint8_t* out_pauli);
```

`moonlab_z2_lgt_1d_build` allocates `*out_paulis` and `*out_coeffs`;
caller frees both via `free()`.  Output qubit count is
`2 * num_matter_sites - 1`.

`moonlab_z2_lgt_1d_gauss_law` writes the interior Gauss-law operator
`G_x = X_{2x-1} Z_{2x} X_{2x+1}` for `1 <= site_x <= N - 2` into
`out_pauli` (caller-provided buffer of length `2 * N - 1`).

## Status code registry

```c
typedef int moonlab_status_t;

#define MOONLAB_STATUS_SUCCESS          ( 0)
#define MOONLAB_STATUS_ERR_INVALID      (-1)
#define MOONLAB_STATUS_ERR_QUBIT        (-2)
#define MOONLAB_STATUS_ERR_OOM          (-3)
#define MOONLAB_STATUS_ERR_BACKEND      (-4)
#define MOONLAB_STATUS_ERR_MODULE_BASE  (-100)

const char* moonlab_status_string(int module, int status);
```

`module` enumerates the per-module status namespace (`0=GENERIC,
1=CA_MPS, 2=CA_MPS_VAR_D, 3=CA_MPS_STAB_WARMSTART, 4=CA_PEPS,
5=TN_STATE, 6=TN_GATE, 7=TN_MEASURE, 8=TENSOR, 9=CONTRACT,
10=SVD_COMPRESS, 11=CLIFFORD, 12=PARTITION, 13=DIST_GATE,
14=MPI_BRIDGE`).  Returns a static string for canonical codes; never
NULL.

## Example: gauge-aware warmstart on 1+1D Z2 LGT

```c
#include "applications/moonlab_export.h"
#include "applications/hep/lattice_z2_1d.h"

z2_lgt_config_t cfg = {.num_matter_sites = 4, .t_hop = 1.0,
                       .h_link = 0.5, .mass = 0.0,
                       .gauss_penalty = 0.0};

uint8_t* paulis = NULL;
double*  coeffs = NULL;
uint32_t T = 0, nq = 0;
moonlab_z2_lgt_1d_build(4, 1.0, 0.5, 0.0, 0.0,
                          &paulis, &coeffs, &T, &nq);

uint32_t k = cfg.num_matter_sites - 2;
uint8_t* gens = calloc((size_t)k * nq, 1);
for (uint32_t i = 0; i < k; i++) {
    moonlab_z2_lgt_1d_gauss_law(4, i + 1, &gens[(size_t)i * nq]);
}

moonlab_ca_mps_t* s = moonlab_ca_mps_create(nq, 32);
double E = 0.0;
moonlab_ca_mps_var_d_run(s, paulis, coeffs, T,
                           /*max_outer=*/25, /*dtau=*/0.10,
                           /*imag_steps=*/4, /*clifford_passes=*/8,
                           /*composite_2gate=*/1,
                           /*warmstart=*/4,         /* STABILIZER_SUBGROUP */
                           gens, k, &E);

moonlab_ca_mps_free(s);
free(gens);
free(paulis);
free(coeffs);
```

## See also

- `documents/algorithms/ca-mps-var-d.md` -- algorithmic walk-through.
- `MATH.md` §10-12 -- math foundations.
- `docs/research/ca_mps.md` -- full design + benchmarks.
- `docs/research/var_d_lattice_gauge_theory.md` -- Z2 LGT theorem +
  open directions.
- `documents/api/python/ca_mps.md` -- Python bindings.
