# Clifford-Assisted MPS (CA-MPS), variational-D, and the gauge-aware warmstart

`@since` v0.2.1.  Source: `src/algorithms/tensor_network/ca_mps*.{c,h}`,
`src/applications/hep/lattice_z2_1d.{c,h}`.

This page covers three interlocking primitives that ship together in
v0.2.1:

1. **CA-MPS** -- a hybrid `|psi> = D|phi>` representation that splits
   a quantum state into a Clifford prefactor `D` (an Aaronson-Gottesman
   tableau) and an MPS factor `|phi>`.  Headline benchmark: 64x
   bond-dim advantage + 13884x speedup vs plain MPS at n=12 on a
   stabilizer-rich circuit.
2. **Variational-D (var-D)** -- alternating ground-state search that
   updates `D` greedily in the Clifford basis and `|phi>` via
   imag-time Trotter sweep.  Validated on TFIM, XXZ Heisenberg, and
   kagome 12-site frustrated AFM.
3. **Gauge-aware stabilizer-subgroup warmstart** -- a symplectic
   Gauss-Jordan Clifford builder that takes any list of commuting
   Pauli generators and emits an O(n^2)-gate Clifford circuit that
   places `D|0^n>` in their simultaneous +1 eigenspace.  First HEP
   application: 1+1D Z2 lattice gauge theory.

Math foundations are in `MATH.md` §10-12; design write-ups in
`docs/research/ca_mps.md` and
`docs/research/var_d_lattice_gauge_theory.md`.

## CA-MPS hybrid representation

For an n-qubit state |psi>, store the pair
`(D : Clifford circuit on n qubits, |phi> : MPS on n qubits)`
with the contract that `|psi> = D|phi>`.

Clifford gates `U` update only `D`: `D := U * D`.  Cost is the AG
tableau cost (O(n) for 1-qubit, O(n) for 2-qubit), no MPS work.

Non-Clifford gates of the form `exp(-i theta P/2)` for a Pauli string
`P` are applied via tableau-side conjugation `P' = D^dagger P D`,
followed by a Pauli-rotation MPO applied to `|phi>`:

```c
moonlab_ca_mps_t* s = moonlab_ca_mps_create(n, /*max_bond=*/32);

// Clifford layer: free in MPS cost.
for (uint32_t q = 0; q < n; q++) moonlab_ca_mps_h(s, q);
moonlab_ca_mps_cnot(s, 0, 1);

// Non-Clifford rotation: pushed into the MPS.
moonlab_ca_mps_rz(s, 3, 0.7);

// Imag-time evolution under a Pauli string (non-unitary).
uint8_t P[12] = {3,3, 0,0,0,0,0,0,0,0,0,0};   // Z_0 Z_1
moonlab_ca_mps_imag_pauli_rotation(s, P, 0.01);
moonlab_ca_mps_normalize(s);

// Expectation values.
double _Complex e;
moonlab_ca_mps_expect_pauli(s, P, &e);

moonlab_ca_mps_free(s);
```

## Variational-D (var-D)

Given a Pauli-sum Hamiltonian `H = sum_k c_k P_k`, var-D searches
`(D, |phi>)` to minimise

```
E(D, |phi>) = <psi|H|psi>
            = sum_k c_k <phi| (D^dagger P_k D) |phi>
```

The optimisation alternates two updates:

1. **D-update**: at fixed `|phi>`, enumerate single- and two-qubit
   Clifford candidates `g in {H, S, S^dagger, CNOT, CZ, SWAP}`.  For
   each candidate compute the energy delta and accept the largest
   decrease.  Optional 2-gate composite moves help escape 1-gate
   local minima (e.g. the Kramers-Wannier dual basis transform on
   TFIM, which requires `H_all + CNOT-chain` and has no productive
   single-gate descent direction from `D = I`).

2. **|phi>-update**: first-order Trotter sweep over the Pauli sum,
   each `exp(-dtau * c_k * P_k)` applied via the imag-time
   Pauli-rotation MPO with intermediate renormalisation every 4
   terms (without renormalisation, the MPS norm drifts and the
   variational energy estimate falls below the true ground state).

The alternating loop is monotone in E at each accepted gate and at
each Trotter cycle, so it converges; the fixed point need not be the
global ground state because the greedy gate search has stationary
points wherever no Clifford reduces the energy at fixed `|phi>`.

Warmstart options seed `D` in productive Clifford basins:

| Code | Name | What it builds |
|------|------|----------------|
| 0 | `IDENTITY` | `D = I`. |
| 1 | `H_ALL` | `H` on every qubit. |
| 2 | `DUAL_TFIM` | `H_ALL` then a CNOT chain.  TFIM's Kramers-Wannier dual basis. |
| 3 | `FERRO_TFIM` | `H` on qubit 0 then a CNOT chain.  Cat-state encoder; needed in TFIM's deep ferro regime. |
| 4 | `STABILIZER_SUBGROUP` | Gauge-aware: see below. |

```c
ca_mps_var_d_alt_config_t cfg = ca_mps_var_d_alt_config_default();
cfg.warmstart = CA_MPS_WARMSTART_DUAL_TFIM;
cfg.composite_2gate = 1;
cfg.max_outer_iters = 25;

ca_mps_var_d_alt_result_t res = {0};
moonlab_ca_mps_optimize_var_d_alternating(
    state, paulis, coeffs, num_terms, &cfg, &res);
```

## Gauge-aware warmstart

For Hamiltonians whose physical sector is the simultaneous +1
eigenspace of a commuting set of Pauli generators (lattice gauge
theory Gauss-law operators, surface/toric/repetition-code stabilizers,
any abelian symmetry projector), the warmstart Clifford `D_S` can be
built exactly via Aaronson-Gottesman symplectic Gauss-Jordan
elimination on the Pauli tableau.

Encode each generator as a row of an F2 tableau `[X | Z | r]` of
shape `(k, 2n+1)`.  Three operations:

- **Row swap** (free, just reorders the generating set).
- **Row XOR** (free, multiplies one generator into another).
- **Column ops** (apply a Clifford gate, mutate the column on every
  row): `H` swaps the (x, z) pair; `S` maps `(x, z) -> (x, x XOR z)`;
  `CNOT(c, t)` maps `x_t ^= x_c`, `z_c ^= z_t`.  Phase bit `r` updates
  per the standard AG conjugation rules.

The reduction loop pivots row-by-row on a fresh qubit, rotating each
non-trivial entry to X via H/S, clearing the row's other qubits via
CNOT, eliminating the column from other rows by row XOR, and finally
rotating the pivot from X back to Z.  After k pivots the tableau is
canonical: row p is +/- Z on a distinct qubit.

The state `|b>` with `b_{q_p} = r_p` (negative-phase bits) is in the
simultaneous +1 eigenspace of every transformed generator.  Applying
the inverse-and-reversed Clifford circuit to `|b>` then yields a
state in the simultaneous +1 eigenspace of the **original**
generators.

Cost: O(n^2) Clifford gates, O(k * n^2) tableau ops.  See
`MATH.md` §12 for the full proof and
`src/algorithms/tensor_network/ca_mps_var_d_stab_warmstart.c` for the
implementation.

```c
#include "algorithms/tensor_network/ca_mps_var_d_stab_warmstart.h"

// Generators g_0, ..., g_{k-1} laid out (k, n) row-major in
// Moonlab's Pauli-byte encoding (0=I, 1=X, 2=Y, 3=Z), pairwise
// commuting and independent.
moonlab_ca_mps_apply_stab_subgroup_warmstart(state, generators, k);
```

## First HEP application: 1+1D Z2 lattice gauge theory

The Schwinger-style 1D chain with N matter sites + (N - 1) link
qubits (total `2N - 1`).  Hamiltonian (after JW with parallel
transport, with exactly gauge-invariant kinetic terms):

```
H = -(t/2) sum_x [ X_{2x} Y_{2x+1} Y_{2x+2}
                 - Y_{2x} Y_{2x+1} X_{2x+2} ]   (matter + gauge link)
    - h sum_x Z_{2x+1}                          (electric field)
    + (m/2) sum_x (-1)^x Z_{2x}                 (staggered mass)
    + lambda sum_{x=1..N-2} (I - G_x)            (Gauss-law penalty)
```

with the interior Gauss-law operator
`G_x = X_{2x-1} Z_{2x} X_{2x+1}`.  Each kinetic-term Pauli string
commutes term-by-term with every `G_y`, so `H` preserves the gauge
sector exactly under any unitary or imag-time evolution.

```c
#include "applications/hep/lattice_z2_1d.h"

z2_lgt_config_t cfg = {.num_matter_sites = 4, .t_hop = 1.0,
                       .h_link = 0.5, .mass = 0.0,
                       .gauss_penalty = 0.0};
uint8_t* paulis;
double* coeffs;
uint32_t T, nq;
z2_lgt_1d_build_pauli_sum(&cfg, &paulis, &coeffs, &T, &nq);

// Stack the interior Gauss-law operators as the warmstart subgroup.
const uint32_t k = cfg.num_matter_sites - 2;
uint8_t* gens = calloc((size_t)k * nq, 1);
for (uint32_t i = 0; i < k; i++) {
    z2_lgt_1d_gauss_law_pauli(&cfg, i + 1, &gens[(size_t)i * nq]);
}

// var-D with the gauge-aware warmstart.
moonlab_ca_mps_t* s = moonlab_ca_mps_create(nq, 32);
ca_mps_var_d_alt_config_t vcfg = ca_mps_var_d_alt_config_default();
vcfg.warmstart                = CA_MPS_WARMSTART_STABILIZER_SUBGROUP;
vcfg.warmstart_stab_paulis    = gens;
vcfg.warmstart_stab_num_gens  = k;
ca_mps_var_d_alt_result_t res = {0};
moonlab_ca_mps_optimize_var_d_alternating(
    s, paulis, coeffs, T, &vcfg, &res);
```

## Bindings

The full v0.2.1 ABI is exposed via:

- **C** -- `src/applications/moonlab_export.h`:
  `moonlab_ca_mps_var_d_run`, `moonlab_ca_mps_gauge_warmstart`,
  `moonlab_z2_lgt_1d_build`, `moonlab_z2_lgt_1d_gauss_law`,
  `moonlab_status_string`.
- **Python** -- `bindings/python/moonlab/ca_mps.py`: class `CAMPS`
  plus `var_d_run`, `gauge_warmstart`, `z2_lgt_1d_build`,
  `z2_lgt_1d_gauss_law`, `status_string`.
- **Rust** -- `bindings/rust/moonlab/src/ca_mps.rs`: `CaMps`,
  `VarDConfig`, `Warmstart`, `var_d_run`, `gauge_warmstart`,
  `z2_lgt_1d_build`, `z2_lgt_1d_gauss_law`, `status_string`.
- **JavaScript / WASM** -- `bindings/javascript/packages/core/src/ca-mps.ts`:
  class `CaMps` plus `varDRun`, `gaugeWarmstart`, `z2Lgt1dBuild`,
  `z2Lgt1dGaussLaw`, `statusString`.

## Validation

- `tests/unit/test_ca_mps_var_d.c`,
  `tests/unit/test_ca_mps_var_d_alt.c`,
  `tests/unit/test_ca_mps_var_d_composite.c` -- core var-D unit tests.
- `tests/unit/test_gauge_warmstart.c` -- pins +1 eigenvalue on Bell,
  GHZ-3, four interior Z2 LGT Gauss-law operators of N=4, plus
  rejection of anti-commuting input.
- `tests/unit/test_z2_lgt_pauli_sum.c` -- term-by-term commutativity
  of the kinetic terms with the Gauss-law operators.
- `tests/abi/test_moonlab_export_abi.c` -- dlsym-finds and
  behaviourally exercises every v0.2.1 ABI entry point.
- `bindings/python/tests/test_ca_mps.py` -- six Python smoke +
  correctness tests.

## See also

- Math: `MATH.md` §10-12.
- Design: `docs/research/ca_mps.md`.
- Z2 LGT: `docs/research/var_d_lattice_gauge_theory.md`.
- Examples: `examples/tensor_network/ca_mps_var_d_*.c`,
  `examples/hep/z2_gauge_var_d.c`.
