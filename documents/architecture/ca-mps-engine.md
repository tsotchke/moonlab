# CA-MPS Engine

The Clifford-Assisted MPS (CA-MPS) engine implements the hybrid
`|psi> = D|phi>` state representation, the variational-D
ground-state search, and the gauge-aware stabilizer-subgroup
warmstart.  It is the headline tensor-network primitive that ships
in **v0.2.1**.

## Why hybrid?

The Aaronson-Gottesman tableau and the matrix-product-state machinery
each handle a different slice of quantum-state structure efficiently:

- **Tableau**: stabilizer states (and Clifford evolutions on them) at
  `O(n)` bit-ops per gate.  Cannot represent non-Clifford content.
- **MPS**: arbitrary states, but bond dimension grows with the
  amount of **non-Clifford** entanglement.  Stabilizer-rich states
  blow up bond dimension unnecessarily.

CA-MPS factors any state as `|psi> = D|phi>` where `D` is a Clifford
unitary stored in the tableau and `|phi>` is the residual MPS.
Clifford gates absorb into `D` (free in MPS cost).  Non-Clifford
gates of the form `exp(-i theta P/2)` for a Pauli string `P` are
conjugated through the tableau (`P' = D^dagger P D` in `O(n^2)` bit
ops) and applied to `|phi>` as a Pauli-rotation MPO.

For circuits with high Clifford fraction, `|phi>`'s bond dimension
stays at the **non-Clifford weight** of the circuit — which can be
1 even on stabilizer states with maximal entanglement.

Headline benchmark at `n = 12`: **64x bond-dim advantage and 13884x
speedup** over plain MPS on a random Clifford circuit
(`tests/performance/bench_ca_mps.c`).

## Module layout

```
src/algorithms/tensor_network/
  ca_mps.{c,h}                       -- state handle + gate surface
  ca_mps_var_d.{c,h}                 -- alternating ground-state search
  ca_mps_var_d_stab_warmstart.{c,h}  -- gauge-aware Clifford prep
  ca_peps.{c,h}                      -- 2D scaffold (gates in v0.3)

src/applications/hep/
  lattice_z2_1d.{c,h}                -- 1+1D Z2 LGT Pauli sum

src/utils/
  moonlab_status.{c,h}               -- centralised status registry
```

The Clifford backend lives in `src/backends/clifford/` and is reused
unchanged from earlier releases.

## Variational-D ground-state search

Given a Pauli-sum Hamiltonian `H = sum_k c_k P_k`, var-D minimises
the variational energy
`E(D, |phi>) = sum_k c_k <phi| D^dagger P_k D |phi>` by alternating
two updates:

1. **D-update** (greedy local Clifford): enumerate one-qubit
   `{H, S, S^dagger}` and two-qubit `{CNOT, CZ, SWAP}` candidates;
   accept the largest energy decrease.  Optional 2-gate composite
   moves help escape one-gate local minima where the right descent
   direction requires two consecutive accepts that look bad in
   step 1 (TFIM critical point, kagome AFM).

2. **|phi>-update** (imag-time Trotter): apply
   `exp(-dtau * c_k * P_k)` to `|phi>` for each Pauli term, with
   intermediate renormalisation every 4 terms.

The alternating loop is monotone in E and converges; the fixed point
is local, not necessarily global.  Empirical performance:

- TFIM at criticality: matches DMRG to `< 1e-3` relative energy at
  `n = 12` with `DUAL_TFIM` warmstart.
- XXZ Heisenberg gapless regime: stalls at a ferromagnetic-cluster
  local minimum at `n = 10` without `FERRO_TFIM` warmstart.
- Kagome 12-site frustrated AFM: matches Läuchli et al. PRB 83,
  212401 (2011) Table I cluster "12" reference to `< 1e-2`.

## Gauge-aware warmstart (Aaronson-Gottesman symplectic Gauss-Jordan)

For Hamiltonians whose physical sector is the simultaneous +1
eigenspace of a commuting set of Pauli generators
`{g_0, ..., g_{k-1}}`, the warmstart Clifford `D_S` is built by
direct symplectic Gauss-Jordan elimination on the Pauli tableau:

1. Encode each generator as a row of an F2 tableau `[X | Z | r]`.
2. Pivot row-by-row on a fresh qubit; rotate non-trivial entries to
   X via `H` / `S`; clear the row's other qubits via `CNOT`;
   eliminate the column from other rows by row XOR; rotate the
   pivot from X back to Z.
3. After `k` pivots the tableau is canonical (each row is `+/- Z`
   on a distinct qubit).  The state `|b>` with `b_{q_p} = r_p` is
   in the simultaneous +1 eigenspace of the transformed generators.

Apply the inverse-and-reversed Clifford circuit to `|b>` to recover
a state in the +1 eigenspace of the **original** generators.

Cost: `O(n^2)` Clifford gates emitted, `O(k * n^2)` tableau ops.
Generalises across LGT Gauss-law operators, surface/toric/repetition
codes, and any abelian symmetry projector.

## 1+1D Z2 lattice gauge theory

The Schwinger-style chain on `2N - 1` qubits (N matter + (N - 1)
links).  Hamiltonian uses **exactly gauge-invariant** kinetic terms
(`X_{2x} Y_{2x+1} Y_{2x+2}` and `Y_{2x} Y_{2x+1} X_{2x+2}`) — each
piece commutes with every interior Gauss-law operator
`G_x = X_{2x-1} Z_{2x} X_{2x+1}` term-by-term, so `H` preserves the
gauge sector exactly under any unitary or imag-time evolution.

The lambda penalty `lambda * sum_x (I - G_x)` is therefore redundant
for physics inside the +1 sector; it remains in the API for
compatibility with prior releases that needed it for energetic
projection.

## Stable ABI

Five new entry points join `src/applications/moonlab_export.h` since
v0.2.1, all dlsym-pinned by `tests/abi/test_moonlab_export_abi.c`:

- `moonlab_ca_mps_var_d_run`
- `moonlab_ca_mps_gauge_warmstart`
- `moonlab_z2_lgt_1d_build`
- `moonlab_z2_lgt_1d_gauss_law`
- `moonlab_status_string`

## See also

- `tensor-network-engine.md` -- the underlying MPS / DMRG machinery.
- `documents/algorithms/ca-mps-var-d.md` -- algorithmic walk-through.
- `documents/api/c/ca-mps.md` -- C API reference.
- `documents/api/python/ca_mps.md` -- Python bindings.
- `MATH.md` §10-12 -- math foundations.
- `docs/research/ca_mps.md` -- full design + benchmarks.
- `docs/research/var_d_lattice_gauge_theory.md` -- Z2 LGT theorem.
