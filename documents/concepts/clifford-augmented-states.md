# Clifford-augmented quantum states

A short conceptual primer on the hybrid `|psi> = D|phi>` state
representation used by Moonlab's CA-MPS engine.

## Motivation

A quantum state on `n` qubits has up to `2^n` complex amplitudes.
Two well-known compression techniques each capture a different slice:

- **Stabilizer states** (Aaronson-Gottesman tableau): every state
  reachable from `|0^n>` by Clifford gates.  Stored in `O(n^2)` bits.
  Compatible with arbitrary Clifford evolution.  Cannot represent
  any non-Clifford content.
- **Matrix-product states** (MPS) with bond dimension `chi`: every
  state with bounded entanglement between left and right halves.
  Stored in `O(n * chi^2)` complex numbers.  Compatible with
  arbitrary gate evolution (with cost depending on entanglement
  growth).

Stabilizer-rich states often have **a lot** of entanglement (a
GHZ state at `n = 100` is maximally entangled across many cuts) but
zero **non-Clifford** complexity.  Plain MPS handles them poorly: the
GHZ state at `n = 100` has bond dimension 2 across the central cut
yet still requires the full `O(n * 4)` MPS data.  Pure stabilizer
methods handle them perfectly but fail the moment a single non-
Clifford rotation enters.

## The hybrid: factor out the Clifford layer

Write any state as `|psi> = D|phi>` where:

- `D` is a Clifford unitary, stored as an Aaronson-Gottesman tableau.
- `|phi>` is the **non-Clifford residual** as an MPS.

A Clifford gate `U` updates only the tableau: `D := U * D`.  No MPS
work.  This is **free** in MPS bond dimension.

A non-Clifford gate is parametrised as `exp(-i theta P/2)` for a
Pauli string `P`.  Clifford conjugation maps any Pauli to another
Pauli, so we compute `P' = D^dagger P D` (in `O(n^2)` tableau ops)
and then apply `exp(-i theta P'/2)` to `|phi>` as a Pauli-rotation
MPO.

For circuits with high Clifford fraction, `|phi>`'s bond dimension
tracks **only** the non-Clifford content, regardless of the total
state entanglement.  GHZ at `n = 100`: `D` is the GHZ-encoder
Clifford (`O(n)` bits), `|phi> = |0^n>` has bond dimension 1.

## Variational-D (var-D)

CA-MPS works in the other direction too: when looking for the ground
state of a Hamiltonian `H`, search over `D` to **minimise** the
entanglement that `|phi>` has to carry.  The ground state itself
becomes `|psi_GS> = D_*|phi_*>` for some optimal Clifford `D_*` plus
low-entanglement residual `|phi_*>`.

Concretely, alternate:

1. **D-update**: greedy local Clifford search to reduce the
   variational energy `<psi|H|psi> = <phi|D^dagger H D|phi>` at fixed
   `|phi>`.
2. **|phi>-update**: imag-time Trotter sweep on the Hamiltonian-
   under-the-current-`D` (equivalently, conjugating each Pauli term
   of `H` through `D^dagger ... D` and applying it to `|phi>`).

For TFIM, this finds the Kramers-Wannier-dual basis automatically:
`D = H_all + CNOT_chain` reduces the bipartite half-cut entropy of
the TFIM critical-point ground state by `5x to 50x` across the phase
diagram.

## Gauge-aware warmstart

Some Hamiltonians have a known stabilizer structure (lattice gauge
theory Gauss-law operators, surface code stabilizers, repetition
code, abelian symmetry projectors).  In those cases, an **explicit**
Clifford `D_S` can be built by symplectic Gauss-Jordan elimination
on the Pauli tableau of the generators, such that `D_S|0^n>` already
sits inside the desired physical subspace.  var-D then only has to
capture the residual non-stabilizer dynamics on top.

## When CA-MPS helps

CA-MPS provides the largest speedups when:

- A **large fraction** of the gates is Clifford (random Clifford
  circuits, stabilizer-rich quantum-error-correction simulations,
  variational ansaetze with mostly entangling Cliffords + a few
  parametrised rotations).
- The target ground state has known **abelian stabilizer
  structure** (lattice gauge theory, surface/toric/repetition
  codes, fixed-charge / fixed-Sz sectors).
- The system size makes plain MPS bond-dim grow but the
  non-Clifford complexity stays bounded.

CA-MPS provides little advantage when the circuit is dominated by
non-Clifford rotations (deep variational ansaetze with parameter
counts comparable to qubit count, generic chemistry circuits without
obvious stabilizer structure).

## See also

- `documents/concepts/tensor-networks.md` -- MPS background.
- `documents/algorithms/ca-mps-var-d.md` -- algorithm reference.
- `documents/architecture/ca-mps-engine.md` -- engine architecture.
- `MATH.md` §10-12 -- math foundations.
