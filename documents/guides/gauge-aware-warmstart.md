# Gauge-aware warmstart guide

How to use the v0.2.1 stabilizer-subgroup warmstart to project a
CA-MPS state into the +1 eigenspace of any commuting set of Pauli
generators -- lattice gauge theory Gauss-law operators, surface /
toric / repetition code stabilizers, fixed-charge symmetry sectors.

## When you'd reach for this

Three common scenarios:

1. **Lattice gauge theory simulation.**  The physical sector is the
   simultaneous +1 eigenspace of the Gauss-law operators `G_x`.
   Naive variational methods leak out of this sector under imag-time
   evolution; the gauge-aware warmstart starts you exactly inside it.
2. **Quantum error correction code simulation.**  Surface / toric /
   repetition / colour codes all have abelian stabilizer subgroups.
   The warmstart prepares the code subspace.
3. **Symmetry projection.**  Any abelian commuting subgroup of the
   Pauli group defines a sector (e.g. fixed parity, fixed charge,
   fixed `Sz`).  The warmstart prepares an arbitrary state in that
   sector.

## How the warmstart builds the Clifford

Aaronson-Gottesman symplectic Gauss-Jordan elimination on the Pauli
tableau:

- Encode each generator as a row of an F2 tableau `[X | Z | r]`.
- Pivot row-by-row on a fresh qubit; rotate non-trivial entries to X
  via `H` / `S`; clear the row's other qubits via `CNOT`; eliminate
  the column from other rows by row XOR; rotate the pivot from X back
  to Z.
- After `k` pivots the tableau is canonical: row `p` is `+/- Z` on
  a distinct qubit `q_p`.  The state `|b>` with `b_{q_p} = r_p` is in
  the simultaneous +1 eigenspace of the transformed generators.
- The inverse-and-reversed Clifford circuit applied to `|b>` recovers
  a state in the +1 eigenspace of the **original** generators.

Cost: `O(n^2)` Clifford gates emitted, `O(k * n^2)` tableau ops.  The
emitted gates are absorbed into the CA-MPS Clifford prefactor `D` so
they cost nothing in MPS bond dimension.

## Calling it from C

```c
#include "applications/moonlab_export.h"

moonlab_ca_mps_t* state = moonlab_ca_mps_create(num_qubits, 32);

/* Generators: row-major (k, num_qubits) bytes, encoding
 *   0=I, 1=X, 2=Y, 3=Z.
 * Must be pairwise commuting and linearly independent. */
const uint8_t generators[/* k * num_qubits */] = { /* ... */ };

int rc = moonlab_ca_mps_gauge_warmstart(state, generators, k);
if (rc != 0) {
    fprintf(stderr, "warmstart failed: %s\n",
            moonlab_status_string(/*module=CA_MPS_STAB_WARMSTART*/3, rc));
    /* Common failure: rc == -1 (CA_MPS_ERR_INVALID), meaning the
     * generators don't pairwise commute or aren't independent. */
}

/* state->D|0^n> is now in the +1 eigenspace of every generator. */
```

## Calling it from Python

```python
import numpy as np
from moonlab import CAMPS, gauge_warmstart

# GHZ-3 stabilizers: {XXX, ZZI, IZZ}.
generators = np.array([
    [1, 1, 1],   # X X X
    [3, 3, 0],   # Z Z I
    [0, 3, 3],   # I Z Z
], dtype=np.uint8)

state = CAMPS(num_qubits=3, max_bond_dim=8)
gauge_warmstart(state, generators)
# state is now the GHZ-3 state (|000> + |111>) / sqrt(2).
```

## Calling it from Rust

```rust
use moonlab::{CaMps, gauge_warmstart};

// Bell-pair stabilizers {XX, ZZ}.
let generators: [u8; 4] = [
    1, 1,    // X X
    3, 3,    // Z Z
];

let mut state = CaMps::new(2, 8)?;
gauge_warmstart(&mut state, &generators, /*num_gens=*/2)?;
// state.D|00> is the Bell state.
```

## Calling it from JS / WASM

```typescript
import { CaMps, gaugeWarmstart } from '@moonlab/quantum-core';

const generators = new Uint8Array([
  1, 1,    // X X
  3, 3,    // Z Z
]);

const state = await CaMps.create(2, 8);
gaugeWarmstart(state, generators, /*numGens=*/2);
// state.norm === 1.0
state.dispose();
```

## Diagnosing failures

The most common rejection: the supplied generators don't pairwise
commute.  Check this directly with the symplectic-form parity test:
two Pauli strings `P` and `Q` commute iff
`sum_q (x_P[q] & z_Q[q]) XOR (z_P[q] & x_Q[q])` is even.

The second-most-common rejection: linear dependence.  If two
generators `g_i` and `g_j` differ by a scalar (e.g. `g_j = -g_i`),
or if `g_k = g_i * g_j` for some pair, the symplectic Gauss-Jordan
will fail to find an unused pivot qubit on row `k` and return
`CA_MPS_ERR_INVALID`.

The error path is unit-tested by case 5 of
`tests/unit/test_gauge_warmstart.c` (anti-commuting `{X_0, Z_0}` ->
`CA_MPS_ERR_INVALID`).

## What the warmstart does NOT do

It only prepares the **+1 eigenspace projection at the start**.  If
the Hamiltonian's terms don't commute with the generators, imag-time
evolution can drift the state out of the sector.

For 1+1D Z2 LGT this matters: until v0.2.1, the kinetic terms were
written in their bare Jordan-Wigner form (`X_{2x} X_{2x+1} X_{2x+2}`),
which anti-commutes with `G_x`.  v0.2.1 ships exactly gauge-invariant
kinetic terms (`X_{2x} Y_{2x+1} Y_{2x+2}` and `Y_{2x} Y_{2x+1} X_{2x+2}`)
that commute with every `G_y` term-by-term.  See
`docs/research/var_d_lattice_gauge_theory.md` for the algebra.

For a model whose Hamiltonian inherently doesn't commute with the
target stabilizers, you need **either** an exactly gauge-invariant
rewrite of `H` (preferred) **or** a gauge-aware var-D inner loop
that constrains the greedy gate search (open research item).

## See also

- `documents/algorithms/ca-mps-var-d.md` -- algorithm reference.
- `documents/api/c/ca-mps.md` -- C API.
- `MATH.md` §12 -- algorithm proof.
- `docs/research/var_d_lattice_gauge_theory.md` -- Z2 LGT case study.
