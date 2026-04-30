# Tutorial 11: 1+1D Z2 lattice gauge theory with var-D

End-to-end walk-through of the 1+1D Z2 lattice gauge theory shipped
in **v0.2.1**: build the matter + gauge-link Pauli sum, project into
the gauge-invariant subspace via the gauge-aware warmstart, and
search for the ground state with var-D.

By the end of this tutorial you will:

- Construct the Z2 LGT Hamiltonian on a small qubit chain.
- Prepare a state in the +1 eigenspace of every interior Gauss-law
  operator using the symplectic Gauss-Jordan warmstart.
- Run the variational-D alternating loop on top of that state.
- Confirm the gauge-invariance pin: `<G_x>` stays at `+1` throughout.

## Background

The lattice qubit layout is interleaved matter / link:

```
matter site 0 -- link 0 -- matter site 1 -- link 1 -- ... -- matter site N-1
qubit:    0          1          2              3            2N-2
```

Total qubits: `2N - 1`.

The Hamiltonian (Jordan-Wigner with parallel transport, exactly
gauge-invariant kinetic terms):

```
H = -(t/2) sum_x [ X_{2x} Y_{2x+1} Y_{2x+2}
                 - Y_{2x} Y_{2x+1} X_{2x+2} ]   (matter + gauge link)
    - h sum_x Z_{2x+1}                          (electric field)
    + (m/2) sum_x (-1)^x Z_{2x}                 (staggered mass)
    + lambda sum_{x=1..N-2} (I - G_x)            (Gauss-law penalty)
```

Interior Gauss-law operator: `G_x = X_{2x-1} Z_{2x} X_{2x+1}`.

The kinetic terms commute with every `G_y` term-by-term (anti-commute
count = 2 each = even), so `H` preserves the gauge sector exactly.

## Setup (Python)

```python
import numpy as np
from moonlab import (
    CAMPS, var_d_run, gauge_warmstart, status_string,
    z2_lgt_1d_build, z2_lgt_1d_gauss_law,
    WARMSTART_STABILIZER_SUBGROUP,
)
```

## Step 1: Build the Hamiltonian

```python
N = 4
paulis, coeffs = z2_lgt_1d_build(
    N=N, t=1.0, h=0.5, m=0.5, gauss_penalty=0.0)

print(f"qubits: {paulis.shape[1]}  ({2 * N - 1} expected)")
print(f"terms : {paulis.shape[0]}")
```

`gauss_penalty=0.0` is fine because v0.2.1's kinetic terms are
exactly gauge-invariant -- the lambda penalty is redundant inside
the +1 sector.

## Step 2: Stack the Gauss-law generators

```python
gens = np.stack([z2_lgt_1d_gauss_law(N=N, site_x=x)
                 for x in range(1, N - 1)])
print("gens shape =", gens.shape)   # (N - 2, 2N - 1) = (2, 7)
```

Each row is the Pauli-byte encoding of one interior `G_x`.

## Step 3: Apply the gauge-aware warmstart (sanity check)

You can apply just the warmstart, without var-D, to confirm the
state lands in the gauge sector:

```python
nq = 2 * N - 1
state = CAMPS(num_qubits=nq, max_bond_dim=32)

gauge_warmstart(state, gens)

# At this point state.D|0^n> is in the simultaneous +1 eigenspace
# of every G_x.  Verified by tests/unit/test_gauge_warmstart.c
# case 4 (four Gauss-law operators of N=4 -> all <G_x> = +1).
print("norm =", state.norm)   # 1.0 within float tolerance.
```

## Step 4: Run var-D with the gauge-aware warmstart

```python
state = CAMPS(num_qubits=nq, max_bond_dim=32)
energy = var_d_run(
    state, paulis, coeffs,
    max_outer_iters=10,
    imag_time_dtau=0.10,
    imag_time_steps_per_outer=4,
    clifford_passes_per_outer=8,
    composite_2gate=True,
    warmstart=WARMSTART_STABILIZER_SUBGROUP,
    stab_paulis=gens,
)
print(f"final variational energy = {energy:.6f}")
```

Internally the loop alternates a greedy local-Clifford D-update with
imag-time `|phi>`-update; the imag-time step preserves the gauge
sector because every Pauli term in `H` commutes with every `G_x`.

## Step 5: Verify gauge invariance after the loop

The `var_d_run` call above runs 10 outer iters with imag-time on
`|phi>` between each.  If the kinetic terms anti-commuted with the
Gauss-law operators (as in pre-v0.2.1 builds), the state would
drift.  In v0.2.1 it stays exactly inside the +1 sector.  This is
pinned in `tests/unit/test_z2_lgt_pauli_sum.c` (term-by-term
commutativity check) and validated end-to-end in
`examples/hep/z2_gauge_var_d.c`.

## Larger systems

Scale `N` up:

- `N = 4`: 7 qubits, 2 interior Gauss-law constraints (this tutorial).
- `N = 6`: 11 qubits, 4 constraints.
- `N = 8`: 15 qubits, 6 constraints.  Use larger `max_bond_dim`
  (64 or 128).

For `N >= 6`, expect the alternating loop to need `max_outer_iters
= 25` to 50 with `composite_2gate=True` to converge.

## What to read next

- `documents/algorithms/ca-mps-var-d.md` -- the algorithmic reference
  for var-D and the warmstart.
- `documents/guides/gauge-aware-warmstart.md` -- how to use the
  warmstart on other stabilizer-coded problems (surface code, toric
  code, repetition code).
- `docs/research/var_d_lattice_gauge_theory.md` -- the math write-up
  including the gauge-invariance proof.
- `examples/hep/z2_gauge_var_d.c` -- the parameter-sweep demo
  driver.
