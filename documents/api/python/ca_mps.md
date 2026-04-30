# Clifford-Assisted MPS (CA-MPS) — Python API

The `moonlab.ca_mps` module wraps the v0.2.1 stable C ABI for the
Clifford-Assisted MPS pipeline: hybrid `|psi> = D|phi>` state
representation, variational-D ground-state search, gauge-aware
stabilizer-subgroup warmstart, and the 1+1D Z2 lattice gauge theory
helpers.

**Module**: `moonlab.ca_mps` (also re-exported as `moonlab.CAMPS`,
`moonlab.var_d_run`, etc.)

**Since**: v0.2.1.

## Quick example: gauge-aware ground-state prep on Z2 LGT

```python
import numpy as np
from moonlab import (
    CAMPS, var_d_run, z2_lgt_1d_build, z2_lgt_1d_gauss_law,
    WARMSTART_STABILIZER_SUBGROUP,
)

# Build the Z2 LGT Hamiltonian on 4 matter sites (7 qubits).
paulis, coeffs = z2_lgt_1d_build(N=4, t=1.0, h=0.5, m=0.5,
                                  gauss_penalty=0.0)

# Stack the interior Gauss-law operators as a (k, n) bytes block.
gens = np.stack([z2_lgt_1d_gauss_law(N=4, site_x=x) for x in (1, 2)])

# Run var-D with the gauge-aware warmstart.
state = CAMPS(num_qubits=7, max_bond_dim=32)
energy = var_d_run(state, paulis, coeffs,
                    warmstart=WARMSTART_STABILIZER_SUBGROUP,
                    stab_paulis=gens)
print(f"final energy = {energy:.6f}")
```

## CAMPS class

```python
class CAMPS:
    def __init__(self, num_qubits: int, max_bond_dim: int = 32) -> None: ...

    @property
    def num_qubits(self) -> int: ...
    @property
    def bond_dim(self) -> int: ...
    @property
    def norm(self) -> float: ...

    # Clifford gates: tableau-only, no MPS cost.
    def h(self, q: int) -> None: ...
    def s(self, q: int) -> None: ...
    def sdag(self, q: int) -> None: ...
    def x(self, q: int) -> None: ...
    def y(self, q: int) -> None: ...
    def z(self, q: int) -> None: ...
    def cnot(self, c: int, t: int) -> None: ...
    def cz(self, a: int, b: int) -> None: ...
    def swap(self, a: int, b: int) -> None: ...

    # Non-Clifford rotations: pushed into the MPS factor.
    def rx(self, q: int, theta: float) -> None: ...
    def ry(self, q: int, theta: float) -> None: ...
    def rz(self, q: int, theta: float) -> None: ...
    def t_gate(self, q: int) -> None: ...
    def t_dagger(self, q: int) -> None: ...
    def phase(self, q: int, theta: float) -> None: ...

    def normalize(self) -> None: ...
```

The handle is owned and freed automatically on garbage collection.
All methods raise `RuntimeError` on a non-zero return from the C
ABI; the message includes the canonical `moonlab_status_to_string`
output.

## Free functions

```python
WARMSTART_IDENTITY              = 0
WARMSTART_H_ALL                 = 1
WARMSTART_DUAL_TFIM             = 2
WARMSTART_FERRO_TFIM            = 3
WARMSTART_STABILIZER_SUBGROUP   = 4

def gauge_warmstart(state: CAMPS, paulis: ArrayLike) -> None: ...

def var_d_run(state: CAMPS,
               paulis: ArrayLike,
               coeffs: ArrayLike,
               max_outer_iters: int = 25,
               imag_time_dtau: float = 0.10,
               imag_time_steps_per_outer: int = 4,
               clifford_passes_per_outer: int = 8,
               composite_2gate: bool = False,
               warmstart: int = WARMSTART_IDENTITY,
               stab_paulis: ArrayLike | None = None) -> float: ...

def z2_lgt_1d_build(N: int,
                     t: float = 1.0, h: float = 1.0,
                     m: float = 0.0,
                     gauss_penalty: float = 0.0
                     ) -> tuple[np.ndarray, np.ndarray]: ...

def z2_lgt_1d_gauss_law(N: int, site_x: int) -> np.ndarray: ...

def status_string(module: int, status: int) -> str: ...
```

`paulis` is a `(num_terms, num_qubits)` `np.uint8` array with byte
encoding `0=I, 1=X, 2=Y, 3=Z`.  `coeffs` is a `(num_terms,)`
`np.float64` array.

When `warmstart == WARMSTART_STABILIZER_SUBGROUP`, supply
`stab_paulis` of shape `(num_gens, num_qubits)`.

`status_string` mirrors the C `moonlab_status_string`:

```python
>>> from moonlab.ca_mps import status_string, StatusModule  # implicit module enum
>>> status_string(module=1, status=0)
'SUCCESS'
>>> status_string(module=1, status=-1)
'ERR_INVALID'
```

## Tests

Six smoke + correctness tests under
`bindings/python/tests/test_ca_mps.py` exercise:

- `CAMPS` lifecycle + basic gates.
- `status_string` canonical codes.
- `z2_lgt_1d_build` shape + `z2_lgt_1d_gauss_law` byte layout.
- `gauge_warmstart` on Bell stabilizers (norm preserved).
- Anti-commuting input rejection.
- Short var-D run on a 4-qubit toy TFIM.

Run via the Moonlab CMake suite:

```bash
ctest -L "" -R python_bindings_pytest --output-on-failure
```

## See also

- C API: `documents/api/c/ca-mps.md`.
- Algorithm walk-through: `documents/algorithms/ca-mps-var-d.md`.
- Math: `MATH.md` §10-12.
- Z2 LGT: `docs/research/var_d_lattice_gauge_theory.md`.
