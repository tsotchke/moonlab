# Matrix-Product Density Operator (MPDO) — API reference

`src/quantum/noise_mpdo.{c,h}` — polynomial-cost noisy circuit
simulator using the matrix-product representation of the density
matrix.  Suited to local-noise circuits in quasi-1D layouts up to
~100 qubits at single-qubit error rates of 1e-3.

Status: scaffold (v0.3.0).  Single-qubit Kraus channels exposed;
two-qubit Kraus + SVD bond truncation deferred to v0.3.x.

## Concepts

The MPDO stores a noisy n-qubit density matrix `rho` as an MPS-of-
superoperators.  Each site tensor has a 4-dimensional physical leg
holding the vectorised local 2x2 density block:

```
vec(rho_i) = (rho_00, rho_01, rho_10, rho_11)
```

and a left/right virtual bond of dimension `chi` capturing site-site
correlations.  A single-qubit Kraus channel acts as a 4x4
superoperator on the physical leg in O(chi^2) time without changing
the bond.  Two-qubit channels and gates require local SVD truncation
(landing in v0.3.x).

The bond-dim cap is fixed at `moonlab_mpdo_create` time; truncation
during multi-site operations (when implemented) will not exceed it.

## Lifecycle

```c
typedef struct moonlab_mpdo_t moonlab_mpdo_t;

typedef enum {
    MPDO_SUCCESS    = 0,
    MPDO_ERR_INVALID = -1,
    MPDO_ERR_QUBIT   = -2,
    MPDO_ERR_OOM     = -3,
    MPDO_ERR_BACKEND = -4,
} mpdo_error_t;

moonlab_mpdo_t* moonlab_mpdo_create(uint32_t num_qubits,
                                     uint32_t max_bond_dim);
void            moonlab_mpdo_free(moonlab_mpdo_t* m);
moonlab_mpdo_t* moonlab_mpdo_clone(const moonlab_mpdo_t* m);
```

Initial state is `|0...0><0...0|` at chi = 1.  Recommended
`max_bond_dim = 32` for ~50-qubit local-noise circuits.

## Introspection

```c
uint32_t moonlab_mpdo_num_qubits(const moonlab_mpdo_t*);
uint32_t moonlab_mpdo_max_bond_dim(const moonlab_mpdo_t*);
uint32_t moonlab_mpdo_current_bond_dim(const moonlab_mpdo_t*);
double   moonlab_mpdo_trace(const moonlab_mpdo_t*);
```

`Tr(rho)` should always equal 1 to within roundoff for a CPTP-evolved
state; deviations indicate truncation error or implementation bugs.

## General Kraus channels

```c
typedef double _Complex mpdo_complex_t;

mpdo_error_t moonlab_mpdo_apply_kraus_1q(moonlab_mpdo_t* state,
                                          uint32_t qubit,
                                          const mpdo_complex_t* kraus,
                                          uint32_t num_kraus);
```

Kraus operators stored as a flat row-major `[num_kraus, 2, 2]` array.
The channel is `rho -> sum_a K_a rho K_a^dagger`.  Caller must ensure
`sum_a K_a^dag K_a = I` (use
`noise_kraus_completeness_deviation` from `noise.h` to verify).

## Named single-qubit channels

Each wraps the right Kraus rep using conventions from
`src/quantum/noise.h`:

```c
mpdo_error_t moonlab_mpdo_apply_depolarizing_1q     (state, qubit, p);
mpdo_error_t moonlab_mpdo_apply_amplitude_damping_1q(state, qubit, gamma);
mpdo_error_t moonlab_mpdo_apply_phase_damping_1q    (state, qubit, lambda);
mpdo_error_t moonlab_mpdo_apply_bit_flip_1q         (state, qubit, p);
mpdo_error_t moonlab_mpdo_apply_phase_flip_1q       (state, qubit, p);
mpdo_error_t moonlab_mpdo_apply_bit_phase_flip_1q   (state, qubit, p);
```

| channel | Kraus | physical effect |
|---|---|---|
| depolarizing | `(1-p) I + p/3 (X.X + Y.Y + Z.Z)` | `<Z> -> 1 - 4p/3` on `|0>` |
| amplitude_damping (T1) | `K_0 = diag(1, sqrt(1-gamma)), K_1 = sqrt(gamma) |0><1|` | gamma=1 resets to `|0>` |
| phase_damping (T2) | `K_0 = diag(1, sqrt(1-lambda)), K_1 = sqrt(lambda) |1><1|` | preserves `<Z>`, kills `<X>, <Y>` |
| bit_flip | `(1-p) I + p X` | `<Z> -> (1-2p) <Z>` |
| phase_flip | `(1-p) I + p Z` | `<X> -> (1-2p) <X>` |
| bit_phase_flip | `(1-p) I + p Y` | flips both `<X>` and `<Z>` |

## Observables

```c
mpdo_error_t moonlab_mpdo_expect_pauli_1q(const moonlab_mpdo_t* state,
                                           uint32_t qubit,
                                           uint8_t  pauli_code,
                                           double*  out_expval);
```

`pauli_code` ∈ {0=I, 1=X, 2=Y, 3=Z}.  Returns `Tr(rho * P_q)` as a
real number.

## Example

```c
#include "moonlab/quantum/noise_mpdo.h"
#include <stdio.h>

int main(void) {
    moonlab_mpdo_t* m = moonlab_mpdo_create(/*qubits=*/3,
                                            /*max_bond=*/16);
    /* Apply X to qubit 1 then depolarising at p=0.4. */
    const mpdo_complex_t X[4] = { 0.0, 1.0, 1.0, 0.0 };
    moonlab_mpdo_apply_kraus_1q(m, 1, X, 1);
    moonlab_mpdo_apply_depolarizing_1q(m, 1, 0.4);

    double z = 0.0;
    moonlab_mpdo_expect_pauli_1q(m, 1, /*Z=*/3, &z);
    printf("<Z_1> = %.6f (expected -1 + 4*0.4/3 = -0.4667)\n", z);
    /* prints -0.4667 to machine precision */

    moonlab_mpdo_free(m);
    return 0;
}
```

## Roadmap (v0.3.x)

- Two-qubit Kraus channels with SVD bond truncation
- Wrappers for the existing two-qubit composite-Pauli channel
- High-level `moonlab_mpdo_simulate_noisy(circuit, noise_model, shots)`
  entry point per the v0.3 plan §2A
- Python binding parity (currently accessible via raw ctypes against
  `libquantumsim`)

## See also

- `src/quantum/noise.h` — pure-state Kraus channel reference
  implementation; conventions match exactly.
- `tests/unit/test_mpdo_smoke.c` — 9-case smoke at 1e-12 tolerance.
- `docs/reference/qgt-api.md` — sister v0.3 module (topology).
