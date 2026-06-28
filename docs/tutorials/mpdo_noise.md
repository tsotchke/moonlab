# Tutorial: Noisy circuit simulation with the MPDO engine

The matrix-product density operator (MPDO) representation, introduced
by Verstraete, Garcia-Ripoll, and Cirac [1] and developed for noisy
circuit simulation by Werner et al. [2], encodes a many-body density
matrix as a tensor train of vectorised local density blocks.  The
Moonlab MPDO engine implements this representation with bond
dimension `chi` capped at construction time.  The cost of a
single-qubit Kraus channel is `O(n chi^2)`, in contrast to the
`O(4^n)` scaling of dense density-matrix simulation, enabling
simulation of order-100 qubits in quasi-1D layouts at single-qubit
error rates near `10^-3`.

This tutorial covers the v0.3 single-qubit surface: state
initialisation, named completely-positive trace-preserving (CPTP)
channels, user-supplied Kraus operators, and observable readout.
All numerical results are pinned to machine precision (`10^-12`) by
`tests/unit/test_mpdo_smoke.c`.

Prerequisites:

- Moonlab built with `-DQSIM_BUILD_TESTS=ON`.
- Working knowledge of the Kraus operator-sum representation [3]
  and the standard single-qubit noise channels (depolarising,
  amplitude damping, dephasing).

Header: `src/quantum/noise_mpdo.h`.  Full API reference:
[../reference/mpdo-api.md](../reference/mpdo-api.md).

## 1. The initial state

```c
#include "moonlab/quantum/noise_mpdo.h"
#include <stdio.h>

int main(void) {
    /* 4 qubits, bond cap chi = 16. */
    moonlab_mpdo_t* m = moonlab_mpdo_create(/*qubits=*/4, /*max_bond=*/16);
    printf("Tr(rho) = %.12f (expected 1)\n",     moonlab_mpdo_trace(m));

    double z;
    moonlab_mpdo_expect_pauli_1q(m, /*qubit=*/2, /*Z=*/3, &z);
    printf("<Z_2> on |0000> = %+.6f (expected +1)\n", z);

    moonlab_mpdo_free(m);
}
```

The MPDO is initialised in the product state
`|0...0><0...0|` at bond dimension `chi = 1`, so `Tr(rho) = 1`
and `<Z_q> = +1` for every qubit by construction.

## 2. Named single-qubit channels

The library exposes the canonical single-qubit Kraus channels with
operators consistent with [4] and the conventions used in
`src/quantum/noise.h`.  Each named channel is CPTP by construction;
no completeness check is required of the user.

```c
moonlab_mpdo_apply_depolarizing_1q     (m, qubit, p);       /* p in [0,1] */
moonlab_mpdo_apply_amplitude_damping_1q(m, qubit, gamma);   /* T_1 */
moonlab_mpdo_apply_phase_damping_1q    (m, qubit, lambda);  /* T_2 */
moonlab_mpdo_apply_bit_flip_1q         (m, qubit, p);
moonlab_mpdo_apply_phase_flip_1q       (m, qubit, p);
moonlab_mpdo_apply_bit_phase_flip_1q   (m, qubit, p);
```

| channel | Kraus rep | analytical effect |
|---|---|---|
| depolarising | `(1-p) I + p/3 (X.X + Y.Y + Z.Z)` | `<Z>` -> `<Z>` * (1 - 4p/3) |
| amplitude damping | `K_0 = diag(1, sqrt(1-gamma)), K_1 = sqrt(gamma) |0><1|` | `gamma=1` resets to `|0>` |
| phase damping | `K_0 = diag(1, sqrt(1-lambda)), K_1 = sqrt(lambda) |1><1|` | preserves `<Z>`, kills `<X>, <Y>` |
| bit_flip | `(1-p) I + p X` | `<Z>` -> `<Z>` * (1-2p) |
| phase_flip | `(1-p) I + p Z` | `<X>` -> `<X>` * (1-2p) |

## 3. Worked example: the depolarising channel

```c
#include "moonlab/quantum/noise_mpdo.h"
#include <stdio.h>

int main(void) {
    moonlab_mpdo_t* m = moonlab_mpdo_create(1, 16);
    moonlab_mpdo_apply_depolarizing_1q(m, /*qubit=*/0, /*p=*/0.4);

    double z;
    moonlab_mpdo_expect_pauli_1q(m, 0, /*Z=*/3, &z);
    /* expected 1 - 4 * 0.4 / 3 = 0.4667; we measure 0.466667. */
    printf("<Z> after depolarising p=0.4: %.6f\n", z);

    moonlab_mpdo_free(m);
}
```

The contraction `<Z> -> <Z>(1 - 4p/3)` follows from the Pauli-twirl
identity for the symmetric depolarising channel and serves as a
canonical cross-check when validating custom calibration models.

## 4. User-supplied Kraus operators

Each named channel is a thin wrapper around the general entry point
`moonlab_mpdo_apply_kraus_1q`, which accepts an arbitrary
operator-sum decomposition.  The convention is a row-major
`[num_kraus, 2, 2]` flat array of `mpdo_complex_t`
(`double _Complex`):

```c
/* Pauli X as a single-Kraus channel (i.e. a unitary X gate). */
const mpdo_complex_t X[4] = { 0.0, 1.0,
                              1.0, 0.0 };
moonlab_mpdo_apply_kraus_1q(m, /*qubit=*/0, X, /*num_kraus=*/1);
```

A two-Kraus example for the symmetric depolarising channel at
arbitrary `p` (equivalent to `apply_depolarizing_1q`):

```c
const double a = sqrt(1.0 - p);
const double b = sqrt(p / 3.0);
const mpdo_complex_t k[4 * 4] = {
    /* sqrt(1-p) I */     a, 0.0,    0.0, a,
    /* sqrt(p/3) X */     0.0, b,    b,   0.0,
    /* sqrt(p/3) Y */     0.0, -b * _Complex_I,
                          b * _Complex_I, 0.0,
    /* sqrt(p/3) Z */     b, 0.0,    0.0, -b,
};
moonlab_mpdo_apply_kraus_1q(m, qubit, k, /*num_kraus=*/4);
```

Trace preservation (`sum_a K_a^dag K_a = I`) is the user's
responsibility for arbitrary Kraus decompositions; the helper
`noise_kraus_completeness_deviation` in `src/quantum/noise.h`
returns the operator-norm deviation and should be checked before
trusting any calibration import.

## 5. Composing unitaries and noise

A realistic noisy circuit alternates ideal unitary gates with named
channels.  The following sequence applies a Hadamard gate followed
by a dephasing channel and measures the resulting transverse
coherence:

```c
moonlab_mpdo_t* m = moonlab_mpdo_create(2, 16);

/* Hadamard on qubit 0 (Kraus = {H}, single op). */
const double h = 1.0 / sqrt(2.0);
const mpdo_complex_t H[4] = { h,  h,
                              h, -h };
moonlab_mpdo_apply_kraus_1q(m, 0, H, 1);

/* Phase damping at lambda = 0.2 on the same qubit. */
moonlab_mpdo_apply_phase_damping_1q(m, 0, /*lambda=*/0.2);

/* Now <X_0> = sqrt(1 - 0.2) ~ 0.8944 and <Z_0> = 0. */
double x0, z0;
moonlab_mpdo_expect_pauli_1q(m, 0, /*X=*/1, &x0);
moonlab_mpdo_expect_pauli_1q(m, 0, /*Z=*/3, &z0);
printf("<X_0> = %.4f (expected 0.8944)\n", x0);
printf("<Z_0> = %.4f (expected 0)\n",      z0);
```

## 6. Observables and trace

```c
double trace = moonlab_mpdo_trace(m);                      /* always 1 */
double e_x;
moonlab_mpdo_expect_pauli_1q(m, qubit, /*X=*/1, &e_x);
double e_y;
moonlab_mpdo_expect_pauli_1q(m, qubit, /*Y=*/2, &e_y);
double e_z;
moonlab_mpdo_expect_pauli_1q(m, qubit, /*Z=*/3, &e_z);
```

`Tr(rho)` must equal 1 to roundoff for a CPTP-evolved state.  Any
deviation indicates either (a) a non-CPTP user-supplied Kraus
decomposition or (b) loss of weight to bond truncation.  In v0.3.0
the single-qubit path performs no truncation, so any deviation
reflects the supplied Kraus operators.

## 7. Language bindings

The single-qubit MPDO surface ships with first-class Python and Rust
bindings in v0.3.0.  The Python module is a thin wrapper that
preserves the C semantics:

```python
from moonlab.mpdo import Mpdo

rho = Mpdo(num_qubits=4, max_bond_dim=16)
rho.apply_depolarizing(qubit=0, p=0.4)
print(rho.expect_pauli(0, 'Z'))   # 0.466667 ...
```

The Rust binding (`moonlab::mpdo::Mpdo`) provides RAII handle
management, typed Pauli codes, and the same six named channels.
Both bindings are validated against the C reference at `1e-12`
tolerance.

## 8. Roadmap

The remaining MPDO surface is scheduled for forthcoming releases:

- Two-qubit Kraus channels with singular-value-decomposition bond
  truncation [2].
- Two-qubit composite-Pauli channel wrappers.
- A high-level entry point
  `moonlab_mpdo_simulate_noisy(circuit, noise_model, shots)` driven
  by a calibrated noise model.

For two-qubit noise in the interim, the dense state-vector
implementation in `src/quantum/noise.c` supports any qubit count up
to 32 and exposes the same named channels with identical
conventions.

## References

[1] F. Verstraete, J. J. Garcia-Ripoll, and J. I. Cirac,
    "Matrix product density operators: Simulation of finite-
    temperature and dissipative systems",
    Phys. Rev. Lett. **93**, 207204 (2004).

[2] A. H. Werner, D. Jaschke, P. Silvi, M. Kliesch, T. Calarco,
    J. Eisert, and S. Montangero, "Positive tensor network
    approach for simulating open quantum many-body systems",
    Phys. Rev. Lett. **116**, 237201 (2016).

[3] K. Kraus, *States, Effects, and Operations: Fundamental
    Notions of Quantum Theory*, Lecture Notes in Physics, vol. 190,
    Springer (1983).

[4] M. A. Nielsen and I. L. Chuang, *Quantum Computation and
    Quantum Information*, Cambridge University Press, 10th
    Anniversary Edition, 2010, chapter 8 (Quantum noise and quantum
    operations).

## See also

- [getting_started.md](getting_started.md): a first-build
  walkthrough.
- [topological_band_structure.md](topological_band_structure.md):
  the companion v0.3 module (quantum geometric tensor).
- `tests/unit/test_mpdo_smoke.c`: the nine-case validation suite
  that pins every numerical claim in this tutorial to `10^-12`.
- `src/quantum/noise.h`: pure-state Kraus channel reference
  implementation with conventions identical to the MPDO engine,
  enabling small-system cross-checks.
