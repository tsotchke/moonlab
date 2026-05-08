# Tutorial: Noisy circuits with MPDO

The matrix-product density operator (MPDO) engine added in v0.3.0 is
Moonlab's polynomial-cost noise simulator.  Where the dense state-
vector path costs O(2^n) per gate, MPDO costs O(n * chi^2) per
single-qubit channel, so you can simulate hundreds of qubits at
single-qubit error rates around 1e-3 in quasi-1D layouts.

This tutorial walks through the full v0.3 single-qubit surface:
initial state, named channels, custom Kraus operators, and
expectation-value readout.  Every numerical claim is checked by
`tests/unit/test_mpdo_smoke.c` to 1e-12.

Prerequisites:

- Moonlab built with `-DQSIM_BUILD_TESTS=ON`.
- Familiarity with the Kraus operator-sum representation and named
  noise channels (depolarising, amplitude damping, phase damping).

Header: `src/quantum/noise_mpdo.h`.  Full reference:
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

The MPDO is initialised in the product `|0...0><0...0|` at chi = 1.
You always have `Tr(rho) = 1` and `<Z_q> = +1` for every qubit.

## 2. Named single-qubit channels

These wrap the standard Kraus reps and match `src/quantum/noise.h`
exactly.  All of them are CPTP by construction (you do not need to
verify completeness yourself).

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

## 3. Worked example — depolarising channel

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

The Pauli twirl `<Z> -> <Z>(1 - 4p/3)` is not a heuristic — it falls
straight out of the operator-sum identity for the depolarising
channel and is the gold-standard cross-check whenever you are
debugging a custom calibration model.

## 4. Custom Kraus operators

Every named channel is a wrapper around `moonlab_mpdo_apply_kraus_1q`,
so you can also feed your own operators.  The convention is row-major
`[num_kraus, 2, 2]` flat array of `mpdo_complex_t` (= `double _Complex`):

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

Trace preservation (`sum_a K_a^dag K_a = I`) is your responsibility
when you bring your own Kraus rep.  Use
`noise_kraus_completeness_deviation` from `src/quantum/noise.h` to
quantify it before you trust your calibration import.

## 5. Composing gates and noise

A typical noisy circuit alternates ideal unitaries with named channels.
The following sequence simulates a `H` gate followed by phase damping
on an otherwise-decohering qubit:

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

## 6. Reading observables and traces

```c
double trace = moonlab_mpdo_trace(m);                      /* always 1 */
double e_x;
moonlab_mpdo_expect_pauli_1q(m, qubit, /*X=*/1, &e_x);
double e_y;
moonlab_mpdo_expect_pauli_1q(m, qubit, /*Y=*/2, &e_y);
double e_z;
moonlab_mpdo_expect_pauli_1q(m, qubit, /*Z=*/3, &e_z);
```

`Tr(rho)` should always be 1 to roundoff for a CPTP-evolved state.
Drift away from 1 is the canary that either (a) you fed a non-CPTP
Kraus rep, or (b) bond truncation lost too much weight.  In v0.3.0 the
single-qubit path never truncates, so all drift is your Kraus rep.

## 7. Roadmap (v0.3.x)

The remaining MPDO surface lands in v0.3.x:

- Two-qubit Kraus channels with SVD bond truncation
- Two-qubit composite-Pauli wrappers
- High-level `moonlab_mpdo_simulate_noisy(circuit, noise_model, shots)`
  driving the full noise model from a calibration file
- Python binding parity (currently accessible via raw ctypes against
  `libquantumsim`)

If you need two-qubit noise *today*, the dense-state-vector + Kraus
sampler in `src/quantum/noise.c` works at any qubit count up to 32 and
exposes the same named channels with identical conventions.

## See also

- [getting_started.md](getting_started.md) — first-build walkthrough.
- [topological_band_structure.md](topological_band_structure.md) —
  companion v0.3 module (quantum geometric tensor).
- `tests/unit/test_mpdo_smoke.c` — the 9-case smoke that pins every
  expression in this tutorial to 1e-12.
- `src/quantum/noise.h` — pure-state Kraus channel reference, same
  conventions as MPDO so you can cross-check on small systems.
