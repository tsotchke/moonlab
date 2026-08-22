# Quantum annealing — v1.2.1

Moonlab v1.2.1 implements closed-system transverse-field quantum annealing as
a real statevector evolution. It is suitable for algorithm research,
small-instance ground-truth validation, schedule studies, QUBO/Ising
cross-checking, and validation before submitting an embedded problem to a
hardware provider.

It does not claim to emulate a particular D-Wave processor. Hardware graph
minor embedding, freeze-out, calibration bias, finite-temperature bath
dynamics, chain breaks, and provider timing are separate backend concerns.

## Hamiltonian and evolution

The engine starts in the exact ground state of the driver, `|+>^n`, and evolves

```text
H(s) = -A(s) driver_strength sum_i X_i
       + B(s) problem_strength H_problem,       s=t/T.
```

The Ising problem is

```text
H_problem = offset + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j.
```

Every step is evaluated at its midpoint. The default symmetric Strang step is

```text
exp(-i dt A H_driver / 2)
exp(-i dt B H_problem)
exp(-i dt A H_driver / 2).
```

Moonlab implements the driver with `RX` rotations and each `ZZ` problem term
with `CNOT-RZ-CNOT`, reusing the same tested exponentials as QAOA. Set
`second_order=false` to select the first-order product formula explicitly.

Built-in schedules all enforce `A(0)=1`, `B(0)=0`, `A(1)=0`, `B(1)=1`:

| Schedule | `A(s)` | `B(s)` |
|---|---:|---:|
| Linear | `1-s` | `s` |
| Quadratic | `1-s^2` | `s^2` |
| Cosine | `cos^2(pi s/2)` | `sin^2(pi s/2)` |

## QUBO convention

The QUBO API uses the complete row-major matrix:

```text
E(x) = offset + sum_i sum_j x_i Q[i,j] x_j,  x_i in {0,1}.
```

`Q` may be asymmetric. During the exact map `x_i=(1-z_i)/2`, Moonlab combines
`Q[i,j]+Q[j,i]`; no triangle is silently ignored. `moonlab_qubo_to_ising`
exposes the conversion, and the unit oracle checks equality for every
bitstring of an asymmetric three-variable example.

## Results and honesty boundary

The quantum evolution produces the state and sample distribution. Moonlab then
enumerates the already-materialized computational basis to report:

- best sampled and most-likely bitstrings;
- exact ground energy, one representative ground bitstring, and degeneracy;
- exact final classical problem gap;
- expected and residual energy;
- total probability on the ground-state manifold;
- final state norm;
- every retained sample and its classical energy;
- the effective non-zero seed.

Enumeration is diagnostic only. It does not replace, repair, or steer the
quantum evolution, and the best sampled result remains distinct from the exact
ground-state metadata.

## C

The full module is `<quantumsim/algorithms/quantum_annealing.h>`. The stable
ABI 0.8.0 also provides plain-array one-shots:

```c
#include <moonlab/moonlab_export.h>

double Q[4] = {-1, 1, 1, -1};       /* (x0+x1-1)^2, with offset 1 */
moonlab_anneal_summary_v1 result;
uint64_t samples[128];
double energies[128];

int rc = moonlab_anneal_qubo_v1(
    2, Q, 1.0,
    12.0, 1200, 128, 0x123456789abcdef0ULL,
    2,                 /* cosine */
    1.0, 1.0, 1,      /* driver, problem, second-order */
    &result, samples, energies);
```

See `examples/applications/quantum_annealing_demo.c`.

## Python

```python
from moonlab.annealing import AnnealConfig, anneal_qubo

result = anneal_qubo(
    [[-1.0, 1.0], [1.0, -1.0]], offset=1.0,
    config=AnnealConfig(total_time=12, num_steps=1200,
                        num_samples=128, seed=0x123456789abcdef0),
)
assert result.best_energy == 0.0
```

## Rust

```rust
use moonlab::annealing::{anneal_qubo, AnnealConfig};

let q = [-1.0, 1.0, 1.0, -1.0];
let result = anneal_qubo(&q, 1.0, &AnnealConfig::default())?;
```

## JavaScript / WebAssembly

```ts
import { annealQubo } from '@tsotchkecorp/moonlab';

const result = await annealQubo([-1, 1, 1, -1], 1, {
  totalTime: 12, numSteps: 1200, numSamples: 128,
  seed: 0x123456789abcdef0n, schedule: 'cosine',
});
```

Bitstrings and seeds are `bigint` in JavaScript so values above `2^53` are
never rounded.

## Verification contract

The focused native gate proves:

1. every schedule endpoint, monotonicity, and partition-of-unity identity;
2. exact QUBO/Ising parity on all bitstrings of an asymmetric matrix;
3. second-order statevector evolution against an independent 100,000-step RK4
   integration of the one-qubit time-dependent Schrödinger equation;
4. norm preservation, analytic ground energy/degeneracy/gap, and successful
   slow-anneal ground-state preparation;
5. byte-identical samples for equal seeds;
6. fail-closed validation of matrices, schedules, steps, and sample indices;
7. stable ABI symbol resolution and numerical smoke;
8. Python, Rust, and JavaScript marshalling/result parity;
9. ASan/UBSan ABI-boundary replay for bounded fuzzed QUBOs.

The ICC `moonlab-v1.2.1-quantum-annealing` target additionally requires a
clean-tree runtime trace before release certification.
