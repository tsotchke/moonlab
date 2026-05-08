# Tutorial: Getting started

This tutorial takes you from a fresh checkout to running your first
quantum simulation.  Every command was verified against v0.3.0 on
macOS arm64 and Ubuntu 24.04 x86_64.

## 1. Build the library

Moonlab is CMake-driven.  An out-of-tree build keeps your source tree
clean:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

That gives you `libquantumsim` plus every example, test, and benchmark
target by default.  Useful flags:

| Flag | Default | What it gates |
|---|---|---|
| `-DQSIM_BUILD_TESTS=ON` | ON | CTest targets under `tests/` |
| `-DQSIM_BUILD_EXAMPLES=ON` | ON | Programs under `examples/` |
| `-DQSIM_BUILD_BENCHMARKS=ON` | ON | Programs under `benchmarks/` |
| `-DQSIM_WERROR=ON` | OFF | Treat warnings as errors |
| `-DQSIM_ENABLE_SANITIZERS=ON` | OFF | ASAN + UBSAN (Debug builds) |
| `-DQSIM_ENABLE_MPI=ON` | OFF | Distributed state-vector path |

After the build finishes you can run the unit tests with:

```sh
ctest --test-dir build --output-on-failure -j
```

A green run is 17/17 topology, plus all base subsystems (state vector,
gates, measurement, MPS, DMRG, var-D, MPDO, QGT, QRNG, PQC).

## 2. Your first quantum program

The classical "hello world" of quantum computing is the Hadamard gate:

```c
#include "moonlab/quantum/state.h"
#include "moonlab/quantum/gates.h"
#include "moonlab/quantum/measurement.h"
#include <stdio.h>

int main(void) {
    quantum_state_t state;
    quantum_state_init(&state, /*num_qubits=*/1);

    /* H|0> = (|0> + |1>) / sqrt(2). */
    gate_hadamard(&state, /*qubit=*/0);

    double p0 = quantum_state_get_probability(&state, 0);
    double p1 = quantum_state_get_probability(&state, 1);
    printf("After H|0>: P(0) = %.4f, P(1) = %.4f\n", p0, p1);
    /* prints 0.5000 / 0.5000 to roundoff. */

    quantum_state_free(&state);
    return 0;
}
```

A worked version of the same program (with measurement statistics and
ASCII bar charts) lives at `examples/basic/hello_quantum.c` and is
already built for you:

```sh
./build/examples/basic/hello_quantum
```

## 3. Two qubits, one gate, one Bell pair

```c
quantum_state_t state;
quantum_state_init(&state, /*num_qubits=*/2);

gate_hadamard(&state, 0);
gate_cnot(&state, /*control=*/0, /*target=*/1);
/* state is (|00> + |11>) / sqrt(2). */

for (size_t i = 0; i < 4; ++i) {
    complex_t a = quantum_state_get_amplitude(&state, i);
    printf("|%zu>: %+.4f %+.4fi\n", i, creal(a), cimag(a));
}
quantum_state_free(&state);
```

This is the canonical entangled state.  Run
`./build/examples/basic/bell_state` to see the same program with full
CHSH-inequality verification (`S` should land near 2*sqrt(2) = 2.828).

## 4. Where to go next

- `examples/basic/` — GHZ state, teleportation, full gate tour.
- `examples/algorithms/` — Grover's search, QFT, QAOA, VQE on H_2.
- `examples/topological/` — six topological models with computed
  Chern / Z_2 / winding invariants.  Walked through in
  [topological_band_structure.md](topological_band_structure.md).
- `examples/applications/` — Bell-test QRNG, post-quantum KEM
  (FIPS 203), CHSH multi-run aggregator.
- [mpdo_noise.md](mpdo_noise.md) — simulate noisy circuits without
  shooting Monte Carlo trajectories.

If you prefer Python, every example under `examples/basic/` has a `.py`
sibling using the bindings at `bindings/python/moonlab/`.

## Troubleshooting

- **Link error on `libquantumsim`**: confirm your build directory
  contains `build/libquantumsim.dylib` (macOS) or `.so` (Linux).
  `cmake --install build --prefix /usr/local` puts it on the dynamic
  loader path system-wide.
- **MPI build fails**: requires Homebrew OpenMPI or Linux MPICH;
  verify with `mpicc --version`.
- **WebGPU demo doesn't load**: rebuild the WASM artefacts via
  `bindings/javascript/scripts/build_wasm.sh`; serve over `http`, not
  `file://` (Chrome/Safari refuse WebGPU on local files).
- **ctest timeout on `unit_qgt_*`**: expected at L >= 12; the
  benchmark targets scan smaller grids.  Re-run with `-j 1` if
  parallel CTest contention is the suspect.
