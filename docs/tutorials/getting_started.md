# Tutorial: Getting started

This tutorial takes a fresh checkout to a running quantum simulation
in three short steps. It targets MoonLab 1.1.0 and the same CMake build
contract exercised by Linux, macOS, Windows x64, and Windows ARM64 CI.

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
| `-DQSIM_WERROR=ON` | ON | Treat warnings as errors |
| `-DQSIM_ENABLE_SANITIZERS=ON` | OFF | ASAN + UBSAN (Debug builds) |
| `-DQSIM_ENABLE_MPI=ON` | OFF | Distributed state-vector path |

After the build finishes you can run the unit tests with:

```sh
ctest --test-dir build --output-on-failure -j
```

CTest labels let you select a subsystem (`core`, `tn`, `algorithms`, `topology`,
`crypto`, `distributed`, `abi`, or `bindings`). For example:

```sh
ctest --test-dir build -L "core|abi" --output-on-failure -j 4
```

### Windows

Windows builds require ClangCL rather than the MSVC C frontend. The supported
driver builds, tests, packages, and verifies an external consumer:

```powershell
.\scripts\build_windows_artifact.ps1 `
  -Arch x64 `
  -BuildDir build-windows-x64 `
  -Output .\dist\moonlab-local-windows-x64.zip
```

Pass `-Arch arm64` for native Windows ARM64. See
[`docs/WINDOWS.md`](../WINDOWS.md) for the raw CMake command and release ZIP
layout.

## 2. Your first quantum program

The canonical introductory program in quantum computing prepares a
single-qubit superposition with the Hadamard gate `H` and verifies
the resulting probability distribution against the Born rule:

```c
#include "quantum/state.h"
#include "quantum/gates.h"
#include "quantum/measurement.h"
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
./build/hello_quantum
```

## 3. Bell pair preparation and verification

The two-qubit entangled Bell state `|Phi^+> = (|00> + |11>) / sqrt(2)`
is prepared by a Hadamard followed by a CNOT:

```c
quantum_state_t state;
quantum_state_init(&state, /*num_qubits=*/2);

gate_hadamard(&state, 0);
gate_cnot(&state, /*control=*/0, /*target=*/1);

for (size_t i = 0; i < 4; ++i) {
    complex_t a = quantum_state_get_amplitude(&state, i);
    printf("|%zu>: %+.4f %+.4fi\n", i, creal(a), cimag(a));
}
quantum_state_free(&state);
```

The example program `./build/examples/basic/bell_state` extends this
preparation with a full CHSH-inequality verification: the measured
correlator `S` violates the classical bound `|S| <= 2` and approaches
the Tsirelson bound `2 sqrt(2) ~= 2.828`.  In Moonlab this verifies that
the simulated state and measurement path reproduce the quantum prediction;
it is not physical-device or device-independent entanglement certification.

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
  contains `build/libquantumsim.dylib` (macOS), `.so` (Linux), or
  `build/Release/quantumsim.dll` (Windows multi-config builds).
  `cmake --install build --prefix /usr/local` puts it on the dynamic
  loader path system-wide.
- **MPI build fails**: requires Homebrew OpenMPI or Linux MPICH;
  verify with `mpicc --version`.
- **Windows complex-arithmetic compile errors**: delete the build directory
  and reconfigure with the Visual Studio generator plus `-T ClangCL`.
- **WebGPU demo doesn't load**: rebuild the WASM artefacts via
  `bindings/javascript/scripts/build_wasm.sh`; serve over `http`, not
  `file://` (Chrome/Safari refuse WebGPU on local files).
- **`ctest` timeouts on `unit_qgt_*`**: expected at lattice sizes
  `L >= 12`; the benchmark targets use smaller grids.  Re-run with
  `-j 1` if parallel CTest contention is the suspected cause.
