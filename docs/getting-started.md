# Getting started with Moonlab

This is the 15-minute onboarding tutorial for MoonLab v1.1. It walks you from
`git clone` to a working Bell circuit submitted via each of the four bindings
(C / Python / Rust / JavaScript), and then shows how to run the same circuit
through the cloud control plane.

## 1. What Moonlab is

Moonlab is a distributed quantum compute platform: a C99 state-vector engine
plus four first-class language bindings (C, Python, Rust, JavaScript/Node), a
control-plane line protocol over TCP for submitting circuits to remote
workers, a Docker quickstart for the full stack, and bridges to the sibling
libraries `libirrep` (QEC + representation theory) and `SbNN` (equivariant
neural networks). One simulation kernel, every language, one wire format.

## 2. Prerequisites

- CMake 3.20 or newer
- A C11 compiler (Clang on macOS, GCC 9+ on Linux, ClangCL on Windows)
- OpenBLAS / LAPACK on Linux; Accelerate is auto-detected on macOS and
  Windows packages use the in-tree QR/SVD fallbacks
- Optional: Python 3.10+, Rust 1.86+, Node 18+ for the corresponding bindings
- Optional: OpenSSL 1.1+ for TLS-wrapped control-plane transport
- Release targets: macOS (ARM64, x86-64), Linux (ARM64, x86-64), and Windows
  (ARM64, x64). See [WINDOWS.md](WINDOWS.md) for the native toolchain contract.

## 3. Build from source (5 min)

On Linux, the Eshkol-style universal driver detects the distribution family,
installs the native dependency names, and performs a portable release build:

```sh
git clone https://github.com/tsotchke/moonlab
cd moonlab
scripts/build-linux.sh --ctest --verify-package
```

It supports Debian-family distributions (Debian, Ubuntu, Mint, Pop!_OS and
derivatives), Fedora/RHEL-family distributions (Fedora, RHEL, Rocky, Alma and
CentOS Stream), Arch-family distributions, openSUSE/SLES, Alpine/musl, and
NixOS through an ephemeral `nix-shell`.
Use `--no-install-deps` when the required development packages are already
managed by the host. The release gate verifies clean current/floor images from
all mainstream families. Its canonical Debian 12/13 and Ubuntu
22.04/24.04/26.04 matrix runs on both hosted x86-64 and ARM64, requires every
focused test (including the entropy health test) to execute without skips, and
publishes source/image/artifact-bound evidence. Physical mesh hosts add NixOS
and Jetson validation.

The platform-neutral manual path remains:

```sh
git clone https://github.com/tsotchke/moonlab
cd moonlab
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure -LE "long|memory_heavy" -j 4
```

Common optional flags:

- `-DQSIM_ENABLE_MPI=ON` -- distributed state vector across MPI ranks
- `-DQSIM_ENABLE_LIBIRREP=ON` -- link the sibling QEC + symmetry library
  (requires `LIBIRREP_ROOT` to point at a built libirrep tree)
- `-DQSIM_ENABLE_METAL=ON` -- Metal GPU backend (macOS, on by default)
- `-DQSIM_ENABLE_CUDA=ON` -- native CUDA state-vector backend (MoonLab 1.1)
- `-DQSIM_ENABLE_OPENMP=ON` -- OpenMP parallel gate kernels (on by default)

The build produces `build/libquantumsim.{dylib,so}` on Unix or
`build/Release/quantumsim.dll` with a multi-config Windows generator. To
install system-wide:

```sh
sudo cmake --install build           # headers -> /usr/local/include/quantumsim/
```

On Windows, use the end-to-end package driver instead of the Unix command:

```powershell
.\scripts\build_windows_artifact.ps1 `
  -Arch x64 `
  -BuildDir build-windows-x64 `
  -Output .\dist\moonlab-local-windows-x64.zip
```

## 4. Bell circuit from C

Save this as `bell.c`:

```c
#include <quantumsim/quantum/state.h>
#include <quantumsim/quantum/gates.h>
#include <stdio.h>

int main(void) {
    quantum_state_t *s = quantum_state_create(2);
    if (!s) { fprintf(stderr, "alloc failed\n"); return 1; }

    gate_hadamard(s, 0);    /* H on qubit 0 */
    gate_cnot(s, 0, 1);     /* CNOT(control=0, target=1) */

    for (uint64_t i = 0; i < 4; i++) {
        printf("P(|%llu>) = %.4f\n",
               (unsigned long long)i,
               quantum_state_get_probability(s, i));
    }

    quantum_state_destroy(s);
    return 0;
}
```

Build and run:

```sh
cc bell.c -lquantumsim -o bell
./bell
```

Expected output:

```
P(|0>) = 0.5000
P(|1>) = 0.0000
P(|2>) = 0.0000
P(|3>) = 0.5000
```

That is the Bell state `(|00> + |11>) / sqrt(2)`.

## 5. Same circuit from Python

```sh
pip install -e bindings/python    # editable install against the local build
```

```python
from moonlab import QuantumState

state = QuantumState(2)
state.h(0).cnot(0, 1)
print(state.probabilities())   # [0.5, 0.0, 0.0, 0.5]
```

The Python `QuantumState` mirrors the C API one-to-one: every gate method
(`h`, `x`, `y`, `z`, `s`, `t`, `cnot`, `cz`, `swap`, `toffoli`, ...) returns
`self` for fluent chaining and raises `QuantumError` on a non-zero status from
the underlying C call.

## 6. Same circuit from Rust

In `Cargo.toml`:

```toml
[dependencies]
moonlab = "1.1"
```

In `src/main.rs`:

```rust
use moonlab::QuantumState;

fn main() {
    let mut s = QuantumState::new(2).expect("alloc");
    s.h(0).cnot(0, 1);
    println!("{:?}", s.probabilities());
    // [0.5, 0.0, 0.0, 0.5]
}
```

Note that the Rust gate methods (`h`, `cnot`, `cx`, `rz`, ...) take `&mut self`
and return `&mut Self` for chaining; argument validation happens inside, and
out-of-range qubits become a no-op rather than panicking. Constructors and
queries (`new`, `prob_zero`, `entanglement_entropy`, ...) return `Result`.

The crate discovers a Homebrew/native SDK through `pkg-config quantumsim`.
For a custom SDK, set both `MOONLAB_LIB_DIR` and `MOONLAB_INCLUDE_DIR`; when
working in this source tree those can point to `$(pwd)/build` and
`$(pwd)/src`, respectively.

## 7. Same circuit from JS (Node)

```sh
cd bindings/javascript/packages/core
pnpm install
pnpm build
```

```typescript
import { QuantumState } from '@moonlab/quantum-core';

const s = await QuantumState.create({ numQubits: 2 });
s.h(0).cnot(0, 1);
console.log(s.getProbabilities());   // Float64Array [0.5, 0, 0, 0.5]
s.dispose();
```

The JS binding runs the same C kernel compiled to WebAssembly via Emscripten.
`QuantumState.create` is async because it instantiates the WASM module on
first use; subsequent creates are synchronous against the cached module.
Always call `dispose()` to free WASM-side memory.

## 8. The cloud platform in one command

The control plane (`moonlab_control_server`) accepts serialized circuits over
TCP, dispatches them to worker simulators, and ships metrics to Prometheus.
It is built from `-DQSIM_ENABLE_CONTROL_PLANE=ON`, the default for a
from-source or Homebrew build on non-Windows platforms. The published
Python wheels build with `QSIM_ENABLE_CONTROL_PLANE=OFF` (see
`bindings/python/pyproject.toml`) to keep the wheel dependency-free, so the
walkthrough below needs a repo build (or the Homebrew formula), not
`pip install moonlab`. The whole stack ships in `deploy/docker/`:

```sh
docker compose -f deploy/docker/docker-compose.yml up --build
```

This brings up:

- `moonlab-control` -- circuit dispatcher on `127.0.0.1:7070`
- `moonlab-exporter` -- Prometheus exporter on `127.0.0.1:9090`
- `prometheus` -- scraper + UI on `http://127.0.0.1:9091`

Submit the same Bell circuit from Python:

```python
from moonlab.control_plane import submit_circuit
from moonlab.qgtl import QgtlCircuit, GateType

c = (QgtlCircuit(num_qubits=2)
     .add_gate(GateType.H,    target=0)
     .add_gate(GateType.CNOT, target=1, control=0))

probs = submit_circuit("127.0.0.1", 7070, c.serialize())
print(probs)   # [0.5, 0.0, 0.0, 0.5]
```

The Rust and JS bindings expose the same wire protocol via
`moonlab::control_plane::submit_circuit` and
`@moonlab/quantum-core/control-plane` respectively. The serialized form is the
moonlab-circuit v1 text format, identical across all bindings.

Open `http://127.0.0.1:9091` to watch shot rate, gate latency, and worker
queue depth update in real time as you submit more jobs.

## 9. Where to go next

- `docs/CONTROL_PLANE.md` -- deployment guide for the control plane, including
  TLS, HMAC auth, and multi-worker scheduling
- `docs/PARITY_MATRIX.md` -- which subsystems are exposed in which binding
- `docs/STABLE_ABI.md` -- the v1.0 stability contract for the C ABI
- `docs/INTEGRATION_libirrep_SbNN.md` -- bridges to the sibling libraries
- `examples/` -- 52 runnable demos across all bindings (start with
  `examples/basic/bell_state.c` for an extended version of the Bell circuit
  above with entropy and CHSH analysis)
- `bindings/python/README.md`, `bindings/rust/moonlab/README.md`,
  `bindings/javascript/README.md` -- language-specific deep dives
