# Getting started with Moonlab

This is the 15-minute onboarding tutorial for Moonlab v1.0. It walks you from
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

- CMake 3.18 or newer
- A C99 compiler (clang on macOS, gcc 9+ on Linux)
- OpenBLAS / LAPACK (Accelerate is auto-detected on macOS)
- Optional: Python 3.10+, Rust 1.70+, Node 18+ for the corresponding bindings
- Optional: OpenSSL 1.1+ for TLS-wrapped control-plane transport
- Validated on macOS (arm64, x86_64) and Linux (x86_64). Windows is unsupported.

## 3. Build from source (5 min)

```sh
git clone https://github.com/tsotchke/moonlab
cd moonlab
cmake -B build -DQSIM_ENABLE_TLS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure   # ~3 minutes, 114 tests
```

All 114 tests should pass. Common optional flags:

- `-DQSIM_ENABLE_MPI=ON` -- distributed state vector across MPI ranks
- `-DQSIM_ENABLE_LIBIRREP=ON` -- link the sibling QEC + symmetry library
  (requires `LIBIRREP_ROOT` to point at a built libirrep tree)
- `-DQSIM_ENABLE_METAL=ON` -- Metal GPU backend (macOS, on by default)
- `-DQSIM_ENABLE_OPENMP=ON` -- OpenMP parallel gate kernels (on by default)

The build produces `build/libquantumsim.{dylib,so}`. To install system-wide:

```sh
sudo cmake --install build           # headers -> /usr/local/include/quantumsim/
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
moonlab = { path = "bindings/rust/moonlab" }
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

Set `MOONLAB_LIB_DIR=$(pwd)/build` if the binding cannot locate
`libquantumsim` at runtime.

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
The whole stack ships in `deploy/docker/`:

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
- `examples/` -- ~80 runnable demos across all bindings (start with
  `examples/basic/bell_state.c` for an extended version of the Bell circuit
  above with entropy and CHSH analysis)
- `bindings/python/README.md`, `bindings/rust/moonlab/README.md`,
  `bindings/javascript/README.md` -- language-specific deep dives
