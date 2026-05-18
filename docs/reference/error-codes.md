# Error codes reference

Moonlab uses a layered return-code convention rather than a single
flat error registry.  This document records the convention itself,
the canonical generic codes, and the per-module enums that subsystem
APIs return.  Every claim here cross-references the source header
that defines it; if a code is not listed below, it is not part of
the public surface.

## Convention

Defined in
[`src/utils/moonlab_status.h`](../../src/utils/moonlab_status.h):

- `0` always means success.
- Negative integers always mean failure.
- The codes `-1`, `-2`, `-3`, `-4` are reserved across the library
  for `INVALID`, `QUBIT`, `OOM`, `BACKEND` respectively.  Modules
  that don't use a concept simply skip that code.
- Module-specific extensions occupy `-100` and below, leaving the
  `(-99, 0)` window clear of cross-module collisions.

The generic typedef `moonlab_status_t` (an `int`, see
`src/utils/moonlab_status.h:51`) is for code that composes calls
across modules and wants a single return type.  Existing per-module
`*_error_t` enums are not deprecated -- new code may use either.

## Generic canonical codes

From `src/utils/moonlab_status.h:53-64`:

| Macro                              | Value | Meaning                       |
|------------------------------------|------:|-------------------------------|
| `MOONLAB_STATUS_SUCCESS`           |     0 | success                       |
| `MOONLAB_STATUS_ERR_INVALID`       |    -1 | invalid argument / state      |
| `MOONLAB_STATUS_ERR_QUBIT`         |    -2 | bad qubit index               |
| `MOONLAB_STATUS_ERR_OOM`           |    -3 | out of memory                 |
| `MOONLAB_STATUS_ERR_BACKEND`       |    -4 | backend operation failed      |
| `MOONLAB_STATUS_ERR_MODULE_BASE`   |  -100 | first module-extension code   |

Use `moonlab_status_ok(status)` (an inline helper in the same
header) to test for success when the per-module enum identity is not
in scope.

## Quantum-core return codes

`src/quantum/state.h:69-76` defines `qs_error_t`, the enum returned
by the state-vector and gate APIs in `src/quantum/`:

| Name                          | Value |
|-------------------------------|------:|
| `QS_SUCCESS`                  |     0 |
| `QS_ERROR_INVALID_QUBIT`      |    -1 |
| `QS_ERROR_INVALID_STATE`      |    -2 |
| `QS_ERROR_NOT_NORMALIZED`     |    -3 |
| `QS_ERROR_OUT_OF_MEMORY`      |    -4 |
| `QS_ERROR_INVALID_DIMENSION`  |    -5 |

`qs_error_t` is the return type of every `gate_*` function in
`src/quantum/gates.h` and of state-lifecycle helpers like
`quantum_state_init`, `quantum_state_clone`, `quantum_state_normalize`,
and `quantum_state_from_amplitudes` (see `src/quantum/state.h`).

## Per-module enums

Each subsystem header carries its own enum.  Codes `0`, `-1`, `-2`,
`-3`, `-4` keep the canonical meaning above wherever the module has
a corresponding concept; additional negative codes are
module-specific and documented in the listed header.

| Module                        | Enum                   | Header                                                  |
|-------------------------------|------------------------|---------------------------------------------------------|
| State vector + gates          | `qs_error_t`           | `src/quantum/state.h`                                   |
| MPDO noise sim                | `noise_mpdo_error_t`   | `src/quantum/noise_mpdo.h`                              |
| CA-MPS                        | `ca_mps_error_t`       | `src/algorithms/tensor_network/ca_mps.h`                |
| CA-MPS variational-D          | `ca_mps_error_t`       | `src/algorithms/tensor_network/ca_mps_var_d.h`          |
| CA-PEPS                       | `ca_peps_error_t`      | `src/algorithms/tensor_network/ca_peps.h`               |
| TN state (MPS / MPO)          | `tn_state_error_t`     | `src/algorithms/tensor_network/tn_state.h`              |
| TN gate dispatch              | `tn_gate_error_t`      | `src/algorithms/tensor_network/tn_gates.h`              |
| TN measurement                | `tn_measure_error_t`   | `src/algorithms/tensor_network/tn_measurement.h`        |
| Tensor primitive              | `tensor_error_t`       | `src/algorithms/tensor_network/tensor.h`                |
| Network contraction           | `contract_error_t`     | `src/algorithms/tensor_network/contraction.h`           |
| SVD compression               | `svd_compress_error_t` | `src/algorithms/tensor_network/svd_compress.h`          |
| Clifford backend              | `clifford_error_t`     | `src/backends/clifford/clifford.h`                      |
| State partition (MPI)         | `partition_error_t`    | `src/distributed/state_partition.h`                     |
| Distributed gates (MPI)       | `dist_gate_error_t`    | `src/distributed/distributed_gates.h`                   |
| MPI bridge                    | `mpi_bridge_error_t`   | `src/distributed/mpi_bridge.h`                          |
| GPU backend                   | `gpu_error_t`          | `src/optimization/gpu/gpu_backend.h`                    |

Notable module-specific extensions:

- `tn_state_error_t` adds `TRUNCATION (-5)`, `CONTRACTION_FAILED (-6)`,
  `NORMALIZATION (-7)`, `INVALID_CONFIG (-8)`, and
  `ENTANGLEMENT_TOO_HIGH (-9)` for SVD / bond-cap related failures
  (`src/algorithms/tensor_network/tn_state.h:122-132`).
- `tensor_error_t` exposes the widest range (-1 .. -13), separating
  rank/dimension/index errors, SVD failure, GPU sync failure, and
  `NOT_IMPLEMENTED (-9)` (`src/algorithms/tensor_network/tensor.h:66-80`).
- `ca_peps_error_t` uses `CA_PEPS_ERR_NOT_IMPLEMENTED = -100` for
  CA-PEPS surfaces still on the roadmap
  (`src/algorithms/tensor_network/ca_peps.h:51-57`).
- `mpi_bridge_error_t` adds `TIMEOUT (-7)` for collective ops with
  a deadline (`src/distributed/mpi_bridge.h:82-90`).

## Diagnostic helper

`src/utils/moonlab_status.h:98` declares:

```c
const char *moonlab_status_to_string(moonlab_status_module_t module,
                                     moonlab_status_t status);
```

The module enum (`moonlab_status_module_t`, defined in the same
header at `:73-89`) identifies which per-module table the code
should be looked up in.  Pass `MOONLAB_MODULE_GENERIC` to print the
canonical codes (success / `INVALID` / `QUBIT` / `OOM` / `BACKEND`).

Modules that maintain their own `*_error_string` helper (e.g.
`contract_error_string` in `contraction.h:460`,
`svd_compress_error_string` in `svd_compress.h:359`) are equivalent
single-module shortcuts.

## Usage

### C

```c
#include "quantum/state.h"
#include "quantum/gates.h"
#include "utils/moonlab_status.h"

quantum_state_t state;
qs_error_t status = quantum_state_init(&state, 4);
if (status != QS_SUCCESS) {
    fprintf(stderr, "init failed: %s\n",
            moonlab_status_to_string(MOONLAB_MODULE_GENERIC,
                                     (moonlab_status_t)status));
    return 1;
}

status = gate_hadamard(&state, 0);
if (status != QS_SUCCESS) {
    /* handle */
}
```

### Python

The Python bindings raise `QuantumError`
(`bindings/python/moonlab/core.py:245`) on non-zero return codes
from the C ABI; callers see exceptions, not raw negative integers.

### Rust

The Rust bindings convert C return codes into
`Result<T, QuantumError>` at the function boundary; the variant
hierarchy is defined in `bindings/rust/moonlab/src/error.rs` (see
the `pub enum QuantumError` declaration).

## See also

- [Configuration options](configuration-options.md) -- runtime
  configuration knobs whose validators surface these codes.
- [TDVP API](tdvp-api.md) and [QGT API](qgt-api.md) for examples
  of subsystem-level error reporting in algorithm modules.
