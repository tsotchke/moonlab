# Stable ABI contract -- v1.0

## Scope

This document defines what moonlab v1.0 guarantees to consumers of
the C library (the FFI surface) and to consumers of the four binding
languages (Python / Rust / JS / wire protocol).

**The contract**: every C symbol tagged `MOONLAB_API` in a public
header under `src/` is part of the v1.0 surface.  Within the 1.x
release series moonlab MAY:

- add new `MOONLAB_API` symbols
- add new fields to opaque types (caller never sees the layout)
- add new enum values in additive-only fashion
- accept inputs that previously returned `MOONLAB_*_BAD_ARG` (loosen)

moonlab MAY NOT (without a v2.0 major bump):

- remove or rename an existing `MOONLAB_API` symbol
- change a function signature
- change the numeric value of an existing enum entry
- reject inputs that previously succeeded (tighten)
- change the meaning of a return code

The same guarantees apply transitively to the Python / Rust / JS
binding surfaces, modulo idiomatic adaptation (e.g. Rust's
`Result<T, Error>` vs C's `int` return code).

### Header install namespace (v1.0 historical wart)

Headers install into `<prefix>/include/quantumsim/` (not
`/moonlab/`) -- a residue from the project's pre-rename name when
the C library was called `libquantumsim`.  v1.0 keeps the
`quantumsim/` install path so existing FFI consumers don't break;
the `moonlab` brand applies to the project, the binding crates,
and the public API symbols (`MOONLAB_API`, `moonlab_*`), not the
include directory.

Application code includes headers as:

```c
#include <quantumsim/quantum/state.h>
#include <quantumsim/control/control_plane.h>
```

A migration to `<moonlab/...>` is on the v2.0 list; this is a
v1.0 ABI-stability commitment (the include path cannot move
inside the 1.x line).

## Symbol catalog by module

`grep -rn "^MOONLAB_API " src/ --include="*.h"` returns 224
declarations as of v1.0.  Distribution by header (highest first):

| Header                                       | Symbols |
|----------------------------------------------|---------|
| `src/applications/moonlab_export.h`          |  36     |
| `src/integration/libirrep_bridge.h`          |  19     |
| `src/control/control_plane.h`                |  19     |
| `src/algorithms/quantum_geometry/qgt.h`      |  18     |
| `src/quantum/gates.h`                        |  17     |
| `src/quantum/noise_mpdo.h`                   |  15     |
| `src/algorithms/tensor_network/ca_mps.h`     |  15     |
| `src/distributed/scheduler.h`                |  14     |
| `src/algorithms/qaoa.h`                      |  13     |
| `src/algorithms/vqe.h`                       |  12     |
| `src/applications/moonlab_qgtl_backend.h`    |  11     |
| `src/quantum/state.h`                        |   7     |
| `src/algorithms/topology_realspace/chern_kpm.h` | 7   |
| `src/algorithms/bell_tests.h`                |   6     |
| `src/algorithms/grover.h`                    |   5     |
| `src/backends/clifford/clifford.h`           |   4     |
| `src/applications/decoder_bench.h`           |   3     |
| (others, 1 each)                             |   3     |

The authoritative list is the source.  This doc is not the
catalogue -- treat the `MOONLAB_API` annotation as the contract.

## Wire protocol contract

The control-plane line protocol is frozen at v1.0.  See
`docs/CONTROL_PLANE.md` for the full reference; the v1.0 guarantees:

- Verbs `CIRCUIT`, `SHOTS`, `HEALTH`, `METRICS`, `AUTH`,
  `CIRCUIT_AUTH` keep their existing wire formats.
- Reply framing tokens (`OK <count>`, `SAMPLES <count>`,
  `METRICS <bytes>`, `ERR <code> <msg>`) keep their existing units.
- Status codes `MOONLAB_CONTROL_*` (-400, -401, -402, -403, -405,
  -407, -408, -409) are stable.  New codes may be added in the
  -4xx range; existing codes do not change meaning.

Adding a new verb is a minor-version change.  Removing or renaming a
verb is a major-version change.

## Status code catalogues

Each module defines an enum of return codes.  At v1.0 the following
ranges are frozen:

| Range          | Module                        | Header                           |
|----------------|-------------------------------|----------------------------------|
| `0`            | success (all modules)         | -                                |
| `-1` ... `-4`  | generic (`moonlab_status_t`)  | `src/utils/moonlab_status.h`     |
| `-100` ...     | per-module extensions         | per-module                       |
| `-201`         | `MOONLAB_LIBIRREP_NOT_BUILT`  | `src/integration/libirrep_bridge.h` |
| `-301` ...     | QGTL backend                  | `src/applications/moonlab_qgtl_backend.h` |
| `-400` ... `-409` | control plane              | `src/control/control_plane.h`    |
| `-501` ... `-504` | scheduler                  | `src/distributed/scheduler.h`    |
| `-601` ...     | decoder bench                 | `src/applications/decoder_bench.h` |

The full mapping is in `docs/reference/error-codes.md`.  Within 1.x,
adding a new code in an existing range is additive; changing the
meaning of an existing code is a v2.0-only operation.

## What's NOT covered

The contract intentionally does NOT cover:

- **Internal struct layouts**.  Every opaque pointer
  (`moonlab_ca_mps_t`, `moonlab_control_server_t`, etc.) is opaque
  even within 1.x.  Inspecting via sizeof or pointer arithmetic is
  undefined.
- **Performance characteristics**.  Algorithmic complexity bounds
  documented in the header @brief blocks are guidance, not promises;
  moonlab may choose a slower-asymptotic-better-constant
  implementation if it gives faster wall-clock on the realistic
  workload.
- **Behaviour under undocumented inputs**.  Passing a NULL where the
  doc says "must be non-NULL" is undefined.
- **Opt-in build flags**.  `QSIM_ENABLE_LIBIRREP`, `QSIM_ENABLE_MPI`,
  `QSIM_ENABLE_TLS`, `QSIM_ENABLE_CUDA`, `QSIM_ENABLE_OPENCL`,
  `QSIM_ENABLE_VULKAN`, `QSIM_ENABLE_CUQUANTUM`, `QSIM_ENABLE_ESHKOL`,
  `QSIM_WERROR` may change default value or be renamed across 1.x.
- **Environment variables**.  `MOONLAB_CONTROL_LOG`,
  `MOONLAB_CONTROL_LOG_FORMAT` are guidance; values may be widened.

## Deprecation policy

Within 1.x, a symbol marked deprecated continues to work but emits a
build-time `[[deprecated]]` warning when consumed.  The symbol may be
removed in the next major version.

```c
[[deprecated("renamed to foo_v2; will remove in 2.0")]]
MOONLAB_API int moonlab_foo(...);

/* New replacement. */
MOONLAB_API int moonlab_foo_v2(...);
```

Bindings (Python / Rust / JS) follow their idiomatic deprecation
mechanism (Python `DeprecationWarning`, Rust `#[deprecated]`, JS
`@deprecated` JSDoc tag).

## Binding-language ABIs

| Language | Crate / package                   | Semver        |
|----------|-----------------------------------|---------------|
| C        | `libquantumsim.{so,dylib,dll}`    | v1.0          |
| Python   | `moonlab` (pip)                   | follows v1.0  |
| Rust     | `moonlab` + `moonlab-sys` crates  | follows v1.0  |
| JS       | `@moonlab/quantum-core`           | follows v1.0  |

When `moonlab` v1.0 ships, the binding crates rev to a 1.x compatible
version (e.g. `moonlab-rs 1.0.0`, `@moonlab/quantum-core 1.0.0`).
Breaking changes in the language idiom of a single binding (e.g.
switching Rust's `Vec<f64>` to `Box<[f64]>`) are allowed but rare;
each binding's CHANGELOG records them with semver discipline.

## See also

- `docs/PARITY_MATRIX.md` -- which capability is wired through which
  binding
- `docs/CONTROL_PLANE.md` -- wire protocol reference + ops runbook
- `docs/reference/error-codes.md` -- per-module status code listing
- `docs/INTEGRATION_libirrep_SbNN.md` -- v1.0 commitment for the two
  optional sibling-library bridges
