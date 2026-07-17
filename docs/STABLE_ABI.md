# Stable ABI contract -- v1.x

**Current package:** 1.1.0
**Current ABI:** 0.5.0

## Scope

This document defines what MoonLab 1.x guarantees to consumers of
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

### Header install namespaces

The full source-shaped header tree installs into
`<prefix>/include/quantumsim/`, a residue from the project's pre-rename name
when the C library was called `libquantumsim`. MoonLab 1.x keeps this path so
existing consumers do not break. The stable downstream ABI headers also
install under `<prefix>/include/moonlab/` for new consumers.

Application code includes headers as:

```c
#include <quantumsim/quantum/state.h>
#include <quantumsim/control/control_plane.h>
#include <moonlab/moonlab_export.h>
```

Neither installed namespace may move incompatibly inside the 1.x line.

## Symbol catalog by module

The table below is a current snapshot (355 `MOONLAB_API`-tagged declarations
across the tree as of ABI 0.5.0), refreshed against `moonlab_export.h` and
every public header. It supersedes the earlier v1.0.3 snapshot this document
used to carry, which undercounted `moonlab_export.h` itself (36 listed vs.
70 actual) and predated `moonlab_vqe_gradient` (ABI 0.4.0) and
`moonlab_qrng_get_status` plus the conditioned QRNG contract (ABI 0.5.0). The
authoritative current list is always every `MOONLAB_API` declaration in a
public header; `tests/abi/test_moonlab_export_abi.c` loads and functionally
smokes the committed export surface on Unix and Windows.

| Header                                       | Symbols |
|----------------------------------------------|---------|
| `src/applications/moonlab_export.h`          |  70     |
| `src/algorithms/tensor_network/ca_mps.h`     |  37     |
| `src/algorithms/vqe.h`                       |  30     |
| `src/distributed/scheduler.h`                |  26     |
| `src/control/control_plane.h`                |  21     |
| `src/integration/libirrep_bridge.h`          |  19     |
| `src/algorithms/quantum_geometry/qgt.h`      |  18     |
| `src/quantum/gates.h`                        |  17     |
| `src/quantum/noise_mpdo.h`                   |  15     |
| `src/crypto/mlkem/mlkem.h`                   |  15     |
| `src/algorithms/qaoa.h`                      |  13     |
| `src/applications/moonlab_qgtl_backend.h`    |  11     |
| `src/quantum/state.h`                        |  10     |
| `src/applications/decoder_bench.h`           |   9     |
| `src/utils/audit_buffer.h`                   |   7     |
| `src/applications/vendor_noise_backend.h`    |   7     |
| `src/algorithms/topology_realspace/chern_kpm.h` | 7   |
| `src/algorithms/bell_tests.h`                |   6     |
| `src/algorithms/grover.h`                    |   5     |
| `src/utils/token_bucket.h`                   |   4     |
| `src/backends/clifford/clifford.h`           |   4     |
| (others: `quantum_entropy.h`, `entanglement.h`, `mwpm_exact.h`) | 4 |

The authoritative list is the source.  This doc is not the
catalogue -- treat the `MOONLAB_API` annotation as the contract.

### v1.0.3 additions

Three modules grew runtime registries; eleven new public symbols.
All are frozen by the v1.0 ABI policy below: new names and meanings
are stable across the v1.x line.

`scheduler.h`:
  - `moonlab_scheduler_set_completion_hook(fn, ctx)` + the
    `moonlab_completion_hook_fn` typedef.

`vendor_noise_backend.h`:
  - `moonlab_register_vendor_noise_profile(name, profile)`
  - `moonlab_unregister_vendor_noise_profile(name)`
  - `moonlab_lookup_vendor_noise_profile(name)`
  - `moonlab_num_vendor_noise_profiles()`
  - `moonlab_list_vendor_noise_profiles(out_names, max)`

`decoder_bench.h`:
  - `moonlab_register_decoder(name, fn, ctx, description)`
    + the `moonlab_decoder_fn` typedef
    + the `moonlab_decoder_entry_t` struct
  - `moonlab_unregister_decoder(name)`
  - `moonlab_lookup_decoder(name)`
  - `moonlab_decoder_decode_by_name(name, in)`
  - `moonlab_num_decoders()`
  - `moonlab_list_decoders(out_names, max)`

#### v1.0.3 multi-tenant additions (second wave)

Seven additional public symbols added during the multi-tenant
arc, all frozen under the same v1.x ABI policy:

`control_plane.h`:
  - `moonlab_control_submit_circuit_auth_tenant(host, port,
    tenant_id, secret, secret_len, body, body_len,
    out_probs, out_n)`
  - `moonlab_control_server_set_admission_hook(server, fn, ctx)`
    + the `moonlab_admission_hook_fn` typedef

`scheduler.h`:
  - `moonlab_scheduler_set_request_context(tenant_id, request_id)`
  - `moonlab_scheduler_current_tenant_id()`
  - `moonlab_scheduler_current_request_id()`
  - `moonlab_scheduler_fire_completion_hook(job, results,
    backend_name)`

`utils/token_bucket.h`:
  - `moonlab_token_bucket_init(bkt, burst, refill_per_sec)`
  - `moonlab_token_bucket_take(bkt, n)`
  - `moonlab_token_bucket_refill(bkt, n)`
  - `moonlab_token_bucket_peek(bkt)`
    + the `moonlab_token_bucket_t` struct (caller-owned storage)

`utils/audit_buffer.h`:
  - `moonlab_audit_buffer_init(buf, slots, record_size, capacity)`
  - `moonlab_audit_buffer_destroy(buf)`
  - `moonlab_audit_buffer_push(buf, record)` -> 1 clean / 0 dropped
  - `moonlab_audit_buffer_pop(buf, out)` -> 1 on success / 0 empty
  - `moonlab_audit_buffer_len(buf)`
  - `moonlab_audit_buffer_drops(buf)`
  - `moonlab_audit_buffer_reset_drops(buf)`
    + the `moonlab_audit_buffer_t` struct (caller-owned storage +
      caller-owned slots block; capacity does NOT need to be a
      power of two -- mutex-guarded ring uses `% capacity`).
    + struct layout includes a `pthread_mutex_t lock` and an
      `_Atomic int state` field added in v1.0.5; do NOT rely on
      the layout being identical across patch versions, since the
      mutex/state are implementation details.  Treat the struct
      as caller-allocated opaque storage and access it ONLY via
      the public API.  Caller MUST zero-initialise the struct
      before the first `init()` (e.g. `audit_buffer_t b = {0};`).

See `docs/EXTENSION_SURFACES.md` for the integration guide that
shows each surface with C / Python / Rust / JavaScript snippets.
See `examples/extensions/open_core_overlay_demo.c` (C) and
`examples/extensions/python_overlay_demo.py` (Python) for runnable
overlays that exercise the full plug-in arc.

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

| Language | Crate / package                   | Current version |
|----------|-----------------------------------|------------------|
| C        | `libquantumsim.{so,dylib,dll}`    | package 1.1.0, stable ABI 0.5.0 |
| Python   | `moonlab` (pip)                   | follows the package version (1.1.0) |
| Rust     | `moonlab` + `moonlab-sys` crates  | follows the package version (1.1.0) |
| JS       | `@moonlab/quantum-core`           | follows the package version (1.1.0) |

Each binding crate/package revs alongside the C library's package version
(currently 1.1.0) and stays within the same 1.x compatibility line as the
stable C ABI (currently 0.5.0). Breaking changes in the language idiom of a
single binding (e.g. switching Rust's `Vec<f64>` to `Box<[f64]>`) are
allowed but rare; each binding's CHANGELOG records them with semver
discipline.

## See also

- `docs/PARITY_MATRIX.md` -- which capability is wired through which
  binding
- `docs/CONTROL_PLANE.md` -- wire protocol reference + ops runbook
- `docs/reference/error-codes.md` -- per-module status code listing
- `docs/INTEGRATION_libirrep_SbNN.md` -- v1.0 commitment for the two
  optional sibling-library bridges
