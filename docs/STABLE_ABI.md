# Stable ABI contract -- v1.x

**Current package:** 1.2.1
**Current ABI:** 0.7.0

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

The table below counts the `MOONLAB_API` declarations in each public header.
`scripts/check_stable_abi_counts.sh` verifies it against the source and fails
the release smoke on any mismatch, which is new in v1.2.1 and was overdue: ten
of these rows had drifted, some badly (`grover.h` read 5 against 14 real
declarations, `quantum/state.h` 10 against 25, `quantum/gates.h` 17 against 34)
because nothing tied the prose to the headers. A wrong count is not cosmetic --
this table is what a downstream consumer reads to size the surface they are
binding against.

As of ABI 0.6.0 the second-tier binding surface (the `moonlab_diff_*` autograd
API, the dense state/gate/measurement calls, and the distributed headers) is
also `MOONLAB_API`-tagged, so it survives a hidden-visibility build. The
authoritative current list is always every `MOONLAB_API` declaration in a
public header; `tests/abi/test_moonlab_export_abi.c` loads and functionally
smokes the committed export surface on Unix and Windows.

### ABI 0.5.0 and 0.6.0 additions

ABI 0.5.0 added `moonlab_qrng_get_status` and the `moonlab_qrng_status_t`
conditioned-QRNG status contract.

ABI 0.6.0 (this release):

- Promoted `moonlab_ca_mps_conjugate_pauli` to the stable surface, declared in
  `moonlab_export.h` with a plain `int` return (independent of the internal
  `ca_mps_error_t` enum).
- Declared the seven QGT topology one-shots already implemented in
  `moonlab_export_lean.c`: `moonlab_ssh_winding`, `moonlab_kitaev_chain_z2`,
  `moonlab_chern_qwz_proj`, `moonlab_chern_qwz_pt`, `moonlab_kane_mele_z2`,
  `moonlab_bhz_z2`, `moonlab_hofstadter_chern`.
- Introduced the `moonlab_complex_double` header-boundary carrier so
  `moonlab_export.h` parses under MSVC (which lacks C99 `double _Complex`)
  while keeping the C99 ABI byte-for-byte identical; the CA-MPS observable
  signatures use it through pointers only.
- Promoted the second tier of binding-consumed symbols to `MOONLAB_API` at
  their declaration sites (the `quantum_state_*`, `gate_*`, `measurement_*`,
  `moonlab_diff_*`, QAOA, Grover, surface-code, and distributed families), so
  the full binding + Eshkol-RFC surface survives a `QSIM_HIDDEN_VISIBILITY=ON`
  build.

| Header                                       | Symbols |
|----------------------------------------------|---------|
| `src/applications/moonlab_export.h`          |     84  |
| `src/algorithms/tensor_network/ca_mps.h`     |     37  |
| `src/algorithms/vqe.h`                       |     37  |
| `src/distributed/scheduler.h`                |     26  |
| `src/control/control_plane.h`                |     21  |
| `src/qec/stim_circuit.h`                     |     20  |
| `src/integration/libirrep_bridge.h`          |     19  |
| `src/algorithms/quantum_geometry/qgt.h`      |     32  |
| `src/quantum/gates.h`                        |     34  |
| `src/quantum/noise_mpdo.h`                   |     15  |
| `src/qec/stim_dem.h`                         |     15  |
| `src/crypto/mlkem/mlkem.h`                   |     21  |
| `src/algorithms/qaoa.h`                      |     30  |
| `src/applications/moonlab_qgtl_backend.h`    |     11  |
| `src/quantum/state.h`                        |     25  |
| `src/applications/decoder_bench.h`           |      9  |
| `src/utils/audit_buffer.h`                   |      7  |
| `src/applications/vendor_noise_backend.h`    |      7  |
| `src/algorithms/topology_realspace/chern_kpm.h`|      7  |
| `src/algorithms/bell_tests.h`                |      9  |
| `src/algorithms/grover.h`                    |     14  |
| `src/utils/token_bucket.h`                   |      4  |
| `src/backends/clifford/clifford.h`           |     18  |
| (others: `quantum_entropy.h`, `entanglement.h`, `mwpm_exact.h`) | 4 |

The authoritative list is the source.  This doc is not the
catalogue -- treat the `MOONLAB_API` annotation as the contract.

### Stability tiers

`MOONLAB_API` is the ABI contract; the `@stability` tag in a symbol's
doc comment says how settled its *semantics* are. Three tiers are in
use, and no others:

| Tier        | Meaning                                                        |
|-------------|----------------------------------------------------------------|
| `stable`    | Frozen. Signature and semantics break only at a major version.  |
| `evolving`  | Signature is stable within the minor series; semantics may extend compatibly. |
| `beta`      | May change in any release. No compatibility promise. Promotion to `evolving` requires no signature change at the time of promotion. |

The tiers are normative, not descriptive: they state what a consumer may
rely on, and the release process is bound by them.

How the tier is assigned:

- **`stable`** -- every symbol declared in
  `src/applications/moonlab_export.h`. That header is the committed
  downstream ABI, versioned by `MOONLAB_ABI_VERSION_*` and pinned by
  `tests/abi/test_moonlab_export_abi.c`, which dlopens and smokes the
  surface exactly as a consumer does. A symbol declared both there and in
  its module header carries `stable` at both declaration sites; the two
  must not disagree.
- **`evolving`** -- every other `MOONLAB_API` symbol that has shipped in a
  tagged release. These are covered by the v1.x no-break policy above.
- **`beta`** -- `MOONLAB_API` symbols added since the last tagged release
  and not yet shipped. This is the only tier whose signature may still
  move, and it is where a symbol lives during the release cycle that
  introduces it.

An `evolving` or `stable` symbol that has shipped cannot be removed,
renamed, or re-signatured before v2.0. A `beta` symbol carries no such
promise until it ships, which is exactly why the tier exists: the
alternative is to freeze a signature before anyone has used it.

### Stim ecosystem interop additions

Thirty-seven new symbols across three headers, all additive. No existing
signature changed, and `moonlab_export.h` is untouched, so the
`MOONLAB_ABI_VERSION_*` triple of the stable downstream export surface is
unchanged.

`src/qec/stim_circuit.h` -- 20 symbols. Import and export of Stim's `.stim`
circuit format, plus lowering onto the Pauli-frame sampler:

  - `moonlab_stim_circuit_parse`, `..._parse_file`, `..._free`,
    `moonlab_stim_circuit_to_text`, `moonlab_stim_text_free` -- the format
    round trip. Serialisation takes a `flatten` flag selecting whether
    `REPEAT` blocks are preserved or expanded; both forms re-parse to a
    semantically identical circuit.
  - `moonlab_stim_circuit_num_qubits`, `..._num_measurements`,
    `..._num_detectors`, `..._num_observables`, `..._num_ticks`,
    `..._qubit_coords`, `..._detector_coords` -- introspection, with
    `SHIFT_COORDS` offsets already folded into the reported coordinates.
  - `moonlab_stim_circuit_num_ops`, `..._num_channel_args`, `..._lower`,
    `..._detector_csr`, `..._observable_csr`,
    `..._measurement_inversions` -- lowering to `pf_circuit_op_t` plus the
    CSR parity sets the detector sampler and the decoders consume.
  - `moonlab_stim_circuit_sample_measurements`,
    `..._sample_detectors` -- one-call sampling, mirroring Stim's
    `compile_sampler()` and `compile_detector_sampler()` layouts.
  - `moonlab_stim_status_t` and `moonlab_stim_error_t` carry the failure
    code, 1-based source line, and a message naming the offending token.
    Unsupported input is always rejected, never skipped.

`src/qec/stim_dem.h` -- 15 symbols. Import and export of Stim's detector
error model format against moonlab's edge-list decoders:

  - `moonlab_dem_parse`, `..._parse_file`, `..._free`, `moonlab_dem_to_text`,
    `moonlab_dem_text_from_edges` -- the DEM round trip. Export is the exact
    inverse of the import merge, so an edge list survives a trip through DEM
    text, and the emitted text loads directly into PyMatching.
  - `moonlab_dem_num_detectors`, `..._num_observables`, `..._num_edges`,
    `..._num_correlations`, `..._num_hyperedges`, `..._num_detectorless`,
    `..._edges`, `..._correlations`, `..._detector_coords` -- the edge-list
    view in `moonlab_uf_decoder_new()` layout. Components that are not
    graphlike are counted and reported rather than dropped.
  - `moonlab_dem_make_uf_decoder` -- builds a plain or correlated decoder
    straight from a parsed model, so Stim's `^` decompositions reach the
    two-pass correlated decoder as correlation links.

`src/backends/clifford/pauli_frame.h` -- 2 symbols, plus two additive
`pf_op_kind_t` enumerators:

  - `pauli_frame_batch_sample_circuit_ex`,
    `pauli_frame_batch_sample_detectors_ex` -- the existing samplers plus a
    channel-argument table. The original entry points delegate to these and
    are unchanged.
  - `PF_OP_PAULI_CHANNEL_1 = 17` and `PF_OP_PAULI_CHANNEL_2 = 18` extend
    `pf_op_kind_t` additively; values 0..16 keep their numbers, and
    `pf_circuit_op_t` keeps its layout, so the JS binding's mirrored
    `PfOpKind` and every existing op array stay valid. The channel ops carry
    the base index of their probabilities in the table in
    `pf_circuit_op_t::p`, which is why no struct field was added.

@stability beta for all thirty-seven: the signatures are frozen by the ABI
policy above, and the tier will move once the format coverage has had a
release in the field.

### v1.2.1 additions

`src/algorithms/vqe.h` (quantum-geometry verbs and the custom-ansatz
constructor; all additive, no existing signature changed):

  - `vqe_compute_berry_curvature(solver, parameters, berry_out)` --
    the antisymmetric imaginary half of the quantum geometric tensor,
    `F_ij = -2 Im[<d_i psi|d_j psi> - <d_i psi|psi><psi|d_j psi>]`,
    sharing the exact analytic derivatives of `vqe_compute_qgt`.
  - `vqe_create_custom_ansatz(num_qubits, num_parameters, apply,
    user_data)` + the `vqe_custom_ansatz_fn` callback typedef --
    constructs a `VQE_ANSATZ_CUSTOM` ansatz from a caller-supplied
    circuit. `vqe_apply_ansatz` routes to it, so the whole ideal-state
    surface (`vqe_compute_energy`, `vqe_compute_gradient`, `vqe_solve`,
    `vqe_compute_qgt`, `vqe_compute_berry_curvature`) accepts it. The
    geometric verbs differentiate a custom circuit by central
    differences, since its gate structure is opaque to the library.
    `vqe_apply_ansatz_noisy` routes to the callback too, handing it the
    noise model and entropy source: the callback owns its gate sequence,
    so it is the only place the per-gate error channels can be
    interleaved correctly.
  - `vqe_apply_single_qubit_noise(state, qubit, noise, entropy)` and
    `vqe_apply_two_qubit_noise(state, q1, q2, noise, entropy)` -- the
    exact channel composition the built-in noisy ansaetze apply after
    each gate, exposed so a custom circuit reproduces built-in noise
    semantics rather than approximating them. No-ops when `noise` is
    NULL, so one circuit body serves the ideal and the noisy path.

`src/algorithms/quantum_geometry/qgt.h` (pointwise two-band geometry;
all additive, no existing signature changed):

  - `qgt_dsigma_metric_curvature(d, dx, dy, g, omega_xy)` -- closed-form,
    gauge-free lower-band Fubini-Study metric and Berry curvature of
    `H(k) = d(k).sigma` from the analytic `d` and its momentum gradients.
    Machine-precision: no eigenvector, no finite difference.
  - `qgt_curvature_at(sys, k, dk, g, omega_xy)` -- the same tensor for any
    2-band Bloch callback via the projector trace
    `Q = Tr[P- dH_mu P+ dH_nu]/(DeltaE)^2`, with `dH` from a central
    difference of the Hamiltonian. Gauge-free, `O(dk^2)`-accurate.
  - `qgt_set_dsigma(sys, f)`, `qgt_dsigma_at(sys, k, d, dx, dy)`,
    `qgt_exact_curvature_at(sys, k, g, omega_xy)` + the `qgt_dsigma_fn`
    callback typedef -- the analytic d-vector a system may carry, and the
    routed exact path. `qgt_model_qwz` and `qgt_model_haldane` populate it
    at construction, so the closed form is reachable from a model handle
    without the caller re-deriving `d(k)`. A system with no analytic
    d-vector returns `-3` from both accessors.
  - `qgt_create_dsigma(bloch, dsigma, user)` -- `qgt_create` plus
    `qgt_set_dsigma` in one call, so an FFI consumer does not hold the
    handle across two calls and unwind the first on the second's failure.

Both entry points return `-2` at a band touching, tested relative to the
scale of their inputs (`1e-12` times the Hamiltonian trace scale, matching
the module's existing `lower_eigvec_2x2` threshold, or `1e-12` times the L1
scale of `d` and its gradients).

`src/applications/moonlab_export.h` (ABI 0.7.0) promotes six of these to the
committed downstream surface, so QGTL, libirrep, and SbNN reach the quantum
geometry over FFI without marshalling an opaque handle:

  - `moonlab_vqe_qgt(solver, parameters, qgt_out, num_parameters)` --
    the Fubini-Study metric, written **row-major**: element (i, j) at index
    `i * num_parameters + j`. The matrix is symmetric so the layout is its
    own transpose, but the index formula is the committed contract. This is
    the entry the Eshkol custom-VJP tape node binds against.
  - `moonlab_vqe_berry_curvature(...)` -- same shape, the curvature half.
  - `moonlab_vqe_natural_gradient(solver, parameters, gradient,
    regularization, direction_out, num_parameters)`.
  - `moonlab_dsigma_metric_curvature(d, dx, dy, g_out, omega_out)`.
  - `moonlab_qwz_curvature_at(m, k, g_out, omega_out)` and
    `moonlab_haldane_curvature_at(t1, t2, phi, m_stagger, k, g_out,
    omega_out)` -- model parameters in, arrays out, following the ABI 0.6.0
    topology one-shot convention exactly.

All three VQE entries validate the caller's `num_parameters` against the
solver's ansatz and return -1 / -2 / -3 for NULL argument / count mismatch /
internal failure, matching `moonlab_vqe_gradient`. The band-geometry entries
pass through the module status codes: 0, -1 on a bad argument, -2 at a band
touching. `tests/abi/test_moonlab_export_abi.c` dlopens and exercises all six.

Everything else added in this cycle lands in `src/algorithms/vqe.h` and
`src/algorithms/quantum_geometry/qgt.h`, which are version-pinned rather than
frozen.

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
| `-501` ... `-507` | scheduler                  | `src/distributed/scheduler.h`    |
| `-401` ... `-404` | decoder bench              | `src/applications/decoder_bench.h` |

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
| C        | `libquantumsim.{so,dylib,dll}`    | package 1.2.0, stable ABI 0.6.0 |
| Python   | `moonlab` (pip)                   | follows the package version (1.2.0) |
| Rust     | `moonlab` + `moonlab-sys` crates  | follows the package version (1.2.0) |
| JS       | `@tsotchkecorp/moonlab`           | follows the package version (1.2.0) |

Each binding crate/package revs alongside the C library's package version
(currently 1.2.0) and stays within the same 1.x compatibility line as the
stable C ABI (currently 0.6.0). Breaking changes in the language idiom of a
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
