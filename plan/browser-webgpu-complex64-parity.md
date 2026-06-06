# Browser WebGPU Complex64 Parity Blocker

Status: implementation note for the MoonLab `ulg` branch. No browser WebGPU
runtime patch is applied here because the available backend work lives on a
divergent `webgpu` branch and the precision/parity contract is not yet
accepted for the reduced ULG fixture path.

## Current Branch State

- `bindings/javascript/packages/core` on `ulg` has no browser WebGPU runtime
  source, no JS WebGPU API export, no `HAS_WEBGPU` build switch, no Asyncify
  wiring, and no `_gpu_*` / `_tensor_gpu_webgpu_available` exports.
- Current branch WebGPU artifacts are stale no-backend evidence only:
  - `bindings/javascript/packages/core/artifacts/webgpu_smoke/results.json`
  - `bindings/javascript/packages/core/artifacts/webgpu_unified_eval/results.json`
  - `bindings/javascript/packages/core/artifacts/webgpu_eval/results.json`
- The active magnetar runtime contract remains reduced fixture plumbing:
  `ulg.magnetar.fidelity-runtime-scope.v0`,
  `fullFidelityMagnetarSimulation = false`, and
  `fullPhysicsValidation = false`.

## Blocker

The old `webgpu` branch backend uses complex64 browser buffers while the
current MoonLab CPU/WASM reference path uses float64 interleaved amplitudes.
In `webgpu:src/optimization/gpu/backends/gpu_webgpu.c`, the WGSL kernels store
amplitudes and probabilities as `array<vec2<f32>>`; the JS bridge copies
`HEAPF64` into `Float32Array`, dispatches the GPU work, and writes `Float32`
results back into `HEAPF64`.

The old parity script
`webgpu:bindings/javascript/packages/core/scripts/webgpu-unified-eval.mjs`
compares those complex64 GPU results against a float64 CPU reference with an
absolute tolerance of `1e-5`. It also includes `phase` in random operation
streams while the old WebGPU backend routes phase through a CPU fallback, so
native WebGPU coverage can be mixed with fallback coverage unless it is tracked
per operation.

That makes a direct backend port unsafe for this branch. It would introduce a
precision downgrade and a broad Emscripten/browser runtime surface before the
reduced fixture contract states what WebGPU parity means.

## Smallest Safe Next Patch

Add a browser-only precision/parity probe before enabling runtime acceleration.
The probe should emit a deterministic reduced fixture artifact rather than a
full physics claim.

Suggested artifact contract:

```json
{
  "schema": "moonlab.webgpu.complex64-parity-scope.v0",
  "backend": "webgpu",
  "amplitudeRepresentation": "complex64-interleaved-f32",
  "referenceRepresentation": "wasm-float64-interleaved",
  "maxProbabilityAbsDiff": 0.00001,
  "fullFidelityMagnetarSimulation": false,
  "fullPhysicsValidation": false,
  "reducedFixtureOnly": true,
  "nativeCoverageRequired": [
    "hadamard",
    "pauli_x",
    "pauli_z",
    "cnot",
    "compute_probabilities"
  ],
  "nativeCoverageExcluded": [
    "phase"
  ]
}
```

Implementation slice:

1. Add a browser-gated parity probe script under
   `bindings/javascript/packages/core/scripts/`, for example
   `webgpu-complex64-parity.mjs`.
2. Add a small TypeScript contract builder under
   `bindings/javascript/packages/core/src/`, for example
   `webgpu-complex64-parity.ts`, that records the fields above and per-op
   native/fallback coverage.
3. Add a package script such as `webgpu:complex64:parity` that requires a real
   browser WebGPU adapter when `MOONLAB_WEBGPU_PARITY_REQUIRE_BACKEND=1` is set
   and otherwise writes an explicit `backendAvailable = false` artifact.
4. Only after that artifact exists, port the old branch C backend in a separate
   commit with explicit `HAS_WEBGPU`, Asyncify, export-list, and browser-only
   tests.

The first parity probe should avoid claiming magnetar simulation validation.
It should only demonstrate that reduced deterministic quantum fixtures can
round-trip through browser complex64 WebGPU kernels within the accepted
tolerance.

## Acceptance Commands

Current docs-only note:

```sh
git diff --check
```

Future parity patch:

```sh
pnpm --dir bindings/javascript/packages/core build:ts
pnpm --dir bindings/javascript/packages/core exec vitest run src/__tests__/ulg-quantum-response-artifact.test.ts
MOONLAB_WEBGPU_PARITY_REQUIRE_BACKEND=1 pnpm --dir bindings/javascript/packages/core webgpu:complex64:parity
```

## Risks

- `complex64` browser results may pass reduced fixtures while still being
  inappropriate for broader float64-sensitive workloads.
- Browser secure-context and adapter availability can make CI evidence
  host-dependent unless no-backend artifacts are clearly separated from
  required-backend runs.
- Asyncify and Emscripten export changes are broad runtime changes and should
  not be bundled with the contract/probe patch.
- Phase and any other CPU-fallback operations must be counted separately from
  native WebGPU coverage.
- Passing this parity probe must not be interpreted as full magnetar physics
  validation.
