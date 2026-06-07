# Browser WebGPU Complex64 Parity Blocker

Status: implementation note for the MoonLab `ulg` branch. No full browser
WebGPU runtime patch is applied here because the available backend work lives
on a divergent `webgpu` branch. The current branch does have a reduced
complex64 browser-probe contract and smoke harness that can pass only for the
declared fixture operations.

## Current Branch State

- `bindings/javascript/packages/core` on `ulg` has a reduced-fixture WebGPU
  complex64 parity-scope contract and CLI. It now has standalone
  browser-executable probe helpers for `compute_probabilities`, `hadamard`,
  `pauli_x`, `pauli_z`, and `cnot`, plus explicit browser adapter/device
  preflight evidence. It still has no full browser WebGPU runtime source, no
  JS WebGPU API export, no `HAS_WEBGPU` build switch, no Asyncify wiring, and
  no `_gpu_*` / `_tensor_gpu_webgpu_available` exports.
- Checked-in older WebGPU artifacts are stale no-backend evidence only:
  - `bindings/javascript/packages/core/artifacts/webgpu_smoke/results.json`
  - `bindings/javascript/packages/core/artifacts/webgpu_unified_eval/results.json`
  - `bindings/javascript/packages/core/artifacts/webgpu_eval/results.json`
- The current browser-smoke command can emit fresh browser-context evidence to
  a caller-supplied path. On the local Chrome run for this slice it acquired a
  device, covered all declared reduced operations, and set
  `webgpuParity.passed=true`.
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

The first safe patch has been implemented as a contract/preflight CLI before
runtime acceleration. It emits a deterministic reduced fixture artifact rather
than a full physics claim.

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

Implemented slice:

1. Added `bindings/javascript/packages/core/src/webgpu-complex64-parity.ts`.
2. Added `bindings/javascript/packages/core/scripts/webgpu-complex64-parity.mjs`.
3. Added package script `webgpu:complex64:parity`.
4. Added focused unit coverage for no-backend, required-backend, and overclaim
   rejection paths.
5. Added a browser-executable `compute_probabilities` WGSL probe over the
   reduced complex64 fixture states. This records only probability-kernel
   coverage when a real browser WebGPU adapter is present.
6. Tightened the contract so `webgpuParity.passed` requires all required native
   operations to be covered; a passing probability-kernel probe alone remains
   partial evidence.
7. Added standalone browser-executable native-operation WGSL probe helpers for
   `hadamard`, `pauli_x`, `pauli_z`, and `cnot`. These report fixture coverage
   only when a real browser WebGPU adapter/device executes them; the default
   Node/no-adapter artifact keeps them `executed=false` and `covered=false`.
8. Added `moonlab.webgpu.complex64-browser-backend-preflight.v0` adapter/device
   evidence. It records `navigatorGpuAvailable`, `adapterAvailable`,
   `deviceAcquired`, adapter info when available, and the current preflight
   stage. Kernel probes are skipped unless the preflight acquires a real device.
9. Added a dependency-free browser smoke harness and Chrome-compatible runner
   so the same probe path can execute from a localhost browser context when a
   real `navigator.gpu` adapter is available. The runner extracts the browser
   artifact JSON and keeps required-backend mode nonzero unless probe execution
   is actually recorded.

The next patch after this browser harness should stay minimal: decide whether
to check in a stable browser-smoke evidence artifact or continue toward a real
runtime backend adapter. Keep that separate from broad backend imports and avoid
counting `phase` CPU fallback as native WebGPU coverage.

The first parity probe should avoid claiming magnetar simulation validation.
It should only demonstrate that reduced deterministic quantum fixtures can
round-trip through browser complex64 WebGPU kernels within the accepted
tolerance. The current probes cover reduced standalone probability and
operation fixtures when a real adapter executes them; they are not a full
MoonLab browser WebGPU runtime backend.

## Acceptance Commands

Current parity-scope patch:

```sh
pnpm --dir bindings/javascript/packages/core build:ts
pnpm --dir bindings/javascript/packages/core exec vitest run src/__tests__/webgpu-complex64-parity.test.ts
pnpm --dir bindings/javascript/packages/core exec vitest run src/__tests__/ulg-quantum-response-artifact.test.ts
pnpm --dir bindings/javascript/packages/core test
pnpm --dir bindings/javascript/packages/core webgpu:complex64:parity -- --out /tmp/moonlab-webgpu-complex64-parity.json
pnpm --dir bindings/javascript/packages/core webgpu:complex64:browser-smoke -- --out /tmp/moonlab-webgpu-complex64-browser-smoke.json
pnpm --dir bindings/javascript/packages/core webgpu:complex64:browser-smoke -- --require-backend --out /tmp/moonlab-webgpu-complex64-browser-smoke-required.json
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
