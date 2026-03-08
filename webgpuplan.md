# Moonlab WASM WebGPU Plan

## Objective
Add **WebGPU support to the WASM build** with a staged rollout that is safe, testable, and continuously verifiable without manual intervention.

## Current State (as of March 7, 2026)
- WASM build includes unified GPU backend + WebGPU backend sources and exports.
- Tensor GPU context abstraction in `src/algorithms/tensor_network/tensor.c` can select WebGPU in WASM runtimes.
- Tensor-network accelerated compute paths in `tn_gates.c` / `tn_measurement.c` are still Metal-only; WebGPU currently falls back to CPU there.
- Unified GPU backend WebGPU path has native WGSL dispatch for a first operation set and CPU fallback for unsupported/runtime-guarded scenarios.

## Implementation Progress (March 7, 2026)
- Completed:
  - WebGPU backend scaffolding + WASM build integration (`HAS_WEBGPU`, exports, TS API surface).
  - Unified GPU backend session API in JS with backend/native introspection.
  - Native WGSL dispatch path for unified backend operations:
    - `hadamard`
    - `pauli_x`
    - `pauli_z`
    - `cnot`
    - `compute_probabilities`
  - Automated non-interactive loops:
    - `scripts/webgpu-eval.mjs` (tensor/runtime parity checks)
    - `scripts/webgpu-unified-smoke.mjs` (backend/native smoke)
    - `scripts/webgpu-unified-eval.mjs` (randomized unified backend parity loop)
- In progress:
  - Deno-native WebGPU dispatch enablement (currently guarded off due Asyncify/runtime instability in Deno path).
  - Tensor-network compute offload parity for WebGPU in `tn_gates.c`/`tn_measurement.c` (currently Metal-only fast path).

## Design Principles
- Keep CPU path as correctness reference and fallback.
- Add WebGPU incrementally: availability/context/buffer lifecycle first, compute kernels second.
- Make every phase testable in automation.
- Prefer deterministic parity checks over pure performance claims.

## Implementation Phases

### Phase 0: Backend Scaffolding (WASM-safe)
Goal: Introduce WebGPU backend plumbing in WASM without destabilizing existing behavior.

Changes:
- Add `src/optimization/gpu/backends/gpu_webgpu.h`
- Add `src/optimization/gpu/backends/gpu_webgpu.c`
- Add WASM compile definitions (`HAS_WEBGPU`, `QSIM_HAS_WEBGPU`) in Emscripten CMake.
- Add `QSIM_BACKEND_GPU_WEBGPU` in config enums/string conversion/detection paths.

Acceptance:
- WASM build succeeds with WebGPU support enabled.
- Existing unit/integration tests remain green.

### Phase 1: Tensor GPU Context Integration (WASM)
Goal: Allow tensor subsystem to initialize/select WebGPU in WASM.

Changes:
- Update `src/algorithms/tensor_network/tensor.c` GPU context to include backend type `webgpu`.
- Add WebGPU buffer alloc/sync/free branches in:
  - `tensor_gpu_alloc`
  - `tensor_sync_to_gpu`
  - `tensor_sync_to_cpu`
  - `tensor_gpu_free`
- Add backend introspection helpers in `tensor.h/.c` so JS can query active backend.

Acceptance:
- Tensor GPU context reports WebGPU backend in browser environments with `navigator.gpu`.
- CPU behavior remains unchanged if WebGPU unavailable.

### Phase 2: WASM Export + JS API Surface
Goal: Expose backend controls/status for automation and runtime selection.

Changes:
- Add C exports for backend status/init (and any readiness probes) to `exports.txt`.
- Add TypeScript API module (e.g. `src/webgpu.ts`) with:
  - `isWebGPUAvailable()`
  - `initializeWebGPUBackend()`
  - `getActiveTensorGPUBackend()`
- Re-export from `src/index.ts`.

Acceptance:
- Node: functions return deterministic “unavailable” semantics.
- Browser secure context: availability and init path return consistent state.

### Phase 3: WebGPU Compute Offload (First Kernel Slice)
Goal: Offload one high-value tensor workload while preserving correctness.

Initial kernel target (from Metal/CUDA parity):
- Adjacent 2-qubit gate path in MPS (`apply_gate_2q_adjacent`) where bond dimension exceeds threshold.

Changes:
- Introduce WGSL kernels and dispatch path for a minimal, bounded operation set.
- Keep CPU fallback and runtime guardrails.
- Implement strict numerical tolerance checks against CPU result per operation.

Acceptance:
- For supported ops/shapes, backend produces output within tolerance.
- Unsupported shapes/ops automatically route to CPU.

### Phase 4: Expand Coverage
Goal: Broaden WebGPU acceleration beyond first slice.

Targets:
- Additional two-qubit operations and selected measurement kernels.
- Batched workloads where dispatch overhead is amortized.

Acceptance:
- Extended op set passes parity suite with stable tolerances.

### Phase 5: Performance Tuning + Hardening
Goal: Move from “working” to “production-grade”.

Targets:
- Kernel fusion where practical.
- Workgroup sizing heuristics based on adapter limits.
- Reduced host-device copy churn and better buffer reuse.
- Detailed telemetry/logging hooks for automated perf regression detection.

Acceptance:
- Perf benchmarks show improvement over CPU fallback on supported browsers/GPUs.
- No correctness regressions in parity suite.

## Testing Strategy (No Human Interaction)

### 1) Deterministic Correctness Suite
- Reuse existing integration tests as baseline.
- Add WebGPU-specific integration tests that:
  - Assert availability/init semantics.
  - Run fixed seeded circuits and compare probabilities/amplitudes against CPU path.
- Tolerance bands:
  - probabilities: `abs diff <= 1e-8` for scalar checks
  - amplitude vectors: `L2 relative error <= 1e-6` (tunable by op class)

### 2) Randomized Parity Eval Loop
Implement an automated eval runner:
- Generate seeded random circuits (vary qubits/depth/gate mix).
- Execute each case in:
  - CPU/reference mode
  - WebGPU-enabled mode
- Compare:
  - probability distribution error metrics
  - key expectation values
  - runtime and memory stats (where available)
- Emit machine-readable JSON summary + nonzero exit on threshold violation.

Suggested location:
- `bindings/javascript/packages/core/scripts/webgpu-eval.mjs`

Suggested command:
- `pnpm --filter @moonlab/quantum-core webgpu:eval`

### 3) CI/Automation Topology
- Tier A (always-on): Node-based smoke/integration checks (WebGPU unavailable expected in many runners).
- Tier B (WebGPU-capable runner): browser-driven eval loop for parity + perf smoke.
- Fail-fast rules:
  - Any correctness tolerance breach fails pipeline.
  - Backend init regressions fail pipeline.

## Eval Loop Specification
For each seeded test case:
1. Create circuit with constrained random gate set.
2. Run CPU reference path.
3. Run WebGPU-enabled path.
4. Compare outputs and record metrics.
5. Aggregate by gate family, qubit count, depth bucket.

Outputs:
- `artifacts/webgpu_eval/results.json`
- `artifacts/webgpu_eval/summary.txt`

Pass criteria:
- 0 correctness failures.
- Aggregate drift below configured tolerance.
- Optional perf checks treated as warnings initially, then promoted to failures after stabilization.

## Risks and Mitigations
- Browser/WebGPU async initialization complexity:
  - Mitigation: explicit init API and readiness checks.
- Numeric drift from precision/layout differences:
  - Mitigation: operation-class tolerances + seeded parity loops.
- Feature fragmentation across browsers:
  - Mitigation: capability probing + strict CPU fallback.
- CI environment variability:
  - Mitigation: separate baseline (always-on) and WebGPU-capable tiers.

## Definition of Done (Incremental)
- Phase 1 done:
  - WASM exposes WebGPU backend selection/status.
  - Tensor GPU context supports WebGPU backend lifecycle.
  - Automated tests verify init + fallback behavior.
- Phase 3 done:
  - At least one MPS compute path runs on WebGPU and matches CPU within tolerances.
  - Eval loop runs unattended and enforces parity thresholds.
