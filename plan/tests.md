# Testing Strategy

## ULG Quantum Response Artifact Readiness

Targeted validation for this slice:

- Run the JavaScript core unit tests for schema helper behavior:
  `pnpm test:unit -- src/__tests__/ulg-quantum-response-artifact.test.ts`
- Run the JavaScript core integration test for the WASM-backed Bell-state artifact:
  `pnpm test:integration -- src/__tests__/ulg-quantum-response-artifact.integration.test.ts`
- Build the JavaScript core before CLI validation so `dist/index.mjs`, `dist/moonlab.js`, and `dist/moonlab.wasm` are present:
  `pnpm build`
- Emit and validate a JSON artifact against the ULG schema:
  `pnpm ulg:artifact -- --schema /home/cos/projects/ulg/ulg-gpu-abi/src/schemas/quantum_response_artifact.schema.json --out artifacts/ulg_bell_response.json`

This readiness slice does not exercise networking, relay processes, or GPU scheduling.

## ULG Magnetar Dipole Ising Calibration Probe

Targeted validation for this slice:

- Run the JavaScript core unit tests for schema and magnetar dipole helper behavior:
  `pnpm test:unit -- src/__tests__/ulg-quantum-response-artifact.test.ts`
- Run the JavaScript core integration tests for the WASM Ising wrapper and magnetar artifact:
  `pnpm test:integration -- src/__tests__/ising-model.integration.test.ts src/__tests__/ulg-quantum-response-artifact.integration.test.ts`
- Build the JavaScript core before CLI validation so `dist/index.mjs`, `dist/moonlab.js`, and `dist/moonlab.wasm` are present:
  `pnpm build`
- Emit and validate the magnetar calibration artifact against the ULG schema:
  `pnpm ulg:artifact -- --probe magnetar-dipole-ising --schema /home/cos/projects/ulg/ulg-gpu-abi/src/schemas/quantum_response_artifact.schema.json --out /tmp/moonlab-ulg-magnetar-dipole-ising.json`
- Re-run the Bell artifact CLI to confirm the default readiness path still works:
  `pnpm ulg:artifact -- --schema /home/cos/projects/ulg/ulg-gpu-abi/src/schemas/quantum_response_artifact.schema.json --out /tmp/moonlab-ulg-bell-response.json`

This calibration slice remains CPU/WASM-only. It does not exercise networking, relay processes, GPU scheduling, plasma dynamics, radiation transport, relativistic corrections, or MHD evolution.

## PeerCompute Magnetar Reference Contract

Targeted validation for the reference/tolerance handoff fields:

- Run the JavaScript core integration test that builds the magnetar artifact and asserts `outputs.reference`:
  `pnpm test:integration -- src/__tests__/ulg-quantum-response-artifact.integration.test.ts`
- Emit and validate the magnetar artifact against the ULG schema, then inspect `outputs.reference.schema`, `outputs.reference.role`, `outputs.reference.observables`, and `outputs.reference.tolerances`:
  `pnpm ulg:artifact -- --probe magnetar-dipole-ising --schema /home/cos/projects/ulg/ulg-gpu-abi/src/schemas/quantum_response_artifact.schema.json --out /tmp/moonlab-sidecar-magnetar-reference.json`

This remains an additive MoonLab artifact contract. PeerCompute now has a consumer-side parser/assertion path for the summary fields.

## Magnetar Reference Family Inventory

Targeted validation for the calibrated-reference inventory:

- Run the JavaScript core integration test that builds the magnetar artifact and asserts `outputs.references[]` ids/families, downstream role, the ready scoped analytic `magnetosphere-mhd` entry, its contract/unit hashes, field maps/tolerances/observed deltas, pass validation, and the remaining blocked PIC/radiation/relativity entries:
  `pnpm test:integration -- src/__tests__/ulg-quantum-response-artifact.integration.test.ts`
- Emit and validate the magnetar artifact against the ULG schema, then inspect `outputs.references`:
  `pnpm ulg:artifact -- --probe magnetar-dipole-ising --schema /home/cos/projects/ulg/ulg-gpu-abi/src/schemas/quantum_response_artifact.schema.json --out /tmp/moonlab-magnetar-reference-inventory.json`
- The analytic entry's `fieldObservedDeltas` must use the same keys as `fieldTolerances` (`magneticFieldTeslaRel`, `normalizedFieldAbs`, `radialPowerLawAbs`, `divergenceProxyAbs`) so PeerCompute/Multiscale can verify every observed delta is within tolerance.

This inventory now carries one reduced analytic magnetosphere dipole-field reference. It still does not provide full MHD, PIC kinetic plasma, radiation transport, or relativistic correction scientific readiness; those entries remain blockers until calibrated benchmark data, validation runs, field maps, units hashes, and tolerance contracts exist.
