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

## Supplied Magnetar Reference Contracts

Targeted validation for optional calibrated-reference contract input:

- Run the JavaScript core integration test that supplies a ready radiation transport reference and asserts it replaces the `radiation-transport` placeholder while the existing analytic magnetosphere entry remains ready:
  `pnpm test:integration -- src/__tests__/ulg-quantum-response-artifact.integration.test.ts`
- Run the JavaScript core unit suite to ensure the additive builder changes do not regress Bell or magnetar helper behavior:
  `pnpm test:unit -- src/__tests__/ulg-quantum-response-artifact.test.ts`
- Build the JavaScript core so DTS generation verifies the optional reference input types:
  `pnpm build`
- Emit a magnetar artifact with CLI-supplied reference contracts and inspect the ready counts and provenance:
  `pnpm ulg:artifact -- --probe magnetar-dipole-ising --references /tmp/moonlab-supplied-references.json --schema /home/cos/projects/ulg/ulg-gpu-abi/src/schemas/quantum_response_artifact.schema.json --out /tmp/moonlab-supplied-reference-artifact.json`

Current result on 2026-06-06: focused integration passed `45/45`, focused unit passed `95/95`, package build passed with the existing package export-order warning, and the CLI smoke emitted four calibrated family entries with two ready/scientific entries after a supplied radiation contract.

## Magnetar Reference Contract Validator

Targeted validation for standalone calibrated-reference contract reports:

- Run the JavaScript core unit suite with focused validator cases:
  `pnpm test:unit -- src/__tests__/ulg-quantum-response-artifact.test.ts`
- Build the JavaScript core before CLI validation so `dist/index.mjs`, `dist/moonlab.js`, and `dist/moonlab.wasm` are present:
  `pnpm build`
- Run the JavaScript core integration suite with the CLI report assertion:
  `pnpm test:integration -- src/__tests__/ulg-quantum-response-artifact.integration.test.ts`
- Manually smoke the validator CLI against an invalid supplied reference both without and with `--strict`; non-strict should exit `0`, strict should exit `1`, and the JSON should report missing hashes plus tolerance failures.

Current result on 2026-06-06: focused unit passed `100/100`, full package build passed with the existing package export-order warning, integration passed `46/46` after the full build restored `dist/moonlab.js` and `dist/moonlab.wasm`, and the manual invalid-reference CLI smoke returned non-strict exit `0` and strict exit `1`.

## Checked-In Magnetar Reference Contracts

Targeted validation for the checked-in reduced four-family reference payload:

- Strictly validate the payload:
  `pnpm ulg:artifact -- --validate-references references/magnetar-calibrated-reference-contracts.json --strict`
- Run the JavaScript core integration suite, which now proves the checked-in payload emits four ready/scientific calibrated reference entries when merged with the built-in analytic MHD reference:
  `pnpm test:integration -- src/__tests__/ulg-quantum-response-artifact.integration.test.ts`
- Run the JavaScript core unit suite for non-regression:
  `pnpm test:unit -- src/__tests__/ulg-quantum-response-artifact.test.ts`
- Build the package so DTS and WASM artifacts remain valid:
  `pnpm build`
- Emit a magnetar artifact with the checked-in references and inspect the counts:
  `pnpm ulg:artifact -- --probe magnetar-dipole-ising --references references/magnetar-calibrated-reference-contracts.json --schema /home/cos/projects/ulg/ulg-gpu-abi/src/schemas/quantum_response_artifact.schema.json --out /tmp/moonlab-four-family-reference-artifact.json`

Current result on 2026-06-06: integration passed `48/48`, unit passed `100/100`, strict validator reported `reference-contract-suite-ready` with ready count `4`, package build passed with the existing package export-order warning, and the artifact smoke reported reference/scientific coverage counts `4/4` with no blockers.
