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
