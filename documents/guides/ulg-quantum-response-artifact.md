# ULG Quantum Response Artifact

Moonlab can emit a minimal ULG `QuantumResponseArtifact` readiness artifact from the JavaScript core package. The helper runs a local two-qubit Bell-state task through the built WASM core, records exact-probability parity against the Bell reference, and marks that no networking or GPU scheduling was started.

From `bindings/javascript/packages/core`:

```bash
pnpm build
pnpm ulg:artifact -- --schema /home/cos/projects/ulg/ulg-gpu-abi/src/schemas/quantum_response_artifact.schema.json --out artifacts/ulg_bell_response.json
```

The emitted JSON includes the schema-required fields plus:

- `validation`: schema compatibility, Bell parity, no-networking, and no-GPU-scheduling checks.
- `parity`: expected and observed Bell probabilities with max delta and tolerance.
- `provenance`: runtime, package, circuit, CLI, WASM asset inspection, and schema path metadata.

This slice is intentionally CPU/WASM-only. It does not open sockets, start relay clients, or claim ownership of GPU placement.
