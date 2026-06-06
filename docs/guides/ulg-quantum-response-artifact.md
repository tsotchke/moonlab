# ULG Quantum Response Artifact

Moonlab can emit minimal ULG `QuantumResponseArtifact` readiness artifacts from the JavaScript core package. The default helper runs a local two-qubit Bell-state task through the built WASM core, records exact-probability parity against the Bell reference, and marks that no networking or GPU scheduling was started.

From `bindings/javascript/packages/core`:

```bash
pnpm build
pnpm ulg:artifact -- --schema /home/cos/projects/ulg/ulg-gpu-abi/src/schemas/quantum_response_artifact.schema.json --out artifacts/ulg_bell_response.json
```

Moonlab can also emit a small magnetar-relevant calibration probe:

```bash
pnpm ulg:artifact -- --probe magnetar-dipole-ising --schema /home/cos/projects/ulg/ulg-gpu-abi/src/schemas/quantum_response_artifact.schema.json --out artifacts/ulg_magnetar_dipole_ising.json
```

That probe maps a default magnetar surface field of `1e11` tesla at a `10 km` stellar radius onto three normalized radial Ising fields using `B(r)=B_surface*(R/r)^3`, then evaluates the resulting nearest-neighbor Ising model through the Moonlab WASM core. It validates the WASM energies against a JavaScript reference calculation.

The emitted JSON includes the schema-required fields plus:

- `validation`: schema compatibility, probe parity, no-networking, and no-GPU-scheduling checks.
- `parity`: expected and observed Bell probabilities or Ising energies with max delta and tolerance.
- `provenance`: runtime, package, circuit or core primitive, CLI, WASM asset inspection, and schema path metadata.
- `outputs.reference`: for the magnetar probe, a PeerCompute-facing reference contract with `schema`, `role`, `contractHash`, `hamiltonian`, `observables.groundState`, `observables.energySpectrum`, `tolerances.energyAbs`, and validation deltas.

These slices are intentionally CPU/WASM-only. They do not open sockets, start relay clients, or claim ownership of GPU placement. The magnetar dipole Ising probe is a calibration and handoff primitive, not a full magnetar simulation; it does not include plasma, radiation, relativistic, or MHD evolution.
