# Implementation Status

## ULG Quantum Response Artifact Readiness

- [x] Read workspace instructions before editing.
- [x] Confirm current branch and avoid branch switching.
- [x] Add additive Bell-state `QuantumResponseArtifact` builder.
- [x] Add CLI emitter with schema validation hook.
- [x] Add unit and integration tests.
- [x] Add docs for the readiness artifact command.
- [x] Run targeted build/tests and schema validation.
- [x] Commit locally after passing validation.

Current blocker: none.

## ULG Magnetar Dipole Ising Calibration Probe

- [x] Read workspace instructions and MoonLab plan/log/tests before editing.
- [x] Identify next bounded magnetar-relevant gap after WASM readiness.
- [x] Expose the existing WASM Ising evaluator through the JavaScript core package.
- [x] Add additive ULG magnetar dipole Ising calibration artifact builder.
- [x] Add CLI probe selection for Bell vs magnetar calibration artifacts.
- [x] Add unit/integration tests and guide updates.
- [x] Run targeted unit/integration/build/schema validation.
- [x] Commit locally after passing validation.

Current blocker: none.

## PeerCompute Magnetar Reference Contract

- [x] Inspect the emitted magnetar dipole Ising artifact shape.
- [x] Identify the narrow PeerCompute consumption gap: explicit reference/tolerance fields.
- [x] Add nested `outputs.reference` with Hamiltonian, reference spectrum, ground state, and absolute energy tolerance.
- [x] Add integration assertions and guide updates for the reference contract.

Current blocker: PeerCompute still needs to ingest `outputs.reference` as a reference/tolerance input rather than only summarizing the artifact as a generic MoonLab calibration.
