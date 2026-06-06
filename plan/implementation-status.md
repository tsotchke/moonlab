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

Current blocker: none for the normalized Ising reference handoff; PeerCompute now ingests the `outputs.reference` summary.

## Magnetar Reference Family Inventory

- [x] Read workspace and ULG agent instructions before editing.
- [x] Confirm MoonLab has no local `AGENTS.md` file and inspect current plan/log/test notes.
- [x] Inspect the magnetar quantum-response artifact emission path for `outputs.reference`.
- [x] Add additive `outputs.references[]` inventory entries for magnetosphere MHD, PIC kinetic plasma, radiation transport, and relativistic correction families.
- [x] Mark every inventory entry as not ready and not scientifically covered, with missing validation status and blockers.
- [x] Align inventory ids, family names, role, and blocker ids with the PeerCompute/Multiscale tolerance-suite contract.
- [x] Add focused integration assertions and guide updates for the inventory semantics.

Current blocker: the inventory is intentionally not a scientific reference set. Calibrated magnetosphere MHD, PIC kinetic plasma, radiation transport, and relativistic correction benchmark data, validation runs, field maps, units hashes, and tolerance contracts are still missing.
