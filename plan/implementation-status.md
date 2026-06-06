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

## Analytic Magnetosphere Reference

- [x] Add a scoped `magnetosphere-mhd` analytic dipole-field reference entry.
- [x] Provide solver id, contract hash, units hash, field maps, field tolerances, observed deltas, pass validation, and evidence text.
- [x] Align observed-delta keys with tolerance keys so downstream PeerCompute/Multiscale readiness checks can evaluate the scoped reference.
- [x] Keep the scope explicit as an analytic exterior dipole-field benchmark, not full resistive-MHD or force-free magnetosphere validation.
- [x] Keep PIC kinetic plasma, radiation transport, and relativistic correction entries blocked.

Current blocker: full magnetar scientific readiness still needs calibrated PIC kinetic plasma, radiation transport, relativistic correction, and fuller MHD/force-free benchmark coverage beyond the analytic dipole-field reference.

## Supplied Magnetar Reference Contracts

- [x] Add an optional builder input for externally supplied calibrated reference contracts.
- [x] Add CLI `--references <json>` support for JSON arrays, `{ "references": [...] }`, and `{ "outputs": { "references": [...] } }` shapes.
- [x] Merge supplied contracts by inventory `id` or `family`.
- [x] Require ready/scientific flags, solver id, SHA-256 contract/unit hashes, field maps, field tolerances, observed deltas, pass validation, and observed deltas within tolerance before replacing a blocker.
- [x] Keep invalid supplied contracts blocked instead of granting scientific coverage.
- [x] Add integration coverage and CLI smoke validation.

Current blocker: real calibrated PIC, radiation, relativity, and fuller MHD/force-free reference contract files still need to be generated and validated before full magnetar scientific readiness can advance.

## Magnetar Reference Contract Validator

- [x] Read the current supplied-reference merge rules and keep validation aligned with them.
- [x] Export `validateMagnetarReferenceContracts()` from the JavaScript core package.
- [x] Report the four magnetar calibrated-reference families with ready/scientific flags, hash validity, field-map/tolerance/delta readiness, tolerance failures, blockers, and errors.
- [x] Report unknown ids/families and duplicate family submissions without silently dropping them.
- [x] Add CLI `--validate-references <json>` support with `--strict` controlling nonzero exit behavior.
- [x] Add focused unit tests for valid supplied references, missing hashes, empty field maps, unknown ids/families, and deltas exceeding tolerance.
- [x] Add CLI integration coverage against the built package export.
- [x] Run focused unit/integration/build/manual CLI validation.

Current blocker: the validator can certify contract shape and tolerance deltas, but it still needs real calibrated PIC, radiation, relativity, and fuller MHD/force-free reference files before full magnetar scientific readiness can pass.
