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

## Checked-In Magnetar Reference Contracts

- [x] Add `references/magnetar-calibrated-reference-contracts.json` to the JavaScript core package.
- [x] Provide reduced PIC kinetic plasma, grey-radiation, and post-Newtonian scalar contracts that complement the built-in analytic magnetosphere reference.
- [x] Keep every supplied contract explicitly scoped as reduced tolerance plumbing, not a calibrated full-physics solver benchmark.
- [x] Include the `references` directory in the package file list.
- [x] Add integration coverage proving the checked-in payload validates strictly and emits a four-family ready magnetar artifact.
- [x] Run focused unit/integration/build/CLI validation.

Current blocker: the four-family contract payload clears MoonLab reference inventory readiness for PeerCompute tolerance plumbing, but it is still reduced scalar reference data. Full magnetar scientific readiness still requires authoritative calibrated PIC, radiation transport, GR/GRMHD, and full MHD/force-free validation.

## Normalized Magnetar Reference Suite

- [x] Identify the next bounded MoonLab-side gap after the reduced checked-in contracts: a producer artifact that emits the merged canonical four-family reference suite, not only validation pass/fail or a full quantum-response artifact.
- [x] Export `normalizeMagnetarReferenceContractSuite()` from the JavaScript core package.
- [x] Add CLI `--normalize-references <json>` support with `--strict` behavior matching the validator.
- [x] Tighten supplied `contractHash` and `unitsHash` checks to full 64-hex SHA-256 digests and normalize accepted hashes to lowercase.
- [x] Add focused unit and integration coverage for normalized-suite output.

Current blocker: normalized-suite artifacts still normalize reduced scalar contracts only. Replacing them with authoritative external PIC, radiation transport, GR/GRMHD, and full MHD/force-free contracts remains a future data/science task.

## Magnetar Fidelity Runtime Scope Contract

- [x] Add `ulg.magnetar.fidelity-runtime-scope.v0` to the normalized reference
  suite and every canonical magnetar calibrated reference.
- [x] Keep reduced fixture references explicit with
  `fullFidelityMagnetarSimulation = false` and
  `fullPhysicsValidation = false`.
- [x] Preserve scope metadata when evaluating supplied reference contracts.
- [x] Add unit and integration coverage for suite-level and per-reference scope
  propagation.
- [x] Run focused TypeScript build, unit tests, integration tests, and WASM build
  before staging into ULG.

Current blocker: the scope contract prevents overclaiming, but the payload still
contains reduced scalar fixture data. Full magnetar validation still requires
authoritative PIC, radiation transport, GR/GRMHD, and full MHD/force-free
reference data.

## Browser WebGPU Complex64 Parity Blocker

- [x] Inspect the current `ulg` branch for browser WebGPU source, build wiring,
  JS API exports, and runtime artifacts.
- [x] Confirm the current branch contains only stale no-backend WebGPU artifacts
  and no active browser WebGPU runtime.
- [x] Inspect the old `webgpu` branch backend, shader representation, parity
  script, and branch divergence without switching branches.
- [x] Identify the precision blocker: old browser WebGPU kernels use
  complex64/interleaved `vec2<f32>` buffers while the CPU/WASM reference path is
  float64/interleaved.
- [x] Keep the next step bounded to a precision/parity contract and browser
  probe before any runtime backend port.
- [x] Add the `moonlab.webgpu.complex64-parity-scope.v0` contract builder,
  package export, CLI command, and focused unit coverage.
- [x] Preserve reduced fixture scope with
  `fullFidelityMagnetarSimulation = false` and
  `fullPhysicsValidation = false`.
- [x] Emit explicit no-backend scope artifacts by default, with required-backend
  mode failing when no browser WebGPU adapter/runtime parity execution exists.
- [x] Add the first browser-executable WGSL helper for reduced complex64
  `compute_probabilities` kernel probing.
- [x] Keep `compute_probabilities` probe evidence partial: it can mark only
  that native operation covered, and a full `webgpuParity.passed` claim now
  requires all required native operations to be covered.
- [x] Add bounded browser-executable native-operation probe kernels for
  `hadamard`, `pauli_x`, and `pauli_z`, with fixture-level complex64 amplitude
  checks when a real browser WebGPU adapter is available.
- [x] Preserve default no-backend evidence: in Node/no-adapter runs the native
  operation probe still records `executed=false`, `covered=false`, and
  `native-operation-probe-not-executed` for each declared gate.

Current blocker: MoonLab now has the reduced-fixture WebGPU complex64 parity
scope artifact, CLI, and a browser-executable `compute_probabilities` WGSL
probe plus standalone native-operation WGSL probes for `hadamard`, `pauli_x`,
and `pauli_z`. It still has no full MoonLab browser WebGPU runtime backend,
no local browser adapter execution evidence in the Node CLI path, and no
native `cnot` operation probe. Required-backend parity remains blocked until a
browser adapter records native coverage for every required operation
(`hadamard`, `pauli_x`, `pauli_z`, `cnot`, and `compute_probabilities`),
without counting `phase` CPU fallback as native coverage.

## Canonical Normalized Reference Suite Export

- [x] Identify the next reduced-reference hardening gap after the normalized
  suite and WebGPU blocker note: downstream systems need byte-stable suite JSON
  they can hash or diff without reimplementing MoonLab canonicalization.
- [x] Add opt-in CLI `--canonical` output for artifact, reference-validation,
  and normalized-reference-suite JSON.
- [x] Add integration coverage that emits the checked-in normalized reference
  suite canonically and pins its SHA-256 digest.
- [x] Keep the canonical suite scoped to reduced fixture data with
  `fullFidelityMagnetarSimulation = false` and
  `fullPhysicsValidation = false`.

Current blocker: canonical export hardens the reduced reference artifact
handoff, but it still only covers reduced scalar contracts. Full magnetar
readiness remains blocked on authoritative PIC, radiation transport, GR/GRMHD,
and full MHD/force-free validation data.
