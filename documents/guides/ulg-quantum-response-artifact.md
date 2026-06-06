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

The magnetar probe can merge externally supplied calibrated reference contracts:

```bash
pnpm ulg:artifact -- --probe magnetar-dipole-ising --references magnetar-reference-contracts.json --schema /home/cos/projects/ulg/ulg-gpu-abi/src/schemas/quantum_response_artifact.schema.json --out artifacts/ulg_magnetar_dipole_ising.json
```

Moonlab ships a reduced three-family contract payload that complements the
built-in analytic magnetosphere reference:

```bash
pnpm ulg:artifact -- --validate-references references/magnetar-calibrated-reference-contracts.json --strict
pnpm ulg:artifact -- --probe magnetar-dipole-ising --references references/magnetar-calibrated-reference-contracts.json --schema /home/cos/projects/ulg/ulg-gpu-abi/src/schemas/quantum_response_artifact.schema.json --out artifacts/ulg_magnetar_dipole_ising.json
```

The references file may be a JSON array, `{ "references": [...] }`, or `{ "outputs": { "references": [...] } }`. A supplied entry only replaces its inventory placeholder when it names a known `id` or `family`, declares `ready: true` and `scientificCoverage: true`, includes a solver id, SHA-256 contract and units hashes, field maps, tolerances, observed deltas, and pass validation, with every observed delta within its tolerance. Invalid supplied entries leave the original blocker in place and add a supplied-reference readiness blocker.

Reference contracts can also be validated without emitting a new artifact:

```bash
pnpm ulg:artifact -- --validate-references magnetar-reference-contracts.json
pnpm ulg:artifact -- --validate-references magnetar-reference-contracts.json --strict
```

The validation report uses schema `moonlab.magnetar.reference-contract-validation-report.v0` and reports each calibrated family, supplied/ready flags, hash validity, field-map/tolerance/delta readiness, tolerance failures, blockers, and errors. Without `--strict`, invalid or partial reports still exit `0` so the JSON can be inspected. With `--strict`, the CLI exits nonzero unless all four magnetar reference families are ready and no supplied reference is invalid or unknown.

Reference contracts can also be normalized into a standalone four-family suite:

```bash
pnpm ulg:artifact -- --normalize-references references/magnetar-calibrated-reference-contracts.json --strict --out artifacts/magnetar_normalized_reference_suite.json
pnpm ulg:artifact -- --normalize-references references/magnetar-calibrated-reference-contracts.json --strict --canonical --out artifacts/magnetar_normalized_reference_suite.canonical.json
```

The normalized-suite report uses schema `moonlab.magnetar.normalized-reference-suite.v0`, merges valid supplied contracts with the built-in analytic magnetosphere reference, emits exactly the four magnetar reference families in canonical order, includes the validation report, and preserves blockers/errors for invalid or partial inputs. Supplied `contractHash` and `unitsHash` values must be full `sha256:` digests with 64 hex characters; accepted hashes are normalized to lowercase.

Use `--canonical` when downstream ULG or PeerCompute checks need a stable
byte-for-byte JSON payload to hash or diff. Canonical output sorts object keys
recursively and emits a single JSON line plus a trailing newline; it does not
change the reduced fixture scope or add full-physics validation claims.
The checked-in reduced reference suite currently canonicalizes to SHA-256
`sha256:e88c1ba87216aca7b8df77e7f7347c3e1cc506ab5d1b3c06979cc92b4a925b65`
when the trailing newline is excluded from the digest.

The emitted JSON includes the schema-required fields plus:

- `validation`: schema compatibility, probe parity, no-networking, and no-GPU-scheduling checks.
- `parity`: expected and observed Bell probabilities or Ising energies with max delta and tolerance.
- `provenance`: runtime, package, circuit or core primitive, CLI, WASM asset inspection, and schema path metadata.
- `outputs.reference`: for the magnetar probe, a PeerCompute-facing reference contract with `schema`, `role`, `contractHash`, `hamiltonian`, `observables.groundState`, `observables.energySpectrum`, `tolerances.energyAbs`, and validation deltas.
- `outputs.references`: for the magnetar probe, an inventory of calibrated magnetosphere MHD, PIC kinetic plasma, radiation transport, and relativistic correction reference families. The magnetosphere entry now provides a scoped analytic dipole-field benchmark with solver id `moonlab-analytic-dipole-field-v0`, field maps/tolerances/deltas, SHA-256 contract and units hashes, `ready: true`, and `scientificCoverage: true` for that reduced exterior-field reference only. The checked-in `references/magnetar-calibrated-reference-contracts.json` file supplies reduced PIC, grey-radiation, and post-Newtonian scalar contracts that can make all four inventory entries ready for PeerCompute tolerance plumbing.

These slices are intentionally CPU/WASM-only. They do not open sockets, start relay clients, or claim ownership of GPU placement. The magnetar dipole Ising probe, analytic dipole-field reference, and reduced supplied contracts are calibration and handoff primitives, not a full magnetar simulation; they do not include calibrated PIC, spectral radiation transport, GR/GRMHD, or full MHD evolution.
