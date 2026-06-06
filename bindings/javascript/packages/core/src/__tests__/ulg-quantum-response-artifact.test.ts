import { describe, expect, it } from 'vitest';
import {
  buildMagnetarDipoleIsingInput,
  buildMagnetarDipoleIsingModel,
  canonicalJson,
  DEFAULT_ULG_QUANTUM_RESPONSE_SCHEMA,
  evaluateIsingReferenceEnergy,
  normalizeMagnetarReferenceContractSuite,
  validateMagnetarReferenceContracts,
  validateUlgQuantumResponseArtifact,
  type UlgMagnetarReferenceFamilyInventoryEntry,
  type UlgQuantumResponseArtifact,
} from '../ulg-quantum-response-artifact';

const VALID_CONTRACT_HASH = `sha256:${'a'.repeat(64)}`;
const VALID_UNITS_HASH = `sha256:${'b'.repeat(64)}`;
const VALID_UPPERCASE_CONTRACT_HASH = `sha256:${'C'.repeat(64)}`;

describe('ULG QuantumResponseArtifact schema helpers', () => {
  it('canonicalizes JSON objects with stable key ordering', () => {
    const left = canonicalJson({ b: 2, a: { d: 4, c: 3 } });
    const right = canonicalJson({ a: { c: 3, d: 4 }, b: 2 });

    expect(left).toBe(right);
    expect(left).toBe('{"a":{"c":3,"d":4},"b":2}');
  });

  it('validates the required ULG schema surface', () => {
    const artifact: UlgQuantumResponseArtifact = {
      artifactId: 'moonlab-bell-state-smoke-test',
      sourceService: 'moonlab',
      taskKind: 'bell-state-smoke',
      inputHash: 'sha256:test',
      method: 'moonlab-js-core-wasm',
      representation: 'state-vector-probability-distribution',
      outputs: {},
      uncertainty: {},
      validation: {
        schemaCompatible: true,
        schemaTitle: DEFAULT_ULG_QUANTUM_RESPONSE_SCHEMA.title,
        checks: [],
        errors: [],
      },
      provenance: {},
      parity: {},
    };

    expect(validateUlgQuantumResponseArtifact(artifact).valid).toBe(true);
  });

  it('reports schema mismatches', () => {
    const result = validateUlgQuantumResponseArtifact({
      artifactId: '',
      sourceService: 'other',
    });

    expect(result.valid).toBe(false);
    expect(result.errors).toContain('missing required field: taskKind');
    expect(result.errors).toContain('field sourceService must equal "moonlab"');
    expect(result.errors).toContain('field artifactId must be at least 1 character(s)');
  });

  it('builds a deterministic magnetar dipole Ising calibration input', () => {
    const input = buildMagnetarDipoleIsingInput();
    const model = buildMagnetarDipoleIsingModel(input);

    expect(input.surfaceMagneticFieldTesla).toBe(1e11);
    expect(input.radialSamplesMeters).toEqual([10_000, 15_000, 20_000]);
    expect(input.bitstrings).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(model.fieldScaleTesla).toBe(1e11);
    expect(model.localFields[0]).toBe(-1);
    expect(model.localFields[1]).toBeCloseTo(-8 / 27, 12);
    expect(model.localFields[2]).toBe(-0.125);
    expect(model.couplings).toEqual([
      { qubit1: 0, qubit2: 1, value: -0.125 },
      { qubit1: 1, qubit2: 2, value: -0.125 },
    ]);
    expect(evaluateIsingReferenceEnergy(model, 0)).toBeLessThan(
      evaluateIsingReferenceEnergy(model, 7)
    );
  });

  it('rejects invalid magnetar dipole probe inputs', () => {
    expect(() =>
      buildMagnetarDipoleIsingInput({ radialSamplesMeters: [20_000, 10_000] })
    ).toThrow('radialSamplesMeters must be sorted');
    expect(() =>
      buildMagnetarDipoleIsingInput({ radialSamplesMeters: [9_000] })
    ).toThrow('radialSamplesMeters must be at or outside stellarRadiusMeters');
    expect(() =>
      buildMagnetarDipoleIsingInput({ couplingStrength: -1 })
    ).toThrow('couplingStrength must be non-negative');
  });

  it('validates supplied calibrated reference contracts without requiring a full suite', () => {
    const report = validateMagnetarReferenceContracts([
      readyReferenceContract({
        id: 'radiation-transport-reference',
        family: 'radiation-transport',
        solverId: 'moonlab-grey-radiation-transport-reference-v0',
      }),
    ]);

    const radiation = report.entries.find((entry) => entry.family === 'radiation-transport');
    expect(report.status).toBe('reference-contract-suite-partial');
    expect(report.ready).toBe(false);
    expect(report.invalidSuppliedCount).toBe(0);
    expect(report.suppliedReadyCount).toBe(1);
    expect(radiation?.supplied).toBe(true);
    expect(radiation?.ready).toBe(true);
    expect(radiation?.contractHashValid).toBe(true);
    expect(radiation?.unitsHashValid).toBe(true);
    expect(radiation?.fieldDeltasWithinTolerance).toBe(true);
    expect(radiation?.errors).toEqual([]);
  });

  it('normalizes supplied reference contracts into a four-family suite', () => {
    const suite = normalizeMagnetarReferenceContractSuite({
      references: [
        readyReferenceContract({
          id: 'radiation-transport-reference',
          family: 'radiation-transport',
          solverId: 'moonlab-grey-radiation-transport-reference-v0',
          contractHash: VALID_UPPERCASE_CONTRACT_HASH,
        }),
      ],
    });
    const radiation = suite.references.find((entry) => entry.family === 'radiation-transport');
    const radiationValidation = suite.validation.entries.find((entry) => (
      entry.family === 'radiation-transport'
    ));

    expect(suite.schema).toBe('moonlab.magnetar.normalized-reference-suite.v0');
    expect(suite.status).toBe('reference-contract-suite-partial');
    expect(suite.ready).toBe(false);
    expect(suite.familyCount).toBe(4);
    expect(suite.readyCount).toBe(2);
    expect(suite.suppliedCount).toBe(1);
    expect(suite.source.builtInReferenceFamilies).toEqual(['magnetosphere-mhd']);
    expect(suite.references.map((entry) => entry.family)).toEqual([
      'magnetosphere-mhd',
      'pic-kinetic-plasma',
      'radiation-transport',
      'relativistic-correction',
    ]);
    expect(radiation?.ready).toBe(true);
    expect(radiation?.contractHash).toBe(VALID_UPPERCASE_CONTRACT_HASH.toLowerCase());
    expect(radiationValidation?.supplied).toBe(true);
    expect(radiationValidation?.ready).toBe(true);
  });

  it('reports missing contract and unit hashes on supplied references', () => {
    const report = validateMagnetarReferenceContracts([
      readyReferenceContract({
        id: 'pic-kinetic-plasma-reference',
        family: 'pic-kinetic-plasma',
        contractHash: null,
        unitsHash: null,
      }),
    ]);
    const pic = report.entries.find((entry) => entry.family === 'pic-kinetic-plasma');

    expect(report.status).toBe('reference-contract-suite-invalid');
    expect(report.invalidSuppliedCount).toBe(1);
    expect(pic?.ready).toBe(false);
    expect(pic?.contractHashValid).toBe(false);
    expect(pic?.unitsHashValid).toBe(false);
    expect(pic?.errors).toContain('contractHash must be a sha256 digest');
    expect(pic?.errors).toContain('unitsHash must be a sha256 digest');
  });

  it('requires supplied contract hashes to be full SHA-256 hex digests', () => {
    const report = validateMagnetarReferenceContracts([
      readyReferenceContract({
        id: 'pic-kinetic-plasma-reference',
        family: 'pic-kinetic-plasma',
        contractHash: 'sha256:reference-contract',
        unitsHash: 'sha256:reference-units',
      }),
    ]);
    const pic = report.entries.find((entry) => entry.family === 'pic-kinetic-plasma');

    expect(report.status).toBe('reference-contract-suite-invalid');
    expect(pic?.ready).toBe(false);
    expect(pic?.contractHashValid).toBe(false);
    expect(pic?.unitsHashValid).toBe(false);
    expect(pic?.errors).toContain('contractHash must be a sha256 digest');
    expect(pic?.errors).toContain('unitsHash must be a sha256 digest');
  });

  it('reports empty field maps on supplied references', () => {
    const report = validateMagnetarReferenceContracts([
      readyReferenceContract({
        id: 'relativistic-correction-reference',
        family: 'relativistic-correction',
        fieldMap: {},
      }),
    ]);
    const relativity = report.entries.find((entry) => entry.family === 'relativistic-correction');

    expect(report.status).toBe('reference-contract-suite-invalid');
    expect(relativity?.ready).toBe(false);
    expect(relativity?.fieldMapReady).toBe(false);
    expect(relativity?.errors).toContain('fieldMap must be a non-empty object');
  });

  it('reports unknown supplied reference families and ids', () => {
    const report = validateMagnetarReferenceContracts([
      {
        id: 'unknown-reference',
        family: 'unknown-family',
        ready: true,
      },
    ]);

    expect(report.status).toBe('reference-contract-suite-invalid');
    expect(report.unknownCount).toBe(1);
    expect(report.unknownReferences[0]).toMatchObject({
      id: 'unknown-reference',
      family: 'unknown-family',
    });
    expect(report.unknownReferences[0].errors).toContain(
      'reference contract id or family is not part of the magnetar reference inventory'
    );
  });

  it('reports observed deltas that exceed supplied tolerances', () => {
    const report = validateMagnetarReferenceContracts([
      readyReferenceContract({
        id: 'radiation-transport-reference',
        family: 'radiation-transport',
        fieldTolerances: {
          opticalDepthAbs: 1e-6,
        },
        fieldObservedDeltas: {
          opticalDepthAbs: 2e-6,
        },
      }),
    ]);
    const radiation = report.entries.find((entry) => entry.family === 'radiation-transport');

    expect(report.status).toBe('reference-contract-suite-invalid');
    expect(radiation?.ready).toBe(false);
    expect(radiation?.fieldDeltasWithinTolerance).toBe(false);
    expect(radiation?.toleranceFailures).toEqual([
      {
        field: 'opticalDepthAbs',
        observed: 2e-6,
        tolerance: 1e-6,
      },
    ]);
    expect(radiation?.errors).toContain('fieldObservedDeltas must be within fieldTolerances');
  });
});

function readyReferenceContract(
  overrides: Partial<UlgMagnetarReferenceFamilyInventoryEntry> = {}
): Partial<UlgMagnetarReferenceFamilyInventoryEntry> {
  return {
    id: 'radiation-transport-reference',
    family: 'radiation-transport',
    provider: 'moonlab',
    solverId: 'moonlab-reference-solver-v0',
    schema: 'moonlab.magnetar.calibrated-reference.v0',
    role: 'peercompute-scientific-tolerance-input',
    contractHash: VALID_CONTRACT_HASH,
    unitsHash: VALID_UNITS_HASH,
    fieldMap: {
      opticalDepth: 'outputs.reference.opticalDepth',
    },
    fieldTolerances: {
      opticalDepthAbs: 1e-6,
    },
    fieldObservedDeltas: {
      opticalDepthAbs: 0,
    },
    label: 'Supplied calibrated reference',
    status: 'calibrated-reference-ready',
    ready: true,
    scientificCoverage: true,
    scope: 'supplied-calibrated-reference-contract',
    validationStatus: 'pass',
    validation: {
      status: 'pass',
      evidence: ['Supplied calibrated reference contract.'],
    },
    blocker: null,
    blockers: [],
    ...overrides,
  };
}
