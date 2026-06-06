import { describe, expect, it } from 'vitest';
import {
  buildMagnetarDipoleIsingInput,
  buildMagnetarDipoleIsingModel,
  canonicalJson,
  DEFAULT_ULG_QUANTUM_RESPONSE_SCHEMA,
  evaluateIsingReferenceEnergy,
  validateUlgQuantumResponseArtifact,
  type UlgQuantumResponseArtifact,
} from '../ulg-quantum-response-artifact';

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
});
