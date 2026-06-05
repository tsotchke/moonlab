import { describe, expect, it } from 'vitest';
import {
  canonicalJson,
  DEFAULT_ULG_QUANTUM_RESPONSE_SCHEMA,
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
});
