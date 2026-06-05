import { describe, expect, it } from 'vitest';
import { buildUlgBellStateArtifact } from '../ulg-quantum-response-artifact';

describe('ULG QuantumResponseArtifact Bell state readiness', () => {
  it('runs the core WASM Bell task and emits parity-ready artifact JSON', async () => {
    const artifact = await buildUlgBellStateArtifact({
      createdAt: '2026-06-05T00:00:00.000Z',
      inputHash: 'sha256:test-input',
    });

    expect(artifact.sourceService).toBe('moonlab');
    expect(artifact.taskKind).toBe('bell-state-smoke');
    expect(artifact.validation.schemaCompatible).toBe(true);
    expect(artifact.parity.passed).toBe(true);
    expect(artifact.provenance.networkingStarted).toBe(false);
    expect(artifact.provenance.gpuSchedulingStarted).toBe(false);
    expect(artifact.outputs.probabilities).toEqual([0.5, 0, 0, 0.5]);
  });
});
