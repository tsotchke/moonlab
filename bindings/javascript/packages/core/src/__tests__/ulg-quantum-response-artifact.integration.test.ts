import { describe, expect, it } from 'vitest';
import {
  buildUlgBellStateArtifact,
  buildUlgMagnetarDipoleIsingArtifact,
} from '../ulg-quantum-response-artifact';

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

  it('runs the WASM Ising primitive for a magnetar dipole calibration artifact', async () => {
    const artifact = await buildUlgMagnetarDipoleIsingArtifact({
      createdAt: '2026-06-05T00:00:00.000Z',
      inputHash: 'sha256:test-magnetar-input',
    });
    const outputs = artifact.outputs as {
      radialSamples: Array<{ radiusMeters: number; magneticFieldTesla: number }>;
      evaluations: Array<{
        bitstring: number;
        bitString: string;
        observedEnergy: number;
        referenceEnergy: number;
      }>;
      summary: {
        groundState: { bitstring: number; bitString: string; observedEnergy: number };
      };
    };

    expect(artifact.sourceService).toBe('moonlab');
    expect(artifact.taskKind).toBe('magnetar-dipole-ising-calibration');
    expect(artifact.method).toBe('moonlab-js-core-wasm-ising-evaluator');
    expect(artifact.validation.schemaCompatible).toBe(true);
    expect(artifact.parity.passed).toBe(true);
    expect(artifact.provenance.networkingStarted).toBe(false);
    expect(artifact.provenance.gpuSchedulingStarted).toBe(false);
    expect(outputs.radialSamples[0].magneticFieldTesla).toBe(1e11);
    expect(outputs.radialSamples[1].magneticFieldTesla / 1e11).toBeCloseTo((2 / 3) ** 3, 12);
    expect(outputs.summary.groundState.bitstring).toBe(0);
    expect(outputs.summary.groundState.bitString).toBe('000');
    expect(outputs.evaluations[0].observedEnergy).toBe(outputs.evaluations[0].referenceEnergy);
  });
});
