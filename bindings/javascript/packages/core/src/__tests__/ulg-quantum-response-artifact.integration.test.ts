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
      isingModel: {
        localFields: number[];
        couplings: Array<{ qubit1: number; qubit2: number; value: number }>;
      };
      evaluations: Array<{
        bitstring: number;
        bitString: string;
        observedEnergy: number;
        referenceEnergy: number;
      }>;
      reference: {
        schema: string;
        role: string;
        contractHash: string;
        energyUnits: string;
        hamiltonian: {
          localFields: number[];
          couplings: Array<{ qubit1: number; qubit2: number; value: number }>;
        };
        observables: {
          groundState: { bitString: string; referenceEnergy: number };
          energySpectrum: Array<{ bitString: string; referenceEnergy: number }>;
        };
        tolerances: {
          energyAbs: number;
          maxObservedEnergyDelta: number;
        };
        validation: {
          parityPassed: boolean;
          evaluatedBitstrings: number;
        };
      };
      references: Array<{
        family: string;
        status: string;
        ready: boolean;
        scientificCoverage: boolean;
        scope: string;
        validation: {
          status: string;
          evidence: string[];
        };
        blockers: string[];
      }>;
      summary: {
        groundState: {
          bitstring: number;
          bitString: string;
          observedEnergy: number;
          referenceEnergy: number;
          energyUnits: string;
        };
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
    expect(outputs.summary.groundState.referenceEnergy).toBe(
      outputs.summary.groundState.observedEnergy
    );
    expect(outputs.summary.groundState.energyUnits).toBe('normalized-ising');
    expect(outputs.reference.schema).toBe('moonlab.magnetar-dipole-ising-reference.v0');
    expect(outputs.reference.role).toBe('peercompute-reference-tolerance-input');
    expect(outputs.reference.contractHash).toBe(artifact.inputHash);
    expect(outputs.reference.energyUnits).toBe('normalized-ising');
    expect(outputs.reference.hamiltonian.localFields).toEqual(outputs.isingModel.localFields);
    expect(outputs.reference.hamiltonian.couplings).toEqual(outputs.isingModel.couplings);
    expect(outputs.reference.observables.groundState.bitString).toBe('000');
    expect(outputs.reference.observables.groundState.referenceEnergy).toBe(
      outputs.summary.groundState.referenceEnergy
    );
    expect(outputs.reference.observables.energySpectrum).toHaveLength(8);
    expect(outputs.reference.tolerances.energyAbs).toBe(1e-9);
    expect(outputs.reference.tolerances.maxObservedEnergyDelta).toBe(0);
    expect(outputs.reference.validation.parityPassed).toBe(true);
    expect(outputs.reference.validation.evaluatedBitstrings).toBe(8);
    expect(outputs.references.map((reference) => reference.id)).toEqual([
      'magnetosphere-mhd-reference',
      'pic-kinetic-plasma-reference',
      'radiation-transport-reference',
      'relativistic-correction-reference',
    ]);
    expect(outputs.references.map((reference) => reference.family)).toEqual([
      'magnetosphere-mhd',
      'pic-kinetic-plasma',
      'radiation-transport',
      'relativistic-correction',
    ]);
    for (const reference of outputs.references) {
      expect(reference.schema).toBe('moonlab.magnetar.calibrated-reference.v0');
      expect(reference.role).toBe('peercompute-scientific-tolerance-input');
      expect(reference.contractHash).toBeNull();
      expect(reference.unitsHash).toBeNull();
      expect(reference.fieldMap).toBeNull();
      expect(reference.fieldTolerances).toBeNull();
      expect(reference.fieldObservedDeltas).toBeNull();
      expect(reference.status).toBe('calibrated-reference-missing');
      expect(reference.ready).toBe(false);
      expect(reference.scientificCoverage).toBe(false);
      expect(reference.scope).toBe('inventory-only-not-scientific-reference');
      expect(reference.validationStatus).toBe('missing');
      expect(reference.validation.status).toBe('missing');
      expect(reference.validation.evidence).toEqual([]);
      expect(reference.blocker).toMatch(/^calibrated-.*-reference-missing$/);
      expect(reference.blockers.length).toBeGreaterThan(0);
    }
    expect(outputs.evaluations[0].observedEnergy).toBe(outputs.evaluations[0].referenceEnergy);
  });
});
