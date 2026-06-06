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
        id: string;
        family: string;
        schema: string;
        role: string;
        solverId: string | null;
        contractHash: string | null;
        unitsHash: string | null;
        fieldMap: Record<string, unknown> | null;
        fieldTolerances: Record<string, unknown> | null;
        fieldObservedDeltas: Record<string, unknown> | null;
        status: string;
        ready: boolean;
        scientificCoverage: boolean;
        scope: string;
        validationStatus: string;
        validation: {
          status: string;
          evidence: string[];
        };
        blocker: string | null;
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
    const magnetosphereReference = outputs.references[0];
    expect(magnetosphereReference.schema).toBe('moonlab.magnetar.calibrated-reference.v0');
    expect(magnetosphereReference.role).toBe('peercompute-scientific-tolerance-input');
    expect(magnetosphereReference.solverId).toBe('moonlab-analytic-dipole-field-v0');
    expect(magnetosphereReference.contractHash).toBe(artifact.inputHash);
    expect(magnetosphereReference.unitsHash).toBe(
      'sha256:b9ef2d46ec5f2d0c1fb8a2866012e9340a67f188ebc8a579b93ce61e72f4b4a5'
    );
    expect(magnetosphereReference.fieldMap).toMatchObject({
      radiusMeters: 'outputs.radialSamples[].radiusMeters',
      magneticFieldTesla: 'outputs.radialSamples[].magneticFieldTesla',
    });
    expect(magnetosphereReference.fieldTolerances).toMatchObject({
      magneticFieldTeslaRel: 1e-12,
      normalizedFieldAbs: 1e-12,
    });
    expect(magnetosphereReference.fieldObservedDeltas).toMatchObject({
      magneticFieldTeslaRel: 0,
      normalizedFieldAbs: 0,
      radialPowerLawAbs: 0,
      divergenceProxyAbs: 0,
    });
    expect(magnetosphereReference.status).toBe('calibrated-reference-ready');
    expect(magnetosphereReference.ready).toBe(true);
    expect(magnetosphereReference.scientificCoverage).toBe(true);
    expect(magnetosphereReference.scope).toBe('analytic-dipole-magnetosphere-reference-not-full-mhd');
    expect(magnetosphereReference.validationStatus).toBe('pass');
    expect(magnetosphereReference.validation.status).toBe('pass');
    expect(magnetosphereReference.validation.evidence.length).toBeGreaterThan(0);
    expect(magnetosphereReference.blocker).toBeNull();
    expect(magnetosphereReference.blockers).toEqual([]);

    for (const reference of outputs.references.slice(1)) {
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

  it('merges supplied calibrated reference contracts into the magnetar inventory', async () => {
    const artifact = await buildUlgMagnetarDipoleIsingArtifact({
      createdAt: '2026-06-05T00:00:00.000Z',
      inputHash: 'sha256:test-magnetar-input',
      references: [
        {
          id: 'radiation-transport-reference',
          family: 'radiation-transport',
          provider: 'moonlab',
          solverId: 'moonlab-grey-radiation-transport-reference-v0',
          schema: 'moonlab.magnetar.calibrated-reference.v0',
          role: 'peercompute-scientific-tolerance-input',
          contractHash: 'sha256:radiation-transport-contract',
          unitsHash: 'sha256:radiation-transport-units',
          fieldMap: {
            opticalDepth: 'outputs.radiationReference.opticalDepth',
            luminosityFlux: 'outputs.radiationReference.luminosityFlux',
          },
          fieldTolerances: {
            opticalDepthAbs: 1e-6,
            luminosityFluxRel: 1e-6,
          },
          fieldObservedDeltas: {
            opticalDepthAbs: 0,
            luminosityFluxRel: 0,
          },
          label: 'Supplied grey radiation transport reference',
          status: 'calibrated-reference-ready',
          ready: true,
          scientificCoverage: true,
          scope: 'supplied-calibrated-reference-contract',
          validationStatus: 'pass',
          validation: {
            status: 'pass',
            evidence: ['Supplied reduced grey-radiation benchmark contract.'],
          },
          blocker: null,
          blockers: [],
        },
      ],
    });
    const outputs = artifact.outputs as {
      references: Array<{
        id: string;
        family: string;
        solverId: string | null;
        status: string;
        ready: boolean;
        scientificCoverage: boolean;
        scope: string;
        validationStatus: string;
        blocker: string | null;
      }>;
    };

    const radiationReference = outputs.references.find((reference) => reference.family === 'radiation-transport');
    expect(radiationReference?.id).toBe('radiation-transport-reference');
    expect(radiationReference?.solverId).toBe('moonlab-grey-radiation-transport-reference-v0');
    expect(radiationReference?.status).toBe('calibrated-reference-ready');
    expect(radiationReference?.ready).toBe(true);
    expect(radiationReference?.scientificCoverage).toBe(true);
    expect(radiationReference?.scope).toBe('supplied-calibrated-reference-contract');
    expect(radiationReference?.validationStatus).toBe('pass');
    expect(radiationReference?.blocker).toBeNull();
    expect(outputs.references.filter((reference) => reference.ready)).toHaveLength(2);
  });
});
