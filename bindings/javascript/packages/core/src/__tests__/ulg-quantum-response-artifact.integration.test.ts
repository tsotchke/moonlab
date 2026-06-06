import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  buildUlgBellStateArtifact,
  buildUlgMagnetarDipoleIsingArtifact,
} from '../ulg-quantum-response-artifact';

const testFilePath = fileURLToPath(import.meta.url);
const coreRoot = resolve(dirname(testFilePath), '../..');
const emitArtifactCli = resolve(coreRoot, 'scripts/emit-ulg-quantum-response-artifact.mjs');
const magnetarReferenceContractsPath = resolve(
  coreRoot,
  'references/magnetar-calibrated-reference-contracts.json'
);
const TEST_MAGNETAR_INPUT_HASH = `sha256:${'0'.repeat(64)}`;
const VALID_RADIATION_CONTRACT_HASH = `sha256:${'1'.repeat(64)}`;
const VALID_RADIATION_UNITS_HASH = `sha256:${'2'.repeat(64)}`;

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
      inputHash: TEST_MAGNETAR_INPUT_HASH,
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
      inputHash: TEST_MAGNETAR_INPUT_HASH,
      references: [
        {
          id: 'radiation-transport-reference',
          family: 'radiation-transport',
          provider: 'moonlab',
          solverId: 'moonlab-grey-radiation-transport-reference-v0',
          schema: 'moonlab.magnetar.calibrated-reference.v0',
          role: 'peercompute-scientific-tolerance-input',
          contractHash: VALID_RADIATION_CONTRACT_HASH,
          unitsHash: VALID_RADIATION_UNITS_HASH,
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

  it('validates supplied reference contracts through the artifact CLI', () => {
    const tempDir = mkdtempSync(join(tmpdir(), 'moonlab-reference-contracts-'));
    const referencesPath = join(tempDir, 'references.json');
    writeFileSync(
      referencesPath,
      JSON.stringify(
        [
          readyCliReference(
            'pic-kinetic-plasma-reference',
            'pic-kinetic-plasma',
            'moonlab-pic-reference-v0'
          ),
          readyCliReference(
            'radiation-transport-reference',
            'radiation-transport',
            'moonlab-radiation-reference-v0'
          ),
          readyCliReference(
            'relativistic-correction-reference',
            'relativistic-correction',
            'moonlab-relativity-reference-v0'
          ),
        ],
        null,
        2
      )
    );

    const stdout = execFileSync(
      process.execPath,
      [emitArtifactCli, '--validate-references', referencesPath, '--strict'],
      {
        cwd: coreRoot,
        encoding: 'utf8',
      }
    );
    const report = JSON.parse(stdout) as {
      schema: string;
      status: string;
      ready: boolean;
      familyCount: number;
      readyCount: number;
      suppliedCount: number;
      suppliedReadyCount: number;
      invalidSuppliedCount: number;
      unknownCount: number;
      entries: Array<{
        family: string;
        supplied: boolean;
        ready: boolean;
      }>;
    };

    expect(report.schema).toBe('moonlab.magnetar.reference-contract-validation-report.v0');
    expect(report.status).toBe('reference-contract-suite-ready');
    expect(report.ready).toBe(true);
    expect(report.familyCount).toBe(4);
    expect(report.readyCount).toBe(4);
    expect(report.suppliedCount).toBe(3);
    expect(report.suppliedReadyCount).toBe(3);
    expect(report.invalidSuppliedCount).toBe(0);
    expect(report.unknownCount).toBe(0);
    expect(report.entries.find((entry) => entry.family === 'magnetosphere-mhd')?.supplied).toBe(false);
    expect(report.entries.filter((entry) => entry.supplied && entry.ready)).toHaveLength(3);
  });

  it('emits a four-family magnetar artifact from checked-in reference contracts', async () => {
    const references = JSON.parse(readFileSync(magnetarReferenceContractsPath, 'utf8'));
    const artifact = await buildUlgMagnetarDipoleIsingArtifact({
      createdAt: '2026-06-05T00:00:00.000Z',
      inputHash: TEST_MAGNETAR_INPUT_HASH,
      references: references.references,
    });
    const outputs = artifact.outputs as {
      references: Array<{
        id: string;
        family: string;
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
        blocker: string | null;
      }>;
    };

    expect(outputs.references).toHaveLength(4);
    expect(outputs.references.every((reference) => reference.ready)).toBe(true);
    expect(outputs.references.every((reference) => reference.scientificCoverage)).toBe(true);
    expect(outputs.references.every((reference) => reference.validationStatus === 'pass')).toBe(true);
    expect(outputs.references.every((reference) => reference.contractHash?.startsWith('sha256:'))).toBe(true);
    expect(outputs.references.every((reference) => reference.unitsHash?.startsWith('sha256:'))).toBe(true);
    expect(outputs.references.every((reference) => reference.fieldMap != null)).toBe(true);
    expect(outputs.references.every((reference) => reference.fieldTolerances != null)).toBe(true);
    expect(outputs.references.every((reference) => reference.fieldObservedDeltas != null)).toBe(true);
    expect(outputs.references.every((reference) => reference.blocker == null)).toBe(true);
    expect(outputs.references.map((reference) => reference.family)).toEqual([
      'magnetosphere-mhd',
      'pic-kinetic-plasma',
      'radiation-transport',
      'relativistic-correction',
    ]);
    expect(outputs.references.find((reference) => (
      reference.family === 'pic-kinetic-plasma'
    ))?.solverId).toBe('moonlab-reduced-pic-kinetic-plasma-reference-v0');
    expect(outputs.references.find((reference) => (
      reference.family === 'radiation-transport'
    ))?.solverId).toBe('moonlab-reduced-grey-radiation-reference-v0');
    expect(outputs.references.find((reference) => (
      reference.family === 'relativistic-correction'
    ))?.solverId).toBe('moonlab-reduced-post-newtonian-reference-v0');
  });

  it('strictly validates the checked-in magnetar reference contract asset', () => {
    const stdout = execFileSync(
      process.execPath,
      [emitArtifactCli, '--validate-references', magnetarReferenceContractsPath, '--strict'],
      {
        cwd: coreRoot,
        encoding: 'utf8',
      }
    );
    const report = JSON.parse(stdout) as {
      status: string;
      ready: boolean;
      familyCount: number;
      readyCount: number;
      suppliedCount: number;
      suppliedReadyCount: number;
      invalidSuppliedCount: number;
      unknownCount: number;
      blockers: string[];
    };

    expect(report.status).toBe('reference-contract-suite-ready');
    expect(report.ready).toBe(true);
    expect(report.familyCount).toBe(4);
    expect(report.readyCount).toBe(4);
    expect(report.suppliedCount).toBe(3);
    expect(report.suppliedReadyCount).toBe(3);
    expect(report.invalidSuppliedCount).toBe(0);
    expect(report.unknownCount).toBe(0);
    expect(report.blockers).toEqual([]);
  });

  it('normalizes checked-in magnetar reference contracts through the artifact CLI', () => {
    const stdout = execFileSync(
      process.execPath,
      [emitArtifactCli, '--normalize-references', magnetarReferenceContractsPath, '--strict'],
      {
        cwd: coreRoot,
        encoding: 'utf8',
      }
    );
    const suite = JSON.parse(stdout) as {
      schema: string;
      status: string;
      ready: boolean;
      familyCount: number;
      readyCount: number;
      suppliedCount: number;
      suppliedReadyCount: number;
      invalidSuppliedCount: number;
      unknownCount: number;
      source: {
        builtInReferenceFamilies: string[];
      };
      references: Array<{
        family: string;
        ready: boolean;
        scientificCoverage: boolean;
        contractHash: string | null;
        unitsHash: string | null;
      }>;
      validation: {
        ready: boolean;
        errors: string[];
      };
      blockers: string[];
      errors: string[];
    };

    expect(suite.schema).toBe('moonlab.magnetar.normalized-reference-suite.v0');
    expect(suite.status).toBe('reference-contract-suite-ready');
    expect(suite.ready).toBe(true);
    expect(suite.familyCount).toBe(4);
    expect(suite.readyCount).toBe(4);
    expect(suite.suppliedCount).toBe(3);
    expect(suite.suppliedReadyCount).toBe(3);
    expect(suite.invalidSuppliedCount).toBe(0);
    expect(suite.unknownCount).toBe(0);
    expect(suite.source.builtInReferenceFamilies).toEqual(['magnetosphere-mhd']);
    expect(suite.references).toHaveLength(4);
    expect(suite.references.map((reference) => reference.family)).toEqual([
      'magnetosphere-mhd',
      'pic-kinetic-plasma',
      'radiation-transport',
      'relativistic-correction',
    ]);
    expect(suite.references.every((reference) => reference.ready)).toBe(true);
    expect(suite.references.every((reference) => reference.scientificCoverage)).toBe(true);
    expect(suite.references.every((reference) => (
      reference.contractHash?.match(/^sha256:[0-9a-f]{64}$/)
    ))).toBe(true);
    expect(suite.references.every((reference) => (
      reference.unitsHash?.match(/^sha256:[0-9a-f]{64}$/)
    ))).toBe(true);
    expect(suite.validation.ready).toBe(true);
    expect(suite.validation.errors).toEqual([]);
    expect(suite.blockers).toEqual([]);
    expect(suite.errors).toEqual([]);
  });
});

function readyCliReference(id: string, family: string, solverId: string) {
  return {
    id,
    family,
    provider: 'moonlab',
    solverId,
    schema: 'moonlab.magnetar.calibrated-reference.v0',
    role: 'peercompute-scientific-tolerance-input',
    contractHash: `sha256:${'3'.repeat(64)}`,
    unitsHash: `sha256:${'4'.repeat(64)}`,
    fieldMap: {
      scalar: `outputs.${family}.scalar`,
    },
    fieldTolerances: {
      scalarAbs: 1e-6,
    },
    fieldObservedDeltas: {
      scalarAbs: 0,
    },
    label: `${family} supplied reference`,
    status: 'calibrated-reference-ready',
    ready: true,
    scientificCoverage: true,
    scope: 'supplied-calibrated-reference-contract',
    validationStatus: 'pass',
    validation: {
      status: 'pass',
      evidence: [`${family} reduced reference contract.`],
    },
    blocker: null,
    blockers: [],
  };
}
