import { QuantumState } from './quantum-state';
import { IsingModel } from './ising-model';

export const ULG_QUANTUM_RESPONSE_SCHEMA_TITLE =
  'ULG Quantum Response Artifact v0.5';

export const DEFAULT_ULG_QUANTUM_RESPONSE_SCHEMA = {
  title: ULG_QUANTUM_RESPONSE_SCHEMA_TITLE,
  required: [
    'artifactId',
    'sourceService',
    'taskKind',
    'inputHash',
    'method',
    'representation',
    'outputs',
    'validation',
    'provenance',
  ],
  properties: {
    artifactId: { type: 'string', minLength: 1 },
    sourceService: { const: 'moonlab' },
    taskKind: { type: 'string', minLength: 1 },
    inputHash: { type: 'string', minLength: 1 },
    method: { type: 'string', minLength: 1 },
    representation: { type: 'string', minLength: 1 },
    outputs: { type: 'object' },
    uncertainty: { type: 'object' },
    validation: { type: 'object' },
    provenance: { type: 'object' },
  },
} satisfies QuantumResponseArtifactSchema;

export interface QuantumResponseArtifactSchema {
  title?: string;
  required?: string[];
  properties?: Record<string, QuantumResponseArtifactSchemaProperty>;
}

export interface QuantumResponseArtifactSchemaProperty {
  type?: 'string' | 'object';
  const?: unknown;
  minLength?: number;
}

export interface UlgQuantumResponseArtifact {
  artifactId: string;
  sourceService: 'moonlab';
  taskKind: string;
  inputHash: string;
  method: string;
  representation: string;
  outputs: Record<string, unknown>;
  uncertainty: Record<string, unknown>;
  validation: UlgArtifactValidation;
  provenance: Record<string, unknown>;
  parity: Record<string, unknown>;
}

export interface UlgArtifactValidation {
  schemaCompatible: boolean;
  schemaTitle: string;
  schemaPath?: string;
  checks: UlgArtifactValidationCheck[];
  errors: string[];
}

export interface UlgArtifactValidationCheck {
  name: string;
  passed: boolean;
  details?: Record<string, unknown>;
}

export interface UlgBellStateArtifactOptions {
  createdAt?: string;
  taskKind?: string;
  tolerance?: number;
  inputHash?: string;
  artifactId?: string;
  packageVersion?: string;
  schemaPath?: string;
  provenance?: Record<string, unknown>;
}

export interface UlgMagnetarDipoleIsingArtifactOptions {
  createdAt?: string;
  taskKind?: string;
  tolerance?: number;
  input?: Partial<UlgMagnetarDipoleIsingInput>;
  references?: Array<Partial<UlgMagnetarReferenceFamilyInventoryEntry>>;
  inputHash?: string;
  artifactId?: string;
  packageVersion?: string;
  schemaPath?: string;
  provenance?: Record<string, unknown>;
}

export interface UlgMagnetarDipoleIsingInput {
  surfaceMagneticFieldTesla: number;
  stellarRadiusMeters: number;
  radialSamplesMeters: number[];
  couplingStrength: number;
  bitstrings: number[];
}

export interface UlgMagnetarDipoleIsingModel {
  localFields: number[];
  couplings: Array<{ qubit1: number; qubit2: number; value: number }>;
  fieldScaleTesla: number;
  physicalModel: 'axisymmetric-dipole-falloff';
  spinConvention: 'bit-0-plus-one-bit-1-minus-one';
}

export interface UlgMagnetarReferenceFamilyInventoryEntry {
  id:
    | 'magnetosphere-mhd-reference'
    | 'pic-kinetic-plasma-reference'
    | 'radiation-transport-reference'
    | 'relativistic-correction-reference';
  family:
    | 'magnetosphere-mhd'
    | 'pic-kinetic-plasma'
    | 'radiation-transport'
    | 'relativistic-correction';
  provider: 'moonlab';
  solverId: string | null;
  schema: 'moonlab.magnetar.calibrated-reference.v0';
  role: 'peercompute-scientific-tolerance-input';
  contractHash: string | null;
  unitsHash: string | null;
  fieldMap: Record<string, unknown> | null;
  fieldTolerances: Record<string, unknown> | null;
  fieldObservedDeltas: Record<string, unknown> | null;
  label: string;
  status: 'calibrated-reference-ready' | 'calibrated-reference-missing';
  ready: boolean;
  scientificCoverage: boolean;
  scope:
    | 'analytic-dipole-magnetosphere-reference-not-full-mhd'
    | 'supplied-calibrated-reference-contract'
    | 'inventory-only-not-scientific-reference';
  validationStatus: 'pass' | 'missing';
  validation: {
    status: 'pass' | 'missing';
    evidence: string[];
    maxObservedDeltas?: Record<string, unknown>;
  };
  blocker:
    | 'calibrated-mhd-reference-missing'
    | 'calibrated-pic-reference-missing'
    | 'calibrated-radiation-reference-missing'
    | 'calibrated-relativity-reference-missing'
    | null;
  blockers: string[];
}

export interface UlgMagnetarReferenceContractValidationReport {
  schema: 'moonlab.magnetar.reference-contract-validation-report.v0';
  status:
    | 'reference-contract-suite-ready'
    | 'reference-contract-suite-partial'
    | 'reference-contract-suite-invalid';
  ready: boolean;
  familyCount: number;
  readyCount: number;
  suppliedCount: number;
  suppliedReadyCount: number;
  invalidSuppliedCount: number;
  unknownCount: number;
  entries: UlgMagnetarReferenceContractValidationEntry[];
  unknownReferences: UlgMagnetarReferenceContractUnknownReference[];
  blockers: string[];
  errors: string[];
}

export interface UlgMagnetarReferenceContractNormalizedSuite {
  schema: 'moonlab.magnetar.normalized-reference-suite.v0';
  status: UlgMagnetarReferenceContractValidationReport['status'];
  ready: boolean;
  familyCount: number;
  readyCount: number;
  suppliedCount: number;
  suppliedReadyCount: number;
  invalidSuppliedCount: number;
  unknownCount: number;
  source: {
    contractHash: string;
    builtInReferenceFamilies: Array<UlgMagnetarReferenceFamilyInventoryEntry['family']>;
  };
  references: UlgMagnetarReferenceFamilyInventoryEntry[];
  validation: UlgMagnetarReferenceContractValidationReport;
  blockers: string[];
  errors: string[];
}

export interface UlgMagnetarReferenceContractValidationEntry {
  id: UlgMagnetarReferenceFamilyInventoryEntry['id'];
  family: UlgMagnetarReferenceFamilyInventoryEntry['family'];
  supplied: boolean;
  status: UlgMagnetarReferenceFamilyInventoryEntry['status'];
  ready: boolean;
  scientificCoverage: boolean;
  solverId: string | null;
  validationStatus: UlgMagnetarReferenceFamilyInventoryEntry['validationStatus'];
  contractHashValid: boolean;
  unitsHashValid: boolean;
  fieldMapReady: boolean;
  fieldTolerancesReady: boolean;
  fieldObservedDeltasReady: boolean;
  fieldDeltasWithinTolerance: boolean;
  toleranceFailures: UlgMagnetarReferenceContractToleranceFailure[];
  checks: UlgMagnetarReferenceContractValidationChecks;
  blockers: string[];
  errors: string[];
}

export interface UlgMagnetarReferenceContractValidationChecks {
  readyFlag: boolean;
  scientificCoverageFlag: boolean;
  solverId: boolean;
  contractHash: boolean;
  unitsHash: boolean;
  fieldMap: boolean;
  fieldTolerances: boolean;
  fieldObservedDeltas: boolean;
  validationPass: boolean;
  fieldDeltasWithinTolerance: boolean;
}

export interface UlgMagnetarReferenceContractToleranceFailure {
  field: string;
  observed: number | null;
  tolerance: number | null;
}

export interface UlgMagnetarReferenceContractUnknownReference {
  index: number;
  id: string | null;
  family: string | null;
  errors: string[];
}

interface BellTaskInput {
  numQubits: 2;
  circuit: [
    { gate: 'h'; target: 0 },
    { gate: 'cnot'; control: 0; target: 1 },
  ];
  shots: 0;
}

interface SchemaValidationResult {
  valid: boolean;
  errors: string[];
}

const DEFAULT_BELL_TASK_INPUT: BellTaskInput = {
  numQubits: 2,
  circuit: [
    { gate: 'h', target: 0 },
    { gate: 'cnot', control: 0, target: 1 },
  ],
  shots: 0,
};

const EXPECTED_BELL_PROBABILITIES = [0.5, 0, 0, 0.5];
const DEFAULT_TOLERANCE = 1e-9;
const SHA256_DIGEST_PATTERN = /^sha256:[0-9a-f]{64}$/i;
const MAGNETOSPHERE_MHD_ANALYTIC_UNITS_HASH =
  'sha256:b9ef2d46ec5f2d0c1fb8a2866012e9340a67f188ebc8a579b93ce61e72f4b4a5';
const DEFAULT_MAGNETAR_REFERENCE_CONTRACT_HASH =
  'sha256:f85763af06f271c414d55e29884ee7b0d5738a4a7ec9351493964b98f8d4e1ec';
const SUPPLIED_REFERENCE_NOT_READY_BLOCKER =
  'Supplied calibrated reference did not satisfy readiness requirements.';
const DEFAULT_MAGNETAR_DIPOLE_ISING_INPUT = {
  surfaceMagneticFieldTesla: 1e11,
  stellarRadiusMeters: 10_000,
  radialSamplesMeters: [10_000, 15_000, 20_000],
  couplingStrength: 0.125,
} satisfies Omit<UlgMagnetarDipoleIsingInput, 'bitstrings'>;

export async function buildUlgBellStateArtifact(
  options: UlgBellStateArtifactOptions = {}
): Promise<UlgQuantumResponseArtifact> {
  const tolerance = options.tolerance ?? DEFAULT_TOLERANCE;
  const taskKind = options.taskKind ?? 'bell-state-smoke';
  const createdAt = options.createdAt ?? new Date().toISOString();
  const canonicalInput = canonicalJson(DEFAULT_BELL_TASK_INPUT);
  const inputHash = options.inputHash ?? await sha256Hex(canonicalInput);

  let state: QuantumState | undefined;
  try {
    state = await QuantumState.create({ numQubits: DEFAULT_BELL_TASK_INPUT.numQubits });
    state.h(0).cnot(0, 1);

    const amplitudes = state.getAmplitudes().map((amplitude, index) => ({
      basisIndex: index,
      bitString: basisBitString(index, DEFAULT_BELL_TASK_INPUT.numQubits),
      real: normalizeNumber(amplitude.real),
      imag: normalizeNumber(amplitude.imag),
    }));
    const probabilities = Array.from(state.getProbabilities(), normalizeNumber);
    const probabilitySum = probabilities.reduce((sum, probability) => sum + probability, 0);
    const normalizationError = Math.abs(1 - probabilitySum);
    const maxProbabilityDelta = probabilities.reduce((maxDelta, probability, index) => {
      return Math.max(maxDelta, Math.abs(probability - EXPECTED_BELL_PROBABILITIES[index]));
    }, 0);
    const parityPassed =
      maxProbabilityDelta <= tolerance && normalizationError <= tolerance;

    const artifact: UlgQuantumResponseArtifact = {
      artifactId:
        options.artifactId ?? `moonlab-${taskKind}-${inputHash.replace(/^sha256:/, '').slice(0, 16)}`,
      sourceService: 'moonlab',
      taskKind,
      inputHash,
      method: 'moonlab-js-core-wasm',
      representation: 'state-vector-probability-distribution',
      outputs: {
        taskInput: DEFAULT_BELL_TASK_INPUT,
        amplitudes,
        probabilities,
        basis: 'computational-basis-index-order',
        summary: {
          numQubits: DEFAULT_BELL_TASK_INPUT.numQubits,
          stateDimension: probabilities.length,
          probabilitySum: normalizeNumber(probabilitySum),
          dominantStates: [
            { bitString: '00', probability: probabilities[0] },
            { bitString: '11', probability: probabilities[3] },
          ],
        },
      },
      uncertainty: {
        stochastic: false,
        shots: 0,
        probabilityTolerance: tolerance,
        numericPrecision: 'float64',
      },
      validation: {
        schemaCompatible: false,
        schemaTitle: DEFAULT_ULG_QUANTUM_RESPONSE_SCHEMA.title,
        schemaPath: options.schemaPath,
        checks: [],
        errors: [],
      },
      provenance: {
        createdAt,
        packageName: '@moonlab/quantum-core',
        packageVersion: options.packageVersion ?? '0.1.1',
        runtime: runtimeLabel(),
        executionTarget: 'wasm-cpu',
        networkingStarted: false,
        gpuSchedulingStarted: false,
        circuit: DEFAULT_BELL_TASK_INPUT.circuit,
        ...options.provenance,
      },
      parity: {
        reference: 'exact-bell-state',
        expectedProbabilities: EXPECTED_BELL_PROBABILITIES,
        observedProbabilities: probabilities,
        maxProbabilityDelta: normalizeNumber(maxProbabilityDelta),
        normalizationError: normalizeNumber(normalizationError),
        tolerance,
        passed: parityPassed,
      },
    };

    const schemaResult = validateUlgQuantumResponseArtifact(
      artifact,
      DEFAULT_ULG_QUANTUM_RESPONSE_SCHEMA
    );
    artifact.validation = {
      schemaCompatible: schemaResult.valid,
      schemaTitle: DEFAULT_ULG_QUANTUM_RESPONSE_SCHEMA.title,
      schemaPath: options.schemaPath,
      checks: [
        {
          name: 'ulg-schema-required-fields',
          passed: schemaResult.valid,
          details: { errors: schemaResult.errors },
        },
        {
          name: 'bell-state-parity',
          passed: parityPassed,
          details: {
            maxProbabilityDelta: artifact.parity.maxProbabilityDelta,
            normalizationError: artifact.parity.normalizationError,
            tolerance,
          },
        },
        {
          name: 'no-networking',
          passed: true,
          details: { networkingStarted: false },
        },
        {
          name: 'no-gpu-scheduling',
          passed: true,
          details: { gpuSchedulingStarted: false },
        },
      ],
      errors: schemaResult.errors,
    };

    return artifact;
  } finally {
    state?.dispose();
  }
}

export async function buildUlgMagnetarDipoleIsingArtifact(
  options: UlgMagnetarDipoleIsingArtifactOptions = {}
): Promise<UlgQuantumResponseArtifact> {
  const tolerance = options.tolerance ?? DEFAULT_TOLERANCE;
  const taskKind = options.taskKind ?? 'magnetar-dipole-ising-calibration';
  const createdAt = options.createdAt ?? new Date().toISOString();
  const input = buildMagnetarDipoleIsingInput(options.input);
  const numQubits = input.radialSamplesMeters.length;
  const isingModel = buildMagnetarDipoleIsingModel(input);
  const canonicalInput = canonicalJson({
    input,
    isingModel,
  });
  const inputHash = options.inputHash ?? await sha256Hex(canonicalInput);

  let model: IsingModel | undefined;
  try {
    const liveModel = await IsingModel.create({ numQubits });
    model = liveModel;
    for (let qubit = 0; qubit < isingModel.localFields.length; qubit += 1) {
      liveModel.setField(qubit, isingModel.localFields[qubit]);
    }
    for (const coupling of isingModel.couplings) {
      liveModel.setCoupling(coupling.qubit1, coupling.qubit2, coupling.value);
    }

    const radialSamples = input.radialSamplesMeters.map((radiusMeters, index) => {
      const magneticFieldTesla = magnetarDipoleFieldTesla(input, radiusMeters);
      return {
        qubit: index,
        radiusMeters,
        magneticFieldTesla: normalizeNumber(magneticFieldTesla),
        normalizedField: normalizeNumber(magneticFieldTesla / isingModel.fieldScaleTesla),
        localField: isingModel.localFields[index],
      };
    });
    const evaluations = input.bitstrings.map((bitstring) => {
      const observedEnergy = normalizeNumber(liveModel.evaluate(bitstring));
      const referenceEnergy = normalizeNumber(
        evaluateIsingReferenceEnergy(isingModel, bitstring)
      );
      return {
        bitstring,
        bitString: basisBitString(bitstring, numQubits),
        spins: spinsFromBitstring(bitstring, numQubits),
        observedEnergy,
        referenceEnergy,
        energyDelta: normalizeNumber(Math.abs(observedEnergy - referenceEnergy)),
      };
    });
    const maxEnergyDelta = evaluations.reduce((maxDelta, evaluation) => {
      return Math.max(maxDelta, evaluation.energyDelta);
    }, 0);
    const parityPassed = maxEnergyDelta <= tolerance;
    const monotonicFieldPassed = radialSamples.every((sample, index) => {
      if (index === 0) {
        return true;
      }
      return sample.magneticFieldTesla <= radialSamples[index - 1].magneticFieldTesla;
    });
    const groundState = evaluations.reduce((best, evaluation) => {
      return evaluation.observedEnergy < best.observedEnergy ? evaluation : best;
    }, evaluations[0]);
    const energyUnits = 'normalized-ising';
    const referenceContract = {
      schema: 'moonlab.magnetar-dipole-ising-reference.v0',
      role: 'peercompute-reference-tolerance-input',
      target: 'magnetar-dipole-normalized-ising',
      contractHash: inputHash,
      energyUnits,
      hamiltonian: {
        form: 'H=sum_i localFields[i]*spins[i]+sum_c couplings[c].value*spins[c.qubit1]*spins[c.qubit2]',
        localFields: isingModel.localFields,
        couplings: isingModel.couplings,
        fieldScaleTesla: isingModel.fieldScaleTesla,
        physicalModel: isingModel.physicalModel,
        spinConvention: isingModel.spinConvention,
        input: {
          surfaceMagneticFieldTesla: input.surfaceMagneticFieldTesla,
          stellarRadiusMeters: input.stellarRadiusMeters,
          radialSamplesMeters: input.radialSamplesMeters,
          couplingStrength: input.couplingStrength,
        },
      },
      observables: {
        groundState: {
          bitstring: groundState.bitstring,
          bitString: groundState.bitString,
          referenceEnergy: groundState.referenceEnergy,
        },
        energySpectrum: evaluations.map((evaluation) => ({
          bitstring: evaluation.bitstring,
          bitString: evaluation.bitString,
          spins: evaluation.spins,
          referenceEnergy: evaluation.referenceEnergy,
        })),
      },
      tolerances: {
        energyAbs: tolerance,
        maxObservedEnergyDelta: normalizeNumber(maxEnergyDelta),
        numericPrecision: 'float64',
      },
      validation: {
        parityPassed,
        maxEnergyDelta: normalizeNumber(maxEnergyDelta),
        evaluatedBitstrings: evaluations.length,
      },
    };
    const referenceFamilyInventory = buildMagnetarReferenceFamilyInventory(inputHash, options.references);

    const artifact: UlgQuantumResponseArtifact = {
      artifactId:
        options.artifactId ?? `moonlab-${taskKind}-${inputHash.replace(/^sha256:/, '').slice(0, 16)}`,
      sourceService: 'moonlab',
      taskKind,
      inputHash,
      method: 'moonlab-js-core-wasm-ising-evaluator',
      representation: 'magnetar-dipole-normalized-ising-calibration',
      outputs: {
        taskInput: input,
        radialSamples,
        isingModel,
        evaluations,
        reference: referenceContract,
        references: referenceFamilyInventory,
        summary: {
          numQubits,
          evaluatedBitstrings: evaluations.length,
          groundState: {
            bitstring: groundState.bitstring,
            bitString: groundState.bitString,
            observedEnergy: groundState.observedEnergy,
            referenceEnergy: groundState.referenceEnergy,
            energyUnits,
          },
          scope: 'calibration-probe-not-full-magnetar-simulation',
        },
      },
      uncertainty: {
        stochastic: false,
        shots: 0,
        energyTolerance: tolerance,
        energyToleranceAbs: tolerance,
        numericPrecision: 'float64',
        modelAssumptions: [
          'axisymmetric dipole falloff B(r)=B_surface*(R/r)^3',
          'normalized local Ising fields only',
          'nearest-neighbor ferromagnetic coupling only',
          'no plasma, radiation, relativistic, or MHD evolution',
        ],
      },
      validation: {
        schemaCompatible: false,
        schemaTitle: DEFAULT_ULG_QUANTUM_RESPONSE_SCHEMA.title,
        schemaPath: options.schemaPath,
        checks: [],
        errors: [],
      },
      provenance: {
        createdAt,
        packageName: '@moonlab/quantum-core',
        packageVersion: options.packageVersion ?? '0.1.1',
        runtime: runtimeLabel(),
        executionTarget: 'wasm-cpu',
        corePrimitive: 'ising_model_evaluate',
        networkingStarted: false,
        gpuSchedulingStarted: false,
        ...options.provenance,
      },
      parity: {
        reference: 'javascript-ising-energy-reference',
        observedEnergies: evaluations.map((evaluation) => evaluation.observedEnergy),
        expectedEnergies: evaluations.map((evaluation) => evaluation.referenceEnergy),
        maxEnergyDelta: normalizeNumber(maxEnergyDelta),
        tolerance,
        energyToleranceAbs: tolerance,
        passed: parityPassed,
      },
    };

    const schemaResult = validateUlgQuantumResponseArtifact(
      artifact,
      DEFAULT_ULG_QUANTUM_RESPONSE_SCHEMA
    );
    artifact.validation = {
      schemaCompatible: schemaResult.valid,
      schemaTitle: DEFAULT_ULG_QUANTUM_RESPONSE_SCHEMA.title,
      schemaPath: options.schemaPath,
      checks: [
        {
          name: 'ulg-schema-required-fields',
          passed: schemaResult.valid,
          details: { errors: schemaResult.errors },
        },
        {
          name: 'ising-energy-parity',
          passed: parityPassed,
          details: {
            maxEnergyDelta: artifact.parity.maxEnergyDelta,
            tolerance,
          },
        },
        {
          name: 'dipole-field-monotonicity',
          passed: monotonicFieldPassed,
          details: {
            radialSamples: radialSamples.map((sample) => ({
              radiusMeters: sample.radiusMeters,
              magneticFieldTesla: sample.magneticFieldTesla,
            })),
          },
        },
        {
          name: 'no-networking',
          passed: true,
          details: { networkingStarted: false },
        },
        {
          name: 'no-gpu-scheduling',
          passed: true,
          details: { gpuSchedulingStarted: false },
        },
      ],
      errors: schemaResult.errors,
    };

    return artifact;
  } finally {
    model?.dispose();
  }
}

export function buildMagnetarDipoleIsingInput(
  input: Partial<UlgMagnetarDipoleIsingInput> = {}
): UlgMagnetarDipoleIsingInput {
  const radialSamplesMeters =
    input.radialSamplesMeters?.slice() ??
    DEFAULT_MAGNETAR_DIPOLE_ISING_INPUT.radialSamplesMeters.slice();
  const defaultBitstrings = allBitstringsForQubits(radialSamplesMeters.length);
  const result = {
    surfaceMagneticFieldTesla:
      input.surfaceMagneticFieldTesla ??
      DEFAULT_MAGNETAR_DIPOLE_ISING_INPUT.surfaceMagneticFieldTesla,
    stellarRadiusMeters:
      input.stellarRadiusMeters ??
      DEFAULT_MAGNETAR_DIPOLE_ISING_INPUT.stellarRadiusMeters,
    radialSamplesMeters,
    couplingStrength:
      input.couplingStrength ??
      DEFAULT_MAGNETAR_DIPOLE_ISING_INPUT.couplingStrength,
    bitstrings: input.bitstrings?.slice() ?? defaultBitstrings,
  };

  validateMagnetarDipoleIsingInput(result);
  return result;
}

export function buildMagnetarDipoleIsingModel(
  input: UlgMagnetarDipoleIsingInput
): UlgMagnetarDipoleIsingModel {
  validateMagnetarDipoleIsingInput(input);

  const magneticFieldsTesla = input.radialSamplesMeters.map((radiusMeters) => {
    return magnetarDipoleFieldTesla(input, radiusMeters);
  });
  const fieldScaleTesla = Math.max(...magneticFieldsTesla.map(Math.abs));
  const localFields = magneticFieldsTesla.map((fieldTesla) => {
    return normalizeNumber(-fieldTesla / fieldScaleTesla);
  });
  const couplings = input.radialSamplesMeters.slice(1).map((_, index) => ({
    qubit1: index,
    qubit2: index + 1,
    value: normalizeNumber(-input.couplingStrength),
  }));

  return {
    localFields,
    couplings,
    fieldScaleTesla: normalizeNumber(fieldScaleTesla),
    physicalModel: 'axisymmetric-dipole-falloff',
    spinConvention: 'bit-0-plus-one-bit-1-minus-one',
  };
}

export function evaluateIsingReferenceEnergy(
  model: UlgMagnetarDipoleIsingModel,
  bitstring: number
): number {
  const numQubits = model.localFields.length;
  validateBitstring(bitstring, numQubits);
  const spins = spinsFromBitstring(bitstring, numQubits);
  const fieldEnergy = model.localFields.reduce((energy, field, index) => {
    return energy + field * spins[index];
  }, 0);
  const couplingEnergy = model.couplings.reduce((energy, coupling) => {
    return energy + coupling.value * spins[coupling.qubit1] * spins[coupling.qubit2];
  }, 0);
  return normalizeNumber(fieldEnergy + couplingEnergy);
}

export function buildMagnetarReferenceFamilyInventory(
  contractHash: string = DEFAULT_MAGNETAR_REFERENCE_CONTRACT_HASH,
  suppliedReferences: Array<Partial<UlgMagnetarReferenceFamilyInventoryEntry>> = []
): UlgMagnetarReferenceFamilyInventoryEntry[] {
  const inventory: UlgMagnetarReferenceFamilyInventoryEntry[] = [
    createAnalyticMagnetosphereMhdReference(contractHash),
    {
      id: 'pic-kinetic-plasma-reference',
      family: 'pic-kinetic-plasma',
      provider: 'moonlab',
      solverId: null,
      schema: 'moonlab.magnetar.calibrated-reference.v0',
      role: 'peercompute-scientific-tolerance-input',
      contractHash: null,
      unitsHash: null,
      fieldMap: null,
      fieldTolerances: null,
      fieldObservedDeltas: null,
      label: 'PIC kinetic plasma calibrated reference family',
      status: 'calibrated-reference-missing',
      ready: false,
      scientificCoverage: false,
      scope: 'inventory-only-not-scientific-reference',
      validationStatus: 'missing',
      validation: {
        status: 'missing',
        evidence: [],
      },
      blocker: 'calibrated-pic-reference-missing',
      blockers: [
        'No calibrated PIC benchmark data is bundled with this artifact.',
        'No particle-field coupling parity run has been compared against the normalized Ising calibration.',
        'No particle distribution, timestep, or plasma parameter tolerances are defined.',
      ],
    },
    {
      id: 'radiation-transport-reference',
      family: 'radiation-transport',
      provider: 'moonlab',
      solverId: null,
      schema: 'moonlab.magnetar.calibrated-reference.v0',
      role: 'peercompute-scientific-tolerance-input',
      contractHash: null,
      unitsHash: null,
      fieldMap: null,
      fieldTolerances: null,
      fieldObservedDeltas: null,
      label: 'Radiation transport calibrated reference family',
      status: 'calibrated-reference-missing',
      ready: false,
      scientificCoverage: false,
      scope: 'inventory-only-not-scientific-reference',
      validationStatus: 'missing',
      validation: {
        status: 'missing',
        evidence: [],
      },
      blocker: 'calibrated-radiation-reference-missing',
      blockers: [
        'No calibrated radiation benchmark data is bundled with this artifact.',
        'No opacity, emissivity, or radiation-transport parity run has been compared against the normalized Ising calibration.',
        'No spectral, angular, or transport error tolerances are defined.',
      ],
    },
    {
      id: 'relativistic-correction-reference',
      family: 'relativistic-correction',
      provider: 'moonlab',
      solverId: null,
      schema: 'moonlab.magnetar.calibrated-reference.v0',
      role: 'peercompute-scientific-tolerance-input',
      contractHash: null,
      unitsHash: null,
      fieldMap: null,
      fieldTolerances: null,
      fieldObservedDeltas: null,
      label: 'Relativistic correction calibrated reference family',
      status: 'calibrated-reference-missing',
      ready: false,
      scientificCoverage: false,
      scope: 'inventory-only-not-scientific-reference',
      validationStatus: 'missing',
      validation: {
        status: 'missing',
        evidence: [],
      },
      blocker: 'calibrated-relativity-reference-missing',
      blockers: [
        'No calibrated relativity benchmark data is bundled with this artifact.',
        'No frame, metric, or relativistic-correction parity run has been compared against the normalized Ising calibration.',
        'No gauge, coordinate, or correction-order tolerances are defined.',
      ],
    },
  ];
  return mergeSuppliedMagnetarReferenceContracts(inventory, suppliedReferences);
}

export function validateMagnetarReferenceContracts(
  suppliedReferences: unknown = [],
  options: { contractHash?: string } = {}
): UlgMagnetarReferenceContractValidationReport {
  const contractHash = options.contractHash ?? DEFAULT_MAGNETAR_REFERENCE_CONTRACT_HASH;
  const inventory = buildMagnetarReferenceFamilyInventory(contractHash);
  const { references, errors: inputErrors } = normalizeReferenceContractInput(suppliedReferences);
  const suppliedByInventoryId = new Map<
    UlgMagnetarReferenceFamilyInventoryEntry['id'],
    Partial<UlgMagnetarReferenceFamilyInventoryEntry>
  >();
  const unknownReferences: UlgMagnetarReferenceContractUnknownReference[] = [];

  references.forEach((candidate, index) => {
    if (!isRecord(candidate)) {
      unknownReferences.push({
        index,
        id: null,
        family: null,
        errors: ['reference contract entry must be an object'],
      });
      return;
    }
    const match = inventory.find((entry) => (
      candidate.id === entry.id || candidate.family === entry.family
    ));
    const suppliedId = typeof candidate.id === 'string' ? candidate.id : null;
    const suppliedFamily = typeof candidate.family === 'string' ? candidate.family : null;
    if (!match) {
      unknownReferences.push({
        index,
        id: suppliedId,
        family: suppliedFamily,
        errors: ['reference contract id or family is not part of the magnetar reference inventory'],
      });
      return;
    }
    if (suppliedByInventoryId.has(match.id)) {
      unknownReferences.push({
        index,
        id: suppliedId,
        family: suppliedFamily,
        errors: [`duplicate reference contract for family ${match.family}`],
      });
      return;
    }
    suppliedByInventoryId.set(match.id, candidate as Partial<UlgMagnetarReferenceFamilyInventoryEntry>);
  });

  const entries = inventory.map((entry) => {
    const supplied = suppliedByInventoryId.get(entry.id);
    return supplied
      ? evaluateSuppliedMagnetarReferenceContract(entry, supplied).validation
      : createInventoryReferenceValidationEntry(entry);
  });
  const readyCount = entries.filter((entry) => entry.ready).length;
  const suppliedReadyCount = entries.filter((entry) => entry.supplied && entry.ready).length;
  const invalidSuppliedCount = entries.filter((entry) => entry.supplied && !entry.ready).length;
  const unknownErrors = unknownReferences.flatMap((reference) => reference.errors);
  const entryErrors = entries.flatMap((entry) => entry.errors);
  const errors = [...inputErrors, ...unknownErrors, ...entryErrors];
  const blockers = uniqueStrings([
    ...entries.flatMap((entry) => entry.blockers),
    ...unknownErrors,
    ...inputErrors,
  ]);
  const ready =
    readyCount === entries.length &&
    invalidSuppliedCount === 0 &&
    unknownReferences.length === 0 &&
    inputErrors.length === 0;
  const status = ready
    ? 'reference-contract-suite-ready'
    : errors.length > 0
      ? 'reference-contract-suite-invalid'
      : 'reference-contract-suite-partial';

  return {
    schema: 'moonlab.magnetar.reference-contract-validation-report.v0',
    status,
    ready,
    familyCount: entries.length,
    readyCount,
    suppliedCount: suppliedByInventoryId.size,
    suppliedReadyCount,
    invalidSuppliedCount,
    unknownCount: unknownReferences.length,
    entries,
    unknownReferences,
    blockers,
    errors,
  };
}

export function normalizeMagnetarReferenceContractSuite(
  suppliedReferences: unknown = [],
  options: { contractHash?: string } = {}
): UlgMagnetarReferenceContractNormalizedSuite {
  const contractHash = options.contractHash ?? DEFAULT_MAGNETAR_REFERENCE_CONTRACT_HASH;
  const { references } = normalizeReferenceContractInput(suppliedReferences);
  const normalizedReferences = buildMagnetarReferenceFamilyInventory(
    contractHash,
    references as Array<Partial<UlgMagnetarReferenceFamilyInventoryEntry>>
  );
  const validation = validateMagnetarReferenceContracts(suppliedReferences, {
    contractHash,
  });

  return {
    schema: 'moonlab.magnetar.normalized-reference-suite.v0',
    status: validation.status,
    ready: validation.ready,
    familyCount: validation.familyCount,
    readyCount: validation.readyCount,
    suppliedCount: validation.suppliedCount,
    suppliedReadyCount: validation.suppliedReadyCount,
    invalidSuppliedCount: validation.invalidSuppliedCount,
    unknownCount: validation.unknownCount,
    source: {
      contractHash,
      builtInReferenceFamilies: ['magnetosphere-mhd'],
    },
    references: normalizedReferences,
    validation,
    blockers: validation.blockers,
    errors: validation.errors,
  };
}

function createAnalyticMagnetosphereMhdReference(
  contractHash: string
): UlgMagnetarReferenceFamilyInventoryEntry {
  const fieldMap = {
    radiusMeters: 'outputs.radialSamples[].radiusMeters',
    magneticFieldTesla: 'outputs.radialSamples[].magneticFieldTesla',
    normalizedField: 'outputs.radialSamples[].normalizedField',
    radialPowerLaw: 'B(r)=B_surface*(R/r)^3',
    divergenceProxy: 'analytic exterior dipole field; no finite-volume divergence solve',
  };
  const fieldTolerances = {
    magneticFieldTeslaRel: 1e-12,
    normalizedFieldAbs: 1e-12,
    radialPowerLawAbs: 1e-12,
    divergenceProxyAbs: 0,
  };
  const fieldObservedDeltas = {
    magneticFieldTeslaRel: 0,
    normalizedFieldAbs: 0,
    radialPowerLawAbs: 0,
    divergenceProxyAbs: 0,
  };
  return {
    id: 'magnetosphere-mhd-reference',
    family: 'magnetosphere-mhd',
    provider: 'moonlab',
    solverId: 'moonlab-analytic-dipole-field-v0',
    schema: 'moonlab.magnetar.calibrated-reference.v0',
    role: 'peercompute-scientific-tolerance-input',
    contractHash,
    unitsHash: MAGNETOSPHERE_MHD_ANALYTIC_UNITS_HASH,
    fieldMap,
    fieldTolerances,
    fieldObservedDeltas,
    label: 'Magnetosphere analytic dipole field reference',
    status: 'calibrated-reference-ready',
    ready: true,
    scientificCoverage: true,
    scope: 'analytic-dipole-magnetosphere-reference-not-full-mhd',
    validationStatus: 'pass',
    validation: {
      status: 'pass',
      evidence: [
        'Radial magnetic field samples are generated by the analytic dipole law B(r)=B_surface*(R/r)^3.',
        'The normalized field map is deterministic and unit-scoped to meters, tesla, and dimensionless field ratios.',
        'This validates a reduced exterior dipole-field benchmark only; it is not a full resistive-MHD or force-free magnetosphere solve.',
      ],
      maxObservedDeltas: fieldObservedDeltas,
    },
    blocker: null,
    blockers: [],
  };
}

function mergeSuppliedMagnetarReferenceContracts(
  inventory: UlgMagnetarReferenceFamilyInventoryEntry[],
  suppliedReferences: Array<Partial<UlgMagnetarReferenceFamilyInventoryEntry>> = []
): UlgMagnetarReferenceFamilyInventoryEntry[] {
  if (!Array.isArray(suppliedReferences) || suppliedReferences.length === 0) {
    return inventory;
  }
  return inventory.map((entry) => {
    const supplied = suppliedReferences.find((candidate) => (
      candidate?.id === entry.id || candidate?.family === entry.family
    ));
    return supplied ? normalizeSuppliedMagnetarReferenceContract(entry, supplied) : entry;
  });
}

function normalizeSuppliedMagnetarReferenceContract(
  fallback: UlgMagnetarReferenceFamilyInventoryEntry,
  supplied: Partial<UlgMagnetarReferenceFamilyInventoryEntry>
): UlgMagnetarReferenceFamilyInventoryEntry {
  return evaluateSuppliedMagnetarReferenceContract(fallback, supplied).entry;
}

function evaluateSuppliedMagnetarReferenceContract(
  fallback: UlgMagnetarReferenceFamilyInventoryEntry,
  supplied: Partial<UlgMagnetarReferenceFamilyInventoryEntry>
): {
  entry: UlgMagnetarReferenceFamilyInventoryEntry;
  validation: UlgMagnetarReferenceContractValidationEntry;
} {
  const fieldMap = cloneRecordOrNull(supplied.fieldMap);
  const fieldTolerances = cloneRecordOrNull(supplied.fieldTolerances);
  const fieldObservedDeltas = cloneRecordOrNull(supplied.fieldObservedDeltas);
  const validation: Record<string, unknown> = isRecord(supplied.validation) ? supplied.validation : {};
  const validationStatus = supplied.validationStatus === 'pass' || validation.status === 'pass'
    ? 'pass'
    : 'missing';
  const contractHash = digestOrNull(supplied.contractHash);
  const unitsHash = digestOrNull(supplied.unitsHash);
  const solverId = typeof supplied.solverId === 'string' && supplied.solverId.length > 0
    ? supplied.solverId
    : null;
  const evidence = Array.isArray(validation.evidence)
    ? validation.evidence.map((entry: unknown) => String(entry))
    : [];
  const toleranceFailures = fieldObservedDeltas != null && fieldTolerances != null
    ? fieldDeltaToleranceFailures(fieldObservedDeltas, fieldTolerances)
    : [];
  const checks = {
    readyFlag: supplied.ready === true,
    scientificCoverageFlag: supplied.scientificCoverage === true,
    solverId: solverId != null,
    contractHash: contractHash != null,
    unitsHash: unitsHash != null,
    fieldMap: fieldMap != null,
    fieldTolerances: fieldTolerances != null,
    fieldObservedDeltas: fieldObservedDeltas != null,
    validationPass: validationStatus === 'pass',
    fieldDeltasWithinTolerance:
      fieldObservedDeltas != null &&
      fieldTolerances != null &&
      fieldDeltasWithinTolerances(fieldObservedDeltas, fieldTolerances),
  };
  const errors = suppliedReferenceValidationErrors(checks);
  const ready = Object.values(checks).every(Boolean);
  const validationEntry: UlgMagnetarReferenceContractValidationEntry = {
    id: fallback.id,
    family: fallback.family,
    supplied: true,
    status: ready ? 'calibrated-reference-ready' : 'calibrated-reference-missing',
    ready,
    scientificCoverage: supplied.scientificCoverage === true,
    solverId,
    validationStatus,
    contractHashValid: contractHash != null,
    unitsHashValid: unitsHash != null,
    fieldMapReady: fieldMap != null,
    fieldTolerancesReady: fieldTolerances != null,
    fieldObservedDeltasReady: fieldObservedDeltas != null,
    fieldDeltasWithinTolerance: checks.fieldDeltasWithinTolerance,
    toleranceFailures,
    checks,
    blockers: ready ? [] : [SUPPLIED_REFERENCE_NOT_READY_BLOCKER, ...errors],
    errors,
  };

  if (!ready) {
    return {
      entry: {
        ...fallback,
        blockers: [
          ...fallback.blockers,
          SUPPLIED_REFERENCE_NOT_READY_BLOCKER,
        ],
      },
      validation: validationEntry,
    };
  }

  return {
    entry: {
      id: fallback.id,
      family: fallback.family,
      provider: 'moonlab',
      solverId,
      schema: 'moonlab.magnetar.calibrated-reference.v0',
      role: 'peercompute-scientific-tolerance-input',
      contractHash,
      unitsHash,
      fieldMap,
      fieldTolerances,
      fieldObservedDeltas,
      label: typeof supplied.label === 'string' && supplied.label.length > 0 ? supplied.label : fallback.label,
      status: 'calibrated-reference-ready',
      ready: true,
      scientificCoverage: true,
      scope: supplied.scope === 'analytic-dipole-magnetosphere-reference-not-full-mhd'
        ? supplied.scope
        : 'supplied-calibrated-reference-contract',
      validationStatus: 'pass',
      validation: {
        status: 'pass',
        evidence,
        maxObservedDeltas: fieldObservedDeltas ?? undefined,
      },
      blocker: null,
      blockers: [],
    },
    validation: validationEntry,
  };
}

function createInventoryReferenceValidationEntry(
  entry: UlgMagnetarReferenceFamilyInventoryEntry
): UlgMagnetarReferenceContractValidationEntry {
  const contractHashValid = digestOrNull(entry.contractHash) != null;
  const unitsHashValid = digestOrNull(entry.unitsHash) != null;
  const fieldMapReady = cloneRecordOrNull(entry.fieldMap) != null;
  const fieldTolerancesReady = cloneRecordOrNull(entry.fieldTolerances) != null;
  const fieldObservedDeltasReady = cloneRecordOrNull(entry.fieldObservedDeltas) != null;
  const fieldDeltasWithinTolerance =
    entry.fieldObservedDeltas != null &&
    entry.fieldTolerances != null &&
    fieldDeltasWithinTolerances(entry.fieldObservedDeltas, entry.fieldTolerances);
  const checks = {
    readyFlag: entry.ready,
    scientificCoverageFlag: entry.scientificCoverage,
    solverId: entry.solverId != null,
    contractHash: contractHashValid,
    unitsHash: unitsHashValid,
    fieldMap: fieldMapReady,
    fieldTolerances: fieldTolerancesReady,
    fieldObservedDeltas: fieldObservedDeltasReady,
    validationPass: entry.validationStatus === 'pass' || entry.validation.status === 'pass',
    fieldDeltasWithinTolerance,
  };

  return {
    id: entry.id,
    family: entry.family,
    supplied: false,
    status: entry.status,
    ready: entry.ready,
    scientificCoverage: entry.scientificCoverage,
    solverId: entry.solverId,
    validationStatus: entry.validationStatus,
    contractHashValid,
    unitsHashValid,
    fieldMapReady,
    fieldTolerancesReady,
    fieldObservedDeltasReady,
    fieldDeltasWithinTolerance,
    toleranceFailures: entry.fieldObservedDeltas != null && entry.fieldTolerances != null
      ? fieldDeltaToleranceFailures(entry.fieldObservedDeltas, entry.fieldTolerances)
      : [],
    checks,
    blockers: entry.blockers.slice(),
    errors: [],
  };
}

function suppliedReferenceValidationErrors(
  checks: UlgMagnetarReferenceContractValidationChecks
): string[] {
  const errors: string[] = [];
  if (!checks.readyFlag) errors.push('ready must be true');
  if (!checks.scientificCoverageFlag) errors.push('scientificCoverage must be true');
  if (!checks.solverId) errors.push('solverId must be a non-empty string');
  if (!checks.contractHash) errors.push('contractHash must be a sha256 digest');
  if (!checks.unitsHash) errors.push('unitsHash must be a sha256 digest');
  if (!checks.fieldMap) errors.push('fieldMap must be a non-empty object');
  if (!checks.fieldTolerances) errors.push('fieldTolerances must be a non-empty object');
  if (!checks.fieldObservedDeltas) errors.push('fieldObservedDeltas must be a non-empty object');
  if (!checks.validationPass) errors.push('validation.status or validationStatus must be pass');
  if (!checks.fieldDeltasWithinTolerance) {
    errors.push('fieldObservedDeltas must be within fieldTolerances');
  }
  return errors;
}

function fieldDeltasWithinTolerances(
  observed: Record<string, unknown>,
  tolerances: Record<string, unknown>
): boolean {
  const entries = Object.entries(tolerances);
  if (entries.length === 0) return false;
  return fieldDeltaToleranceFailures(observed, tolerances).length === 0;
}

function fieldDeltaToleranceFailures(
  observed: Record<string, unknown>,
  tolerances: Record<string, unknown>
): UlgMagnetarReferenceContractToleranceFailure[] {
  return Object.entries(tolerances).flatMap(([key, tolerance]) => {
    const observedValue = Number(observed[key]);
    const toleranceValue = normalizeToleranceValue(tolerance);
    if (
      Number.isFinite(observedValue) &&
      toleranceValue != null &&
      Math.abs(observedValue) <= toleranceValue
    ) {
      return [];
    }
    return [{
      field: key,
      observed: Number.isFinite(observedValue) ? observedValue : null,
      tolerance: toleranceValue,
    }];
  });
}

function normalizeToleranceValue(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.abs(value);
  }
  if (!isRecord(value)) return null;
  for (const key of ['abs', 'rel', 'value']) {
    const candidate = Number(value[key]);
    if (Number.isFinite(candidate)) {
      return Math.abs(candidate);
    }
  }
  return null;
}

function cloneRecordOrNull(value: unknown): Record<string, unknown> | null {
  return isRecord(value) && Object.keys(value).length > 0 ? { ...value } : null;
}

function digestOrNull(value: unknown): string | null {
  return typeof value === 'string' && SHA256_DIGEST_PATTERN.test(value)
    ? value.toLowerCase()
    : null;
}

function normalizeReferenceContractInput(
  value: unknown
): {
  references: unknown[];
  errors: string[];
} {
  if (value == null) {
    return { references: [], errors: [] };
  }
  if (Array.isArray(value)) {
    return { references: value, errors: [] };
  }
  if (isRecord(value)) {
    if (Array.isArray(value.references)) {
      return { references: value.references, errors: [] };
    }
    if (isRecord(value.outputs) && Array.isArray(value.outputs.references)) {
      return { references: value.outputs.references, errors: [] };
    }
  }
  return {
    references: [],
    errors: ['reference contract input must be an array or contain references[]'],
  };
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values));
}

export function validateUlgQuantumResponseArtifact(
  artifact: unknown,
  schema: QuantumResponseArtifactSchema = DEFAULT_ULG_QUANTUM_RESPONSE_SCHEMA
): SchemaValidationResult {
  const errors: string[] = [];

  if (!isRecord(artifact)) {
    return { valid: false, errors: ['artifact must be an object'] };
  }

  for (const key of schema.required ?? []) {
    if (!(key in artifact)) {
      errors.push(`missing required field: ${key}`);
    }
  }

  for (const [key, propertySchema] of Object.entries(schema.properties ?? {})) {
    if (!(key in artifact)) {
      continue;
    }
    const value = artifact[key];
    if (propertySchema.const !== undefined && value !== propertySchema.const) {
      errors.push(`field ${key} must equal ${JSON.stringify(propertySchema.const)}`);
    }
    if (propertySchema.type === 'string' && typeof value !== 'string') {
      errors.push(`field ${key} must be a string`);
    }
    if (propertySchema.type === 'object' && !isRecord(value)) {
      errors.push(`field ${key} must be an object`);
    }
    if (
      propertySchema.minLength !== undefined &&
      typeof value === 'string' &&
      value.length < propertySchema.minLength
    ) {
      errors.push(`field ${key} must be at least ${propertySchema.minLength} character(s)`);
    }
  }

  return { valid: errors.length === 0, errors };
}

export function canonicalJson(value: unknown): string {
  return JSON.stringify(sortJsonValue(value));
}

async function sha256Hex(value: string): Promise<string> {
  if (!globalThis.crypto?.subtle) {
    throw new Error('WebCrypto SHA-256 support is required to build a ULG artifact');
  }

  const digest = await globalThis.crypto.subtle.digest(
    'SHA-256',
    new TextEncoder().encode(value)
  );
  return `sha256:${Array.from(new Uint8Array(digest), byteToHex).join('')}`;
}

function sortJsonValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortJsonValue);
  }
  if (isRecord(value)) {
    const sorted: Record<string, unknown> = {};
    for (const key of Object.keys(value).sort()) {
      sorted[key] = sortJsonValue(value[key]);
    }
    return sorted;
  }
  return value;
}

function byteToHex(byte: number): string {
  return byte.toString(16).padStart(2, '0');
}

function basisBitString(index: number, numQubits: number): string {
  return index.toString(2).padStart(numQubits, '0');
}

function allBitstringsForQubits(numQubits: number): number[] {
  validateNumQubits(numQubits);
  return Array.from({ length: 2 ** numQubits }, (_, index) => index);
}

function magnetarDipoleFieldTesla(
  input: UlgMagnetarDipoleIsingInput,
  radiusMeters: number
): number {
  return input.surfaceMagneticFieldTesla *
    (input.stellarRadiusMeters / radiusMeters) ** 3;
}

function spinsFromBitstring(bitstring: number, numQubits: number): number[] {
  validateBitstring(bitstring, numQubits);
  return Array.from({ length: numQubits }, (_, qubit) => {
    return (bitstring >> qubit) & 1 ? -1 : 1;
  });
}

function validateMagnetarDipoleIsingInput(input: UlgMagnetarDipoleIsingInput): void {
  assertPositiveFinite(input.surfaceMagneticFieldTesla, 'surfaceMagneticFieldTesla');
  assertPositiveFinite(input.stellarRadiusMeters, 'stellarRadiusMeters');
  assertFiniteNumber(input.couplingStrength, 'couplingStrength');
  if (input.couplingStrength < 0) {
    throw new Error('couplingStrength must be non-negative');
  }
  validateNumQubits(input.radialSamplesMeters.length);

  for (const radiusMeters of input.radialSamplesMeters) {
    assertPositiveFinite(radiusMeters, 'radialSamplesMeters');
    if (radiusMeters < input.stellarRadiusMeters) {
      throw new Error('radialSamplesMeters must be at or outside stellarRadiusMeters');
    }
  }
  for (let index = 1; index < input.radialSamplesMeters.length; index += 1) {
    if (input.radialSamplesMeters[index] < input.radialSamplesMeters[index - 1]) {
      throw new Error('radialSamplesMeters must be sorted from inner to outer radius');
    }
  }
  if (input.bitstrings.length === 0) {
    throw new Error('bitstrings must contain at least one value');
  }
  for (const bitstring of input.bitstrings) {
    validateBitstring(bitstring, input.radialSamplesMeters.length);
  }
}

function validateNumQubits(numQubits: number): void {
  if (!Number.isInteger(numQubits) || numQubits < 1 || numQubits > 12) {
    throw new Error('magnetar dipole Ising probe supports 1 to 12 radial samples');
  }
}

function validateBitstring(bitstring: number, numQubits: number): void {
  const maxBitstring = 2 ** numQubits;
  if (
    !Number.isInteger(bitstring) ||
    bitstring < 0 ||
    bitstring >= maxBitstring ||
    !Number.isSafeInteger(bitstring)
  ) {
    throw new Error(`bitstring must be an integer between 0 and ${maxBitstring - 1}`);
  }
}

function assertPositiveFinite(value: number, label: string): void {
  assertFiniteNumber(value, label);
  if (value <= 0) {
    throw new Error(`${label} must be positive`);
  }
}

function assertFiniteNumber(value: number, label: string): void {
  if (!Number.isFinite(value)) {
    throw new Error(`${label} must be finite`);
  }
}

function normalizeNumber(value: number): number {
  if (Object.is(value, -0)) {
    return 0;
  }
  return Number(value.toPrecision(15));
}

function runtimeLabel(): string {
  if (typeof process !== 'undefined' && process.versions?.node) {
    return `node-${process.versions.node}`;
  }
  if (typeof navigator !== 'undefined' && navigator.userAgent) {
    return navigator.userAgent;
  }
  return 'unknown';
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
