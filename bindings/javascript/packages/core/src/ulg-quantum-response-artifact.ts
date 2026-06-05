import { QuantumState } from './quantum-state';

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
