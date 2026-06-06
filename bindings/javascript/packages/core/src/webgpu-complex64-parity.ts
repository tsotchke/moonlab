export const MOONLAB_WEBGPU_COMPLEX64_PARITY_SCOPE_SCHEMA =
  'moonlab.webgpu.complex64-parity-scope.v0';
export const MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF = 1e-5;

export const MOONLAB_WEBGPU_COMPLEX64_NATIVE_COVERAGE_REQUIRED = [
  'hadamard',
  'pauli_x',
  'pauli_z',
  'cnot',
  'compute_probabilities',
] as const;

export const MOONLAB_WEBGPU_COMPLEX64_NATIVE_COVERAGE_EXCLUDED = [
  'phase',
] as const;

type NativeCoverageRequired =
  typeof MOONLAB_WEBGPU_COMPLEX64_NATIVE_COVERAGE_REQUIRED[number];
type NativeCoverageExcluded =
  typeof MOONLAB_WEBGPU_COMPLEX64_NATIVE_COVERAGE_EXCLUDED[number];
type FixtureOperationName = NativeCoverageRequired | NativeCoverageExcluded;

interface ComplexValue {
  real: number;
  imag: number;
}

interface FixtureOperation {
  operation: FixtureOperationName;
  qubit?: number;
  control?: number;
  target?: number;
}

interface ReducedFixtureDefinition {
  fixtureId: string;
  qubitCount: number;
  operations: FixtureOperation[];
}

export interface MoonlabWebGpuComplex64BackendDetection {
  available: boolean;
  runtime: string;
  reason?: string;
  adapterInfo?: Record<string, unknown>;
}

export interface MoonlabWebGpuComplex64NativeCoverageEntry {
  operation: NativeCoverageRequired;
  required: true;
  covered: boolean;
  fallbackAllowed: false;
  status:
    | 'covered-by-browser-webgpu'
    | 'not-run-backend-unavailable'
    | 'not-run-runtime-backend-not-wired';
}

export interface MoonlabWebGpuComplex64FallbackCoverageEntry {
  operation: NativeCoverageExcluded;
  excludedFromNativeCoverage: true;
  fallbackAllowed: true;
  status: 'cpu-fallback-excluded-from-native-parity';
  reason: string;
}

export interface MoonlabWebGpuComplex64ReducedFixtureResult {
  fixtureId: string;
  qubitCount: number;
  operations: FixtureOperation[];
  referenceRepresentation: 'wasm-float64-interleaved';
  complex64Representation: 'complex64-interleaved-f32';
  referenceProbabilities: number[];
  complex64Probabilities: number[];
  maxProbabilityAbsDiff: number;
  passed: boolean;
}

export interface MoonlabWebGpuComplex64ParityValidation {
  valid: boolean;
  errors: string[];
  checks: Array<{
    name: string;
    passed: boolean;
    details?: Record<string, unknown>;
  }>;
}

export interface MoonlabWebGpuComplex64ParityScopeArtifact {
  schema: typeof MOONLAB_WEBGPU_COMPLEX64_PARITY_SCOPE_SCHEMA;
  artifactKind: 'browser-webgpu-complex64-parity-scope';
  generatedAt: string;
  status:
    | 'scope-ready-backend-unavailable'
    | 'scope-ready-backend-detected'
    | 'blocked-webgpu-backend-required';
  contractReady: boolean;
  backend: 'webgpu';
  backendAvailable: boolean;
  requireBackend: boolean;
  backendDetection: MoonlabWebGpuComplex64BackendDetection;
  amplitudeRepresentation: 'complex64-interleaved-f32';
  referenceRepresentation: 'wasm-float64-interleaved';
  maxProbabilityAbsDiff: number;
  maxAmplitudeAbsDiff: number;
  reducedFixtureOnly: true;
  fullFidelityMagnetarSimulation: false;
  fullPhysicsValidation: false;
  fidelityRuntimeScope: {
    schema: 'ulg.magnetar.fidelity-runtime-scope.v0';
    fidelityTier: 'reduced-calibrated-runtime-fixture';
    runtimeScope: 'browser-webgpu-complex64-reduced-fixture-parity';
    readinessClaim: 'integration-tolerance-gate-only';
    reducedCalibratedRuntimeFixture: true;
    hostRuntimeSmokeFixture: true;
    fullFidelityMagnetarSimulation: false;
    fullPhysicsValidation: false;
  };
  nativeCoverageRequired: NativeCoverageRequired[];
  nativeCoverageExcluded: NativeCoverageExcluded[];
  coverage: {
    nativeWebGpu: MoonlabWebGpuComplex64NativeCoverageEntry[];
    cpuFallbackExcluded: MoonlabWebGpuComplex64FallbackCoverageEntry[];
  };
  reducedFixtures: MoonlabWebGpuComplex64ReducedFixtureResult[];
  complex64Preflight: {
    mode: 'cpu-complex64-rounding-preflight';
    executed: true;
    passed: boolean;
    maxProbabilityAbsDiff: number;
    tolerance: number;
  };
  webgpuParity: {
    executed: boolean;
    passed: boolean;
    maxProbabilityAbsDiff: number | null;
    tolerance: number;
    reason: string;
  };
  blockers: string[];
  contractValidation: MoonlabWebGpuComplex64ParityValidation;
}

export interface BuildMoonlabWebGpuComplex64ParityScopeOptions {
  generatedAt?: string;
  backendDetection?: MoonlabWebGpuComplex64BackendDetection;
  backendAvailable?: boolean;
  requireBackend?: boolean;
  coveredNativeOperations?: NativeCoverageRequired[];
  webgpuParity?: {
    executed: boolean;
    passed: boolean;
    maxProbabilityAbsDiff: number;
    reason?: string;
  };
}

export function buildMoonlabWebGpuComplex64ParityScope(
  options: BuildMoonlabWebGpuComplex64ParityScopeOptions = {}
): MoonlabWebGpuComplex64ParityScopeArtifact {
  const backendAvailable =
    options.backendAvailable ?? options.backendDetection?.available ?? false;
  const requireBackend = options.requireBackend ?? false;
  const backendDetection = options.backendDetection ?? {
    available: backendAvailable,
    runtime: backendAvailable ? 'browser-webgpu' : 'node-or-browser-without-webgpu',
    reason: backendAvailable
      ? undefined
      : 'navigator.gpu adapter unavailable; no browser WebGPU backend was executed',
  };
  const reducedFixtures = buildReducedFixtureResults();
  const preflightMaxDiff = Math.max(
    ...reducedFixtures.map((fixture) => fixture.maxProbabilityAbsDiff)
  );
  const coveredOperations = new Set(options.coveredNativeOperations ?? []);
  const nativeCoverage = MOONLAB_WEBGPU_COMPLEX64_NATIVE_COVERAGE_REQUIRED.map(
    (operation): MoonlabWebGpuComplex64NativeCoverageEntry => {
      const covered = coveredOperations.has(operation);
      return {
        operation,
        required: true,
        covered,
        fallbackAllowed: false,
        status: covered
          ? 'covered-by-browser-webgpu'
          : backendAvailable
            ? 'not-run-runtime-backend-not-wired'
            : 'not-run-backend-unavailable',
      };
    }
  );
  const webgpuParity = options.webgpuParity ?? {
    executed: false,
    passed: false,
    maxProbabilityAbsDiff: null,
    reason: backendAvailable
      ? 'browser WebGPU adapter was detected, but no native MoonLab WebGPU runtime is wired on this branch'
      : 'browser WebGPU adapter unavailable; emitted contract/preflight artifact only',
  };
  const blockers = buildBlockers({
    backendAvailable,
    requireBackend,
    nativeCoverage,
    webgpuParity,
  });
  const artifactWithoutValidation = {
    schema: MOONLAB_WEBGPU_COMPLEX64_PARITY_SCOPE_SCHEMA,
    artifactKind: 'browser-webgpu-complex64-parity-scope',
    generatedAt: options.generatedAt ?? new Date().toISOString(),
    status: requireBackend && !backendAvailable
      ? 'blocked-webgpu-backend-required'
      : backendAvailable
        ? 'scope-ready-backend-detected'
        : 'scope-ready-backend-unavailable',
    contractReady: true,
    backend: 'webgpu',
    backendAvailable,
    requireBackend,
    backendDetection,
    amplitudeRepresentation: 'complex64-interleaved-f32',
    referenceRepresentation: 'wasm-float64-interleaved',
    maxProbabilityAbsDiff: MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF,
    maxAmplitudeAbsDiff: MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF,
    reducedFixtureOnly: true,
    fullFidelityMagnetarSimulation: false,
    fullPhysicsValidation: false,
    fidelityRuntimeScope: {
      schema: 'ulg.magnetar.fidelity-runtime-scope.v0',
      fidelityTier: 'reduced-calibrated-runtime-fixture',
      runtimeScope: 'browser-webgpu-complex64-reduced-fixture-parity',
      readinessClaim: 'integration-tolerance-gate-only',
      reducedCalibratedRuntimeFixture: true,
      hostRuntimeSmokeFixture: true,
      fullFidelityMagnetarSimulation: false,
      fullPhysicsValidation: false,
    },
    nativeCoverageRequired: [...MOONLAB_WEBGPU_COMPLEX64_NATIVE_COVERAGE_REQUIRED],
    nativeCoverageExcluded: [...MOONLAB_WEBGPU_COMPLEX64_NATIVE_COVERAGE_EXCLUDED],
    coverage: {
      nativeWebGpu: nativeCoverage,
      cpuFallbackExcluded: MOONLAB_WEBGPU_COMPLEX64_NATIVE_COVERAGE_EXCLUDED.map(
        (operation): MoonlabWebGpuComplex64FallbackCoverageEntry => ({
          operation,
          excludedFromNativeCoverage: true,
          fallbackAllowed: true,
          status: 'cpu-fallback-excluded-from-native-parity',
          reason:
            'phase is excluded from the reduced native WebGPU parity requirement because the old backend routed it through CPU fallback',
        })
      ),
    },
    reducedFixtures,
    complex64Preflight: {
      mode: 'cpu-complex64-rounding-preflight',
      executed: true,
      passed: preflightMaxDiff <= MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF,
      maxProbabilityAbsDiff: preflightMaxDiff,
      tolerance: MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF,
    },
    webgpuParity: {
      executed: webgpuParity.executed,
      passed: webgpuParity.passed,
      maxProbabilityAbsDiff: webgpuParity.maxProbabilityAbsDiff,
      tolerance: MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF,
      reason: webgpuParity.reason ?? '',
    },
    blockers,
  } satisfies Omit<MoonlabWebGpuComplex64ParityScopeArtifact, 'contractValidation'>;

  return {
    ...artifactWithoutValidation,
    contractValidation: validateMoonlabWebGpuComplex64ParityScope(artifactWithoutValidation),
  };
}

export function validateMoonlabWebGpuComplex64ParityScope(
  value: unknown
): MoonlabWebGpuComplex64ParityValidation {
  const artifact = value as Partial<MoonlabWebGpuComplex64ParityScopeArtifact>;
  const checks: MoonlabWebGpuComplex64ParityValidation['checks'] = [];
  const errors: string[] = [];

  addCheck(checks, errors, 'schema', artifact.schema === MOONLAB_WEBGPU_COMPLEX64_PARITY_SCOPE_SCHEMA);
  addCheck(checks, errors, 'backend', artifact.backend === 'webgpu');
  addCheck(
    checks,
    errors,
    'amplitude-representation',
    artifact.amplitudeRepresentation === 'complex64-interleaved-f32'
  );
  addCheck(
    checks,
    errors,
    'reference-representation',
    artifact.referenceRepresentation === 'wasm-float64-interleaved'
  );
  addCheck(
    checks,
    errors,
    'probability-tolerance',
    typeof artifact.maxProbabilityAbsDiff === 'number'
      && artifact.maxProbabilityAbsDiff <= MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF
  );
  addCheck(checks, errors, 'reduced-fixture-only', artifact.reducedFixtureOnly === true);
  addCheck(
    checks,
    errors,
    'no-full-fidelity-magnetar-simulation',
    artifact.fullFidelityMagnetarSimulation === false
      && artifact.fidelityRuntimeScope?.fullFidelityMagnetarSimulation === false
  );
  addCheck(
    checks,
    errors,
    'no-full-physics-validation',
    artifact.fullPhysicsValidation === false
      && artifact.fidelityRuntimeScope?.fullPhysicsValidation === false
  );
  addCheck(
    checks,
    errors,
    'native-coverage-required',
    arraysEqual(
      artifact.nativeCoverageRequired,
      MOONLAB_WEBGPU_COMPLEX64_NATIVE_COVERAGE_REQUIRED
    )
  );
  addCheck(
    checks,
    errors,
    'native-coverage-excluded',
    arraysEqual(
      artifact.nativeCoverageExcluded,
      MOONLAB_WEBGPU_COMPLEX64_NATIVE_COVERAGE_EXCLUDED
    )
  );
  addCheck(
    checks,
    errors,
    'phase-excluded-from-native-coverage',
    artifact.coverage?.cpuFallbackExcluded?.some((entry) => (
      entry.operation === 'phase'
      && entry.excludedFromNativeCoverage === true
      && entry.status === 'cpu-fallback-excluded-from-native-parity'
    )) === true
  );
  addCheck(
    checks,
    errors,
    'webgpu-pass-requires-execution',
    artifact.webgpuParity?.passed !== true || artifact.webgpuParity.executed === true
  );
  addCheck(
    checks,
    errors,
    'complex64-preflight-within-tolerance',
    artifact.complex64Preflight?.executed === true
      && artifact.complex64Preflight.passed === true
      && artifact.complex64Preflight.maxProbabilityAbsDiff
        <= MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF
  );

  return {
    valid: errors.length === 0,
    errors,
    checks,
  };
}

function buildBlockers({
  backendAvailable,
  requireBackend,
  nativeCoverage,
  webgpuParity,
}: {
  backendAvailable: boolean;
  requireBackend: boolean;
  nativeCoverage: MoonlabWebGpuComplex64NativeCoverageEntry[];
  webgpuParity: {
    executed: boolean;
    passed: boolean;
  };
}): string[] {
  const blockers: string[] = [];
  if (!backendAvailable) {
    blockers.push('browser-webgpu-adapter-unavailable');
  }
  if (requireBackend && !backendAvailable) {
    blockers.push('required-browser-webgpu-backend-missing');
  }
  if (nativeCoverage.some((entry) => !entry.covered)) {
    blockers.push('native-webgpu-operation-coverage-not-yet-recorded');
  }
  if (!webgpuParity.executed) {
    blockers.push('browser-webgpu-kernel-parity-not-executed');
  } else if (!webgpuParity.passed) {
    blockers.push('browser-webgpu-kernel-parity-failed');
  }
  return blockers;
}

function buildReducedFixtureResults(): MoonlabWebGpuComplex64ReducedFixtureResult[] {
  return reducedFixtureDefinitions().map((fixture) => {
    const referenceState = runFixture(fixture, false);
    const complex64State = runFixture(fixture, true);
    const referenceProbabilities = probabilities(referenceState, false);
    const complex64Probabilities = probabilities(complex64State, true);
    const maxProbabilityAbsDiff = maxAbsDiff(referenceProbabilities, complex64Probabilities);

    return {
      fixtureId: fixture.fixtureId,
      qubitCount: fixture.qubitCount,
      operations: fixture.operations,
      referenceRepresentation: 'wasm-float64-interleaved',
      complex64Representation: 'complex64-interleaved-f32',
      referenceProbabilities,
      complex64Probabilities,
      maxProbabilityAbsDiff,
      passed: maxProbabilityAbsDiff <= MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF,
    };
  });
}

function reducedFixtureDefinitions(): ReducedFixtureDefinition[] {
  return [
    {
      fixtureId: 'bell-2q-hadamard-cnot-probabilities',
      qubitCount: 2,
      operations: [
        { operation: 'hadamard', qubit: 0 },
        { operation: 'cnot', control: 0, target: 1 },
        { operation: 'compute_probabilities' },
      ],
    },
    {
      fixtureId: 'pauli-xz-2q-probabilities',
      qubitCount: 2,
      operations: [
        { operation: 'pauli_x', qubit: 1 },
        { operation: 'pauli_z', qubit: 1 },
        { operation: 'compute_probabilities' },
      ],
    },
    {
      fixtureId: 'entangled-3q-h-x-z-cnot-probabilities',
      qubitCount: 3,
      operations: [
        { operation: 'hadamard', qubit: 0 },
        { operation: 'pauli_x', qubit: 2 },
        { operation: 'pauli_z', qubit: 2 },
        { operation: 'cnot', control: 0, target: 1 },
        { operation: 'compute_probabilities' },
      ],
    },
  ];
}

function runFixture(fixture: ReducedFixtureDefinition, complex64: boolean): ComplexValue[] {
  let state = initialState(fixture.qubitCount);
  for (const op of fixture.operations) {
    if (op.operation === 'compute_probabilities') {
      continue;
    }
    state = applyOperation(state, fixture.qubitCount, op, complex64);
    if (complex64) {
      state = state.map(froundComplex);
    }
  }
  return state;
}

function initialState(qubitCount: number): ComplexValue[] {
  const state = Array.from({ length: 2 ** qubitCount }, () => ({ real: 0, imag: 0 }));
  state[0] = { real: 1, imag: 0 };
  return state;
}

function applyOperation(
  state: ComplexValue[],
  qubitCount: number,
  op: FixtureOperation,
  complex64: boolean
): ComplexValue[] {
  if (op.operation === 'hadamard') {
    return applySingleQubitMatrix(state, qubitCount, requiredQubit(op), [
      { real: 1 / Math.sqrt(2), imag: 0 },
      { real: 1 / Math.sqrt(2), imag: 0 },
      { real: 1 / Math.sqrt(2), imag: 0 },
      { real: -1 / Math.sqrt(2), imag: 0 },
    ], complex64);
  }
  if (op.operation === 'pauli_x') {
    return applySingleQubitMatrix(state, qubitCount, requiredQubit(op), [
      { real: 0, imag: 0 },
      { real: 1, imag: 0 },
      { real: 1, imag: 0 },
      { real: 0, imag: 0 },
    ], complex64);
  }
  if (op.operation === 'pauli_z') {
    return applySingleQubitMatrix(state, qubitCount, requiredQubit(op), [
      { real: 1, imag: 0 },
      { real: 0, imag: 0 },
      { real: 0, imag: 0 },
      { real: -1, imag: 0 },
    ], complex64);
  }
  if (op.operation === 'cnot') {
    return applyCnot(state, requiredControl(op), requiredTarget(op));
  }
  throw new Error(`Unsupported reduced fixture operation: ${op.operation}`);
}

function applySingleQubitMatrix(
  state: ComplexValue[],
  qubitCount: number,
  qubit: number,
  matrix: [ComplexValue, ComplexValue, ComplexValue, ComplexValue],
  complex64: boolean
): ComplexValue[] {
  assertQubit(qubit, qubitCount);
  const output = state.map((amplitude) => ({ ...amplitude }));
  const stride = 2 ** qubit;
  const period = stride * 2;
  for (let base = 0; base < state.length; base += period) {
    for (let offset = 0; offset < stride; offset += 1) {
      const zeroIndex = base + offset;
      const oneIndex = zeroIndex + stride;
      const zero = state[zeroIndex];
      const one = state[oneIndex];
      output[zeroIndex] = addComplex(multiplyComplex(matrix[0], zero), multiplyComplex(matrix[1], one));
      output[oneIndex] = addComplex(multiplyComplex(matrix[2], zero), multiplyComplex(matrix[3], one));
      if (complex64) {
        output[zeroIndex] = froundComplex(output[zeroIndex]);
        output[oneIndex] = froundComplex(output[oneIndex]);
      }
    }
  }
  return output;
}

function applyCnot(state: ComplexValue[], control: number, target: number): ComplexValue[] {
  if (control === target) {
    throw new Error('cnot control and target must differ');
  }
  const output = state.map((amplitude) => ({ ...amplitude }));
  for (let index = 0; index < state.length; index += 1) {
    if (((index >> control) & 1) === 1 && ((index >> target) & 1) === 0) {
      const flipped = index | (1 << target);
      output[index] = { ...state[flipped] };
      output[flipped] = { ...state[index] };
    }
  }
  return output;
}

function probabilities(state: ComplexValue[], complex64: boolean): number[] {
  return state.map((amplitude) => {
    const probability = amplitude.real * amplitude.real + amplitude.imag * amplitude.imag;
    return complex64 ? Math.fround(probability) : probability;
  });
}

function multiplyComplex(left: ComplexValue, right: ComplexValue): ComplexValue {
  return {
    real: left.real * right.real - left.imag * right.imag,
    imag: left.real * right.imag + left.imag * right.real,
  };
}

function addComplex(left: ComplexValue, right: ComplexValue): ComplexValue {
  return {
    real: left.real + right.real,
    imag: left.imag + right.imag,
  };
}

function froundComplex(value: ComplexValue): ComplexValue {
  return {
    real: Math.fround(value.real),
    imag: Math.fround(value.imag),
  };
}

function maxAbsDiff(left: number[], right: number[]): number {
  return left.reduce((maxDiff, value, index) => (
    Math.max(maxDiff, Math.abs(value - (right[index] ?? Number.NaN)))
  ), 0);
}

function requiredQubit(op: FixtureOperation): number {
  if (typeof op.qubit !== 'number') {
    throw new Error(`${op.operation} fixture operation requires qubit`);
  }
  return op.qubit;
}

function requiredControl(op: FixtureOperation): number {
  if (typeof op.control !== 'number') {
    throw new Error(`${op.operation} fixture operation requires control`);
  }
  return op.control;
}

function requiredTarget(op: FixtureOperation): number {
  if (typeof op.target !== 'number') {
    throw new Error(`${op.operation} fixture operation requires target`);
  }
  return op.target;
}

function assertQubit(qubit: number, qubitCount: number): void {
  if (!Number.isInteger(qubit) || qubit < 0 || qubit >= qubitCount) {
    throw new Error(`qubit ${qubit} out of range for ${qubitCount} qubits`);
  }
}

function arraysEqual(left: readonly string[] | undefined, right: readonly string[]): boolean {
  return Array.isArray(left)
    && left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function addCheck(
  checks: MoonlabWebGpuComplex64ParityValidation['checks'],
  errors: string[],
  name: string,
  passed: boolean
): void {
  checks.push({ name, passed });
  if (!passed) {
    errors.push(`${name} check failed`);
  }
}
