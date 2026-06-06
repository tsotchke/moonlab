export const MOONLAB_WEBGPU_COMPLEX64_PARITY_SCOPE_SCHEMA =
  'moonlab.webgpu.complex64-parity-scope.v0';
export const MOONLAB_WEBGPU_COMPLEX64_PROBABILITY_KERNEL_PROBE_SCHEMA =
  'moonlab.webgpu.complex64-probability-kernel-probe.v0';
export const MOONLAB_WEBGPU_COMPLEX64_NATIVE_OPERATION_PROBE_SCHEMA =
  'moonlab.webgpu.complex64-native-operation-probe.v0';
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

export const MOONLAB_WEBGPU_COMPLEX64_NATIVE_OPERATION_PROBE_OPERATIONS = [
  'hadamard',
  'pauli_x',
  'pauli_z',
] as const;

type NativeCoverageRequired =
  typeof MOONLAB_WEBGPU_COMPLEX64_NATIVE_COVERAGE_REQUIRED[number];
type NativeCoverageExcluded =
  typeof MOONLAB_WEBGPU_COMPLEX64_NATIVE_COVERAGE_EXCLUDED[number];
type NativeOperationProbeOperation =
  typeof MOONLAB_WEBGPU_COMPLEX64_NATIVE_OPERATION_PROBE_OPERATIONS[number];
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

interface NativeOperationProbeFixtureDefinition {
  fixtureId: string;
  qubitCount: number;
  operation: FixtureOperation & { operation: NativeOperationProbeOperation };
  inputState: ComplexValue[];
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

export interface MoonlabBrowserWebGpuComplex64ProbabilityKernelProbeFixture {
  fixtureId: string;
  qubitCount: number;
  amplitudeCount: number;
  referenceRepresentation: 'wasm-float64-interleaved';
  complex64Representation: 'complex64-interleaved-f32';
  browserWebGpuKernel: 'compute_probabilities';
  referenceProbabilities: number[];
  browserWebGpuProbabilities: number[];
  maxProbabilityAbsDiff: number;
  passed: boolean;
}

export interface MoonlabBrowserWebGpuComplex64ProbabilityKernelProbe {
  schema: typeof MOONLAB_WEBGPU_COMPLEX64_PROBABILITY_KERNEL_PROBE_SCHEMA;
  probeKind: 'browser-webgpu-complex64-probability-kernel';
  kernel: 'compute_probabilities';
  executed: boolean;
  passed: boolean;
  coveredNativeOperations: Array<Extract<NativeCoverageRequired, 'compute_probabilities'>>;
  fixtureResults: MoonlabBrowserWebGpuComplex64ProbabilityKernelProbeFixture[];
  maxProbabilityAbsDiff: number | null;
  tolerance: number;
  reason: string;
}

export interface MoonlabBrowserWebGpuComplex64NativeOperationProbeFixture {
  fixtureId: string;
  qubitCount: number;
  amplitudeCount: number;
  operation: NativeOperationProbeOperation;
  referenceRepresentation: 'cpu-complex64-interleaved-f32';
  complex64Representation: 'complex64-interleaved-f32';
  browserWebGpuKernel: NativeOperationProbeOperation;
  inputAmplitudes: number[];
  referenceAmplitudes: number[];
  browserWebGpuAmplitudes: number[];
  maxAmplitudeAbsDiff: number;
  passed: boolean;
}

export interface MoonlabBrowserWebGpuComplex64NativeOperationProbeOperationResult {
  operation: NativeOperationProbeOperation;
  executed: boolean;
  passed: boolean;
  covered: boolean;
  fixtureResults: MoonlabBrowserWebGpuComplex64NativeOperationProbeFixture[];
  maxAmplitudeAbsDiff: number | null;
  tolerance: number;
  blocker?: string;
  reason: string;
}

export interface MoonlabBrowserWebGpuComplex64NativeOperationProbe {
  schema: typeof MOONLAB_WEBGPU_COMPLEX64_NATIVE_OPERATION_PROBE_SCHEMA;
  probeKind: 'browser-webgpu-complex64-native-operation-probe';
  executed: boolean;
  passed: boolean;
  coveredNativeOperations: NativeOperationProbeOperation[];
  operationResults: MoonlabBrowserWebGpuComplex64NativeOperationProbeOperationResult[];
  maxAmplitudeAbsDiff: number | null;
  tolerance: number;
  reason: string;
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
  browserKernelProbe: MoonlabBrowserWebGpuComplex64ProbabilityKernelProbe;
  browserNativeOperationProbe: MoonlabBrowserWebGpuComplex64NativeOperationProbe;
  blockers: string[];
  contractValidation: MoonlabWebGpuComplex64ParityValidation;
}

export interface BuildMoonlabWebGpuComplex64ParityScopeOptions {
  generatedAt?: string;
  backendDetection?: MoonlabWebGpuComplex64BackendDetection;
  backendAvailable?: boolean;
  requireBackend?: boolean;
  coveredNativeOperations?: NativeCoverageRequired[];
  browserKernelProbe?: MoonlabBrowserWebGpuComplex64ProbabilityKernelProbe;
  browserNativeOperationProbe?: MoonlabBrowserWebGpuComplex64NativeOperationProbe;
  webgpuParity?: {
    executed: boolean;
    passed: boolean;
    maxProbabilityAbsDiff: number | null;
    reason?: string;
  };
}

export interface RunMoonlabBrowserWebGpuComplex64ProbabilityKernelProbeOptions {
  gpu?: MoonlabBrowserWebGpu | null;
  runtime?: string;
  workgroupSize?: number;
}

export interface RunMoonlabBrowserWebGpuComplex64NativeOperationProbeOptions {
  gpu?: MoonlabBrowserWebGpu | null;
  runtime?: string;
  workgroupSize?: number;
}

export interface BuildMoonlabWebGpuComplex64ParityScopeWithBrowserProbeOptions
  extends Omit<
    BuildMoonlabWebGpuComplex64ParityScopeOptions,
    | 'backendDetection'
    | 'backendAvailable'
    | 'coveredNativeOperations'
    | 'browserKernelProbe'
    | 'browserNativeOperationProbe'
  > {
  gpu?: MoonlabBrowserWebGpu | null;
  runtime?: string;
  workgroupSize?: number;
}

export interface MoonlabBrowserWebGpu {
  requestAdapter(): Promise<MoonlabBrowserWebGpuAdapter | null>;
}

export interface MoonlabBrowserWebGpuAdapter {
  requestDevice(): Promise<MoonlabBrowserWebGpuDevice>;
  requestAdapterInfo?(): Promise<Record<string, unknown>>;
}

export interface MoonlabBrowserWebGpuDevice {
  createBuffer(options: {
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }): MoonlabBrowserWebGpuBuffer;
  createShaderModule(options: { code: string }): unknown;
  createComputePipeline(options: {
    layout: 'auto';
    compute: {
      module: unknown;
      entryPoint: string;
    };
  }): MoonlabBrowserWebGpuComputePipeline;
  createBindGroup(options: {
    layout: unknown;
    entries: Array<{
      binding: number;
      resource: {
        buffer: MoonlabBrowserWebGpuBuffer;
      };
    }>;
  }): unknown;
  createCommandEncoder(): MoonlabBrowserWebGpuCommandEncoder;
  queue: {
    submit(commandBuffers: unknown[]): void;
    onSubmittedWorkDone?(): Promise<void>;
  };
  destroy?(): void;
}

export interface MoonlabBrowserWebGpuBuffer {
  getMappedRange(): ArrayBuffer;
  unmap(): void;
  mapAsync(mode: number): Promise<void>;
  destroy?(): void;
}

export interface MoonlabBrowserWebGpuComputePipeline {
  getBindGroupLayout(index: number): unknown;
}

export interface MoonlabBrowserWebGpuCommandEncoder {
  beginComputePass(): MoonlabBrowserWebGpuComputePassEncoder;
  copyBufferToBuffer(
    source: MoonlabBrowserWebGpuBuffer,
    sourceOffset: number,
    destination: MoonlabBrowserWebGpuBuffer,
    destinationOffset: number,
    size: number
  ): void;
  finish(): unknown;
}

export interface MoonlabBrowserWebGpuComputePassEncoder {
  setPipeline(pipeline: MoonlabBrowserWebGpuComputePipeline): void;
  setBindGroup(index: number, bindGroup: unknown): void;
  dispatchWorkgroups(workgroupCountX: number): void;
  end(): void;
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
  const browserKernelProbe = options.browserKernelProbe
    ?? buildNotExecutedProbabilityKernelProbe(
      backendAvailable
        ? 'browser WebGPU probability kernel probe not executed; native MoonLab gate kernels are not wired on this branch'
        : 'browser WebGPU probability kernel probe not executed because no adapter/runtime was available'
    );
  const browserNativeOperationProbe = options.browserNativeOperationProbe
    ?? buildNotExecutedNativeOperationProbe(
      backendAvailable
        ? 'browser WebGPU native operation probe not executed for declared operations on this branch'
        : 'browser WebGPU native operation probe not executed because no adapter/runtime was available'
    );
  const coveredOperations = new Set<NativeCoverageRequired>([
    ...(options.coveredNativeOperations ?? []),
    ...(browserKernelProbe.executed && browserKernelProbe.passed
      ? browserKernelProbe.coveredNativeOperations
      : []),
    ...(browserNativeOperationProbe.executed
      ? browserNativeOperationProbe.coveredNativeOperations
      : []),
  ]);
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
    browserKernelProbe,
    browserNativeOperationProbe,
    blockers,
  } satisfies Omit<MoonlabWebGpuComplex64ParityScopeArtifact, 'contractValidation'>;

  return {
    ...artifactWithoutValidation,
    contractValidation: validateMoonlabWebGpuComplex64ParityScope(artifactWithoutValidation),
  };
}

export async function buildMoonlabWebGpuComplex64ParityScopeWithBrowserProbe(
  options: BuildMoonlabWebGpuComplex64ParityScopeWithBrowserProbeOptions = {}
): Promise<MoonlabWebGpuComplex64ParityScopeArtifact> {
  const browserKernelProbe = await runMoonlabBrowserWebGpuComplex64ProbabilityKernelProbe({
    gpu: options.gpu,
    runtime: options.runtime,
    workgroupSize: options.workgroupSize,
  });
  const browserNativeOperationProbe =
    await runMoonlabBrowserWebGpuComplex64NativeOperationProbe({
      gpu: options.gpu,
      runtime: options.runtime,
      workgroupSize: options.workgroupSize,
    });
  const browserProbeExecuted =
    browserKernelProbe.executed || browserNativeOperationProbe.executed;
  const backendDetection: MoonlabWebGpuComplex64BackendDetection = {
    available: browserProbeExecuted,
    runtime: options.runtime ?? detectRuntime(),
    reason: joinProbeReasons(browserKernelProbe.reason, browserNativeOperationProbe.reason),
  };

  return buildMoonlabWebGpuComplex64ParityScope({
    generatedAt: options.generatedAt,
    requireBackend: options.requireBackend,
    backendDetection,
    browserKernelProbe,
    browserNativeOperationProbe,
    webgpuParity: {
      executed: false,
      passed: false,
      maxProbabilityAbsDiff: browserKernelProbe.maxProbabilityAbsDiff,
      reason: browserProbeExecuted
        ? 'browser WebGPU probe(s) ran, but full required native operation coverage is incomplete; full reduced-fixture WebGPU parity was not executed'
        : joinProbeReasons(browserKernelProbe.reason, browserNativeOperationProbe.reason),
    },
  });
}

export async function runMoonlabBrowserWebGpuComplex64ProbabilityKernelProbe(
  options: RunMoonlabBrowserWebGpuComplex64ProbabilityKernelProbeOptions = {}
): Promise<MoonlabBrowserWebGpuComplex64ProbabilityKernelProbe> {
  const gpu = options.gpu === undefined ? detectGlobalBrowserWebGpu() : options.gpu;
  if (!gpu || typeof gpu.requestAdapter !== 'function') {
    return buildNotExecutedProbabilityKernelProbe(
      'navigator.gpu.requestAdapter is unavailable in this JavaScript runtime'
    );
  }

  let adapter: MoonlabBrowserWebGpuAdapter | null;
  try {
    adapter = await gpu.requestAdapter();
  } catch (error) {
    return buildNotExecutedProbabilityKernelProbe(
      `navigator.gpu.requestAdapter failed: ${errorMessage(error)}`
    );
  }

  if (!adapter) {
    return buildNotExecutedProbabilityKernelProbe('navigator.gpu.requestAdapter returned no adapter');
  }

  let device: MoonlabBrowserWebGpuDevice;
  try {
    device = await adapter.requestDevice();
  } catch (error) {
    return buildNotExecutedProbabilityKernelProbe(
      `GPUAdapter.requestDevice failed: ${errorMessage(error)}`
    );
  }

  try {
    const constants = browserWebGpuConstants();
    const workgroupSize = normalizeWorkgroupSize(options.workgroupSize);
    const fixtureResults: MoonlabBrowserWebGpuComplex64ProbabilityKernelProbeFixture[] = [];
    for (const fixture of reducedFixtureDefinitions()) {
      const complex64State = runFixture(fixture, true);
      const referenceProbabilities = probabilities(complex64State, true);
      const browserWebGpuProbabilities = await runProbabilityKernelFixture({
        device,
        constants,
        state: complex64State,
        workgroupSize,
      });
      const maxProbabilityAbsDiff = maxAbsDiff(
        referenceProbabilities,
        browserWebGpuProbabilities
      );
      fixtureResults.push({
        fixtureId: fixture.fixtureId,
        qubitCount: fixture.qubitCount,
        amplitudeCount: complex64State.length,
        referenceRepresentation: 'wasm-float64-interleaved',
        complex64Representation: 'complex64-interleaved-f32',
        browserWebGpuKernel: 'compute_probabilities',
        referenceProbabilities,
        browserWebGpuProbabilities,
        maxProbabilityAbsDiff,
        passed: maxProbabilityAbsDiff <= MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF,
      });
    }

    const maxProbabilityAbsDiff = Math.max(
      ...fixtureResults.map((fixture) => fixture.maxProbabilityAbsDiff)
    );
    const passed = fixtureResults.every((fixture) => fixture.passed);
    return {
      schema: MOONLAB_WEBGPU_COMPLEX64_PROBABILITY_KERNEL_PROBE_SCHEMA,
      probeKind: 'browser-webgpu-complex64-probability-kernel',
      kernel: 'compute_probabilities',
      executed: true,
      passed,
      coveredNativeOperations: passed ? ['compute_probabilities'] : [],
      fixtureResults,
      maxProbabilityAbsDiff,
      tolerance: MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF,
      reason: passed
        ? 'browser WebGPU compute_probabilities kernel matched reduced complex64 fixture probabilities'
        : 'browser WebGPU compute_probabilities kernel exceeded reduced fixture tolerance',
    };
  } catch (error) {
    return buildFailedProbabilityKernelProbe(
      `browser WebGPU compute_probabilities kernel failed: ${errorMessage(error)}`
    );
  } finally {
    device.destroy?.();
  }
}

export async function runMoonlabBrowserWebGpuComplex64NativeOperationProbe(
  options: RunMoonlabBrowserWebGpuComplex64NativeOperationProbeOptions = {}
): Promise<MoonlabBrowserWebGpuComplex64NativeOperationProbe> {
  const gpu = options.gpu === undefined ? detectGlobalBrowserWebGpu() : options.gpu;
  if (!gpu || typeof gpu.requestAdapter !== 'function') {
    return buildNotExecutedNativeOperationProbe(
      'navigator.gpu.requestAdapter is unavailable in this JavaScript runtime'
    );
  }

  let adapter: MoonlabBrowserWebGpuAdapter | null;
  try {
    adapter = await gpu.requestAdapter();
  } catch (error) {
    return buildNotExecutedNativeOperationProbe(
      `navigator.gpu.requestAdapter failed: ${errorMessage(error)}`
    );
  }

  if (!adapter) {
    return buildNotExecutedNativeOperationProbe(
      'navigator.gpu.requestAdapter returned no adapter'
    );
  }

  let device: MoonlabBrowserWebGpuDevice;
  try {
    device = await adapter.requestDevice();
  } catch (error) {
    return buildNotExecutedNativeOperationProbe(
      `GPUAdapter.requestDevice failed: ${errorMessage(error)}`
    );
  }

  try {
    const constants = browserWebGpuConstants();
    const workgroupSize = normalizeWorkgroupSize(options.workgroupSize);
    const operationResults: MoonlabBrowserWebGpuComplex64NativeOperationProbeOperationResult[] = [];
    for (const operation of MOONLAB_WEBGPU_COMPLEX64_NATIVE_OPERATION_PROBE_OPERATIONS) {
      operationResults.push(
        await runNativeOperationProbeOperation({
          device,
          constants,
          operation,
          workgroupSize,
        })
      );
    }

    const coveredNativeOperations = operationResults
      .filter((result) => result.covered)
      .map((result) => result.operation);
    const maxAmplitudeAbsDiff = maxNullable(
      operationResults.map((result) => result.maxAmplitudeAbsDiff)
    );
    const passed = operationResults.length > 0
      && operationResults.every((result) => result.passed);
    return {
      schema: MOONLAB_WEBGPU_COMPLEX64_NATIVE_OPERATION_PROBE_SCHEMA,
      probeKind: 'browser-webgpu-complex64-native-operation-probe',
      executed: operationResults.some((result) => result.executed),
      passed,
      coveredNativeOperations,
      operationResults,
      maxAmplitudeAbsDiff,
      tolerance: MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF,
      reason: passed
        ? `browser WebGPU native operation kernels (${MOONLAB_WEBGPU_COMPLEX64_NATIVE_OPERATION_PROBE_OPERATIONS.join(', ')}) matched reduced complex64 fixture amplitudes`
        : 'one or more browser WebGPU native operation kernels exceeded reduced fixture tolerance',
    };
  } catch (error) {
    return buildFailedNativeOperationProbe(
      `browser WebGPU native operation probe failed: ${errorMessage(error)}`
    );
  } finally {
    device.destroy?.();
  }
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
    'webgpu-pass-requires-full-native-coverage',
    artifact.webgpuParity?.passed !== true
      || artifact.coverage?.nativeWebGpu?.every((entry) => entry.covered === true) === true
  );
  addCheck(
    checks,
    errors,
    'webgpu-pass-requires-executed-native-probe-evidence',
    artifact.webgpuParity?.passed !== true
      || MOONLAB_WEBGPU_COMPLEX64_NATIVE_COVERAGE_REQUIRED.every((operation) => (
        hasExecutedNativeOperationEvidence(artifact, operation)
      ))
  );
  addCheck(
    checks,
    errors,
    'browser-probability-kernel-probe-contract',
    artifact.browserKernelProbe?.schema
      === MOONLAB_WEBGPU_COMPLEX64_PROBABILITY_KERNEL_PROBE_SCHEMA
      && artifact.browserKernelProbe.kernel === 'compute_probabilities'
      && (artifact.browserKernelProbe.passed !== true
        || artifact.browserKernelProbe.executed === true)
      && (!artifact.browserKernelProbe.passed
        || artifact.browserKernelProbe.coveredNativeOperations.includes('compute_probabilities'))
  );
  addCheck(
    checks,
    errors,
    'browser-native-operation-probe-contract',
    artifact.browserNativeOperationProbe?.schema
      === MOONLAB_WEBGPU_COMPLEX64_NATIVE_OPERATION_PROBE_SCHEMA
      && artifact.browserNativeOperationProbe.probeKind
        === 'browser-webgpu-complex64-native-operation-probe'
      && (artifact.browserNativeOperationProbe.passed !== true
        || artifact.browserNativeOperationProbe.executed === true)
      && artifact.browserNativeOperationProbe.operationResults.every((entry) => (
        MOONLAB_WEBGPU_COMPLEX64_NATIVE_OPERATION_PROBE_OPERATIONS.includes(entry.operation)
        && (entry.passed !== true || entry.executed === true)
        && (!entry.covered || (entry.executed === true && entry.passed === true))
        && (!entry.covered
          || artifact.browserNativeOperationProbe?.coveredNativeOperations.includes(
            entry.operation
          ))
      ))
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

interface BrowserWebGpuConstants {
  bufferUsage: {
    MAP_READ: number;
    COPY_SRC: number;
    COPY_DST: number;
    STORAGE: number;
  };
  mapMode: {
    READ: number;
  };
}

function buildNotExecutedProbabilityKernelProbe(
  reason: string
): MoonlabBrowserWebGpuComplex64ProbabilityKernelProbe {
  return {
    schema: MOONLAB_WEBGPU_COMPLEX64_PROBABILITY_KERNEL_PROBE_SCHEMA,
    probeKind: 'browser-webgpu-complex64-probability-kernel',
    kernel: 'compute_probabilities',
    executed: false,
    passed: false,
    coveredNativeOperations: [],
    fixtureResults: [],
    maxProbabilityAbsDiff: null,
    tolerance: MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF,
    reason,
  };
}

function buildFailedProbabilityKernelProbe(
  reason: string
): MoonlabBrowserWebGpuComplex64ProbabilityKernelProbe {
  return {
    ...buildNotExecutedProbabilityKernelProbe(reason),
    executed: true,
  };
}

function buildNotExecutedNativeOperationProbe(
  reason: string
): MoonlabBrowserWebGpuComplex64NativeOperationProbe {
  return {
    schema: MOONLAB_WEBGPU_COMPLEX64_NATIVE_OPERATION_PROBE_SCHEMA,
    probeKind: 'browser-webgpu-complex64-native-operation-probe',
    executed: false,
    passed: false,
    coveredNativeOperations: [],
    operationResults: MOONLAB_WEBGPU_COMPLEX64_NATIVE_OPERATION_PROBE_OPERATIONS.map(
      (operation): MoonlabBrowserWebGpuComplex64NativeOperationProbeOperationResult => ({
        operation,
        executed: false,
        passed: false,
        covered: false,
        fixtureResults: [],
        maxAmplitudeAbsDiff: null,
        tolerance: MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF,
        blocker: 'native-operation-probe-not-executed',
        reason,
      })
    ),
    maxAmplitudeAbsDiff: null,
    tolerance: MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF,
    reason,
  };
}

function buildFailedNativeOperationProbe(
  reason: string
): MoonlabBrowserWebGpuComplex64NativeOperationProbe {
  return {
    ...buildNotExecutedNativeOperationProbe(reason),
    executed: true,
    operationResults: MOONLAB_WEBGPU_COMPLEX64_NATIVE_OPERATION_PROBE_OPERATIONS.map(
      (operation): MoonlabBrowserWebGpuComplex64NativeOperationProbeOperationResult => ({
        operation,
        executed: true,
        passed: false,
        covered: false,
        fixtureResults: [],
        maxAmplitudeAbsDiff: null,
        tolerance: MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF,
        blocker: 'native-operation-probe-failed',
        reason,
      })
    ),
  };
}

function normalizeWorkgroupSize(workgroupSize: number | undefined): number {
  if (workgroupSize === undefined || !Number.isFinite(workgroupSize)) {
    return 64;
  }
  return Math.max(1, Math.floor(workgroupSize));
}

function detectGlobalBrowserWebGpu(): MoonlabBrowserWebGpu | null {
  const nav = globalThis.navigator;
  const maybeGpu = nav && typeof nav === 'object' && 'gpu' in nav
    ? (nav as { gpu?: unknown }).gpu
    : undefined;
  if (!maybeGpu || typeof maybeGpu !== 'object' || !('requestAdapter' in maybeGpu)) {
    return null;
  }
  return maybeGpu as MoonlabBrowserWebGpu;
}

function detectRuntime(): string {
  if (typeof window !== 'undefined' && typeof document !== 'undefined') {
    return 'browser';
  }
  if (typeof process !== 'undefined' && process.versions?.node) {
    return `node-${process.versions.node}`;
  }
  return 'unknown-js-runtime';
}

function browserWebGpuConstants(): BrowserWebGpuConstants {
  const webGpuGlobal = globalThis as typeof globalThis & {
    GPUBufferUsage?: Partial<BrowserWebGpuConstants['bufferUsage']>;
    GPUMapMode?: Partial<BrowserWebGpuConstants['mapMode']>;
  };
  return {
    bufferUsage: {
      MAP_READ: webGpuGlobal.GPUBufferUsage?.MAP_READ ?? 1,
      COPY_SRC: webGpuGlobal.GPUBufferUsage?.COPY_SRC ?? 4,
      COPY_DST: webGpuGlobal.GPUBufferUsage?.COPY_DST ?? 8,
      STORAGE: webGpuGlobal.GPUBufferUsage?.STORAGE ?? 128,
    },
    mapMode: {
      READ: webGpuGlobal.GPUMapMode?.READ ?? 1,
    },
  };
}

async function runProbabilityKernelFixture({
  device,
  constants,
  state,
  workgroupSize,
}: {
  device: MoonlabBrowserWebGpuDevice;
  constants: BrowserWebGpuConstants;
  state: ComplexValue[];
  workgroupSize: number;
}): Promise<number[]> {
  const amplitudeData = new Float32Array(state.length * 2);
  state.forEach((amplitude, index) => {
    amplitudeData[index * 2] = Math.fround(amplitude.real);
    amplitudeData[index * 2 + 1] = Math.fround(amplitude.imag);
  });

  const inputBuffer = device.createBuffer({
    size: amplitudeData.byteLength,
    usage: constants.bufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  new Float32Array(inputBuffer.getMappedRange()).set(amplitudeData);
  inputBuffer.unmap();

  const outputByteLength = state.length * Float32Array.BYTES_PER_ELEMENT;
  const outputBuffer = device.createBuffer({
    size: outputByteLength,
    usage: constants.bufferUsage.STORAGE | constants.bufferUsage.COPY_SRC,
  });
  const readBuffer = device.createBuffer({
    size: outputByteLength,
    usage: constants.bufferUsage.COPY_DST | constants.bufferUsage.MAP_READ,
  });

  try {
    const shaderModule = device.createShaderModule({
      code: probabilityKernelWgsl(state.length, workgroupSize),
    });
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: inputBuffer },
        },
        {
          binding: 1,
          resource: { buffer: outputBuffer },
        },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(state.length / workgroupSize));
    pass.end();
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputByteLength);
    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone?.();
    await readBuffer.mapAsync(constants.mapMode.READ);
    const mapped = readBuffer.getMappedRange();
    const result = Array.from(new Float32Array(mapped.slice(0, outputByteLength)));
    readBuffer.unmap();
    return result;
  } finally {
    inputBuffer.destroy?.();
    outputBuffer.destroy?.();
    readBuffer.destroy?.();
  }
}

function probabilityKernelWgsl(elementCount: number, workgroupSize: number): string {
  return `
@group(0) @binding(0) var<storage, read> amplitudes: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> outputProbabilities: array<f32>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= ${elementCount}u) {
    return;
  }
  let amplitude = amplitudes[index];
  outputProbabilities[index] = amplitude.x * amplitude.x + amplitude.y * amplitude.y;
}
`;
}

async function runNativeOperationProbeOperation({
  device,
  constants,
  operation,
  workgroupSize,
}: {
  device: MoonlabBrowserWebGpuDevice;
  constants: BrowserWebGpuConstants;
  operation: NativeOperationProbeOperation;
  workgroupSize: number;
}): Promise<MoonlabBrowserWebGpuComplex64NativeOperationProbeOperationResult> {
  const fixtureResults: MoonlabBrowserWebGpuComplex64NativeOperationProbeFixture[] = [];
  for (const fixture of nativeOperationProbeFixtureDefinitions(operation)) {
    const inputState = fixture.inputState.map(froundComplex);
    const referenceState = applyOperation(
      inputState,
      fixture.qubitCount,
      fixture.operation,
      true
    );
    const browserWebGpuState = await runNativeOperationKernelFixture({
      device,
      constants,
      state: inputState,
      operation: fixture.operation,
      qubitCount: fixture.qubitCount,
      workgroupSize,
    });
    const inputAmplitudes = flattenComplexValues(inputState, true);
    const referenceAmplitudes = flattenComplexValues(referenceState, true);
    const browserWebGpuAmplitudes = flattenComplexValues(browserWebGpuState, true);
    const maxAmplitudeAbsDiff = maxAbsDiff(referenceAmplitudes, browserWebGpuAmplitudes);

    fixtureResults.push({
      fixtureId: fixture.fixtureId,
      qubitCount: fixture.qubitCount,
      amplitudeCount: inputState.length,
      operation,
      referenceRepresentation: 'cpu-complex64-interleaved-f32',
      complex64Representation: 'complex64-interleaved-f32',
      browserWebGpuKernel: operation,
      inputAmplitudes,
      referenceAmplitudes,
      browserWebGpuAmplitudes,
      maxAmplitudeAbsDiff,
      passed: maxAmplitudeAbsDiff <= MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF,
    });
  }

  const maxAmplitudeAbsDiff = Math.max(
    ...fixtureResults.map((fixture) => fixture.maxAmplitudeAbsDiff)
  );
  const passed = fixtureResults.every((fixture) => fixture.passed);
  return {
    operation,
    executed: true,
    passed,
    covered: passed,
    fixtureResults,
    maxAmplitudeAbsDiff,
    tolerance: MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF,
    reason: passed
      ? `browser WebGPU native ${operation} operation kernel matched reduced complex64 fixture amplitudes`
      : `browser WebGPU native ${operation} operation kernel exceeded reduced fixture tolerance`,
  };
}

async function runNativeOperationKernelFixture({
  device,
  constants,
  state,
  operation,
  qubitCount,
  workgroupSize,
}: {
  device: MoonlabBrowserWebGpuDevice;
  constants: BrowserWebGpuConstants;
  state: ComplexValue[];
  operation: FixtureOperation;
  qubitCount: number;
  workgroupSize: number;
}): Promise<ComplexValue[]> {
  const amplitudeData = new Float32Array(flattenComplexValues(state, true));
  const inputBuffer = device.createBuffer({
    size: amplitudeData.byteLength,
    usage: constants.bufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  new Float32Array(inputBuffer.getMappedRange()).set(amplitudeData);
  inputBuffer.unmap();

  const outputByteLength = amplitudeData.byteLength;
  const outputBuffer = device.createBuffer({
    size: outputByteLength,
    usage: constants.bufferUsage.STORAGE | constants.bufferUsage.COPY_SRC,
  });
  const readBuffer = device.createBuffer({
    size: outputByteLength,
    usage: constants.bufferUsage.COPY_DST | constants.bufferUsage.MAP_READ,
  });

  try {
    const shaderModule = device.createShaderModule({
      code: nativeOperationKernelWgsl({
        operation,
        elementCount: state.length,
        qubitCount,
        workgroupSize,
      }),
    });
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: inputBuffer },
        },
        {
          binding: 1,
          resource: { buffer: outputBuffer },
        },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(state.length / workgroupSize));
    pass.end();
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputByteLength);
    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone?.();
    await readBuffer.mapAsync(constants.mapMode.READ);
    const mapped = readBuffer.getMappedRange();
    const result = complexValuesFromFloat32(
      new Float32Array(mapped.slice(0, outputByteLength))
    );
    readBuffer.unmap();
    return result;
  } finally {
    inputBuffer.destroy?.();
    outputBuffer.destroy?.();
    readBuffer.destroy?.();
  }
}

function nativeOperationKernelWgsl({
  operation,
  elementCount,
  qubitCount,
  workgroupSize,
}: {
  operation: FixtureOperation;
  elementCount: number;
  qubitCount: number;
  workgroupSize: number;
}): string {
  if (operation.operation === 'hadamard') {
    const qubit = requiredQubit(operation);
    assertQubit(qubit, qubitCount);
    return hadamardKernelWgsl(elementCount, qubit, workgroupSize);
  }
  if (operation.operation === 'pauli_x') {
    const qubit = requiredQubit(operation);
    assertQubit(qubit, qubitCount);
    return pauliXKernelWgsl(elementCount, qubit, workgroupSize);
  }
  if (operation.operation === 'pauli_z') {
    const qubit = requiredQubit(operation);
    assertQubit(qubit, qubitCount);
    return pauliZKernelWgsl(elementCount, qubit, workgroupSize);
  }
  throw new Error(`Unsupported native operation probe kernel: ${operation.operation}`);
}

function hadamardKernelWgsl(
  elementCount: number,
  qubit: number,
  workgroupSize: number
): string {
  const qubitMask = 1 << qubit;
  return `
@group(0) @binding(0) var<storage, read> inputAmplitudes: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> outputAmplitudes: array<vec2<f32>>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let zeroIndex = globalId.x;
  if (zeroIndex >= ${elementCount}u || (zeroIndex & ${qubitMask}u) != 0u) {
    return;
  }
  let oneIndex = zeroIndex | ${qubitMask}u;
  let zero = inputAmplitudes[zeroIndex];
  let one = inputAmplitudes[oneIndex];
  let invSqrt2 = 0.70710678118654752440f;
  outputAmplitudes[zeroIndex] = vec2<f32>(
    (zero.x + one.x) * invSqrt2,
    (zero.y + one.y) * invSqrt2
  );
  outputAmplitudes[oneIndex] = vec2<f32>(
    (zero.x - one.x) * invSqrt2,
    (zero.y - one.y) * invSqrt2
  );
}
`;
}

function pauliXKernelWgsl(
  elementCount: number,
  qubit: number,
  workgroupSize: number
): string {
  const qubitMask = 1 << qubit;
  return `
@group(0) @binding(0) var<storage, read> inputAmplitudes: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> outputAmplitudes: array<vec2<f32>>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let zeroIndex = globalId.x;
  if (zeroIndex >= ${elementCount}u || (zeroIndex & ${qubitMask}u) != 0u) {
    return;
  }
  let oneIndex = zeroIndex | ${qubitMask}u;
  outputAmplitudes[zeroIndex] = inputAmplitudes[oneIndex];
  outputAmplitudes[oneIndex] = inputAmplitudes[zeroIndex];
}
`;
}

function pauliZKernelWgsl(
  elementCount: number,
  qubit: number,
  workgroupSize: number
): string {
  const qubitMask = 1 << qubit;
  return `
@group(0) @binding(0) var<storage, read> inputAmplitudes: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> outputAmplitudes: array<vec2<f32>>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let index = globalId.x;
  if (index >= ${elementCount}u) {
    return;
  }
  let amplitude = inputAmplitudes[index];
  if ((index & ${qubitMask}u) == 0u) {
    outputAmplitudes[index] = amplitude;
    return;
  }
  outputAmplitudes[index] = vec2<f32>(-amplitude.x, -amplitude.y);
}
`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
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

function nativeOperationProbeFixtureDefinitions(
  operation: NativeOperationProbeOperation
): NativeOperationProbeFixtureDefinition[] {
  if (operation === 'hadamard') {
    return [
      {
        fixtureId: 'hadamard-1q-zero-state-amplitudes',
        qubitCount: 1,
        operation: { operation: 'hadamard', qubit: 0 },
        inputState: initialState(1),
      },
      {
        fixtureId: 'hadamard-2q-qubit-1-complex-amplitudes',
        qubitCount: 2,
        operation: { operation: 'hadamard', qubit: 1 },
        inputState: [
          { real: Math.SQRT1_2, imag: 0 },
          { real: 0, imag: 0 },
          { real: 0, imag: Math.SQRT1_2 },
          { real: 0, imag: 0 },
        ],
      },
    ];
  }
  if (operation === 'pauli_x') {
    return [
      {
        fixtureId: 'pauli-x-1q-zero-state-amplitudes',
        qubitCount: 1,
        operation: { operation: 'pauli_x', qubit: 0 },
        inputState: initialState(1),
      },
      {
        fixtureId: 'pauli-x-2q-qubit-1-complex-amplitudes',
        qubitCount: 2,
        operation: { operation: 'pauli_x', qubit: 1 },
        inputState: [
          { real: 0.25, imag: -0.5 },
          { real: 0.125, imag: 0.375 },
          { real: -0.75, imag: 0.0625 },
          { real: 0.5, imag: -0.25 },
        ],
      },
    ];
  }
  if (operation === 'pauli_z') {
    return [
      {
        fixtureId: 'pauli-z-1q-one-state-amplitudes',
        qubitCount: 1,
        operation: { operation: 'pauli_z', qubit: 0 },
        inputState: [
          { real: 0, imag: 0 },
          { real: 1, imag: 0 },
        ],
      },
      {
        fixtureId: 'pauli-z-2q-qubit-1-complex-amplitudes',
        qubitCount: 2,
        operation: { operation: 'pauli_z', qubit: 1 },
        inputState: [
          { real: 0.25, imag: -0.5 },
          { real: 0.125, imag: 0.375 },
          { real: -0.75, imag: 0.0625 },
          { real: 0.5, imag: -0.25 },
        ],
      },
    ];
  }
  throw new Error(`Unsupported native operation probe fixture: ${operation}`);
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

function flattenComplexValues(state: ComplexValue[], complex64: boolean): number[] {
  return state.flatMap((amplitude) => {
    const real = complex64 ? Math.fround(amplitude.real) : amplitude.real;
    const imag = complex64 ? Math.fround(amplitude.imag) : amplitude.imag;
    return [real, imag];
  });
}

function complexValuesFromFloat32(values: Float32Array): ComplexValue[] {
  const state: ComplexValue[] = [];
  for (let index = 0; index < values.length; index += 2) {
    state.push({
      real: Math.fround(values[index] ?? 0),
      imag: Math.fround(values[index + 1] ?? 0),
    });
  }
  return state;
}

function maxAbsDiff(left: number[], right: number[]): number {
  return left.reduce((maxDiff, value, index) => (
    Math.max(maxDiff, Math.abs(value - (right[index] ?? Number.NaN)))
  ), 0);
}

function maxNullable(values: Array<number | null>): number | null {
  const numbers = values.filter((value): value is number => typeof value === 'number');
  return numbers.length === 0 ? null : Math.max(...numbers);
}

function joinProbeReasons(...reasons: string[]): string {
  return [...new Set(reasons.filter((reason) => reason.length > 0))].join('; ');
}

function hasExecutedNativeOperationEvidence(
  artifact: Partial<MoonlabWebGpuComplex64ParityScopeArtifact>,
  operation: NativeCoverageRequired
): boolean {
  if (operation === 'compute_probabilities') {
    return artifact.browserKernelProbe?.executed === true
      && artifact.browserKernelProbe.passed === true
      && artifact.browserKernelProbe.coveredNativeOperations.includes(operation);
  }
  return artifact.browserNativeOperationProbe?.operationResults.some((result) => (
    result.operation === operation
    && result.executed === true
    && result.passed === true
    && result.covered === true
  )) === true;
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
