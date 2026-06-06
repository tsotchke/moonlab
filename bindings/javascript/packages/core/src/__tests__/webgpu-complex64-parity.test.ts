import { describe, expect, it } from 'vitest';
import {
  buildMoonlabWebGpuComplex64ParityScopeWithBrowserProbe,
  buildMoonlabWebGpuComplex64ParityScope,
  MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF,
  MOONLAB_WEBGPU_COMPLEX64_NATIVE_COVERAGE_EXCLUDED,
  MOONLAB_WEBGPU_COMPLEX64_NATIVE_COVERAGE_REQUIRED,
  MOONLAB_WEBGPU_COMPLEX64_PARITY_SCOPE_SCHEMA,
  MOONLAB_WEBGPU_COMPLEX64_PROBABILITY_KERNEL_PROBE_SCHEMA,
  runMoonlabBrowserWebGpuComplex64ProbabilityKernelProbe,
  validateMoonlabWebGpuComplex64ParityScope,
  type MoonlabBrowserWebGpuComplex64ProbabilityKernelProbe,
  type MoonlabWebGpuComplex64ParityScopeArtifact,
} from '../webgpu-complex64-parity';

describe('WebGPU complex64 parity scope contract', () => {
  it('emits an explicit reduced-fixture no-backend scope artifact', () => {
    const artifact = buildMoonlabWebGpuComplex64ParityScope({
      generatedAt: '2026-06-06T20:00:00.000Z',
      backendDetection: {
        available: false,
        runtime: 'node-test',
        reason: 'test runtime has no browser WebGPU adapter',
      },
    });

    expect(artifact.schema).toBe(MOONLAB_WEBGPU_COMPLEX64_PARITY_SCOPE_SCHEMA);
    expect(artifact.status).toBe('scope-ready-backend-unavailable');
    expect(artifact.contractReady).toBe(true);
    expect(artifact.backend).toBe('webgpu');
    expect(artifact.backendAvailable).toBe(false);
    expect(artifact.reducedFixtureOnly).toBe(true);
    expect(artifact.fullFidelityMagnetarSimulation).toBe(false);
    expect(artifact.fullPhysicsValidation).toBe(false);
    expect(artifact.fidelityRuntimeScope.fullFidelityMagnetarSimulation).toBe(false);
    expect(artifact.fidelityRuntimeScope.fullPhysicsValidation).toBe(false);
    expect(artifact.nativeCoverageRequired).toEqual(
      MOONLAB_WEBGPU_COMPLEX64_NATIVE_COVERAGE_REQUIRED
    );
    expect(artifact.nativeCoverageExcluded).toEqual(
      MOONLAB_WEBGPU_COMPLEX64_NATIVE_COVERAGE_EXCLUDED
    );
    expect(artifact.coverage.cpuFallbackExcluded).toEqual([
      expect.objectContaining({
        operation: 'phase',
        excludedFromNativeCoverage: true,
      }),
    ]);
    expect(artifact.coverage.nativeWebGpu.every((entry) => entry.covered === false)).toBe(true);
    expect(artifact.complex64Preflight.passed).toBe(true);
    expect(artifact.complex64Preflight.maxProbabilityAbsDiff).toBeLessThanOrEqual(
      MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF
    );
    expect(artifact.webgpuParity.executed).toBe(false);
    expect(artifact.webgpuParity.passed).toBe(false);
    expect(artifact.browserKernelProbe).toEqual(
      expect.objectContaining({
        schema: MOONLAB_WEBGPU_COMPLEX64_PROBABILITY_KERNEL_PROBE_SCHEMA,
        kernel: 'compute_probabilities',
        executed: false,
        passed: false,
        coveredNativeOperations: [],
      })
    );
    expect(artifact.blockers).toContain('browser-webgpu-adapter-unavailable');
    expect(artifact.blockers).toContain('browser-webgpu-kernel-parity-not-executed');
    expect(artifact.contractValidation.valid).toBe(true);
  });

  it('marks a required missing backend as blocked without changing reduced scope', () => {
    const artifact = buildMoonlabWebGpuComplex64ParityScope({
      generatedAt: '2026-06-06T20:00:00.000Z',
      requireBackend: true,
      backendAvailable: false,
    });

    expect(artifact.status).toBe('blocked-webgpu-backend-required');
    expect(artifact.requireBackend).toBe(true);
    expect(artifact.blockers).toContain('required-browser-webgpu-backend-missing');
    expect(artifact.fullFidelityMagnetarSimulation).toBe(false);
    expect(artifact.fullPhysicsValidation).toBe(false);
    expect(validateMoonlabWebGpuComplex64ParityScope(artifact).valid).toBe(true);
  });

  it('rejects full-physics overclaims and fake WebGPU parity passes', () => {
    const artifact = buildMoonlabWebGpuComplex64ParityScope({
      generatedAt: '2026-06-06T20:00:00.000Z',
    });
    const invalid = {
      ...artifact,
      fullFidelityMagnetarSimulation: true,
      fullPhysicsValidation: true,
      webgpuParity: {
        ...artifact.webgpuParity,
        passed: true,
        executed: false,
      },
    } as unknown as MoonlabWebGpuComplex64ParityScopeArtifact;
    const validation = validateMoonlabWebGpuComplex64ParityScope(invalid);

    expect(validation.valid).toBe(false);
    expect(validation.errors).toContain('no-full-fidelity-magnetar-simulation check failed');
    expect(validation.errors).toContain('no-full-physics-validation check failed');
    expect(validation.errors).toContain('webgpu-pass-requires-execution check failed');
  });

  it('records a partial browser probability-kernel probe without claiming full parity', () => {
    const browserKernelProbe: MoonlabBrowserWebGpuComplex64ProbabilityKernelProbe = {
      schema: MOONLAB_WEBGPU_COMPLEX64_PROBABILITY_KERNEL_PROBE_SCHEMA,
      probeKind: 'browser-webgpu-complex64-probability-kernel',
      kernel: 'compute_probabilities',
      executed: true,
      passed: true,
      coveredNativeOperations: ['compute_probabilities'],
      fixtureResults: [
        {
          fixtureId: 'test-fixture',
          qubitCount: 1,
          amplitudeCount: 2,
          referenceRepresentation: 'wasm-float64-interleaved',
          complex64Representation: 'complex64-interleaved-f32',
          browserWebGpuKernel: 'compute_probabilities',
          referenceProbabilities: [1, 0],
          browserWebGpuProbabilities: [1, 0],
          maxProbabilityAbsDiff: 0,
          passed: true,
        },
      ],
      maxProbabilityAbsDiff: 0,
      tolerance: MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF,
      reason: 'test browser probability kernel matched',
    };
    const artifact = buildMoonlabWebGpuComplex64ParityScope({
      generatedAt: '2026-06-06T20:00:00.000Z',
      backendAvailable: true,
      browserKernelProbe,
    });

    expect(artifact.browserKernelProbe.executed).toBe(true);
    expect(artifact.coverage.nativeWebGpu).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          operation: 'compute_probabilities',
          covered: true,
          status: 'covered-by-browser-webgpu',
        }),
        expect.objectContaining({
          operation: 'hadamard',
          covered: false,
          status: 'not-run-runtime-backend-not-wired',
        }),
      ])
    );
    expect(artifact.webgpuParity.executed).toBe(false);
    expect(artifact.webgpuParity.passed).toBe(false);
    expect(artifact.blockers).toContain('native-webgpu-operation-coverage-not-yet-recorded');
    expect(artifact.contractValidation.valid).toBe(true);
  });

  it('rejects a WebGPU parity pass when native gate coverage is partial', () => {
    const artifact = buildMoonlabWebGpuComplex64ParityScope({
      generatedAt: '2026-06-06T20:00:00.000Z',
      backendAvailable: true,
      coveredNativeOperations: ['compute_probabilities'],
    });
    const invalid = {
      ...artifact,
      webgpuParity: {
        ...artifact.webgpuParity,
        executed: true,
        passed: true,
        maxProbabilityAbsDiff: 0,
      },
    } as MoonlabWebGpuComplex64ParityScopeArtifact;
    const validation = validateMoonlabWebGpuComplex64ParityScope(invalid);

    expect(validation.valid).toBe(false);
    expect(validation.errors).toContain(
      'webgpu-pass-requires-full-native-coverage check failed'
    );
  });

  it('allows failed browser probability-kernel evidence without marking coverage', () => {
    const browserKernelProbe: MoonlabBrowserWebGpuComplex64ProbabilityKernelProbe = {
      schema: MOONLAB_WEBGPU_COMPLEX64_PROBABILITY_KERNEL_PROBE_SCHEMA,
      probeKind: 'browser-webgpu-complex64-probability-kernel',
      kernel: 'compute_probabilities',
      executed: true,
      passed: false,
      coveredNativeOperations: [],
      fixtureResults: [],
      maxProbabilityAbsDiff: null,
      tolerance: MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF,
      reason: 'test probability kernel failed',
    };
    const artifact = buildMoonlabWebGpuComplex64ParityScope({
      generatedAt: '2026-06-06T20:00:00.000Z',
      backendAvailable: true,
      browserKernelProbe,
    });

    expect(artifact.coverage.nativeWebGpu).toContainEqual(
      expect.objectContaining({
        operation: 'compute_probabilities',
        covered: false,
      })
    );
    expect(artifact.webgpuParity.passed).toBe(false);
    expect(artifact.contractValidation.valid).toBe(true);
  });

  it('exposes a no-adapter browser probability-kernel probe path', async () => {
    const probe = await runMoonlabBrowserWebGpuComplex64ProbabilityKernelProbe({
      gpu: null,
    });

    expect(probe.schema).toBe(MOONLAB_WEBGPU_COMPLEX64_PROBABILITY_KERNEL_PROBE_SCHEMA);
    expect(probe.executed).toBe(false);
    expect(probe.passed).toBe(false);
    expect(probe.coveredNativeOperations).toEqual([]);
    expect(probe.reason).toContain('navigator.gpu.requestAdapter is unavailable');
  });

  it('builds a probe-aware scope artifact without executing WebGPU outside an adapter', async () => {
    const artifact = await buildMoonlabWebGpuComplex64ParityScopeWithBrowserProbe({
      generatedAt: '2026-06-06T20:00:00.000Z',
      gpu: null,
    });

    expect(artifact.browserKernelProbe.executed).toBe(false);
    expect(artifact.webgpuParity.executed).toBe(false);
    expect(artifact.webgpuParity.passed).toBe(false);
    expect(artifact.blockers).toContain('browser-webgpu-adapter-unavailable');
    expect(artifact.contractValidation.valid).toBe(true);
  });
});
