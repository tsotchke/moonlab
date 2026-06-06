import { describe, expect, it } from 'vitest';
import {
  buildMoonlabWebGpuComplex64ParityScope,
  MOONLAB_WEBGPU_COMPLEX64_MAX_PROBABILITY_ABS_DIFF,
  MOONLAB_WEBGPU_COMPLEX64_NATIVE_COVERAGE_EXCLUDED,
  MOONLAB_WEBGPU_COMPLEX64_NATIVE_COVERAGE_REQUIRED,
  MOONLAB_WEBGPU_COMPLEX64_PARITY_SCOPE_SCHEMA,
  validateMoonlabWebGpuComplex64ParityScope,
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
});
