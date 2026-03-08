import { describe, expect, it } from 'vitest';
import {
  getActiveTensorGPUBackend,
  initializeWebGPUBackend,
  isTensorGPUAvailable,
  isWebGPUAvailable,
} from '../webgpu';

describe('WebGPU Backend Integration', () => {
  it('returns a boolean for runtime WebGPU availability', async () => {
    const available = await isWebGPUAvailable();
    expect(typeof available).toBe('boolean');
  });

  it('initializes backend without throwing', async () => {
    const initialized = await initializeWebGPUBackend();
    expect(typeof initialized).toBe('boolean');
  });

  it('returns known backend label', async () => {
    const backend = await getActiveTensorGPUBackend();
    expect(['none', 'metal', 'cuda', 'webgpu', 'unknown']).toContain(backend);
  });

  it('returns a boolean for tensor GPU availability', async () => {
    const available = await isTensorGPUAvailable();
    expect(typeof available).toBe('boolean');
  });
});
