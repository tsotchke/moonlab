/**
 * WebGPU / tensor GPU backend helpers for WASM builds.
 */

import type { MoonlabModule } from './memory';
import { getModule } from './wasm-loader';

export type TensorGpuBackend = 'none' | 'metal' | 'cuda' | 'webgpu' | 'unknown';

const BACKEND_NAMES: Record<number, TensorGpuBackend> = {
  0: 'none',
  1: 'metal',
  2: 'cuda',
  3: 'webgpu',
};

function runtimeWebGPUSupported(): boolean {
  if (typeof navigator === 'undefined') {
    return false;
  }
  const nav = navigator as Navigator & { gpu?: unknown };
  return typeof nav.gpu !== 'undefined';
}

function tensorBackendFromCode(code: number): TensorGpuBackend {
  return BACKEND_NAMES[code] ?? 'unknown';
}

function canQueryBackend(module: MoonlabModule): module is MoonlabModule & {
  _tensor_gpu_backend_type: (ctxPtr: number) => number;
} {
  return typeof module._tensor_gpu_backend_type === 'function';
}

/**
 * Returns whether WebGPU appears available in this runtime + WASM backend.
 */
export async function isWebGPUAvailable(): Promise<boolean> {
  const module = await getModule();
  if (typeof module._tensor_gpu_get_context === 'function' && canQueryBackend(module)) {
    const ctxPtr = module._tensor_gpu_get_context();
    if (!ctxPtr) {
      return false;
    }
    return module._tensor_gpu_backend_type(ctxPtr) === 3;
  }
  if (typeof module._tensor_gpu_webgpu_available === 'function') {
    return module._tensor_gpu_webgpu_available() !== 0;
  }
  return runtimeWebGPUSupported();
}

/**
 * Initializes tensor GPU backend and returns true when WebGPU backend is active.
 */
export async function initializeWebGPUBackend(): Promise<boolean> {
  const module = await getModule();
  if (typeof module._tensor_gpu_get_context !== 'function' || !canQueryBackend(module)) {
    return false;
  }

  const ctxPtr = module._tensor_gpu_get_context();
  if (!ctxPtr) {
    return false;
  }

  return module._tensor_gpu_backend_type(ctxPtr) === 3;
}

/**
 * Returns active tensor GPU backend name.
 */
export async function getActiveTensorGPUBackend(): Promise<TensorGpuBackend> {
  const module = await getModule();
  if (!canQueryBackend(module)) {
    return 'unknown';
  }
  const backendCode = module._tensor_gpu_backend_type(0);
  return tensorBackendFromCode(backendCode);
}

/**
 * Returns whether any tensor GPU backend is available.
 */
export async function isTensorGPUAvailable(): Promise<boolean> {
  const module = await getModule();
  if (typeof module._tensor_gpu_available !== 'function') {
    return false;
  }
  return module._tensor_gpu_available() !== 0;
}
