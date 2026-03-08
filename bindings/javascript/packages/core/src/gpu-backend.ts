import type { Complex } from './complex';
import { WasmMemory, type MoonlabModule } from './memory';
import { getModule } from './wasm-loader';

export const GPU_BACKEND_NONE = 0;
export const GPU_BACKEND_METAL = 1;
export const GPU_BACKEND_WEBGPU = 2;
export const GPU_BACKEND_OPENCL = 3;
export const GPU_BACKEND_VULKAN = 4;
export const GPU_BACKEND_CUDA = 5;
export const GPU_BACKEND_CUQUANTUM = 6;
export const GPU_BACKEND_AUTO = 7;

export type GPUBackendTypeCode =
  | 0
  | 1
  | 2
  | 3
  | 4
  | 5
  | 6
  | 7;

export function backendTypeName(code: number): string {
  switch (code) {
    case GPU_BACKEND_NONE:
      return 'none';
    case GPU_BACKEND_METAL:
      return 'metal';
    case GPU_BACKEND_WEBGPU:
      return 'webgpu';
    case GPU_BACKEND_OPENCL:
      return 'opencl';
    case GPU_BACKEND_VULKAN:
      return 'vulkan';
    case GPU_BACKEND_CUDA:
      return 'cuda';
    case GPU_BACKEND_CUQUANTUM:
      return 'cuquantum';
    case GPU_BACKEND_AUTO:
      return 'auto';
    default:
      return 'unknown';
  }
}

export async function isUnifiedGPUApiAvailable(): Promise<boolean> {
  const module = await getModule();
  return (
    typeof module._gpu_compute_init === 'function' &&
    typeof module._gpu_compute_free === 'function' &&
    typeof module._gpu_get_backend_type === 'function' &&
    typeof module._gpu_buffer_create_from_data === 'function' &&
    typeof module._gpu_buffer_read === 'function' &&
    typeof module._gpu_buffer_write === 'function'
  );
}

export class GPUBackendSession {
  private module: MoonlabModule;
  private memory: WasmMemory;
  private ctxPtr: number;
  private disposed = false;

  private constructor(module: MoonlabModule, ctxPtr: number) {
    this.module = module;
    this.memory = new WasmMemory(module);
    this.ctxPtr = ctxPtr;
  }

  static async create(preferred: GPUBackendTypeCode = GPU_BACKEND_AUTO): Promise<GPUBackendSession | null> {
    const module = await getModule();
    if (typeof module._gpu_compute_init !== 'function') {
      return null;
    }
    const ctxPtr = module._gpu_compute_init(preferred);
    if (!ctxPtr) {
      return null;
    }
    return new GPUBackendSession(module, ctxPtr);
  }

  get pointer(): number {
    return this.ctxPtr;
  }

  get backendType(): number {
    this.ensureNotDisposed();
    if (typeof this.module._gpu_get_backend_type !== 'function') {
      return GPU_BACKEND_NONE;
    }
    return this.module._gpu_get_backend_type(this.ctxPtr);
  }

  get backendName(): string {
    return backendTypeName(this.backendType);
  }

  get nativeAccelerated(): boolean {
    this.ensureNotDisposed();
    if (typeof this.module._gpu_is_native_accelerated !== 'function') {
      return false;
    }
    return this.module._gpu_is_native_accelerated(this.ctxPtr) !== 0;
  }

  createBufferFromInterleaved(data: Float64Array): number {
    this.ensureNotDisposed();
    if (typeof this.module._gpu_buffer_create_from_data !== 'function') {
      throw new Error('Unified GPU buffer API not available');
    }
    const bytes = data.byteLength;
    const ptr = this.memory.alloc(bytes);
    this.memory.writeFloat64Array(ptr, data);
    const bufferPtr = this.module._gpu_buffer_create_from_data(this.ctxPtr, ptr, bytes);
    this.memory.free(ptr);
    if (!bufferPtr) {
      throw new Error('gpu_buffer_create_from_data failed');
    }
    return bufferPtr;
  }

  createBufferFromComplex(data: Complex[]): number {
    const interleaved = new Float64Array(data.length * 2);
    for (let i = 0; i < data.length; i++) {
      interleaved[i * 2] = data[i].real;
      interleaved[i * 2 + 1] = data[i].imag;
    }
    return this.createBufferFromInterleaved(interleaved);
  }

  readInterleavedBuffer(bufferPtr: number, complexCount: number): Float64Array {
    this.ensureNotDisposed();
    if (typeof this.module._gpu_buffer_read !== 'function') {
      throw new Error('Unified GPU buffer read API not available');
    }
    const bytes = complexCount * 16;
    const outPtr = this.memory.alloc(bytes);
    const rc = this.module._gpu_buffer_read(bufferPtr, outPtr, bytes, 0);
    if (rc !== 0) {
      this.memory.free(outPtr);
      throw new Error(`gpu_buffer_read failed: ${rc}`);
    }
    const out = this.memory.readFloat64Array(outPtr, complexCount * 2);
    this.memory.free(outPtr);
    return out;
  }

  readComplexBuffer(bufferPtr: number, complexCount: number): Complex[] {
    const interleaved = this.readInterleavedBuffer(bufferPtr, complexCount);
    const out: Complex[] = [];
    for (let i = 0; i < complexCount; i++) {
      out.push({
        real: interleaved[i * 2],
        imag: interleaved[i * 2 + 1],
      });
    }
    return out;
  }

  freeBuffer(bufferPtr: number): void {
    if (typeof this.module._gpu_buffer_free === 'function' && bufferPtr) {
      this.module._gpu_buffer_free(bufferPtr);
    }
  }

  hadamard(bufferPtr: number, qubit: number, stateDim: number): number {
    this.ensureNotDisposed();
    if (typeof this.module._gpu_hadamard_u32 === 'function') {
      return this.module._gpu_hadamard_u32(this.ctxPtr, bufferPtr, qubit, stateDim >>> 0);
    }
    if (typeof this.module._gpu_hadamard !== 'function') return -7;
    return this.module._gpu_hadamard(this.ctxPtr, bufferPtr, qubit, BigInt(stateDim));
  }

  hadamardAll(bufferPtr: number, numQubits: number, stateDim: number): number {
    this.ensureNotDisposed();
    if (typeof this.module._gpu_hadamard_all_u32 === 'function') {
      return this.module._gpu_hadamard_all_u32(this.ctxPtr, bufferPtr, numQubits, stateDim >>> 0);
    }
    if (typeof this.module._gpu_hadamard_all !== 'function') return -7;
    return this.module._gpu_hadamard_all(this.ctxPtr, bufferPtr, numQubits, BigInt(stateDim));
  }

  pauliX(bufferPtr: number, qubit: number, stateDim: number): number {
    this.ensureNotDisposed();
    if (typeof this.module._gpu_pauli_x_u32 === 'function') {
      return this.module._gpu_pauli_x_u32(this.ctxPtr, bufferPtr, qubit, stateDim >>> 0);
    }
    if (typeof this.module._gpu_pauli_x !== 'function') return -7;
    return this.module._gpu_pauli_x(this.ctxPtr, bufferPtr, qubit, BigInt(stateDim));
  }

  pauliZ(bufferPtr: number, qubit: number, stateDim: number): number {
    this.ensureNotDisposed();
    if (typeof this.module._gpu_pauli_z_u32 === 'function') {
      return this.module._gpu_pauli_z_u32(this.ctxPtr, bufferPtr, qubit, stateDim >>> 0);
    }
    if (typeof this.module._gpu_pauli_z !== 'function') return -7;
    return this.module._gpu_pauli_z(this.ctxPtr, bufferPtr, qubit, BigInt(stateDim));
  }

  phase(bufferPtr: number, qubit: number, theta: number, stateDim: number): number {
    this.ensureNotDisposed();
    if (typeof this.module._gpu_phase_u32 === 'function') {
      return this.module._gpu_phase_u32(this.ctxPtr, bufferPtr, qubit, theta, stateDim >>> 0);
    }
    if (typeof this.module._gpu_phase !== 'function') return -7;
    return this.module._gpu_phase(this.ctxPtr, bufferPtr, qubit, theta, BigInt(stateDim));
  }

  cnot(bufferPtr: number, control: number, target: number, stateDim: number): number {
    this.ensureNotDisposed();
    if (typeof this.module._gpu_cnot_u32 === 'function') {
      return this.module._gpu_cnot_u32(this.ctxPtr, bufferPtr, control, target, stateDim >>> 0);
    }
    if (typeof this.module._gpu_cnot !== 'function') return -7;
    return this.module._gpu_cnot(this.ctxPtr, bufferPtr, control, target, BigInt(stateDim));
  }

  computeProbabilities(amplitudesBufferPtr: number, stateDim: number): Float64Array {
    this.ensureNotDisposed();
    if (typeof this.module._gpu_buffer_create !== 'function' ||
        typeof this.module._gpu_compute_probabilities !== 'function' ||
        typeof this.module._gpu_buffer_read !== 'function') {
      throw new Error('Unified GPU probability API not available');
    }

    const probabilitiesBuffer = this.module._gpu_buffer_create(this.ctxPtr, stateDim * 8);
    if (!probabilitiesBuffer) {
      throw new Error('gpu_buffer_create failed for probabilities');
    }

    try {
      const rc = typeof this.module._gpu_compute_probabilities_u32 === 'function'
        ? this.module._gpu_compute_probabilities_u32(
            this.ctxPtr,
            amplitudesBufferPtr,
            probabilitiesBuffer,
            stateDim >>> 0
          )
        : this.module._gpu_compute_probabilities(
            this.ctxPtr,
            amplitudesBufferPtr,
            probabilitiesBuffer,
            BigInt(stateDim)
          );
      if (rc !== 0) {
        throw new Error(`gpu_compute_probabilities failed: ${rc}`);
      }

      const outPtr = this.memory.alloc(stateDim * 8);
      const readRc = this.module._gpu_buffer_read(probabilitiesBuffer, outPtr, stateDim * 8, 0);
      if (readRc !== 0) {
        this.memory.free(outPtr);
        throw new Error(`gpu_buffer_read probabilities failed: ${readRc}`);
      }
      const out = this.memory.readFloat64Array(outPtr, stateDim);
      this.memory.free(outPtr);
      return out;
    } finally {
      this.freeBuffer(probabilitiesBuffer);
    }
  }

  sumSquaredMagnitudes(amplitudesBufferPtr: number, stateDim: number): number {
    this.ensureNotDisposed();
    if (typeof this.module._gpu_sum_squared_magnitudes !== 'function') {
      throw new Error('gpu_sum_squared_magnitudes not available');
    }

    const outPtr = this.memory.allocFloat64Array(1);
    const rc = typeof this.module._gpu_sum_squared_magnitudes_u32 === 'function'
      ? this.module._gpu_sum_squared_magnitudes_u32(
          this.ctxPtr,
          amplitudesBufferPtr,
          stateDim >>> 0,
          outPtr
        )
      : this.module._gpu_sum_squared_magnitudes(
          this.ctxPtr,
          amplitudesBufferPtr,
          BigInt(stateDim),
          outPtr
        );
    if (rc !== 0) {
      this.memory.free(outPtr);
      throw new Error(`gpu_sum_squared_magnitudes failed: ${rc}`);
    }
    const result = this.memory.readFloat64(outPtr);
    this.memory.free(outPtr);
    return result;
  }

  normalize(amplitudesBufferPtr: number, norm: number, stateDim: number): number {
    this.ensureNotDisposed();
    if (typeof this.module._gpu_normalize_u32 === 'function') {
      return this.module._gpu_normalize_u32(this.ctxPtr, amplitudesBufferPtr, norm, stateDim >>> 0);
    }
    if (typeof this.module._gpu_normalize !== 'function') return -7;
    return this.module._gpu_normalize(this.ctxPtr, amplitudesBufferPtr, norm, BigInt(stateDim));
  }

  dispose(): void {
    if (this.disposed) return;
    if (typeof this.module._gpu_compute_free === 'function' && this.ctxPtr) {
      this.module._gpu_compute_free(this.ctxPtr);
    }
    this.memory.freeAll();
    this.disposed = true;
  }

  private ensureNotDisposed(): void {
    if (this.disposed) {
      throw new Error('GPUBackendSession has been disposed');
    }
  }
}
