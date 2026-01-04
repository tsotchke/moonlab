/**
 * WASM Memory Management
 *
 * Handles allocation, deallocation, and data transfer between
 * JavaScript and WebAssembly linear memory.
 */

import type { Complex } from './complex';

/**
 * Memory manager for WASM allocations
 */
export class WasmMemory {
  private module: MoonlabModule;
  private allocations: Map<number, { size: number; type: string }> = new Map();

  constructor(module: MoonlabModule) {
    this.module = module;
  }

  /**
   * Allocate raw bytes
   */
  alloc(bytes: number): number {
    const ptr = this.module._malloc(bytes);
    if (ptr === 0) {
      throw new Error(`WASM memory allocation failed: ${bytes} bytes`);
    }
    this.allocations.set(ptr, { size: bytes, type: 'raw' });
    return ptr;
  }

  /**
   * Allocate memory for complex array (interleaved real/imag)
   * Each complex number is 16 bytes (2 × Float64)
   */
  allocComplexArray(length: number): number {
    const bytes = length * 16;
    const ptr = this.alloc(bytes);
    this.allocations.set(ptr, { size: bytes, type: 'complex[]' });
    return ptr;
  }

  /**
   * Allocate memory for Float64 array
   */
  allocFloat64Array(length: number): number {
    const bytes = length * 8;
    const ptr = this.alloc(bytes);
    this.allocations.set(ptr, { size: bytes, type: 'f64[]' });
    return ptr;
  }

  /**
   * Allocate memory for Int32 array
   */
  allocInt32Array(length: number): number {
    const bytes = length * 4;
    const ptr = this.alloc(bytes);
    this.allocations.set(ptr, { size: bytes, type: 'i32[]' });
    return ptr;
  }

  /**
   * Free allocated memory
   */
  free(ptr: number): void {
    if (this.allocations.has(ptr)) {
      this.module._free(ptr);
      this.allocations.delete(ptr);
    }
  }

  /**
   * Free all tracked allocations
   */
  freeAll(): void {
    for (const ptr of this.allocations.keys()) {
      this.module._free(ptr);
    }
    this.allocations.clear();
  }

  /**
   * Get allocation info for debugging
   */
  getAllocations(): Map<number, { size: number; type: string }> {
    return new Map(this.allocations);
  }

  /**
   * Read complex array from WASM memory
   */
  readComplexArray(ptr: number, length: number): Complex[] {
    const result: Complex[] = [];
    const heap = this.module.HEAPF64;
    const offset = ptr / 8;

    for (let i = 0; i < length; i++) {
      result.push({
        real: heap[offset + i * 2],
        imag: heap[offset + i * 2 + 1],
      });
    }
    return result;
  }

  /**
   * Write complex array to WASM memory
   */
  writeComplexArray(ptr: number, data: Complex[]): void {
    const heap = this.module.HEAPF64;
    const offset = ptr / 8;

    for (let i = 0; i < data.length; i++) {
      heap[offset + i * 2] = data[i].real;
      heap[offset + i * 2 + 1] = data[i].imag;
    }
  }

  /**
   * Read Float64 array from WASM memory (returns a copy)
   */
  readFloat64Array(ptr: number, length: number): Float64Array {
    const offset = ptr / 8;
    return new Float64Array(this.module.HEAPF64.buffer, offset * 8, length).slice();
  }

  /**
   * Write Float64 array to WASM memory
   */
  writeFloat64Array(ptr: number, data: Float64Array | number[]): void {
    const heap = this.module.HEAPF64;
    const offset = ptr / 8;

    for (let i = 0; i < data.length; i++) {
      heap[offset + i] = data[i];
    }
  }

  /**
   * Read Int32 array from WASM memory
   */
  readInt32Array(ptr: number, length: number): Int32Array {
    const offset = ptr / 4;
    return new Int32Array(this.module.HEAP32.buffer, offset * 4, length).slice();
  }

  /**
   * Write Int32 array to WASM memory
   */
  writeInt32Array(ptr: number, data: Int32Array | number[]): void {
    const heap = this.module.HEAP32;
    const offset = ptr / 4;

    for (let i = 0; i < data.length; i++) {
      heap[offset + i] = data[i];
    }
  }

  /**
   * Read a single Float64 value
   */
  readFloat64(ptr: number): number {
    return this.module.HEAPF64[ptr / 8];
  }

  /**
   * Write a single Float64 value
   */
  writeFloat64(ptr: number, value: number): void {
    this.module.HEAPF64[ptr / 8] = value;
  }

  /**
   * Read a single Int32 value
   */
  readInt32(ptr: number): number {
    return this.module.HEAP32[ptr / 4];
  }

  /**
   * Write a single Int32 value
   */
  writeInt32(ptr: number, value: number): void {
    this.module.HEAP32[ptr / 4] = value;
  }

  /**
   * Read a pointer value (depends on pointer size)
   */
  readPointer(ptr: number): number {
    // Assuming 32-bit pointers in WASM
    return this.module.HEAP32[ptr / 4];
  }

  /**
   * Copy bytes within WASM memory
   */
  copy(dest: number, src: number, bytes: number): void {
    const destArr = new Uint8Array(this.module.HEAP8.buffer, dest, bytes);
    const srcArr = new Uint8Array(this.module.HEAP8.buffer, src, bytes);
    destArr.set(srcArr);
  }

  /**
   * Zero out memory region
   */
  zero(ptr: number, bytes: number): void {
    const arr = new Uint8Array(this.module.HEAP8.buffer, ptr, bytes);
    arr.fill(0);
  }
}

/**
 * WASM Module type definition
 */
export interface MoonlabModule {
  // Memory
  HEAP8: Int8Array;
  HEAPU8: Uint8Array;
  HEAP16: Int16Array;
  HEAPU16: Uint16Array;
  HEAP32: Int32Array;
  HEAPU32: Uint32Array;
  HEAPF32: Float32Array;
  HEAPF64: Float64Array;

  // Memory management
  _malloc(size: number): number;
  _free(ptr: number): void;

  // State management
  _quantum_state_init(statePtr: number, numQubits: number): number;
  _quantum_state_free(statePtr: number): void;
  _quantum_state_reset(statePtr: number): void;
  _quantum_state_clone(destPtr: number, srcPtr: number): number;
  _quantum_state_normalize(statePtr: number): number;
  _quantum_state_get_probability(statePtr: number, basisState: number): number;
  _quantum_state_entropy(statePtr: number): number;
  _quantum_state_purity(statePtr: number): number;
  _quantum_state_fidelity(state1Ptr: number, state2Ptr: number): number;

  // Single-qubit gates
  _gate_hadamard(statePtr: number, qubit: number): number;
  _gate_pauli_x(statePtr: number, qubit: number): number;
  _gate_pauli_y(statePtr: number, qubit: number): number;
  _gate_pauli_z(statePtr: number, qubit: number): number;
  _gate_s(statePtr: number, qubit: number): number;
  _gate_s_dagger(statePtr: number, qubit: number): number;
  _gate_t(statePtr: number, qubit: number): number;
  _gate_t_dagger(statePtr: number, qubit: number): number;
  _gate_rx(statePtr: number, qubit: number, angle: number): number;
  _gate_ry(statePtr: number, qubit: number, angle: number): number;
  _gate_rz(statePtr: number, qubit: number, angle: number): number;
  _gate_phase(statePtr: number, qubit: number, angle: number): number;
  _gate_u3(
    statePtr: number,
    qubit: number,
    theta: number,
    phi: number,
    lambda: number
  ): number;

  // Two-qubit gates
  _gate_cnot(statePtr: number, control: number, target: number): number;
  _gate_cz(statePtr: number, control: number, target: number): number;
  _gate_cy(statePtr: number, control: number, target: number): number;
  _gate_swap(statePtr: number, qubit1: number, qubit2: number): number;
  _gate_cphase(
    statePtr: number,
    control: number,
    target: number,
    angle: number
  ): number;
  _gate_crx(
    statePtr: number,
    control: number,
    target: number,
    angle: number
  ): number;
  _gate_cry(
    statePtr: number,
    control: number,
    target: number,
    angle: number
  ): number;
  _gate_crz(
    statePtr: number,
    control: number,
    target: number,
    angle: number
  ): number;

  // Multi-qubit gates
  _gate_toffoli(
    statePtr: number,
    control1: number,
    control2: number,
    target: number
  ): number;
  _gate_fredkin(
    statePtr: number,
    control: number,
    target1: number,
    target2: number
  ): number;
  _gate_qft(statePtr: number, qubitsPtr: number, numQubits: number): number;
  _gate_iqft(statePtr: number, qubitsPtr: number, numQubits: number): number;

  // Measurement
  _measurement_probability_one(statePtr: number, qubit: number): number;
  _measurement_probability_zero(statePtr: number, qubit: number): number;
  _measurement_all_probabilities(statePtr: number, probsPtr: number): void;
  _measurement_probability_distribution(statePtr: number, distPtr: number): void;
  _measurement_single_qubit(
    statePtr: number,
    qubit: number,
    randomValue: number
  ): number;
  _measurement_all_qubits(statePtr: number, randomValue: number): number;
  _measurement_expectation_z(statePtr: number, qubit: number): number;
  _measurement_expectation_x(statePtr: number, qubit: number): number;
  _measurement_expectation_y(statePtr: number, qubit: number): number;
  _measurement_correlation_zz(
    statePtr: number,
    qubit1: number,
    qubit2: number
  ): number;

  // Grover
  _grover_search(statePtr: number, configPtr: number, entropyPtr: number): number;
  _grover_optimal_iterations(numQubits: number): number;
  _grover_oracle(statePtr: number, markedState: number): number;
  _grover_diffusion(statePtr: number): number;
  _grover_iteration(statePtr: number, markedState: number): number;

  // Bell states
  _create_bell_state(
    statePtr: number,
    qubit1: number,
    qubit2: number,
    type: number
  ): number;
  _create_bell_state_phi_plus(
    statePtr: number,
    qubit1: number,
    qubit2: number
  ): number;
  _create_bell_state_phi_minus(
    statePtr: number,
    qubit1: number,
    qubit2: number
  ): number;
  _create_bell_state_psi_plus(
    statePtr: number,
    qubit1: number,
    qubit2: number
  ): number;
  _create_bell_state_psi_minus(
    statePtr: number,
    qubit1: number,
    qubit2: number
  ): number;

  // Ready promise
  ready: Promise<MoonlabModule>;

  // Helpers from post.js
  getSecureRandom(): number;
  getSecureRandomBytes(count: number): Uint8Array;
  readComplexArray(ptr: number, length: number): Complex[];
  writeComplexArray(ptr: number, data: Complex[]): void;
  readFloat64Array(ptr: number, length: number): Float64Array;
  allocComplexArray(length: number): number;
  allocFloat64Array(length: number): number;

  // Version
  version: { core: string; wasm: boolean };
}
