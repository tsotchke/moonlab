/**
 * Tensor Network (MPS) bindings for Moonlab WASM
 *
 * Provides a minimal JS/TS surface over the tensor-network solvers
 * (MPS state, gates, measurement) and a convenience DMRG ground-state
 * solver for the transverse-field Ising model.
 *
 * These wrappers stay close to the C API: pointers are managed and freed,
 * and amplitudes/statevectors are only materialized on demand for small
 * systems to avoid blowing up memory.
 */

import type { Complex } from './complex';
import { WasmMemory, type MoonlabModule } from './memory';
import { getModule } from './wasm-loader';

/**
 * Options for creating a tensor-network MPS state
 */
export interface TensorNetworkOptions {
  numQubits: number;
}

/**
 * Result from a DMRG ground-state solve (TFIM convenience)
 */
export interface DMRGResult {
  state: TensorNetworkState;
  energy?: number;
  energyVariance?: number;
  numSweeps?: number;
  converged?: boolean;
}

/**
 * TensorNetworkState wraps `tn_mps_state_t *`
 */
export class TensorNetworkState {
  private module: MoonlabModule;
  private memory: WasmMemory;
  private ptr: number;
  private _numQubits: number;
  private disposed = false;

  private constructor(module: MoonlabModule, memory: WasmMemory, ptr: number, numQubits: number) {
    this.module = module;
    this.memory = memory;
    this.ptr = ptr;
    this._numQubits = numQubits;
  }

  /**
   * Create from an existing native pointer (internal use).
   */
  static fromPointer(
    module: MoonlabModule,
    memory: WasmMemory,
    ptr: number,
    numQubits: number
  ): TensorNetworkState {
    return new TensorNetworkState(module, memory, ptr, numQubits);
  }

  /**
   * Create |00..0> tensor-network state
   */
  static async create(options: TensorNetworkOptions): Promise<TensorNetworkState> {
    if (options.numQubits < 1) {
      throw new Error('numQubits must be >= 1');
    }
    const module = await getModule();
    const memory = new WasmMemory(module);
    const ptr = module._tn_mps_create_zero(options.numQubits, 0);
    if (!ptr) {
      throw new Error('tn_mps_create_zero failed');
    }
    return new TensorNetworkState(module, memory, ptr, options.numQubits);
  }

  /**
   * Create from computational basis integer
   */
  static async fromBasis(numQubits: number, basisState: number): Promise<TensorNetworkState> {
    const module = await getModule();
    const memory = new WasmMemory(module);
    const ptr = module._tn_mps_create_basis(numQubits, basisState, 0);
    if (!ptr) {
      throw new Error('tn_mps_create_basis failed');
    }
    return new TensorNetworkState(module, memory, ptr, numQubits);
  }

  /**
   * Number of qubits
   */
  get numQubits(): number {
    return this._numQubits;
  }

  /**
   * Dispose the underlying MPS
   */
  dispose(): void {
    if (!this.disposed && this.ptr) {
      this.module._tn_mps_free(this.ptr);
      this.memory.freeAll();
      this.disposed = true;
    }
  }

  // =========================================================================
  // Gates (subset)
  // =========================================================================

  h(qubit: number): this {
    this.checkQubit(qubit);
    this.module._tn_apply_h(this.ptr, qubit);
    return this;
  }

  x(qubit: number): this {
    this.checkQubit(qubit);
    this.module._tn_apply_x(this.ptr, qubit);
    return this;
  }

  y(qubit: number): this {
    this.checkQubit(qubit);
    this.module._tn_apply_y(this.ptr, qubit);
    return this;
  }

  z(qubit: number): this {
    this.checkQubit(qubit);
    this.module._tn_apply_z(this.ptr, qubit);
    return this;
  }

  rx(qubit: number, theta: number): this {
    this.checkQubit(qubit);
    this.module._tn_apply_rx(this.ptr, qubit, theta);
    return this;
  }

  ry(qubit: number, theta: number): this {
    this.checkQubit(qubit);
    this.module._tn_apply_ry(this.ptr, qubit, theta);
    return this;
  }

  rz(qubit: number, theta: number): this {
    this.checkQubit(qubit);
    this.module._tn_apply_rz(this.ptr, qubit, theta);
    return this;
  }

  cnot(control: number, target: number): this {
    this.checkQubit(control);
    this.checkQubit(target);
    this.module._tn_apply_cnot(this.ptr, control, target);
    return this;
  }

  cz(control: number, target: number): this {
    this.checkQubit(control);
    this.checkQubit(target);
    this.module._tn_apply_cz(this.ptr, control, target);
    return this;
  }

  swap(q1: number, q2: number): this {
    this.checkQubit(q1);
    this.checkQubit(q2);
    this.module._tn_apply_swap(this.ptr, q1, q2);
    return this;
  }

  rzz(q1: number, q2: number, theta: number): this {
    this.checkQubit(q1);
    this.checkQubit(q2);
    this.module._tn_apply_rzz(this.ptr, q1, q2, theta);
    return this;
  }

  /**
   * Normalize MPS
   */
  normalize(): this {
    this.module._tn_mps_normalize(this.ptr);
    return this;
  }

  // =========================================================================
  // Measurement / inspection
  // =========================================================================

  /**
   * Probability of bitstring (returns exact value without collapse)
   */
  bitstringProbability(bitstring: number): number {
    return this.module._tn_measure_bitstring_probability(this.ptr, bitstring);
  }

  /**
   * Convert to full statevector (only feasible for small n)
   */
  toStateVector(maxQubits: number = 24): Complex[] {
    if (this._numQubits > maxQubits) {
      throw new Error(`Refusing to materialize statevector for ${this._numQubits} qubits (max ${maxQubits})`);
    }
    const dim = 1 << this._numQubits;
    const outPtr = this.memory.allocComplexArray(dim);
    const rc = this.module._tn_mps_to_statevector(this.ptr, outPtr);
    if (rc !== 0) {
      this.memory.free(outPtr);
      throw new Error(`tn_mps_to_statevector failed with code ${rc}`);
    }
    const amps = this.memory.readComplexArray(outPtr, dim);
    this.memory.free(outPtr);
    return amps;
  }

  // =========================================================================
  // Internals
  // =========================================================================

  /** WASM pointer (internal use) */
  get pointer(): number {
    return this.ptr;
  }

  private checkQubit(q: number): void {
    if (q < 0 || q >= this._numQubits) {
      throw new Error(`Invalid qubit index ${q}`);
    }
  }
}

// ===========================================================================
// DMRG convenience: TFIM ground state
// ===========================================================================

export interface TFIMOptions {
  numSites: number;
  g: number; // transverse field ratio h/J
  maxStatevectorQubits?: number;
}

/**
 * Solve TFIM ground state with DMRG and return an MPS handle.
 * This exercises the WASM solvers to produce an actual ground state
 * (discrete Schrödinger solver on a 1D spin chain).
 */
export async function dmrgTFIMGroundState(options: TFIMOptions): Promise<DMRGResult> {
  const module = await getModule();
  const memory = new WasmMemory(module);

  if (options.numSites < 2) {
    throw new Error('numSites must be >= 2');
  }

  // Allocate space for dmrg_result_t* out-param
  const resultPtrPtr = memory.alloc(4);
  memory.writeInt32(resultPtrPtr, 0);

  const mpsPtr = module._dmrg_tfim_ground_state(options.numSites, options.g, 0, resultPtrPtr);
  if (!mpsPtr) {
    memory.free(resultPtrPtr);
    throw new Error('dmrg_tfim_ground_state failed (null MPS pointer)');
  }

  const tnState = TensorNetworkState.fromPointer(module, memory, mpsPtr, options.numSites);

  // Parse a small subset of dmrg_result_t fields (best-effort)
  const resultStructPtr = memory.readPointer(resultPtrPtr);
  let energy: number | undefined;
  let variance: number | undefined;
  let numSweeps: number | undefined;
  let converged: boolean | undefined;

  if (resultStructPtr) {
    const base = resultStructPtr / 8; // HEAPF64 index for doubles
    energy = module.HEAPF64[base + 0];
    variance = module.HEAPF64[base + 1];
    // num_sweeps is uint32 at offset 16 bytes -> HEAP32 index
    numSweeps = module.HEAP32[(resultStructPtr + 16) >> 2];
    // converged is a bool; stored just after total_time (double at offset 24)
    const convergedByte = module.HEAP8[resultStructPtr + 32];
    converged = !!convergedByte;
    module._dmrg_result_free(resultStructPtr);
  }

  memory.free(resultPtrPtr);

  return {
    state: tnState,
    energy,
    energyVariance: variance,
    numSweeps,
    converged,
  };
}
