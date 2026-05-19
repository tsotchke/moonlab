/**
 * Single-qubit gate-fusion DAG binding (C-side since v0.2.0, JS
 * binding since v0.4.7).
 *
 * Wraps ``src/optimization/fusion/fusion.{c,h}`` so JavaScript code
 * can build a symbolic circuit, run the run-length single-qubit
 * fuser over it, and execute the fused schedule on a
 * {@link QuantumState}.  Mirrors the Python ``moonlab.fusion``
 * (v0.4.4) and Rust ``moonlab::fusion`` (v0.4.6) surfaces.
 *
 * The fuser collapses adjacent single-qubit gates on the same qubit
 * into one 2x2 matrix product, dropping repeated full-state passes.
 * On a five-layer hardware-efficient ansatz at n = 16 the fused
 * execution is roughly 2.2x faster than the unfused dispatch; the
 * full performance discussion lives in
 * ``src/optimization/fusion/fusion.h``.
 *
 * @example
 * ```typescript
 * import { QuantumState, FusedCircuit } from '@moonlab/quantum-core';
 *
 * const circuit = await FusedCircuit.create(4);
 * circuit.h(0).rz(0, 0.3).rx(0, 0.7).cnot(0, 1).rz(1, 0.4);
 * const { fused, stats } = circuit.compile();
 * console.log(stats);
 *
 * const state = await QuantumState.create({ numQubits: 4 });
 * fused.execute(state);
 *
 * fused.dispose();
 * circuit.dispose();
 * state.dispose();
 * ```
 *
 * References:
 *  - T. Haener and D. S. Steiger, "0.5 Petabyte Simulation of a
 *    45-Qubit Quantum Circuit", SC17 (2017), arXiv:1704.01127.
 *  - Y. Suzuki et al., "Qulacs: a fast and versatile quantum circuit
 *    simulator for research purpose", Quantum 5, 559 (2021).
 */

import { QuantumState } from './quantum-state';
import { getModule } from './wasm-loader';

/** Diagnostic counts from {@link FusedCircuit.compile}.  Mirrors
 *  ``fuse_stats_t`` in ``src/optimization/fusion/fusion.h``. */
export interface FuseStats {
  /** Symbol-list length of the input circuit. */
  originalGates: number;
  /** Symbol-list length of the fused output. */
  fusedGates: number;
  /** Number of pair-wise 2x2 multiplications the fuser performed;
   *  equals (run-length - 1) summed across every fused run.  Zero
   *  when no single-qubit gate ran into a same-qubit successor
   *  before a multi-qubit barrier. */
  mergesApplied: number;
}

/** Tuple returned by {@link FusedCircuit.compile}. */
export interface FuseCompileResult {
  /** Owned fused-circuit handle; the caller must `dispose()` it. */
  fused: FusedCircuit;
  /** Summary statistics from the fuser pass. */
  stats: FuseStats;
}

type FusionModule = {
  _fuse_circuit_create: (n: number) => number;
  _fuse_circuit_free: (h: number) => void;
  _fuse_circuit_len: (h: number) => number;
  _fuse_circuit_num_qubits: (h: number) => number;
  _fuse_compile: (h: number, statsPtr: number) => number;
  _fuse_execute: (h: number, statePtr: number) => number;
  _malloc: (n: number) => number;
  _free: (p: number) => void;
  HEAPU32: Uint32Array;
} & Record<string, (...args: number[]) => number>;

/**
 * Owned WASM handle to a fusion-DAG circuit.
 *
 * Build the circuit with the fluent gate-append methods, then call
 * {@link compile} to produce a fused circuit and {@link execute} to
 * apply it to a {@link QuantumState}.  Call {@link dispose} to
 * release the underlying C handle.
 */
export class FusedCircuit {
  private handle: number;
  private mod: FusionModule;
  private nq: number;
  private freed = false;

  private constructor(handle: number, mod: FusionModule, numQubits: number) {
    this.handle = handle;
    this.mod = mod;
    this.nq = numQubits;
  }

  /** Allocate an empty fusion circuit on ``numQubits`` qubits. */
  static async create(numQubits: number): Promise<FusedCircuit> {
    if (numQubits < 1) {
      throw new RangeError(`numQubits must be >= 1, got ${numQubits}`);
    }
    const mod = (await getModule()) as unknown as FusionModule;
    const h = mod._fuse_circuit_create(numQubits);
    if (!h) throw new Error('fuse_circuit_create returned NULL');
    return new FusedCircuit(h, mod, numQubits);
  }

  /** Release the underlying WASM allocation. */
  dispose(): void {
    if (this.freed) return;
    this.mod._fuse_circuit_free(this.handle);
    this.freed = true;
  }

  /** Number of qubits the circuit was created for. */
  get numQubits(): number { return this.nq; }

  /** Number of symbolic gates currently in the circuit. */
  get length(): number {
    return this.mod._fuse_circuit_len(this.handle);
  }

  // ---- Single-qubit gates (fluent) --------------------------------------
  h(q: number): this   { this._call1('fuse_append_h', q);   return this; }
  x(q: number): this   { this._call1('fuse_append_x', q);   return this; }
  y(q: number): this   { this._call1('fuse_append_y', q);   return this; }
  z(q: number): this   { this._call1('fuse_append_z', q);   return this; }
  s(q: number): this   { this._call1('fuse_append_s', q);   return this; }
  sdg(q: number): this { this._call1('fuse_append_sdg', q); return this; }
  t(q: number): this   { this._call1('fuse_append_t', q);   return this; }
  tdg(q: number): this { this._call1('fuse_append_tdg', q); return this; }

  phase(q: number, theta: number): this {
    this._call1p('fuse_append_phase', q, theta); return this;
  }
  rx(q: number, theta: number): this {
    this._call1p('fuse_append_rx', q, theta); return this;
  }
  ry(q: number, theta: number): this {
    this._call1p('fuse_append_ry', q, theta); return this;
  }
  rz(q: number, theta: number): this {
    this._call1p('fuse_append_rz', q, theta); return this;
  }

  u3(q: number, theta: number, phi: number, lambda: number): this {
    this._guard1(q);
    const rc = this.mod._fuse_append_u3(this.handle, q, theta, phi, lambda);
    if (rc !== 0) throw new Error(`fuse_append_u3 rc=${rc}`);
    return this;
  }

  // ---- Two-qubit gates (fluent) -----------------------------------------
  cnot(ctrl: number, tgt: number): this {
    this._call2('fuse_append_cnot', ctrl, tgt); return this;
  }
  cz(ctrl: number, tgt: number): this {
    this._call2('fuse_append_cz', ctrl, tgt); return this;
  }
  cy(ctrl: number, tgt: number): this {
    this._call2('fuse_append_cy', ctrl, tgt); return this;
  }
  swap(a: number, b: number): this {
    this._call2('fuse_append_swap', a, b); return this;
  }

  cphase(ctrl: number, tgt: number, theta: number): this {
    this._call2p('fuse_append_cphase', ctrl, tgt, theta); return this;
  }
  crx(ctrl: number, tgt: number, theta: number): this {
    this._call2p('fuse_append_crx', ctrl, tgt, theta); return this;
  }
  cry(ctrl: number, tgt: number, theta: number): this {
    this._call2p('fuse_append_cry', ctrl, tgt, theta); return this;
  }
  crz(ctrl: number, tgt: number, theta: number): this {
    this._call2p('fuse_append_crz', ctrl, tgt, theta); return this;
  }

  // ---- Compile + execute -------------------------------------------------

  /** Run the run-length single-qubit fuser.  Returns an owned new
   *  {@link FusedCircuit} (caller must `dispose()`) plus a
   *  {@link FuseStats} summary. */
  compile(): FuseCompileResult {
    // fuse_stats_t = three size_t fields = 12 bytes on WASM32.
    const STATS_BYTES = 12;
    const statsPtr = this.mod._malloc(STATS_BYTES);
    try {
      const h = this.mod._fuse_compile(this.handle, statsPtr);
      if (!h) throw new Error('fuse_compile returned NULL');
      const idx = statsPtr >> 2;
      const stats: FuseStats = {
        originalGates: this.mod.HEAPU32[idx],
        fusedGates:    this.mod.HEAPU32[idx + 1],
        mergesApplied: this.mod.HEAPU32[idx + 2],
      };
      return { fused: new FusedCircuit(h, this.mod, this.nq), stats };
    } finally {
      this.mod._free(statsPtr);
    }
  }

  /** Apply this circuit to ``state`` in place.  Works on both fused
   *  and unfused circuits. */
  execute(state: QuantumState): void {
    const statePtr = state._internal_state_pointer();
    const rc = this.mod._fuse_execute(this.handle, statePtr);
    if (rc !== 0) throw new Error(`fuse_execute rc=${rc}`);
  }

  // ---- Internal helpers --------------------------------------------------

  private _guard1(q: number): void {
    if (this.freed) throw new Error('FusedCircuit is disposed');
    if (q < 0 || q >= this.nq) {
      throw new RangeError(`qubit ${q} out of [0, ${this.nq})`);
    }
  }

  private _guard2(a: number, b: number): void {
    if (this.freed) throw new Error('FusedCircuit is disposed');
    if (a < 0 || a >= this.nq || b < 0 || b >= this.nq) {
      throw new RangeError(`qubits ${a},${b} out of [0, ${this.nq})`);
    }
  }

  private _call1(fnName: string, q: number): void {
    this._guard1(q);
    const rc = this.mod[`_${fnName}`](this.handle, q);
    if (rc !== 0) throw new Error(`${fnName}(${q}) rc=${rc}`);
  }

  private _call1p(fnName: string, q: number, theta: number): void {
    this._guard1(q);
    const rc = this.mod[`_${fnName}`](this.handle, q, theta);
    if (rc !== 0) throw new Error(`${fnName}(${q},${theta}) rc=${rc}`);
  }

  private _call2(fnName: string, a: number, b: number): void {
    this._guard2(a, b);
    const rc = this.mod[`_${fnName}`](this.handle, a, b);
    if (rc !== 0) throw new Error(`${fnName}(${a},${b}) rc=${rc}`);
  }

  private _call2p(fnName: string, a: number, b: number, theta: number): void {
    this._guard2(a, b);
    const rc = this.mod[`_${fnName}`](this.handle, a, b, theta);
    if (rc !== 0) throw new Error(`${fnName}(${a},${b},${theta}) rc=${rc}`);
  }
}
