/**
 * QGTL-shaped circuit-ingestion JS/WASM binding -- since v0.6.8.
 *
 * Mirrors the Python (v0.6.8) + Rust (v0.6.8) bindings.  Numerically
 * stable gate-type enum (matches QGTL `gate_type_t`) so codes copied
 * from QGTL examples translate unchanged.
 *
 * The dist/moonlab.wasm in the repo today predates v0.6.6 and lacks
 * the moonlab_qgtl_* symbols; the auto-skip pattern used for the
 * libirrep binding applies here too -- once the WASM is rebuilt the
 * tests pick up the real assertions.
 *
 * @example
 * ```typescript
 * import { QgtlCircuit, GateType } from '@moonlab/quantum-core';
 *
 * const c = await QgtlCircuit.create(2);
 * c.addGate(GateType.H, 0);
 * c.addGate(GateType.CNOT, 1, 0);
 * const r = await c.execute({ returnProbabilities: true });
 * console.log(r.probabilities[0], r.probabilities[3]);  // 0.5, 0.5
 * c.dispose();
 * ```
 */

import { getModule } from './wasm-loader';

/** Numerically matches `moonlab_qgtl_gate_t` in
 *  `src/applications/moonlab_qgtl_backend.h`. */
export enum GateType {
  I    = 0,
  X    = 1,
  Y    = 2,
  Z    = 3,
  H    = 4,
  S    = 5,
  T    = 6,
  RX   = 7,
  RY   = 8,
  RZ   = 9,
  CNOT = 10,
  CY   = 11,
  CZ   = 12,
  SWAP = 13,
}

export const MOONLAB_QGTL_OK = 0;
export const MOONLAB_QGTL_BAD_ARG = -301;
export const MOONLAB_QGTL_OOM = -302;
export const MOONLAB_QGTL_UNSUPPORTED = -303;
export const MOONLAB_QGTL_INTERNAL = -304;

export class QgtlError extends Error {
  readonly rc: number;
  constructor(message: string, rc: number) {
    super(message);
    this.name = 'QgtlError';
    this.rc = rc;
  }
}

export interface QgtlExecOptions {
  numShots?: number;
  rngSeed?: bigint;
  returnProbabilities?: boolean;
}

export interface QgtlResults {
  numQubits: number;
  numShots: number;
  outcomes: BigUint64Array | null;
  probabilities: Float64Array | null;
}

type QgtlModule = {
  _moonlab_qgtl_circuit_create: (n: number) => number;
  _moonlab_qgtl_circuit_free: (h: number) => void;
  _moonlab_qgtl_add_gate: (h: number, type: number, target: number, control: number, params: number) => number;
  _moonlab_qgtl_execute: (h: number, opts: number, out: number) => number;
  _moonlab_qgtl_results_free: (r: number) => void;
  _moonlab_qgtl_circuit_num_qubits: (h: number) => number;
  _moonlab_qgtl_circuit_num_gates: (h: number) => number;
  _malloc: (n: number) => number;
  _free: (p: number) => void;
  HEAP32: Int32Array;
  HEAPU32: Uint32Array;
  HEAPF64: Float64Array;
};

export class QgtlCircuit {
  private handle: number;
  private mod: QgtlModule;
  private freed = false;
  private _n: number;

  private constructor(handle: number, mod: QgtlModule, numQubits: number) {
    this.handle = handle;
    this.mod = mod;
    this._n = numQubits;
  }

  static async create(numQubits: number): Promise<QgtlCircuit> {
    const mod = (await getModule()) as unknown as QgtlModule;
    const h = mod._moonlab_qgtl_circuit_create(numQubits);
    if (!h) {
      throw new QgtlError(
        `circuit_create(${numQubits}) returned NULL (num_qubits must be in [1, 32])`,
        MOONLAB_QGTL_BAD_ARG,
      );
    }
    return new QgtlCircuit(h, mod, numQubits);
  }

  dispose(): void {
    if (this.freed) return;
    this.mod._moonlab_qgtl_circuit_free(this.handle);
    this.freed = true;
  }

  get numQubits(): number { return this._n; }
  get numGates(): number {
    return this.mod._moonlab_qgtl_circuit_num_gates(this.handle);
  }

  addGate(type: GateType, target: number, control: number = -1, params: number[] = []): this {
    let paramsPtr = 0;
    if (params.length > 0) {
      paramsPtr = this.mod._malloc(8 * params.length);
      const view = this.mod.HEAPF64;
      for (let i = 0; i < params.length; i++) {
        view[(paramsPtr >> 3) + i] = params[i];
      }
    }
    try {
      const rc = this.mod._moonlab_qgtl_add_gate(this.handle, type, target, control, paramsPtr);
      if (rc !== 0) {
        throw new QgtlError(
          `addGate(${GateType[type]}, target=${target}, control=${control}): rc=${rc}`,
          rc,
        );
      }
    } finally {
      if (paramsPtr) this.mod._free(paramsPtr);
    }
    return this;
  }

  async execute(opts: QgtlExecOptions = {}): Promise<QgtlResults> {
    const numShots = opts.numShots ?? 0;
    const rngSeed = opts.rngSeed ?? 0n;
    const returnProbs = opts.returnProbabilities ?? false;

    // moonlab_qgtl_exec_options_t: { int, uint64, int } -- 24 bytes with padding.
    const optsPtr = this.mod._malloc(24);
    // moonlab_qgtl_results_t: { int, int, uint64*, double* } -- 24 bytes on wasm32.
    const resPtr = this.mod._malloc(24);
    try {
      this.mod.HEAP32[optsPtr >> 2] = numShots;
      // rng_seed at offset 8 (uint64 alignment). Split into two i32s.
      this.mod.HEAPU32[(optsPtr >> 2) + 2] = Number(rngSeed & 0xFFFFFFFFn);
      this.mod.HEAPU32[(optsPtr >> 2) + 3] = Number((rngSeed >> 32n) & 0xFFFFFFFFn);
      this.mod.HEAP32[(optsPtr >> 2) + 4] = returnProbs ? 1 : 0;

      // Zero the results struct.
      for (let i = 0; i < 6; i++) this.mod.HEAP32[(resPtr >> 2) + i] = 0;

      const rc = this.mod._moonlab_qgtl_execute(this.handle, optsPtr, resPtr);
      if (rc !== 0) throw new QgtlError(`execute: rc=${rc}`, rc);

      const numQubits = this.mod.HEAP32[resPtr >> 2];
      const numShotsOut = this.mod.HEAP32[(resPtr >> 2) + 1];
      const outcomesPtr = this.mod.HEAPU32[(resPtr >> 2) + 2];
      const probsPtr = this.mod.HEAPU32[(resPtr >> 2) + 3];

      let outcomes: BigUint64Array | null = null;
      if (numShotsOut > 0 && outcomesPtr) {
        outcomes = new BigUint64Array(numShotsOut);
        for (let i = 0; i < numShotsOut; i++) {
          const lo = BigInt(this.mod.HEAPU32[(outcomesPtr >> 2) + 2 * i]);
          const hi = BigInt(this.mod.HEAPU32[(outcomesPtr >> 2) + 2 * i + 1]);
          outcomes[i] = lo | (hi << 32n);
        }
      }
      let probabilities: Float64Array | null = null;
      if (returnProbs && probsPtr) {
        const dim = 1 << numQubits;
        probabilities = new Float64Array(
          this.mod.HEAPF64.subarray(probsPtr >> 3, (probsPtr >> 3) + dim),
        );
      }
      this.mod._moonlab_qgtl_results_free(resPtr);
      return { numQubits, numShots: numShotsOut, outcomes, probabilities };
    } finally {
      this.mod._free(optsPtr);
      this.mod._free(resPtr);
    }
  }
}
