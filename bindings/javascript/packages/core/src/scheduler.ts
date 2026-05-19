/**
 * Distributed-scheduler JS/WASM binding -- since v0.7.1.
 *
 * Wraps `src/distributed/scheduler.{c,h}` (v0.7.0).  Mirrors the
 * Python + Rust bindings: a Job class with fluent builders, an
 * async execute() that runs the worker fan-out, JSON serialisation
 * for over-the-wire dispatch.
 *
 * @example
 * ```typescript
 * import { Job, QgtlGateType } from '@moonlab/quantum-core';
 *
 * const j = await Job.create(2);
 * j.addGate(QgtlGateType.H, 0);
 * j.addGate(QgtlGateType.CNOT, 1, 0);
 * j.setNumShots(1024).setNumWorkers(4).setRngSeed(0xdeadbeefn);
 * const r = await j.execute();
 * console.log(r.outcomes.length);  // 1024
 * j.dispose();
 * ```
 */

import { getModule } from './wasm-loader';
import { GateType } from './qgtl';

export const MOONLAB_SCHED_OK = 0;
export const MOONLAB_SCHED_BAD_ARG = -501;
export const MOONLAB_SCHED_OOM = -502;
export const MOONLAB_SCHED_INTERNAL = -503;
export const MOONLAB_SCHED_BUFFER_TOO_SMALL = -504;

export class SchedulerError extends Error {
  readonly rc: number;
  constructor(message: string, rc: number) {
    super(message);
    this.name = 'SchedulerError';
    this.rc = rc;
  }
}

export interface JobResults {
  numQubits: number;
  totalShots: number;
  outcomes: BigUint64Array;
  numWorkersUsed: number;
  workerSeconds: Float64Array;
}

type SchedulerModule = {
  _moonlab_job_create: (n: number) => number;
  _moonlab_job_free: (h: number) => void;
  _moonlab_job_add_gate: (h: number, type: number, target: number, control: number, params: number) => number;
  _moonlab_job_set_num_shots: (h: number, n: number) => number;
  _moonlab_job_set_num_workers: (h: number, n: number) => number;
  _moonlab_job_set_rng_seed: (h: number, seed_lo: number, seed_hi: number) => number;
  _moonlab_job_num_qubits: (h: number) => number;
  _moonlab_job_num_gates: (h: number) => number;
  _moonlab_job_num_shots: (h: number) => number;
  _moonlab_job_num_workers: (h: number) => number;
  _moonlab_scheduler_run: (h: number, res: number) => number;
  _moonlab_job_results_free: (res: number) => void;
  _moonlab_job_to_json: (h: number, buf: number, bufsize: number) => number;
  _malloc: (n: number) => number;
  _free: (p: number) => void;
  HEAP32: Int32Array;
  HEAPU32: Uint32Array;
  HEAPF64: Float64Array;
  HEAPU8: Uint8Array;
};

function _check(rc: number, ctx: string): void {
  if (rc !== MOONLAB_SCHED_OK) {
    throw new SchedulerError(`${ctx}: rc=${rc}`, rc);
  }
}

export class Job {
  private handle: number;
  private mod: SchedulerModule;
  private freed = false;
  private _n: number;

  private constructor(handle: number, mod: SchedulerModule, n: number) {
    this.handle = handle;
    this.mod = mod;
    this._n = n;
  }

  static async create(numQubits: number): Promise<Job> {
    const mod = (await getModule()) as unknown as SchedulerModule;
    const h = mod._moonlab_job_create(numQubits);
    if (!h) {
      throw new SchedulerError(
        `job_create(${numQubits}): NULL (num_qubits must be in [1, 32])`,
        MOONLAB_SCHED_BAD_ARG,
      );
    }
    return new Job(h, mod, numQubits);
  }

  dispose(): void {
    if (this.freed) return;
    this.mod._moonlab_job_free(this.handle);
    this.freed = true;
  }

  get numQubits(): number { return this._n; }
  get numGates(): number {
    return this.mod._moonlab_job_num_gates(this.handle);
  }
  get numShots(): number {
    return this.mod._moonlab_job_num_shots(this.handle);
  }
  get numWorkers(): number {
    return this.mod._moonlab_job_num_workers(this.handle);
  }

  addGate(type: GateType, target: number, control: number = -1, params: number[] = []): this {
    let paramsPtr = 0;
    if (params.length > 0) {
      paramsPtr = this.mod._malloc(8 * params.length);
      for (let i = 0; i < params.length; i++) {
        this.mod.HEAPF64[(paramsPtr >> 3) + i] = params[i];
      }
    }
    try {
      _check(
        this.mod._moonlab_job_add_gate(this.handle, type, target, control, paramsPtr),
        `addGate(${GateType[type]}, target=${target}, control=${control})`,
      );
    } finally {
      if (paramsPtr) this.mod._free(paramsPtr);
    }
    return this;
  }

  setNumShots(n: number): this {
    _check(this.mod._moonlab_job_set_num_shots(this.handle, n),
           `setNumShots(${n})`);
    return this;
  }

  setNumWorkers(n: number): this {
    _check(this.mod._moonlab_job_set_num_workers(this.handle, n),
           `setNumWorkers(${n})`);
    return this;
  }

  setRngSeed(seed: bigint): this {
    /* Emscripten exposes uint64 args as two i32s when JS interop. */
    const lo = Number(seed & 0xFFFFFFFFn);
    const hi = Number((seed >> 32n) & 0xFFFFFFFFn);
    _check(this.mod._moonlab_job_set_rng_seed(this.handle, lo, hi),
           `setRngSeed`);
    return this;
  }

  async execute(): Promise<JobResults> {
    /* moonlab_job_results_t: { int, int, uint64*, int, double* } -- 24 bytes
     * on wasm32 (int4 + int4 + ptr4 + int4 + ptr4 = 20, padded to 24). */
    const resPtr = this.mod._malloc(24);
    try {
      for (let i = 0; i < 6; i++) this.mod.HEAP32[(resPtr >> 2) + i] = 0;
      _check(this.mod._moonlab_scheduler_run(this.handle, resPtr), 'scheduler_run');
      const numQubits = this.mod.HEAP32[resPtr >> 2];
      const totalShots = this.mod.HEAP32[(resPtr >> 2) + 1];
      const outcomesPtr = this.mod.HEAPU32[(resPtr >> 2) + 2];
      const numWorkersUsed = this.mod.HEAP32[(resPtr >> 2) + 3];
      const workerSecondsPtr = this.mod.HEAPU32[(resPtr >> 2) + 4];

      const outcomes = new BigUint64Array(totalShots);
      for (let i = 0; i < totalShots; i++) {
        const lo = BigInt(this.mod.HEAPU32[(outcomesPtr >> 2) + 2 * i]);
        const hi = BigInt(this.mod.HEAPU32[(outcomesPtr >> 2) + 2 * i + 1]);
        outcomes[i] = lo | (hi << 32n);
      }
      const workerSeconds = new Float64Array(
        this.mod.HEAPF64.subarray(
          workerSecondsPtr >> 3,
          (workerSecondsPtr >> 3) + numWorkersUsed,
        ),
      );

      this.mod._moonlab_job_results_free(resPtr);
      return { numQubits, totalShots, outcomes, numWorkersUsed, workerSeconds };
    } finally {
      this.mod._free(resPtr);
    }
  }

  async toJson(): Promise<string> {
    const needed = this.mod._moonlab_job_to_json(this.handle, 0, 0);
    if (needed < 0) {
      throw new SchedulerError(`to_json size-probe: rc=${needed}`, needed);
    }
    const cap = needed + 1;
    const buf = this.mod._malloc(cap);
    try {
      const written = this.mod._moonlab_job_to_json(this.handle, buf, cap);
      if (written < 0) {
        throw new SchedulerError(`to_json: rc=${written}`, written);
      }
      return new TextDecoder('utf-8').decode(
        this.mod.HEAPU8.subarray(buf, buf + written),
      );
    } finally {
      this.mod._free(buf);
    }
  }
}
