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
  _moonlab_job_set_rng_seed: (h: number, seed: bigint) => number;
  _moonlab_job_num_qubits: (h: number) => number;
  _moonlab_job_num_gates: (h: number) => number;
  _moonlab_job_num_shots: (h: number) => number;
  _moonlab_job_num_workers: (h: number) => number;
  _moonlab_scheduler_run: (h: number, res: number) => number;
  _moonlab_job_results_free: (res: number) => void;
  _moonlab_job_to_json: (h: number, buf: number, bufsize: number) => number;
  // Vendor-noise profile registry (since v1.0.3).
  _moonlab_register_vendor_noise_profile?: (name: number, prof: number) => number;
  _moonlab_unregister_vendor_noise_profile?: (name: number) => number;
  _moonlab_lookup_vendor_noise_profile?: (name: number) => number;
  _moonlab_num_vendor_noise_profiles?: () => number;
  _moonlab_list_vendor_noise_profiles?: (out: number, max: number) => number;
  // Scheduler completion hook (since v1.0.3).
  _moonlab_scheduler_set_completion_hook?: (fn: number, ctx: number) => number;
  _malloc: (n: number) => number;
  _free: (p: number) => void;
  HEAP32: Int32Array;
  HEAPU32: Uint32Array;
  HEAPF64: Float64Array;
  HEAPU8: Uint8Array;
  UTF8ToString: (ptr: number) => string;
  stringToUTF8: (s: string, ptr: number, max: number) => void;
  addFunction?: (fn: Function, sig: string) => number;
  removeFunction?: (ptr: number) => void;
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
    /* Emscripten 3.0+ defaults to WASM_BIGINT, so i64 maps to a single
     * BigInt arg.  Pass the seed directly. */
    _check(this.mod._moonlab_job_set_rng_seed(this.handle, seed),
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

// ===================================================================
// Vendor-noise profile runtime registry (since v1.0.3)
// ===================================================================

/** Hardware-noise profile snapshot.  Mirror of the C struct. */
export interface VendorNoiseProfile {
  pGate1q: number;
  pGate2q: number;
  pReadout: number;
  description: string;
}

function jsStringToHeapSched(mod: SchedulerModule, s: string | undefined): number {
  if (!s) return 0;
  const len = new TextEncoder().encode(s).length + 1;
  const ptr = mod._malloc(len);
  mod.stringToUTF8(s, ptr, len);
  return ptr;
}

/** Whether this WASM build exports the v1.0.3 noise-profile registry. */
export async function vendorNoiseProfileRegistryAvailable(): Promise<boolean> {
  const mod = (await getModule()) as unknown as SchedulerModule;
  return typeof mod._moonlab_register_vendor_noise_profile === 'function';
}

export async function registerVendorNoiseProfile(
  name: string,
  profile: VendorNoiseProfile,
): Promise<void> {
  const mod = (await getModule()) as unknown as SchedulerModule;
  if (!mod._moonlab_register_vendor_noise_profile) {
    throw new SchedulerError(
      'WASM build lacks the v1.0.3 vendor-noise profile registry',
      MOONLAB_SCHED_BAD_ARG,
    );
  }
  // moonlab_vendor_noise_profile_t: 3 doubles + 1 char* = 32 bytes.
  const profPtr = mod._malloc(32);
  const namePtr = jsStringToHeapSched(mod, name);
  const descPtr = jsStringToHeapSched(mod, profile.description);
  try {
    mod.HEAPF64[profPtr >> 3]       = profile.pGate1q;
    mod.HEAPF64[(profPtr >> 3) + 1] = profile.pGate2q;
    mod.HEAPF64[(profPtr >> 3) + 2] = profile.pReadout;
    mod.HEAPU32[(profPtr >> 2) + 6] = descPtr;
    const rc = mod._moonlab_register_vendor_noise_profile(namePtr, profPtr);
    if (rc !== MOONLAB_SCHED_OK) {
      throw new SchedulerError(
        `register_vendor_noise_profile(${name}): rc=${rc}`, rc);
    }
  } finally {
    mod._free(profPtr);
    mod._free(namePtr);
    if (descPtr) mod._free(descPtr);
  }
}

export async function unregisterVendorNoiseProfile(name: string): Promise<void> {
  const mod = (await getModule()) as unknown as SchedulerModule;
  if (!mod._moonlab_unregister_vendor_noise_profile) return;
  const namePtr = jsStringToHeapSched(mod, name);
  try {
    const rc = mod._moonlab_unregister_vendor_noise_profile(namePtr);
    if (rc !== MOONLAB_SCHED_OK) {
      throw new SchedulerError(
        `unregister_vendor_noise_profile(${name}): rc=${rc}`, rc);
    }
  } finally {
    mod._free(namePtr);
  }
}

export async function lookupVendorNoiseProfile(
  name: string,
): Promise<VendorNoiseProfile | null> {
  const mod = (await getModule()) as unknown as SchedulerModule;
  if (!mod._moonlab_lookup_vendor_noise_profile) return null;
  const namePtr = jsStringToHeapSched(mod, name);
  try {
    const p = mod._moonlab_lookup_vendor_noise_profile(namePtr);
    if (p === 0) return null;
    const pf64 = p >> 3;
    const descPtr = mod.HEAPU32[(p >> 2) + 6];
    return {
      pGate1q: mod.HEAPF64[pf64],
      pGate2q: mod.HEAPF64[pf64 + 1],
      pReadout: mod.HEAPF64[pf64 + 2],
      description: descPtr ? mod.UTF8ToString(descPtr) : '',
    };
  } finally {
    mod._free(namePtr);
  }
}

export async function listVendorNoiseProfiles(): Promise<string[]> {
  const mod = (await getModule()) as unknown as SchedulerModule;
  if (!mod._moonlab_num_vendor_noise_profiles ||
      !mod._moonlab_list_vendor_noise_profiles) return [];
  const n = mod._moonlab_num_vendor_noise_profiles();
  if (n <= 0) return [];
  const buf = mod._malloc(n * 4);
  try {
    const written = mod._moonlab_list_vendor_noise_profiles(buf, n);
    const names: string[] = [];
    for (let i = 0; i < written; i++) {
      const ptr = mod.HEAPU32[(buf >> 2) + i];
      if (ptr) names.push(mod.UTF8ToString(ptr));
    }
    return names;
  } finally {
    mod._free(buf);
  }
}

// ===================================================================
// Scheduler completion hook (since v1.0.3)
// ===================================================================

/** Subset of the job-results struct surfaced to the JS callback. */
export interface CompletionInfo {
  numQubits: number;
  totalShots: number;
  backendName: string | null;
}

export type CompletionCallback = (info: CompletionInfo) => void;

// One-slot global state mirroring the C runtime's single hook.
let activeHookFnPtr: number | null = null;
let activeCallback: CompletionCallback | null = null;

/** Install a JS callback that fires after every successful
 *  scheduler.run().  Failed runs do not fire the hook.  Pass `null`
 *  to clear.  Since v1.0.3. */
export async function setCompletionHook(
  callback: CompletionCallback | null,
): Promise<void> {
  const mod = (await getModule()) as unknown as SchedulerModule;
  if (!mod._moonlab_scheduler_set_completion_hook) {
    throw new SchedulerError(
      'WASM build lacks the v1.0.3 scheduler completion hook',
      MOONLAB_SCHED_BAD_ARG,
    );
  }
  // Always tear down any prior installation first.
  if (activeHookFnPtr !== null) {
    mod.removeFunction?.(activeHookFnPtr);
    activeHookFnPtr = null;
    activeCallback = null;
  }
  if (callback === null) {
    const rc = mod._moonlab_scheduler_set_completion_hook(0, 0);
    if (rc !== MOONLAB_SCHED_OK) {
      throw new SchedulerError(`clear_completion_hook: rc=${rc}`, rc);
    }
    return;
  }
  if (!mod.addFunction) {
    throw new SchedulerError(
      'WASM build lacks addFunction; rebuild with ALLOW_TABLE_GROWTH=1',
      MOONLAB_SCHED_BAD_ARG,
    );
  }
  // C signature: void(*)(const job*, const results*, const char*, void*).
  // emscripten sig 'viiii' (void, 4 i32 args; pointers + ctx).
  const trampoline = (
    _jobPtr: number, resultsPtr: number, backendNamePtr: number, _ctx: number,
  ) => {
    try {
      if (!activeCallback || resultsPtr === 0) return;
      // moonlab_job_results_t: { num_qubits, total_shots, outcomes*,
      // num_workers_used, worker_seconds* }.  We only read the int fields.
      const r32 = resultsPtr >> 2;
      const info: CompletionInfo = {
        numQubits: mod.HEAP32[r32],
        totalShots: mod.HEAP32[r32 + 1],
        backendName: backendNamePtr ? mod.UTF8ToString(backendNamePtr) : null,
      };
      activeCallback(info);
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error('JS completion hook threw:', e);
    }
  };
  activeHookFnPtr = mod.addFunction(trampoline, 'viiii');
  activeCallback = callback;
  const rc = mod._moonlab_scheduler_set_completion_hook(activeHookFnPtr, 0);
  if (rc !== MOONLAB_SCHED_OK) {
    mod.removeFunction?.(activeHookFnPtr);
    activeHookFnPtr = null;
    activeCallback = null;
    throw new SchedulerError(`set_completion_hook: rc=${rc}`, rc);
  }
}

export async function clearCompletionHook(): Promise<void> {
  return setCompletionHook(null);
}
