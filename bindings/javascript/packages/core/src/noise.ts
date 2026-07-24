/**
 * Noise channel bindings.
 *
 * Apply realistic decoherence channels to a {@link QuantumState}:
 * depolarising, amplitude-damping, phase-damping, pure-dephasing,
 * bit-flip, phase-flip, bit-phase-flip, thermal relaxation, plus
 * classical readout error and the composite per-device noise model
 * (`noise_model_t` + `noise_apply_model`).
 *
 * All channels are Monte-Carlo trajectory unravellings of the
 * Kraus-operator CPTP map: each call realises ONE Kraus branch selected
 * by a uniform-[0, 1) random draw, exactly as the C engine defines it
 * (`src/quantum/noise.{c,h}`).  Averaging the reduced density operator
 * over many trajectories recovers the channel.  The state is mutated in
 * place; norm is preserved up to floating-point precision.
 *
 * Every C channel takes the random draw(s) as trailing argument(s).
 * The JS layer supplies them: by default from a crypto-backed uniform
 * generator, or from the optional trailing parameter so tests and
 * reproducible pipelines can inject their own stream.  Mirrors Python
 * `moonlab.noise` (seedable numpy Generator) and Rust `moonlab::noise`
 * (caller-supplied `random_value`).
 *
 * WASM exports declared in
 * `bindings/javascript/packages/core/emscripten/exports.txt`.
 *
 * @since v0.2.1 (channel surface); v1.2.0 (correct random-draw
 *   plumbing, readout error, DeviceNoiseModel)
 */

import type { QuantumState } from './quantum-state';
import { getModule } from './wasm-loader';

interface NoiseModule {
  _noise_depolarizing_single: (s: number, q: number, p: number, rv: number) => void;
  _noise_depolarizing_two_qubit: (s: number, q1: number, q2: number, p: number, rv: number) => void;
  _noise_amplitude_damping: (s: number, q: number, gamma: number, rv: number) => void;
  _noise_phase_damping: (s: number, q: number, gamma: number, rv: number) => void;
  _noise_pure_dephasing: (s: number, q: number, sigma: number, randomPhase: number) => void;
  _noise_bit_flip: (s: number, q: number, p: number, rv: number) => void;
  _noise_phase_flip: (s: number, q: number, p: number, rv: number) => void;
  _noise_bit_phase_flip: (s: number, q: number, p: number, rv: number) => void;
  _noise_thermal_relaxation: (
    s: number, q: number, t1: number, t2: number, t: number, rvsPtr: number,
  ) => void;
  _noise_readout_error: (outcome: number, e01: number, e10: number, rv: number) => number;
  _noise_model_create: () => number;
  _noise_model_destroy: (m: number) => void;
  _noise_model_create_realistic: (
    t1Us: number, t2Us: number, gateError: number, readoutError: number,
  ) => number;
  _noise_model_set_depolarizing: (m: number, rate: number) => void;
  _noise_model_set_amplitude_damping: (m: number, rate: number) => void;
  _noise_model_set_phase_damping: (m: number, rate: number) => void;
  _noise_model_set_thermal: (m: number, t1: number, t2: number) => void;
  _noise_model_set_gate_time: (m: number, t: number) => void;
  _noise_model_set_readout_error: (m: number, e01: number, e10: number) => void;
  _noise_model_set_enabled: (m: number, enabled: number) => void;
  _noise_apply_model: (s: number, q: number, m: number, rvsPtr: number) => void;
  _noise_apply_model_two_qubit: (
    s: number, q1: number, q2: number, m: number, rvsPtr: number,
  ) => void;
  _malloc: (n: number) => number;
  _free: (p: number) => void;
  HEAPF64: Float64Array;
}

function modAndPtr(state: QuantumState): { mod: NoiseModule; ptr: number } {
  const mod = state._internal_module() as unknown as NoiseModule;
  const ptr = state._internal_state_pointer();
  return { mod, ptr };
}

/** Uniform-[0, 1) source used when the caller does not supply a draw. */
export type RandomSource = () => number;

/** Crypto-backed uniform-[0, 1) draw (53-bit mantissa). */
function cryptoUniform(): number {
  const buf = new Uint32Array(2);
  globalThis.crypto.getRandomValues(buf);
  // 26 + 27 bits -> uniform double in [0, 1) with full 53-bit precision.
  return ((buf[0] >>> 6) * 134217728 + (buf[1] >>> 5)) / 9007199254740992;
}

const defaultRng: RandomSource = cryptoUniform;

function draw(rv: number | undefined): number {
  return rv === undefined ? defaultRng() : rv;
}

/** Standard-normal draw (Box-Muller over the default uniform source). */
function normalDraw(): number {
  let u = defaultRng();
  if (u <= 0) u = Number.MIN_VALUE;
  const v = defaultRng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Single-qubit depolarising channel with probability `p`.
 *  `randomValue` is the uniform-[0, 1) branch selector; omit it to use
 *  the crypto-backed default. */
export function depolarizing(
  state: QuantumState, qubit: number, p: number, randomValue?: number,
): void {
  const { mod, ptr } = modAndPtr(state);
  mod._noise_depolarizing_single(ptr, qubit, p, draw(randomValue));
}

/** Two-qubit depolarising channel with probability `p`. */
export function depolarizing2(
  state: QuantumState, qubit1: number, qubit2: number, p: number,
  randomValue?: number,
): void {
  const { mod, ptr } = modAndPtr(state);
  mod._noise_depolarizing_two_qubit(ptr, qubit1, qubit2, p, draw(randomValue));
}

/** Amplitude-damping channel (T1 relaxation) with rate `gamma` in [0, 1]. */
export function amplitudeDamping(
  state: QuantumState, qubit: number, gamma: number, randomValue?: number,
): void {
  const { mod, ptr } = modAndPtr(state);
  mod._noise_amplitude_damping(ptr, qubit, gamma, draw(randomValue));
}

/** Phase-damping channel with rate `gamma` in [0, 1]. */
export function phaseDamping(
  state: QuantumState, qubit: number, gamma: number, randomValue?: number,
): void {
  const { mod, ptr } = modAndPtr(state);
  mod._noise_phase_damping(ptr, qubit, gamma, draw(randomValue));
}

/** Gaussian pure-dephasing: multiplies the qubit's |1> amplitudes by
 *  `exp(i * sigma * randomPhase)`.  `randomPhase` should be a standard
 *  normal draw; omit it to use a Box-Muller draw over the default
 *  source, making the accumulated phase variance `sigma^2` per call. */
export function pureDephasing(
  state: QuantumState, qubit: number, sigma: number, randomPhase?: number,
): void {
  const { mod, ptr } = modAndPtr(state);
  mod._noise_pure_dephasing(
    ptr, qubit, sigma, randomPhase === undefined ? normalDraw() : randomPhase,
  );
}

/** Bit-flip channel with probability `p`. */
export function bitFlip(
  state: QuantumState, qubit: number, p: number, randomValue?: number,
): void {
  const { mod, ptr } = modAndPtr(state);
  mod._noise_bit_flip(ptr, qubit, p, draw(randomValue));
}

/** Phase-flip channel with probability `p`. */
export function phaseFlip(
  state: QuantumState, qubit: number, p: number, randomValue?: number,
): void {
  const { mod, ptr } = modAndPtr(state);
  mod._noise_phase_flip(ptr, qubit, p, draw(randomValue));
}

/** Bit-phase-flip channel with probability `p`. */
export function bitPhaseFlip(
  state: QuantumState, qubit: number, p: number, randomValue?: number,
): void {
  const { mod, ptr } = modAndPtr(state);
  mod._noise_bit_phase_flip(ptr, qubit, p, draw(randomValue));
}

/** Thermal relaxation with T1, T2 timescales over duration `t`.  The C
 *  channel consumes two uniform draws (T1 decay branch, T2 dephasing
 *  branch); pass `randomValues` to inject them. */
export function thermalRelaxation(
  state: QuantumState,
  qubit: number,
  t1: number,
  t2: number,
  duration: number,
  randomValues?: readonly [number, number],
): void {
  const { mod, ptr } = modAndPtr(state);
  const rvs = randomValues ?? [defaultRng(), defaultRng()];
  const rvsPtr = mod._malloc(16);
  try {
    mod.HEAPF64[rvsPtr >> 3] = rvs[0];
    mod.HEAPF64[(rvsPtr >> 3) + 1] = rvs[1];
    mod._noise_thermal_relaxation(ptr, qubit, t1, t2, duration, rvsPtr);
  } finally {
    mod._free(rvsPtr);
  }
}

/** Classical readout error: flips a measured `outcome` (0 or 1) with
 *  probability `error0to1` (reported 1 given true 0) respectively
 *  `error1to0` (reported 0 given true 1).  Pure post-processing -- the
 *  quantum state is untouched.  Returns the reported outcome. */
export async function readoutError(
  outcome: number, error0to1: number, error1to0: number, randomValue?: number,
): Promise<number> {
  const mod = (await getModule()) as unknown as NoiseModule;
  return mod._noise_readout_error(outcome, error0to1, error1to0, draw(randomValue));
}

/** Options for {@link DeviceNoiseModel.create}. */
export interface DeviceNoiseModelOptions {
  /** Single-qubit depolarising rate applied after each gate. */
  depolarizingRate?: number;
  /** Two-qubit depolarising rate (via `noise_model_create_realistic`'s
   *  gate-error scaling this is 10x the single-qubit rate when the
   *  realistic constructor is used; here it is whatever the C model
   *  holds -- set the single-qubit channels individually). */
  amplitudeDampingRate?: number;
  /** Phase-damping rate. */
  phaseDampingRate?: number;
  /** T1 relaxation time (same time unit as `gateTime`). */
  t1?: number;
  /** T2 dephasing time (same time unit as `gateTime`). */
  t2?: number;
  /** Gate duration driving the thermal-relaxation contribution. */
  gateTime?: number;
  /** Readout error probabilities [P(1|0), P(0|1)]. */
  readoutError?: readonly [number, number];
  /** Uniform-[0, 1) source for the per-application draws (defaults to
   *  the crypto-backed generator). */
  rng?: RandomSource;
}

/**
 * Composite per-device noise profile: depolarising + damping + thermal
 * relaxation applied together after each gate, mirroring Python
 * `moonlab.noise.DeviceNoiseModel`.  Wraps the C `noise_model_t` and
 * `noise_apply_model` / `noise_apply_model_two_qubit` entries.
 *
 * @example
 * ```typescript
 * const model = await DeviceNoiseModel.create({
 *   depolarizingRate: 1e-3, t1: 50e-6, t2: 30e-6, gateTime: 20e-9,
 * });
 * const state = await QuantumState.create({ numQubits: 2 });
 * state.h(0);
 * model.applySingle(state, 0);    // one noise trajectory after the gate
 * model.dispose();
 * ```
 */
export class DeviceNoiseModel {
  private ptr: number;
  private mod: NoiseModule;
  private rng: RandomSource;
  private freed = false;

  private constructor(ptr: number, mod: NoiseModule, rng: RandomSource) {
    this.ptr = ptr;
    this.mod = mod;
    this.rng = rng;
  }

  /** Build a model from explicit per-channel rates. */
  static async create(options: DeviceNoiseModelOptions = {}): Promise<DeviceNoiseModel> {
    const mod = (await getModule()) as unknown as NoiseModule;
    const ptr = mod._noise_model_create();
    if (!ptr) throw new Error('noise_model_create returned NULL');
    const m = new DeviceNoiseModel(ptr, mod, options.rng ?? defaultRng);
    if (options.depolarizingRate !== undefined) {
      mod._noise_model_set_depolarizing(ptr, options.depolarizingRate);
    }
    if (options.amplitudeDampingRate !== undefined) {
      mod._noise_model_set_amplitude_damping(ptr, options.amplitudeDampingRate);
    }
    if (options.phaseDampingRate !== undefined) {
      mod._noise_model_set_phase_damping(ptr, options.phaseDampingRate);
    }
    if (options.t1 !== undefined && options.t2 !== undefined) {
      mod._noise_model_set_thermal(ptr, options.t1, options.t2);
    }
    if (options.gateTime !== undefined) {
      mod._noise_model_set_gate_time(ptr, options.gateTime);
    }
    if (options.readoutError !== undefined) {
      mod._noise_model_set_readout_error(
        ptr, options.readoutError[0], options.readoutError[1],
      );
    }
    return m;
  }

  /** Build a model from device-style figures of merit: T1/T2 in
   *  microseconds, a per-gate error rate, and a readout error
   *  probability (`noise_model_create_realistic`). */
  static async realistic(
    t1Us: number, t2Us: number, gateError: number, readoutErrorProb: number,
    rng?: RandomSource,
  ): Promise<DeviceNoiseModel> {
    const mod = (await getModule()) as unknown as NoiseModule;
    const ptr = mod._noise_model_create_realistic(t1Us, t2Us, gateError, readoutErrorProb);
    if (!ptr) throw new Error('noise_model_create_realistic returned NULL');
    return new DeviceNoiseModel(ptr, mod, rng ?? defaultRng);
  }

  /** Internal: WASM pointer to the underlying `noise_model_t`. */
  _internal_ptr(): number {
    if (this.freed) throw new Error('DeviceNoiseModel disposed');
    return this.ptr;
  }

  /** Enable / disable the model (disabled models are no-ops). */
  setEnabled(enabled: boolean): void {
    this.mod._noise_model_set_enabled(this._internal_ptr(), enabled ? 1 : 0);
  }

  private applyWithDraws(
    stateFn: (rvsPtr: number) => void, numDraws: number,
    randomValues?: readonly number[],
  ): void {
    const mod = this.mod;
    const rvsPtr = mod._malloc(numDraws * 8);
    try {
      for (let i = 0; i < numDraws; i++) {
        mod.HEAPF64[(rvsPtr >> 3) + i] =
          randomValues !== undefined && i < randomValues.length
            ? randomValues[i]
            : this.rng();
      }
      stateFn(rvsPtr);
    } finally {
      mod._free(rvsPtr);
    }
  }

  /** Apply one noise trajectory to `qubit` (call after a single-qubit
   *  gate).  Consumes up to 6 uniform draws. */
  applySingle(state: QuantumState, qubit: number, randomValues?: readonly number[]): void {
    const ptr = this._internal_ptr();
    const { mod, ptr: statePtr } = modAndPtr(state);
    this.applyWithDraws(
      (rvsPtr) => mod._noise_apply_model(statePtr, qubit, ptr, rvsPtr),
      6, randomValues,
    );
  }

  /** Apply one noise trajectory to a two-qubit gate on `qubitA`,
   *  `qubitB`.  Consumes up to 10 uniform draws. */
  applyTwo(
    state: QuantumState, qubitA: number, qubitB: number,
    randomValues?: readonly number[],
  ): void {
    const ptr = this._internal_ptr();
    const { mod, ptr: statePtr } = modAndPtr(state);
    this.applyWithDraws(
      (rvsPtr) => mod._noise_apply_model_two_qubit(statePtr, qubitA, qubitB, ptr, rvsPtr),
      10, randomValues,
    );
  }

  /** Release the underlying C allocation. */
  dispose(): void {
    if (this.freed) return;
    this.mod._noise_model_destroy(this.ptr);
    this.freed = true;
  }
}
