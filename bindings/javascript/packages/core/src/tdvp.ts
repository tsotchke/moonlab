/**
 * Adaptive-bond two-site TDVP WebAssembly binding (since v0.4.3).
 *
 * Thin wrapper around the `moonlab_tdvp_*` stable-ABI surface in
 * `src/applications/moonlab_tdvp_export.c`.  The C side bundles the
 * MPS state, Hamiltonian MPO, internal `tdvp_engine_t` (including
 * per-bond PID controller slots when the adaptive controller is
 * enabled), and a step-by-step `tdvp_history_t` behind a single
 * opaque handle, so this TS layer never has to marshal the nested
 * `tdvp_config_t` byte-by-byte.
 *
 * Highlights:
 *  - {@link TdvpEngine.createTfim} / {@link TdvpEngine.createHeisenberg}
 *    convenience constructors for the two MPO families exported by the
 *    library.  When `adaptiveTargetEntropy > 0` the entropy-feedback
 *    PID controller is enabled with the reference-paper gains
 *    (arXiv:2604.03960); pass 0 to stay on the legacy fixed-cap path.
 *  - {@link TdvpEngine.step} and {@link TdvpEngine.evolveTo}.
 *  - Accessors: `currentTime`, `currentEnergy`, `currentNorm`,
 *    `currentMaxBondDim`, `numBonds`, `bondChi(bond)`.
 *  - {@link TdvpEngine.history} accessors:
 *    `numSteps`, `getStep(step)`, `getBondChi(step)`.
 *
 * @example
 * ```typescript
 * import { TdvpEngine, EvolutionType } from '@moonlab/quantum-core';
 *
 * const engine = await TdvpEngine.createTfim({
 *   numSites: 8,
 *   J: 1.0,
 *   h: 1.0,
 *   initialBondDim: 8,
 *   maxBondDim: 32,
 *   dt: 0.05,
 *   evolution: EvolutionType.ImaginaryTime,
 *   adaptiveTargetEntropy: 1e-3,
 * });
 *
 * for (let step = 0; step < 30; step++) {
 *   engine.step();
 * }
 * console.log(`E = ${engine.currentEnergy.toFixed(4)}, chi = ${engine.currentMaxBondDim}`);
 *
 * engine.dispose();
 * ```
 */

import { getModule } from './wasm-loader';

/** Direction of TDVP evolution. */
export enum EvolutionType {
  /** `exp(-i H dt)` -- unitary, norm-preserving by construction. */
  RealTime = 0,
  /** `exp(-H dt)` -- non-unitary; projects onto the ground state at
   *  long times. */
  ImaginaryTime = 1,
}

/** Common options for both {@link TdvpEngine.createHeisenberg} and
 *  {@link TdvpEngine.createTfim}. */
export interface TdvpCommonOptions {
  /** Chain length (>= 2). */
  numSites: number;
  /** Initial MPS bond width; clamped to `maxBondDim` and to a floor
   *  of 4. */
  initialBondDim?: number;
  /** Outer-bound bond-dim cap (also the adaptive controller's
   *  `chi_ceiling` when adaptive is on). */
  maxBondDim?: number;
  /** Time-step size (positive). */
  dt?: number;
  /** Real-time vs imag-time evolution. */
  evolution?: EvolutionType;
  /** Entropy-error budget for the PID controller.  Pass `0` (or omit)
   *  to keep the legacy fixed-bond path. */
  adaptiveTargetEntropy?: number;
}

/** Options for {@link TdvpEngine.createHeisenberg}. */
export interface HeisenbergOptions extends TdvpCommonOptions {
  /** Exchange coupling. */
  J?: number;
  /** XXZ anisotropy (1.0 = isotropic Heisenberg). */
  Delta?: number;
  /** Longitudinal field. */
  h?: number;
}

/** Options for {@link TdvpEngine.createTfim}. */
export interface TfimOptions extends TdvpCommonOptions {
  /** Z-Z exchange coupling. */
  J?: number;
  /** Transverse field strength. */
  h?: number;
}

/** A single recorded step from the engine's history. */
export interface TdvpHistoryStep {
  time: number;
  energy: number;
  norm: number;
}

/** Owned WASM handle to a TDVP engine.  Construct via
 *  {@link TdvpEngine.createTfim} / {@link TdvpEngine.createHeisenberg}
 *  and release with {@link TdvpEngine.dispose}. */
export class TdvpEngine {
  private handle: number;
  private mod: ReturnType<typeof getModule> extends Promise<infer T> ? T : never;
  private nb: number;
  private freed = false;

  private constructor(handle: number, mod: TdvpEngine['mod'], numBonds: number) {
    this.handle = handle;
    this.mod = mod;
    this.nb = numBonds;
  }

  /** Build an adaptive-bond TDVP engine on a Heisenberg-XXZ chain
   *
   *   H = J * sum_i (X X + Y Y + Delta Z Z) - h * sum_i Z
   */
  static async createHeisenberg(opts: HeisenbergOptions): Promise<TdvpEngine> {
    return TdvpEngine._build(opts, async (mod) => {
      const m = mod as unknown as {
        _moonlab_tdvp_create_heisenberg: (
          n: number, J: number, Delta: number, h: number,
          chiInit: number, chiMax: number, dt: number,
          imag: number, eps: number,
        ) => number;
      };
      return m._moonlab_tdvp_create_heisenberg(
        opts.numSites,
        opts.J ?? 1.0,
        opts.Delta ?? 1.0,
        opts.h ?? 0.0,
        opts.initialBondDim ?? 8,
        opts.maxBondDim ?? 32,
        opts.dt ?? 0.05,
        opts.evolution === EvolutionType.ImaginaryTime ? 1 : 0,
        opts.adaptiveTargetEntropy ?? 0.0,
      );
    });
  }

  /** Build an adaptive-bond TDVP engine on a transverse-field Ising chain
   *
   *   H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i
   */
  static async createTfim(opts: TfimOptions): Promise<TdvpEngine> {
    return TdvpEngine._build(opts, async (mod) => {
      const m = mod as unknown as {
        _moonlab_tdvp_create_tfim: (
          n: number, J: number, h: number,
          chiInit: number, chiMax: number, dt: number,
          imag: number, eps: number,
        ) => number;
      };
      return m._moonlab_tdvp_create_tfim(
        opts.numSites,
        opts.J ?? 1.0,
        opts.h ?? 1.0,
        opts.initialBondDim ?? 8,
        opts.maxBondDim ?? 32,
        opts.dt ?? 0.05,
        opts.evolution === EvolutionType.ImaginaryTime ? 1 : 0,
        opts.adaptiveTargetEntropy ?? 0.0,
      );
    });
  }

  /** Internal: shared construction path for both convenience builders. */
  private static async _build(
    opts: TdvpCommonOptions,
    spawn: (mod: unknown) => Promise<number>,
  ): Promise<TdvpEngine> {
    if (opts.numSites < 2) throw new Error('numSites must be >= 2');
    const mod = await getModule();
    const handle = await spawn(mod);
    if (!handle) throw new Error('moonlab_tdvp_create_* returned NULL');
    const numBonds = (mod as unknown as {
      _moonlab_tdvp_num_bonds: (h: number) => number;
    })._moonlab_tdvp_num_bonds(handle);
    return new TdvpEngine(handle, mod as TdvpEngine['mod'], numBonds);
  }

  /** Release the underlying WASM allocation.  Safe to call twice. */
  dispose(): void {
    if (this.freed) return;
    (this.mod as unknown as { _moonlab_tdvp_engine_free: (h: number) => void })
      ._moonlab_tdvp_engine_free(this.handle);
    this.freed = true;
  }

  /** Advance the engine by one TDVP step. */
  step(): void {
    const rc = (this.mod as unknown as {
      _moonlab_tdvp_step: (h: number) => number;
    })._moonlab_tdvp_step(this.handle);
    if (rc !== 0) throw new Error(`moonlab_tdvp_step rc=${rc}`);
  }

  /** Advance until the engine's clock reaches @p targetTime. */
  evolveTo(targetTime: number): void {
    const rc = (this.mod as unknown as {
      _moonlab_tdvp_evolve_to: (h: number, t: number) => number;
    })._moonlab_tdvp_evolve_to(this.handle, targetTime);
    if (rc !== 0) throw new Error(`moonlab_tdvp_evolve_to rc=${rc}`);
  }

  /** Current accumulated evolution time. */
  get currentTime(): number {
    return (this.mod as unknown as {
      _moonlab_tdvp_current_time: (h: number) => number;
    })._moonlab_tdvp_current_time(this.handle);
  }

  /** Current variational energy `<psi|H|psi>`. */
  get currentEnergy(): number {
    return (this.mod as unknown as {
      _moonlab_tdvp_current_energy: (h: number) => number;
    })._moonlab_tdvp_current_energy(this.handle);
  }

  /** Current MPS norm `<psi|psi>^{1/2}`. */
  get currentNorm(): number {
    return (this.mod as unknown as {
      _moonlab_tdvp_current_norm: (h: number) => number;
    })._moonlab_tdvp_current_norm(this.handle);
  }

  /** Peak bond dimension across the MPS factor. */
  get currentMaxBondDim(): number {
    return (this.mod as unknown as {
      _moonlab_tdvp_current_max_bond_dim: (h: number) => number;
    })._moonlab_tdvp_current_max_bond_dim(this.handle);
  }

  /** Number of inter-site bonds (`numSites - 1`). */
  get numBonds(): number {
    return this.nb;
  }

  /** PID-selected chi for bond @p bond, or 0 when the adaptive
   *  controller is disabled. */
  bondChi(bond: number): number {
    if (bond < 0 || bond >= this.nb) {
      throw new RangeError(`bond ${bond} out of [0, ${this.nb})`);
    }
    return (this.mod as unknown as {
      _moonlab_tdvp_bond_chi: (h: number, b: number) => number;
    })._moonlab_tdvp_bond_chi(this.handle, bond);
  }

  /** Number of steps recorded in the engine's history. */
  get historyNumSteps(): number {
    return (this.mod as unknown as {
      _moonlab_tdvp_history_num_steps: (h: number) => number;
    })._moonlab_tdvp_history_num_steps(this.handle);
  }

  /** Read step @p step from the engine's history. */
  historyStep(step: number): TdvpHistoryStep {
    if (step < 0 || step >= this.historyNumSteps) {
      throw new RangeError(`step ${step} out of [0, ${this.historyNumSteps})`);
    }
    const m = this.mod as unknown as {
      _moonlab_tdvp_history_get_step: (
        h: number, s: number, tp: number, ep: number, np: number,
      ) => number;
      _malloc: (n: number) => number;
      _free: (p: number) => void;
      HEAPF64: Float64Array;
    };
    const tPtr = m._malloc(8);
    const ePtr = m._malloc(8);
    const nPtr = m._malloc(8);
    try {
      const rc = m._moonlab_tdvp_history_get_step(
        this.handle, step, tPtr, ePtr, nPtr);
      if (rc !== 0) {
        throw new Error(`moonlab_tdvp_history_get_step rc=${rc}`);
      }
      return {
        time: m.HEAPF64[tPtr >> 3],
        energy: m.HEAPF64[ePtr >> 3],
        norm: m.HEAPF64[nPtr >> 3],
      };
    } finally {
      m._free(tPtr); m._free(ePtr); m._free(nPtr);
    }
  }

  /** Read the per-bond chi snapshot for step @p step.  Returns a
   *  fresh `Uint32Array` of length `numBonds`. */
  historyBondChi(step: number): Uint32Array {
    if (step < 0 || step >= this.historyNumSteps) {
      throw new RangeError(`step ${step} out of [0, ${this.historyNumSteps})`);
    }
    const m = this.mod as unknown as {
      _moonlab_tdvp_history_get_bond_chi: (
        h: number, s: number, buf: number, cap: number,
      ) => number;
      _malloc: (n: number) => number;
      _free: (p: number) => void;
      HEAPU32: Uint32Array;
    };
    const bytes = this.nb * 4;
    const buf = m._malloc(bytes);
    try {
      const rc = m._moonlab_tdvp_history_get_bond_chi(
        this.handle, step, buf, this.nb);
      if (rc !== 0) {
        throw new Error(`moonlab_tdvp_history_get_bond_chi rc=${rc}`);
      }
      return new Uint32Array(m.HEAPU32.buffer, buf, this.nb).slice();
    } finally {
      m._free(buf);
    }
  }
}
