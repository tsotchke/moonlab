/**
 * Surface code (Clifford-tableau variant) binding -- since v0.5.14.
 *
 * Wraps ``src/algorithms/topological/topological.{c,h}``'s
 * polynomial-scaling stabiliser-formalism surface code.  Mirrors
 * Python ``moonlab.surface_code`` (v0.5.13) + Rust
 * ``moonlab::surface_code`` (v0.5.12).  The dense state-vector
 * variant is limited to tiny `d`; the Clifford-tableau path is
 * what makes threshold sweeps on `d in {3, 5, 7}` tractable from
 * the browser.
 *
 * Decoding is **not** part of this surface yet; callers retrieve
 * raw syndrome data via {@link SurfaceCode.syndromeWeight} and
 * plug their own decoder (e.g. a minimum-weight perfect matching
 * library compiled to WASM).
 *
 * @example
 * ```typescript
 * import { SurfaceCode } from '@moonlab/quantum-core';
 *
 * const code = await SurfaceCode.create(3, 42n);
 * const q = code.dataIndex(1, 1);
 * code.applyError(q, 'X');
 * code.measureZSyndromes();
 * console.log(`weight = ${code.syndromeWeight}`);
 * code.dispose();
 * ```
 */

import { getModule } from './wasm-loader';

/** Pauli error type for {@link SurfaceCode.applyError}. */
export type PauliError = 'X' | 'Y' | 'Z';

type SurfaceCodeModule = {
  _surface_code_clifford_create: (distance: number, rngSeed: bigint) => number;
  _surface_code_clifford_free: (h: number) => void;
  _surface_code_clifford_data_index: (
    h: number, row: number, col: number,
  ) => number;
  _surface_code_clifford_apply_error: (
    h: number, q: number, errorType: number,
  ) => number;
  _surface_code_clifford_measure_z_syndromes: (h: number) => number;
  _surface_code_clifford_measure_x_syndromes: (h: number) => number;
  _surface_code_clifford_syndrome_weight: (h: number) => number;
};

/**
 * Owned WASM handle to a rotated-lattice surface code at distance
 * ``d`` with ``d^2`` data qubits and ``2(d-1)^2`` ancillas.
 */
export class SurfaceCode {
  private handle: number;
  private mod: SurfaceCodeModule;
  private _distance: number;
  private freed = false;

  private constructor(handle: number, mod: SurfaceCodeModule, distance: number) {
    this.handle = handle;
    this.mod = mod;
    this._distance = distance;
  }

  /** Allocate a rotated surface code.  `distance` must be odd and
   *  `>= 3`; the C side rejects even or trivial distances.
   *  `rngSeed` is a splitmix64 state for ancilla-mediated
   *  measurement; pass a fixed seed for reproducibility. */
  static async create(distance: number, rngSeed: bigint = 0n): Promise<SurfaceCode> {
    if (distance < 3 || distance % 2 === 0) {
      throw new RangeError(
        `surface code distance must be odd and >= 3, got ${distance}`,
      );
    }
    const mod = (await getModule()) as unknown as SurfaceCodeModule;
    const h = mod._surface_code_clifford_create(distance, rngSeed);
    if (!h) throw new Error('surface_code_clifford_create returned NULL');
    return new SurfaceCode(h, mod, distance);
  }

  /** Release the underlying allocation. */
  dispose(): void {
    if (this.freed) return;
    this.mod._surface_code_clifford_free(this.handle);
    this.freed = true;
  }

  /** Code distance `d`. */
  get distance(): number { return this._distance; }

  /** Number of physical data qubits = `d^2`. */
  get numDataQubits(): number { return this._distance * this._distance; }

  /** Number of ancillas in each parity sector = `(d - 1)^2`.
   *  Total ancillas across both sectors = `2 (d - 1)^2`. */
  get numAncillasPerSector(): number {
    const m = this._distance - 1;
    return m * m;
  }

  /** Linear data-qubit index from `(row, col)` in `[0, d)`. */
  dataIndex(row: number, col: number): number {
    this._guardLive();
    if (row < 0 || row >= this._distance || col < 0 || col >= this._distance) {
      throw new RangeError(
        `(row, col) = (${row}, ${col}) out of [0, ${this._distance})`,
      );
    }
    return this.mod._surface_code_clifford_data_index(this.handle, row, col);
  }

  /** Apply a single-qubit Pauli error on data qubit `qubit`. */
  applyError(qubit: number, errorType: PauliError): void {
    this._guardLive();
    if (qubit < 0 || qubit >= this.numDataQubits) {
      throw new RangeError(
        `data qubit ${qubit} out of [0, ${this.numDataQubits})`,
      );
    }
    if (errorType !== 'X' && errorType !== 'Y' && errorType !== 'Z') {
      throw new TypeError(
        `errorType must be 'X', 'Y', or 'Z', got ${errorType}`,
      );
    }
    const rc = this.mod._surface_code_clifford_apply_error(
      this.handle, qubit, errorType.charCodeAt(0),
    );
    if (rc !== 0) throw new Error(`apply_error rc=${rc}`);
  }

  /** Measure all `(d - 1)^2` Z-type stabilisers (ZZZZ on four
   *  data qubits around each interior vertex). */
  measureZSyndromes(): void {
    this._guardLive();
    const rc = this.mod._surface_code_clifford_measure_z_syndromes(this.handle);
    if (rc !== 0) throw new Error(`measure_z_syndromes rc=${rc}`);
  }

  /** Measure all `(d - 1)^2` X-type stabilisers (XXXX on four
   *  data qubits around each interior face). */
  measureXSyndromes(): void {
    this._guardLive();
    const rc = this.mod._surface_code_clifford_measure_x_syndromes(this.handle);
    if (rc !== 0) throw new Error(`measure_x_syndromes rc=${rc}`);
  }

  /** Set-bit count across both X and Z syndromes (diagnostic).
   *  Zero on a logical-zero start; non-zero after an undetectable
   *  Pauli error. */
  get syndromeWeight(): number {
    this._guardLive();
    return this.mod._surface_code_clifford_syndrome_weight(this.handle);
  }

  private _guardLive(): void {
    if (this.freed) throw new Error('SurfaceCode is disposed');
  }
}
