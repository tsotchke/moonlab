/**
 * libirrep QEC zoo binding -- since v0.6.5.
 *
 * Wraps the eight CSS-code factories from
 * `src/integration/libirrep_bridge.{c,h}`.  Mirrors the Python
 * (`moonlab.libirrep_qec`, v0.6.3) and Rust
 * (`moonlab::libirrep_qec`, v0.6.4) bindings.  One class
 * (`LibirrepQecCode`) with eight named factories covers:
 *
 *   - rotated surface code,
 *   - Kitaev 2D toric code,
 *   - 2D color codes (Steane [[7,1,3]], Hamming [[15,7,3]]),
 *   - IBM bivariate-bicycle qLDPC (Bravyi-Nature 627 Gross-72/144/288),
 *   - Tillich-Zemor hypergraph product (d in {3, 4, 5}).
 *
 * The bridge is optional at C build time.  The WASM build does NOT
 * link libirrep (it lives outside the WASM toolchain), so every
 * factory raises `LibirrepNotBuiltError` in the browser today.
 * Probe with `await LibirrepQecCode.isAvailable()` first; the call
 * is cheap (one WASM call, no allocation) and lets consumers fall
 * back to the in-tree surface-code path.
 *
 * @example
 * ```typescript
 * import { LibirrepQecCode } from '@moonlab/quantum-core';
 *
 * if (await LibirrepQecCode.isAvailable()) {
 *   const code = await LibirrepQecCode.bb72_12_6();
 *   console.log(code.numQubits, code.logicalQubits, code.distance);
 *   code.dispose();
 * }
 * ```
 */

import { getModule } from './wasm-loader';

/** Bridge status codes mirroring `libirrep_bridge.h`. */
export const MOONLAB_LIBIRREP_OK = 0;
export const MOONLAB_LIBIRREP_NOT_BUILT = -201;
export const MOONLAB_LIBIRREP_BAD_ARG = -202;
export const MOONLAB_LIBIRREP_INTERNAL = -203;
export const MOONLAB_LIBIRREP_OOM = -204;

/** Base error class for any libirrep bridge failure. */
export class LibirrepError extends Error {
  readonly rc: number;
  constructor(message: string, rc: number) {
    super(message);
    this.name = 'LibirrepError';
    this.rc = rc;
  }
}

/** Raised when the C build did not link libirrep
 *  (rebuild with `-DQSIM_ENABLE_LIBIRREP=ON`). */
export class LibirrepNotBuiltError extends LibirrepError {
  constructor(ctx: string) {
    super(
      `${ctx}: moonlab was compiled without libirrep ` +
        `(rebuild with -DQSIM_ENABLE_LIBIRREP=ON).`,
      MOONLAB_LIBIRREP_NOT_BUILT,
    );
    this.name = 'LibirrepNotBuiltError';
  }
}

type LibirrepModule = {
  _moonlab_libirrep_available: () => number;
  _moonlab_libirrep_surface_code_new: (d: number, outPtr: number) => number;
  _moonlab_libirrep_toric_code_new: (lx: number, ly: number, outPtr: number) => number;
  _moonlab_libirrep_color_steane_new: (outPtr: number) => number;
  _moonlab_libirrep_color_hamming_15_7_3_new: (outPtr: number) => number;
  _moonlab_libirrep_bb_72_12_6_new: (outPtr: number) => number;
  _moonlab_libirrep_bb_144_12_12_new: (outPtr: number) => number;
  _moonlab_libirrep_bb_288_12_18_new: (outPtr: number) => number;
  _moonlab_libirrep_hgp_repetition_new: (d: number, outPtr: number) => number;
  _moonlab_libirrep_qec_free: (h: number) => void;
  _moonlab_libirrep_qec_n_qubits: (h: number) => number;
  _moonlab_libirrep_qec_n_x_stabs: (h: number) => number;
  _moonlab_libirrep_qec_n_z_stabs: (h: number) => number;
  _moonlab_libirrep_qec_logical_qubits: (h: number) => number;
  _moonlab_libirrep_qec_distance: (h: number) => number;
  _moonlab_libirrep_qec_get_x_check_row: (
    h: number,
    row: number,
    buf: number,
  ) => number;
  _moonlab_libirrep_qec_get_z_check_row: (
    h: number,
    row: number,
    buf: number,
  ) => number;
  _malloc: (size: number) => number;
  _free: (ptr: number) => void;
  HEAP32: Int32Array;
  HEAPU8: Uint8Array;
};

function _raiseFor(rc: number, ctx: string): void {
  if (rc === MOONLAB_LIBIRREP_OK) return;
  if (rc === MOONLAB_LIBIRREP_NOT_BUILT) throw new LibirrepNotBuiltError(ctx);
  throw new LibirrepError(`${ctx}: rc=${rc}`, rc);
}

/** Owned WASM handle to a CSS code built via libirrep. */
export class LibirrepQecCode {
  private handle: number;
  private mod: LibirrepModule;
  private freed = false;

  private constructor(handle: number, mod: LibirrepModule) {
    this.handle = handle;
    this.mod = mod;
  }

  /** Whether the WASM build was linked with libirrep (the default
   *  WASM build is NOT linked). */
  static async isAvailable(): Promise<boolean> {
    const mod = (await getModule()) as unknown as LibirrepModule;
    return mod._moonlab_libirrep_available() === 1;
  }

  private static async _factory(
    builder: (mod: LibirrepModule, outPtr: number) => number,
    ctx: string,
  ): Promise<LibirrepQecCode> {
    const mod = (await getModule()) as unknown as LibirrepModule;
    const outPtr = mod._malloc(4); /* sizeof(void*) on wasm32 = 4 */
    try {
      const rc = builder(mod, outPtr);
      _raiseFor(rc, ctx);
      const handle = mod.HEAP32[outPtr >> 2];
      if (!handle) throw new LibirrepError(`${ctx}: handle is NULL`, MOONLAB_LIBIRREP_INTERNAL);
      return new LibirrepQecCode(handle, mod);
    } finally {
      mod._free(outPtr);
    }
  }

  /** Rotated surface code at the given (odd) distance, `>= 2`. */
  static async surface(distance: number): Promise<LibirrepQecCode> {
    return LibirrepQecCode._factory(
      (mod, outPtr) => mod._moonlab_libirrep_surface_code_new(distance, outPtr),
      `surface(distance=${distance})`,
    );
  }

  /** Kitaev 2D toric code on the `Lx x Ly` torus. */
  static async toric(lx: number, ly: number): Promise<LibirrepQecCode> {
    return LibirrepQecCode._factory(
      (mod, outPtr) => mod._moonlab_libirrep_toric_code_new(lx, ly, outPtr),
      `toric(Lx=${lx}, Ly=${ly})`,
    );
  }

  /** Steane [[7, 1, 3]] 2D color code. */
  static async steane(): Promise<LibirrepQecCode> {
    return LibirrepQecCode._factory(
      (mod, outPtr) => mod._moonlab_libirrep_color_steane_new(outPtr),
      'steane()',
    );
  }

  /** [[15, 7, 3]] Hamming-based CSS code. */
  static async hamming15_7_3(): Promise<LibirrepQecCode> {
    return LibirrepQecCode._factory(
      (mod, outPtr) => mod._moonlab_libirrep_color_hamming_15_7_3_new(outPtr),
      'hamming15_7_3()',
    );
  }

  /** IBM Gross-72 bivariate-bicycle qLDPC code [[72, 12, 6]]. */
  static async bb72_12_6(): Promise<LibirrepQecCode> {
    return LibirrepQecCode._factory(
      (mod, outPtr) => mod._moonlab_libirrep_bb_72_12_6_new(outPtr),
      'bb72_12_6()',
    );
  }

  /** IBM Gross-144 bivariate-bicycle qLDPC code [[144, 12, 12]]. */
  static async bb144_12_12(): Promise<LibirrepQecCode> {
    return LibirrepQecCode._factory(
      (mod, outPtr) => mod._moonlab_libirrep_bb_144_12_12_new(outPtr),
      'bb144_12_12()',
    );
  }

  /** IBM Gross-288 bivariate-bicycle qLDPC code [[288, 12, 18]]. */
  static async bb288_12_18(): Promise<LibirrepQecCode> {
    return LibirrepQecCode._factory(
      (mod, outPtr) => mod._moonlab_libirrep_bb_288_12_18_new(outPtr),
      'bb288_12_18()',
    );
  }

  /** Tillich-Zemor hypergraph product of two `[d, 1, d]` repetition
   *  codes.  Only `d in {3, 4, 5}` is supported. */
  static async hgpRepetition(d: 3 | 4 | 5): Promise<LibirrepQecCode> {
    return LibirrepQecCode._factory(
      (mod, outPtr) => mod._moonlab_libirrep_hgp_repetition_new(d, outPtr),
      `hgpRepetition(d=${d})`,
    );
  }

  /** Release the underlying C handle. */
  dispose(): void {
    if (this.freed) return;
    this.mod._moonlab_libirrep_qec_free(this.handle);
    this.freed = true;
  }

  private _guardLive(): void {
    if (this.freed) throw new Error('LibirrepQecCode is disposed');
  }

  /** Number of physical qubits `n`. */
  get numQubits(): number {
    this._guardLive();
    return this.mod._moonlab_libirrep_qec_n_qubits(this.handle);
  }

  /** Number of X-stabiliser generators. */
  get numXStabs(): number {
    this._guardLive();
    return this.mod._moonlab_libirrep_qec_n_x_stabs(this.handle);
  }

  /** Number of Z-stabiliser generators. */
  get numZStabs(): number {
    this._guardLive();
    return this.mod._moonlab_libirrep_qec_n_z_stabs(this.handle);
  }

  /** Number of logical qubits `k = n - rank(H_X) - rank(H_Z)`. */
  get logicalQubits(): number {
    this._guardLive();
    return this.mod._moonlab_libirrep_qec_logical_qubits(this.handle);
  }

  /** Brute-force code distance.  Memoised by the C bridge after the
   *  first call.  Only tractable up to `n ~ 25, d <= 5`. */
  get distance(): number {
    this._guardLive();
    return this.mod._moonlab_libirrep_qec_distance(this.handle);
  }

  /** Length-`numQubits` `0/1` support vector of the X-stabiliser
   *  at the given row. */
  xCheckRow(row: number): Uint8Array {
    this._guardLive();
    const n = this.numQubits;
    const buf = this.mod._malloc(n);
    try {
      const rc = this.mod._moonlab_libirrep_qec_get_x_check_row(
        this.handle,
        row,
        buf,
      );
      _raiseFor(rc, `xCheckRow(${row})`);
      return new Uint8Array(this.mod.HEAPU8.subarray(buf, buf + n));
    } finally {
      this.mod._free(buf);
    }
  }

  /** Length-`numQubits` `0/1` support vector of the Z-stabiliser
   *  at the given row. */
  zCheckRow(row: number): Uint8Array {
    this._guardLive();
    const n = this.numQubits;
    const buf = this.mod._malloc(n);
    try {
      const rc = this.mod._moonlab_libirrep_qec_get_z_check_row(
        this.handle,
        row,
        buf,
      );
      _raiseFor(rc, `zCheckRow(${row})`);
      return new Uint8Array(this.mod.HEAPU8.subarray(buf, buf + n));
    } finally {
      this.mod._free(buf);
    }
  }
}
