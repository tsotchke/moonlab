/**
 * Clifford-Assisted PEPS (2D) binding (C-side since v0.2.1, JS
 * binding since v0.4.12).
 *
 * Wraps ``src/algorithms/tensor_network/ca_peps.{c,h}``, the 2D
 * generalisation of CA-MPS.  Same Clifford-tableau + physical-MPS
 * split: Clifford gates are free (O(n) tableau bit-ops), and
 * non-Clifford gates conjugate through to Pauli rotations on the
 * physical factor.  Mirrors Python ``moonlab.ca_peps.CAPEPS`` and
 * Rust ``moonlab::ca_peps::CaPeps``.
 *
 * @example
 * ```typescript
 * import { CaPeps, PauliCode } from '@moonlab/quantum-core';
 *
 * // 2x3 lattice with bond-dim cap chi = 8.
 * const state = await CaPeps.create(2, 3, 8);
 * state.h(0).cnot(0, 1);
 * const z0 = state.expectPauliSingle(0, PauliCode.Z);
 * console.log(`<Z_0> = ${z0.toFixed(6)}`);
 * state.dispose();
 * ```
 */

import { getModule } from './wasm-loader';
import { PauliCode } from './mpdo';

// Re-export so `import { PauliCode } from '@moonlab/quantum-core'` keeps
// working when callers reach for CaPeps observables.
export { PauliCode } from './mpdo';

type CaPepsModule = {
  _moonlab_ca_peps_create: (lx: number, ly: number, chi: number) => number;
  _moonlab_ca_peps_free: (h: number) => void;
  _moonlab_ca_peps_clone: (h: number) => number;
  _moonlab_ca_peps_lx: (h: number) => number;
  _moonlab_ca_peps_ly: (h: number) => number;
  _moonlab_ca_peps_num_qubits: (h: number) => number;
  _moonlab_ca_peps_max_bond_dim: (h: number) => number;
  _moonlab_ca_peps_current_bond_dim: (h: number) => number;
  _moonlab_ca_peps_max_half_cut_entropy: (h: number) => number;
  _moonlab_ca_peps_norm: (h: number) => number;
  _moonlab_ca_peps_normalize: (h: number) => number;
  _moonlab_ca_peps_h: (h: number, q: number) => number;
  _moonlab_ca_peps_s: (h: number, q: number) => number;
  _moonlab_ca_peps_sdag: (h: number, q: number) => number;
  _moonlab_ca_peps_x: (h: number, q: number) => number;
  _moonlab_ca_peps_y: (h: number, q: number) => number;
  _moonlab_ca_peps_z: (h: number, q: number) => number;
  _moonlab_ca_peps_cnot: (h: number, c: number, t: number) => number;
  _moonlab_ca_peps_cz: (h: number, a: number, b: number) => number;
  _moonlab_ca_peps_rx: (h: number, q: number, theta: number) => number;
  _moonlab_ca_peps_ry: (h: number, q: number, theta: number) => number;
  _moonlab_ca_peps_rz: (h: number, q: number, theta: number) => number;
  _moonlab_ca_peps_t_gate: (h: number, q: number) => number;
  _moonlab_ca_peps_t_dagger: (h: number, q: number) => number;
  _moonlab_ca_peps_phase: (h: number, q: number, theta: number) => number;
  _moonlab_ca_peps_prob_z: (h: number, q: number, outPtr: number) => number;
  _moonlab_ca_peps_expect_pauli: (
    h: number,
    pauliPtr: number,
    outPtr: number,
  ) => number;
  _malloc: (n: number) => number;
  _free: (p: number) => void;
  HEAPU8: Uint8Array;
  HEAPF64: Float64Array;
};

/**
 * Owned WASM handle to a CA-PEPS state.
 *
 * Linear-index addressing throughout: qubit `q = x + Lx * y`.
 * Two-qubit gates require adjacent linear indices; vertical
 * neighbours sit `Lx` apart in linear order.
 */
export class CaPeps {
  private handle: number;
  private mod: CaPepsModule;
  private freed = false;

  private constructor(handle: number, mod: CaPepsModule) {
    this.handle = handle;
    this.mod = mod;
  }

  /** Allocate a CA-PEPS on `Lx x Ly` with per-bond chi cap
   *  `chiBond`.  Initial state `|0...0>` with `D = I`. */
  static async create(lx: number, ly: number, chiBond: number): Promise<CaPeps> {
    if (lx < 1 || ly < 1) {
      throw new RangeError(`lx=${lx}, ly=${ly}: both must be >= 1`);
    }
    if (chiBond < 1) {
      throw new RangeError(`chiBond must be >= 1, got ${chiBond}`);
    }
    const mod = (await getModule()) as unknown as CaPepsModule;
    const h = mod._moonlab_ca_peps_create(lx, ly, chiBond);
    if (!h) throw new Error('moonlab_ca_peps_create returned NULL');
    return new CaPeps(h, mod);
  }

  /** Release the underlying WASM allocation. */
  dispose(): void {
    if (this.freed) return;
    this.mod._moonlab_ca_peps_free(this.handle);
    this.freed = true;
  }

  /** Deep copy. */
  clone(): CaPeps {
    this._guardLive();
    const h = this.mod._moonlab_ca_peps_clone(this.handle);
    if (!h) throw new Error('moonlab_ca_peps_clone returned NULL');
    return new CaPeps(h, this.mod);
  }

  get lx(): number { this._guardLive(); return this.mod._moonlab_ca_peps_lx(this.handle); }
  get ly(): number { this._guardLive(); return this.mod._moonlab_ca_peps_ly(this.handle); }
  get numQubits(): number {
    this._guardLive();
    return this.mod._moonlab_ca_peps_num_qubits(this.handle);
  }
  get maxBondDim(): number {
    this._guardLive();
    return this.mod._moonlab_ca_peps_max_bond_dim(this.handle);
  }
  get currentBondDim(): number {
    this._guardLive();
    return this.mod._moonlab_ca_peps_current_bond_dim(this.handle);
  }
  /** `<phi | phi>` of the physical factor; ~1 modulo truncation. */
  get norm(): number {
    this._guardLive();
    return this.mod._moonlab_ca_peps_norm(this.handle);
  }
  /** Maximum half-cut von Neumann entanglement entropy of `|phi>`
   *  across all bipartitions, in nats. */
  get maxHalfCutEntropy(): number {
    this._guardLive();
    return this.mod._moonlab_ca_peps_max_half_cut_entropy(this.handle);
  }

  normalize(): this {
    this._check(
      this.mod._moonlab_ca_peps_normalize(this.handle),
      'normalize',
    );
    return this;
  }

  // ---- Clifford gates ---------------------------------------------------
  h(q: number): this   { return this._call1('h', q); }
  s(q: number): this   { return this._call1('s', q); }
  sdag(q: number): this { return this._call1('sdag', q); }
  x(q: number): this   { return this._call1('x', q); }
  y(q: number): this   { return this._call1('y', q); }
  z(q: number): this   { return this._call1('z', q); }
  cnot(control: number, target: number): this {
    return this._call2('cnot', control, target);
  }
  cz(a: number, b: number): this { return this._call2('cz', a, b); }

  // ---- Non-Clifford gates ----------------------------------------------
  rx(q: number, theta: number): this { return this._call1p('rx', q, theta); }
  ry(q: number, theta: number): this { return this._call1p('ry', q, theta); }
  rz(q: number, theta: number): this { return this._call1p('rz', q, theta); }
  t(q: number): this    { return this._call1('t_gate', q); }
  tdg(q: number): this  { return this._call1('t_dagger', q); }
  phase(q: number, theta: number): this {
    return this._call1p('phase', q, theta);
  }

  // ---- Observables -----------------------------------------------------

  /** Marginal `P(Z_q = +1)`. */
  probZ(q: number): number {
    this._guardQubit(q);
    const out = this.mod._malloc(8);
    try {
      this._check(
        this.mod._moonlab_ca_peps_prob_z(this.handle, q, out),
        'prob_z',
      );
      return this.mod.HEAPF64[out >> 3];
    } finally {
      this.mod._free(out);
    }
  }

  /** `<psi | P | psi>` for an n-qubit Pauli string.  Entry per
   *  qubit is `PauliCode` (0=I, 1=X, 2=Y, 3=Z).  Length must equal
   *  `numQubits`.  Returns `[re, im]` of the complex expectation. */
  expectPauli(pauli: readonly PauliCode[] | Uint8Array): [number, number] {
    this._guardLive();
    const n = this.numQubits;
    if (pauli.length !== n) {
      throw new RangeError(
        `pauli.length=${pauli.length} != numQubits=${n}`,
      );
    }
    const pauliPtr = this.mod._malloc(n);
    const outPtr = this.mod._malloc(16);  // double _Complex = 2 doubles
    try {
      for (let i = 0; i < n; i++) {
        this.mod.HEAPU8[pauliPtr + i] = pauli[i];
      }
      this._check(
        this.mod._moonlab_ca_peps_expect_pauli(this.handle, pauliPtr, outPtr),
        'expect_pauli',
      );
      const idx = outPtr >> 3;
      return [this.mod.HEAPF64[idx], this.mod.HEAPF64[idx + 1]];
    } finally {
      this.mod._free(pauliPtr);
      this.mod._free(outPtr);
    }
  }

  /** `<psi | P_q | psi>` for a single-site Pauli on qubit `q`,
   *  identity elsewhere.  Returns the real part only (single-site
   *  Pauli expectations are real up to roundoff). */
  expectPauliSingle(q: number, pauli: PauliCode): number {
    this._guardQubit(q);
    const n = this.numQubits;
    const ops = new Uint8Array(n);  // zero-init = identity
    ops[q] = pauli;
    return this.expectPauli(ops)[0];
  }

  // ---- Internal helpers ------------------------------------------------

  private _guardLive(): void {
    if (this.freed) throw new Error('CaPeps is disposed');
  }

  private _guardQubit(q: number): void {
    this._guardLive();
    const n = this.numQubits;
    if (q < 0 || q >= n) {
      throw new RangeError(`qubit ${q} out of [0, ${n})`);
    }
  }

  private _call1(name: string, q: number): this {
    this._guardQubit(q);
    const m = this.mod as unknown as Record<string, (...args: number[]) => number>;
    this._check(m[`_moonlab_ca_peps_${name}`](this.handle, q), name);
    return this;
  }

  private _call1p(name: string, q: number, theta: number): this {
    this._guardQubit(q);
    const m = this.mod as unknown as Record<string, (...args: number[]) => number>;
    this._check(m[`_moonlab_ca_peps_${name}`](this.handle, q, theta), name);
    return this;
  }

  private _call2(name: string, a: number, b: number): this {
    this._guardQubit(a);
    this._guardQubit(b);
    const m = this.mod as unknown as Record<string, (...args: number[]) => number>;
    this._check(m[`_moonlab_ca_peps_${name}`](this.handle, a, b), name);
    return this;
  }

  private _check(rc: number, fn: string): void {
    if (rc !== 0) throw new Error(`moonlab_ca_peps_${fn} rc=${rc}`);
  }
}
