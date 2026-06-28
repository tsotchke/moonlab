/**
 * Standalone Aaronson-Gottesman Clifford tableau binding (since
 * v0.4.5).
 *
 * Wraps ``src/backends/clifford/clifford.{c,h}`` so JavaScript code
 * can simulate arbitrary-qubit Clifford circuits in polynomial time.
 * Useful for n > 32 GHZ / Bell / stabilizer-code experiments that
 * exceed the dense state-vector ceiling.
 *
 * The {@link CliffordTableau} class mirrors the Python
 * ``moonlab.clifford.Clifford`` and the Rust
 * ``moonlab::backends::clifford`` surface.  Gate methods are
 * fluent (return ``this``); single-qubit measurement and full
 * computational-basis sampling go through {@link
 * CliffordTableau.measure} and {@link CliffordTableau.sampleAll}.
 *
 * The C measurement entry point takes a splitmix64 RNG state
 * pointer; this wrapper hides that behind ``Math.random``-seeded
 * 64-bit splitmix when the caller does not pass an explicit seed.
 *
 * @example
 * ```typescript
 * import { CliffordTableau } from '@moonlab/quantum-core';
 *
 * // GHZ state on 64 qubits: H(0); CNOT(0, i) for i in 1..n-1.
 * const c = await CliffordTableau.create(64);
 * c.h(0);
 * for (let i = 1; i < 64; i++) c.cnot(0, i);
 *
 * // Sample one bitstring: all zeros or all ones, with equal prob.
 * const sample = c.sampleAll();
 * console.log(`bits = 0x${sample.bits.toString(16)}`);
 *
 * c.dispose();
 * ```
 */

import { getModule } from './wasm-loader';

/** Result of a single-qubit Z-basis measurement. */
export interface MeasureResult {
  /** Measurement outcome, 0 or 1. */
  outcome: 0 | 1;
  /** ``"deterministic"`` if every stabiliser commuted with Z_q (the
   *  tableau alone fixed the outcome), ``"random"`` if a stabiliser
   *  anticommuted and the result was drawn from the RNG state. */
  outcomeKind: "deterministic" | "random";
}

/** Result of a full computational-basis sample. */
export interface SampleAllResult {
  /** Packed bits: bit ``q`` is the outcome on qubit ``q``.  At most
   *  64 qubits fit in this representation; for larger tableaus call
   *  {@link CliffordTableau.measure} per qubit. */
  bits: bigint;
}

/**
 * Owned WASM handle to a Clifford tableau.
 */
export class CliffordTableau {
  private handle: number;
  private mod: ReturnType<typeof getModule> extends Promise<infer T> ? T : never;
  private nq: number;
  private rngState: bigint;
  private freed = false;

  private constructor(handle: number,
                       mod: CliffordTableau['mod'],
                       numQubits: number) {
    this.handle = handle;
    this.mod = mod;
    this.nq = numQubits;
    // Seed splitmix64 RNG.  64-bit seed combined from two
    // Math.random() draws so the upper and lower halves are
    // independent; the C side advances the state per measurement.
    const hi = BigInt(Math.floor(Math.random() * 0x100000000));
    const lo = BigInt(Math.floor(Math.random() * 0x100000000));
    this.rngState = (hi << 32n) ^ lo ^ 1n;  // never zero
  }

  /** Allocate a fresh tableau in |0..0> on ``numQubits`` qubits. */
  static async create(numQubits: number): Promise<CliffordTableau> {
    if (numQubits < 1) {
      throw new RangeError(`numQubits must be >= 1, got ${numQubits}`);
    }
    const mod = await getModule();
    const h = (mod as unknown as {
      _clifford_tableau_create: (n: number) => number;
    })._clifford_tableau_create(numQubits);
    if (!h) throw new Error('clifford_tableau_create returned NULL');
    return new CliffordTableau(h, mod as CliffordTableau['mod'], numQubits);
  }

  /** Release the underlying WASM allocation. */
  dispose(): void {
    if (this.freed) return;
    (this.mod as unknown as { _clifford_tableau_free: (h: number) => void })
      ._clifford_tableau_free(this.handle);
    this.freed = true;
  }

  /** Number of qubits the tableau was created for. */
  get numQubits(): number { return this.nq; }

  /** Override the splitmix64 RNG state used by ``measure`` /
   *  ``sampleAll``.  Both calls advance the state in place; this
   *  setter lets the caller pin it for reproducibility. */
  setRngSeed(seed: bigint): this {
    this.rngState = seed === 0n ? 1n : seed;
    return this;
  }

  // ---- Single-qubit Clifford gates --------------------------------------
  h(q: number): this   { this._call1('clifford_h', q);     return this; }
  s(q: number): this   { this._call1('clifford_s', q);     return this; }
  sdag(q: number): this { this._call1('clifford_s_dag', q); return this; }
  x(q: number): this   { this._call1('clifford_x', q);     return this; }
  y(q: number): this   { this._call1('clifford_y', q);     return this; }
  z(q: number): this   { this._call1('clifford_z', q);     return this; }

  // ---- Two-qubit Clifford gates -----------------------------------------
  cnot(ctrl: number, tgt: number): this {
    this._call2('clifford_cnot', ctrl, tgt); return this;
  }
  cz(a: number, b: number): this {
    this._call2('clifford_cz', a, b); return this;
  }
  swap(a: number, b: number): this {
    this._call2('clifford_swap', a, b); return this;
  }

  /** Z-basis measurement on qubit ``q``.  Mutates the tableau and
   *  advances the internal RNG state. */
  measure(q: number): MeasureResult {
    if (q < 0 || q >= this.nq) {
      throw new RangeError(`qubit ${q} out of [0, ${this.nq})`);
    }
    const m = this.mod as unknown as {
      _clifford_measure: (
        h: number, q: number, rngPtr: number, outPtr: number, kindPtr: number,
      ) => number;
      _malloc: (n: number) => number;
      _free: (p: number) => void;
      HEAPU32: Uint32Array;
      HEAP32: Int32Array;
    };
    const rngPtr  = m._malloc(8);
    const outPtr  = m._malloc(4);
    const kindPtr = m._malloc(4);
    try {
      // Write the 64-bit RNG state to WASM memory (little-endian).
      const idx32 = rngPtr >> 2;
      m.HEAPU32[idx32]     = Number(this.rngState & 0xffffffffn);
      m.HEAPU32[idx32 + 1] = Number((this.rngState >> 32n) & 0xffffffffn);
      const rc = m._clifford_measure(
        this.handle, q, rngPtr, outPtr, kindPtr);
      if (rc !== 0) {
        throw new Error(`clifford_measure rc=${rc}`);
      }
      // Read the (mutated) RNG state back.
      const lo = BigInt(m.HEAPU32[idx32]);
      const hi = BigInt(m.HEAPU32[idx32 + 1]);
      this.rngState = (hi << 32n) | lo;

      const outcomeRaw = m.HEAP32[outPtr >> 2];
      const kindRaw    = m.HEAP32[kindPtr >> 2];
      return {
        outcome: (outcomeRaw === 0 ? 0 : 1) as 0 | 1,
        outcomeKind: kindRaw === 0 ? "deterministic" : "random",
      };
    } finally {
      m._free(rngPtr); m._free(outPtr); m._free(kindPtr);
    }
  }

  /** Draw an n-bit computational-basis sample.  Mutates the
   *  tableau and advances the RNG state.  Caps at 64 qubits because
   *  the C entry point returns a single ``uint64_t``. */
  sampleAll(): SampleAllResult {
    if (this.nq > 64) {
      throw new RangeError(
        `sampleAll() supports up to 64 qubits; got ${this.nq}.  ` +
        `Use measure(q) in a loop for wider tableaus.`);
    }
    const m = this.mod as unknown as {
      _clifford_sample_all: (h: number, rngPtr: number, resultPtr: number) => number;
      _malloc: (n: number) => number;
      _free: (p: number) => void;
      HEAPU32: Uint32Array;
    };
    const rngPtr    = m._malloc(8);
    const resultPtr = m._malloc(8);
    try {
      const rngIdx = rngPtr >> 2;
      m.HEAPU32[rngIdx]     = Number(this.rngState & 0xffffffffn);
      m.HEAPU32[rngIdx + 1] = Number((this.rngState >> 32n) & 0xffffffffn);
      const rc = m._clifford_sample_all(this.handle, rngPtr, resultPtr);
      if (rc !== 0) {
        throw new Error(`clifford_sample_all rc=${rc}`);
      }
      const rngLo = BigInt(m.HEAPU32[rngIdx]);
      const rngHi = BigInt(m.HEAPU32[rngIdx + 1]);
      this.rngState = (rngHi << 32n) | rngLo;

      const resIdx = resultPtr >> 2;
      const lo = BigInt(m.HEAPU32[resIdx]);
      const hi = BigInt(m.HEAPU32[resIdx + 1]);
      return { bits: (hi << 32n) | lo };
    } finally {
      m._free(rngPtr); m._free(resultPtr);
    }
  }

  // ---- Internal helpers --------------------------------------------------

  private _call1(fnName: string, q: number): void {
    if (q < 0 || q >= this.nq) {
      throw new RangeError(`qubit ${q} out of [0, ${this.nq})`);
    }
    const m = this.mod as unknown as Record<string, (...a: number[]) => number>;
    const rc = m[`_${fnName}`](this.handle, q);
    if (rc !== 0) throw new Error(`${fnName}(${q}) rc=${rc}`);
  }

  private _call2(fnName: string, a: number, b: number): void {
    if (a < 0 || a >= this.nq || b < 0 || b >= this.nq) {
      throw new RangeError(`qubits ${a},${b} out of [0, ${this.nq})`);
    }
    const m = this.mod as unknown as Record<string, (...a: number[]) => number>;
    const rc = m[`_${fnName}`](this.handle, a, b);
    if (rc !== 0) throw new Error(`${fnName}(${a},${b}) rc=${rc}`);
  }
}
