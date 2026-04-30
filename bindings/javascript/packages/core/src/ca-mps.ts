/**
 * Clifford-Assisted MPS (CA-MPS) WebAssembly bindings.
 *
 * Hybrid `|psi> = D|phi>` representation that absorbs the Clifford
 * structure of a circuit into a tableau and only pushes non-Clifford
 * rotations into the MPS factor.  See `docs/research/ca_mps.md`.
 *
 * Highlights:
 *  - {@link CaMps} class: the state handle plus the Clifford and
 *    non-Clifford gate surface.
 *  - {@link varDRun}: variational-D ground-state search (alternating
 *    greedy Clifford + imag-time loop).
 *  - {@link gaugeWarmstart}: standalone Aaronson-Gottesman symplectic-
 *    Gauss-Jordan Clifford preparation on any abelian Pauli stabilizer
 *    subgroup.
 *  - {@link z2Lgt1dBuild} / {@link z2Lgt1dGaussLaw}: 1+1D Z2 lattice
 *    gauge theory Pauli sum + Gauss-law accessor.
 *  - {@link statusString}: diagnostic stringifier for any Moonlab
 *    status code.
 *
 * All symbols bind to the v0.2.1 stable ABI in
 * `src/applications/moonlab_export.h`.
 */

import { getModule } from './wasm-loader';

/** Warmstart strategies for {@link varDRun}. */
export enum Warmstart {
  Identity = 0,
  HAll = 1,
  DualTfim = 2,
  FerroTfim = 3,
  /** Gauge-aware: caller supplies commuting Pauli generators in
   * `stabPaulis` of {@link varDRun}. */
  StabilizerSubgroup = 4,
}

/** Module identifiers for {@link statusString}. */
export enum StatusModule {
  Generic = 0,
  CaMps = 1,
  CaMpsVarD = 2,
  CaMpsStabWarmstart = 3,
  CaPeps = 4,
  TnState = 5,
  TnGate = 6,
  TnMeasure = 7,
  Tensor = 8,
  Contract = 9,
  SvdCompress = 10,
  Clifford = 11,
  Partition = 12,
  DistGate = 13,
  MpiBridge = 14,
}

/** Pauli byte encoding: 0=I, 1=X, 2=Y, 3=Z. */
export type PauliByte = 0 | 1 | 2 | 3;

/** Configuration for {@link varDRun}. */
export interface VarDConfig {
  maxOuterIters?: number;
  imagTimeDtau?: number;
  imagTimeStepsPerOuter?: number;
  cliffordPassesPerOuter?: number;
  composite2gate?: boolean;
  warmstart?: Warmstart;
  /** Required when `warmstart === StabilizerSubgroup`.  Row-major
   * `(numGens, numQubits)` flat Uint8Array of Pauli bytes. */
  stabPaulis?: Uint8Array;
  /** Number of generator rows in `stabPaulis`. */
  stabNumGens?: number;
}

/** Result of a {@link z2Lgt1dBuild} call. */
export interface Z2LgtPauliSum {
  /** Row-major `(numTerms, numQubits)` flat Uint8Array. */
  paulis: Uint8Array;
  /** Length-`numTerms` Float64Array. */
  coeffs: Float64Array;
  /** Equals `2 * numMatterSites - 1`. */
  numQubits: number;
  /** Equals `paulis.length / numQubits`. */
  numTerms: number;
}

/** Owned WASM handle to a Clifford-Assisted MPS state. */
export class CaMps {
  /** WASM pointer (opaque). */
  private handle: number;
  private mod: ReturnType<typeof getModule> extends Promise<infer T> ? T : never;
  private freed: boolean = false;

  /** Internal: use {@link CaMps.create} instead. */
  private constructor(
    handle: number,
    mod: CaMps['mod'],
  ) {
    this.handle = handle;
    this.mod = mod;
  }

  /** Allocate a CA-MPS state on `numQubits` with the given truncation
   * cap. */
  static async create(
    numQubits: number,
    maxBondDim: number = 32,
  ): Promise<CaMps> {
    const mod = await getModule();
    const h = (mod as unknown as { _moonlab_ca_mps_create: (n: number, b: number) => number })
      ._moonlab_ca_mps_create(numQubits, maxBondDim);
    if (!h) throw new Error('moonlab_ca_mps_create returned NULL');
    return new CaMps(h, mod);
  }

  /** Release the underlying WASM allocation. */
  dispose(): void {
    if (this.freed) return;
    (this.mod as unknown as { _moonlab_ca_mps_free: (h: number) => void })
      ._moonlab_ca_mps_free(this.handle);
    this.freed = true;
  }

  get numQubits(): number {
    return (this.mod as unknown as { _moonlab_ca_mps_num_qubits: (h: number) => number })
      ._moonlab_ca_mps_num_qubits(this.handle);
  }

  get bondDim(): number {
    return (this.mod as unknown as { _moonlab_ca_mps_current_bond_dim: (h: number) => number })
      ._moonlab_ca_mps_current_bond_dim(this.handle);
  }

  get norm(): number {
    return (this.mod as unknown as { _moonlab_ca_mps_norm: (h: number) => number })
      ._moonlab_ca_mps_norm(this.handle);
  }

  // --- Clifford gates: tableau-only, no MPS cost. ---
  h(q: number): this { this.call1q('h', q); return this; }
  s(q: number): this { this.call1q('s', q); return this; }
  sdag(q: number): this { this.call1q('sdag', q); return this; }
  x(q: number): this { this.call1q('x', q); return this; }
  y(q: number): this { this.call1q('y', q); return this; }
  z(q: number): this { this.call1q('z', q); return this; }
  cnot(c: number, t: number): this { this.call2q('cnot', c, t); return this; }
  cz(a: number, b: number): this { this.call2q('cz', a, b); return this; }
  swap(a: number, b: number): this { this.call2q('swap', a, b); return this; }

  // --- Non-Clifford rotations: pushed into the MPS factor. ---
  rx(q: number, theta: number): this { this.call1qParam('rx', q, theta); return this; }
  ry(q: number, theta: number): this { this.call1qParam('ry', q, theta); return this; }
  rz(q: number, theta: number): this { this.call1qParam('rz', q, theta); return this; }
  tGate(q: number): this { this.call1q('t_gate', q); return this; }
  tDagger(q: number): this { this.call1q('t_dagger', q); return this; }
  phase(q: number, theta: number): this { this.call1qParam('phase', q, theta); return this; }

  /** Renormalise after non-unitary evolution (e.g. imag-time). */
  normalize(): this {
    const fn = `_moonlab_ca_mps_normalize` as const;
    const m = this.mod as unknown as Record<string, (...a: number[]) => number>;
    const rc = m[fn](this.handle);
    if (rc !== 0) {
      throw new Error(
        `moonlab_ca_mps_normalize -> ${statusString(StatusModule.CaMps, rc)}`);
    }
    return this;
  }

  /** Apply the gauge-aware warmstart Clifford in place.  See
   * {@link gaugeWarmstart}. */
  gaugeWarmstart(paulis: Uint8Array, numGens: number): this {
    if (paulis.length !== this.numQubits * numGens) {
      throw new Error(
        `gaugeWarmstart: paulis.length=${paulis.length} != numGens*numQubits=${this.numQubits * numGens}`);
    }
    const m = this.mod as unknown as {
      _moonlab_ca_mps_gauge_warmstart: (h: number, p: number, n: number) => number;
      HEAPU8: Uint8Array;
      _malloc: (n: number) => number;
      _free: (p: number) => void;
    };
    const ptr = m._malloc(paulis.length);
    try {
      m.HEAPU8.set(paulis, ptr);
      const rc = m._moonlab_ca_mps_gauge_warmstart(this.handle, ptr, numGens);
      if (rc !== 0) {
        throw new Error(
          `moonlab_ca_mps_gauge_warmstart -> ${statusString(StatusModule.CaMpsStabWarmstart, rc)}`);
      }
    } finally {
      m._free(ptr);
    }
    return this;
  }

  // --- Internal helpers ---
  private call1q(name: string, q: number): void {
    const fn = `_moonlab_ca_mps_${name}` as const;
    const m = this.mod as unknown as Record<string, (...a: number[]) => number>;
    const rc = m[fn](this.handle, q);
    if (rc !== 0) {
      throw new Error(
        `moonlab_ca_mps_${name}(${q}) -> ${statusString(StatusModule.CaMps, rc)}`);
    }
  }
  private call2q(name: string, a: number, b: number): void {
    const fn = `_moonlab_ca_mps_${name}` as const;
    const m = this.mod as unknown as Record<string, (...a: number[]) => number>;
    const rc = m[fn](this.handle, a, b);
    if (rc !== 0) {
      throw new Error(
        `moonlab_ca_mps_${name}(${a},${b}) -> ${statusString(StatusModule.CaMps, rc)}`);
    }
  }
  private call1qParam(name: string, q: number, theta: number): void {
    const fn = `_moonlab_ca_mps_${name}` as const;
    const m = this.mod as unknown as Record<string, (...a: number[]) => number>;
    const rc = m[fn](this.handle, q, theta);
    if (rc !== 0) {
      throw new Error(
        `moonlab_ca_mps_${name}(${q},${theta}) -> ${statusString(StatusModule.CaMps, rc)}`);
    }
  }

  /** Used by free functions in this module to access the WASM module. */
  _internal_module(): CaMps['mod'] { return this.mod; }
  _internal_handle(): number { return this.handle; }
}

/** Run the variational-D alternating ground-state search. */
export async function varDRun(
  state: CaMps,
  paulis: Uint8Array,
  coeffs: Float64Array,
  numTerms: number,
  cfg: VarDConfig = {},
): Promise<number> {
  const n = state.numQubits;
  if (paulis.length !== n * numTerms || coeffs.length !== numTerms) {
    throw new Error('varDRun: shape mismatch');
  }

  const m = state._internal_module() as unknown as {
    _moonlab_ca_mps_var_d_run: (
      h: number, p: number, c: number, T: number,
      maxOuter: number, dtau: number, imagSteps: number,
      cliffordPasses: number, composite: number, warmstart: number,
      stab: number, stabN: number, outE: number,
    ) => number;
    HEAPU8: Uint8Array;
    HEAPF64: Float64Array;
    _malloc: (n: number) => number;
    _free: (p: number) => void;
  };

  const pPtr = m._malloc(paulis.length);
  const cPtr = m._malloc(coeffs.length * 8);
  const ePtr = m._malloc(8);
  let stabPtr = 0;
  const stabBytes = (cfg.warmstart === Warmstart.StabilizerSubgroup
    && cfg.stabPaulis) ? cfg.stabPaulis.length : 0;
  if (stabBytes) {
    stabPtr = m._malloc(stabBytes);
    m.HEAPU8.set(cfg.stabPaulis!, stabPtr);
  }

  try {
    m.HEAPU8.set(paulis, pPtr);
    m.HEAPF64.set(coeffs, cPtr / 8);
    const rc = m._moonlab_ca_mps_var_d_run(
      state._internal_handle(),
      pPtr, cPtr, numTerms,
      cfg.maxOuterIters ?? 25,
      cfg.imagTimeDtau ?? 0.10,
      cfg.imagTimeStepsPerOuter ?? 4,
      cfg.cliffordPassesPerOuter ?? 8,
      cfg.composite2gate ? 1 : 0,
      cfg.warmstart ?? Warmstart.Identity,
      stabPtr,
      cfg.stabNumGens ?? 0,
      ePtr,
    );
    if (rc !== 0) {
      throw new Error(
        `moonlab_ca_mps_var_d_run -> ${statusString(StatusModule.CaMpsVarD, rc)}`);
    }
    return m.HEAPF64[ePtr / 8];
  } finally {
    m._free(pPtr);
    m._free(cPtr);
    m._free(ePtr);
    if (stabPtr) m._free(stabPtr);
  }
}

/** Apply the gauge-aware warmstart Clifford to a CA-MPS state. */
export function gaugeWarmstart(
  state: CaMps,
  paulis: Uint8Array,
  numGens: number,
): void {
  state.gaugeWarmstart(paulis, numGens);
}

/** Build the 1+1D Z2 LGT Pauli sum on N matter sites. */
export async function z2Lgt1dBuild(
  numMatterSites: number,
  tHop: number = 1.0,
  hLink: number = 1.0,
  mass: number = 0.0,
  gaussPenalty: number = 0.0,
): Promise<Z2LgtPauliSum> {
  const mod = await getModule();
  const m = mod as unknown as {
    _moonlab_z2_lgt_1d_build: (
      N: number, t: number, h: number, mass: number, gp: number,
      outP: number, outC: number, outT: number, outQ: number,
    ) => number;
    HEAPU8: Uint8Array;
    HEAPU32: Uint32Array;
    HEAPF64: Float64Array;
    _malloc: (n: number) => number;
    _free: (p: number) => void;
  };
  const outPP = m._malloc(4);   // pointer-to-pointer-to-uint8
  const outCC = m._malloc(4);   // pointer-to-pointer-to-double
  const outT = m._malloc(4);
  const outQ = m._malloc(4);
  try {
    const rc = m._moonlab_z2_lgt_1d_build(
      numMatterSites, tHop, hLink, mass, gaussPenalty,
      outPP, outCC, outT, outQ);
    if (rc !== 0) {
      throw new Error(`moonlab_z2_lgt_1d_build returned ${rc}`);
    }
    const pPtr = m.HEAPU32[outPP / 4];
    const cPtr = m.HEAPU32[outCC / 4];
    const T = m.HEAPU32[outT / 4];
    const Q = m.HEAPU32[outQ / 4];
    const total = T * Q;
    const paulis = new Uint8Array(m.HEAPU8.buffer, pPtr, total).slice();
    const coeffs = new Float64Array(m.HEAPF64.buffer, cPtr, T).slice();
    m._free(pPtr);
    m._free(cPtr);
    return { paulis, coeffs, numQubits: Q, numTerms: T };
  } finally {
    m._free(outPP);
    m._free(outCC);
    m._free(outT);
    m._free(outQ);
  }
}

/** Return the interior Gauss-law operator at matter site `siteX`. */
export async function z2Lgt1dGaussLaw(
  numMatterSites: number,
  siteX: number,
): Promise<Uint8Array> {
  const mod = await getModule();
  const nq = 2 * numMatterSites - 1;
  const m = mod as unknown as {
    _moonlab_z2_lgt_1d_gauss_law: (N: number, x: number, p: number) => number;
    HEAPU8: Uint8Array;
    _malloc: (n: number) => number;
    _free: (p: number) => void;
  };
  const ptr = m._malloc(nq);
  try {
    const rc = m._moonlab_z2_lgt_1d_gauss_law(numMatterSites, siteX, ptr);
    if (rc !== 0) {
      throw new Error(`moonlab_z2_lgt_1d_gauss_law(N=${numMatterSites}, x=${siteX}) returned ${rc}`);
    }
    return new Uint8Array(m.HEAPU8.buffer, ptr, nq).slice();
  } finally {
    m._free(ptr);
  }
}

/** Pretty-print a Moonlab status code.  Synchronous since the WASM
 * module is expected to already be loaded by the time errors occur. */
export function statusString(module: StatusModule, status: number): string {
  // Best-effort: if the module isn't loaded yet, return a generic
  // string rather than blocking on async load.
  try {
    // `getModule` returns a Promise; we can't await here without making
    // this function async.  Use the cached module from a recent CaMps
    // create -- callers that hit this path before any allocation will
    // get a generic string.
    const cached = (globalThis as unknown as {
      __moonlab_module?: unknown;
    }).__moonlab_module as
      | undefined
      | { _moonlab_status_string?: (m: number, s: number) => number;
          UTF8ToString?: (p: number) => string };
    if (cached?._moonlab_status_string && cached.UTF8ToString) {
      const p = cached._moonlab_status_string(module, status);
      return cached.UTF8ToString(p);
    }
  } catch { /* fallthrough */ }
  return `<module=${module} status=${status}>`;
}
