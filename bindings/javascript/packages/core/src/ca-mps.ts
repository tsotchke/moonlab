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

/**
 * Module-level cache of the loaded WASM instance, populated on the
 * first {@link CaMps.create} or {@link z2Lgt1dBuild} call.  Used by
 * {@link statusString} to remain synchronous (the underlying status
 * code lookup needs the WASM module's `_moonlab_status_to_string` and
 * `UTF8ToString`).
 */
let _cachedModule: unknown = null;
async function _resolveModule(): Promise<unknown> {
  if (_cachedModule) return _cachedModule;
  _cachedModule = await getModule();
  return _cachedModule;
}

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
    const mod = await _resolveModule();
    const h = (mod as unknown as { _moonlab_ca_mps_create: (n: number, b: number) => number })
      ._moonlab_ca_mps_create(numQubits, maxBondDim);
    if (!h) throw new Error('moonlab_ca_mps_create returned NULL');
    return new CaMps(h, mod as CaMps['mod']);
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
      // The direct internal name; the moonlab_* wrapper lives in
      // moonlab_qrng_export.c which we keep out of the WASM build.
      _moonlab_ca_mps_apply_stab_subgroup_warmstart: (
        h: number, p: number, n: number,
      ) => number;
      HEAPU8: Uint8Array;
      _malloc: (n: number) => number;
      _free: (p: number) => void;
    };
    const ptr = m._malloc(paulis.length);
    try {
      m.HEAPU8.set(paulis, ptr);
      const rc = m._moonlab_ca_mps_apply_stab_subgroup_warmstart(
        this.handle, ptr, numGens);
      if (rc !== 0) {
        throw new Error(
          `gaugeWarmstart -> ${statusString(StatusModule.CaMpsStabWarmstart, rc)}`);
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

/** Detailed result of {@link varDRun}. */
export interface VarDResult {
  /** Final variational energy `<psi|H|psi>` after the alternating loop. */
  finalEnergy: number;
  /** Initial variational energy (before any updates). */
  initialEnergy: number;
  /** Max half-cut entanglement entropy of `|phi>` at the start. */
  initialPhiEntropy: number;
  /** Max half-cut entanglement entropy of `|phi>` at the end. */
  finalPhiEntropy: number;
  /** Total Clifford gates accepted across all outer iterations. */
  totalGatesAdded: number;
  /** Outer iterations actually executed (>= 1, <= maxOuterIters). */
  outerIterations: number;
  /** True if the loop converged (energy delta below convergenceEps)
   * rather than hitting the iter cap. */
  converged: boolean;
}

/* WASM32 layout of `ca_mps_var_d_alt_config_t` (56 bytes total):
 *   [ 0..4 )  int     max_outer_iters
 *   [ 8..16)  double  imag_time_dtau
 *   [16..20)  int     imag_time_steps_per_outer
 *   [20..24)  int     clifford_passes_per_outer
 *   [24..32)  double  convergence_eps
 *   [32..36)  int     include_2q_gates
 *   [36..40)  int     composite_2gate
 *   [40..44)  enum    warmstart
 *   [44..48)  ptr     warmstart_stab_paulis
 *   [48..52)  u32     warmstart_stab_num_gens
 *   [52..56)  int     verbose
 */
const VAR_D_ALT_CONFIG_BYTES = 56;
/* WASM32 layout of `ca_mps_var_d_alt_result_t` (48 bytes incl. tail pad):
 *   [ 0.. 8) double  initial_energy
 *   [ 8..16) double  final_energy
 *   [16..24) double  initial_phi_entropy
 *   [24..32) double  final_phi_entropy
 *   [32..36) int     total_gates_added
 *   [36..40) int     outer_iterations
 *   [40..44) int     converged
 */
const VAR_D_ALT_RESULT_BYTES = 48;

/** Run the variational-D alternating ground-state search.
 *
 * Wires directly to `moonlab_ca_mps_optimize_var_d_alternating` in
 * `ca_mps_var_d.c`.  The config + result structs are marshalled into
 * the WASM heap by hand using the layouts pinned at the top of this
 * file -- adjust them here if the C struct ever grows a field.
 *
 * Returns a {@link VarDResult} summary; the most-accessed value
 * (final energy) is also written through the resolver.
 */
export async function varDRun(
  state: CaMps,
  paulis: Uint8Array,
  coeffs: Float64Array,
  numTerms: number,
  cfg: VarDConfig = {},
): Promise<VarDResult> {
  const n = state.numQubits;
  if (paulis.length !== n * numTerms || coeffs.length !== numTerms) {
    throw new Error(
      `varDRun: paulis.length=${paulis.length} or coeffs.length=${coeffs.length}` +
      ` does not match num_terms=${numTerms} * num_qubits=${n}`);
  }

  const m = state._internal_module() as unknown as {
    _moonlab_ca_mps_optimize_var_d_alternating: (
      state_ptr: number,
      paulis_ptr: number,
      coeffs_ptr: number,
      num_terms: number,
      config_ptr: number,
      result_ptr: number,
    ) => number;
    HEAPU8: Uint8Array;
    HEAP32: Int32Array;
    HEAPU32: Uint32Array;
    HEAPF64: Float64Array;
    _malloc: (n: number) => number;
    _free: (p: number) => void;
  };

  // Allocate input buffers.
  const paulisPtr = m._malloc(paulis.length);
  const coeffsPtr = m._malloc(coeffs.length * 8);
  const configPtr = m._malloc(VAR_D_ALT_CONFIG_BYTES);
  const resultPtr = m._malloc(VAR_D_ALT_RESULT_BYTES);

  let stabPaulisPtr = 0;
  const wantStab = cfg.warmstart === Warmstart.StabilizerSubgroup;
  const stabBytes = (wantStab && cfg.stabPaulis) ? cfg.stabPaulis.length : 0;
  if (wantStab && (!cfg.stabPaulis || cfg.stabNumGens === undefined)) {
    m._free(paulisPtr);
    m._free(coeffsPtr);
    m._free(configPtr);
    m._free(resultPtr);
    throw new Error(
      'varDRun: warmstart=StabilizerSubgroup requires stabPaulis + stabNumGens');
  }
  if (stabBytes) {
    stabPaulisPtr = m._malloc(stabBytes);
  }

  try {
    // Copy Pauli + coeffs into WASM heap.
    m.HEAPU8.set(paulis, paulisPtr);
    m.HEAPF64.set(coeffs, coeffsPtr / 8);
    if (stabPaulisPtr) {
      m.HEAPU8.set(cfg.stabPaulis!, stabPaulisPtr);
    }

    // Marshal the config struct.  Match `ca_mps_var_d_alt_config_default`.
    const i32 = m.HEAP32;
    const u32 = m.HEAPU32;
    const f64 = m.HEAPF64;
    const cfg32 = configPtr >>> 2;
    const cfgF64 = configPtr >>> 3;
    // Zero the struct first to put predictable bytes in the padding.
    m.HEAPU8.fill(0, configPtr, configPtr + VAR_D_ALT_CONFIG_BYTES);

    i32[cfg32 + 0]  = cfg.maxOuterIters ?? 30;                 // offset 0
    f64[cfgF64 + 1] = cfg.imagTimeDtau ?? 0.10;                 // offset 8
    i32[cfg32 + 4]  = cfg.imagTimeStepsPerOuter ?? 5;           // offset 16
    i32[cfg32 + 5]  = cfg.cliffordPassesPerOuter ?? 10;         // offset 20
    f64[cfgF64 + 3] = 1e-7;                                     // offset 24 (convergenceEps)
    i32[cfg32 + 8]  = 1;                                        // offset 32 (include_2q_gates)
    i32[cfg32 + 9]  = cfg.composite2gate ? 1 : 0;               // offset 36
    i32[cfg32 + 10] = cfg.warmstart ?? Warmstart.Identity;      // offset 40
    u32[cfg32 + 11] = stabPaulisPtr;                            // offset 44
    u32[cfg32 + 12] = wantStab ? (cfg.stabNumGens ?? 0) : 0;    // offset 48
    i32[cfg32 + 13] = 0;                                        // offset 52 (verbose)

    // Zero the result struct so an early-out leaves predictable values.
    m.HEAPU8.fill(0, resultPtr, resultPtr + VAR_D_ALT_RESULT_BYTES);

    const rc = m._moonlab_ca_mps_optimize_var_d_alternating(
      state._internal_handle(),
      paulisPtr,
      coeffsPtr,
      numTerms,
      configPtr,
      resultPtr,
    );

    if (rc !== 0) {
      throw new Error(
        `varDRun (optimize_var_d_alternating) -> ${statusString(StatusModule.CaMpsVarD, rc)}`);
    }

    // Read the result struct.
    const resF64 = resultPtr >>> 3;
    const res32  = resultPtr >>> 2;
    return {
      initialEnergy:     f64[resF64 + 0],
      finalEnergy:       f64[resF64 + 1],
      initialPhiEntropy: f64[resF64 + 2],
      finalPhiEntropy:   f64[resF64 + 3],
      totalGatesAdded:   i32[res32  + 8],
      outerIterations:   i32[res32  + 9],
      converged:         i32[res32  + 10] !== 0,
    };
  } finally {
    m._free(paulisPtr);
    m._free(coeffsPtr);
    m._free(configPtr);
    m._free(resultPtr);
    if (stabPaulisPtr) m._free(stabPaulisPtr);
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

/* WASM32 layout of `z2_lgt_config_t` (40 bytes total):
 *   [ 0.. 4) u32     num_matter_sites
 *   [ 8..16) double  t_hop
 *   [16..24) double  h_link
 *   [24..32) double  mass
 *   [32..40) double  gauss_penalty
 */
const Z2_LGT_CONFIG_BYTES = 40;

/** Allocate + fill a `z2_lgt_config_t` in WASM heap.  Caller frees. */
function fillZ2Config(
  m: {
    HEAPU8: Uint8Array;
    HEAPU32: Uint32Array;
    HEAPF64: Float64Array;
    _malloc: (n: number) => number;
  },
  numMatterSites: number,
  tHop: number,
  hLink: number,
  mass: number,
  gaussPenalty: number,
): number {
  const ptr = m._malloc(Z2_LGT_CONFIG_BYTES);
  m.HEAPU8.fill(0, ptr, ptr + Z2_LGT_CONFIG_BYTES);
  m.HEAPU32[(ptr >>> 2) + 0] = numMatterSites;   // offset 0
  const f64 = ptr >>> 3;
  m.HEAPF64[f64 + 1] = tHop;                      // offset 8
  m.HEAPF64[f64 + 2] = hLink;                     // offset 16
  m.HEAPF64[f64 + 3] = mass;                      // offset 24
  m.HEAPF64[f64 + 4] = gaussPenalty;              // offset 32
  return ptr;
}

/** Build the 1+1D Z2 LGT Pauli sum on N matter sites.
 *
 * Wires directly to `z2_lgt_1d_build_pauli_sum` in `lattice_z2_1d.c`
 * via a stack-allocated `z2_lgt_config_t` in the WASM heap.  The C
 * builder allocates `out_paulis` and `out_coeffs` via `calloc`; we
 * copy them out into JS-owned `Uint8Array` / `Float64Array` and
 * `free` the C-side allocations.
 */
export async function z2Lgt1dBuild(
  numMatterSites: number,
  tHop: number = 1.0,
  hLink: number = 1.0,
  mass: number = 0.0,
  gaussPenalty: number = 0.0,
): Promise<Z2LgtPauliSum> {
  const mod = await _resolveModule();
  const m = mod as unknown as {
    _z2_lgt_1d_build_pauli_sum: (
      cfgPtr: number, outP: number, outC: number, outT: number, outQ: number,
    ) => number;
    HEAPU8: Uint8Array;
    HEAPU32: Uint32Array;
    HEAPF64: Float64Array;
    _malloc: (n: number) => number;
    _free: (p: number) => void;
  };
  const cfgPtr = fillZ2Config(m, numMatterSites, tHop, hLink, mass, gaussPenalty);
  const outPP = m._malloc(4);   // pointer-to-pointer-to-uint8
  const outCC = m._malloc(4);   // pointer-to-pointer-to-double
  const outT = m._malloc(4);
  const outQ = m._malloc(4);
  try {
    const rc = m._z2_lgt_1d_build_pauli_sum(cfgPtr, outPP, outCC, outT, outQ);
    if (rc !== 0) {
      throw new Error(`z2_lgt_1d_build_pauli_sum returned ${rc}`);
    }
    const pPtr = m.HEAPU32[outPP / 4];
    const cPtr = m.HEAPU32[outCC / 4];
    const T = m.HEAPU32[outT / 4];
    const Q = m.HEAPU32[outQ / 4];
    const total = T * Q;
    // Slice copies into JS-owned arrays so we can free the C heap
    // backing immediately.
    const paulis = new Uint8Array(m.HEAPU8.buffer, pPtr, total).slice();
    const coeffs = new Float64Array(m.HEAPF64.buffer, cPtr, T).slice();
    m._free(pPtr);
    m._free(cPtr);
    return { paulis, coeffs, numQubits: Q, numTerms: T };
  } finally {
    m._free(cfgPtr);
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
  const mod = await _resolveModule();
  const nq = 2 * numMatterSites - 1;
  const m = mod as unknown as {
    _z2_lgt_1d_gauss_law_pauli: (cfgPtr: number, x: number, p: number) => number;
    HEAPU8: Uint8Array;
    HEAPU32: Uint32Array;
    HEAPF64: Float64Array;
    _malloc: (n: number) => number;
    _free: (p: number) => void;
  };
  // Internal entry takes a config-struct pointer; only num_matter_sites
  // is read for the gauss-law accessor, but we fill all fields with
  // sensible zeros to be safe.
  const cfgPtr = fillZ2Config(m, numMatterSites, 0, 0, 0, 0);
  const ptr = m._malloc(nq);
  try {
    const rc = m._z2_lgt_1d_gauss_law_pauli(cfgPtr, siteX, ptr);
    if (rc !== 0) {
      throw new Error(
        `z2_lgt_1d_gauss_law_pauli(N=${numMatterSites}, x=${siteX}) returned ${rc}`);
    }
    return new Uint8Array(m.HEAPU8.buffer, ptr, nq).slice();
  } finally {
    m._free(ptr);
    m._free(cfgPtr);
  }
}

/** Pretty-print a Moonlab status code.
 *
 * Wires directly to `moonlab_status_to_string` in `moonlab_status.c`.
 * Synchronous: relies on the module-level WASM cache populated on the
 * first {@link CaMps.create} (or other CA-MPS API call).  If called
 * cold (before any allocation), returns a `<module=N status=K>`
 * placeholder rather than blocking on async load.
 */
export function statusString(module: StatusModule, status: number): string {
  try {
    const cached = _cachedModule as
      | null
      | { _moonlab_status_to_string?: (m: number, s: number) => number;
          UTF8ToString?: (p: number) => string };
    if (cached?._moonlab_status_to_string && cached.UTF8ToString) {
      const p = cached._moonlab_status_to_string(module, status);
      return cached.UTF8ToString(p);
    }
  } catch { /* fallthrough */ }
  return `<module=${module} status=${status}>`;
}
