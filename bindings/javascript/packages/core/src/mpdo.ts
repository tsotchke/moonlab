/**
 * Matrix-Product Density Operator (MPDO) mixed-state simulator
 * binding (C-side since v0.3.0, JS binding since v0.4.10).
 *
 * Wraps ``src/quantum/noise_mpdo.{c,h}`` so JavaScript can simulate
 * open-system dynamics with single-qubit Kraus channels on an
 * MPS-compressed density matrix.  Mirrors the v0.3.0 Python
 * ``moonlab.mpdo.MPDO`` and Rust ``moonlab::mpdo::Mpdo`` surfaces.
 *
 * Channel parameter conventions match
 * ``src/quantum/noise.h`` so the noise model stays consistent
 * across MPDO and pure-state simulation paths.  Bond dimensions
 * start at ``chi = 1`` (product state) and grow with applied
 * channels, capped at ``maxBondDim``.
 *
 * @example
 * ```typescript
 * import { Mpdo, PauliCode } from '@moonlab/quantum-core';
 *
 * const rho = await Mpdo.create(4, 16);  // 4 qubits, chi_max = 16
 * rho.applyDepolarizing(0, 0.05);
 * rho.applyAmplitudeDamping(1, 0.1);
 *
 * const z0 = rho.expectPauli(0, PauliCode.Z);
 * console.log(`<Z_0> = ${z0.toFixed(6)}`);
 * console.log(`Tr(rho) = ${rho.trace().toFixed(6)}`); // ~1
 *
 * rho.dispose();
 * ```
 */

import { getModule } from './wasm-loader';

/** Pauli operator code for {@link Mpdo.expectPauli}. */
export enum PauliCode {
  I = 0,
  X = 1,
  Y = 2,
  Z = 3,
}

type MpdoModule = {
  _moonlab_mpdo_create: (numQubits: number, maxBondDim: number) => number;
  _moonlab_mpdo_free: (h: number) => void;
  _moonlab_mpdo_clone: (h: number) => number;
  _moonlab_mpdo_num_qubits: (h: number) => number;
  _moonlab_mpdo_max_bond_dim: (h: number) => number;
  _moonlab_mpdo_current_bond_dim: (h: number) => number;
  _moonlab_mpdo_trace: (h: number) => number;
  _moonlab_mpdo_apply_depolarizing_1q: (h: number, q: number, p: number) => number;
  _moonlab_mpdo_apply_amplitude_damping_1q: (h: number, q: number, g: number) => number;
  _moonlab_mpdo_apply_phase_damping_1q: (h: number, q: number, l: number) => number;
  _moonlab_mpdo_apply_bit_flip_1q: (h: number, q: number, p: number) => number;
  _moonlab_mpdo_apply_phase_flip_1q: (h: number, q: number, p: number) => number;
  _moonlab_mpdo_apply_bit_phase_flip_1q: (h: number, q: number, p: number) => number;
  _moonlab_mpdo_apply_kraus_1q: (
    h: number,
    q: number,
    krausPtr: number,
    numKraus: number,
  ) => number;
  _moonlab_mpdo_expect_pauli_1q: (
    h: number,
    q: number,
    pauli: number,
    outPtr: number,
  ) => number;
  _malloc: (n: number) => number;
  _free: (p: number) => void;
  HEAPF64: Float64Array;
};

/**
 * Owned WASM handle to an MPDO mixed-state simulator.
 *
 * Build with {@link create} and tear down with {@link dispose}.
 * Channels and observables mirror the Python and Rust surfaces;
 * see the module-level example for usage.
 */
export class Mpdo {
  private handle: number;
  private mod: MpdoModule;
  private nq: number;
  private chiMax: number;
  private freed = false;

  private constructor(
    handle: number,
    mod: MpdoModule,
    numQubits: number,
    maxBondDim: number,
  ) {
    this.handle = handle;
    this.mod = mod;
    this.nq = numQubits;
    this.chiMax = maxBondDim;
  }

  /** Allocate a product-state MPDO on ``numQubits`` qubits with
   *  bond-dimension cap ``maxBondDim``.  The initial state is
   *  `|0...0> <0...0|`. */
  static async create(numQubits: number, maxBondDim: number): Promise<Mpdo> {
    if (numQubits < 1) {
      throw new RangeError(`numQubits must be >= 1, got ${numQubits}`);
    }
    if (maxBondDim < 1) {
      throw new RangeError(`maxBondDim must be >= 1, got ${maxBondDim}`);
    }
    const mod = (await getModule()) as unknown as MpdoModule;
    const h = mod._moonlab_mpdo_create(numQubits, maxBondDim);
    if (!h) throw new Error('moonlab_mpdo_create returned NULL');
    return new Mpdo(h, mod, numQubits, maxBondDim);
  }

  /** Release the underlying WASM allocation. */
  dispose(): void {
    if (this.freed) return;
    this.mod._moonlab_mpdo_free(this.handle);
    this.freed = true;
  }

  /** Deep copy of this MPDO.  Caller owns the returned handle. */
  clone(): Mpdo {
    this._guardLive();
    const h = this.mod._moonlab_mpdo_clone(this.handle);
    if (!h) throw new Error('moonlab_mpdo_clone returned NULL');
    return new Mpdo(h, this.mod, this.nq, this.chiMax);
  }

  /** Number of qubits in the MPDO. */
  get numQubits(): number { return this.nq; }

  /** Bond-dimension cap configured at construction. */
  get maxBondDim(): number { return this.chiMax; }

  /** Current maximum bond dimension across all bonds.  Grows as
   *  channels are applied, capped by {@link maxBondDim}. */
  get currentBondDim(): number {
    this._guardLive();
    return this.mod._moonlab_mpdo_current_bond_dim(this.handle);
  }

  /** Trace of the density matrix, Tr(rho).  Should be 1 for a
   *  properly normalised state; deviations indicate truncation
   *  error. */
  trace(): number {
    this._guardLive();
    return this.mod._moonlab_mpdo_trace(this.handle);
  }

  // ---- Named single-qubit channels -------------------------------------

  /** Single-qubit depolarising channel,
   *  `rho |-> (1-p) rho + (p/3) (X rho X + Y rho Y + Z rho Z)`. */
  applyDepolarizing(qubit: number, probability: number): this {
    this._guardQubit(qubit);
    this._check(
      this.mod._moonlab_mpdo_apply_depolarizing_1q(this.handle, qubit, probability),
      'apply_depolarizing_1q',
    );
    return this;
  }

  /** Amplitude damping (T1 relaxation) with parameter ``gamma``. */
  applyAmplitudeDamping(qubit: number, gamma: number): this {
    this._guardQubit(qubit);
    this._check(
      this.mod._moonlab_mpdo_apply_amplitude_damping_1q(this.handle, qubit, gamma),
      'apply_amplitude_damping_1q',
    );
    return this;
  }

  /** Phase damping (pure dephasing) with parameter ``lambda``. */
  applyPhaseDamping(qubit: number, lambda: number): this {
    this._guardQubit(qubit);
    this._check(
      this.mod._moonlab_mpdo_apply_phase_damping_1q(this.handle, qubit, lambda),
      'apply_phase_damping_1q',
    );
    return this;
  }

  /** Bit-flip channel `(1-p) I + p X`. */
  applyBitFlip(qubit: number, probability: number): this {
    this._guardQubit(qubit);
    this._check(
      this.mod._moonlab_mpdo_apply_bit_flip_1q(this.handle, qubit, probability),
      'apply_bit_flip_1q',
    );
    return this;
  }

  /** Phase-flip channel `(1-p) I + p Z`. */
  applyPhaseFlip(qubit: number, probability: number): this {
    this._guardQubit(qubit);
    this._check(
      this.mod._moonlab_mpdo_apply_phase_flip_1q(this.handle, qubit, probability),
      'apply_phase_flip_1q',
    );
    return this;
  }

  /** Bit-phase-flip channel `(1-p) I + p Y`. */
  applyBitPhaseFlip(qubit: number, probability: number): this {
    this._guardQubit(qubit);
    this._check(
      this.mod._moonlab_mpdo_apply_bit_phase_flip_1q(this.handle, qubit, probability),
      'apply_bit_phase_flip_1q',
    );
    return this;
  }

  /** Apply an arbitrary single-qubit Kraus channel.  ``kraus`` is a
   *  flat row-major interleaved-complex array of length
   *  ``numKraus * 4 * 2`` (4 entries per 2x2 matrix, 2 doubles per
   *  complex). */
  applyKraus(qubit: number, kraus: Float64Array, numKraus: number): this {
    this._guardQubit(qubit);
    if (kraus.length !== numKraus * 4 * 2) {
      throw new RangeError(
        `kraus.length=${kraus.length} mismatches numKraus * 4 * 2 = ${numKraus * 4 * 2}`,
      );
    }
    const bytes = kraus.length * 8;
    const ptr = this.mod._malloc(bytes);
    try {
      // Copy doubles into WASM heap; HEAPF64 indexes by 8 bytes.
      const idx = ptr >> 3;
      for (let i = 0; i < kraus.length; i++) {
        this.mod.HEAPF64[idx + i] = kraus[i];
      }
      this._check(
        this.mod._moonlab_mpdo_apply_kraus_1q(this.handle, qubit, ptr, numKraus),
        'apply_kraus_1q',
      );
    } finally {
      this.mod._free(ptr);
    }
    return this;
  }

  // ---- Observables -----------------------------------------------------

  /** Single-site Pauli expectation `Tr(rho * P_q)` where ``P`` is
   *  one of `{I, X, Y, Z}` on qubit ``qubit`` and identity
   *  elsewhere. */
  expectPauli(qubit: number, pauli: PauliCode): number {
    this._guardQubit(qubit);
    const outPtr = this.mod._malloc(8);
    try {
      this._check(
        this.mod._moonlab_mpdo_expect_pauli_1q(
          this.handle, qubit, pauli, outPtr,
        ),
        'expect_pauli_1q',
      );
      return this.mod.HEAPF64[outPtr >> 3];
    } finally {
      this.mod._free(outPtr);
    }
  }

  // ---- Internal helpers ------------------------------------------------

  private _guardLive(): void {
    if (this.freed) throw new Error('Mpdo is disposed');
  }

  private _guardQubit(q: number): void {
    this._guardLive();
    if (q < 0 || q >= this.nq) {
      throw new RangeError(`qubit ${q} out of [0, ${this.nq})`);
    }
  }

  private _check(rc: number, fn: string): void {
    if (rc !== 0) throw new Error(`moonlab_mpdo_${fn} rc=${rc}`);
  }
}
