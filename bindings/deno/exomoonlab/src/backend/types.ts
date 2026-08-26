/**
 * The backend seam.
 *
 * MoonLab's simulation core is reachable two ways: natively, through the
 * stable C ABI (`MOONLAB_API`, ABI 0.7.0), and in a browser, through the
 * Emscripten build that re-exports that same C surface. Both call the *same*
 * functions, which is why exomoonlab can be one application rather than two.
 *
 * This module is the only place that fact is expressed as a type. Everything
 * above it -- windows, painters, themes -- talks to `MoonLabBackend` and never
 * learns which implementation it got.
 *
 * Every operation is async on purpose. `ShellApp.frame()` is synchronous and
 * runs every frame; a 20-qubit run cannot happen inside it. Making the seam
 * async at the type level means no caller can accidentally block the frame
 * loop, even when a given implementation happens to answer immediately.
 */

/** Which implementation answered. */
export type BackendKind = "native" | "wasm";

/**
 * What a backend can actually do, resolved by probing rather than assumed.
 * A caller treats an absent capability exactly like `false`.
 */
export interface BackendCapabilities {
  /** Largest qubit count this backend will attempt. */
  readonly maxQubits: number;
  /** `quantum_state_create`/`_destroy` are exported (ULG addition). */
  readonly allocatingConstructor: boolean;
  /** Exported function count, where the backend can report one. */
  readonly exportedFunctions?: number;
  /**
   * Berry curvature and Chern numbers are reachable.
   *
   * Native builds carry the whole quantum-geometry module; the WASM build
   * only does when it was compiled after those symbols were added to
   * `emscripten/exports.txt`. A window that needs it asks first and says so
   * plainly when the answer is no, rather than drawing an empty grid.
   */
  readonly bandGeometry: boolean;
  /** `quantum_state_from_amplitudes` is reachable. */
  readonly amplitudeUpload: boolean;
}

/** A two-band model with an analytic d-vector, so curvature is exact. */
export type BandModel =
  | { readonly kind: "qwz"; readonly m: number }
  | {
    readonly kind: "haldane";
    readonly t1: number;
    readonly t2: number;
    readonly phi: number;
    readonly mStagger: number;
  };

/** Berry curvature sampled over the Brillouin zone, with its integral. */
export interface BerryGrid {
  /** Momenta per axis; the field is `n * n`, row-major. */
  readonly n: number;
  readonly curvature: Float64Array;
  /**
   * The integrated Chern number. Exact at any finite `n` provided the band
   * stays gapped across the grid (Fukui-Hatsugai-Suzuki), which is why a
   * coarse grid still reports a clean integer.
   */
  readonly chern: number;
  readonly min: number;
  readonly max: number;
}

/**
 * An opaque handle to a live quantum state.
 *
 * The pointer stays inside the implementation deliberately: a native pointer
 * and a WASM heap offset are not the same kind of thing, and nothing above
 * this seam should be in a position to confuse them.
 */
export interface StateHandle {
  readonly numQubits: number;
  readonly stateDim: number;
}

/**
 * The narrow surface phase 1 needs: state lifecycle, the gates required to
 * build a Bell state, and the read-outs the equivalence harness compares.
 *
 * Deliberately small. It grows when a window needs something, not before.
 */
export interface MoonLabBackend {
  readonly kind: BackendKind;
  /** Human-readable provenance -- which library or artifact answered. */
  readonly description: string;
  readonly capabilities: BackendCapabilities;

  createState(numQubits: number): Promise<StateHandle>;
  destroyState(state: StateHandle): Promise<void>;
  reset(state: StateHandle): Promise<void>;
  normalize(state: StateHandle): Promise<void>;

  hadamard(state: StateHandle, qubit: number): Promise<void>;
  pauliX(state: StateHandle, qubit: number): Promise<void>;
  pauliZ(state: StateHandle, qubit: number): Promise<void>;
  cnot(state: StateHandle, control: number, target: number): Promise<void>;

  /**
   * Overwrites the state with a supplied amplitude vector.
   *
   * `amplitudes` is interleaved real/imaginary, length `2 * stateDim`, matching
   * the C `complex_t*`. Present only when
   * `capabilities.amplitudeUpload` is true.
   */
  loadAmplitudes?(state: StateHandle, amplitudes: Float64Array): Promise<void>;

  /** Probability of one computational basis state. */
  probability(state: StateHandle, basisIndex: number): Promise<number>;
  /** Probability that a single qubit measures 1, without collapsing. */
  probabilityOne(state: StateHandle, qubit: number): Promise<number>;
  entropy(state: StateHandle): Promise<number>;
  purity(state: StateHandle): Promise<number>;

  /**
   * Berry curvature over the Brillouin zone for a two-band model.
   *
   * Present only when `capabilities.bandGeometry` is true.
   */
  berryGrid?(model: BandModel, n: number): Promise<BerryGrid>;

  /** Releases the library handle. Safe to call twice. */
  dispose(): Promise<void>;
}

/** Raised when a backend cannot be constructed; carries why, for the probe. */
export class BackendUnavailableError extends Error {
  constructor(kind: BackendKind, reason: string, options?: { cause?: unknown }) {
    super(`${kind} backend unavailable: ${reason}`, options);
    this.name = "BackendUnavailableError";
  }
}
