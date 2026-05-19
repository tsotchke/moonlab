/**
 * Bell-inequality test bindings (C-side since v0.2.0; JS binding
 * since v0.5.4).
 *
 * Wraps ``src/algorithms/bell_tests.{c,h}`` so JavaScript callers
 * can prepare any of the four canonical Bell pairs, run the CHSH
 * inequality at Tsirelson-optimal angles, and run the multi-qubit
 * Mermin variants (GHZ + Klyshko).  Mirrors the Python
 * ``moonlab.algorithms.BellTest`` and Rust ``moonlab::bell``
 * surfaces.
 *
 * The C-side tests sample measurement outcomes through a
 * ``quantum_entropy_ctx_t``; this module leases one for each call
 * via an internal RAII helper and frees it on return.  In the WASM
 * build the entropy stack is backed by ``getentropy(3)`` which
 * emscripten polyfills via ``crypto.getRandomValues()``.
 *
 * @example
 * ```typescript
 * import { QuantumState, BellState, chshTest } from '@moonlab/quantum-core';
 *
 * const state = await QuantumState.create({ numQubits: 2 });
 * await createBellState(state, 0, 1, BellState.PhiPlus);
 * const r = await chshTest(state, 0, 1, 4000);
 * console.log(`CHSH S = ${r.chshValue.toFixed(4)} (Tsirelson = ${r.quantumBound.toFixed(4)})`);
 * state.dispose();
 * ```
 */

import { getModule } from './wasm-loader';
import { QuantumState } from './quantum-state';

/** Bell-state index for {@link createBellState}.  Mirrors
 *  ``bell_state_type_t`` in the C header. */
export enum BellState {
  /** `|Phi+> = (|00> + |11>) / sqrt(2)` -- the canonical Bell pair. */
  PhiPlus = 0,
  /** `|Phi-> = (|00> - |11>) / sqrt(2)`. */
  PhiMinus = 1,
  /** `|Psi+> = (|01> + |10>) / sqrt(2)`. */
  PsiPlus = 2,
  /** `|Psi-> = (|01> - |10>) / sqrt(2)` -- the singlet. */
  PsiMinus = 3,
}

/** Result of a Bell-test run.  Mirrors ``bell_test_result_t`` in the
 *  C header. */
export interface BellTestResult {
  /** CHSH `S` (or Mermin `|M|` for the GHZ variant). */
  chshValue: number;
  /** `E(a, b)`. */
  correlationAB: number;
  /** `E(a, b')`. */
  correlationABprime: number;
  /** `E(a', b)`. */
  correlationAprimeB: number;
  /** `E(a', b')`. */
  correlationAprimeBprime: number;
  /** Classical (LHV) bound; 2.0 for CHSH, 2.0 for Mermin-GHZ. */
  classicalBound: number;
  /** Quantum (Tsirelson) bound; `2 * sqrt(2)` for CHSH, 4.0 for
   *  Mermin-GHZ. */
  quantumBound: number;
  /** One-tail p-value against the classical-bound null. */
  pValue: number;
  /** Standard error of `chshValue`. */
  standardError: number;
  /** Total measurement pairs the test consumed. */
  measurements: number;
  /** `true` if `chshValue > classicalBound`. */
  violatesClassical: boolean;
  /** `true` if `chshValue` is within ~0.05 of `quantumBound`. */
  confirmsQuantum: boolean;
  /** `true` if `pValue < 0.01`. */
  statisticallySignificant: boolean;
}

type BellModule = {
  _create_bell_state: (statePtr: number, q1: number, q2: number, type: number) => number;
  _bell_get_optimal_settings: (settingsPtr: number) => void;
  _bell_test_chsh: (
    resultPtr: number, statePtr: number, qa: number, qb: number,
    nMeas: number, settingsPtr: number, entropyPtr: number,
  ) => void;
  _bell_test_mermin_ghz: (
    resultPtr: number, statePtr: number, qa: number, qb: number, qc: number,
    nMeas: number, entropyPtr: number,
  ) => void;
  _bell_test_mermin_klyshko: (
    statePtr: number, numQubits: number, nMeas: number, entropyPtr: number,
  ) => number;
  _quantum_entropy_ctx_create_hw: () => number;
  _quantum_entropy_ctx_destroy: (ctx: number) => void;
  _malloc: (n: number) => number;
  _free: (p: number) => void;
  HEAPF64: Float64Array;
  HEAP32: Int32Array;
};

// `bell_test_result_t` is 17 fields packed into 128 bytes.  The
// double fields appear first in order: chsh_value, E(a,b), E(a,b'),
// E(a',b), E(a',b'), classical_bound, quantum_bound, p_value,
// standard_error.  Then size_t measurements, then 3 int flags, then
// 4 uint64_t counts.
function readBellResult(m: BellModule, resultPtr: number): BellTestResult {
  const idx = resultPtr >> 3;  // HEAPF64 index
  const d = m.HEAPF64;
  const i32 = resultPtr >> 2;  // HEAP32 index for the size_t / int / uint64 tail
  // size_t measurements lives at byte offset 72 (9 doubles) on
  // wasm32 -- size_t is 4 bytes there.  int flags follow at 76, 80,
  // 84.  We read measurements as a uint32.
  return {
    chshValue:               d[idx + 0],
    correlationAB:           d[idx + 1],
    correlationABprime:      d[idx + 2],
    correlationAprimeB:      d[idx + 3],
    correlationAprimeBprime: d[idx + 4],
    classicalBound:          d[idx + 5],
    quantumBound:            d[idx + 6],
    pValue:                  d[idx + 7],
    standardError:           d[idx + 8],
    measurements:            m.HEAP32[i32 + 18],  // byte 72 / 4
    violatesClassical:        m.HEAP32[i32 + 19] !== 0,
    confirmsQuantum:          m.HEAP32[i32 + 20] !== 0,
    statisticallySignificant: m.HEAP32[i32 + 21] !== 0,
  };
}

async function withEntropy<T>(fn: (m: BellModule, entropy: number) => T): Promise<T> {
  const m = (await getModule()) as unknown as BellModule;
  const entropy = m._quantum_entropy_ctx_create_hw();
  if (!entropy) throw new Error('quantum_entropy_ctx_create_hw returned NULL');
  try {
    return fn(m, entropy);
  } finally {
    m._quantum_entropy_ctx_destroy(entropy);
  }
}

/** Prepare the named Bell pair on the given qubits of `state`.
 *  Maps directly to `create_bell_state(state, q1, q2, type)`. */
export async function createBellState(
  state: QuantumState, qubit1: number, qubit2: number, type: BellState,
): Promise<void> {
  const m = (await getModule()) as unknown as BellModule;
  const rc = m._create_bell_state(
    state._internal_state_pointer(), qubit1, qubit2, type,
  );
  if (rc !== 0) throw new Error(`create_bell_state rc=${rc}`);
}

/** Run the CHSH inequality test on `state` at the Tsirelson-optimal
 *  angles `(a = 0, a' = pi/2, b = pi/4, b' = -pi/4)`.  Returns the
 *  full result struct including the `S` parameter, the four
 *  correlations, the classical / quantum bounds, and significance
 *  flags. */
export async function chshTest(
  state: QuantumState, qubitA: number, qubitB: number, numMeasurements: number,
): Promise<BellTestResult> {
  return withEntropy((m, entropy) => {
    const settingsPtr = m._malloc(32);  // 4 doubles
    const resultPtr = m._malloc(128);  // bell_test_result_t
    try {
      m._bell_get_optimal_settings(settingsPtr);
      m._bell_test_chsh(
        resultPtr, state._internal_state_pointer(),
        qubitA, qubitB, numMeasurements, settingsPtr, entropy,
      );
      return readBellResult(m, resultPtr);
    } finally {
      m._free(resultPtr);
      m._free(settingsPtr);
    }
  });
}

/** Run the Mermin inequality on a 3-qubit GHZ state at the indicated
 *  qubits.  Polynomial `M = <XYY> + <YXY> + <YYX> - <XXX>`; classical
 *  bound `|M| <= 2`, quantum bound `|M| = 4`.  The returned
 *  [`BellTestResult`] reuses `chshValue` for `|M|` and the four
 *  correlation fields for `{XYY, YXY, YYX, XXX}` in that order. */
export async function merminGhzTest(
  state: QuantumState,
  qubitA: number, qubitB: number, qubitC: number,
  numMeasurements: number,
): Promise<BellTestResult> {
  return withEntropy((m, entropy) => {
    const resultPtr = m._malloc(128);
    try {
      m._bell_test_mermin_ghz(
        resultPtr, state._internal_state_pointer(),
        qubitA, qubitB, qubitC, numMeasurements, entropy,
      );
      return readBellResult(m, resultPtr);
    } finally {
      m._free(resultPtr);
    }
  });
}

/** Run the N-qubit Mermin-Klyshko inequality on the first `numQubits`
 *  qubits of `state`.  Returns the normalised `|M_N|` value such that
 *  the classical (LHV) bound is 1.0 and the ideal GHZ quantum value
 *  is `2^((N-1)/2)`.  For `N = 2` this coincides with
 *  `CHSH / (2 sqrt(2))`. */
export async function merminKlyshkoTest(
  state: QuantumState, numQubits: number, numMeasurements: number,
): Promise<number> {
  return withEntropy((m, entropy) => {
    return m._bell_test_mermin_klyshko(
      state._internal_state_pointer(), numQubits, numMeasurements, entropy,
    );
  });
}
