/**
 * Grover's search bindings (C-side since v0.2.0; JS binding since
 * v0.5.4).
 *
 * Wraps ``src/algorithms/grover.{c,h}`` so JavaScript callers can
 * run the full search entry point plus the optimal-iteration
 * helper.  Searches a marked basis state in ``O(sqrt(N))`` queries
 * where ``N = 2^numQubits``.  Mirrors the Python
 * ``moonlab.algorithms.Grover`` and Rust ``moonlab::grover``
 * surfaces.
 *
 * @example
 * ```typescript
 * import { QuantumState, groverSearch } from '@moonlab/quantum-core';
 *
 * const state = await QuantumState.create({ numQubits: 4 });
 * const r = await groverSearch(state, 0b1010n);
 * console.log(`P(success) = ${r.successProbability.toFixed(3)}`);
 * state.dispose();
 * ```
 */

import { getModule } from './wasm-loader';
import { QuantumState } from './quantum-state';

/** Snapshot of one Grover-search run.  Mirrors ``grover_result_t``
 *  in the C header. */
export interface GroverResult {
  /** Most-probable measurement outcome after the final Grover
   *  iteration. */
  foundState: bigint;
  /** Probability of measuring the marked state on a noiseless
   *  projective measurement. */
  successProbability: number;
  /** Number of oracle queries the algorithm performed. */
  oracleCalls: number;
  /** Number of Grover iterations the algorithm performed. */
  iterationsPerformed: number;
  /** `|<target|final>|^2`. */
  fidelity: number;
  /** `true` when the most-probable outcome equals the marked state. */
  foundMarkedState: boolean;
}

type GroverModule = {
  _grover_search: (
    resultPtr: number, statePtr: number, configPtr: number, entropyPtr: number,
  ) => void;
  _grover_optimal_iterations: (numQubits: number) => number;
  _quantum_entropy_ctx_create_hw: () => number;
  _quantum_entropy_ctx_destroy: (ctx: number) => void;
  _malloc: (n: number) => number;
  _free: (p: number) => void;
  HEAPF64: Float64Array;
  HEAP32: Int32Array;
  HEAPU32: Uint32Array;
};

/** Return the optimal number of Grover iterations on a `numQubits`-
 *  qubit register: `floor(pi * sqrt(N) / 4)` where `N = 2^numQubits`. */
export async function groverOptimalIterations(numQubits: number): Promise<number> {
  const m = (await getModule()) as unknown as GroverModule;
  return m._grover_optimal_iterations(numQubits);
}

/** Run Grover's search on `state`, looking for `markedState`.  When
 *  `numIterations` is `undefined` (the default), the optimal
 *  `floor(pi sqrt(N) / 4)` iteration count is used.  The state is
 *  mutated to the post-iteration superposition. */
export async function groverSearch(
  state: QuantumState,
  markedState: bigint,
  numIterations?: number,
): Promise<GroverResult> {
  const m = (await getModule()) as unknown as GroverModule;
  const entropy = m._quantum_entropy_ctx_create_hw();
  if (!entropy) throw new Error('quantum_entropy_ctx_create_hw returned NULL');

  // grover_config_t layout on wasm32:
  //   size_t   num_qubits             @ 0  (4 bytes)
  //   uint64_t marked_state           @ 8  (8 bytes; 4-byte align gap)
  //   size_t   num_iterations         @ 16 (4 bytes)
  //   int      use_optimal_iterations @ 20 (4 bytes)
  const configPtr = m._malloc(24);
  // grover_result_t layout (40 bytes on wasm32):
  //   uint64_t found_state            @ 0  (8 bytes)
  //   double   success_probability    @ 8  (8 bytes)
  //   size_t   oracle_calls           @ 16 (4 bytes)
  //   size_t   iterations_performed   @ 20 (4 bytes)
  //   double   fidelity               @ 24 (8 bytes; aligned)
  //   int      found_marked_state     @ 32 (4 bytes)
  const resultPtr = m._malloc(40);
  try {
    const numQubits = state.numQubits;
    const useOptimal = numIterations === undefined ? 1 : 0;
    const iters = numIterations ?? 0;
    m.HEAPU32[configPtr >> 2]       = numQubits;
    // Pack marked_state as a u64; HEAPU32 little-endian on wasm32.
    m.HEAPU32[(configPtr + 8) >> 2]     = Number(markedState & 0xffffffffn);
    m.HEAPU32[(configPtr + 12) >> 2]    = Number((markedState >> 32n) & 0xffffffffn);
    m.HEAPU32[(configPtr + 16) >> 2]    = iters;
    m.HEAP32[(configPtr + 20) >> 2]     = useOptimal;

    m._grover_search(resultPtr, state._internal_state_pointer(), configPtr, entropy);

    const foundLo = BigInt(m.HEAPU32[resultPtr >> 2]);
    const foundHi = BigInt(m.HEAPU32[(resultPtr + 4) >> 2]);
    const foundState = (foundHi << 32n) | foundLo;
    const successProbability = m.HEAPF64[(resultPtr + 8) >> 3];
    const oracleCalls = m.HEAPU32[(resultPtr + 16) >> 2];
    const iterationsPerformed = m.HEAPU32[(resultPtr + 20) >> 2];
    const fidelity = m.HEAPF64[(resultPtr + 24) >> 3];
    const foundMarkedState = m.HEAP32[(resultPtr + 32) >> 2] !== 0;

    return {
      foundState,
      successProbability,
      oracleCalls,
      iterationsPerformed,
      fidelity,
      foundMarkedState,
    };
  } finally {
    m._free(resultPtr);
    m._free(configPtr);
    m._quantum_entropy_ctx_destroy(entropy);
  }
}
