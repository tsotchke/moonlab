/**
 * Entanglement metric bindings.
 *
 * Compute bipartite entanglement measures on a {@link QuantumState}:
 * von Neumann entropy of the reduced density matrix on a chosen
 * subsystem, mutual information `I(A:B)`, and (for two-qubit states)
 * Wootters concurrence and negativity.
 *
 * Underlying C symbols live in `src/quantum/entanglement.{c,h}`.
 *
 * @since v0.2.1
 */

import type { QuantumState } from './quantum-state';

interface EntanglementModule {
  _entanglement_entropy_bipartition: (
    s: number, qubits_b_ptr: number, num_b: number,
  ) => number;
  _entanglement_mutual_information: (
    s: number, qubits_a_ptr: number, num_a: number,
    qubits_b_ptr: number, num_b: number,
  ) => number;
  _entanglement_concurrence_2qubit: (s: number) => number;
  _entanglement_negativity_2qubit: (s: number) => number;
  HEAP32: Int32Array;
  _malloc: (n: number) => number;
  _free: (p: number) => void;
}

/**
 * Internal: marshal a JS number array as a `const int[]` in the WASM
 * heap, run `body`, and free the buffer.  Returns whatever `body`
 * returns.
 */
function withInt32Array<T>(
  mod: EntanglementModule, arr: number[], body: (ptr: number, len: number) => T,
): T {
  const ptr = mod._malloc(arr.length * 4);
  try {
    const off = ptr >>> 2;
    for (let i = 0; i < arr.length; i++) mod.HEAP32[off + i] = arr[i] | 0;
    return body(ptr, arr.length);
  } finally {
    mod._free(ptr);
  }
}

function modAndPtr(state: QuantumState): { mod: EntanglementModule; ptr: number } {
  const mod = state._internal_module() as unknown as EntanglementModule;
  const ptr = state._internal_state_pointer();
  return { mod, ptr };
}

/**
 * Bipartite von Neumann entropy of subsystem B via partial trace.
 *
 * `qubitsB` is the subsystem-B qubit list (the complement is
 * traced out).  Returns S(rho_B) in nats (log base e), where
 * S(rho) = -Tr(rho log rho).
 */
export function vonNeumannEntropy(
  state: QuantumState, qubitsB: number[],
): number {
  const { mod, ptr } = modAndPtr(state);
  return withInt32Array(mod, qubitsB,
    (qPtr, n) => mod._entanglement_entropy_bipartition(ptr, qPtr, n));
}

/**
 * Mutual information `I(A:B) = S(A) + S(B) - S(AB)` between two
 * disjoint subsystems.  On a pure state of A u B, this reduces to
 * `2 * S(A)`.  Qubits outside A u B are traced out first.
 *
 * Returned in bits (log base 2).
 */
export function mutualInformation(
  state: QuantumState, qubitsA: number[], qubitsB: number[],
): number {
  const { mod, ptr } = modAndPtr(state);
  return withInt32Array(mod, qubitsA,
    (aPtr, na) => withInt32Array(mod, qubitsB,
      (bPtr, nb) => mod._entanglement_mutual_information(
        ptr, aPtr, na, bPtr, nb)));
}

/**
 * Wootters concurrence on a two-qubit pure state.  Returns 0 for
 * separable states and 1 for maximally entangled (e.g. Bell).  Only
 * meaningful when `state.numQubits === 2`.
 */
export function concurrence2Qubit(state: QuantumState): number {
  const { mod, ptr } = modAndPtr(state);
  return mod._entanglement_concurrence_2qubit(ptr);
}

/**
 * Negativity on a two-qubit pure state, defined as the absolute sum
 * of negative eigenvalues of the partial transpose.  Returns 0 for
 * separable, 0.5 for maximally entangled.
 */
export function negativity2Qubit(state: QuantumState): number {
  const { mod, ptr } = modAndPtr(state);
  return mod._entanglement_negativity_2qubit(ptr);
}
