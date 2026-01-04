/**
 * @moonlab/quantum-core
 *
 * WebAssembly quantum computing simulator core library.
 * Provides low-level access to quantum state manipulation,
 * gates, and measurement operations.
 *
 * @example
 * ```typescript
 * import { QuantumState, Circuit } from '@moonlab/quantum-core';
 *
 * // Create a 2-qubit Bell state using fluent API
 * const state = await QuantumState.create({ numQubits: 2 });
 * state.h(0).cnot(0, 1);
 * console.log(state.getProbabilities());  // [0.5, 0, 0, 0.5]
 * state.dispose();
 *
 * // Or use Circuit builder for reusable circuits
 * const circuit = new Circuit(2)
 *   .h(0)
 *   .cnot(0, 1);
 *
 * const state2 = await QuantumState.create({ numQubits: 2 });
 * circuit.apply(state2);
 * state2.dispose();
 * ```
 *
 * @packageDocumentation
 */

// ============================================================================
// Core Classes
// ============================================================================

export { QuantumState } from './quantum-state';
export type { QuantumStateOptions } from './quantum-state';

export { Circuit } from './circuit';
export type {
  Gate,
  GateType,
  SingleQubitGate,
  TwoQubitGate,
  ThreeQubitGate,
  MultiQubitGate,
  ParameterizedGate,
  CircuitStats,
} from './circuit';

// ============================================================================
// Complex Number Utilities
// ============================================================================

export {
  complex,
  magnitude,
  magnitudeSquared,
  phase,
  conjugate,
  add,
  subtract,
  multiply,
  scale,
  divide,
  exp,
  fromPolar,
  toPolar,
  equals,
  toString,
  fromInterleaved,
  toInterleaved,
  innerProduct,
  norm,
  normalize,
  ZERO,
  ONE,
  I,
} from './complex';
export type { Complex } from './complex';

// ============================================================================
// WASM Module Management
// ============================================================================

export { getModule, isLoaded, resetModule, preload } from './wasm-loader';
export type { LoadOptions } from './wasm-loader';

export { WasmMemory } from './memory';
export type { MoonlabModule } from './memory';

// ============================================================================
// Version Info
// ============================================================================

export const VERSION = '0.1.1';

// ============================================================================
// Type-only Exports
// ============================================================================

/**
 * Basis state represented as a binary string
 * e.g., "00", "01", "10", "11" for 2 qubits
 */
export type BasisState = string;

/**
 * Measurement result with basis state and probability
 */
export interface MeasurementResult {
  basisState: number;
  bitString: BasisState;
  probability: number;
}

/**
 * State vector as array of complex amplitudes
 */
export type StateVector = import('./complex').Complex[];

/**
 * Probability distribution over basis states
 */
export type ProbabilityDistribution = Float64Array;
