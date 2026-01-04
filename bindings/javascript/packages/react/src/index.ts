/**
 * @moonlab/quantum-react
 *
 * React hooks and components for quantum computing applications.
 *
 * @example
 * ```tsx
 * import {
 *   useQuantumState,
 *   useCircuit,
 *   BlochSphere,
 *   AmplitudeBars,
 *   CircuitDiagram
 * } from '@moonlab/quantum-react';
 *
 * function QuantumApp() {
 *   const { amplitudes, numQubits, applyGate } = useQuantumState({ numQubits: 2 });
 *   const { circuit, addGate } = useCircuit({ numQubits: 2 });
 *
 *   const handleBellState = () => {
 *     applyGate('h', 0);
 *     applyGate('cnot', 0, 1);
 *     addGate('h', 0);
 *     addGate('cnot', 0, 1);
 *   };
 *
 *   return (
 *     <div>
 *       <button onClick={handleBellState}>Create Bell State</button>
 *       <AmplitudeBars amplitudes={amplitudes} numQubits={numQubits} />
 *       <CircuitDiagram circuit={circuit} />
 *     </div>
 *   );
 * }
 * ```
 *
 * @packageDocumentation
 */

// ============================================================================
// Hooks
// ============================================================================

export {
  useQuantumState,
  useCircuit,
} from './hooks';

export type {
  UseQuantumStateOptions,
  UseQuantumStateReturn,
  UseCircuitOptions,
  UseCircuitReturn,
} from './hooks';

// ============================================================================
// Components
// ============================================================================

export {
  BlochSphere,
  AmplitudeBars,
  CircuitDiagram,
} from './components';

export type {
  BlochSphereProps,
  AmplitudeBarsProps,
  CircuitDiagramProps,
} from './components';

// ============================================================================
// Re-exports from Core
// ============================================================================

export {
  QuantumState,
  Circuit,
  type Complex,
  type QuantumStateOptions,
} from '@moonlab/quantum-core';

// ============================================================================
// Version Info
// ============================================================================

export const VERSION = '1.0.0';
