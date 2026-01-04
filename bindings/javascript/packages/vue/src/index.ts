/**
 * @moonlab/quantum-vue
 *
 * Vue 3 composables and components for quantum computing applications.
 *
 * @example
 * ```vue
 * <script setup lang="ts">
 * import {
 *   useQuantumState,
 *   useCircuit,
 *   BlochSphere,
 *   AmplitudeBars,
 *   CircuitDiagram
 * } from '@moonlab/quantum-vue';
 *
 * const { amplitudes, numQubits, applyGate } = useQuantumState({ numQubits: 2 });
 * const { circuit, addGate } = useCircuit({ numQubits: 2 });
 *
 * const handleBellState = () => {
 *   applyGate('h', 0);
 *   applyGate('cnot', 0, 1);
 *   addGate('h', 0);
 *   addGate('cnot', 0, 1);
 * };
 * </script>
 *
 * <template>
 *   <div>
 *     <button @click="handleBellState">Create Bell State</button>
 *     <AmplitudeBars :amplitudes="amplitudes" :num-qubits="numQubits" />
 *     <CircuitDiagram :circuit="circuit" />
 *   </div>
 * </template>
 * ```
 *
 * @packageDocumentation
 */

// ============================================================================
// Composables
// ============================================================================

export {
  useQuantumState,
  useCircuit,
} from './composables';

export type {
  UseQuantumStateOptions,
  UseQuantumStateReturn,
  UseCircuitOptions,
  UseCircuitReturn,
} from './composables';

// ============================================================================
// Components
// ============================================================================

export {
  BlochSphere,
  AmplitudeBars,
  CircuitDiagram,
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
