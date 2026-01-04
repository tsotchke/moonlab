/**
 * useQuantumState Composable
 *
 * Vue 3 composable for managing quantum state with reactivity.
 */

import { ref, computed, onMounted, onUnmounted, watch, type Ref } from 'vue';
import { QuantumState, type Complex } from '@moonlab/quantum-core';

export interface UseQuantumStateOptions {
  /**
   * Number of qubits (required)
   */
  numQubits: number;

  /**
   * Initial amplitudes (optional)
   */
  initialAmplitudes?: Complex[];

  /**
   * Auto-initialize on mount (default: true)
   */
  autoInit?: boolean;
}

export interface UseQuantumStateReturn {
  /**
   * The quantum state instance (null if not initialized)
   */
  state: Ref<QuantumState | null>;

  /**
   * Whether the state is currently loading
   */
  loading: Ref<boolean>;

  /**
   * Error message if initialization failed
   */
  error: Ref<string | null>;

  /**
   * Current amplitudes (reactive)
   */
  amplitudes: Ref<Complex[]>;

  /**
   * Current probabilities (reactive)
   */
  probabilities: Ref<number[]>;

  /**
   * Number of qubits
   */
  numQubits: Ref<number>;

  /**
   * Initialize or reinitialize the state
   */
  initialize: () => Promise<void>;

  /**
   * Reset state to |0...0⟩
   */
  reset: () => void;

  /**
   * Apply a gate and update reactive values
   */
  applyGate: (gate: string, ...args: (number | number[])[]) => void;

  /**
   * Measure a qubit (collapses state)
   */
  measure: (qubit: number) => number;

  /**
   * Measure all qubits
   */
  measureAll: () => number;

  /**
   * Force refresh of reactive values
   */
  refresh: () => void;

  /**
   * Dispose of the quantum state
   */
  dispose: () => void;
}

/**
 * Vue 3 composable for quantum state management
 *
 * @example
 * ```vue
 * <script setup lang="ts">
 * import { useQuantumState } from '@moonlab/quantum-vue';
 *
 * const { amplitudes, probabilities, applyGate, loading } = useQuantumState({
 *   numQubits: 2
 * });
 *
 * const createBellState = () => {
 *   applyGate('h', 0);
 *   applyGate('cnot', 0, 1);
 * };
 * </script>
 *
 * <template>
 *   <div v-if="loading">Loading...</div>
 *   <div v-else>
 *     <button @click="createBellState">Bell State</button>
 *     <pre>{{ probabilities }}</pre>
 *   </div>
 * </template>
 * ```
 */
export function useQuantumState(
  options: UseQuantumStateOptions
): UseQuantumStateReturn {
  const { numQubits: initialQubits, initialAmplitudes, autoInit = true } = options;

  const state = ref<QuantumState | null>(null);
  const loading = ref(autoInit);
  const error = ref<string | null>(null);
  const amplitudes = ref<Complex[]>([]);
  const probabilities = ref<number[]>([]);
  const numQubits = ref(initialQubits);

  // Update reactive values from current state
  const refresh = () => {
    if (state.value && !state.value.isDisposed) {
      amplitudes.value = state.value.getAmplitudes();
      probabilities.value = Array.from(state.value.getProbabilities());
    }
  };

  // Initialize the quantum state
  const initialize = async () => {
    loading.value = true;
    error.value = null;

    try {
      // Dispose previous state if exists
      if (state.value && !state.value.isDisposed) {
        state.value.dispose();
      }

      const newState = await QuantumState.create({
        numQubits: numQubits.value,
        amplitudes: initialAmplitudes,
      });

      state.value = newState;
      refresh();
    } catch (err) {
      error.value = err instanceof Error ? err.message : String(err);
      state.value = null;
    } finally {
      loading.value = false;
    }
  };

  // Reset to |0...0⟩
  const reset = () => {
    if (state.value && !state.value.isDisposed) {
      state.value.reset();
      refresh();
    }
  };

  // Apply a gate
  const applyGate = (gate: string, ...args: (number | number[])[]) => {
    if (!state.value || state.value.isDisposed) return;

    const s = state.value;

    switch (gate.toLowerCase()) {
      case 'h':
        s.h(args[0] as number);
        break;
      case 'x':
        s.x(args[0] as number);
        break;
      case 'y':
        s.y(args[0] as number);
        break;
      case 'z':
        s.z(args[0] as number);
        break;
      case 's':
        s.s(args[0] as number);
        break;
      case 'sdg':
        s.sdg(args[0] as number);
        break;
      case 't':
        s.t(args[0] as number);
        break;
      case 'tdg':
        s.tdg(args[0] as number);
        break;
      case 'rx':
        s.rx(args[0] as number, args[1] as number);
        break;
      case 'ry':
        s.ry(args[0] as number, args[1] as number);
        break;
      case 'rz':
        s.rz(args[0] as number, args[1] as number);
        break;
      case 'phase':
        s.phase(args[0] as number, args[1] as number);
        break;
      case 'u3':
        s.u3(
          args[0] as number,
          args[1] as number,
          args[2] as number,
          args[3] as number
        );
        break;
      case 'cnot':
      case 'cx':
        s.cnot(args[0] as number, args[1] as number);
        break;
      case 'cz':
        s.cz(args[0] as number, args[1] as number);
        break;
      case 'cy':
        s.cy(args[0] as number, args[1] as number);
        break;
      case 'swap':
        s.swap(args[0] as number, args[1] as number);
        break;
      case 'cphase':
        s.cphase(args[0] as number, args[1] as number, args[2] as number);
        break;
      case 'toffoli':
      case 'ccx':
        s.toffoli(args[0] as number, args[1] as number, args[2] as number);
        break;
      case 'fredkin':
      case 'cswap':
        s.fredkin(args[0] as number, args[1] as number, args[2] as number);
        break;
      case 'qft':
        s.qft(args[0] as number[]);
        break;
      case 'iqft':
        s.iqft(args[0] as number[]);
        break;
      default:
        console.warn(`Unknown gate: ${gate}`);
    }

    refresh();
  };

  // Measure a qubit
  const measure = (qubit: number): number => {
    if (!state.value || state.value.isDisposed) return 0;
    const result = state.value.measure(qubit);
    refresh();
    return result;
  };

  // Measure all qubits
  const measureAll = (): number => {
    if (!state.value || state.value.isDisposed) return 0;
    const result = state.value.measureAll();
    refresh();
    return result;
  };

  // Dispose
  const dispose = () => {
    if (state.value && !state.value.isDisposed) {
      state.value.dispose();
      state.value = null;
      amplitudes.value = [];
      probabilities.value = [];
    }
  };

  // Auto-initialize on mount
  onMounted(() => {
    if (autoInit) {
      initialize();
    }
  });

  // Cleanup on unmount
  onUnmounted(() => {
    if (state.value && !state.value.isDisposed) {
      state.value.dispose();
    }
  });

  return {
    state,
    loading,
    error,
    amplitudes,
    probabilities,
    numQubits,
    initialize,
    reset,
    applyGate,
    measure,
    measureAll,
    refresh,
    dispose,
  };
}
