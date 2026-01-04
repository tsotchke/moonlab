/**
 * useCircuit Composable
 *
 * Vue 3 composable for building and managing quantum circuits.
 */

import { ref, computed, type Ref, type ComputedRef } from 'vue';
import { Circuit, type CircuitStats } from '@moonlab/quantum-core';

export interface UseCircuitOptions {
  /**
   * Number of qubits (required)
   */
  numQubits: number;

  /**
   * Circuit name (optional)
   */
  name?: string;

  /**
   * Initial circuit JSON (optional)
   */
  initialCircuit?: ReturnType<Circuit['toJSON']>;
}

export interface UseCircuitReturn {
  /**
   * The circuit instance (reactive)
   */
  circuit: Ref<Circuit>;

  /**
   * Circuit statistics (computed)
   */
  stats: ComputedRef<CircuitStats>;

  /**
   * Number of gates in the circuit (computed)
   */
  gateCount: ComputedRef<number>;

  /**
   * Add a gate to the circuit
   */
  addGate: (gate: string, ...args: (number | number[])[]) => void;

  /**
   * Remove the last gate
   */
  undoLastGate: () => void;

  /**
   * Clear all gates
   */
  clear: () => void;

  /**
   * Export as OpenQASM
   */
  toQASM: () => string;

  /**
   * Export as JSON
   */
  toJSON: () => ReturnType<Circuit['toJSON']>;

  /**
   * Import from JSON
   */
  fromJSON: (json: ReturnType<Circuit['toJSON']>) => void;

  /**
   * Create a new circuit with different qubits
   */
  resize: (numQubits: number) => void;
}

/**
 * Vue 3 composable for quantum circuit building
 *
 * @example
 * ```vue
 * <script setup lang="ts">
 * import { useCircuit } from '@moonlab/quantum-vue';
 *
 * const { circuit, stats, addGate, undoLastGate, clear, toQASM } = useCircuit({
 *   numQubits: 3
 * });
 * </script>
 *
 * <template>
 *   <div>
 *     <div>Gates: {{ stats.totalGates }}</div>
 *     <button @click="addGate('h', 0)">Add H(0)</button>
 *     <button @click="addGate('cnot', 0, 1)">Add CNOT</button>
 *     <button @click="undoLastGate">Undo</button>
 *     <button @click="clear">Clear</button>
 *     <pre>{{ toQASM() }}</pre>
 *   </div>
 * </template>
 * ```
 */
export function useCircuit(options: UseCircuitOptions): UseCircuitReturn {
  const { numQubits: initialQubits, name, initialCircuit } = options;

  // Version for reactivity tracking
  const version = ref(0);

  // Create initial circuit
  const circuit = ref<Circuit>(
    initialCircuit ? Circuit.fromJSON(initialCircuit) : new Circuit(initialQubits, name)
  );

  // Computed stats
  const stats = computed(() => {
    // Access version to trigger reactivity
    void version.value;
    return circuit.value.getStats();
  });

  // Gate count
  const gateCount = computed(() => {
    void version.value;
    return circuit.value.length;
  });

  // Add a gate
  const addGate = (gate: string, ...args: (number | number[])[]) => {
    const c = circuit.value;

    switch (gate.toLowerCase()) {
      case 'h':
        c.h(args[0] as number);
        break;
      case 'x':
        c.x(args[0] as number);
        break;
      case 'y':
        c.y(args[0] as number);
        break;
      case 'z':
        c.z(args[0] as number);
        break;
      case 's':
        c.s(args[0] as number);
        break;
      case 'sdg':
        c.sdg(args[0] as number);
        break;
      case 't':
        c.t(args[0] as number);
        break;
      case 'tdg':
        c.tdg(args[0] as number);
        break;
      case 'rx':
        c.rx(args[0] as number, args[1] as number);
        break;
      case 'ry':
        c.ry(args[0] as number, args[1] as number);
        break;
      case 'rz':
        c.rz(args[0] as number, args[1] as number);
        break;
      case 'phase':
        c.phase(args[0] as number, args[1] as number);
        break;
      case 'u3':
        c.u3(
          args[0] as number,
          args[1] as number,
          args[2] as number,
          args[3] as number
        );
        break;
      case 'cnot':
      case 'cx':
        c.cnot(args[0] as number, args[1] as number);
        break;
      case 'cz':
        c.cz(args[0] as number, args[1] as number);
        break;
      case 'cy':
        c.cy(args[0] as number, args[1] as number);
        break;
      case 'swap':
        c.swap(args[0] as number, args[1] as number);
        break;
      case 'cphase':
        c.cphase(args[0] as number, args[1] as number, args[2] as number);
        break;
      case 'toffoli':
      case 'ccx':
        c.toffoli(args[0] as number, args[1] as number, args[2] as number);
        break;
      case 'fredkin':
      case 'cswap':
        c.fredkin(args[0] as number, args[1] as number, args[2] as number);
        break;
      case 'qft':
        c.qft(args[0] as number[]);
        break;
      case 'iqft':
        c.iqft(args[0] as number[]);
        break;
      case 'barrier':
        c.barrier(args[0] as number[] | undefined);
        break;
      case 'measure':
        c.measure(args[0] as number, args[1] as number | undefined);
        break;
      default:
        console.warn(`Unknown gate: ${gate}`);
    }

    version.value++;
  };

  // Undo last gate
  const undoLastGate = () => {
    if (circuit.value.length === 0) return;

    // Create a new circuit without the last gate
    const json = circuit.value.toJSON();
    json.gates = json.gates.slice(0, -1);
    circuit.value = Circuit.fromJSON(json);
    version.value++;
  };

  // Clear all gates
  const clear = () => {
    circuit.value = new Circuit(circuit.value.numQubits, circuit.value.name);
    version.value++;
  };

  // Export as QASM
  const toQASM = () => circuit.value.toQASM();

  // Export as JSON
  const toJSON = () => circuit.value.toJSON();

  // Import from JSON
  const fromJSON = (json: ReturnType<Circuit['toJSON']>) => {
    circuit.value = Circuit.fromJSON(json);
    version.value++;
  };

  // Resize (create new circuit with different qubits)
  const resize = (numQubits: number) => {
    circuit.value = new Circuit(numQubits, circuit.value.name);
    version.value++;
  };

  return {
    circuit,
    stats,
    gateCount,
    addGate,
    undoLastGate,
    clear,
    toQASM,
    toJSON,
    fromJSON,
    resize,
  };
}
