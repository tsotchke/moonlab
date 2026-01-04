/**
 * useCircuit Hook
 *
 * React hook for building and managing quantum circuits.
 */

import { useState, useCallback, useMemo } from 'react';
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
   * The circuit instance
   */
  circuit: Circuit;

  /**
   * Circuit statistics
   */
  stats: CircuitStats;

  /**
   * Number of gates in the circuit
   */
  gateCount: number;

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
 * React hook for quantum circuit building
 *
 * @example
 * ```tsx
 * function CircuitBuilder() {
 *   const {
 *     circuit,
 *     stats,
 *     addGate,
 *     undoLastGate,
 *     clear,
 *     toQASM
 *   } = useCircuit({ numQubits: 3 });
 *
 *   return (
 *     <div>
 *       <div>Gates: {stats.totalGates}</div>
 *       <button onClick={() => addGate('h', 0)}>Add H(0)</button>
 *       <button onClick={() => addGate('cnot', 0, 1)}>Add CNOT</button>
 *       <button onClick={undoLastGate}>Undo</button>
 *       <button onClick={clear}>Clear</button>
 *       <pre>{toQASM()}</pre>
 *     </div>
 *   );
 * }
 * ```
 */
export function useCircuit(options: UseCircuitOptions): UseCircuitReturn {
  const { numQubits: initialQubits, name, initialCircuit } = options;

  // Create initial circuit
  const [circuit, setCircuit] = useState<Circuit>(() => {
    if (initialCircuit) {
      return Circuit.fromJSON(initialCircuit);
    }
    return new Circuit(initialQubits, name);
  });

  // Track version for re-renders
  const [version, setVersion] = useState(0);

  // Compute stats
  const stats = useMemo(() => circuit.getStats(), [circuit, version]);

  // Gate count
  const gateCount = circuit.length;

  // Add a gate
  const addGate = useCallback(
    (gate: string, ...args: (number | number[])[]) => {
      const c = circuit;

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

      setVersion((v) => v + 1);
    },
    [circuit]
  );

  // Undo last gate
  const undoLastGate = useCallback(() => {
    if (circuit.length === 0) return;

    // Create a new circuit without the last gate
    const json = circuit.toJSON();
    json.gates = json.gates.slice(0, -1);
    setCircuit(Circuit.fromJSON(json));
    setVersion((v) => v + 1);
  }, [circuit]);

  // Clear all gates
  const clear = useCallback(() => {
    setCircuit(new Circuit(circuit.numQubits, circuit.name));
    setVersion((v) => v + 1);
  }, [circuit.numQubits, circuit.name]);

  // Export as QASM
  const toQASM = useCallback(() => circuit.toQASM(), [circuit, version]);

  // Export as JSON
  const toJSON = useCallback(() => circuit.toJSON(), [circuit, version]);

  // Import from JSON
  const fromJSON = useCallback(
    (json: ReturnType<Circuit['toJSON']>) => {
      setCircuit(Circuit.fromJSON(json));
      setVersion((v) => v + 1);
    },
    []
  );

  // Resize (create new circuit with different qubits)
  const resize = useCallback(
    (numQubits: number) => {
      setCircuit(new Circuit(numQubits, circuit.name));
      setVersion((v) => v + 1);
    },
    [circuit.name]
  );

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
