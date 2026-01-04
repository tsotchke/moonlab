/**
 * useQuantumState Hook
 *
 * React hook for managing quantum state with automatic cleanup.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
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
  state: QuantumState | null;

  /**
   * Whether the state is currently loading
   */
  loading: boolean;

  /**
   * Error message if initialization failed
   */
  error: string | null;

  /**
   * Current amplitudes (reactive)
   */
  amplitudes: Complex[];

  /**
   * Current probabilities (reactive)
   */
  probabilities: number[];

  /**
   * Number of qubits
   */
  numQubits: number;

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
  applyGate: (
    gate: string,
    ...args: (number | number[])[]
  ) => void;

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
 * React hook for quantum state management
 *
 * @example
 * ```tsx
 * function QuantumDemo() {
 *   const { state, probabilities, applyGate, loading } = useQuantumState({
 *     numQubits: 2
 *   });
 *
 *   if (loading) return <div>Loading...</div>;
 *
 *   return (
 *     <div>
 *       <button onClick={() => applyGate('h', 0)}>H(0)</button>
 *       <button onClick={() => applyGate('cnot', 0, 1)}>CNOT(0,1)</button>
 *       <pre>{JSON.stringify(probabilities, null, 2)}</pre>
 *     </div>
 *   );
 * }
 * ```
 */
export function useQuantumState(
  options: UseQuantumStateOptions
): UseQuantumStateReturn {
  const { numQubits, initialAmplitudes, autoInit = true } = options;

  const [state, setState] = useState<QuantumState | null>(null);
  const [loading, setLoading] = useState(autoInit);
  const [error, setError] = useState<string | null>(null);
  const [amplitudes, setAmplitudes] = useState<Complex[]>([]);
  const [probabilities, setProbabilities] = useState<number[]>([]);

  const stateRef = useRef<QuantumState | null>(null);

  // Update reactive values from current state
  const refresh = useCallback(() => {
    if (stateRef.current && !stateRef.current.isDisposed) {
      const amps = stateRef.current.getAmplitudes();
      const probs = Array.from(stateRef.current.getProbabilities());
      setAmplitudes(amps);
      setProbabilities(probs);
    }
  }, []);

  // Initialize the quantum state
  const initialize = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      // Dispose previous state if exists
      if (stateRef.current && !stateRef.current.isDisposed) {
        stateRef.current.dispose();
      }

      const newState = await QuantumState.create({
        numQubits,
        amplitudes: initialAmplitudes,
      });

      stateRef.current = newState;
      setState(newState);
      refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setState(null);
    } finally {
      setLoading(false);
    }
  }, [numQubits, initialAmplitudes, refresh]);

  // Reset to |0...0⟩
  const reset = useCallback(() => {
    if (stateRef.current && !stateRef.current.isDisposed) {
      stateRef.current.reset();
      refresh();
    }
  }, [refresh]);

  // Apply a gate
  const applyGate = useCallback(
    (gate: string, ...args: (number | number[])[]) => {
      if (!stateRef.current || stateRef.current.isDisposed) return;

      const s = stateRef.current;

      // Map gate names to methods
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
    },
    [refresh]
  );

  // Measure a qubit
  const measure = useCallback(
    (qubit: number): number => {
      if (!stateRef.current || stateRef.current.isDisposed) return 0;
      const result = stateRef.current.measure(qubit);
      refresh();
      return result;
    },
    [refresh]
  );

  // Measure all qubits
  const measureAll = useCallback((): number => {
    if (!stateRef.current || stateRef.current.isDisposed) return 0;
    const result = stateRef.current.measureAll();
    refresh();
    return result;
  }, [refresh]);

  // Dispose
  const dispose = useCallback(() => {
    if (stateRef.current && !stateRef.current.isDisposed) {
      stateRef.current.dispose();
      stateRef.current = null;
      setState(null);
      setAmplitudes([]);
      setProbabilities([]);
    }
  }, []);

  // Auto-initialize on mount
  useEffect(() => {
    if (autoInit) {
      initialize();
    }

    // Cleanup on unmount
    return () => {
      if (stateRef.current && !stateRef.current.isDisposed) {
        stateRef.current.dispose();
      }
    };
  }, [autoInit, initialize]);

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
