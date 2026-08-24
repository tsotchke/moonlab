/**
 * The circuit catalog.
 *
 * Shared by the console and the equivalence harness so the thing the user
 * watches and the thing CI checks are literally the same circuits. Each is
 * parametric in qubit count, because "does it stay responsive at 20 qubits"
 * is a question about the same circuit at a different size, not a different
 * circuit.
 */

import type { MoonLabBackend, StateHandle } from "../backend/mod.ts";

export interface Circuit {
  readonly id: string;
  readonly name: string;
  /** Smallest register the circuit is meaningful on. */
  readonly minQubits: number;
  readonly defaultQubits: number;
  readonly apply: (
    backend: MoonLabBackend,
    state: StateHandle,
    numQubits: number,
  ) => Promise<void>;
}

export const CIRCUITS: readonly Circuit[] = [
  {
    id: "ground",
    name: "ground state",
    minQubits: 1,
    defaultQubits: 3,
    apply: () => Promise.resolve(),
  },
  {
    id: "x0",
    name: "X on qubit 0",
    minQubits: 1,
    defaultQubits: 3,
    apply: (b, s) => b.pauliX(s, 0),
  },
  {
    id: "uniform",
    name: "uniform superposition",
    minQubits: 1,
    defaultQubits: 3,
    apply: async (b, s, n) => {
      for (let q = 0; q < n; q++) await b.hadamard(s, q);
    },
  },
  {
    id: "bell",
    name: "Bell pair",
    minQubits: 2,
    defaultQubits: 2,
    apply: async (b, s) => {
      await b.hadamard(s, 0);
      await b.cnot(s, 0, 1);
    },
  },
  {
    id: "ghz",
    name: "GHZ state",
    minQubits: 2,
    defaultQubits: 4,
    apply: async (b, s, n) => {
      await b.hadamard(s, 0);
      for (let q = 1; q < n; q++) await b.cnot(s, 0, q);
    },
  },
  {
    id: "z-phase",
    name: "Z phase (probabilities unchanged)",
    minQubits: 1,
    defaultQubits: 2,
    apply: async (b, s) => {
      await b.hadamard(s, 0);
      await b.pauliZ(s, 0);
    },
  },
  {
    id: "ladder",
    name: "entangling ladder",
    minQubits: 2,
    defaultQubits: 5,
    apply: async (b, s, n) => {
      for (let q = 0; q < n; q++) await b.hadamard(s, q);
      for (let q = 0; q + 1 < n; q++) await b.cnot(s, q, q + 1);
      await b.pauliZ(s, Math.min(2, n - 1));
      await b.normalize(s);
    },
  },
];

export function circuitById(id: string): Circuit {
  const found = CIRCUITS.find((c) => c.id === id);
  if (!found) throw new Error(`unknown circuit: ${id}`);
  return found;
}
