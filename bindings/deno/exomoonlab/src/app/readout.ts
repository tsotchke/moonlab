/**
 * Computing what the probabilities window shows.
 *
 * This runs as a job, never in `frame()`. It reduces a state to something
 * small enough to paint: the strongest basis states plus a few scalars.
 *
 * Scanning is bounded. Reading every basis probability is one call per
 * amplitude, so a 20-qubit register would mean over a million round trips
 * through FFI or the WASM boundary. Past `scanLimit` the scan stops and the
 * readout says so, rather than freezing while it insists on completeness.
 * Per-qubit marginals are unaffected -- those are one call per qubit and come
 * from the full state either way.
 */

import type { MoonLabBackend } from "../backend/mod.ts";
import type { Circuit } from "./circuits.ts";
import type { BasisEntry, ProbabilityReadout } from "../ui/probabilities.ts";

/** Basis states scanned before the readout stops and reports partial cover. */
export const DEFAULT_SCAN_LIMIT = 4096;

export interface ReadoutOptions {
  readonly numQubits: number;
  /** How many of the strongest basis states to keep. */
  readonly topCount?: number;
  readonly scanLimit?: number;
}

export async function computeReadout(
  backend: MoonLabBackend,
  circuit: Circuit,
  options: ReadoutOptions,
): Promise<ProbabilityReadout> {
  const numQubits = Math.max(circuit.minQubits, options.numQubits);
  const topCount = options.topCount ?? 16;
  const scanLimit = options.scanLimit ?? DEFAULT_SCAN_LIMIT;

  const state = await backend.createState(numQubits);
  try {
    await circuit.apply(backend, state, numQubits);

    const stateDim = state.stateDim;
    const scanned = Math.min(stateDim, scanLimit);

    // A bounded min-heap would be tidier; at these sizes a sorted insert into
    // a topCount-length array is simpler and measurably fine.
    const top: BasisEntry[] = [];
    let coveredProbability = 0;
    let smallestKept = 0;

    for (let index = 0; index < scanned; index++) {
      const probability = await backend.probability(state, index);
      if (probability <= 0) continue;
      if (top.length < topCount) {
        top.push({ index, probability });
        top.sort((a, b) => b.probability - a.probability);
        smallestKept = top[top.length - 1].probability;
      } else if (probability > smallestKept) {
        top[top.length - 1] = { index, probability };
        top.sort((a, b) => b.probability - a.probability);
        smallestKept = top[top.length - 1].probability;
      }
    }
    coveredProbability = top.reduce((sum, entry) => sum + entry.probability, 0);

    const qubitOnes: number[] = [];
    for (let q = 0; q < numQubits; q++) {
      qubitOnes.push(await backend.probabilityOne(state, q));
    }

    return {
      numQubits,
      stateDim,
      scanned,
      top,
      coveredProbability,
      entropy: await backend.entropy(state),
      purity: await backend.purity(state),
      qubitOnes,
    };
  } finally {
    await backend.destroyState(state);
  }
}
