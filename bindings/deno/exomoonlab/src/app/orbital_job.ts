/**
 * Round-tripping an orbital density through the simulator.
 *
 * The density is known in closed form, so pushing it through a state vector
 * and reading the probabilities back is a measurement with a known answer:
 * the L1 distance between what MoonLab returns and what the analytic solution
 * says should be at the level of floating-point noise. It is the same check
 * the project's web demo makes, and it is the reason this window is a MoonLab
 * demo rather than a plot of a formula.
 */

import type { MoonLabBackend } from "../backend/mod.ts";
import { type DensitySlice, densitySlice, type Orbital, suggestedExtent } from "./orbital.ts";

export interface OrbitalResult {
  readonly slice: DensitySlice;
  /** L1 distance between MoonLab's probabilities and the analytic density. */
  readonly drift?: number;
  /** Qubits used for the round-trip, when it ran. */
  readonly qubits?: number;
}

/** Largest power-of-two dimension not exceeding `count`. */
function fittedQubits(count: number, maxQubits: number): number {
  let qubits = 1;
  while (2 ** (qubits + 1) <= count && qubits + 1 <= maxQubits) qubits++;
  return qubits;
}

export async function computeOrbital(
  backend: MoonLabBackend,
  orbital: Orbital,
  size: number,
): Promise<OrbitalResult> {
  const slice = densitySlice(orbital, size, suggestedExtent(orbital));

  // The round-trip needs the backend to accept an amplitude vector. When it
  // cannot, the picture is still correct -- only the cross-check is missing,
  // and the window says so rather than inventing a drift figure.
  if (!backend.capabilities.amplitudeUpload || !backend.loadAmplitudes) {
    return { slice };
  }

  const qubits = fittedQubits(slice.density.length, backend.capabilities.maxQubits);
  const dim = 2 ** qubits;

  // Amplitudes are sqrt of probability; the leading `dim` cells of the slice
  // are what fits, renormalised so the vector is a legal state.
  let total = 0;
  for (let i = 0; i < dim; i++) total += slice.density[i];
  if (!(total > 0)) return { slice };

  const amplitudes = new Float64Array(dim * 2);
  const expected = new Float64Array(dim);
  for (let i = 0; i < dim; i++) {
    const probability = slice.density[i] / total;
    expected[i] = probability;
    amplitudes[i * 2] = Math.sqrt(probability);
  }

  const state = await backend.createState(qubits);
  try {
    await backend.loadAmplitudes(state, amplitudes);
    let drift = 0;
    for (let i = 0; i < dim; i++) {
      drift += Math.abs((await backend.probability(state, i)) - expected[i]);
    }
    return { slice, drift, qubits };
  } finally {
    await backend.destroyState(state);
  }
}
