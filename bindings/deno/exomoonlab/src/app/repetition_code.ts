/**
 * The repetition code, and its decoding threshold.
 *
 * The smallest error-correcting code that shows the effect worth showing: as
 * the code gets longer, the logical error rate falls below the physical one --
 * but only while the physical rate is under threshold. Above it, adding
 * distance makes things *worse*, and the curves cross. That crossing is the
 * whole point of the picture.
 *
 * MoonLab's union-find decoder does the matching; this file only builds the
 * detector error model, samples shots, and scores the answers.
 */

import { type DetectorGraph, type MoonLabBackend, UF_BOUNDARY } from "../backend/mod.ts";

/**
 * A distance-d repetition code: d data qubits, d-1 parity detectors.
 *
 * Edge j is a flip of data qubit j. It lights the detectors either side of
 * it, and the two boundary edges light only one. Edge 0 is the one declared
 * to flip the logical observable, matching the convention in MoonLab's own
 * uf_decoder test.
 */
export function repetitionCodeGraph(distance: number): DetectorGraph {
  if (distance < 2) throw new RangeError("distance must be at least 2");
  const numDetectors = distance - 1;
  const numEdges = distance;
  const edgeA = new Uint32Array(numEdges);
  const edgeB = new Uint32Array(numEdges);
  const edgeWeight = new Float64Array(numEdges);
  const edgeObs = new BigUint64Array(numEdges);

  for (let j = 0; j < numEdges; j++) {
    if (j === 0) {
      edgeA[j] = 0;
      edgeB[j] = UF_BOUNDARY;
      edgeObs[j] = 1n;
    } else if (j === numEdges - 1) {
      edgeA[j] = numDetectors - 1;
      edgeB[j] = UF_BOUNDARY;
      edgeObs[j] = 0n;
    } else {
      edgeA[j] = j - 1;
      edgeB[j] = j;
      edgeObs[j] = 0n;
    }
    edgeWeight[j] = 1;
  }
  return { numDetectors, numObservables: 1, edgeA, edgeB, edgeWeight, edgeObs };
}

/** A deterministic 32-bit PRNG, so a run is reproducible and testable. */
export function makeRng(seed: number): () => number {
  let state = (seed >>> 0) || 0x9e3779b9;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 0x100000000;
  };
}

export interface SampledShots {
  /** Detector-major: detector i of shot s at `i * shots + s`. */
  readonly detectors: Uint8Array;
  /** True logical flip per shot. */
  readonly truth: Uint8Array;
  readonly shots: number;
}

/**
 * Samples independent bit flips and derives the syndrome they produce.
 *
 * The truth is the parity of the observable-flipping edges that actually
 * fired, which is what the decoder is then asked to reconstruct from the
 * detectors alone.
 */
export function sampleShots(
  graph: DetectorGraph,
  distance: number,
  physicalError: number,
  shots: number,
  rng: () => number,
): SampledShots {
  const detectors = new Uint8Array(graph.numDetectors * shots);
  const truth = new Uint8Array(shots);

  for (let s = 0; s < shots; s++) {
    let logical = 0;
    const lit = new Uint8Array(graph.numDetectors);
    for (let j = 0; j < distance; j++) {
      if (rng() >= physicalError) continue;
      if (graph.edgeObs[j] === 1n) logical ^= 1;
      const a = graph.edgeA[j];
      const b = graph.edgeB[j];
      if (a !== UF_BOUNDARY) lit[a] ^= 1;
      if (b !== UF_BOUNDARY) lit[b] ^= 1;
    }
    truth[s] = logical;
    for (let i = 0; i < graph.numDetectors; i++) detectors[i * shots + s] = lit[i];
  }
  return { detectors, truth, shots };
}

export interface DecodePoint {
  readonly physical: number;
  readonly logical: number;
  readonly shots: number;
}

/** One (physical rate -> logical rate) measurement at a given distance. */
export async function measurePoint(
  backend: MoonLabBackend,
  distance: number,
  physicalError: number,
  shots: number,
  seed: number,
): Promise<DecodePoint> {
  const graph = repetitionCodeGraph(distance);
  const sample = sampleShots(graph, distance, physicalError, shots, makeRng(seed));
  const predicted = await backend.decodeBatch!(graph, sample.detectors, shots);
  let wrong = 0;
  for (let s = 0; s < shots; s++) if ((predicted[s] & 1) !== sample.truth[s]) wrong++;
  return { physical: physicalError, logical: wrong / shots, shots };
}

export interface ThresholdCurve {
  readonly distances: readonly number[];
  /** One series per distance, each a list of points. */
  readonly series: readonly (readonly DecodePoint[])[];
  readonly rates: readonly number[];
}

/** Sweeps physical error rate for several code distances. */
export async function measureThreshold(
  backend: MoonLabBackend,
  distances: readonly number[],
  rates: readonly number[],
  shots: number,
  seed = 12345,
): Promise<ThresholdCurve> {
  const series: DecodePoint[][] = [];
  for (const [di, distance] of distances.entries()) {
    const points: DecodePoint[] = [];
    for (const [ri, rate] of rates.entries()) {
      // Vary the seed per point so the distances are not correlated samples of
      // the same noise, which would flatten the crossing.
      points.push(await measurePoint(backend, distance, rate, shots, seed + di * 1000 + ri));
    }
    series.push(points);
  }
  return { distances, series, rates };
}
