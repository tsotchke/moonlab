/** Complete transverse-field Ising/QUBO annealing (Moonlab v1.2.1). */

import { getModule } from './wasm-loader';

export type AnnealSchedule = 'linear' | 'quadratic' | 'cosine';

export interface AnnealConfig {
  totalTime?: number;
  numSteps?: number;
  numSamples?: number;
  /** Zero asks Moonlab to assign and report a seed. */
  seed?: bigint;
  schedule?: AnnealSchedule;
  driverStrength?: number;
  problemStrength?: number;
  secondOrder?: boolean;
}

export interface AnnealResult {
  numQubits: number;
  effectiveSeed: bigint;
  bestBitstring: bigint;
  mostLikelyBitstring: bigint;
  groundBitstring: bigint;
  groundDegeneracy: number;
  bestEnergy: number;
  groundEnergy: number;
  expectedEnergy: number;
  successProbability: number;
  residualEnergy: number;
  problemGap: number;
  finalNorm: number;
  samples: bigint[];
  sampleEnergies: number[];
}

export interface AnnealingWasmModule {
  HEAPU8: Uint8Array;
  HEAPF64: Float64Array;
  _malloc(bytes: number): number;
  _free(ptr: number): void;
  _moonlab_anneal_ising_v1(
    n: number, h: number, j: number, offset: number,
    totalTime: number, steps: number, samples: number, seed: bigint,
    schedule: number, driver: number, problem: number, secondOrder: number,
    summary: number, samplesOut: number, energiesOut: number,
  ): number;
  _moonlab_anneal_qubo_v1(
    n: number, q: number, offset: number,
    totalTime: number, steps: number, samples: number, seed: bigint,
    schedule: number, driver: number, problem: number, secondOrder: number,
    summary: number, samplesOut: number, energiesOut: number,
  ): number;
}

const SUMMARY_BYTES = 96;
const scheduleCode: Record<AnnealSchedule, number> = {
  linear: 0, quadratic: 1, cosine: 2,
};

function resolved(config: AnnealConfig = {}) {
  const result = {
    totalTime: config.totalTime ?? 10,
    numSteps: config.numSteps ?? 1000,
    numSamples: config.numSamples ?? 1024,
    seed: config.seed ?? 0n,
    schedule: scheduleCode[config.schedule ?? 'cosine'],
    driverStrength: config.driverStrength ?? 1,
    problemStrength: config.problemStrength ?? 1,
    secondOrder: config.secondOrder ?? true,
  };
  if (!(result.totalTime > 0) || !Number.isInteger(result.numSteps) ||
      result.numSteps < 1 || result.numSteps > 10_000_000 ||
      !Number.isInteger(result.numSamples) || result.numSamples < 1 ||
      result.numSamples > 10_000_000 || !Number.isInteger(result.schedule) ||
      result.seed < 0n ||
      result.seed > 0xffff_ffff_ffff_ffffn ||
      !(result.driverStrength > 0) || !(result.problemStrength > 0)) {
    throw new RangeError('invalid annealing configuration');
  }
  return result;
}

function squareDimension(matrix: readonly number[]): number {
  const n = Math.sqrt(matrix.length);
  if (!Number.isInteger(n) || n < 1) {
    throw new RangeError('matrix must be non-empty and square');
  }
  return n;
}

function writeF64(mod: AnnealingWasmModule, values: readonly number[]): number {
  const ptr = mod._malloc(values.length * 8);
  if (!ptr) throw new Error('Moonlab WASM allocation failed');
  mod.HEAPF64.set(values, ptr >>> 3);
  return ptr;
}

function parseResult(mod: AnnealingWasmModule, summary: number,
                     samplesPtr: number, energiesPtr: number,
                     n: number, count: number): AnnealResult {
  const view = new DataView(mod.HEAPU8.buffer);
  const u64 = (offset: number) => view.getBigUint64(summary + offset, true);
  const f64 = (offset: number) => view.getFloat64(summary + offset, true);
  const samples: bigint[] = [];
  const sampleEnergies: number[] = [];
  for (let i = 0; i < count; i++) {
    samples.push(view.getBigUint64(samplesPtr + i * 8, true));
    sampleEnergies.push(view.getFloat64(energiesPtr + i * 8, true));
  }
  return {
    numQubits: n,
    effectiveSeed: u64(0),
    bestBitstring: u64(8),
    mostLikelyBitstring: u64(16),
    groundBitstring: u64(24),
    groundDegeneracy: view.getUint32(summary + 32, true),
    bestEnergy: f64(40),
    groundEnergy: f64(48),
    expectedEnergy: f64(56),
    successProbability: f64(64),
    residualEnergy: f64(72),
    problemGap: f64(80),
    finalNorm: f64(88),
    samples,
    sampleEnergies,
  };
}

function buffers(mod: AnnealingWasmModule, count: number) {
  const summary = mod._malloc(SUMMARY_BYTES);
  const samples = mod._malloc(count * 8);
  const energies = mod._malloc(count * 8);
  if (!summary || !samples || !energies) {
    if (summary) mod._free(summary);
    if (samples) mod._free(samples);
    if (energies) mod._free(energies);
    throw new Error('Moonlab WASM allocation failed');
  }
  return { summary, samples, energies };
}

export async function annealIsing(
  fields: readonly number[], couplings: readonly number[], offset = 0,
  config: AnnealConfig = {},
): Promise<AnnealResult> {
  return annealIsingWithModule(
    (await getModule()) as unknown as AnnealingWasmModule,
    fields, couplings, offset, config);
}

/** Synchronous dependency-injected variant for embedders and unit tests. */
export function annealIsingWithModule(mod: AnnealingWasmModule,
                                      fields: readonly number[],
                                      couplings: readonly number[],
                                      offset = 0,
                                      config: AnnealConfig = {}): AnnealResult {
  const n = squareDimension(couplings);
  if (fields.length !== n) throw new RangeError('fields length must match matrix');
  const c = resolved(config);
  const h = writeF64(mod, fields), j = writeF64(mod, couplings);
  const out = buffers(mod, c.numSamples);
  try {
    const rc = mod._moonlab_anneal_ising_v1(
      n, h, j, offset, c.totalTime, c.numSteps, c.numSamples, c.seed,
      c.schedule, c.driverStrength, c.problemStrength,
      c.secondOrder ? 1 : 0, out.summary, out.samples, out.energies);
    if (rc !== 0) throw new Error(`moonlab_anneal_ising_v1 failed: ${rc}`);
    return parseResult(mod, out.summary, out.samples, out.energies, n, c.numSamples);
  } finally {
    mod._free(h); mod._free(j);
    mod._free(out.summary); mod._free(out.samples); mod._free(out.energies);
  }
}

export async function annealQubo(
  qubo: readonly number[], offset = 0, config: AnnealConfig = {},
): Promise<AnnealResult> {
  return annealQuboWithModule(
    (await getModule()) as unknown as AnnealingWasmModule,
    qubo, offset, config);
}

/** Synchronous dependency-injected variant for embedders and unit tests. */
export function annealQuboWithModule(mod: AnnealingWasmModule,
                                     qubo: readonly number[],
                                     offset = 0,
                                     config: AnnealConfig = {}): AnnealResult {
  const n = squareDimension(qubo);
  const c = resolved(config);
  const q = writeF64(mod, qubo);
  const out = buffers(mod, c.numSamples);
  try {
    const rc = mod._moonlab_anneal_qubo_v1(
      n, q, offset, c.totalTime, c.numSteps, c.numSamples, c.seed,
      c.schedule, c.driverStrength, c.problemStrength,
      c.secondOrder ? 1 : 0, out.summary, out.samples, out.energies);
    if (rc !== 0) throw new Error(`moonlab_anneal_qubo_v1 failed: ${rc}`);
    return parseResult(mod, out.summary, out.samples, out.energies, n, c.numSamples);
  } finally {
    mod._free(q);
    mod._free(out.summary); mod._free(out.samples); mod._free(out.energies);
  }
}
