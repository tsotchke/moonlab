import { describe, expect, it } from 'vitest';
import { annealQubo } from '../annealing';

describe('quantum annealing wasm integration', () => {
  it('solves and deterministically replays a two-variable QUBO', async () => {
    const config = {
      totalTime: 12, numSteps: 1200, numSamples: 128,
      seed: 0x1234_5678_9abc_def0n,
      schedule: 'cosine' as const,
    };
    const q = [-1, 1, 1, -1];
    const first = await annealQubo(q, 1, config);
    const second = await annealQubo(q, 1, config);
    expect(first.samples).toEqual(second.samples);
    expect(first.sampleEnergies).toEqual(second.sampleEnergies);
    expect(first.groundDegeneracy).toBe(2);
    expect(first.groundEnergy).toBeCloseTo(0, 12);
    expect(first.problemGap).toBeCloseTo(1, 12);
    expect(first.bestEnergy).toBeCloseTo(0, 12);
    expect(first.successProbability).toBeGreaterThan(0.95);
    expect(first.finalNorm).toBeCloseTo(1, 10);
  });
});
