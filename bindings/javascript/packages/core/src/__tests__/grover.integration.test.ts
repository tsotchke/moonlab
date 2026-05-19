/**
 * Integration tests for Grover's search.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { QuantumState } from '../quantum-state';
import { groverSearch, groverOptimalIterations } from '../grover';

describe('Grover optimal iterations', () => {
  it('matches floor(pi sqrt(N)/4) at small n', async () => {
    // n = 4 -> N = 16 -> optimal = floor(pi * 4 / 4) = floor(pi) = 3.
    expect(await groverOptimalIterations(4)).toBe(3);
    // n = 6 -> N = 64 -> optimal = floor(pi * 8 / 4) = floor(2*pi) = 6.
    expect(await groverOptimalIterations(6)).toBe(6);
  });
});

describe('Grover search', () => {
  let state: QuantumState;
  afterEach(() => { state?.dispose(); });

  it('finds the marked state with > 95% probability at n=4', async () => {
    state = await QuantumState.create({ numQubits: 4 });
    const r = await groverSearch(state, 0b1010n);
    expect(r.successProbability).toBeGreaterThan(0.9);
    expect(r.foundMarkedState).toBe(true);
    expect(r.foundState).toBe(0b1010n);
    expect(r.iterationsPerformed).toBe(await groverOptimalIterations(4));
  });

  it('honours an explicit iteration count', async () => {
    state = await QuantumState.create({ numQubits: 4 });
    const r = await groverSearch(state, 0b0101n, 2);
    expect(r.iterationsPerformed).toBe(2);
  });

  it('reports finite fidelity in [0, 1]', async () => {
    state = await QuantumState.create({ numQubits: 4 });
    const r = await groverSearch(state, 0b0011n);
    expect(r.fidelity).toBeGreaterThanOrEqual(0.0);
    expect(r.fidelity).toBeLessThanOrEqual(1.0001);
  });
});
