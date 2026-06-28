/**
 * Integration tests for topological invariants.  All 7 wrappers
 * go through the moonlab_export_lean.c convenience layer.
 */

import { describe, it, expect } from 'vitest';
import {
  qwzChern, chernQwzProj, chernQwzParallelTransport,
  sshWinding, kitaevChainZ2, kaneMeleZ2, bhzZ2, hofstadterChern,
} from '../topology';

describe('QWZ Chern number', () => {
  it('returns -1 in topological phase (m=1)', async () => {
    expect(await qwzChern({ m: 1.0, n: 16 })).toBe(-1);
    expect(await chernQwzProj({ m: 1.0, n: 16 })).toBe(-1);
    expect(await chernQwzParallelTransport({ m: 1.0, n: 16 })).toBe(-1);
  });

  it('returns 0 in trivial phase (m=3)', async () => {
    expect(await qwzChern({ m: 3.0, n: 16 })).toBe(0);
    expect(await chernQwzProj({ m: 3.0, n: 16 })).toBe(0);
  });

  it('three integrators agree at a given phase point', async () => {
    const a = await qwzChern({ m: -1.0, n: 16 });
    const b = await chernQwzProj({ m: -1.0, n: 16 });
    const c = await chernQwzParallelTransport({ m: -1.0, n: 16 });
    expect(a).toBe(b);
    expect(b).toBe(c);
  });
});

describe('SSH winding', () => {
  it('topological when |t2| > |t1|', async () => {
    expect(await sshWinding({ t1: 0.5, t2: 1.0, n: 64 })).toBe(1);
  });

  it('trivial when |t1| > |t2|', async () => {
    expect(await sshWinding({ t1: 1.0, t2: 0.5, n: 64 })).toBe(0);
  });
});

describe('Kitaev chain Z2', () => {
  it('topological when |mu| < 2|t| and delta != 0', async () => {
    expect(await kitaevChainZ2({ t: 1.0, mu: 0.5, delta: 1.0 })).toBe(1);
  });

  it('trivial when |mu| > 2|t|', async () => {
    expect(await kitaevChainZ2({ t: 1.0, mu: 3.0, delta: 1.0 })).toBe(0);
  });
});

describe('Kane-Mele Z2', () => {
  it('QSH phase at lambdaSo dominant', async () => {
    expect(await kaneMeleZ2({
      t: 1.0, lambdaSo: 0.1, lambdaR: 0.0, lambdaV: 0.1, n: 8,
    })).toBe(1);
  });

  it('trivial when lambdaV >> 3 sqrt(3) lambdaSo', async () => {
    // 3 sqrt(3) ~= 5.2; lambdaV/lambdaSo = 10 > 5.2 -> trivial.
    expect(await kaneMeleZ2({
      t: 1.0, lambdaSo: 0.05, lambdaR: 0.0, lambdaV: 0.5, n: 8,
    })).toBe(0);
  });
});

describe('BHZ Z2', () => {
  it('QSH for 0 < M/B < 8 (lattice regularised window)', async () => {
    expect(await bhzZ2({ A: 1.0, B: 1.0, M: 2.0, n: 8 })).toBe(1);
  });

  it('trivial when M/B is outside the topological window', async () => {
    expect(await bhzZ2({ A: 1.0, B: 1.0, M: -1.0, n: 8 })).toBe(0);
  });
});

describe('Hofstadter Chern', () => {
  it('lowest band at phi = 1/3 has Chern +1', async () => {
    expect(await hofstadterChern({
      t: 1.0, p: 1, q: 3, nOccupied: 1, n: 16,
    })).toBe(1);
  });
});
