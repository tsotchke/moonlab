/**
 * Integration tests for the adaptive-bond TDVP engine.
 *
 * Run `pnpm build:wasm` first.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { TdvpEngine, EvolutionType } from '../tdvp';

describe('TdvpEngine TFIM lifecycle', () => {
  let e: TdvpEngine | null = null;
  afterEach(() => { e?.dispose(); e = null; });

  it('creates a TFIM engine of the requested size', async () => {
    e = await TdvpEngine.createTfim({
      numSites: 6, J: 1.0, h: 1.0,
      initialBondDim: 4, maxBondDim: 16,
      dt: 0.05, evolution: EvolutionType.RealTime,
    });
    expect(e.numBonds).toBe(5);  // N - 1 bonds on an open chain
    expect(e.currentTime).toBeCloseTo(0.0, 12);
    expect(e.currentNorm).toBeCloseTo(1.0, 8);
  });

  it('rejects numSites < 2', async () => {
    await expect(TdvpEngine.createTfim({ numSites: 1 } as any))
      .rejects.toThrow(/numSites/);
  });

  it('dispose is idempotent', async () => {
    e = await TdvpEngine.createTfim({ numSites: 4 });
    e.dispose();
    e.dispose();
  });
});

describe('TdvpEngine TFIM real-time evolution', () => {
  it('one step advances time by dt', async () => {
    const e = await TdvpEngine.createTfim({
      numSites: 4, J: 1.0, h: 1.0,
      dt: 0.05, evolution: EvolutionType.RealTime,
    });
    e.step();
    expect(e.currentTime).toBeCloseTo(0.05, 8);
    e.dispose();
  });

  it('evolveTo lands on the target time', async () => {
    const e = await TdvpEngine.createTfim({
      numSites: 4, J: 1.0, h: 1.0,
      dt: 0.05, evolution: EvolutionType.RealTime,
    });
    e.evolveTo(0.3);
    expect(e.currentTime).toBeGreaterThanOrEqual(0.3 - 1e-9);
    // Bond dimensions may have grown.
    expect(e.currentMaxBondDim).toBeGreaterThanOrEqual(1);
    e.dispose();
  });

  it('norm stays near 1 through 5 real-time steps', async () => {
    const e = await TdvpEngine.createTfim({
      numSites: 4, J: 1.0, h: 0.5,
      dt: 0.05, evolution: EvolutionType.RealTime,
    });
    for (let i = 0; i < 5; i++) e.step();
    expect(Math.abs(e.currentNorm - 1.0)).toBeLessThan(0.05);
    e.dispose();
  });

  it('history records steps as they happen', async () => {
    const e = await TdvpEngine.createTfim({
      numSites: 4, J: 1.0, h: 0.5,
      dt: 0.05, evolution: EvolutionType.RealTime,
    });
    e.step();
    e.step();
    e.step();
    expect(e.historyNumSteps).toBe(3);
    const s0 = e.historyStep(0);
    expect(s0.time).toBeCloseTo(0.05, 8);
    expect(typeof s0.energy).toBe('number');
    expect(typeof s0.norm).toBe('number');
    e.dispose();
  });
});

describe('TdvpEngine TFIM imaginary-time ground state', () => {
  it('drives energy downward (cooling)', async () => {
    const e = await TdvpEngine.createTfim({
      numSites: 6, J: 1.0, h: 1.0,
      initialBondDim: 4, maxBondDim: 16,
      dt: 0.05, evolution: EvolutionType.ImaginaryTime,
    });
    const e0 = e.currentEnergy;
    e.evolveTo(0.5);
    const e1 = e.currentEnergy;
    expect(e1).toBeLessThan(e0);
    e.dispose();
  });
});

describe('TdvpEngine Heisenberg', () => {
  it('creates and steps a Heisenberg-XXZ engine', async () => {
    const e = await TdvpEngine.createHeisenberg({
      numSites: 4, J: 1.0, Delta: 1.0, h: 0.0,
      initialBondDim: 4, maxBondDim: 16,
      dt: 0.02, evolution: EvolutionType.ImaginaryTime,
    });
    expect(e.numBonds).toBe(3);
    e.step();
    expect(e.currentTime).toBeCloseTo(0.02, 8);
    e.dispose();
  });
});

describe('TdvpEngine bond inspection', () => {
  it('bondChi returns per-bond bond dim within [0, maxBondDim]', async () => {
    // bondChi reports the per-bond SVD-truncated chi; bonds the
    // engine hasn't visited yet report 0 -- the C-side TDVP sweep
    // updates bond_states lazily as the sweep visits each link.
    const e = await TdvpEngine.createTfim({
      numSites: 6, J: 1.0, h: 0.5,
      initialBondDim: 4, maxBondDim: 16,
      dt: 0.05, evolution: EvolutionType.RealTime,
    });
    e.evolveTo(0.2);
    for (let b = 0; b < e.numBonds; b++) {
      const chi = e.bondChi(b);
      expect(chi).toBeGreaterThanOrEqual(0);
      expect(chi).toBeLessThanOrEqual(16);
    }
    e.dispose();
  });

  it('rejects bondChi out of range', async () => {
    const e = await TdvpEngine.createTfim({ numSites: 4 });
    expect(() => e.bondChi(100)).toThrow();
    e.dispose();
  });
});
