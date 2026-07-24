/**
 * Integration tests for the Pauli-frame batch shot sampler.
 *
 * Numeric checks against known distributions: GHZ shot correlations,
 * deterministic circuits, and noise-channel rates within a > 4-sigma
 * statistical tolerance.
 */

import { describe, it, expect } from 'vitest';
import * as pf from '../pauli-frame';

const SHOTS = 2000;

describe('pauliFrame.sampleCircuit', () => {
  it('deterministic circuit: X then measure reads 1 on every shot', async () => {
    const ops = [pf.x(0), pf.measure(0), pf.measure(1)];
    const { numMeasurements, samples } = await pf.sampleCircuit(2, ops, 100, { seed: 7 });
    expect(numMeasurements).toBe(2);
    for (let s = 0; s < 100; s++) {
      expect(samples[0 * 100 + s]).toBe(1);   // X-flipped qubit
      expect(samples[1 * 100 + s]).toBe(0);   // untouched qubit
    }
  });

  it('GHZ_3: perfect intra-shot correlation, ~50/50 marginal', async () => {
    const ops = [
      pf.h(0), pf.cnot(0, 1), pf.cnot(1, 2),
      pf.measure(0), pf.measure(1), pf.measure(2),
    ];
    expect(await pf.numMeasurements(ops)).toBe(3);

    const { numMeasurements, samples } = await pf.sampleCircuit(3, ops, SHOTS, { seed: 42 });
    expect(numMeasurements).toBe(3);

    let ones = 0;
    for (let s = 0; s < SHOTS; s++) {
      const m0 = samples[0 * SHOTS + s];
      const m1 = samples[1 * SHOTS + s];
      const m2 = samples[2 * SHOTS + s];
      // GHZ collapse: all three measurements agree exactly, every shot.
      expect(m1).toBe(m0);
      expect(m2).toBe(m0);
      ones += m0;
    }
    // Marginal is Bernoulli(1/2): sd = sqrt(0.25 / N) ~ 0.011;
    // 0.05 is ~4.5 sigma.
    expect(ones / SHOTS).toBeGreaterThan(0.5 - 0.05);
    expect(ones / SHOTS).toBeLessThan(0.5 + 0.05);
  });

  it('seeded runs are reproducible; different seeds differ', async () => {
    const ops = [
      pf.h(0), pf.cnot(0, 1),
      pf.measure(0), pf.measure(1),
    ];
    const a = await pf.sampleCircuit(2, ops, 500, { seed: 123 });
    const b = await pf.sampleCircuit(2, ops, 500, { seed: 123 });
    const c = await pf.sampleCircuit(2, ops, 500, { seed: 456 });
    expect(Array.from(a.samples)).toEqual(Array.from(b.samples));
    expect(Array.from(a.samples)).not.toEqual(Array.from(c.samples));
  });

  it('X_ERROR fires at its configured rate', async () => {
    const p = 0.2;
    const ops = [pf.xError(0, p), pf.measure(0)];
    const { samples } = await pf.sampleCircuit(1, ops, SHOTS, { seed: 9 });
    let ones = 0;
    for (let s = 0; s < SHOTS; s++) ones += samples[s];
    // sd = sqrt(0.2 * 0.8 / N) ~ 0.0089; 0.045 is ~5 sigma.
    expect(ones / SHOTS).toBeGreaterThan(p - 0.045);
    expect(ones / SHOTS).toBeLessThan(p + 0.045);
  });

  it('measureNoisy flips the record at its configured rate', async () => {
    const p = 0.15;
    const ops = [pf.measureNoisy(0, p)];
    const { samples } = await pf.sampleCircuit(1, ops, SHOTS, { seed: 11 });
    let ones = 0;
    for (let s = 0; s < SHOTS; s++) ones += samples[s];
    // sd = sqrt(0.15 * 0.85 / N) ~ 0.008; 0.04 is ~5 sigma.
    expect(ones / SHOTS).toBeGreaterThan(p - 0.04);
    expect(ones / SHOTS).toBeLessThan(p + 0.04);
  });

  it('depolarize1 flips a |0> readout with rate 2p/3', async () => {
    const p = 0.3;
    const ops = [pf.depolarize1(0, p), pf.measure(0)];
    const { samples } = await pf.sampleCircuit(1, ops, SHOTS, { seed: 13 });
    let ones = 0;
    for (let s = 0; s < SHOTS; s++) ones += samples[s];
    const expected = (2 * p) / 3;   // X and Y flip; Z does not.
    // sd = sqrt(0.2 * 0.8 / N) ~ 0.0089; 0.045 is ~5 sigma.
    expect(ones / SHOTS).toBeGreaterThan(expected - 0.045);
    expect(ones / SHOTS).toBeLessThan(expected + 0.045);
  });
});

describe('pauliFrame.sampleDetectors', () => {
  it('noiseless circuit: every detector reads 0', async () => {
    const ops = [
      pf.h(0), pf.cnot(0, 1),
      pf.measure(0), pf.measure(1),
    ];
    // One detector: parity of the two (perfectly correlated) readouts.
    const det = await pf.sampleDetectors(2, ops, [[0, 1]], SHOTS, { seed: 21 });
    for (let s = 0; s < SHOTS; s++) expect(det[s]).toBe(0);
  });

  it('single X error mechanism fires its detectors together', async () => {
    const p = 0.25;
    // Repetition-style readout: X_ERROR on the middle qubit flips m1,
    // firing both D0 = m0 xor m1 and D1 = m1 xor m2 in the same shot.
    const ops = [
      pf.xError(1, p),
      pf.measure(0), pf.measure(1), pf.measure(2),
    ];
    const det = await pf.sampleDetectors(
      3, ops, [[0, 1], [1, 2]], SHOTS, { seed: 33 },
    );
    let fires = 0;
    for (let s = 0; s < SHOTS; s++) {
      const d0 = det[0 * SHOTS + s];
      const d1 = det[1 * SHOTS + s];
      expect(d1).toBe(d0);   // the same error mechanism drives both
      fires += d0;
    }
    // sd = sqrt(0.25 * 0.75 / N) ~ 0.0097; 0.05 is ~5 sigma.
    expect(fires / SHOTS).toBeGreaterThan(p - 0.05);
    expect(fires / SHOTS).toBeLessThan(p + 0.05);
  });
});

describe('pauliFrame SIMD introspection', () => {
  it('reports a known backend and a positive lane width', async () => {
    const backend = await pf.simdBackend();
    expect(['neon', 'avx2', 'avx512', 'scalar']).toContain(backend);
    expect(await pf.simdLanes()).toBeGreaterThanOrEqual(1);
  });
});
