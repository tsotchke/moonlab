/**
 * Integration tests for the Mpdo class (matrix-product density
 * operator mixed-state simulator).
 *
 * Run `pnpm build:wasm` first.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { Mpdo, PauliCode } from '../mpdo';

describe('Mpdo lifecycle', () => {
  let m: Mpdo | null = null;
  afterEach(() => { m?.dispose(); m = null; });

  it('creates an MPDO of the requested shape', async () => {
    m = await Mpdo.create(4, 16);
    expect(m.numQubits).toBe(4);
    expect(m.maxBondDim).toBe(16);
  });

  it('initial state is a product state (currentBondDim = 1)', async () => {
    m = await Mpdo.create(4, 16);
    expect(m.currentBondDim).toBe(1);
  });

  it('initial trace equals 1', async () => {
    m = await Mpdo.create(4, 16);
    expect(m.trace()).toBeCloseTo(1.0, 10);
  });

  it('rejects bad dimensions', async () => {
    await expect(Mpdo.create(0, 4)).rejects.toThrow(/numQubits/);
    await expect(Mpdo.create(4, 0)).rejects.toThrow(/maxBondDim/);
  });

  it('dispose is idempotent', async () => {
    m = await Mpdo.create(2, 4);
    m.dispose();
    m.dispose();
  });

  it('clone is independent of the source', async () => {
    m = await Mpdo.create(2, 4);
    m.applyDepolarizing(0, 0.1);
    const c = m.clone();
    expect(c.numQubits).toBe(2);
    // After applying another channel only to the clone, the
    // original's <Z_0> should differ from the clone's.
    c.applyDepolarizing(0, 0.5);
    const z_orig = m.expectPauli(0, PauliCode.Z);
    const z_clone = c.expectPauli(0, PauliCode.Z);
    expect(Math.abs(z_orig - z_clone)).toBeGreaterThan(0.01);
    c.dispose();
  });
});

describe('Mpdo channel correctness on single qubit', () => {
  it('zero-probability bit-flip is identity', async () => {
    const m = await Mpdo.create(1, 4);
    const z_before = m.expectPauli(0, PauliCode.Z);
    m.applyBitFlip(0, 0.0);
    const z_after = m.expectPauli(0, PauliCode.Z);
    expect(z_after).toBeCloseTo(z_before, 10);
    m.dispose();
  });

  it('full amplitude damping resets <Z> to +1 (|0> ground)', async () => {
    const m = await Mpdo.create(1, 4);
    m.applyAmplitudeDamping(0, 1.0);
    expect(m.expectPauli(0, PauliCode.Z)).toBeCloseTo(1.0, 6);
    m.dispose();
  });

  it('full depolarising channel sends <Z> to 0', async () => {
    // p = 3/4 sends every state to I/2 (maximally mixed); <Z> = 0.
    const m = await Mpdo.create(1, 4);
    m.applyDepolarizing(0, 0.75);
    expect(Math.abs(m.expectPauli(0, PauliCode.Z))).toBeLessThan(1e-6);
    m.dispose();
  });

  it('trace stays at 1 under any single-qubit channel', async () => {
    const m = await Mpdo.create(3, 8);
    m.applyDepolarizing(0, 0.1);
    m.applyAmplitudeDamping(1, 0.2);
    m.applyPhaseDamping(2, 0.05);
    m.applyBitFlip(0, 0.05);
    m.applyPhaseFlip(1, 0.05);
    m.applyBitPhaseFlip(2, 0.05);
    expect(m.trace()).toBeCloseTo(1.0, 6);
    m.dispose();
  });

  it('PauliCode.I expectation is exactly 1 on any state', async () => {
    const m = await Mpdo.create(3, 4);
    m.applyDepolarizing(0, 0.1);  // perturb
    expect(m.expectPauli(1, PauliCode.I)).toBeCloseTo(1.0, 12);
    m.dispose();
  });
});

describe('Mpdo range checks', () => {
  it('rejects out-of-range qubit on every applier', async () => {
    const m = await Mpdo.create(2, 4);
    expect(() => m.applyDepolarizing(5, 0.1)).toThrow(/qubit 5/);
    expect(() => m.applyAmplitudeDamping(5, 0.1)).toThrow(/qubit 5/);
    expect(() => m.applyPhaseFlip(5, 0.1)).toThrow(/qubit 5/);
    expect(() => m.expectPauli(5, PauliCode.Z)).toThrow(/qubit 5/);
    m.dispose();
  });
});
