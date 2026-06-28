/**
 * Integration tests for the CaPeps 2D Clifford-Assisted PEPS
 * simulator.
 *
 * Run `pnpm build:wasm` first.  Linear qubit index = x + Lx * y.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { CaPeps, PauliCode } from '../ca-peps';

describe('CaPeps lifecycle', () => {
  let s: CaPeps | null = null;
  afterEach(() => { s?.dispose(); s = null; });

  it('creates a 2x3 lattice with chi=8', async () => {
    s = await CaPeps.create(2, 3, 8);
    expect(s.lx).toBe(2);
    expect(s.ly).toBe(3);
    expect(s.numQubits).toBe(6);
    expect(s.maxBondDim).toBe(8);
  });

  it('initial norm is 1', async () => {
    s = await CaPeps.create(2, 2, 4);
    expect(s.norm).toBeCloseTo(1.0, 10);
  });

  it('rejects bad dimensions', async () => {
    await expect(CaPeps.create(0, 3, 4)).rejects.toThrow();
    await expect(CaPeps.create(3, 0, 4)).rejects.toThrow();
    await expect(CaPeps.create(2, 2, 0)).rejects.toThrow();
  });

  it('clone is independent of the source', async () => {
    s = await CaPeps.create(2, 2, 4);
    s.h(0);
    const c = s.clone();
    // h(0) again on s; toggling back to |0>.  c stays on H|0>.
    s.h(0);
    const z_orig = s.expectPauliSingle(0, PauliCode.Z);
    const z_clone = c.expectPauliSingle(0, PauliCode.Z);
    expect(z_orig).toBeCloseTo(1.0, 8);
    expect(Math.abs(z_clone)).toBeLessThan(1e-6);
    c.dispose();
  });
});

describe('CaPeps gate semantics', () => {
  it('|0...0> has <Z_q> = 1 on every qubit', async () => {
    const s = await CaPeps.create(2, 2, 4);
    for (let q = 0; q < 4; q++) {
      expect(s.expectPauliSingle(q, PauliCode.Z)).toBeCloseTo(1.0, 10);
    }
    s.dispose();
  });

  it('Hadamard on qubit 0 zeros <Z_0> but keeps <Z_1> = 1', async () => {
    const s = await CaPeps.create(2, 2, 4);
    s.h(0);
    expect(Math.abs(s.expectPauliSingle(0, PauliCode.Z))).toBeLessThan(1e-10);
    expect(s.expectPauliSingle(1, PauliCode.Z)).toBeCloseTo(1.0, 10);
    s.dispose();
  });

  it('Bell pair via H(0) + CNOT(0,1) has perfect ZZ correlation', async () => {
    const s = await CaPeps.create(2, 1, 4);
    s.h(0).cnot(0, 1);
    const [re, im] = s.expectPauli([PauliCode.Z, PauliCode.Z]);
    expect(re).toBeCloseTo(1.0, 10);
    expect(Math.abs(im)).toBeLessThan(1e-10);
    s.dispose();
  });

  it('Bell pair single-qubit <Z_0> = 0', async () => {
    const s = await CaPeps.create(2, 1, 4);
    s.h(0).cnot(0, 1);
    expect(Math.abs(s.expectPauliSingle(0, PauliCode.Z))).toBeLessThan(1e-10);
    s.dispose();
  });

  it('probZ on |0> is 1, on H|0> is 0.5', async () => {
    const s = await CaPeps.create(2, 1, 4);
    expect(s.probZ(0)).toBeCloseTo(1.0, 10);
    s.h(0);
    expect(s.probZ(0)).toBeCloseTo(0.5, 10);
    s.dispose();
  });

  it('non-Clifford gates apply', async () => {
    const s = await CaPeps.create(2, 1, 4);
    s.rx(0, Math.PI / 4);   // some non-Clifford rotation
    s.t(0);                  // T gate
    s.tdg(0);                // T-dagger
    s.phase(0, 0.123);
    expect(s.norm).toBeGreaterThan(0.9);  // physical state
    s.dispose();
  });
});

describe('CaPeps validation', () => {
  it('rejects pauli string of wrong length', async () => {
    const s = await CaPeps.create(2, 2, 4);
    expect(() => s.expectPauli([PauliCode.Z, PauliCode.Z])).toThrow(/numQubits=4/);
    s.dispose();
  });

  it('rejects out-of-range qubit', async () => {
    const s = await CaPeps.create(2, 2, 4);
    expect(() => s.h(10)).toThrow(/qubit 10/);
    expect(() => s.expectPauliSingle(10, PauliCode.Z)).toThrow(/qubit 10/);
    s.dispose();
  });

  it('throws after dispose', async () => {
    const s = await CaPeps.create(2, 2, 4);
    s.dispose();
    expect(() => s.h(0)).toThrow(/disposed/);
  });
});
