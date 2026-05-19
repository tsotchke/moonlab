/**
 * Integration tests for the surface code binding (since v0.5.14).
 * Mirrors the Rust + Python suites; pins the JS wrapper against
 * regression with the same physical-meaningful invariants.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { SurfaceCode } from '../surface-code';

describe('SurfaceCode lifecycle', () => {
  let code: SurfaceCode | null = null;
  afterEach(() => { code?.dispose(); code = null; });

  it('distance-3 has the expected layout', async () => {
    code = await SurfaceCode.create(3, 1n);
    expect(code.distance).toBe(3);
    expect(code.numDataQubits).toBe(9);
    expect(code.numAncillasPerSector).toBe(4);
  });

  it('rejects even or too-small distance', async () => {
    await expect(SurfaceCode.create(2, 1n)).rejects.toThrow(/odd and >= 3/);
    await expect(SurfaceCode.create(4, 1n)).rejects.toThrow(/odd and >= 3/);
    await expect(SurfaceCode.create(1, 1n)).rejects.toThrow(/odd and >= 3/);
  });

  it('dispose is idempotent', async () => {
    code = await SurfaceCode.create(3, 1n);
    code.dispose();
    code.dispose();
  });
});

describe('SurfaceCode lattice geometry', () => {
  it('dataIndex maps each (row, col) to a distinct linear index', async () => {
    const code = await SurfaceCode.create(3, 1n);
    const indices = new Set<number>();
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const idx = code.dataIndex(r, c);
        expect(idx).toBeGreaterThanOrEqual(0);
        expect(idx).toBeLessThan(9);
        indices.add(idx);
      }
    }
    expect(indices.size).toBe(9);
    code.dispose();
  });

  it('dataIndex rejects out-of-range coordinates', async () => {
    const code = await SurfaceCode.create(3, 1n);
    expect(() => code.dataIndex(3, 0)).toThrow(/out of \[0, 3\)/);
    expect(() => code.dataIndex(0, 3)).toThrow(/out of \[0, 3\)/);
    code.dispose();
  });
});

describe('SurfaceCode syndrome measurement', () => {
  it('Z-stabiliser measurement is idempotent on |0...0>', async () => {
    const code = await SurfaceCode.create(3, 42n);
    code.measureZSyndromes();
    const w0 = code.syndromeWeight;
    code.measureZSyndromes();
    expect(code.syndromeWeight).toBe(w0);
    code.dispose();
  });

  it('X error lights Z stabilisers', async () => {
    const code = await SurfaceCode.create(3, 42n);
    const q = code.dataIndex(1, 1);
    code.applyError(q, 'X');
    code.measureZSyndromes();
    expect(code.syndromeWeight).toBeGreaterThan(0);
    code.dispose();
  });

  it('Z error lights X stabilisers', async () => {
    const code = await SurfaceCode.create(3, 7n);
    const q = code.dataIndex(1, 1);
    code.applyError(q, 'Z');
    code.measureXSyndromes();
    expect(code.syndromeWeight).toBeGreaterThan(0);
    code.dispose();
  });
});

describe('SurfaceCode error injection', () => {
  it('rejects unknown error type', async () => {
    const code = await SurfaceCode.create(3, 1n);
    expect(() => code.applyError(0, 'W' as never)).toThrow(/X.*Y.*Z/);
    code.dispose();
  });

  it('rejects out-of-range qubit', async () => {
    const code = await SurfaceCode.create(3, 1n);
    expect(() => code.applyError(100, 'X')).toThrow(/out of/);
    code.dispose();
  });
});
