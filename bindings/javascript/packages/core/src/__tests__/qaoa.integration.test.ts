/**
 * Integration tests for the QAOA wrapper.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { Graph, IsingModel, QaoaSolver } from '../qaoa';

describe('Graph', () => {
  it('builds a triangle without errors', async () => {
    const g = await Graph.create(3, [[0, 1, 1.0], [1, 2, 1.0], [2, 0, 1.0]]);
    g.dispose();
  });

  it('rejects an invalid edge index', async () => {
    await expect(
      Graph.create(3, [[0, 5, 1.0]])
    ).rejects.toThrow(/graph_add_edge/);
  });
});

describe('IsingModel MaxCut encoding', () => {
  let g: Graph | null = null;
  let ising: IsingModel | null = null;
  afterEach(() => {
    ising?.dispose(); ising = null;
    g?.dispose();    g = null;
  });

  it('triangle: all-zero (no cut) energy > one-flipped (2-edge cut)', async () => {
    g = await Graph.create(3, [[0, 1, 1.0], [1, 2, 1.0], [2, 0, 1.0]]);
    ising = await IsingModel.fromMaxcut(g);
    const e_000 = ising.evaluate(0n);
    const e_101 = ising.evaluate(5n);
    expect(e_000).toBeGreaterThan(e_101);
    expect(Number.isFinite(e_101)).toBe(true);
  });
});

describe('QaoaSolver on triangle MaxCut', () => {
  it('runs a p=1 sweep and returns plausible gamma/beta angles', async () => {
    const g = await Graph.create(3, [[0, 1, 1.0], [1, 2, 1.0], [2, 0, 1.0]]);
    const ising = await IsingModel.fromMaxcut(g);
    const solver = await QaoaSolver.create(ising, 1);
    const r = solver.solve();
    expect(r.numLayers).toBe(1);
    expect(r.optimalGamma.length).toBe(1);
    expect(r.optimalBeta.length).toBe(1);
    expect(Number.isFinite(r.bestEnergy)).toBe(true);
    expect(Number.isFinite(r.approximationRatio)).toBe(true);
    solver.dispose();
    ising.dispose();
    g.dispose();
  });

  it('computeExpectation returns a finite value at fixed angles', async () => {
    const g = await Graph.create(2, [[0, 1, 1.0]]);
    const ising = await IsingModel.fromMaxcut(g);
    const solver = await QaoaSolver.create(ising, 1);
    const e = solver.computeExpectation(
      new Float64Array([0.3]),
      new Float64Array([0.5]),
    );
    expect(Number.isFinite(e)).toBe(true);
    solver.dispose();
    ising.dispose();
    g.dispose();
  });

  it('rejects numLayers < 1', async () => {
    const g = await Graph.create(2, [[0, 1, 1.0]]);
    const ising = await IsingModel.fromMaxcut(g);
    await expect(QaoaSolver.create(ising, 0)).rejects.toThrow(/numLayers/);
    ising.dispose();
    g.dispose();
  });
});
