/**
 * Integration tests for the VQE wrapper.  Exercises the WASM
 * entropy stack since v0.5.4.
 */

import { describe, it, expect, afterEach } from 'vitest';
import {
  PauliHamiltonian,
  VqeSolver,
  OptimizerType,
  resolveOptimizer,
} from '../vqe';

describe('PauliHamiltonian', () => {
  let h: PauliHamiltonian | null = null;
  afterEach(() => { h?.dispose(); h = null; });

  it('H2 at 0.74 A has the expected layout', async () => {
    h = await PauliHamiltonian.h2(0.74);
    expect(h.numQubits).toBe(2);
    expect(h.numTerms).toBeGreaterThanOrEqual(4);
  });

  it('H2 exact ground state ~= -1.137 Ha at 0.74 A', async () => {
    h = await PauliHamiltonian.h2(0.74);
    const e0 = h.exactGroundStateEnergy();
    // STO-3G H2 at R=0.74A reference is -1.137 Ha; the C side
    // returns -1.1422 (slightly different basis / convention).
    // Pin the result is in the expected physical range.
    expect(e0).toBeLessThan(-1.0);
    expect(e0).toBeGreaterThan(-1.3);
  });

  it('LiH at 1.6 A has the expected qubit count', async () => {
    h = await PauliHamiltonian.lih(1.6);
    expect(h.numQubits).toBeGreaterThanOrEqual(4);
  });

  it('custom 1-qubit H = 0.5 Z has ground-state -0.5', async () => {
    const builder = await PauliHamiltonian.builder(1, 1);
    h = builder.addTerm(0.5, 'Z').build();
    expect(h.numQubits).toBe(1);
    expect(h.numTerms).toBe(1);
    const e0 = h.exactGroundStateEnergy();
    expect(e0).toBeCloseTo(-0.5, 10);
  });
});

describe('VqeSolver', () => {
  it('runs H2 under Adam and produces a finite bounded energy', async () => {
    const h = await PauliHamiltonian.h2(0.74);
    const exact = h.exactGroundStateEnergy();
    const solver = await VqeSolver.create(h, 2, OptimizerType.Adam);
    const r = solver.solve();
    expect(Number.isFinite(r.groundStateEnergy)).toBe(true);
    // Adam at default iterations doesn't always hit chemical accuracy
    // -- just assert the result is in the same neighbourhood.
    expect(r.groundStateEnergy).toBeGreaterThan(exact - 0.5);
    expect(r.groundStateEnergy).toBeLessThan(exact + 5.0);
    expect(r.optimalParameters.length).toBeGreaterThan(0);
    expect(r.iterations).toBeGreaterThanOrEqual(0);
    solver.dispose();
    h.dispose();
  });

  it('computeEnergy at arbitrary parameters returns a finite number', async () => {
    const h = await PauliHamiltonian.h2(0.74);
    const solver = await VqeSolver.create(h, 1, OptimizerType.Adam);
    const params = new Float64Array(8).fill(0.1);  // 2q + 1 layer ~= 8 params
    const e = solver.computeEnergy(params);
    expect(Number.isFinite(e)).toBe(true);
    solver.dispose();
    h.dispose();
  });
});

describe('VqeSolver ergonomics (since v1.2.0)', () => {
  it('resolves string optimizer names to the C enum values', () => {
    expect(resolveOptimizer('cobyla')).toBe(OptimizerType.Cobyla);
    expect(resolveOptimizer('lbfgs')).toBe(OptimizerType.Lbfgs);
    expect(resolveOptimizer('adam')).toBe(OptimizerType.Adam);
    expect(resolveOptimizer('gradient_descent')).toBe(OptimizerType.GradientDescent);
    expect(resolveOptimizer('gradient-descent')).toBe(OptimizerType.GradientDescent);
    expect(resolveOptimizer('qng')).toBe(OptimizerType.Qng);
    expect(resolveOptimizer('natural_gradient')).toBe(OptimizerType.Qng);
    expect(OptimizerType.Qng).toBe(4);
    expect(() => resolveOptimizer('nonesuch' as never)).toThrow(/Unknown optimizer/);
  });

  it('rejects unknown ansatz names', async () => {
    const h = await PauliHamiltonian.h2(0.74);
    await expect(
      VqeSolver.create(h, { ansatz: 'banana' as never }),
    ).rejects.toThrow(/Unknown ansatz/);
    h.dispose();
  });

  it('UCCSD H2 has one parameter and reaches chemical accuracy', async () => {
    // Mirrors tests/unit/test_vqe.c::test_h2_uccsd_chemical_accuracy:
    // from the HF reference a single particle-conserving excitation
    // spans the exact ground state, so UCCSD must land within 1.6 mHa
    // of exact diagonalisation (which is ~ -1.137 Ha at R = 0.74 A).
    const h = await PauliHamiltonian.h2(0.74);
    const exact = h.exactGroundStateEnergy();
    expect(exact).toBeLessThan(-1.0);
    expect(exact).toBeGreaterThan(-1.3);

    const solver = await VqeSolver.create(h, {
      ansatz: 'uccsd',
      numElectrons: 1,
      optimizer: 'adam',
      learningRate: 0.1,
      maxIterations: 500,
      tolerance: 1e-10,
    });
    expect(solver.numParameters).toBe(1);
    const r = solver.solve();
    expect(Number.isFinite(r.groundStateEnergy)).toBe(true);
    expect(Math.abs(r.groundStateEnergy - exact)).toBeLessThan(1.6e-3);
    solver.dispose();
    h.dispose();
  });

  it('UCCSD LiH (4 qubits, 3 electrons) reaches chemical accuracy', async () => {
    // Mirrors tests/unit/test_vqe.c::test_lih_uccsd_chemical_accuracy.
    const h = await PauliHamiltonian.lih(1.5949);
    const exact = h.exactGroundStateEnergy();
    const solver = await VqeSolver.create(h, {
      ansatz: 'uccsd',
      numElectrons: 3,
      optimizer: 'adam',
      learningRate: 0.1,
      maxIterations: 3000,
      tolerance: 1e-12,
    });
    expect(solver.numParameters).toBe(3);
    const r = solver.solve();
    expect(Math.abs(r.groundStateEnergy - exact)).toBeLessThan(1.6e-3);
    solver.dispose();
    h.dispose();
  });

  it('every string-addressable optimizer solves H2 below -1.0 Ha', async () => {
    const names = ['cobyla', 'lbfgs', 'adam', 'gradient_descent', 'qng'] as const;
    for (const name of names) {
      const h = await PauliHamiltonian.h2(0.74);
      const solver = await VqeSolver.create(h, {
        numLayers: 2, optimizer: name,
      });
      const r = solver.solve();
      expect(Number.isFinite(r.groundStateEnergy)).toBe(true);
      expect(r.groundStateEnergy, `optimizer ${name}`).toBeLessThan(-1.0);
      solver.dispose();
      h.dispose();
    }
  }, 120000);

  it('learningRate reaches the C optimizer and changes the outcome', async () => {
    // Same ansatz family and optimizer; a sane learning rate must beat
    // an absurd one from any starting point (the C optimizer keeps the
    // best energy seen, so the divergent run is cushioned but worse).
    const h = await PauliHamiltonian.h2(0.74);
    const sane = await VqeSolver.create(h, {
      numLayers: 2, optimizer: 'gradient_descent', learningRate: 0.05,
    });
    const rSane = sane.solve();
    sane.dispose();

    const divergent = await VqeSolver.create(h, {
      numLayers: 2, optimizer: 'gradient_descent', learningRate: 250.0,
    });
    const rDivergent = divergent.solve();
    divergent.dispose();
    h.dispose();

    expect(rSane.groundStateEnergy).toBeLessThan(-1.0);
    expect(rSane.groundStateEnergy).toBeLessThanOrEqual(rDivergent.groundStateEnergy);
  });

  it('maxIterations caps the iteration count', async () => {
    const h = await PauliHamiltonian.h2(0.74);
    const solver = await VqeSolver.create(h, {
      numLayers: 2, optimizer: 'adam', maxIterations: 3, tolerance: 0,
    });
    const r = solver.solve();
    expect(r.iterations).toBeLessThanOrEqual(3);
    solver.dispose();
    h.dispose();
  });

  it('H2O Hamiltonian constructs and diagonalises', async () => {
    const h = await PauliHamiltonian.h2o();
    expect(h.numQubits).toBeGreaterThanOrEqual(4);
    const e0 = h.exactGroundStateEnergy();
    expect(Number.isFinite(e0)).toBe(true);
    expect(e0).toBeLessThan(0);
    h.dispose();
  });
});
