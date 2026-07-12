import { describe, expect, it } from 'vitest';
import { Grover, VQE, createH2Hamiltonian } from '../index';

describe('Grover', () => {
  it('amplifies the marked state', async () => {
    const grover = await Grover.create({ numQubits: 3, markedState: 5 });
    try {
      const result = grover.search();
      expect(result.markedState).toBe(5);
      expect(result.foundState).toBe(5);
      expect(result.oracleCalls).toBe(result.iterations);
      expect(result.successProbability).toBeGreaterThan(0.9);
      expect(result.topStates[0].index).toBe(5);
    } finally {
      grover.dispose();
    }
  });

  it('rejects invalid marked states', async () => {
    await expect(Grover.create({ numQubits: 3, markedState: 8 })).rejects.toThrow();
  });
});

describe('VQE', () => {
  it('builds an H2 Hamiltonian and optimizes a variational ansatz', async () => {
    const hamiltonian = createH2Hamiltonian({ bondDistance: 0.74, basis: 'sto-3g' });
    expect(hamiltonian.terms.length).toBeGreaterThan(0);

    const vqe = await VQE.create({
      hamiltonian,
      ansatz: 'uccsd',
      optimizer: 'cobyla',
      maxIterations: 48,
    });

    const result = vqe.solve();
    expect(result.energy).toBeLessThan(-1);
    expect(result.evaluations).toBeGreaterThanOrEqual(48);
    expect(result.parameters).toHaveLength(1);
    expect(result.probabilities).toHaveLength(4);
    expect(result.topStates.length).toBeGreaterThan(0);
    vqe.dispose();
  });
});
