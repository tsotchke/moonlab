import { describe, expect, it } from 'vitest';
import { IsingModel } from '../ising-model';

describe('IsingModel WASM wrapper', () => {
  it('evaluates fields and couplings through the WASM Ising primitive', async () => {
    const model = await IsingModel.create({ numQubits: 3 });

    try {
      model
        .setField(0, -1)
        .setField(1, -0.5)
        .setField(2, -0.25)
        .setCoupling(0, 1, -0.125)
        .setCoupling(1, 2, -0.125);

      expect(model.evaluate(0)).toBeCloseTo(-2, 12);
      expect(model.evaluate(7)).toBeCloseTo(1.5, 12);
      expect(model.evaluate(1)).toBeCloseTo(0.25, 12);
    } finally {
      model.dispose();
    }

    expect(model.isDisposed).toBe(true);
    expect(() => model.evaluate(0)).toThrow('IsingModel has been disposed');
  });
});
