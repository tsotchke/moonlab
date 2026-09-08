import { describe, expect, it } from 'vitest';
import { annealQuboWithModule, type AnnealingWasmModule } from '../annealing';

function fakeModule() {
  const memory = new ArrayBuffer(65536);
  let next = 1024;
  let observed: unknown[] = [];
  const module: AnnealingWasmModule = {
    HEAPU8: new Uint8Array(memory),
    HEAPF64: new Float64Array(memory),
    _malloc(bytes) { const p = next; next += (bytes + 7) & ~7; return p; },
    _free() {},
    _moonlab_anneal_ising_v1() { return -99; },
    _moonlab_anneal_qubo_v1(
      n, _q, offset, total, steps, count, seed, schedule,
      driver, problem, second, summary, samples, energies,
    ) {
      observed = [n, offset, total, steps, count, seed, schedule,
                  driver, problem, second];
      const v = new DataView(memory);
      v.setBigUint64(summary + 0, seed, true);
      v.setBigUint64(summary + 8, 1n, true);
      v.setBigUint64(summary + 16, 2n, true);
      v.setBigUint64(summary + 24, 1n, true);
      v.setUint32(summary + 32, 2, true);
      for (const [off, value] of [[40, 0], [48, 0], [56, 0.1], [64, 0.9],
                                  [72, 0.1], [80, 1], [88, 1]] as const) {
        v.setFloat64(summary + off, value, true);
      }
      for (let i = 0; i < count; i++) {
        v.setBigUint64(samples + i * 8, BigInt(i & 1 ? 2 : 1), true);
        v.setFloat64(energies + i * 8, 0, true);
      }
      return 0;
    },
  };
  return { module, observed: () => observed };
}

describe('quantum annealing binding', () => {
  it('marshals the full config and parses uint64 samples', () => {
    const fake = fakeModule();
    const result = annealQuboWithModule(fake.module, [-1, 1, 1, -1], 1, {
      totalTime: 12, numSteps: 1200, numSamples: 4,
      seed: 0x1234_5678_9abc_def0n, schedule: 'cosine',
      driverStrength: 1.5, problemStrength: 0.75, secondOrder: true,
    });
    expect(fake.observed()).toEqual([
      2, 1, 12, 1200, 4, 0x1234_5678_9abc_def0n, 2, 1.5, 0.75, 1,
    ]);
    expect(result.groundDegeneracy).toBe(2);
    expect(result.successProbability).toBe(0.9);
    expect(result.samples).toEqual([1n, 2n, 1n, 2n]);
  });

  it('rejects a non-square QUBO before calling wasm', () => {
    const fake = fakeModule();
    expect(() => annealQuboWithModule(fake.module, [1, 2, 3])).toThrow(/square/);
  });
});
