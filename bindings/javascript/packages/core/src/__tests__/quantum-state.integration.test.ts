/**
 * Integration tests for QuantumState class
 *
 * These tests require the WASM module to be built first.
 * Run `pnpm build` before running these tests.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { QuantumState } from '../quantum-state';
import { magnitudeSquared } from '../complex';

describe('QuantumState Creation', () => {
  let state: QuantumState;

  afterEach(() => {
    state?.dispose();
  });

  it('creates single qubit state', async () => {
    state = await QuantumState.create({ numQubits: 1 });
    expect(state.numQubits).toBe(1);
    expect(state.stateDim).toBe(2);
  });

  it('creates multi-qubit state', async () => {
    state = await QuantumState.create({ numQubits: 4 });
    expect(state.numQubits).toBe(4);
    expect(state.stateDim).toBe(16);
  });

  it('initializes to |0...0> state', async () => {
    state = await QuantumState.create({ numQubits: 2 });
    const probs = state.getProbabilities();
    expect(probs[0]).toBeCloseTo(1); // |00> has probability 1
    expect(probs[1]).toBeCloseTo(0);
    expect(probs[2]).toBeCloseTo(0);
    expect(probs[3]).toBeCloseTo(0);
  });

  it('rejects invalid qubit counts', async () => {
    await expect(QuantumState.create({ numQubits: 0 })).rejects.toThrow();
    await expect(QuantumState.create({ numQubits: 31 })).rejects.toThrow();
  });

  it('creates state with initial amplitudes', async () => {
    // |+> state
    const amp = 1 / Math.SQRT2;
    state = await QuantumState.create({
      numQubits: 1,
      amplitudes: [{ real: amp, imag: 0 }, { real: amp, imag: 0 }],
    });
    const probs = state.getProbabilities();
    expect(probs[0]).toBeCloseTo(0.5);
    expect(probs[1]).toBeCloseTo(0.5);
  });
});

describe('Single-Qubit Gates', () => {
  let state: QuantumState;

  afterEach(() => {
    state?.dispose();
  });

  it('applies Hadamard gate', async () => {
    state = await QuantumState.create({ numQubits: 1 });
    state.h(0);
    const probs = state.getProbabilities();
    expect(probs[0]).toBeCloseTo(0.5);
    expect(probs[1]).toBeCloseTo(0.5);
  });

  it('applies X gate (NOT)', async () => {
    state = await QuantumState.create({ numQubits: 1 });
    state.x(0);
    const probs = state.getProbabilities();
    expect(probs[0]).toBeCloseTo(0); // |0> -> |1>
    expect(probs[1]).toBeCloseTo(1);
  });

  it('X gate is self-inverse', async () => {
    state = await QuantumState.create({ numQubits: 1 });
    state.x(0).x(0);
    const probs = state.getProbabilities();
    expect(probs[0]).toBeCloseTo(1); // Back to |0>
  });

  it('applies Y gate', async () => {
    state = await QuantumState.create({ numQubits: 1 });
    state.y(0);
    const probs = state.getProbabilities();
    expect(probs[0]).toBeCloseTo(0);
    expect(probs[1]).toBeCloseTo(1);
  });

  it('applies Z gate (no effect on |0>)', async () => {
    state = await QuantumState.create({ numQubits: 1 });
    state.z(0);
    const probs = state.getProbabilities();
    expect(probs[0]).toBeCloseTo(1); // Z|0> = |0>
  });

  it('Z gate adds phase to |1>', async () => {
    state = await QuantumState.create({ numQubits: 1 });
    state.x(0).z(0); // |1> -> -|1>
    const amps = state.getAmplitudes();
    expect(amps[1].real).toBeCloseTo(-1);
  });

  it('applies S gate', async () => {
    state = await QuantumState.create({ numQubits: 1 });
    state.x(0).s(0); // |1> -> i|1>
    const amps = state.getAmplitudes();
    expect(amps[1].real).toBeCloseTo(0);
    expect(amps[1].imag).toBeCloseTo(1);
  });

  it('applies T gate', async () => {
    state = await QuantumState.create({ numQubits: 1 });
    state.x(0).t(0); // |1> -> e^(iπ/4)|1>
    const amps = state.getAmplitudes();
    expect(magnitudeSquared(amps[1])).toBeCloseTo(1);
  });

  it('applies rotation gates', async () => {
    state = await QuantumState.create({ numQubits: 1 });
    state.rx(0, Math.PI); // Rx(π)|0> = -i|1>
    const probs = state.getProbabilities();
    expect(probs[0]).toBeCloseTo(0);
    expect(probs[1]).toBeCloseTo(1);
  });

  it('Ry rotation creates superposition', async () => {
    state = await QuantumState.create({ numQubits: 1 });
    state.ry(0, Math.PI / 2);
    const probs = state.getProbabilities();
    expect(probs[0]).toBeCloseTo(0.5);
    expect(probs[1]).toBeCloseTo(0.5);
  });

  it('validates qubit index', async () => {
    state = await QuantumState.create({ numQubits: 2 });
    expect(() => state.h(-1)).toThrow();
    expect(() => state.h(2)).toThrow();
  });
});

describe('Two-Qubit Gates', () => {
  let state: QuantumState;

  afterEach(() => {
    state?.dispose();
  });

  it('applies CNOT gate', async () => {
    state = await QuantumState.create({ numQubits: 2 });
    state.x(0).cnot(0, 1); // |10> -> |11>
    const probs = state.getProbabilities();
    expect(probs[0b00]).toBeCloseTo(0);
    expect(probs[0b11]).toBeCloseTo(1);
  });

  it('CNOT creates Bell state', async () => {
    state = await QuantumState.create({ numQubits: 2 });
    state.h(0).cnot(0, 1); // (|00> + |11>)/√2
    const probs = state.getProbabilities();
    expect(probs[0b00]).toBeCloseTo(0.5);
    expect(probs[0b01]).toBeCloseTo(0);
    expect(probs[0b10]).toBeCloseTo(0);
    expect(probs[0b11]).toBeCloseTo(0.5);
  });

  it('applies CZ gate', async () => {
    state = await QuantumState.create({ numQubits: 2 });
    state.x(0).x(1).cz(0, 1); // |11> -> -|11>
    const amps = state.getAmplitudes();
    expect(amps[0b11].real).toBeCloseTo(-1);
  });

  it('applies SWAP gate', async () => {
    state = await QuantumState.create({ numQubits: 2 });
    state.x(0).swap(0, 1); // |10> -> |01>
    const probs = state.getProbabilities();
    expect(probs[0b01]).toBeCloseTo(1);
  });

  it('validates control != target', async () => {
    state = await QuantumState.create({ numQubits: 2 });
    expect(() => state.cnot(0, 0)).toThrow();
  });
});

describe('Three-Qubit Gates', () => {
  let state: QuantumState;

  afterEach(() => {
    state?.dispose();
  });

  it('applies Toffoli gate', async () => {
    state = await QuantumState.create({ numQubits: 3 });
    state.x(0).x(1).toffoli(0, 1, 2); // |110> -> |111>
    const probs = state.getProbabilities();
    expect(probs[0b111]).toBeCloseTo(1);
  });

  it('Toffoli only flips when both controls are 1', async () => {
    state = await QuantumState.create({ numQubits: 3 });
    state.x(0).toffoli(0, 1, 2); // |100> -> |100> (q1 is 0)
    const probs = state.getProbabilities();
    expect(probs[0b100]).toBeCloseTo(1);
  });

  it('applies Fredkin gate', async () => {
    state = await QuantumState.create({ numQubits: 3 });
    state.x(0).x(1).fredkin(0, 1, 2); // |110> -> |101>
    const probs = state.getProbabilities();
    expect(probs[0b101]).toBeCloseTo(1);
  });
});

describe('State Operations', () => {
  let state: QuantumState;

  afterEach(() => {
    state?.dispose();
  });

  it('resets state to |0...0>', async () => {
    state = await QuantumState.create({ numQubits: 2 });
    state.h(0).h(1).reset();
    const probs = state.getProbabilities();
    expect(probs[0]).toBeCloseTo(1);
  });

  it('clones state', async () => {
    state = await QuantumState.create({ numQubits: 2 });
    state.h(0).cnot(0, 1);

    const cloned = await state.clone();
    const origProbs = state.getProbabilities();
    const clonedProbs = cloned.getProbabilities();

    for (let i = 0; i < 4; i++) {
      expect(clonedProbs[i]).toBeCloseTo(origProbs[i]);
    }

    cloned.dispose();
  });

  it('normalizes state', async () => {
    state = await QuantumState.create({ numQubits: 1 });
    // Set unnormalized amplitudes
    state.setAmplitudes([{ real: 3, imag: 0 }, { real: 4, imag: 0 }]);
    state.normalize();

    const probs = state.getProbabilities();
    const total = probs[0] + probs[1];
    expect(total).toBeCloseTo(1);
  });
});

describe('Measurement', () => {
  let state: QuantumState;

  afterEach(() => {
    state?.dispose();
  });

  it('measures definite state correctly', async () => {
    state = await QuantumState.create({ numQubits: 1 });
    state.x(0); // |1>
    const result = state.measure(0);
    expect(result).toBe(1);
  });

  it('measureAll returns basis state', async () => {
    state = await QuantumState.create({ numQubits: 3 });
    state.x(0).x(2); // |101>
    const result = state.measureAll();
    expect(result).toBe(0b101);
  });

  it('probabilityZero and probabilityOne are consistent', async () => {
    state = await QuantumState.create({ numQubits: 1 });
    state.h(0);
    const p0 = state.probabilityZero(0);
    const p1 = state.probabilityOne(0);
    expect(p0 + p1).toBeCloseTo(1);
    expect(p0).toBeCloseTo(0.5);
  });
});

describe('Expectation Values', () => {
  let state: QuantumState;

  afterEach(() => {
    state?.dispose();
  });

  it('expectation Z for |0> is 1', async () => {
    state = await QuantumState.create({ numQubits: 1 });
    expect(state.expectationZ(0)).toBeCloseTo(1);
  });

  it('expectation Z for |1> is -1', async () => {
    state = await QuantumState.create({ numQubits: 1 });
    state.x(0);
    expect(state.expectationZ(0)).toBeCloseTo(-1);
  });

  it('expectation Z for |+> is 0', async () => {
    state = await QuantumState.create({ numQubits: 1 });
    state.h(0);
    expect(state.expectationZ(0)).toBeCloseTo(0);
  });

  it('expectation X for |+> is 1', async () => {
    state = await QuantumState.create({ numQubits: 1 });
    state.h(0);
    expect(state.expectationX(0)).toBeCloseTo(1);
  });

  it('ZZ correlation for Bell state', async () => {
    state = await QuantumState.create({ numQubits: 2 });
    state.h(0).cnot(0, 1);
    // Bell state has perfect ZZ correlation
    expect(state.correlationZZ(0, 1)).toBeCloseTo(1);
  });
});

describe('State Properties', () => {
  let state: QuantumState;

  afterEach(() => {
    state?.dispose();
  });

  it('purity of pure state is 1', async () => {
    state = await QuantumState.create({ numQubits: 2 });
    state.h(0).cnot(0, 1);
    expect(state.purity()).toBeCloseTo(1);
  });

  it('calculates fidelity between identical states', async () => {
    state = await QuantumState.create({ numQubits: 2 });
    state.h(0).cnot(0, 1);

    const state2 = await QuantumState.create({ numQubits: 2 });
    state2.h(0).cnot(0, 1);

    expect(state.fidelity(state2)).toBeCloseTo(1);
    state2.dispose();
  });

  it('fidelity of orthogonal states is 0', async () => {
    state = await QuantumState.create({ numQubits: 1 });
    // |0>

    const state2 = await QuantumState.create({ numQubits: 1 });
    state2.x(0); // |1>

    expect(state.fidelity(state2)).toBeCloseTo(0);
    state2.dispose();
  });
});

describe('Disposal and Error Handling', () => {
  it('throws when using disposed state', async () => {
    const state = await QuantumState.create({ numQubits: 1 });
    state.dispose();

    expect(state.isDisposed).toBe(true);
    expect(() => state.h(0)).toThrow('disposed');
    expect(() => state.getProbabilities()).toThrow('disposed');
  });

  it('double dispose is safe', async () => {
    const state = await QuantumState.create({ numQubits: 1 });
    state.dispose();
    expect(() => state.dispose()).not.toThrow();
  });
});

describe('Fluent API', () => {
  let state: QuantumState;

  afterEach(() => {
    state?.dispose();
  });

  it('supports method chaining', async () => {
    state = await QuantumState.create({ numQubits: 3 });
    state.h(0).cnot(0, 1).cnot(1, 2).x(2).z(0);

    // Should not throw, just verify chain works
    expect(state.numQubits).toBe(3);
  });
});
