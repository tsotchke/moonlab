/**
 * Integration tests for the FusedCircuit gate-fusion DAG.
 *
 * Run `pnpm build:wasm` first.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { FusedCircuit } from '../fusion';
import { QuantumState } from '../quantum-state';

describe('FusedCircuit lifecycle', () => {
  let c: FusedCircuit | null = null;
  afterEach(() => { c?.dispose(); c = null; });

  it('creates an empty circuit on numQubits qubits', async () => {
    c = await FusedCircuit.create(4);
    expect(c.numQubits).toBe(4);
    expect(c.length).toBe(0);
  });

  it('rejects numQubits < 1', async () => {
    await expect(FusedCircuit.create(0)).rejects.toThrow(/numQubits/);
  });

  it('dispose is idempotent', async () => {
    c = await FusedCircuit.create(2);
    c.dispose();
    c.dispose();
  });
});

describe('FusedCircuit gate appenders', () => {
  it('fluent gate chain grows .length linearly', async () => {
    const c = await FusedCircuit.create(2);
    c.h(0).rz(0, 0.3).rx(0, 0.7).cnot(0, 1).rz(1, 0.4);
    expect(c.length).toBe(5);
    c.dispose();
  });

  it('rejects qubit out of range', async () => {
    const c = await FusedCircuit.create(2);
    expect(() => c.h(5)).toThrow(/qubit 5/);
    expect(() => c.cnot(0, 5)).toThrow(/qubits 0,5/);
    c.dispose();
  });

  it('supports u3 with three parameters', async () => {
    const c = await FusedCircuit.create(1);
    c.u3(0, 0.1, 0.2, 0.3);
    expect(c.length).toBe(1);
    c.dispose();
  });

  it('supports two-qubit parameterised gates', async () => {
    const c = await FusedCircuit.create(2);
    c.cphase(0, 1, 0.5).crx(0, 1, 0.3).cry(0, 1, 0.4).crz(0, 1, 0.5);
    expect(c.length).toBe(4);
    c.dispose();
  });
});

describe('FusedCircuit compile + execute', () => {
  it('fuses adjacent same-qubit single-qubit gates', async () => {
    // Three consecutive 1q gates on qubit 0 should fuse to one FUSED_1Q.
    const c = await FusedCircuit.create(2);
    c.h(0).rz(0, 0.3).rx(0, 0.7);
    const { fused, stats } = c.compile();
    expect(stats.originalGates).toBe(3);
    expect(stats.fusedGates).toBe(1);
    expect(stats.mergesApplied).toBe(2);  // run-length 3 -> 2 merges
    fused.dispose();
    c.dispose();
  });

  it('does not fuse across a multi-qubit barrier on the same qubit', async () => {
    // H(0); CNOT(0,1); RZ(0, theta) -- the CNOT touches qubit 0 so flushes.
    const c = await FusedCircuit.create(2);
    c.h(0).cnot(0, 1).rz(0, 0.3);
    const { fused, stats } = c.compile();
    expect(stats.mergesApplied).toBe(0);
    fused.dispose();
    c.dispose();
  });

  it('executes a Bell-pair fused circuit on QuantumState', async () => {
    const state = await QuantumState.create({ numQubits: 2 });
    const c = await FusedCircuit.create(2);
    c.h(0).cnot(0, 1);
    const { fused } = c.compile();
    fused.execute(state);
    const p = state.getProbabilities();
    // |Phi+> = (|00> + |11>) / sqrt 2 -> P(00) = P(11) = 0.5; others = 0.
    expect(p[0]).toBeCloseTo(0.5, 8);
    expect(p[1]).toBeCloseTo(0.0, 8);
    expect(p[2]).toBeCloseTo(0.0, 8);
    expect(p[3]).toBeCloseTo(0.5, 8);
    fused.dispose();
    c.dispose();
    state.dispose();
  });

  it('fused and unfused give identical state-vector probabilities', async () => {
    const stateA = await QuantumState.create({ numQubits: 3 });
    const stateB = await QuantumState.create({ numQubits: 3 });
    const buildC = async (state: QuantumState) => {
      const c = await FusedCircuit.create(3);
      c.h(0).rz(0, 0.31).rx(0, 0.52).cnot(0, 1)
        .h(1).ry(1, 0.4).cnot(1, 2)
        .rz(2, 0.23);
      return c;
    };
    const cA = await buildC(stateA);
    cA.execute(stateA);
    const cB = await buildC(stateB);
    const { fused } = cB.compile();
    fused.execute(stateB);
    const pA = stateA.getProbabilities();
    const pB = stateB.getProbabilities();
    for (let i = 0; i < 8; i++) {
      expect(pA[i]).toBeCloseTo(pB[i], 9);
    }
    fused.dispose();
    cA.dispose(); cB.dispose();
    stateA.dispose(); stateB.dispose();
  });
});
