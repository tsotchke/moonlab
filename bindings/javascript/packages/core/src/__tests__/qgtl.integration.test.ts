/**
 * JS integration tests for QGTL ingestion (since v0.6.8).
 *
 * The dist/moonlab.wasm in the repo today predates v0.6.6 (no
 * moonlab_qgtl_* symbols), so the suite auto-skips until a fresh
 * WASM build picks them up.
 */

import { describe, it, expect } from 'vitest';
import { QgtlCircuit, GateType, QgtlError } from '../qgtl';
import { getModule } from '../wasm-loader';

async function hasQgtlSymbols(): Promise<boolean> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const mod = (await getModule()) as any;
  return typeof mod._moonlab_qgtl_circuit_create === 'function';
}

describe('QgtlCircuit Bell pair', () => {
  it('produces P(|00>) = P(|11>) = 0.5', async () => {
    if (!(await hasQgtlSymbols())) return;
    const c = await QgtlCircuit.create(2);
    try {
      c.addGate(GateType.H, 0);
      c.addGate(GateType.CNOT, 1, 0);
      expect(c.numGates).toBe(2);
      const r = await c.execute({ returnProbabilities: true });
      expect(r.probabilities).not.toBeNull();
      expect(Math.abs(r.probabilities![0] - 0.5)).toBeLessThan(1e-9);
      expect(Math.abs(r.probabilities![3] - 0.5)).toBeLessThan(1e-9);
      expect(r.probabilities![1]).toBeLessThan(1e-9);
      expect(r.probabilities![2]).toBeLessThan(1e-9);
    } finally {
      c.dispose();
    }
  });
});

describe('QgtlCircuit shot sampling', () => {
  it('Bell sampling: only |00>/|11>, ~50/50', async () => {
    if (!(await hasQgtlSymbols())) return;
    const c = await QgtlCircuit.create(2);
    try {
      c.addGate(GateType.H, 0);
      c.addGate(GateType.CNOT, 1, 0);
      const r = await c.execute({ numShots: 1024, rngSeed: 0xdeadbeefn });
      expect(r.outcomes).not.toBeNull();
      expect(r.outcomes!.length).toBe(1024);
      let n00 = 0, n11 = 0, nother = 0;
      for (const o of r.outcomes!) {
        if (o === 0n) n00++;
        else if (o === 3n) n11++;
        else nother++;
      }
      expect(nother).toBe(0);
      expect(Math.abs(n00 - 512)).toBeLessThan(80);
    } finally {
      c.dispose();
    }
  });
});

describe('QgtlCircuit error paths', () => {
  it('rejects num_qubits = 0', async () => {
    if (!(await hasQgtlSymbols())) return;
    await expect(QgtlCircuit.create(0)).rejects.toBeInstanceOf(QgtlError);
  });

  it('rejects CNOT control == target', async () => {
    if (!(await hasQgtlSymbols())) return;
    const c = await QgtlCircuit.create(2);
    try {
      expect(() => c.addGate(GateType.CNOT, 0, 0)).toThrow(QgtlError);
    } finally {
      c.dispose();
    }
  });
});

describe('QgtlCircuit gate-type enum', () => {
  it('numeric values match QGTL gate_type_t', () => {
    expect(GateType.I).toBe(0);
    expect(GateType.H).toBe(4);
    expect(GateType.CNOT).toBe(10);
    expect(GateType.SWAP).toBe(13);
  });
});
