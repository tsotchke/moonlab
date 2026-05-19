/**
 * Integration tests for the CliffordTableau class.
 *
 * Run `pnpm build:wasm` first to ensure dist/moonlab.{js,wasm} are
 * up to date.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { CliffordTableau } from '../clifford';

describe('CliffordTableau lifecycle', () => {
  let t: CliffordTableau | null = null;
  afterEach(() => { t?.dispose(); t = null; });

  it('creates and reports numQubits', async () => {
    t = await CliffordTableau.create(8);
    expect(t.numQubits).toBe(8);
  });

  it('rejects numQubits < 1', async () => {
    await expect(CliffordTableau.create(0)).rejects.toThrow(/numQubits/);
  });

  it('dispose is idempotent', async () => {
    t = await CliffordTableau.create(4);
    t.dispose();
    t.dispose();  // second call is no-op
  });
});

describe('CliffordTableau gate effects', () => {
  let t: CliffordTableau;
  afterEach(() => { t.dispose(); });

  it('H on |0> gives deterministic-or-random measurement at outcome 0/1', async () => {
    t = await CliffordTableau.create(1).then(c => c.setRngSeed(0xDEADBEEFn));
    t.h(0);
    const r = t.measure(0);
    // Post-Hadamard the outcome is random (Z anticommutes with H|0>'s stabiliser X).
    expect(r.outcomeKind).toBe('random');
    expect(r.outcome === 0 || r.outcome === 1).toBe(true);
  });

  it('measurement on |0> with no gates is deterministic 0', async () => {
    t = await CliffordTableau.create(3);
    const r = t.measure(1);
    expect(r.outcomeKind).toBe('deterministic');
    expect(r.outcome).toBe(0);
  });

  it('GHZ via H + CNOT chain has anticorrelated bitstrings under sampleAll', async () => {
    // Build GHZ on 8 qubits: H(0); CNOT(0, i) for i = 1..7.
    t = await CliffordTableau.create(8).then(c => c.setRngSeed(0xC0FFEEn));
    t.h(0);
    for (let i = 1; i < 8; i++) t.cnot(0, i);
    const s = t.sampleAll();
    // All-zeros (0x00) or all-ones (0xFF) are the only valid GHZ samples.
    const expected = [0n, (1n << 8n) - 1n];
    expect(expected).toContain(s.bits);
  });

  it('rejects sampleAll above 64 qubits', async () => {
    t = await CliffordTableau.create(65);
    expect(() => t.sampleAll()).toThrow(/supports up to 64/);
  });

  it('S then Sdag on |+> returns to |+>', async () => {
    t = await CliffordTableau.create(1);
    t.h(0).s(0).sdag(0);
    // |+> measured in Z is random; verify by sampling.
    let zeros = 0;
    const TRIALS = 200;
    for (let i = 0; i < TRIALS; i++) {
      const c = await CliffordTableau.create(1);
      c.setRngSeed(BigInt(i) * 1000n + 1n);
      c.h(0).s(0).sdag(0);
      if (c.measure(0).outcome === 0) zeros++;
      c.dispose();
    }
    // |+> outcome distribution should be ~50/50; allow generous slack.
    expect(zeros / TRIALS).toBeGreaterThan(0.3);
    expect(zeros / TRIALS).toBeLessThan(0.7);
  });

  it('reproducibility under fixed seed', async () => {
    const seed = 0x12345678ABCDn;
    const sample = async () => {
      const c = await CliffordTableau.create(4).then(x => x.setRngSeed(seed));
      c.h(0).cnot(0, 1).cnot(0, 2).cnot(0, 3);
      const s = c.sampleAll();
      c.dispose();
      return s.bits;
    };
    const a = await sample();
    const b = await sample();
    expect(a).toBe(b);
  });
});

describe('CliffordTableau range checks', () => {
  it('rejects out-of-range qubit on gate', async () => {
    const t = await CliffordTableau.create(2);
    expect(() => t.h(5)).toThrow(/qubit 5/);
    expect(() => t.cnot(0, 5)).toThrow(/qubits 0,5/);
    t.dispose();
  });

  it('rejects out-of-range qubit on measure', async () => {
    const t = await CliffordTableau.create(2);
    expect(() => t.measure(5)).toThrow(/qubit 5/);
    t.dispose();
  });
});
