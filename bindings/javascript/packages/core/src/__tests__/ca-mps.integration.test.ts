/**
 * Integration tests for the CaMps Clifford-assisted MPS binding.
 *
 * Currently exercises the Born-rule sequential sampler
 * `moonlab_ca_mps_sample_z` against analytic Bell-pair and GHZ_4
 * support sets.  Mirrors the Python and Rust tests in
 *   bindings/python/tests/test_ca_mps.py
 *   bindings/rust/moonlab/tests/ca_mps_sample_z_e2e.rs
 *
 * Run `pnpm build:wasm` first.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { CaMps, varDRun, Warmstart } from '../ca-mps';

/** Reproducible LCG matching the Rust test so distributions line up. */
class Lcg {
  private state: number;
  constructor(seed: number) { this.state = seed >>> 0; }
  next(): number {
    this.state = (Math.imul(this.state, 1_664_525) + 1_013_904_223) >>> 0;
    return (this.state >>> 8) / 16_777_216;
  }
  fill(n: number): Float64Array {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = this.next();
    return out;
  }
}

describe('CaMps.sampleZ Born-rule sequential sampler', () => {
  let s: CaMps | null = null;
  afterEach(() => { s?.dispose(); s = null; });

  it('Bell pair: support is exactly {00, 11} with rough 50/50 split', async () => {
    s = await CaMps.create(2, 16);
    s.h(0).cnot(0, 1);

    const shots = 4096;
    const rng = new Lcg(0xDEAD_BEEF);
    const randoms = rng.fill(shots * 2);
    const bits = s.sampleZ(shots, randoms);

    expect(bits.length).toBe(shots * 2);

    let n00 = 0;
    let n11 = 0;
    let other = 0;
    for (let i = 0; i < shots; i++) {
      const b0 = bits[i * 2];
      const b1 = bits[i * 2 + 1];
      if (b0 === 0 && b1 === 0) n00++;
      else if (b0 === 1 && b1 === 1) n11++;
      else other++;
    }
    expect(other).toBe(0);
    const p00 = n00 / shots;
    const p11 = n11 / shots;
    expect(Math.abs(p00 - 0.5)).toBeLessThan(0.08);
    expect(Math.abs(p11 - 0.5)).toBeLessThan(0.08);
  });

  it('GHZ_4: every shot is all-zeros or all-ones', async () => {
    const n = 4;
    s = await CaMps.create(n, 16);
    s.h(0);
    for (let k = 1; k < n; k++) s.cnot(k - 1, k);

    const shots = 4096;
    const rng = new Lcg(0xCAFE_BABE);
    const randoms = rng.fill(shots * n);
    const bits = s.sampleZ(shots, randoms);

    for (let i = 0; i < shots; i++) {
      const row = bits.slice(i * n, (i + 1) * n);
      const allZero = row.every(b => b === 0);
      const allOnes = row.every(b => b === 1);
      expect(allZero || allOnes).toBe(true);
    }
  });

  it('zero shots returns empty array without allocating in WASM', async () => {
    s = await CaMps.create(3, 8);
    const bits = s.sampleZ(0, new Float64Array(0));
    expect(bits.length).toBe(0);
  });

  it('rejects randomValues of the wrong length', async () => {
    s = await CaMps.create(3, 8);
    const tooShort = new Float64Array(5);   // need 4 * 3 = 12
    expect(() => s!.sampleZ(4, tooShort)).toThrow(/randomValues.length/);
  });
});

describe('varDRun (alternating Clifford + imag-time minimisation)', () => {
  let s: CaMps | null = null;
  afterEach(() => { s?.dispose(); s = null; });

  function tfimPauliSum(n: number, g: number): {
    paulis: Uint8Array; coeffs: Float64Array; numTerms: number;
  } {
    // H = -sum_i Z_i Z_{i+1} - g sum_i X_i.  Convention: Pauli byte
    // 1=X, 2=Y, 3=Z (matches the C-side enum used by var-D).
    const numTerms = (n - 1) + n;
    const paulis = new Uint8Array(numTerms * n);
    const coeffs = new Float64Array(numTerms);
    let row = 0;
    for (let i = 0; i < n - 1; i++) {
      paulis[row * n + i] = 3;
      paulis[row * n + i + 1] = 3;
      coeffs[row] = -1.0;
      row++;
    }
    for (let i = 0; i < n; i++) {
      paulis[row * n + i] = 1;
      coeffs[row] = -g;
      row++;
    }
    return { paulis, coeffs, numTerms };
  }

  it('TFIM critical point N=4 returns a finite energy below initial', async () => {
    s = await CaMps.create(4, 16);
    const { paulis, coeffs, numTerms } = tfimPauliSum(4, 1.0);
    const result = await varDRun(s, paulis, coeffs, numTerms, {
      maxOuterIters: 4,
      imagTimeDtau: 0.1,
      imagTimeStepsPerOuter: 3,
      cliffordPassesPerOuter: 4,
      warmstart: Warmstart.HAll,
    });
    expect(Number.isFinite(result.initialEnergy)).toBe(true);
    expect(Number.isFinite(result.finalEnergy)).toBe(true);
    expect(result.finalEnergy).toBeLessThanOrEqual(result.initialEnergy);
    expect(result.outerIterations).toBeGreaterThan(0);
  });

  it('rejects mismatched Pauli/coeff shapes', async () => {
    s = await CaMps.create(3, 8);
    const paulis = new Uint8Array(3);  // 1 term, n=3
    const coeffs = new Float64Array(2); // mismatch
    await expect(
      varDRun(s, paulis, coeffs, 1, { maxOuterIters: 1 }),
    ).rejects.toThrow(/does not match/);
  });
});
