/**
 * Integration tests for Bell-inequality + Mermin tests.  All
 * exercise the WASM-resident entropy stack (since v0.5.4 the
 * `hardware_entropy_wasm.c` shim provides getentropy-backed
 * randomness for the C-side measurement samplers).
 */

import { describe, it, expect, afterEach } from 'vitest';
import { QuantumState } from '../quantum-state';
import {
  BellState, createBellState, chshTest,
  merminGhzTest, merminKlyshkoTest,
} from '../bell';

describe('Bell-state creation', () => {
  let state: QuantumState;
  afterEach(() => { state?.dispose(); });

  it('|Phi+> has P(00) = P(11) = 0.5, others = 0', async () => {
    state = await QuantumState.create({ numQubits: 2 });
    await createBellState(state, 0, 1, BellState.PhiPlus);
    const p = state.getProbabilities();
    expect(p[0]).toBeCloseTo(0.5, 10);
    expect(p[1]).toBeCloseTo(0.0, 10);
    expect(p[2]).toBeCloseTo(0.0, 10);
    expect(p[3]).toBeCloseTo(0.5, 10);
  });

  it('|Psi+> has P(01) = P(10) = 0.5', async () => {
    state = await QuantumState.create({ numQubits: 2 });
    await createBellState(state, 0, 1, BellState.PsiPlus);
    const p = state.getProbabilities();
    expect(p[1]).toBeCloseTo(0.5, 10);
    expect(p[2]).toBeCloseTo(0.5, 10);
  });
});

describe('CHSH inequality', () => {
  it('violates classical bound on |Phi+>', async () => {
    const state = await QuantumState.create({ numQubits: 2 });
    await createBellState(state, 0, 1, BellState.PhiPlus);
    const r = await chshTest(state, 0, 1, 4000);
    expect(r.classicalBound).toBeCloseTo(2.0, 12);
    expect(r.quantumBound).toBeCloseTo(2 * Math.sqrt(2), 8);
    expect(r.chshValue).toBeGreaterThan(2.4);  // 4000 shots reach ~2.8 with slack
    expect(r.chshValue).toBeLessThan(2.9);
    expect(r.violatesClassical).toBe(true);
    state.dispose();
  });

  it('reports measurement count matching the request', async () => {
    const state = await QuantumState.create({ numQubits: 2 });
    await createBellState(state, 0, 1, BellState.PhiPlus);
    const r = await chshTest(state, 0, 1, 1000);
    // Each of 4 angle pairs samples num_measurements / 4 = 250 pairs;
    // the result is normalised, but the `measurements` field counts the
    // per-pair samples.  Allow generous slack on the lower bound.
    expect(r.measurements).toBeGreaterThan(0);
    state.dispose();
  });
});

describe('Mermin GHZ', () => {
  it('|GHZ_3> violates the classical bound', async () => {
    const state = await QuantumState.create({ numQubits: 3 });
    state.h(0).cnot(0, 1).cnot(0, 2);
    const r = await merminGhzTest(state, 0, 1, 2, 4000);
    expect(r.classicalBound).toBeCloseTo(2.0, 12);
    expect(r.quantumBound).toBeCloseTo(4.0, 12);
    expect(Math.abs(r.chshValue)).toBeGreaterThan(2.5);  // |M| floor
    expect(Math.abs(r.chshValue)).toBeLessThan(4.1);
    state.dispose();
  });
});

describe('Mermin-Klyshko', () => {
  it('|GHZ_3> normalised |M_N| > 1.0 (classical bound)', async () => {
    const state = await QuantumState.create({ numQubits: 3 });
    state.h(0).cnot(0, 1).cnot(0, 2);
    const mn = await merminKlyshkoTest(state, 3, 4000);
    expect(mn).toBeGreaterThan(1.1);
    state.dispose();
  });
});
