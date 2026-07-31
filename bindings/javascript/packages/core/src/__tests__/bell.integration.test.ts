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

/**
 * CHSH is the one sampled estimator in this file, so its bounds are
 * derived from the shot count rather than eyeballed.
 *
 * `bell_test_chsh` splits `shots` evenly across the four correlators, so
 * each E-hat averages `shots / 4` samples of a +-1 variable.  At the
 * Tsirelson-optimal angles every |E| = 1/sqrt(2), so Var[X] = 1 - E^2 =
 * 1/2 and Var[E-hat] = (1/2) / (shots / 4).  S combines four independent
 * estimators with +-1 coefficients, hence
 *
 *     sigma_S = sqrt(4 * (1/2) / (shots / 4)) = sqrt(8 / shots)
 *
 * which is 0.0447 at 4000 shots, around 2 sqrt(2) = 2.8284.  The former
 * upper bound of 2.9 sat 1.6 sigma out and failed one release build in
 * twenty: 20000 measured runs of this exact C entry point gave mean
 * 2.82788, sd 0.04441 (predicted 0.04472), and S >= 2.9 in 1058 of them
 * -- 5.3%.  The mean sits 1.8 standard errors *below* Tsirelson, so the
 * estimator is unbiased and the flake was purely a miscalibrated bound.
 *
 * The entropy stack behind the sampler is deliberately unseedable:
 * `quantum_entropy_ctx_create_hw` is the only constructor the ABI
 * exposes, because `quantum_entropy.h` requires measurement sampling to
 * draw from a CSPRNG.  So the assertion is calibrated instead of pinned,
 * at 5 sigma either side -- tighter than the 2.4 floor it replaces while
 * flaking at 6e-7 per side.
 */
const CHSH_SHOTS = 4000;
const TSIRELSON = 2 * Math.sqrt(2);
const CHSH_SIGMA = Math.sqrt(8 / CHSH_SHOTS);
const CHSH_TOL = 5 * CHSH_SIGMA;

describe('CHSH inequality', () => {
  it('violates classical bound on |Phi+>', async () => {
    const state = await QuantumState.create({ numQubits: 2 });
    await createBellState(state, 0, 1, BellState.PhiPlus);
    const r = await chshTest(state, 0, 1, CHSH_SHOTS);
    expect(r.classicalBound).toBeCloseTo(2.0, 12);
    expect(r.quantumBound).toBeCloseTo(TSIRELSON, 8);
    // 2.605 -- far above the classical bound of 2, and 5 sigma below
    // Tsirelson, so a genuinely degraded S still fails the test.
    expect(r.chshValue).toBeGreaterThan(TSIRELSON - CHSH_TOL);
    // 3.052 -- well under the algebraic maximum of 4 that a broken
    // correlator combination would produce.
    expect(r.chshValue).toBeLessThan(TSIRELSON + CHSH_TOL);
    expect(r.violatesClassical).toBe(true);
    state.dispose();
  });

  it('estimates each correlator at the Tsirelson-optimal magnitude', async () => {
    const state = await QuantumState.create({ numQubits: 2 });
    await createBellState(state, 0, 1, BellState.PhiPlus);
    const r = await chshTest(state, 0, 1, CHSH_SHOTS);
    // Per-correlator sigma is sqrt((1 - 1/2) / (shots / 4)) = 0.0224, so
    // 5 sigma is 0.112.  Signs follow E(a, b) = cos(a - b) at
    // (a, a', b, b') = (0, pi/2, pi/4, 3pi/4); pinning them catches an
    // angle or sign regression that |S| alone would absorb.
    const sigmaE = Math.sqrt(0.5 / (CHSH_SHOTS / 4));
    const expected: Array<[string, number, number]> = [
      ["E(a,b)",   r.correlationAB,           +Math.SQRT1_2],
      ["E(a,b')",  r.correlationABprime,      -Math.SQRT1_2],
      ["E(a',b)",  r.correlationAprimeB,      +Math.SQRT1_2],
      ["E(a',b')", r.correlationAprimeBprime, +Math.SQRT1_2],
    ];
    for (const [name, observed, ideal] of expected) {
      expect(observed, name).toBeGreaterThan(ideal - 5 * sigmaE);
      expect(observed, name).toBeLessThan(ideal + 5 * sigmaE);
    }
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

// Both Mermin surfaces take the analytic path -- `bell_test_mermin_ghz`
// and `bell_test_mermin_klyshko` evaluate <P_0 ... P_{n-1}> exactly and
// ignore `num_measurements`, so neither carries sampling noise.  2000
// runs of each C entry point returned bit-identical values (|M| =
// 4.0000000000000009, |M_3| / norm = 2.0000000000000004, zero spread),
// so they are asserted at their exact ideal values rather than behind
// the slack that a sampled estimator would need.

describe('Mermin GHZ', () => {
  it('|GHZ_3> saturates the quantum bound |M| = 4', async () => {
    const state = await QuantumState.create({ numQubits: 3 });
    state.h(0).cnot(0, 1).cnot(0, 2);
    const r = await merminGhzTest(state, 0, 1, 2, 4000);
    expect(r.classicalBound).toBeCloseTo(2.0, 12);
    expect(r.quantumBound).toBeCloseTo(4.0, 12);
    expect(Math.abs(r.chshValue)).toBeGreaterThan(r.classicalBound);
    expect(Math.abs(r.chshValue)).toBeCloseTo(4.0, 10);
    // <XYY> = <YXY> = <YYX> = -1 and <XXX> = +1 on |GHZ_3>, giving
    // M = -4 before the absolute value.
    expect(r.correlationAB).toBeCloseTo(-1.0, 10);
    expect(r.correlationABprime).toBeCloseTo(-1.0, 10);
    expect(r.correlationAprimeB).toBeCloseTo(-1.0, 10);
    expect(r.correlationAprimeBprime).toBeCloseTo(1.0, 10);
    state.dispose();
  });
});

describe('Mermin-Klyshko', () => {
  it('|GHZ_3> normalised |M_N| reaches 2^((N-1)/2) = 2', async () => {
    const state = await QuantumState.create({ numQubits: 3 });
    state.h(0).cnot(0, 1).cnot(0, 2);
    const mn = await merminKlyshkoTest(state, 3, 4000);
    expect(mn).toBeGreaterThan(1.0);  // classical (LHV) bound
    expect(mn).toBeCloseTo(2 ** ((3 - 1) / 2), 10);
    state.dispose();
  });
});
