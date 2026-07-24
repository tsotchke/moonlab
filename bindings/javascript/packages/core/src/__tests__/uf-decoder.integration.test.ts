/**
 * Integration tests for the union-find QEC decoder.
 *
 * Ports the numeric cases from tests/unit/test_uf_decoder.c (repetition
 * chain, boundary-hub regression, empty-syndrome + thread equivalence)
 * and adds a sampler -> decoder end-to-end check over the Pauli-frame
 * detector sampler.
 */

import { describe, it, expect } from 'vitest';
import { UfDecoder, UF_BOUNDARY } from '../uf-decoder';
import * as pf from '../pauli-frame';

describe('UfDecoder', () => {
  it('decodes the hand-checkable repetition chain exactly', async () => {
    // D0 -- boundary (flips the observable), D0 -- D1, D1 -- boundary.
    const d = await UfDecoder.create(2, 1, [
      { a: 0, b: UF_BOUNDARY, weight: 1.0, observables: 1 },
      { a: 0, b: 1, weight: 1.0, observables: 0 },
      { a: 1, b: UF_BOUNDARY, weight: 1.0, observables: 0 },
    ]);
    expect(d.numEdges).toBe(3);

    // Four shots, one per syndrome, detector-major -- same vectors as
    // tests/unit/test_uf_decoder.c::test_repetition_chain.
    const det = new Uint8Array([
      /* D0 over shots */ 0, 1, 1, 0,
      /* D1 over shots */ 0, 0, 1, 1,
    ]);
    const out = d.decodeBatch(det, 4);
    expect(Array.from(out)).toEqual([0, 1, 0, 0]);
    d.dispose();
  });

  it('keeps distinct boundary nodes (no boundary hub)', async () => {
    // Two independent defect pairs, each joined by a zero-observable
    // edge and each adjacent to the boundary through an observable-
    // flipping edge.  Correct answer: obs = 0.
    const d = await UfDecoder.create(4, 1, [
      { a: 0, b: 1, observables: 0 },
      { a: 2, b: 3, observables: 0 },
      { a: 0, b: UF_BOUNDARY, observables: 1 },
      { a: 1, b: UF_BOUNDARY, observables: 1 },
      { a: 2, b: UF_BOUNDARY, observables: 1 },
      { a: 3, b: UF_BOUNDARY, observables: 1 },
    ]);
    const det = new Uint8Array([1, 1, 1, 1]);   // all four detectors lit
    const out = d.decodeBatch(det, 1);
    expect(out[0]).toBe(0);
    d.dispose();
  });

  it('empty syndromes decode to no correction; thread count is inert', async () => {
    const d = await UfDecoder.create(2, 1, [
      { a: 0, b: UF_BOUNDARY, observables: 1 },
      { a: 0, b: 1, observables: 0 },
      { a: 1, b: UF_BOUNDARY, observables: 0 },
    ]);
    const SHOTS = 512;
    const det = new Uint8Array(2 * SHOTS);
    for (let s = 0; s < SHOTS; s++) {
      det[0 * SHOTS + s] = s & 1;
      det[1 * SHOTS + s] = (s >> 1) & 1;
    }
    const oneThread = d.decodeBatch(det, SHOTS, 1);
    const allCores = d.decodeBatch(det, SHOTS, 0);
    expect(Array.from(allCores)).toEqual(Array.from(oneThread));
    expect(oneThread[0]).toBe(0);   // shot 0 has an empty syndrome
    d.dispose();
  });

  it('rejects mis-sized detector batches', async () => {
    const d = await UfDecoder.create(2, 1, [
      { a: 0, b: UF_BOUNDARY, observables: 1 },
      { a: 0, b: 1 },
      { a: 1, b: UF_BOUNDARY },
    ]);
    expect(() => d.decodeBatch(new Uint8Array(7), 4)).toThrow(/length/);
    d.dispose();
  });

  it('decodes Pauli-frame detector samples end to end', async () => {
    // Three-qubit repetition readout with an X error only on qubit 0.
    // D0 = m0 xor m1 fires exactly when the error occurred; D1 never
    // fires.  The decoder must report the observable flip (= m0) shot
    // for shot.
    const p = 0.3;
    const SHOTS = 1500;
    const ops = [
      pf.xError(0, p),
      pf.measure(0), pf.measure(1), pf.measure(2),
    ];
    const { samples } = await pf.sampleCircuit(3, ops, SHOTS, { seed: 77 });
    const det = await pf.sampleDetectors(
      3, ops, [[0, 1], [1, 2]], SHOTS, { seed: 77 },
    );

    const d = await UfDecoder.create(2, 1, [
      { a: 0, b: UF_BOUNDARY, observables: 1 },  // X on q0: fires D0, flips m0
      { a: 0, b: 1, observables: 0 },            // X on q1: fires D0 + D1
      { a: 1, b: UF_BOUNDARY, observables: 0 },  // X on q2: fires D1
    ]);
    const obs = d.decodeBatch(det, SHOTS);
    d.dispose();

    let errors = 0;
    for (let s = 0; s < SHOTS; s++) {
      // Same seed -> the detector run reproduces the sample run, so the
      // decoded observable must equal the observed m0 flip exactly.
      expect(det[0 * SHOTS + s]).toBe(samples[0 * SHOTS + s]);
      expect(det[1 * SHOTS + s]).toBe(0);
      if (obs[s] !== samples[0 * SHOTS + s]) errors++;
    }
    expect(errors).toBe(0);
  });
});
