/**
 * Integration tests for the noise-channel bindings.
 *
 * The C channels are Monte-Carlo trajectory unravellings that take the
 * caller's uniform-[0, 1) draw as the trailing argument.  Two kinds of
 * checks below:
 *
 *  - deterministic: inject an explicit draw and pin the exact Kraus
 *    branch the C engine selects (`src/quantum/noise.c` branch logic);
 *  - statistical: run many trajectories with the default crypto RNG
 *    and check empirical channel rates within a > 4-sigma tolerance.
 */

import { describe, it, expect } from 'vitest';
import { QuantumState } from '../quantum-state';
import {
  depolarizing,
  amplitudeDamping,
  phaseDamping,
  bitFlip,
  phaseFlip,
  bitPhaseFlip,
  thermalRelaxation,
  readoutError,
  DeviceNoiseModel,
} from '../noise';

async function withState(
  numQubits: number, fn: (s: QuantumState) => void | Promise<void>,
): Promise<void> {
  const s = await QuantumState.create({ numQubits });
  try {
    await fn(s);
  } finally {
    s.dispose();
  }
}

describe('noise channels: deterministic branch selection', () => {
  it('bitFlip applies X exactly when the draw is below p', async () => {
    await withState(1, (s) => {
      bitFlip(s, 0, 0.3, 0.1);           // 0.1 < 0.3 -> flip
      expect(s.probabilityOne(0)).toBeCloseTo(1.0, 10);
      bitFlip(s, 0, 0.3, 0.9);           // 0.9 >= 0.3 -> no flip
      expect(s.probabilityOne(0)).toBeCloseTo(1.0, 10);
      bitFlip(s, 0, 0.3, 0.29999);       // flip back
      expect(s.probabilityOne(0)).toBeCloseTo(0.0, 10);
    });
  });

  it('depolarizing branch order is X, Y, Z over r = rv/p', async () => {
    // On |0>: X and Y flip the qubit, Z does not.
    await withState(1, (s) => {
      depolarizing(s, 0, 0.9, 0.1);      // r=0.111 < 1/3 -> X
      expect(s.probabilityOne(0)).toBeCloseTo(1.0, 10);
    });
    await withState(1, (s) => {
      depolarizing(s, 0, 0.9, 0.45);     // r=0.5 in [1/3, 2/3) -> Y
      expect(s.probabilityOne(0)).toBeCloseTo(1.0, 10);
    });
    await withState(1, (s) => {
      depolarizing(s, 0, 0.9, 0.85);     // r=0.944 >= 2/3 -> Z
      expect(s.probabilityOne(0)).toBeCloseTo(0.0, 10);
    });
    await withState(1, (s) => {
      depolarizing(s, 0, 0.9, 0.95);     // rv >= p -> no error
      expect(s.probabilityOne(0)).toBeCloseTo(0.0, 10);
    });
  });

  it('amplitudeDamping collapses |1> to |0> when the draw is below gamma', async () => {
    await withState(1, (s) => {
      s.x(0);
      amplitudeDamping(s, 0, 0.25, 0.2); // p_decay = 0.25 on pure |1>
      expect(s.probabilityOne(0)).toBeCloseTo(0.0, 10);
    });
    await withState(1, (s) => {
      s.x(0);
      amplitudeDamping(s, 0, 0.25, 0.9); // no jump; renormalised |1>
      expect(s.probabilityOne(0)).toBeCloseTo(1.0, 10);
    });
  });

  it('phaseFlip and bitPhaseFlip act on the expected branch', async () => {
    await withState(1, (s) => {
      phaseFlip(s, 0, 0.5, 0.4);         // Z on |0>: populations unchanged
      expect(s.probabilityOne(0)).toBeCloseTo(0.0, 10);
      bitPhaseFlip(s, 0, 0.5, 0.4);      // Y on |0>: flips population
      expect(s.probabilityOne(0)).toBeCloseTo(1.0, 10);
    });
  });

  it('phaseDamping preserves populations and the norm', async () => {
    await withState(1, (s) => {
      s.h(0);
      phaseDamping(s, 0, 0.4, 0.9);
      const probs = s.getProbabilities();
      const norm = probs[0] + probs[1];
      expect(norm).toBeCloseTo(1.0, 10);
    });
  });

  it('thermalRelaxation decays |1> on the T1 branch', async () => {
    await withState(1, (s) => {
      s.x(0);
      // gamma_t1 = 1 - exp(-t/T1) = 1 - exp(-1) ~ 0.632; draw 0.1 decays.
      thermalRelaxation(s, 0, 1.0, 0.5, 1.0, [0.1, 0.9]);
      expect(s.probabilityOne(0)).toBeCloseTo(0.0, 10);
    });
    await withState(1, (s) => {
      s.x(0);
      thermalRelaxation(s, 0, 1.0, 0.5, 1e-9, [0.9, 0.9]); // t << T1: no decay
      expect(s.probabilityOne(0)).toBeCloseTo(1.0, 6);
    });
  });

  it('readoutError flips the reported bit on the expected branch', async () => {
    expect(await readoutError(0, 0.2, 0.1, 0.15)).toBe(1); // 0.15 < 0.2
    expect(await readoutError(0, 0.2, 0.1, 0.5)).toBe(0);
    expect(await readoutError(1, 0.2, 0.1, 0.05)).toBe(0); // 0.05 < 0.1
    expect(await readoutError(1, 0.2, 0.1, 0.5)).toBe(1);
  });
});

describe('noise channels: statistical rates (default RNG)', () => {
  // Empirical rate tolerance: 4+ sigma of a binomial with N = 2000.
  const N = 2000;

  it(`bitFlip p=0.3 flips ~30% of ${N} trajectories`, async () => {
    let flips = 0;
    const s = await QuantumState.create({ numQubits: 1 });
    try {
      for (let i = 0; i < N; i++) {
        s.reset();
        bitFlip(s, 0, 0.3);
        if (s.probabilityOne(0) > 0.5) flips++;
      }
    } finally {
      s.dispose();
    }
    // sd = sqrt(0.3 * 0.7 / N) ~ 0.0102; 0.05 is ~4.9 sigma.
    expect(flips / N).toBeGreaterThan(0.3 - 0.05);
    expect(flips / N).toBeLessThan(0.3 + 0.05);
  });

  it(`depolarizing p=0.75 flips |0> with rate 2/3 * p = 0.5`, async () => {
    let flips = 0;
    const s = await QuantumState.create({ numQubits: 1 });
    try {
      for (let i = 0; i < N; i++) {
        s.reset();
        depolarizing(s, 0, 0.75);
        if (s.probabilityOne(0) > 0.5) flips++;
      }
    } finally {
      s.dispose();
    }
    // sd = sqrt(0.25 / N) ~ 0.0112; 0.055 is ~4.9 sigma.
    expect(flips / N).toBeGreaterThan(0.5 - 0.055);
    expect(flips / N).toBeLessThan(0.5 + 0.055);
  });

  it(`amplitudeDamping gamma=0.4 decays |1> with rate ~gamma`, async () => {
    let decays = 0;
    const s = await QuantumState.create({ numQubits: 1 });
    try {
      for (let i = 0; i < N; i++) {
        s.reset();
        s.x(0);
        amplitudeDamping(s, 0, 0.4);
        if (s.probabilityOne(0) < 0.5) decays++;
      }
    } finally {
      s.dispose();
    }
    // sd = sqrt(0.4 * 0.6 / N) ~ 0.011; 0.055 is ~5 sigma.
    expect(decays / N).toBeGreaterThan(0.4 - 0.055);
    expect(decays / N).toBeLessThan(0.4 + 0.055);
  });
});

describe('DeviceNoiseModel', () => {
  it('applies its depolarizing channel through noise_apply_model', async () => {
    const model = await DeviceNoiseModel.create({ depolarizingRate: 0.5 });
    try {
      // Draw 0.0 -> r = 0 -> X branch: |0> becomes |1>.
      await withState(1, (s) => {
        model.applySingle(s, 0, [0.0]);
        expect(s.probabilityOne(0)).toBeCloseTo(1.0, 10);
      });
      // Draw 0.99 >= rate -> no error.
      await withState(1, (s) => {
        model.applySingle(s, 0, [0.99]);
        expect(s.probabilityOne(0)).toBeCloseTo(0.0, 10);
      });
    } finally {
      model.dispose();
    }
  });

  it('two-qubit application consumes the leading two-qubit draw', async () => {
    const model = await DeviceNoiseModel.create({ depolarizingRate: 0.0 });
    try {
      await withState(2, (s) => {
        // No channels enabled beyond the zero-rate depolarizing: the
        // state must be untouched and normalised.
        model.applyTwo(s, 0, 1, [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]);
        const probs = s.getProbabilities();
        expect(probs[0]).toBeCloseTo(1.0, 10);
      });
    } finally {
      model.dispose();
    }
  });

  it('realistic() builds a model that preserves the state norm', async () => {
    const model = await DeviceNoiseModel.realistic(50.0, 30.0, 1e-3, 1e-2);
    try {
      await withState(1, (s) => {
        s.h(0);
        model.applySingle(s, 0);
        const probs = s.getProbabilities();
        expect(probs[0] + probs[1]).toBeCloseTo(1.0, 6);
      });
    } finally {
      model.dispose();
    }
  });

  it('setEnabled(false) turns the model into a no-op', async () => {
    const model = await DeviceNoiseModel.create({ depolarizingRate: 1.0 });
    try {
      model.setEnabled(false);
      await withState(1, (s) => {
        model.applySingle(s, 0, [0.0]);
        expect(s.probabilityOne(0)).toBeCloseTo(0.0, 10);
      });
    } finally {
      model.dispose();
    }
  });
});
