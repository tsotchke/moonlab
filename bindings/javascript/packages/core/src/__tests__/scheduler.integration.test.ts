/**
 * Distributed scheduler integration tests (since v0.7.1).
 *
 * Auto-skip when the loaded moonlab.wasm predates v0.7.0 -- the
 * existing dist/moonlab.wasm in the repo today lacks moonlab_job_*
 * symbols.  Once a fresh WASM build picks them up, the real
 * assertions activate.
 */

import { describe, it, expect } from 'vitest';
import { Job, SchedulerError } from '../scheduler';
import { GateType } from '../qgtl';
import { getModule } from '../wasm-loader';

async function hasSchedulerSymbols(): Promise<boolean> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const mod = (await getModule()) as any;
  return typeof mod._moonlab_job_create === 'function';
}

describe('Job execute', () => {
  it('Bell pair, single worker, 256 shots', async () => {
    if (!(await hasSchedulerSymbols())) return;
    const j = await Job.create(2);
    try {
      j.addGate(GateType.H, 0)
       .addGate(GateType.CNOT, 1, 0)
       .setNumShots(256)
       .setNumWorkers(1)
       .setRngSeed(0xdeadbeefn);
      const r = await j.execute();
      expect(r.totalShots).toBe(256);
      expect(r.numWorkersUsed).toBe(1);
      let nOther = 0;
      for (const o of r.outcomes) {
        if (o !== 0n && o !== 3n) nOther++;
      }
      expect(nOther).toBe(0);
    } finally {
      j.dispose();
    }
  });

  it('Bell pair, 4 workers, 1024 shots', async () => {
    if (!(await hasSchedulerSymbols())) return;
    const j = await Job.create(2);
    try {
      j.addGate(GateType.H, 0)
       .addGate(GateType.CNOT, 1, 0)
       .setNumShots(1024)
       .setNumWorkers(4)
       .setRngSeed(0xdeadbeefn);
      const r = await j.execute();
      expect(r.numWorkersUsed).toBe(4);
      expect(r.workerSeconds.length).toBe(4);
      let n00 = 0, n11 = 0, nOther = 0;
      for (const o of r.outcomes) {
        if (o === 0n) n00++;
        else if (o === 3n) n11++;
        else nOther++;
      }
      expect(nOther).toBe(0);
      expect(n00 + n11).toBe(1024);
      expect(Math.abs(n00 - 512)).toBeLessThan(80);
    } finally {
      j.dispose();
    }
  });
});

describe('Job JSON', () => {
  it('serialises with moonlab/job/v0.7.0 schema', async () => {
    if (!(await hasSchedulerSymbols())) return;
    const j = await Job.create(2);
    try {
      j.addGate(GateType.H, 0)
       .addGate(GateType.CNOT, 1, 0)
       .setNumShots(256)
       .setNumWorkers(2);
      const s = await j.toJson();
      expect(s).toContain('"schema": "moonlab/job/v0.7.0"');
      expect(s).toContain('"num_qubits": 2');
      expect(s).toContain('"num_shots": 256');
    } finally {
      j.dispose();
    }
  });
});

describe('Job error paths', () => {
  it('rejects num_qubits = 0', async () => {
    if (!(await hasSchedulerSymbols())) return;
    await expect(Job.create(0)).rejects.toBeInstanceOf(SchedulerError);
  });
});
