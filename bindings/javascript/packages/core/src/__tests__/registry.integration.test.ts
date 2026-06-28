/**
 * v1.0.3 plug-in surface integration tests for JS bindings:
 *
 *   - decoder runtime registry (register / lookup / list /
 *     decode_by_name)
 *   - vendor-noise profile registry
 *   - scheduler completion hook
 *
 * Auto-skip when the loaded moonlab.wasm predates v1.0.3 (the
 * symbols + addFunction runtime aren't exported yet).  Once a
 * fresh WASM build picks them up, the real assertions activate.
 */

import { describe, it, expect } from 'vitest';
import {
  decoderRegistryAvailable,
  listDecoders,
  lookupDecoder,
  registerDecoder,
  unregisterDecoder,
  decodeByName,
  type CodeGeometry,
} from '../decoder';
import {
  vendorNoiseProfileRegistryAvailable,
  registerVendorNoiseProfile,
  lookupVendorNoiseProfile,
  unregisterVendorNoiseProfile,
  listVendorNoiseProfiles,
  setCompletionHook,
  clearCompletionHook,
  Job,
} from '../scheduler';
import { GateType } from '../qgtl';
import { getModule } from '../wasm-loader';

async function hasAddFunction(): Promise<boolean> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const mod = (await getModule()) as any;
  return typeof mod.addFunction === 'function';
}

describe('decoder runtime registry', () => {
  it('lists the five built-in decoders when available', async () => {
    if (!(await decoderRegistryAvailable())) return;
    const names = await listDecoders();
    for (const required of [
      'greedy', 'mwpm_exact', 'sbnn',
      'libirrep_single_shot', 'pymatching',
    ]) {
      expect(names).toContain(required);
    }
  });

  it('lookup returns null for unknown names', async () => {
    if (!(await decoderRegistryAvailable())) return;
    expect(await lookupDecoder('definitely-not-real')).toBeNull();
  });

  it('register + dispatch + unregister round-trips', async () => {
    if (!(await decoderRegistryAvailable())) return;
    if (!(await hasAddFunction())) return;
    const sentinel = 0xA5;
    await registerDecoder(
      'js-sentinel-test',
      (code, _syndromes, _seed) => {
        const out = new Uint8Array(code.numQubits);
        out[0] = sentinel;
        return out;
      },
      'round-trip sanity decoder',
    );
    try {
      expect(await listDecoders()).toContain('js-sentinel-test');
      const entry = await lookupDecoder('js-sentinel-test');
      expect(entry?.description).toContain('round-trip');

      const d = 3;
      const code: CodeGeometry = {
        distance: d, numQubits: 2 * d * d, isToric: true,
      };
      const corr = await decodeByName(
        'js-sentinel-test', code, new Uint8Array(d * d),
      );
      expect(corr[0]).toBe(sentinel);
      for (let i = 1; i < corr.length; i++) expect(corr[i]).toBe(0);
    } finally {
      await unregisterDecoder('js-sentinel-test');
    }
    expect(await listDecoders()).not.toContain('js-sentinel-test');
  });
});

describe('vendor-noise profile registry', () => {
  it('contains the six baked-in profiles when available', async () => {
    if (!(await vendorNoiseProfileRegistryAvailable())) return;
    const names = await listVendorNoiseProfiles();
    for (const required of [
      'ibm-falcon-emu', 'rigetti-aspen-emu', 'ionq-forte-emu',
      'ibm-falcon', 'rigetti-aspen', 'ionq-forte',
    ]) {
      expect(names).toContain(required);
    }
  });

  it('register + lookup + unregister round-trips', async () => {
    if (!(await vendorNoiseProfileRegistryAvailable())) return;
    await registerVendorNoiseProfile('js-test-profile', {
      pGate1q: 0.0015, pGate2q: 0.012, pReadout: 0.018,
      description: 'JS-test snapshot',
    });
    try {
      const back = await lookupVendorNoiseProfile('js-test-profile');
      expect(back).not.toBeNull();
      expect(back!.pGate2q).toBeCloseTo(0.012, 9);
      expect(back!.description).toContain('snapshot');
    } finally {
      await unregisterVendorNoiseProfile('js-test-profile');
    }
    expect(await lookupVendorNoiseProfile('js-test-profile')).toBeNull();
  });
});

describe('scheduler completion hook', () => {
  it('fires after a successful run with the right args', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const mod = (await getModule()) as any;
    if (typeof mod._moonlab_scheduler_set_completion_hook !== 'function') return;
    if (!(await hasAddFunction())) return;

    let count = 0;
    let lastQubits = -1;
    let lastShots = -1;
    let lastBackend: string | null = '';
    await setCompletionHook((info) => {
      count++;
      lastQubits = info.numQubits;
      lastShots = info.totalShots;
      lastBackend = info.backendName;
    });
    try {
      const j = await Job.create(2);
      try {
        j.addGate(GateType.H, 0).addGate(GateType.CNOT, 1, 0)
          .setNumShots(128).setNumWorkers(1).setRngSeed(0xdeadbeefn);
        await j.execute();
      } finally {
        j.dispose();
      }
      expect(count).toBe(1);
      expect(lastQubits).toBe(2);
      expect(lastShots).toBe(128);
      // Default backend = "simulator".
      expect(lastBackend).toBe('simulator');
    } finally {
      await clearCompletionHook();
    }

    // After clear, the hook must NOT fire again.
    const j2 = await Job.create(1);
    try {
      j2.addGate(GateType.H, 0).setNumShots(32).setNumWorkers(1);
      await j2.execute();
    } finally {
      j2.dispose();
    }
    expect(count).toBe(1);
  });
});
