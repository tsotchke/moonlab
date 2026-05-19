/**
 * Integration tests for the libirrep QEC zoo binding (since v0.6.5).
 *
 * The WASM build links the bridge symbols (via the `src/integration/
 * libirrep_bridge.c` stub path) but does NOT link libirrep itself --
 * libirrep is a native-only dependency.  So `isAvailable()` is
 * expected to return false in the browser today, and every factory
 * raises `LibirrepNotBuiltError`.  These tests pin that contract so
 * a future "real WASM libirrep" build can flip the bit cleanly.
 */

import { describe, it, expect } from 'vitest';
import {
  LibirrepQecCode,
  LibirrepNotBuiltError,
} from '../libirrep-qec';
import { getModule } from '../wasm-loader';

/* The dist/moonlab.wasm in the repo today predates v0.6.5 and
 * doesn't ship the libirrep bridge symbols.  Auto-skip the suite
 * when the symbol isn't present so vitest stays green until the
 * next WASM rebuild picks the new exports up.  Once that ships
 * the skip path no-ops and the real assertions run. */
async function hasLibirrepSymbols(): Promise<boolean> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const mod = (await getModule()) as any;
  return typeof mod._moonlab_libirrep_available === 'function';
}

describe('LibirrepQecCode probe', () => {
  it('isAvailable() is false in the default WASM build', async () => {
    if (!(await hasLibirrepSymbols())) return;
    const available = await LibirrepQecCode.isAvailable();
    expect(available).toBe(false);
  });
});

describe('LibirrepQecCode factory error paths', () => {
  it('surface(3) raises LibirrepNotBuiltError', async () => {
    if (!(await hasLibirrepSymbols())) return;
    await expect(LibirrepQecCode.surface(3)).rejects.toBeInstanceOf(
      LibirrepNotBuiltError,
    );
  });

  it('toric(3, 3) raises LibirrepNotBuiltError', async () => {
    if (!(await hasLibirrepSymbols())) return;
    await expect(LibirrepQecCode.toric(3, 3)).rejects.toBeInstanceOf(
      LibirrepNotBuiltError,
    );
  });

  it('steane() raises LibirrepNotBuiltError', async () => {
    if (!(await hasLibirrepSymbols())) return;
    await expect(LibirrepQecCode.steane()).rejects.toBeInstanceOf(
      LibirrepNotBuiltError,
    );
  });

  it('bb72_12_6() raises LibirrepNotBuiltError', async () => {
    if (!(await hasLibirrepSymbols())) return;
    await expect(LibirrepQecCode.bb72_12_6()).rejects.toBeInstanceOf(
      LibirrepNotBuiltError,
    );
  });

  it('hgpRepetition(3) raises LibirrepNotBuiltError', async () => {
    if (!(await hasLibirrepSymbols())) return;
    await expect(LibirrepQecCode.hgpRepetition(3)).rejects.toBeInstanceOf(
      LibirrepNotBuiltError,
    );
  });

  it('error carries the rebuild-with hint', async () => {
    if (!(await hasLibirrepSymbols())) return;
    try {
      await LibirrepQecCode.steane();
      expect.fail('factory should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(LibirrepNotBuiltError);
      const msg = (e as Error).message;
      expect(msg).toContain('compiled without libirrep');
      expect(msg).toContain('QSIM_ENABLE_LIBIRREP');
    }
  });
});
