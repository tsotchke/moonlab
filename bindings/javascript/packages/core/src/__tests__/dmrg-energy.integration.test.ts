/**
 * Integration tests for the DMRG scalar-energy convenience wrappers.
 *
 * Mirrors `bindings/python/tests/test_dmrg.py` and the Rust
 * `tests/dmrg_energy.rs` for cross-language parity on
 * `moonlab_dmrg_tfim_energy` / `moonlab_dmrg_heisenberg_energy`.
 */

import { describe, it, expect } from 'vitest';
import {
  dmrgTFIMGroundEnergy,
  dmrgHeisenbergGroundEnergy,
} from '../tensor-network';

describe('dmrgTFIMGroundEnergy', () => {
  it('returns a finite negative energy at the critical point', async () => {
    const e = await dmrgTFIMGroundEnergy(8, 1.0, 32, 10);
    expect(Number.isFinite(e)).toBe(true);
    expect(e).toBeLessThan(0);
  });

  it('larger transverse field yields a more negative energy', async () => {
    const eSmall = await dmrgTFIMGroundEnergy(8, 0.1, 32, 10);
    const eLarge = await dmrgTFIMGroundEnergy(8, 2.0, 32, 10);
    expect(Number.isFinite(eSmall)).toBe(true);
    expect(Number.isFinite(eLarge)).toBe(true);
    expect(eLarge).toBeLessThan(eSmall);
  });

  it.each([
    { numSites: 1, g: 1.0, chi: 16, sweeps: 5 },
    { numSites: 4, g: 1.0, chi: 0, sweeps: 5 },
    { numSites: 4, g: 1.0, chi: 16, sweeps: 0 },
  ])('returns DBL_MAX sentinel for invalid input %j', async (cfg) => {
    const e = await dmrgTFIMGroundEnergy(cfg.numSites, cfg.g, cfg.chi, cfg.sweeps);
    expect(e).toBeGreaterThan(1e300);
  });
});

describe('dmrgHeisenbergGroundEnergy', () => {
  it('isotropic Heisenberg returns a finite energy', async () => {
    const e = await dmrgHeisenbergGroundEnergy(8, 1.0, 1.0, 0.0, 32, 10);
    expect(Number.isFinite(e)).toBe(true);
  });

  it('XX-only chain (Delta=0) returns a finite energy', async () => {
    const e = await dmrgHeisenbergGroundEnergy(8, 1.0, 0.0, 0.0, 32, 10);
    expect(Number.isFinite(e)).toBe(true);
  });
});
