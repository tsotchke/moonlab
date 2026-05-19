/**
 * Topological invariants bindings (C-side since v0.2.x; JS binding
 * since v0.5.6).
 *
 * Wraps the topology surface in ``src/algorithms/quantum_geometry/qgt.h``
 * + the convenience entry points in
 * ``src/applications/moonlab_export_lean.c``.  Each function bundles
 * model construction, invariant calculation, and cleanup into one
 * call returning the integer invariant; no opaque struct pointer
 * crosses the FFI boundary.  Mirrors the Python ``moonlab.topology``
 * and Rust ``moonlab::topology`` surfaces.
 *
 * @example
 * ```typescript
 * import { qwzChern, sshWinding, kaneMeleZ2 } from '@moonlab/quantum-core';
 *
 * console.log(await qwzChern({ m: 1.0, n: 16 }));      // -1 (topological)
 * console.log(await sshWinding({ t1: 0.5, t2: 1.0 })); // 1 (topological)
 * console.log(await kaneMeleZ2({ lambdaSo: 0.1, lambdaV: 0.1 })); // 1
 * ```
 */

import { getModule } from './wasm-loader';

type TopologyModule = {
  _moonlab_qwz_chern: (m: number, N: number, out: number) => number;
  _moonlab_chern_qwz_proj: (m: number, N: number) => number;
  _moonlab_chern_qwz_pt: (m: number, N: number) => number;
  _moonlab_ssh_winding: (t1: number, t2: number, N: number) => number;
  _moonlab_kitaev_chain_z2: (t: number, mu: number, delta: number) => number;
  _moonlab_kane_mele_z2: (
    t: number, lambdaSo: number, lambdaR: number, lambdaV: number, N: number,
  ) => number;
  _moonlab_bhz_z2: (A: number, B: number, M: number, N: number) => number;
  _moonlab_hofstadter_chern: (
    t: number, p: number, q: number, nOccupied: number, N: number,
  ) => number;
};

// INT_MIN sentinel used by the C convenience wrappers to signal
// failure (bad args, alloc failure, integrator non-convergence).
const INT_MIN = -2147483648;

function check(value: number, label: string): number {
  if (value === INT_MIN) {
    throw new Error(`${label} failed (returned INT_MIN sentinel)`);
  }
  return value;
}

// ---- 2D Chern -----------------------------------------------------------

/** Qi-Wu-Zhang (2008) two-band model Chern number via the
 *  Fukui-Hatsugai-Suzuki link-variable integrator.  Topological with
 *  `|m| < 2` (returns `-1`), trivial outside (returns `0`).
 *
 *  `n` is the Brillouin-zone grid side; minimum 4 for numerical
 *  convergence. */
export async function qwzChern(opts: { m: number; n?: number }): Promise<number> {
  const mod = (await getModule()) as unknown as TopologyModule;
  const n = opts.n ?? 16;
  return check(mod._moonlab_qwz_chern(opts.m, n, 0), 'qwzChern');
}

/** QWZ Chern number via the gauge-free projector-trace integrator
 *  `F_xy(k) = -2 Im Tr[ P (d_x P) (d_y P) ]`.  Equivalent to
 *  {@link qwzChern} on every gapped phase point; retained for
 *  cross-validation. */
export async function chernQwzProj(opts: { m: number; n?: number }): Promise<number> {
  const mod = (await getModule()) as unknown as TopologyModule;
  const n = opts.n ?? 16;
  return check(mod._moonlab_chern_qwz_proj(opts.m, n), 'chernQwzProj');
}

/** QWZ Chern number via the parallel-transport-gauge eigenvector
 *  integrator.  Equivalent to {@link qwzChern} on gapped phases. */
export async function chernQwzParallelTransport(
  opts: { m: number; n?: number },
): Promise<number> {
  const mod = (await getModule()) as unknown as TopologyModule;
  const n = opts.n ?? 16;
  return check(mod._moonlab_chern_qwz_pt(opts.m, n), 'chernQwzParallelTransport');
}

// ---- 1D invariants ------------------------------------------------------

/** Integer winding number of the SSH model via the 1D Zak phase.
 *  Topological (winding = `+1`) when `|t2| > |t1|`. */
export async function sshWinding(
  opts: { t1: number; t2: number; n?: number },
): Promise<number> {
  const mod = (await getModule()) as unknown as TopologyModule;
  const n = opts.n ?? 64;
  return check(mod._moonlab_ssh_winding(opts.t1, opts.t2, n), 'sshWinding');
}

/** Z_2 invariant of the Kitaev p-wave chain (BdG Majorana).
 *  Topological (`1`) for `|mu| < 2|t|` and non-zero `delta`. */
export async function kitaevChainZ2(
  opts: { t?: number; mu: number; delta?: number },
): Promise<number> {
  const mod = (await getModule()) as unknown as TopologyModule;
  return check(
    mod._moonlab_kitaev_chain_z2(opts.t ?? 1.0, opts.mu, opts.delta ?? 1.0),
    'kitaevChainZ2',
  );
}

// ---- n-band Z_2 invariants ---------------------------------------------

/** Z_2 invariant of the Kane-Mele model on the honeycomb lattice.
 *  Returns `1` (quantum spin Hall) for
 *  `|lambdaV| < 3 sqrt(3) |lambdaSo|`, `0` otherwise.
 *
 *  `n` is the BZ grid side; must be even and `>= 8`. */
export async function kaneMeleZ2(
  opts: { t?: number; lambdaSo: number; lambdaR?: number; lambdaV: number; n?: number },
): Promise<number> {
  const mod = (await getModule()) as unknown as TopologyModule;
  const n = opts.n ?? 8;
  return check(
    mod._moonlab_kane_mele_z2(
      opts.t ?? 1.0, opts.lambdaSo, opts.lambdaR ?? 0.0, opts.lambdaV, n,
    ),
    'kaneMeleZ2',
  );
}

/** Z_2 invariant of the Bernevig-Hughes-Zhang 4-band model.
 *  Returns `1` (QSH) for `0 < M/B < 8` in this lattice
 *  regularisation, `0` otherwise. */
export async function bhzZ2(
  opts: { A: number; B: number; M: number; n?: number },
): Promise<number> {
  const mod = (await getModule()) as unknown as TopologyModule;
  const n = opts.n ?? 8;
  return check(mod._moonlab_bhz_z2(opts.A, opts.B, opts.M, n), 'bhzZ2');
}

/** Sub-band Chern number of the Harper-Hofstadter model at flux
 *  `phi = p / q`.  For `phi = 1/q` the lowest band has Chern `+1`
 *  (TKNN 1982).  `nOccupied` selects how many lowest sub-bands to
 *  integrate over; must be in `[1, q-1]`. */
export async function hofstadterChern(
  opts: { t?: number; p: number; q: number; nOccupied?: number; n?: number },
): Promise<number> {
  const mod = (await getModule()) as unknown as TopologyModule;
  const nOcc = opts.nOccupied ?? 1;
  const n = opts.n ?? 16;
  return check(
    mod._moonlab_hofstadter_chern(opts.t ?? 1.0, opts.p, opts.q, nOcc, n),
    'hofstadterChern',
  );
}
