/**
 * Hydrogenic orbitals — the Schrödinger solutions.
 *
 * MoonLab's C library carries quantum chemistry (Jordan-Wigner, molecular
 * Hamiltonians, UCCSD) but not the closed-form hydrogenic wavefunctions, so
 * these are computed here, the same way the project's own web demo does it.
 * What MoonLab then does with the result is the interesting part: the density
 * is loaded into a state vector as amplitudes and the probabilities are read
 * back, so the simulator round-trips a distribution whose exact answer we
 * already know. The L1 drift between the two is a correctness measurement,
 * not decoration.
 */

/** Associated Laguerre L^alpha_k(x), by the standard upward recurrence. */
export function assocLaguerre(k: number, alpha: number, x: number): number {
  if (k === 0) return 1;
  if (k === 1) return 1 + alpha - x;
  let previous = 1;
  let current = 1 + alpha - x;
  for (let i = 2; i <= k; i++) {
    const next = ((2 * i - 1 + alpha - x) * current - (i - 1 + alpha) * previous) / i;
    previous = current;
    current = next;
  }
  return current;
}

function factorial(n: number): number {
  let result = 1;
  for (let i = 2; i <= n; i++) result *= i;
  return result;
}

/** Associated Legendre P^m_l(x) for m >= 0, by recurrence. */
export function assocLegendre(l: number, m: number, x: number): number {
  let pmm = 1;
  if (m > 0) {
    const somx2 = Math.sqrt(Math.max(0, 1 - x * x));
    let fact = 1;
    for (let i = 1; i <= m; i++) {
      pmm *= -fact * somx2;
      fact += 2;
    }
  }
  if (l === m) return pmm;
  let pmmp1 = x * (2 * m + 1) * pmm;
  if (l === m + 1) return pmmp1;
  let pll = 0;
  for (let ll = m + 2; ll <= l; ll++) {
    pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
    pmm = pmmp1;
    pmmp1 = pll;
  }
  return pll;
}

/** Real spherical harmonic Y_lm(theta, phi). */
export function realSphericalHarmonic(
  l: number,
  m: number,
  theta: number,
  phi: number,
): number {
  const absM = Math.abs(m);
  const norm = Math.sqrt(
    ((2 * l + 1) / (4 * Math.PI)) * (factorial(l - absM) / factorial(l + absM)),
  );
  const legendre = assocLegendre(l, absM, Math.cos(theta));
  if (m === 0) return norm * legendre;
  const azimuth = m > 0 ? Math.cos(absM * phi) : Math.sin(absM * phi);
  return Math.SQRT2 * norm * legendre * azimuth;
}

/** Radial part R_nl(r), in Bohr radii, for nuclear charge Z. */
export function radialWavefunction(n: number, l: number, z: number, r: number): number {
  const rho = (2 * z * r) / n;
  const prefactor = Math.sqrt(
    ((2 * z) / n) ** 3 * (factorial(n - l - 1) / (2 * n * factorial(n + l))),
  );
  return prefactor * Math.exp(-rho / 2) * rho ** l * assocLaguerre(n - l - 1, 2 * l + 1, rho);
}

export interface Orbital {
  readonly n: number;
  readonly l: number;
  readonly m: number;
  readonly z: number;
}

/** The conventional label: 1s, 2p, 3d… */
export function orbitalLabel(o: Orbital): string {
  const shells = "spdfgh";
  return `${o.n}${shells[o.l] ?? "?"}${o.m === 0 ? "" : ` (m=${o.m > 0 ? "+" : ""}${o.m})`}`;
}

export function orbitalIsValid(o: Orbital): boolean {
  return o.n >= 1 && o.l >= 0 && o.l < o.n && Math.abs(o.m) <= o.l;
}

export interface DensitySlice {
  readonly size: number;
  /** |psi|^2 on a `size * size` grid, row-major, already normalised to sum 1. */
  readonly density: Float64Array;
  /** Half-width of the slice in Bohr radii. */
  readonly extent: number;
  readonly max: number;
}

/**
 * Samples |psi|^2 on the x-z plane through the nucleus.
 *
 * That plane is chosen because it is the one where the m-dependence of a real
 * orbital is visible: p_z has its two lobes along z, d_{z^2} its torus, and a
 * slice through x-y would show several of them as featureless rings.
 */
export function densitySlice(orbital: Orbital, size: number, extent: number): DensitySlice {
  const density = new Float64Array(size * size);
  let max = 0;
  let total = 0;
  for (let row = 0; row < size; row++) {
    // Row 0 is +z at the top, so the picture matches the usual orientation.
    const zc = extent * (1 - (2 * row) / (size - 1));
    for (let column = 0; column < size; column++) {
      const xc = extent * ((2 * column) / (size - 1) - 1);
      const r = Math.hypot(xc, zc);
      const theta = r === 0 ? 0 : Math.acos(zc / r);
      const phi = xc >= 0 ? 0 : Math.PI;
      const psi = radialWavefunction(orbital.n, orbital.l, orbital.z, r) *
        realSphericalHarmonic(orbital.l, orbital.m, theta, phi);
      const value = psi * psi;
      density[row * size + column] = value;
      total += value;
      if (value > max) max = value;
    }
  }
  if (total > 0) {
    for (let i = 0; i < density.length; i++) density[i] /= total;
    max /= total;
  }
  return { size, density, extent, max };
}

/** A sensible viewing window: orbitals spread roughly as n^2 Bohr radii. */
export function suggestedExtent(orbital: Orbital): number {
  return (2.5 * orbital.n * orbital.n) / orbital.z;
}

// ---------------------------------------------------------------------------
// Multi-electron corrections
//
// Ported from the project's web demo (bindings/javascript/demo, the Schrödinger
// page) so the terminal console shows the same physics under the same names.
// They are approximations, deliberately: a Slater-style screening rule, a
// scalar relativistic contraction, an angular spin-orbit weighting, and a small
// configuration-mixing sum. None is a solved Hartree-Fock orbital, and the UI
// says which are active rather than implying an exact result.
// ---------------------------------------------------------------------------

export const FINE_STRUCTURE_ALPHA = 1 / 137.035999084;

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/** 2n^2. */
export function shellCapacity(n: number): number {
  return 2 * n * n;
}

function electronsBeforeShell(n: number): number {
  let total = 0;
  for (let i = 1; i < n; i++) total += shellCapacity(i);
  return total;
}

export function electronsInShell(z: number, n: number): number {
  if (n < 1) return 0;
  return clamp(z - electronsBeforeShell(n), 0, shellCapacity(n));
}

/** Slater-style effective nuclear charge with a small exchange term. */
export function effectiveNuclearCharge(z: number, n: number, l: number, enabled: boolean): number {
  if (!enabled) return z;
  const inShellN = electronsInShell(z, n);
  const inShellNMinus1 = n > 1 ? electronsInShell(z, n - 1) : 0;
  const sameShell = Math.max(0, inShellN - 1);
  const lowerShell = Math.max(0, z - inShellN - inShellNMinus1);
  const nMinus1Coeff = l <= 1 ? 0.85 : 1.0;
  const exchangeTerm = l > 0 ? 0.02 * ((2 * l + 1) / Math.max(1, n)) : 0;
  const shielding = 0.35 * sameShell + nMinus1Coeff * inShellNMinus1 + lowerShell;
  return clamp(z - shielding + exchangeTerm, 1, z);
}

/** Scalar relativistic contraction of the effective charge. */
export function applyRelativisticContraction(
  zEff: number,
  n: number,
  l: number,
  enabled: boolean,
): number {
  if (!enabled) return zEff;
  const beta = (FINE_STRUCTURE_ALPHA * zEff) ** 2;
  return zEff * (1 + (beta * 0.32) / (Math.max(1, n) * (l + 1)));
}

/** Angular spin-orbit weighting; identity for l = 0 or m = 0. */
export function spinOrbitDensityFactor(
  theta: number,
  n: number,
  l: number,
  m: number,
  zEff: number,
  enabled: boolean,
): number {
  if (!enabled || l === 0 || m === 0) return 1;
  const beta = (FINE_STRUCTURE_ALPHA * zEff) ** 2;
  const coupling = (beta * 0.28 * l) / Math.max(1, n * n);
  return clamp(1 + coupling * Math.cos(theta) * (m / Math.max(1, l)), 0.35, 1.65);
}

export interface CorrelationTerm {
  readonly n: number;
  readonly l: number;
  readonly m: number;
  readonly weight: number;
}

/** Configuration-mixing terms: neighbouring l, and the next shell up. */
export function buildCorrelationTerms(
  n: number,
  l: number,
  m: number,
  maxN = 6,
): CorrelationTerm[] {
  const terms: CorrelationTerm[] = [];
  const add = (tn: number, tl: number, weight: number) => {
    if (tn < 1 || tn > maxN || tl < 0 || tl >= tn || weight <= 0) return;
    terms.push({ n: tn, l: tl, m: clamp(m, -tl, tl), weight });
  };
  if (l - 1 >= 0) add(n, l - 1, 0.28);
  if (l + 1 < n) add(n, l + 1, 0.28);
  add(n + 1, l, 0.14);
  return terms;
}

/** Which corrections are switched on. */
export interface OrbitalPhysics {
  readonly screeningExchange: boolean;
  readonly relativisticSpinOrbit: boolean;
  readonly correlationMixing: boolean;
}

export const NO_CORRECTIONS: OrbitalPhysics = {
  screeningExchange: false,
  relativisticSpinOrbit: false,
  correlationMixing: false,
};

export function activeCorrections(physics: OrbitalPhysics): string[] {
  const active: string[] = [];
  if (physics.screeningExchange) active.push("screening+exchange");
  if (physics.relativisticSpinOrbit) active.push("relativistic/spin-orbit");
  if (physics.correlationMixing) active.push("correlation/mixing");
  return active;
}

/**
 * |psi|^2 on the x-z plane with the corrections applied.
 *
 * With every correction off this is exactly {@link densitySlice} for Z = 1,
 * which the tests assert -- an approximation that quietly changed the
 * hydrogenic baseline would be worse than not offering it.
 */
export function correctedDensitySlice(
  orbital: Orbital,
  physics: OrbitalPhysics,
  size: number,
  extent: number,
): DensitySlice {
  const zEffBase = effectiveNuclearCharge(
    orbital.z,
    orbital.n,
    orbital.l,
    physics.screeningExchange,
  );
  const zEff = applyRelativisticContraction(
    zEffBase,
    orbital.n,
    orbital.l,
    physics.relativisticSpinOrbit,
  );
  const terms = physics.correlationMixing
    ? buildCorrelationTerms(orbital.n, orbital.l, orbital.m)
    : [];

  const density = new Float64Array(size * size);
  let max = 0;
  let total = 0;
  for (let row = 0; row < size; row++) {
    const zc = extent * (1 - (2 * row) / (size - 1));
    for (let column = 0; column < size; column++) {
      const xc = extent * ((2 * column) / (size - 1) - 1);
      const r = Math.hypot(xc, zc);
      const theta = r === 0 ? 0 : Math.acos(zc / r);
      const phi = xc >= 0 ? 0 : Math.PI;

      const base = radialWavefunction(orbital.n, orbital.l, zEff, r) *
        realSphericalHarmonic(orbital.l, orbital.m, theta, phi);
      let value = base * base;
      for (const term of terms) {
        const psi = radialWavefunction(term.n, term.l, zEff, r) *
          realSphericalHarmonic(term.l, term.m, theta, phi);
        value += term.weight * psi * psi;
      }
      value *= spinOrbitDensityFactor(
        theta,
        orbital.n,
        orbital.l,
        orbital.m,
        zEff,
        physics.relativisticSpinOrbit,
      );

      density[row * size + column] = value;
      total += value;
      if (value > max) max = value;
    }
  }
  if (total > 0) {
    for (let i = 0; i < density.length; i++) density[i] /= total;
    max /= total;
  }
  return { size, density, extent, max };
}

// ---------------------------------------------------------------------------
// The 3-D cloud
//
// A slice through the x-z plane shows an orbital's nodal structure exactly,
// which is why it was the first view. What it cannot show is the shape: a
// d_xy and a d_x2-y2 have identical x-z slices and look nothing alike. For
// that the density has to be sampled as a volume and projected, and the
// projection has to be rotatable or the reader is back to guessing.
// ---------------------------------------------------------------------------

export interface DensityVolume {
  readonly size: number;
  /** |psi|^2 on a size^3 grid, index (((z * size) + y) * size) + x. */
  readonly density: Float64Array;
  readonly extent: number;
  readonly max: number;
}

/** Samples |psi|^2 over a cube, with the same corrections the slice uses. */
export function densityVolume(
  orbital: Orbital,
  physics: OrbitalPhysics,
  size: number,
  extent: number,
): DensityVolume {
  const zEffBase = effectiveNuclearCharge(
    orbital.z,
    orbital.n,
    orbital.l,
    physics.screeningExchange,
  );
  const zEff = applyRelativisticContraction(
    zEffBase,
    orbital.n,
    orbital.l,
    physics.relativisticSpinOrbit,
  );
  const terms = physics.correlationMixing
    ? buildCorrelationTerms(orbital.n, orbital.l, orbital.m)
    : [];

  const density = new Float64Array(size * size * size);
  let max = 0;
  const step = (2 * extent) / (size - 1);
  for (let iz = 0; iz < size; iz++) {
    const zc = -extent + iz * step;
    for (let iy = 0; iy < size; iy++) {
      const yc = -extent + iy * step;
      for (let ix = 0; ix < size; ix++) {
        const xc = -extent + ix * step;
        const r = Math.sqrt(xc * xc + yc * yc + zc * zc);
        const theta = r === 0 ? 0 : Math.acos(zc / r);
        const phi = Math.atan2(yc, xc);

        const base = radialWavefunction(orbital.n, orbital.l, zEff, r) *
          realSphericalHarmonic(orbital.l, orbital.m, theta, phi);
        let value = base * base;
        for (const term of terms) {
          const psi = radialWavefunction(term.n, term.l, zEff, r) *
            realSphericalHarmonic(term.l, term.m, theta, phi);
          value += term.weight * psi * psi;
        }
        value *= spinOrbitDensityFactor(
          theta,
          orbital.n,
          orbital.l,
          orbital.m,
          zEff,
          physics.relativisticSpinOrbit,
        );
        density[(iz * size + iy) * size + ix] = value;
        if (value > max) max = value;
      }
    }
  }
  return { size, density, extent, max };
}

export interface Projection {
  readonly width: number;
  readonly height: number;
  /** Column density, row-major, normalised so the peak is 1. */
  readonly image: Float64Array;
}

/**
 * Projects the volume along the view axis after a yaw/pitch rotation.
 *
 * Forward splatting rather than ray marching: every voxel is rotated once and
 * added to the pixel it lands on, which costs one pass over the volume and
 * needs no interpolation along a ray. The result is column density -- what an
 * X-ray of the cloud would show -- so a lobe pointing at the viewer reads
 * bright and the nodal planes stay dark.
 */
export function projectVolume(
  volume: DensityVolume,
  yaw: number,
  pitch: number,
  width: number,
  height: number,
): Projection {
  const image = new Float64Array(width * height);
  const size = volume.size;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  // The cube's half-diagonal, so no rotation can push a voxel off the canvas.
  const reach = Math.SQRT2;
  const half = (size - 1) / 2;

  for (let iz = 0; iz < size; iz++) {
    const z = (iz - half) / half;
    for (let iy = 0; iy < size; iy++) {
      const y = (iy - half) / half;
      const rowBase = (iz * size + iy) * size;
      for (let ix = 0; ix < size; ix++) {
        const value = volume.density[rowBase + ix];
        if (value <= 0) continue;
        const x = (ix - half) / half;

        // Yaw about the vertical axis, then pitch about the horizontal one.
        const x1 = x * cy + y * sy;
        const y1 = -x * sy + y * cy;
        const y2 = y1 * cp + z * sp;
        const z2 = -y1 * sp + z * cp;

        // x1 across, z2 up; y2 is depth and is summed away.
        const u = Math.round(((x1 / reach + 1) / 2) * (width - 1));
        const v = Math.round(((1 - z2 / reach) / 2) * (height - 1));
        if (u < 0 || u >= width || v < 0 || v >= height) continue;
        image[v * width + u] += value;
      }
    }
  }

  let peak = 0;
  for (const v of image) if (v > peak) peak = v;
  if (peak > 0) { for (let i = 0; i < image.length; i++) image[i] /= peak; }
  return { width, height, image };
}
