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
