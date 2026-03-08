import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { dmrgWeightsInWorker, ensureMoonlabWorker, probabilitiesFromAmplitudes } from '../workers/moonlabClient';
import { ElementPicker } from './ElementPicker';
import { ELEMENTS, type Atom } from './elements';
import './Orbitals.css';

// Preload Moonlab once per module load (avoids React strict-mode double effects)
const orbitalPreload =
  typeof window !== 'undefined'
    ? ensureMoonlabWorker().catch((err) => {
        console.error('Moonlab worker preload failed', err);
        throw err;
      })
    : Promise.resolve();

const L_LABELS = ['s', 'p', 'd', 'f', 'g'];
const MOBILE_COLLAPSE_QUERY = '(max-width: 900px)';
const GRID_COLOR = '#2b2e31';
const MAX_GRID = 32;
const MIN_GRID = 4;
const BASE_GRID = 32;
const BASE_STATES = BASE_GRID * BASE_GRID * BASE_GRID;
const DEFAULT_QUBITS = Math.ceil(Math.log2(BASE_STATES)); // 15 qubits
const MAX_QUBITS_UI = 32;
const SAFE_QUBITS = 24;
const MAX_QUBITS_RUNTIME = 32; // cap at 2^32 amplitudes (use with care)
const MAX_N_UI = 10;
const FINE_STRUCTURE_ALPHA = 1 / 137.035999084;
const DMRG_MIN_SITES = 4;
const DMRG_MAX_SITES = 12;
const DEFAULT_DMRG_SITES = 6;
const DEFAULT_ATOM = ELEMENTS.find((e) => e.symbol === 'Fe') || ELEMENTS[0];
const DEFAULT_N = 5;
const DEFAULT_L = 2;
const DEFAULT_M = 2;
const DEFAULT_POINT_COUNT = 494000;
const POINT_SIZE_MIN_UI = 0.01;
const POINT_SIZE_MAX_UI = 0.15;
const DEFAULT_POINT_SIZE = POINT_SIZE_MIN_UI;
const OPACITY_MIN_UI = 0.1;
const OPACITY_MAX_UI = 0.9;
const DEFAULT_OPACITY = 0.2;
const DEFAULT_CLOUD_COLOR = '#ffffff';
const DEFAULT_SHELL_COLOR_A = '#ffffff';
const DEFAULT_SHELL_COLOR_B = '#612bde';
const PROBABILITY_DRIFT_FALLBACK_L1 = 5e-3;
const DMRG_INFLUENCE_BLEND = 0.35;
const DMRG_INFLUENCE_GAMMA = 0.65;
const ADAPTIVE_EXTENT_MAX_PASSES = 8;
const ADAPTIVE_EXTENT_BOUNDARY_TARGET = 0.08;
const ADAPTIVE_EXTENT_GROWTH_FACTOR = 1.6;
const ADAPTIVE_EXTENT_MAX = 40;

const CONTROL_TOOLTIPS = {
  atom: 'Choose the element (atomic number Z). Higher Z pulls the cloud inward and increases radial decay.',
  chooseElement: 'Open the periodic table to select a different element.',
  randomQuantumState: 'Pick a random element and a valid random quantum tuple (n, l, m).',
  n: `Sets the principal quantum number (1-${MAX_N_UI}); higher n increases orbital size and radial nodes.`,
  screeningExchange:
    'Approximate multi-electron screening/exchange using shell-based effective nuclear charge (Zeff).',
  relativisticSpinOrbit:
    'Applies relativistic radial contraction and a spin-orbit-inspired density splitting for high-Z atoms.',
  correlationMixing:
    'Mixes neighboring orbital configurations to approximate many-electron correlation effects.',
  qubits: 'Controls WASM state size; more qubits increase lattice resolution and memory/time cost.',
  allowHighQubits: 'Unlocks 25-32 qubits; requires very high memory (64 GB+).',
  useDmrg: 'Run TFIM ground-state solver in WASM to modulate orbital sampling.',
  dmrgSites: 'Number of sites in the TFIM chain; more sites cost more compute.',
  dmrgG: 'Transverse field ratio g/J; shifts the TFIM ground state.',
  l: 'Orbital shape selector (s, p, d, f, ...).',
  m: 'Orbital orientation (magnetic quantum number).',
  guides: 'Show or hide the lattice grid and Cartesian axes.',
  background: 'Toggle the pixelated moon backdrop behind the simulation.',
  cloudColor: 'Set the tint used for the orbital point cloud.',
  shellColors: 'Alternate between two colors for successive energy shells.',
  shellColorA: 'Primary shell color (even shells).',
  shellColorB: 'Secondary shell color (odd shells).',
  pointCount: 'Number of sampled points; higher values yield a denser cloud.',
  pointSize: 'Rendered size of each point sprite.',
  opacity: 'Cloud transparency; lower values are more transparent.',
  regenerate: 'Recompute the orbital cloud with current settings.',
  rotation: 'Toggle auto-rotation of the scene.',
};

interface PhysicsModelOptions {
  screeningExchange: boolean;
  relativisticSpinOrbit: boolean;
  correlationMixing: boolean;
}

interface CloudParams {
  atom: Atom;
  n: number;
  l: number;
  m: number;
  pointCount: number;
  extent: number;
  gridSize: number;
  physics: PhysicsModelOptions;
}

interface ProbabilityGrid {
  positions: { x: number; y: number; z: number }[];
  probabilities: number[];
  elapsedMs: number;
}

const clamp = (value: number, min: number, max: number) =>
  Math.min(Math.max(value, min), max);

const randomInt = (min: number, max: number): number =>
  Math.floor(Math.random() * (max - min + 1)) + min;

const hexToRgb = (hex: string) => {
  const cleaned = hex.trim().replace('#', '');
  const expanded = cleaned.length === 3
    ? cleaned.split('').map((ch) => ch + ch).join('')
    : cleaned;
  if (expanded.length !== 6) {
    return { r: 1, g: 1, b: 1 };
  }
  const value = Number.parseInt(expanded, 16);
  return {
    r: ((value >> 16) & 255) / 255,
    g: ((value >> 8) & 255) / 255,
    b: (value & 255) / 255,
  };
};

const factorial = (n: number): number => {
  if (n < 0) return 1;
  let result = 1;
  for (let i = 2; i <= n; i++) {
    result *= i;
  }
  return result;
};

const binomial = (n: number, k: number): number => {
  if (k < 0 || k > n) return 0;
  return factorial(n) / (factorial(k) * factorial(n - k));
};

// Associated Laguerre polynomial L_p^k(x)
const assocLaguerre = (p: number, k: number, x: number): number => {
  let sum = 0;
  for (let m = 0; m <= p; m++) {
    const coeff = ((-1) ** m) * binomial(p + k, p - m) / factorial(m);
    sum += coeff * x ** m;
  }
  return sum;
};

// Associated Legendre polynomial P_l^m(x) for m >= 0
const associatedLegendre = (l: number, m: number, x: number): number => {
  const absX = clamp(x, -1, 1);
  let pmm = 1;
  if (m > 0) {
    const somx2 = Math.sqrt((1 - absX) * (1 + absX));
    let fact = 1;
    for (let i = 1; i <= m; i++) {
      pmm *= -fact * somx2;
      fact += 2;
    }
  }
  if (l === m) return pmm;

  let pmmp1 = absX * (2 * m + 1) * pmm;
  if (l === m + 1) return pmmp1;

  let pll = 0;
  for (let ll = m + 2; ll <= l; ll++) {
    pll = ((2 * ll - 1) * absX * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
    pmm = pmmp1;
    pmmp1 = pll;
  }
  return pll;
};

// Real-valued spherical harmonics (for visualization)
const realSphericalHarmonic = (l: number, m: number, theta: number, phi: number): number => {
  const absM = Math.abs(m);
  const plm = associatedLegendre(l, absM, Math.cos(theta));
  const norm =
    Math.sqrt(((2 * l + 1) / (4 * Math.PI)) * (factorial(l - absM) / factorial(l + absM)));

  if (m > 0) return Math.sqrt(2) * norm * plm * Math.cos(absM * phi);
  if (m < 0) return Math.sqrt(2) * norm * plm * Math.sin(absM * phi);
  return norm * plm;
};

const radialComponent = (n: number, l: number, r: number, Z: number): number => {
  const rho = (2 * Z * r) / n;
  const prefactor =
    Math.sqrt(((2 * Z) / n) ** 3 * (factorial(n - l - 1) / (2 * n * factorial(n + l))));
  const laguerre = assocLaguerre(n - l - 1, 2 * l + 1, rho);
  const radial = prefactor * Math.exp(-rho / 2) * rho ** l * laguerre;
  return radial;
};

const shellCapacity = (n: number) => 2 * n * n;

const electronsBeforeShell = (n: number): number => {
  let total = 0;
  for (let k = 1; k < n; k++) {
    total += shellCapacity(k);
  }
  return total;
};

const electronsInShell = (Z: number, n: number): number => {
  if (n < 1) return 0;
  const before = electronsBeforeShell(n);
  return clamp(Z - before, 0, shellCapacity(n));
};

const effectiveNuclearCharge = (atom: Atom, n: number, l: number, enabled: boolean): number => {
  if (!enabled) return atom.Z;

  const inShellN = electronsInShell(atom.Z, n);
  const inShellNMinus1 = n > 1 ? electronsInShell(atom.Z, n - 1) : 0;
  const sameShell = Math.max(0, inShellN - 1);
  const lowerShell = Math.max(0, atom.Z - inShellN - inShellNMinus1);

  const sameCoeff = 0.35;
  const nMinus1Coeff = l <= 1 ? 0.85 : 1.0;
  const exchangeTerm = l > 0 ? 0.02 * ((2 * l + 1) / Math.max(1, n)) : 0;

  const shielding = sameCoeff * sameShell + nMinus1Coeff * inShellNMinus1 + lowerShell;
  return clamp(atom.Z - shielding + exchangeTerm, 1, atom.Z);
};

const applyRelativisticContraction = (
  zEff: number,
  n: number,
  l: number,
  enabled: boolean
): number => {
  if (!enabled) return zEff;
  const beta = Math.pow(FINE_STRUCTURE_ALPHA * zEff, 2);
  const contraction = 1 + (beta * 0.32) / (Math.max(1, n) * (l + 1));
  return zEff * contraction;
};

const spinOrbitDensityFactor = (
  theta: number,
  n: number,
  l: number,
  m: number,
  zEff: number,
  enabled: boolean
): number => {
  if (!enabled || l === 0 || m === 0) return 1;
  const beta = Math.pow(FINE_STRUCTURE_ALPHA * zEff, 2);
  const coupling = (beta * 0.28 * l) / Math.max(1, n * n);
  const orientation = Math.cos(theta) * (m / Math.max(1, l));
  return clamp(1 + coupling * orientation, 0.35, 1.65);
};

type CorrelationTerm = {
  n: number;
  l: number;
  m: number;
  weight: number;
};

const buildCorrelationTerms = (n: number, l: number, m: number): CorrelationTerm[] => {
  const terms: CorrelationTerm[] = [];
  const addTerm = (termN: number, termL: number, weight: number) => {
    if (termN < 1 || termN > MAX_N_UI) return;
    if (termL < 0 || termL >= termN) return;
    if (weight <= 0) return;
    terms.push({
      n: termN,
      l: termL,
      m: clamp(m, -termL, termL),
      weight,
    });
  };

  if (l - 1 >= 0) addTerm(n, l - 1, 0.28);
  if (l + 1 < n) addTerm(n, l + 1, 0.28);
  if (n - 1 >= 1) addTerm(n - 1, Math.min(l, n - 2), 0.44);

  let totalWeight = 0;
  for (let i = 0; i < terms.length; i++) {
    totalWeight += terms[i].weight;
  }
  if (totalWeight <= 0) return [];
  for (let i = 0; i < terms.length; i++) {
    terms[i].weight /= totalWeight;
  }
  return terms;
};

const correlationMixingStrength = (atom: Atom): number => clamp(0.04 + atom.Z * 0.0012, 0.04, 0.18);

const buildProbabilityGrid = async (params: CloudParams): Promise<ProbabilityGrid> => {
  const start = performance.now();
  const stateCount = params.gridSize * params.gridSize * params.gridSize;
  const positions: { x: number; y: number; z: number }[] = new Array(stateCount);
  const probabilities: number[] = new Array(stateCount);

  const spacing = (params.extent * 2) / (params.gridSize - 1);
  const baseZEff = effectiveNuclearCharge(
    params.atom,
    params.n,
    params.l,
    params.physics.screeningExchange
  );
  const baseRadialZ = applyRelativisticContraction(
    baseZEff,
    params.n,
    params.l,
    params.physics.relativisticSpinOrbit
  );

  const correlationTerms = params.physics.correlationMixing
    ? buildCorrelationTerms(params.n, params.l, params.m)
    : [];
  const correlationStrength =
    params.physics.correlationMixing && correlationTerms.length > 0
      ? correlationMixingStrength(params.atom)
      : 0;
  const preparedCorrelationTerms =
    correlationStrength > 0
      ? correlationTerms.map((term) => {
          const termZEff = effectiveNuclearCharge(
            params.atom,
            term.n,
            term.l,
            params.physics.screeningExchange
          );
          return {
            ...term,
            zEff: termZEff,
            radialZ: applyRelativisticContraction(
              termZEff,
              term.n,
              term.l,
              params.physics.relativisticSpinOrbit
            ),
          };
        })
      : [];

  let totalProb = 0;
  for (let idx = 0; idx < stateCount; idx++) {
    const z = Math.floor(idx / (params.gridSize * params.gridSize));
    const y = Math.floor((idx - z * params.gridSize * params.gridSize) / params.gridSize);
    const x = idx % params.gridSize;

    const posX = -params.extent + x * spacing;
    const posY = -params.extent + y * spacing;
    const posZ = -params.extent + z * spacing;

    const r = Math.sqrt(posX * posX + posY * posY + posZ * posZ);
    const theta = r === 0 ? 0 : Math.acos(clamp(posZ / (r || 1), -1, 1));
    const phi = Math.atan2(posY, posX);
    const rr = Math.max(r, 1e-5);
    const radial = radialComponent(params.n, params.l, rr, baseRadialZ);
    const ylm = realSphericalHarmonic(params.l, params.m, theta, phi);
    const soFactor = spinOrbitDensityFactor(
      theta,
      params.n,
      params.l,
      params.m,
      baseZEff,
      params.physics.relativisticSpinOrbit
    );
    let prob = Math.max(0, radial * radial * ylm * ylm * soFactor);

    if (correlationStrength > 0) {
      let mixedProb = 0;
      for (let i = 0; i < preparedCorrelationTerms.length; i++) {
        const term = preparedCorrelationTerms[i];
        const termRadial = radialComponent(term.n, term.l, rr, term.radialZ);
        const termYlm = realSphericalHarmonic(term.l, term.m, theta, phi);
        const termSoFactor = spinOrbitDensityFactor(
          theta,
          term.n,
          term.l,
          term.m,
          term.zEff,
          params.physics.relativisticSpinOrbit
        );
        mixedProb += term.weight * Math.max(0, termRadial * termRadial * termYlm * termYlm * termSoFactor);
      }
      prob = (1 - correlationStrength) * prob + correlationStrength * mixedProb;
    }

    positions[idx] = { x: posX, y: posY, z: posZ };
    probabilities[idx] = prob;
    totalProb += prob;
  }

  const norm = totalProb > 0 ? totalProb : 1;
  for (let i = 0; i < stateCount; i++) {
    probabilities[i] = probabilities[i] > 0 ? probabilities[i] / norm : 0;
  }

  // Default normalized probabilities for base lattice; actual dimension handled in regenerate
  const elapsedMs = performance.now() - start;
  return { positions, probabilities, elapsedMs };
};

const alignToDimension = (
  probabilities: number[],
  positions: { x: number; y: number; z: number }[],
  targetDim: number
): { probs: number[]; pos: { x: number; y: number; z: number }[] } => {
  const baseLen = probabilities.length;
  if (targetDim === baseLen) return { probs: probabilities, pos: positions };

  if (targetDim < baseLen) {
    const slicedProbs = probabilities.slice(0, targetDim);
    const slicedPos = positions.slice(0, targetDim);
    const total = slicedProbs.reduce((a, b) => a + b, 0);
    const norm = total > 0 ? total : 1;
    return { probs: slicedProbs.map((p) => p / norm), pos: slicedPos };
  }

  // targetDim > baseLen: pad with zeros at origin
  const paddedProbs = probabilities.slice();
  const paddedPos = positions.slice();
  const padCount = targetDim - baseLen;
  for (let i = 0; i < padCount; i++) {
    paddedProbs.push(0);
    paddedPos.push({ x: 0, y: 0, z: 0 });
  }
  return { probs: paddedProbs, pos: paddedPos };
};

const nextPowerOfTwo = (value: number): number => {
  if (!Number.isFinite(value) || value <= 1) return 1;
  let power = 1;
  while (power < value && power < 0x80000000) {
    power <<= 1;
  }
  return power;
};

const boundaryMassFraction = (probabilities: ArrayLike<number>, gridSize: number): number => {
  if (gridSize <= 2) return 0;
  const area = gridSize * gridSize;
  let boundaryMass = 0;
  let totalMass = 0;

  for (let i = 0; i < probabilities.length; i++) {
    const p = probabilities[i] ?? 0;
    if (p <= 0) continue;
    totalMass += p;

    const z = Math.floor(i / area);
    const yz = i - z * area;
    const y = Math.floor(yz / gridSize);
    const x = yz - y * gridSize;
    if (
      x === 0 ||
      y === 0 ||
      z === 0 ||
      x === gridSize - 1 ||
      y === gridSize - 1 ||
      z === gridSize - 1
    ) {
      boundaryMass += p;
    }
  }

  return totalMass > 0 ? boundaryMass / totalMass : 0;
};

const samplePoints = (
  probabilities: ArrayLike<number>,
  positions: { x: number; y: number; z: number }[],
  count: number,
  extent: number,
  gridSize: number
): Float32Array => {
  const points = new Float32Array(count * 3);

  // Preferred path: sample continuously from grid cells using cell-average mass.
  // This avoids point clouds clumping at lattice node centers (checkerboard artifacts).
  const nodeCount = gridSize * gridSize * gridSize;
  if (gridSize > 1 && probabilities.length === nodeCount) {
    const nodeIndex = (x: number, y: number, z: number) => z * gridSize * gridSize + y * gridSize + x;

    // Smooth the node field with a separable 1-2-1 kernel in 3D.
    // This removes harsh voxel boundaries while preserving global orbital structure.
    const smoothed = new Float64Array(nodeCount);
    for (let z = 0; z < gridSize; z++) {
      for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
          let sum = 0;
          let wsum = 0;
          for (let dz = -1; dz <= 1; dz++) {
            const zz = clamp(z + dz, 0, gridSize - 1);
            const wz = dz === 0 ? 2 : 1;
            for (let dy = -1; dy <= 1; dy++) {
              const yy = clamp(y + dy, 0, gridSize - 1);
              const wy = dy === 0 ? 2 : 1;
              for (let dx = -1; dx <= 1; dx++) {
                const xx = clamp(x + dx, 0, gridSize - 1);
                const wx = dx === 0 ? 2 : 1;
                const w = wx * wy * wz;
                sum += probabilities[nodeIndex(xx, yy, zz)] * w;
                wsum += w;
              }
            }
          }
          smoothed[nodeIndex(x, y, z)] = wsum > 0 ? sum / wsum : probabilities[nodeIndex(x, y, z)];
        }
      }
    }

    const cellsPerAxis = gridSize - 1;
    const cellsPerLayer = cellsPerAxis * cellsPerAxis;
    const cellCount = cellsPerAxis * cellsPerAxis * cellsPerAxis;
    const cellCdf = new Float64Array(cellCount);

    let totalCellMass = 0;
    let ci = 0;
    for (let z = 0; z < cellsPerAxis; z++) {
      for (let y = 0; y < cellsPerAxis; y++) {
        for (let x = 0; x < cellsPerAxis; x++) {
          const p000 = smoothed[nodeIndex(x, y, z)];
          const p100 = smoothed[nodeIndex(x + 1, y, z)];
          const p010 = smoothed[nodeIndex(x, y + 1, z)];
          const p110 = smoothed[nodeIndex(x + 1, y + 1, z)];
          const p001 = smoothed[nodeIndex(x, y, z + 1)];
          const p101 = smoothed[nodeIndex(x + 1, y, z + 1)];
          const p011 = smoothed[nodeIndex(x, y + 1, z + 1)];
          const p111 = smoothed[nodeIndex(x + 1, y + 1, z + 1)];
          const cellMass = (p000 + p100 + p010 + p110 + p001 + p101 + p011 + p111) * 0.125;
          totalCellMass += cellMass;
          cellCdf[ci++] = totalCellMass;
        }
      }
    }

    if (totalCellMass > 0) {
      const spacing = (extent * 2) / (gridSize - 1);
      for (let i = 0; i < count; i++) {
        const r = Math.random() * totalCellMass;
        let low = 0;
        let high = cellCdf.length - 1;
        while (low < high) {
          const mid = Math.floor((low + high) / 2);
          if (r <= cellCdf[mid]) {
            high = mid;
          } else {
            low = mid + 1;
          }
        }

        const z = Math.floor(low / cellsPerLayer);
        const yz = low - z * cellsPerLayer;
        const y = Math.floor(yz / cellsPerAxis);
        const x = yz - y * cellsPerAxis;

        const p000 = smoothed[nodeIndex(x, y, z)];
        const p100 = smoothed[nodeIndex(x + 1, y, z)];
        const p010 = smoothed[nodeIndex(x, y + 1, z)];
        const p110 = smoothed[nodeIndex(x + 1, y + 1, z)];
        const p001 = smoothed[nodeIndex(x, y, z + 1)];
        const p101 = smoothed[nodeIndex(x + 1, y, z + 1)];
        const p011 = smoothed[nodeIndex(x, y + 1, z + 1)];
        const p111 = smoothed[nodeIndex(x + 1, y + 1, z + 1)];
        const maxCorner = Math.max(p000, p100, p010, p110, p001, p101, p011, p111);

        // Rejection sample inside the selected voxel using trilinear density.
        // This removes piecewise-constant "block" look inside each cell.
        let u = Math.random();
        let v = Math.random();
        let w = Math.random();
        if (maxCorner > 0) {
          for (let attempt = 0; attempt < 6; attempt++) {
            u = Math.random();
            v = Math.random();
            w = Math.random();
            const ux = 1 - u;
            const vy = 1 - v;
            const wz = 1 - w;
            const trilinear =
              p000 * ux * vy * wz +
              p100 * u * vy * wz +
              p010 * ux * v * wz +
              p110 * u * v * wz +
              p001 * ux * vy * w +
              p101 * u * vy * w +
              p011 * ux * v * w +
              p111 * u * v * w;
            if (Math.random() * maxCorner <= trilinear) {
              break;
            }
          }
        }

        points[i * 3] = -extent + (x + u) * spacing;
        points[i * 3 + 1] = -extent + (y + v) * spacing;
        points[i * 3 + 2] = -extent + (z + w) * spacing;
      }
      return points;
    }
  }

  // Fallback for non-cubic aligned dimensions.
  const cdf = new Float64Array(probabilities.length);
  let total = 0;
  for (let i = 0; i < probabilities.length; i++) {
    total += probabilities[i];
    cdf[i] = total;
  }
  if (total <= 0) {
    for (let i = 0; i < count; i++) {
      points[i * 3] = (Math.random() - 0.5) * 2;
      points[i * 3 + 1] = (Math.random() - 0.5) * 2;
      points[i * 3 + 2] = (Math.random() - 0.5) * 2;
    }
    return points;
  }

  const spacing = (extent * 2) / Math.max(1, gridSize - 1);
  const jitterSigma = spacing * 0.2;
  const gaussianRandom = () => {
    const u1 = Math.max(1e-12, Math.random());
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };

  for (let i = 0; i < count; i++) {
    const r = Math.random() * total;
    let low = 0;
    let high = cdf.length - 1;
    while (low < high) {
      const mid = Math.floor((low + high) / 2);
      if (r <= cdf[mid]) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }

    const pos = positions[low];
    points[i * 3] = pos.x + gaussianRandom() * jitterSigma;
    points[i * 3 + 1] = pos.y + gaussianRandom() * jitterSigma;
    points[i * 3 + 2] = pos.z + gaussianRandom() * jitterSigma;
  }

  return points;
};

const extentForAtom = (n: number, Z: number) => {
  const base = 3 + n * 0.7;
  const compression = Math.max(1, Z / 4);
  return base / compression;
};

const estimateSpatialExtent = (
  atom: Atom,
  n: number,
  l: number,
  physics: PhysicsModelOptions
): number => {
  let extent = extentForAtom(n, atom.Z);
  const adaptiveExtentEnabled =
    physics.screeningExchange || physics.relativisticSpinOrbit || physics.correlationMixing;
  if (adaptiveExtentEnabled) {
    const seedZEff = effectiveNuclearCharge(atom, n, l, physics.screeningExchange);
    const seedRadialZ = applyRelativisticContraction(seedZEff, n, l, physics.relativisticSpinOrbit);
    const spreadScale = (n * n) / Math.max(0.5, seedRadialZ);
    const seededExtent = clamp(1.5 + spreadScale * 0.55, extent, ADAPTIVE_EXTENT_MAX);
    extent = Math.max(extent, seededExtent);
    if (physics.correlationMixing) {
      extent = Math.min(ADAPTIVE_EXTENT_MAX, extent * 1.08);
    }
  }
  return extent;
};

const tunePointStyleForExtent = (
  extent: number
): {
  pointSize: number;
  opacity: number;
} => {
  const spreadNorm = clamp((extent - 1.5) / (ADAPTIVE_EXTENT_MAX * 0.35), 0, 1);
  const pointSize = clamp(
    POINT_SIZE_MIN_UI + spreadNorm * 0.085,
    POINT_SIZE_MIN_UI,
    POINT_SIZE_MAX_UI
  );
  const opacity = clamp(OPACITY_MIN_UI + spreadNorm * 0.6, OPACITY_MIN_UI, OPACITY_MAX_UI);
  return { pointSize, opacity };
};

const nucleusRadiusForExtent = (extent: number) => clamp(extent * 0.003, 0.006, 0.02);

const isPowerOfTwo = (value: number) => value > 0 && (value & (value - 1)) === 0;

const dmrgWeightIndexForGrid = (
  linearIndex: number,
  gridSide: number,
  dmrgLength: number
): number => {
  if (dmrgLength <= 0) return 0;
  const area = gridSide * gridSide;
  const z = Math.floor(linearIndex / area);
  const yz = linearIndex - z * area;
  const y = Math.floor(yz / gridSide);
  const x = yz - y * gridSide;

  // Spatial hash avoids 1D modulo aliasing (visible axis-locked striping/blocks).
  const hx = Math.imul(x + 1, 73856093);
  const hy = Math.imul(y + 1, 19349663);
  const hz = Math.imul(z + 1, 83492791);
  const hash = (hx ^ hy ^ hz) >>> 0;

  if (isPowerOfTwo(dmrgLength)) {
    return hash & (dmrgLength - 1);
  }
  return hash % dmrgLength;
};

const buildDmrgInfluenceSampler = (
  dmrgWeights: Float64Array,
  gridSide: number
): ((linearIndex: number) => number) => {
  const dmrgLength = dmrgWeights.length;
  if (dmrgLength === 0) {
    return () => 1;
  }

  let mean = 0;
  for (let i = 0; i < dmrgLength; i++) {
    mean += dmrgWeights[i];
  }
  mean = mean > 0 ? mean / dmrgLength : 1;

  const cells = Math.max(1, gridSide - 1);
  const side = Math.round(Math.cbrt(dmrgLength));
  const hasCubeLayout = side * side * side === dmrgLength;
  const dmrgIndex3D = (x: number, y: number, z: number) => z * side * side + y * side + x;

  const soften = (rawWeight: number) => {
    const normalized = mean > 0 ? rawWeight / mean : 1;
    const compressed = Math.pow(Math.max(normalized, 1e-12), DMRG_INFLUENCE_GAMMA);
    return 1 + DMRG_INFLUENCE_BLEND * (compressed - 1);
  };

  if (hasCubeLayout) {
    return (linearIndex: number) => {
      const area = gridSide * gridSide;
      const z = Math.floor(linearIndex / area);
      const yz = linearIndex - z * area;
      const y = Math.floor(yz / gridSide);
      const x = yz - y * gridSide;

      const tx = (x / cells) * (side - 1);
      const ty = (y / cells) * (side - 1);
      const tz = (z / cells) * (side - 1);

      const x0 = Math.floor(tx);
      const y0 = Math.floor(ty);
      const z0 = Math.floor(tz);
      const x1 = Math.min(side - 1, x0 + 1);
      const y1 = Math.min(side - 1, y0 + 1);
      const z1 = Math.min(side - 1, z0 + 1);

      const ux = tx - x0;
      const uy = ty - y0;
      const uz = tz - z0;

      const w000 = dmrgWeights[dmrgIndex3D(x0, y0, z0)];
      const w100 = dmrgWeights[dmrgIndex3D(x1, y0, z0)];
      const w010 = dmrgWeights[dmrgIndex3D(x0, y1, z0)];
      const w110 = dmrgWeights[dmrgIndex3D(x1, y1, z0)];
      const w001 = dmrgWeights[dmrgIndex3D(x0, y0, z1)];
      const w101 = dmrgWeights[dmrgIndex3D(x1, y0, z1)];
      const w011 = dmrgWeights[dmrgIndex3D(x0, y1, z1)];
      const w111 = dmrgWeights[dmrgIndex3D(x1, y1, z1)];

      const vx00 = w000 * (1 - ux) + w100 * ux;
      const vx10 = w010 * (1 - ux) + w110 * ux;
      const vx01 = w001 * (1 - ux) + w101 * ux;
      const vx11 = w011 * (1 - ux) + w111 * ux;
      const vxy0 = vx00 * (1 - uy) + vx10 * uy;
      const vxy1 = vx01 * (1 - uy) + vx11 * uy;
      const raw = vxy0 * (1 - uz) + vxy1 * uz;

      return soften(raw);
    };
  }

  // Fallback for non-cubic DMRG lengths: hashed mapping with softened influence.
  return (linearIndex: number) => {
    const idx = dmrgWeightIndexForGrid(linearIndex, gridSide, dmrgLength);
    return soften(dmrgWeights[idx]);
  };
};

const createPointSpriteTexture = (): THREE.Texture => {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    const fallback = new THREE.Texture();
    fallback.needsUpdate = true;
    return fallback;
  }

  const r = size / 2;
  const gradient = ctx.createRadialGradient(r, r, 0, r, r, r);
  gradient.addColorStop(0, 'rgba(255,255,255,1)');
  gradient.addColorStop(0.45, 'rgba(255,255,255,0.9)');
  gradient.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
};

const fitCameraToExtent = (
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  extent: number
) => {
  const safeExtent = Math.max(0.5, extent);
  const targetDistance = Math.max(8, safeExtent * 2.8);
  const near = Math.max(0.01, safeExtent * 0.01);
  const far = Math.max(300, safeExtent * 160);

  const target = controls.target.clone();
  const dir = camera.position.clone().sub(target);
  if (dir.lengthSq() < 1e-6) {
    dir.set(1, 1, 1).normalize();
  } else {
    dir.normalize();
  }

  const currentDistance = camera.position.distanceTo(target);
  if (currentDistance < targetDistance * 0.7 || currentDistance > targetDistance * 4.5) {
    camera.position.copy(target.clone().add(dir.multiplyScalar(targetDistance)));
  }

  camera.near = near;
  camera.far = far;
  camera.updateProjectionMatrix();

  controls.minDistance = Math.max(0.5, safeExtent * 0.12);
  controls.maxDistance = Math.max(60, safeExtent * 12);
  controls.update();
};

const OrbitalDemo: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const cloudRef = useRef<THREE.Points>();
  const sceneRef = useRef<THREE.Scene>();
  const rendererRef = useRef<THREE.WebGLRenderer>();
  const cameraRef = useRef<THREE.PerspectiveCamera>();
  const materialRef = useRef<THREE.PointsMaterial>();
  const controlsRef = useRef<OrbitControls>();
  const gridRef = useRef<THREE.GridHelper | null>(null);
  const axesRef = useRef<THREE.AxesHelper | null>(null);
  const nucleusRef = useRef<THREE.Mesh | null>(null);
  const pointTextureRef = useRef<THREE.Texture | null>(null);
  const rotatingRef = useRef<boolean>(true);
  const [atom, setAtom] = useState<Atom>(() => DEFAULT_ATOM);
  const [n, setN] = useState<number>(DEFAULT_N);
  const [l, setL] = useState<number>(DEFAULT_L);
  const [m, setM] = useState<number>(DEFAULT_M);
  const [qubits, setQubits] = useState<number>(
    Math.max(4, Math.min(MAX_QUBITS_UI, DEFAULT_QUBITS))
  );
  const [allowHighQubits, setAllowHighQubits] = useState<boolean>(false);
  const [pointCount, setPointCount] = useState<number>(DEFAULT_POINT_COUNT);
  const [pointSize, setPointSize] = useState<number>(DEFAULT_POINT_SIZE);
  const [opacity, setOpacity] = useState<number>(DEFAULT_OPACITY);
  const [cloudColor, setCloudColor] = useState<string>(DEFAULT_CLOUD_COLOR);
  const [useShellColors, setUseShellColors] = useState<boolean>(true);
  const [shellColorA, setShellColorA] = useState<string>(DEFAULT_SHELL_COLOR_A);
  const [shellColorB, setShellColorB] = useState<string>(DEFAULT_SHELL_COLOR_B);
  const [isRotating, setIsRotating] = useState<boolean>(true);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [lastStats, setLastStats] = useState<{ elapsedMs: number; generated: number } | null>(null);
  const [useDmrg, setUseDmrg] = useState<boolean>(false);
  const [dmrgSites, setDmrgSites] = useState<number>(DEFAULT_DMRG_SITES);
  const [dmrgG, setDmrgG] = useState<number>(1.0);
  const [dmrgWeights, setDmrgWeights] = useState<Float64Array | null>(null);
  const [dmrgStats, setDmrgStats] = useState<{
    elapsedMs: number;
    sites: number;
    energy?: number;
    variance?: number;
  } | null>(null);
  const [isDmrgRunning, setIsDmrgRunning] = useState<boolean>(false);
  const [pointsBuffer, setPointsBuffer] = useState<Float32Array | null>(null);
  const [currentExtent, setCurrentExtent] = useState<number>(extentForAtom(DEFAULT_N, DEFAULT_ATOM.Z));
  const [isCollapsed, setIsCollapsed] = useState<boolean>(false);
  const [isPickerOpen, setIsPickerOpen] = useState<boolean>(false);
  const [showGuides, setShowGuides] = useState<boolean>(false);
  const [showBackground, setShowBackground] = useState<boolean>(false);
  const [useScreeningExchange, setUseScreeningExchange] = useState<boolean>(false);
  const [useRelativisticSpinOrbit, setUseRelativisticSpinOrbit] = useState<boolean>(false);
  const [useCorrelationMixing, setUseCorrelationMixing] = useState<boolean>(false);
  const dmrgRunId = useRef(0);
  const pageStyle = useMemo(
    () => ({
      backgroundColor: '#000',
      backgroundImage: showBackground ? 'var(--moon-bg-image)' : 'none',
      backgroundPosition: 'center bottom',
      backgroundSize: '100vw auto',
      backgroundRepeat: 'no-repeat',
    }),
    [showBackground]
  );
  const cloudRgb = useMemo(() => hexToRgb(cloudColor), [cloudColor]);
  const shellRgbA = useMemo(() => hexToRgb(shellColorA), [shellColorA]);
  const shellRgbB = useMemo(() => hexToRgb(shellColorB), [shellColorB]);
  const radialNodeRadii = useMemo(() => {
    if (!useShellColors) return [];
    const nodeCount = Math.max(0, n - l - 1);
    if (nodeCount === 0) return [];
    const zEff = effectiveNuclearCharge(atom, n, l, useScreeningExchange);
    const radialZ = applyRelativisticContraction(zEff, n, l, useRelativisticSpinOrbit);
    const maxR = Math.max(0.1, currentExtent * 1.2);
    const steps = 2048;
    const nodes: number[] = [];
    let prevR = 1e-4;
    let prevVal = radialComponent(n, l, prevR, radialZ);
    for (let i = 1; i <= steps; i++) {
      const r = (i / steps) * maxR;
      const val = radialComponent(n, l, r, radialZ);
      if (prevVal === 0) {
        prevVal = val;
        prevR = r;
        continue;
      }
      if (prevVal * val < 0) {
        const t = Math.abs(prevVal) / (Math.abs(prevVal) + Math.abs(val));
        nodes.push(prevR + t * (r - prevR));
        if (nodes.length >= nodeCount) break;
      }
      prevVal = val;
      prevR = r;
    }
    return nodes;
  }, [useShellColors, n, l, atom, currentExtent, useScreeningExchange, useRelativisticSpinOrbit]);

  const lOptions = useMemo(() => Array.from({ length: n }, (_, i) => i), [n]);
  const mOptions = useMemo(() => Array.from({ length: l * 2 + 1 }, (_, i) => i - l), [l]);
  const randomizeQuantumState = () => {
    const nextAtom = ELEMENTS[randomInt(0, ELEMENTS.length - 1)] ?? DEFAULT_ATOM;
    const nextN = randomInt(1, MAX_N_UI);
    const nextL = randomInt(0, nextN - 1);
    const nextM = randomInt(-nextL, nextL);
    const physics: PhysicsModelOptions = {
      screeningExchange: useScreeningExchange,
      relativisticSpinOrbit: useRelativisticSpinOrbit,
      correlationMixing: useCorrelationMixing,
    };
    const estimatedExtent = estimateSpatialExtent(nextAtom, nextN, nextL, physics);
    const tunedStyle = tunePointStyleForExtent(estimatedExtent);

    setAtom(nextAtom);
    setN(nextN);
    setL(nextL);
    setM(nextM);
    setPointSize(tunedStyle.pointSize);
    setOpacity(tunedStyle.opacity);
  };

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const media = window.matchMedia(MOBILE_COLLAPSE_QUERY);
    const apply = (matches: boolean) => setIsCollapsed(matches);
    apply(media.matches);
    const handler = (event: MediaQueryListEvent) => apply(event.matches);
    if (typeof media.addEventListener === 'function') {
      media.addEventListener('change', handler);
      return () => media.removeEventListener('change', handler);
    }
    media.addListener(handler);
    return () => media.removeListener(handler);
  }, []);

  useEffect(() => {
    setM((prev) => clamp(prev, -l, l));
  }, [l]);

  useEffect(() => {
    if (!useDmrg) {
      setDmrgWeights(null);
      setDmrgStats(null);
      setIsDmrgRunning(false);
      return;
    }

    const runId = ++dmrgRunId.current;
    let cancelled = false;
    const sites = clamp(dmrgSites, DMRG_MIN_SITES, DMRG_MAX_SITES);

    setIsDmrgRunning(true);
    const start = performance.now();

    const solve = async () => {
      await orbitalPreload;
      const result = await dmrgWeightsInWorker({ numSites: sites, g: dmrgG });
      if (cancelled || dmrgRunId.current !== runId) return;
      setDmrgWeights(result.weights);
      setDmrgStats({
        elapsedMs: result.elapsedMs ?? performance.now() - start,
        sites,
        energy: result.energy,
        variance: result.variance,
      });
    };

    solve()
      .catch((err) => {
        if (cancelled || dmrgRunId.current !== runId) return;
        console.error('DMRG solver failed', err);
        setDmrgWeights(null);
        setDmrgStats({
          elapsedMs: performance.now() - start,
          sites,
        });
      })
      .finally(() => {
        if (!cancelled && dmrgRunId.current === runId) {
          setIsDmrgRunning(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [useDmrg, dmrgSites, dmrgG]);

  // Initialize Three.js scene
  useEffect(() => {
    if (!mountRef.current) return;
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    const scene = new THREE.Scene();
    scene.background = null;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.05, 1000);
    camera.position.set(6, 6, 6);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    mountRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.8;
    controls.enablePan = false;
    controls.target.set(0, 0, 0);
    controls.update();

    const ambient = new THREE.AmbientLight('#c2c2c2', 0.35);
    scene.add(ambient);
    const dir = new THREE.DirectionalLight('#f0f0f0', 0.55);
    dir.position.set(4, 5, 2);
    scene.add(dir);

    const initialExtent = extentForAtom(n, atom.Z);
    const initialDim = Math.pow(2, Math.min(Math.max(8, DEFAULT_QUBITS), MAX_QUBITS_RUNTIME));
    const initialGridSide = Math.min(MAX_GRID, Math.max(MIN_GRID, Math.floor(Math.cbrt(initialDim))));
    const initialSpacing = (initialExtent * 2) / (initialGridSide - 1);
    const initialSize = initialSpacing * (initialGridSide - 1);

    const nucleus = new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 32),
      new THREE.MeshBasicMaterial({ color: '#d8d8d8' })
    );
    const nucleusRadius = nucleusRadiusForExtent(initialExtent);
    nucleus.scale.setScalar(nucleusRadius);
    scene.add(nucleus);
    nucleusRef.current = nucleus;

    const grid = new THREE.GridHelper(initialSize, initialGridSide - 1, GRID_COLOR, GRID_COLOR);
    grid.position.y = -initialExtent;
    grid.visible = showGuides;
    scene.add(grid);
    gridRef.current = grid;

    const axes = new THREE.AxesHelper(initialExtent * 1.2);
    axes.position.set(0, -initialExtent, 0);
    const axesMaterial = axes.material as THREE.LineBasicMaterial;
    axesMaterial.vertexColors = false;
    axesMaterial.color.set('#7d7f82');
    axes.visible = showGuides;
    scene.add(axes);
    axesRef.current = axes;

    const pointTexture = createPointSpriteTexture();
    pointTextureRef.current = pointTexture;

    const material = new THREE.PointsMaterial({
      size: pointSize,
      sizeAttenuation: true,
      transparent: true,
      opacity,
      vertexColors: true,
      map: pointTexture,
      alphaTest: 0.06,
      depthWrite: false,
    });

    sceneRef.current = scene;
    rendererRef.current = renderer;
    cameraRef.current = camera;
    controlsRef.current = controls;
    materialRef.current = material;

    fitCameraToExtent(camera, controls, initialExtent);

    const handleResize = () => {
      if (!rendererRef.current || !camera || !mountRef.current) return;
      const w = mountRef.current.clientWidth;
      const h = mountRef.current.clientHeight;
      rendererRef.current.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', handleResize);

    const animate = () => {
      requestAnimationFrame(animate);
      if (controlsRef.current) {
        controlsRef.current.autoRotate = rotatingRef.current;
        controlsRef.current.update();
      }
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      window.removeEventListener('resize', handleResize);
      if (mountRef.current?.contains(renderer.domElement)) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
      material.dispose();
      cameraRef.current = undefined;
      if (pointTextureRef.current) {
        pointTextureRef.current.dispose();
        pointTextureRef.current = null;
      }
      controls.dispose();
    };
  }, []);

  useEffect(() => {
    if (gridRef.current) {
      gridRef.current.visible = showGuides;
    }
    if (axesRef.current) {
      axesRef.current.visible = showGuides;
    }
  }, [showGuides]);

  useEffect(() => {
    rotatingRef.current = isRotating;
    if (controlsRef.current) {
      controlsRef.current.autoRotate = isRotating;
    }
  }, [isRotating]);

  // Update point cloud when data changes
  useEffect(() => {
    if (!sceneRef.current || !materialRef.current || !pointsBuffer) return;

    if (cloudRef.current) {
      sceneRef.current.remove(cloudRef.current);
      cloudRef.current.geometry.dispose();
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(pointsBuffer, 3));

    const colors = new Float32Array(pointsBuffer.length);
    const maxR = currentExtent || 1;
    for (let i = 0; i < pointsBuffer.length; i += 3) {
      const x = pointsBuffer[i];
      const y = pointsBuffer[i + 1];
      const z = pointsBuffer[i + 2];
      const r = Math.sqrt(x * x + y * y + z * z);
      const t = clamp(r / (maxR * 1.2), 0, 1);
      const shade = 0.9 - 0.45 * t;
      let shellIndex = 0;
      for (let j = 0; j < radialNodeRadii.length; j++) {
        if (r >= radialNodeRadii[j]) {
          shellIndex += 1;
        } else {
          break;
        }
      }
      const baseColor = useShellColors
        ? (shellIndex % 2 === 0 ? shellRgbA : shellRgbB)
        : cloudRgb;
      colors[i] = shade * baseColor.r;
      colors[i + 1] = shade * baseColor.g;
      colors[i + 2] = shade * baseColor.b;
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const points = new THREE.Points(geometry, materialRef.current);
    cloudRef.current = points;
    sceneRef.current.add(points);

    materialRef.current.size = pointSize;
    materialRef.current.opacity = opacity;
  }, [
    pointsBuffer,
    pointSize,
    opacity,
    currentExtent,
    cloudRgb,
    shellRgbA,
    shellRgbB,
    useShellColors,
    n,
    l,
    atom.Z,
    radialNodeRadii,
  ]);

  const regenerate = async (targetQubits: number = qubits) => {
    const capped = allowHighQubits ? targetQubits : Math.min(targetQubits, SAFE_QUBITS);
    const effectiveQubits = Math.min(capped, MAX_QUBITS_RUNTIME);
    setIsGenerating(true);
    let extent = extentForAtom(n, atom.Z);
    try {
      await orbitalPreload;
      const targetDim = Math.pow(2, effectiveQubits);
      const gridSide = Math.min(MAX_GRID, Math.max(MIN_GRID, Math.floor(Math.cbrt(targetDim))));
      const physics = {
        screeningExchange: useScreeningExchange,
        relativisticSpinOrbit: useRelativisticSpinOrbit,
        correlationMixing: useCorrelationMixing,
      };
      const adaptiveExtentEnabled =
        physics.screeningExchange || physics.relativisticSpinOrbit || physics.correlationMixing;
      if (adaptiveExtentEnabled) {
        extent = estimateSpatialExtent(atom, n, l, physics);
      }

      let grid: ProbabilityGrid | null = null;
      let boundaryMass = 0;
      for (let pass = 0; pass < ADAPTIVE_EXTENT_MAX_PASSES; pass++) {
        grid = await buildProbabilityGrid({
          atom,
          n,
          l,
          m,
          pointCount,
          extent,
          gridSize: gridSide,
          physics,
        });
        if (!adaptiveExtentEnabled) break;

        boundaryMass = boundaryMassFraction(grid.probabilities, gridSide);
        if (boundaryMass <= ADAPTIVE_EXTENT_BOUNDARY_TARGET || extent >= ADAPTIVE_EXTENT_MAX) {
          break;
        }
        extent = Math.min(ADAPTIVE_EXTENT_MAX, extent * ADAPTIVE_EXTENT_GROWTH_FACTOR);
      }
      if (!grid) {
        throw new Error('Failed to build probability grid');
      }

      if (adaptiveExtentEnabled && boundaryMass > ADAPTIVE_EXTENT_BOUNDARY_TARGET) {
        console.warn(
          `[orbitals] Boundary mass remained high after extent adaptation (${boundaryMass.toExponential(3)} at extent=${extent.toFixed(2)})`
        );
      }

      setCurrentExtent(extent);
      if (cameraRef.current && controlsRef.current) {
        fitCameraToExtent(cameraRef.current, controlsRef.current, extent);
      }
      const spacing = (extent * 2) / (gridSide - 1);
      const gridSizeWorld = spacing * (gridSide - 1);

      const baseLen = grid.probabilities.length;
      const solverDim = targetDim <= baseLen ? targetDim : nextPowerOfTwo(baseLen);
      const solverQubits = Math.round(Math.log2(solverDim));
      if (solverDim !== targetDim) {
        console.info(
          `[orbitals] Capping solver dimension from 2^${effectiveQubits} (${targetDim.toLocaleString()}) to ${solverDim.toLocaleString()} (grid ${gridSide}^3)`
        );
      }

      const aligned = alignToDimension(grid.probabilities, grid.positions, solverDim);

      const dmrgInfluence = useDmrg && dmrgWeights && dmrgWeights.length > 0 ? dmrgWeights : null;
      const dmrgSampleWeight = dmrgInfluence
        ? buildDmrgInfluenceSampler(dmrgInfluence, gridSide)
        : (() => 1);

      const amplitudes = new Float64Array(solverDim * 2);
      const expectedProbs = new Float64Array(solverDim);
      let total = 0;
      for (let i = 0; i < solverDim; i++) {
        const baseProb = aligned.probs[i];
        if (baseProb <= 0) continue;
        total += baseProb * dmrgSampleWeight(i);
      }
      const safeTotal = total > 0 ? total : 1;
      const norm = Math.sqrt(safeTotal);
      for (let i = 0; i < solverDim; i++) {
        const baseProb = aligned.probs[i];
        if (baseProb <= 0) {
          expectedProbs[i] = 0;
          amplitudes[i * 2] = 0;
          amplitudes[i * 2 + 1] = 0;
          continue;
        }
        const prob = baseProb * dmrgSampleWeight(i);
        expectedProbs[i] = prob / safeTotal;
        const amp = prob > 0 ? Math.sqrt(prob) / norm : 0;
        amplitudes[i * 2] = amp;
        amplitudes[i * 2 + 1] = 0;
      }

      const moonlabProbs = await probabilitiesFromAmplitudes({
        numQubits: solverQubits,
        amplitudes,
      });

      let l1Drift = 0;
      for (let i = 0; i < solverDim; i++) {
        l1Drift += Math.abs(moonlabProbs[i] - expectedProbs[i]);
      }

      const probsForSampling =
        Number.isFinite(l1Drift) && l1Drift <= PROBABILITY_DRIFT_FALLBACK_L1
          ? moonlabProbs
          : expectedProbs;
      if (probsForSampling !== moonlabProbs) {
        console.warn(
          `[orbitals] Probability drift too high (L1=${l1Drift.toExponential(3)}); using analytic fallback`
        );
      }

      // Always sample from the physical base lattice (gridSide^3). High-qubit runs
      // use padded state dimensions, and sampling those padded entries reintroduces
      // voxel aliasing artifacts.
      const probsForSamplingGrid = new Float64Array(baseLen);
      let gridMass = 0;
      for (let i = 0; i < baseLen; i++) {
        const p = probsForSampling[i] ?? 0;
        probsForSamplingGrid[i] = p;
        gridMass += p;
      }
      if (gridMass > 0) {
        for (let i = 0; i < baseLen; i++) {
          probsForSamplingGrid[i] /= gridMass;
        }
      } else {
        // Ultimate fallback: analytic base-grid probabilities are guaranteed non-negative.
        for (let i = 0; i < baseLen; i++) {
          probsForSamplingGrid[i] = grid.probabilities[i];
        }
      }

      const paddedMass = Math.max(0, 1 - gridMass);
      if (paddedMass > 1e-5) {
        console.warn(
          `[orbitals] Probability leakage into padded states: ${paddedMass.toExponential(3)}`
        );
      }

      const sampled = samplePoints(probsForSamplingGrid, grid.positions, pointCount, extent, gridSide);
      setPointsBuffer(sampled);
      setLastStats({ elapsedMs: grid.elapsedMs, generated: pointCount });

    // Update grid helper to match current lattice
    if (sceneRef.current) {
      if (gridRef.current) {
        sceneRef.current.remove(gridRef.current);
      }
      const newGrid = new THREE.GridHelper(gridSizeWorld, gridSide - 1, GRID_COLOR, GRID_COLOR);
      newGrid.position.y = -extent;
      newGrid.visible = showGuides;
      sceneRef.current.add(newGrid);
      gridRef.current = newGrid;
    }

      // Update axes position/size relative to extent
    if (sceneRef.current && axesRef.current) {
      axesRef.current.position.set(0, -extent, 0);
      axesRef.current.scale.setScalar(Math.max(1, spacing * gridSide * 0.25));
    }

      // Update nucleus scale relative to cloud extent
      if (nucleusRef.current) {
        nucleusRef.current.scale.setScalar(nucleusRadiusForExtent(extent));
      }
    } catch (err) {
      console.error('Failed to build orbital cloud', err);
    } finally {
      setIsGenerating(false);
    }
  };

  useEffect(() => {
    void regenerate(qubits);
  }, [
    atom,
    n,
    l,
    m,
    pointCount,
    qubits,
    useDmrg,
    dmrgWeights,
    useScreeningExchange,
    useRelativisticSpinOrbit,
    useCorrelationMixing,
  ]);

  const isInitialLoad = pointsBuffer === null;
  const showOverlay = isInitialLoad || isGenerating || (useDmrg && isDmrgRunning);
  const overlayLabel = isInitialLoad
    ? 'Initializing orbital cloud…'
    : isGenerating
      ? 'Building orbital…'
      : 'Solving Schrödinger (DMRG)…';
  const activeAtomicCorrections = [
    useScreeningExchange ? 'screening/exchange' : '',
    useRelativisticSpinOrbit ? 'relativistic/spin-orbit' : '',
    useCorrelationMixing ? 'correlation/mixing' : '',
  ]
    .filter(Boolean)
    .join(' • ');

  return (
    <div className="orbital-page" style={pageStyle}>
      <div className="orbital-viewport" ref={mountRef}></div>

      <div className={`orbital-controls ${isCollapsed ? 'collapsed' : ''}`}>
        <div className="controls-header">
          <div className="controls-header-left">
            <div className="pill">Schrödinger • Three.js • Moonlab</div>
            <h1 className="section-title">Schrodinger Sim</h1>
          </div>
          <button
            className="collapse-btn"
            onClick={() => setIsCollapsed((prev) => !prev)}
            title={isCollapsed ? 'Expand controls' : 'Collapse controls'}
          >
            {isCollapsed ? '+' : '−'}
          </button>
        </div>

        <div className="controls-body">
          <div className="control-grid">
            <div className="control">
              <span title={CONTROL_TOOLTIPS.atom}>Select Atom</span>
              <div className="atom-row">
                <div className="atom-chip" title={`${atom.name}. ${CONTROL_TOOLTIPS.atom}`}>
                  <span className="atom-symbol">{atom.symbol}</span>
                  <span className="atom-name">{atom.name}</span>
                  <span className="atom-z">Z = {atom.Z}</span>
                </div>
                <button
                  className="btn btn-secondary"
                  type="button"
                  onClick={() => setIsPickerOpen(true)}
                  title={CONTROL_TOOLTIPS.chooseElement}
                >
                  Choose Element
                </button>
                <button
                  className="btn btn-secondary"
                  type="button"
                  onClick={randomizeQuantumState}
                  title={CONTROL_TOOLTIPS.randomQuantumState}
                >
                  Random Element + n/l/m
                </button>
              </div>
            </div>

            <label className="control">
              <span title={CONTROL_TOOLTIPS.n}>Principal Quantum Number (n)</span>
              <input
                type="range"
                min={1}
                max={MAX_N_UI}
                value={n}
                title={CONTROL_TOOLTIPS.n}
                onChange={(e) => {
                  const next = Number(e.target.value);
                  setN(next);
                  setL((prev) => clamp(prev, 0, next - 1));
                }}
              />
              <div className="control-value">n = {n}</div>
            </label>

            <div className="control-group">
              <label className="control checkbox-control">
                <input
                  type="checkbox"
                  checked={useScreeningExchange}
                  title={CONTROL_TOOLTIPS.screeningExchange}
                  onChange={(e) => setUseScreeningExchange(e.target.checked)}
                />
                <span title={CONTROL_TOOLTIPS.screeningExchange}>Screening + exchange correction</span>
              </label>

              <label className="control checkbox-control">
                <input
                  type="checkbox"
                  checked={useRelativisticSpinOrbit}
                  title={CONTROL_TOOLTIPS.relativisticSpinOrbit}
                  onChange={(e) => setUseRelativisticSpinOrbit(e.target.checked)}
                />
                <span title={CONTROL_TOOLTIPS.relativisticSpinOrbit}>
                  Relativistic + spin-orbit correction
                </span>
              </label>

              <label className="control checkbox-control">
                <input
                  type="checkbox"
                  checked={useCorrelationMixing}
                  title={CONTROL_TOOLTIPS.correlationMixing}
                  onChange={(e) => setUseCorrelationMixing(e.target.checked)}
                />
                <span title={CONTROL_TOOLTIPS.correlationMixing}>Correlation + configuration mixing</span>
              </label>
            </div>

            <label className="control">
              <span title={CONTROL_TOOLTIPS.qubits}>WASM Qubits (min 4, UI max 32)</span>
              <input
                type="range"
                min={4}
                max={allowHighQubits ? MAX_QUBITS_UI : SAFE_QUBITS}
                value={Math.min(qubits, allowHighQubits ? MAX_QUBITS_UI : SAFE_QUBITS)}
                title={CONTROL_TOOLTIPS.qubits}
                onChange={(e) => setQubits(Number(e.target.value))}
              />
              <div className="control-value">
                {qubits} requested • effective ≤ {MAX_QUBITS_RUNTIME} (state dim up to 2^{MAX_QUBITS_RUNTIME})
              </div>
            </label>

            <label className="control checkbox-control">
              <input
                type="checkbox"
                checked={allowHighQubits}
                title={CONTROL_TOOLTIPS.allowHighQubits}
                onChange={(e) => {
                  setAllowHighQubits(e.target.checked);
                  if (!e.target.checked && qubits > SAFE_QUBITS) {
                    setQubits(SAFE_QUBITS);
                  }
                }}
              />
              <span title={CONTROL_TOOLTIPS.allowHighQubits}>I have ≥64 GB RAM (enable 25–32 qubits)</span>
            </label>

            <div className="control-group">
              <label className="control checkbox-control">
                <input
                  type="checkbox"
                  checked={useDmrg}
                  title={CONTROL_TOOLTIPS.useDmrg}
                  onChange={(e) => setUseDmrg(e.target.checked)}
                />
                <span title={CONTROL_TOOLTIPS.useDmrg}>Use DMRG solver (TFIM ground state)</span>
              </label>

              {useDmrg && (
                <>
                  <label className="control">
                    <span title={CONTROL_TOOLTIPS.dmrgSites}>DMRG Chain Length</span>
                    <input
                      type="range"
                      min={DMRG_MIN_SITES}
                      max={DMRG_MAX_SITES}
                      value={dmrgSites}
                      title={CONTROL_TOOLTIPS.dmrgSites}
                      onChange={(e) => setDmrgSites(Number(e.target.value))}
                    />
                    <div className="control-value">{dmrgSites} sites</div>
                  </label>

                  <label className="control">
                    <span title={CONTROL_TOOLTIPS.dmrgG}>TFIM Field Ratio (g)</span>
                    <input
                      type="range"
                      min={0.2}
                      max={2.5}
                      step={0.05}
                      value={dmrgG}
                      title={CONTROL_TOOLTIPS.dmrgG}
                      onChange={(e) => setDmrgG(Number(e.target.value))}
                    />
                    <div className="control-value">g = {dmrgG.toFixed(2)}</div>
                  </label>
                </>
              )}
            </div>

            <label className="control checkbox-control">
              <input
                type="checkbox"
                checked={showGuides}
                title={CONTROL_TOOLTIPS.guides}
                onChange={(e) => setShowGuides(e.target.checked)}
              />
              <span title={CONTROL_TOOLTIPS.guides}>Show cartesian grid + axes</span>
            </label>

            <label className="control checkbox-control">
              <input
                type="checkbox"
                checked={showBackground}
                title={CONTROL_TOOLTIPS.background}
                onChange={(e) => setShowBackground(e.target.checked)}
              />
              <span title={CONTROL_TOOLTIPS.background}>Show moon background</span>
            </label>

            <label className="control">
              <span title={CONTROL_TOOLTIPS.l}>Angular Momentum (l)</span>
              <select value={l} onChange={(e) => setL(Number(e.target.value))} title={CONTROL_TOOLTIPS.l}>
                {lOptions.map((val) => (
                  <option key={val} value={val}>
                    l = {val} ({L_LABELS[val] || 'higher'})
                  </option>
                ))}
              </select>
            </label>

            <label className="control">
              <span title={CONTROL_TOOLTIPS.m}>Magnetic Quantum Number (m)</span>
              <select value={m} onChange={(e) => setM(Number(e.target.value))} title={CONTROL_TOOLTIPS.m}>
                {mOptions.map((val) => (
                  <option key={val} value={val}>
                    m = {val}
                  </option>
                ))}
              </select>
            </label>

            <label className="control">
              <span title={CONTROL_TOOLTIPS.pointCount}>Point Density</span>
              <input
                type="range"
                min={4000}
                max={1000000}
                step={1000}
                value={pointCount}
                title={CONTROL_TOOLTIPS.pointCount}
                onChange={(e) => setPointCount(Number(e.target.value))}
              />
              <div className="control-value">{pointCount.toLocaleString()} samples</div>
            </label>

            <label className="control">
              <span title={CONTROL_TOOLTIPS.pointSize}>Point Size</span>
              <input
                type="range"
                min={POINT_SIZE_MIN_UI}
                max={POINT_SIZE_MAX_UI}
                step={0.005}
                value={pointSize}
                title={CONTROL_TOOLTIPS.pointSize}
                onChange={(e) => setPointSize(Number(e.target.value))}
              />
              <div className="control-value">{pointSize.toFixed(3)}</div>
            </label>

            <label className="control">
              <span title={CONTROL_TOOLTIPS.opacity}>Opacity</span>
              <input
                type="range"
                min={OPACITY_MIN_UI}
                max={OPACITY_MAX_UI}
                step={0.05}
                value={opacity}
                title={CONTROL_TOOLTIPS.opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
              />
              <div className="control-value">{opacity.toFixed(2)}</div>
            </label>

            <div className="control-group">
              <label className="control checkbox-control">
                <input
                  type="checkbox"
                  checked={useShellColors}
                  title={CONTROL_TOOLTIPS.shellColors}
                  onChange={(e) => setUseShellColors(e.target.checked)}
                />
                <span title={CONTROL_TOOLTIPS.shellColors}>Alternate shell colors</span>
              </label>

              {useShellColors ? (
                <label className="control color-control">
                  <span title={CONTROL_TOOLTIPS.shellColorA}>Shell Colors</span>
                  <div className="color-row">
                    <input
                      type="color"
                      value={shellColorA}
                      title={CONTROL_TOOLTIPS.shellColorA}
                      onChange={(e) => setShellColorA(e.target.value)}
                    />
                    <input
                      type="color"
                      value={shellColorB}
                      title={CONTROL_TOOLTIPS.shellColorB}
                      onChange={(e) => setShellColorB(e.target.value)}
                    />
                    <div className="control-value">
                      {shellColorA.toUpperCase()} / {shellColorB.toUpperCase()}
                    </div>
                  </div>
                </label>
              ) : (
                <label className="control color-control">
                  <span title={CONTROL_TOOLTIPS.cloudColor}>Cloud Color</span>
                  <div className="color-row">
                    <input
                      type="color"
                      value={cloudColor}
                      title={CONTROL_TOOLTIPS.cloudColor}
                      onChange={(e) => setCloudColor(e.target.value)}
                    />
                    <div className="control-value">{cloudColor.toUpperCase()}</div>
                  </div>
                </label>
              )}
            </div>
          </div>

          <div className="button-row">
            <button
              className="btn btn-primary"
              onClick={() => void regenerate()}
              disabled={isGenerating}
              title={CONTROL_TOOLTIPS.regenerate}
            >
              {isGenerating ? 'Generating…' : 'Regenerate Cloud'}
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => setIsRotating((prev) => !prev)}
              title={CONTROL_TOOLTIPS.rotation}
            >
              {isRotating ? 'Pause Rotation' : 'Resume Rotation'}
            </button>
          </div>

          <div className="stats">
            <div>
              <div className="stat-label">Energy Level</div>
              <div className="stat-value">
                n = {n}, l = {l} ({L_LABELS[l] || 'g+'}), m = {m}
              </div>
            </div>
            <div>
              <div className="stat-label">Atom</div>
              <div className="stat-value">
                {atom.name} (Z = {atom.Z})
              </div>
            </div>
            <div>
              <div className="stat-label">Atomic Model</div>
              <div className="stat-value">
                {activeAtomicCorrections || 'hydrogenic baseline'}
              </div>
            </div>
            <div>
              <div className="stat-label">Computation</div>
              <div className="stat-value">
                {lastStats
                  ? `${Math.round(lastStats.elapsedMs)} ms • ${lastStats.generated.toLocaleString()} points`
                  : 'waiting for first run'}
              </div>
            </div>
            <div>
              <div className="stat-label">DMRG Solver</div>
              <div className="stat-value">
                {useDmrg
                  ? isDmrgRunning
                    ? 'solving TFIM ground state…'
                    : dmrgStats
                      ? `sites ${dmrgStats.sites} • E0 ≈ ${
                          dmrgStats.energy !== undefined ? dmrgStats.energy.toFixed(4) : 'n/a'
                        } • var ${
                          dmrgStats.variance !== undefined ? dmrgStats.variance.toFixed(4) : 'n/a'
                        } • ${Math.round(dmrgStats.elapsedMs)} ms`
                      : 'waiting for first solve'
                  : 'disabled'}
              </div>
            </div>
          </div>

          <div className="about-card">
            <h3>Schrödinger Solver (DMRG)</h3>
            <p>
              The TFIM chain is solved in WASM using the tensor-network DMRG solver to compute a ground-state wavefunction.
              Its probability distribution modulates the orbital sampling when enabled.
            </p>
            <ul>
              <li>Status: {useDmrg ? (isDmrgRunning ? 'solving…' : 'ready') : 'disabled'}</li>
              <li>Chain: {dmrgSites} sites (g = {dmrgG.toFixed(2)})</li>
              <li>
                Energy: {dmrgStats?.energy !== undefined ? dmrgStats.energy.toFixed(6) : 'n/a'} • Variance:{' '}
                {dmrgStats?.variance !== undefined ? dmrgStats.variance.toFixed(6) : 'n/a'}
              </li>
            </ul>
          </div>

            <div className="about-card">
              <h3>About this simulation</h3>
              <p>
                We discretize an adaptive 3D lattice (scales with qubits up to 32³) and compute hydrogen-like orbital
                amplitudes analytically on the CPU. Moonlab then builds a quantum state vector in WASM, normalizes it,
                and samples measurement probabilities to drive the point cloud. If DMRG is enabled, Moonlab’s tensor-network
                solver runs a TFIM ground-state calculation (a Schrödinger eigenproblem for a 1D spin chain) and uses that
                quantum state’s probabilities to modulate the orbital distribution. Optional checkboxes apply additional
                screening/exchange, relativistic/spin-orbit, and correlation/mixing approximations on top of the
                hydrogenic baseline. Three.js renders the resulting |ψ|² density with orbit controls.
              </p>
              <ul>
                <li>Adjust n, l, m to see shapes evolve (s, p, d, f…)</li>
                <li>Atom choice changes nuclear charge (Z) and radial decay</li>
                <li>Point density / size / opacity balance fidelity and performance</li>
            </ul>
          </div>
        </div>
      </div>

      <ElementPicker
        elements={ELEMENTS}
        isOpen={isPickerOpen}
        onClose={() => setIsPickerOpen(false)}
        onSelect={(el) => setAtom(el)}
        selected={atom}
      />

      {showOverlay && (
        <div className="overlay">
          <div className="overlay-content">
            <img
              className="loading-gif"
              src={`${import.meta.env.BASE_URL}moonlab_glitch.gif`}
              alt="Moonlab loading animation"
            />
            <div className="overlay-text">{overlayLabel}</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default OrbitalDemo;
