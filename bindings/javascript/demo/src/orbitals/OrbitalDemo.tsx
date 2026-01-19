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
const MAX_GRID = 32;
const MIN_GRID = 4;
const BASE_GRID = 32;
const BASE_STATES = BASE_GRID * BASE_GRID * BASE_GRID;
const DEFAULT_QUBITS = Math.ceil(Math.log2(BASE_STATES)); // 15 qubits
const MAX_QUBITS_UI = 32;
const SAFE_QUBITS = 24;
const MAX_QUBITS_RUNTIME = 32; // cap at 2^32 amplitudes (use with care)
const DMRG_MIN_SITES = 4;
const DMRG_MAX_SITES = 12;
const DEFAULT_DMRG_SITES = 6;
const DEFAULT_ATOM = ELEMENTS.find((e) => e.symbol === 'Fe') || ELEMENTS[0];
const DEFAULT_N = 5;
const DEFAULT_L = 2;
const DEFAULT_M = 2;
const DEFAULT_POINT_COUNT = 494000;
const DEFAULT_POINT_SIZE = 0.01;
const DEFAULT_OPACITY = 0.15;

const CONTROL_TOOLTIPS = {
  atom: 'Choose the element (atomic number Z). Higher Z pulls the cloud inward and increases radial decay.',
  chooseElement: 'Open the periodic table to select a different element.',
  n: 'Sets the principal quantum number; higher n increases orbital size and radial nodes.',
  qubits: 'Controls WASM state size; more qubits increase lattice resolution and memory/time cost.',
  allowHighQubits: 'Unlocks 25-32 qubits; requires very high memory (64 GB+).',
  useDmrg: 'Run TFIM ground-state solver in WASM to modulate orbital sampling.',
  dmrgSites: 'Number of sites in the TFIM chain; more sites cost more compute.',
  dmrgG: 'Transverse field ratio g/J; shifts the TFIM ground state.',
  l: 'Orbital shape selector (s, p, d, f, ...).',
  m: 'Orbital orientation (magnetic quantum number).',
  pointCount: 'Number of sampled points; higher values yield a denser cloud.',
  pointSize: 'Rendered size of each point sprite.',
  opacity: 'Cloud transparency; lower values are more transparent.',
  regenerate: 'Recompute the orbital cloud with current settings.',
  rotation: 'Toggle auto-rotation of the scene.',
};

interface CloudParams {
  atom: Atom;
  n: number;
  l: number;
  m: number;
  pointCount: number;
  extent: number;
  gridSize: number;
}

interface ProbabilityGrid {
  positions: { x: number; y: number; z: number }[];
  probabilities: number[];
  elapsedMs: number;
}

const clamp = (value: number, min: number, max: number) =>
  Math.min(Math.max(value, min), max);

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

const buildProbabilityGrid = async (params: CloudParams): Promise<ProbabilityGrid> => {
  const start = performance.now();
  const stateCount = params.gridSize * params.gridSize * params.gridSize;
  const positions: { x: number; y: number; z: number }[] = new Array(stateCount);
  const amplitudes: Complex[] = new Array(stateCount);
  const probabilities: number[] = new Array(stateCount);

  const spacing = (params.extent * 2) / (params.gridSize - 1);

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

    const radial = radialComponent(params.n, params.l, Math.max(r, 1e-5), params.atom.Z);
    const ylm = realSphericalHarmonic(params.l, params.m, theta, phi);
    const prob = Math.max(0, radial * radial * ylm * ylm);

    positions[idx] = { x: posX, y: posY, z: posZ };
    probabilities[idx] = prob;
    totalProb += prob;
  }

  const norm = totalProb > 0 ? Math.sqrt(totalProb) : 1;
  for (let i = 0; i < stateCount; i++) {
    const amp = probabilities[i] > 0 ? Math.sqrt(probabilities[i]) / norm : 0;
    amplitudes[i] = { real: amp, imag: 0 };
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

const samplePoints = (
  probabilities: ArrayLike<number>,
  positions: { x: number; y: number; z: number }[],
  count: number,
  extent: number,
  gridSize: number
): Float32Array => {
  const cdf = new Float64Array(probabilities.length);
  let total = 0;
  for (let i = 0; i < probabilities.length; i++) {
    total += probabilities[i];
    cdf[i] = total;
  }

  const points = new Float32Array(count * 3);
  if (total <= 0) {
    for (let i = 0; i < count; i++) {
      points[i * 3] = (Math.random() - 0.5) * 2;
      points[i * 3 + 1] = (Math.random() - 0.5) * 2;
      points[i * 3 + 2] = (Math.random() - 0.5) * 2;
    }
    return points;
  }

  // Calculate jitter amount based on grid spacing
  // Use Gaussian jitter with stddev = spacing to smooth grid artifacts
  const spacing = (extent * 2) / (gridSize - 1);

  // Box-Muller transform for Gaussian random numbers
  const gaussianRandom = () => {
    const u1 = Math.random();
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
    // Add Gaussian jitter to create smooth distribution around grid points
    points[i * 3] = pos.x + gaussianRandom() * spacing;
    points[i * 3 + 1] = pos.y + gaussianRandom() * spacing;
    points[i * 3 + 2] = pos.z + gaussianRandom() * spacing;
  }

  return points;
};

const extentForAtom = (n: number, Z: number) => {
  const base = 3 + n * 0.7;
  const compression = Math.max(1, Z / 4);
  return base / compression;
};

const nucleusRadiusForExtent = (extent: number) => clamp(extent * 0.003, 0.006, 0.02);

const OrbitalDemo: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const cloudRef = useRef<THREE.Points>();
  const sceneRef = useRef<THREE.Scene>();
  const rendererRef = useRef<THREE.WebGLRenderer>();
  const materialRef = useRef<THREE.PointsMaterial>();
  const controlsRef = useRef<OrbitControls>();
  const gridRef = useRef<THREE.GridHelper | null>(null);
  const axesRef = useRef<THREE.AxesHelper | null>(null);
  const nucleusRef = useRef<THREE.Mesh | null>(null);
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
  const [isRotating, setIsRotating] = useState<boolean>(true);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [lastStats, setLastStats] = useState<{ elapsedMs: number; generated: number } | null>(null);
  const [useDmrg, setUseDmrg] = useState<boolean>(true);
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
  const dmrgRunId = useRef(0);

  const lOptions = useMemo(() => Array.from({ length: n }, (_, i) => i), [n]);
  const mOptions = useMemo(() => Array.from({ length: l * 2 + 1 }, (_, i) => i - l), [l]);

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
    scene.background = new THREE.Color('#050716');

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(6, 6, 6);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mountRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.8;

    const ambient = new THREE.AmbientLight('#88aaff', 0.4);
    scene.add(ambient);
    const dir = new THREE.DirectionalLight('#aaddff', 0.6);
    dir.position.set(4, 5, 2);
    scene.add(dir);

    const initialExtent = extentForAtom(n, atom.Z);
    const initialDim = Math.pow(2, Math.min(Math.max(8, DEFAULT_QUBITS), MAX_QUBITS_RUNTIME));
    const initialGridSide = Math.min(MAX_GRID, Math.max(MIN_GRID, Math.floor(Math.cbrt(initialDim))));
    const initialSpacing = (initialExtent * 2) / (initialGridSide - 1);
    const initialSize = initialSpacing * (initialGridSide - 1);

    const nucleus = new THREE.Mesh(
      new THREE.SphereGeometry(1, 32, 32),
      new THREE.MeshBasicMaterial({ color: '#ff7ac3' })
    );
    const nucleusRadius = nucleusRadiusForExtent(initialExtent);
    nucleus.scale.setScalar(nucleusRadius);
    scene.add(nucleus);
    nucleusRef.current = nucleus;

    const grid = new THREE.GridHelper(initialSize, initialGridSide - 1, '#0b3b5a', '#0b3b5a');
    grid.position.y = -initialExtent;
    scene.add(grid);
    gridRef.current = grid;

    const axes = new THREE.AxesHelper(initialExtent * 1.2);
    axes.position.set(0, -initialExtent, 0);
    scene.add(axes);
    axesRef.current = axes;

    const material = new THREE.PointsMaterial({
      size: pointSize,
      sizeAttenuation: true,
      transparent: true,
      opacity,
      vertexColors: true,
      depthWrite: false,
    });

    sceneRef.current = scene;
    rendererRef.current = renderer;
    controlsRef.current = controls;
    materialRef.current = material;

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
      controls.dispose();
    };
  }, []);

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
    const color = new THREE.Color();
    const maxR = currentExtent || 1;
    for (let i = 0; i < pointsBuffer.length; i += 3) {
      const x = pointsBuffer[i];
      const y = pointsBuffer[i + 1];
      const z = pointsBuffer[i + 2];
      const r = Math.sqrt(x * x + y * y + z * z);
      const t = clamp(r / (maxR * 1.2), 0, 1);
      color.setHSL(0.62 - 0.3 * t, 0.85, 0.55 + 0.2 * (1 - t));
      colors[i] = color.r;
      colors[i + 1] = color.g;
      colors[i + 2] = color.b;
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const points = new THREE.Points(geometry, materialRef.current);
    cloudRef.current = points;
    sceneRef.current.add(points);

    materialRef.current.size = pointSize;
    materialRef.current.opacity = opacity;
  }, [pointsBuffer, pointSize, opacity, currentExtent]);

  const regenerate = async (targetQubits: number = qubits) => {
    const capped = allowHighQubits ? targetQubits : Math.min(targetQubits, SAFE_QUBITS);
    const effectiveQubits = Math.min(capped, MAX_QUBITS_RUNTIME);
    setIsGenerating(true);
    const extent = extentForAtom(n, atom.Z);
    setCurrentExtent(extent);
    try {
      await orbitalPreload;
      const targetDim = Math.pow(2, effectiveQubits);
      const gridSide = Math.min(MAX_GRID, Math.max(MIN_GRID, Math.floor(Math.cbrt(targetDim))));
      const spacing = (extent * 2) / (gridSide - 1);
      const gridSizeWorld = spacing * (gridSide - 1);

      const grid = await buildProbabilityGrid({
        atom,
        n,
        l,
        m,
        pointCount,
        extent,
        gridSize: gridSide,
      });
      const aligned = alignToDimension(grid.probabilities, grid.positions, targetDim);

      const dmrgInfluence = useDmrg && dmrgWeights && dmrgWeights.length > 0 ? dmrgWeights : null;

      const amplitudes = new Float64Array(targetDim * 2);
      let total = 0;
      for (let i = 0; i < targetDim; i++) {
        const weight = dmrgInfluence ? dmrgInfluence[i % dmrgInfluence.length] : 1;
        total += aligned.probs[i] * weight;
      }
      const norm = total > 0 ? Math.sqrt(total) : 1;
      for (let i = 0; i < targetDim; i++) {
        const weight = dmrgInfluence ? dmrgInfluence[i % dmrgInfluence.length] : 1;
        const prob = aligned.probs[i] * weight;
        const amp = prob > 0 ? Math.sqrt(prob) / norm : 0;
        amplitudes[i * 2] = amp;
        amplitudes[i * 2 + 1] = 0;
      }

      const probs = await probabilitiesFromAmplitudes({
        numQubits: effectiveQubits,
        amplitudes,
      });

      const sampled = samplePoints(probs, aligned.pos, pointCount, extent, gridSide);
      setPointsBuffer(sampled);
      setLastStats({ elapsedMs: grid.elapsedMs, generated: pointCount });

      // Update grid helper to match current lattice
      if (sceneRef.current) {
        if (gridRef.current) {
          sceneRef.current.remove(gridRef.current);
        }
        const newGrid = new THREE.GridHelper(gridSizeWorld, gridSide - 1, '#0b3b5a', '#0b3b5a');
        newGrid.position.y = -extent;
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
  }, [atom, n, l, m, pointCount, qubits, useDmrg, dmrgWeights]);

  const isInitialLoad = pointsBuffer === null;
  const showOverlay = isInitialLoad || isGenerating || (useDmrg && isDmrgRunning);
  const overlayLabel = isInitialLoad
    ? 'Initializing orbital cloud…'
    : isGenerating
      ? 'Building orbital…'
      : 'Solving Schrödinger (DMRG)…';

  return (
    <div className="orbital-page">
      <div className="orbital-viewport" ref={mountRef}>
        {showOverlay && (
          <div className="overlay">
            <div className="overlay-content">
              <div className="quantum-spinner overlay-spinner" aria-hidden="true">
                <div className="spinner-ring"></div>
                <div className="spinner-core"></div>
              </div>
              <div className="overlay-text">{overlayLabel}</div>
            </div>
          </div>
        )}
      </div>

      <div className={`orbital-controls ${isCollapsed ? 'collapsed' : ''}`}>
        <div className="controls-header">
          <div className="controls-header-left">
            <div className="pill">Schrödinger • Three.js • Moonlab</div>
            <h1 className="section-title">Quantum Orbital Explorer</h1>
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
              </div>
            </div>

            <label className="control">
              <span title={CONTROL_TOOLTIPS.n}>Principal Quantum Number (n)</span>
              <input
                type="range"
                min={1}
                max={5}
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

            <label className="control checkbox-control">
              <input
                type="checkbox"
                checked={useDmrg}
                title={CONTROL_TOOLTIPS.useDmrg}
                onChange={(e) => setUseDmrg(e.target.checked)}
              />
              <span title={CONTROL_TOOLTIPS.useDmrg}>Use DMRG solver (TFIM ground state)</span>
            </label>

            <label className="control">
              <span title={CONTROL_TOOLTIPS.dmrgSites}>DMRG Chain Length</span>
              <input
                type="range"
                min={DMRG_MIN_SITES}
                max={DMRG_MAX_SITES}
                value={dmrgSites}
                title={CONTROL_TOOLTIPS.dmrgSites}
                onChange={(e) => setDmrgSites(Number(e.target.value))}
                disabled={!useDmrg}
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
                disabled={!useDmrg}
              />
              <div className="control-value">g = {dmrgG.toFixed(2)}</div>
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
                min={0.01}
                max={0.15}
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
                min={0.1}
                max={0.9}
                step={0.05}
                value={opacity}
                title={CONTROL_TOOLTIPS.opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
              />
              <div className="control-value">{opacity.toFixed(2)}</div>
            </label>
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
                quantum state’s probabilities to modulate the orbital distribution. Three.js renders the resulting |ψ|²
                density with orbit controls.
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
    </div>
  );
};

export default OrbitalDemo;
