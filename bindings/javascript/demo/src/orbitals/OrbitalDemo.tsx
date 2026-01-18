import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { QuantumState, preload, type Complex } from '@moonlab/quantum-core';
import { ElementPicker } from './ElementPicker';
import { ELEMENTS, type Atom } from './elements';
import './Orbitals.css';

// Preload Moonlab once per module load (avoids React strict-mode double effects)
const orbitalPreload =
  typeof window !== 'undefined'
    ? preload().catch((err) => {
        console.error('Moonlab preload failed', err);
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
  probabilities: number[],
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

const OrbitalDemo: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const cloudRef = useRef<THREE.Points>();
  const sceneRef = useRef<THREE.Scene>();
  const rendererRef = useRef<THREE.WebGLRenderer>();
  const materialRef = useRef<THREE.PointsMaterial>();
  const controlsRef = useRef<OrbitControls>();
  const gridRef = useRef<THREE.GridHelper | null>(null);
  const axesRef = useRef<THREE.AxesHelper | null>(null);
  const rotatingRef = useRef<boolean>(true);
  const [atom, setAtom] = useState<Atom>(() => ELEMENTS.find((e) => e.symbol === 'N') || ELEMENTS[0]);
  const [n, setN] = useState<number>(4);
  const [l, setL] = useState<number>(2);
  const [m, setM] = useState<number>(0);
  const [qubits, setQubits] = useState<number>(
    Math.max(4, Math.min(MAX_QUBITS_UI, DEFAULT_QUBITS))
  );
  const [allowHighQubits, setAllowHighQubits] = useState<boolean>(false);
  const [pointCount, setPointCount] = useState<number>(30000);
  const [pointSize, setPointSize] = useState<number>(0.05);
  const [opacity, setOpacity] = useState<number>(0.5);
  const [isRotating, setIsRotating] = useState<boolean>(true);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [lastStats, setLastStats] = useState<{ elapsedMs: number; generated: number } | null>(null);
  const [pointsBuffer, setPointsBuffer] = useState<Float32Array | null>(null);
  const [currentExtent, setCurrentExtent] = useState<number>(extentForAtom(4, 7));
  const [isCollapsed, setIsCollapsed] = useState<boolean>(false);
  const [isPickerOpen, setIsPickerOpen] = useState<boolean>(false);

  const lOptions = useMemo(() => Array.from({ length: n }, (_, i) => i), [n]);
  const mOptions = useMemo(() => Array.from({ length: l * 2 + 1 }, (_, i) => i - l), [l]);

  useEffect(() => {
    setM((prev) => clamp(prev, -l, l));
  }, [l]);

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

    const nucleus = new THREE.Mesh(
      new THREE.SphereGeometry(0.12, 32, 32),
      new THREE.MeshBasicMaterial({ color: '#ff7ac3' })
    );
    scene.add(nucleus);

    const initialExtent = extentForAtom(n, atom.Z);
    const initialDim = Math.pow(2, Math.min(Math.max(8, DEFAULT_QUBITS), MAX_QUBITS_RUNTIME));
    const initialGridSide = Math.min(MAX_GRID, Math.max(MIN_GRID, Math.floor(Math.cbrt(initialDim))));
    const initialSpacing = (initialExtent * 2) / (initialGridSide - 1);
    const initialSize = initialSpacing * (initialGridSide - 1);

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

  const regenerate = async (targetQubits: number) => {
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

    // Normalize via Moonlab state to respect measurement semantics
    const amplitudes: Complex[] = new Array(targetDim);
    let total = 0;
    for (let i = 0; i < targetDim; i++) {
      total += aligned.probs[i];
    }
    const norm = total > 0 ? Math.sqrt(total) : 1;
    for (let i = 0; i < targetDim; i++) {
      const amp = aligned.probs[i] > 0 ? Math.sqrt(aligned.probs[i]) / norm : 0;
      amplitudes[i] = { real: amp, imag: 0 };
    }

    const state = await QuantumState.create({ numQubits: effectiveQubits });
    state.setAmplitudes(amplitudes);
    const probs = Array.from(state.getProbabilities());
    state.dispose();

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
  } catch (err) {
    console.error('Failed to build orbital cloud', err);
  } finally {
      setIsGenerating(false);
    }
  };

useEffect(() => {
  void regenerate(qubits);
}, [atom, n, l, m, pointCount, qubits]);

  return (
    <div className="orbital-page">
      <div className="orbital-viewport" ref={mountRef}>
        {isGenerating && <div className="overlay">Building orbital…</div>}
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
              <span>Select Atom</span>
              <div className="atom-row">
                <div className="atom-chip" title={atom.name}>
                  <span className="atom-symbol">{atom.symbol}</span>
                  <span className="atom-name">{atom.name}</span>
                  <span className="atom-z">Z = {atom.Z}</span>
                </div>
                <button className="btn btn-secondary" type="button" onClick={() => setIsPickerOpen(true)}>
                  Choose Element
                </button>
              </div>
            </div>

            <label className="control">
              <span>Principal Quantum Number (n)</span>
              <input
                type="range"
                min={1}
                max={5}
                value={n}
                onChange={(e) => {
                  const next = Number(e.target.value);
                  setN(next);
                  setL((prev) => clamp(prev, 0, next - 1));
                }}
              />
              <div className="control-value">n = {n}</div>
            </label>

            <label className="control">
              <span>WASM Qubits (min 4, UI max 32)</span>
              <input
                type="range"
                min={4}
                max={allowHighQubits ? MAX_QUBITS_UI : SAFE_QUBITS}
                value={Math.min(qubits, allowHighQubits ? MAX_QUBITS_UI : SAFE_QUBITS)}
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
                onChange={(e) => {
                  setAllowHighQubits(e.target.checked);
                  if (!e.target.checked && qubits > SAFE_QUBITS) {
                    setQubits(SAFE_QUBITS);
                  }
                }}
              />
              <span>I have ≥64 GB RAM (enable 25–32 qubits)</span>
            </label>

            <label className="control">
              <span>Angular Momentum (l)</span>
              <select value={l} onChange={(e) => setL(Number(e.target.value))}>
                {lOptions.map((val) => (
                  <option key={val} value={val}>
                    l = {val} ({L_LABELS[val] || 'higher'})
                  </option>
                ))}
              </select>
            </label>

            <label className="control">
              <span>Magnetic Quantum Number (m)</span>
              <select value={m} onChange={(e) => setM(Number(e.target.value))}>
                {mOptions.map((val) => (
                  <option key={val} value={val}>
                    m = {val}
                  </option>
                ))}
              </select>
            </label>

            <label className="control">
              <span>Point Density</span>
              <input
                type="range"
                min={4000}
                max={1000000}
                step={5000}
                value={pointCount}
                onChange={(e) => setPointCount(Number(e.target.value))}
              />
              <div className="control-value">{pointCount.toLocaleString()} samples</div>
            </label>

            <label className="control">
              <span>Point Size</span>
              <input
                type="range"
                min={0.01}
                max={0.15}
                step={0.005}
                value={pointSize}
                onChange={(e) => setPointSize(Number(e.target.value))}
              />
              <div className="control-value">{pointSize.toFixed(3)}</div>
            </label>

            <label className="control">
              <span>Opacity</span>
              <input
                type="range"
                min={0.1}
                max={0.9}
                step={0.05}
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
              />
              <div className="control-value">{opacity.toFixed(2)}</div>
            </label>
          </div>

          <div className="button-row">
            <button className="btn btn-primary" onClick={() => void regenerate()} disabled={isGenerating}>
              {isGenerating ? 'Generating…' : 'Regenerate Cloud'}
            </button>
            <button className="btn btn-secondary" onClick={() => setIsRotating((prev) => !prev)}>
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
          </div>

            <div className="about-card">
              <h3>About this simulation</h3>
              <p>
                We discretize an adaptive 3D lattice (scales with qubits up to 32³), assign hydrogen-like orbital
                amplitudes with Moonlab (up to {MAX_QUBITS_RUNTIME} qubits runtime), let the WASM backend normalize
                the distribution, and sample measurement outcomes to drive the point cloud. Three.js renders the
                resulting |ψ|² density with orbit controls.
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
