import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { QuantumState, preload, type Complex } from '@moonlab/quantum-core';
import './Orbitals.css';

// Preload Moonlab once per module load (avoids React strict-mode double effects)
const orbitalPreload =
  typeof window !== 'undefined'
    ? preload().catch((err) => {
        console.error('Moonlab preload failed', err);
        throw err;
      })
    : Promise.resolve();

type Atom = { symbol: string; name: string; Z: number };

const ATOMS: Atom[] = [
  // Period 1
  { symbol: 'H', name: 'Hydrogen', Z: 1 },
  { symbol: 'He', name: 'Helium', Z: 2 },
  // Period 2
  { symbol: 'Li', name: 'Lithium', Z: 3 },
  { symbol: 'Be', name: 'Beryllium', Z: 4 },
  { symbol: 'B', name: 'Boron', Z: 5 },
  { symbol: 'C', name: 'Carbon', Z: 6 },
  { symbol: 'N', name: 'Nitrogen', Z: 7 },
  { symbol: 'O', name: 'Oxygen', Z: 8 },
  { symbol: 'F', name: 'Fluorine', Z: 9 },
  { symbol: 'Ne', name: 'Neon', Z: 10 },
  // Period 3
  { symbol: 'Na', name: 'Sodium', Z: 11 },
  { symbol: 'Mg', name: 'Magnesium', Z: 12 },
  { symbol: 'Al', name: 'Aluminum', Z: 13 },
  { symbol: 'Si', name: 'Silicon', Z: 14 },
  { symbol: 'P', name: 'Phosphorus', Z: 15 },
  { symbol: 'S', name: 'Sulfur', Z: 16 },
  { symbol: 'Cl', name: 'Chlorine', Z: 17 },
  { symbol: 'Ar', name: 'Argon', Z: 18 },
  // Period 4
  { symbol: 'K', name: 'Potassium', Z: 19 },
  { symbol: 'Ca', name: 'Calcium', Z: 20 },
  { symbol: 'Sc', name: 'Scandium', Z: 21 },
  { symbol: 'Ti', name: 'Titanium', Z: 22 },
  { symbol: 'V', name: 'Vanadium', Z: 23 },
  { symbol: 'Cr', name: 'Chromium', Z: 24 },
  { symbol: 'Mn', name: 'Manganese', Z: 25 },
  { symbol: 'Fe', name: 'Iron', Z: 26 },
  { symbol: 'Co', name: 'Cobalt', Z: 27 },
  { symbol: 'Ni', name: 'Nickel', Z: 28 },
  { symbol: 'Cu', name: 'Copper', Z: 29 },
  { symbol: 'Zn', name: 'Zinc', Z: 30 },
  { symbol: 'Ga', name: 'Gallium', Z: 31 },
  { symbol: 'Ge', name: 'Germanium', Z: 32 },
  { symbol: 'As', name: 'Arsenic', Z: 33 },
  { symbol: 'Se', name: 'Selenium', Z: 34 },
  { symbol: 'Br', name: 'Bromine', Z: 35 },
  { symbol: 'Kr', name: 'Krypton', Z: 36 },
  // Period 5
  { symbol: 'Rb', name: 'Rubidium', Z: 37 },
  { symbol: 'Sr', name: 'Strontium', Z: 38 },
  { symbol: 'Y', name: 'Yttrium', Z: 39 },
  { symbol: 'Zr', name: 'Zirconium', Z: 40 },
  { symbol: 'Nb', name: 'Niobium', Z: 41 },
  { symbol: 'Mo', name: 'Molybdenum', Z: 42 },
  { symbol: 'Tc', name: 'Technetium', Z: 43 },
  { symbol: 'Ru', name: 'Ruthenium', Z: 44 },
  { symbol: 'Rh', name: 'Rhodium', Z: 45 },
  { symbol: 'Pd', name: 'Palladium', Z: 46 },
  { symbol: 'Ag', name: 'Silver', Z: 47 },
  { symbol: 'Cd', name: 'Cadmium', Z: 48 },
  { symbol: 'In', name: 'Indium', Z: 49 },
  { symbol: 'Sn', name: 'Tin', Z: 50 },
  { symbol: 'Sb', name: 'Antimony', Z: 51 },
  { symbol: 'Te', name: 'Tellurium', Z: 52 },
  { symbol: 'I', name: 'Iodine', Z: 53 },
  { symbol: 'Xe', name: 'Xenon', Z: 54 },
  // Period 6
  { symbol: 'Cs', name: 'Cesium', Z: 55 },
  { symbol: 'Ba', name: 'Barium', Z: 56 },
  { symbol: 'La', name: 'Lanthanum', Z: 57 },
  { symbol: 'Ce', name: 'Cerium', Z: 58 },
  { symbol: 'Pr', name: 'Praseodymium', Z: 59 },
  { symbol: 'Nd', name: 'Neodymium', Z: 60 },
  { symbol: 'Pm', name: 'Promethium', Z: 61 },
  { symbol: 'Sm', name: 'Samarium', Z: 62 },
  { symbol: 'Eu', name: 'Europium', Z: 63 },
  { symbol: 'Gd', name: 'Gadolinium', Z: 64 },
  { symbol: 'Tb', name: 'Terbium', Z: 65 },
  { symbol: 'Dy', name: 'Dysprosium', Z: 66 },
  { symbol: 'Ho', name: 'Holmium', Z: 67 },
  { symbol: 'Er', name: 'Erbium', Z: 68 },
  { symbol: 'Tm', name: 'Thulium', Z: 69 },
  { symbol: 'Yb', name: 'Ytterbium', Z: 70 },
  { symbol: 'Lu', name: 'Lutetium', Z: 71 },
  { symbol: 'Hf', name: 'Hafnium', Z: 72 },
  { symbol: 'Ta', name: 'Tantalum', Z: 73 },
  { symbol: 'W', name: 'Tungsten', Z: 74 },
  { symbol: 'Re', name: 'Rhenium', Z: 75 },
  { symbol: 'Os', name: 'Osmium', Z: 76 },
  { symbol: 'Ir', name: 'Iridium', Z: 77 },
  { symbol: 'Pt', name: 'Platinum', Z: 78 },
  { symbol: 'Au', name: 'Gold', Z: 79 },
  { symbol: 'Hg', name: 'Mercury', Z: 80 },
  { symbol: 'Tl', name: 'Thallium', Z: 81 },
  { symbol: 'Pb', name: 'Lead', Z: 82 },
  { symbol: 'Bi', name: 'Bismuth', Z: 83 },
  { symbol: 'Po', name: 'Polonium', Z: 84 },
  { symbol: 'At', name: 'Astatine', Z: 85 },
  { symbol: 'Rn', name: 'Radon', Z: 86 },
  // Period 7
  { symbol: 'Fr', name: 'Francium', Z: 87 },
  { symbol: 'Ra', name: 'Radium', Z: 88 },
  { symbol: 'Ac', name: 'Actinium', Z: 89 },
  { symbol: 'Th', name: 'Thorium', Z: 90 },
  { symbol: 'Pa', name: 'Protactinium', Z: 91 },
  { symbol: 'U', name: 'Uranium', Z: 92 },
  { symbol: 'Np', name: 'Neptunium', Z: 93 },
  { symbol: 'Pu', name: 'Plutonium', Z: 94 },
  { symbol: 'Am', name: 'Americium', Z: 95 },
  { symbol: 'Cm', name: 'Curium', Z: 96 },
  { symbol: 'Bk', name: 'Berkelium', Z: 97 },
  { symbol: 'Cf', name: 'Californium', Z: 98 },
  { symbol: 'Es', name: 'Einsteinium', Z: 99 },
  { symbol: 'Fm', name: 'Fermium', Z: 100 },
  { symbol: 'Md', name: 'Mendelevium', Z: 101 },
  { symbol: 'No', name: 'Nobelium', Z: 102 },
  { symbol: 'Lr', name: 'Lawrencium', Z: 103 },
  { symbol: 'Rf', name: 'Rutherfordium', Z: 104 },
  { symbol: 'Db', name: 'Dubnium', Z: 105 },
  { symbol: 'Sg', name: 'Seaborgium', Z: 106 },
  { symbol: 'Bh', name: 'Bohrium', Z: 107 },
  { symbol: 'Hs', name: 'Hassium', Z: 108 },
  { symbol: 'Mt', name: 'Meitnerium', Z: 109 },
  { symbol: 'Ds', name: 'Darmstadtium', Z: 110 },
  { symbol: 'Rg', name: 'Roentgenium', Z: 111 },
  { symbol: 'Cn', name: 'Copernicium', Z: 112 },
  { symbol: 'Nh', name: 'Nihonium', Z: 113 },
  { symbol: 'Fl', name: 'Flerovium', Z: 114 },
  { symbol: 'Mc', name: 'Moscovium', Z: 115 },
  { symbol: 'Lv', name: 'Livermorium', Z: 116 },
  { symbol: 'Ts', name: 'Tennessine', Z: 117 },
  { symbol: 'Og', name: 'Oganesson', Z: 118 },
];

const L_LABELS = ['s', 'p', 'd', 'f', 'g'];
const GRID_SIZE = 16; // 16x16x16 grid -> 4096 lattice points
const NUM_STATES = GRID_SIZE * GRID_SIZE * GRID_SIZE;
const NUM_QUBITS = Math.ceil(Math.log2(NUM_STATES)); // 12 qubits

interface CloudParams {
  atom: Atom;
  n: number;
  l: number;
  m: number;
  pointCount: number;
  extent: number;
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
  const positions: { x: number; y: number; z: number }[] = new Array(NUM_STATES);
  const amplitudes: Complex[] = new Array(NUM_STATES);
  const probabilities: number[] = new Array(NUM_STATES);

  const spacing = (params.extent * 2) / (GRID_SIZE - 1);

  let totalProb = 0;
  for (let idx = 0; idx < NUM_STATES; idx++) {
    const z = Math.floor(idx / (GRID_SIZE * GRID_SIZE));
    const y = Math.floor((idx - z * GRID_SIZE * GRID_SIZE) / GRID_SIZE);
    const x = idx % GRID_SIZE;

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
  for (let i = 0; i < NUM_STATES; i++) {
    const amp = probabilities[i] > 0 ? Math.sqrt(probabilities[i]) / norm : 0;
    amplitudes[i] = { real: amp, imag: 0 };
  }

  const state = await QuantumState.create({ numQubits: NUM_QUBITS });
  state.setAmplitudes(amplitudes);
  const probs = Array.from(state.getProbabilities());
  state.dispose();

  const elapsedMs = performance.now() - start;
  return { positions, probabilities: probs, elapsedMs };
};

const samplePoints = (
  probabilities: number[],
  positions: { x: number; y: number; z: number }[],
  count: number,
  extent: number
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
  const spacing = (extent * 2) / (GRID_SIZE - 1);

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
  const rotatingRef = useRef<boolean>(true);
  const [atom, setAtom] = useState<Atom>(ATOMS[6]); // Nitrogen default
  const [n, setN] = useState<number>(4);
  const [l, setL] = useState<number>(2);
  const [m, setM] = useState<number>(0);
  const [pointCount, setPointCount] = useState<number>(30000);
  const [pointSize, setPointSize] = useState<number>(0.05);
  const [opacity, setOpacity] = useState<number>(0.5);
  const [isRotating, setIsRotating] = useState<boolean>(true);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [lastStats, setLastStats] = useState<{ elapsedMs: number; generated: number } | null>(null);
  const [pointsBuffer, setPointsBuffer] = useState<Float32Array | null>(null);
  const [currentExtent, setCurrentExtent] = useState<number>(extentForAtom(4, 7));
  const [isCollapsed, setIsCollapsed] = useState<boolean>(false);

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

    const grid = new THREE.GridHelper(10, 20, '#0b3b5a', '#0b3b5a');
    grid.position.y = -2.5;
    scene.add(grid);

    const axes = new THREE.AxesHelper(4);
    axes.position.set(0, -2.5, 0);
    scene.add(axes);

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

  const regenerate = async () => {
    setIsGenerating(true);
    const extent = extentForAtom(n, atom.Z);
    setCurrentExtent(extent);
    try {
      await orbitalPreload;
      const grid = await buildProbabilityGrid({
        atom,
        n,
        l,
        m,
        pointCount,
        extent,
      });
      const sampled = samplePoints(grid.probabilities, grid.positions, pointCount, extent);
      setPointsBuffer(sampled);
      setLastStats({ elapsedMs: grid.elapsedMs, generated: pointCount });
    } catch (err) {
      console.error('Failed to build orbital cloud', err);
    } finally {
      setIsGenerating(false);
    }
  };

  useEffect(() => {
    void regenerate();
  }, [atom, n, l, m, pointCount]);

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
            <label className="control">
              <span>Select Atom</span>
              <select
                value={atom.symbol}
                onChange={(e) => {
                  const next = ATOMS.find((a) => a.symbol === e.target.value);
                  if (next) setAtom(next);
                }}
              >
                {ATOMS.map((a) => (
                  <option key={a.symbol} value={a.symbol}>
                    {a.name} (Z = {a.Z})
                  </option>
                ))}
              </select>
            </label>

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
              We discretize a small 3D lattice, assign hydrogen-like orbital amplitudes with Moonlab
              ({NUM_QUBITS} qubits, {NUM_STATES} basis states), let the WASM backend normalize the
              distribution, and sample measurement outcomes to drive the point cloud. Three.js renders
              the resulting |ψ|² density with orbit controls.
            </p>
            <ul>
              <li>Adjust n, l, m to see shapes evolve (s, p, d, f…)</li>
              <li>Atom choice changes nuclear charge (Z) and radial decay</li>
              <li>Point density / size / opacity balance fidelity and performance</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OrbitalDemo;
