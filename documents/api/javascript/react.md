# Archived Moonlab Documentation: React Integration

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# React Integration

React hooks and components for quantum simulation.

**Package**: `@moonlab/quantum-react`

## Overview

The `@moonlab/quantum-react` package provides idiomatic React integration for quantum simulation:

- **Hooks**: `useQuantumState`, `useCircuit`, `useMeasurement`
- **Components**: Pre-built visualization components
- **Context**: Provider for shared quantum resources
- **TypeScript**: Full type definitions included

## Installation

[archived fence delimiter: ```bash]
npm install @moonlab/quantum-react @moonlab/quantum-core @moonlab/quantum-viz
# or
yarn add @moonlab/quantum-react @moonlab/quantum-core @moonlab/quantum-viz
# or
pnpm add @moonlab/quantum-react @moonlab/quantum-core @moonlab/quantum-viz
[archived fence delimiter: ```]

## Quick Start

[archived fence delimiter: ```tsx]
import React from 'react';
import {
  QuantumProvider,
  useQuantumState,
  AmplitudeBars
} from '@moonlab/quantum-react';

function BellStateDemo() {
  const { state, isLoading, error } = useQuantumState({ numQubits: 2 });

  if (isLoading) return <div>Loading WASM...</div>;
  if (error) return <div>Error: {error.message}</div>;

  // Apply gates
  state.h(0).cnot(0, 1);

  return (
    <div>
      <h2>Bell State: (|00⟩ + |11⟩)/√2</h2>
      <AmplitudeBars state={state} width={400} height={200} />
    </div>
  );
}

function App() {
  return (
    <QuantumProvider>
      <BellStateDemo />
    </QuantumProvider>
  );
}

export default App;
[archived fence delimiter: ```]

## QuantumProvider

Context provider for quantum simulation resources.

### Usage

[archived fence delimiter: ```tsx]
import { QuantumProvider } from '@moonlab/quantum-react';

function App() {
  return (
    <QuantumProvider
      wasmPath="/wasm/moonlab.wasm"
      onLoad={() => console.log('WASM loaded')}
      onError={(err) => console.error('Load failed:', err)}
    >
      <YourApp />
    </QuantumProvider>
  );
}
[archived fence delimiter: ```]

### Props

[archived fence delimiter: ```typescript]
interface QuantumProviderProps {
  children: React.ReactNode;
  wasmPath?: string;           // Custom WASM path
  onLoad?: () => void;         // Load success callback
  onError?: (error: Error) => void;  // Load error callback
}
[archived fence delimiter: ```]

### useQuantumContext

Access provider context directly.

[archived fence delimiter: ```tsx]
import { useQuantumContext } from '@moonlab/quantum-react';

function DebugInfo() {
  const { isLoaded, version, memoryUsage } = useQuantumContext();

  return (
    <div>
      <p>WASM Loaded: {isLoaded ? 'Yes' : 'No'}</p>
      <p>Version: {version}</p>
      <p>Memory: {memoryUsage} bytes</p>
    </div>
  );
}
[archived fence delimiter: ```]

## Hooks

### useQuantumState

Create and manage a quantum state.

[archived fence delimiter: ```typescript]
function useQuantumState(options: UseQuantumStateOptions): UseQuantumStateResult
[archived fence delimiter: ```]

**Options**:
[archived fence delimiter: ```typescript]
interface UseQuantumStateOptions {
  numQubits: number;           // Number of qubits (1-30)
  initialAmplitudes?: Complex[];  // Optional initial state
}
[archived fence delimiter: ```]

**Result**:
[archived fence delimiter: ```typescript]
interface UseQuantumStateResult {
  state: QuantumState | null;  // The quantum state (null while loading)
  isLoading: boolean;          // Whether WASM is loading
  error: Error | null;         // Any error that occurred
  reset: () => void;           // Reset to |0...0⟩
  probabilities: Float64Array; // Current probabilities
  measure: (qubit: number) => number;  // Measure qubit
  measureAll: () => number;    // Measure all qubits
}
[archived fence delimiter: ```]

**Example**:
[archived fence delimiter: ```tsx]
import { useQuantumState } from '@moonlab/quantum-react';

function QuantumCircuit() {
  const {
    state,
    isLoading,
    error,
    reset,
    probabilities,
    measureAll
  } = useQuantumState({ numQubits: 3 });

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  const handleRun = () => {
    state.h(0).cnot(0, 1).cnot(1, 2);
  };

  const handleMeasure = () => {
    const result = measureAll();
    console.log(`Measured: ${result.toString(2).padStart(3, '0')}`);
  };

  return (
    <div>
      <button onClick={handleRun}>Create GHZ</button>
      <button onClick={handleMeasure}>Measure</button>
      <button onClick={reset}>Reset</button>
      <pre>{JSON.stringify(Array.from(probabilities), null, 2)}</pre>
    </div>
  );
}
[archived fence delimiter: ```]

### useCircuit

Build reusable quantum circuits.

[archived fence delimiter: ```typescript]
function useCircuit(numQubits: number): UseCircuitResult
[archived fence delimiter: ```]

**Result**:
[archived fence delimiter: ```typescript]
interface UseCircuitResult {
  circuit: Circuit;            // Circuit builder
  gates: GateRecord[];         // List of applied gates
  stats: CircuitStats;         // Gate counts, depth
  clear: () => void;           // Clear circuit
  apply: (state: QuantumState) => void;  // Apply to state
}

interface GateRecord {
  name: string;
  qubits: number[];
  params?: number[];
  timestamp: number;
}

interface CircuitStats {
  numGates: number;
  depth: number;
  singleQubitGates: number;
  twoQubitGates: number;
  threeQubitGates: number;
}
[archived fence delimiter: ```]

**Example**:
[archived fence delimiter: ```tsx]
import { useCircuit, useQuantumState } from '@moonlab/quantum-react';

function CircuitBuilder() {
  const { state } = useQuantumState({ numQubits: 4 });
  const { circuit, gates, stats, clear, apply } = useCircuit(4);

  const addHadamard = (qubit: number) => {
    circuit.h(qubit);
  };

  const addCNOT = (control: number, target: number) => {
    circuit.cnot(control, target);
  };

  const runCircuit = () => {
    if (state) {
      state.reset();
      apply(state);
    }
  };

  return (
    <div>
      <div>Gates: {stats.numGates}, Depth: {stats.depth}</div>
      <div>
        {[0, 1, 2, 3].map(q => (
          <button key={q} onClick={() => addHadamard(q)}>
            H({q})
          </button>
        ))}
      </div>
      <button onClick={runCircuit}>Run</button>
      <button onClick={clear}>Clear</button>
    </div>
  );
}
[archived fence delimiter: ```]

### useMeasurement

Perform repeated measurements with statistics.

[archived fence delimiter: ```typescript]
function useMeasurement(state: QuantumState | null): UseMeasurementResult
[archived fence delimiter: ```]

**Result**:
[archived fence delimiter: ```typescript]
interface UseMeasurementResult {
  measure: (shots: number) => MeasurementResults;
  results: MeasurementResults | null;
  histogram: Map<number, number>;  // State → count
  isRunning: boolean;
}

interface MeasurementResults {
  shots: number;
  counts: Map<number, number>;
  mostFrequent: number;
  probabilities: Map<number, number>;
}
[archived fence delimiter: ```]

**Example**:
[archived fence delimiter: ```tsx]
import { useQuantumState, useMeasurement } from '@moonlab/quantum-react';

function MeasurementDemo() {
  const { state } = useQuantumState({ numQubits: 2 });
  const { measure, results, histogram } = useMeasurement(state);

  const runExperiment = () => {
    if (state) {
      state.reset().h(0).cnot(0, 1);
      measure(1000);  // 1000 shots
    }
  };

  return (
    <div>
      <button onClick={runExperiment}>Run 1000 Shots</button>
      {results && (
        <div>
          <p>Most frequent: |{results.mostFrequent.toString(2).padStart(2, '0')}⟩</p>
          {Array.from(histogram.entries()).map(([state, count]) => (
            <div key={state}>
              |{state.toString(2).padStart(2, '0')}⟩: {count} ({(count/1000*100).toFixed(1)}%)
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
[archived fence delimiter: ```]

### useBlochCoordinates

Extract Bloch sphere coordinates from single-qubit state.

[archived fence delimiter: ```typescript]
function useBlochCoordinates(state: QuantumState | null, qubit?: number): BlochCoords
[archived fence delimiter: ```]

**Result**:
[archived fence delimiter: ```typescript]
interface BlochCoords {
  theta: number;   // Polar angle (0 to π)
  phi: number;     // Azimuthal angle (0 to 2π)
  x: number;       // Cartesian x
  y: number;       // Cartesian y
  z: number;       // Cartesian z
}
[archived fence delimiter: ```]

**Example**:
[archived fence delimiter: ```tsx]
import { useQuantumState, useBlochCoordinates } from '@moonlab/quantum-react';

function BlochDisplay() {
  const { state } = useQuantumState({ numQubits: 1 });
  const coords = useBlochCoordinates(state);

  if (state) {
    state.h(0).t(0);
  }

  return (
    <div>
      <p>θ = {coords.theta.toFixed(3)}</p>
      <p>φ = {coords.phi.toFixed(3)}</p>
      <p>Cartesian: ({coords.x.toFixed(2)}, {coords.y.toFixed(2)}, {coords.z.toFixed(2)})</p>
    </div>
  );
}
[archived fence delimiter: ```]

## Components

### AmplitudeBars

Bar chart of state probabilities.

[archived fence delimiter: ```tsx]
import { AmplitudeBars } from '@moonlab/quantum-react';

<AmplitudeBars
  state={quantumState}
  width={600}
  height={300}
  showLabels={true}
  showValues={true}
  theme="dark"
  animate={true}
  highlightStates={[0, 7]}
/>
[archived fence delimiter: ```]

**Props**:
[archived fence delimiter: ```typescript]
interface AmplitudeBarsProps {
  state: QuantumState;         // Quantum state to visualize
  width?: number;              // Chart width (default: 400)
  height?: number;             // Chart height (default: 200)
  showLabels?: boolean;        // Show basis state labels (default: true)
  showValues?: boolean;        // Show probability values (default: true)
  maxBars?: number;            // Max bars to display (default: 32)
  threshold?: number;          // Min probability to show (default: 0.001)
  theme?: 'light' | 'dark' | ThemeConfig;  // Color theme
  animate?: boolean;           // Animate transitions (default: true)
  highlightStates?: number[];  // States to highlight
  onClick?: (state: number) => void;  // Click handler
  className?: string;          // CSS class
  style?: React.CSSProperties; // Inline styles
}
[archived fence delimiter: ```]

### BlochSphere

Single-qubit Bloch sphere visualization.

[archived fence delimiter: ```tsx]
import { BlochSphere } from '@moonlab/quantum-react';

<BlochSphere
  state={singleQubitState}
  size={300}
  mode="3d"
  showAxes={true}
  showLabels={true}
  autoRotate={false}
  theme="dark"
/>
[archived fence delimiter: ```]

**Props**:
[archived fence delimiter: ```typescript]
interface BlochSphereProps {
  state: QuantumState;         // Single-qubit state
  size?: number;               // Canvas size (default: 200)
  mode?: '2d' | '3d';          // Rendering mode (default: '2d')
  showAxes?: boolean;          // Show axes (default: true)
  showLabels?: boolean;        // Show labels (default: true)
  autoRotate?: boolean;        // 3D auto-rotation (default: false)
  theme?: 'light' | 'dark' | ThemeConfig;
  className?: string;
  style?: React.CSSProperties;
}
[archived fence delimiter: ```]

### CircuitDiagram

Quantum circuit visualization.

[archived fence delimiter: ```tsx]
import { CircuitDiagram } from '@moonlab/quantum-react';

<CircuitDiagram
  state={quantumState}
  width={600}
  gateStyle="ibm"
  showMeasurements={true}
  interactive={true}
  onGateClick={(gate) => console.log('Clicked:', gate)}
/>
[archived fence delimiter: ```]

**Props**:
[archived fence delimiter: ```typescript]
interface CircuitDiagramProps {
  state: QuantumState;           // State with gate history
  width?: number;                // Diagram width (auto if not set)
  height?: number;               // Diagram height (auto if not set)
  gateStyle?: 'default' | 'ibm' | 'google' | 'minimal';
  showMeasurements?: boolean;    // Show measurement symbols
  interactive?: boolean;         // Enable click/hover
  onGateClick?: (gate: GateInfo) => void;
  onGateHover?: (gate: GateInfo | null) => void;
  className?: string;
  style?: React.CSSProperties;
}
[archived fence delimiter: ```]

### StateVector

Display full state vector with amplitudes.

[archived fence delimiter: ```tsx]
import { StateVector } from '@moonlab/quantum-react';

<StateVector
  state={quantumState}
  format="ket"
  precision={4}
  threshold={0.001}
  showPhase={true}
/>
[archived fence delimiter: ```]

**Props**:
[archived fence delimiter: ```typescript]
interface StateVectorProps {
  state: QuantumState;
  format?: 'ket' | 'table' | 'matrix';  // Display format
  precision?: number;            // Decimal places (default: 4)
  threshold?: number;            // Min amplitude to show
  showPhase?: boolean;           // Show phase angles
  showProbability?: boolean;     // Show |α|²
  className?: string;
  style?: React.CSSProperties;
}
[archived fence delimiter: ```]

### MeasurementHistogram

Histogram of measurement results.

[archived fence delimiter: ```tsx]
import { MeasurementHistogram } from '@moonlab/quantum-react';

<MeasurementHistogram
  counts={measurementCounts}
  shots={1000}
  width={500}
  height={300}
  sortBy="value"
/>
[archived fence delimiter: ```]

**Props**:
[archived fence delimiter: ```typescript]
interface MeasurementHistogramProps {
  counts: Map<number, number>;   // State → count
  shots: number;                 // Total measurements
  width?: number;
  height?: number;
  sortBy?: 'state' | 'value';    // Sort order
  maxBars?: number;              // Limit displayed bars
  theme?: 'light' | 'dark';
  className?: string;
  style?: React.CSSProperties;
}
[archived fence delimiter: ```]

## Algorithm Components

### GroverDemo

Interactive Grover's algorithm demonstration.

[archived fence delimiter: ```tsx]
import { GroverDemo } from '@moonlab/quantum-react';

<GroverDemo
  numQubits={4}
  onComplete={(result) => console.log('Found:', result)}
/>
[archived fence delimiter: ```]

### BellTestDemo

CHSH Bell inequality test.

[archived fence delimiter: ```tsx]
import { BellTestDemo } from '@moonlab/quantum-react';

<BellTestDemo
  numShots={10000}
  onResult={(chsh) => console.log('CHSH:', chsh)}
/>
[archived fence delimiter: ```]

### VQEDemo

Variational Quantum Eigensolver visualization.

[archived fence delimiter: ```tsx]
import { VQEDemo } from '@moonlab/quantum-react';

<VQEDemo
  molecule="H2"
  numLayers={2}
  onConverge={(energy) => console.log('Energy:', energy)}
/>
[archived fence delimiter: ```]

## Theming

### Built-in Themes

[archived fence delimiter: ```tsx]
import { ThemeProvider, themes } from '@moonlab/quantum-react';

function App() {
  return (
    <ThemeProvider theme={themes.dark}>
      <QuantumProvider>
        <YourApp />
      </QuantumProvider>
    </ThemeProvider>
  );
}
[archived fence delimiter: ```]

### Custom Theme

[archived fence delimiter: ```tsx]
import { ThemeProvider, createTheme } from '@moonlab/quantum-react';

const myTheme = createTheme({
  colors: {
    primary: '#6366f1',
    secondary: '#8b5cf6',
    background: '#0f172a',
    surface: '#1e293b',
    text: '#f1f5f9',
    textSecondary: '#94a3b8',
    success: '#22c55e',
    error: '#ef4444',
    grid: '#334155',
  },
  fonts: {
    body: 'Inter, sans-serif',
    mono: 'JetBrains Mono, monospace',
  },
  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
  },
});

function App() {
  return (
    <ThemeProvider theme={myTheme}>
      <QuantumProvider>
        <YourApp />
      </QuantumProvider>
    </ThemeProvider>
  );
}
[archived fence delimiter: ```]

### useTheme Hook

[archived fence delimiter: ```tsx]
import { useTheme } from '@moonlab/quantum-react';

function CustomComponent() {
  const theme = useTheme();

  return (
    <div style={{
      backgroundColor: theme.colors.surface,
      color: theme.colors.text,
      padding: theme.spacing.md,
    }}>
      Custom styled content
    </div>
  );
}
[archived fence delimiter: ```]

## Complete Example

[archived fence delimiter: ```tsx]
import React, { useState } from 'react';
import {
  QuantumProvider,
  ThemeProvider,
  useQuantumState,
  useCircuit,
  useMeasurement,
  AmplitudeBars,
  CircuitDiagram,
  BlochSphere,
  MeasurementHistogram,
  themes,
} from '@moonlab/quantum-react';

function QuantumLab() {
  const { state, isLoading, error, reset } = useQuantumState({ numQubits: 3 });
  const { circuit, stats, clear, apply } = useCircuit(3);
  const { measure, results, histogram } = useMeasurement(state);
  const [selectedQubit, setSelectedQubit] = useState(0);

  if (isLoading) return <div className="loading">Loading quantum core...</div>;
  if (error) return <div className="error">Error: {error.message}</div>;

  const handleAddGate = (gate: string) => {
    switch (gate) {
      case 'H':
        circuit.h(selectedQubit);
        break;
      case 'X':
        circuit.x(selectedQubit);
        break;
      case 'CNOT':
        circuit.cnot(selectedQubit, (selectedQubit + 1) % 3);
        break;
    }
  };

  const handleRun = () => {
    reset();
    apply(state);
  };

  const handleMeasure = () => {
    measure(1000);
  };

  return (
    <div className="quantum-lab">
      <header>
        <h1>Quantum Lab</h1>
        <div className="stats">
          Gates: {stats.numGates} | Depth: {stats.depth}
        </div>
      </header>

      <div className="controls">
        <div className="qubit-selector">
          {[0, 1, 2].map(q => (
            <button
              key={q}
              className={q === selectedQubit ? 'active' : ''}
              onClick={() => setSelectedQubit(q)}
            >
              Q{q}
            </button>
          ))}
        </div>

        <div className="gate-buttons">
          <button onClick={() => handleAddGate('H')}>H</button>
          <button onClick={() => handleAddGate('X')}>X</button>
          <button onClick={() => handleAddGate('CNOT')}>CNOT</button>
        </div>

        <div className="action-buttons">
          <button onClick={handleRun}>Run Circuit</button>
          <button onClick={handleMeasure}>Measure (1000x)</button>
          <button onClick={() => { clear(); reset(); }}>Reset</button>
        </div>
      </div>

      <div className="visualizations">
        <div className="panel">
          <h2>Circuit</h2>
          <CircuitDiagram state={state} gateStyle="ibm" />
        </div>

        <div className="panel">
          <h2>Probabilities</h2>
          <AmplitudeBars state={state} width={400} height={200} />
        </div>

        {results && (
          <div className="panel">
            <h2>Measurement Results</h2>
            <MeasurementHistogram
              counts={histogram}
              shots={results.shots}
              width={400}
              height={200}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function App() {
  return (
    <ThemeProvider theme={themes.dark}>
      <QuantumProvider>
        <QuantumLab />
      </QuantumProvider>
    </ThemeProvider>
  );
}

export default App;
[archived fence delimiter: ```]

## TypeScript Support

Full TypeScript definitions are included:

[archived fence delimiter: ```typescript]
import type {
  QuantumState,
  Complex,
  GateInfo,
  CircuitStats,
  MeasurementResults,
  ThemeConfig,
  BlochCoords,
} from '@moonlab/quantum-react';
[archived fence delimiter: ```]

## Server-Side Rendering

The WASM module requires browser APIs. For SSR frameworks:

[archived fence delimiter: ```tsx]
import dynamic from 'next/dynamic';

const QuantumProvider = dynamic(
  () => import('@moonlab/quantum-react').then(mod => mod.QuantumProvider),
  { ssr: false }
);

const AmplitudeBars = dynamic(
  () => import('@moonlab/quantum-react').then(mod => mod.AmplitudeBars),
  { ssr: false }
);
[archived fence delimiter: ```]

## Performance Tips

1. **Memoize expensive operations**:
[archived fence delimiter: ```tsx]
const probabilities = useMemo(
  () => state?.getProbabilities(),
  [state, /* gate operations */]
);
[archived fence delimiter: ```]

2. **Use `React.memo` for visualization components**:
[archived fence delimiter: ```tsx]
const MemoizedBars = React.memo(AmplitudeBars);
[archived fence delimiter: ```]

3. **Dispose states when unmounting**:
[archived fence delimiter: ```tsx]
useEffect(() => {
  return () => {
    state?.dispose();
  };
}, [state]);
[archived fence delimiter: ```]

4. **Limit re-renders with `useCallback`**:
[archived fence delimiter: ```tsx]
const handleMeasure = useCallback(() => {
  if (state) {
    measure(1000);
  }
}, [state, measure]);
[archived fence delimiter: ```]

## See Also

- [Core API](core.md) - Quantum simulation core
- [Visualization API](viz.md) - Low-level visualization
- [Vue Integration](vue.md) - Vue composables

```
