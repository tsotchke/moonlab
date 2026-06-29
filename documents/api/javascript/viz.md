# Archived Moonlab Documentation: JavaScript Visualization API

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# JavaScript Visualization API

Interactive quantum state visualizations for the browser.

**Package**: `@moonlab/quantum-viz`

## Overview

The `@moonlab/quantum-viz` package provides beautiful, interactive visualizations for quantum states and circuits:

- **BlochSphere**: 2D/3D single-qubit state visualization
- **AmplitudeBars**: Probability distribution bar charts
- **CircuitDiagram**: Quantum circuit rendering
- **StateCity**: 3D amplitude "cityscape" plots

## Installation

[archived fence delimiter: ```bash]
npm install @moonlab/quantum-viz
# or
yarn add @moonlab/quantum-viz
# or
pnpm add @moonlab/quantum-viz
[archived fence delimiter: ```]

## Quick Start

[archived fence delimiter: ```typescript]
import { BlochSphere, AmplitudeBars } from '@moonlab/quantum-viz';
import { QuantumState } from '@moonlab/quantum-core';

// Create quantum state
const state = await QuantumState.create({ numQubits: 2 });
state.h(0).cnot(0, 1);

// Render probability bars
const container = document.getElementById('viz');
const bars = new AmplitudeBars(container, {
  width: 600,
  height: 300
});
bars.update(state.getProbabilities());

// Render Bloch sphere for single qubit
const singleQubit = await QuantumState.create({ numQubits: 1 });
singleQubit.h(0).t(0);

const sphereContainer = document.getElementById('bloch');
const sphere = new BlochSphere(sphereContainer, { size: 300 });
sphere.update(singleQubit);
[archived fence delimiter: ```]

## BlochSphere

2D or 3D visualization of single-qubit states on the Bloch sphere.

### Constructor

[archived fence delimiter: ```typescript]
new BlochSphere(container: HTMLElement, options?: BlochSphereOptions)
[archived fence delimiter: ```]

**Parameters**:
[archived fence delimiter: ```typescript]
interface BlochSphereOptions {
  size?: number;           // Canvas size in pixels (default: 200)
  mode?: '2d' | '3d';      // Rendering mode (default: '2d')
  showAxes?: boolean;      // Show X, Y, Z axes (default: true)
  showLabels?: boolean;    // Show |0⟩, |1⟩, |+⟩, etc. (default: true)
  animate?: boolean;       // Enable rotation animation (default: false)
  colors?: BlochColors;    // Custom color scheme
}

interface BlochColors {
  background?: string;     // Canvas background
  sphere?: string;         // Sphere surface color
  axes?: string;           // Axis line color
  state?: string;          // State vector color
  labels?: string;         // Text label color
}
[archived fence delimiter: ```]

### Methods

#### update

[archived fence delimiter: ```typescript]
update(state: QuantumState | BlochCoordinates): void
[archived fence delimiter: ```]

Update the displayed state.

**Parameters**:
- `state`: Single-qubit QuantumState or explicit Bloch coordinates

[archived fence delimiter: ```typescript]
interface BlochCoordinates {
  theta: number;  // Polar angle (0 to π)
  phi: number;    // Azimuthal angle (0 to 2π)
}
[archived fence delimiter: ```]

**Example**:
[archived fence delimiter: ```typescript]
const sphere = new BlochSphere(container, { size: 300, mode: '2d' });

// From QuantumState
const state = await QuantumState.create({ numQubits: 1 });
state.h(0);
sphere.update(state);

// From coordinates
sphere.update({ theta: Math.PI / 4, phi: Math.PI / 2 });
[archived fence delimiter: ```]

#### setColors

[archived fence delimiter: ```typescript]
setColors(colors: Partial<BlochColors>): void
[archived fence delimiter: ```]

Update color scheme.

[archived fence delimiter: ```typescript]
sphere.setColors({
  background: '#1a1a2e',
  state: '#ff6b6b',
  axes: '#4ecdc4'
});
[archived fence delimiter: ```]

#### resize

[archived fence delimiter: ```typescript]
resize(size: number): void
[archived fence delimiter: ```]

Resize the visualization.

#### dispose

[archived fence delimiter: ```typescript]
dispose(): void
[archived fence delimiter: ```]

Clean up resources.

### Properties

[archived fence delimiter: ```typescript]
class BlochSphere {
  readonly canvas: HTMLCanvasElement;
  readonly size: number;
  theta: number;   // Current polar angle
  phi: number;     // Current azimuthal angle
}
[archived fence delimiter: ```]

## BlochSphere3D

WebGL-accelerated 3D Bloch sphere with interactive rotation.

### Constructor

[archived fence delimiter: ```typescript]
new BlochSphere3D(container: HTMLElement, options?: BlochSphere3DOptions)
[archived fence delimiter: ```]

**Parameters**:
[archived fence delimiter: ```typescript]
interface BlochSphere3DOptions extends BlochSphereOptions {
  enableZoom?: boolean;      // Mouse wheel zoom (default: true)
  enableRotate?: boolean;    // Click-drag rotation (default: true)
  autoRotate?: boolean;      // Continuous rotation (default: false)
  autoRotateSpeed?: number;  // Rotation speed (default: 1.0)
  quality?: 'low' | 'medium' | 'high';  // Mesh quality (default: 'medium')
}
[archived fence delimiter: ```]

### Methods

Inherits all BlochSphere methods plus:

#### setCamera

[archived fence delimiter: ```typescript]
setCamera(position: { x: number; y: number; z: number }): void
[archived fence delimiter: ```]

Set camera position.

#### resetCamera

[archived fence delimiter: ```typescript]
resetCamera(): void
[archived fence delimiter: ```]

Reset to default view.

#### startAnimation

[archived fence delimiter: ```typescript]
startAnimation(): void
[archived fence delimiter: ```]

Start continuous rendering loop.

#### stopAnimation

[archived fence delimiter: ```typescript]
stopAnimation(): void
[archived fence delimiter: ```]

Stop rendering loop.

### Example

[archived fence delimiter: ```typescript]
const sphere3d = new BlochSphere3D(container, {
  size: 400,
  autoRotate: true,
  autoRotateSpeed: 0.5,
  quality: 'high'
});

const state = await QuantumState.create({ numQubits: 1 });
state.rx(0, Math.PI / 6);
sphere3d.update(state);
sphere3d.startAnimation();
[archived fence delimiter: ```]

## AmplitudeBars

Bar chart visualization of probability distribution.

### Constructor

[archived fence delimiter: ```typescript]
new AmplitudeBars(container: HTMLElement, options?: AmplitudeBarsOptions)
[archived fence delimiter: ```]

**Parameters**:
[archived fence delimiter: ```typescript]
interface AmplitudeBarsOptions {
  width?: number;          // Chart width (default: 400)
  height?: number;         // Chart height (default: 200)
  showLabels?: boolean;    // Basis state labels (default: true)
  showValues?: boolean;    // Probability values (default: true)
  maxBars?: number;        // Max bars to display (default: 32)
  threshold?: number;      // Min probability to show (default: 0.001)
  colors?: BarColors;      // Color scheme
  animation?: boolean;     // Animate transitions (default: true)
  animationDuration?: number;  // Transition ms (default: 300)
}

interface BarColors {
  background?: string;
  bar?: string | string[];   // Single color or gradient stops
  label?: string;
  value?: string;
  grid?: string;
}
[archived fence delimiter: ```]

### Methods

#### update

[archived fence delimiter: ```typescript]
update(probabilities: Float64Array | number[]): void
[archived fence delimiter: ```]

Update displayed probabilities.

[archived fence delimiter: ```typescript]
const bars = new AmplitudeBars(container, { width: 600, height: 300 });

const state = await QuantumState.create({ numQubits: 3 });
state.h(0).cnot(0, 1).cnot(1, 2);

bars.update(state.getProbabilities());
[archived fence delimiter: ```]

#### highlight

[archived fence delimiter: ```typescript]
highlight(indices: number[]): void
[archived fence delimiter: ```]

Highlight specific basis states.

[archived fence delimiter: ```typescript]
bars.highlight([0, 7]);  // Highlight |000⟩ and |111⟩
[archived fence delimiter: ```]

#### clearHighlight

[archived fence delimiter: ```typescript]
clearHighlight(): void
[archived fence delimiter: ```]

Remove all highlights.

#### setColors

[archived fence delimiter: ```typescript]
setColors(colors: Partial<BarColors>): void
[archived fence delimiter: ```]

Update color scheme.

[archived fence delimiter: ```typescript]
bars.setColors({
  bar: ['#667eea', '#764ba2'],  // Gradient
  background: '#1a1a2e'
});
[archived fence delimiter: ```]

#### resize

[archived fence delimiter: ```typescript]
resize(width: number, height: number): void
[archived fence delimiter: ```]

Resize the chart.

#### dispose

[archived fence delimiter: ```typescript]
dispose(): void
[archived fence delimiter: ```]

Clean up resources.

### Complex Amplitudes Mode

[archived fence delimiter: ```typescript]
interface AmplitudeDisplayOptions {
  mode?: 'probability' | 'amplitude' | 'phase';
}

const bars = new AmplitudeBars(container, {
  width: 600,
  height: 300,
  mode: 'amplitude'  // Show |α|, not |α|²
});

// Update with complex amplitudes
bars.updateComplex(state.getAmplitudes());
[archived fence delimiter: ```]

## CircuitDiagram

Render quantum circuits as SVG.

### Constructor

[archived fence delimiter: ```typescript]
new CircuitDiagram(container: HTMLElement, options?: CircuitDiagramOptions)
[archived fence delimiter: ```]

**Parameters**:
[archived fence delimiter: ```typescript]
interface CircuitDiagramOptions {
  width?: number;           // SVG width (default: auto)
  height?: number;          // SVG height (default: auto)
  gateWidth?: number;       // Gate box width (default: 40)
  gateHeight?: number;      // Gate box height (default: 30)
  wireSpacing?: number;     // Vertical space (default: 50)
  gateSpacing?: number;     // Horizontal space (default: 20)
  style?: CircuitStyle;     // Visual style
  interactive?: boolean;    // Enable click handlers (default: false)
}

type CircuitStyle = 'default' | 'ibm' | 'google' | 'minimal';
[archived fence delimiter: ```]

### Methods

#### fromState

[archived fence delimiter: ```typescript]
static fromState(
  state: QuantumState,
  container: HTMLElement,
  options?: CircuitDiagramOptions
): CircuitDiagram
[archived fence delimiter: ```]

Create diagram from QuantumState gate history.

[archived fence delimiter: ```typescript]
const state = await QuantumState.create({ numQubits: 3 });
state.h(0).cnot(0, 1).cnot(1, 2);

const diagram = CircuitDiagram.fromState(state, container, {
  style: 'ibm',
  interactive: true
});
[archived fence delimiter: ```]

#### addGate

[archived fence delimiter: ```typescript]
addGate(gate: GateDefinition): this
[archived fence delimiter: ```]

Add gate to circuit.

[archived fence delimiter: ```typescript]
interface GateDefinition {
  name: string;              // Gate name ('H', 'X', 'CNOT', etc.)
  qubits: number[];          // Target qubits
  params?: number[];         // Parameters (for rotation gates)
  label?: string;            // Custom label
}
[archived fence delimiter: ```]

[archived fence delimiter: ```typescript]
const diagram = new CircuitDiagram(container, { style: 'minimal' });

diagram
  .addGate({ name: 'H', qubits: [0] })
  .addGate({ name: 'CNOT', qubits: [0, 1] })
  .addGate({ name: 'RZ', qubits: [1], params: [Math.PI / 4] })
  .render();
[archived fence delimiter: ```]

#### addMeasurement

[archived fence delimiter: ```typescript]
addMeasurement(qubit: number, classicalBit?: number): this
[archived fence delimiter: ```]

Add measurement symbol.

#### addBarrier

[archived fence delimiter: ```typescript]
addBarrier(qubits?: number[]): this
[archived fence delimiter: ```]

Add visual barrier.

#### render

[archived fence delimiter: ```typescript]
render(): void
[archived fence delimiter: ```]

Render the circuit to SVG.

#### toSVG

[archived fence delimiter: ```typescript]
toSVG(): string
[archived fence delimiter: ```]

Get SVG markup string.

[archived fence delimiter: ```typescript]
const svgString = diagram.toSVG();
// Use for download, server-side rendering, etc.
[archived fence delimiter: ```]

#### download

[archived fence delimiter: ```typescript]
download(filename?: string): void
[archived fence delimiter: ```]

Download as SVG file.

[archived fence delimiter: ```typescript]
diagram.download('my-circuit.svg');
[archived fence delimiter: ```]

#### clear

[archived fence delimiter: ```typescript]
clear(): void
[archived fence delimiter: ```]

Clear all gates.

#### dispose

[archived fence delimiter: ```typescript]
dispose(): void
[archived fence delimiter: ```]

Clean up resources.

### Gate Click Handlers

[archived fence delimiter: ```typescript]
const diagram = new CircuitDiagram(container, { interactive: true });

diagram.on('gateClick', (event) => {
  console.log(`Clicked: ${event.gate.name} on qubits ${event.gate.qubits}`);
});

diagram.on('gateHover', (event) => {
  // Show tooltip
});
[archived fence delimiter: ```]

## StateCity

3D "city" visualization of state amplitudes.

### Constructor

[archived fence delimiter: ```typescript]
new StateCity(container: HTMLElement, options?: StateCityOptions)
[archived fence delimiter: ```]

**Parameters**:
[archived fence delimiter: ```typescript]
interface StateCityOptions {
  width?: number;
  height?: number;
  colorMode?: 'probability' | 'phase';  // Color by amplitude or phase
  enableRotate?: boolean;
  enableZoom?: boolean;
}
[archived fence delimiter: ```]

### Methods

#### update

[archived fence delimiter: ```typescript]
update(amplitudes: Complex[]): void
[archived fence delimiter: ```]

Update displayed amplitudes.

[archived fence delimiter: ```typescript]
const city = new StateCity(container, {
  width: 600,
  height: 400,
  colorMode: 'phase'
});

const state = await QuantumState.create({ numQubits: 4 });
state.h(0).h(1).cz(0, 1);

city.update(state.getAmplitudes());
[archived fence delimiter: ```]

## Color Schemes

### Built-in Themes

[archived fence delimiter: ```typescript]
import { themes } from '@moonlab/quantum-viz';

const bars = new AmplitudeBars(container, {
  colors: themes.dark
});

// Available themes:
// themes.light - Light background
// themes.dark  - Dark background
// themes.ibm   - IBM Quantum style
// themes.google - Cirq style
[archived fence delimiter: ```]

### Custom Theme

[archived fence delimiter: ```typescript]
import { createTheme } from '@moonlab/quantum-viz';

const myTheme = createTheme({
  primary: '#6366f1',
  secondary: '#8b5cf6',
  background: '#0f0f23',
  text: '#e2e8f0',
  grid: '#334155'
});

const bars = new AmplitudeBars(container, { colors: myTheme.bars });
const sphere = new BlochSphere(container, { colors: myTheme.bloch });
[archived fence delimiter: ```]

## Animation Utilities

### AnimatedState

Track state evolution with smooth transitions.

[archived fence delimiter: ```typescript]
import { AnimatedState } from '@moonlab/quantum-viz';

const animated = new AnimatedState({
  duration: 500,
  easing: 'easeInOutQuad'
});

const bars = new AmplitudeBars(container);
animated.attach(bars);

// Updates animate smoothly
const state = await QuantumState.create({ numQubits: 2 });
animated.setState(state);  // Renders initial state

state.h(0);
animated.setState(state);  // Animates to new probabilities

state.cnot(0, 1);
animated.setState(state);  // Animates again
[archived fence delimiter: ```]

### Timeline

Step through circuit execution.

[archived fence delimiter: ```typescript]
import { Timeline } from '@moonlab/quantum-viz';

const state = await QuantumState.create({ numQubits: 3 });
const timeline = new Timeline(state);

// Record steps
timeline.record(() => state.h(0));
timeline.record(() => state.cnot(0, 1));
timeline.record(() => state.cnot(1, 2));

// Playback
timeline.step();      // Apply H(0)
timeline.step();      // Apply CNOT(0,1)
timeline.stepBack();  // Undo CNOT(0,1)
timeline.reset();     // Back to |000⟩
[archived fence delimiter: ```]

## Complete Example

[archived fence delimiter: ```typescript]
import { QuantumState } from '@moonlab/quantum-core';
import {
  BlochSphere,
  AmplitudeBars,
  CircuitDiagram,
  themes
} from '@moonlab/quantum-viz';

async function visualizeGHZ() {
  // Create 3-qubit GHZ state
  const state = await QuantumState.create({ numQubits: 3 });
  state.h(0).cnot(0, 1).cnot(1, 2);

  // Circuit diagram
  const circuitContainer = document.getElementById('circuit')!;
  const circuit = CircuitDiagram.fromState(state, circuitContainer, {
    style: 'ibm',
    gateWidth: 50
  });

  // Probability bars
  const barsContainer = document.getElementById('bars')!;
  const bars = new AmplitudeBars(barsContainer, {
    width: 600,
    height: 250,
    colors: themes.dark.bars
  });
  bars.update(state.getProbabilities());
  bars.highlight([0, 7]);  // Highlight |000⟩ and |111⟩

  // Single qubit Bloch sphere
  const qubit0 = await QuantumState.create({ numQubits: 1 });
  qubit0.h(0);

  const sphereContainer = document.getElementById('bloch')!;
  const sphere = new BlochSphere(sphereContainer, {
    size: 250,
    mode: '2d',
    colors: themes.dark.bloch
  });
  sphere.update(qubit0);

  // Cleanup on page unload
  window.addEventListener('beforeunload', () => {
    circuit.dispose();
    bars.dispose();
    sphere.dispose();
    state.dispose();
    qubit0.dispose();
  });
}

visualizeGHZ();
[archived fence delimiter: ```]

## Browser Support

| Browser | Minimum Version |
|---------|-----------------|
| Chrome | 60+ |
| Firefox | 55+ |
| Safari | 11+ |
| Edge | 79+ |

3D visualizations (BlochSphere3D, StateCity) require WebGL support.

## See Also

- [Core API](core.md) - Quantum simulation core
- [React Integration](react.md) - React components
- [Vue Integration](vue.md) - Vue composables

```
