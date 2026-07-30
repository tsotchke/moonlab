# JavaScript API Reference

WebAssembly-compiled quantum simulation with React and Vue components for building interactive web applications.

## Overview

| Package | Description |
|---------|-------------|
| [@tsotchkecorp/moonlab](core.md) | Core quantum simulation (WASM) |
| [@tsotchkecorp/moonlab-viz](viz.md) | D3.js-based visualization |
| [@tsotchkecorp/moonlab-react](react.md) | React components |
| [@tsotchkecorp/moonlab-vue](vue.md) | Vue components |

## Installation

```bash
# npm
npm install @tsotchkecorp/moonlab @tsotchkecorp/moonlab-viz

# pnpm
pnpm add @tsotchkecorp/moonlab @tsotchkecorp/moonlab-viz

# yarn
yarn add @tsotchkecorp/moonlab @tsotchkecorp/moonlab-viz
```

For React or Vue:
```bash
npm install @tsotchkecorp/moonlab-react  # or @tsotchkecorp/moonlab-vue
```

## Quick Start

### ES Modules

```javascript
import { QuantumState } from '@tsotchkecorp/moonlab';

async function main() {
  // Create 2-qubit state
  const state = await QuantumState.create({ numQubits: 2 });

  // Create Bell state
  state.h(0).cnot(0, 1);

  // Get probabilities
  const probs = state.getProbabilities();
  console.log('P(|00⟩):', probs[0].toFixed(4));  // 0.5000
  console.log('P(|11⟩):', probs[3].toFixed(4));  // 0.5000

  // Measure all qubits
  const result = state.measureAll();
  console.log(`Measured: |${result.toString(2).padStart(2, '0')}⟩`);

  // Clean up WebAssembly memory
  state.dispose();
}

main();
```

### CommonJS (Node.js)

```javascript
const { QuantumState } = require('@tsotchkecorp/moonlab');

async function main() {
  const state = await QuantumState.create({ numQubits: 4 });
  state.h(0).h(1).h(2).h(3);
  console.log('Created uniform superposition');
  state.dispose();
}

main();
```

## QuantumState Class

### Factory Method

QuantumState uses an async factory pattern (constructor is private).

```typescript
class QuantumState {
    static async create(options: QuantumStateOptions): Promise<QuantumState>;
}

interface QuantumStateOptions {
    numQubits: number;       // 1-30 qubits
    amplitudes?: Complex[];  // Optional initial amplitudes
}
```

**Parameters**:
- `numQubits`: Number of qubits (1-28 in browser, 1-30 in Node.js)

**Throws**: `Error` if memory allocation fails or qubits out of range

### Properties

```typescript
interface QuantumState {
    readonly numQubits: number;
    readonly stateDim: number;
}
```

### Gate Methods

All gate methods return `this` for chaining.

#### Single-Qubit Gates

```typescript
class QuantumState {
    x(qubit: number): this;
    y(qubit: number): this;
    z(qubit: number): this;
    h(qubit: number): this;
    s(qubit: number): this;
    sDagger(qubit: number): this;
    t(qubit: number): this;
    tDagger(qubit: number): this;
}
```

#### Rotation Gates

```typescript
class QuantumState {
    rx(qubit: number, theta: number): this;
    ry(qubit: number, theta: number): this;
    rz(qubit: number, theta: number): this;
    phase(qubit: number, theta: number): this;
}
```

#### Multi-Qubit Gates

```typescript
class QuantumState {
    cnot(control: number, target: number): this;
    cz(control: number, target: number): this;
    swap(qubit1: number, qubit2: number): this;
    cphase(control: number, target: number, theta: number): this;
    toffoli(control1: number, control2: number, target: number): this;
}
```

### State Access

```typescript
class QuantumState {
    // Get all amplitudes as Complex[]
    getAmplitudes(): Complex[];

    // Get probabilities
    getProbabilities(): Float64Array;

    // Get single probability
    probability(basisState: number): number;
}
```

### Measurement

```typescript
class QuantumState {
    // Measure single qubit (collapses state)
    measure(qubit: number): number;  // Returns 0 or 1

    // Measure all qubits (collapses state)
    measureAll(): number;  // Returns basis state index
}
```

### State Operations

```typescript
class QuantumState {
    // Reset to |0...0⟩
    reset(): this;

    // Clone the state
    async clone(): Promise<QuantumState>;

    // Entanglement entropy
    entropy(): number;

    // Free WebAssembly memory (REQUIRED when done)
    dispose(): void;
}
```

## TypeScript Support

Full TypeScript definitions included:

```typescript
import { QuantumState, Complex } from '@tsotchkecorp/moonlab';

interface Complex {
    re: number;
    im: number;
}

// Async usage required
const state = await QuantumState.create({ numQubits: 4 });
const amps: Complex[] = state.getAmplitudes();
const prob: number = state.probability(0);
```

## Visualization (@tsotchkecorp/moonlab-viz)

### CircuitDiagram

```typescript
import { CircuitDiagram } from '@tsotchkecorp/moonlab-viz';

const diagram = new CircuitDiagram('#container', {
    numQubits: 3,
    width: 600,
    height: 200
});

diagram.addGate('H', 0);
diagram.addGate('CNOT', 0, 1);
diagram.addGate('CNOT', 1, 2);
diagram.render();
```

### StateVisualization

```typescript
import { StateVisualization } from '@tsotchkecorp/moonlab-viz';

const viz = new StateVisualization('#container', state, {
    type: 'bar',  // 'bar' | 'bloch' | 'city'
    width: 400,
    height: 300
});

// Update when state changes
state.h(0);
viz.update(state);
```

### BlochSphere

```typescript
import { BlochSphere } from '@tsotchkecorp/moonlab-viz';

const bloch = new BlochSphere('#container', {
    width: 300,
    height: 300,
    showAxes: true
});

bloch.setState(state, 0);  // Show qubit 0
```

## React Components (@tsotchkecorp/moonlab-react)

### QuantumCircuit

```jsx
import { QuantumCircuit, Gate, Measure } from '@tsotchkecorp/moonlab-react';

function MyCircuit() {
    const [result, setResult] = useState(null);

    return (
        <QuantumCircuit qubits={3} onMeasure={setResult}>
            <Gate type="H" qubit={0} />
            <Gate type="CNOT" control={0} target={1} />
            <Gate type="CNOT" control={1} target={2} />
            <Measure qubits={[0, 1, 2]} />
        </QuantumCircuit>
    );
}
```

### useQuantumState Hook

```jsx
import { useQuantumState } from '@tsotchkecorp/moonlab-react';

function BellStateDemo() {
    const { state, gates, measure, probabilities } = useQuantumState(2);

    const createBellState = () => {
        gates.h(0);
        gates.cnot(0, 1);
    };

    return (
        <div>
            <button onClick={createBellState}>Create Bell State</button>
            <button onClick={measure}>Measure</button>
            <ProbabilityChart data={probabilities} />
        </div>
    );
}
```

### StateVector

```jsx
import { StateVector } from '@tsotchkecorp/moonlab-react';

<StateVector
    state={state}
    format="polar"  // 'cartesian' | 'polar' | 'probability'
    precision={4}
/>
```

## Vue Components (@tsotchkecorp/moonlab-vue)

### QuantumCircuit

```vue
<template>
    <QuantumCircuit :qubits="3" @measure="onMeasure">
        <Gate type="H" :qubit="0" />
        <Gate type="CNOT" :control="0" :target="1" />
        <Measure :qubits="[0, 1, 2]" />
    </QuantumCircuit>
</template>

<script setup>
import { QuantumCircuit, Gate, Measure } from '@tsotchkecorp/moonlab-vue';

const onMeasure = (result) => {
    console.log('Measured:', result);
};
</script>
```

### useQuantum Composable

```vue
<script setup>
import { useQuantum } from '@tsotchkecorp/moonlab-vue';

const { state, h, cnot, measure, probabilities } = useQuantum(2);

const createBellState = () => {
    h(0);
    cnot(0, 1);
};
</script>
```

## Async Initialization

The factory pattern handles WASM initialization automatically:

```javascript
import { QuantumState, preload } from '@tsotchkecorp/moonlab';

async function simulate() {
    // Optional: preload WASM module for faster first use
    await preload();

    const state = await QuantumState.create({ numQubits: 20 });
    // ... operations ...
    state.dispose();
}
```

## Web Worker Support

Run simulations in a Web Worker:

```javascript
// worker.js
import { QuantumState } from '@tsotchkecorp/moonlab';

self.onmessage = async (e) => {
    const { numQubits, circuit } = e.data;

    const state = await QuantumState.create({ numQubits });

    for (const gate of circuit) {
        state[gate.type](...gate.args);
    }

    const probs = Array.from(state.getProbabilities());
    state.dispose();

    self.postMessage({ probabilities: probs });
};

// main.js
const worker = new Worker('./worker.js', { type: 'module' });
worker.postMessage({
    numQubits: 10,
    circuit: [
        { type: 'h', args: [0] },
        { type: 'cnot', args: [0, 1] }
    ]
});
```

## Memory Management

WebAssembly requires manual memory management:

```javascript
const state = await QuantumState.create({ numQubits: 10 });

try {
    // ... operations ...
} finally {
    state.dispose();  // Always dispose!
}
```

With async/await pattern:
```javascript
async function simulate() {
    const state = await QuantumState.create({ numQubits: 10 });
    try {
        state.h(0);
        return state.measureAll();
    } finally {
        state.dispose();
    }
}
```

## Browser Compatibility

| Browser | Version | Notes |
|---------|---------|-------|
| Chrome | 57+ | Full support |
| Firefox | 52+ | Full support |
| Safari | 11+ | Full support |
| Edge | 79+ | Full support |

Requires WebAssembly support. Check with:
```javascript
const wasmSupported = typeof WebAssembly !== 'undefined';
```

## Bundle Size

| Package | Size (gzipped) |
|---------|----------------|
| @tsotchkecorp/moonlab | ~150 KB |
| @tsotchkecorp/moonlab-viz | ~45 KB |
| @tsotchkecorp/moonlab-react | ~12 KB |
| @tsotchkecorp/moonlab-vue | ~10 KB |

## See Also

- [Core API](core.md) - Complete reference
- [Visualization](viz.md) - Chart and diagram APIs
- [React Components](react.md) - React integration
- [Vue Components](vue.md) - Vue integration
- [Demo Site](https://moonlab.dev/demo) - Interactive examples
