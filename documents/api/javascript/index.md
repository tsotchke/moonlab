# Archived Moonlab Documentation: JavaScript API Reference

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# JavaScript API Reference

WebAssembly-compiled quantum simulation with React and Vue components for building interactive web applications.

## Overview

| Package | Description |
|---------|-------------|
| [@moonlab/quantum-core](core.md) | Core quantum simulation (WASM) |
| [@moonlab/quantum-viz](viz.md) | D3.js-based visualization |
| [@moonlab/quantum-react](react.md) | React components |
| [@moonlab/quantum-vue](vue.md) | Vue components |

## Installation

[archived fence delimiter: ```bash]
# npm
npm install @moonlab/quantum-core @moonlab/quantum-viz

# pnpm
pnpm add @moonlab/quantum-core @moonlab/quantum-viz

# yarn
yarn add @moonlab/quantum-core @moonlab/quantum-viz
[archived fence delimiter: ```]

For React or Vue:
[archived fence delimiter: ```bash]
npm install @moonlab/quantum-react  # or @moonlab/quantum-vue
[archived fence delimiter: ```]

## Quick Start

### ES Modules

[archived fence delimiter: ```javascript]
import { QuantumState } from '@moonlab/quantum-core';

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
[archived fence delimiter: ```]

### CommonJS (Node.js)

[archived fence delimiter: ```javascript]
const { QuantumState } = require('@moonlab/quantum-core');

async function main() {
  const state = await QuantumState.create({ numQubits: 4 });
  state.h(0).h(1).h(2).h(3);
  console.log('Created uniform superposition');
  state.dispose();
}

main();
[archived fence delimiter: ```]

## QuantumState Class

### Factory Method

QuantumState uses an async factory pattern (constructor is private).

[archived fence delimiter: ```typescript]
class QuantumState {
    static async create(options: QuantumStateOptions): Promise<QuantumState>;
}

interface QuantumStateOptions {
    numQubits: number;       // 1-30 qubits
    amplitudes?: Complex[];  // Optional initial amplitudes
}
[archived fence delimiter: ```]

**Parameters**:
- `numQubits`: Number of qubits (1-28 in browser, 1-30 in Node.js)

**Throws**: `Error` if memory allocation fails or qubits out of range

### Properties

[archived fence delimiter: ```typescript]
interface QuantumState {
    readonly numQubits: number;
    readonly stateDim: number;
}
[archived fence delimiter: ```]

### Gate Methods

All gate methods return `this` for chaining.

#### Single-Qubit Gates

[archived fence delimiter: ```typescript]
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
[archived fence delimiter: ```]

#### Rotation Gates

[archived fence delimiter: ```typescript]
class QuantumState {
    rx(qubit: number, theta: number): this;
    ry(qubit: number, theta: number): this;
    rz(qubit: number, theta: number): this;
    phase(qubit: number, theta: number): this;
}
[archived fence delimiter: ```]

#### Multi-Qubit Gates

[archived fence delimiter: ```typescript]
class QuantumState {
    cnot(control: number, target: number): this;
    cz(control: number, target: number): this;
    swap(qubit1: number, qubit2: number): this;
    cphase(control: number, target: number, theta: number): this;
    toffoli(control1: number, control2: number, target: number): this;
}
[archived fence delimiter: ```]

### State Access

[archived fence delimiter: ```typescript]
class QuantumState {
    // Get all amplitudes as Complex[]
    getAmplitudes(): Complex[];

    // Get probabilities
    getProbabilities(): Float64Array;

    // Get single probability
    probability(basisState: number): number;
}
[archived fence delimiter: ```]

### Measurement

[archived fence delimiter: ```typescript]
class QuantumState {
    // Measure single qubit (collapses state)
    measure(qubit: number): number;  // Returns 0 or 1

    // Measure all qubits (collapses state)
    measureAll(): number;  // Returns basis state index
}
[archived fence delimiter: ```]

### State Operations

[archived fence delimiter: ```typescript]
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
[archived fence delimiter: ```]

## TypeScript Support

Full TypeScript definitions included:

[archived fence delimiter: ```typescript]
import { QuantumState, Complex } from '@moonlab/quantum-core';

interface Complex {
    re: number;
    im: number;
}

// Async usage required
const state = await QuantumState.create({ numQubits: 4 });
const amps: Complex[] = state.getAmplitudes();
const prob: number = state.probability(0);
[archived fence delimiter: ```]

## Visualization (@moonlab/quantum-viz)

### CircuitDiagram

[archived fence delimiter: ```typescript]
import { CircuitDiagram } from '@moonlab/quantum-viz';

const diagram = new CircuitDiagram('#container', {
    numQubits: 3,
    width: 600,
    height: 200
});

diagram.addGate('H', 0);
diagram.addGate('CNOT', 0, 1);
diagram.addGate('CNOT', 1, 2);
diagram.render();
[archived fence delimiter: ```]

### StateVisualization

[archived fence delimiter: ```typescript]
import { StateVisualization } from '@moonlab/quantum-viz';

const viz = new StateVisualization('#container', state, {
    type: 'bar',  // 'bar' | 'bloch' | 'city'
    width: 400,
    height: 300
});

// Update when state changes
state.h(0);
viz.update(state);
[archived fence delimiter: ```]

### BlochSphere

[archived fence delimiter: ```typescript]
import { BlochSphere } from '@moonlab/quantum-viz';

const bloch = new BlochSphere('#container', {
    width: 300,
    height: 300,
    showAxes: true
});

bloch.setState(state, 0);  // Show qubit 0
[archived fence delimiter: ```]

## React Components (@moonlab/quantum-react)

### QuantumCircuit

[archived fence delimiter: ```jsx]
import { QuantumCircuit, Gate, Measure } from '@moonlab/quantum-react';

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
[archived fence delimiter: ```]

### useQuantumState Hook

[archived fence delimiter: ```jsx]
import { useQuantumState } from '@moonlab/quantum-react';

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
[archived fence delimiter: ```]

### StateVector

[archived fence delimiter: ```jsx]
import { StateVector } from '@moonlab/quantum-react';

<StateVector
    state={state}
    format="polar"  // 'cartesian' | 'polar' | 'probability'
    precision={4}
/>
[archived fence delimiter: ```]

## Vue Components (@moonlab/quantum-vue)

### QuantumCircuit

[archived fence delimiter: ```vue]
<template>
    <QuantumCircuit :qubits="3" @measure="onMeasure">
        <Gate type="H" :qubit="0" />
        <Gate type="CNOT" :control="0" :target="1" />
        <Measure :qubits="[0, 1, 2]" />
    </QuantumCircuit>
</template>

<script setup>
import { QuantumCircuit, Gate, Measure } from '@moonlab/quantum-vue';

const onMeasure = (result) => {
    console.log('Measured:', result);
};
</script>
[archived fence delimiter: ```]

### useQuantum Composable

[archived fence delimiter: ```vue]
<script setup>
import { useQuantum } from '@moonlab/quantum-vue';

const { state, h, cnot, measure, probabilities } = useQuantum(2);

const createBellState = () => {
    h(0);
    cnot(0, 1);
};
</script>
[archived fence delimiter: ```]

## Async Initialization

The factory pattern handles WASM initialization automatically:

[archived fence delimiter: ```javascript]
import { QuantumState, preload } from '@moonlab/quantum-core';

async function simulate() {
    // Optional: preload WASM module for faster first use
    await preload();

    const state = await QuantumState.create({ numQubits: 20 });
    // ... operations ...
    state.dispose();
}
[archived fence delimiter: ```]

## Web Worker Support

Run simulations in a Web Worker:

[archived fence delimiter: ```javascript]
// worker.js
import { QuantumState } from '@moonlab/quantum-core';

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
[archived fence delimiter: ```]

## Memory Management

WebAssembly requires manual memory management:

[archived fence delimiter: ```javascript]
const state = await QuantumState.create({ numQubits: 10 });

try {
    // ... operations ...
} finally {
    state.dispose();  // Always dispose!
}
[archived fence delimiter: ```]

With async/await pattern:
[archived fence delimiter: ```javascript]
async function simulate() {
    const state = await QuantumState.create({ numQubits: 10 });
    try {
        state.h(0);
        return state.measureAll();
    } finally {
        state.dispose();
    }
}
[archived fence delimiter: ```]

## Browser Compatibility

| Browser | Version | Notes |
|---------|---------|-------|
| Chrome | 57+ | Full support |
| Firefox | 52+ | Full support |
| Safari | 11+ | Full support |
| Edge | 79+ | Full support |

Requires WebAssembly support. Check with:
[archived fence delimiter: ```javascript]
const wasmSupported = typeof WebAssembly !== 'undefined';
[archived fence delimiter: ```]

## Bundle Size

| Package | Size (gzipped) |
|---------|----------------|
| @moonlab/quantum-core | ~150 KB |
| @moonlab/quantum-viz | ~45 KB |
| @moonlab/quantum-react | ~12 KB |
| @moonlab/quantum-vue | ~10 KB |

## See Also

- [Core API](core.md) - Complete reference
- [Visualization](viz.md) - Chart and diagram APIs
- [React Components](react.md) - React integration
- [Vue Components](vue.md) - Vue integration
- [Demo Site](https://moonlab.dev/demo) - Interactive examples
```
