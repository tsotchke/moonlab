# Vue Components API

Vue.js components for quantum circuit visualization and interaction.

## Installation

```bash
npm install @moonlab/vue
```

```javascript
// main.js
import { createApp } from 'vue'
import MoonlabVue from '@moonlab/vue'
import App from './App.vue'

const app = createApp(App)
app.use(MoonlabVue)
app.mount('#app')
```

## Components

### `<QuantumCircuit>`

Interactive quantum circuit editor and visualizer.

#### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `qubits` | `number` | `4` | Number of qubits |
| `gates` | `Gate[]` | `[]` | Initial gate list |
| `editable` | `boolean` | `true` | Allow editing |
| `showMeasurements` | `boolean` | `true` | Show measurement gates |
| `theme` | `'light' \| 'dark'` | `'light'` | Color theme |
| `gridSize` | `number` | `40` | Grid cell size in pixels |

#### Events

| Event | Payload | Description |
|-------|---------|-------------|
| `@gate-added` | `{ gate, position }` | Gate added to circuit |
| `@gate-removed` | `{ gate, position }` | Gate removed |
| `@gate-moved` | `{ gate, from, to }` | Gate moved |
| `@circuit-changed` | `Gate[]` | Circuit modified |
| `@run` | `void` | Run button clicked |

#### Slots

| Slot | Props | Description |
|------|-------|-------------|
| `toolbar` | `{ addGate, clear, run }` | Custom toolbar |
| `gate` | `{ gate, x, y }` | Custom gate rendering |

#### Example

```vue
<template>
  <QuantumCircuit
    :qubits="4"
    :gates="initialGates"
    editable
    theme="dark"
    @circuit-changed="onCircuitChange"
    @run="runSimulation"
  >
    <template #toolbar="{ addGate, clear, run }">
      <button @click="addGate('H', 0)">Add H</button>
      <button @click="clear">Clear</button>
      <button @click="run">Run</button>
    </template>
  </QuantumCircuit>
</template>

<script setup>
import { ref } from 'vue'
import { QuantumCircuit } from '@moonlab/vue'

const initialGates = ref([
  { type: 'H', qubit: 0, step: 0 },
  { type: 'CNOT', control: 0, target: 1, step: 1 }
])

function onCircuitChange(gates) {
  console.log('Circuit updated:', gates)
}

async function runSimulation() {
  // Call simulation API
}
</script>
```

### `<BlochSphere>`

3D Bloch sphere visualization for single-qubit states.

#### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `state` | `[number, number]` | `[1, 0]` | State vector [α, β] |
| `theta` | `number` | `0` | Polar angle (alternative to state) |
| `phi` | `number` | `0` | Azimuthal angle |
| `size` | `number` | `300` | Component size in pixels |
| `showAxes` | `boolean` | `true` | Show X, Y, Z axes |
| `showLabels` | `boolean` | `true` | Show axis labels |
| `animate` | `boolean` | `false` | Animate state changes |
| `interactive` | `boolean` | `false` | Allow drag rotation |

#### Events

| Event | Payload | Description |
|-------|---------|-------------|
| `@state-changed` | `{ theta, phi }` | User changed state |
| `@rotation` | `{ rx, ry }` | Sphere rotated |

#### Example

```vue
<template>
  <BlochSphere
    :state="qubitState"
    :size="400"
    animate
    interactive
    @state-changed="onStateChange"
  />
</template>

<script setup>
import { ref, computed } from 'vue'
import { BlochSphere } from '@moonlab/vue'

const theta = ref(Math.PI / 4)
const phi = ref(0)

const qubitState = computed(() => {
  const alpha = Math.cos(theta.value / 2)
  const beta = Math.sin(theta.value / 2) * Math.exp(1i * phi.value)
  return [alpha, beta]
})

function onStateChange({ theta: t, phi: p }) {
  theta.value = t
  phi.value = p
}
</script>
```

### `<StateVector>`

Visualize quantum state amplitudes.

#### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `amplitudes` | `Complex[]` | Required | State amplitudes |
| `format` | `'bar' \| 'table' \| 'circle'` | `'bar'` | Display format |
| `showPhase` | `boolean` | `true` | Show phase information |
| `binaryLabels` | `boolean` | `true` | Use binary basis labels |
| `precision` | `number` | `4` | Decimal precision |
| `maxBasisStates` | `number` | `32` | Max states to show |

#### Events

| Event | Payload | Description |
|-------|---------|-------------|
| `@basis-click` | `{ index, label }` | Basis state clicked |

#### Example

```vue
<template>
  <StateVector
    :amplitudes="state.amplitudes"
    format="bar"
    showPhase
    @basis-click="onBasisClick"
  />
</template>

<script setup>
import { StateVector } from '@moonlab/vue'

function onBasisClick({ index, label }) {
  console.log(`Clicked basis state ${label} (index ${index})`)
}
</script>
```

### `<ProbabilityDistribution>`

Bar chart of measurement probabilities.

#### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `probabilities` | `number[]` | Required | Probability values |
| `labels` | `string[]` | Auto | Basis state labels |
| `threshold` | `number` | `0.001` | Hide below threshold |
| `sorted` | `boolean` | `false` | Sort by probability |
| `barColor` | `string` | `'#3b82f6'` | Bar fill color |
| `height` | `number` | `200` | Chart height |

#### Example

```vue
<template>
  <ProbabilityDistribution
    :probabilities="probs"
    sorted
    :threshold="0.01"
  />
</template>

<script setup>
import { ref } from 'vue'
import { ProbabilityDistribution } from '@moonlab/vue'

const probs = ref([0.5, 0.0, 0.0, 0.5])  // Bell state
</script>
```

### `<MeasurementHistogram>`

Histogram of measurement results.

#### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `counts` | `Record<string, number>` | Required | Count per outcome |
| `shots` | `number` | Sum of counts | Total shots |
| `showPercentage` | `boolean` | `true` | Show percentages |
| `maxBars` | `number` | `16` | Maximum bars |
| `sorted` | `boolean` | `true` | Sort by count |

#### Example

```vue
<template>
  <MeasurementHistogram
    :counts="measurementCounts"
    :shots="1000"
    showPercentage
    sorted
  />
</template>

<script setup>
import { ref } from 'vue'
import { MeasurementHistogram } from '@moonlab/vue'

const measurementCounts = ref({
  '00': 512,
  '11': 488
})
</script>
```

### `<GatePalette>`

Draggable gate palette for circuit building.

#### Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `gates` | `string[]` | All gates | Available gates |
| `columns` | `number` | `4` | Grid columns |
| `draggable` | `boolean` | `true` | Enable drag |

#### Events

| Event | Payload | Description |
|-------|---------|-------------|
| `@gate-selected` | `string` | Gate type selected |
| `@gate-drag-start` | `{ type, event }` | Drag started |

#### Example

```vue
<template>
  <GatePalette
    :gates="['H', 'X', 'Y', 'Z', 'CNOT', 'CZ', 'SWAP']"
    @gate-selected="onGateSelect"
  />
</template>

<script setup>
import { GatePalette } from '@moonlab/vue'

function onGateSelect(gateType) {
  console.log('Selected gate:', gateType)
}
</script>
```

## Composables

### `useQuantumState`

Reactive quantum state management.

```vue
<script setup>
import { useQuantumState } from '@moonlab/vue'

const { state, apply, measure, reset, amplitudes, probabilities } = useQuantumState(4)

// Apply gates
apply('H', 0)
apply('CNOT', 0, 1)

// Get reactive data
console.log(amplitudes.value)
console.log(probabilities.value)

// Measure
const result = await measure()

// Reset to |0...0⟩
reset()
</script>
```

#### Returns

| Property | Type | Description |
|----------|------|-------------|
| `state` | `Ref<QuantumState>` | State object |
| `apply` | `(gate, ...qubits) => void` | Apply gate |
| `measure` | `(qubit?) => Promise<number>` | Measure qubit(s) |
| `reset` | `() => void` | Reset to ground state |
| `amplitudes` | `ComputedRef<Complex[]>` | State amplitudes |
| `probabilities` | `ComputedRef<number[]>` | Probabilities |
| `entanglementEntropy` | `ComputedRef<number>` | Entropy of qubit 0 |

### `useCircuitBuilder`

Circuit building and manipulation.

```vue
<script setup>
import { useCircuitBuilder } from '@moonlab/vue'

const {
  gates,
  addGate,
  removeGate,
  moveGate,
  clear,
  toQASM,
  fromQASM,
  undo,
  redo
} = useCircuitBuilder(4)

// Build circuit
addGate('H', { qubit: 0, step: 0 })
addGate('CNOT', { control: 0, target: 1, step: 1 })

// Export to QASM
const qasm = toQASM()

// Undo/redo
undo()
redo()
</script>
```

### `useSimulation`

Manage simulation execution.

```vue
<script setup>
import { useSimulation } from '@moonlab/vue'

const {
  isRunning,
  progress,
  result,
  error,
  run,
  cancel
} = useSimulation()

async function runCircuit(gates) {
  const result = await run(gates, { shots: 1000 })
  console.log(result.counts)
}
</script>
```

## Directives

### `v-gate-drop`

Enable gate dropping on elements.

```vue
<template>
  <div
    v-gate-drop="{ onDrop: handleGateDrop }"
    class="drop-zone"
  >
    Drop gates here
  </div>
</template>

<script setup>
function handleGateDrop(gate, position) {
  console.log(`Dropped ${gate.type} at position`, position)
}
</script>
```

### `v-circuit-keyboard`

Keyboard shortcuts for circuit editing.

```vue
<template>
  <div v-circuit-keyboard="circuitRef">
    <QuantumCircuit ref="circuitRef" />
  </div>
</template>
```

Shortcuts:
- `Delete/Backspace`: Remove selected gate
- `Ctrl+Z`: Undo
- `Ctrl+Y`: Redo
- `Ctrl+C`: Copy gate
- `Ctrl+V`: Paste gate

## Theming

### Custom Theme

```vue
<script setup>
import { MoonlabTheme } from '@moonlab/vue'

const customTheme = {
  colors: {
    background: '#1a1a2e',
    foreground: '#eaeaea',
    primary: '#00d9ff',
    gate: {
      H: '#ff6b6b',
      X: '#4ecdc4',
      CNOT: '#45b7d1'
    }
  },
  fonts: {
    main: 'Inter, sans-serif',
    mono: 'Fira Code, monospace'
  }
}
</script>

<template>
  <MoonlabTheme :theme="customTheme">
    <QuantumCircuit :qubits="4" />
  </MoonlabTheme>
</template>
```

### Dark Mode

```vue
<template>
  <MoonlabTheme :dark="isDark">
    <QuantumCircuit theme="auto" />
  </MoonlabTheme>
</template>

<script setup>
import { useDark } from '@vueuse/core'

const isDark = useDark()
</script>
```

## TypeScript Support

Full TypeScript definitions included:

```typescript
import type {
  Gate,
  GateType,
  CircuitProps,
  StateVectorProps,
  BlochSphereProps,
  Complex
} from '@moonlab/vue'

const gate: Gate = {
  type: 'CNOT' as GateType,
  control: 0,
  target: 1,
  step: 0
}
```

## Complete Example

```vue
<template>
  <div class="quantum-app">
    <div class="sidebar">
      <GatePalette @gate-selected="selectedGate = $event" />
    </div>

    <div class="main">
      <QuantumCircuit
        ref="circuit"
        :qubits="numQubits"
        :gates="gates"
        @circuit-changed="gates = $event"
      />

      <div class="controls">
        <button @click="runSimulation" :disabled="isRunning">
          {{ isRunning ? 'Running...' : 'Run' }}
        </button>
        <button @click="circuit.clear()">Clear</button>
      </div>
    </div>

    <div class="results" v-if="result">
      <h3>Results</h3>
      <MeasurementHistogram :counts="result.counts" :shots="1000" />
      <StateVector :amplitudes="result.finalState" format="bar" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import {
  QuantumCircuit,
  GatePalette,
  MeasurementHistogram,
  StateVector,
  useSimulation
} from '@moonlab/vue'
import type { Gate } from '@moonlab/vue'

const numQubits = ref(4)
const gates = ref<Gate[]>([])
const selectedGate = ref<string | null>(null)
const circuit = ref()

const { isRunning, result, run } = useSimulation()

async function runSimulation() {
  await run(gates.value, {
    qubits: numQubits.value,
    shots: 1000
  })
}
</script>

<style scoped>
.quantum-app {
  display: grid;
  grid-template-columns: 200px 1fr 300px;
  gap: 1rem;
  height: 100vh;
}
</style>
```

## See Also

- [JavaScript Core API](core.md) - Core WASM bindings
- [React Components](react.md) - React version
- [Visualization API](viz.md) - Rendering primitives
- [Vue.js Documentation](https://vuejs.org/) - Vue.js framework

