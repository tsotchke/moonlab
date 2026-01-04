# JavaScript Core API

Complete reference for the WebAssembly quantum simulation core.

**Package**: `@moonlab/quantum-core`

## Overview

The `@moonlab/quantum-core` package provides WebAssembly-powered quantum simulation for JavaScript/TypeScript applications. Features:

- **WebAssembly Performance**: Near-native speed in the browser
- **Fluent API**: Method chaining for circuit building
- **Full Gate Set**: Complete universal quantum gate library
- **Complex Number Support**: Full complex arithmetic utilities
- **Memory Safe**: Automatic WASM memory management

## Installation

```bash
npm install @moonlab/quantum-core
# or
yarn add @moonlab/quantum-core
# or
pnpm add @moonlab/quantum-core
```

## Quick Start

```typescript
import { QuantumState, Circuit } from '@moonlab/quantum-core';

// Create a 2-qubit Bell state using fluent API
const state = await QuantumState.create({ numQubits: 2 });
state.h(0).cnot(0, 1);

console.log(state.getProbabilities());  // Float64Array [0.5, 0, 0, 0.5]

// IMPORTANT: Dispose when done
state.dispose();

// Or use Circuit builder for reusable circuits
const circuit = new Circuit(2)
  .h(0)
  .cnot(0, 1);

const state2 = await QuantumState.create({ numQubits: 2 });
circuit.apply(state2);
state2.dispose();
```

## QuantumState

Main class for quantum state manipulation.

### Static Methods

#### create

```typescript
static async create(options: QuantumStateOptions): Promise<QuantumState>
```

Create a new quantum state initialized to $|0\ldots0\rangle$.

**Parameters**:
```typescript
interface QuantumStateOptions {
  numQubits: number;       // 1-30 qubits
  amplitudes?: Complex[];  // Optional initial amplitudes
}
```

**Returns**: Promise resolving to QuantumState

**Example**:
```typescript
// Basic creation
const state = await QuantumState.create({ numQubits: 4 });

// With initial amplitudes
const bellState = await QuantumState.create({
  numQubits: 2,
  amplitudes: [
    { re: 1/Math.sqrt(2), im: 0 },  // |00⟩
    { re: 0, im: 0 },                // |01⟩
    { re: 0, im: 0 },                // |10⟩
    { re: 1/Math.sqrt(2), im: 0 },  // |11⟩
  ]
});
```

### Properties

```typescript
class QuantumState {
  readonly numQubits: number;    // Number of qubits
  readonly stateDim: number;     // State dimension (2^n)
  readonly isDisposed: boolean;  // Whether disposed
}
```

### State Operations

#### reset

```typescript
reset(): this
```

Reset state to $|0\ldots0\rangle$.

```typescript
state.h(0).cnot(0, 1);
state.reset();  // Back to |00⟩
```

#### clone

```typescript
async clone(): Promise<QuantumState>
```

Create a deep copy of the state.

```typescript
const copy = await state.clone();
// state and copy are independent
```

#### normalize

```typescript
normalize(): this
```

Normalize the state vector to unit length.

### Single-Qubit Gates

All gates return `this` for method chaining.

```typescript
// Pauli gates
state.x(qubit);    // Pauli-X (NOT)
state.y(qubit);    // Pauli-Y
state.z(qubit);    // Pauli-Z

// Hadamard
state.h(qubit);    // Creates superposition

// Phase gates
state.s(qubit);    // S gate (√Z)
state.sdg(qubit);  // S† gate
state.t(qubit);    // T gate (π/8)
state.tdg(qubit);  // T† gate

// Rotation gates
state.rx(qubit, angle);  // X-axis rotation
state.ry(qubit, angle);  // Y-axis rotation
state.rz(qubit, angle);  // Z-axis rotation
state.phase(qubit, angle);  // Phase gate

// General unitary
state.u3(qubit, theta, phi, lambda);  // U3 gate
```

**Example**:
```typescript
const state = await QuantumState.create({ numQubits: 3 });

// Method chaining
state
  .h(0)
  .rx(1, Math.PI / 4)
  .t(2)
  .phase(0, Math.PI / 2);
```

### Two-Qubit Gates

```typescript
// Controlled gates
state.cnot(control, target);   // CNOT (CX)
state.cx(control, target);     // Alias for CNOT
state.cz(control, target);     // Controlled-Z
state.cy(control, target);     // Controlled-Y

// SWAP
state.swap(qubit1, qubit2);

// Controlled rotations
state.crx(control, target, angle);
state.cry(control, target, angle);
state.crz(control, target, angle);
state.cphase(control, target, angle);
```

### Three-Qubit Gates

```typescript
// Toffoli (CCNOT)
state.toffoli(control1, control2, target);
state.ccx(control1, control2, target);  // Alias

// Fredkin (CSWAP)
state.fredkin(control, target1, target2);
state.cswap(control, target1, target2);  // Alias
```

### Multi-Qubit Operations

```typescript
// Quantum Fourier Transform
state.qft([0, 1, 2, 3]);    // Apply to qubits 0-3
state.iqft([0, 1, 2, 3]);   // Inverse QFT
```

### State Queries

#### getAmplitudes

```typescript
getAmplitudes(): Complex[]
```

Get all complex amplitudes.

```typescript
const amps = state.getAmplitudes();
console.log(amps[0]);  // { re: number, im: number }
```

#### setAmplitudes

```typescript
setAmplitudes(amplitudes: Complex[]): this
```

Set state amplitudes (must match state dimension).

#### getProbabilities

```typescript
getProbabilities(): Float64Array
```

Get probability distribution.

```typescript
const probs = state.getProbabilities();
// probs[i] = probability of measuring |i⟩
```

#### probability

```typescript
probability(basisState: number): number
```

Get probability of specific basis state.

```typescript
const p00 = state.probability(0);  // P(|00⟩)
const p11 = state.probability(3);  // P(|11⟩) for 2 qubits
```

### Measurement

#### measure

```typescript
measure(qubit: number): number
```

Measure single qubit (collapses state).

**Returns**: 0 or 1

```typescript
const result = state.measure(0);
console.log(`Qubit 0 measured: ${result}`);
```

#### measureAll

```typescript
measureAll(): number
```

Measure all qubits (collapses state).

**Returns**: Basis state index (0 to 2^n - 1)

```typescript
const result = state.measureAll();
console.log(`Measured: |${result.toString(2).padStart(state.numQubits, '0')}⟩`);
```

#### probabilityZero / probabilityOne

```typescript
probabilityZero(qubit: number): number
probabilityOne(qubit: number): number
```

Get single-qubit measurement probabilities (non-destructive).

### Expectation Values

```typescript
// Single-qubit Pauli expectations
const zExp = state.expectationZ(qubit);  // ⟨Z⟩
const xExp = state.expectationX(qubit);  // ⟨X⟩
const yExp = state.expectationY(qubit);  // ⟨Y⟩

// Two-qubit correlation
const zzCorr = state.correlationZZ(qubit1, qubit2);  // ⟨Z₁Z₂⟩
```

### State Properties

```typescript
// Von Neumann entropy
const s = state.entropy();

// Purity (1 for pure states)
const p = state.purity();

// Fidelity with another state
const f = state.fidelity(otherState);
```

### Memory Management

```typescript
dispose(): void
```

**IMPORTANT**: Must be called when done with state to free WASM memory.

```typescript
const state = await QuantumState.create({ numQubits: 4 });
try {
  state.h(0).cnot(0, 1);
  // ... use state
} finally {
  state.dispose();
}
```

## Circuit

Reusable circuit builder.

### Constructor

```typescript
new Circuit(numQubits: number)
```

### Gate Methods

Same as QuantumState but builds circuit without executing.

```typescript
const circuit = new Circuit(3)
  .h(0)
  .cnot(0, 1)
  .cnot(1, 2);
```

### apply

```typescript
apply(state: QuantumState): void
```

Apply circuit to a quantum state.

```typescript
const state = await QuantumState.create({ numQubits: 3 });
circuit.apply(state);
```

### Circuit Statistics

```typescript
interface CircuitStats {
  numGates: number;
  depth: number;
  singleQubitGates: number;
  twoQubitGates: number;
  threeQubitGates: number;
}

const stats = circuit.getStats();
```

## Complex Numbers

Utilities for complex arithmetic.

### Types

```typescript
interface Complex {
  re: number;  // Real part
  im: number;  // Imaginary part
}
```

### Constants

```typescript
import { ZERO, ONE, I } from '@moonlab/quantum-core';

ZERO  // { re: 0, im: 0 }
ONE   // { re: 1, im: 0 }
I     // { re: 0, im: 1 }
```

### Creation

```typescript
import { complex, fromPolar } from '@moonlab/quantum-core';

const c = complex(1, 2);  // 1 + 2i
const p = fromPolar(1, Math.PI / 4);  // e^(iπ/4)
```

### Operations

```typescript
import {
  add, subtract, multiply, divide, scale,
  conjugate, magnitude, magnitudeSquared, phase,
  exp, equals, toString
} from '@moonlab/quantum-core';

const a = complex(1, 2);
const b = complex(3, 4);

add(a, b);           // a + b
subtract(a, b);      // a - b
multiply(a, b);      // a * b
divide(a, b);        // a / b
scale(a, 2);         // 2a
conjugate(a);        // a*
magnitude(a);        // |a|
magnitudeSquared(a); // |a|²
phase(a);            // arg(a)
exp(a);              // e^a
equals(a, b);        // a == b
toString(a);         // "1 + 2i"
```

### Array Operations

```typescript
import { innerProduct, norm, normalize } from '@moonlab/quantum-core';

const v = [complex(1, 0), complex(0, 1)];

innerProduct(v, v);  // ⟨v|v⟩
norm(v);             // ||v||
normalize(v);        // v / ||v||
```

## WASM Module Management

### Preloading

```typescript
import { preload, isLoaded } from '@moonlab/quantum-core';

// Preload WASM module
await preload();

// Check if loaded
if (isLoaded()) {
  // Module ready
}
```

### Load Options

```typescript
interface LoadOptions {
  wasmPath?: string;  // Custom path to WASM file
}

await preload({ wasmPath: '/custom/path/moonlab.wasm' });
```

## TypeScript Types

```typescript
// Basis state as binary string
type BasisState = string;  // e.g., "00", "01", "10", "11"

// Measurement result
interface MeasurementResult {
  basisState: number;
  bitString: BasisState;
  probability: number;
}

// State vector
type StateVector = Complex[];

// Probability distribution
type ProbabilityDistribution = Float64Array;
```

## Complete Example

```typescript
import { QuantumState, Circuit, complex } from '@moonlab/quantum-core';

async function quantumTeleportation() {
  // Create 3-qubit state
  const state = await QuantumState.create({ numQubits: 3 });

  // Prepare state to teleport on qubit 0
  state.h(0).t(0);

  // Create entangled pair between qubits 1 and 2
  state.h(1).cnot(1, 2);

  // Bell measurement on qubits 0 and 1
  state.cnot(0, 1).h(0);

  // Measure qubits 0 and 1
  const m0 = state.measure(0);
  const m1 = state.measure(1);

  // Apply corrections to qubit 2
  if (m1 === 1) state.x(2);
  if (m0 === 1) state.z(2);

  // Qubit 2 now has the original state from qubit 0
  console.log('Teleportation complete!');
  console.log(`Measurements: m0=${m0}, m1=${m1}`);
  console.log('Final probabilities:', state.getProbabilities());

  state.dispose();
}

quantumTeleportation();
```

## Browser Support

| Browser | Minimum Version |
|---------|-----------------|
| Chrome | 57+ |
| Firefox | 52+ |
| Safari | 11+ |
| Edge | 16+ |

Requires WebAssembly support.

## See Also

- [Visualization API](viz.md) - Quantum state visualizations
- [React Integration](react.md) - React hooks and components
- [Vue Integration](../api/javascript/vue.md) - Vue composables

