import React, { useState } from 'react';
import {
  ensureMoonlabWorker,
  runCircuitInWorker,
  runExampleAlgorithmInWorker,
  type WorkerGate,
} from '../workers/moonlabClient';
import './Examples.css';

interface Example {
  id: string;
  title: string;
  description: string;
  code: string;
  runnable?: boolean;
}

const EXAMPLES: Example[] = [
  {
    id: 'bell-state',
    title: 'Bell State (Entanglement)',
    description: 'Create a maximally entangled Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2',
    code: `import { QuantumState } from '@moonlab/quantum-core';

// Create a 2-qubit state
const state = await QuantumState.create({ numQubits: 2 });

// Apply Hadamard to qubit 0 (creates superposition)
state.h(0);

// Apply CNOT with qubit 0 as control, qubit 1 as target
state.cnot(0, 1);

// Now the qubits are entangled!
const probs = state.getProbabilities();
console.log(probs);  // [0.5, 0, 0, 0.5]

// Measuring qubit 0 instantly determines qubit 1
state.dispose();`,
    runnable: true,
  },
  {
    id: 'superposition',
    title: 'Uniform Superposition',
    description: 'Create equal superposition over all basis states using Hadamard gates',
    code: `import { QuantumState } from '@moonlab/quantum-core';

// Create a 3-qubit state (8 basis states)
const state = await QuantumState.create({ numQubits: 3 });

// Apply Hadamard to all qubits
state.h(0).h(1).h(2);

// Each of the 8 states has equal probability
const probs = state.getProbabilities();
console.log(probs);  // [0.125, 0.125, ..., 0.125]

// Verify normalization
const total = probs.reduce((a, b) => a + b, 0);
console.log(\`Total probability: \${total}\`);  // 1.0

state.dispose();`,
    runnable: true,
  },
  {
    id: 'grover',
    title: "Grover's Search Algorithm",
    description: 'Find a marked item in an unsorted database with quadratic speedup',
    code: `import { QuantumState } from '@moonlab/quantum-core';
import { Grover } from '@moonlab/quantum-algorithms';

// Search in a space of 1024 items (10 qubits)
const grover = await Grover.create({
  numQubits: 10,
  markedState: 42,  // The item we're searching for
});

// Run the search (uses optimal number of iterations)
const result = grover.search();

console.log(\`Found: \${result.foundState}\`);          // 42
console.log(\`Success prob: \${result.successProbability}\`);  // ~96.9%
console.log(\`Oracle calls: \${result.oracleCalls}\`);  // ~25 (vs 1024 classically!)

grover.dispose();`,
    runnable: true,
  },
  {
    id: 'phase-kickback',
    title: 'Phase Kickback',
    description: 'Demonstrate the phase kickback phenomenon used in quantum algorithms',
    code: `import { QuantumState } from '@moonlab/quantum-core';

// Create 2-qubit state
const state = await QuantumState.create({ numQubits: 2 });

// Prepare control in superposition
state.h(0);

// Prepare target in |1⟩ state
state.x(1);

// Apply controlled-Z
state.cz(0, 1);

// The phase kicks back to the control qubit!
// Control qubit now has a relative phase

// Apply Hadamard to see the phase
state.h(0);

const probs = state.getProbabilities();
console.log(probs);

state.dispose();`,
    runnable: true,
  },
  {
    id: 'quantum-teleportation',
    title: 'Quantum Teleportation',
    description: 'Teleport a quantum state using entanglement and classical communication',
    code: `import { QuantumState } from '@moonlab/quantum-core';

// 3 qubits: q0=state to teleport, q1,q2=entangled pair
const state = await QuantumState.create({ numQubits: 3 });

// Prepare the state to teleport (arbitrary state on q0)
state.h(0).t(0);  // |ψ⟩ = H·T|0⟩

// Create Bell pair between q1 and q2
state.h(1).cnot(1, 2);

// Bell measurement on q0 and q1
state.cnot(0, 1).h(0);

// Measure q0 and q1 (classical bits)
const m0 = state.measure(0);
const m1 = state.measure(1);

// Apply corrections to q2 based on measurement
if (m1) state.x(2);
if (m0) state.z(2);

// q2 now contains the original state |ψ⟩!
console.log('Teleportation complete!');

state.dispose();`,
    runnable: true,
  },
  {
    id: 'vqe-h2',
    title: 'VQE: H₂ Ground State Energy',
    description: 'Find the ground state energy of hydrogen molecule using variational quantum eigensolver',
    code: `import { VQE, createH2Hamiltonian } from '@moonlab/quantum-algorithms';

// Create H2 Hamiltonian at bond distance 0.74 Å
const hamiltonian = createH2Hamiltonian({
  bondDistance: 0.74,
  basis: 'sto-3g'
});

// Create VQE solver
const vqe = await VQE.create({
  hamiltonian,
  ansatz: 'uccsd',     // Unitary Coupled Cluster
  optimizer: 'cobyla',
  maxIterations: 100
});

// Run optimization
const result = vqe.solve();

console.log(\`Ground state energy: \${result.energy} Ha\`);
console.log(\`Chemical accuracy: \${result.chemicalAccuracy}\`);
console.log(\`Iterations: \${result.iterations}\`);

vqe.dispose();`,
    runnable: true,
  },
  {
    id: 'ca-mps-stabilizer',
    title: 'CA-MPS: Stabilizer Circuit',
    description: 'Pure-Clifford GHZ state via Clifford-Assisted MPS. The Clifford structure goes into the tableau (O(n) bit ops); the MPS stays at bond dimension 1. 64x bond-dim advantage and 13884x speedup vs plain MPS at n=12 (since v0.2.1).',
    code: `import { CaMps } from '@moonlab/quantum-core';

// 12-qubit CA-MPS, max MPS bond dim 32. The Clifford prefactor D
// stores the Aaronson-Gottesman tableau; |phi> stays at bond 1
// for any pure-Clifford circuit.
const state = await CaMps.create(12, 32);

// 12-qubit GHZ: H on qubit 0, then a CNOT chain.
state.h(0);
for (let q = 0; q + 1 < 12; q++) state.cnot(q, q + 1);

console.log('num_qubits =', state.numQubits);
console.log('bond_dim   =', state.bondDim);   // expected: 1
console.log('norm       =', state.norm);       // expected: 1.0

state.dispose();`,
    runnable: false,
  },
  {
    id: 'gauge-warmstart-bell',
    title: 'Gauge-Aware Warmstart: Bell stabilizers',
    description: 'Aaronson-Gottesman symplectic-Gauss-Jordan Clifford prep on the abelian stabilizer subgroup {XX, ZZ}. The resulting state is in the simultaneous +1 eigenspace of every generator -- the Bell state |Phi+>. Generalises to LGT Gauss-law operators, surface/toric/repetition codes (since v0.2.1).',
    code: `import { CaMps, gaugeWarmstart } from '@moonlab/quantum-core';

// Bell-pair stabilizer subgroup S = {XX, ZZ}.
// Pauli-byte encoding: 0=I, 1=X, 2=Y, 3=Z.
const generators = new Uint8Array([
  1, 1,    // X X
  3, 3,    // Z Z
]);

const state = await CaMps.create(2, 8);
gaugeWarmstart(state, generators, /*numGens=*/2);

// state.D|00> is now in the +1 eigenspace of both XX and ZZ,
// i.e. the Bell state (|00> + |11>) / sqrt(2).
console.log('norm =', state.norm);   // 1.0
state.dispose();`,
    runnable: false,
  },
  {
    id: 'z2-lgt-build',
    title: 'Z2 Lattice Gauge Theory: Pauli sum builder',
    description: 'Build the 1+1D Z2 LGT Hamiltonian on N matter sites. Exactly gauge-invariant kinetic terms (XYY/YYX form) -- each piece commutes with every interior Gauss-law operator G_x = X_{2x-1} Z_{2x} X_{2x+1}. First HEP application of the gauge-aware warmstart (since v0.2.1).',
    code: `import { z2Lgt1dBuild, z2Lgt1dGaussLaw } from '@moonlab/quantum-core';

// N = 4 matter sites -> 7 qubits (4 matter + 3 link).
const ham = await z2Lgt1dBuild(4, /*t=*/1.0, /*h=*/0.5, /*m=*/0.0,
                                  /*gauss_penalty=*/0.0);

console.log('num_qubits =', ham.numQubits);  // 7
console.log('num_terms  =', ham.numTerms);   // matter + electric + mass

// Interior Gauss-law operator at matter site x = 1.
const G1 = await z2Lgt1dGaussLaw(4, 1);
// Bytes are 0,1,3,1,0,0,0 -- X on qubit 1, Z on qubit 2,
// X on qubit 3, identity elsewhere.
console.log('G_1 =', Array.from(G1));`,
    runnable: false,
  },
];

const Examples: React.FC = () => {
  const logoUrl = `${import.meta.env.BASE_URL}ml-logo.png`;
  const [selectedExample, setSelectedExample] = useState<string>(EXAMPLES[0].id);
  const [copied, setCopied] = useState(false);
  const [runOutput, setRunOutput] = useState<Record<string, string>>({});
  const [runError, setRunError] = useState<Record<string, string>>({});
  const [runningId, setRunningId] = useState<string | null>(null);

  const currentExample = EXAMPLES.find(e => e.id === selectedExample) || EXAMPLES[0];

  const formatProbabilities = (probs: Float64Array | number[], numQubits: number): string => {
    const lines: string[] = [];
    for (let i = 0; i < probs.length; i++) {
      const prob = probs[i];
      if (prob < 0.0005) continue;
      const label = i.toString(2).padStart(numQubits, '0');
      lines.push(`|${label}⟩: ${(prob * 100).toFixed(2)}%`);
    }
    if (lines.length === 0) {
      return 'No non-zero probabilities found.';
    }
    return lines.join('\n');
  };

  const formatTopStates = (states: Array<{ bitstring: string; probability: number }>): string => {
    if (!states.length) return 'No high-probability states found.';
    return states
      .map((state) => `|${state.bitstring}⟩: ${(state.probability * 100).toFixed(2)}%`)
      .join('\n');
  };

  const runExample = async () => {
    if (!currentExample.runnable) return;
    setRunningId(currentExample.id);
    setRunError((prev) => ({ ...prev, [currentExample.id]: '' }));
    try {
      await ensureMoonlabWorker();
      let numQubits = 0;
      let gates: WorkerGate[] = [];

      if (
        currentExample.id === 'grover' ||
        currentExample.id === 'quantum-teleportation' ||
        currentExample.id === 'vqe-h2'
      ) {
        const response = await runExampleAlgorithmInWorker({
          id: currentExample.id as 'grover' | 'quantum-teleportation' | 'vqe-h2',
          cleanupAfterRun: true,
        });

        let output = '';
        if (response.algorithm === 'grover') {
          output = [
            `Marked state: ${response.markedState} (|${response.markedState.toString(2).padStart(response.numQubits, '0')}⟩)`,
            `Found state: ${response.foundState} (|${response.foundState.toString(2).padStart(response.numQubits, '0')}⟩)`,
            `Iterations: ${response.iterations}`,
            `Oracle calls: ${response.oracleCalls}`,
            `Success probability at marked state: ${(response.successProbability * 100).toFixed(3)}%`,
            '',
            'Top states:',
            formatTopStates(response.topStates),
          ].join('\n');
        } else if (response.algorithm === 'quantum-teleportation') {
          output = [
            `Measured bits: m0=${response.measurementBits.m0}, m1=${response.measurementBits.m1}`,
            `Teleportation fidelity (Bloch overlap): ${response.fidelity.toFixed(6)}`,
            `Source Bloch vector: (${response.sourceBloch.x.toFixed(4)}, ${response.sourceBloch.y.toFixed(4)}, ${response.sourceBloch.z.toFixed(4)})`,
            `Target Bloch vector: (${response.targetBloch.x.toFixed(4)}, ${response.targetBloch.y.toFixed(4)}, ${response.targetBloch.z.toFixed(4)})`,
            '',
            'Top states:',
            formatTopStates(response.topStates),
          ].join('\n');
        } else if (response.algorithm === 'vqe-h2') {
          output = [
            `Bond distance: ${response.bondDistance.toFixed(4)} Å`,
            `Estimated ground energy: ${response.energyHartree.toFixed(9)} Ha`,
            `Reference FCI energy: ${response.referenceEnergyHartree.toFixed(9)} Ha`,
            `Chemical accuracy error: ${response.chemicalAccuracyKcalMol.toFixed(4)} kcal/mol`,
            `Within chemical accuracy (<= 1 kcal/mol): ${response.convergedToChemicalAccuracy ? 'yes' : 'no'}`,
            `Iterations: ${response.iterations}`,
            `Energy evaluations: ${response.evaluations}`,
            `Optimal parameters: [${Array.from(response.parameters)
              .map((value) => value.toFixed(5))
              .join(', ')}]`,
            '',
            'Top basis states:',
            formatTopStates(response.topStates),
          ].join('\n');
        }

        setRunOutput((prev) => ({ ...prev, [currentExample.id]: output }));
        return;
      }

      if (currentExample.id === 'bell-state') {
        numQubits = 2;
        gates = [
          { type: 'H', qubit: 0 },
          { type: 'CNOT', qubit: 1, controlQubit: 0 },
        ];
      } else if (currentExample.id === 'superposition') {
        numQubits = 3;
        gates = [
          { type: 'H', qubit: 0 },
          { type: 'H', qubit: 1 },
          { type: 'H', qubit: 2 },
        ];
      } else if (currentExample.id === 'phase-kickback') {
        numQubits = 2;
        gates = [
          { type: 'H', qubit: 0 },
          { type: 'X', qubit: 1 },
          { type: 'CZ', qubit: 1, controlQubit: 0 },
          { type: 'H', qubit: 0 },
        ];
      }

      if (numQubits === 0) throw new Error('Example is not runnable in the browser build.');

      const result = await runCircuitInWorker({ numQubits, gates, cleanupAfterRun: true });
      if (result.warnings.length) {
        console.warn('Example warnings:', result.warnings);
      }
      const output = formatProbabilities(result.probabilities, numQubits);
      setRunOutput((prev) => ({ ...prev, [currentExample.id]: output }));
    } catch (error) {
      setRunError((prev) => ({
        ...prev,
        [currentExample.id]: error instanceof Error ? error.message : String(error),
      }));
    } finally {
      setRunningId(null);
    }
  };

  const copyCode = async () => {
    await navigator.clipboard.writeText(currentExample.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="examples">
      <div className="section-header">
        <img className="section-logo" src={logoUrl} alt="" aria-hidden="true" />
        <div className="section-header-text">
          <h1 className="section-title">Code Examples</h1>
          <p className="section-description">
            Copy-paste code snippets to get started quickly with quantum computing.
          </p>
        </div>
      </div>

      <div className="examples-layout">
        <aside className="examples-sidebar">
          {EXAMPLES.map(example => (
            <button
              key={example.id}
              className={`example-btn ${selectedExample === example.id ? 'selected' : ''}`}
              onClick={() => setSelectedExample(example.id)}
            >
              <span className="example-title">{example.title}</span>
              <span className="example-desc">{example.description}</span>
            </button>
          ))}
        </aside>

        <main className="example-content">
          <div className="example-header">
            <h2>{currentExample.title}</h2>
            <div className="example-actions">
              {currentExample.runnable && (
                <button
                  className="btn btn-primary"
                  onClick={runExample}
                  disabled={runningId === currentExample.id}
                >
                  {runningId === currentExample.id ? 'Running…' : 'Run Example'}
                </button>
              )}
              <button className="btn btn-secondary" onClick={copyCode}>
                {copied ? 'Copied!' : 'Copy Code'}
              </button>
            </div>
          </div>

          <p className="example-description">{currentExample.description}</p>

          <div className="code-container">
            <pre className="code-block">
              <code>{currentExample.code}</code>
            </pre>
          </div>

          <div className="output-container">
            <h3>Output</h3>
            {currentExample.runnable ? (
              <pre className="output-block">
                {runError[currentExample.id]
                  ? `Error: ${runError[currentExample.id]}`
                  : runOutput[currentExample.id] || 'Run the example to compute output in WASM.'}
              </pre>
            ) : (
              <pre className="output-block">
                This example depends on algorithms not bundled in the browser WASM build yet.
              </pre>
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

export default Examples;
