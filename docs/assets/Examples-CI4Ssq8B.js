import{r as l,j as a}from"./index-BnghuY-9.js";import{e as C,a as E,r as w}from"./moonlabClient-D2nZaPpU.js";const c=[{id:"bell-state",title:"Bell State (Entanglement)",description:"Create a maximally entangled Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2",code:`import { QuantumState } from '@moonlab/quantum-core';

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
state.dispose();`,runnable:!0},{id:"superposition",title:"Uniform Superposition",description:"Create equal superposition over all basis states using Hadamard gates",code:`import { QuantumState } from '@moonlab/quantum-core';

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

state.dispose();`,runnable:!0},{id:"grover",title:"Grover's Search Algorithm",description:"Find a marked item in an unsorted database with quadratic speedup",code:`import { QuantumState } from '@moonlab/quantum-core';
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

grover.dispose();`,runnable:!0},{id:"phase-kickback",title:"Phase Kickback",description:"Demonstrate the phase kickback phenomenon used in quantum algorithms",code:`import { QuantumState } from '@moonlab/quantum-core';

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

state.dispose();`,runnable:!0},{id:"quantum-teleportation",title:"Quantum Teleportation",description:"Teleport a quantum state using entanglement and classical communication",code:`import { QuantumState } from '@moonlab/quantum-core';

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

state.dispose();`,runnable:!0},{id:"vqe-h2",title:"VQE: H₂ Ground State Energy",description:"Find the ground state energy of hydrogen molecule using variational quantum eigensolver",code:`import { VQE, createH2Hamiltonian } from '@moonlab/quantum-algorithms';

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

vqe.dispose();`,runnable:!0}],H=()=>{const q="./ml-logo.png",[m,f]=l.useState(c[0].id),[S,p]=l.useState(!1),[$,b]=l.useState({}),[h,g]=l.useState({}),[x,y]=l.useState(null),o=c.find(t=>t.id===m)||c[0],v=(t,s)=>{const i=[];for(let n=0;n<t.length;n++){const e=t[n];if(e<5e-4)continue;const r=n.toString(2).padStart(s,"0");i.push(`|${r}⟩: ${(e*100).toFixed(2)}%`)}return i.length===0?"No non-zero probabilities found.":i.join(`
`)},u=t=>t.length?t.map(s=>`|${s.bitstring}⟩: ${(s.probability*100).toFixed(2)}%`).join(`
`):"No high-probability states found.",j=async()=>{if(o.runnable){y(o.id),g(t=>({...t,[o.id]:""}));try{await C();let t=0,s=[];if(o.id==="grover"||o.id==="quantum-teleportation"||o.id==="vqe-h2"){const e=await E({id:o.id,cleanupAfterRun:!0});let r="";e.algorithm==="grover"?r=[`Marked state: ${e.markedState} (|${e.markedState.toString(2).padStart(e.numQubits,"0")}⟩)`,`Found state: ${e.foundState} (|${e.foundState.toString(2).padStart(e.numQubits,"0")}⟩)`,`Iterations: ${e.iterations}`,`Oracle calls: ${e.oracleCalls}`,`Success probability at marked state: ${(e.successProbability*100).toFixed(3)}%`,"","Top states:",u(e.topStates)].join(`
`):e.algorithm==="quantum-teleportation"?r=[`Measured bits: m0=${e.measurementBits.m0}, m1=${e.measurementBits.m1}`,`Teleportation fidelity (Bloch overlap): ${e.fidelity.toFixed(6)}`,`Source Bloch vector: (${e.sourceBloch.x.toFixed(4)}, ${e.sourceBloch.y.toFixed(4)}, ${e.sourceBloch.z.toFixed(4)})`,`Target Bloch vector: (${e.targetBloch.x.toFixed(4)}, ${e.targetBloch.y.toFixed(4)}, ${e.targetBloch.z.toFixed(4)})`,"","Top states:",u(e.topStates)].join(`
`):e.algorithm==="vqe-h2"&&(r=[`Bond distance: ${e.bondDistance.toFixed(4)} Å`,`Estimated ground energy: ${e.energyHartree.toFixed(9)} Ha`,`Reference FCI energy: ${e.referenceEnergyHartree.toFixed(9)} Ha`,`Chemical accuracy error: ${e.chemicalAccuracyKcalMol.toFixed(4)} kcal/mol`,`Within chemical accuracy (<= 1 kcal/mol): ${e.convergedToChemicalAccuracy?"yes":"no"}`,`Iterations: ${e.iterations}`,`Energy evaluations: ${e.evaluations}`,`Optimal parameters: [${Array.from(e.parameters).map(d=>d.toFixed(5)).join(", ")}]`,"","Top basis states:",u(e.topStates)].join(`
`)),b(d=>({...d,[o.id]:r}));return}if(o.id==="bell-state"?(t=2,s=[{type:"H",qubit:0},{type:"CNOT",qubit:1,controlQubit:0}]):o.id==="superposition"?(t=3,s=[{type:"H",qubit:0},{type:"H",qubit:1},{type:"H",qubit:2}]):o.id==="phase-kickback"&&(t=2,s=[{type:"H",qubit:0},{type:"X",qubit:1},{type:"CZ",qubit:1,controlQubit:0},{type:"H",qubit:0}]),t===0)throw new Error("Example is not runnable in the browser build.");const i=await w({numQubits:t,gates:s,cleanupAfterRun:!0});i.warnings.length&&console.warn("Example warnings:",i.warnings);const n=v(i.probabilities,t);b(e=>({...e,[o.id]:n}))}catch(t){g(s=>({...s,[o.id]:t instanceof Error?t.message:String(t)}))}finally{y(null)}}},k=async()=>{await navigator.clipboard.writeText(o.code),p(!0),setTimeout(()=>p(!1),2e3)};return a.jsxs("div",{className:"examples",children:[a.jsxs("div",{className:"section-header",children:[a.jsx("img",{className:"section-logo",src:q,alt:"","aria-hidden":"true"}),a.jsxs("div",{className:"section-header-text",children:[a.jsx("h1",{className:"section-title",children:"Code Examples"}),a.jsx("p",{className:"section-description",children:"Copy-paste code snippets to get started quickly with quantum computing."})]})]}),a.jsxs("div",{className:"examples-layout",children:[a.jsx("aside",{className:"examples-sidebar",children:c.map(t=>a.jsxs("button",{className:`example-btn ${m===t.id?"selected":""}`,onClick:()=>f(t.id),children:[a.jsx("span",{className:"example-title",children:t.title}),a.jsx("span",{className:"example-desc",children:t.description})]},t.id))}),a.jsxs("main",{className:"example-content",children:[a.jsxs("div",{className:"example-header",children:[a.jsx("h2",{children:o.title}),a.jsxs("div",{className:"example-actions",children:[o.runnable&&a.jsx("button",{className:"btn btn-primary",onClick:j,disabled:x===o.id,children:x===o.id?"Running…":"Run Example"}),a.jsx("button",{className:"btn btn-secondary",onClick:k,children:S?"Copied!":"Copy Code"})]})]}),a.jsx("p",{className:"example-description",children:o.description}),a.jsx("div",{className:"code-container",children:a.jsx("pre",{className:"code-block",children:a.jsx("code",{children:o.code})})}),a.jsxs("div",{className:"output-container",children:[a.jsx("h3",{children:"Output"}),o.runnable?a.jsx("pre",{className:"output-block",children:h[o.id]?`Error: ${h[o.id]}`:$[o.id]||"Run the example to compute output in WASM."}):a.jsx("pre",{className:"output-block",children:"This example depends on algorithms not bundled in the browser WASM build yet."})]})]})]})]})};export{H as default};
//# sourceMappingURL=Examples-CI4Ssq8B.js.map
