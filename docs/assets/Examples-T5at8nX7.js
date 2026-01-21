import{r as i,j as e}from"./index-CtOx95dX.js";import{e as C,r as w}from"./moonlabClient-D-rb-4O8.js";const l=[{id:"bell-state",title:"Bell State (Entanglement)",description:"Create a maximally entangled Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2",code:`import { QuantumState } from '@moonlab/quantum-core';

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

grover.dispose();`,runnable:!1},{id:"phase-kickback",title:"Phase Kickback",description:"Demonstrate the phase kickback phenomenon used in quantum algorithms",code:`import { QuantumState } from '@moonlab/quantum-core';

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

state.dispose();`,runnable:!1},{id:"vqe-h2",title:"VQE: H₂ Ground State Energy",description:"Find the ground state energy of hydrogen molecule using variational quantum eigensolver",code:`import { VQE, createH2Hamiltonian } from '@moonlab/quantum-algorithms';

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

vqe.dispose();`,runnable:!1}],E=()=>{const h="./moonlab.png",[c,g]=i.useState(l[0].id),[q,u]=i.useState(!1),[x,y]=i.useState({}),[d,m]=i.useState({}),[p,b]=i.useState(null),a=l.find(t=>t.id===c)||l[0],f=(t,s)=>{const o=[];for(let n=0;n<t.length;n++){const r=t[n];if(r<5e-4)continue;const v=n.toString(2).padStart(s,"0");o.push(`|${v}⟩: ${(r*100).toFixed(2)}%`)}return o.length===0?"No non-zero probabilities found.":o.join(`
`)},S=async()=>{if(a.runnable){b(a.id),m(t=>({...t,[a.id]:""}));try{await C();let t=0,s=[];if(a.id==="bell-state"?(t=2,s=[{type:"H",qubit:0},{type:"CNOT",qubit:1,controlQubit:0}]):a.id==="superposition"?(t=3,s=[{type:"H",qubit:0},{type:"H",qubit:1},{type:"H",qubit:2}]):a.id==="phase-kickback"&&(t=2,s=[{type:"H",qubit:0},{type:"X",qubit:1},{type:"CZ",qubit:1,controlQubit:0},{type:"H",qubit:0}]),t===0)throw new Error("Example is not runnable in the browser build.");const o=await w({numQubits:t,gates:s});o.warnings.length&&console.warn("Example warnings:",o.warnings);const n=f(o.probabilities,t);y(r=>({...r,[a.id]:n}))}catch(t){m(s=>({...s,[a.id]:t instanceof Error?t.message:String(t)}))}finally{b(null)}}},j=async()=>{await navigator.clipboard.writeText(a.code),u(!0),setTimeout(()=>u(!1),2e3)};return e.jsxs("div",{className:"examples",children:[e.jsxs("div",{className:"section-header",children:[e.jsx("img",{className:"section-logo",src:h,alt:"","aria-hidden":"true"}),e.jsxs("div",{className:"section-header-text",children:[e.jsx("h1",{className:"section-title",children:"Code Examples"}),e.jsx("p",{className:"section-description",children:"Copy-paste code snippets to get started quickly with quantum computing."})]})]}),e.jsxs("div",{className:"examples-layout",children:[e.jsx("aside",{className:"examples-sidebar",children:l.map(t=>e.jsxs("button",{className:`example-btn ${c===t.id?"selected":""}`,onClick:()=>g(t.id),children:[e.jsx("span",{className:"example-title",children:t.title}),e.jsx("span",{className:"example-desc",children:t.description})]},t.id))}),e.jsxs("main",{className:"example-content",children:[e.jsxs("div",{className:"example-header",children:[e.jsx("h2",{children:a.title}),e.jsxs("div",{className:"example-actions",children:[a.runnable&&e.jsx("button",{className:"btn btn-primary",onClick:S,disabled:p===a.id,children:p===a.id?"Running…":"Run Example"}),e.jsx("button",{className:"btn btn-secondary",onClick:j,children:q?"Copied!":"Copy Code"})]})]}),e.jsx("p",{className:"example-description",children:a.description}),e.jsx("div",{className:"code-container",children:e.jsx("pre",{className:"code-block",children:e.jsx("code",{children:a.code})})}),e.jsxs("div",{className:"output-container",children:[e.jsx("h3",{children:"Output"}),a.runnable?e.jsx("pre",{className:"output-block",children:d[a.id]?`Error: ${d[a.id]}`:x[a.id]||"Run the example to compute output in WASM."}):e.jsx("pre",{className:"output-block",children:"This example depends on algorithms not bundled in the browser WASM build yet."})]})]})]})]})};export{E as default};
//# sourceMappingURL=Examples-T5at8nX7.js.map
