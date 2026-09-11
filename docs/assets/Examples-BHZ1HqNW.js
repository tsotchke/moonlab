import{r as l,j as a}from"./index-DCaC2bRl.js";import{e as $,a as j,r as E}from"./moonlabClient-D2nZaPpU.js";const c=[{id:"bell-state",title:"Bell State (Entanglement)",description:"Create a maximally entangled Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2",code:`import { QuantumState } from '@tsotchkecorp/moonlab';

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
state.dispose();`,runnable:!0},{id:"superposition",title:"Uniform Superposition",description:"Create equal superposition over all basis states using Hadamard gates",code:`import { QuantumState } from '@tsotchkecorp/moonlab';

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

state.dispose();`,runnable:!0},{id:"grover",title:"Grover's Search Algorithm",description:"Find a marked item in an unsorted database with quadratic speedup",code:`import { QuantumState } from '@tsotchkecorp/moonlab';
import { Grover } from '@tsotchkecorp/moonlab-algorithms';

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

grover.dispose();`,runnable:!0},{id:"phase-kickback",title:"Phase Kickback",description:"Demonstrate the phase kickback phenomenon used in quantum algorithms",code:`import { QuantumState } from '@tsotchkecorp/moonlab';

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

state.dispose();`,runnable:!0},{id:"quantum-teleportation",title:"Quantum Teleportation",description:"Teleport a quantum state using entanglement and classical communication",code:`import { QuantumState } from '@tsotchkecorp/moonlab';

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

state.dispose();`,runnable:!0},{id:"vqe-h2",title:"VQE: H₂ Ground State Energy",description:"Find the ground state energy of hydrogen molecule using variational quantum eigensolver",code:`import { VQE, createH2Hamiltonian } from '@tsotchkecorp/moonlab-algorithms';

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

vqe.dispose();`,runnable:!0},{id:"ca-mps-stabilizer",title:"CA-MPS: Stabilizer Circuit",description:"Pure-Clifford GHZ state via Clifford-Assisted MPS. The Clifford structure goes into the tableau (O(n) bit ops); the MPS stays at bond dimension 1. 64x bond-dim advantage and 13884x speedup vs plain MPS at n=12 (since v0.2.1).",code:`import { CaMps } from '@tsotchkecorp/moonlab';

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

state.dispose();`,runnable:!1},{id:"gauge-warmstart-bell",title:"Gauge-Aware Warmstart: Bell stabilizers",description:"Aaronson-Gottesman symplectic-Gauss-Jordan Clifford prep on the abelian stabilizer subgroup {XX, ZZ}. The resulting state is in the simultaneous +1 eigenspace of every generator -- the Bell state |Phi+>. Generalises to LGT Gauss-law operators, surface/toric/repetition codes (since v0.2.1).",code:`import { CaMps, gaugeWarmstart } from '@tsotchkecorp/moonlab';

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
state.dispose();`,runnable:!1},{id:"z2-lgt-build",title:"Z2 Lattice Gauge Theory: Pauli sum builder",description:"Build the 1+1D Z2 LGT Hamiltonian on N matter sites. Exactly gauge-invariant kinetic terms (XYY/YYX form) -- each piece commutes with every interior Gauss-law operator G_x = X_{2x-1} Z_{2x} X_{2x+1}. First HEP application of the gauge-aware warmstart (since v0.2.1).",code:`import { z2Lgt1dBuild, z2Lgt1dGaussLaw } from '@tsotchkecorp/moonlab';

// N = 4 matter sites -> 7 qubits (4 matter + 3 link).
const ham = await z2Lgt1dBuild(4, /*t=*/1.0, /*h=*/0.5, /*m=*/0.0,
                                  /*gauss_penalty=*/0.0);

console.log('num_qubits =', ham.numQubits);  // 7
console.log('num_terms  =', ham.numTerms);   // matter + electric + mass

// Interior Gauss-law operator at matter site x = 1.
const G1 = await z2Lgt1dGaussLaw(4, 1);
// Bytes are 0,1,3,1,0,0,0 -- X on qubit 1, Z on qubit 2,
// X on qubit 3, identity elsewhere.
console.log('G_1 =', Array.from(G1));`,runnable:!1}],T=()=>{const y="./ml-logo.png",[m,q]=l.useState(c[0].id),[S,p]=l.useState(!1),[v,b]=l.useState({}),[h,g]=l.useState({}),[x,f]=l.useState(null),s=c.find(e=>e.id===m)||c[0],C=(e,o)=>{const i=[];for(let r=0;r<e.length;r++){const t=e[r];if(t<5e-4)continue;const n=r.toString(2).padStart(o,"0");i.push(`|${n}⟩: ${(t*100).toFixed(2)}%`)}return i.length===0?"No non-zero probabilities found.":i.join(`
`)},u=e=>e.length?e.map(o=>`|${o.bitstring}⟩: ${(o.probability*100).toFixed(2)}%`).join(`
`):"No high-probability states found.",k=async()=>{if(s.runnable){f(s.id),g(e=>({...e,[s.id]:""}));try{await $();let e=0,o=[];if(s.id==="grover"||s.id==="quantum-teleportation"||s.id==="vqe-h2"){const t=await j({id:s.id,cleanupAfterRun:!0});let n="";t.algorithm==="grover"?n=[`Marked state: ${t.markedState} (|${t.markedState.toString(2).padStart(t.numQubits,"0")}⟩)`,`Found state: ${t.foundState} (|${t.foundState.toString(2).padStart(t.numQubits,"0")}⟩)`,`Iterations: ${t.iterations}`,`Oracle calls: ${t.oracleCalls}`,`Success probability at marked state: ${(t.successProbability*100).toFixed(3)}%`,"","Top states:",u(t.topStates)].join(`
`):t.algorithm==="quantum-teleportation"?n=[`Measured bits: m0=${t.measurementBits.m0}, m1=${t.measurementBits.m1}`,`Teleportation fidelity (Bloch overlap): ${t.fidelity.toFixed(6)}`,`Source Bloch vector: (${t.sourceBloch.x.toFixed(4)}, ${t.sourceBloch.y.toFixed(4)}, ${t.sourceBloch.z.toFixed(4)})`,`Target Bloch vector: (${t.targetBloch.x.toFixed(4)}, ${t.targetBloch.y.toFixed(4)}, ${t.targetBloch.z.toFixed(4)})`,"","Top states:",u(t.topStates)].join(`
`):t.algorithm==="vqe-h2"&&(n=[`Bond distance: ${t.bondDistance.toFixed(4)} Å`,`Estimated ground energy: ${t.energyHartree.toFixed(9)} Ha`,`Reference FCI energy: ${t.referenceEnergyHartree.toFixed(9)} Ha`,`Chemical accuracy error: ${t.chemicalAccuracyKcalMol.toFixed(4)} kcal/mol`,`Within chemical accuracy (<= 1 kcal/mol): ${t.convergedToChemicalAccuracy?"yes":"no"}`,`Iterations: ${t.iterations}`,`Energy evaluations: ${t.evaluations}`,`Optimal parameters: [${Array.from(t.parameters).map(d=>d.toFixed(5)).join(", ")}]`,"","Top basis states:",u(t.topStates)].join(`
`)),b(d=>({...d,[s.id]:n}));return}if(s.id==="bell-state"?(e=2,o=[{type:"H",qubit:0},{type:"CNOT",qubit:1,controlQubit:0}]):s.id==="superposition"?(e=3,o=[{type:"H",qubit:0},{type:"H",qubit:1},{type:"H",qubit:2}]):s.id==="phase-kickback"&&(e=2,o=[{type:"H",qubit:0},{type:"X",qubit:1},{type:"CZ",qubit:1,controlQubit:0},{type:"H",qubit:0}]),e===0)throw new Error("Example is not runnable in the browser build.");const i=await E({numQubits:e,gates:o,cleanupAfterRun:!0});i.warnings.length&&console.warn("Example warnings:",i.warnings);const r=C(i.probabilities,e);b(t=>({...t,[s.id]:r}))}catch(e){g(o=>({...o,[s.id]:e instanceof Error?e.message:String(e)}))}finally{f(null)}}},w=async()=>{await navigator.clipboard.writeText(s.code),p(!0),setTimeout(()=>p(!1),2e3)};return a.jsxs("div",{className:"examples",children:[a.jsxs("div",{className:"section-header",children:[a.jsx("img",{className:"section-logo",src:y,alt:"","aria-hidden":"true"}),a.jsxs("div",{className:"section-header-text",children:[a.jsx("h1",{className:"section-title",children:"Code Examples"}),a.jsx("p",{className:"section-description",children:"Copy-paste code snippets to get started quickly with quantum computing."})]})]}),a.jsxs("div",{className:"examples-layout",children:[a.jsx("aside",{className:"examples-sidebar",children:c.map(e=>a.jsxs("button",{className:`example-btn ${m===e.id?"selected":""}`,onClick:()=>q(e.id),children:[a.jsx("span",{className:"example-title",children:e.title}),a.jsx("span",{className:"example-desc",children:e.description})]},e.id))}),a.jsxs("main",{className:"example-content",children:[a.jsxs("div",{className:"example-header",children:[a.jsx("h2",{children:s.title}),a.jsxs("div",{className:"example-actions",children:[s.runnable&&a.jsx("button",{className:"btn btn-primary",onClick:k,disabled:x===s.id,children:x===s.id?"Running…":"Run Example"}),a.jsx("button",{className:"btn btn-secondary",onClick:w,children:S?"Copied!":"Copy Code"})]})]}),a.jsx("p",{className:"example-description",children:s.description}),a.jsx("div",{className:"code-container",children:a.jsx("pre",{className:"code-block",children:a.jsx("code",{children:s.code})})}),a.jsxs("div",{className:"output-container",children:[a.jsx("h3",{children:"Output"}),s.runnable?a.jsx("pre",{className:"output-block",children:h[s.id]?`Error: ${h[s.id]}`:v[s.id]||"Run the example to compute output in WASM."}):a.jsx("pre",{className:"output-block",children:"This example depends on algorithms not bundled in the browser WASM build yet."})]})]})]})]})};export{T as default};
//# sourceMappingURL=Examples-BHZ1HqNW.js.map
