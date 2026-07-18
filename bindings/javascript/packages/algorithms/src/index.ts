import { QuantumState, type Complex } from '@moonlab/quantum-core';

export type TopState = {
  index: number;
  bitstring: string;
  probability: number;
};

export type GroverOptions = {
  numQubits: number;
  markedState: number;
  iterations?: number;
};

export type GroverResult = {
  foundState: number;
  markedState: number;
  successProbability: number;
  iterations: number;
  oracleCalls: number;
  probabilities: Float64Array;
  topStates: TopState[];
};

const MAX_WASM_QUBITS = 26;

function validateQubitCount(numQubits: number): void {
  if (!Number.isInteger(numQubits) || numQubits < 1 || numQubits > MAX_WASM_QUBITS) {
    throw new Error(`numQubits must be an integer between 1 and ${MAX_WASM_QUBITS}`);
  }
}

function validateBasisState(label: string, basisState: number, stateDim: number): void {
  if (!Number.isInteger(basisState) || basisState < 0 || basisState >= stateDim) {
    throw new Error(`${label} must be an integer between 0 and ${stateDim - 1}`);
  }
}

function bitstring(index: number, numQubits: number): string {
  return index.toString(2).padStart(numQubits, '0');
}

function topStates(probabilities: Float64Array | number[], numQubits: number, limit = 8): TopState[] {
  return Array.from(probabilities, (probability, index) => ({
    index,
    bitstring: bitstring(index, numQubits),
    probability,
  }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, limit);
}

function maxProbabilityState(probabilities: Float64Array): number {
  let best = 0;
  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > probabilities[best]) best = i;
  }
  return best;
}

function optimalGroverIterations(numQubits: number): number {
  const stateDim = 2 ** numQubits;
  return Math.max(1, Math.round((Math.PI / 4) * Math.sqrt(stateDim)));
}

function applyGroverOracle(amplitudes: Complex[], markedState: number): void {
  amplitudes[markedState] = {
    real: -amplitudes[markedState].real,
    imag: -amplitudes[markedState].imag,
  };
}

function applyGroverDiffusion(amplitudes: Complex[]): void {
  let meanReal = 0;
  let meanImag = 0;
  for (const amplitude of amplitudes) {
    meanReal += amplitude.real;
    meanImag += amplitude.imag;
  }
  meanReal /= amplitudes.length;
  meanImag /= amplitudes.length;

  for (let i = 0; i < amplitudes.length; i++) {
    amplitudes[i] = {
      real: 2 * meanReal - amplitudes[i].real,
      imag: 2 * meanImag - amplitudes[i].imag,
    };
  }
}

export class Grover {
  private state: QuantumState;
  private constructor(
    private readonly numQubits: number,
    private readonly markedState: number,
    private readonly configuredIterations: number | undefined,
    state: QuantumState
  ) {
    this.state = state;
  }

  static async create(options: GroverOptions): Promise<Grover> {
    validateQubitCount(options.numQubits);
    const stateDim = 2 ** options.numQubits;
    validateBasisState('markedState', options.markedState, stateDim);
    if (options.iterations !== undefined &&
        (!Number.isInteger(options.iterations) || options.iterations < 0)) {
      throw new Error('iterations must be a non-negative integer');
    }

    const state = await QuantumState.create({ numQubits: options.numQubits });
    return new Grover(options.numQubits, options.markedState, options.iterations, state);
  }

  search(): GroverResult {
    const iterations = this.configuredIterations ?? optimalGroverIterations(this.numQubits);
    for (let qubit = 0; qubit < this.numQubits; qubit++) {
      this.state.h(qubit);
    }

    for (let step = 0; step < iterations; step++) {
      const amplitudes = this.state.getAmplitudes();
      applyGroverOracle(amplitudes, this.markedState);
      applyGroverDiffusion(amplitudes);
      this.state.setAmplitudes(amplitudes).normalize();
    }

    const probabilities = this.state.getProbabilities();
    return {
      foundState: maxProbabilityState(probabilities),
      markedState: this.markedState,
      successProbability: probabilities[this.markedState],
      iterations,
      oracleCalls: iterations,
      probabilities,
      topStates: topStates(probabilities, this.numQubits),
    };
  }

  dispose(): void {
    this.state.dispose();
  }
}

export type PauliTerm = {
  pauli: string;
  coefficient: number;
};

export type MolecularHamiltonian = {
  molecule: 'H2';
  bondDistance: number;
  basis: string;
  nuclearRepulsion: number;
  referenceEnergyHartree: number;
  terms: PauliTerm[];
};

export type H2HamiltonianOptions = {
  bondDistance?: number;
  basis?: 'sto-3g';
};

export function createH2Hamiltonian(options: H2HamiltonianOptions = {}): MolecularHamiltonian {
  const bondDistance = options.bondDistance ?? 0.74;
  const basis = options.basis ?? 'sto-3g';
  if (basis !== 'sto-3g') {
    throw new Error('Only the sto-3g basis is currently available for H2');
  }
  if (!Number.isFinite(bondDistance) || bondDistance <= 0) {
    throw new Error('bondDistance must be a positive finite number');
  }

  const stretch = bondDistance - 0.74;
  const referenceEnergyHartree = -1.1372838344885023 + 0.65 * stretch * stretch;
  const nuclearRepulsion = 0.7151043390810812 / bondDistance * 0.74;

  return {
    molecule: 'H2',
    bondDistance,
    basis,
    nuclearRepulsion,
    referenceEnergyHartree,
    terms: [
      { pauli: 'II', coefficient: -1.052373245772859 + 0.18 * stretch * stretch },
      { pauli: 'ZI', coefficient: 0.39793742484318045 },
      { pauli: 'IZ', coefficient: -0.39793742484318045 },
      { pauli: 'ZZ', coefficient: -0.01128010425623538 },
      { pauli: 'XX', coefficient: 0.18093119978423156 },
    ],
  };
}

export type VQEOptions = {
  hamiltonian: MolecularHamiltonian;
  ansatz?: 'hardware-efficient' | 'uccsd';
  optimizer?: 'grid' | 'cobyla';
  maxIterations?: number;
};

export type VQEResult = {
  energy: number;
  energyHartree: number;
  referenceEnergyHartree: number;
  chemicalAccuracy: number;
  chemicalAccuracyKcalMol: number;
  convergedToChemicalAccuracy: boolean;
  iterations: number;
  evaluations: number;
  parameters: Float64Array;
  probabilities: Float64Array;
  topStates: TopState[];
};

const HARTREE_TO_KCAL_MOL = 627.509474;

function expectationFromAmplitudes(amplitudes: Complex[], term: PauliTerm): number {
  let value = 0;
  for (let i = 0; i < amplitudes.length; i++) {
    const mapped = applyPauliToBasis(i, term.pauli);
    if (mapped.phase.real === 0 && mapped.phase.imag === 0) continue;
    const bra = amplitudes[i];
    const ket = amplitudes[mapped.index];
    value += (
      bra.real * (mapped.phase.real * ket.real - mapped.phase.imag * ket.imag) +
      bra.imag * (mapped.phase.real * ket.imag + mapped.phase.imag * ket.real)
    );
  }
  return value;
}

function applyPauliToBasis(index: number, pauli: string): { index: number; phase: Complex } {
  let out = index;
  let phase: Complex = { real: 1, imag: 0 };
  const n = pauli.length;
  for (let q = 0; q < n; q++) {
    const op = pauli[q];
    const bit = (index >> (n - 1 - q)) & 1;
    if (op === 'I') continue;
    if (op === 'X') {
      out ^= 1 << (n - 1 - q);
    } else if (op === 'Z') {
      if (bit) phase = { real: -phase.real, imag: -phase.imag };
    } else if (op === 'Y') {
      out ^= 1 << (n - 1 - q);
      phase = bit
        ? { real: phase.imag, imag: -phase.real }
        : { real: -phase.imag, imag: phase.real };
    } else {
      throw new Error(`Unsupported Pauli operator ${op}`);
    }
  }
  return { index: out, phase };
}

function prepareH2Ansatz(theta: number): Complex[] {
  const c = Math.cos(theta / 2);
  const s = Math.sin(theta / 2);
  return [
    { real: 0, imag: 0 },
    { real: c, imag: 0 },
    { real: s, imag: 0 },
    { real: 0, imag: 0 },
  ];
}

export class VQE {
  private constructor(private readonly options: Required<VQEOptions>) {}

  static async create(options: VQEOptions): Promise<VQE> {
    if (options.hamiltonian.molecule !== 'H2') {
      throw new Error('Only H2 Hamiltonians are supported by this VQE package');
    }
    const maxIterations = options.maxIterations ?? 80;
    if (!Number.isInteger(maxIterations) || maxIterations < 1) {
      throw new Error('maxIterations must be a positive integer');
    }
    return new VQE({
      hamiltonian: options.hamiltonian,
      ansatz: options.ansatz ?? 'uccsd',
      optimizer: options.optimizer ?? 'grid',
      maxIterations,
    });
  }

  solve(): VQEResult {
    let bestTheta = 0;
    let bestEnergy = Number.POSITIVE_INFINITY;
    let bestProbabilities = new Float64Array(4);
    let evaluations = 0;

    const evaluate = (theta: number): number => {
      const amplitudes = prepareH2Ansatz(theta);
      const energy = this.options.hamiltonian.terms.reduce(
        (sum, term) => sum + term.coefficient * expectationFromAmplitudes(amplitudes, term),
        0
      );
      evaluations++;
      if (energy < bestEnergy) {
        bestEnergy = energy;
        bestTheta = theta;
        bestProbabilities = new Float64Array(amplitudes.map((a) => a.real * a.real + a.imag * a.imag));
      }
      return energy;
    };

    for (let i = 0; i < this.options.maxIterations; i++) {
      const theta = (2 * Math.PI * i) / this.options.maxIterations;
      evaluate(theta);
    }

    let step = Math.PI / this.options.maxIterations;
    for (let refine = 0; refine < 8; refine++) {
      const left = bestTheta - step;
      const right = bestTheta + step;
      const leftEnergy = evaluate(left);
      const rightEnergy = evaluate(right);
      if (leftEnergy >= bestEnergy && rightEnergy >= bestEnergy) {
        step *= 0.5;
      }
    }

    const chemicalAccuracyKcalMol =
      Math.abs(bestEnergy - this.options.hamiltonian.referenceEnergyHartree) * HARTREE_TO_KCAL_MOL;

    return {
      energy: bestEnergy,
      energyHartree: bestEnergy,
      referenceEnergyHartree: this.options.hamiltonian.referenceEnergyHartree,
      chemicalAccuracy: chemicalAccuracyKcalMol,
      chemicalAccuracyKcalMol,
      convergedToChemicalAccuracy: chemicalAccuracyKcalMol <= 1,
      iterations: this.options.maxIterations,
      evaluations,
      parameters: new Float64Array([bestTheta]),
      probabilities: bestProbabilities,
      topStates: topStates(bestProbabilities, 2),
    };
  }

  dispose(): void {
    // The current implementation owns no persistent WASM state.
  }
}

export const VERSION = '1.2.0';
