/**
 * Variational Quantum Eigensolver (C-side since v0.2.0; JS binding
 * since v0.5.5; ergonomics since v1.2.0).
 *
 * Wraps ``src/algorithms/vqe.{c,h}`` around the Pauli-Hamiltonian
 * builder, the hardware-efficient and UCCSD ansaetze, the five
 * classical optimizers (Adam / L-BFGS / COBYLA / gradient-descent /
 * quantum natural gradient), the optimizer hyperparameter surface, and
 * the ``vqe_solve`` driver.  Mirrors the Python
 * ``moonlab.algorithms.VQE`` ergonomics (string optimizers,
 * ``ansatz: 'uccsd'``, hyperparameter overrides).
 *
 * @example
 * ```typescript
 * import { PauliHamiltonian, VqeSolver } from '@moonlab/quantum-core';
 *
 * const h = await PauliHamiltonian.h2(0.74);
 * console.log('exact E_0 =', h.exactGroundStateEnergy(), 'Ha');
 *
 * const solver = await VqeSolver.create(h, {
 *   ansatz: 'uccsd',
 *   numElectrons: 1,
 *   optimizer: 'adam',
 *   learningRate: 0.1,
 *   maxIterations: 500,
 * });
 * const r = solver.solve();
 * console.log(`VQE E = ${r.groundStateEnergy.toFixed(4)} Ha`);
 * solver.dispose();
 * h.dispose();
 * ```
 */

import { getModule } from './wasm-loader';

/** Classical optimizer for {@link VqeSolver}.  Maps to
 *  ``vqe_optimizer_type_t`` in the C header. */
export enum OptimizerType {
  /** Constrained gradient-free optimizer. */
  Cobyla = 0,
  /** Limited-memory BFGS (quasi-Newton). */
  Lbfgs = 1,
  /** Adam adaptive-moment optimizer (default). */
  Adam = 2,
  /** Plain gradient descent. */
  GradientDescent = 3,
  /** Quantum natural gradient (Fubini-Study metric preconditioning,
   *  since v1.2.0). */
  Qng = 4,
}

/** String spellings accepted wherever an optimizer is selected.
 *  Mirrors Python's ``VQE(optimizer='qng')`` surface. */
export type OptimizerName =
  | 'cobyla'
  | 'lbfgs'
  | 'adam'
  | 'gradient_descent'
  | 'gradient-descent'
  | 'qng'
  | 'natural_gradient'
  | 'natural-gradient';

const OPTIMIZER_NAMES: Record<string, OptimizerType> = {
  cobyla: OptimizerType.Cobyla,
  lbfgs: OptimizerType.Lbfgs,
  adam: OptimizerType.Adam,
  gradient_descent: OptimizerType.GradientDescent,
  'gradient-descent': OptimizerType.GradientDescent,
  qng: OptimizerType.Qng,
  natural_gradient: OptimizerType.Qng,
  'natural-gradient': OptimizerType.Qng,
};

/** Resolve a string or enum optimizer spec to the C enum value. */
export function resolveOptimizer(spec: OptimizerType | OptimizerName): OptimizerType {
  if (typeof spec === 'number') return spec;
  const resolved = OPTIMIZER_NAMES[spec.trim().toLowerCase()];
  if (resolved === undefined) {
    throw new Error(
      `Unknown optimizer '${spec}'. Valid: ${Object.keys(OPTIMIZER_NAMES).join(', ')}`,
    );
  }
  return resolved;
}

/** Ansatz families constructible through {@link VqeSolver.create}. */
export type AnsatzName = 'hardware_efficient' | 'hardware-efficient' | 'hea' | 'uccsd';

/** Options for {@link VqeSolver.create}.  Every field is optional;
 *  defaults reproduce the historical hardware-efficient + Adam
 *  behaviour. */
export interface VqeSolverOptions {
  /** Ansatz family (default `'hardware_efficient'`). */
  ansatz?: AnsatzName;
  /** Circuit depth for the hardware-efficient ansatz (default 2). */
  numLayers?: number;
  /** Electron count for the UCCSD ansatz (default `numQubits / 2`,
   *  i.e. half filling -- same default as Python). */
  numElectrons?: number;
  /** Optimizer as enum value or string name (default Adam). */
  optimizer?: OptimizerType | OptimizerName;
  /** Maximum optimizer iterations (default: C per-optimizer default,
   *  e.g. 1000 for Adam). */
  maxIterations?: number;
  /** Convergence tolerance (default: C per-optimizer default). */
  tolerance?: number;
  /** Gradient step size (Adam / gradient descent / QNG). */
  learningRate?: number;
  /** Adam first-moment decay (default 0.9). */
  beta1?: number;
  /** Adam second-moment decay (default 0.999). */
  beta2?: number;
  /** Adam numerical epsilon (default 1e-8). */
  epsilon?: number;
  /** QNG metric Tikhonov shift (default 1e-3). */
  qngRegularization?: number;
}

/** Result of one VQE solve.  Mirrors ``vqe_result_t`` minus the raw
 *  pointer; `optimalParameters` is copied into an owned `Float64Array`. */
export interface VqeResult {
  /** Final ground-state energy in Hartree (includes nuclear repulsion). */
  groundStateEnergy: number;
  /** Same energy converted to kcal/mol. */
  groundStateEnergyKcalMol: number;
  /** Optimal variational parameters at convergence. */
  optimalParameters: Float64Array;
  /** Number of optimizer iterations consumed. */
  iterations: number;
  /** Final gradient norm at exit. */
  convergenceTolerance: number;
  /** `true` if the optimizer met the tolerance criterion. */
  converged: boolean;
}

type VqeModule = {
  _vqe_create_h2_hamiltonian: (R: number) => number;
  _vqe_create_lih_hamiltonian: (R: number) => number;
  _vqe_create_h2o_hamiltonian: () => number;
  _vqe_exact_ground_state_energy: (h: number) => number;
  _vqe_hartree_to_kcalmol: (E: number) => number;
  _pauli_hamiltonian_create: (n: number, k: number) => number;
  _pauli_hamiltonian_add_term: (h: number, c: number, s: number, i: number) => number;
  _pauli_hamiltonian_free: (h: number) => void;
  _vqe_create_hardware_efficient_ansatz: (n: number, L: number) => number;
  _vqe_create_uccsd_ansatz: (n: number, ne: number) => number;
  _vqe_ansatz_free: (a: number) => void;
  _vqe_optimizer_create: (t: number) => number;
  _vqe_optimizer_set_hyperparams: (
    o: number, lr: number, b1: number, b2: number, eps: number, reg: number,
  ) => void;
  _vqe_optimizer_free: (o: number) => void;
  _vqe_solver_create: (h: number, a: number, o: number, e: number) => number;
  _vqe_solver_free: (s: number) => void;
  _vqe_solve: (resultPtr: number, s: number) => void;
  _vqe_result_free: (resultPtr: number) => void;
  _vqe_compute_energy: (s: number, params: number) => number;
  _quantum_entropy_ctx_create_hw: () => number;
  _quantum_entropy_ctx_destroy: (c: number) => void;
  _malloc: (n: number) => number;
  _free: (p: number) => void;
  HEAPF64: Float64Array;
  HEAP32: Int32Array;
  HEAPU32: Uint32Array;
  HEAPU8: Uint8Array;
  stringToUTF8: (s: string, ptr: number, max: number) => void;
};

/* vqe_optimizer_t field offsets on wasm32 (offsets pinned in the
 * source, same pattern as the var-D config marshalling in ca-mps.ts):
 *   0  int     type
 *   4  size_t  max_iterations
 *   8  double  tolerance
 *  16  double  learning_rate
 *  24  int     verbose
 *  32  double  beta1
 *  40  double  beta2
 *  48  double  epsilon
 *  56  double  qng_regularization
 * The double hyperparameters go through the C setter
 * `vqe_optimizer_set_hyperparams` (NaN = keep default); only the two
 * fields the setter does not cover are poked directly. */
const OPT_OFF_MAX_ITERATIONS = 4;
const OPT_OFF_TOLERANCE = 8;

/* vqe_ansatz_t field offset on wasm32: num_parameters (size_t) at 12. */
const ANSATZ_OFF_NUM_PARAMETERS = 12;

/** Pauli-string Hamiltonian over `numQubits` qubits. */
export class PauliHamiltonian {
  private ptr: number;
  private mod: VqeModule;
  private freed = false;

  /** Internal: WASM pointer to the underlying ``pauli_hamiltonian_t``. */
  _internal_ptr(): number {
    if (this.freed) throw new Error('PauliHamiltonian disposed');
    return this.ptr;
  }

  private constructor(ptr: number, mod: VqeModule) {
    this.ptr = ptr;
    this.mod = mod;
  }

  /** H2 molecule Hamiltonian in the STO-3G basis under Jordan-Wigner. */
  static async h2(bondDistance: number): Promise<PauliHamiltonian> {
    const mod = (await getModule()) as unknown as VqeModule;
    const ptr = mod._vqe_create_h2_hamiltonian(bondDistance);
    if (!ptr) throw new Error('vqe_create_h2_hamiltonian returned NULL');
    return new PauliHamiltonian(ptr, mod);
  }

  /** LiH molecule Hamiltonian (4 qubits) at the given bond distance. */
  static async lih(bondDistance: number): Promise<PauliHamiltonian> {
    const mod = (await getModule()) as unknown as VqeModule;
    const ptr = mod._vqe_create_lih_hamiltonian(bondDistance);
    if (!ptr) throw new Error('vqe_create_lih_hamiltonian returned NULL');
    return new PauliHamiltonian(ptr, mod);
  }

  /** H2O molecule Hamiltonian at its equilibrium geometry. */
  static async h2o(): Promise<PauliHamiltonian> {
    const mod = (await getModule()) as unknown as VqeModule;
    const ptr = mod._vqe_create_h2o_hamiltonian();
    if (!ptr) throw new Error('vqe_create_h2o_hamiltonian returned NULL');
    return new PauliHamiltonian(ptr, mod);
  }

  /** Open a fluent builder for a custom Pauli-sum Hamiltonian. */
  static async builder(numQubits: number, numTerms: number): Promise<PauliHamiltonianBuilder> {
    const mod = (await getModule()) as unknown as VqeModule;
    const ptr = mod._pauli_hamiltonian_create(numQubits, numTerms);
    if (!ptr) throw new Error('pauli_hamiltonian_create returned NULL');
    return new PauliHamiltonianBuilder(ptr, mod, numTerms);
  }

  /** Read `num_qubits` from the underlying struct.  size_t at byte 0. */
  get numQubits(): number {
    if (this.freed) throw new Error('PauliHamiltonian disposed');
    return this.mod.HEAPU32[this.ptr >> 2];
  }

  /** Read `num_terms` from the underlying struct.  size_t at byte 4. */
  get numTerms(): number {
    if (this.freed) throw new Error('PauliHamiltonian disposed');
    return this.mod.HEAPU32[(this.ptr + 4) >> 2];
  }

  /** Exact ground-state energy by direct diagonalisation
   *  (O(4^n)).  Intended for small (n <= 10) Hamiltonians as a
   *  reference value for VQE convergence checks.  Returns `NaN`
   *  on internal failure. */
  exactGroundStateEnergy(): number {
    if (this.freed) throw new Error('PauliHamiltonian disposed');
    return this.mod._vqe_exact_ground_state_energy(this.ptr);
  }

  /** Release the underlying allocation. */
  dispose(): void {
    if (this.freed) return;
    this.mod._pauli_hamiltonian_free(this.ptr);
    this.freed = true;
  }
}

/** Fluent builder for a custom {@link PauliHamiltonian}. */
export class PauliHamiltonianBuilder {
  private ptr: number;
  private mod: VqeModule;
  private capacity: number;
  private next = 0;
  private dropped = false;

  /** @internal */
  constructor(ptr: number, mod: VqeModule, capacity: number) {
    this.ptr = ptr;
    this.mod = mod;
    this.capacity = capacity;
  }

  /** Append one Pauli-string term `coefficient * P` where `pauli` is a
   *  string over `{I, X, Y, Z}` of length `numQubits`. */
  addTerm(coefficient: number, pauli: string): this {
    if (this.next >= this.capacity) {
      throw new Error(`Hamiltonian capacity ${this.capacity} exceeded`);
    }
    // Pauli strings are pure ASCII so UTF-8 byte length equals
    // string length.  +1 for the NUL terminator.
    const len = pauli.length + 1;
    const strPtr = this.mod._malloc(len);
    try {
      this.mod.stringToUTF8(pauli, strPtr, len);
      const rc = this.mod._pauli_hamiltonian_add_term(
        this.ptr, coefficient, strPtr, this.next,
      );
      if (rc !== 0) throw new Error(`pauli_hamiltonian_add_term rc=${rc}`);
      this.next++;
    } finally {
      this.mod._free(strPtr);
    }
    return this;
  }

  /** Consume the builder and return the finished Hamiltonian. */
  build(): PauliHamiltonian {
    if (this.dropped) throw new Error('Builder already consumed');
    this.dropped = true;
    // Hand the raw pointer over to a fresh PauliHamiltonian.
    return Object.assign(Object.create((PauliHamiltonian as any).prototype), {
      ptr: this.ptr, mod: this.mod, freed: false,
    });
  }
}

/** VQE solver bundling Hamiltonian + ansatz + optimizer + entropy
 *  ctx.  Cleanup is automatic on {@link dispose}. */
export class VqeSolver {
  private solver: number;
  private ansatz: number;
  private optimizer: number;
  private entropy: number;
  private hamiltonian: PauliHamiltonian;
  private mod: VqeModule;
  private freed = false;

  private constructor(
    solver: number, ansatz: number, optimizer: number, entropy: number,
    hamiltonian: PauliHamiltonian, mod: VqeModule,
  ) {
    this.solver = solver;
    this.ansatz = ansatz;
    this.optimizer = optimizer;
    this.entropy = entropy;
    this.hamiltonian = hamiltonian;
    this.mod = mod;
  }

  /** Construct a VQE solver over `hamiltonian`.
   *
   *  Preferred form: `VqeSolver.create(h, options)` with the
   *  {@link VqeSolverOptions} bag (ansatz family, optimizer by string
   *  or enum, hyperparameters).  The legacy positional form
   *  `VqeSolver.create(h, numLayers, optimizerType)` remains
   *  supported.
   *
   *  The solver takes shared ownership of `hamiltonian` for its
   *  lifetime; the caller can dispose `hamiltonian` after the solver
   *  is disposed. */
  static async create(
    hamiltonian: PauliHamiltonian,
    optionsOrNumLayers?: VqeSolverOptions | number,
    legacyOptimizerType?: OptimizerType,
  ): Promise<VqeSolver> {
    const options: VqeSolverOptions =
      typeof optionsOrNumLayers === 'number'
        ? { numLayers: optionsOrNumLayers, optimizer: legacyOptimizerType ?? OptimizerType.Adam }
        : { ...(optionsOrNumLayers ?? {}) };

    const mod = (await getModule()) as unknown as VqeModule;
    const numQubits = hamiltonian.numQubits;
    const optimizerType = resolveOptimizer(options.optimizer ?? OptimizerType.Adam);

    const entropy = mod._quantum_entropy_ctx_create_hw();
    if (!entropy) throw new Error('quantum_entropy_ctx_create_hw returned NULL');

    const ansatzName = (options.ansatz ?? 'hardware_efficient').trim().toLowerCase();
    let ansatz: number;
    if (ansatzName === 'uccsd') {
      const numElectrons = options.numElectrons ?? Math.floor(numQubits / 2);
      ansatz = mod._vqe_create_uccsd_ansatz(numQubits, numElectrons);
      if (!ansatz) {
        mod._quantum_entropy_ctx_destroy(entropy);
        throw new Error('vqe_create_uccsd_ansatz returned NULL');
      }
    } else if (
      ansatzName === 'hardware_efficient' ||
      ansatzName === 'hardware-efficient' ||
      ansatzName === 'hea'
    ) {
      ansatz = mod._vqe_create_hardware_efficient_ansatz(
        numQubits, options.numLayers ?? 2,
      );
      if (!ansatz) {
        mod._quantum_entropy_ctx_destroy(entropy);
        throw new Error('vqe_create_hardware_efficient_ansatz returned NULL');
      }
    } else {
      mod._quantum_entropy_ctx_destroy(entropy);
      throw new Error(
        `Unknown ansatz '${options.ansatz}'. Valid: 'hardware_efficient', 'uccsd'`,
      );
    }

    const optimizer = mod._vqe_optimizer_create(optimizerType);
    if (!optimizer) {
      mod._vqe_ansatz_free(ansatz);
      mod._quantum_entropy_ctx_destroy(entropy);
      throw new Error('vqe_optimizer_create returned NULL');
    }

    // Route the double hyperparameters through the C setter (NaN slots
    // keep the per-optimizer defaults) and poke the two integer/double
    // fields the setter does not cover.
    if (
      options.learningRate !== undefined || options.beta1 !== undefined ||
      options.beta2 !== undefined || options.epsilon !== undefined ||
      options.qngRegularization !== undefined
    ) {
      mod._vqe_optimizer_set_hyperparams(
        optimizer,
        options.learningRate ?? NaN,
        options.beta1 ?? NaN,
        options.beta2 ?? NaN,
        options.epsilon ?? NaN,
        options.qngRegularization ?? NaN,
      );
    }
    if (options.maxIterations !== undefined) {
      mod.HEAPU32[(optimizer + OPT_OFF_MAX_ITERATIONS) >> 2] = options.maxIterations;
    }
    if (options.tolerance !== undefined) {
      mod.HEAPF64[(optimizer + OPT_OFF_TOLERANCE) >> 3] = options.tolerance;
    }

    const solver = mod._vqe_solver_create(
      hamiltonian._internal_ptr(), ansatz, optimizer, entropy,
    );
    if (!solver) {
      mod._vqe_optimizer_free(optimizer);
      mod._vqe_ansatz_free(ansatz);
      mod._quantum_entropy_ctx_destroy(entropy);
      throw new Error('vqe_solver_create returned NULL');
    }
    return new VqeSolver(solver, ansatz, optimizer, entropy, hamiltonian, mod);
  }

  /** Number of variational parameters in the underlying ansatz. */
  get numParameters(): number {
    if (this.freed) throw new Error('VqeSolver disposed');
    return this.mod.HEAPU32[(this.ansatz + ANSATZ_OFF_NUM_PARAMETERS) >> 2];
  }

  /** Run the classical-quantum optimisation loop to convergence. */
  solve(): VqeResult {
    if (this.freed) throw new Error('VqeSolver disposed');
    // vqe_result_t is 64 bytes on wasm32; emcc passes the struct
    // through a hidden first argument when returning by value.
    const resultPtr = this.mod._malloc(64);
    try {
      this.mod._vqe_solve(resultPtr, this.solver);
      // Field offsets confirmed by C-side probe under emcc 4.0.22:
      //  0  double  ground_state_energy
      //  8  ptr     optimal_parameters
      // 12  size_t  num_parameters
      // 16  size_t  iterations
      // 20  pad
      // 24  double  convergence_tolerance
      // 32  int     converged
      const groundStateEnergy   = this.mod.HEAPF64[(resultPtr +  0) >> 3];
      const paramsArrPtr        = this.mod.HEAPU32[(resultPtr +  8) >> 2];
      const numParameters       = this.mod.HEAPU32[(resultPtr + 12) >> 2];
      const iterations          = this.mod.HEAPU32[(resultPtr + 16) >> 2];
      const convergenceTolerance = this.mod.HEAPF64[(resultPtr + 24) >> 3];
      const converged            = this.mod.HEAP32[(resultPtr + 32) >> 2] !== 0;

      const params = paramsArrPtr && numParameters > 0
        ? new Float64Array(
            this.mod.HEAPF64.buffer, paramsArrPtr, numParameters,
          ).slice()  // copy out of WASM memory before vqe_result_free below
        : new Float64Array(0);

      const eKcalMol = this.mod._vqe_hartree_to_kcalmol(groundStateEnergy);

      return {
        groundStateEnergy,
        groundStateEnergyKcalMol: eKcalMol,
        optimalParameters: params,
        iterations,
        convergenceTolerance,
        converged,
      };
    } finally {
      // vqe_solve returns a result that owns optimal_parameters; release
      // the owned array through the C entry, then the struct storage.
      this.mod._vqe_result_free(resultPtr);
      this.mod._free(resultPtr);
    }
  }

  /** Compute `E(theta) = <psi(theta) | H | psi(theta)>` at an
   *  arbitrary parameter vector. */
  computeEnergy(parameters: Float64Array): number {
    if (this.freed) throw new Error('VqeSolver disposed');
    const bytes = parameters.length * 8;
    const ptr = this.mod._malloc(bytes);
    try {
      const idx = ptr >> 3;
      for (let i = 0; i < parameters.length; i++) {
        this.mod.HEAPF64[idx + i] = parameters[i];
      }
      return this.mod._vqe_compute_energy(this.solver, ptr);
    } finally {
      this.mod._free(ptr);
    }
  }

  /** Number of qubits the underlying Hamiltonian acts on. */
  get numQubits(): number {
    return this.hamiltonian.numQubits;
  }

  /** Release every C-side allocation owned by this solver. */
  dispose(): void {
    if (this.freed) return;
    this.mod._vqe_solver_free(this.solver);
    this.mod._vqe_optimizer_free(this.optimizer);
    this.mod._vqe_ansatz_free(this.ansatz);
    this.mod._quantum_entropy_ctx_destroy(this.entropy);
    this.freed = true;
  }
}
