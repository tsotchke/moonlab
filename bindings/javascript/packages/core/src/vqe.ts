/**
 * Variational Quantum Eigensolver (C-side since v0.2.0; JS binding
 * since v0.5.5).
 *
 * Wraps ``src/algorithms/vqe.{c,h}`` around the Pauli-Hamiltonian
 * builder, the hardware-efficient ansatz, the four classical
 * optimizers (Adam / L-BFGS / COBYLA / gradient-descent), and the
 * ``vqe_solve`` driver.  Mirrors the Python
 * ``moonlab.algorithms.VQE`` and Rust ``moonlab::vqe`` surfaces.
 *
 * @example
 * ```typescript
 * import { PauliHamiltonian, VqeSolver, OptimizerType } from '@moonlab/quantum-core';
 *
 * const h = await PauliHamiltonian.h2(0.74);
 * console.log('exact E_0 =', h.exactGroundStateEnergy(), 'Ha');
 *
 * const solver = await VqeSolver.create(h, 2, OptimizerType.Adam);
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
  _vqe_exact_ground_state_energy: (h: number) => number;
  _vqe_hartree_to_kcalmol: (E: number) => number;
  _pauli_hamiltonian_create: (n: number, k: number) => number;
  _pauli_hamiltonian_add_term: (h: number, c: number, s: number, i: number) => number;
  _pauli_hamiltonian_free: (h: number) => void;
  _vqe_create_hardware_efficient_ansatz: (n: number, L: number) => number;
  _vqe_ansatz_free: (a: number) => void;
  _vqe_optimizer_create: (t: number) => number;
  _vqe_optimizer_free: (o: number) => void;
  _vqe_solver_create: (h: number, a: number, o: number, e: number) => number;
  _vqe_solver_free: (s: number) => void;
  _vqe_solve: (resultPtr: number, s: number) => void;
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

  /** Construct a VQE solver over `hamiltonian` using a hardware-
   *  efficient ansatz of depth `numLayers` and the given classical
   *  optimizer.  The solver takes shared ownership of `hamiltonian`
   *  for its lifetime; the caller can dispose `hamiltonian` after
   *  the solver is disposed. */
  static async create(
    hamiltonian: PauliHamiltonian,
    numLayers: number,
    optimizerType: OptimizerType,
  ): Promise<VqeSolver> {
    const mod = (await getModule()) as unknown as VqeModule;
    const entropy = mod._quantum_entropy_ctx_create_hw();
    if (!entropy) throw new Error('quantum_entropy_ctx_create_hw returned NULL');
    const ansatz = mod._vqe_create_hardware_efficient_ansatz(
      hamiltonian.numQubits, numLayers,
    );
    if (!ansatz) {
      mod._quantum_entropy_ctx_destroy(entropy);
      throw new Error('vqe_create_hardware_efficient_ansatz returned NULL');
    }
    const optimizer = mod._vqe_optimizer_create(optimizerType);
    if (!optimizer) {
      mod._vqe_ansatz_free(ansatz);
      mod._quantum_entropy_ctx_destroy(entropy);
      throw new Error('vqe_optimizer_create returned NULL');
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
          ).slice()  // copy out of WASM memory before VQE solver frees it
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
