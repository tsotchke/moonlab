/**
 * Quantum Approximate Optimization Algorithm (C-side since v0.2.0;
 * JS binding since v0.5.5).
 *
 * Wraps ``src/algorithms/qaoa.{c,h}`` around the Ising-model
 * encoding, the MaxCut graph helper, and the ``qaoa_solve`` driver.
 * Mirrors the Python ``moonlab.algorithms.QAOA`` and Rust
 * ``moonlab::qaoa`` surfaces.
 *
 * @example
 * ```typescript
 * import { Graph, IsingModel, QaoaSolver } from '@moonlab/quantum-core';
 *
 * // Triangle MaxCut: optimal cut value = 2.
 * const g = await Graph.create(3, [[0, 1, 1.0], [1, 2, 1.0], [2, 0, 1.0]]);
 * const ising = await IsingModel.fromMaxcut(g);
 * const solver = await QaoaSolver.create(ising, 1);
 * const r = solver.solve();
 * console.log(`best energy = ${r.bestEnergy.toFixed(3)}`);
 * solver.dispose();
 * ```
 */

import { getModule } from './wasm-loader';

/** QAOA optimisation result.  Mirrors ``qaoa_result_t`` minus the
 *  raw pointers; `optimalGamma` / `optimalBeta` / `energyHistory`
 *  are copied into owned `Float64Array`s. */
export interface QaoaResult {
  /** Bitstring of the best-energy sample seen during optimisation. */
  bestBitstring: bigint;
  /** Energy at `bestBitstring`. */
  bestEnergy: number;
  /** Number of optimiser iterations consumed. */
  numIterations: number;
  /** `true` if the optimiser hit the convergence tolerance. */
  converged: boolean;
  /** Solution quality vs the global optimum. */
  approximationRatio: number;
  /** Optimal cost-Hamiltonian angles `gamma_1, ..., gamma_p`. */
  optimalGamma: Float64Array;
  /** Optimal mixer-Hamiltonian angles `beta_1, ..., beta_p`. */
  optimalBeta: Float64Array;
  /** QAOA depth `p`. */
  numLayers: number;
  /** Total shot count consumed across the run. */
  totalMeasurements: number;
}

type QaoaModule = {
  _graph_create: (numVertices: number, numEdges: number) => number;
  _graph_free: (g: number) => void;
  _graph_add_edge: (g: number, idx: number, u: number, v: number, w: number) => number;
  _ising_model_create: (numQubits: number) => number;
  _ising_model_free: (ising: number) => void;
  _ising_model_set_coupling: (ising: number, i: number, j: number, v: number) => number;
  _ising_model_set_field: (ising: number, i: number, v: number) => number;
  _ising_model_evaluate: (ising: number, bitstring: bigint) => number;
  _ising_encode_maxcut: (graph: number) => number;
  _qaoa_solver_create: (ising: number, numLayers: number, entropy: number) => number;
  _qaoa_solver_free: (solver: number) => void;
  _qaoa_solve: (resultPtr: number, solver: number) => void;
  _qaoa_compute_expectation: (solver: number, gamma: number, beta: number) => number;
  _quantum_entropy_ctx_create_hw: () => number;
  _quantum_entropy_ctx_destroy: (c: number) => void;
  _malloc: (n: number) => number;
  _free: (p: number) => void;
  HEAPF64: Float64Array;
  HEAP32: Int32Array;
  HEAPU32: Uint32Array;
};

/** Weighted undirected graph used to encode MaxCut. */
export class Graph {
  private ptr: number;
  private mod: QaoaModule;
  private freed = false;

  /** @internal */
  _internal_ptr(): number {
    if (this.freed) throw new Error('Graph disposed');
    return this.ptr;
  }

  private constructor(ptr: number, mod: QaoaModule) {
    this.ptr = ptr;
    this.mod = mod;
  }

  /** Build a graph on `numVertices` vertices with the given
   *  weighted edges `[u, v, weight]`. */
  static async create(
    numVertices: number,
    edges: ReadonlyArray<readonly [number, number, number]>,
  ): Promise<Graph> {
    const mod = (await getModule()) as unknown as QaoaModule;
    const ptr = mod._graph_create(numVertices, edges.length);
    if (!ptr) throw new Error('graph_create returned NULL');
    for (let i = 0; i < edges.length; i++) {
      const [u, v, w] = edges[i];
      const rc = mod._graph_add_edge(ptr, i, u, v, w);
      if (rc !== 0) {
        mod._graph_free(ptr);
        throw new Error(`graph_add_edge(idx=${i}, u=${u}, v=${v}) rc=${rc}`);
      }
    }
    return new Graph(ptr, mod);
  }

  /** Release the underlying allocation. */
  dispose(): void {
    if (this.freed) return;
    this.mod._graph_free(this.ptr);
    this.freed = true;
  }
}

/** Ising-model Hamiltonian
 *  `H = sum_{i<j} J_{ij} s_i s_j + sum_i h_i s_i`. */
export class IsingModel {
  private ptr: number;
  private mod: QaoaModule;
  private freed = false;

  /** @internal */
  _internal_ptr(): number {
    if (this.freed) throw new Error('IsingModel disposed');
    return this.ptr;
  }

  private constructor(ptr: number, mod: QaoaModule) {
    this.ptr = ptr;
    this.mod = mod;
  }

  /** Empty Ising model on `numQubits` spins.  Add couplings / fields
   *  via {@link setCoupling} / {@link setField}. */
  static async create(numQubits: number): Promise<IsingModel> {
    const mod = (await getModule()) as unknown as QaoaModule;
    const ptr = mod._ising_model_create(numQubits);
    if (!ptr) throw new Error('ising_model_create returned NULL');
    return new IsingModel(ptr, mod);
  }

  /** Encode a MaxCut problem on `graph` as the equivalent Ising
   *  model: `H = sum_{(i,j) in E} w_{ij} (1 - s_i s_j) / 2`. */
  static async fromMaxcut(graph: Graph): Promise<IsingModel> {
    const mod = (await getModule()) as unknown as QaoaModule;
    const ptr = mod._ising_encode_maxcut(graph._internal_ptr());
    if (!ptr) throw new Error('ising_encode_maxcut returned NULL');
    return new IsingModel(ptr, mod);
  }

  /** Set the symmetric coupling `J_{ij}`. */
  setCoupling(i: number, j: number, value: number): this {
    if (this.freed) throw new Error('IsingModel disposed');
    const rc = this.mod._ising_model_set_coupling(this.ptr, i, j, value);
    if (rc !== 0) throw new Error(`ising_model_set_coupling(${i},${j}) rc=${rc}`);
    return this;
  }

  /** Set the on-site field `h_i`. */
  setField(i: number, value: number): this {
    if (this.freed) throw new Error('IsingModel disposed');
    const rc = this.mod._ising_model_set_field(this.ptr, i, value);
    if (rc !== 0) throw new Error(`ising_model_set_field(${i}) rc=${rc}`);
    return this;
  }

  /** Evaluate the classical Ising energy on the bitstring
   *  `bits[i] = (bitstring >> i) & 1`. */
  evaluate(bitstring: bigint): number {
    if (this.freed) throw new Error('IsingModel disposed');
    return this.mod._ising_model_evaluate(this.ptr, bitstring);
  }

  /** Release the underlying allocation. */
  dispose(): void {
    if (this.freed) return;
    this.mod._ising_model_free(this.ptr);
    this.freed = true;
  }
}

/** QAOA solver bundling `(ising, numLayers, entropy)`. */
export class QaoaSolver {
  private solver: number;
  private entropy: number;
  private ising: IsingModel;
  private mod: QaoaModule;
  private freed = false;

  private constructor(
    solver: number, entropy: number, ising: IsingModel, mod: QaoaModule,
  ) {
    this.solver = solver;
    this.entropy = entropy;
    this.ising = ising;
    this.mod = mod;
  }

  /** Build a QAOA solver at depth `numLayers` over `ising`. */
  static async create(ising: IsingModel, numLayers: number): Promise<QaoaSolver> {
    if (numLayers < 1) throw new RangeError('numLayers must be >= 1');
    const mod = (await getModule()) as unknown as QaoaModule;
    const entropy = mod._quantum_entropy_ctx_create_hw();
    if (!entropy) throw new Error('quantum_entropy_ctx_create_hw returned NULL');
    const solver = mod._qaoa_solver_create(
      ising._internal_ptr(), numLayers, entropy,
    );
    if (!solver) {
      mod._quantum_entropy_ctx_destroy(entropy);
      throw new Error('qaoa_solver_create returned NULL');
    }
    return new QaoaSolver(solver, entropy, ising, mod);
  }

  /** Run the QAOA optimisation loop to convergence. */
  solve(): QaoaResult {
    if (this.freed) throw new Error('QaoaSolver disposed');
    // qaoa_result_t is 64 bytes on wasm32; emcc passes the struct
    // through a hidden first argument when returning by value.
    const resultPtr = this.mod._malloc(64);
    try {
      this.mod._qaoa_solve(resultPtr, this.solver);
      // Field offsets confirmed by C-side probe under emcc 4.0.22:
      //  0  uint64_t best_bitstring
      //  8  double   best_energy
      // 16  ptr      energy_history
      // 20  size_t   num_iterations
      // 24  int      converged
      // 32  double   approximation_ratio
      // 40  ptr      optimal_gamma
      // 44  ptr      optimal_beta
      // 48  size_t   num_layers
      // 52  size_t   total_measurements
      const bitLo = BigInt(this.mod.HEAPU32[resultPtr >> 2]);
      const bitHi = BigInt(this.mod.HEAPU32[(resultPtr + 4) >> 2]);
      const bestBitstring     = (bitHi << 32n) | bitLo;
      const bestEnergy        = this.mod.HEAPF64[(resultPtr + 8) >> 3];
      const numIterations     = this.mod.HEAPU32[(resultPtr + 20) >> 2];
      const converged         = this.mod.HEAP32[(resultPtr + 24) >> 2] !== 0;
      const approximationRatio = this.mod.HEAPF64[(resultPtr + 32) >> 3];
      const gammaPtr          = this.mod.HEAPU32[(resultPtr + 40) >> 2];
      const betaPtr           = this.mod.HEAPU32[(resultPtr + 44) >> 2];
      const numLayers         = this.mod.HEAPU32[(resultPtr + 48) >> 2];
      const totalMeasurements = this.mod.HEAPU32[(resultPtr + 52) >> 2];

      const copyOut = (ptr: number, n: number): Float64Array => (
        ptr && n > 0
          ? new Float64Array(this.mod.HEAPF64.buffer, ptr, n).slice()
          : new Float64Array(0)
      );
      return {
        bestBitstring,
        bestEnergy,
        numIterations,
        converged,
        approximationRatio,
        optimalGamma: copyOut(gammaPtr, numLayers),
        optimalBeta: copyOut(betaPtr, numLayers),
        numLayers,
        totalMeasurements,
      };
    } finally {
      this.mod._free(resultPtr);
    }
  }

  /** Evaluate `<H_C>(gamma, beta)` at the given angles without
   *  running the optimiser. */
  computeExpectation(gamma: Float64Array, beta: Float64Array): number {
    if (this.freed) throw new Error('QaoaSolver disposed');
    const gPtr = this.mod._malloc(gamma.length * 8);
    const bPtr = this.mod._malloc(beta.length * 8);
    try {
      for (let i = 0; i < gamma.length; i++) {
        this.mod.HEAPF64[(gPtr >> 3) + i] = gamma[i];
      }
      for (let i = 0; i < beta.length; i++) {
        this.mod.HEAPF64[(bPtr >> 3) + i] = beta[i];
      }
      return this.mod._qaoa_compute_expectation(this.solver, gPtr, bPtr);
    } finally {
      this.mod._free(gPtr);
      this.mod._free(bPtr);
    }
  }

  /** Release every C-side allocation owned by this solver. */
  dispose(): void {
    if (this.freed) return;
    this.mod._qaoa_solver_free(this.solver);
    this.mod._quantum_entropy_ctx_destroy(this.entropy);
    this.freed = true;
  }
}
