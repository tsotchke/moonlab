/**
 * QuantumState Class
 *
 * High-level interface for quantum state manipulation.
 * Provides a fluent API for applying gates and measuring qubits.
 */

import type { Complex } from './complex';
import { WasmMemory, type MoonlabModule } from './memory';
import { getModule } from './wasm-loader';

/**
 * Size of quantum_state_t struct in bytes
 * This should match the C struct layout
 */
const STATE_STRUCT_SIZE = 256;

/**
 * Offset of amplitudes pointer in quantum_state_t
 * WASM32 struct layout:
 *   size_t num_qubits     @ offset 0  (4 bytes)
 *   size_t state_dim      @ offset 4  (4 bytes)
 *   complex_t *amplitudes @ offset 8  (4 bytes pointer)
 */
const AMPLITUDES_OFFSET = 8;

/**
 * Offset of num_qubits in quantum_state_t
 */
const NUM_QUBITS_OFFSET = 0;

/**
 * Options for creating a quantum state
 */
export interface QuantumStateOptions {
  /**
   * Number of qubits (1-20 recommended, up to 30 possible)
   */
  numQubits: number;

  /**
   * Initial amplitudes (optional, defaults to |0...0⟩)
   */
  amplitudes?: Complex[];
}

/**
 * Quantum state class with fluent API
 *
 * @example
 * ```typescript
 * const state = await QuantumState.create({ numQubits: 2 });
 * state.h(0).cnot(0, 1);  // Create Bell state
 * const probs = state.getProbabilities();
 * state.dispose();
 * ```
 */
export class QuantumState {
  private module: MoonlabModule;
  private memory: WasmMemory;
  private statePtr: number;
  private _numQubits: number;
  private _disposed: boolean = false;

  /**
   * Private constructor - use QuantumState.create() instead
   */
  private constructor(
    module: MoonlabModule,
    memory: WasmMemory,
    statePtr: number,
    numQubits: number
  ) {
    this.module = module;
    this.memory = memory;
    this.statePtr = statePtr;
    this._numQubits = numQubits;
  }

  /**
   * Create a new quantum state
   */
  static async create(options: QuantumStateOptions): Promise<QuantumState> {
    if (options.numQubits < 1 || options.numQubits > 30) {
      throw new Error('numQubits must be between 1 and 30');
    }

    const module = await getModule();
    const memory = new WasmMemory(module);

    // Allocate quantum_state_t struct
    const statePtr = memory.alloc(STATE_STRUCT_SIZE);

    // Initialize the state
    const result = module._quantum_state_init(statePtr, options.numQubits);
    if (result !== 0) {
      memory.free(statePtr);
      throw new Error(`Failed to initialize quantum state: error ${result}`);
    }

    const state = new QuantumState(module, memory, statePtr, options.numQubits);

    // Set initial amplitudes if provided
    if (options.amplitudes) {
      state.setAmplitudes(options.amplitudes);
    }

    return state;
  }

  // =========================================================================
  // Properties
  // =========================================================================

  /**
   * Number of qubits in this state
   */
  get numQubits(): number {
    return this._numQubits;
  }

  /**
   * Dimension of state space (2^n)
   */
  get stateDim(): number {
    return 1 << this._numQubits;
  }

  /**
   * Check if the state has been disposed
   */
  get isDisposed(): boolean {
    return this._disposed;
  }

  // =========================================================================
  // State Operations
  // =========================================================================

  /**
   * Reset to |0...0⟩ state
   */
  reset(): this {
    this.checkDisposed();
    this.module._quantum_state_reset(this.statePtr);
    return this;
  }

  /**
   * Clone this state
   */
  async clone(): Promise<QuantumState> {
    this.checkDisposed();
    return QuantumState.create({
      numQubits: this._numQubits,
      amplitudes: this.getAmplitudes(),
    });
  }

  /**
   * Normalize the state vector
   */
  normalize(): this {
    this.checkDisposed();
    this.module._quantum_state_normalize(this.statePtr);
    return this;
  }

  // =========================================================================
  // Single-Qubit Gates
  // =========================================================================

  /**
   * Hadamard gate - creates superposition
   * H|0⟩ = (|0⟩ + |1⟩)/√2
   */
  h(qubit: number): this {
    this.checkDisposed();
    this.checkQubit(qubit);
    this.module._gate_hadamard(this.statePtr, qubit);
    return this;
  }

  /**
   * Pauli-X gate (NOT gate)
   * X|0⟩ = |1⟩, X|1⟩ = |0⟩
   */
  x(qubit: number): this {
    this.checkDisposed();
    this.checkQubit(qubit);
    this.module._gate_pauli_x(this.statePtr, qubit);
    return this;
  }

  /**
   * Pauli-Y gate
   * Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
   */
  y(qubit: number): this {
    this.checkDisposed();
    this.checkQubit(qubit);
    this.module._gate_pauli_y(this.statePtr, qubit);
    return this;
  }

  /**
   * Pauli-Z gate (phase flip)
   * Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
   */
  z(qubit: number): this {
    this.checkDisposed();
    this.checkQubit(qubit);
    this.module._gate_pauli_z(this.statePtr, qubit);
    return this;
  }

  /**
   * S gate (√Z, phase gate)
   * S|0⟩ = |0⟩, S|1⟩ = i|1⟩
   */
  s(qubit: number): this {
    this.checkDisposed();
    this.checkQubit(qubit);
    this.module._gate_s(this.statePtr, qubit);
    return this;
  }

  /**
   * S-dagger gate (S†)
   */
  sdg(qubit: number): this {
    this.checkDisposed();
    this.checkQubit(qubit);
    this.module._gate_s_dagger(this.statePtr, qubit);
    return this;
  }

  /**
   * T gate (√S, π/8 gate)
   */
  t(qubit: number): this {
    this.checkDisposed();
    this.checkQubit(qubit);
    this.module._gate_t(this.statePtr, qubit);
    return this;
  }

  /**
   * T-dagger gate (T†)
   */
  tdg(qubit: number): this {
    this.checkDisposed();
    this.checkQubit(qubit);
    this.module._gate_t_dagger(this.statePtr, qubit);
    return this;
  }

  /**
   * Rotation around X-axis
   * Rx(θ) = exp(-iθX/2)
   */
  rx(qubit: number, angle: number): this {
    this.checkDisposed();
    this.checkQubit(qubit);
    this.module._gate_rx(this.statePtr, qubit, angle);
    return this;
  }

  /**
   * Rotation around Y-axis
   * Ry(θ) = exp(-iθY/2)
   */
  ry(qubit: number, angle: number): this {
    this.checkDisposed();
    this.checkQubit(qubit);
    this.module._gate_ry(this.statePtr, qubit, angle);
    return this;
  }

  /**
   * Rotation around Z-axis
   * Rz(θ) = exp(-iθZ/2)
   */
  rz(qubit: number, angle: number): this {
    this.checkDisposed();
    this.checkQubit(qubit);
    this.module._gate_rz(this.statePtr, qubit, angle);
    return this;
  }

  /**
   * Phase gate P(θ)
   * P(θ)|0⟩ = |0⟩, P(θ)|1⟩ = e^(iθ)|1⟩
   */
  phase(qubit: number, angle: number): this {
    this.checkDisposed();
    this.checkQubit(qubit);
    this.module._gate_phase(this.statePtr, qubit, angle);
    return this;
  }

  /**
   * General single-qubit unitary U3(θ, φ, λ)
   */
  u3(qubit: number, theta: number, phi: number, lambda: number): this {
    this.checkDisposed();
    this.checkQubit(qubit);
    this.module._gate_u3(this.statePtr, qubit, theta, phi, lambda);
    return this;
  }

  // =========================================================================
  // Two-Qubit Gates
  // =========================================================================

  /**
   * Controlled-NOT (CNOT) gate
   * Flips target if control is |1⟩
   */
  cnot(control: number, target: number): this {
    this.checkDisposed();
    this.checkQubit(control);
    this.checkQubit(target);
    if (control === target) {
      throw new Error('Control and target must be different qubits');
    }
    this.module._gate_cnot(this.statePtr, control, target);
    return this;
  }

  /**
   * Alias for CNOT
   */
  cx(control: number, target: number): this {
    return this.cnot(control, target);
  }

  /**
   * Controlled-Z gate
   */
  cz(control: number, target: number): this {
    this.checkDisposed();
    this.checkQubit(control);
    this.checkQubit(target);
    if (control === target) {
      throw new Error('Control and target must be different qubits');
    }
    this.module._gate_cz(this.statePtr, control, target);
    return this;
  }

  /**
   * Controlled-Y gate
   */
  cy(control: number, target: number): this {
    this.checkDisposed();
    this.checkQubit(control);
    this.checkQubit(target);
    if (control === target) {
      throw new Error('Control and target must be different qubits');
    }
    this.module._gate_cy(this.statePtr, control, target);
    return this;
  }

  /**
   * SWAP gate - exchanges two qubits
   */
  swap(qubit1: number, qubit2: number): this {
    this.checkDisposed();
    this.checkQubit(qubit1);
    this.checkQubit(qubit2);
    if (qubit1 === qubit2) {
      return this; // SWAP(q,q) is identity
    }
    this.module._gate_swap(this.statePtr, qubit1, qubit2);
    return this;
  }

  /**
   * Controlled phase gate CP(θ)
   */
  cphase(control: number, target: number, angle: number): this {
    this.checkDisposed();
    this.checkQubit(control);
    this.checkQubit(target);
    if (control === target) {
      throw new Error('Control and target must be different qubits');
    }
    this.module._gate_cphase(this.statePtr, control, target, angle);
    return this;
  }

  /**
   * Controlled Rx gate
   */
  crx(control: number, target: number, angle: number): this {
    this.checkDisposed();
    this.checkQubit(control);
    this.checkQubit(target);
    if (control === target) {
      throw new Error('Control and target must be different qubits');
    }
    this.module._gate_crx(this.statePtr, control, target, angle);
    return this;
  }

  /**
   * Controlled Ry gate
   */
  cry(control: number, target: number, angle: number): this {
    this.checkDisposed();
    this.checkQubit(control);
    this.checkQubit(target);
    if (control === target) {
      throw new Error('Control and target must be different qubits');
    }
    this.module._gate_cry(this.statePtr, control, target, angle);
    return this;
  }

  /**
   * Controlled Rz gate
   */
  crz(control: number, target: number, angle: number): this {
    this.checkDisposed();
    this.checkQubit(control);
    this.checkQubit(target);
    if (control === target) {
      throw new Error('Control and target must be different qubits');
    }
    this.module._gate_crz(this.statePtr, control, target, angle);
    return this;
  }

  // =========================================================================
  // Three-Qubit Gates
  // =========================================================================

  /**
   * Toffoli gate (CCNOT)
   * Flips target if both controls are |1⟩
   */
  toffoli(control1: number, control2: number, target: number): this {
    this.checkDisposed();
    this.checkQubit(control1);
    this.checkQubit(control2);
    this.checkQubit(target);
    if (control1 === control2 || control1 === target || control2 === target) {
      throw new Error('All qubits must be different');
    }
    this.module._gate_toffoli(this.statePtr, control1, control2, target);
    return this;
  }

  /**
   * Alias for Toffoli
   */
  ccx(control1: number, control2: number, target: number): this {
    return this.toffoli(control1, control2, target);
  }

  /**
   * Fredkin gate (CSWAP)
   * Swaps targets if control is |1⟩
   */
  fredkin(control: number, target1: number, target2: number): this {
    this.checkDisposed();
    this.checkQubit(control);
    this.checkQubit(target1);
    this.checkQubit(target2);
    if (control === target1 || control === target2 || target1 === target2) {
      throw new Error('All qubits must be different');
    }
    this.module._gate_fredkin(this.statePtr, control, target1, target2);
    return this;
  }

  /**
   * Alias for Fredkin
   */
  cswap(control: number, target1: number, target2: number): this {
    return this.fredkin(control, target1, target2);
  }

  // =========================================================================
  // Multi-Qubit Operations
  // =========================================================================

  /**
   * Quantum Fourier Transform on specified qubits
   */
  qft(qubits: number[]): this {
    this.checkDisposed();
    for (const q of qubits) {
      this.checkQubit(q);
    }

    const ptr = this.memory.allocInt32Array(qubits.length);
    this.memory.writeInt32Array(ptr, qubits);
    this.module._gate_qft(this.statePtr, ptr, qubits.length);
    this.memory.free(ptr);
    return this;
  }

  /**
   * Inverse Quantum Fourier Transform
   */
  iqft(qubits: number[]): this {
    this.checkDisposed();
    for (const q of qubits) {
      this.checkQubit(q);
    }

    const ptr = this.memory.allocInt32Array(qubits.length);
    this.memory.writeInt32Array(ptr, qubits);
    this.module._gate_iqft(this.statePtr, ptr, qubits.length);
    this.memory.free(ptr);
    return this;
  }

  // =========================================================================
  // State Queries
  // =========================================================================

  /**
   * Get all amplitudes as complex numbers
   */
  getAmplitudes(): Complex[] {
    this.checkDisposed();
    const amplitudesPtr = this.memory.readPointer(
      this.statePtr + AMPLITUDES_OFFSET
    );
    return this.memory.readComplexArray(amplitudesPtr, this.stateDim);
  }

  /**
   * Set amplitudes (must have correct dimension)
   */
  setAmplitudes(amplitudes: Complex[]): this {
    this.checkDisposed();
    if (amplitudes.length !== this.stateDim) {
      throw new Error(
        `Expected ${this.stateDim} amplitudes, got ${amplitudes.length}`
      );
    }
    const amplitudesPtr = this.memory.readPointer(
      this.statePtr + AMPLITUDES_OFFSET
    );
    this.memory.writeComplexArray(amplitudesPtr, amplitudes);
    return this;
  }

  /**
   * Get probability distribution for all basis states
   */
  getProbabilities(): Float64Array {
    this.checkDisposed();
    const ptr = this.memory.allocFloat64Array(this.stateDim);
    this.module._measurement_probability_distribution(this.statePtr, ptr);
    const probs = this.memory.readFloat64Array(ptr, this.stateDim);
    this.memory.free(ptr);
    return probs;
  }

  /**
   * Get probability of a specific basis state
   */
  probability(basisState: number): number {
    this.checkDisposed();
    if (basisState < 0 || basisState >= this.stateDim) {
      throw new Error(`Basis state must be between 0 and ${this.stateDim - 1}`);
    }
    return this.module._quantum_state_get_probability(this.statePtr, basisState);
  }

  // =========================================================================
  // Measurement
  // =========================================================================

  /**
   * Measure a single qubit (collapses state)
   * @returns 0 or 1
   */
  measure(qubit: number): number {
    this.checkDisposed();
    this.checkQubit(qubit);
    const randomValue = Math.random();
    return this.module._measurement_single_qubit(
      this.statePtr,
      qubit,
      randomValue
    );
  }

  /**
   * Measure all qubits (collapses state)
   * @returns Basis state index (0 to 2^n - 1)
   */
  measureAll(): number {
    this.checkDisposed();
    const randomValue = Math.random();
    return this.module._measurement_all_qubits(this.statePtr, randomValue);
  }

  /**
   * Get probability of measuring |0⟩ on a qubit (non-destructive)
   */
  probabilityZero(qubit: number): number {
    this.checkDisposed();
    this.checkQubit(qubit);
    return this.module._measurement_probability_zero(this.statePtr, qubit);
  }

  /**
   * Get probability of measuring |1⟩ on a qubit (non-destructive)
   */
  probabilityOne(qubit: number): number {
    this.checkDisposed();
    this.checkQubit(qubit);
    return this.module._measurement_probability_one(this.statePtr, qubit);
  }

  // =========================================================================
  // Expectation Values
  // =========================================================================

  /**
   * Expectation value of Z operator on a qubit
   * ⟨Z⟩ = P(0) - P(1)
   */
  expectationZ(qubit: number): number {
    this.checkDisposed();
    this.checkQubit(qubit);
    return this.module._measurement_expectation_z(this.statePtr, qubit);
  }

  /**
   * Expectation value of X operator on a qubit
   */
  expectationX(qubit: number): number {
    this.checkDisposed();
    this.checkQubit(qubit);
    return this.module._measurement_expectation_x(this.statePtr, qubit);
  }

  /**
   * Expectation value of Y operator on a qubit
   */
  expectationY(qubit: number): number {
    this.checkDisposed();
    this.checkQubit(qubit);
    return this.module._measurement_expectation_y(this.statePtr, qubit);
  }

  /**
   * Two-qubit ZZ correlation ⟨Z_i Z_j⟩
   */
  correlationZZ(qubit1: number, qubit2: number): number {
    this.checkDisposed();
    this.checkQubit(qubit1);
    this.checkQubit(qubit2);
    return this.module._measurement_correlation_zz(
      this.statePtr,
      qubit1,
      qubit2
    );
  }

  // =========================================================================
  // State Properties
  // =========================================================================

  /**
   * Von Neumann entropy of the state
   */
  entropy(): number {
    this.checkDisposed();
    return this.module._quantum_state_entropy(this.statePtr);
  }

  /**
   * Purity of the state (1 for pure states)
   */
  purity(): number {
    this.checkDisposed();
    return this.module._quantum_state_purity(this.statePtr);
  }

  /**
   * Fidelity with another state
   * F = |⟨ψ|φ⟩|²
   */
  fidelity(other: QuantumState): number {
    this.checkDisposed();
    other.checkDisposed();
    if (this._numQubits !== other._numQubits) {
      throw new Error('States must have same number of qubits');
    }
    return this.module._quantum_state_fidelity(this.statePtr, other.statePtr);
  }

  // =========================================================================
  // Cleanup
  // =========================================================================

  /**
   * Dispose of the state and free memory
   * Must be called when done with the state
   */
  dispose(): void {
    if (!this._disposed) {
      this.module._quantum_state_free(this.statePtr);
      this.memory.free(this.statePtr);
      this._disposed = true;
    }
  }

  // =========================================================================
  // Internal Helpers
  // =========================================================================

  private checkDisposed(): void {
    if (this._disposed) {
      throw new Error('QuantumState has been disposed');
    }
  }

  private checkQubit(qubit: number): void {
    if (qubit < 0 || qubit >= this._numQubits) {
      throw new Error(
        `Qubit index ${qubit} out of range [0, ${this._numQubits - 1}]`
      );
    }
  }

  /**
   * Get the internal pointer (for advanced usage)
   * @internal
   */
  get _ptr(): number {
    return this.statePtr;
  }
}
