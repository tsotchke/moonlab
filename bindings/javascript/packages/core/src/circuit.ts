/**
 * Circuit Builder
 *
 * Provides a declarative way to build quantum circuits that can be
 * applied to quantum states. Circuits are reusable and can be combined.
 */

import type { QuantumState } from './quantum-state';

// ============================================================================
// Gate Type Definitions
// ============================================================================

/**
 * Single-qubit gate types
 */
export type SingleQubitGateType =
  | 'h'
  | 'x'
  | 'y'
  | 'z'
  | 's'
  | 'sdg'
  | 't'
  | 'tdg';

/**
 * Parameterized single-qubit gate types
 */
export type ParameterizedGateType = 'rx' | 'ry' | 'rz' | 'phase' | 'u3';

/**
 * Two-qubit gate types
 */
export type TwoQubitGateType =
  | 'cnot'
  | 'cx'
  | 'cz'
  | 'cy'
  | 'swap'
  | 'cphase'
  | 'crx'
  | 'cry'
  | 'crz';

/**
 * Three-qubit gate types
 */
export type ThreeQubitGateType = 'toffoli' | 'ccx' | 'fredkin' | 'cswap';

/**
 * Multi-qubit gate types
 */
export type MultiQubitGateType = 'qft' | 'iqft';

/**
 * All gate types
 */
export type GateType =
  | SingleQubitGateType
  | ParameterizedGateType
  | TwoQubitGateType
  | ThreeQubitGateType
  | MultiQubitGateType
  | 'barrier'
  | 'measure';

// ============================================================================
// Gate Definitions
// ============================================================================

/**
 * Single-qubit gate (no parameters)
 */
export interface SingleQubitGate {
  type: SingleQubitGateType;
  qubit: number;
}

/**
 * Parameterized single-qubit gate
 */
export interface ParameterizedGate {
  type: ParameterizedGateType;
  qubit: number;
  params: number[];
}

/**
 * Two-qubit gate
 */
export interface TwoQubitGate {
  type: TwoQubitGateType;
  control: number;
  target: number;
  param?: number;
}

/**
 * Three-qubit gate
 */
export interface ThreeQubitGate {
  type: ThreeQubitGateType;
  qubits: [number, number, number];
}

/**
 * Multi-qubit gate (variable number of qubits)
 */
export interface MultiQubitGate {
  type: MultiQubitGateType;
  qubits: number[];
}

/**
 * Barrier (for visualization, doesn't affect state)
 */
export interface BarrierGate {
  type: 'barrier';
  qubits: number[];
}

/**
 * Measurement gate
 */
export interface MeasureGate {
  type: 'measure';
  qubit: number;
  classicalBit?: number;
}

/**
 * Union type for all gates
 */
export type Gate =
  | SingleQubitGate
  | ParameterizedGate
  | TwoQubitGate
  | ThreeQubitGate
  | MultiQubitGate
  | BarrierGate
  | MeasureGate;

// ============================================================================
// Circuit Statistics
// ============================================================================

/**
 * Statistics about a circuit
 */
export interface CircuitStats {
  numQubits: number;
  depth: number;
  totalGates: number;
  singleQubitGates: number;
  twoQubitGates: number;
  threeQubitGates: number;
  measurements: number;
  gateBreakdown: Record<string, number>;
}

// ============================================================================
// Circuit Class
// ============================================================================

/**
 * Quantum Circuit Builder
 *
 * Build reusable quantum circuits with a fluent API.
 *
 * @example
 * ```typescript
 * const bellCircuit = new Circuit(2)
 *   .h(0)
 *   .cnot(0, 1);
 *
 * const state = await QuantumState.create({ numQubits: 2 });
 * bellCircuit.apply(state);
 * ```
 */
export class Circuit {
  private _numQubits: number;
  private _gates: Gate[] = [];
  private _name?: string;

  /**
   * Create a new circuit
   * @param numQubits Number of qubits in the circuit
   * @param name Optional name for the circuit
   */
  constructor(numQubits: number, name?: string) {
    if (numQubits < 1 || numQubits > 30) {
      throw new Error('numQubits must be between 1 and 30');
    }
    this._numQubits = numQubits;
    this._name = name;
  }

  // =========================================================================
  // Properties
  // =========================================================================

  /**
   * Number of qubits in the circuit
   */
  get numQubits(): number {
    return this._numQubits;
  }

  /**
   * Circuit name
   */
  get name(): string | undefined {
    return this._name;
  }

  /**
   * Get all gates in the circuit
   */
  get gates(): readonly Gate[] {
    return this._gates;
  }

  /**
   * Number of gates in the circuit
   */
  get length(): number {
    return this._gates.length;
  }

  // =========================================================================
  // Single-Qubit Gates
  // =========================================================================

  /**
   * Hadamard gate
   */
  h(qubit: number): this {
    this.validateQubit(qubit);
    this._gates.push({ type: 'h', qubit });
    return this;
  }

  /**
   * Pauli-X gate (NOT)
   */
  x(qubit: number): this {
    this.validateQubit(qubit);
    this._gates.push({ type: 'x', qubit });
    return this;
  }

  /**
   * Pauli-Y gate
   */
  y(qubit: number): this {
    this.validateQubit(qubit);
    this._gates.push({ type: 'y', qubit });
    return this;
  }

  /**
   * Pauli-Z gate
   */
  z(qubit: number): this {
    this.validateQubit(qubit);
    this._gates.push({ type: 'z', qubit });
    return this;
  }

  /**
   * S gate (sqrt Z)
   */
  s(qubit: number): this {
    this.validateQubit(qubit);
    this._gates.push({ type: 's', qubit });
    return this;
  }

  /**
   * S-dagger gate
   */
  sdg(qubit: number): this {
    this.validateQubit(qubit);
    this._gates.push({ type: 'sdg', qubit });
    return this;
  }

  /**
   * T gate (sqrt S)
   */
  t(qubit: number): this {
    this.validateQubit(qubit);
    this._gates.push({ type: 't', qubit });
    return this;
  }

  /**
   * T-dagger gate
   */
  tdg(qubit: number): this {
    this.validateQubit(qubit);
    this._gates.push({ type: 'tdg', qubit });
    return this;
  }

  // =========================================================================
  // Parameterized Single-Qubit Gates
  // =========================================================================

  /**
   * Rotation around X-axis
   */
  rx(qubit: number, angle: number): this {
    this.validateQubit(qubit);
    this._gates.push({ type: 'rx', qubit, params: [angle] });
    return this;
  }

  /**
   * Rotation around Y-axis
   */
  ry(qubit: number, angle: number): this {
    this.validateQubit(qubit);
    this._gates.push({ type: 'ry', qubit, params: [angle] });
    return this;
  }

  /**
   * Rotation around Z-axis
   */
  rz(qubit: number, angle: number): this {
    this.validateQubit(qubit);
    this._gates.push({ type: 'rz', qubit, params: [angle] });
    return this;
  }

  /**
   * Phase gate P(angle)
   */
  phase(qubit: number, angle: number): this {
    this.validateQubit(qubit);
    this._gates.push({ type: 'phase', qubit, params: [angle] });
    return this;
  }

  /**
   * General single-qubit unitary U3(theta, phi, lambda)
   */
  u3(qubit: number, theta: number, phi: number, lambda: number): this {
    this.validateQubit(qubit);
    this._gates.push({ type: 'u3', qubit, params: [theta, phi, lambda] });
    return this;
  }

  // =========================================================================
  // Two-Qubit Gates
  // =========================================================================

  /**
   * Controlled-NOT gate
   */
  cnot(control: number, target: number): this {
    this.validateQubit(control);
    this.validateQubit(target);
    this.validateDifferent(control, target);
    this._gates.push({ type: 'cnot', control, target });
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
    this.validateQubit(control);
    this.validateQubit(target);
    this.validateDifferent(control, target);
    this._gates.push({ type: 'cz', control, target });
    return this;
  }

  /**
   * Controlled-Y gate
   */
  cy(control: number, target: number): this {
    this.validateQubit(control);
    this.validateQubit(target);
    this.validateDifferent(control, target);
    this._gates.push({ type: 'cy', control, target });
    return this;
  }

  /**
   * SWAP gate
   */
  swap(qubit1: number, qubit2: number): this {
    this.validateQubit(qubit1);
    this.validateQubit(qubit2);
    this._gates.push({ type: 'swap', control: qubit1, target: qubit2 });
    return this;
  }

  /**
   * Controlled phase gate
   */
  cphase(control: number, target: number, angle: number): this {
    this.validateQubit(control);
    this.validateQubit(target);
    this.validateDifferent(control, target);
    this._gates.push({ type: 'cphase', control, target, param: angle });
    return this;
  }

  /**
   * Controlled Rx gate
   */
  crx(control: number, target: number, angle: number): this {
    this.validateQubit(control);
    this.validateQubit(target);
    this.validateDifferent(control, target);
    this._gates.push({ type: 'crx', control, target, param: angle });
    return this;
  }

  /**
   * Controlled Ry gate
   */
  cry(control: number, target: number, angle: number): this {
    this.validateQubit(control);
    this.validateQubit(target);
    this.validateDifferent(control, target);
    this._gates.push({ type: 'cry', control, target, param: angle });
    return this;
  }

  /**
   * Controlled Rz gate
   */
  crz(control: number, target: number, angle: number): this {
    this.validateQubit(control);
    this.validateQubit(target);
    this.validateDifferent(control, target);
    this._gates.push({ type: 'crz', control, target, param: angle });
    return this;
  }

  // =========================================================================
  // Three-Qubit Gates
  // =========================================================================

  /**
   * Toffoli gate (CCNOT)
   */
  toffoli(control1: number, control2: number, target: number): this {
    this.validateQubit(control1);
    this.validateQubit(control2);
    this.validateQubit(target);
    this.validateAllDifferent([control1, control2, target]);
    this._gates.push({ type: 'toffoli', qubits: [control1, control2, target] });
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
   */
  fredkin(control: number, target1: number, target2: number): this {
    this.validateQubit(control);
    this.validateQubit(target1);
    this.validateQubit(target2);
    this.validateAllDifferent([control, target1, target2]);
    this._gates.push({ type: 'fredkin', qubits: [control, target1, target2] });
    return this;
  }

  /**
   * Alias for Fredkin
   */
  cswap(control: number, target1: number, target2: number): this {
    return this.fredkin(control, target1, target2);
  }

  // =========================================================================
  // Multi-Qubit Gates
  // =========================================================================

  /**
   * Quantum Fourier Transform
   */
  qft(qubits: number[]): this {
    for (const q of qubits) {
      this.validateQubit(q);
    }
    this._gates.push({ type: 'qft', qubits: [...qubits] });
    return this;
  }

  /**
   * Inverse Quantum Fourier Transform
   */
  iqft(qubits: number[]): this {
    for (const q of qubits) {
      this.validateQubit(q);
    }
    this._gates.push({ type: 'iqft', qubits: [...qubits] });
    return this;
  }

  // =========================================================================
  // Special Operations
  // =========================================================================

  /**
   * Add a barrier (for visualization, no effect on state)
   */
  barrier(qubits?: number[]): this {
    const qs = qubits ?? Array.from({ length: this._numQubits }, (_, i) => i);
    for (const q of qs) {
      this.validateQubit(q);
    }
    this._gates.push({ type: 'barrier', qubits: qs });
    return this;
  }

  /**
   * Measure a qubit
   */
  measure(qubit: number, classicalBit?: number): this {
    this.validateQubit(qubit);
    this._gates.push({ type: 'measure', qubit, classicalBit });
    return this;
  }

  /**
   * Measure all qubits
   */
  measureAll(): this {
    for (let i = 0; i < this._numQubits; i++) {
      this._gates.push({ type: 'measure', qubit: i, classicalBit: i });
    }
    return this;
  }

  // =========================================================================
  // Circuit Composition
  // =========================================================================

  /**
   * Append another circuit to this one
   * @param other Circuit to append
   * @param offset Qubit offset for the appended circuit
   */
  append(other: Circuit, offset: number = 0): this {
    if (offset + other.numQubits > this._numQubits) {
      throw new Error(
        `Cannot append ${other.numQubits}-qubit circuit at offset ${offset} to ${this._numQubits}-qubit circuit`
      );
    }

    for (const gate of other._gates) {
      this._gates.push(this.offsetGate(gate, offset));
    }

    return this;
  }

  /**
   * Repeat the circuit n times
   */
  repeat(n: number): this {
    if (n < 1) {
      throw new Error('Repeat count must be at least 1');
    }

    const originalGates = [...this._gates];
    for (let i = 1; i < n; i++) {
      this._gates.push(...originalGates.map((g) => ({ ...g })));
    }

    return this;
  }

  /**
   * Create inverse (dagger) of the circuit
   */
  inverse(): Circuit {
    const inverse = new Circuit(this._numQubits, `${this._name ?? 'circuit'}†`);

    // Reverse gates and apply inverse to each
    for (let i = this._gates.length - 1; i >= 0; i--) {
      const gate = this._gates[i];
      inverse._gates.push(this.invertGate(gate));
    }

    return inverse;
  }

  // =========================================================================
  // Application
  // =========================================================================

  /**
   * Apply this circuit to a quantum state
   */
  apply(state: QuantumState): QuantumState {
    if (state.numQubits !== this._numQubits) {
      throw new Error(
        `Circuit has ${this._numQubits} qubits but state has ${state.numQubits}`
      );
    }

    for (const gate of this._gates) {
      this.applyGate(state, gate);
    }

    return state;
  }

  // =========================================================================
  // Statistics
  // =========================================================================

  /**
   * Get circuit statistics
   */
  getStats(): CircuitStats {
    const gateBreakdown: Record<string, number> = {};
    let singleQubitGates = 0;
    let twoQubitGates = 0;
    let threeQubitGates = 0;
    let measurements = 0;

    for (const gate of this._gates) {
      gateBreakdown[gate.type] = (gateBreakdown[gate.type] || 0) + 1;

      if (gate.type === 'measure') {
        measurements++;
      } else if (gate.type === 'barrier') {
        // Don't count barriers
      } else if (this.isSingleQubitGate(gate)) {
        singleQubitGates++;
      } else if (this.isTwoQubitGate(gate)) {
        twoQubitGates++;
      } else if (this.isThreeQubitGate(gate)) {
        threeQubitGates++;
      }
    }

    return {
      numQubits: this._numQubits,
      depth: this.calculateDepth(),
      totalGates:
        singleQubitGates + twoQubitGates + threeQubitGates + measurements,
      singleQubitGates,
      twoQubitGates,
      threeQubitGates,
      measurements,
      gateBreakdown,
    };
  }

  /**
   * Calculate circuit depth
   */
  private calculateDepth(): number {
    const qubitDepths = new Array(this._numQubits).fill(0);

    for (const gate of this._gates) {
      if (gate.type === 'barrier') continue;

      const qubits = this.getGateQubits(gate);
      const maxDepth = Math.max(...qubits.map((q) => qubitDepths[q]));

      for (const q of qubits) {
        qubitDepths[q] = maxDepth + 1;
      }
    }

    return Math.max(...qubitDepths);
  }

  // =========================================================================
  // Serialization
  // =========================================================================

  /**
   * Convert circuit to JSON
   */
  toJSON(): { numQubits: number; name?: string; gates: Gate[] } {
    return {
      numQubits: this._numQubits,
      name: this._name,
      gates: this._gates,
    };
  }

  /**
   * Create circuit from JSON
   */
  static fromJSON(json: {
    numQubits: number;
    name?: string;
    gates: Gate[];
  }): Circuit {
    const circuit = new Circuit(json.numQubits, json.name);
    circuit._gates = json.gates;
    return circuit;
  }

  /**
   * Convert to OpenQASM 2.0 string
   */
  toQASM(): string {
    const lines: string[] = [
      'OPENQASM 2.0;',
      'include "qelib1.inc";',
      `qreg q[${this._numQubits}];`,
      `creg c[${this._numQubits}];`,
      '',
    ];

    for (const gate of this._gates) {
      lines.push(this.gateToQASM(gate));
    }

    return lines.join('\n');
  }

  // =========================================================================
  // Private Helpers
  // =========================================================================

  private validateQubit(qubit: number): void {
    if (qubit < 0 || qubit >= this._numQubits) {
      throw new Error(
        `Qubit ${qubit} out of range [0, ${this._numQubits - 1}]`
      );
    }
  }

  private validateDifferent(q1: number, q2: number): void {
    if (q1 === q2) {
      throw new Error('Control and target must be different qubits');
    }
  }

  private validateAllDifferent(qubits: number[]): void {
    const unique = new Set(qubits);
    if (unique.size !== qubits.length) {
      throw new Error('All qubits must be different');
    }
  }

  private isSingleQubitGate(
    gate: Gate
  ): gate is SingleQubitGate | ParameterizedGate {
    return (
      'qubit' in gate &&
      !('control' in gate) &&
      gate.type !== 'measure'
    );
  }

  private isTwoQubitGate(gate: Gate): gate is TwoQubitGate {
    return 'control' in gate && 'target' in gate;
  }

  private isThreeQubitGate(gate: Gate): gate is ThreeQubitGate {
    return (
      'qubits' in gate &&
      Array.isArray(gate.qubits) &&
      gate.qubits.length === 3 &&
      (gate.type === 'toffoli' ||
        gate.type === 'ccx' ||
        gate.type === 'fredkin' ||
        gate.type === 'cswap')
    );
  }

  private getGateQubits(gate: Gate): number[] {
    if ('qubit' in gate) {
      return [gate.qubit];
    }
    if ('control' in gate && 'target' in gate) {
      return [gate.control, gate.target];
    }
    if ('qubits' in gate) {
      return gate.qubits;
    }
    return [];
  }

  private offsetGate(gate: Gate, offset: number): Gate {
    if ('qubit' in gate) {
      return { ...gate, qubit: gate.qubit + offset };
    }
    if ('control' in gate && 'target' in gate) {
      return {
        ...gate,
        control: gate.control + offset,
        target: gate.target + offset,
      };
    }
    if ('qubits' in gate) {
      const offsetQubits = gate.qubits.map((q) => q + offset);
      return { ...gate, qubits: offsetQubits } as Gate;
    }
    return gate;
  }

  private invertGate(gate: Gate): Gate {
    // Most gates are self-inverse or need angle negation
    switch (gate.type) {
      case 's':
        return { ...gate, type: 'sdg' };
      case 'sdg':
        return { ...gate, type: 's' };
      case 't':
        return { ...gate, type: 'tdg' };
      case 'tdg':
        return { ...gate, type: 't' };
      case 'rx':
      case 'ry':
      case 'rz':
      case 'phase':
        return {
          ...gate,
          params: [(gate as ParameterizedGate).params[0] * -1],
        } as ParameterizedGate;
      case 'u3': {
        const [theta, phi, lambda] = (gate as ParameterizedGate).params;
        return {
          ...gate,
          params: [-theta, -lambda, -phi],
        } as ParameterizedGate;
      }
      case 'cphase':
      case 'crx':
      case 'cry':
      case 'crz':
        return {
          ...gate,
          param: -(gate as TwoQubitGate).param!,
        } as TwoQubitGate;
      default:
        // Self-inverse gates (H, X, Y, Z, CNOT, CZ, SWAP, etc.)
        return { ...gate };
    }
  }

  private applyGate(state: QuantumState, gate: Gate): void {
    switch (gate.type) {
      // Single-qubit gates
      case 'h':
        state.h((gate as SingleQubitGate).qubit);
        break;
      case 'x':
        state.x((gate as SingleQubitGate).qubit);
        break;
      case 'y':
        state.y((gate as SingleQubitGate).qubit);
        break;
      case 'z':
        state.z((gate as SingleQubitGate).qubit);
        break;
      case 's':
        state.s((gate as SingleQubitGate).qubit);
        break;
      case 'sdg':
        state.sdg((gate as SingleQubitGate).qubit);
        break;
      case 't':
        state.t((gate as SingleQubitGate).qubit);
        break;
      case 'tdg':
        state.tdg((gate as SingleQubitGate).qubit);
        break;

      // Parameterized single-qubit gates
      case 'rx':
        state.rx(
          (gate as ParameterizedGate).qubit,
          (gate as ParameterizedGate).params[0]
        );
        break;
      case 'ry':
        state.ry(
          (gate as ParameterizedGate).qubit,
          (gate as ParameterizedGate).params[0]
        );
        break;
      case 'rz':
        state.rz(
          (gate as ParameterizedGate).qubit,
          (gate as ParameterizedGate).params[0]
        );
        break;
      case 'phase':
        state.phase(
          (gate as ParameterizedGate).qubit,
          (gate as ParameterizedGate).params[0]
        );
        break;
      case 'u3': {
        const p = (gate as ParameterizedGate).params;
        state.u3((gate as ParameterizedGate).qubit, p[0], p[1], p[2]);
        break;
      }

      // Two-qubit gates
      case 'cnot':
      case 'cx':
        state.cnot(
          (gate as TwoQubitGate).control,
          (gate as TwoQubitGate).target
        );
        break;
      case 'cz':
        state.cz((gate as TwoQubitGate).control, (gate as TwoQubitGate).target);
        break;
      case 'cy':
        state.cy((gate as TwoQubitGate).control, (gate as TwoQubitGate).target);
        break;
      case 'swap':
        state.swap(
          (gate as TwoQubitGate).control,
          (gate as TwoQubitGate).target
        );
        break;
      case 'cphase':
        state.cphase(
          (gate as TwoQubitGate).control,
          (gate as TwoQubitGate).target,
          (gate as TwoQubitGate).param!
        );
        break;
      case 'crx':
        state.crx(
          (gate as TwoQubitGate).control,
          (gate as TwoQubitGate).target,
          (gate as TwoQubitGate).param!
        );
        break;
      case 'cry':
        state.cry(
          (gate as TwoQubitGate).control,
          (gate as TwoQubitGate).target,
          (gate as TwoQubitGate).param!
        );
        break;
      case 'crz':
        state.crz(
          (gate as TwoQubitGate).control,
          (gate as TwoQubitGate).target,
          (gate as TwoQubitGate).param!
        );
        break;

      // Three-qubit gates
      case 'toffoli':
      case 'ccx': {
        const [c1, c2, t] = (gate as ThreeQubitGate).qubits;
        state.toffoli(c1, c2, t);
        break;
      }
      case 'fredkin':
      case 'cswap': {
        const [c, t1, t2] = (gate as ThreeQubitGate).qubits;
        state.fredkin(c, t1, t2);
        break;
      }

      // Multi-qubit gates
      case 'qft':
        state.qft((gate as MultiQubitGate).qubits);
        break;
      case 'iqft':
        state.iqft((gate as MultiQubitGate).qubits);
        break;

      // Measurement
      case 'measure':
        state.measure((gate as MeasureGate).qubit);
        break;

      // Barrier - no-op
      case 'barrier':
        break;
    }
  }

  private gateToQASM(gate: Gate): string {
    switch (gate.type) {
      case 'h':
        return `h q[${(gate as SingleQubitGate).qubit}];`;
      case 'x':
        return `x q[${(gate as SingleQubitGate).qubit}];`;
      case 'y':
        return `y q[${(gate as SingleQubitGate).qubit}];`;
      case 'z':
        return `z q[${(gate as SingleQubitGate).qubit}];`;
      case 's':
        return `s q[${(gate as SingleQubitGate).qubit}];`;
      case 'sdg':
        return `sdg q[${(gate as SingleQubitGate).qubit}];`;
      case 't':
        return `t q[${(gate as SingleQubitGate).qubit}];`;
      case 'tdg':
        return `tdg q[${(gate as SingleQubitGate).qubit}];`;
      case 'rx':
        return `rx(${(gate as ParameterizedGate).params[0]}) q[${(gate as ParameterizedGate).qubit}];`;
      case 'ry':
        return `ry(${(gate as ParameterizedGate).params[0]}) q[${(gate as ParameterizedGate).qubit}];`;
      case 'rz':
        return `rz(${(gate as ParameterizedGate).params[0]}) q[${(gate as ParameterizedGate).qubit}];`;
      case 'phase':
        return `u1(${(gate as ParameterizedGate).params[0]}) q[${(gate as ParameterizedGate).qubit}];`;
      case 'u3': {
        const p = (gate as ParameterizedGate).params;
        return `u3(${p[0]},${p[1]},${p[2]}) q[${(gate as ParameterizedGate).qubit}];`;
      }
      case 'cnot':
      case 'cx':
        return `cx q[${(gate as TwoQubitGate).control}],q[${(gate as TwoQubitGate).target}];`;
      case 'cz':
        return `cz q[${(gate as TwoQubitGate).control}],q[${(gate as TwoQubitGate).target}];`;
      case 'cy':
        return `cy q[${(gate as TwoQubitGate).control}],q[${(gate as TwoQubitGate).target}];`;
      case 'swap':
        return `swap q[${(gate as TwoQubitGate).control}],q[${(gate as TwoQubitGate).target}];`;
      case 'cphase':
        return `cu1(${(gate as TwoQubitGate).param}) q[${(gate as TwoQubitGate).control}],q[${(gate as TwoQubitGate).target}];`;
      case 'crx':
        return `crx(${(gate as TwoQubitGate).param}) q[${(gate as TwoQubitGate).control}],q[${(gate as TwoQubitGate).target}];`;
      case 'cry':
        return `cry(${(gate as TwoQubitGate).param}) q[${(gate as TwoQubitGate).control}],q[${(gate as TwoQubitGate).target}];`;
      case 'crz':
        return `crz(${(gate as TwoQubitGate).param}) q[${(gate as TwoQubitGate).control}],q[${(gate as TwoQubitGate).target}];`;
      case 'toffoli':
      case 'ccx': {
        const [c1, c2, t] = (gate as ThreeQubitGate).qubits;
        return `ccx q[${c1}],q[${c2}],q[${t}];`;
      }
      case 'fredkin':
      case 'cswap': {
        const [c, t1, t2] = (gate as ThreeQubitGate).qubits;
        return `cswap q[${c}],q[${t1}],q[${t2}];`;
      }
      case 'barrier': {
        const qs = (gate as BarrierGate).qubits.map((q) => `q[${q}]`).join(',');
        return `barrier ${qs};`;
      }
      case 'measure': {
        const m = gate as MeasureGate;
        const cbit = m.classicalBit ?? m.qubit;
        return `measure q[${m.qubit}] -> c[${cbit}];`;
      }
      default:
        return `// Unknown gate: ${gate.type}`;
    }
  }
}

// ============================================================================
// Common Circuits
// ============================================================================

/**
 * Create a Bell state circuit (|00⟩ + |11⟩) / sqrt(2)
 */
export function bellCircuit(): Circuit {
  return new Circuit(2, 'Bell').h(0).cnot(0, 1);
}

/**
 * Create a GHZ state circuit (|000...0⟩ + |111...1⟩) / sqrt(2)
 */
export function ghzCircuit(numQubits: number): Circuit {
  const circuit = new Circuit(numQubits, 'GHZ').h(0);
  for (let i = 1; i < numQubits; i++) {
    circuit.cnot(i - 1, i);
  }
  return circuit;
}

/**
 * Create a Quantum Fourier Transform circuit
 */
export function qftCircuit(numQubits: number): Circuit {
  return new Circuit(numQubits, 'QFT').qft(
    Array.from({ length: numQubits }, (_, i) => i)
  );
}

/**
 * Create a uniform superposition circuit (Hadamard on all qubits)
 */
export function superpositionCircuit(numQubits: number): Circuit {
  const circuit = new Circuit(numQubits, 'Superposition');
  for (let i = 0; i < numQubits; i++) {
    circuit.h(i);
  }
  return circuit;
}
