/**
 * Tests for Circuit class
 */

import { describe, it, expect } from 'vitest';
import {
  Circuit,
  bellCircuit,
  ghzCircuit,
  qftCircuit,
  superpositionCircuit,
  type Gate,
  type SingleQubitGate,
  type ParameterizedGate,
  type TwoQubitGate,
  type ThreeQubitGate,
  type MultiQubitGate,
} from '../circuit';

describe('Circuit Creation', () => {
  it('creates circuit with specified qubits', () => {
    const circuit = new Circuit(4);
    expect(circuit.numQubits).toBe(4);
    expect(circuit.length).toBe(0);
  });

  it('creates circuit with name', () => {
    const circuit = new Circuit(2, 'MyCircuit');
    expect(circuit.name).toBe('MyCircuit');
  });

  it('throws for invalid qubit count', () => {
    expect(() => new Circuit(0)).toThrow('numQubits must be between 1 and 30');
    expect(() => new Circuit(31)).toThrow('numQubits must be between 1 and 30');
  });

  it('allows maximum 30 qubits', () => {
    const circuit = new Circuit(30);
    expect(circuit.numQubits).toBe(30);
  });
});

describe('Single-Qubit Gates', () => {
  it('adds Hadamard gate', () => {
    const circuit = new Circuit(2).h(0);
    expect(circuit.length).toBe(1);
    expect(circuit.gates[0]).toEqual({ type: 'h', qubit: 0 });
  });

  it('adds Pauli gates', () => {
    const circuit = new Circuit(3).x(0).y(1).z(2);
    expect(circuit.length).toBe(3);
    expect(circuit.gates[0]).toEqual({ type: 'x', qubit: 0 });
    expect(circuit.gates[1]).toEqual({ type: 'y', qubit: 1 });
    expect(circuit.gates[2]).toEqual({ type: 'z', qubit: 2 });
  });

  it('adds S and T gates', () => {
    const circuit = new Circuit(2).s(0).t(1).sdg(0).tdg(1);
    expect(circuit.gates.map((g) => g.type)).toEqual(['s', 't', 'sdg', 'tdg']);
  });

  it('validates qubit index', () => {
    const circuit = new Circuit(2);
    expect(() => circuit.h(-1)).toThrow('Qubit -1 out of range');
    expect(() => circuit.h(2)).toThrow('Qubit 2 out of range');
  });

  it('supports fluent API chaining', () => {
    const circuit = new Circuit(2).h(0).x(1).y(0).z(1);
    expect(circuit.length).toBe(4);
  });
});

describe('Parameterized Single-Qubit Gates', () => {
  it('adds rotation gates', () => {
    const circuit = new Circuit(1).rx(0, Math.PI).ry(0, Math.PI / 2).rz(0, Math.PI / 4);
    expect(circuit.length).toBe(3);
    expect((circuit.gates[0] as ParameterizedGate).params[0]).toBe(Math.PI);
    expect((circuit.gates[1] as ParameterizedGate).params[0]).toBe(Math.PI / 2);
    expect((circuit.gates[2] as ParameterizedGate).params[0]).toBe(Math.PI / 4);
  });

  it('adds phase gate', () => {
    const circuit = new Circuit(1).phase(0, Math.PI);
    const gate = circuit.gates[0] as ParameterizedGate;
    expect(gate.type).toBe('phase');
    expect(gate.params[0]).toBe(Math.PI);
  });

  it('adds U3 gate with three parameters', () => {
    const circuit = new Circuit(1).u3(0, Math.PI, Math.PI / 2, Math.PI / 4);
    const gate = circuit.gates[0] as ParameterizedGate;
    expect(gate.type).toBe('u3');
    expect(gate.params).toEqual([Math.PI, Math.PI / 2, Math.PI / 4]);
  });
});

describe('Two-Qubit Gates', () => {
  it('adds CNOT gate', () => {
    const circuit = new Circuit(2).cnot(0, 1);
    const gate = circuit.gates[0] as TwoQubitGate;
    expect(gate.type).toBe('cnot');
    expect(gate.control).toBe(0);
    expect(gate.target).toBe(1);
  });

  it('adds CX alias for CNOT', () => {
    const circuit = new Circuit(2).cx(0, 1);
    expect(circuit.gates[0].type).toBe('cnot');
  });

  it('adds CZ and CY gates', () => {
    const circuit = new Circuit(2).cz(0, 1).cy(1, 0);
    expect(circuit.gates[0].type).toBe('cz');
    expect(circuit.gates[1].type).toBe('cy');
  });

  it('adds SWAP gate', () => {
    const circuit = new Circuit(2).swap(0, 1);
    const gate = circuit.gates[0] as TwoQubitGate;
    expect(gate.type).toBe('swap');
  });

  it('adds controlled rotation gates', () => {
    const circuit = new Circuit(2)
      .cphase(0, 1, Math.PI)
      .crx(0, 1, Math.PI / 2)
      .cry(0, 1, Math.PI / 4)
      .crz(0, 1, Math.PI / 8);

    expect(circuit.gates.map((g) => g.type)).toEqual(['cphase', 'crx', 'cry', 'crz']);
    expect((circuit.gates[0] as TwoQubitGate).param).toBe(Math.PI);
  });

  it('validates different qubits for CNOT', () => {
    const circuit = new Circuit(2);
    expect(() => circuit.cnot(0, 0)).toThrow('Control and target must be different');
  });
});

describe('Three-Qubit Gates', () => {
  it('adds Toffoli gate', () => {
    const circuit = new Circuit(3).toffoli(0, 1, 2);
    const gate = circuit.gates[0] as ThreeQubitGate;
    expect(gate.type).toBe('toffoli');
    expect(gate.qubits).toEqual([0, 1, 2]);
  });

  it('adds CCX alias for Toffoli', () => {
    const circuit = new Circuit(3).ccx(0, 1, 2);
    expect(circuit.gates[0].type).toBe('toffoli');
  });

  it('adds Fredkin gate', () => {
    const circuit = new Circuit(3).fredkin(0, 1, 2);
    const gate = circuit.gates[0] as ThreeQubitGate;
    expect(gate.type).toBe('fredkin');
    expect(gate.qubits).toEqual([0, 1, 2]);
  });

  it('adds CSWAP alias for Fredkin', () => {
    const circuit = new Circuit(3).cswap(0, 1, 2);
    expect(circuit.gates[0].type).toBe('fredkin');
  });

  it('validates all qubits are different', () => {
    const circuit = new Circuit(3);
    expect(() => circuit.toffoli(0, 0, 1)).toThrow('All qubits must be different');
    expect(() => circuit.toffoli(0, 1, 0)).toThrow('All qubits must be different');
    expect(() => circuit.fredkin(1, 1, 2)).toThrow('All qubits must be different');
  });
});

describe('Multi-Qubit Gates', () => {
  it('adds QFT gate', () => {
    const circuit = new Circuit(4).qft([0, 1, 2, 3]);
    const gate = circuit.gates[0] as MultiQubitGate;
    expect(gate.type).toBe('qft');
    expect(gate.qubits).toEqual([0, 1, 2, 3]);
  });

  it('adds inverse QFT gate', () => {
    const circuit = new Circuit(3).iqft([0, 1, 2]);
    const gate = circuit.gates[0] as MultiQubitGate;
    expect(gate.type).toBe('iqft');
    expect(gate.qubits).toEqual([0, 1, 2]);
  });

  it('validates qubit indices in QFT', () => {
    const circuit = new Circuit(3);
    expect(() => circuit.qft([0, 1, 3])).toThrow('Qubit 3 out of range');
  });
});

describe('Special Operations', () => {
  it('adds barrier to all qubits', () => {
    const circuit = new Circuit(3).barrier();
    expect(circuit.gates[0]).toEqual({ type: 'barrier', qubits: [0, 1, 2] });
  });

  it('adds barrier to specific qubits', () => {
    const circuit = new Circuit(4).barrier([1, 3]);
    expect(circuit.gates[0]).toEqual({ type: 'barrier', qubits: [1, 3] });
  });

  it('adds measurement', () => {
    const circuit = new Circuit(2).measure(0).measure(1, 5);
    expect(circuit.gates[0]).toEqual({ type: 'measure', qubit: 0, classicalBit: undefined });
    expect(circuit.gates[1]).toEqual({ type: 'measure', qubit: 1, classicalBit: 5 });
  });

  it('adds measureAll', () => {
    const circuit = new Circuit(3).measureAll();
    expect(circuit.length).toBe(3);
    expect(circuit.gates.every((g) => g.type === 'measure')).toBe(true);
  });
});

describe('Circuit Composition', () => {
  it('appends another circuit', () => {
    const base = new Circuit(4).h(0);
    const toAppend = new Circuit(2).cnot(0, 1);
    base.append(toAppend, 2); // Offset by 2

    expect(base.length).toBe(2);
    expect((base.gates[1] as TwoQubitGate).control).toBe(2);
    expect((base.gates[1] as TwoQubitGate).target).toBe(3);
  });

  it('throws when appended circuit doesn\'t fit', () => {
    const base = new Circuit(3);
    const toAppend = new Circuit(2);
    expect(() => base.append(toAppend, 2)).toThrow('Cannot append');
  });

  it('repeats circuit', () => {
    const circuit = new Circuit(2).h(0).cnot(0, 1);
    circuit.repeat(3);
    expect(circuit.length).toBe(6);
  });

  it('creates inverse circuit', () => {
    const original = new Circuit(2).s(0).t(1).rx(0, Math.PI);
    const inverse = original.inverse();

    expect(inverse.gates[0].type).toBe('rx'); // Reversed order
    expect(inverse.gates[1].type).toBe('tdg'); // T inverted to Tdg
    expect(inverse.gates[2].type).toBe('sdg'); // S inverted to Sdg
    expect((inverse.gates[0] as ParameterizedGate).params[0]).toBe(-Math.PI); // Angle negated
  });

  it('inverse of self-inverse gates remains same type', () => {
    const original = new Circuit(2).h(0).x(1).cnot(0, 1);
    const inverse = original.inverse();

    // H, X, CNOT are self-inverse
    expect(inverse.gates.map((g) => g.type)).toEqual(['cnot', 'x', 'h']);
  });
});

describe('Circuit Statistics', () => {
  it('calculates gate counts', () => {
    const circuit = new Circuit(3)
      .h(0).h(1).h(2) // 3 single-qubit
      .cnot(0, 1).cz(1, 2) // 2 two-qubit
      .toffoli(0, 1, 2) // 1 three-qubit
      .measure(0);

    const stats = circuit.getStats();
    expect(stats.numQubits).toBe(3);
    expect(stats.singleQubitGates).toBe(3);
    expect(stats.twoQubitGates).toBe(2);
    expect(stats.threeQubitGates).toBe(1);
    expect(stats.measurements).toBe(1);
    expect(stats.totalGates).toBe(7);
  });

  it('calculates gate breakdown', () => {
    const circuit = new Circuit(2).h(0).h(1).cnot(0, 1);
    const stats = circuit.getStats();
    expect(stats.gateBreakdown).toEqual({ h: 2, cnot: 1 });
  });

  it('calculates circuit depth', () => {
    // Parallel gates on different qubits have depth 1
    const parallel = new Circuit(3).h(0).h(1).h(2);
    expect(parallel.getStats().depth).toBe(1);

    // Sequential gates on same qubit increase depth
    const sequential = new Circuit(1).h(0).x(0).y(0);
    expect(sequential.getStats().depth).toBe(3);
  });
});

describe('Serialization', () => {
  it('converts to JSON and back', () => {
    const original = new Circuit(2, 'Test').h(0).cnot(0, 1).measure(0);
    const json = original.toJSON();
    const restored = Circuit.fromJSON(json);

    expect(restored.numQubits).toBe(original.numQubits);
    expect(restored.name).toBe(original.name);
    expect(restored.gates).toEqual(original.gates);
  });

  it('converts to OpenQASM', () => {
    const circuit = new Circuit(2).h(0).cnot(0, 1).measure(0);
    const qasm = circuit.toQASM();

    expect(qasm).toContain('OPENQASM 2.0;');
    expect(qasm).toContain('qreg q[2];');
    expect(qasm).toContain('creg c[2];');
    expect(qasm).toContain('h q[0];');
    expect(qasm).toContain('cx q[0],q[1];');
    expect(qasm).toContain('measure q[0] -> c[0];');
  });

  it('converts parameterized gates to QASM', () => {
    const circuit = new Circuit(1).rx(0, Math.PI).u3(0, Math.PI, Math.PI / 2, Math.PI / 4);
    const qasm = circuit.toQASM();

    expect(qasm).toContain(`rx(${Math.PI}) q[0];`);
    expect(qasm).toContain(`u3(${Math.PI},${Math.PI / 2},${Math.PI / 4}) q[0];`);
  });
});

describe('Common Circuits', () => {
  it('creates Bell circuit', () => {
    const bell = bellCircuit();
    expect(bell.numQubits).toBe(2);
    expect(bell.name).toBe('Bell');
    expect(bell.length).toBe(2);
    expect(bell.gates[0].type).toBe('h');
    expect(bell.gates[1].type).toBe('cnot');
  });

  it('creates GHZ circuit', () => {
    const ghz = ghzCircuit(4);
    expect(ghz.numQubits).toBe(4);
    expect(ghz.name).toBe('GHZ');
    expect(ghz.length).toBe(4); // 1 H + 3 CNOTs
    expect(ghz.gates[0].type).toBe('h');
    expect(ghz.gates.slice(1).every((g) => g.type === 'cnot')).toBe(true);
  });

  it('creates QFT circuit', () => {
    const qft = qftCircuit(3);
    expect(qft.numQubits).toBe(3);
    expect(qft.name).toBe('QFT');
    expect(qft.gates[0].type).toBe('qft');
  });

  it('creates superposition circuit', () => {
    const sup = superpositionCircuit(4);
    expect(sup.numQubits).toBe(4);
    expect(sup.name).toBe('Superposition');
    expect(sup.length).toBe(4);
    expect(sup.gates.every((g) => g.type === 'h')).toBe(true);
  });
});

describe('Edge Cases', () => {
  it('handles single qubit circuit', () => {
    const circuit = new Circuit(1).h(0).x(0).measure(0);
    expect(circuit.getStats().numQubits).toBe(1);
    expect(circuit.getStats().depth).toBe(2); // measure doesn't add depth to gate count
  });

  it('handles empty circuit', () => {
    const circuit = new Circuit(5);
    expect(circuit.length).toBe(0);
    expect(circuit.getStats().totalGates).toBe(0);
    expect(circuit.getStats().depth).toBe(0);
  });

  it('handles repeated append', () => {
    const circuit = new Circuit(4);
    const sub = new Circuit(2).h(0).h(1);

    circuit.append(sub, 0);
    circuit.append(sub, 2);

    expect(circuit.length).toBe(4);
    expect((circuit.gates[2] as SingleQubitGate).qubit).toBe(2);
    expect((circuit.gates[3] as SingleQubitGate).qubit).toBe(3);
  });
});
