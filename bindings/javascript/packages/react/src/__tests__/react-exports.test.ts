import { describe, expect, it } from 'vitest';
import {
  AmplitudeBars,
  BlochSphere,
  Circuit,
  CircuitDiagram,
  QuantumState,
  VERSION,
  useCircuit,
  useQuantumState,
} from '../index';

describe('@moonlab/quantum-react exports', () => {
  it('exposes components, hooks, core types, and version', () => {
    expect(typeof BlochSphere).toBe('function');
    expect(typeof AmplitudeBars).toBe('function');
    expect(typeof CircuitDiagram).toBe('function');
    expect(typeof useQuantumState).toBe('function');
    expect(typeof useCircuit).toBe('function');
    expect(typeof QuantumState.create).toBe('function');
    expect(typeof Circuit).toBe('function');
    expect(VERSION).toMatch(/^\d+\.\d+\.\d+$/);
  });
});
