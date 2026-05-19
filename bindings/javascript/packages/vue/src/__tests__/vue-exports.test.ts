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

describe('@moonlab/quantum-vue exports', () => {
  it('exposes components, composables, core types, and version', () => {
    expect(BlochSphere.name).toBe('BlochSphere');
    expect(AmplitudeBars.name).toBe('AmplitudeBars');
    expect(CircuitDiagram.name).toBe('CircuitDiagram');
    expect(typeof useQuantumState).toBe('function');
    expect(typeof useCircuit).toBe('function');
    expect(typeof QuantumState.create).toBe('function');
    expect(typeof Circuit).toBe('function');
    expect(VERSION).toMatch(/^\d+\.\d+\.\d+$/);
  });
});
