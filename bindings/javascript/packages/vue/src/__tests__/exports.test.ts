import { describe, expect, it } from 'vitest';
import {
  VERSION,
  AmplitudeBars,
  BlochSphere,
  CircuitDiagram,
  useCircuit,
  useQuantumState,
} from '../index';

describe('@moonlab/quantum-vue public surface', () => {
  it('reports the release version and exports composables', () => {
    expect(VERSION).toBe('1.1.0');
    expect(useQuantumState).toBeTypeOf('function');
    expect(useCircuit).toBeTypeOf('function');
  });

  it('defines stable component names and runtime props', () => {
    const components = [BlochSphere, AmplitudeBars, CircuitDiagram];
    expect(components.map((component) => component.name)).toEqual([
      'BlochSphere',
      'AmplitudeBars',
      'CircuitDiagram',
    ]);
    for (const component of components) {
      expect(component.props).toHaveProperty('width');
      expect(component.props).toHaveProperty('height');
      expect(component.setup).toBeTypeOf('function');
    }
  });
});
