import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import {
  VERSION,
  AmplitudeBars,
  BlochSphere,
  CircuitDiagram,
  useCircuit,
  useQuantumState,
} from '../index';

describe('@moonlab/quantum-react public surface', () => {
  it('reports the release version and exports hooks', () => {
    expect(VERSION).toBe('1.1.0');
    expect(useQuantumState).toBeTypeOf('function');
    expect(useCircuit).toBeTypeOf('function');
  });

  it('server-renders all visualization components without a browser', () => {
    const components = [BlochSphere, AmplitudeBars, CircuitDiagram];
    for (const Component of components) {
      const markup = renderToStaticMarkup(React.createElement(Component));
      expect(markup).toContain('<canvas');
    }
  });
});
