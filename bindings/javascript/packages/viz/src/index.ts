/**
 * @moonlab/quantum-viz
 *
 * Beautiful quantum state visualizations for web applications.
 * Supports both Canvas 2D and WebGL 3D rendering.
 *
 * @example
 * ```typescript
 * import { BlochSphere, AmplitudeBars } from '@moonlab/quantum-viz';
 * import { QuantumState } from '@moonlab/quantum-core';
 *
 * // Create visualizations
 * const sphere = new BlochSphere('bloch-canvas');
 * const bars = new AmplitudeBars('amplitude-canvas');
 *
 * // Create quantum state
 * const state = await QuantumState.create({ numQubits: 2 });
 * state.h(0).cnot(0, 1);
 *
 * // Visualize
 * bars.setState(state.getAmplitudes(), state.numQubits);
 *
 * // For single qubit visualization on Bloch sphere
 * const singleQubit = await QuantumState.create({ numQubits: 1 });
 * singleQubit.h(0);
 * const amps = singleQubit.getAmplitudes();
 * sphere.setState([amps[0], amps[1]]);
 * ```
 *
 * @packageDocumentation
 */

// ============================================================================
// Canvas 2D Visualizations
// ============================================================================

export {
  BlochSphere,
  AmplitudeBars,
  CircuitDiagram,
} from './canvas';

export type {
  BlochSphereOptions,
  AmplitudeBarsOptions,
  CircuitDiagramOptions,
} from './canvas';

// ============================================================================
// Types and Utilities
// ============================================================================

export type {
  // Color types
  RGB,
  RGBA,
  HSL,
  HexColor,
  Color,

  // Geometry
  Point2D,
  Point3D,
  Size,
  BoundingBox,

  // Visualization options
  BaseVisualizationOptions,
  ColorSchemeOptions,

  // Quantum types
  AmplitudeInfo,
  BlochState,

  // Events
  VisualizationEventType,
  EventListener,
  InteractionEvent,
} from './types';

export {
  // Color utilities
  rgbToHex,
  hexToRgb,
  hslToRgb,
  phaseToHue,
  amplitudeToColor,

  // Quantum utilities
  indexToBitString,
  amplitudesToInfo,
  amplitudesToBlochState,

  // Animation utilities
  easeInOutCubic,
  lerp,
  clamp,
} from './types';

// ============================================================================
// WebGL 3D Visualizations
// ============================================================================

export { BlochSphere3D } from './webgl';
export type { BlochSphere3DOptions } from './webgl';

// ============================================================================
// Version Info
// ============================================================================

export const VERSION = '1.0.0';
