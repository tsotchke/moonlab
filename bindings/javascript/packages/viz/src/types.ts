/**
 * Common Types for Quantum Visualizations
 */

import type { Complex } from '@moonlab/quantum-core';

// ============================================================================
// Color Types
// ============================================================================

/**
 * RGB color as [r, g, b] with values 0-255
 */
export type RGB = [number, number, number];

/**
 * RGBA color as [r, g, b, a] with r,g,b 0-255 and a 0-1
 */
export type RGBA = [number, number, number, number];

/**
 * HSL color as [h, s, l] with h 0-360, s,l 0-100
 */
export type HSL = [number, number, number];

/**
 * Color as hex string (#RRGGBB or #RRGGBBAA)
 */
export type HexColor = string;

/**
 * Any supported color format
 */
export type Color = RGB | RGBA | HSL | HexColor;

// ============================================================================
// Geometry Types
// ============================================================================

/**
 * 2D point
 */
export interface Point2D {
  x: number;
  y: number;
}

/**
 * 3D point
 */
export interface Point3D {
  x: number;
  y: number;
  z: number;
}

/**
 * Size in pixels
 */
export interface Size {
  width: number;
  height: number;
}

/**
 * Bounding box
 */
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

// ============================================================================
// Visualization Options
// ============================================================================

/**
 * Base options for all visualizations
 */
export interface BaseVisualizationOptions {
  /**
   * Canvas width in pixels (default: 400)
   */
  width?: number;

  /**
   * Canvas height in pixels (default: 400)
   */
  height?: number;

  /**
   * Background color (default: transparent)
   */
  backgroundColor?: Color;

  /**
   * Enable animations (default: true)
   */
  animated?: boolean;

  /**
   * Animation duration in ms (default: 300)
   */
  animationDuration?: number;

  /**
   * Device pixel ratio for high-DPI displays (default: auto)
   */
  pixelRatio?: number;
}

/**
 * Color scheme options
 */
export interface ColorSchemeOptions {
  /**
   * Color scheme preset
   */
  scheme?: 'default' | 'dark' | 'light' | 'rainbow' | 'monochrome';

  /**
   * Primary color for states/bars
   */
  primaryColor?: Color;

  /**
   * Secondary color for accents
   */
  secondaryColor?: Color;

  /**
   * Text color
   */
  textColor?: Color;

  /**
   * Grid/axis color
   */
  gridColor?: Color;
}

// ============================================================================
// Quantum State Representation
// ============================================================================

/**
 * Amplitude with phase information for visualization
 */
export interface AmplitudeInfo {
  /**
   * Basis state index (0 to 2^n - 1)
   */
  index: number;

  /**
   * Basis state as binary string
   */
  bitString: string;

  /**
   * Complex amplitude
   */
  amplitude: Complex;

  /**
   * Probability (|amplitude|^2)
   */
  probability: number;

  /**
   * Phase angle in radians
   */
  phase: number;
}

/**
 * Single qubit state on Bloch sphere
 */
export interface BlochState {
  /**
   * Theta angle (polar, 0 to pi)
   */
  theta: number;

  /**
   * Phi angle (azimuthal, 0 to 2pi)
   */
  phi: number;

  /**
   * Cartesian coordinates
   */
  x: number;
  y: number;
  z: number;
}

// ============================================================================
// Event Types
// ============================================================================

/**
 * Visualization event types
 */
export type VisualizationEventType =
  | 'click'
  | 'hover'
  | 'stateChange'
  | 'animationStart'
  | 'animationEnd';

/**
 * Event listener callback
 */
export type EventListener<T = unknown> = (event: T) => void;

/**
 * Click/hover event data
 */
export interface InteractionEvent {
  type: 'click' | 'hover';
  point: Point2D;
  basisState?: number;
  qubit?: number;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Convert RGB to hex string
 */
export function rgbToHex(rgb: RGB | RGBA): HexColor {
  const [r, g, b, a] = rgb.length === 4 ? rgb : [...rgb, 1];
  const hex = [r, g, b]
    .map((c) => Math.round(c).toString(16).padStart(2, '0'))
    .join('');
  if (a !== 1) {
    return `#${hex}${Math.round(a * 255).toString(16).padStart(2, '0')}`;
  }
  return `#${hex}`;
}

/**
 * Convert hex to RGB
 */
export function hexToRgb(hex: HexColor): RGB {
  const match = hex.match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})?$/i);
  if (!match) {
    throw new Error(`Invalid hex color: ${hex}`);
  }
  return [parseInt(match[1], 16), parseInt(match[2], 16), parseInt(match[3], 16)];
}

/**
 * Convert HSL to RGB
 */
export function hslToRgb(hsl: HSL): RGB {
  const [h, s, l] = [hsl[0] / 360, hsl[1] / 100, hsl[2] / 100];
  let r: number, g: number, b: number;

  if (s === 0) {
    r = g = b = l;
  } else {
    const hue2rgb = (p: number, q: number, t: number): number => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };

    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }

  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

/**
 * Convert phase (radians) to hue (0-360)
 * Maps -π to +π onto the color wheel
 */
export function phaseToHue(phase: number): number {
  // Normalize phase to [0, 2π]
  const normalized = ((phase % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  // Convert to hue [0, 360]
  return (normalized / (2 * Math.PI)) * 360;
}

/**
 * Get color for a quantum amplitude based on phase
 * Uses phase-as-hue coloring with lightness based on probability
 */
export function amplitudeToColor(amplitude: Complex, maxProbability: number = 1): RGBA {
  const probability = amplitude.real * amplitude.real + amplitude.imag * amplitude.imag;
  const phase = Math.atan2(amplitude.imag, amplitude.real);

  const hue = phaseToHue(phase);
  const saturation = 80;
  const lightness = 30 + 40 * Math.sqrt(probability / maxProbability);

  const rgb = hslToRgb([hue, saturation, lightness]);
  const alpha = Math.max(0.2, probability / maxProbability);

  return [...rgb, alpha];
}

/**
 * Convert basis state index to bit string
 */
export function indexToBitString(index: number, numQubits: number): string {
  return index.toString(2).padStart(numQubits, '0');
}

/**
 * Convert complex amplitudes to AmplitudeInfo array
 */
export function amplitudesToInfo(amplitudes: Complex[], numQubits: number): AmplitudeInfo[] {
  return amplitudes.map((amplitude, index) => {
    const probability = amplitude.real * amplitude.real + amplitude.imag * amplitude.imag;
    const phase = Math.atan2(amplitude.imag, amplitude.real);

    return {
      index,
      bitString: indexToBitString(index, numQubits),
      amplitude,
      probability,
      phase,
    };
  });
}

/**
 * Calculate Bloch sphere coordinates from amplitudes
 * Only valid for single-qubit states
 */
export function amplitudesToBlochState(amplitudes: [Complex, Complex]): BlochState {
  const [a0, a1] = amplitudes;

  // |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
  const r0 = Math.sqrt(a0.real * a0.real + a0.imag * a0.imag);
  const r1 = Math.sqrt(a1.real * a1.real + a1.imag * a1.imag);

  // Global phase (from |0⟩ coefficient)
  const globalPhase = Math.atan2(a0.imag, a0.real);

  // Remove global phase from |1⟩ coefficient to get relative phase
  const a1Phase = Math.atan2(a1.imag, a1.real);
  const phi = a1Phase - globalPhase;

  // θ from probability amplitudes
  const theta = 2 * Math.acos(Math.min(1, r0));

  // Bloch sphere coordinates
  const x = Math.sin(theta) * Math.cos(phi);
  const y = Math.sin(theta) * Math.sin(phi);
  const z = Math.cos(theta);

  return { theta, phi, x, y, z };
}

/**
 * Easing function for animations
 */
export function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/**
 * Linear interpolation
 */
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Clamp value to range
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
