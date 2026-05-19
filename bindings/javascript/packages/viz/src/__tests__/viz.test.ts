import { describe, expect, it } from 'vitest';
import {
  AmplitudeBars,
  BlochSphere,
  CircuitDiagram,
  VERSION,
  amplitudeToColor,
  amplitudesToBlochState,
  amplitudesToInfo,
  clamp,
  hexToRgb,
  hslToRgb,
  indexToBitString,
  phaseToHue,
  rgbToHex,
} from '../index';

describe('@moonlab/quantum-viz exports', () => {
  it('exports the visualization classes and version', () => {
    expect(typeof BlochSphere).toBe('function');
    expect(typeof AmplitudeBars).toBe('function');
    expect(typeof CircuitDiagram).toBe('function');
    expect(VERSION).toMatch(/^\d+\.\d+\.\d+$/);
  });
});

describe('color utilities', () => {
  it('converts between RGB, hex, HSL, and phase colors', () => {
    expect(rgbToHex([255, 16, 0])).toBe('#ff1000');
    expect(rgbToHex([255, 16, 0, 0.5])).toBe('#ff100080');
    expect(hexToRgb('#0a14ff')).toEqual([10, 20, 255]);
    expect(hslToRgb([0, 100, 50])).toEqual([255, 0, 0]);
    expect(phaseToHue(Math.PI)).toBe(180);

    const color = amplitudeToColor({ real: 0, imag: 1 }, 1);
    expect(color).toHaveLength(4);
    expect(color[3]).toBeGreaterThan(0);
    expect(color[3]).toBeLessThanOrEqual(1);
  });
});

describe('quantum visualization utilities', () => {
  it('maps amplitudes into labeled probability records', () => {
    const info = amplitudesToInfo(
      [
        { real: Math.SQRT1_2, imag: 0 },
        { real: 0, imag: Math.SQRT1_2 },
      ],
      1
    );

    expect(info).toHaveLength(2);
    expect(info[0]).toMatchObject({ index: 0, bitString: '0' });
    expect(info[0].probability).toBeCloseTo(0.5);
    expect(info[1]).toMatchObject({ index: 1, bitString: '1' });
    expect(info[1].phase).toBeCloseTo(Math.PI / 2);
  });

  it('derives Bloch coordinates and common helpers', () => {
    const plus = amplitudesToBlochState([
      { real: Math.SQRT1_2, imag: 0 },
      { real: Math.SQRT1_2, imag: 0 },
    ]);

    expect(plus.x).toBeCloseTo(1);
    expect(plus.y).toBeCloseTo(0);
    expect(plus.z).toBeCloseTo(0);
    expect(indexToBitString(5, 4)).toBe('0101');
    expect(clamp(12, -2, 4)).toBe(4);
  });
});
