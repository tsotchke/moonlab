import { describe, expect, it } from 'vitest';
import {
  VERSION,
  amplitudesToBlochState,
  amplitudesToInfo,
  clamp,
  hexToRgb,
  indexToBitString,
  rgbToHex,
} from '../index';

describe('@moonlab/quantum-viz public utilities', () => {
  it('reports the release version and round-trips colors', () => {
    expect(VERSION).toBe('1.2.0');
    expect(rgbToHex([12, 34, 56])).toBe('#0c2238');
    expect(hexToRgb('#0c2238')).toEqual([12, 34, 56]);
  });

  it('derives basis labels and normalized probabilities', () => {
    const amplitudes = [
      { real: Math.SQRT1_2, imag: 0 },
      { real: 0, imag: 0 },
      { real: 0, imag: 0 },
      { real: 0, imag: Math.SQRT1_2 },
    ];
    const info = amplitudesToInfo(amplitudes, 2);
    expect(indexToBitString(3, 2)).toBe('11');
    expect(info.map((entry) => entry.bitString)).toEqual(['00', '01', '10', '11']);
    expect(info[0].probability).toBeCloseTo(0.5);
    expect(info[3].probability).toBeCloseTo(0.5);
  });

  it('maps single-qubit amplitudes onto the Bloch sphere', () => {
    const state = amplitudesToBlochState([
      { real: Math.SQRT1_2, imag: 0 },
      { real: Math.SQRT1_2, imag: 0 },
    ]);
    expect(state.x).toBeCloseTo(1);
    expect(state.y).toBeCloseTo(0);
    expect(state.z).toBeCloseTo(0);
    expect(clamp(3, -1, 1)).toBe(1);
  });
});
