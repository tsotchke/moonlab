/**
 * Tests for complex number utilities
 */

import { describe, it, expect } from 'vitest';
import {
  complex,
  ZERO,
  ONE,
  I,
  magnitude,
  magnitudeSquared,
  phase,
  conjugate,
  add,
  subtract,
  multiply,
  scale,
  divide,
  exp,
  fromPolar,
  toPolar,
  equals,
  toString,
  fromInterleaved,
  toInterleaved,
  innerProduct,
  norm,
  normalize,
  type Complex,
} from '../complex';

describe('Complex Number Creation', () => {
  it('creates complex number with real and imaginary parts', () => {
    const c = complex(3, 4);
    expect(c.real).toBe(3);
    expect(c.imag).toBe(4);
  });

  it('creates real number when imaginary part omitted', () => {
    const c = complex(5);
    expect(c.real).toBe(5);
    expect(c.imag).toBe(0);
  });

  it('provides correct constants', () => {
    expect(ZERO).toEqual({ real: 0, imag: 0 });
    expect(ONE).toEqual({ real: 1, imag: 0 });
    expect(I).toEqual({ real: 0, imag: 1 });
  });
});

describe('Complex Magnitude and Phase', () => {
  it('calculates magnitude correctly', () => {
    // 3-4-5 triangle
    expect(magnitude(complex(3, 4))).toBe(5);
    expect(magnitude(complex(0, 1))).toBe(1);
    expect(magnitude(complex(1, 0))).toBe(1);
    expect(magnitude(complex(0, 0))).toBe(0);
  });

  it('calculates magnitude squared correctly', () => {
    expect(magnitudeSquared(complex(3, 4))).toBe(25);
    expect(magnitudeSquared(complex(1, 1))).toBe(2);
  });

  it('calculates phase correctly', () => {
    expect(phase(complex(1, 0))).toBeCloseTo(0);
    expect(phase(complex(0, 1))).toBeCloseTo(Math.PI / 2);
    expect(phase(complex(-1, 0))).toBeCloseTo(Math.PI);
    expect(phase(complex(0, -1))).toBeCloseTo(-Math.PI / 2);
    expect(phase(complex(1, 1))).toBeCloseTo(Math.PI / 4);
  });
});

describe('Complex Conjugate', () => {
  it('negates imaginary part', () => {
    const c = complex(3, 4);
    const conj = conjugate(c);
    expect(conj.real).toBe(3);
    expect(conj.imag).toBe(-4);
  });

  it('leaves real numbers unchanged', () => {
    const c = complex(5, 0);
    const conj = conjugate(c);
    expect(conj).toEqual({ real: 5, imag: 0 });
  });
});

describe('Complex Arithmetic', () => {
  describe('Addition', () => {
    it('adds complex numbers correctly', () => {
      const a = complex(1, 2);
      const b = complex(3, 4);
      const result = add(a, b);
      expect(result).toEqual({ real: 4, imag: 6 });
    });

    it('handles negative numbers', () => {
      const a = complex(-1, -2);
      const b = complex(3, 4);
      const result = add(a, b);
      expect(result).toEqual({ real: 2, imag: 2 });
    });
  });

  describe('Subtraction', () => {
    it('subtracts complex numbers correctly', () => {
      const a = complex(5, 7);
      const b = complex(2, 3);
      const result = subtract(a, b);
      expect(result).toEqual({ real: 3, imag: 4 });
    });
  });

  describe('Multiplication', () => {
    it('multiplies complex numbers correctly', () => {
      // (1+2i)(3+4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i
      const a = complex(1, 2);
      const b = complex(3, 4);
      const result = multiply(a, b);
      expect(result.real).toBeCloseTo(-5);
      expect(result.imag).toBeCloseTo(10);
    });

    it('i * i = -1', () => {
      const result = multiply(I, I);
      expect(result.real).toBeCloseTo(-1);
      expect(result.imag).toBeCloseTo(0);
    });

    it('multiplies by conjugate to get magnitude squared', () => {
      const c = complex(3, 4);
      const result = multiply(c, conjugate(c));
      expect(result.real).toBeCloseTo(25);
      expect(result.imag).toBeCloseTo(0);
    });
  });

  describe('Scalar Multiplication', () => {
    it('scales both parts', () => {
      const c = complex(3, 4);
      const result = scale(c, 2);
      expect(result).toEqual({ real: 6, imag: 8 });
    });

    it('handles negative scalars', () => {
      const c = complex(3, 4);
      const result = scale(c, -1);
      expect(result).toEqual({ real: -3, imag: -4 });
    });
  });

  describe('Division', () => {
    it('divides complex numbers correctly', () => {
      // (1+2i)/(3+4i) = (1+2i)(3-4i)/25 = (3 - 4i + 6i + 8)/25 = (11 + 2i)/25
      const a = complex(1, 2);
      const b = complex(3, 4);
      const result = divide(a, b);
      expect(result.real).toBeCloseTo(11 / 25);
      expect(result.imag).toBeCloseTo(2 / 25);
    });

    it('throws on division by zero', () => {
      expect(() => divide(complex(1, 2), ZERO)).toThrow('Division by zero');
    });

    it('divides by real number', () => {
      const a = complex(4, 6);
      const b = complex(2, 0);
      const result = divide(a, b);
      expect(result).toEqual({ real: 2, imag: 3 });
    });
  });
});

describe('Exponential and Polar Forms', () => {
  it('computes e^(iθ) correctly', () => {
    // e^(i*0) = 1
    const e0 = exp(0);
    expect(e0.real).toBeCloseTo(1);
    expect(e0.imag).toBeCloseTo(0);

    // e^(i*π/2) = i
    const ePiOver2 = exp(Math.PI / 2);
    expect(ePiOver2.real).toBeCloseTo(0);
    expect(ePiOver2.imag).toBeCloseTo(1);

    // e^(i*π) = -1
    const ePi = exp(Math.PI);
    expect(ePi.real).toBeCloseTo(-1);
    expect(ePi.imag).toBeCloseTo(0);
  });

  it('converts from polar form', () => {
    const c = fromPolar(5, Math.atan2(4, 3)); // magnitude 5, angle of 3+4i
    expect(c.real).toBeCloseTo(3);
    expect(c.imag).toBeCloseTo(4);
  });

  it('converts to polar form', () => {
    const [r, theta] = toPolar(complex(3, 4));
    expect(r).toBeCloseTo(5);
    expect(theta).toBeCloseTo(Math.atan2(4, 3));
  });

  it('round-trips through polar form', () => {
    const original = complex(3, 4);
    const [r, theta] = toPolar(original);
    const restored = fromPolar(r, theta);
    expect(restored.real).toBeCloseTo(original.real);
    expect(restored.imag).toBeCloseTo(original.imag);
  });
});

describe('Equality and String Conversion', () => {
  it('compares equal complex numbers', () => {
    expect(equals(complex(3, 4), complex(3, 4))).toBe(true);
  });

  it('compares unequal complex numbers', () => {
    expect(equals(complex(3, 4), complex(3, 5))).toBe(false);
    expect(equals(complex(3, 4), complex(4, 4))).toBe(false);
  });

  it('uses tolerance for approximate equality', () => {
    const a = complex(1, 2);
    const b = complex(1 + 1e-12, 2 - 1e-12);
    expect(equals(a, b, 1e-10)).toBe(true);
    expect(equals(a, b, 1e-15)).toBe(false);
  });

  it('formats complex numbers as strings', () => {
    expect(toString(complex(3, 4), 1)).toBe('3.0 + 4.0i');
    expect(toString(complex(3, -4), 1)).toBe('3.0 - 4.0i');
    expect(toString(complex(0, 0), 1)).toBe('0.0 + 0.0i');
  });
});

describe('Array Operations', () => {
  describe('Interleaved Conversion', () => {
    it('converts interleaved array to complex array', () => {
      const interleaved = new Float64Array([1, 2, 3, 4, 5, 6]);
      const complexArray = fromInterleaved(interleaved);
      expect(complexArray).toEqual([
        { real: 1, imag: 2 },
        { real: 3, imag: 4 },
        { real: 5, imag: 6 },
      ]);
    });

    it('converts complex array to interleaved array', () => {
      const complexArray: Complex[] = [
        { real: 1, imag: 2 },
        { real: 3, imag: 4 },
      ];
      const interleaved = toInterleaved(complexArray);
      expect(Array.from(interleaved)).toEqual([1, 2, 3, 4]);
    });

    it('round-trips through interleaved form', () => {
      const original: Complex[] = [
        { real: 1, imag: 2 },
        { real: 3, imag: 4 },
        { real: 5, imag: 6 },
      ];
      const restored = fromInterleaved(toInterleaved(original));
      expect(restored).toEqual(original);
    });
  });

  describe('Inner Product', () => {
    it('computes inner product correctly', () => {
      // <(1+i), (1+i)> = (1-i)(1+i) = 1 + 1 = 2
      const v = [complex(1, 1)];
      const result = innerProduct(v, v);
      expect(result.real).toBeCloseTo(2);
      expect(result.imag).toBeCloseTo(0);
    });

    it('computes inner product of orthogonal vectors', () => {
      // |0> = (1, 0), |1> = (0, 1) are orthogonal in C^2
      const v0 = [complex(1, 0), complex(0, 0)];
      const v1 = [complex(0, 0), complex(1, 0)];
      const result = innerProduct(v0, v1);
      expect(result.real).toBeCloseTo(0);
      expect(result.imag).toBeCloseTo(0);
    });

    it('throws on mismatched lengths', () => {
      const a = [complex(1, 0)];
      const b = [complex(1, 0), complex(0, 1)];
      expect(() => innerProduct(a, b)).toThrow('Vectors must have same length');
    });
  });

  describe('Norm and Normalize', () => {
    it('computes norm correctly', () => {
      // ||(3, 4)|| = 5
      const v = [complex(3, 0), complex(4, 0)];
      expect(norm(v)).toBeCloseTo(5);
    });

    it('computes norm of complex vector', () => {
      // ||(1+i)/√2|| = 1
      const v = [complex(1 / Math.SQRT2, 1 / Math.SQRT2)];
      expect(norm(v)).toBeCloseTo(1);
    });

    it('normalizes vector to unit length', () => {
      const v = [complex(3, 0), complex(4, 0)];
      const normalized = normalize(v);
      expect(norm(normalized)).toBeCloseTo(1);
      expect(normalized[0].real).toBeCloseTo(0.6);
      expect(normalized[1].real).toBeCloseTo(0.8);
    });

    it('throws when normalizing zero vector', () => {
      const v = [ZERO, ZERO];
      expect(() => normalize(v)).toThrow('Cannot normalize zero vector');
    });
  });
});

describe('Quantum Mechanics Applications', () => {
  it('represents |+> state correctly', () => {
    // |+> = (|0> + |1>)/√2 = (1/√2, 1/√2)
    const plus: Complex[] = [
      complex(1 / Math.SQRT2, 0),
      complex(1 / Math.SQRT2, 0),
    ];
    expect(norm(plus)).toBeCloseTo(1); // normalized
  });

  it('represents |-> state correctly', () => {
    // |-> = (|0> - |1>)/√2 = (1/√2, -1/√2)
    const minus: Complex[] = [
      complex(1 / Math.SQRT2, 0),
      complex(-1 / Math.SQRT2, 0),
    ];
    expect(norm(minus)).toBeCloseTo(1);
  });

  it('|+> and |-> are orthogonal', () => {
    const plus: Complex[] = [
      complex(1 / Math.SQRT2, 0),
      complex(1 / Math.SQRT2, 0),
    ];
    const minus: Complex[] = [
      complex(1 / Math.SQRT2, 0),
      complex(-1 / Math.SQRT2, 0),
    ];
    const ip = innerProduct(plus, minus);
    expect(ip.real).toBeCloseTo(0);
    expect(ip.imag).toBeCloseTo(0);
  });

  it('probability from amplitude squared', () => {
    // |1/√2|² = 1/2
    const amplitude = complex(1 / Math.SQRT2, 0);
    expect(magnitudeSquared(amplitude)).toBeCloseTo(0.5);
  });

  it('handles phase factors correctly', () => {
    // e^(iπ/4) has magnitude 1
    const phased = exp(Math.PI / 4);
    expect(magnitude(phased)).toBeCloseTo(1);
  });
});
