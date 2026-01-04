/**
 * Complex number utilities for quantum state representation.
 *
 * Quantum amplitudes are complex numbers of the form a + bi.
 */

/**
 * Complex number interface
 */
export interface Complex {
  real: number;
  imag: number;
}

/**
 * Create a complex number
 */
export function complex(real: number, imag: number = 0): Complex {
  return { real, imag };
}

/**
 * Complex zero
 */
export const ZERO: Complex = { real: 0, imag: 0 };

/**
 * Complex one
 */
export const ONE: Complex = { real: 1, imag: 0 };

/**
 * Complex i (imaginary unit)
 */
export const I: Complex = { real: 0, imag: 1 };

/**
 * Calculate magnitude (absolute value) |z| = sqrt(a² + b²)
 */
export function magnitude(c: Complex): number {
  return Math.sqrt(c.real * c.real + c.imag * c.imag);
}

/**
 * Calculate magnitude squared |z|² = a² + b²
 * This is the probability in quantum mechanics
 */
export function magnitudeSquared(c: Complex): number {
  return c.real * c.real + c.imag * c.imag;
}

/**
 * Calculate phase angle arg(z) = atan2(b, a)
 */
export function phase(c: Complex): number {
  return Math.atan2(c.imag, c.real);
}

/**
 * Complex conjugate z* = a - bi
 */
export function conjugate(c: Complex): Complex {
  return { real: c.real, imag: -c.imag };
}

/**
 * Complex addition z1 + z2
 */
export function add(a: Complex, b: Complex): Complex {
  return {
    real: a.real + b.real,
    imag: a.imag + b.imag,
  };
}

/**
 * Complex subtraction z1 - z2
 */
export function subtract(a: Complex, b: Complex): Complex {
  return {
    real: a.real - b.real,
    imag: a.imag - b.imag,
  };
}

/**
 * Complex multiplication z1 * z2 = (a1*a2 - b1*b2) + (a1*b2 + a2*b1)i
 */
export function multiply(a: Complex, b: Complex): Complex {
  return {
    real: a.real * b.real - a.imag * b.imag,
    imag: a.real * b.imag + a.imag * b.real,
  };
}

/**
 * Scalar multiplication s * z
 */
export function scale(c: Complex, s: number): Complex {
  return {
    real: c.real * s,
    imag: c.imag * s,
  };
}

/**
 * Complex division z1 / z2
 */
export function divide(a: Complex, b: Complex): Complex {
  const denom = b.real * b.real + b.imag * b.imag;
  if (denom === 0) {
    throw new Error('Division by zero');
  }
  return {
    real: (a.real * b.real + a.imag * b.imag) / denom,
    imag: (a.imag * b.real - a.real * b.imag) / denom,
  };
}

/**
 * Complex exponential e^(iθ) = cos(θ) + i*sin(θ)
 */
export function exp(theta: number): Complex {
  return {
    real: Math.cos(theta),
    imag: Math.sin(theta),
  };
}

/**
 * Create complex from polar form r*e^(iθ)
 */
export function fromPolar(r: number, theta: number): Complex {
  return {
    real: r * Math.cos(theta),
    imag: r * Math.sin(theta),
  };
}

/**
 * Convert to polar form [r, θ]
 */
export function toPolar(c: Complex): [number, number] {
  return [magnitude(c), phase(c)];
}

/**
 * Check if two complex numbers are approximately equal
 */
export function equals(a: Complex, b: Complex, tolerance: number = 1e-10): boolean {
  return (
    Math.abs(a.real - b.real) < tolerance && Math.abs(a.imag - b.imag) < tolerance
  );
}

/**
 * Format complex number as string
 */
export function toString(c: Complex, precision: number = 4): string {
  const r = c.real.toFixed(precision);
  const i = Math.abs(c.imag).toFixed(precision);
  const sign = c.imag >= 0 ? '+' : '-';
  return `${r} ${sign} ${i}i`;
}

/**
 * Convert interleaved Float64Array to Complex array
 * WASM stores complex as [real0, imag0, real1, imag1, ...]
 */
export function fromInterleaved(arr: Float64Array): Complex[] {
  const result: Complex[] = [];
  for (let i = 0; i < arr.length; i += 2) {
    result.push({ real: arr[i], imag: arr[i + 1] });
  }
  return result;
}

/**
 * Convert Complex array to interleaved Float64Array
 */
export function toInterleaved(complexArray: Complex[]): Float64Array {
  const result = new Float64Array(complexArray.length * 2);
  for (let i = 0; i < complexArray.length; i++) {
    result[i * 2] = complexArray[i].real;
    result[i * 2 + 1] = complexArray[i].imag;
  }
  return result;
}

/**
 * Calculate inner product <a|b> = Σ conj(a_i) * b_i
 */
export function innerProduct(a: Complex[], b: Complex[]): Complex {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same length');
  }
  let result = ZERO;
  for (let i = 0; i < a.length; i++) {
    result = add(result, multiply(conjugate(a[i]), b[i]));
  }
  return result;
}

/**
 * Calculate norm ||v|| = sqrt(<v|v>)
 */
export function norm(v: Complex[]): number {
  let sum = 0;
  for (const c of v) {
    sum += magnitudeSquared(c);
  }
  return Math.sqrt(sum);
}

/**
 * Normalize a complex vector to unit length
 */
export function normalize(v: Complex[]): Complex[] {
  const n = norm(v);
  if (n === 0) {
    throw new Error('Cannot normalize zero vector');
  }
  return v.map((c) => scale(c, 1 / n));
}
