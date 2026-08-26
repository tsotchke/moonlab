/**
 * The diverging ramp for signed fields.
 *
 * Berry curvature has a sign that matters -- it is polarity, not magnitude --
 * so the scale is diverging: two hues with a *neutral* midpoint, equal steps
 * per arm, and no hue at the middle. A rainbow would invent structure the data
 * does not have, and a single-hue sequential ramp would throw the sign away.
 *
 * The poles come from the active theme rather than a fixed pair, because a
 * theme belongs to the shell and hardcoding colours here would look wrong in
 * seventeen of the eighteen. What is fixed is the *shape*: neutral centre,
 * monotone lightness out to each pole, which is the property that makes a
 * diverging ramp readable. `rampIsMonotone` exists so that is tested rather
 * than asserted.
 */

import type { Rgb } from "./cells.ts";

/** WCAG relative luminance, 0..1. */
export function luminance([r, g, b]: Rgb): number {
  const channel = (value: number) => {
    const v = value / 255;
    return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

function mix(a: Rgb, b: Rgb, t: number): Rgb {
  const clamp = (v: number) => Math.max(0, Math.min(255, Math.round(v)));
  return [
    clamp(a[0] + (b[0] - a[0]) * t),
    clamp(a[1] + (b[1] - a[1]) * t),
    clamp(a[2] + (b[2] - a[2]) * t),
  ];
}

/** The neutral centre: the theme's own grey, never one of the two hues. */
export function neutralMidpoint(surface: Rgb, muted: Rgb): Rgb {
  return mix(surface, muted, 0.45);
}

export interface DivergingRamp {
  /** Steps from the negative pole through neutral to the positive pole. */
  readonly steps: readonly Rgb[];
  /** Index of the neutral midpoint. */
  readonly mid: number;
}

/**
 * Builds a diverging ramp with `perArm` steps either side of the midpoint,
 * so the total is `2 * perArm + 1` and the arms are symmetric by construction.
 */
export function divergingRamp(
  negative: Rgb,
  neutral: Rgb,
  positive: Rgb,
  perArm = 6,
): DivergingRamp {
  const steps: Rgb[] = [];
  for (let i = perArm; i >= 1; i--) steps.push(mix(neutral, negative, i / perArm));
  steps.push(neutral);
  for (let i = 1; i <= perArm; i++) steps.push(mix(neutral, positive, i / perArm));
  return { steps, mid: perArm };
}

/**
 * Lightness must move monotonically from the midpoint out along each arm, or
 * the reader cannot rank magnitudes within a sign. This is the check that
 * applies to a diverging ramp -- the categorical CVD validator fails ramps by
 * design, because their steps sit deliberately close together.
 */
export function rampIsMonotone(ramp: DivergingRamp): boolean {
  const l = ramp.steps.map(luminance);
  const centre = l[ramp.mid];
  const arm = (indices: number[]) => {
    let previous = centre;
    let direction = 0;
    for (const index of indices) {
      const value = l[index];
      const delta = value - previous;
      if (Math.abs(delta) < 1e-9) return false;
      const sign = Math.sign(delta);
      if (direction === 0) direction = sign;
      else if (sign !== direction) return false;
      previous = value;
    }
    return true;
  };
  const negative = Array.from({ length: ramp.mid }, (_, i) => ramp.mid - 1 - i);
  const positive = Array.from({ length: ramp.mid }, (_, i) => ramp.mid + 1 + i);
  return arm(negative) && arm(positive);
}

/**
 * Picks a step for `value`.
 *
 * The scale is symmetric about zero -- `extent` is used for both arms -- so a
 * field that is mostly positive still shows its negative lobe at a comparable
 * intensity, and zero always lands exactly on the neutral step.
 */
export function rampColor(ramp: DivergingRamp, value: number, extent: number): Rgb {
  if (!(extent > 0) || !Number.isFinite(value)) return ramp.steps[ramp.mid];
  const t = Math.max(-1, Math.min(1, value / extent));
  const offset = Math.round(t * ramp.mid);
  return ramp.steps[ramp.mid + offset];
}

/**
 * A sequential ramp for magnitude.
 *
 * |psi|^2 is a magnitude, not a polarity: it has no meaningful zero to diverge
 * about, so it takes one hue running light to dark rather than the two-hue
 * diverging scale the Berry curvature uses. Getting this distinction wrong is
 * the most common way a correct number ends up in a misleading picture.
 */
export function sequentialRamp(surface: Rgb, hue: Rgb, steps = 12): readonly Rgb[] {
  const ramp: Rgb[] = [];
  for (let i = 0; i < steps; i++) ramp.push(mix(surface, hue, i / (steps - 1)));
  return ramp;
}

/** True when luminance moves in one direction across the whole ramp. */
export function sequentialIsMonotone(ramp: readonly Rgb[]): boolean {
  let direction = 0;
  for (let i = 1; i < ramp.length; i++) {
    const delta = luminance(ramp[i]) - luminance(ramp[i - 1]);
    if (Math.abs(delta) < 1e-9) return false;
    const sign = Math.sign(delta);
    if (direction === 0) direction = sign;
    else if (sign !== direction) return false;
  }
  return true;
}

/** Picks a step for a value in `[0, max]`, with a gamma to lift faint detail. */
export function sequentialColor(
  ramp: readonly Rgb[],
  value: number,
  max: number,
  gamma = 0.45,
): Rgb {
  if (!(max > 0) || !(value > 0)) return ramp[0];
  const t = Math.min(1, (value / max) ** gamma);
  return ramp[Math.round(t * (ramp.length - 1))];
}
