/**
 * Band geometry: the ramp, the physics, and the window.
 *
 * The physics assertions are the valuable ones. The Qi-Wu-Zhang model has a
 * documented phase diagram -- C = +1 for -2 < m < 0, C = -1 for 0 < m < 2,
 * C = 0 outside -- so the Chern numbers this backend reports are checkable
 * against theory rather than against a recorded snapshot of themselves.
 */

import { openNativeBackend } from "../src/backend/mod.ts";
import {
  divergingRamp,
  luminance,
  neutralMidpoint,
  rampColor,
  rampIsMonotone,
} from "../src/ui/colormap.ts";
import { fittedGridSize, paintBandGeometry, paletteRamp } from "../src/ui/band_geometry.ts";
import { Surface } from "../src/ui/cells.ts";
import { desktopPalette, THEMES } from "../src/ui/theme.ts";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(`assertion failed: ${message}`);
}

Deno.test("the diverging ramp is monotone in lightness on every theme", () => {
  // The check that applies to a diverging ramp is lightness monotonicity out
  // from the midpoint; the categorical CVD validator fails ramps by design.
  for (const theme of THEMES) {
    const ramp = paletteRamp(desktopPalette(theme));
    assert(rampIsMonotone(ramp), `ramp not monotone for theme "${theme.id}"`);
  }
});

Deno.test("the ramp centre is neutral, not a hue", () => {
  const palette = desktopPalette(THEMES[0]);
  const mid = neutralMidpoint(palette.surface, palette.muted);
  const [r, g, b] = mid;
  // "Neutral" here means no channel dominates: a diverging midpoint carrying a
  // hue reads as a third category rather than as zero.
  const spread = Math.max(r, g, b) - Math.min(r, g, b);
  assert(spread < 40, `midpoint is not neutral enough: ${mid} spread ${spread}`);
});

Deno.test("zero maps to the midpoint and the arms are symmetric", () => {
  const ramp = divergingRamp([0, 0, 255], [128, 128, 128], [255, 0, 0], 6);
  assert(ramp.steps.length === 13, `expected 13 steps, got ${ramp.steps.length}`);
  assert(rampColor(ramp, 0, 1) === ramp.steps[ramp.mid], "zero is not the midpoint");
  const low = rampColor(ramp, -1, 1);
  const high = rampColor(ramp, 1, 1);
  assert(low === ramp.steps[0], "negative extreme is not the first step");
  assert(high === ramp.steps[ramp.steps.length - 1], "positive extreme is not the last");
  assert(luminance(low) !== luminance(high), "the two poles are indistinguishable");
});

Deno.test("QWZ Chern numbers match the published phase diagram", async () => {
  const backend = await openNativeBackend();
  try {
    assert(backend.capabilities.bandGeometry, "native backend lacks band geometry");
    const cases: Array<[number, number]> = [
      [-3.0, 0],
      [-1.5, 1],
      [-0.5, 1],
      [0.5, -1],
      [1.5, -1],
      [3.0, 0],
    ];
    for (const [m, expected] of cases) {
      const grid = await backend.berryGrid!({ kind: "qwz", m }, 24);
      const chern = Math.round(grid.chern);
      assert(
        chern === expected,
        `QWZ m=${m}: expected C=${expected}, got ${chern} (raw ${grid.chern})`,
      );
      // FHS gives an exact integer on a gapped band, so the raw value should
      // be an integer to well within rounding, not merely round to one.
      assert(
        Math.abs(grid.chern - chern) < 1e-6,
        `QWZ m=${m}: chern ${grid.chern} is not integral`,
      );
      assert(grid.n === 24 && grid.curvature.length === 24 * 24, "grid shape wrong");
    }
  } finally {
    await backend.dispose();
  }
});

Deno.test("curvature is antisymmetric in the mass parameter", async () => {
  const backend = await openNativeBackend();
  try {
    // QWZ at +m and -m are related by a sign flip of the curvature, so the
    // extents should mirror. This catches a field that is silently all zeros.
    const negative = await backend.berryGrid!({ kind: "qwz", m: -1 }, 16);
    const positive = await backend.berryGrid!({ kind: "qwz", m: 1 }, 16);
    assert(negative.max > 0, "negative-mass field has no positive lobe");
    assert(positive.min < 0, "positive-mass field has no negative lobe");
    assert(
      Math.abs(negative.max + positive.min) < 1e-9,
      `extents not mirrored: ${negative.max} vs ${positive.min}`,
    );
  } finally {
    await backend.dispose();
  }
});

Deno.test("the window says so when a backend cannot do band geometry", () => {
  const surface = new Surface(60, 14);
  paintBandGeometry(surface, { column: 0, row: 0, width: 60, height: 14 }, {
    modelLabel: "Qi-Wu-Zhang",
    paramLabel: "m = -1.00",
    busy: false,
    unavailable: "the wasm backend does not export the quantum-geometry symbols",
  }, desktopPalette(THEMES[0]));
  const text = surface.toText();
  assert(text.includes("band geometry unavailable"), `no honest notice:\n${text}`);
  assert(text.includes("does not export"), `reason missing:\n${text}`);
  assert(!text.includes("C ="), `claimed a Chern number it does not have:\n${text}`);
});

Deno.test("the grid shrinks to fit rather than overflowing", () => {
  assert(fittedGridSize({ column: 0, row: 0, width: 20, height: 30 }, 32) === 9, "width bound");
  assert(fittedGridSize({ column: 0, row: 0, width: 200, height: 12 }, 32) === 7, "height bound");
  assert(fittedGridSize({ column: 0, row: 0, width: 200, height: 60 }, 32) === 32, "requested");
});
