/**
 * The Schrödinger window.
 *
 * The orbital assertions are checkable against textbook physics rather than
 * against a snapshot of our own output: a 2p has a nodal plane, an s orbital
 * is isotropic, and every density integrates to one.
 */

import { openNativeBackend } from "../src/backend/mod.ts";
import {
  applyRelativisticContraction,
  buildCorrelationTerms,
  correctedDensitySlice,
  densitySlice,
  effectiveNuclearCharge,
  NO_CORRECTIONS,
  orbitalIsValid,
  orbitalLabel,
  radialWavefunction,
  suggestedExtent,
} from "../src/app/orbital.ts";
import { densityToRgb } from "../src/ui/graphics.ts";
import { computeOrbital } from "../src/app/orbital_job.ts";
import { sequentialColor, sequentialIsMonotone, sequentialRamp } from "../src/ui/colormap.ts";
import { desktopPalette, THEMES } from "../src/ui/theme.ts";
import { orbitalRamp } from "../src/ui/orbital_view.ts";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(`assertion failed: ${message}`);
}

const at = (s: ReturnType<typeof densitySlice>, row: number, col: number) =>
  s.density[row * s.size + col];

Deno.test("every density normalises to one", () => {
  for (const o of [{ n: 1, l: 0, m: 0, z: 1 }, { n: 3, l: 2, m: 0, z: 1 }]) {
    const slice = densitySlice(o, 41, suggestedExtent(o));
    let sum = 0;
    for (const v of slice.density) sum += v;
    assert(Math.abs(sum - 1) < 1e-12, `${orbitalLabel(o)} sums to ${sum}`);
  }
});

Deno.test("s orbitals peak at the nucleus and are isotropic", () => {
  const slice = densitySlice(
    { n: 1, l: 0, m: 0, z: 1 },
    41,
    suggestedExtent({ n: 1, l: 0, m: 0, z: 1 }),
  );
  const mid = 20;
  assert(at(slice, mid, mid) === slice.max, "1s does not peak at the nucleus");
  // Equal distance along x and along z must give equal density.
  assert(
    Math.abs(at(slice, mid, mid + 8) - at(slice, mid - 8, mid)) < 1e-15,
    "1s is not isotropic",
  );
});

Deno.test("2p has a nodal plane at z = 0 and lobes along z", () => {
  const o = { n: 2, l: 1, m: 0, z: 1 };
  const slice = densitySlice(o, 41, suggestedExtent(o));
  const mid = 20;
  assert(at(slice, mid, mid) === 0, "2p is nonzero at the nucleus");
  // The whole z = 0 row is the nodal plane.
  for (let column = 0; column < slice.size; column++) {
    assert(at(slice, mid, column) < 1e-20, `2p nodal plane broken at column ${column}`);
  }
  assert(at(slice, mid - 8, mid) > 0, "2p has no lobe along +z");
});

Deno.test("radial wavefunctions decay", () => {
  const values = [0.5, 1, 2, 4, 8].map((r) => radialWavefunction(1, 0, 1, r));
  assert(values.every((v, i) => i === 0 || v < values[i - 1]), "1s radial is not monotone");
});

Deno.test("the sequential ramp is monotone on every theme", () => {
  for (const theme of THEMES) {
    const ramp = orbitalRamp(desktopPalette(theme));
    assert(sequentialIsMonotone(ramp), `ramp not monotone for "${theme.id}"`);
  }
  const ramp = sequentialRamp([0, 0, 0], [255, 255, 255], 8);
  assert(sequentialColor(ramp, 0, 1) === ramp[0], "zero is not the low end");
  assert(sequentialColor(ramp, 1, 1) === ramp[ramp.length - 1], "max is not the high end");
});

Deno.test("MoonLab round-trips the density to machine precision", async () => {
  const backend = await openNativeBackend();
  try {
    assert(backend.capabilities.amplitudeUpload, "native backend cannot upload amplitudes");
    for (const o of [{ n: 1, l: 0, m: 0, z: 1 }, { n: 2, l: 1, m: 0, z: 1 }]) {
      const result = await computeOrbital(backend, o, 32);
      assert(result.drift !== undefined, `${orbitalLabel(o)}: no round-trip ran`);
      // The analytic answer is known, so this is a real accuracy measurement.
      assert(
        result.drift! < 1e-12,
        `${orbitalLabel(o)}: L1 drift ${result.drift} is too large`,
      );
    }
  } finally {
    await backend.dispose();
  }
});

Deno.test("the offered orbitals are all physically valid", () => {
  // l < n and |m| <= l; a typo here would render as a silently empty slice.
  const offered = [
    { n: 1, l: 0, m: 0, z: 1 },
    { n: 2, l: 0, m: 0, z: 1 },
    { n: 2, l: 1, m: 0, z: 1 },
    { n: 2, l: 1, m: 1, z: 1 },
    { n: 3, l: 0, m: 0, z: 1 },
    { n: 3, l: 1, m: 0, z: 1 },
    { n: 3, l: 2, m: 0, z: 1 },
    { n: 3, l: 2, m: 2, z: 1 },
    { n: 4, l: 3, m: 0, z: 1 },
  ];
  for (const o of offered) {
    assert(orbitalIsValid(o), `invalid quantum numbers: ${JSON.stringify(o)}`);
  }
});

Deno.test("corrections off reproduce the hydrogenic baseline exactly", () => {
  // An approximation that quietly moved the uncorrected result would be worse
  // than not offering one.
  const o = { n: 3, l: 1, m: 0, z: 1 };
  const plain = densitySlice(o, 33, suggestedExtent(o));
  const corrected = correctedDensitySlice(o, NO_CORRECTIONS, 33, suggestedExtent(o));
  for (let i = 0; i < plain.density.length; i++) {
    assert(
      Math.abs(plain.density[i] - corrected.density[i]) < 1e-15,
      `baseline drift at ${i}`,
    );
  }
});

Deno.test("screening lowers the effective charge, never below one", () => {
  // Slater: a 1s electron in helium sees less than the full +2.
  assert(effectiveNuclearCharge(2, 1, 0, true) < 2, "helium 1s is unscreened");
  assert(effectiveNuclearCharge(2, 1, 0, true) >= 1, "screening drove Zeff below 1");
  assert(effectiveNuclearCharge(1, 1, 0, true) === 1, "hydrogen should be unshielded");
  assert(effectiveNuclearCharge(26, 3, 2, true) < 26, "iron 3d is unscreened");
  // Disabled is the identity.
  assert(effectiveNuclearCharge(26, 3, 2, false) === 26, "disabled screening changed Zeff");
});

Deno.test("relativistic contraction increases Zeff and grows with Z", () => {
  const light = applyRelativisticContraction(2, 1, 0, true) / 2;
  const heavy = applyRelativisticContraction(80, 1, 0, true) / 80;
  assert(heavy > light, "contraction does not grow with Z");
  assert(applyRelativisticContraction(80, 1, 0, false) === 80, "disabled changed Zeff");
});

Deno.test("correlation mixing only offers valid configurations", () => {
  for (const [n, l, m] of [[3, 1, 0], [4, 2, 1], [2, 0, 0], [1, 0, 0]] as const) {
    for (const term of buildCorrelationTerms(n, l, m)) {
      assert(term.l < term.n, `invalid term l=${term.l} n=${term.n}`);
      assert(Math.abs(term.m) <= term.l, `invalid term m=${term.m} l=${term.l}`);
      assert(term.weight > 0, "non-positive weight");
    }
  }
});

Deno.test("corrections change the picture when switched on", () => {
  const o = { n: 3, l: 2, m: 1, z: 26 };
  const plain = correctedDensitySlice(o, NO_CORRECTIONS, 25, suggestedExtent(o));
  const full = correctedDensitySlice(
    o,
    {
      screeningExchange: true,
      relativisticSpinOrbit: true,
      correlationMixing: true,
    },
    25,
    suggestedExtent(o),
  );
  let delta = 0;
  for (let i = 0; i < plain.density.length; i++) {
    delta += Math.abs(plain.density[i] - full.density[i]);
  }
  assert(delta > 1e-3, `corrections had no visible effect (L1 ${delta})`);
  // Still a normalised density.
  let sum = 0;
  for (const v of full.density) sum += v;
  assert(Math.abs(sum - 1) < 1e-12, `corrected density sums to ${sum}`);
});

Deno.test("the RGB encoder produces one pixel per sample", () => {
  const o = { n: 2, l: 1, m: 0, z: 1 };
  const slice = densitySlice(o, 32, suggestedExtent(o));
  const ramp = orbitalRamp(desktopPalette(THEMES[0]));
  const image = densityToRgb(slice, ramp, 32);
  assert(image.width === 32 && image.height === 32, "wrong image size");
  assert(image.data.length === 32 * 32 * 3, `expected RGB24, got ${image.data.length} bytes`);
  // The nodal plane must be the ramp's low end, not an arbitrary colour.
  const mid = 16;
  const offset = (mid * 32 + 4) * 3;
  const [r, g, b] = ramp[0];
  assert(
    image.data[offset] === r && image.data[offset + 1] === g && image.data[offset + 2] === b,
    "nodal plane is not the ramp floor",
  );
});
