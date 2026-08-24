/**
 * Headless rendering.
 *
 * The app composes cells, so it can be driven and inspected with no terminal
 * and no browser -- which is the useful consequence of the presenter seam, and
 * the reason these assertions can run in CI.
 */

import { MoonLabApp } from "../src/app/console_app.ts";
import { Surface } from "../src/ui/cells.ts";
import { bar, basisLabel } from "../src/ui/probabilities.ts";
import { openNativeBackend } from "../src/backend/mod.ts";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(`assertion failed: ${message}`);
}

const SIZE = { columns: 96, rows: 26 };

/**
 * Renders frames until the pending job settles.
 *
 * Waiting on the absence of "computing…" is not enough: once a result exists
 * the window deliberately keeps showing it while the next one computes, so
 * that string never returns and a stale frame would be mistaken for a fresh
 * one. The subtitle's status field is the honest signal.
 */
async function settle(app: MoonLabApp, frames = 400): Promise<string> {
  let text = "";
  for (let i = 0; i < frames; i++) {
    text = renderToText(app);
    if (!text.includes("computing…") && !text.includes("· running")) return text;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error(`job did not settle within ${frames} frames:\n${text}`);
}

function renderToText(app: MoonLabApp): string {
  const frame = app.frame(0, SIZE);
  return frame.map((row) => row.map((cell) => cell.char).join("").trimEnd()).join("\n");
}

Deno.test("bar renders proportionally and clamps", () => {
  assert(bar(0, 10) === "", "zero is empty");
  assert(bar(1, 10) === "█".repeat(10), "one fills the width");
  assert(bar(2, 10) === "█".repeat(10), "above one clamps");
  assert(bar(0.5, 10).length > 0 && bar(0.5, 10).length <= 10, "half fits");
  // A tiny probability must still be visible, not rounded away to nothing.
  assert(bar(0.02, 10) !== "", "small fractions still paint");
});

Deno.test("basisLabel pads to the register width", () => {
  assert(basisLabel(0, 3) === "|000⟩", `got ${basisLabel(0, 3)}`);
  assert(basisLabel(5, 3) === "|101⟩", `got ${basisLabel(5, 3)}`);
});

Deno.test("Surface clips writes at its edges", () => {
  const s = new Surface(6, 2);
  s.write(4, 0, "abcdef");
  assert(s.toText().split("\n")[0] === "    ab", `got ${JSON.stringify(s.toText())}`);
  s.write(0, 99, "offgrid");
  s.write(-5, 0, "offgrid");
  assert(s.toText().split("\n").length === 2, "no rows appeared");
});

Deno.test("app renders a Bell pair headlessly", async () => {
  const backend = await openNativeBackend();
  const app = new MoonLabApp({ backend });
  try {
    await app.init();
    const text = await settle(app);

    assert(text.includes("Bell pair"), `title missing:\n${text}`);
    assert(text.includes("native backend"), `backend line missing:\n${text}`);
    assert(text.includes("|00⟩"), `|00> row missing:\n${text}`);
    assert(text.includes("|11⟩"), `|11> row missing:\n${text}`);
    // The two Bell terms carry the whole distribution, at half each.
    const halves = [...text.matchAll(/50\.000%/g)].length;
    assert(halves === 2, `expected two 50.000% rows, saw ${halves}:\n${text}`);
    assert(text.includes("purity 1.000000"), `purity missing:\n${text}`);
    // Regression: the two 0.5s sum to a hair over 1 in floating point, which
    // rendered the remainder as "-0.000% of the mass" before it was clamped.
    assert(!text.includes("-0.000%"), `negative residual mass:\n${text}`);
    assert(
      text.includes("none with measurable probability"),
      `zero-probability states described wrongly:\n${text}`,
    );
    assert(text.includes("j/k circuit"), `help line missing:\n${text}`);
  } finally {
    await backend.dispose();
  }
});

Deno.test("keys switch circuit and register size without blocking", async () => {
  const backend = await openNativeBackend();
  const app = new MoonLabApp({ backend });
  try {
    await app.init();
    await settle(app);

    app.key({ key: "j" });
    // The frame immediately after a key must still render -- never throw and
    // never wait on the job it just started.
    const during = renderToText(app);
    assert(during.length > 0, "frame during a pending job rendered");

    const after = await settle(app);
    assert(!after.includes("Bell pair"), `circuit did not change:\n${after}`);

    app.key({ key: "+" });
    const grown = await settle(app);
    assert(/\d+ qubits/.test(grown), `qubit count missing:\n${grown}`);
  } finally {
    await backend.dispose();
  }
});

Deno.test("a large register stays renderable via a bounded scan", async () => {
  const backend = await openNativeBackend();
  // 16 qubits is 65536 amplitudes against a 512-state scan budget.
  const app = new MoonLabApp({ backend, scanLimit: 512 });
  try {
    await app.init();
    await settle(app);
    for (let i = 0; i < 14; i++) app.key({ key: "+" });
    const text = await settle(app, 600);

    assert(text.includes("scan bounded"), `bounded scan not disclosed:\n${text}`);
    assert(!text.includes("of the mass"), `claimed mass coverage from a partial scan:\n${text}`);
  } finally {
    await backend.dispose();
  }
});
