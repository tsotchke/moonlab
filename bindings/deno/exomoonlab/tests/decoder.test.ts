/**
 * The QEC decoder window.
 *
 * The valuable assertions are the physics ones, and they are checkable against
 * theory rather than a snapshot: the repetition code's threshold is p = 0.5,
 * so below it a longer code must do better and above it a longer code must do
 * worse. A decoder that silently did nothing would still produce a smooth
 * curve; only the crossing proves it is decoding.
 */

import { openNativeBackend } from "../src/backend/mod.ts";
import { UF_BOUNDARY } from "../src/backend/mod.ts";
import {
  makeRng,
  measurePoint,
  measureThreshold,
  repetitionCodeGraph,
  sampleShots,
} from "../src/app/repetition_code.ts";

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(`assertion failed: ${message}`);
}

Deno.test("the repetition-code graph matches MoonLab's own d=3 fixture", () => {
  // src/qec/uf_decoder.h's test builds exactly this: D0-boundary flipping the
  // observable, D0-D1 flipping nothing, D1-boundary flipping nothing.
  const g = repetitionCodeGraph(3);
  assert(g.numDetectors === 2, `expected 2 detectors, got ${g.numDetectors}`);
  assert(g.edgeA.length === 3, `expected 3 edges, got ${g.edgeA.length}`);
  assert(g.edgeA[0] === 0 && g.edgeB[0] === UF_BOUNDARY, "edge 0 is not D0-boundary");
  assert(g.edgeA[1] === 0 && g.edgeB[1] === 1, "edge 1 is not D0-D1");
  assert(g.edgeA[2] === 1 && g.edgeB[2] === UF_BOUNDARY, "edge 2 is not D1-boundary");
  assert(g.edgeObs[0] === 1n, "edge 0 should flip the observable");
  assert(g.edgeObs[1] === 0n && g.edgeObs[2] === 0n, "only edge 0 should flip it");
});

Deno.test("every graph is well formed across distances", () => {
  for (const d of [2, 3, 5, 9, 17]) {
    const g = repetitionCodeGraph(d);
    assert(g.numDetectors === d - 1, `d=${d}: wrong detector count`);
    assert(g.edgeA.length === d, `d=${d}: wrong edge count`);
    for (let j = 0; j < d; j++) {
      const a = g.edgeA[j];
      assert(a < g.numDetectors, `d=${d}: edge ${j} start ${a} out of range`);
      const b = g.edgeB[j];
      assert(b === UF_BOUNDARY || b < g.numDetectors, `d=${d}: edge ${j} end out of range`);
    }
  }
});

Deno.test("sampling is deterministic and its syndrome matches the errors", () => {
  const g = repetitionCodeGraph(5);
  const a = sampleShots(g, 5, 0.2, 200, makeRng(7));
  const b = sampleShots(g, 5, 0.2, 200, makeRng(7));
  assert(a.truth.every((v, i) => v === b.truth[i]), "same seed gave different truth");
  assert(a.detectors.every((v, i) => v === b.detectors[i]), "same seed gave different syndrome");
  // At p = 0 nothing fires, so nothing lights and nothing flips.
  const clean = sampleShots(g, 5, 0, 50, makeRng(1));
  assert(clean.detectors.every((v) => v === 0), "p=0 lit a detector");
  assert(clean.truth.every((v) => v === 0), "p=0 flipped the logical");
});

Deno.test("a clean syndrome decodes to no correction", async () => {
  const backend = await openNativeBackend();
  try {
    assert(backend.capabilities.decoder, "native backend lacks the decoder");
    const point = await measurePoint(backend, 5, 0, 200, 3);
    assert(point.logical === 0, `p=0 gave a logical error rate of ${point.logical}`);
  } finally {
    await backend.dispose();
  }
});

Deno.test("the threshold is where the distances cross", async () => {
  const backend = await openNativeBackend();
  try {
    // The repetition code's threshold is p = 0.5 exactly.
    const curve = await measureThreshold(backend, [3, 9], [0.25, 0.65], 4000);
    const [short, long] = curve.series;

    // Below threshold the longer code wins...
    assert(
      long[0].logical < short[0].logical,
      `below threshold d=9 (${long[0].logical}) should beat d=3 (${short[0].logical})`,
    );
    // ...and above it, the longer code loses. That inversion is the whole
    // point, and a no-op decoder could not produce it.
    assert(
      long[1].logical > short[1].logical,
      `above threshold d=9 (${long[1].logical}) should be worse than d=3 (${short[1].logical})`,
    );
  } finally {
    await backend.dispose();
  }
});

Deno.test("decoding beats doing nothing, below threshold", async () => {
  const backend = await openNativeBackend();
  try {
    // With no correction the logical error is just the parity of the single
    // observable-flipping edge, i.e. p itself. Decoding must do better.
    const p = 0.15;
    const point = await measurePoint(backend, 9, p, 4000, 11);
    assert(
      point.logical < p / 2,
      `decoded rate ${point.logical} is no better than leaving the errors alone (${p})`,
    );
  } finally {
    await backend.dispose();
  }
});
