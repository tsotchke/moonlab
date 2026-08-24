/**
 * Native vs WASM equivalence.
 *
 * The claim this file exists to check: because both backends call the same
 * `MOONLAB_API` C functions, they are not merely similar, they are the same
 * computation reached two ways. exomoonlab being one application instead of
 * two rests on that, so it is tested rather than assumed.
 *
 * Tolerance. Both paths here are the float64 CPU implementation, so agreement
 * should be near-exact; 1e-12 leaves room for compiler reassociation between
 * an -Ofast native build and Emscripten's, and nothing more. This is NOT the
 * looser complex64 tolerance that governs the WebGPU path -- that is a
 * different comparison against a different reference.
 */

/**
 * Local assertions rather than jsr:@std/assert: this test must run in CI
 * without network access, and two helpers are not worth a remote dependency.
 */
function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(`assertion failed: ${message}`);
}

function assertAlmostEquals(
  actual: number,
  expected: number,
  tolerance: number,
  what: string,
): void {
  const delta = Math.abs(actual - expected);
  if (!(delta <= tolerance)) {
    throw new Error(
      `${what}: wasm=${actual} native=${expected} differ by ${delta.toExponential(3)} ` +
        `(tolerance ${tolerance.toExponential(3)})`,
    );
  }
}
import type { MoonLabBackend, StateHandle } from "../src/backend/mod.ts";
import { openNativeBackend, openWasmBackend } from "../src/backend/mod.ts";

const TOLERANCE = 1e-12;

/** A circuit as a sequence of ops, applied to a fresh state. */
interface Circuit {
  readonly name: string;
  readonly qubits: number;
  readonly apply: (b: MoonLabBackend, s: StateHandle) => Promise<void>;
}

const CIRCUITS: Circuit[] = [
  {
    name: "ground state |000>",
    qubits: 3,
    apply: () => Promise.resolve(),
  },
  {
    name: "X on qubit 0",
    qubits: 3,
    apply: (b, s) => b.pauliX(s, 0),
  },
  {
    name: "uniform superposition (H on all 3)",
    qubits: 3,
    apply: async (b, s) => {
      for (let q = 0; q < 3; q++) await b.hadamard(s, q);
    },
  },
  {
    name: "Bell pair (H0, CNOT 0->1)",
    qubits: 2,
    apply: async (b, s) => {
      await b.hadamard(s, 0);
      await b.cnot(s, 0, 1);
    },
  },
  {
    name: "GHZ over 4 qubits",
    qubits: 4,
    apply: async (b, s) => {
      await b.hadamard(s, 0);
      for (let q = 1; q < 4; q++) await b.cnot(s, 0, q);
    },
  },
  {
    name: "Z is a no-op on probabilities",
    qubits: 2,
    apply: async (b, s) => {
      await b.hadamard(s, 0);
      await b.pauliZ(s, 0);
    },
  },
  {
    name: "entangling ladder, 5 qubits",
    qubits: 5,
    apply: async (b, s) => {
      for (let q = 0; q < 5; q++) await b.hadamard(s, q);
      for (let q = 0; q + 1 < 5; q++) await b.cnot(s, q, q + 1);
      await b.pauliZ(s, 2);
      await b.normalize(s);
    },
  },
];

/** Everything we can read back without touching the struct layout. */
interface Readout {
  readonly probabilities: number[];
  readonly probabilityOne: number[];
  readonly entropy: number;
  readonly purity: number;
}

async function run(backend: MoonLabBackend, circuit: Circuit): Promise<Readout> {
  const state = await backend.createState(circuit.qubits);
  try {
    await circuit.apply(backend, state);
    const probabilities: number[] = [];
    for (let i = 0; i < state.stateDim; i++) {
      probabilities.push(await backend.probability(state, i));
    }
    const probabilityOne: number[] = [];
    for (let q = 0; q < circuit.qubits; q++) {
      probabilityOne.push(await backend.probabilityOne(state, q));
    }
    return {
      probabilities,
      probabilityOne,
      entropy: await backend.entropy(state),
      purity: await backend.purity(state),
    };
  } finally {
    await backend.destroyState(state);
  }
}

async function tryOpen(
  open: () => Promise<MoonLabBackend>,
): Promise<MoonLabBackend | { unavailable: string }> {
  try {
    return await open();
  } catch (cause) {
    return { unavailable: cause instanceof Error ? cause.message : String(cause) };
  }
}

Deno.test("native and WASM backends agree", async (t) => {
  const native = await tryOpen(openNativeBackend);
  const wasm = await tryOpen(openWasmBackend);

  // A missing backend is reported, never quietly passed over: an equivalence
  // suite that skipped half of itself and still said "ok" would be worse than
  // no suite at all.
  if ("unavailable" in native || "unavailable" in wasm) {
    const missing = [
      "unavailable" in native ? `native: ${native.unavailable}` : null,
      "unavailable" in wasm ? `wasm: ${wasm.unavailable}` : null,
    ].filter(Boolean).join("\n\n");
    throw new Error(
      `equivalence needs both backends; could not open:\n\n${missing}`,
    );
  }

  await t.step("provenance", () => {
    console.log(`    native: ${native.description}`);
    console.log(`    wasm:   ${wasm.description}`);
    console.log(
      `    wasm exports: ${wasm.capabilities.exportedFunctions}, ` +
        `allocating ctor: native=${native.capabilities.allocatingConstructor} ` +
        `wasm=${wasm.capabilities.allocatingConstructor}`,
    );
  });

  let worst = 0;
  for (const circuit of CIRCUITS) {
    await t.step(circuit.name, async () => {
      const a = await run(native, circuit);
      const b = await run(wasm, circuit);

      assert(
        a.probabilities.length === b.probabilities.length,
        `state dimension differs: ${a.probabilities.length} vs ${b.probabilities.length}`,
      );

      for (let i = 0; i < a.probabilities.length; i++) {
        worst = Math.max(worst, Math.abs(a.probabilities[i] - b.probabilities[i]));
        assertAlmostEquals(
          b.probabilities[i],
          a.probabilities[i],
          TOLERANCE,
          `P(|${i.toString(2).padStart(circuit.qubits, "0")}>)`,
        );
      }
      for (let q = 0; q < a.probabilityOne.length; q++) {
        worst = Math.max(worst, Math.abs(a.probabilityOne[q] - b.probabilityOne[q]));
        assertAlmostEquals(b.probabilityOne[q], a.probabilityOne[q], TOLERANCE, `P(q${q}=1)`);
      }
      worst = Math.max(worst, Math.abs(a.entropy - b.entropy));
      worst = Math.max(worst, Math.abs(a.purity - b.purity));
      assertAlmostEquals(b.entropy, a.entropy, TOLERANCE, "entropy");
      assertAlmostEquals(b.purity, a.purity, TOLERANCE, "purity");

      // Probabilities must also be internally consistent, or "they agree"
      // could just mean they are wrong in the same way.
      const total = a.probabilities.reduce((sum, p) => sum + p, 0);
      assertAlmostEquals(total, 1, 1e-9, "native probabilities sum to 1");
    });
  }

  await t.step("summary", () => {
    console.log(`    largest deviation across all circuits: ${worst.toExponential(3)}`);
  });

  await native.dispose();
  await wasm.dispose();
});
