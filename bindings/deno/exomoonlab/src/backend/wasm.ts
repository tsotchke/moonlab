/**
 * WASM backend: the Emscripten build of the same C core.
 *
 * Loading note. The Emscripten glue is a `-sMODULARIZE` UMD file whose
 * CommonJS branch is guarded by `typeof module === 'object'`. The artifacts
 * live inside packages marked `"type": "module"`, so Deno resolves the `.js`
 * as ESM, that branch never runs, and `createRequire()` yields an empty
 * object. Evaluating the source with a synthetic `module`/`exports` avoids
 * depending on where the artifact sits or what its package.json claims.
 *
 * Allocation note. The prebuilt artifacts predate ULG's `quantum_state_create`
 * export, so the struct is allocated here with `_malloc` and initialised with
 * `_quantum_state_init`. The size is upstream's own constant from
 * `packages/core/src/quantum-state.ts`: a deliberate over-allocation, because
 * the real layout differs between wasm32 and a 64-bit host and must not be
 * hardcoded per-target. When a build exporting `quantum_state_create` is
 * available this backend prefers it and the constant stops mattering.
 */

import { dirname } from "node:path";
import { createRequire } from "node:module";
import {
  type BackendCapabilities,
  BackendUnavailableError,
  type MoonLabBackend,
  type StateHandle,
} from "./types.ts";

const QS_SUCCESS = 0;

/** Upstream's over-allocation for quantum_state_t; see the note above. */
const STATE_STRUCT_SIZE = 256;

type WasmModule = Record<string, (...args: never[]) => unknown>;

/** Candidate glue files, freshest build first. */
export function wasmGlueCandidates(): string[] {
  const explicit = Deno.env.get("MOONLAB_WASM");
  const repoRoot = new URL("../../../../../", import.meta.url).pathname;
  return [
    ...(explicit ? [explicit] : []),
    // A local `pnpm build:wasm` lands here and is newest.
    `${repoRoot}bindings/javascript/packages/core/dist/moonlab.js`,
    // Tracked prebuilt, currently the only one that exists without emcc.
    `${repoRoot}bindings/javascript/demo/public/moonlab.js`,
    `${repoRoot}docs/moonlab.js`,
  ];
}

async function instantiate(gluePath: string): Promise<WasmModule> {
  const source = await Deno.readTextFile(gluePath);
  const here = dirname(gluePath);
  const mod: { exports: Record<string, unknown> } = { exports: {} };
  const require = createRequire(gluePath);
  new Function("module", "exports", "require", "__dirname", "__filename", source)(
    mod,
    mod.exports,
    require,
    here,
    gluePath,
  );
  const factory = (mod.exports.default ?? mod.exports) as
    | ((options: unknown) => Promise<WasmModule>)
    | undefined;
  if (typeof factory !== "function") {
    throw new Error(`${gluePath} did not export a MODULARIZE factory`);
  }
  return await factory({ locateFile: (file: string) => `${here}/${file}` });
}

interface WasmState extends StateHandle {
  readonly ptr: number;
  /** True when allocated by us and needing an explicit _free. */
  readonly manual: boolean;
}

function asWasm(state: StateHandle): WasmState {
  const w = state as WasmState;
  if (typeof w.ptr !== "number") {
    throw new TypeError("state handle did not come from the WASM backend");
  }
  return w;
}

export async function openWasmBackend(): Promise<MoonLabBackend> {
  const tried: string[] = [];
  let m: WasmModule | undefined;
  let path = "";
  for (const candidate of wasmGlueCandidates()) {
    try {
      m = await instantiate(candidate);
      path = candidate;
      break;
    } catch (cause) {
      tried.push(`${candidate}: ${cause instanceof Error ? cause.message : cause}`);
    }
  }
  if (!m) {
    throw new BackendUnavailableError(
      "wasm",
      `no loadable Emscripten build. Tried:\n  ${tried.join("\n  ")}\n` +
        "Build one with: cd bindings/javascript/packages/core && pnpm build:wasm (needs emcc)",
    );
  }

  const fn = <T>(name: string): T => {
    const f = m![name];
    if (typeof f !== "function") throw new Error(`${path} does not export ${name}`);
    return f as T;
  };

  const malloc = fn<(n: number) => number>("_malloc");
  const free = fn<(p: number) => void>("_free");
  const stateInit = fn<(p: number, n: number) => number>("_quantum_state_init");
  const stateFree = fn<(p: number) => void>("_quantum_state_free");
  const stateReset = fn<(p: number) => void>("_quantum_state_reset");
  const stateNormalize = fn<(p: number) => number>("_quantum_state_normalize");
  const hadamard = fn<(p: number, q: number) => number>("_gate_hadamard");
  const pauliX = fn<(p: number, q: number) => number>("_gate_pauli_x");
  const pauliZ = fn<(p: number, q: number) => number>("_gate_pauli_z");
  const cnot = fn<(p: number, c: number, t: number) => number>("_gate_cnot");
  // uint64_t reaches wasm as a BigInt: this build has WASM_BIGINT enabled.
  const getProbability = fn<(p: number, i: bigint) => number>("_quantum_state_get_probability");
  const probabilityOne = fn<(p: number, q: number) => number>("_measurement_probability_one");
  const entropy = fn<(p: number) => number>("_quantum_state_entropy");
  const purity = fn<(p: number) => number>("_quantum_state_purity");

  const createFn = m["_quantum_state_create"] as ((n: number) => number) | undefined;
  const destroyFn = m["_quantum_state_destroy"] as ((p: number) => void) | undefined;
  const hasAllocatingCtor = typeof createFn === "function" && typeof destroyFn === "function";

  const capabilities: BackendCapabilities = {
    // wasm32 memory ceiling bites long before the native one does.
    maxQubits: 24,
    allocatingConstructor: hasAllocatingCtor,
    exportedFunctions: Object.keys(m).filter((k) => k.startsWith("_")).length,
  };

  const check = (code: number, what: string): void => {
    if (code !== QS_SUCCESS) throw new Error(`${what} failed with qs_error_t ${code}`);
  };

  return {
    kind: "wasm",
    description: `MoonLab WASM via ${path}${hasAllocatingCtor ? "" : " (no allocating ctor)"}`,
    capabilities,

    createState(numQubits: number): Promise<StateHandle> {
      if (!Number.isInteger(numQubits) || numQubits < 1) {
        return Promise.reject(new RangeError("numQubits must be a positive integer"));
      }
      if (numQubits > capabilities.maxQubits) {
        return Promise.reject(
          new RangeError(`${numQubits} qubits exceeds maxQubits ${capabilities.maxQubits}`),
        );
      }
      if (hasAllocatingCtor) {
        const ptr = createFn!(numQubits);
        if (ptr === 0) {
          return Promise.reject(new Error(`quantum_state_create(${numQubits}) returned NULL`));
        }
        return Promise.resolve({ numQubits, stateDim: 2 ** numQubits, ptr, manual: false });
      }
      const ptr = malloc(STATE_STRUCT_SIZE);
      if (ptr === 0) return Promise.reject(new Error("malloc for quantum_state_t failed"));
      const rc = stateInit(ptr, numQubits);
      if (rc !== QS_SUCCESS) {
        free(ptr);
        return Promise.reject(new Error(`quantum_state_init failed with qs_error_t ${rc}`));
      }
      return Promise.resolve({ numQubits, stateDim: 2 ** numQubits, ptr, manual: true });
    },

    destroyState(state: StateHandle): Promise<void> {
      const w = asWasm(state);
      if (w.manual) {
        stateFree(w.ptr);
        free(w.ptr);
      } else {
        destroyFn!(w.ptr);
      }
      return Promise.resolve();
    },

    reset(state) {
      stateReset(asWasm(state).ptr);
      return Promise.resolve();
    },
    normalize(state) {
      check(stateNormalize(asWasm(state).ptr), "quantum_state_normalize");
      return Promise.resolve();
    },
    hadamard(state, qubit) {
      check(hadamard(asWasm(state).ptr, qubit), "gate_hadamard");
      return Promise.resolve();
    },
    pauliX(state, qubit) {
      check(pauliX(asWasm(state).ptr, qubit), "gate_pauli_x");
      return Promise.resolve();
    },
    pauliZ(state, qubit) {
      check(pauliZ(asWasm(state).ptr, qubit), "gate_pauli_z");
      return Promise.resolve();
    },
    cnot(state, control, target) {
      check(cnot(asWasm(state).ptr, control, target), "gate_cnot");
      return Promise.resolve();
    },
    probability(state, basisIndex) {
      return Promise.resolve(getProbability(asWasm(state).ptr, BigInt(basisIndex)));
    },
    probabilityOne(state, qubit) {
      return Promise.resolve(probabilityOne(asWasm(state).ptr, qubit));
    },
    entropy(state) {
      return Promise.resolve(entropy(asWasm(state).ptr));
    },
    purity(state) {
      return Promise.resolve(purity(asWasm(state).ptr));
    },

    dispose(): Promise<void> {
      // Nothing to release: Emscripten owns its heap for the life of the
      // process, so unlike the native backend there is no library handle to
      // close. Individual states are still freed by destroyState().
      return Promise.resolve();
    },
  } satisfies MoonLabBackend;
}
