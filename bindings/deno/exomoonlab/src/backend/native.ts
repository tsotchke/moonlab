/**
 * Native backend: Deno FFI over libquantumsim.
 *
 * Binds the `MOONLAB_API` symbols directly. This is the faster of the two
 * implementations and the one with the full C surface available, including
 * `quantum_state_create`, which allocates the state struct for us -- so
 * nothing here needs to know the struct's layout. That matters: the layout
 * differs between native (64-bit `size_t`) and wasm32, and hardcoding it in
 * shared code would be a bug waiting for a 32-bit host.
 */

import {
  type BackendCapabilities,
  BackendUnavailableError,
  type MoonLabBackend,
  type StateHandle,
} from "./types.ts";

const QS_SUCCESS = 0;

const SYMBOLS = {
  quantum_state_create: { parameters: ["i32"], result: "pointer" },
  quantum_state_destroy: { parameters: ["pointer"], result: "void" },
  quantum_state_reset: { parameters: ["pointer"], result: "void" },
  quantum_state_normalize: { parameters: ["pointer"], result: "i32" },
  gate_hadamard: { parameters: ["pointer", "i32"], result: "i32" },
  gate_pauli_x: { parameters: ["pointer", "i32"], result: "i32" },
  gate_pauli_z: { parameters: ["pointer", "i32"], result: "i32" },
  gate_cnot: { parameters: ["pointer", "i32", "i32"], result: "i32" },
  quantum_state_get_probability: { parameters: ["pointer", "u64"], result: "f64" },
  measurement_probability_one: { parameters: ["pointer", "i32"], result: "f64" },
  quantum_state_entropy: { parameters: ["pointer"], result: "f64" },
  quantum_state_purity: { parameters: ["pointer"], result: "f64" },
} as const;

/** Where to look for the shared library, most explicit first. */
export function nativeLibraryCandidates(): string[] {
  const explicit = Deno.env.get("MOONLAB_LIB");
  const candidates = explicit ? [explicit] : [];

  // Walk up from this module looking for a built tree. The console lives at
  // bindings/deno/exomoonlab, so the repo root is three levels up.
  const here = new URL(".", import.meta.url).pathname;
  const repoRoot = new URL("../../../../../", import.meta.url).pathname;
  const suffix = Deno.build.os === "darwin" ? "dylib" : "so";
  for (const base of [repoRoot, here]) {
    candidates.push(`${base}build/libquantumsim.${suffix}`);
    candidates.push(`${base}libquantumsim.${suffix}`);
  }
  // Let the platform loader try, for an installed copy.
  candidates.push(`libquantumsim.${suffix}`);
  return candidates;
}

interface NativeState extends StateHandle {
  readonly ptr: Deno.PointerValue;
}

function asNative(state: StateHandle): NativeState {
  const native = state as NativeState;
  if (native.ptr === undefined) {
    throw new TypeError("state handle did not come from the native backend");
  }
  return native;
}

export async function openNativeBackend(): Promise<MoonLabBackend> {
  if (typeof Deno.dlopen !== "function") {
    throw new BackendUnavailableError("native", "Deno.dlopen is not available");
  }

  const tried: string[] = [];
  let lib: Deno.DynamicLibrary<typeof SYMBOLS> | undefined;
  let path = "";
  for (const candidate of nativeLibraryCandidates()) {
    try {
      lib = Deno.dlopen(candidate, SYMBOLS);
      path = candidate;
      break;
    } catch (cause) {
      tried.push(`${candidate}: ${cause instanceof Error ? cause.message : cause}`);
    }
  }
  if (!lib) {
    throw new BackendUnavailableError(
      "native",
      `no loadable libquantumsim. Tried:\n  ${tried.join("\n  ")}\n` +
        "Build it with: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build",
    );
  }

  const fns = lib.symbols;
  const capabilities: BackendCapabilities = {
    // The state vector is 2^n complex doubles; 30 qubits is already 16 GiB.
    // This is a guard against obvious mistakes, not a hardware measurement.
    maxQubits: 28,
    allocatingConstructor: true,
  };

  const check = (code: number, what: string): void => {
    if (code !== QS_SUCCESS) throw new Error(`${what} failed with qs_error_t ${code}`);
  };

  let disposed = false;

  return await Promise.resolve(
    {
      kind: "native",
      description: `libquantumsim via Deno FFI (${path})`,
      capabilities,

      createState(numQubits: number): Promise<StateHandle> {
        if (!Number.isInteger(numQubits) || numQubits < 1) {
          return Promise.reject(new RangeError(`numQubits must be a positive integer`));
        }
        if (numQubits > capabilities.maxQubits) {
          return Promise.reject(
            new RangeError(`${numQubits} qubits exceeds maxQubits ${capabilities.maxQubits}`),
          );
        }
        const ptr = fns.quantum_state_create(numQubits);
        if (ptr === null) {
          return Promise.reject(new Error(`quantum_state_create(${numQubits}) returned NULL`));
        }
        return Promise.resolve({ numQubits, stateDim: 2 ** numQubits, ptr } as NativeState);
      },

      destroyState(state: StateHandle): Promise<void> {
        fns.quantum_state_destroy(asNative(state).ptr);
        return Promise.resolve();
      },

      reset(state: StateHandle): Promise<void> {
        fns.quantum_state_reset(asNative(state).ptr);
        return Promise.resolve();
      },

      normalize(state: StateHandle): Promise<void> {
        check(fns.quantum_state_normalize(asNative(state).ptr), "quantum_state_normalize");
        return Promise.resolve();
      },

      hadamard(state: StateHandle, qubit: number): Promise<void> {
        check(fns.gate_hadamard(asNative(state).ptr, qubit), "gate_hadamard");
        return Promise.resolve();
      },

      pauliX(state: StateHandle, qubit: number): Promise<void> {
        check(fns.gate_pauli_x(asNative(state).ptr, qubit), "gate_pauli_x");
        return Promise.resolve();
      },

      pauliZ(state: StateHandle, qubit: number): Promise<void> {
        check(fns.gate_pauli_z(asNative(state).ptr, qubit), "gate_pauli_z");
        return Promise.resolve();
      },

      cnot(state: StateHandle, control: number, target: number): Promise<void> {
        check(fns.gate_cnot(asNative(state).ptr, control, target), "gate_cnot");
        return Promise.resolve();
      },

      probability(state: StateHandle, basisIndex: number): Promise<number> {
        return Promise.resolve(
          fns.quantum_state_get_probability(asNative(state).ptr, BigInt(basisIndex)),
        );
      },

      probabilityOne(state: StateHandle, qubit: number): Promise<number> {
        return Promise.resolve(fns.measurement_probability_one(asNative(state).ptr, qubit));
      },

      entropy(state: StateHandle): Promise<number> {
        return Promise.resolve(fns.quantum_state_entropy(asNative(state).ptr));
      },

      purity(state: StateHandle): Promise<number> {
        return Promise.resolve(fns.quantum_state_purity(asNative(state).ptr));
      },

      dispose(): Promise<void> {
        if (!disposed) {
          disposed = true;
          lib.close();
        }
        return Promise.resolve();
      },
    } satisfies MoonLabBackend,
  );
}
