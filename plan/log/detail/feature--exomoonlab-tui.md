# Detailed development log: `feature/exomoonlab-tui`

- **Branch:** `feature/exomoonlab-tui`
- **Started:** 2026-08-24
- **Base:** `exomoonlab`
- **Building:** [exomoonlab console interface](../../todo/exomoonlab-console-interface.md)

Implementation of the console planned on `exomoonlab`. The plan, the
architecture, and the phase list live there and are not restated here; this log
records what was actually attempted on this branch.

Code lands in `bindings/deno/exomoonlab`.

## Starting position

Phase 1 — the backend seam and the equivalence harness — has not been started.
Nothing under `bindings/deno/` exists yet.

Two facts from the `exomoonlab` survey shape the first move:

- exotui's `ShellPresenter` seam already runs one application under both a
  console and a browser host, so this branch writes one application, not two.
- MoonLab's stable C ABI is re-exported to WASM at 424 symbols, so both
  backends call the same C functions. The equivalence harness exists to prove
  that claim rather than assume it.

The known blocker for the browser half is unchanged: building `moonlab.wasm`
needs the Emscripten SDK, and `emcc` is not installed on this machine.

## 2026-08-24 — Phase 1: the backend seam, and Emscripten turning out not to matter

### Prompt

Go ahead with the JSR pin, that should have everything we need. If we need more
stuff we should develop that in a dedicated branch and push a new release.

### Response and strategy

**The blocker wasn't one.** The plan said phase 1's browser half needed the
Emscripten SDK. Checking before building found three prebuilt WASM artifacts
already tracked in the repo; the newest
(`bindings/javascript/demo/public/moonlab.js`, from "Export the v1.2 C surfaces
from the WASM build") exports 447 functions and covers everything the seam
needs. So both backends ran on day one and the SDK is only needed for a *fresh*
artifact. Worth remembering as a habit: check what the repo already ships
before accepting a stated prerequisite.

**Verifying symbols needs the right tool.** An initial `strings | grep -qx`
check reported `gate_cnot` missing from the WASM build. It is present —
`strings` packs name-section entries with prefixes, so exact-line matching
produces false negatives. Enumerating the export table via
`WebAssembly.Module.exports()` (no instantiation required) is definitive and is
what settled it. The false negative would have sent me building a CNOT
workaround for a function that was there all along.

**Loading the glue.** `createRequire()` returned an empty object. The glue is a
`-sMODULARIZE` UMD file whose CommonJS branch is guarded on
`typeof module === 'object'`, and the artifacts sit inside packages marked
`"type": "module"`, so Deno resolves them as ESM and that branch never runs.
Evaluating the source with a synthetic `module`/`exports` works and, unlike a
`.cjs` rename or a package.json tweak, stays correct wherever the artifact
lives — which matters, since the backend also targets `packages/core/dist`.

**Struct layout deliberately not shared.** `quantum_state_t` has 64-bit
`size_t` fields natively and 32-bit ones under wasm32, so size and offsets
differ by target. Native sidesteps it entirely via `quantum_state_create`; WASM
over-allocates with upstream's own 256-byte constant. No offset arithmetic
exists in shared code, which is why upstream's JS binding using
`AMPLITUDES_OFFSET = 8` (a wasm32 value) is not a contradiction.

**Calling convention.** `uint64_t` reaches the WASM build as a BigInt
(WASM_BIGINT is on); passing a Number throws. Probed rather than assumed.

**Result.** Seven circuits — ground state, single X, uniform superposition,
Bell pair, 4-qubit GHZ, a Z phase that must not move probabilities, and a
5-qubit entangling ladder — agree between backends to **1.1e-16**. The harness
also asserts probabilities sum to 1, so agreement cannot mean "both wrong
identically", and it fails loudly rather than skipping when a backend is
absent. Wired into ctest as `deno_bindings_equivalence`; assertions inlined
instead of pulling `jsr:@std/assert` so it runs hermetically under
`--no-remote`.

exotui is pinned to `jsr:@ubernaut/exotui@^0.6.0` in `deno.jsonc` with a
documented `import_map.local.json` override, but nothing imports it yet — no
lockfile entry until phase 2 does. Per the standing instruction: anything
exotui lacks gets developed on a branch in the exotui repo and released, never
vendored or reached into via `src/`.

**Next.** Phase 2 — one window (amplitudes/probabilities) composed through
`ShellPresenter`, running under both the console and web presenters, with
backend calls off the frame loop.
