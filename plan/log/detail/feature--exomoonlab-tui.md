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

## 2026-08-24 — Phase 2: a window in a terminal, and the exotui gap

### Prompt

Proceed.

### Response and strategy

**Checked the published surface before designing against it.** The plan assumed
`ShellPresenter` and `runShellApp` were importable. At 0.6.0 they are not — they
live in `src/app/shell_presenter.ts`, and none of the fifteen published
entrypoints re-export them. What *is* public, via `@ubernaut/exotui/runtime`, is
`consolePresenter` and `runConsoleShellApp`, because `src/runtime/mod.ts`
re-exports the console presenter and `mod.runtime.ts` re-exports all of that.
So the terminal host works with the pin and the browser host does not.

Verified against JSR itself rather than the local checkout — the working copy is
on a feature branch ahead of the tag, and what matters is what was published.

**Structural typing closed the gap without touching `src/`.** The frame is just
`ReadonlyArray<ReadonlyArray<{char, foreground?, background?, bold?}>>` with
`ShellRgb = readonly [number, number, number]`. Mirroring those shapes in
`src/ui/cells.ts` lets `runConsoleShellApp` accept our frames unchanged;
`deno check main.ts` resolves the real package and confirms it. When exotui
publishes the seam, that file becomes re-exports and nothing else moves.

**The frame loop discipline is the design, so it is enforced by types.** Every
`MoonLabBackend` method is async, so nothing can accidentally be awaited inside
the synchronous `frame()`. Work goes through a `Job` that keeps the last good
value while the next computes, and abandons a superseded run rather than
queueing it.

**Bounded scanning.** Reading every basis probability is one call per amplitude,
so 20 qubits is a million round trips. The readout stops at a scan limit and the
window says the scan was bounded, rather than claiming a share "of the mass" it
did not measure.

**A pty run caught what headless tests missed.** Rendering under
`script -qec "stty rows 30 cols 100; …"` showed `-0.000% of the mass`: two 0.5
probabilities sum to a hair over 1, so the residual went negative. Headless
assertions had checked the numbers, not the sentence. Clamped, zero-probability
states now say "none with measurable probability", and there is a regression
assertion. Worth remembering that a real terminal at a real size renders things
no unit test was looking at.

A second self-inflicted one: the first `settle()` helper waited for "computing…"
to disappear, but the window deliberately keeps showing the previous readout
while recomputing, so that string never returns and stale frames passed as
fresh. It now waits on the job's own status.

**Next.** The browser half and all of phase 3 need the same upstream change:
exotui exporting its web presenter, seam types, shell painters, theme catalog,
backgrounds, and window host. That is a branch and a release in the exotui
repository, not something to work around here.
