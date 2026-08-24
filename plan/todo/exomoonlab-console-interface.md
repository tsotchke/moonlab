# exomoonlab console interface

## Outcome

A themable terminal-desktop console for MoonLab, built on exotui, that runs the
same application two ways: in a terminal against the native library through
Deno FFI, and in a browser against the WASM build. Windows, themes, and
backgrounds come from exotui; the quantum work comes from MoonLab's stable C
ABI; the console itself owns neither.

Architecture and the reasoning behind it: [`../arch/exomoonlab.md`](../arch/exomoonlab.md).

## Context

Three things already exist and should be used rather than reinvented:

- **exotui** (`/home/cos/projects/exotui`) — the presenter seam, the window
  host, seventeen themes, ten animated backgrounds, and exowebtui
  (`examples/web/desktop_app.ts`) as a working dual-host desktop to start from.
- **`bindings/rust/moonlab-tui`** — the existing Ratatui console. Its views
  (dashboard, circuit, amplitudes, Bloch, entropy, Feynman) and modes
  (algorithm browser, run, step-through, free exploration) are the proven
  answer to "what should a MoonLab TUI show".
- **`bindings/javascript/demo`** — the Pages site. Playground, Examples,
  Gallery, Schrödinger orbitals, Topology, and a Web Worker client
  (`src/workers/moonlabClient.ts`) that is the model for the browser backend.

## Phases

Each phase ends somewhere useful. Do not start the next until the current one
runs.

### 1. Backend seam and equivalence harness — **done (2026-08-24)**

Landed in `bindings/deno/exomoonlab`. `MoonLabBackend` covers state lifecycle,
the gates needed for a Bell pair, and the read-outs the harness compares; it is
deliberately narrow and grows when a window needs something.

- [x] Both backends satisfy the same interface and report capabilities honestly.
- [x] An equivalence test runs a fixed circuit set through both and asserts
      agreement. Seven circuits; largest deviation **1.1e-16**, machine epsilon.
- [x] Wired into ctest as `deno_bindings_equivalence`, gated on `deno` like its
      pnpm and cargo siblings.
- [x] No symbol needed adding to `exports.txt`; the existing surface sufficed.

Two things the plan got wrong, corrected here:

- **Emscripten is not required.** Three prebuilt WASM artifacts are already
  tracked in the repository; the newest
  (`bindings/javascript/demo/public/moonlab.js`, 447 exported functions) covers
  everything phase 1 needs. The SDK is only needed to build a *fresh* artifact.
- **The tolerance is tighter than written.** Both backends here are the float64
  CPU path, so they agree to machine epsilon. The looser complex64 tolerance
  governs the WebGPU comparison, which is a different check against a different
  reference.

Carried forward: the prebuilt predates ULG's `quantum_state_create` export, so
the WASM backend reports `allocatingConstructor: false` and allocates via
`malloc` + `quantum_state_init`. It takes the direct path automatically once a
build carrying that export exists.

### 2. One window, both hosts

Get a single MoonLab window — the amplitude/probability view — composing
through `ShellPresenter` and running under the console presenter and the web
presenter with no host-specific code in the application.

- [ ] `deno task exomoonlab` opens it in a terminal.
- [ ] The browser build opens the same window with the same output.
- [ ] Backend calls run off the frame loop; the UI stays responsive during a
      20-qubit run.

### 3. The desktop

Adopt `WorkbenchWindowHostController` and the `workbench_shell` painters so
windows drag, resize, snap, tile, and minimize. Wire the theme catalog and the
animated backgrounds.

- [ ] Windows behave as they do in exowebtui, because they are the same code.
- [ ] Every exotui theme applies, plus a MoonLab-branded one.
- [ ] Theme and layout persist through `presenter.store()` — a file natively,
      IndexedDB in the browser.

### 4. The windows that matter

Ported in this order, each one useful on its own:

- [ ] **Circuit builder** — free exploration; build and run a circuit by hand.
- [ ] **Algorithm browser and runner** — Grover, Bell, GHZ, VQE, QAOA, QFT.
- [ ] **Step-through** — gate at a time, with state after each.
- [ ] **Amplitudes and probabilities** — histogram and table.
- [ ] **Bloch sphere** — single-qubit state.
- [ ] **Entropy and entanglement** — von Neumann and bipartite entropy.
- [ ] **Session inspector** — active backend, ABI version, capabilities,
      running jobs.

Then, as they earn their place: Feynman diagrams, topology invariants,
Schrödinger orbitals, and a tensor-network/DMRG monitor.

### 5. Shipping both

- [ ] A `deno task` runs the terminal console.
- [ ] A web build produces a static bundle carrying `moonlab.wasm`.
- [ ] Both are documented, including the Emscripten prerequisite.
- [ ] CI builds both and runs the equivalence harness.

## Acceptance checks

- [ ] The same `ShellApp` object runs under both presenters with no host
      branching in application code.
- [ ] The equivalence harness passes: native and WASM agree on the fixed
      circuit set.
- [ ] The console never blocks its frame loop on a backend call.
- [ ] No change to the MoonLab core beyond added lines in `exports.txt`.
- [ ] Themes and window layout survive a restart on both hosts.

## Notes

**Decided (2026-08-24).** The console lives in this repository, at
`bindings/deno/exomoonlab`, alongside the other language bindings. It is
therefore versioned with the ABI it calls: a change to `MOONLAB_API` or to
`exports.txt` and the console that consumes it land in the same commit, and the
equivalence harness in phase 1 can run in this repo's CI against the library
built beside it. The cost is that the fork's rebases now carry the console too;
that is accepted.

Implementation happens on `feature/exomoonlab-tui`, branched from
`exomoonlab`.

**Still open.** Should exomoonlab eventually replace
`bindings/rust/moonlab-tui`, or do both stay? This does not block any phase —
decide it once the console can do what the Ratatui one does, not before.

**Known environmental gap.** Building a *fresh* `moonlab.wasm` needs the
Emscripten SDK, and `emcc` is not installed on the current machine — which is
also why `webgpu_unified_smoke` fails locally. This turned out not to block
phase 1, which runs against the tracked prebuilt artifact. It will matter when
the console needs a symbol only a new build exports.
