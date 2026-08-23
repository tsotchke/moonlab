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

### 1. Backend seam and equivalence harness

Define `MoonLabBackend` — state lifecycle, gate application, measurement,
amplitude and probability reads, named algorithms, and a capability record.
Implement both sides: `Deno.dlopen` over `libquantumsim.so` against
`MOONLAB_API`, and the WASM module through `@moonlab/core`.

- [ ] Both backends satisfy the same interface and report capabilities honestly.
- [ ] An equivalence test runs a fixed circuit set through both and asserts
      agreement — exact for integer results, within the declared complex64
      tolerance for amplitudes.
- [ ] Any symbol the console needs that exists in C but not in WASM is added to
      `emscripten/exports.txt` and nowhere else.

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

**Open questions for the user.**

1. Where should the console live — in this repository (say
   `bindings/deno/exomoonlab`), or as its own repository depending on
   `@ubernaut/exotui` and `@moonlab/core`? A separate repository keeps the
   fork's rebases clean; living here keeps it versioned with the ABI it calls.
2. Should exomoonlab eventually replace `bindings/rust/moonlab-tui`, or do both
   stay?

Neither blocks phase 1, which is why phase 1 is first.

**Known environmental gap.** The browser path needs the Emscripten SDK to build
`moonlab.wasm`; `emcc` is not installed on the current machine, and
`webgpu_unified_smoke` fails there for that reason. Phase 2's browser half
cannot be verified until that is installed.
