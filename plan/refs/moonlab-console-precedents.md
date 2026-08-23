# Existing MoonLab consoles

- **Source:** this repository — `bindings/rust/moonlab-tui` and
  `bindings/javascript/demo`
- **Relevant to:** [exomoonlab console interface](../todo/exomoonlab-console-interface.md)
- **Why it matters:** Between them they already answer "what should a MoonLab
  console show". exomoonlab should inherit that vocabulary rather than guess at
  a new one.

## `bindings/rust/moonlab-tui` (Ratatui, terminal only)

Documented at `documents/api/rust/moonlab-tui.md`.

- **Views** (`src/ui/`): `dashboard`, `circuit`, `amplitudes`, `bloch`,
  `entropy`, `feynman`.
- **Modes:** algorithm browser, algorithm running (animated, pausable),
  step-through, free exploration, Feynman diagram browser.
- **Algorithms:** Grover, Bell, GHZ, VQE, QAOA, QFT.
- **Interaction:** vim-style `j`/`k`, `Tab` to cycle panels, `Enter` to run,
  `n` for qubit count, `s`/`f`/`d` to switch mode, `?` for help.

## `bindings/javascript/demo` (React + Vite — the Pages site)

- **Sections:** Playground, Examples, Gallery, Schrödinger orbitals, Topology.
- **`src/workers/moonlabClient.ts`** is the model for the browser backend:
  `runCircuitInWorker`, `probabilitiesFromAmplitudes`, `dmrgWeightsInWorker`,
  `runExampleAlgorithmInWorker`. It keeps the WASM module off the main thread,
  which is exactly what exomoonlab's frame loop requires.
