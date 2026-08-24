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

## YYYY-MM-DD — Short description

### Prompt

<!-- What did the user request or clarify? -->

### Response and strategy

<!-- What was done, why, what happened, and what remains? -->
