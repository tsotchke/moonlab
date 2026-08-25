# exomoonlab

A themable terminal-desktop console for MoonLab, built on [exotui](https://jsr.io/@ubernaut/exotui).
One application, two hosts: a terminal running against the native library, and a browser running
against the WASM build.

Plan and architecture live in `plan/` at the repository root —
[the task](../../../plan/todo/exomoonlab-console-interface.md) and
[the architecture](../../../plan/arch/exomoonlab.md).

## Status: phase 2 of 5, terminal half

The console runs in a terminal against either backend. The browser half is blocked on exotui — see
**The exotui gap** below.

|                      |                                                                  |
| -------------------- | ---------------------------------------------------------------- |
| `MoonLabBackend`     | one interface over MoonLab's stable C ABI                        |
| native backend       | `Deno.dlopen` over `libquantumsim`, using `quantum_state_create` |
| WASM backend         | the Emscripten build, loaded and driven from Deno                |
| equivalence harness  | seven circuits through both, agreeing to **1.1e-16**             |
| probabilities window | bar chart, marginals, entropy and purity                         |
| terminal host        | `runConsoleShellApp` from `@ubernaut/exotui/runtime`             |
| browser host         | **blocked** — exotui 0.6.0 does not export its web presenter     |

## Running

```bash
deno task exomoonlab                  # the console, in a terminal
deno task exomoonlab --backend=wasm   # force the WASM backend
deno task check                       # type-check
deno task test                        # everything, offline
deno task equivalence                 # native vs WASM only
```

Keys: `j`/`k` circuit, `+`/`-` qubits, `r` rerun, `q` quit.

Or through the project's own suite, where it is gated on `deno` being present:

```bash
cd build && ctest -R deno_bindings_equivalence --output-on-failure
```

The native backend needs a built library. It looks at `$MOONLAB_LIB` first, then
`build/libquantumsim.{so,dylib}` at the repository root:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
```

The WASM backend needs an Emscripten artifact. It prefers a fresh `packages/core/dist/moonlab.js`
and otherwise falls back to the prebuilt one tracked at
`bindings/javascript/demo/public/moonlab.js`, so **no Emscripten SDK is required to run the harness
today**. Override with `$MOONLAB_WASM`.

## Depending on exotui

Pinned to the JSR release in `deno.jsonc`:

```jsonc
"@ubernaut/exotui": "jsr:@ubernaut/exotui@^0.6.0"
```

0.6.0 is the first version carrying the `ShellPresenter` seam this console is built on. To work
against a local exotui checkout without editing the pin, create `import_map.local.json`
(gitignored):

```json
{ "imports": { "@ubernaut/exotui": "/home/you/projects/exotui/mod.ts" } }
```

then pass `--import-map=import_map.local.json`. If the console needs something exotui does not
export yet, add it on a branch in the exotui repository and cut a release — do not vendor or reach
into its `src/`.

## Two notes worth keeping

**The struct layout is not shared.** `quantum_state_t` has 64-bit `size_t` fields natively and
32-bit ones under wasm32, so its size and offsets differ by target. The native backend never needs
them (`quantum_state_create` allocates); the WASM backend over-allocates with upstream's own
constant. Nothing computes an offset in shared code.

**The Emscripten glue is evaluated, not imported.** It is a `-sMODULARIZE` UMD file whose CommonJS
branch is guarded on `typeof module === 'object'`, and the artifacts sit inside packages marked
`"type": "module"` — so Deno resolves them as ESM, that branch never runs, and `createRequire()`
returns an empty object. `src/backend/wasm.ts` evaluates the source with a synthetic `module`
instead, which works regardless of where the artifact lives.

## Known gaps

- The tracked prebuilt WASM predates ULG's `quantum_state_create` / `_destroy` exports, so that
  backend reports `capabilities.allocatingConstructor === false` and allocates manually. A build
  made after those `exports.txt` additions will take the direct path.
- Building a fresh `moonlab.wasm` needs the Emscripten SDK, which is not installed on the current
  development machine.

## The exotui gap

An earlier version of this file said the browser host was blocked by exotui. **It was not.** That
check grepped only `mod.web.ts` and missed that `src/web/mod.ts` re-exports `web_presenter.ts` — the
same pattern that puts the console presenter behind `./runtime`. Available in 0.6.0 today:

| Symbol                                                             | Entrypoint                 |
| ------------------------------------------------------------------ | -------------------------- |
| `webPresenter`, `runWebShellApp`                                   | `@ubernaut/exotui/web`     |
| `consolePresenter`, `runConsoleShellApp`                           | `@ubernaut/exotui/runtime` |
| `ShellApp`, `ShellPresenter`, `ShellPresentedFrame`, `runShellApp` | `@ubernaut/exotui/web`     |
| the workbench window host                                          | `@ubernaut/exotui/web`     |

What genuinely was not reachable: the shell painters (`workbench_shell.ts`) and the seventeen-theme
catalog (`shell_theme.ts`). Both are phase 3 concerns, not phase 2. The seam types were reachable
only through `./web`, which is the wrong door for a terminal application.

exotui 0.7.0 adds a `./shell` entrypoint carrying the seam, the painters, the theme catalog, the
window host, and the backgrounds, with the presenters staying in their host-specific homes. When it
publishes, `src/ui/cells.ts` becomes re-exports from `@ubernaut/exotui/shell` and nothing else
moves.
