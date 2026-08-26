# exomoonlab

A themable terminal-desktop console for MoonLab, built on [exotui](https://jsr.io/@ubernaut/exotui).
One application, two hosts: a terminal running against the native library, and a browser running
against the WASM build.

Plan and architecture live in `plan/` at the repository root —
[the task](../../../plan/todo/exomoonlab-console-interface.md) and
[the architecture](../../../plan/arch/exomoonlab.md).

## Status: phase 3 of 5, the desktop

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
# exotui 0.7.0 is not published yet, so use the `dev` tasks. They resolve
# exotui from a sibling checkout via import_map.dev.json.
deno task dev                       # the desktop, in a terminal
deno task dev --backend=wasm        # force the WASM backend
deno task dev --simple              # single-window view instead of the desktop
deno task dev:test                  # every test
deno task dev:build:web             # bundle the browser host into dist/
deno task serve:web                 # serve dist/ on :8787

deno task test:core                 # the exotui-free subset; works with no override
```

Keys: `j`/`k` circuit, `+`/`-` qubits, `t` theme (`T` back), `tab` focus, `m` maximize, `r` rerun,
`q` quit.

Once exotui 0.7.0 publishes, delete `import_map.dev.json` and use the plain tasks —
`deno task exomoonlab`, `deno task test`, `deno task build:web` — which resolve straight from JSR.

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

## The browser host

`deno task build:web` bundles `web.ts` and copies the Emscripten glue and `.wasm` beside it. The
backend resolves the glue as `./moonlab.js` relative to the page, so the three landing as siblings
is the contract, not a convenience.

The WASM backend is host-agnostic by construction: `WasmArtifactSource` splits _reading_ the
artifacts from _binding_ them. Deno reads from disk and hands the glue a real `require`; the browser
fetches and stubs `require` out, because the glue only reaches for it on the Node path it never
takes there. The binding code below that split is identical, which is the point — same C surface,
same numbers, only delivery differs.

One rough edge worth knowing: keys reach the application only once the host's keyboard target has
focus, and exotui 0.6.0 exposes no `focus()` on the host — the target is a hidden textarea its
browser platform creates for on-screen keyboard support. `web.ts` focuses it on load and on
pointerdown so a visitor can type without clicking first. A `focus()` on `WebTuiHost` would let that
workaround go away.

## The desktop

`src/app/desktop.ts` owns no window mechanics. Dragging, resizing, snapping, tiling and the
title-bar controls all come from exotui's `WorkbenchWindowHostController` — the same controller
exowebtui uses, so the behaviour cannot drift from it. This package supplies the window contents, a
palette, and the routing.

Three windows: **Probabilities** (the phase-2 view), **Circuits** (the catalog, with the selection
marked), and **Session** (backend, capabilities, theme).

Keys: `j`/`k` circuit, `+`/`-` qubits, `t` theme (`T` backwards), `tab` focus, `m` maximize, `r`
rerun, `q` quit.

Themes are exotui's `SHELL_THEMES` plus one MoonLab-branded entry — eighteen in all. A theme belongs
to the shell, not to this application, so redefining one here would only guarantee drift. Title-bar
text colour comes from exotui's `shellActiveTitlebarForeground`, which picks dark-on-pale or
light-on-dark by luminance rather than by eye.

Theme, circuit and register size persist through `presenter.store()` — a file under the console
host, IndexedDB in the browser.

### Requires exotui 0.7.0, which is not published

The desktop imports `@ubernaut/exotui/shell`, which does not exist in 0.6.0. Until 0.7.0 publishes,
anything touching the desktop needs the local override:

```bash
deno run --import-map=import_map.local.json --allow-read --allow-ffi --allow-env main.ts
deno test --import-map=import_map.local.json --allow-read --allow-ffi --allow-env
```

`deno task test:core` is the subset that needs no exotui at all — the backend seam and the pure
painters — and is what `ctest` runs, so the project's own gate stays green regardless.

A path override does not carry a dependency's own `imports` the way a JSR dependency would, so
`import_map.local.json` mirrors exotui's npm specifiers as well. That is a property of path
overrides, not a fault in exotui.

### Verified

The terminal desktop is verified in a real terminal (via a pty), and the headless suite covers
window layout, circuit selection, theme cycling and the persistence round-trip. The web bundle
builds and serves, and carries the desktop code — but **the desktop has not been rendered in a
browser**; phase 2's single-window view was the last thing checked there.
