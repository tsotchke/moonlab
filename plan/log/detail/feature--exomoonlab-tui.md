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

## 2026-08-24 — Correction: the browser host was never blocked

### Prompt

Branch exotui from main, widen the public entrypoints, prepare 0.7.0, stop
before publish.

### Response and strategy

**The premise was mine and it was wrong.** I had recorded that exotui 0.6.0 did
not export its web presenter, and planned an upstream release around that. It
does. `src/web/mod.ts` re-exports `web_presenter.ts`, and `mod.web.ts`
re-exports that barrel — exactly the pattern that puts `consolePresenter`
behind `./runtime`. My check grepped only the top-level `mod.web.ts` and stopped
there, which is the same class of mistake as the earlier `strings | grep -qx`
false negative on `gate_cnot`: a cheap check answering a slightly different
question than the one asked, taken as authoritative.

Probing the *published* package is what settled it. Available at 0.6.0:
`webPresenter`, `runWebShellApp`, `consolePresenter`, `runConsoleShellApp`, the
seam types, and the window host. **Phase 2's browser half can be built on the
current pin.**

What was genuinely unreachable: the shell painters (`workbench_shell.ts`) and
the theme catalog (`shell_theme.ts`) — phase 3, not phase 2. And the seam types
escaped only through `./web`, which is the wrong door for a terminal app and an
accident of one barrel importing another rather than a decision.

**The upstream change was still worth making, at a smaller size.** exotui
`feature/publish-shell-seam` adds a `./shell` entrypoint for the host-neutral
half — seam, painters, theme catalog, window host, backgrounds — with the
presenters staying in their host-specific homes so the export map shows the
split instead of hiding it. Additive; nothing moved. `deno task health` is
green at 140/140. Committed, not published.

Two things that repo's gate taught, worth keeping:

- Adding an entrypoint there is seven files. The export map, the closed unions
  and manifest, the packaging doc, the README table, *both* fixtures in the
  stability test, and the budget baseline all have to agree; three tests failed
  until they did.
- The generated `docs/api-reference.md` must be simultaneously byte-identical to
  its generator and `deno fmt`-clean. A 133-character manifest description made
  those two requirements contradict each other, and running `deno fmt` on the
  file "fixed" one by breaking the other. The fix belonged at the source.

**Next.** Phase 2's browser host, on the 0.6.0 pin. It does not depend on 0.7.0
publishing.

## 2026-08-25 — Phase 2 complete: the browser host

### Prompt

Go ahead.

### Response and strategy

**exotui first.** Merged `feature/publish-shell-seam` into exotui's `main` and
tagged `v0.7.0` locally; `deno publish --dry-run` succeeds. Not published —
there are no JSR credentials on this machine, so the real publish needs an
interactive browser OAuth flow only the user can complete.

**Then the browser host, which the 0.6.0 pin already supports.** The work was
not in the UI — `web.ts` is fifteen lines of substance — but in the backend.
`openWasmBackend` assumed Deno: `Deno.readTextFile`, `node:module`,
`node:path`. None of that exists in a tab.

The fix was to name the thing that actually differs. `WasmArtifactSource`
splits *reading* the artifacts from *binding* them: Deno reads from disk and
supplies a real `require`; a browser fetches and stubs `require` out, since the
glue only reaches for it on the Node path it never takes there. Everything
below the split is byte-identical, which is what keeps the two hosts honestly
the same computation. `node:module` became a lazy import so a browser bundle
never pulls it in.

`deno bundle` (2.9, experimental but present) removed the need for esbuild.

**Verifying in a real browser earned its keep, twice.**

First it caught that I was testing the wrong thing: `curl` returned 200 for all
three artifacts on port 8787, so the server looked fine — but the page the
browser showed was another project's. My file-server had failed to bind because
that port was already taken, and the 200s came from whatever owned it. A green
check against the wrong process. Picking a verified-free port fixed it.

Then it caught a real defect that no amount of type-checking would have: the
page rendered perfectly and ignored every keystroke. exotui's browser platform
routes keys through a hidden textarea it creates for on-screen keyboards, and
that has to hold focus. A visitor would have had to click before typing worked.
`web.ts` now focuses it on load and on pointerdown. Recorded as an upstream
follow-up: a `focus()` on `WebTuiHost` retires the workaround.

Final state, confirmed by screenshot: `j` selected GHZ, `+ +` grew it to six
qubits, |000000⟩ and |111111⟩ at 50.000% each over dim 64 with six correct
marginals — computed by the WASM build inside the tab, matching what the
terminal shows from the native library.

**Next.** Phase 3, the desktop — and that is the part that genuinely needs
exotui 0.7.0's `./shell` for the painters and the theme catalog.

## 2026-08-26 — Searching for upstream WebGPU/WASM changes: there are none

### Prompt

"We've got some upstream changes we need to rebase on. They should add better
WebGPU support to the WASM — can you find them and rebase them?"

### Response and strategy

Searched exhaustively and found nothing to rebase. Recorded here because a
null result is worth writing down: the next person should not repeat the hunt.

What was checked, after `git fetch origin --prune --tags`:

| Check | Result |
|---|---|
| Our position vs `origin/master` | 0 behind; master's tip is an ancestor of HEAD |
| Remote branches, via the GitHub API | 13, all fetched — no branch we had not seen |
| Every remote ref vs HEAD | 11 branches carry commits we lack |
| Every commit we lack touching `emscripten/` or a webgpu file | enumerated, ~20 |
| WASM export delta on the newest branches (v1.2.1) | **2 symbols**, both quantum annealing |
| WebGPU additions anywhere | **none** |

The commits that *look* like WebGPU work we lack — "Add WebGPU cnot complex64
probe", "Add browser WebGPU parity smoke harness", and the rest of that series
— are the pre-rebase ULG originals. We carry their replayed equivalents under
different hashes, which is why they show as absent.

The one file named for the topic, `webgpuplan.md`, is a June 29 documentation
sync on `qgtl-vendor-local-moonlab-e578884`, a divergent vendored snapshot
older than master. Not implementation.

The only genuinely new WASM exports upstream are `_moonlab_anneal_ising_v1` and
`_moonlab_anneal_qubo_v1` on the unmerged `v121-*` feature branches — quantum
annealing. Worth picking up when those land on master; nothing to do with
WebGPU.

**WebGPU already reaches our WASM build.** The artifact built this session
exports 34 `gpu_*` functions (`gpu_compute_init`, the buffer surface, the gate
kernels), and `webgpu_unified_smoke` passes — it had failed all session purely
because no artifact existed, not because the support was missing.

Two further avenues were checked before calling the search complete:

- **Is `tsotchke/moonlab` itself a fork with an upstream parent carrying the
  work?** No — the GitHub API reports `fork: false`, no `parent`, no `source`.
  It is the root of its own lineage, so there is no hidden upstream to pull
  from.
- **Is the work in an unmerged pull request?** No. All 20 pull requests on the
  repository are closed; the most recent (#20) is 2026-07-31, and none concerns
  WebGPU. The July run of merged PRs is QGT/VQE geometry work, which we already
  carry.

That exhausts every place the changes could be: branches, tags, loose commits,
the export list, a parent repository, and pull requests. **The upstream WebGPU
changes described do not exist.** "Find them and rebase them" resolves to: found
nothing, so the rebase is a no-op — not skipped, but vacuous.

The user was told this and redirected: "it's ok skip the webgpu moonlab rebase
and focus on the exotui work."

## 2026-08-31 — the cloud turns, and demos become applications

Request: "we should be able to zoom and rotate the Schrodinger probability
cloud. different demos and examples should be launched like applications from
the main menu. leverage the exotui lib wherever possible and extend it when you
can't."

### The cloud is a volume now

The orbital window rendered a slice through the x-z plane. A slice is exact
about nodal structure, which is why it was the first thing built, but it cannot
convey shape — `d_xy` and `d_x2-y2` have *identical* x-z slices and are
physically nothing alike. That is a real limitation, not a cosmetic one.

`densityVolume()` samples |ψ|² over a cube; `projectVolume()` forward-splats
each voxel through a yaw/pitch rotation and accumulates **column density**, the
depth-integrated quantity an X-ray of the cloud would measure. Column density,
not maximum-intensity or a nearest-surface hit, because it keeps the projection
linear: a lobe pointing at the viewer reads bright because there is genuinely
more probability along that ray, and the nodal planes stay dark because no
amount of rotation puts density where the wavefunction has a zero. A
maximum-intensity projection would have washed the nodes out.

Verified against physics rather than against a screenshot: `2p_z` at pitch 0
shows two lobes with a dark plane between them; rotated to pitch 90° they point
along the view axis and merge into a single blob. That is what projecting a
dumbbell onto its own axis must look like, and it is the check that a slice
renderer would have failed.

Cost: one full recompute is ~14ms, so rotation re-runs the whole job instead of
caching and re-projecting a volume. Worth noting the tradeoff was measured, not
assumed — at 14ms there is no frame-loop pressure, and a cache would have added
an invalidation rule (which parameters dirty the volume vs. only the
projection) for no user-visible gain. Revisit only if the sampling grid grows.

`wasd` orbits, `9`/`0` zoom, orientation persists across recomputes.

### Demos are applications

Every window except Probabilities now starts closed, and `` ` `` opens a
launcher listing the six with a one-line description each.

Launching a closed window is the window host's own `restore` command. That
choice matters: a closed window here was never destroyed, it was *unlaunched* —
so its state, geometry and z-order all survive, and there is no second
lifecycle to keep in sync with the host's. Inventing a create-on-launch path
would have meant reconstructing what the host already holds.

**Leveraging exotui rather than reimplementing it** was the explicit
instruction, and the launcher follows it literally. Traversal and wrapping come
from `moveWorkbenchMenuIndex`; Enter and Escape are decided by
`isWorkbenchMenuActivationKey` / `isWorkbenchMenuCloseKey`; drawing is
`paintShellMenuPanel`. Hand-rolling those is exactly how a launcher ends up
quietly disagreeing with the rest of the shell about what Escape means — the
menu would close on a key the window chrome treats as something else, and the
divergence only shows up months later as a bug report about "Escape sometimes
not working."

`handleLauncherKey` returns `{kind: "none" | "state" | "launch"}` so the menu
claims only the keys it actually uses and never swallows the keyboard; `q` and
`t` fall through to the desktop even while the menu is open. There is a test
asserting exactly that, because "the menu ate my keystroke" is the other
classic launcher bug.

### What this required from exotui

`src/shell/workbench_menu.ts` existed but was in **no entrypoint** — `./shell`
could *paint* a menu (`paintShellMenuPanel` was exported) but could not *run*
one, since the traversal and key predicates were unreachable. Exporting them is
`6a122538` on exotui's `feature/publish-menu-surface`, with api-reference and
entrypoint budgets regenerated; api_stability and entrypoint_budgets pass 11/11.

This is the first genuine gap the exotui audit turned up. The earlier 8-lens
adversarial audit (53 agents) had found *zero* missing features — worth
recording that the gap surfaced only when a real consumer tried to build a real
launcher, not from auditing the surface in the abstract.

### Verification

39 tests pass (`deno task dev:test`), `deno lint` clean across 31 files, ctest
3/3 on the Deno-facing gates. The desktop was rendered under a pty and the
start button, hint line and single open window confirmed by reading the frame
text — not by opening a browser, per the standing constraint.

Two tests needed updating rather than fixing: Session and Circuits both assert
on window content, and both windows now start closed, so each launches its app
first. The tests were right; the model beneath them changed.
