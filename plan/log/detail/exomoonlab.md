# Detailed development log: `exomoonlab`

- **Branch:** `exomoonlab`
- **Started:** 2026-08-23
- **Base:** `ulg`, immediately after its rebase onto `origin/master`

## 2026-08-23 — Branch created, console planned

### Prompt

Commit the dirty worktree to `ulg` (leaving out things like `.venv` and adding
them to the gitignore), rebase onto upstream tsotchke, then investigate using
exotui to build a new console interface for MoonLab, starting from something
like exowebtui, with a full themable terminal desktop environment. It should
work natively with the native MoonLab for maximum features and performance, and
also support a web build using the WASM build client-side. Do not change the
MoonLab compiler except to trivially add functionality to the WASM build. Write
the plan down in `moonlab/plan`. First port the vibe-plan planning framework to
`ulg`, then branch `exomoonlab` off `ulg` after the rebase.

### Response and strategy

Work on `ulg` first: two commits for the worktree, the rebase, and the
framework port. That history is recorded in `log-summary.md` and does not
belong here.

Branched `exomoonlab` from `ulg` at `2adc037` and surveyed the ground before
proposing anything.

**What the survey found.** Three things already exist that the plan should
lean on rather than duplicate:

- exotui has a *presenter seam* (`ShellPresenter` / `runShellApp`) with a
  console implementation and a browser implementation. exowebtui
  (`examples/web/desktop_app.ts`) already runs as both a terminal application
  and a web page from one object. This is the single most important finding:
  the dual-host requirement is already solved upstream of us, so exomoonlab
  does not need two applications.
- MoonLab has a stable C ABI (0.7.0, `MOONLAB_API`) and re-exports 424 of those
  symbols to WASM via `emscripten/exports.txt`. Both hosts can therefore reach
  the *same* C functions, which is what makes one application honest rather
  than merely convenient.
- `bindings/rust/moonlab-tui` already is a MoonLab console — Ratatui, terminal
  only, with views for circuits, amplitudes, Bloch spheres, entropy, and
  Feynman diagrams, and modes for browsing, running, stepping, and free
  exploration. Together with the Pages demo's sections it gives a proven
  feature vocabulary. Treating it as a reference rather than a thing to port
  mechanically.

**The design that follows.** One `ShellApp`, one new abstraction
(`MoonLabBackend`) with a native Deno-FFI implementation and a WASM
implementation, chosen by probe at startup rather than at build time. The
frame loop is synchronous, so every backend call goes off-thread — a Web
Worker in the browser (the demo's `moonlabClient.ts` is the model) and
non-blocking FFI natively — with the application rendering the latest result
of each job and never waiting.

Wrote it up as `plan/arch/exomoonlab.md` (architecture and reasoning),
`plan/todo/exomoonlab-console-interface.md` (five phases with acceptance
checks), and two reference files under `plan/refs/`.

**Deliberately deferred.** Two questions are recorded in the task file rather
than answered: whether the console lives in this repository or its own, and
whether it eventually replaces `moonlab-tui`. Neither blocks phase 1, so
neither should hold up starting.

**Known gap.** The browser half cannot be verified on this machine: building
`moonlab.wasm` needs the Emscripten SDK and `emcc` is not installed. This is
also why `webgpu_unified_smoke` fails locally.

## 2026-08-24 — The console lives in this repo

### Prompt

Push `origin/ulg`, and create a new branch off this one for the TUI work; it
should live in this repo.

### Response and strategy

Before publishing, moved the Turborepo untracking commit down from
`exomoonlab` onto `ulg` and rebased `exomoonlab` back on top. It is a
repo-hygiene fix of the same class as the others already on `ulg`, and leaving
it above the fork point would have meant anyone branching from the newly
published `ulg` still carried the stale `.turbo/` artifacts. Git dropped the
duplicate on its own during the rebase.

Force-pushed `ulg` — required, since the rebase onto `origin/master` rewrote
its history. Used `--force-with-lease` pinned to the previously fetched
`7952d7a` so the push would abort if anyone had moved the branch meanwhile.
`ulg-prerebase-backup` still holds the pre-rebase tip.

Answered the first open question from the task file: the console lives here, at
`bindings/deno/exomoonlab`, next to the other language bindings. The deciding
argument is that the console consumes the stable ABI, so it should be versioned
with it — a change to `exports.txt` and the code needing it land in one commit,
and the phase-1 equivalence harness can run against the library built beside
it. The accepted cost is that the fork's rebases now carry the console too.

The second question — whether this supersedes `bindings/rust/moonlab-tui` — is
left open on purpose. It is not answerable until the console can actually do
what the Ratatui one does.

Implementation moves to `feature/exomoonlab-tui`, branched from here, starting
at phase 1 (the backend seam and the equivalence harness).
