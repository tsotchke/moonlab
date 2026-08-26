# Branch specification: everything not in `main`

What this branch changes relative to upstream, and why.

- **Branch:** `feature/exomoonlab-tui`
- **Upstream baseline:** `origin/master` @ `2024c7f` — *Ratchet the stability
  sweep so new public symbols cannot land untagged*
- **Delta:** 46 commits, 86 files, +14,993 / −38
- **Written:** 2026-08-26

Upstream is `tsotchke/moonlab`, which we do not control and which force-pushes
`master`. This branch is rebased onto it rather than merged, so everything below
must survive that repeatedly. See `plan/workflow/version-control.md`.

## Summary

| Area | Change |
|------|--------|
| WASM export surface | 422 → **488** declared exports; built artifact 447 → **515** functions |
| Emscripten build | 4 source files added; chemistry and real-space topology now compile to wasm32 |
| C simulation code | **none** — no `src/**/*.c` or `*.h` modified |
| JavaScript bindings | ULG quantum-response artifact, magnetar reference suite, WebGPU complex64 parity harness |
| New: Deno console | `bindings/deno/exomoonlab` — 27 files, ~3,900 lines |
| Build fixes | 3 defects that broke a clean tree or destroyed tracked files |
| Repository hygiene | 17 build artifacts untracked |
| Planning | vibe-plan framework ported into `plan/` |

**Verification:** `ctest` 185 tests, **0 failures**, 2 skipped by design. Console
suite 25 tests, 0 failures.

---

## 1. The WASM build surface

The largest and most consequential change. MoonLab's C library exports 1,663
symbols; the Emscripten build declared 422 of them. Whole public modules were
native-only, so a browser consumer could not reach them at all.

### 1.1 Export list — 66 symbols added

`bindings/javascript/packages/core/emscripten/exports.txt`, 422 → 488.

| Family | Added | What it unlocks |
|--------|------:|-----------------|
| `qgt_*` | 32 | Berry curvature, quantum metric, Wilson loops, Z₂ invariants, phase-diagram sweeps; models QWZ, Haldane, SSH, BHZ, Kane-Mele, Hofstadter, Kitaev |
| `chern_*` | 14 | FHS Chern integer, local markers, KPM estimator, projector builder |
| `uccsd_*` | 5 | Unitary coupled-cluster ansatz |
| `molecular_*` | 5 | Molecular Hamiltonian construction and qubit mapping |
| `jw_*` | 3 | Jordan-Wigner transforms |
| `quantum_state_*` | 3 | `create`, `destroy` (ULG), `from_amplitudes` |
| `qubit_hamiltonian_*` | 2 | Expectation values, lifecycle |
| `h2_sto3g_pauli_coeffs` | 1 | H₂ STO-3G coefficients vs bond length |
| `grover_quantum_counting` | 1 | Marked-state counting |

Two of these (`quantum_state_create`/`_destroy`) came from the inherited ULG
work; the other 64 were added here.

### 1.2 Emscripten source list — 4 files added

`bindings/javascript/packages/core/emscripten/CMakeLists.txt`:

```
src/algorithms/chemistry/chemistry.c
src/algorithms/topology_realspace/chern_fhs.c
src/algorithms/topology_realspace/chern_kpm.c
src/algorithms/topology_realspace/chern_marker.c
```

**This was found the hard way and is the most transferable lesson here.** The
export list was first validated by checking every symbol against
`nm -D libquantumsim.so`. All 55 resolved. The build then failed anyway:

```
wasm-ld: error: symbol exported via --export not found: chern_fhs_qwz
```

Twenty symbols existed in the native library but their *source files were never
in the Emscripten build*. The two builds compile different source sets, so the
native library says nothing about what wasm32 can link.

> **Verify a WASM export list by building it. Checking against the native
> library will pass and still be wrong.**

Result: the artifact went from **447 to 515** exported functions.

### 1.3 What this does not change

No file under `src/` is modified — not one line of simulation code, no changed
signatures, no altered numerics. The scope constraint was that changes to
MoonLab proper be limited to trivially widening the WASM build, and it holds:
this is an export list and a source list.

---

## 2. Build defects fixed

Three, all of which broke a clean tree or destroyed tracked files. All are
upstream defects reproduced on a pristine `origin/master`, not damage from this
branch.

### 2.1 `make clean` deleted 16 tracked source files

`Makefile`. The clean target ran:

```make
rm -f tests/integration/test_*
```

The glob is unanchored, so it matched `.c` and `.h` sources alongside the
compiled binaries — deleting `test_control_plane*.c`,
`test_gpu_backend_correctness.c`, `test_platform_integration.c` and
`test_tls_keygen.h`. A `make clean` followed by a CMake configure then failed
with *"No SOURCES given to target"* for fifteen targets.

Replaced with a `find` that keeps the intent — remove every compiled
integration binary, including the CMake-only ones no Makefile variable names —
while excluding source extensions.

### 2.2 `test_moonlab_export_abi` did not link libm

`cmake/tests.cmake`. The target linked only `${CMAKE_DL_LIBS}` while every
sibling links `${MATH_LIBRARY}`. The test `dlopen`s the library rather than
linking it, but its own body calls `sin()`/`cos()`, which GCC folds into a
single `sincos()` — a libm symbol. The link failed and took the whole build
with it.

### 2.3 Run targets discarded the caller's library path

`Makefile`. All 32 run targets hardcoded `LD_LIBRARY_PATH=.`, replacing rather
than extending the ambient path, so a build whose dependencies live outside the
default loader search path could not start. Now
`RUNTIME_LIBRARY_PATH = .:$(LD_LIBRARY_PATH)`.

### 2.4 Known-remaining upstream defect

The top-level `Makefile` **cannot build a clean tree**: `src/utils/manifest.o`
includes `moonlab_build_info.h`, which only `cmake/install.cmake` generates.
Reproduced on pristine `origin/master`. **Use CMake.** Not fixed here — it is
upstream's to resolve, and doing so would mean touching the build beyond the
agreed scope.

---

## 3. JavaScript bindings

Inherited from the ULG branch and replayed through the rebase, plus one fix.

### 3.1 ULG quantum response artifact

`src/ulg-quantum-response-artifact.ts` (~1,570 lines) with a CLI emitter, unit
and integration tests, and a guide under `documents/guides/`. Emits
schema-validated `QuantumResponseArtifact` JSON carrying provenance, validation
and parity fields.

### 3.2 Magnetar reference suite

Dipole Ising calibration, contract validator, family inventory, and
normalised/canonical reference exports, with checked-in contracts under
`references/`.

### 3.3 Browser WebGPU complex64 parity

`src/webgpu-complex64-parity.ts` (~2,180 lines) plus a 960-line test, a browser
smoke harness (`.mjs` + `.html`), and recorded evidence under `artifacts/`.
Built probe-by-probe: probability kernel, hadamard, pauli_x, pauli_z, cnot,
backend preflight.

### 3.4 `IsingModel` export collision — fixed

A genuine rebase regression. ULG added `src/ising-model.ts` exporting
`IsingModel`; upstream has since added a *different* `IsingModel` in
`src/qaoa.ts` (since 0.5.5). Both re-exported from the package index, so `tsup`
failed with `TS2300: Duplicate identifier`.

They are not interchangeable — upstream's takes `create(numQubits)` and a
`bigint` `evaluate()` and adds `fromMaxcut()`; ULG's takes `create({ numQubits })`
and a `number` `evaluate()`. Upstream's keeps the bare name; ULG's is re-exported
as `UlgIsingModel`. Every consumer imports these module-locally, so no call site
changed.

---

## 4. New: the exomoonlab console

`bindings/deno/exomoonlab` — 27 files, ~3,900 lines of TypeScript. A themable
terminal-desktop console built on [exotui](https://jsr.io/@ubernaut/exotui).

Sited beside the other language bindings because it consumes the stable ABI and
should be versioned with it: a change to `exports.txt` and the code needing it
land in one commit. Architecture: `plan/arch/exomoonlab.md`.

### 4.1 One application, two hosts

The same object runs in a terminal and in a browser tab. The hosts are one file
each — `main.ts` calls `runConsoleShellApp`, `web.ts` calls `webPresenter` and
`runShellApp` — and nothing else differs.

`ShellApp.frame()` is synchronous and runs ~30×/second, so **every** backend
call is async at the type level and goes through a `Job`; the frame paints the
latest result and never awaits. A superseded run is abandoned rather than
queued.

### 4.2 The backend seam

`MoonLabBackend` has two implementations over the *same* C functions:

| | Native | WASM |
|---|---|---|
| Reached via | `Deno.dlopen` over `libquantumsim` | Emscripten build |
| State allocation | `quantum_state_create` | `malloc` + `quantum_state_init`, or the allocating ctor when present |
| Max qubits | 28 | 24 |

**Struct layouts are never shared.** `qgt_berry_grid_t` is 24 bytes natively and
16 under wasm32 (`size_t` is 4 there); `quantum_state_t`'s offsets differ the
same way. Each backend writes its own down. The native path mostly avoids the
problem by using allocating constructors.

Capabilities are **probed from the live module**, never assumed, so a stale
artifact reports `bandGeometry: false` and the affected window says which
backend cannot do the work rather than drawing something empty.

### 4.3 Windows

| Window | Shows |
|--------|-------|
| Probabilities | Basis-state histogram, per-qubit marginals, entropy, purity |
| Band geometry | Berry curvature over the Brillouin zone; Chern number as the headline |
| Schrödinger | \|ψ\|² for a hydrogenic orbital on the x–z plane; 9 orbitals, 1s→4f |
| Circuits | The shared circuit catalog with the selection marked |
| Session | Backend, capabilities, active theme |

Windows drag, resize, snap, tile and minimize via exotui's
`WorkbenchWindowHostController` — the same controller its own reference desktop
uses, so the behaviour cannot drift. 18 themes; theme, circuit and register size
persist through `presenter.store()` (a file natively, IndexedDB in the browser).

### 4.4 Colour scales

Two different jobs, two different scales, both tested for monotonicity across
all 18 themes:

- **Berry curvature is polarity** → diverging: two hues, *neutral* midpoint,
  equal steps per arm, zero exactly on the neutral step.
- **\|ψ\|² is magnitude** → sequential: one hue, light→dark.

Poles derive from the active theme rather than a fixed pair, because a theme
belongs to the shell. Title-bar foreground comes from exotui's
`shellActiveTitlebarForeground`, which picks dark-on-pale or light-on-dark by
luminance rather than by eye.

### 4.5 Equivalence harness

Wired into `ctest` as `deno_bindings_equivalence`. Runs the same work through
both backends and compares:

| Check | Agreement |
|-------|-----------|
| 7 circuits — probabilities, marginals, entropy, purity | **1.1e-16** |
| Berry curvature at 5 mass values | **2e-16**, identical Chern integers |
| Uploaded amplitude vector round-trip | within 1e-12 |

It **fails rather than skips** when a backend cannot do something — a suite that
skipped half of itself and still reported "ok" would be worse than none. Because
those symbols only reach the artifact from `pnpm build:wasm`, CMake gates
*registration* on `dist/moonlab.wasm` existing, matching how the pnpm and cargo
binding tests are gated: an unbuilt tree does not register the test instead of
failing it.

Physics is asserted against published results, not snapshots of our own output:
QWZ Chern numbers against the model's phase diagram (C = +1 for −2 < m < 0, −1
for 0 < m < 2, 0 beyond) at six mass values and integral to 1e-6; orbital
densities normalise to 1; 2p has an exact nodal plane at z = 0.

---

## 5. Repository hygiene

Seventeen build artifacts untracked (files left on disk):

- 12 compiled binaries — `qsim_test`, `tools/hw_rng_probe`, the test executables
  under `tests/`. Rebuilt by every `make`, dirtying the worktree with megabytes
  of binary churn. Upstream had already untracked them.
- 5 Turborepo artifacts under `bindings/javascript/.turbo/` — a build-cache
  tarball, its metadata, a daemon cookie, two daemon logs from January.

`.turbo/` had been in `.gitignore` since upstream's `954652c`, but ignore rules
do not apply to files already in the index, so the rule never took effect. Same
root cause as the binaries: tracked first, ignored later.

`.gitignore` also gained Python virtualenv patterns.

---

## 6. Planning framework

The vibe-plan layout ported into `plan/`, preserving all four documents already
there via `git mv`:

```
log.md                             -> log/detail/ulg.md      (89K, intact)
tests.md                           -> test/test.md           (+ project-wide sections)
browser-webgpu-complex64-parity.md -> todo/
implementation-status.md            stays, linked from README
```

`AGENTS.md` (with `CLAUDE.md` symlinked to it) carries the reading order plus
the repository rules an agent needs up front: CMake rather than the Makefile,
capping OpenMP so `ctest` does not saturate the machine, and `documents/` rather
than the renamed `docs/`. It defers to `CONTRIBUTING.md` for branch and commit
conventions rather than restating them.

---

## 7. Verification

| Suite | Result |
|-------|--------|
| `ctest` (full) | **185 tests, 0 failures**, 2 skipped by design (`unit_libirrep_*`) |
| Console | **25 tests, 0 failures** |
| Binding smokes | rust, js-vitest, deno, python, version-sync — all pass |

`webgpu_unified_smoke` had failed throughout development on the missing WASM
artifact. It passes now.

Build and test with CMake, capped so the machine stays usable — each test
spawns one OpenMP thread per core, so an uncapped `ctest -jN` uses far more
than `N`:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
cd build && OMP_NUM_THREADS=2 taskset -c 0-7 ctest --output-on-failure -j4
```

---

## 8. Known gaps

Recorded rather than hidden.

- **The built WASM artifact is not committed.** `dist/` is gitignored, so the
  66 added exports only exist for someone who runs `pnpm build:wasm` (needs
  `emcc`; emsdk is installed at `~/emsdk`). The tracked prebuilt at
  `demo/public/moonlab.wasm` is still the older 447-export build. Refreshing it
  would help fresh checkouts but means committing a 1.5 MB binary that conflicts
  on every rebase — deliberately left to a decision rather than taken.
- **`build:ts` cleans `dist/`.** It deletes the WASM artifacts, so the order is
  `build:ts && build:wasm`, as the package's own `build` script does. Running
  them backwards silently leaves the old artifact in place.
- **Kitty image rendering is not wired.** The Schrödinger window detects
  kitty-class terminals and reports it, but renders half-blocks either way:
  exotui's console presenter exposes no image channel, and raw graphics escapes
  fight its diffing painter. Upstream item.
- **The console needs exotui 0.7.1.** Published 0.7.0 lacks two exports found by
  building against it — `createTiledWorkspaceController` and four types used in
  `./shell`'s own signatures. Fixed on exotui's
  `feature/moonlab-console-support`, unpublished. Until then the `dev` tasks use
  `import_map.dev.json`.
- **The Makefile cannot build a clean tree** (§2.4).
