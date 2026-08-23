# Application stack

Record chosen technologies and the reason for choices that are not obvious.
Do not fill this file speculatively.

## Core

| Layer | Choice | Notes |
|-------|--------|-------|
| Language | C11 | `CMAKE_C_STANDARD 11`, required |
| Build | CMake 3.16+ | The supported path; presets in `CMakePresets.json` |
| Build (convenience) | `Makefile` | Not fully maintained — see Current state |
| Version | 1.2.0 | `VERSION.txt` |

Source layout under `src/`: `quantum` (state, gates), `algorithms`,
`applications`, `backends`, `qec`, `mitigation`, `optimization`, `distributed`,
`control`, `crypto`, `integration`, `compat`, `utils`, `visualization`.

## Optional acceleration

OpenMP, LAPACK/OpenBLAS, Metal, CUDA, and MPI are all detected and optional.
Each must degrade to a working CPU path when absent — see
`best-practices.md`.

## Bindings

| Target | Location | Notes |
|--------|----------|-------|
| JavaScript/TypeScript | `bindings/javascript/packages/` | `core`, `algorithms`, `react`, `vue`, `viz` |
| WASM | `bindings/javascript/packages/core/emscripten/` | Emscripten; exports listed in `exports.txt` |
| Python | `bindings/python/` | |
| Rust | `bindings/rust/` | |

## Current state and target state

**Current state.** CMake is the only build that works from a clean tree. The
top-level `Makefile` cannot build `src/utils/manifest.o`, because it needs the
generated `moonlab_build_info.h` that only `cmake/install.cmake` produces. This
is an upstream defect and reproduces on a pristine `origin/master`.

**Target state.** Either the `Makefile` generates `moonlab_build_info.h` the way
CMake does, or it stops advertising targets it cannot build and the
documentation points at CMake alone.

`CONTRIBUTING.md` also tells contributors to branch from `develop`, but no
`develop` branch exists on `origin`. See `plan/workflow/version-control.md`.
