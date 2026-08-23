# Project plan

> This file is user-owned. An LLM may change it only with explicit permission.
>
> Drafted from the repository's current state and the `exomoonlab` kickoff
> brief. Review and correct it; it is the one file assistants must not edit
> unprompted.

## Purpose

MoonLab is a quantum simulation library written in C11, with bindings for
JavaScript/TypeScript, Python, and Rust. It exists to make state-vector and
tensor-network quantum simulation fast on ordinary hardware, and reachable from
the language a user already works in.

This fork (`ulg`, branched from `tsotchke/moonlab`) additionally pursues a
browser-capable WebGPU path and a ULG quantum-response artifact contract.

## Scope

### In scope

- The C11 simulation core: state vector, gates, tensor networks (MPS/MPO/DMRG),
  QEC, error mitigation, and the algorithm library.
- Bindings that expose that core: JavaScript/TypeScript (including the
  Emscripten WASM build), Python, and Rust.
- A WebGPU compute path usable from the browser, at complex64 parity with the
  CPU reference.
- `exomoonlab`: a themable terminal-desktop console interface for MoonLab built
  on exotui/exowebtui, running against the native library for full performance
  and against the WASM build in the browser.

### Out of scope

- Changes to the MoonLab simulation core made on behalf of the console
  interface, beyond trivial additions that expose existing functionality to the
  WASM build. The console consumes MoonLab; it does not reshape it.
- Networking, scheduling, or multi-tenant control-plane behaviour beyond what
  the core already ships.
- Reimplementing physics that upstream already provides.

## Success criteria

- The core builds and its test suite passes on Linux and macOS.
- Each binding exposes the core without diverging from its numerical results.
- The browser WebGPU path matches the CPU reference within the declared
  complex64 tolerance for every operation it claims to support.
- `exomoonlab` runs the same session against either the native library or the
  WASM build, with the backend chosen at startup rather than at build time.

## Constraints

- C11 for the core. CMake is the supported build system; the bare `Makefile`
  is a convenience path and is not fully maintained upstream.
- This is a fork. `origin` is `tsotchke/moonlab` and we do not control it, so
  local work must survive repeated rebases onto upstream.
- Optional acceleration (OpenMP, LAPACK/OpenBLAS, Metal, CUDA, MPI) must stay
  optional: a build without them must still compile and pass.
