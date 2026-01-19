# Moonlab JavaScript Bindings

This folder contains the JavaScript/TypeScript bindings, the WASM build, and the demo app.

## What's here
- `packages/core`: WASM build + JS bindings (`@moonlab/quantum-core`).
- `demo`: Vite demo app (Orbital Explorer and other examples).

## Prerequisites
- Node.js 18+ (20+ recommended)
- `pnpm` (corepack or global install)
  - Corepack (preferred): `corepack enable`
  - Global: `npm i -g pnpm`
- Emscripten SDK (`emcc` on PATH) for WASM builds

## Install dependencies
From the repo root:
```
cd bindings/javascript
pnpm install
```

## Run the demo
```
cd bindings/javascript
pnpm --filter @moonlab/quantum-demo dev
```

## Build for GitHub Pages
This builds the demo into `/docs` at the repo root with relative asset paths:
```
cd bindings/javascript
pnpm --filter @moonlab/quantum-demo build:gh-pages
```
The script will auto-build CLAPACK if the libs are missing (requires network access).
The demo entrypoint will be `/docs/index.html`.

## Recent changes (Tensor Network + DMRG)
- Tensor-network solver sources are compiled into the WASM build (DMRG/TDVP helpers, MPS ops).
- New JS bindings are exposed in `@moonlab/quantum-core`:
  - `TensorNetworkState`
  - `dmrgTFIMGroundState`
- The Orbital demo now uses the DMRG TFIM ground-state distribution to modulate the orbital cloud.

## Emscripten toolchain
- The WASM build is driven by CMake in `packages/core/emscripten`.
- Ensure `emcc` is on PATH (source `emsdk_env.sh`).
- The current build has been tested with Emscripten 4.0.23.

## BLAS/LAPACK requirement (WASM)
Tensor-network solvers require BLAS/LAPACK for SVD and matrix operations. This repo uses a CLAPACK build
as a portable fallback (C-only, no Fortran).

### CLAPACK (recommended)
Script: `packages/core/emscripten/build-clapack-wasm.sh`
- Downloads `clapack-3.2.1-CMAKE.tgz` from netlib.
- Builds with `emcmake`/`emmake`.
- Stages libs and headers in:
  - `packages/core/emscripten/build/deps/clapack-wasm/lib`
  - `packages/core/emscripten/build/deps/clapack-wasm/include`

The CMake build auto-detects this location. To override:
```
export CLAPACK_WASM_ROOT=/path/to/clapack-wasm
```

### OpenBLAS (optional)
If you already have a wasm-built OpenBLAS bundle, point to it:
```
export OPENBLAS_WASM_ROOT=/path/to/openblas-wasm
```
Expected layout:
- `libopenblas.a`
- `include/` with `cblas.h` and `lapacke.h`

If `OPENBLAS_WASM_ROOT` is set and valid, it takes precedence over CLAPACK.

## Build steps
```
cd bindings/javascript/packages/core/emscripten
bash build-clapack-wasm.sh

cd /home/cos/projects/moonlab/bindings/javascript
pnpm --filter @moonlab/quantum-core build
pnpm --filter @moonlab/quantum-demo dev
```

Notes:
- The wasm link may emit warnings about `s_copy`/`s_cat` signature mismatches from `libf2c.a`; these are expected.
- A small `f2c_stub.c` is included in the WASM build to satisfy the `MAIN__` symbol required by `libf2c.a`.
