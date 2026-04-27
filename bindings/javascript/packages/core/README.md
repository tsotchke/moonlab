# @moonlab/core

WebAssembly + WebGPU bindings for the Moonlab quantum simulator.

## Build

The Wasm artefact is produced by Emscripten compiling the C core under
`emscripten/CMakeLists.txt`.

### Prerequisites

- **emsdk** (Emscripten SDK) installed and activated.  Source it before
  building so `$EMSDK` is in the environment:

  ```sh
  # one-time install
  git clone https://github.com/emscripten-core/emsdk.git ~/emsdk
  cd ~/emsdk && ./emsdk install latest && ./emsdk activate latest

  # in every shell where you want to build
  source ~/emsdk/emsdk_env.sh
  ```

  After sourcing, `which emcc` should print a path inside the emsdk
  install and `echo $EMSDK` should print the install root.  The
  `build:wasm` script reads `$EMSDK` directly to locate the
  `Emscripten.cmake` toolchain file.

- **Node 20+** and **pnpm** for the TypeScript build and test suite.

### Commands

| Command | What it does |
|---|---|
| `pnpm build:wasm` | Runs CMake under the Emscripten toolchain at `emscripten/build/`, produces `moonlab.js` + `moonlab.wasm`. |
| `pnpm build:ts` | TypeScript build via tsup (CJS + ESM + `.d.ts` to `dist/`). |
| `pnpm build` | Both of the above, in order. |
| `pnpm test` | vitest run against the Wasm + TS surface. |
| `pnpm webgpu:smoke` | End-to-end WebGPU dispatch sanity check (requires a runtime that exposes `GPU` — Node 22 with `--experimental-webgpu`, or Deno). |

### Build flavours

`emscripten/` ships two BLAS strategies for the linear-algebra core
inside the Wasm build:

- `build-clapack-wasm.sh` — bundles a tiny CLAPACK shim
  (`f2c_stub.c`) so the binary stays small and dependency-free.
- `build-openblas-wasm.sh` — links against an emscripten-built
  OpenBLAS for full-throttle SVD / QR.  Larger artefact, faster math.

`pnpm build:wasm` invokes the default `emscripten/CMakeLists.txt` build
which uses the CLAPACK shim.  Use the matching shell script directly if
you want the OpenBLAS variant.

### Troubleshooting

- *`Could not find toolchain file: $EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake`* —
  `$EMSDK` is unset.  Source `emsdk_env.sh` and retry.

- *The TS bundle compiles but importing `./wasm` fails at runtime* —
  `dist/moonlab.wasm` is stale.  The `webgpu:*` scripts copy the latest
  artefact from `emscripten/build/` automatically; if you ran
  `build:ts` standalone, run `pnpm build:wasm` first or just
  `pnpm build`.
