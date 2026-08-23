# Development summary

Record concise, durable progress, decisions, and pivots that will help someone
resume the project. Do not duplicate task files, commit history, or routine chat.

Add dated entries in reverse chronological order when there is something worth
preserving.

## 2026-08-23

- Rebased `ulg` onto `origin/master`, which had force-updated and moved 788
  commits ahead. 25 of our 26 commits replayed; 15 older ones were already
  upstream and dropped as patch-equivalent.
- Dropped "Fix core WASM readiness blockers" entirely: upstream had
  independently made every fix in it, generally more thoroughly (the purity
  calculation, the `-0` conjugate, the `AMPLITUDES_OFFSET`, the bigint-safe
  `measureAll`, and moving `onRuntimeInitialized` from `post.js` to `pre.js`).
  Keeping it would have registered that callback twice.
- Upstream renamed `docs/` to `documents/` and deleted
  `src/optimization/stride_gates.c` in v0.4.2.
- Ported the vibe-plan planning framework into `plan/`, preserving the four
  documents that were already there: the 89K development log became
  `log/detail/ulg.md`, `tests.md` became `test/test.md`, the WebGPU parity note
  became a task under `todo/`, and `implementation-status.md` stayed put.
- Fixed three build defects found while verifying the rebase:
  - `make clean` ran `rm -f tests/integration/test_*`, an unanchored glob that
    deleted 16 tracked `.c`/`.h` sources.
  - `test_moonlab_export_abi` linked `${CMAKE_DL_LIBS}` but not
    `${MATH_LIBRARY}`, so GCC's `sincos` folding broke the link.
  - `IsingModel` was exported twice from the JS core after the rebase — ours
    from `./ising-model`, upstream's from `./qaoa` — failing the d.ts build.
    Ours is now `UlgIsingModel`.
- Verified: CMake build clean, `ctest` 183/184 passing. The one failure,
  `webgpu_unified_smoke`, needs `pnpm build:wasm` and the Emscripten SDK, which
  is not installed on this machine. Two libirrep tests skip by design.
- Recorded that the top-level `Makefile` cannot build a clean tree — it needs
  the CMake-generated `moonlab_build_info.h`. This reproduces on a pristine
  `origin/master` and is an upstream defect, not a local one.
