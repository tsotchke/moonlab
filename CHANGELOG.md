# Changelog

All notable changes to MoonLab Quantum Simulator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-04-17

Stability pin. Commits a set of build-rot fixes that have been shipping as
local patches, introduces the committed downstream ABI header
`src/applications/moonlab_export.h`, and synchronises the project version
across `VERSION.txt` and CMake. Downstream consumers (notably QGTL, which
binds `moonlab_qrng_bytes` via dlsym) should now pin to this tag.

### Features

- Added `moonlab_abi_version(int* major, int* minor, int* patch)` as a
  runtime feature-discovery probe for downstream consumers. Stable for the
  0.x series.
- Added `src/applications/moonlab_export.h`: the committed, versioned,
  downstream-facing ABI header for `libquantumsim`. Declares
  `moonlab_qrng_bytes` and `moonlab_abi_version`. Future public symbols
  (PQC, quantum-geometry, Chern markers) will be added here as they land.
- Added `tests/abi/test_moonlab_export_abi.c`: a dlsym-based smoke test
  that dlopens `libquantumsim` and exercises the published ABI surface
  exactly as a downstream consumer would. Hooked into CTest as
  `abi_moonlab_export`.
- Added `tests/unit/test_constants.c`: regression test that pins every
  mathematical and physical constant in `src/utils/constants.h` to its
  claimed IEEE 754 value (tolerance 1e-14) plus cross-validation against
  `math.h` ground-truth where applicable. Prevents another silent
  encoding regression like the Tsirelson-bound bug fixed below.
- Added `tests/unit/test_correctness_properties.c`: property-based
  tests for the simulator's core physics — 50 random-circuit trials
  confirm ||psi||^2 = 1 after gate application; 50 random Rx/Ry/Rz
  +/-theta pairs restore the input state to L2 < 1e-12; H*H, X*X, Z*Z,
  CNOT*CNOT are each the identity to 1e-14; create_bell_state_phi_plus
  produces exactly [1/sqrt(2), 0, 0, 1/sqrt(2)] to 1e-14.
- Added `mpi_sendrecv` wrapper in `src/distributed/mpi_bridge.{c,h}`
  (both the HAS_MPI and stub branches). `distributed_gates.c:1107`
  was calling this symbol without a declaration or implementation — the
  MPI build did not link. It now does.

### Build

- New CMake option `QSIM_WERROR` (default OFF): turns on `-Werror` for
  the quantumsim library and its tests while keeping `-Wno-error=pedantic`
  so Homebrew's libomp header — which uses enumerator values outside ISO
  C's range — does not break the build. With `-DQSIM_WERROR=ON` the
  Release build is now warning-clean on macOS arm64.
- `CMakeLists.txt` now reads the canonical version from `VERSION.txt` via
  `file(READ)`. Removes the 0.1.1/1.0.0 drift between the two files.
  `SOVERSION` continues to track `PROJECT_VERSION_MAJOR` so
  `libquantumsim.0.dylib` is produced.
- MPI test executables (`test_distributed_gates`, `test_state_partition`,
  `test_collective_ops`) are now only added if their source file exists
  in `tests/unit/`. Previously the CMake generate step failed when MPI
  was enabled because two of the three sources were never written.
- Existing local build patches (committed):
  - Gated the unified GPU backend dispatcher on at least one GPU backend
    being enabled.
  - Gated `src/optimization/gpu_metal.mm` on Metal being detected AND the
    source existing in the tree.
  - Gated `QSIM_DISTRIBUTED_SOURCES` on `QSIM_ENABLE_MPI=ON`.

### Fixes

- Fixed a switch-case scope error in `gpu_backend.c:1180` (declaration in
  `default:` without braces).
- Added a weak-symbol fallback for `metal_sum_squared_magnitudes` so
  builds that have not yet implemented the real kernel in `gpu_metal.mm`
  still link.
- **Tsirelson-bound constant**: `QC_2SQRT2_HEX` encoded
  `0x400B504F333F9DE6` which decodes to 3.4142135623730950 (= 2 + sqrt(2)),
  not 2*sqrt(2) = 2.8284271247461903. Replaced with
  `0x4006A09E667F3BCD`. `bell_test_confirms_quantum()` had been
  comparing CHSH values against 0.9 * 3.4142 = 3.07 — a threshold no
  quantum system can meet — so the Bell test was silently failing for
  all releases up to and including 0.1.1. After the fix, the Bell test
  reports ~99.4% of quantum maximum at 10k samples and converges to
  ~100% at 100k+ samples.
- **Other physical constants**: `QC_FINE_STRUCTURE_HEX`,
  `QC_RYDBERG_HEX`, and `QC_ELECTRON_G_HEX` were all encoded to values
  that decoded to garbage (0.005 instead of alpha; 1e-163 instead of
  Rydberg; 1e-295 instead of g_e). Replaced with the CODATA-correct
  IEEE 754 encodings. Added `QC_RYDBERG_EV` alongside the atomic-unit
  Rydberg for user convenience. These constants were not referenced
  outside `constants.h` so there is no behavioural change in shipping
  code — only in any future consumer that would have read them.
- **Bit-mixing pseudo-constants**: `QC_HEISENBERG`, `QC_SCHRODINGER`,
  `QC_PAULI_X`, `QC_PAULI_Y`, `QC_PAULI_Z` are relabelled as internal
  QRNG bit-mixing constants, which is what they actually are. Their
  values are unchanged, but the misleading physics names are now flagged
  in a block comment.
- **Test bug in `test_state_purity`**: the old test set four equal
  amplitudes on a 2-qubit state and asserted purity < 1, calling that
  a "mixed state". It is in fact a pure state (purity = 1). The
  underlying simulator is pure-state only. Replaced with correct
  purity-invariant checks across several pure-state shapes.
- Fixed format-string warnings throughout (`%lu` / `%llu` / `%zu`
  mismatches with `uint64_t` / `size_t` / `int`) in
  `tests/health_tests_test.c`, `examples/applications/tsp_logistics.c`,
  `src/algorithms/vqe.c`, `src/applications/hardware_entropy.c`, and
  `src/applications/health_tests.c`. Standardised on `<inttypes.h>`
  `PRIu64` for `uint64_t`.
- Sign-compare warnings fixed by adding explicit `(int)state->num_qubits`
  casts at loop boundaries in `src/quantum/measurement.c`,
  `src/quantum/entanglement.c`, `src/quantum/noise.c`,
  `src/algorithms/grover_optimizer.c`, and `src/algorithms/quantum_rng.c`.
- Unused-parameter warnings fixed via `(void)param;` at the top of
  `gpu_phase`, `gpu_cnot` (where backend-gated case arms may all compile
  out), and the rdrand/rdseed/getrandom wrappers (unused on non-x86_64
  or non-Linux targets).
- Unused-variable warnings (`precision_part` in qpe, `info` in
  simd_dispatch on Apple Silicon, `sqrt_gamma` in noise, `kept_mask`
  / `traced_mask` in state, `measure_mask` in measurement, `log2_size`
  in entropy, `theta` in matrix_math) either removed or suppressed
  with `(void)` / `__attribute__((unused))` depending on intent.
- Macro-redefinition warnings fixed: `M_PI`, `M_PI_2`, `M_PI_4`, and
  `M_SQRT2` are now guarded by `#ifndef` since `<math.h>` defines them
  on macOS and glibc.
- Deprecation in `src/optimization/gpu_metal.mm`: `fastMathEnabled` is
  deprecated in macOS 15.0 in favour of `mathMode`. The new code uses
  `@available` to prefer `mathMode = MTLMathModeSafe` on macOS 15+ and
  falls back to `fastMathEnabled` on older systems, with a local
  `#pragma clang diagnostic` around the fallback call.
- Deprecation in `src/quantum/entanglement.c`: the Apple CLAPACK
  interface (`zheev_` and friends) was deprecated in macOS 13.3 in
  favour of `-DACCELERATE_NEW_LAPACK` headers. Migration requires
  touching every `__CLPK_*` typedef, which is scheduled for Phase 1G;
  a `#pragma clang diagnostic push/ignored/pop` pair around the file
  keeps Release `-Werror` builds green in the meantime.
- 35 `.c` / `.h` files were missing trailing newlines — appended.
- Python bindings (`bindings/python/moonlab/core.py`) hard-coded
  `libquantumsim.so` (Linux name). Replaced with platform-aware library
  name resolution (`.dylib` on macOS, `.dll` on Windows) and a
  multi-directory search path that covers both in-tree `build/` and
  installed layouts.
- Python `__init__.py` unconditionally imported `algorithms.py`, which
  in turn bound against C symbols (`molecular_hamiltonian_create`, …)
  that are not yet exported from the 0.1.2 dylib. The import is now
  wrapped in a try/except so the core `QuantumState`/`Gates`/etc. API
  works; the missing-symbol gap in `algorithms.py` is scheduled for
  0.2 Phase 1G.
- `bindings/javascript/packages/core/emscripten/build-clapack-wasm.sh`
  now works on macOS: `sed -i` usage is compatibility-detected for
  BSD vs GNU, the xerbla.c header patch no longer uses `sed 1i` (which
  BSD sed does not support), job-count comes from `nproc` or
  `sysctl -n hw.ncpu`, and the emcmake step passes
  `CMAKE_POLICY_VERSION_MINIMUM=3.5` to work around CLAPACK 3.2.1's
  ancient CMake minimum.

### Build

- Gated the unified GPU backend dispatcher (`src/optimization/gpu/gpu_backend.c`)
  on at least one GPU backend being enabled. Default macOS builds pick it up
  via Metal; headless Linux builds with no backends no longer fail to link
  against missing backend-specific kernels.
- Gated the Objective-C++ Metal implementation (`src/optimization/gpu_metal.mm`)
  on Metal being detected AND the source existing in the tree.
- Gated `QSIM_DISTRIBUTED_SOURCES` on `QSIM_ENABLE_MPI=ON`. The distributed
  sources reference `mpi_*` helpers declared inside MPI-gated translation
  units; compiling them unconditionally was producing spurious link errors
  in the default build.
- `CMakeLists.txt` now reads the canonical version from `VERSION.txt` via
  `file(READ)`. This removes the 0.1.1/1.0.0 drift between the two files
  and makes version bumps a single-file change. `SOVERSION` continues to
  track `PROJECT_VERSION_MAJOR` so `libquantumsim.0.dylib` is produced for
  the 0.x series.

### Fixes

- Fixed a switch-case scope error in `gpu_backend.c:1180` where a variable
  declaration appeared directly inside `default:` without braces; adding
  braces around the default block resolves the Clang warning and the
  associated UB.
- Added a weak-symbol fallback for `metal_sum_squared_magnitudes` in
  `gpu_backend.c` so builds that have not yet implemented the real kernel
  in `gpu_metal.mm` still link. The real implementation, when added,
  overrides the weak stub at link time.
- **Tsirelson-bound constant bug**: `QC_2SQRT2_HEX` in `src/utils/constants.h`
  encoded `0x400B504F333F9DE6`, which is IEEE 754 3.4142135623730950
  (i.e. 2+sqrt(2)), not 2*sqrt(2) = 2.8284271247461903. The correct
  encoding is `0x4006A09E667F3BCD` (same mantissa as the
  already-correct `QC_SQRT2_HEX` with the exponent field shifted by 1).
  The consequence was that `bell_test_confirms_quantum()` compared CHSH
  values against 0.9 * 3.4142 = 3.07 instead of 0.9 * 2.8284 = 2.55, so
  no real quantum CHSH measurement (bounded above by 2.8284) could ever
  pass the "near maximum" check. Simulator CHSH output was always correct;
  only the reporting and pass/fail logic was broken. After this fix, the
  Bell test reports ~99.4% of quantum maximum at 10k samples and essentially
  100% at 100k+ samples, as expected.

### Testing

- Verified clean-slate build on macOS arm64 in both Release and Debug
  configurations.
- 10 of 11 CTest cases pass after this release. Bell test now reports
  CHSH at ~99.4% of the Tsirelson bound at the default 10k sample size,
  converging to essentially 100% (CHSH = 2.8298 at 100k, 2.8265 at 500k).
  The remaining failure is `unit_quantum_state::test_state_purity`, a
  pre-existing test-side bug: it constructs what it calls a "mixed state"
  by setting 4 equal state-vector amplitudes, but that is in fact a pure
  state (purity = 1), and the underlying simulator is pure-state-only.
  To be addressed in the 0.2 Phase 1G build / CI / housekeeping sweep.

### Known issues carried from 0.1.1

- Benign `%lu` vs `uint64_t` format-string warnings in
  `tests/health_tests_test.c` (6 sites) and
  `examples/applications/tsp_logistics.c` (1 site) on macOS where
  `uint64_t` is `unsigned long long`. Will be cleaned up in 0.2.

---

## [Unreleased]

### ✨ Features

- Initial quantum simulator implementation
- State vector simulation up to 32 qubits
- Complete gate set (H, X, Y, Z, CNOT, CZ, RX, RY, RZ, SWAP, Toffoli)
- Grover's search algorithm with parallel batch search
- VQE (Variational Quantum Eigensolver) for molecular simulation
- QAOA (Quantum Approximate Optimization Algorithm)
- QPE (Quantum Phase Estimation)
- Quantum Random Number Generator (QRNG) with NIST SP 800-90B compliance
- Metal GPU acceleration for macOS (100× speedup)
- SIMD optimization (SSE2, AVX2, ARM NEON)
- OpenMP parallelization
- Python bindings

### 📚 Documentation

- Cloud deployment feasibility analysis
- Scaling beyond 50 qubits research
- Algorithm implementation guides

### 🧪 Testing

- Unit tests for quantum state and gates
- Bell test validation (CHSH inequality)
- NIST health tests for entropy validation
- Integration test suite

---

## Version History

This changelog will be automatically updated by [git-cliff](https://git-cliff.org/)
when new releases are tagged.

To generate the changelog locally:
```bash
git cliff --output CHANGELOG.md
```

To generate for a specific version:
```bash
git cliff --tag v1.0.0 --output CHANGELOG.md
```
