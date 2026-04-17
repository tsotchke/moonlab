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

### Testing

- Verified clean-slate build on macOS arm64 in both Release and Debug
  configurations.
- 9 of 11 CTest cases pass. The two failures (`bell_test`,
  `unit_quantum_state`) are pre-existing issues carried over from 0.1.1:
  `bell_test` has too-strict a CHSH threshold (measured 2.7536, above the
  classical bound but the test asserts closer to Tsirelson 2 sqrt(2)), and
  `unit_quantum_state::test_state_purity` constructs what it calls a
  "mixed state" by setting 4 equal state-vector amplitudes — that is in
  fact a pure state, and the underlying simulator is pure-state only.
  Both will be addressed as part of the 0.2 Phase 1G build / CI / housekeeping
  sweep.

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
