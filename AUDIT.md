# Moonlab Deep Architectural Audit — 2026-04-19

**Status:** Moonlab is a well-documented, literature-grounded research
simulator with a clean C core. It is **not yet** a publishable
scientific instrument, a sellable product, a deployable SaaS, or the
integration backend that QGTL, SbNN, and future research projects
need. This document enumerates the gaps, supported by measurements,
and prioritises the remediation roadmap.

The audit was performed by systematic evidence gathering:
micro-benchmarks of every `bench_*` binary, static code-quality
scanning, test-category classification, deployment-readiness probing,
and research-integration surface review. All claims below are
backed by observable evidence in the tree.

---

## 1. Executive Verdict

Moonlab reaches roughly the **Qulacs tier of single-thread CPU
correctness** on its state-vector core: single-qubit gate at
n=16 takes 20.7 µs, matching published Qulacs numbers. On every
other axis that a real scientific platform needs — reproducibility,
deployment, distributed scaling, observability, research integration
— Moonlab is between **missing** and **prototyped**.

The honest gap list, in one breath:

- **Tensor-network benchmark segfaults (SIGKILL).** The entire TN
  stack's measured throughput is unknown right now.
- **Cross-simulator parity is absent.** Only 1.5 % of tests
  cross-validate; zero tests cite published physics numbers.
- **Deployment blockers are real.** 3 750 occurrences of hardcoded
  Homebrew paths; `docker-compose.yml` references three Dockerfiles
  that do not exist; 1 193 unfiltered library symbols are exposed.
- **Logging is bare `printf`/`fprintf` across eight library files.**
  Cannot be silenced, redirected, or captured as structured
  telemetry.
- **Fifteen distinct error enums** across the tree.
- **Version split:** C core + Rust are `0.2.0-dev`; JS + Python are
  still `0.1.2`.
- **Stable ABI exposes exactly three symbols** (`moonlab_abi_version`,
  `moonlab_qrng_bytes`, `moonlab_qwz_chern`). QGTL / SbNN have no
  first-class path to VQE, DMRG, TDVP, topological invariants beyond
  Chern-QWZ, or tensor-network primitives.

The C code is mostly correct. The platform around it is not.

---

## 2. Measured Performance (Moonlab vs Reference Simulators)

Host: Apple M2 Ultra, 24 cores, Release build, default optimisation.

| Workload | Moonlab (measured) | Reference (published) | Verdict |
|---|---|---|---|
| H @ n=16 | 20.7 µs | Qulacs 20 µs @ n=26 | parity on single-thread CPU |
| CNOT @ n=16 | 16.2 µs | no public reference | **gap** |
| SWAP @ n=16 | 43.4 µs | no public reference | **gap** |
| Quantum Volume @ d=10 | mean HOP 0.848 | Porter-Thomas asymptote 0.847 | correct, as expected |
| Clifford GHZ-3200 prep | 83 ms | Stim GHZ-10000 ~0.4 ms | **≥ 200× slower per gate** |
| Clifford random n=1600 | 18.7 µs/gate | Stim ~0.04 µs/gate | **~500× slower** |
| Gate-fusion speedup @ n=20 | 1.54× | Qulacs fusion ~1.3× | above par on same-qubit runs |
| Chern KPM bulk site @ L=300 | 3.5 µs/site | no public reference | works, uncompared |
| Chern mosaic 4:36 patch L=40 | 520 ms | — | reference-free |
| TN gate throughput | **CRASH** | — | blocker |
| State-vector peak memory @ n=16 | 25.8 MB | Qulacs ~40 MB | lean |
| MPI scaling | not measured | mpiQulacs published | **no data** |
| GPU (Metal) gate throughput | not benchmarked | cuStateVec ~10 TFlop/s | **no data** |

**Interpretation.** Moonlab is credible on dense single-thread CPU
workloads; it is unmeasured on distributed, GPU, and tensor-network
workloads; and it is **orders of magnitude** behind the SOTA
specialists on workloads they target (Stim for Clifford, cuStateVec
for GPU state vector, mpiQulacs for distributed).

**What is missing to make this table honest per-release:**
1. A parity harness against Qulacs + Stim + Qiskit Aer, run in CI.
2. An MPI scaling run at 4, 16, 64 ranks on a cluster.
3. A Metal / CUDA / cuQuantum measured-throughput report.
4. Published FLOP/s and GB/s per gate kernel.

---

## 3. Correctness and Testing Rigour

Moonlab has **343** tests, **13 800 lines** of tests against
**24 500 lines** of source — a 56 % ratio. Classification:

| Category | Count | Share |
|---|---|---|
| Smoke (runs, minimal assertion) | ~103 | 30 % |
| Correctness spot-check (fixed numeric target) | ~228 | 66 % |
| Invariant / property (over randomised inputs) | ~7 | 2 % |
| Cross-validation vs reference implementation | ~5 | **1.5 %** |

**Published-number reproduction:** zero tests cite or reproduce a
published physics or CS benchmark value (e.g. no test asserts that
H₂ ground-state energy at R = 0.74 Å matches the FCI value to within
chemical accuracy, no DMRG test asserts the Bethe-ansatz Heisenberg
energy, no Clifford test compares outputs against Stim).

**Subsystem-by-subsystem test verdict:**

| Subsystem | Verdict |
|---|---|
| Dense state vector, gates | well-tested |
| MPS tensor network | well-tested (incl. vs-exact parity) |
| DMRG | smoke only; no comparison to exact diagonalisation or Bethe ansatz |
| TDVP | **untested** (no dedicated file) |
| QFT | **untested** |
| VQE | smoke + loose-tolerance correctness; no FCI parity |
| QAOA | smoke only |
| QPE | smoke only; no published-eigenvalue parity |
| Clifford backend | well-tested; no Stim parity |
| Surface code (Clifford-backed) | smoke only (syndrome structure) |
| Chern real-space (dense) | well-tested (cross-validated against KPM) |
| Chern real-space (KPM) | well-tested |
| Chern momentum-space (FHS / QGT) | well-tested |
| Chemistry (JW, H2/LiH/H2O) | smoke only |
| MBL | correctness with 5-seed disorder averages (should be 20+) |
| QRNG | well-tested (histograms, chi², SP 800-22 subset) |
| Gate-fusion DAG | well-tested (deterministic-RNG random-circuit parity) |
| Noise channels | well-tested (norm preservation, Kraus completeness) |
| Entanglement measures | well-tested |

**Coverage measurement:** no `gcov` / `llvm-cov` / `lcov` is run in
CI. Branch-coverage is not tracked. No property-based or
differential-fuzzing harness.

**Tests with statistical flake risk (randomisation without voting):**
- `tests/unit/test_correctness_properties.c` — single seed 0xC0FFEE.
- `tests/unit/test_tensor_network.c` — `time(NULL)` seed, single shot.
- `tests/unit/test_mbl.c` — 5 disorder seeds (marginal).

**For a publishable platform, minimum required:**
1. Qulacs-parity harness across 200 randomised circuits at
   n ∈ {4, 8, 12, 16}.
2. Stim-parity harness for every Clifford circuit we claim to run.
3. FCI-reference parity for H₂, LiH, H₂O at equilibrium and two
   stretched geometries.
4. Bethe-ansatz / exact-diag parity for DMRG on L ≤ 14 Heisenberg.
5. Branch coverage ≥ 80 % in CI.
6. Property-based fuzz harness for the gate dispatcher.
7. ≥ 20 disorder seeds for every MBL statistic.
8. Statistical voting on every randomised test (2-of-3 or
   Bonferroni-adjusted).

---

## 4. Code Quality and Architecture

### 4.1 Error-handling chaos

Fifteen distinct error enums, all with negative-for-error convention
but inconsistent zero semantics:

```
qs_error_t, partition_error_t, dist_gate_error_t, mpi_bridge_error_t,
collective_error_t, gpu_error_t, clifford_error_t,
svd_compress_error_t, tn_measure_error_t, tn_state_error_t,
contract_error_t, tensor_error_t, tn_gate_error_t, qrng_v3_error_t,
health_error_t, entropy_error_t
```

A library consumer has to know the right enum for every call.
Unifying to two or three families (or adopting a
`moonlab_error_t` + `moonlab_error_context_t` pattern) is a
publishable-platform minimum.

### 4.2 Library prints directly to stdout/stderr

Eight files emit unconditioned `printf` / `fprintf(stderr, ...)`:

- `visualization/circuit_diagram.c:888,896`
- `visualization/feynman_diagram.c:805`
- `distributed/mpi_bridge.c:486-493, 702, 722` (~10 sites)
- `distributed/state_partition.c:107, 124, 992-999` (~10)
- `distributed/collective_ops.c:1654`
- `optimization/memory_align.c:500-511` (~10)
- `optimization/gpu/backends/gpu_opencl.c:207, 219, 351-354, 388, 444-447` (~8)

This **cannot** be deployed in a server or multi-tenant context.
Every library message must go through a pluggable logger (structured
levels, redirectable, silenceable).

### 4.3 Monolithic files

Eight files exceed 1 000 lines and mix several concerns:

| File | Lines | Concerns mixed |
|---|---|---|
| `tensor_network/dmrg.c` | 2 515 | DMRG loop + effective-Ham construction + Lanczos |
| `tensor_network/tensor.c` | 2 503 | Tensor ops + SVD + GPU sync |
| `tensor_network/tn_measurement.c` | 2 156 | MPS measurement + Born rule + sampling |
| `topological/topological.c` | 2 191 | Anyons + surface code + toric + syndrome |
| `mbl/mbl.c` | 2 123 | MBL diagnostics + level stats + entanglement |
| `algorithms/vqe.c` | 2 091 | VQE loop + ansatz + chemistry Jacobian |
| `optimization/gpu/backends/gpu_webgpu.c` | 1 762 | WebGPU backend + WGSL codegen |
| `visualization/circuit_diagram.c` | 1 407 | ASCII + SVG + LaTeX + parsing |

These should each split into 3–5 modules.

### 4.4 Pointer-ownership contracts are undocumented

Eighteen public functions return a malloc'd pointer with no
docstring note on ownership (`circuit_render_ascii`,
`circuit_render_svg`, `circuit_render_latex`,
`feynman_render_ascii/svg/latex`, `molecular_to_qubit_hamiltonian`,
`mpo_to_matrix`, tensor-SVD decomposition results,
`tn_mps_from_statevector`, `tn_mps_copy`, `dmrg_ground_state`,
`lanczos_ground_state`, `svd_compress`, `contract_find_order`,
`contract_execute`, `skyrmion_braid`, `braid_path_circular`). FFI
consumers will leak or double-free these.

### 4.5 Hot-path allocations

`dmrg.c:852-854` allocates three workspace tensors on every Lanczos
iteration (~100 per ground-state solve). Should pre-allocate a
reusable workspace per DMRG context.

### 4.6 Two eigensolvers, two gate paths, three RNG streams

- Eigensolvers: `hermitian_eigen_decomposition` (unsound on
  complex-Hermitian eigenvectors) + Schulz sign function
  (`chern_marker.c`). A publishable platform has one eigensolver.
- Gate paths: the production stride loop in `quantum/gates.c` +
  the unused SIMD-intrinsic version in `optimization/stride_gates.c`.
- RNG streams: splitmix64 (Clifford) + xorshift (QV, fusion bench)
  + hardware-entropy QRNG (`src/applications/qrng.c`). Inconsistent
  seeding: `cfg.seed = 0` means "use hardware entropy" in some
  modules and "default seed" in others.

### 4.7 `moonlab.algorithms` skipped test

`bindings/python/tests/test_algorithms.py` is `pytest.mark.skip`.
Our Python smoke test asserts the module imports; it does not assert
that VQE returns the right H₂ energy or that QAOA returns the
correct MaxCut approximation ratio.

---

## 5. Deployment and Packaging

### 5.1 Blockers for Docker / SaaS

- **Hardcoded Homebrew / developer paths:** 3 750 occurrences of
  `/Users/tyr/`, `/opt/homebrew/`, `/usr/local/opt/libomp` across
  the tree, including `CMakeLists.txt` rpath entries. A container
  cannot reach host Homebrew. This breaks container portability at
  the first `cmake` call.
- **No Dockerfile exists.** `docker-compose.yml` in the repo
  references three Dockerfiles (`ubuntu/release`, `debian/release`,
  `debian/debug`) that **do not exist** in the tree. The compose
  file also hardcodes `VERSION: 0.1.0` instead of reading
  `VERSION.txt`.
- **Version fragmentation.** C core + Rust crates are `0.2.0-dev`;
  five JS packages and Python `pyproject.toml` are pinned at
  `0.1.2`. A consumer of one module gets a mismatched version of
  the native library.
- **ABI export is unfiltered.** `libquantumsim.dylib` exports
  1 193 global symbols, including OpenMP internals
  (`_gomp_critical_user_*`). No allowlist. Breaking changes in
  any exported symbol would compound into ABI instability across
  versions.
- **License mismatch.** Top-level `LICENSE` is MIT. Several source
  files have Apache 2.0 headers. No SPDX identifiers anywhere. A
  compliance review would flag this.
- **No CI for Windows MSVC or macOS x86_64.** Only macOS arm64 and
  Linux x86_64 are tested.

### 5.2 What works today

- `make install` produces a correct install tree: headers go to
  `include/quantumsim/`, libs to `lib/`, CMake config to
  `share/cmake/quantumsim/`, pkg-config to `lib/pkgconfig/`. A
  downstream CMake `find_package(quantumsim)` consumer would work.
- The stable ABI header (`moonlab_export.h`) is committed and
  dlsym-probed in CI.
- No committed secrets (grepped `.env`, `.pem`, `.key`, tokens:
  zero hits).
- Accelerate / Metal / OpenMP dependencies are all license-compatible
  (Apple proprietary + LLVM/Apache 2.0; no GPL).

---

## 6. Research-Platform Integration

### 6.1 What QGTL / SbNN / research groups actually need, and what we have

| Need | Status |
|---|---|
| Data export (HDF5 / NPZ / Zarr / JSON) | **missing** |
| Reproducibility manifest (git SHA, seed, build, host) | **missing** |
| Deterministic seed cascade across all RNG sites | partial |
| Streaming / intermediate results from long runs | **minimal** (verbose-to-stdout only) |
| Cancellation / timeout / progress callback | **absent** |
| Structured logging abstraction | **absent** (hardcoded `printf`) |
| Physical-units convention documented | ✓ (atomic units, `constants.h`) |
| NumPy integration | ✓ (clean) |
| Pandas / xarray / scipy.sparse integration | **missing** |
| Checkpoint / restore for long jobs (DMRG/TDVP/surface code) | **absent** |
| Stable ABI covering VQE / DMRG / Berry / surface code | **3 symbols only** |

### 6.2 Concrete gaps for QGTL

QGTL today can call `moonlab_qwz_chern` and `moonlab_qrng_bytes`. It
**cannot** access:

- Per-plaquette Berry curvature array (lives in private
  `qgt_berry_grid`).
- Fubini-Study metric at arbitrary momenta (`qgt_metric_at`).
- SSH winding / Zak phase (`qgt_winding_1d`).
- Custom Bloch callbacks (`qgt_create`).
- Chern-KPM marker / bulk map (`chern_kpm_*`).
- Haldane model (`qgt_model_haldane`).
- Anyon fusion / braiding primitives.

Every one of these is implemented and tested. None is on the stable
ABI. QGTL has to vendor Moonlab's private headers to reach them,
defeating the ABI story.

### 6.3 Concrete gaps for SbNN

The Python `moonlab.algorithms` module imports cleanly, but:

- No autograd through VQE / QAOA (the Torch layer in
  `torch_layer.py` is ctypes-callback-based and covers gates, not
  full algorithms).
- No streaming parameter updates from optimisers.
- No way to checkpoint an optimiser state.
- No way to cancel a long batch run.

---

## 7. SaaS / Commercial Readiness

None of the below exist today. Each is a **hard requirement** for
running Moonlab as a service.

- Multi-tenant isolation: no per-request resource accounting, no
  memory quota, no CPU quota.
- Input validation at boundary: the library trusts its inputs.
  Large `num_qubits` values allocate without bounds; malformed
  Pauli strings in VQE silently drop terms (fixed for H₂O in
  CHANGELOG, but the category of bug persists).
- Rate-limiting / DoS protection: no.
- Structured telemetry (OpenTelemetry / Prometheus / etc): no.
- Secrets management (entropy source, any future licence key): no.
- Authentication / authorisation at the API boundary: no (Moonlab is
  a C library, not a service; someone needs to build that layer).
- Cancellation token through every long-running call: no.
- Health / readiness / liveness probes: no.

A realistic SaaS path is a gRPC or HTTP wrapper around the C library
that owns request lifecycle, quotas, auth, and telemetry — the C
library stays pure compute. That wrapper does not exist in the tree.

---

## 8. Prioritised Roadmap

Effort estimates are **my best guess** at single-developer-days with
an LLM pair programmer. They are not promises; they are relative
scaling.

### Tier 0 — Blockers (must land before 0.2.0 release, ~2 weeks)

1. **Remove `/Users/tyr/` and Homebrew hardcodes** from all
   non-Dockerfile sources. Use relative paths + env-override.
   (~1 day)
2. **Create real Dockerfiles** referenced by `docker-compose.yml`
   (Ubuntu 22.04, Debian 12, Debian-debug). (~2 days)
3. **Align versions across all bindings** (C, Rust, Python, JS) to
   `0.2.0-dev`. (~0.5 day)
4. **License hygiene.** One top-level LICENSE (MIT), SPDX header in
   every source file, reconcile Apache-2.0 misattribution. (~1 day)
5. **Fix the TN benchmark crash.** Whatever segfaults
   `bench_tensor_networks` right now is a production bug, not a
   benchmark bug. (~1 day)
6. **ABI symbol allowlist.** Use linker versioning / hidden
   visibility so `libquantumsim.dylib` exports only intended
   public symbols. (~1 day)

### Tier 1 — Publishable (3–4 weeks)

7. **Cross-simulator parity harness**: Qulacs for state-vector,
   Stim for Clifford, OpenFermion/FCI for chemistry. Run in CI.
   (~1 week)
8. **Reproducibility manifest**. Every run emits JSON with git SHA,
   build flags, compiler, hostname, seed, timestamps, input
   checksums. (~1 day)
9. **Deterministic-seed mode** (`QSIM_DETERMINISTIC=1`) that forces
   a single cascade seed through every RNG site. Independent of the
   QRNG cryptographic path. (~2 days)
10. **Published-number benchmark corpus**. 1D Heisenberg DMRG at
    L=100 vs Bethe ansatz. H₂/LiH/H₂O VQE vs FCI (CCSD(T) where
    required). QWZ / Haldane Chern vs exact. Kitaev toric vs Stim.
    Each with a pass/fail threshold. (~1 week)
11. **Pluggable logger.** Replace every `printf`/`fprintf` in
    `src/` with calls to `moonlab_log(level, fmt, ...)`. Default
    backend is stderr; callers can install their own. (~2 days)
12. **Uncertainty quantification on expectation values.** Every
    measurement function gets a `_with_error` variant that returns
    mean + standard error from the shot count. (~2 days)
13. **Fix the Jacobi complex-Hermitian eigensolver** or replace
    with a proper Householder-QL / Jacobi-complex implementation
    (LAPACK `zheevr` on Linux; Accelerate on macOS). Remove the
    Schulz-detour workaround from `chern_marker.c`. (~3 days)

### Tier 2 — Ultra-high-performance (6–8 weeks)

14. **Wire `stride_gates.c` SIMD into the production dispatcher.**
    Measure AVX-512 / NEON / SVE speedup per gate. (~1 week)
15. **Fused 2Q gate kernels.** CNOT / CZ / SWAP / controlled-rotation
    dedicated paths. (~1 week)
16. **Cache-aware gate blocking** at n ≥ 24 for dense SV. (~3 days)
17. **Metal 2Q + measurement kernels.** The Metal backend is
    single-qubit-only today. (~2 weeks)
18. **MPI scaling measurement** at 16 / 64 ranks on a real cluster.
    Publish FLOPs / gate / rank. (~3 days + cluster access)
19. **Benchmark dashboard in CI.** JSON output, per-commit tracking,
    regression alerts. (~1 week)
20. **Parallelise the KPM matvec** via blocked BLAS-3 instead of
    per-site OpenMP (where the overhead hurts). (~1 week)

### Tier 3 — Research platform (6–8 weeks)

21. **HDF5 + NPZ data export** for every output type (state
    vectors, MPS tensors, Berry grids, syndrome history, QV
    outcomes). (~1 week)
22. **Checkpoint / restore API** on `tn_mps_state_t`,
    `surface_code_clifford_t`, VQE / DMRG optimiser state. (~1 week)
23. **Streaming callbacks** on every long-running call (VQE, DMRG,
    TDVP, QAOA, QV). Intermediate-energy, intermediate-entanglement,
    intermediate-syndrome. (~1 week)
24. **Cancellation token** through every long-running call. (~3 days)
25. **Expand stable ABI** to cover Berry curvature grid,
    Fubini-Study metric, SSH winding, Haldane, Chern-KPM, and
    tensor-network MPS / DMRG handles. Committed in
    `moonlab_export.h`. (~1 week)
26. **Units / constants verification module.** Runtime check that
    user inputs land in atomic units; no silent conversion. (~2
    days)
27. **Fragment the monolithic files** (dmrg.c, tensor.c,
    topological.c, vqe.c, gpu_webgpu.c, circuit_diagram.c). (~1
    week)

### Tier 4 — SaaS (6–10 weeks, separate repository)

28. **gRPC / HTTP wrapper** owning request lifecycle, quotas, auth,
    telemetry. Moonlab stays pure compute. (~3 weeks)
29. **Structured telemetry** (OpenTelemetry / Prometheus metrics,
    slog structured logging). (~1 week)
30. **Multi-tenant resource accounting** and per-tenant caps.
    (~2 weeks)
31. **Authentication / authorisation.** OAuth2 / API keys. (~1
    week)
32. **Deployment artifacts.** Helm chart, K8s manifests, Terraform
    for one cloud. (~2 weeks)

### Tier 5 — SOTA algorithmic scope (ongoing)

33. MPO / QTCI Chern mosaic to 10⁶–10⁸ sites (Antão et al. 2026).
34. Adaptive-bond TDVP with entropy feedback.
35. Split-CTMRG PEPS optimisation.
36. BP + cluster-corrected contraction (Alhambra-Tindall et al.).
37. Tree tensor networks, isoTNS variational optimiser.
38. Decision-diagram / TDD / QMDD backend.
39. cuStateVec / cuQuantum full runtime path.
40. Belief-propagation PEPS contraction.

---

## 9. Budget Summary

The realistic effort to move Moonlab from "research prototype" to
"publishable, sellable, deployable, multi-backend integration
platform" is roughly **6 months of focused work** organised in the
tiers above. The first two tiers (blockers + publishable) are
~5–6 weeks and unlock credible scientific publication; Tier 2
(ultra-high-performance) adds ~6–8 weeks and is needed to claim
competitive benchmarks; Tiers 3–5 round out the research platform,
SaaS deployment, and SOTA algorithmic surface. Tier 5 items are
open-ended and best sequenced against the paper targets they
enable.

Moonlab has credible architectural bones. It has not yet had the
focused performance, validation, and deployment work that separates
"good research code" from "a platform other researchers depend on."
This document is the blueprint to close that gap.
