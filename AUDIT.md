# Moonlab Deep Architectural Audit — 2026-04-19

> **2026-04-30 update.** Third pass (commit log `c73cd78..61ec2e0`).
> Major delta since 2026-04-26:
>
> - **var-D mode for CA-MPS shipped** (commits 791dfc4, 5814465,
>   24bc665, plus task #78–84): greedy local-Clifford search,
>   alternating loop, four warmstart options (IDENTITY, H_ALL,
>   DUAL_TFIM, FERRO_TFIM), 2-gate composite moves.
> - **1+1D Z2 lattice gauge theory shipped** (commit d66c712):
>   Pauli-sum builder for matter + gauge-link Hamiltonian, Gauss-law
>   and Wilson-line accessors, demo driver in `examples/hep/`,
>   research write-up.
> - **Gauge-aware stabilizer-subgroup warmstart shipped**
>   (commit 61ec2e0): Aaronson-Gottesman symplectic-Gauss-Jordan
>   Clifford builder, new `STABILIZER_SUBGROUP` warmstart enum,
>   unit-tested on Bell + GHZ + Z2 LGT generators + commutativity
>   reject path.
> - **CA-PEPS scaffold shipped** (commit 4dedb3e): public API stable,
>   gate logic returns `NOT_IMPLEMENTED` until v0.3.
> - **Discrepancy found**: task #73 ("Centralise error enums into
>   one `moonlab_status_t` registry") is marked completed in the
>   task list but no `moonlab_status_t` symbol exists in `src/`.
>   The 41 per-module error enums are still distinct.  Reopened as
>   an open item.
> - **ICC dogfood**: indexed Moonlab (782 files, 346K LOC, 6350
>   symbols), surfaced ~50 dead functions ≥ 30 LOC under `src/`
>   with `--grep-confirm` (no call-shaped textual reference).
>   Triage queue documented in §"Dead-code triage queue" below.
>
> **2026-04-26 update.** A second systematic pass (commit log
> `c73cd78..HEAD`) closed several of the 19 Apr findings.  Status of
> every gap called out below:
>
> - Python / JS version split — fixed.  `setup.py` now reads
>   `VERSION.txt` (`2ac971d`); all five manifests resolve to
>   `0.2.1.dev0`.
> - Logging / `printf`-in-library-files — count down from 8 files to
>   6 (`quantum/state.c`, `utils/{performance_monitor,manifest,
>   config}.c`, `utils/{bench_stats,validation}.h`).  Logging shim
>   still pending.
> - Test seed flake risk — fixed (`0f566a1`): `test_tensor_network.c`
>   now uses a fixed `0xC0FFEE` seed.
> - CI exclusion-by-name — fixed (`8fe2b1d`): all six tier
>   invocations now use `ctest -LE long`; new long tests get
>   auto-excluded by labelling them `long`.
> - CA-MPS shipped without examples — fixed (`a7e6065`): two
>   examples land under `examples/tensor_network/`.
> - Cross-platform tensor_qr / tensor_svd correctness — fixed
>   (`c73cd78`, `365792b`): no-LAPACK platforms now use a real
>   Householder QR + a wide-matrix-aware Jacobi SVD that match
>   LAPACK to machine precision.
> - simd_aligned_memcpy/memset "lying name" — fixed (`7a92303`):
>   debug-build alignment assertion + honest header doc.
> - vqe_solve unchecked mallocs — fixed (`e126b36`).
> - mpi_bridge half-init failure mode — fixed (`96bcb32`).
> - DMRG Lanczos cleanup symmetry — fixed (`fe6998d`).
> - Cumulative truncation metric unbounded under non-unitary
>   evolution — fixed (`eca44ea`): `max_relative_truncation_error`
>   added as a bounded sibling field.
> - QRNG NIST 2-of-3 voting flake — fixed (`fe38742`): widened to
>   3-of-5.
> - kagome ED ASAN timeout — fixed (`88bdc9b`).
>
> **Still open at 2026-04-30**:
>
> - **Centralised `moonlab_status_t` registry**: task list flagged
>   completed but the symbol does not exist in the codebase.  41
>   per-module error enums still distinct.
> - **Stable ABI surface still narrow**: `moonlab_camps_*` and
>   `moonlab_vqe_*` shipped (task #72) but most algorithm modules
>   still lack ABI exposure.  CA-PEPS, var-D, and gauge-warmstart
>   primitives are not yet in `moonlab_export.h`.
> - **CA-PEPS gate logic + 2D contraction**: scaffolded, all gate
>   functions return `CA_PEPS_ERR_NOT_IMPLEMENTED`.  v0.3 milestone.
> - **Z2 LGT exact gauge invariance**: the kinetic terms in
>   `lattice_z2_1d.c` anti-commute with `G_x` term-by-term (lambda
>   penalty enforces gauge invariance only energetically), so the
>   var-D + gauge-aware-warmstart combo is exact at warmstart but
>   drifts under imag-time evolution.  Fix design is documented in
>   `docs/research/var_d_lattice_gauge_theory.md` §"Hamiltonian
>   gauge invariance"; ~30-line Pauli-sum-builder edit + commutativity
>   unit test pending.
> - **MBL `construct_lioms` and `scan_phase_diagram`**: 165 LOC and
>   103 LOC respectively per ICC dead-code report; no in-tree
>   callers.  Either dead or part of an unwired research path.
> - **GPU backends**: CUDA, OpenCL, Vulkan, WebGPU all declared but
>   only Metal is functional.  ICC found `metal_mps_expectation_zz`
>   (177 LOC) with no callers — likely dead code in the Metal layer
>   too.
> - **Distributed primitives**: `partition_fetch_remote` (135 LOC),
>   `partition_scatter_updates` (135 LOC), `dist_mcx` (132 LOC),
>   `partition_plan_2q_exchange` (86 LOC), several
>   `collective_*` ops — all dead per ICC.  MPI bridge ships,
>   distributed engine on top of it does not.
> - **Aarch64 QRNG init slowness**: worked around with test
>   exclusion (task #70 marked completed but the workaround stays).
> - **Coverage tooling**: gcov / llvm-cov tier was added (task #71)
>   but not yet wired into the public CI dashboards.
> - **vqe_solve refactor**: marked complete (task #68) but the file
>   is still ~454 LOC and `vqe_apply_pauli_rotation` (117 LOC) and
>   `vqe_create_uccsd_ansatz` (69 LOC) are flagged dead by ICC —
>   the refactor likely orphaned legacy code paths that should be
>   deleted.
> - **Deployment blockers**: Homebrew paths, missing Dockerfiles
>   for some platform tiers.  Carry-over.
>
> See `MEMORY.md` and the active task list for the remaining queue.

## Dead-code triage queue (ICC `find-dead-code --grep-confirm`)

ICC dogfood (2026-04-30) on the `moonlab` index found ~50 functions
in `src/` with no call-shaped textual reference anywhere in
`src/`, `tests/`, `include/`, `examples/`.  Triage strategy:

1. **Public API**: keep, document.  ICC may have miscounted because
   external bindings (Python, Rust, JS) call these via the stable
   ABI; if not, move to `moonlab_export.h`.
2. **Internal but reachable via dispatch / vtable / function pointer**:
   keep, audit dispatch.
3. **Internal and truly unreachable**: delete.

Top candidates to triage by file:

| File | Function | LOC | Initial classification |
|---|---|---|---|
| `src/algorithms/tensor_network/tn_measurement.c` | `tn_expectation_2q` | 276 | Public API or test helper — confirm |
| `src/algorithms/tensor_network/tensor.c` | `tensor_einsum` | 178 | Likely public — confirm |
| `src/optimization/gpu_metal.mm` | `metal_mps_expectation_zz` | 177 | Probably dead (Metal-only) |
| `src/algorithms/tensor_network/tn_state.c` | `tn_mps_from_statevector` | 175 | Likely public (tests use it?) |
| `src/algorithms/mbl/mbl.c` | `construct_lioms` | 165 | Open research path; unwired |
| `src/algorithms/tensor_network/dmrg.c` | `mpo_skyrmion_create` | 156 | Skyrmion module; likely dead |
| `src/algorithms/tensor_network/svd_compress.c` | `svd_left_canonicalize` | 141 | Public — likely dead |
| `src/optimization/parallel_ops.c` | `grover_parallel_partitioned_search` | 139 | Distributed Grover; unwired |
| `src/distributed/state_partition.c` | `partition_fetch_remote` | 135 | Distributed engine unwired |
| `src/distributed/state_partition.c` | `partition_scatter_updates` | 135 | Same |
| `src/distributed/distributed_gates.c` | `dist_mcx` | 132 | Same |
| `src/algorithms/tensor_network/dmrg.c` | `dmrg_energy_variance` | 128 | Public diagnostic — confirm |
| `src/algorithms/vqe.c` | `vqe_apply_pauli_rotation` | 117 | Possibly orphaned by refactor |
| `src/algorithms/topological/topological.c` | `surface_code_decode_correct` | 110 | Public surface-code primitive |
| `src/optimization/simd_ops.c` | `simd_compute_probabilities` | 109 | SIMD path — confirm runtime dispatch |

The remaining ~35 entries are below 100 LOC.  Each entry should be
either deleted, exposed via the stable ABI, or wired into a real
call site before the v0.2.1 → v0.2.2 → v0.3 ramp.

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

- **Tensor-network DMRG is slow.** Not crashing as initially
  reported — the audit-agent runner hit its own timeout during a
  long bench. DMRG at 16 sites, χ=32, 5 sweeps takes 18.3 s;
  ITensor does this in ~1 s. ~15-20× slower is the real gap. MPS
  gate application and entropy are fine.
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
| MPS gate @ 50 sites, χ=32, H | 0.0 µs (below timer resolution) | — | OK |
| MPS gate @ 50 sites, χ=32, CNOT | 2.4 µs | — | OK |
| MPS entropy @ 100 sites, χ=32 | 158 µs | — | OK |
| DMRG 10 sites, χ=16, 5 sweeps | 676 ms | ITensor ~100 ms | **~7× slower** |
| DMRG 10 sites, χ=32, 5 sweeps | 1.46 s | ITensor ~150 ms | **~10× slower** |
| DMRG 16 sites, χ=32, 5 sweeps | 18.3 s | ITensor ~1 s | **~18× slower** |
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
