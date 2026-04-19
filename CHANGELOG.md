# Changelog

All notable changes to MoonLab Quantum Simulator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Audit / housekeeping

- **Academic-grade documentation pass (audit phase A)**: postdoctoral-
  level prose and verified reference lists added to the P5.x module
  headers -- `clifford.h`, `chern_marker.h`, `chern_kpm.h`, `qgt.h`,
  `quantum_volume.h`, `fusion.h`, and the `surface_code_clifford_t`
  block in `topological.h`. New top-level `ARCHITECTURE.md` describes
  every subsystem, the three computational representations
  (state-vector / tensor-network / Clifford tableau), the ABI story,
  and the full bibliography. Every citation in the new prose was
  verified by live arXiv / DOI fetch during the audit. Corrections
  carried over from the verification pass: Bianco-Resta (2011) lives
  at arXiv:1111.5697 (not 1108.2935 or 1104.5133); Haegeman TDVP PRL
  is arXiv:1103.0936 (not 1103.5869). Hallucinated ID candidates that
  did not resolve to the expected papers were discarded.
- **Version + ABI alignment**: `VERSION.txt` bumped `0.1.2 -> 0.2.0-dev`,
  README citation + badge + `documents/index.md` updated, Rust crates
  (`moonlab`, `moonlab-sys`) bumped to `0.2.0-dev`, Python
  `__version__` bumped. The ABI header had already been at
  `0.2.0`/minor=2; package and ABI now agree.
- **CMake version parser** accepts a pre-release suffix (`-dev`, `-rc1`,
  etc.) in `VERSION.txt` and passes the numeric triple to `project()`.
- **Rust bindings expand to cover the 0.2 surface**: the
  `moonlab-sys` allowlist now includes 67 new bindings for
  `moonlab_qwz_chern`, Clifford, Quantum Volume, Chern marker (dense
  + KPM), QGT + SSH + Wilson, gate-fusion. `moonlab::topology`
  high-level module added (`qwz_chern`, `ssh_winding`, `ChernKpm`),
  with three passing unit tests inside the crate.
- **`matrix_math` ground-truth unit test** (`unit_matrix_math`):
  matmul on 2x2 complex, trace, Frobenius norm, Hermitian check,
  conjugate transpose, eigenvalues on Pauli-Z and a 3x3
  real-symmetric matrix (+ `M v = lambda v` check on the real path
  only), NULL guards. Covers a previously untested `src/utils/`
  subsystem.
- **README caveat** for the complex-Hermitian eigenvector unsoundness
  in `hermitian_eigen_decomposition`; the header already warned but
  a user reading only the README had no signal.
- **`stride_gates.h` docstring** now admits it is a parallel
  experimental module; the production `gate_*` path already uses the
  same stride pattern inline.
- **`moonlab.algorithms` import guard** downgraded from "known
  broken" to "defensive": verified all 66 ctypes signatures resolve
  against the shipping dylib and `from moonlab.algorithms import
  VQE, QAOA, Grover, BellTest` succeeds. Broken-subsystems memory
  note updated; `python_bindings_smoke` now asserts
  `_ALGO_AVAILABLE == True`.

### Features

- **VQE symmetry-preserving ansatz** (`VQE_ANSATZ_SYMMETRY_PRESERVING`).
  Particle-conserving Givens rotations; paired with
  `vqe_create_h2_hamiltonian` and `VQE_OPTIMIZER_COBYLA` now reaches
  chemical accuracy (<0.1 kcal/mol) on H₂ across bond distances 0.5 –
  2.0 Å.
- **VQE exact ground-state reference**. New
  `vqe_exact_ground_state_energy` builds the full Pauli Hamiltonian
  matrix and returns the lowest eigenvalue via shifted power iteration.
  Used by `vqe_h2_molecule` to compare against the Hamiltonian's real
  FCI (not a literature constant that didn't match).
- **`pauli_hamiltonian_t::hf_reference`** field: Hartree-Fock bitmask
  used by `vqe_compute_energy` to prepare the reference state before
  the ansatz runs. Preset for H₂, LiH and H₂O factories.
- **Kraus-completeness validator**. `noise_kraus_completeness_deviation`
  returns `max |Σ K†K − I|` element-wise for every single-qubit channel.
  Test matrix (6 channels × 5 p values) passes at ≤ 2.2×10⁻¹⁶.
- **QRNG**: `QRNG_V3_MODE_BELL_VERIFIED` now force-enables continuous
  Bell monitoring (previously indistinguishable from `DIRECT`).
- **Clifford stabilizer backend** (`src/backends/clifford/`). Aaronson-
  Gottesman tableau (arXiv:quant-ph/0406196), 2n × (2n+1) layout. Gates:
  H, S, S†, X, Y, Z, CNOT, CZ, SWAP. Measurement separates deterministic
  from random branches and reports which. Verified on a 100-qubit GHZ
  (all-or-nothing across 200 shots) and Bell-state correlations
  (1002/0/0/998 over 2000 shots). Header: `src/backends/clifford/clifford.h`.
  Python surface: `moonlab.Clifford`; the existing
  `python_bindings_smoke` test now builds and measures a 100-qubit GHZ
  (beyond the dense simulator's ceiling). Throughput benchmark
  (`bench_clifford`): GHZ-3200 prep in 82 ms, measure-all in 57 ms
  (~18 us/qubit); random 1600-qubit Clifford stream 55 us/gate.
  Scaling is O(n²) per gate as expected.
- **Python bindings** for `quantum_volume_run` as `moonlab.quantum_volume`
  returning a `QuantumVolumeResult` dataclass (width, num_trials,
  mean_hop, stddev_hop, lower_ci_97p5, passed).
- **Local Chern marker, matrix-free KPM**
  (`src/algorithms/topology_realspace/chern_kpm.{c,h}`). Applies the
  projector `P = (I - sign(H))/2` via a Jackson-regularised Chebyshev
  expansion of `sign(H_hat)`, evaluated with a stencil-based QWZ
  matvec. Memory stays O(N); no dense projector is formed. Parity
  with the dense reference: |c_dense - c_kpm| < 0.001 at L=8, m=±1,3
  (n_cheby=80). Scales linearly in N: L=100 (20k-dim, 10k-site
  lattice) in 47 ms/site, L=300 (180k-dim, 90k-site lattice) in
  569 ms/site, returning c(bulk) = +1.0000. The equivalent dense
  projector at L=300 would need ~24 GiB of matrix storage.
- **QGT primitives expansion**: `qgt_wilson_loop` (closed-path Berry
  phase on any user-supplied path, building block for Z_2 / mirror
  Chern / non-Abelian holonomies), `qgt_winding_1d` (integer winding
  number of 1D chiral two-band systems via the Zak phase), and the
  built-in `qgt_model_ssh(t1, t2)` whose topological regime
  (|t2| > |t1|) returns winding = +1, trivial regime returns 0.
  Python: `moonlab.ssh_winding`, `moonlab.berry_grid_qwz`,
  `moonlab.berry_grid_haldane` return NumPy arrays ready for plotting;
  sum(grid)/(2 pi) = Chern at 1e-4 accuracy on the 32x32 QWZ case.
- **Quantum geometric tensor module**
  (`src/algorithms/quantum_geometry/qgt.{c,h}`). Momentum-space
  Berry curvature + Chern number via the Fukui-Hatsugai-Suzuki
  link-variable method (JPSJ 74, 1674, 2005) with gauge-stable
  eigenvector selection (dual-branch picker that avoids spurious
  pi-jumps across h_z = 0). Built-in models: Qi-Wu-Zhang, Haldane.
  Fubini-Study / quantum metric via centered finite differences
  (`qgt_metric_at`); verified PSD and symmetric. Cross-validates
  the real-space Chern marker: QWZ m=+1 gives C=-1 on both paths.
  Haldane topological regime gives C=+-1, trivial gives 0.
- **Stable-ABI entry point** `moonlab_qwz_chern(m, N, out_chern)`
  added to `src/applications/moonlab_export.h`. QGTL, lilirrep
  and SbNN can now probe the Chern number through a single dlsym
  call. ABI minor version bumped 0.1 -> 0.2. The abi-smoke test
  exercises the new symbol at m=+1 (C=-1) and m=+3 (C=0).
- **OpenMP parallelism for Chern mosaics** via
  `chern_kpm_bulk_sum` and a new `chern_kpm_bulk_map` that fills a
  per-site `double[]` map. Sites are embarrassingly parallel; the
  parallel region wraps one site at a time so per-matvec overhead
  stays zero. Measured: 256-site mosaic at L=24 drops from 530 ms to
  40 ms (13x on a 24-thread machine); 1024-site mosaic at L=40 runs
  in 520 ms end-to-end.
- **Python bindings for Chern KPM** as `moonlab.ChernKPM`. Returns
  NumPy arrays for `bulk_map`; exposes `set_cn_modulation(n, Q, V0)`
  for 4-, 8-, 10-fold quasicrystal modulations. The existing
  `python_bindings_smoke` test asserts c(bulk mean) ~ +1.0 on a
  12 x 12 QWZ topological lattice.
- **QRNG statistical test flake fix**: `unit_qrng_statistics` now
  runs each SP 800-22 subtest on 3 independent samples and requires
  2-of-3 passes (per NIST SP 800-22 section 4.2). Drops the
  single-run false-rejection rate from ~6% to ~2x10^-5.
- **Quasicrystal modulation + Chern mosaic** (`chern_kpm_set_modulation`,
  `chern_kpm_cn_modulation` in `chern_kpm.{c,h}`). Attaches a
  spin-independent on-site potential V(r); a helper builds
  `C_n`-rotationally-symmetric cosine modulations
  `V(r) = V_0 Σ cos(q_i · r)`. The matrix-free matvec picks it up
  without allocating any extra matrix memory. Verified: small V_0
  preserves c(bulk) ≈ +1; V_0 = 3 drives the bulk mean to ~0
  (gap closed). `bench_chern_mosaic` computes a full 24x24 bulk
  marker map with C_4 modulation in 0.6 s; the mid-V_0 transition
  regime shows spatial structure consistent with the quasicrystal
  symmetry. This is the first iteration of the plan's §2I Chern
  mosaic capability; the MPO/QTCI backend needed for 10^6+ sites
  remains scheduled as the next iteration.
- **Local Chern marker, dense reference**
  (`src/algorithms/topology_realspace/chern_marker.{c,h}`). Implements
  the Bianco-Resta real-space Chern marker
  `c(r) = -(4π/A_c) Im Σ_s <r,s|P X Q Y P|r,s>` on the Qi-Wu-Zhang
  2-band Chern insulator. Projector onto the filled band is computed
  via Schulz iteration on the matrix sign function
  `P = (I - sign(H))/2`, so no eigendecomposition is needed -- this is
  the same algorithmic path the upcoming KPM/MPO implementation will
  take at 10^6+ sites. Verified: C(bulk) = ±1 in topological regimes
  (m = ±1) and 0 in trivial regimes (|m| = 3) on L=14/L=10 open
  lattices; the existing `hermitian_eigen_decomposition` was confirmed
  to silently corrupt complex-Hermitian inputs (real-Givens only),
  which is why the sign-function path is used here. First ground-truth
  reference for the plan's §2I headline real-space topology capability.
- **Clifford-backed surface code** (`surface_code_clifford_t` in
  `src/algorithms/topological/topological.{c,h}`). A parallel variant
  of the dense `surface_code_t` that tracks stabilizers on the new
  Aaronson-Gottesman tableau. d=15 needs 617 qubits (impossible on the
  dense simulator) and is tractable on the tableau. Syndrome
  measurement is ancilla-mediated: CNOTs onto a dedicated ancilla
  (plus H-wrap for X-stabs), Z-basis measurement, ancilla reset.
  Verified at d=3, 7, 9, 15: a single X error on a data qubit flips
  exactly the four Z-syndromes of the vertices adjacent to it;
  adjacent pairs correctly annihilate at shared vertices.
  Public entry points: `surface_code_clifford_{create,free}`,
  `_apply_error`, `_measure_{x,z}_syndromes`, `_syndrome_weight`.
- **Quantum Volume harness** (`src/applications/quantum_volume.{c,h}`).
  IBM spec (Cross et al., PRA 100, 032328, 2019): width = depth = d,
  d layers of floor(d/2) Haar-random U(4) blocks on shuffled pairs,
  exact heavy-output probability per circuit, mean HOP over N trials
  with 97.5% CI lower bound and pass/fail against the 2/3 threshold.
  Mezzadri 2007 Haar-U(4) generator (Gram-Schmidt + phase correction).
  Noiseless statevector passes widths 3..10 at 100 trials: mean HOP
  converges to the theoretical (1+ln2)/2 ≈ 0.847 with shrinking
  stddev (0.089 at w=3, 0.007 at w=10). `unit_quantum_volume` and
  `bench_quantum_volume` runners.
- **Gate-fusion DAG** (`src/optimization/fusion/`). Builds a circuit as
  a symbolic list (`fuse_circuit_t`), then `fuse_compile` merges runs
  of consecutive single-qubit gates on the same qubit into a single 2x2
  `FUSED_1Q` node. Each two-qubit gate flushes pending 2x2 accumulators
  on the qubits it touches. Measured on a 5-layer HWEA VQE circuit at
  n=16: 315 -> 155 gates, 14.8 ms -> 6.8 ms (2.18x). At n=20:
  395 -> 195, 239 ms -> 167 ms (1.43x). Random-circuit parity to
  L² ≤ 1e-10 vs gate-by-gate. Header: `src/optimization/fusion/fusion.h`.

### Fixes

- **QPE bit-ordering**. `qpe_bitstring_to_phase` now uses the correct
  `y / 2^m` mapping and the IQFT path emits a trailing bit-reversal
  swap. T-gate test recovers φ = 1/8 exactly (confidence 1.0000).
- **MPI distributed state vector**. `partition_state_{create,wrap}`
  read a range from the MPI bridge that was never populated, so every
  rank allocated a zero-size buffer. Now computed from
  `total_amplitudes / size`. Cross-partition H + CNOT + SWAP +
  Toffoli verified on `np=4`.
- **H₂O Pauli Hamiltonian**. 23 malformed strings (wrong length or
  embedded spaces) were silently dropped by `pauli_hamiltonian_add_term`.
  Fixed; H₂O now populates all 38 declared terms.
- **QAOA approximation ratio**. `qaoa_result_t::approximation_ratio`
  was never written (always 0). `qaoa_solve` now brute-force enumerates
  the Ising spectrum (n ≤ 20) and fills it in.
- **VQE COBYLA optimizer**. Inner `simplex[i]` malloc had no NULL
  check; bailed to a segfaulting memcpy on OOM. Now guarded.
- **SIMD dispatch actually reaches gates**. `gate_pauli_x`,
  `gate_pauli_y` and `gate_cnot` now call `simd_complex_swap` /
  `simd_multiply_by_i`. Measured CNOT at n=20: 678 → 214 µs/gate
  (3.17×).
- **`make install` headers were uninstallable**. Relative includes
  into `tools/profiler/` pointed outside the install tree; moved
  `performance_monitor.{c,h}` into `src/utils/` so the header graph
  closes.
- **libomp hardcoded `/opt/homebrew/opt/libomp/lib/libomp.dylib`**
  dependency. `install_name_tool` post-build rewrites it to
  `@rpath/libomp.dylib`; target carries `@loader_path/../lib` +
  Homebrew prefixes as rpath entries.
- **Rust `cargo test` / `cargo build` required `DYLD_LIBRARY_PATH`**.
  `build.rs` on the `moonlab` and `moonlab-tui` crates now emits
  `-Wl,-rpath,<lib_dir>`.
- **JS `Complex.conjugate({real:r, imag:0})`** returned `imag:-0`.
  Normalised to `+0`. `Circuit.calculateDepth` now skips `measure`
  gates (matches the docstring).
- **Python `moonlab.algorithms`**: removed dead bindings to
  non-existent C symbols; fixed use-after-free on re-solve
  (`molecular_hamiltonian_free` called on a `pauli_hamiltonian_t*`);
  `ml.py` now gates torch imports behind `try: import torch`.

### Tests

- Fibonacci anyon braiding: σ₁·σ₁⁻¹ = I verified exactly on a 4-tau
  fusion tree (logical-qubit subspace, dim=2).
- Grover multi-marked (k=3 on n=4): P(marked set) = 0.949 after one
  optimal iteration.
- Noisy VQE: depolarizing(p1=1e-3, p2=1e-2) runs to completion.
- MPI cross-partition H+CNOT, X+SWAP, X+X+Toffoli on np=4.
- Depolarizing(p=3/4) shot-averaged: 〈Z〉 → 0, P(|1⟩) → 0.5.
- MBL level-spacing-ratio pipeline invokes exact diagonalisation and
  `compute_level_statistics` end-to-end.

### Documentation

- `documents/api/c/vqe.md` documents the symmetry-preserving ansatz,
  `hf_reference` pre-condition, and `vqe_exact_ground_state_energy`.
- `src/quantum/state.h` and `src/distributed/mpi_bridge.h` carry
  `@thread-safety` annotations (neither is thread-safe; QRNG is the
  only documented thread-safe API).
- README retired unsubstantiated GPU/MPI speedup tables in favour of
  measured numbers from `bench_state_operations` and the rewritten
  `phase3_phase4_benchmark`.
- `qrng_nist_tests` docstring corrected: runs 3 of the 15 SP 800-22
  tests (monobit, runs, poker), not the full battery.

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
- **QRNG lag-1 serial correlation (rho_1 = 0.567 -> 0.0)**:
  `extract_quantum_entropy` in `src/applications/qrng.c` applied only
  four mixing gates between consecutive measurements. Because
  `quantum_measure_all_fast` collapses the state to a basis vector,
  those four gates were never enough to disperse the amplitudes back
  into the full 2^n-dimensional Hilbert space — most amplitudes
  stayed concentrated near the just-measured state, and the byte
  stream developed a reproducible lag-1 Pearson correlation of
  about 0.57. The new scramble applies one H per qubit, one
  entropy-seeded Rz per qubit, and a CNOT ring coupling all qubits
  between every measurement. On 256 KiB of output the correlation
  now falls within the 5/sqrt(N) ideal-uniform bound (rho_1 ~ 0.002)
  while chi^2 and monobit tests remain healthy.
- **QPE state-preparation multiplied by zero** in
  `src/algorithms/qpe.c`: `qpe_estimate_phase` was multiplying each
  full-state amplitude by the eigenstate amplitude at the
  corresponding system_part instead of tensoring the |+>^m
  precision register with the eigenstate. For any eigenstate with
  no |0>^n component (e.g. the standard T-gate test case |1>) this
  zeroed the QPE state outright and every subsequent measurement
  collapsed to the all-ones bitstring. Fixed to replace amplitudes
  with `(1/sqrt(2^m)) * eigenstate[system_part]`.
- **ASAN race in `health_test_apt`**: `entropy_pool` was calling
  `health_tests_run_batch` from both the background worker thread
  and the direct-generation fallback without shared locking. Added
  `pool->health_mutex` to serialise the two paths. ASAN heap-
  buffer-overflow is now gone.
- **VQE performance**: `vqe_measure_pauli_expectation` was
  sampling 10,000 shots per Pauli term (cloning the full state
  each shot). For ideal simulation the expectation value is
  analytic — O(state_dim * num_qubits) — so the sampling path is
  now the non-default fallback. `vqe_compute_energy` on an H2
  Hamiltonian dropped from 480 ms to 1 ms per call; 1000 ADAM
  iterations now finish in under 3 seconds and converge to
  E = -1.121 Ha, about 4 mHa below the Hartree-Fock reference.
- **Floating-point safety under `-ffast-math`**: `INFINITY` /
  `NAN` sentinels in `vqe.c` and `dmrg.c` are undefined behaviour
  under the library's default `-ffast-math`. Replaced with
  finite `DBL_MAX` sentinels (`VQE_ENERGY_ERROR`,
  `DMRG_ENERGY_ERROR`). `isnan(E)` checks rewritten to
  `!isfinite(E) || E >= ENERGY_ERROR`.
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

- Verified clean-slate build on macOS arm64 in Release, Debug,
  `-DQSIM_WERROR=ON`, and `-DQSIM_ENABLE_MPI=ON` configurations.
- CTest count grew from 11 (of which only 9 passed) to **34** (all
  passing) on a non-MPI macOS arm64 build, and **35** (all passing)
  on a `-DQSIM_ENABLE_MPI=ON` build. `long_evolution` (~7 min) is
  included and passes.

New tests added and wired into CTest:

- `unit_constants` — pins every hex-encoded constant in `src/utils/constants.h`
  to its advertised value at 1e-14 tolerance.
- `unit_correctness_properties` — property-based physics checks:
  norm preservation over random circuits, rotation-inverse
  idempotence, involution of H/X/Z/CNOT squared, bit-exact |Phi+>
  amplitudes from `create_bell_state_phi_plus`.
- `unit_measurement` — projective-collapse conventions, Bell-state
  measurement correlations, `<Z>` expectations, full-distribution
  normalisation.
- `unit_entanglement` — product-state zero entropy, Bell-state
  maximal (1 ebit) entropy, fidelity identical/orthogonal, purity
  across pure states, partial trace of |Phi+> yields rho = I/2.
- `unit_noise` — bit-flip / phase-flip involution, depolarizing p=0
  is no-op, amplitude damping preserves norm, pure dephasing
  preserves populations, norm-preservation across every channel.
- `unit_grover` — success probability >= 0.8 at n = 3, 4, 5 qubits.
- `unit_vqe` — H2 Hamiltonian construction, hardware-efficient
  ansatz build, and a single energy-evaluation smoke through
  `vqe_compute_energy`.
- `unit_qaoa` — square-graph 4-qubit MaxCut solver, approximation
  ratio in [0, 1].
- `unit_qpe` — phase <-> bitstring round-trip, T-gate eigenphase
  invocation (smoke level).
- `unit_chemistry` — molecular_hamiltonian lifecycle,
  `hartree_fock_state` places the right electron count,
  `uccsd_config` lifecycle.
- `unit_topological` — Fibonacci / Ising anyon systems, surface code
  and toric code lifecycle.
- `unit_mbl` — XXZ Hamiltonian + sparse form lifecycle.
- `unit_simd_parity` — Accelerate primitives match a scalar C
  reference at 1e-12 L2-rel; 64-byte alignment confirmed.
- `unit_metal_parity` — Metal GPU kernels (Hadamard, Pauli X, Pauli
  Z) match CPU gates at single-precision tolerance (5e-5 L2-rel);
  SKIPs when Metal unavailable.
- `unit_lattice_2d` — square / triangular / honeycomb lattice
  construction and snake/grid mapping round-trip.
- `unit_skyrmion` — `braid_path_circular` waypoint + total-time smoke.
- `python_bindings_smoke` — Bell state built through the ctypes
  bridge and amplitudes verified against exact 1/sqrt(2).
- `rust_bindings_smoke` — full `cargo test` on `bindings/rust/moonlab`
  (lib + 9 doctests + 44 integration tests) — green.
- `webgpu_unified_smoke` — runs the pre-existing
  `scripts/webgpu-unified-smoke.mjs` randomized parity script.
- `distributed_gates` (MPI build) — `mpirun -np 4` rank/size query,
  allreduce_sum_double, `mpi_sendrecv` round-trip, barrier.

Previously-dormant test sources wired in:

- `unit_tensor_network` (833-line TN suite, already present in
  `tests/unit/test_tensor_network.c` but not in CMake).
- `comprehensive` (577-line end-to-end suite).
- `dmrg`, `mps_vs_exact`, `long_evolution`, `fast_measurement`.

Linked into the library so the TN / chemistry / topological / MBL /
visualization tests can actually resolve their symbols:

- Added `src/algorithms/tensor_network/*.c` (11 files) to
  `QSIM_ALGORITHMS_SOURCES`. Previously the tree had 8 KLOC of
  tensor-network code that was not compiled into `libquantumsim`, so
  every consumer of `tensor_create`, `dmrg_solve`, `tdvp_step`, etc.
  linked to undefined symbols.
- Added `src/algorithms/chemistry/chemistry.c`,
  `src/algorithms/topological/topological.c`,
  `src/algorithms/mbl/mbl.c`,
  `src/visualization/circuit_diagram.c`,
  `src/visualization/feynman_diagram.c`.

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
