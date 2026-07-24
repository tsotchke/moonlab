# Moonlab Quantum Simulator

[![Version](https://img.shields.io/badge/version-1.2.0-blue)]() [![Bell Test](https://img.shields.io/badge/CHSH-violates%20classical-success)](https://en.wikipedia.org/wiki/CHSH_inequality) [![State Vector](https://img.shields.io/badge/State%20Vector-32%20qubits-blue)]() [![PQC](https://img.shields.io/badge/PQC-ML--KEM%20512%2F768%2F1024-brightgreen)]() [![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)]() [![Sanitizers](https://img.shields.io/badge/ASAN%20%2B%20UBSAN-clean-brightgreen)]()

> **Full-stack quantum simulation + quantum-safe cryptography: dense
> state vector (32 qubits), tensor networks, Clifford tableau,
> topological QC, chemistry / VQE with native autograd, error
> mitigation, a Bell-gated and SHAKE256-conditioned hybrid random-byte
> engine, and a FIPS 203 post-quantum KEM seeded by that engine.**

## Community And Commercial Use

This repository is the public **Moonlab Community Edition**. It remains a
useful MIT-licensed simulator/runtime for local use, research, education,
public integrations, and commercial embedding. Commercial Moonlab/QGTL products
should build around this public core with hosted execution, private provider
overlays, enterprise deployment, billing/audit hooks, support, and certified
packages.

See [COMMUNITY_EDITION.md](COMMUNITY_EDITION.md) for the public/private product
boundary.

## Current release: v1.2.0 (2026-07-23)

Version 1.2 hardens the native CUDA state-vector backend and adds bounded MPI
sharding beyond 32 qubits. Distributed gates exchange fixed-size chunks rather
than full remote shards, and the release fleet gate binds an N=33 four-rank,
two-host proof to the exact clean commit and observed 2+2 topology. Ordinary
states remain on the CPU path with 1.x-compatible behavior.

The stable ABI has advanced from 0.3.0 through 0.4.0 (`moonlab_vqe_gradient`,
exposing exact adjoint gradients for supported noise-free ansaetze and
analytic parameter shift otherwise) to **0.6.0**, which adds
`moonlab_qrng_get_status`, honest QRNG capability bits, a certification-
language scrub, hidden-visibility exports for the binding-consumed surface, and
the topology/CA-MPS additions documented in the stable ABI guide. The release
also ships native Windows x64 and ARM64 packages built with ClangCL; both are
tested as relocatable external CMake packages before upload.

| 1.2 deliverable | Contract |
|---|---|
| Bounded CUDA/MPI sharding | Chunked remote gates and expectations with exact N=33, four-rank, two-host attestation |
| Stable ABI 0.6.0 | Hidden-visibility-safe native and binding surface, QGT topology one-shots, and CA-MPS conjugate Pauli |
| Numerical and concurrency hardening | Sanitizer, TSan, no-LAPACK SVD, large-n differential, and adversarial control-plane gates |
| Native CUDA state vector | GPU lifecycle, host/device sync, probabilities, norms, and transparent gate dispatch |
| Windows distribution | `windows-x64.zip` and `windows-arm64.zip` with DLL, import library, headers, and CMake exports |

See [the v1.2.0 release notes](docs/release/v1.2.0-release-notes.md),
[Windows guide](docs/WINDOWS.md), and [full changelog](CHANGELOG.md#120---2026-07-23).

## v1.0 platform foundation

**Open-core extension surfaces.**  Four runtime registries let private
overlays, sibling libraries (QGTL / libirrep / SbNN), and customer
applications plug new behavior into a stock moonlab build without
touching its source:

| Surface                                          | What it adds                                                                                                                  |
|--------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `moonlab_register_backend`                       | New execution backends (live hardware, GPU cluster, alternative simulator) -- dispatch by name from `moonlab_job_set_backend`. |
| `moonlab_register_vendor_noise_profile`          | Live calibration scrapers push device snapshots into the registry; backends look profiles up by name at execute time.          |
| `moonlab_register_decoder`                       | Custom QEC decoders join the same dispatcher as the five slots (`GREEDY`, `MWPM_EXACT`, `SBNN`, `LIBIRREP_SS`, `PYMATCHING`) -- only `GREEDY` and `MWPM_EXACT` are built-in; the other three link an external library and return `MOONLAB_DECODER_NOT_BUILT` otherwise. |
| `moonlab_scheduler_set_completion_hook`          | Synchronous hook for billing meters, audit logs, customer dashboards -- fires after every successful run with `(num_qubits, total_shots, backend_name)`. |

Demo `examples/extensions/open_core_overlay_demo.c` exercises all four
surfaces in one executable (ctest-gated).  Each registry is bound from
Python, Rust, and JS -- see `bindings/{python,rust,javascript}` for the
language-specific signatures.  A public-CI hygiene grep rejects any
`PROPRIETARY:` / `TSOTCHKE-INTERNAL:` markers landing in the public
moonlab tree.

**Multi-tenant productization (second wave).**  The control plane
now speaks an `AUTH <tenant_id>:<hmac>` wire form, propagates the
tenant identity through to the scheduler completion hook for
billing attribution, and exposes a pre-dispatch admission hook that
private overlays use for per-tenant quotas, paid-tier gating, and
emergency lockouts.  Five additional surfaces shipped:

| Surface                                              | What it adds                                                                                  |
|------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `moonlab_control_submit_circuit_auth_tenant`         | C / Python / Rust / JS client emits `AUTH <tenant>:<hmac>\n CIRCUIT <N>\n <body>`.            |
| `moonlab_control_server_set_admission_hook`          | Pre-dispatch refusal (`-405`/`-408`) on quotas, tier gates, lockouts.  Bound from Python + Rust. |
| `moonlab_scheduler_set_request_context` + getters    | Tenant_id + request_id flow into the scheduler thread-local; completion hook reads both.      |
| `moonlab_token_bucket_*`                             | Lock-free per-tenant rate-limit primitive for admission hooks.  Native Python + Rust ports.    |
| `docs/operations/{RUNBOOK,FLEET_DEPLOYMENT}.md`      | SRE runbook + fleet-rollout guide; measured throughput 4674 req/sec Bell 2q HMAC @ P99=1.92ms (single run, single host -- see "Performance numbers" under Current limitations). |

Runnable Python overlay at `examples/extensions/python_overlay_demo.py`
boots an HMAC-secured control plane with a per-tenant TokenBucket-backed
admission hook, submits Bell circuits from three tenants + one banned
tenant + one over-quota tenant, and prints the resulting billing ledger.

## New in v0.7 (2026-05-19)

**v0.7.0** ships the distributed-scheduler MVP -- the first piece
of moonlab's cloud-platform foundation.  `moonlab_job_t` carries
a circuit description + shot count + worker fan-out;
`moonlab_scheduler_run` splits the shots across N OpenMP workers
in-process today, with the same contract shaped to swap MPI /
gRPC / HTTP/2 as the transport in v0.7.1+.  4-worker Bell circuit
verified at 1024 shots: 505 / 519 / 0 -- perfect Bell correlation
across workers.  Schema-versioned JSON job spec
(`moonlab/job/v0.7.0`) enables over-the-wire dispatch.

## New in v0.6 (2026-05-19)

**v0.6.2** extends the CSS-code handle to 6 QEC families behind one
opaque type.  Surface code, toric, 2D color (Steane + Hamming),
IBM bivariate-bicycle qLDPC (the three Bravyi-Nature 627 "Gross"
codes), and hypergraph-product CSS codes.  Every instance is
validated against its published [[n, k, d]] shape.  The
`SurfaceCode` JS / Python / Rust bindings shipped in v0.5.12-14
become QEC-zoo dispatchers in v0.6.3 -- one binding, eight code
families.

**v0.6.1** extends the v0.6.0 bridge with two substantial entry
points.  First, `moonlab_libirrep_heisenberg_sector_e0()` runs
sector-resolved Heisenberg ground-state ED on the orbit-representative
basis: lattice -> space group -> rep table at fixed Sz -> full-reorth
Lanczos with `irrep_heisenberg_apply_in_sector` as the matvec.  At
N = 24 kagome 4x2 the sector dim is ~337k vs 16 777 216 full Hilbert,
making N > 14 ground-state ED a workstation problem instead of an
mpo_to_matrix OOM.  Second, a CSS-code handle layer
(`moonlab_libirrep_qec_t`) starts wrapping libirrep's 18-module QEC
zoo behind one opaque type.

**v0.6.0** opens the libirrep-integration arc.  libirrep is a
production-grade C library covering 18 QEC codes (toric, surface,
color, bivariate bicycle, hypergraph + lifted product, honeycomb +
CSS Floquet, 3D toric, X-cube fracton, HaPPY, single-shot,
Bacon-Shor, Steane * Steane, BdG-skyrmion), rep-theory primitives
(SO(3) / SU(2) / O(3) / SE(3)), and a verified spin-1/2 Heisenberg
sector-ED stack.  Moonlab has been treating it as a paper reference
(the `LIBIRREP_KAGOME12_E0 = -5.44487522` constant in the kagome
tests is a number copied out of libirrep's `PHYSICS_RESULTS.md`);
this release wires in the first real link.

Build with `-DQSIM_ENABLE_LIBIRREP=ON`; detection tries
`find_package(libirrep CONFIG)`, then `pkg-config libirrep`, then
`-DQSIM_LIBIRREP_ROOT=<path>` / `$LIBIRREP_ROOT` env-var pointing
at a source tree with `build/lib/liblibirrep.{a,dylib,so}`.  When
detected, `tests/unit/test_kagome_ed` re-derives the libirrep
reference at runtime (live `irrep_lanczos_eigvals_reorth` on the
`irrep_heisenberg_t` built from `irrep_lattice_build(KAGOME, 2,
2)`) and confirms agreement to machine precision (3.553e-15
disagreement with moonlab's MPO + zheev path).  Default is OFF;
when off the bridge compiles to stubs returning
`MOONLAB_LIBIRREP_NOT_BUILT` and the test prints `(skipped)`.

Next phases: v0.6.1 routes `moonlab_surface_code_clifford_*`
through `irrep_surface_*` + `irrep_css_code_t`, picking up the
other 17 codes for free behind the same JS / Python / Rust binding
surface that v0.5.12-14 shipped.  v0.6.2 expands to wrappers for
the toric / color / BB / X-cube / Floquet / HaPPY family.  v0.6.3+
exposes Clebsch-Gordan + reduction tables to the existing QGT /
DMRG paths.

## New in v0.5 (2026-05-19)

The v0.5 cycle is a production-quality push across all three
language bindings: every algorithm class that exists in Python now
has a working Rust analog and a working JavaScript / WebAssembly
wrapper, with end-to-end integration tests gated by CI.

- **WASM build resurrection (v0.5.0)** — a silent `_Static_assert`
  break that landed in v0.2.4 had been quietly failing the
  emscripten build for ~12 days.  `MOONLAB_MAX_QUBITS` is now
  adaptive (30 on wasm32, 32 on native 64-bit hosts); the
  ABI-export module is split into a qrng-free lean half that the
  WASM build can include, surfacing `moonlab_qwz_chern` +
  `moonlab_abi_version` + DMRG / CA-MPS / Z2-LGT shims to JS.

- **JS binding parity (v0.5.4 - v0.5.6)** — Bell tests / Grover /
  VQE / QAOA / topology are all callable from `@moonlab/quantum-
  core` now.  Hardware entropy backed by `crypto.getRandomValues()`
  via a WASM-only `hardware_entropy_wasm.c` shim unblocks every
  shot-noise-sampling C entry point.  Eight topological-invariant
  helpers in `topology.ts` cover QWZ Chern (3 integrators), SSH
  winding, Kitaev BdG Z_2, Kane-Mele / BHZ Z_2, Hofstadter sub-band
  Chern.

- **JS integration test gate (v0.5.1 + v0.5.3)** — 141 vitest
  integration tests cover every TS wrapper that previously had
  zero coverage; CI now runs them after every WASM rebuild, so
  the kind of silent symbol-export break that bit v0.5.0 cannot
  recur.

- **Rust examples coverage (v0.5.7 + v0.5.8)** — 3/14 -> 14/14
  modules have a runnable `cargo run --example` demo with
  textbook-correct output: CHSH = 2.82 on the Bell pair, H2
  exact ground at -1.142 Ha, Mermin-Klyshko hits the GHZ ideal
  `2^((n-1)/2)` at every n in 2..5, triangle MaxCut at p=3
  converges to 100% approximation ratio, etc.

- **Kane-Mele Rashba silent-correctness fix (v0.5.9)** — the
  C-side `qgt_model_kane_mele` had been silently dropping the
  `lambda_r` parameter, returning the S_z-conserving Z_2
  invariant when the caller passed non-zero Rashba.  Now rejects
  `lambda_r != 0.0` rather than emitting wrong physics; the full
  Pfaffian-based Rashba Z_2 stays on the v0.3.1 milestone list.

- **Rust build hygiene (v0.5.2)** — 326 `unused-unsafe` warnings
  collapsed to 0 by setting the crate-level `unsafe_code` lint to
  `allow` (matching `moonlab-sys`; the FFI surface is unsafe by
  construction).

Full version-by-version notes in [CHANGELOG.md](CHANGELOG.md#050---2026-05-19).

## New in v0.3.0 (2026-05-08)

- **Matrix-product density operator (MPDO) noise simulator** — polynomial-
  cost simulation of noisy circuits with named single-qubit channels
  (depolarising, amplitude damping, phase damping, bit/phase/bit-phase
  flip).  Up to ~100 qubits at single-qubit error rates of 1e-3 in
  quasi-1D layouts.  See `src/quantum/noise_mpdo.{c,h}`.

- **n-band quantum geometric tensor (QGT) module** — opaque-handle
  multi-band Bloch-Hamiltonian primitives + three Berry-grid Chern
  integrators (eigvec FHS, parallel-transport, projector-trace,
  rigorously gauge-free).  Z_2 invariant for 4-band TR-symmetric and
  1D BdG systems.  Cross-checked against the existing real-space
  Bianco-Resta `chern_marker` on QWZ.

- **New topological-band-structure model primitives**: Kane-Mele
  (4-band honeycomb QSH), BHZ (4-band square-lattice TI), Kitaev
  p-wave chain (1D BdG topological superconductor), Harper-Hofstadter
  (q-band magnetic-flux lattice).  Every model reproduces its
  analytical phase boundary exactly.  Hofstadter sub-band Chern
  numbers match the canonical TKNN values.

- **Critical Haldane fix** — the existing 2-band Haldane Hamiltonian's
  NNN antisymmetric sum vanished at the Dirac points in this
  primitive-coord convention; restored to the canonical form, which
  reproduces the textbook `|M| < 3*sqrt(3)*|t2*sin(phi)|` boundary.

See [CHANGELOG.md](CHANGELOG.md#030---2026-05-08) for the full
breakdown.

Moonlab v0.2.0 introduced an end-to-end quantum-simulation-to-PQC
pipeline: `moonlab_qrng_bytes` generates the seeds consumed by
ML-KEM-512 / 768 / 1024 convenience wrappers through the stable ABI.
The release path is now stronger than the original v0.2 implementation:
it continuously health-tests hardware/OS entropy, rejects failed simulated
Bell epochs before delivery, and domain-separates and conditions every
request with SHAKE256. Alongside the cryptography work, 0.2 closes the
Phase 1/2 "completeness" items from the release plan — error
mitigation (ZNE + PEC), POVM measurement, weak measurement, Mermin /
Mermin-Klyshko Bell inequalities, quantum mutual information,
composite and correlated noise channels, DI-QRNG primitives — and
extends the native reverse-mode autograd with controlled rotations
and integrates it directly into the VQE driver.  See
[CHANGELOG.md](CHANGELOG.md) for the per-subsystem state.

> **v0.2.x note**: `hermitian_eigen_decomposition` in `src/utils/matrix_math.c`
> now correctly handles both real-symmetric and complex-Hermitian inputs
> (residual ||H v − λv|| < 1e-14 on the 2×2 smoke; consumed by the
> dense-ED path in `vqe_exact_ground_state_energy`).  The earlier
> real-Givens limitation was fixed in v0.2.0; the README warning is
> retained here as a deprecation note for downstream code that may
> have copied the old workaround.

## Highlights

| Capability | Description |
|------------|-------------|
| **State Vector Engine** | Up to 32 qubits with AMX-aligned buffers + runtime-dispatched SIMD (AVX-512 / AVX2 / NEON / SVE + Apple Accelerate). |
| **Tensor Networks** | MPS, DMRG (2-site with subspace expansion), TDVP, MPO-2D, lattice-2D. Real-space topology via MPO Chebyshev-KPM: local Chern marker on generic 2D models matches dense reference to machine precision. |
| **Clifford-Assisted MPS** | Hybrid `\|psi> = D \|phi>` representation that stores the Clifford part as an O(n) tableau and only the non-Clifford residual as an MPS.  64× bond-dim advantage + 13884× speedup over plain MPS on stabilizer circuits at n=12 (measured once, single host -- not a portable guarantee). Variational-D ground-state search (`ca_mps_var_d.{c,h}`) alternates a greedy local-Clifford D-update with imag-time on `\|phi>` — TFIM/XXZ/kagome AFM oracles ship in `examples/tensor_network/`. CA-PEPS 2D scaffold (`ca_peps.{c,h}`) lands the public API for v0.3. |
| **Gauge-Aware Warmstart** | Aaronson-Gottesman symplectic-Gauss-Jordan Clifford builder (`ca_mps_var_d_stab_warmstart.{c,h}`): takes any list of commuting Pauli generators on n qubits (LGT Gauss-law operators, surface/toric-code stabilizers, repetition-code stabilizers) and emits an O(n^2)-gate Clifford that places `\|0^n>` in the simultaneous +1 eigenspace.  First HEP application: 1+1D Z2 lattice gauge theory (`src/applications/hep/lattice_z2_1d.{c,h}` + `examples/hep/z2_gauge_var_d.c`). |
| **Clifford Backend** | Aaronson–Gottesman tableau simulator: O(n) gates, O(n²) measurement. 3200-qubit GHZ + all-qubits measurement in ~100 ms (measured once, single host). |
| **Chemistry / VQE** | Jordan-Wigner, UCCSD + hardware-efficient ansatz, H₂/LiH/H₂O Pauli Hamiltonians. Native reverse-mode autograd (CRX/CRY/CRZ + all standard rotations); `vqe_compute_gradient` uses adjoint method for HEA noise-free paths — ~5× over parameter-shift on 12 params, linear scaling to 100+. |
| **Quantum Algorithms** | Grover, VQE, QAOA, QPE, CHSH + Mermin + Mermin-Klyshko Bell tests, Shor-ECDLP resource estimator (Gidney/Drake/Boneh 2026). |
| **Error Mitigation** | Zero-noise extrapolation (linear / Richardson / exponential) and probabilistic error cancellation (PEC) primitives. |
| **Measurement** | Projective, POVM (with Kraus-completeness verification), weak-Z measurement with tunable strength, partial, non-collapsing expectations. |
| **Entanglement Metrics** | Von Neumann entropy, Rényi-α, concurrence, negativity, mutual information I(A:B), Schmidt decomposition. |
| **Post-Quantum Cryptography** | FIPS 202 SHA-3 + SHAKE (all KATs pass), FIPS 203 ML-KEM 512 / 768 / 1024 with Fujisaki-Okamoto and implicit rejection, plus QRNG-sourced keygen / encapsulate wrappers. |
| **Quantum RNG** | Thread-safe conditioned hybrid RNG: continuously health-tested hardware/OS entropy, fail-closed simulated Bell epochs, SHAKE256 conditioning, live assurance status, plus Pironio-bound and Toeplitz research primitives. |
| **Noise Models** | Depolarising, amplitude damping, phase damping, bit/phase-flip, thermal relaxation, composite, convex-mixture, correlated two-qubit Pauli. |
| **GPU Acceleration** | Native CUDA state-vector backend (x86-64 NVIDIA and Jetson). Metal compute kernels on macOS (Hadamard / CNOT / probability reduction). WebGPU backend scaffolded. |
| **Distributed State Vector** | Bounded MPI + CUDA sharding beyond 32 qubits: remote one/two-qubit gates, CNOT, and distributed X/Y expectations exchange fixed-size chunks. Release-gated by an N=33 four-rank, two-host fleet proof. |
| **Multi-Language** | C core + Python (ctypes) / Rust / JavaScript bindings.  Python exposes quantum + crypto primitives; 297 pytest test functions. |

## Table of Contents

- [Quick Start](#quick-start)
- [State Vector Simulation](#state-vector-simulation)
- [Tensor Network Methods](#tensor-network-methods)
- [Quantum Algorithms](#quantum-algorithms)
- [Topological Quantum Computing](#topological-quantum-computing)
- [Skyrmion Braiding](#skyrmion-braiding)
- [Quantum Chemistry](#quantum-chemistry)
- [Many-Body Localization](#many-body-localization)
- [Post-Quantum Cryptography](#post-quantum-cryptography)
- [Error Mitigation](#error-mitigation)
- [Additional Shipped Surfaces](#additional-shipped-surfaces)
- [Language Bindings](#language-bindings)
- [Performance](#performance)
- [Building](#building)
- [Documentation](#documentation)
- [Citation](#citation)

## Quick Start

MoonLab is published to the standard registries. The supported package-manager
installs are:

```bash
# Homebrew tap (repository: https://github.com/tsotchke/homebrew-moonlab)
brew tap tsotchke/moonlab
brew install moonlab

# Self-contained Python wheel
pip install moonlab

# JavaScript/WebAssembly core
npm install @moonlab/quantum-core

# Rust TUI (the Homebrew SDK supplies the native library)
cargo install moonlab-tui
```

The Homebrew tap setup is a one-time command. Formula updates are performed by
the release workflow only after building and testing the formula from source.

```bash
# Build (CMake, the canonical path on 0.1.2+)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run the full test suite (~20s, long_evolution adds ~7min and is opt-in)
ctest --test-dir build -E long_evolution --output-on-failure

# Try an example
./build/bell_test_demo
```

Windows uses ClangCL because the simulator relies on C99 complex arithmetic.
The repository driver builds, tests, packages, and consumer-verifies either
native architecture:

```powershell
.\scripts\build_windows_artifact.ps1 `
  -Arch x64 `
  -BuildDir build-windows-x64 `
  -Output .\dist\moonlab-local-windows-x64.zip
```

See [docs/WINDOWS.md](docs/WINDOWS.md) for ARM64 and release-package usage.

Warnings-as-errors CI build (clean on macOS arm64):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQSIM_WERROR=ON
cmake --build build -j
```

Sanitized build (AddressSanitizer + UndefinedBehaviorSanitizer):

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DQSIM_ENABLE_SANITIZERS=ON
cmake --build build-asan -j
ASAN_OPTIONS="detect_leaks=0:halt_on_error=1" \
UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1" \
  ctest --test-dir build-asan -E "long_evolution|python_bindings|rust_bindings|webgpu_unified" \
        --output-on-failure --timeout 300
```

Distributed (MPI) build:

```bash
brew install open-mpi          # macOS; apt-get install -y libopenmpi-dev on Ubuntu
cmake -S . -B build-mpi -DQSIM_ENABLE_MPI=ON
cmake --build build-mpi -j
ctest --test-dir build-mpi -E long_evolution     # mpirun -np 4 distributed_gates
```

The legacy Makefile path (`make all` / `make test`) still works for a
subset of targets, but the CMake build is the source of truth for CI,
warnings discipline, sanitizer builds, install, and all new language
bindings.

### Your First Quantum Program

```c
#include "quantum/state.h"
#include "quantum/gates.h"

int main(void) {
    // Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    quantum_state_t state;
    quantum_state_init(&state, 2);

    gate_hadamard(&state, 0);
    gate_cnot(&state, 0, 1);

    // Verify entanglement (subsystem A = qubit 0)
    int qubit_a = 0;
    double entropy = quantum_state_entanglement_entropy(&state, &qubit_a, 1);
    printf("Entanglement entropy: %.4f (max 1.0)\n", entropy);

    quantum_state_free(&state);
    return 0;
}
```

## State Vector Simulation

Full state vector representation for exact quantum simulation.

### Capabilities

- **32 Qubits**: 4.3 billion amplitude state space (68 GB on high-memory systems)
- **Universal Gates**: Pauli, Hadamard, Phase, Rotation, CNOT, Toffoli, QFT
- **Entanglement Metrics**: Von Neumann entropy, purity, fidelity, partial trace
- **Secure Measurements**: Cryptographic entropy from hardware RNG

### Memory Requirements

| Qubits | Amplitudes | Memory |
|--------|------------|--------|
| 20 | 1,048,576 | 16 MB |
| 24 | 16,777,216 | 256 MB |
| 28 | 268,435,456 | 4.3 GB |
| 32 | 4,294,967,296 | 68.7 GB |

## Tensor Network Methods

Polynomial-scaling simulation for systems beyond state vector limits.

### Matrix Product States (MPS)

```c
#include "algorithms/tensor_network/tn_state.h"
#include "algorithms/tensor_network/tn_gates.h"

// Create 100-qubit MPS with bond dimension 64
tn_state_config_t config = tn_state_config_create(64, 1e-10);
tn_mps_state_t* mps = tn_mps_create_zero(100, &config);

// Apply gates (automatic SVD truncation)
tn_apply_h(mps, 0);
tn_apply_cnot(mps, 0, 1);

// Measure expectation values
double magnetization = tn_expectation_z(mps, 50);
```

### DMRG Ground State

```c
#include "algorithms/tensor_network/dmrg.h"

// Heisenberg chain Hamiltonian as an MPO (100 sites, J=1, Delta=1)
mpo_t* hamiltonian = mpo_heisenberg_create(100, 1.0, 1.0, 0.0);

dmrg_config_t config = dmrg_config_default();
config.max_bond_dim = 128;
config.max_sweeps = 20;
config.energy_tol = 1e-10;

dmrg_result_t* result = dmrg_ground_state(mps, hamiltonian, &config);
printf("Ground state energy: %.10f\n", result->ground_energy);
dmrg_result_free(result);
```

### TDVP Time Evolution

```c
#include "algorithms/tensor_network/tdvp.h"

// Real-time dynamics: two-site TDVP by default (config.variant == TDVP_TWO_SITE)
tdvp_config_t tdvp_cfg = tdvp_config_default();
tdvp_cfg.dt = dt;

tdvp_engine_t* engine = tdvp_engine_create(mps, hamiltonian, &tdvp_cfg);
tdvp_evolve_to(engine, dt * num_steps, NULL);
tdvp_engine_free(engine);
```

### 2D Tensor Networks

```c
#include "algorithms/tensor_network/lattice_2d.h"
#include "algorithms/tensor_network/mpo_2d.h"

// Create 10x10 square lattice with open boundaries
lattice_2d_t* lattice = lattice_2d_create(10, 10, LATTICE_SQUARE, BC_OPEN);

// Apply 2D MPO Hamiltonian
hamiltonian_params_t params = hamiltonian_params_skyrmion_default();
params.J = J_coupling;
mpo_t* H = mpo_2d_create(lattice, &params);
```

### Clifford-Assisted MPS (CA-MPS)

Hybrid representation `|psi> = C |phi>` where `C` is a Clifford unitary
tracked by the Aaronson-Gottesman tableau and `|phi>` is an MPS.  Clifford
gates update only the tableau (O(n) bit ops); non-Clifford gates push
Pauli-string rotations into the MPS factor.  For stabilizer-state-like
circuits CA-MPS uses a bond dimension of 1 regardless of qubit count,
giving up to 64x bond-dim advantage + 13884x speedup over plain MPS at
n=12 (see `tests/performance/bench_ca_mps.c`).

```c
#include "algorithms/tensor_network/ca_mps.h"

// 12-qubit CA-MPS with max MPS bond dim 32
moonlab_ca_mps_t* s = moonlab_ca_mps_create(12, 32);

// Clifford gates: tableau-only, no MPS cost
for (uint32_t q = 0; q < 12; q++) moonlab_ca_mps_h(s, q);
moonlab_ca_mps_cnot(s, 0, 1);

// Non-Clifford: pushed into MPS factor via Pauli-string rotation MPO
moonlab_ca_mps_rz(s, 3, 0.7);

// Imaginary-time ground-state search (non-unitary primitive)
moonlab_ca_mps_imag_pauli_rotation(s, (uint8_t[]){3,3,0,0,0,0,0,0,0,0,0,0}, 0.01);
moonlab_ca_mps_normalize(s);

// Expectation value of an observable Pauli string
double _Complex e;
uint8_t p[12] = {3,0,0,0,0,0,0,0,0,0,0,0};  // Z_0
moonlab_ca_mps_expect_pauli(s, p, &e);
```

See `docs/research/ca_mps.md` for the full theory, gate-application rules,
and benchmark methodology.

### Variational-D (var-D) ground-state search

CA-MPS is also a ground-state-search representation: `|psi_GS> ~ D|phi>`
where `D` is a Clifford basis transform chosen to absorb the
stabilizer-rich entanglement of the target Hamiltonian, leaving `|phi>`
as a low-entanglement MPS.  `moonlab_ca_mps_optimize_var_d_alternating`
implements the alternating optimisation: greedy local-Clifford D-update
+ imag-time `|phi>` evolution.

```c
#include "algorithms/tensor_network/ca_mps_var_d.h"

ca_mps_var_d_alt_config_t cfg = ca_mps_var_d_alt_config_default();
cfg.warmstart                 = CA_MPS_WARMSTART_DUAL_TFIM;  // H_all + CNOT chain
cfg.max_outer_iters           = 25;
cfg.imag_time_steps_per_outer = 4;
cfg.clifford_passes_per_outer = 8;
cfg.composite_2gate           = 1;  // 2-gate composite moves to escape local minima

ca_mps_var_d_alt_result_t res = {0};
moonlab_ca_mps_optimize_var_d_alternating(
    state, paulis, coeffs, num_terms, &cfg, &res);
// res.final_energy, res.final_phi_entropy, res.total_gates_added, ...
```

Warmstart options bias the greedy search toward known-productive Clifford
basins:
- `IDENTITY` -- D starts at I.
- `H_ALL` -- product of H on every qubit.
- `DUAL_TFIM` -- H_all then CNOT chain (Kramers-Wannier dual basis).
- `FERRO_TFIM` -- H + CNOT chain (cat-state encoder).
- `STABILIZER_SUBGROUP` -- gauge-aware: see below.

Validated workloads (in `examples/tensor_network/`): 1D TFIM,
1D XXZ Heisenberg (`ca_mps_var_d_heisenberg.c`), kagome 12-site
frustrated AFM (`ca_mps_var_d_kagome12.c`), comparison vs plain DMRG
(`ca_mps_var_d_vs_plain_dmrg.c`).

### Gauge-aware stabilizer-subgroup warmstart

For Hamiltonians whose physical sector is the +1 eigenspace of a
commuting set of Pauli generators (lattice gauge theory Gauss-law
operators, surface/toric/repetition-code stabilizers, any abelian
symmetry projector), the warmstart Clifford `D_S` can be built
exactly via Aaronson-Gottesman symplectic Gauss-Jordan.  `D_S|0^n>`
is then in the simultaneous +1 eigenspace of every generator and
the var-D loop only has to capture the residual non-stabilizer
dynamics on top.

```c
#include "algorithms/tensor_network/ca_mps_var_d_stab_warmstart.h"

// Generators g_0, ..., g_{k-1} as (k, n) row-major Pauli bytes
// (0=I, 1=X, 2=Y, 3=Z), pairwise commuting and independent.
moonlab_ca_mps_apply_stab_subgroup_warmstart(state, generators, k);
// state->D now stabilises the +1 eigenspace of every g_i.
```

First HEP application: 1+1D Z2 lattice gauge theory.
`src/applications/hep/lattice_z2_1d.{c,h}` builds the matter +
gauge-link Pauli sum; `examples/hep/z2_gauge_var_d.c` runs the
full var-D pipeline with the gauge-aware warmstart.  Math write-up:
`docs/research/var_d_lattice_gauge_theory.md`.

### CA-PEPS (2D)

`src/algorithms/tensor_network/ca_peps.{c,h}` ships the 2D extension
of CA-MPS via a row-major-MPS embedding over `Lx * Ly` qubits.
Clifford gates (H, S, Sdag, X, Y, Z, CNOT, CZ) update the tableau
in-place; non-Clifford rotations and the T family push into the
inner MPS factor the same way CA-MPS does.  Validation lives in
`tests/unit/test_ca_peps.c`: runs a mixed Clifford + rotation
circuit on a 3x3 lattice through both `moonlab_ca_peps_*` and the
equivalent CA-MPS linear index, and checks every Pauli-string
expectation agrees to <1e-10.  Wired into ctest as `unit_ca_peps`.

## Quantum Algorithms

### Grover's Search

```c
#include "algorithms/grover.h"

grover_config_t config = {
    .num_qubits = num_qubits,
    .marked_state = marked_state,
    .use_optimal_iterations = 1
};
grover_result_t result = grover_search(&state, &config, entropy);
// O(√N) queries vs classical O(N)
```

### Variational Quantum Eigensolver (VQE)

```c
#include "algorithms/vqe.h"

vqe_ansatz_t* ansatz = vqe_create_hardware_efficient_ansatz(num_qubits, /*num_layers=*/2);
vqe_optimizer_t* optimizer = vqe_optimizer_create(VQE_OPTIMIZER_LBFGS);
vqe_solver_t* solver = vqe_solver_create(hamiltonian, ansatz, optimizer, entropy);

vqe_result_t result = vqe_solve(solver);
printf("Ground state energy: %.8f Ha\n", result.ground_state_energy);
```

Optimizers: `VQE_OPTIMIZER_COBYLA`, `VQE_OPTIMIZER_LBFGS`, `VQE_OPTIMIZER_ADAM`,
`VQE_OPTIMIZER_GRADIENT_DESCENT`, and `VQE_OPTIMIZER_QNG` (quantum natural
gradient -- preconditions the parameter-shift gradient by the regularised
Fubini-Study metric from `vqe_compute_qgt` / `vqe_natural_gradient_direction`).

### QAOA (Combinatorial Optimization)

```c
#include "algorithms/qaoa.h"

// MaxCut problem
ising_model_t* ising = ising_encode_maxcut(graph);
qaoa_solver_t* solver = qaoa_solver_create(ising, num_layers, entropy);
qaoa_result_t result = qaoa_solve(solver);
printf("Best energy: %.4f (bitstring 0x%llx)\n",
       result.best_energy, (unsigned long long)result.best_bitstring);
```

### Quantum Phase Estimation

```c
#include "algorithms/qpe.h"

qpe_result_t result = qpe_estimate_phase(unitary, eigenstate, precision_qubits, entropy);
double phase = result.estimated_phase;
```

### Bell Test Validation

```c
#include "algorithms/bell_tests.h"

/* CHSH on a Bell pair */
bell_test_result_t r = bell_test_chsh(&state, 0, 1, 10000, NULL, entropy);
printf("CHSH: %.4f (classical <= 2, quantum <= 2.828)\n", r.chsh_value);

/* Mermin polynomial on |GHZ_3>: classical <= 2, quantum max 4 */
bell_test_result_t m = bell_test_mermin_ghz(&ghz3, 0, 1, 2, 10000, entropy);
printf("Mermin |M|: %.4f\n", m.chsh_value);

/* Mermin-Klyshko M_N on |GHZ_N>, normalised so classical <= 1,
   quantum max 2^((N-1)/2). */
double mk = bell_test_mermin_klyshko(&ghz_n, N, 0, NULL);
```

## Topological Quantum Computing

Fault-tolerant quantum computation using anyonic systems.

### Anyon Models

```c
#include "algorithms/topological/topological.h"

// Create Fibonacci anyon system
anyon_system_t* sys = anyon_system_fibonacci();

// Build a fusion tree over the external anyon charges
fusion_tree_t* tree = fusion_tree_create(sys, charges, num_anyons, total_charge);

// Braid adjacent anyons (topological gate): exchanges the anyons at
// `position` and `position + 1`, applying the R-matrix phase and any
// F-matrix basis change to tree->amplitudes in place.
braid_anyons(tree, /*position=*/i, /*clockwise=*/true);
```

### Supported Anyon Types

| Model | Anyons | Universal | Application |
|-------|--------|-----------|-------------|
| Fibonacci | τ, 1 | Yes | Universal TQC |
| Ising | σ, ψ, 1 | No (+ magic) | Majorana fermions |
| SU(2)_k | Multiple | Varies | General TQC |

### Surface Codes

```c
// Create distance-5 surface code
surface_code_t* code = surface_code_create(5);
surface_code_init_logical_zero(code);

// Measure stabilizers
surface_code_measure_X_stabilizers(code);
surface_code_measure_Z_stabilizers(code);

// Decode and correct errors (minimum-weight perfect matching)
qs_error_t status = surface_code_decode_correct(code);
```

A separate Clifford-tableau-backed variant (`surface_code_clifford_t`,
`surface_code_clifford_create(distance, rng_seed)`) simulates the same
protocol in O(n) per gate via the Aaronson-Gottesman tableau and scales
to distance 15+ where the dense `surface_code_t` above is capped near
distance 5 by state-vector memory.

### Toric Codes

```c
// Create toric code on a 6x6 torus
toric_code_t* toric = toric_code_create(6);
toric_code_init_ground_state(toric);

// Create an e-anyon pair via a Z-string between two vertices
toric_code_create_anyon_pair(toric, 'e', 0, 0, 3, 3);

// Braid the anyon pair around each other
toric_code_braid(toric, 0, 0, 3, 3);
```

### Topological Invariants

```c
// Compute topological entanglement entropy from an annulus split into
// three regions A, B, C (S_topo = S_A + S_B + S_C - S_AB - S_BC - S_AC + S_ABC)
double gamma = topological_entanglement_entropy(&state,
                                                  region_A, num_A,
                                                  region_B, num_B,
                                                  region_C, num_C);
// γ = log(D) where D is total quantum dimension
```

## Skyrmion Braiding

Magnetic skyrmion-based topological qubits using real-time dynamics.

### Theory

Skyrmions are topologically protected magnetic structures that can encode quantum information through their braiding. This implementation follows [Psaroudaki & Panagopoulos, Phys. Rev. Lett. 127, 067201 (2021)].

### Usage

```c
#include "algorithms/tensor_network/skyrmion_braiding.h"

// Initialize two skyrmions
skyrmion_t sk1 = { .x = 0.0, .y = 0.0, .charge = 1 };
skyrmion_t sk2 = { .x = 2.0, .y = 0.0, .charge = 1 };

// Define a circular path around the midpoint for one skyrmion to encircle the other
braid_path_t* path = braid_path_circular(/*center_x=*/1.0, /*center_y=*/0.0,
                                          /*radius=*/1.0, BRAID_COUNTERCLOCKWISE,
                                          /*num_segments=*/64, /*velocity=*/1.0);

// Encode a topological qubit in the skyrmion pair (bond_dim=32 MPS)
hamiltonian_params_t params = hamiltonian_params_skyrmion_default();
topo_qubit_t* qubit = topo_qubit_create(lattice, &params,
                                         sk1.x, sk1.y, sk2.x, sk2.y, 32);

// Perform braiding with TDVP time evolution
braid_config_t braid_cfg = braid_config_default();
braid_result_t* result = skyrmion_braid(qubit->mps, qubit->mpo, qubit->lat,
                                         path, &braid_cfg);

// Extract Berry phase (argument of the total accumulated phase)
printf("Berry phase: %.6f\n", carg(result->phase));
printf("Braid succeeded: %s\n", result->success ? "yes" : "no");
```

### Topological Gates

```c
// Apply topological gates via skyrmion braiding
topo_gate_apply(qubit, TOPO_GATE_BRAID, &braid_cfg);        // exp(i*pi/4*sigma)
topo_gate_apply(qubit, TOPO_GATE_DOUBLE_BRAID, &braid_cfg);  // i*sigma
```

## Quantum Chemistry

Molecular simulation with fermionic mappings.

### Jordan-Wigner Transformation

```c
#include "algorithms/chemistry/chemistry.h"

// Jordan-Wigner transform of the hopping term a†_p a_q
fermion_op_t ops[2] = {
    { .type = FERMION_CREATE,     .orbital = p },
    { .type = FERMION_ANNIHILATE, .orbital = q }
};
jw_operator_t hopping = jw_transform_product(ops, 2, num_orbitals);
// hopping.terms[i] is a Pauli string with a double _Complex coefficient
```

### UCCSD Ansatz

```c
// Build UCCSD configuration for molecular simulation
uccsd_config_t* config = uccsd_config_create(/*num_orbitals=*/2, /*num_electrons=*/2);

quantum_state_t state;
quantum_state_init(&state, config->num_orbitals);
hartree_fock_state(&state, config->num_electrons, config->num_orbitals);
uccsd_apply(&state, config);
```

### Molecular Hamiltonians

```c
// H2 molecule in minimal basis (STO-3G, Jordan-Wigner) as a ready-made
// Pauli Hamiltonian -- the direct path VQE consumes.
pauli_hamiltonian_t* H = vqe_create_h2_hamiltonian(bond_length);

vqe_ansatz_t* ansatz = vqe_create_hardware_efficient_ansatz(2, 2);
vqe_optimizer_t* optimizer = vqe_optimizer_create(VQE_OPTIMIZER_LBFGS);
vqe_solver_t* solver = vqe_solver_create(H, ansatz, optimizer, entropy);

vqe_result_t result = vqe_solve(solver);
printf("H2 energy: %.6f Ha\n", result.ground_state_energy);
```

The lower-level geometry path (`molecule_h2(bond_length)` -> `molecule_t*`
-> `molecular_hamiltonian_create` / `molecular_to_qubit_hamiltonian`) builds
the same Hamiltonian from Cartesian atom coordinates for callers who need
custom geometries rather than the pinned equilibrium bond length.

## Many-Body Localization

Disordered quantum systems and thermalization dynamics.

### Disordered Heisenberg Model

```c
#include "algorithms/mbl/mbl.h"

// Create XXZ Hamiltonian with strong disorder (MBL phase)
xxz_hamiltonian_t* H = xxz_hamiltonian_create(/*num_sites=*/16,
                                               /*J=*/1.0, /*delta=*/1.0,
                                               /*disorder_strength=*/5.0,
                                               /*periodic_bc=*/false,
                                               /*seed=*/42);

sparse_hamiltonian_t* sparse = xxz_build_sparse(H);
sparse_hamiltonian_diagonalize(sparse);
```

### Diagnostics

```c
// Level statistics (Poisson vs GOE)
level_statistics_t* stats = compute_level_statistics(sparse->eigenvalues,
                                                       sparse->dim,
                                                       /*filter_edges=*/0.1);
printf("<r> = %.4f\n", stats->mean_ratio);
// <r> ~ 0.39 (Poisson, MBL) vs <r> ~ 0.53 (GOE, thermal)

// Entanglement entropy dynamics for an 8-site subsystem
uint32_t subsystem[8] = {0, 1, 2, 3, 4, 5, 6, 7};
entropy_dynamics_t* dyn = simulate_entropy_dynamics(sparse, &initial_state,
                                                      subsystem, 8,
                                                      /*t_max=*/20.0,
                                                      /*num_steps=*/200);
```

## Post-Quantum Cryptography

Moonlab v0.2 ships a reference implementation of FIPS 202 (SHA-3,
SHAKE) and FIPS 203 (ML-KEM — the NIST-standardised
module-lattice-based KEM). Convenience wrappers obtain seeds from the
health-tested, Bell-gated, SHAKE256-conditioned `moonlab_qrng_bytes` path.
Regulated deployments can instead use the explicit-seed APIs with an approved
DRBG at their validated module boundary. Three parameter sets are
available: ML-KEM-512 (NIST Category 1), ML-KEM-768 (recommended
default), and ML-KEM-1024 (Category 5).

```c
#include <moonlab/moonlab_export.h>

uint8_t ek[MOONLAB_MLKEM768_PUBLICKEYBYTES];
uint8_t dk[MOONLAB_MLKEM768_SECRETKEYBYTES];
uint8_t ct[MOONLAB_MLKEM768_CIPHERTEXTBYTES];
uint8_t K_alice[32], K_bob[32];

// Entropy is drawn from moonlab_qrng_bytes internally.
moonlab_mlkem768_keygen_qrng(ek, dk);
moonlab_mlkem768_encaps_qrng(ct, K_bob, ek);
moonlab_mlkem768_decaps(K_alice, ct, dk);
// K_alice == K_bob
```

Python:

```python
from moonlab.crypto import mlkem
ek, dk   = mlkem.keygen768_qrng()
ct, K_a  = mlkem.encaps768_qrng(ek)
K_b      = mlkem.decaps768(ct, dk)
assert K_a == K_b
```

All NIST FIPS 202 known-answer vectors pass byte-for-byte (SHA-3 224 /
256 / 384 / 512, SHAKE128, SHAKE256 including split-squeeze).  ML-KEM
conformance is validated at two tiers: a self-regression KAT (12
SHA3-256 fingerprints of every artifact at fixed (d, z, m) seeds,
across all three parameter sets) and a NIST-seeded KAT that drives
our in-tree AES-256 SP 800-90A CTR_DRBG from the published NIST
count=0 seed through KeyGen + Encaps.  A FIPS 203 reviewer can hash
the official PQCkemKAT .rsp artifacts with SHA3-256 and compare to
the pinned fingerprints -- match establishes conformance.

Security posture: this is a reference implementation -- constant-time
on non-exotic CPUs, not FIPS-140-certified, and not hardened for
adversarial side-channel environments.  It is suitable for learning,
for integrating the QRNG source into a PQC workflow, and for research
on quantum-safe primitives.  For FIPS-certified production crypto,
integrate with BoringSSL or OpenSSL EVP; the QRNG seed path still
applies.

See `examples/applications/pqc_qrng_demo.c` for a ~100-line
end-to-end demo and `docs/security/pqc.md` for the threat model.

## Error Mitigation

A new `src/mitigation/` subsystem with the two workhorse techniques
for current-generation NISQ hardware:

```c
#include <quantumsim/mitigation/zne.h>

// Suppose fn(lambda, ctx) runs the circuit with noise scaled by lambda
// and returns the measured <O>.
double scales[] = { 1.0, 1.5, 2.0, 3.0 };
double sd = 0.0;
double E_mitigated = zne_mitigate(fn, ctx, scales, 4,
                                   ZNE_EXPONENTIAL, &sd);
```

Three estimators: linear (OLS intercept fit), Richardson (exact
Lagrange interpolation at lambda = 0 -- zero residual on polynomials
of degree <= n-1), and exponential (fit E = a + b exp(-c lambda),
recovers depolarised `<Z>` to 1e-13 in the integration test).

Probabilistic error cancellation primitives (`pec_one_norm_cost`,
`pec_sample_index`, `pec_aggregate`) provide the Monte-Carlo
machinery for caller-supplied quasi-probability decompositions of
inverse noise channels.

## Additional Shipped Surfaces

A few modules ship in the tree without a dedicated walkthrough above.

- **`@moonlab/quantum-algorithms`** (npm) -- a lean, browser-friendly
  package built on `@moonlab/quantum-core`'s WASM state vector. Ships a
  `Grover` class (WASM-backed amplitude amplification, up to 26 qubits) and
  an H2-only `VQE` class that runs a classical grid-search + refinement
  optimizer over a closed-form single-parameter H2 UCCSD ansatz -- it does
  not call into the WASM state vector and does not implement QAOA.
- **`@moonlab/quantum-viz`** (npm) -- Canvas 2D and WebGL 3D quantum-state
  visualizations (`BlochSphere`, `AmplitudeBars`, `CircuitDiagram`) usable
  standalone, independent of the React/Vue framework bindings.
- **`moonlab.ml`** (Python) -- quantum feature maps (angle / amplitude /
  IQP encoding), a `QuantumKernel` + `QSVM` classifier, and a `QuantumPCA`
  dimensionality reducer built on the state-vector primitives. Gradients do
  **not** flow back through these quantum operations (the state is evolved
  imperatively and results are detached); use `moonlab.torch_layer` for a
  trainable quantum layer.
- **`moonlab.torch_layer`** (Python) -- `QuantumLayer` and `QuantumConv1D`
  PyTorch modules trained via the exact parameter-shift rule
  (`ParameterShiftGradient`), documented above under VQE/PyTorch.
- **`moonlab.diff`** (Python) -- `DiffCircuit`, a native reverse-mode
  autograd circuit builder mirroring `src/algorithms/diff/differentiable.h`,
  so Python callers get adjoint gradients (`backward_pauli_sum`) without a
  PyTorch dependency.
- **`moonlab::feynman`** (Rust) -- `FeynmanDiagram` + `ParticleType`
  renderer that emits ASCII, SVG, and LaTeX/TikZ-Feynman diagrams for QFT
  processes (fermion/antifermion/photon/gluon/W/Z/Higgs/ghost/graviton
  lines), independent of the quantum-simulation surface.
- **`examples/applications/qgt_qec_node.c`** -- a single physics story
  tying the quantum geometric tensor to topological error correction: on
  the Qi-Wu-Zhang Chern insulator, the Fubini-Study metric divergence at
  the Dirac-node gap closing and the Chern-number jump are the same
  epsilon^2 = 0 nilpotency that makes a surface-code stabilizer chain
  complex (d1 . d2 = 0) well-defined.

## Language Bindings

### Python (with PyTorch Integration)

```python
import moonlab as ml
import torch
from moonlab.torch_layer import QuantumLayer

# Create quantum state
state = ml.QuantumState(4)
state.h(0).cnot(0, 1).cnot(1, 2).cnot(2, 3)

# PyTorch hybrid layer: parameterized quantum circuit as an nn.Module,
# trained via the exact parameter-shift rule (moonlab.torch_layer.ParameterShiftGradient)
model = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    QuantumLayer(num_qubits=8, depth=2),
    torch.nn.Linear(8, 2),
)

# Train with backpropagation
optimizer = torch.optim.Adam(model.parameters())
output = model(torch.randn(32, 4))
loss = output.sum()
loss.backward()  # quantum gradients computed automatically
```

`moonlab.torch_layer` also ships `QuantumConv1D` (a quantum kernel applied to
sliding windows of a 1D input, the quantum analogue of `nn.Conv1d`).

### VQE in Python

```python
from moonlab.algorithms import VQE

vqe = VQE(num_qubits=4, num_layers=2)
result = vqe.solve_h2(bond_distance=0.74)
print(f"Energy: {result['energy']:.8f} Ha")
# vqe.solve_lih(bond_distance=1.6) and vqe.solve_h2o() are also available
```

### Rust

```rust
use moonlab::QuantumState;

fn main() {
    let mut state = QuantumState::new(4).expect("allocate state");
    state.h(0).cnot(0, 1);

    let entropy = state.entanglement_entropy(&[0]).expect("entropy");
    println!("Entropy: {:.4}", entropy);
}
```

### JavaScript (React)

```jsx
import { useQuantumState, BlochSphere } from '@moonlab/quantum-react';

function QuantumVisualizer() {
    const { amplitudes, applyGate, loading } = useQuantumState({ numQubits: 1 });

    if (loading) return <div>Loading...</div>;

    return (
        <div>
            <BlochSphere amplitudes={amplitudes} />
            <button onClick={() => applyGate('h', 0)}>
                Hadamard
            </button>
        </div>
    );
}
```

`useQuantumState` returns `{ state, loading, error, amplitudes, probabilities,
numQubits, initialize, reset, applyGate, measure, measureAll, refresh, dispose
}`; circuit state (add/undo/export gates) is a separate hook, `useCircuit`.

### Vue

```vue
<template>
    <div>
        <button @click="handleBellState">Create Bell State</button>
        <CircuitDiagram :circuit="circuit" />
    </div>
</template>

<script setup>
import { useQuantumState, useCircuit } from '@moonlab/quantum-vue';

const { amplitudes, numQubits, applyGate } = useQuantumState({ numQubits: 2 });
const { circuit, addGate } = useCircuit({ numQubits: 2 });

const handleBellState = () => {
    applyGate('h', 0);
    applyGate('cnot', 0, 1);
    addGate('h', 0);
    addGate('cnot', 0, 1);
};
</script>
```

## Current limitations

Read this before judging the repo against its headline claims.  The
adversarial audit that produced this list lives in
`docs/audits/adversarial-review-2026-04-19.md`.

- **Chern mosaic**: the full Bianco-Resta local marker
  C(r) = -4 pi * Im Sum_orb <r, orb| P X Q Y P |r, orb>
  now runs end-to-end via the MPO pipeline on a real QWZ 2D
  Chern insulator at L = 4 and reproduces the dense Schulz
  reference to machine precision (|MPO - dense| = 0.0000 at a
  bulk site).  Position operators are the quantics-bit-weighted
  diagonal-sum MPOs.  The sparse-stencil renderer scales to
  L = 300 single-core.  Adaptive QTCI for non-monomial
  modulations is still future work; the linear-in-coordinate
  case (what the Bianco-Resta formula actually uses) is shipped.
- **CHSH / "Bell-verified" QRNG**: prior to 0.2.0 the
  `bell_test_chsh` function silently overwrote the input state with
  `|Phi+>` before measuring, so every CHSH reading was 2.828 by
  fiat.  Fixed this release.  The `moonlab_qrng_bytes`
  BELL_VERIFIED mode now runs its health check on a fresh `|Phi+>`
  temporary rather than on the QRNG's own evolving scratch state;
  treat the resulting CHSH number as a plumbing sanity check, not a
  proof of quantum advantage in the emitted bytes.
- **MPI**: the `distributed_gates` ctest runs at `mpirun -np 4`
  and exercises H, CNOT, SWAP, Toffoli, and a full GHZ chain
  across the partition boundary (norm preservation + specific
  amplitude checks to 1e-10).  What is **not** tested yet:
  multi-node (>1 physical host) scaling, wall-clock comparisons
  against single-host baselines, and any MPI backend other than
  OpenMPI.
- **GPU backends other than Metal, CUDA, and Eshkol**: OpenCL and Vulkan
  compile cleanly and pass a local compile + discovery smoke
  (apt's ocl-icd-loader + PoCL for OpenCL; vulkan-loader + lavapipe for
  Vulkan), but hosted CI does not currently build with either backend
  enabled and no CI runner exercises them against a real GPU -- the
  discovery smoke only verifies backend selection and fallback. The
  native CUDA state-vector path is validated out-of-band on Jetson and
  x86-64 NVIDIA fleet nodes at release time; cuQuantum remains an
  optional, separately provisioned backend.
- **WebGPU / JS**: the CI `wasm-js-tests` job builds moonlab.wasm and the
  TS `@moonlab/quantum-core` package, runs the full vitest unit +
  integration suites against the fresh module, and runs the WebGPU
  unified smoke.  Plain node has no WebGPU runtime, so the smoke
  verifies backend selection + fallback (backend=none), not real GPU
  execution.  The default C-only `ctest` on a fresh clone without
  `-DQSIM_BUILD_JS_DIST=ON` produces no WebGPU coverage -- that is the
  trade-off for not requiring a JS toolchain just to build the library.
- **Platforms**: hosted CI covers Linux x86-64 (plus Linux ARM64 in the
  Linux-compatibility matrix), macOS ARM64, Windows x64, and Windows
  ARM64. The Jetson CUDA workflow is manual-dispatch only -- no
  self-hosted runner is currently enrolled -- so Jetson CUDA coverage
  happens out-of-band via the release mesh smoke. Tagged releases
  additionally build Linux ARM64 and macOS Intel archives. Windows uses
  Visual Studio generators with ClangCL and runs a relocatable-package
  consumer smoke before upload.
- **Performance numbers**: every headline multiplier was measured
  once on one host.  No stddev, no cross-platform reproduction.
  Use the benches below to measure your own hardware; do not
  quote the repository's numbers as portable.

## Performance

The numbers historically quoted here (GPU speedups, MPI scaling, DMRG
wall-clocks) pre-date any reproducible harness that exercises the
full pipeline on a single host configuration, so they have been
retired pending the 0.2 benchmark work (Quantum Volume + CLOPS + XEB +
direct RB as described in `docs/release/` / `MOONLAB_RELEASE_ROADMAP.md`).

Runnable micro-benchmarks ship today:

```bash
./build/bench_state_operations          # dense SV gate throughput
./build/bench_tensor_networks           # MPS / DMRG micro-probes
./build/grover_parallel_benchmark       # Grover scaling across cores
./build/phase3_phase4_benchmark         # Metal kernel sanity
```

Use those to measure your own hardware. A comparative
regression harness against Qiskit-Aer / Qulacs / cuStateVec is
tracked for 0.2.

### Distributed (MPI)

State-vector sharding across MPI ranks is wired end-to-end in
`src/distributed/`: `dist_gate_1q`, `dist_hadamard`, `dist_pauli_*`,
`dist_cnot`, etc. handle both local-partition and cross-partition
gates with the necessary `MPI_Sendrecv` exchange.  The
`tests/integration/test_distributed_*` harnesses validate Bell + GHZ
round-trips on `mpirun -np 2..4`.  Cross-rank scaling above N = 32
qubits (which exercises the dist_* buffer-size path fixed in v0.8.0)
runs on dev hosts but has not yet been published with a peer-host
reproducible scaling table -- that's tracked as part of the
post-v1.0 scaling-honesty pass.

## Building

### Requirements

- **macOS**: 10.15+ on Intel; 11.0+ on Apple Silicon
- **Linux**: GCC 9+ with OpenMP
- **Windows**: Visual Studio 2022+ with CMake and the ClangCL toolset
- **Memory**: 8 GB minimum, 32 GB+ for large simulations

### Build Options

CMake is the canonical build system (`0.1.2+`). Useful options:

| Option | Default | Effect |
|---|---|---|
| `-DCMAKE_BUILD_TYPE=Release\|Debug\|RelWithDebInfo` | `Release` | Standard CMake build type |
| `-DQSIM_ENABLE_METAL=ON` | `ON` on macOS | Metal GPU backend |
| `-DQSIM_ENABLE_MPI=ON` | `OFF` | MPI distributed computing (OpenMPI) |
| `-DQSIM_ENABLE_OPENMP=ON` | `ON` | OpenMP multi-core |
| `-DQSIM_WERROR=ON` | `ON` | `-Werror` build with `-Wpedantic` / `-Wdeprecated-declarations` demoted (libomp + CLAPACK externalities) |
| `-DQSIM_ENABLE_SANITIZERS=ON` | `OFF` | AddressSanitizer + UndefinedBehaviorSanitizer |
| `-DQSIM_ENABLE_AVX512=ON` / `AVX2` / `NEON` / `SVE` | `ON` if available | SIMD path toggles |
| `-DQSIM_BUILD_TESTS=ON` | `ON` | CTest targets |
| `-DQSIM_BUILD_EXAMPLES=ON` | `ON` | `examples/` programs |
| `-DQSIM_BUILD_BENCHMARKS=ON` | `ON` | `benchmarks/` targets |

Legacy `make all` / `make test` still work for a subset of the surface.

### Dependencies

**Required**:
- C compiler (GCC/Clang on Unix; ClangCL on Windows)
- Threads implementation supplied by the platform toolchain

**Optional**:
- OpenMP (multi-core)
- Accelerate framework (macOS AMX)
- Metal (GPU, macOS)
- MPI (distributed)

**macOS OpenMP runtime**: installed `libquantumsim.dylib` artifacts reference
`@rpath/libomp.dylib` and carry only loader-relative rpath entries, so the
consuming process decides which OpenMP runtime satisfies the reference. This
prevents the duplicate-runtime abort (`OMP: Error #15`) when the host
application already ships its own libomp (conda, PyTorch, a different LLVM).
Consumers without an OpenMP runtime on their rpath either place
`libomp.dylib` next to `libquantumsim.dylib`, add libomp's directory to their
link rpath (`-Wl,-rpath,$(brew --prefix libomp)/lib`), or configure Moonlab
with `-DQSIM_EXTRA_RPATH=/path/to/libomp/lib` to pin a directory into the
installed artifact. The Homebrew formula pins libomp's opt path this way;
build-tree binaries reference Homebrew libomp absolutely and need none of it.

## Documentation

Start with the [documentation index](docs/README.md). Current guides include:

- [Getting started](docs/getting-started.md)
- [Windows builds and packages](docs/WINDOWS.md)
- [CI/CD pipelines](docs/CI_CD.md)
- [Tutorials](docs/tutorials/README.md)
- [Stable C ABI](docs/STABLE_ABI.md)
- [Configuration options](docs/reference/configuration-options.md)
- [Architecture](ARCHITECTURE.md) and [platform specification](PLATFORM.md)

## Project Structure

```
moonlab/
├── src/
│   ├── quantum/              # State vector engine
│   ├── algorithms/
│   │   ├── grover.c          # Grover's search
│   │   ├── vqe.c             # Variational eigensolver
│   │   ├── qaoa.c            # Quantum optimization
│   │   ├── qpe.c             # Phase estimation
│   │   ├── tensor_network/   # MPS, DMRG, TDVP, skyrmions
│   │   ├── topological/      # Anyons, surface codes
│   │   ├── chemistry/        # Jordan-Wigner, UCCSD
│   │   └── mbl/              # Many-body localization
│   ├── optimization/         # SIMD, Metal GPU, parallel
│   └── distributed/          # MPI communication
├── bindings/
│   ├── python/               # Python + PyTorch
│   ├── rust/                 # Rust FFI + TUI
│   └── javascript/           # React, Vue, WASM
├── examples/
│   ├── quantum/               # Grover search, GPU benchmarks
│   ├── applications/          # VQE (H2), QAOA (MaxCut), portfolio, QRNG/PQC, qgt_qec_node
│   ├── tensor_network/        # CA-MPS, CA-PEPS, DMRG spin chains, var-D
│   ├── topological/           # QGT models (Kane-Mele, BHZ, Hofstadter, Kitaev)
│   ├── cuda/                  # Native CUDA state-vector demos (QSIM_ENABLE_CUDA)
│   ├── distributed/           # MPI-sharded state vector
│   ├── hep/                   # Z2 lattice-gauge-theory var-D
│   └── extensions/            # Open-core plug-in surfaces (C + Python)
├── tests/                    # Test suite
└── docs/                     # Documentation
```

## Citation

If you use Moonlab in your research, please cite:

```bibtex
@software{tsotchke_moonlab_2026,
    author       = {tsotchke},
    title        = {{Moonlab}: A Quantum Computing Simulation Framework},
    year         = {2026},
    version      = {v1.2.0},
    url          = {https://github.com/tsotchke/moonlab},
    license      = {MIT},
    keywords     = {quantum computing, simulation, tensor networks,
                    Clifford-Assisted MPS, variational-D, lattice gauge
                    theory, Z2 LGT, gauge-aware warmstart,
                    topological quantum computing, DMRG, VQE, QAOA,
                    Chern insulators, quantum geometric tensor}
}
```

## References

**Foundational Textbooks and Reviews**:
- Nielsen, M.A. & Chuang, I.L. (2010). *Quantum Computation and Quantum Information*. Cambridge University Press.
- Preskill, J. (2018). Quantum computing in the NISQ era and beyond. *Quantum*, 2, 79.

**Quantum Algorithms**:
- Shor, P.W. (1994). Algorithms for quantum computation: discrete logarithms and factoring. *Proc. 35th FOCS*, 124-134.
- Grover, L.K. (1996). A fast quantum mechanical algorithm for database search. *Proc. 28th STOC*, 212-219.
- Peruzzo, A. et al. (2014). A variational eigenvalue solver on a photonic quantum processor. *Nat. Commun.* 5, 4213.
- Farhi, E., Goldstone, J., & Gutmann, S. (2014). A quantum approximate optimization algorithm. *arXiv:1411.4028*.
- Kitaev, A.Y. (1995). Quantum measurements and the Abelian stabilizer problem. *arXiv:quant-ph/9511026*.

**Tensor Networks**:
- White, S.R. (1992). Density matrix formulation for quantum renormalization groups. *Phys. Rev. Lett.* 69, 2863.
- Schollwöck, U. (2011). The density-matrix renormalization group in the age of matrix product states. *Ann. Phys.* 326, 96-192.
- Orús, R. (2014). A practical introduction to tensor networks. *Ann. Phys.* 349, 117-158.
- Vidal, G. (2003). Efficient classical simulation of slightly entangled quantum computations. *Phys. Rev. Lett.* 91, 147902.
- Haegeman, J. et al. (2016). Unifying time evolution and optimization with matrix product states. *Phys. Rev. B* 94, 165116.

**Topological Quantum Computing**:
- Kitaev, A. (2003). Fault-tolerant quantum computation by anyons. *Ann. Phys.* 303, 2-30.
- Nayak, C., Simon, S.H., Stern, A., Freedman, M., & Das Sarma, S. (2008). Non-Abelian anyons and topological quantum computation. *Rev. Mod. Phys.* 80, 1083.
- Fowler, A.G., Mariantoni, M., Martinis, J.M., & Cleland, A.N. (2012). Surface codes: Towards practical large-scale quantum computation. *Phys. Rev. A* 86, 032324.

**Skyrmion Physics**:
- Psaroudaki, C. & Panagopoulos, C. (2021). Skyrmion qubits: A new class of quantum logic elements. *Phys. Rev. Lett.* 127, 067201.

**Quantum Chemistry**:
- Jordan, P. & Wigner, E. (1928). Über das Paulische Äquivalenzverbot. *Z. Physik* 47, 631-651.
- McArdle, S., Endo, S., Aspuru-Guzik, A., Benjamin, S.C., & Yuan, X. (2020). Quantum computational chemistry. *Rev. Mod. Phys.* 92, 015003.

**Many-Body Localization**:
- Nandkishore, R. & Huse, D.A. (2015). Many-body localization and thermalization in quantum statistical mechanics. *Annu. Rev. Condens. Matter Phys.* 6, 15-38.
- Abanin, D.A., Altman, E., Bloch, I., & Serbyn, M. (2019). Colloquium: Many-body localization, thermalization, and entanglement. *Rev. Mod. Phys.* 91, 021001.

**Bell Tests and Foundations**:
- Bell, J.S. (1964). On the Einstein Podolsky Rosen paradox. *Physics Physique Физика* 1, 195-200.
- Clauser, J.F., Horne, M.A., Shimony, A., & Holt, R.A. (1969). Proposed experiment to test local hidden-variable theories. *Phys. Rev. Lett.* 23, 880.

**High-Performance Quantum Simulation**:
- Häner, T. & Steiger, D.S. (2017). 0.5 petabyte simulation of a 45-qubit quantum circuit. *Proc. SC17*, Article 33.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

**Moonlab** - From qubits to anyons, from state vectors to tensor networks.

*Built for researchers. Optimized for discovery.*
