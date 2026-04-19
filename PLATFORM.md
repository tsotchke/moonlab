# Moonlab Platform Specification

**Version:** 0.2.0-dev · **Status:** design contract · **Owner:** tsotchke

This document is the architectural contract for Moonlab as a
platform. It is the successor to `ARCHITECTURE.md` (descriptive) and
`AUDIT.md` (diagnostic): PLATFORM.md is *prescriptive*. Every commit
after this point is graded against what this document commits to.

## 0. Mission

**Moonlab is an instrument for doing new physics.**

That is the whole statement. The rest of this document is the
engineering consequence of taking it seriously.

Specifically: Moonlab is the simulator and estimator substrate for
real-space and momentum-space topology, quantum geometry, open and
driven quantum systems, quantum error correction, quantum hardware
design, and the cryptanalytic resource estimates that are the
near-term consequence of the above. It is the computational backend
for QGTL, SbNN, and the rest of the research stack. It is a library
for publication; it is also the kernel a commercial service sits on.

The research contribution is not "we reimplemented X." It is: we
designed a substrate on which a physicist can move from a new
equation to a converged simulation result in hours, at system sizes
that were previously unreachable. The engineering contribution is not
"we're fast." It is: we are measurably competitive with the
state-of-the-art simulator in each regime, and in the regimes where
we are unique (real-space topology, adaptive representation,
differentiable physics) we have no competition.

Moonlab *is* an HPC library — a quantum HPC library. Every substrate
is performance-engineered: vectorised CPU kernels, GPU offload where
the work is parallel enough, distributed execution where problem size
demands it, measurable roofline efficiency on every hot path. The
layer that lowers substrate operations onto actual silicon is
substrate #7 (§2.7); it is the reason we can commit to measured
numbers and not just claims.

## 1. Design Commitments

Every design decision in the tree must serve at least one of the
following. Decisions that serve none are deleted.

1. **New physics first.** A substrate choice that makes a new
   calculation possible outranks a substrate choice that makes an
   existing one 2× faster.
2. **Measured performance, not claimed performance.** Every
   perf-adjacent PR must ship a benchmark that justifies the claim in
   the PR description. The benchmark number goes into the CI
   dashboard. Moonlab is an HPC library; performance is a
   first-class concern at every substrate, not a downstream
   optimisation phase. A substrate is not considered complete until
   it has (a) vectorised CPU kernels with a measured peak-FLOPs
   fraction, (b) a GPU path wherever the work is parallel enough,
   (c) a distributed execution path where problem size demands it,
   and (d) CI-tracked roofline metrics.
3. **Peer-review-grade honesty.** No marketing language in source
   docstrings, no unbacked speedup claims, no subsystem labelled
   "working" that isn't tested against a reference. Every numerical
   tolerance in a test has a justification in `MATH.md`.
4. **Reference-traceable citations.** Every non-trivial algorithmic
   choice cites a verified primary reference. No hallucinated arXiv
   IDs; no "name-year" without a verified DOI or arXiv ID.
5. **Opaque handles, flat arrays at the boundary.** No Moonlab type
   crosses the ABI. QGTL, SbNN, and third-party consumers bind
   through `moonlab_export.h` or `ctypes` without vendoring Moonlab
   internals.
6. **Reproducibility is a platform feature, not a user problem.**
   Every run emits a manifest sufficient to rerun it bit-exactly.
7. **One source of numerical truth.** One eigensolver. One SVD. One
   complex type. One RNG cascade. One error-handling convention.
   Multiple implementations are permitted for performance; multiple
   *semantics* are not.
8. **Substrate before surface.** If adding a new module is easier
   than fixing the substrate underneath it, fix the substrate.

## 2. The Seven Substrates

These are the primitives on which every Moonlab capability sits. The
v0.3 release is the one in which all seven are first-class. Six of
them are *what* the library computes; one (§2.7) is *where and how*
the computation runs, and is the substrate that makes the "quantum
HPC library" claim literal rather than rhetorical.

### 2.1 Hamiltonian algebra (substrate #1)

A typed symbolic operator algebra. A `moonlab_op_t` is a tree of
primitive operators (qubit Pauli, bosonic creation/annihilation,
fermionic creation/annihilation, position, momentum, projector)
combined by sum, scalar product, tensor product, or commutator.
Every Hamiltonian — logical circuit, device pulse, molecular H,
lattice-field H, Floquet drive, dissipator — lives in this algebra.

**API sketch:**

```c
moonlab_op_t* moonlab_op_pauli(const char* pauli_string, size_t n);
moonlab_op_t* moonlab_op_number(size_t mode);
moonlab_op_t* moonlab_op_fermion_c(size_t mode, int dagger);
moonlab_op_t* moonlab_op_add(moonlab_op_t* a, moonlab_op_t* b);
moonlab_op_t* moonlab_op_mul(double complex z, moonlab_op_t* a);
moonlab_op_t* moonlab_op_commutator(moonlab_op_t* a, moonlab_op_t* b);
int           moonlab_op_normalise(moonlab_op_t* a);
void          moonlab_op_free(moonlab_op_t* a);
```

**Status today:** fragmented. Pauli-strings in VQE, MPO in DMRG,
stencil matvec in Chern-KPM, hardcoded matrices in tensor. Migrating
these to the unified algebra is a v0.3 item.

### 2.2 Representation abstraction (substrate #2)

A `moonlab_state_t` that is one of:

- `STATE_DENSE_SV` — dense state vector (current default; ≤ 32 qubits)
- `STATE_MPS` — matrix product state
- `STATE_MPDO` — matrix-product density operator (for noise)
- `STATE_CLIFFORD` — Aaronson-Gottesman tableau
- `STATE_FREE_FERMION` — Gaussian fermionic state (BdG covariance)
- `STATE_BOSONIC_FOCK` — truncated Fock-space CV
- `STATE_TREE` — tree tensor network
- `STATE_NEURAL` — parameterised neural-network ansatz (RBM, CNN,
  transformer, FermiNet-class)

The dispatcher chooses per operator based on (a) operator class
(Clifford, Gaussian, local, generic), (b) current representation,
(c) entanglement forecast, (d) caller hint. States can migrate
between representations on demand: Clifford → SV via magic-state
injection; SV → MPS via iterative SVD; MPS → Free-fermion if the
Jordan-Wigner image becomes Gaussian.

**This is Moonlab's single unique architectural claim.** No
mainstream simulator dispatches representations automatically.
Publishing this substrate alone is a top-tier methods paper.

**Status today:** each representation exists; the unifying
`moonlab_state_t` handle does not. v0.3 item.

### 2.3 Time-evolution integrator (substrate #3)

`moonlab_evolve(state, H, dt, method)` with `method` selecting:

| Method | When |
|---|---|
| Trotter (1st, 2nd, 4th order) | short-range H, small dt |
| Krylov / Lanczos exp | arbitrary H, moderate dim |
| TDVP (1-site, 2-site) | MPS + H as MPO |
| Commutator-Free Magnus (4th, 6th) | strongly time-dependent drive |
| Quantum trajectory (stochastic SE) | Lindblad, large Hilbert |
| Lindblad master equation | direct density-matrix evolution |
| MPDO dissipative | MPS + Lindblad |
| CV truncated Wigner | continuous variable, semiclassical |

Caller writes physics; substrate picks the integrator consistent
with the state representation and the Hamiltonian type.

**Status today:** TDVP exists; Trotter exists; Krylov is buried in
DMRG's Lanczos. Lindblad, Magnus, trajectory, MPDO, truncated
Wigner are all missing. v0.3 item.

### 2.4 Measurement and tomography (substrate #4)

Projective and POVM measurement; weak measurement; shadow
tomography; MPS tomography; neural-network tomography; process
tomography; randomised benchmarking. One API for forward
(state → statistics) and inverse (statistics → state estimate).

**Status today:** projective measurement and basic expectation
values exist. POVM, weak, shadow, NN tomography, process tomography
are missing. v0.3 item.

### 2.5 Compiler / IR (substrate #5)

A three-tier intermediate representation:

- **Logical IR**: device-agnostic gates; native to research code.
- **Physical IR**: hardware-topology-aware; SWAPs inserted,
  connectivity respected, noise-model annotated.
- **Pulse IR**: microwave / optical / laser schedule on a specific
  device Hamiltonian; the output of GRAPE / Krotov / CFME
  optimisation.

Passes: decomposition, fusion (existing), commutation-aware
reordering, noise-aware routing, Pauli-frame tracking, magic-state
distillation lowering, surface-code lowering, T-count minimisation.

Back-ends: Moonlab native execution, OpenQASM 3, QIR (LLVM-IR for
quantum), Qiskit circuit, pulse JSON.

**Status today:** the simulator has a gate API but no IR. Fusion
DAG is the first step. v0.3-v0.4 item.

### 2.6 Resource-tracking context (substrate #6)

A thread-local context accumulating every resource cost as the
computation proceeds: gate count, T-count, circuit depth, qubit
count, code distance, logical error, wall-clock, memory peak,
energy (where measurable), dollar cost. Threaded through every
library call; returned as a structured record at the top of the
call stack.

**Status today:** partially available in ad-hoc fields
(`qv_result_t`, `fuse_stats_t`, `shor_ecdlp_resources_t`).
Unifying into a single context that every substrate writes into is
a v0.3 item.

### 2.7 Execution backend (substrate #7)

**This is what makes Moonlab an HPC library.** The execution backend
is the layer that lowers a substrate operation -- a gate, a matvec,
an SVD, a tensor contraction, an MPO/MPS expectation -- onto actual
silicon. Every higher substrate calls through this layer; no higher
substrate hardcodes a kernel.

The backend surface:

| Tier | Target | Primitives |
|---|---|---|
| CPU | x86_64 + AArch64 | runtime-dispatched SIMD (AVX-512 / AVX2 / NEON / SVE), Apple Accelerate (AMX), OpenMP across cores, cache-aware blocking, NUMA-aware allocation |
| GPU | Apple Metal + NVIDIA CUDA + cuQuantum / cuStateVec + AMD HIP + WebGPU for browsers | parallel gate kernels, batched 2-qubit fusion, on-device SVD / QR, tensor contraction, measurement sampling |
| Distributed | MPI + NCCL / UCX | partitioned state vector, cross-partition gate application, collective measurement, load-balanced TN contraction |
| Heterogeneous | auto-dispatch | CPU/GPU/distributed chosen per op based on problem size, locality, and available hardware; migration between tiers mid-run when the cost model says so |

The backend exposes a narrow, audited API that each substrate
consumes:

```c
/* Apply a 1- or 2-qubit gate to a state at a given representation. */
moonlab_backend_result_t moonlab_bk_apply_gate(
    moonlab_state_t*   state,
    const moonlab_gate_t* gate,
    const moonlab_bk_hints_t* hints);   /* optional: prefer GPU, etc. */

/* Low-level complex BLAS-3 (row- or column-major) with a dispatchable
 * backend. Wraps Accelerate / MKL / cuBLAS / rocBLAS; never reimplements. */
moonlab_backend_result_t moonlab_bk_gemm(
    moonlab_cplx_t alpha,
    const moonlab_buffer_t* A, const moonlab_buffer_t* B,
    moonlab_cplx_t beta,
    moonlab_buffer_t* C);

/* Tensor contraction (einsum-like); dispatches to cuTensorNet where
 * available, falls back to batched GEMM otherwise. */
moonlab_backend_result_t moonlab_bk_contract(
    const char* einsum_spec,
    const moonlab_tensor_t* const* operands, size_t n_operands,
    moonlab_tensor_t* out);
```

**Principles governing the backend:**

1. **We wrap, we do not reinvent**, classical linear algebra. BLAS /
   LAPACK / Eigen / cuBLAS / cuSPARSE / cuTensorNet are dependencies,
   not reimplementations. Moonlab's value-add is the quantum-
   structure-aware dispatch on top, not the matmul.
2. **Every kernel has a measured peak-FLOPs / peak-bandwidth fraction**
   on its target. Roofline analysis for every hot path; CI tracks
   the regressions.
3. **SIMD is in the production path, not a parallel track.** The
   `stride_gates` v0.2 scaffolding gets wired into the dispatcher;
   the scalar loop in `gates.c` goes away.
4. **GPU is first-class, not optional.** Metal, CUDA, and WebGPU are
   on an equal footing with CPU; the same state migrates between
   them. cuStateVec / cuTensorNet / cuSOLVER are consumed where
   they help.
5. **Distributed is not a bolt-on.** MPI partitioning is a
   representation the dispatcher chooses when problem size > node
   capacity. Cross-partition gate application is a backend primitive,
   not a separate library surface.
6. **Backend choice is observable**, never hidden: every execution
   emits a log entry indicating which tier ran the op. Users can
   pin, audit, and reproduce.

**Status today:** fragmented and partial. CPU has a runtime SIMD
dispatcher that the production gate path does *not* use (the
stride_gates.c SIMD kernels exist but are not wired in). Metal has
single-qubit kernels only. CUDA / OpenCL / Vulkan / cuQuantum /
cuStateVec are declared options that have never been CI-tested
against real SDKs. MPI has a partitioned state vector that passes a
cross-partition smoke test but no measured scaling. WebGPU is at
Phase 3 of `webgpuplan.md`. The roofline story is zero: no kernel
has a measured FLOPs / bandwidth number.

Bringing substrate #7 to full first-class status is the single
biggest engineering investment in the v0.3 arc, and the one that
converts "good research code" into "quantum HPC library."

## 3. New-Physics Surface

These are the specific physics questions Moonlab is being designed
to answer. Each is:

- not well-served by existing simulators,
- reachable with the substrate above,
- a publishable result in its own right.

This list is deliberately concrete. Each entry names a physics
target, the substrate it rides on, and a success criterion.

1. **Real-space topology at 10⁸ sites.** C_n-symmetric quasicrystal
   Chern mosaics on a π-flux lattice (Antão, Sun, Fumega, Lado 2026).
   Substrate: (2) MPS representation of the projector, (3) KPM
   time-evolution, (4) local Chern marker measurement. Success: Fig.
   3 of the paper reproduced at 268 × 10⁶ sites in under 24 hours on
   a single workstation.
2. **Non-Hermitian topology.** Local topological invariants for
   PT-symmetric and dissipative lattice models (arXiv 2604.09725).
   Substrate: (1) non-Hermitian H in the algebra, (3) bi-orthogonal
   Lindblad / NH-Schroedinger evolution, (4) dual-QGT measurement.
   Success: reproduce the dual-QGT Chern marker of Dirac-node
   non-Hermitian topology on the honeycomb lattice.
3. **QGT as an optimiser primitive.** Natural-gradient VQE using
   the Fubini-Study metric as the preconditioner (Stokes-Izaac-
   Killoran-Carleo 2020). Substrate: (4) metric measurement, (3)
   parameter update. Success: ≥ 2× convergence-iteration reduction
   vs Adam on H₂, LiH, H₂O, and the transverse-field Ising model at
   its critical point.
4. **Adaptive-representation Hamiltonian evolution.** A state that
   begins as dense SV, auto-compresses to MPS when entanglement
   bound is met, auto-routes Clifford subcircuits through the
   tableau. Substrate: (2) entirely. Success: end-to-end Shor-30bit
   toy run at lower peak memory than any pure-representation path.
5. **Pulse-level device co-simulation.** Transmon-resonator
   Hamiltonian with crosstalk and realistic drive; GRAPE-optimised
   pulse sequence that implements a target two-qubit gate at
   measured 99.99 % fidelity on the simulator's noise model.
   Substrate: (1), (3) Lindblad + Magnus, (4) process tomography,
   (5) pulse IR.
6. **Device-aware fault-tolerant resource estimation.** Per-hardware
   physical-qubit count and wall-clock for a user-supplied logical
   circuit, given a device JSON. Substrate: (5) lowering, (6) cost
   tracking. Used by QGTL for scheduling and by enterprises for
   migration planning. Success: ±30 % agreement with Gidney-Drake-
   Boneh 2026 on secp256k1 ECDLP; already demonstrated for the
   aggregate numbers, device-per-node refinement next.
7. **Differentiable quantum physics.** Every Moonlab output
   (energy, expectation, fidelity, Chern number, entanglement
   entropy) is a PyTorch tensor with `.backward()`. Enables
   gradient-based Hamiltonian learning, quantum control, neural
   ansatz training. Substrate: (3), (4) with autograd hooks.
   Success: train a 4-qubit variational circuit to prepare a
   target state via end-to-end gradient descent from a Python
   loop that never leaves the simulator.
8. **Holographic / toy-AdS simulation.** HaPPY code, holographic
   error-correction codes, bulk-boundary reconstruction at
   publishable system sizes. Substrate: (1), (2) tree tensor
   network, (4) reduced-density-matrix measurement.
9. **Distributed quantum-compute federation.** A physical-layer
   cost model for remote CNOTs across a quantum interconnect;
   substrate for QGTL's scheduler. Substrate: (3) Lindblad + loss,
   (5) pulse IR + remote-gate compilation, (6) cost tracking.
10. **Honest logical/physical resource estimation for arbitrary
    ECDLP and factoring circuits.** Already shipping (see
    `src/algorithms/shor_ecdlp/`). Success: extend to Regev
    factoring, discrete-log on pairing-based curves, lattice
    cryptanalysis.

Each entry above is one paper. Several are multiple.

## 4. Performance Commitments

Every number below is a measurable target that goes into the CI
benchmark dashboard. The claim is only made if the dashboard shows
the number holding within 10 % on the canonical workload.

| Regime | Workload | Target | Reference |
|---|---|---|---|
| Single-qubit gate, CPU SIMD | H on n=26 | ≤ 25 µs | Qulacs 20 µs |
| Two-qubit gate, CPU SIMD | CNOT on n=26 | ≤ 30 µs | Qulacs 28 µs |
| CPU kernel efficiency | H on n=24 | ≥ 50 % of peak bandwidth | roofline |
| Clifford random, tableau | GHZ-10000 | ≤ 10 ms | Stim ~0.4 ms (stretch: parity) |
| DMRG, 1D Heisenberg | L=100, χ=200, 5 sweeps | ≤ 10 s | ITensor ~4 s |
| MPS gate, typical | H @ 50 sites, χ=32 | ≤ 2 µs | ITensor ~1 µs |
| Gate fusion speedup | HWEA at n=20 | ≥ 1.5× vs unfused | (measured) |
| Quantum Volume heavy-output | QV-10 mean HOP | ≥ 0.83 | Porter-Thomas 0.847 |
| Chern-KPM bulk site | L=1024 (1M sites) | ≤ 2 s per site | (no published ref) |
| FHS Chern integer | N=32 grid, QWZ | `lround(c) == C` | (analytical) |
| GPU gate throughput, Metal | H on n=26 | ≤ 5 µs | stretch: cuStateVec ~1 µs |
| GPU gate throughput, CUDA | H on n=28 | ≤ 10 µs | cuStateVec 3-8 µs |
| MPI scaling | n=32 GHZ on np=16 | ≤ 2× single-node at n=28 | mpiQulacs |
| Roofline utilisation | any hot kernel | ≥ 40 % of memory-bandwidth peak | measured |
| Reproducibility manifest | every run | emitted | (new) |
| Qulacs parity | 200 random circuits, n ≤ 16 | ‖Δψ‖ ≤ 1e-10 | (new) |
| Stim parity | 100 Clifford circuits | bit-exact | (new) |
| FCI parity | H₂, LiH, H₂O at equilibrium | ≤ 1 mHa | (new) |
| ITensor parity | DMRG on 1D Heisenberg | ground-state energy ≤ 1e-8 rel | (new) |

Every entry in this table is a CI dashboard item. A perf regression
opens an automatic issue; no PR that breaks a committed threshold
merges without a dedicated update to this section.

## 5. Migration Path

From v0.2.0-dev (today) to v0.3.0 (PLATFORM-compliant):

**Phase 1 (weeks 0-2)**: substrate #6 (resource context) unified;
AUDIT Tier 0 blockers fixed (Dockerfiles, SPDX, TN-bench slowdown
investigated, symbol allowlist).

**Phase 2 (weeks 2-5)**: AUDIT Tier 1 (Qulacs + Stim + OpenFermion
parity harnesses, reproducibility manifest, deterministic-seed mode,
pluggable logger, UQ on measurements). Publishable tier reached.

**Phase 3 (weeks 5-12)**: substrates #1 (Hamiltonian algebra) and
#3 (integrator) unified. Enables the non-Hermitian topology,
natural-gradient VQE, and differentiable-physics items.

**Phase 4 (weeks 12-20)**: substrates #2 (representation) and #5
(compiler IR). Adaptive dispatcher lands. Pulse-level simulation
becomes possible.

**Phase 5 (weeks 20-28)**: substrate #4 (measurement/tomography)
completes. MPO/QTCI Chern mosaic lands (10⁸ site target met).

**Phase 6 (weeks 28-36)**: substrate #7 (execution backend) brought
to first-class status. SIMD kernels wired into the production gate
path; Metal 2Q + measurement kernels; CUDA / cuQuantum integration;
MPI scaling measured at 16 and 64 ranks; WebGPU Phase 4 / Phase 5
completed. Every kernel has a CI roofline entry. This is the phase
that converts "good research code" into "quantum HPC library."

v0.3 releases between phases; v0.3.0 tag when all seven substrates
complete.

## 6. Publication Track

Papers unlocked by v0.3:

1. *Moonlab: an open platform for real-space topology at 10⁸ sites.*
2. *Adaptive-representation state evolution for quantum simulation.*
3. *Non-Hermitian local topological invariants in open-system
   lattice models.*
4. *Natural-gradient VQE with the quantum geometric tensor.*
5. *Honest logical/physical resource estimation for elliptic-curve
   cryptanalysis.* (material already in `shor_ecdlp` module)
6. *Differentiable quantum physics: a unified autograd simulator.*

Papers unlocked by v0.4 (hardware-design surface): pulse-level
co-simulation, device-aware FT resource estimation, federated
quantum-compute cost models.

## 7. Commercial Track

The core library stays MIT. Everything in this document is the
public platform. The commercial layer is a separate repository
(`tsotchke/moonlab-enterprise` or similar) that wraps the platform
with:

- Hosted SaaS (Tier 4 from AUDIT.md).
- Distributed scheduler (QGTL integration).
- Enterprise support tiers.
- Hardware-specific compilation targets (IBM, IonQ, Quantinuum,
  Google, Rigetti, Pasqal, QuEra).
- Migration-planning dashboards (PQC track, threat-timeline, per-
  enterprise device inventory).

The same substrate-6 cost tracking that publishable resource
estimates ride on is the billing surface for the SaaS.

## 8. Non-Goals

What Moonlab is deliberately *not* becoming:

- **Not a reimplementation of classical linear algebra.** BLAS,
  LAPACK, Eigen, cuBLAS, cuSPARSE, cuTensorNet, FFTW live upstream;
  Moonlab consumes them. The value-add is quantum-structure-aware
  dispatch, not matrix multiplication. (This is *not* a claim that
  Moonlab is not HPC — substrate #7 makes it so. It is a claim that
  our HPC contribution sits on top of existing numerical libraries,
  not beneath them.)
- **Not a symbolic CAS.** Moonlab's Hamiltonian algebra is a typed
  DSL for quantum operators, not a general-purpose computer algebra
  system. Use SymPy / Mathematica upstream for symbolic manipulation
  that does not have a direct numerical consequence.
- **Not a circuit designer's IDE.** Circuit authoring is Qiskit's
  and Cirq's niche. Moonlab consumes circuits they produce, and
  produces circuits they can execute.
- **Not a proof assistant.** Formal verification of quantum
  algorithms is a Lean / Coq project.
- **Not a cryptographic attack tool aimed at third parties.**
  Cryptanalysis modules target user-owned keys, synthetic
  instances, and published challenges only. Resource estimation
  for arbitrary curves is a defensive tool: it tells ecosystems
  when to migrate, not how to steal.

---

## Appendix A: Grading rubric

Every future PR is graded against four questions:

1. Which of the six substrates does it build on or extend?
2. Which of the ten new-physics targets does it move closer?
3. What performance commitment does it add, remove, or update? Is
   there a CI benchmark?
4. Does it preserve Design Commitments 1-8? If it breaks one, is
   there a deliberate update to the Commitments section here?

A PR that scores zero on all four is closed without merge. No
exceptions.
