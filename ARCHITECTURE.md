# Moonlab Architecture

This document is the authoritative description of Moonlab's internal
structure, its public interfaces, and the theoretical foundations of
each subsystem. It is intended for contributors and for integrators of
the sibling libraries QGTL, lilirrep and SbNN.

A short physics primer is embedded for each module so the reader does
not have to round-trip to the references; every load-bearing algorithm
nevertheless cites its primary source in a standard reference list at
the end.

## 1. Design Philosophy

Moonlab is organised as a **C core + language-level safe wrappers**.
The C core is the source of truth: every public capability is
exercised by at least one C unit test before any binding is added,
and the stable ABI surface (`src/applications/moonlab_export.h`) is
the promised contract to downstream consumers. Language bindings
(Python, Rust, JavaScript) are convenience layers around that core;
they may lag the C surface but must not extend it.

Three invariants are enforced throughout:

1. **No claims without ground truth.** Every topological invariant is
   cross-validated across independent algorithmic paths (dense Schulz
   sign-function, matrix-free Chebyshev-KPM, momentum-space Fukui-
   Hatsugai-Suzuki link variables); every stabilizer correctness claim
   is tested against a small-qubit dense simulator; every Haar-random
   generator is checked against the expected Porter-Thomas heavy-output
   statistics.
2. **Opaque handles, flat arrays at the boundary.** Public C types are
   forward-declared structs; callers reach into them only through
   functions. Matrices cross the ABI as flat `double*` or `complex_t*`
   in documented layout (row-major, orbital-fastest) with no internal
   Moonlab types needed at the binding site. This is the property
   that lets QGTL / lilirrep / SbNN bind with ctypes / bindgen / WASM
   with no need to vendor Moonlab internals.
3. **Tolerances live in tests, not in the library.** A library function
   that silently rounds a random outcome to a "success" defeats the
   purpose. Tests own the numerical tolerances; the library returns
   exact values with explicit kind flags (random vs deterministic,
   fused vs singular, etc.) so callers can apply their own policy.

## 2. Physical-Dimensionality Stack

Moonlab distinguishes three computational representations of a quantum
state; each is a distinct subsystem with its own entry points.

| Representation | Module | Capacity | Exact? | Best for |
|---|---|---|---|---|
| Dense state vector | `src/quantum/` | n ≤ 32 | yes | small, generic circuits |
| Tensor network (MPS / MPO / TTN) | `src/algorithms/tensor_network/` | n ≈ 100 if weakly entangled | up to truncation | 1D / quasi-1D ground states, Trotter evolution |
| Clifford tableau | `src/backends/clifford/` | n ≳ 10⁴ | yes (Clifford only) | stabilizer circuits, QEC, Pauli-frame workloads |

The three are not interchangeable; they answer different questions.
Gate application at the user-facing layer (`gate_h`, `gate_cnot`, ...)
currently targets the dense representation. The Clifford tableau is
reached directly via `src/backends/clifford/clifford.h`; the tensor
network via `src/algorithms/tensor_network/tn_gates.h`. A unified
dispatcher that chooses automatically based on a circuit's gate set is
planned for 0.3.

## 3. Directory Layout

```
src/
  quantum/          state-vector core (state_t, gates, measurement, entanglement, noise)
  backends/
    clifford/       Aaronson-Gottesman tableau (polynomial-time stabilizer sim)
  algorithms/
    grover.{c,h}    Grover search
    vqe.{c,h}       Variational quantum eigensolver (HEA, UCCSD, symmetry-preserving)
    qaoa.{c,h}      QAOA for Ising / MaxCut / TSP
    qpe.{c,h}       Quantum phase estimation
    chemistry/      Jordan-Wigner, UCCSD, H2 / LiH / H2O
    topological/    Fibonacci/Ising anyons, fusion trees, surface-code scaffold
    topology_realspace/
      chern_marker.{c,h}  dense Bianco-Resta local Chern marker
      chern_kpm.{c,h}     matrix-free Chebyshev-KPM marker (scales to L ~ 300)
    quantum_geometry/
      qgt.{c,h}     quantum geometric tensor, Berry, Wilson loop, SSH winding
    tensor_network/ MPS, MPO, DMRG, TDVP, CTMRG (partial)
    bell_tests.{c,h} CHSH / GHZ / Mermin
    mbl/            many-body localisation
  applications/
    moonlab_export.h   stable ABI surface (dlsym target)
    qrng.{c,h}         v3 quantum RNG (Bell-verified / Grover / direct)
    nist_sp800_22.c    15-test NIST statistical battery
    quantum_volume.{c,h}  IBM Quantum Volume harness
  optimization/
    memory_align.{c,h} AMX-aligned allocation
    simd_*.{c,h}       AVX-512 / AVX2 / NEON / SVE kernels
    stride_gates.{c,h} experimental stride-based SIMD intrinsics (parallel track)
    fusion/
      fusion.{c,h}  gate-fusion DAG (single-qubit run merging)
    gpu/             backend scaffolding: Metal (functional), CUDA/OpenCL/Vulkan/WebGPU (declared)
  utils/
    matrix_math.{c,h}   linear algebra helpers; note complex-Hermitian eigvec
                         path is unsound and documented as such
    quantum_entropy.h   secure entropy handle
  distributed/       MPI bridge + partitioned state vector
  visualization/     circuit + Feynman-diagram rendering
bindings/
  python/ rust/ javascript/    language wrappers (see §5)
tests/
  unit/     per-module C tests
  integration/ cross-module flows
  abi/      dlsym-based stable-ABI probes
  performance/ micro-benchmarks (not in ctest by default)
```

## 4. Subsystem Summaries

### 4.1 State-vector core (`src/quantum/`)

Pure dense state vector `complex_t[2^n]` with alignment-aware allocation
for AMX / AVX-512 / NEON / SVE runtime dispatch. The gate set is the
universal set (Clifford + T) extended with parameterised rotations,
controlled rotations, QFT, IQFT, and Toffoli. Entanglement metrics
(Rényi entropy, concurrence, negativity) live in
`src/quantum/entanglement.{c,h}`; noise (depolarising, amplitude damping,
phase damping, thermal, composite, correlated) in `noise.{c,h}`, with a
Kraus-completeness validator in the unit tests.

### 4.2 Clifford tableau (`src/backends/clifford/`)

Aaronson-Gottesman 2n×(2n+1) binary tableau (Aaronson & Gottesman 2004).
O(n²) memory, O(n) per single-qubit gate, O(n) per CNOT, O(n²) per
measurement. Exact on Clifford circuits; verified at n = 100 (GHZ)
and through a full surface-code module at distance d = 7, 9, 15
(`surface_code_clifford_t` in `src/algorithms/topological/`).

### 4.3 Real-space topology (`src/algorithms/topology_realspace/`)

The Bianco-Resta local Chern marker
`c(r) = −(4π/A_uc) Im ⟨r| P X Q Y P |r⟩` is implemented twice:
- `chern_marker.{c,h}` — dense reference using the Schulz iteration
  on the matrix sign function (no diagonalisation; side-steps the
  known complex-Hermitian eigenvector unsoundness of
  `hermitian_eigen_decomposition`), ceiling ≈ L=20.
- `chern_kpm.{c,h}` — matrix-free Jackson-regularised Chebyshev
  expansion of the sign function applied to the three-projector-
  sandwich expression of the marker. Ceiling L=300 (90k sites) on a
  single core; parallelised across sites via OpenMP.

Both cross-validate to ≤ 10⁻³ on the built-in Qi-Wu-Zhang model.
Quasicrystal mosaics are supported via an on-site cosine modulation
(`chern_kpm_cn_modulation`) in the spirit of Antão et al. (2026).

### 4.4 Momentum-space topology (`src/algorithms/quantum_geometry/`)

Fukui-Hatsugai-Suzuki discretised Chern number and Berry-curvature
grid, the Fubini-Study metric by finite differences, Wilson loops on
user-supplied 2D paths, and the 1D SSH winding number. Built-in
models: QWZ, Haldane (honeycomb, Peierls phase), SSH. The integer-
quantised link-variable method is insensitive to gauge choice and
runs with 32×32 grids giving exact integers on our phase-diagram
tests.

### 4.5 Quantum Volume (`src/applications/quantum_volume.{c,h}`)

The IBM Quantum Volume protocol of Cross et al. (2019). Haar-random
SU(4) layers via the Mezzadri (2007) construction, Fisher-Yates
permutations between layers, exact heavy-output probability from the
full state vector, normal-approximation confidence interval. Widths
3..10 pass with mean HOP converging to the Porter-Thomas asymptote
(1 + ln 2)/2 ≈ 0.847.

### 4.6 Gate-fusion DAG (`src/optimization/fusion/`)

Single-qubit run-length fusion: consecutive 1Q gates on the same
qubit are collapsed to a single 2×2 matrix, flushed when a 2Q gate
touches the qubit. Measured speedup 2.18× at n=16, 1.43× at n=20 on
the hardware-efficient VQE/QAOA layers. Two-qubit block fusion and
commutation-aware reordering are on the 0.3 roadmap.

### 4.7 Tensor networks (`src/algorithms/tensor_network/`)

Matrix-product state / operator plumbing, 2-site DMRG with subspace
expansion and noise decay, TDVP (Haegeman et al. 2011, 2016), 2D
MPO, square / triangular / honeycomb lattice utilities, skyrmion
braiding primitives on top. Modern additions (adaptive-bond TDVP,
split-CTMRG, BP + cluster contraction, isoTNS) are queued for 0.3.

### 4.8 Quantum RNG (`src/applications/qrng*`)

Three-mode v3 engine: direct (quantum simulation of a Hadamard
cascade), Grover-biased, Bell-verified (continuous CHSH monitor
rejects epochs where the measured @f$S@f$ falls below threshold).
Hardware-entropy layer (`hardware_entropy.c`) hits RDSEED / /dev/urandom /
SecRandomCopyBytes. Output is post-processed through a Toeplitz
extractor; NIST SP 800-22 fast subset runs in CI and the full 15-test
battery is available offline.

### 4.9 Stable ABI (`src/applications/moonlab_export.h`)

The promised-stable surface for downstream consumers (dlsym targets).
At 0.2.0-dev this includes:
- `moonlab_abi_version(major, minor, patch)` — feature probe
- `moonlab_qrng_bytes(buf, size)` — QRNG byte stream
- `moonlab_qwz_chern(m, N, out_chern)` — integer Chern of QWZ via FHS

ABI-version bumps (minor) accompany new symbols; existing symbols are
name-and-signature frozen across all 0.x releases. QGTL, lilirrep and
SbNN bind through this surface.

## 5. Language Bindings

### 5.1 Python (`bindings/python/moonlab/`)

Pure ctypes against `libquantumsim.dylib/.so`. Modules: `core`
(QuantumState, Gates, Measurement, QuantumError), `algorithms` (VQE,
QAOA, Grover, BellTest — defensive import guard retained but imports
successfully on a full build), `clifford`, `benchmarks` (quantum_volume),
`topology` (ChernKPM, qwz_chern, berry_grid_qwz, berry_grid_haldane,
ssh_winding), `ml` and `torch_layer` (experimental).

### 5.2 Rust (`bindings/rust/moonlab/` + `moonlab-sys/`)

`moonlab-sys` is the bindgen FFI crate; its allowlist covers 218+
public C symbols including the full 0.2 topology / Clifford / QV /
fusion surfaces. `moonlab` is the safe wrapper crate: `QuantumState`
with RAII + method chaining, `FeynmanDiagram`, and the new `topology`
module (`qwz_chern`, `ssh_winding`, `ChernKpm`).

### 5.3 JavaScript / WebGPU (`bindings/javascript/`)

TypeScript API over an Emscripten WASM build of the C core, plus a
WebGPU compute-backend scaffolding (Phase 3 of `webgpuplan.md` in
progress). Core surface: QuantumState, Circuit, tensor-network
helpers. The 0.2 topology / Clifford surfaces are not yet exposed in
JS.

## 6. Build + Test Story

CMake drives the build. `VERSION.txt` is the canonical version string
and is read at configure time; it accepts a pre-release suffix
(`-dev`, `-rc1`) to keep the Cargo / npm / PyPI ecosystems in sync.

`ctest --test-dir build` runs the full regression (52 targets at the
time of this writing, spanning unit tests, ABI smoke, language-binding
smokes, and fast example runners). Slow benchmarks live in
`tests/performance/` and are built but not auto-registered, so CI
time stays below 10 minutes on a laptop.

The audit memory note (`memory/project_broken_subsystems_v012.md`)
enumerates known issues that are *not* fixed in the current tree. It
is updated in lock-step with the CHANGELOG so a new contributor can
see which warnings are still live.

## 7. References

### Stabilizer formalism & Clifford simulation

- S. Aaronson and D. Gottesman, "Improved simulation of stabilizer
  circuits", Phys. Rev. A 70, 052328 (2004), arXiv:quant-ph/0406196.
- D. Gottesman, "Stabilizer Codes and Quantum Error Correction",
  Caltech PhD thesis (1997), arXiv:quant-ph/9705052.
- D. Gottesman, "The Heisenberg Representation of Quantum Computers",
  arXiv:quant-ph/9807006.
- C. Gidney, "Stim: a fast stabilizer circuit simulator",
  Quantum 5, 497 (2021), arXiv:2103.02202.

### Surface code & fault tolerance

- A. Yu. Kitaev, "Fault-tolerant quantum computation by anyons",
  Ann. Phys. 303, 2 (2003), arXiv:quant-ph/9707021.
- A. G. Fowler, M. Mariantoni, J. M. Martinis and A. N. Cleland,
  "Surface codes: Towards practical large-scale quantum computation",
  Phys. Rev. A 86, 032324 (2012), arXiv:1208.0928.

### Topology (real-space & momentum-space)

- R. Bianco and R. Resta, "Mapping topological order in coordinate
  space", Phys. Rev. B 84, 241106(R) (2011), arXiv:1111.5697.
- T. V. C. Antão, Y. Sun, A. O. Fumega and J. L. Lado,
  "Tensor Network Method for Real-Space Topology in Quasicrystal
  Chern Mosaics", Phys. Rev. Lett. 136, 156601 (2026),
  doi:10.1103/hhdf-xpwg.
- T. Fukui, Y. Hatsugai and H. Suzuki, "Chern numbers in discretized
  Brillouin zone", J. Phys. Soc. Jpn. 74, 1674 (2005),
  arXiv:cond-mat/0503172.
- A. Weisse, G. Wellein, A. Alvermann and H. Fehske,
  "The kernel polynomial method", Rev. Mod. Phys. 78, 275 (2006),
  arXiv:cond-mat/0504627.
- N. J. Higham, "Functions of Matrices: Theory and Computation",
  SIAM (2008).

### Quantum geometry

- J. P. Provost and G. Vallee, "Riemannian structure on manifolds of
  quantum states", Commun. Math. Phys. 76, 289 (1980),
  doi:10.1007/BF02193559.
- M. V. Berry, "Quantal phase factors accompanying adiabatic changes",
  Proc. R. Soc. Lond. A 392, 45 (1984), doi:10.1098/rspa.1984.0023.
- J. Zak, "Berry's phase for energy bands in solids",
  Phys. Rev. Lett. 62, 2747 (1989),
  doi:10.1103/PhysRevLett.62.2747.

### Models

- F. D. M. Haldane, "Model for a quantum Hall effect without Landau
  levels", Phys. Rev. Lett. 61, 2015 (1988),
  doi:10.1103/PhysRevLett.61.2015.
- X.-L. Qi, Y.-S. Wu and S.-C. Zhang, "Topological quantization of
  the spin Hall effect", Phys. Rev. B 74, 085308 (2006),
  arXiv:cond-mat/0505308.
- W. P. Su, J. R. Schrieffer and A. J. Heeger, "Solitons in
  polyacetylene", Phys. Rev. Lett. 42, 1698 (1979),
  doi:10.1103/PhysRevLett.42.1698.
- A. Yu. Kitaev, "Unpaired Majorana fermions in quantum wires",
  Phys.-Usp. 44, 131 (2001), arXiv:cond-mat/0010440.

### Variational algorithms

- E. Farhi, J. Goldstone, S. Gutmann, "A Quantum Approximate
  Optimization Algorithm", arXiv:1411.4028 (2014).
- R. Cleve, A. Ekert, C. Macchiavello and M. Mosca, "Quantum
  algorithms revisited", Proc. R. Soc. Lond. A 454, 339 (1998),
  arXiv:quant-ph/9708016.

### Tensor networks

- J. Haegeman, J. I. Cirac, T. J. Osborne, I. Pižorn, H. Verschelde
  and F. Verstraete, "Time-dependent variational principle for
  quantum lattices", Phys. Rev. Lett. 107, 070601 (2011),
  arXiv:1103.0936.
- J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken and
  F. Verstraete, "Unifying time evolution and optimization with
  matrix product states", Phys. Rev. B 94, 165116 (2016),
  arXiv:1408.5056.

### Benchmarks

- A. W. Cross, L. S. Bishop, S. Sheldon, P. D. Nation and J. M.
  Gambetta, "Validating quantum computers using randomized model
  circuits", Phys. Rev. A 100, 032328 (2019), arXiv:1811.12926.
- F. Mezzadri, "How to generate random matrices from the classical
  compact groups", Notices of the AMS 54, 592 (2007),
  arXiv:math-ph/0609050.
- T. Häner and D. S. Steiger, "0.5 Petabyte Simulation of a 45-Qubit
  Quantum Circuit", SC17 (2017), arXiv:1704.01127.
- Y. Suzuki et al., "Qulacs: a fast and versatile quantum circuit
  simulator for research purpose", Quantum 5, 559 (2021),
  arXiv:2011.13524.

All references above were verified against live arXiv / DOI sources
during the 2026-04-18 audit. Future additions should carry the same
verification bar.
