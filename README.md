# Moonlab Quantum Simulator

[![Version](https://img.shields.io/badge/version-0.2.0-blue)]() [![Bell Test](https://img.shields.io/badge/CHSH-violates%20classical-success)](https://en.wikipedia.org/wiki/CHSH_inequality) [![State Vector](https://img.shields.io/badge/State%20Vector-32%20qubits-blue)]() [![PQC](https://img.shields.io/badge/PQC-ML--KEM%20512%2F768%2F1024-brightgreen)]() [![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux-lightgrey)]() [![Sanitizers](https://img.shields.io/badge/ASAN%20%2B%20UBSAN-clean-brightgreen)]()

> **Full-stack quantum simulation + quantum-safe cryptography: dense
> state vector (32 qubits), tensor networks, Clifford tableau,
> topological QC, chemistry / VQE with native autograd, error
> mitigation, Bell-verified QRNG, and a FIPS 203 post-quantum KEM
> seeded by that QRNG.**

Moonlab v0.2.0 is the first release with an honest end-to-end
"quantum-source to quantum-safe" pipeline in one library: the
Bell-verified quantum RNG generates 32-byte seeds that feed directly
into NIST-standard ML-KEM-512 / 768 / 1024 key generation via a single
stable-ABI call.  Alongside the cryptography work, 0.2 closes the
Phase 1/2 "completeness" items from the release plan — error
mitigation (ZNE + PEC), POVM measurement, weak measurement, Mermin /
Mermin-Klyshko Bell inequalities, quantum mutual information,
composite and correlated noise channels, DI-QRNG primitives — and
extends the native reverse-mode autograd with controlled rotations
and integrates it directly into the VQE driver.  See
[CHANGELOG.md](CHANGELOG.md) for the per-subsystem state.

> **Known limitation (v0.1.x–0.2.x):** `hermitian_eigen_decomposition`
> in `src/utils/matrix_math.c` uses real-valued Givens rotations; it
> returns correct *eigenvalues* but not eigenvectors for complex
> Hermitian matrices. Current in-tree callers only consume eigenvalues,
> so they're unaffected. For projector / Berry-curvature / Wilson-loop
> work on complex-Hermitian operators, use the matrix-sign (Schulz)
> path in `src/algorithms/topology_realspace/chern_marker.c` as a
> template. Tracked for a proper complex-Jacobi / Householder-QL
> rewrite in 0.3.

## Highlights

| Capability | Description |
|------------|-------------|
| **State Vector Engine** | Up to 32 qubits with AMX-aligned buffers + runtime-dispatched SIMD (AVX-512 / AVX2 / NEON / SVE + Apple Accelerate). |
| **Tensor Networks** | MPS, DMRG (2-site with subspace expansion), TDVP, MPO-2D, lattice-2D. Real-space topology via MPO Chebyshev-KPM: local Chern marker on generic 2D models matches dense reference to machine precision. |
| **Clifford Backend** | Aaronson–Gottesman tableau simulator: O(n) gates, O(n²) measurement. 3200-qubit GHZ + all-qubits measurement in ~100 ms. |
| **Chemistry / VQE** | Jordan-Wigner, UCCSD + hardware-efficient ansatz, H₂/LiH/H₂O Pauli Hamiltonians. Native reverse-mode autograd (CRX/CRY/CRZ + all standard rotations); `vqe_compute_gradient` uses adjoint method for HEA noise-free paths — ~5× over parameter-shift on 12 params, linear scaling to 100+. |
| **Quantum Algorithms** | Grover, VQE, QAOA, QPE, CHSH + Mermin + Mermin-Klyshko Bell tests, Shor-ECDLP resource estimator (Gidney/Drake/Boneh 2026). |
| **Error Mitigation** | Zero-noise extrapolation (linear / Richardson / exponential) and probabilistic error cancellation (PEC) primitives. |
| **Measurement** | Projective, POVM (with Kraus-completeness verification), weak-Z measurement with tunable strength, partial, non-collapsing expectations. |
| **Entanglement Metrics** | Von Neumann entropy, Rényi-α, concurrence, negativity, mutual information I(A:B), Schmidt decomposition. |
| **Post-Quantum Cryptography** | FIPS 202 SHA-3 + SHAKE (all KATs pass), FIPS 203 ML-KEM 512 / 768 / 1024 with Fujisaki-Okamoto and implicit rejection, plus QRNG-sourced keygen / encapsulate wrappers. |
| **Quantum RNG** | v3 QRNG with Bell-verified mode; device-independent primitives: Pironio min-entropy bound + Toeplitz extractor. |
| **Noise Models** | Depolarising, amplitude damping, phase damping, bit/phase-flip, thermal relaxation, composite, convex-mixture, correlated two-qubit Pauli. |
| **GPU Acceleration** | Metal compute kernels on macOS (Hadamard / CNOT / probability reduction). WebGPU backend scaffolded. |
| **Multi-Language** | C core + Python (ctypes) / Rust / JavaScript bindings.  Python exposes quantum + crypto primitives; 120+ pytest cases. |

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
- [Language Bindings](#language-bindings)
- [Performance](#performance)
- [Building](#building)
- [Documentation](#documentation)
- [Citation](#citation)

## Quick Start

```bash
# Build (CMake, the canonical path on 0.1.2+)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run the full test suite (~20s, long_evolution adds ~7min and is opt-in)
ctest --test-dir build -E long_evolution --output-on-failure

# Try an example
./build/bell_test_demo
```

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

    // Verify entanglement
    double entropy = quantum_entanglement_entropy(&state, 0);
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
tn_mps_t* mps = tn_mps_create(100, 64);

// Apply gates (automatic SVD truncation)
tn_mps_apply_single(mps, 0, GATE_H);
tn_mps_apply_two(mps, 0, 1, GATE_CNOT);

// Measure expectation values
double magnetization = tn_mps_expectation_z(mps, 50);
```

### DMRG Ground State

```c
#include "algorithms/tensor_network/dmrg.h"

// Heisenberg chain Hamiltonian
dmrg_params_t params = {
    .num_sites = 100,
    .max_bond_dim = 128,
    .num_sweeps = 20,
    .tolerance = 1e-10
};

dmrg_result_t result = dmrg_ground_state(&hamiltonian, &params);
printf("Ground state energy: %.10f\n", result.energy);
```

### TDVP Time Evolution

```c
#include "algorithms/tensor_network/tdvp.h"

// Real-time dynamics
tdvp_evolve(mps, hamiltonian, dt, num_steps, TDVP_TWO_SITE);
```

### 2D Tensor Networks

```c
#include "algorithms/tensor_network/lattice_2d.h"
#include "algorithms/tensor_network/mpo_2d.h"

// Create 10x10 lattice
lattice_2d_t* lattice = lattice_2d_create(10, 10, LATTICE_SQUARE);

// Apply 2D MPO Hamiltonian
mpo_2d_t* H = mpo_2d_heisenberg(lattice, J_coupling);
```

## Quantum Algorithms

### Grover's Search

```c
grover_result_t result = grover_search(&state, marked_state, num_qubits);
// O(√N) queries vs classical O(N)
```

### Variational Quantum Eigensolver (VQE)

```c
#include "algorithms/vqe.h"

vqe_config_t config = {
    .ansatz = VQE_ANSATZ_UCCSD,
    .optimizer = VQE_OPT_BFGS,
    .max_iterations = 100
};

vqe_result_t result = vqe_minimize(&hamiltonian, &config);
printf("Ground state energy: %.8f Ha\n", result.energy);
```

### QAOA (Combinatorial Optimization)

```c
#include "algorithms/qaoa.h"

// MaxCut problem
qaoa_result_t result = qaoa_maxcut(&graph, num_layers);
printf("Best cut: %zu edges\n", result.best_cut);
```

### Quantum Phase Estimation

```c
#include "algorithms/qpe.h"

double phase = qpe_estimate(&unitary, precision_qubits, &state);
```

### Bell Test Validation

```c
#include "algorithms/bell_tests.h"

/* CHSH on a Bell pair */
bell_test_result_t r = bell_test_chsh(&state, 0, 1, 10000, NULL, entropy);
printf("CHSH: %.4f (classical <= 2, quantum <= 2.828)\n", r.chsh_value);

/* Mermin polynomial on |GHZ_3>: classical <= 2, quantum max 4 */
bell_test_result_t m = bell_test_mermin_ghz(&ghz3, 0, 1, 2, 0, entropy);
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
anyon_system_t* sys = anyon_system_fibonacci(num_anyons);

// Braid anyons (topological gate)
braid_anyons(sys, i, j, BRAID_COUNTERCLOCKWISE);

// Compute resulting unitary
complex_t* U = fusion_tree_to_unitary(sys->fusion_tree);
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
surface_code_t* code = surface_code_create(5, 5, BOUNDARY_PLANAR);

// Measure stabilizers
surface_code_measure_stabilizers(code);

// Decode and correct errors
int success = surface_code_decode_mwpm(code);
```

### Toric Codes

```c
// Create toric code on 6x6 lattice
toric_code_t* toric = toric_code_create(6, 6);

// Measure plaquette and star operators
toric_code_measure_plaquettes(toric);
toric_code_measure_stars(toric);

// Apply logical X on first logical qubit
toric_code_logical_x(toric, 0);
```

### Topological Invariants

```c
// Compute topological entanglement entropy
double gamma = topological_entanglement_entropy(&state, &region);
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

// Define circular braiding path
braid_path_t* path = braid_path_circular(sk1, sk2, num_steps);

// Perform braiding with TDVP time evolution
topo_qubit_t* qubit = topo_qubit_create(mps, sk1, sk2);
braid_result_t result = skyrmion_braid(qubit, path, &params);

// Extract Berry phase
printf("Berry phase: %.6f\n", result.berry_phase);
printf("Fidelity: %.6f\n", result.fidelity);
```

### Topological Gates

```c
// Apply topological gates via skyrmion exchange
topo_gate_apply(qubit, TOPO_GATE_EXCHANGE);  // π rotation
topo_gate_apply(qubit, TOPO_GATE_BRAID);     // Braiding unitary
```

## Quantum Chemistry

Molecular simulation with fermionic mappings.

### Jordan-Wigner Transformation

```c
#include "algorithms/chemistry/chemistry.h"

// Create fermionic operator a†_p a_q
fermion_op_t op = fermion_op_hopping(p, q);

// Transform to qubit Hamiltonian
jw_operator_t* jw = jordan_wigner_transform(&op);

// Get Pauli string representation
pauli_string_t* paulis = jw_to_pauli_strings(jw);
```

### UCCSD Ansatz

```c
// Build UCCSD circuit for molecular simulation
uccsd_params_t params = {
    .num_electrons = 2,
    .num_orbitals = 4,
    .singles = true,
    .doubles = true
};

circuit_t* ansatz = uccsd_circuit(&params);
```

### Molecular Hamiltonians

```c
// H2 molecule in minimal basis
molecular_hamiltonian_t* H = molecular_hamiltonian_h2(bond_length);

// Run VQE
vqe_result_t result = vqe_minimize(H, &vqe_config);
printf("H2 energy: %.6f Ha\n", result.energy);
```

## Many-Body Localization

Disordered quantum systems and thermalization dynamics.

### Disordered Heisenberg Model

```c
#include "algorithms/mbl/mbl.h"

// Create XXZ Hamiltonian with disorder
xxz_params_t params = {
    .num_sites = 16,
    .Jxy = 1.0,
    .Jz = 1.0,
    .disorder_strength = 5.0,  // Strong disorder (MBL phase)
    .seed = 42
};

xxz_hamiltonian_t* H = xxz_hamiltonian_create(&params);
```

### Diagnostics

```c
// Level statistics (Poisson vs GOE)
double r = level_spacing_ratio(eigenvalues, num_eigenvalues);
// r ≈ 0.39 (Poisson, MBL) vs r ≈ 0.53 (GOE, thermal)

// Entanglement entropy dynamics
double S = entanglement_entropy_half_chain(state, num_sites);
```

## Post-Quantum Cryptography

Moonlab v0.2 ships a reference implementation of FIPS 202 (SHA-3,
SHAKE) and FIPS 203 (ML-KEM — the NIST-standardised
module-lattice-based KEM), seeded by the same Bell-verified quantum
RNG that exports `moonlab_qrng_bytes`.  Three parameter sets are
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
#include <moonlab/mitigation/zne.h>

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

## Language Bindings

### Python (with PyTorch Integration)

```python
import moonlab as ml
import torch

# Create quantum state
state = ml.QuantumState(4)
state.h(0).cnot(0, 1).cnot(1, 2).cnot(2, 3)

# PyTorch hybrid layer
class QuantumLayer(torch.nn.Module):
    def __init__(self, num_qubits):
        super().__init__()
        self.circuit = ml.ParameterizedCircuit(num_qubits)
        self.params = torch.nn.Parameter(torch.randn(num_qubits * 3))

    def forward(self, x):
        return self.circuit.run(x, self.params)

# Train with backpropagation
model = QuantumLayer(4)
optimizer = torch.optim.Adam(model.parameters())
```

### VQE in Python

```python
from moonlab.algorithms import VQE

vqe = VQE(
    hamiltonian=H_molecular,
    ansatz='uccsd',
    optimizer='BFGS'
)
result = vqe.minimize()
print(f"Energy: {result.energy:.8f} Ha")
```

### Rust

```rust
use moonlab::{QuantumState, Gate};

fn main() {
    let mut state = QuantumState::new(4);
    state.apply(Gate::H, 0);
    state.apply(Gate::CNOT, (0, 1));

    let entropy = state.entanglement_entropy(0);
    println!("Entropy: {:.4}", entropy);
}
```

### JavaScript (React)

```jsx
import { useQuantumState, BlochSphere } from '@moonlab/quantum-react';

function QuantumVisualizer() {
    const [state, dispatch] = useQuantumState(1);

    return (
        <div>
            <BlochSphere state={state} />
            <button onClick={() => dispatch({ type: 'H', qubit: 0 })}>
                Hadamard
            </button>
        </div>
    );
}
```

### Vue

```vue
<template>
    <circuit-diagram :circuit="circuit" />
</template>

<script setup>
import { useQuantumState } from '@moonlab/quantum-vue';
const { state, circuit } = useQuantumState(2);
</script>
```

## Limitations (as of 0.2.0)

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
- **GPU backends other than Metal + Eshkol**: OpenCL and Vulkan now
  compile cleanly and pass a "compile + discovery smoke" CI tier on
  Linux (apt's ocl-icd-loader + PoCL for OpenCL; vulkan-loader +
  lavapipe for Vulkan) but are not exercised against a real GPU in
  CI -- the smoke test just verifies backend selection and
  fallback.  CUDA and cuQuantum have 1000+ LOC implementations but
  no CI runner has NVIDIA hardware; treat them as compile-only.
- **WebGPU / JS**: CI has a dedicated `WASM / WebGPU smoke` tier
  that builds the TS `@moonlab/quantum-core` package and runs the
  unified smoke end-to-end.  The default C-only `ctest` on a fresh
  clone without `-DQSIM_BUILD_JS_DIST=ON` produces no WebGPU
  coverage -- that is the trade-off for not requiring a JS toolchain
  just to build the library.
- **Platforms**: Linux x86_64, Linux aarch64, macOS arm64, and
  macOS x86_64 all have CI tiers that build the full tree and run
  `ctest -E long_evolution`; macOS arm64 additionally runs Release,
  Debug, -Werror, ASAN+UBSAN, and MPI-enabled variants.  Windows
  is unsupported.
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

`distributed_gates` exercises the MPI bridge: init, allreduce,
sendrecv and barrier round-trip on ≥2 ranks. End-to-end distributed
state-vector gate application across partitions is still in progress —
the scaling table that used to live here was not reproducibly
measured and has been removed.

## Building

### Requirements

- **macOS**: 10.15+ (Apple Silicon recommended)
- **Linux**: GCC 9+ with OpenMP
- **Memory**: 8 GB minimum, 32 GB+ for large simulations

### Build Options

CMake is the canonical build system (`0.1.2+`). Useful options:

| Option | Default | Effect |
|---|---|---|
| `-DCMAKE_BUILD_TYPE=Release\|Debug\|RelWithDebInfo` | `Release` | Standard CMake build type |
| `-DQSIM_ENABLE_METAL=ON` | `ON` on macOS | Metal GPU backend |
| `-DQSIM_ENABLE_MPI=ON` | `OFF` | MPI distributed computing (OpenMPI) |
| `-DQSIM_ENABLE_OPENMP=ON` | `ON` | OpenMP multi-core |
| `-DQSIM_WERROR=ON` | `OFF` | `-Werror` build with `-Wpedantic` / `-Wdeprecated-declarations` demoted (libomp + CLAPACK externalities) |
| `-DQSIM_ENABLE_SANITIZERS=ON` | `OFF` | AddressSanitizer + UndefinedBehaviorSanitizer |
| `-DQSIM_ENABLE_AVX512=ON` / `AVX2` / `NEON` / `SVE` | `ON` if available | SIMD path toggles |
| `-DQSIM_BUILD_TESTS=ON` | `ON` | CTest targets |
| `-DQSIM_BUILD_EXAMPLES=ON` | `ON` | `examples/` programs |
| `-DQSIM_BUILD_BENCHMARKS=ON` | `ON` | `benchmarks/` targets |

Legacy `make all` / `make test` still work for a subset of the surface.

### Dependencies

**Required**:
- C compiler (GCC/Clang)
- POSIX threads

**Optional**:
- OpenMP (multi-core)
- Accelerate framework (macOS AMX)
- Metal (GPU, macOS)
- MPI (distributed)

## Documentation

Full documentation is available in the [docs/](docs/) directory:

- [Getting Started](docs/getting-started/index.md)
- [Concepts](docs/concepts/index.md)
- [Tutorials](docs/tutorials/index.md)
- [API Reference](docs/api/index.md)
- [Algorithm Deep Dives](docs/algorithms/index.md)
- [Performance Guide](docs/performance/index.md)

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
│   ├── basic/                # Hello quantum, Bell states
│   ├── algorithms/           # VQE, QAOA, Grover
│   ├── tensor_network/       # Spin chains, DMRG
│   └── applications/         # Portfolio, QRNG
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
    version      = {v0.2.0},
    url          = {https://github.com/tsotchke/moonlab},
    license      = {MIT},
    keywords     = {quantum computing, simulation, tensor networks,
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
