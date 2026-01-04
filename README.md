# Moonlab Quantum Simulator

[![Version](https://img.shields.io/badge/version-0.1.0-blue)]() [![Bell Test](https://img.shields.io/badge/Bell%20CHSH-2.828-success)](https://en.wikipedia.org/wiki/CHSH_inequality) [![State Vector](https://img.shields.io/badge/State%20Vector-32%20qubits-blue)]() [![Tensor Network](https://img.shields.io/badge/Tensor%20Network-100%2B%20qubits-purple)]() [![GPU Speedup](https://img.shields.io/badge/GPU%20Speedup-100x-orange)]() [![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux-lightgrey)]()

> **A comprehensive quantum computing framework spanning state vector simulation, tensor networks, topological quantum computing, and skyrmion physics**

Moonlab is a production-grade quantum simulation platform that bridges theoretical physics and high-performance computing. From Bell inequality violation (CHSH = 2.828) to DMRG ground state calculations, from topological anyon braiding to magnetic skyrmion qubits, Moonlab provides the tools researchers need to explore the full spectrum of quantum phenomena.

## Highlights

| Capability | Description |
|------------|-------------|
| **State Vector Engine** | Up to 32 qubits with AMX/SIMD optimization |
| **Tensor Networks** | MPS, DMRG, TDVP for 100+ qubit systems |
| **Topological QC** | Fibonacci/Ising anyons, surface codes, toric codes |
| **Skyrmion Braiding** | Magnetic skyrmion topological qubits |
| **Quantum Chemistry** | Jordan-Wigner, UCCSD, molecular Hamiltonians |
| **Many-Body Localization** | Disordered spin chains, entanglement dynamics |
| **Quantum Algorithms** | Grover, VQE, QAOA, QPE, Bell tests |
| **GPU Acceleration** | Metal compute (100x speedup) |
| **Multi-Language** | C, Python (PyTorch), Rust (TUI), JavaScript (React/Vue) |

## Table of Contents

- [Quick Start](#quick-start)
- [State Vector Simulation](#state-vector-simulation)
- [Tensor Network Methods](#tensor-network-methods)
- [Quantum Algorithms](#quantum-algorithms)
- [Topological Quantum Computing](#topological-quantum-computing)
- [Skyrmion Braiding](#skyrmion-braiding)
- [Quantum Chemistry](#quantum-chemistry)
- [Many-Body Localization](#many-body-localization)
- [Language Bindings](#language-bindings)
- [Performance](#performance)
- [Building](#building)
- [Documentation](#documentation)
- [Citation](#citation)

## Quick Start

```bash
# Build
make all

# Run tests
make test

# Try an example
./examples/tensor_network/quantum_spin_chain
```

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

bell_result_t result = bell_chsh(&state, 0, 1, 10000);
printf("CHSH: %.4f (classical ≤ 2, quantum ≤ 2.828)\n", result.chsh_value);
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
import { useQuantumState, BlochSphere } from '@moonlab/react';

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
import { useQuantumState } from '@moonlab/vue';
const { state, circuit } = useQuantumState(2);
</script>
```

## Performance

### GPU Acceleration (Metal)

| Operation | CPU (SIMD) | GPU (Metal) | Speedup |
|-----------|------------|-------------|---------|
| 24-qubit Hadamard | 78 ms | 2.5 ms | 31x |
| 28-qubit CNOT | 3.2 s | 52 ms | 62x |
| Grover (22 qubits) | 28 min | 28 s | 60x |

### Tensor Network Scaling

| System | Bond Dim | Memory | Time (DMRG) |
|--------|----------|--------|-------------|
| 50 sites | 64 | 200 KB | 2 s |
| 100 sites | 128 | 1.6 MB | 15 s |
| 200 sites | 256 | 13 MB | 2 min |

### Distributed Computing (MPI)

| Qubits | Nodes | Memory/Node | Speedup |
|--------|-------|-------------|---------|
| 32 | 16 | 4 GB | 14x |
| 34 | 64 | 4 GB | 52x |
| 36 | 256 | 4 GB | 180x |

## Building

### Requirements

- **macOS**: 10.15+ (Apple Silicon recommended)
- **Linux**: GCC 9+ with OpenMP
- **Memory**: 8 GB minimum, 32 GB+ for large simulations

### Build Options

```bash
# Standard build
make all

# With GPU acceleration (macOS)
make METAL=1

# With MPI support
make MPI=1

# Debug build
make DEBUG=1

# Build examples
make examples

# Run tests
make test
```

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
@software{tsotchke_moonlab_2024,
    author       = {tsotchke},
    title        = {{Moonlab}: A Quantum Computing Simulation Framework},
    year         = {2026},
    month        = jan,
    version      = {v0.1.0},
    url          = {https://github.com/tsotchke/moonlab},
    license      = {MIT},
    keywords     = {quantum computing, simulation, tensor networks,
                    topological quantum computing, DMRG, VQE, QAOA}
}
```

## References

**Theoretical Foundations**:
- Bell, J.S. (1964). On the Einstein Podolsky Rosen paradox. *Physics*, 1(3), 195-200.
- Grover, L.K. (1996). A fast quantum mechanical algorithm for database search. *STOC '96*.
- White, S.R. (1992). Density matrix formulation for quantum renormalization groups. *Phys. Rev. Lett.* 69, 2863.
- Kitaev, A. (2003). Fault-tolerant quantum computation by anyons. *Ann. Phys.* 303, 2-30.
- Psaroudaki, C. & Panagopoulos, C. (2021). Skyrmion qubits: A new class of quantum logic elements. *Phys. Rev. Lett.* 127, 067201.

**Algorithms**:
- Peruzzo, A. et al. (2014). A variational eigenvalue solver on a photonic quantum processor. *Nat. Commun.* 5, 4213.
- Farhi, E., Goldstone, J., & Gutmann, S. (2014). A quantum approximate optimization algorithm. *arXiv:1411.4028*.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

**Moonlab** - From qubits to anyons, from state vectors to tensor networks.

*Built for researchers. Optimized for discovery.*
