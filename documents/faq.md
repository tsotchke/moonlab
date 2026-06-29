# Archived Moonlab Documentation: Frequently Asked Questions

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Frequently Asked Questions

Common questions about Moonlab Quantum Simulator, organized by topic.

## General Questions

### What is Moonlab Quantum Simulator?

Moonlab is a high-performance quantum computing simulation library. It provides state-vector simulation for up to 32 qubits, comprehensive algorithm implementations (Grover, VQE, QAOA, DMRG), and bindings for multiple programming languages (C, Python, Rust, JavaScript).

### Is this a real quantum computer?

No. Moonlab is a **classical simulator** of quantum computation. It runs on conventional computers (CPUs and GPUs) and simulates how a quantum computer would behave. This is useful for:

- Developing and testing quantum algorithms
- Educational purposes
- Research that doesn't require actual quantum hardware
- Hybrid quantum-classical algorithm development

### How many qubits can I simulate?

Moonlab supports up to 32 qubits with sufficient memory:

| Qubits | Memory Required |
|--------|-----------------|
| 20 | 16 MB |
| 25 | 512 MB |
| 28 | 4 GB |
| 30 | 16 GB |
| 32 | 64 GB |

For larger systems with limited entanglement, tensor network methods (DMRG, MPS) can simulate 100+ qubits.

### What platforms are supported?

- **macOS**: Full support including Metal GPU acceleration on Apple Silicon (M1-M4)
- **Linux**: Full support with CUDA/Vulkan GPU options
- **Windows**: Via WSL2 (Windows Subsystem for Linux)
- **Web browsers**: Via WebAssembly

### Is Moonlab open source?

Yes. Moonlab is open-source software. See the LICENSE file in the repository for details.

## Installation

### How do I install Moonlab?

[archived fence delimiter: ```bash]
git clone https://github.com/tsotchke/moonlab.git
cd quantum-simulator
make
[archived fence delimiter: ```]

See [Installation](installation.md) for detailed platform-specific instructions.

### Why is the build failing?

Common causes:

1. **Missing compiler**: Install Xcode Command Line Tools (macOS) or build-essential (Linux)
2. **Missing OpenMP**: Install libomp (`brew install libomp` on macOS)
3. **Old compiler**: Need GCC 9+ or Clang 11+

Run `make clean && make` to retry after fixing issues.

### How do I install Python bindings?

[archived fence delimiter: ```bash]
make  # Build C library first
cd bindings/python
pip install -e .
[archived fence delimiter: ```]

### How do I install JavaScript packages?

[archived fence delimiter: ```bash]
cd bindings/javascript
pnpm install
pnpm build
[archived fence delimiter: ```]

## Usage

### How do I create a quantum state?

**C**:
[archived fence delimiter: ```c]
quantum_state_t state;
quantum_state_init(&state, 4);  // 4 qubits
[archived fence delimiter: ```]

**Python**:
[archived fence delimiter: ```python]
from moonlab import QuantumState
state = QuantumState(4)
[archived fence delimiter: ```]

**JavaScript**:
[archived fence delimiter: ```javascript]
import { QuantumState } from '@moonlab/quantum-core';
const state = await QuantumState.create({ numQubits: 4 });
[archived fence delimiter: ```]

### How do I create a Bell state?

Apply Hadamard to qubit 0, then CNOT from 0 to 1:

[archived fence delimiter: ```python]
state = QuantumState(2)
state.h(0).cnot(0, 1)
[archived fence delimiter: ```]

This creates $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$.

### How do I measure qubits?

[archived fence delimiter: ```python]
result = state.measure()  # Collapses state, returns outcome
print(f"Measured: |{result:02b}⟩")
[archived fence delimiter: ```]

**Important**: Measurement collapses the quantum state. Use `clone()` first if you need to preserve the original state.

### How do I check the probability of a specific outcome?

[archived fence delimiter: ```python]
probs = state.probabilities()
print(f"P(|00⟩) = {probs[0]}")
print(f"P(|11⟩) = {probs[3]}")
[archived fence delimiter: ```]

Or for a single outcome:
[archived fence delimiter: ```python]
p = state.probability(0)  # Probability of |00...0⟩
[archived fence delimiter: ```]

### How do I calculate entanglement entropy?

[archived fence delimiter: ```python]
entropy = state.entanglement_entropy([0])  # Entropy of qubit 0
print(f"S = {entropy:.4f} bits")
[archived fence delimiter: ```]

For a Bell state, entropy is 1.0 bit (maximum for 2 qubits).

### What gates are available?

**Single-qubit**: X, Y, Z, H, S, S†, T, T†, Rx, Ry, Rz, Phase
**Two-qubit**: CNOT, CZ, SWAP, CPhase
**Three-qubit**: Toffoli (CCX), Fredkin (CSWAP)

See [Gate Reference](reference/gate-reference.md) for matrices and usage.

### How do I apply a custom unitary?

Currently, you need to decompose custom unitaries into available gates. Universal gate sets like {H, T, CNOT} can approximate any unitary.

## Performance

### How do I enable GPU acceleration?

On macOS with Apple Silicon, Metal is enabled by default. Verify:

[archived fence delimiter: ```bash]
make metal_gpu
./examples/quantum/metal_gpu_benchmark
[archived fence delimiter: ```]

For CUDA (NVIDIA GPUs), see [GPU Acceleration Guide](guides/gpu-acceleration.md).

### Why is my simulation slow?

1. **Too many qubits**: Memory bandwidth limits large simulations
2. **No parallelization**: Ensure OpenMP is installed (`brew install libomp`)
3. **Debug build**: Use `make` (release) not `make CFLAGS="-g -O0"`
4. **Inefficient algorithm**: Reduce circuit depth, use tensor networks for 1D systems

### How much memory do I need?

State vector simulation requires $2^n \times 16$ bytes for $n$ qubits. For 25 qubits, that's 512 MB. For 30 qubits, 16 GB.

### Can I use multiple CPU cores?

Yes, via OpenMP. Set thread count:

[archived fence delimiter: ```bash]
export OMP_NUM_THREADS=8
[archived fence delimiter: ```]

### Can I use distributed computing (MPI)?

Yes, for simulations exceeding single-machine memory. See [Distributed Simulation Guide](guides/distributed-simulation.md).

## Algorithms

### How do I run Grover's search?

[archived fence delimiter: ```python]
from moonlab.algorithms import grover_search

def oracle(state):
    # Mark target state (e.g., |1010⟩)
    state.cz(1, 3)  # Simplified oracle

result = grover_search(4, oracle)
[archived fence delimiter: ```]

See [Grover's Algorithm](algorithms/grovers-algorithm.md) for complete documentation.

### How do I use VQE?

[archived fence delimiter: ```python]
from moonlab.algorithms import vqe
import numpy as np

def ansatz(state, params):
    state.ry(0, params[0])
    state.cnot(0, 1)
    state.ry(1, params[1])

energy, params = vqe(hamiltonian, ansatz, np.random.randn(2))
[archived fence delimiter: ```]

See [VQE Algorithm](algorithms/vqe-algorithm.md) for molecular simulation examples.

### How do I use QAOA?

[archived fence delimiter: ```python]
from moonlab.algorithms import qaoa

result = qaoa(cost_hamiltonian, mixer_hamiltonian, p=3)
[archived fence delimiter: ```]

See [QAOA Algorithm](algorithms/qaoa-algorithm.md) for optimization problems.

### What is DMRG and when should I use it?

DMRG (Density Matrix Renormalization Group) is a tensor network method for simulating 1D systems with limited entanglement. Use it when:

- System is 1D or quasi-1D
- Entanglement follows area law (ground states of gapped Hamiltonians)
- You need more than ~30 qubits

See [DMRG Algorithm](algorithms/dmrg-algorithm.md).

## Troubleshooting

### "Library not found" error

Set the library path:

[archived fence delimiter: ```bash]
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)  # Linux
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(pwd)  # macOS
[archived fence delimiter: ```]

### "Invalid qubit" error

Qubit indices are 0-based and must be less than `num_qubits`:

[archived fence delimiter: ```python]
state = QuantumState(4)  # Qubits 0, 1, 2, 3
state.x(4)  # Error! Max qubit is 3
[archived fence delimiter: ```]

### "Out of memory" error

Reduce the number of qubits or use tensor network methods:

[archived fence delimiter: ```python]
# Instead of 32 qubits with state vector:
from moonlab.tensor import MPS
mps = MPS(100, bond_dimension=64)  # 100 qubits with MPS
[archived fence delimiter: ```]

### Results don't match expectations

1. **Qubit ordering**: Moonlab uses LSB convention: `|q_{n-1}...q_1 q_0\rangle`
2. **Normalization**: Check `state.probabilities().sum()` equals 1.0
3. **Phase**: Global phase doesn't affect measurement probabilities
4. **Randomness**: Measurement is probabilistic—run multiple times

### Python import error

[archived fence delimiter: ```python]
ImportError: cannot find libquantumsim.so
[archived fence delimiter: ```]

Build the library and set path:
[archived fence delimiter: ```bash]
make
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)
[archived fence delimiter: ```]

### WebAssembly memory error

In browsers, memory is limited. Reduce qubits or use Web Workers:

[archived fence delimiter: ```javascript]
// Max ~25 qubits in browser
const state = await QuantumState.create({ numQubits: 20 });  // OK
const big = await QuantumState.create({ numQubits: 30 });    // May fail
[archived fence delimiter: ```]

## Contributing

### How do I report a bug?

Open an issue on [GitHub](https://github.com/tsotchke/moonlab/issues) with:
- Moonlab version
- Platform (OS, CPU)
- Minimal reproduction code
- Expected vs. actual behavior

### How do I contribute code?

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

See [Contributing Guide](contributing/index.md).

### How do I request a feature?

Open a GitHub issue with the "enhancement" label describing:
- Use case
- Proposed solution
- Alternatives considered

## More Help

- [Troubleshooting](troubleshooting.md) - Detailed solutions
- [API Reference](api/index.md) - Complete documentation
- [Tutorials](tutorials/index.md) - Step-by-step guides
- [GitHub Discussions](https://github.com/tsotchke/moonlab/discussions) - Community help
```
