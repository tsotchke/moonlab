# Moonlab Quantum Simulator

A high-performance quantum computing simulation library designed for researchers, developers, and educators in quantum information science. Moonlab provides state-vector simulation for up to 32 qubits with hardware-accelerated performance on Apple Silicon, comprehensive algorithm implementations, and bindings for Python, Rust, and JavaScript.

## Key Features

- **High-Fidelity Simulation**: Full state-vector simulation with double-precision complex amplitudes
- **32-Qubit Capacity**: Simulate quantum systems with up to $2^{32}$ basis states
- **Bell Inequality Verification**: CHSH parameter $S = 2\sqrt{2} \approx 2.828$, demonstrating genuine quantum correlations
- **Hardware Acceleration**: Metal GPU acceleration for Apple Silicon (M1-M4), SIMD optimizations (AVX-512, NEON)
- **Comprehensive Algorithms**: Grover's search, VQE, QAOA, QPE, DMRG, and tensor network methods
- **Multi-Language Support**: C core with Python, Rust, and JavaScript/WebAssembly bindings
- **Distributed Computing**: MPI-based distributed simulation for large-scale computations

## Quick Navigation

| Section | Description |
|---------|-------------|
| [Quick Start](quickstart.md) | Get running in 5 minutes |
| [Installation](installation.md) | Platform-specific installation instructions |
| [Getting Started](getting-started/index.md) | Comprehensive beginner's guide |
| [Concepts](concepts/index.md) | Theoretical foundations |
| [Tutorials](tutorials/index.md) | Step-by-step learning path |
| [API Reference](api/index.md) | Complete API documentation |
| [Algorithms](algorithms/index.md) | Algorithm implementations and theory |
| [Examples](examples/index.md) | Code examples and applications |

## Documentation by Language

<div class="grid-container">

### C Library
The core simulation engine. Optimal for performance-critical applications and embedded systems.
- [C API Reference](api/c/index.md)
- [Building from Source](guides/building-from-source.md)

### Python
High-level interface with NumPy integration and PyTorch quantum layers.
- [Python API Reference](api/python/index.md)
- [Quantum ML Module](api/python/ml.md)

### Rust
Safe, ergonomic bindings with full ownership semantics.
- [Rust API Reference](api/rust/index.md)
- [Terminal UI Application](api/rust/moonlab-tui.md)

### JavaScript
WebAssembly-compiled core with React and Vue components.
- [JavaScript API Reference](api/javascript/index.md)
- [Visualization Library](api/javascript/viz.md)

</div>

## Algorithm Implementations

| Algorithm | Description | Complexity |
|-----------|-------------|------------|
| [Grover's Search](algorithms/grovers-algorithm.md) | Unstructured database search | $O(\sqrt{N})$ |
| [VQE](algorithms/vqe-algorithm.md) | Variational Quantum Eigensolver for molecular ground states | Variational |
| [QAOA](algorithms/qaoa-algorithm.md) | Quantum Approximate Optimization Algorithm | Variational |
| [QPE](algorithms/qpe-algorithm.md) | Quantum Phase Estimation | $O(1/\epsilon)$ |
| [DMRG](algorithms/dmrg-algorithm.md) | Density Matrix Renormalization Group | $O(\chi^3)$ |

## Performance Highlights

- **State-Vector Operations**: Up to 40× speedup with Metal GPU acceleration
- **Grover's Algorithm**: Batch processing of 76 parallel searches on M2 Ultra
- **Tensor Networks**: DMRG sweeps with bond dimension $\chi = 256$ in real-time
- **Memory Efficiency**: 64-byte aligned allocations for optimal SIMD/AMX performance

## Mathematical Foundations

Moonlab implements quantum mechanics on finite-dimensional Hilbert spaces. For an $n$-qubit system, the state space is $\mathcal{H} = (\mathbb{C}^2)^{\otimes n}$ with dimension $2^n$. A pure state is represented as:

$$|\psi\rangle = \sum_{i=0}^{2^n-1} \alpha_i |i\rangle, \quad \sum_{i=0}^{2^n-1} |\alpha_i|^2 = 1$$

where $\alpha_i \in \mathbb{C}$ are probability amplitudes and $|i\rangle$ denotes the computational basis state corresponding to the binary representation of $i$.

## Getting Help

- [FAQ](faq.md) - Frequently asked questions
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [Contributing](contributing/index.md) - How to contribute
- [GitHub Issues](https://github.com/tsotchke/moonlab/issues) - Report bugs and request features

## License

Moonlab Quantum Simulator is open-source software. See the LICENSE file for details.

---

**Version**: 0.1.1
**Last Updated**: January 2026
