# Changelog

All notable changes to MoonLab Quantum Simulator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
