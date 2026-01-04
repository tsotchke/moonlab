# Algorithm Documentation

Deep dives into quantum algorithms implemented in Moonlab.

## Overview

This section provides comprehensive documentation of quantum algorithms, including mathematical foundations, implementation details, complexity analysis, and practical usage guidance.

## Algorithm Categories

### Search Algorithms

| Algorithm | Speedup | Use Case |
|-----------|---------|----------|
| [Grover's Search](grovers-algorithm.md) | $O(\sqrt{N})$ | Unstructured search |

### Variational Algorithms

| Algorithm | Type | Use Case |
|-----------|------|----------|
| [VQE](vqe-algorithm.md) | Hybrid | Molecular ground states |
| [QAOA](qaoa-algorithm.md) | Hybrid | Combinatorial optimization |

### Eigenvalue Algorithms

| Algorithm | Output | Use Case |
|-----------|--------|----------|
| [QPE](qpe-algorithm.md) | Phase estimation | Eigenvalue extraction |

### Tensor Network Methods

| Algorithm | Scaling | Use Case |
|-----------|---------|----------|
| [DMRG](dmrg-algorithm.md) | $O(n\chi^3)$ | 1D ground states |
| [TDVP](tdvp-algorithm.md) | $O(n\chi^3)$ | Real-time dynamics |

### Topological Quantum Computing

| Algorithm | Approach | Use Case |
|-----------|----------|----------|
| [Topological Computing](topological-computing.md) | Anyons | Fault-tolerant qubits |
| [Skyrmion Braiding](skyrmion-braiding.md) | Magnetic skyrmions | Topological gates |

### Fundamental Tests

| Protocol | Purpose |
|----------|---------|
| [Bell/CHSH Test](bell-chsh-test.md) | Entanglement verification |

## Algorithm Selection Guide

### Finding Ground States

```
Is your system 1D or quasi-1D?
├─ Yes: Use DMRG (handles 100+ sites)
└─ No: Is it small enough for state vector?
    ├─ Yes (≤30 qubits): Use VQE
    └─ No: Consider hybrid classical-quantum methods
```

### Optimization Problems

```
Is the problem naturally quadratic?
├─ Yes: Use QAOA directly
└─ No: Can it be reduced to QUBO?
    ├─ Yes: Transform and use QAOA
    └─ No: Consider specialized encodings
```

### Search Problems

```
Is the search space unstructured?
├─ Yes: Use Grover's algorithm
└─ No: Does it have exploitable structure?
    ├─ Yes: Use problem-specific algorithm
    └─ No: Grover's still provides speedup
```

## Common Patterns

### Variational Loop

All variational algorithms (VQE, QAOA) share this structure:

```
1. Initialize parameters θ
2. Prepare quantum state |ψ(θ)⟩
3. Measure expectation value ⟨H⟩
4. Classical optimizer updates θ
5. Repeat until convergence
```

### Oracle Construction

Grover's algorithm and many others require oracles:

```
Oracle marks solutions: |x⟩ → (-1)^f(x)|x⟩

Construction approaches:
- Phase kickback with ancilla
- Direct phase manipulation
- Compiled arithmetic circuits
```

### Amplitude Amplification

Generalization of Grover's technique:

$$\mathcal{Q} = -AS_0A^{-1}S_f$$

where $A$ is any preparation operator.

## Complexity Summary

| Algorithm | Query Complexity | Gate Complexity | Classical Comparison |
|-----------|-----------------|-----------------|---------------------|
| Grover | $O(\sqrt{N})$ | $O(\sqrt{N} \cdot n)$ | $O(N)$ |
| VQE | Problem-dependent | $O(p \cdot n)$ per iteration | Exponential |
| QAOA | $O(p)$ | $O(p \cdot m)$ | NP-hard problems |
| QPE | $O(1/\epsilon)$ | $O(n^2/\epsilon)$ | Exponential |
| DMRG | N/A (classical) | N/A | Polynomial in $\chi$ |

Where:
- $N = 2^n$ (search space size)
- $n$ = number of qubits
- $p$ = circuit depth / layers
- $m$ = number of edges (for QAOA on graphs)
- $\chi$ = bond dimension (for DMRG)
- $\epsilon$ = precision

## Implementation Quality

All algorithms in Moonlab are implemented with:

1. **Numerical Stability**: Careful handling of floating-point arithmetic
2. **Performance Optimization**: SIMD vectorization, GPU acceleration
3. **Error Handling**: Comprehensive validation and error messages
4. **Testing**: Extensive unit and integration tests

## Further Reading

- [Tutorials](../tutorials/index.md) - Step-by-step algorithm tutorials
- [C API Reference](../api/c/index.md) - Low-level implementation details
- [Python API](../api/python/algorithms.md) - High-level algorithm interface

## References

1. Grover, L. K. (1996). "A fast quantum mechanical algorithm for database search." STOC '96.
2. Peruzzo, A. et al. (2014). "A variational eigenvalue solver on a photonic quantum processor." Nature Communications.
3. Farhi, E. et al. (2014). "A Quantum Approximate Optimization Algorithm." arXiv:1411.4028.
4. Kitaev, A. (1995). "Quantum measurements and the Abelian stabilizer problem." arXiv:quant-ph/9511026.
5. White, S. R. (1992). "Density matrix formulation for quantum renormalization groups." Physical Review Letters.

