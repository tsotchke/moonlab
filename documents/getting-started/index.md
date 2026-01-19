# Getting Started

Welcome to quantum computing with Moonlab. This guide provides a structured learning path from foundational concepts to practical simulation skills.

## Learning Roadmap

```
                    ┌─────────────────────────────────────────┐
                    │          Getting Started                │
                    │                                         │
    ┌───────────────┼──────────────┬──────────────────────────┤
    ▼               ▼              ▼                          ▼
┌────────┐   ┌────────────┐   ┌────────────┐          ┌────────────┐
│Prereqs │ → │ QM Primer  │ → │Linear Alg. │    →     │First Sim   │
└────────┘   └────────────┘   └────────────┘          └────────────┘
    │               │              │                          │
    │    Background knowledge      │        Hands-on          │
    └──────────────────────────────┴──────────────────────────┘
                                   │
                                   ▼
                          ┌────────────────┐
                          │   Next Steps   │
                          │ • Tutorials    │
                          │ • Algorithms   │
                          │ • API Docs     │
                          └────────────────┘
```

## Path Selection

Choose your starting point based on your background:

### Path A: Complete Beginner
If you're new to quantum computing, follow this sequence:
1. [Prerequisites](prerequisites.md) - Required mathematical background
2. [Linear Algebra Review](linear-algebra-review.md) - Vectors, matrices, tensor products
3. [Quantum Mechanics Primer](quantum-mechanics-primer.md) - Essential physics concepts
4. [First Simulation](first-simulation.md) - Build your first circuit
5. [Next Steps](next-steps.md) - Continue learning

### Path B: Physics/Math Background
If you have quantum mechanics or advanced mathematics experience:
1. [Prerequisites](prerequisites.md) - Verify your background
2. [First Simulation](first-simulation.md) - Jump into coding
3. [Next Steps](next-steps.md) - Explore advanced topics

### Path C: Software Developer
If you're comfortable with programming but new to quantum:
1. [Quantum Mechanics Primer](quantum-mechanics-primer.md) - Core concepts
2. [First Simulation](first-simulation.md) - Practical coding
3. [Next Steps](next-steps.md) - API and algorithm deep-dives

## What You'll Learn

By the end of this guide, you will understand:

| Concept | Description |
|---------|-------------|
| **Qubits** | The fundamental unit of quantum information |
| **Superposition** | How qubits exist in multiple states simultaneously |
| **Quantum Gates** | Operations that transform quantum states |
| **Entanglement** | Non-classical correlations between qubits |
| **Measurement** | Extracting classical information from quantum states |

And you'll be able to:

- Create and manipulate quantum states in code
- Apply quantum gates to build circuits
- Measure quantum states and interpret results
- Calculate entanglement entropy and fidelity
- Run simulations up to 32 qubits

## Time Investment

| Section | Estimated Time |
|---------|---------------|
| Prerequisites | 10 minutes (review) |
| Linear Algebra Review | 30-60 minutes |
| Quantum Mechanics Primer | 45-90 minutes |
| First Simulation | 30-45 minutes |

Total: approximately 2-4 hours for the complete beginner path.

## Section Overview

### [Prerequisites](prerequisites.md)
Mathematical and programming knowledge required for quantum computing. Covers linear algebra basics, complex numbers, probability theory, and programming requirements.

### [Linear Algebra Review](linear-algebra-review.md)
A focused review of the linear algebra essential for quantum computing: vector spaces, matrices, eigenvalues, and tensor products. Includes worked examples with quantum interpretations.

### [Quantum Mechanics Primer](quantum-mechanics-primer.md)
The physics concepts underlying quantum computation: state vectors, the Born rule, unitary evolution, and the measurement postulate. No prior physics required.

### [First Simulation](first-simulation.md)
A complete walkthrough of building, running, and understanding your first quantum simulation. Covers state initialization, gate application, and measurement.

### [Next Steps](next-steps.md)
Guidance on continuing your quantum computing journey: tutorials, algorithm implementations, and advanced topics.

## Quick Reference

### The Qubit

A qubit is described by a unit vector in $\mathbb{C}^2$:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \quad |\alpha|^2 + |\beta|^2 = 1$$

where $\alpha, \beta \in \mathbb{C}$ are probability amplitudes.

### Common Gates

| Gate | Matrix | Effect |
|------|--------|--------|
| X (NOT) | $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ | Bit flip: $\|0\rangle \leftrightarrow \|1\rangle$ |
| H (Hadamard) | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ | Creates superposition |
| CNOT | $\begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\ 0&0&0&1 \\ 0&0&1&0 \end{pmatrix}$ | Entangles two qubits |

### Bell State Circuit

```
|0⟩ ──[H]──●──   →   |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
           │
|0⟩ ───────⊕──
```

## Getting Help

- [FAQ](../faq.md) - Frequently asked questions
- [Troubleshooting](../troubleshooting.md) - Common issues and solutions
- [Glossary](../concepts/glossary.md) - Terminology reference
- [GitHub Issues](https://github.com/tsotchke/moonlab/issues) - Report bugs or ask questions
