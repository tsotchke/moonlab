# Prerequisites

This document outlines the mathematical and programming knowledge needed to effectively use Moonlab Quantum Simulator. Assess your current knowledge and identify areas for review.

## Mathematical Background

### Essential: Complex Numbers

Quantum computing fundamentally relies on complex numbers. You should be comfortable with:

**Basic operations**
- Addition, multiplication, division of complex numbers
- Complex conjugate: $z^* = a - bi$ for $z = a + bi$
- Modulus (absolute value): $|z| = \sqrt{a^2 + b^2}$

**Euler's formula**
$$e^{i\theta} = \cos\theta + i\sin\theta$$

This appears constantly in quantum gates. For example, the phase gate applies $e^{i\phi}$.

**Polar form**
$$z = re^{i\theta} = r(\cos\theta + i\sin\theta)$$

Useful for understanding rotation gates and global phases.

**Self-assessment**: Can you compute $(1+i)(1-i)$? What is $|e^{i\pi/4}|$?

Answers: $(1+i)(1-i) = 2$; $|e^{i\pi/4}| = 1$ (all points on the unit circle have modulus 1).

### Essential: Linear Algebra

Quantum mechanics is linear algebra over $\mathbb{C}$. Required concepts:

**Vectors and inner products**
- Column vectors in $\mathbb{C}^n$
- Inner product: $\langle u|v\rangle = \sum_i u_i^* v_i$
- Norm: $\|v\| = \sqrt{\langle v|v\rangle}$

**Matrices**
- Matrix multiplication
- Transpose: $(A^T)_{ij} = A_{ji}$
- Conjugate transpose (adjoint): $(A^\dagger)_{ij} = A_{ji}^*$

**Special matrices**
- Unitary: $U^\dagger U = UU^\dagger = I$
- Hermitian: $H = H^\dagger$
- Eigenvalues and eigenvectors

**Tensor products**
- $|a\rangle \otimes |b\rangle$ (also written $|a\rangle|b\rangle$ or $|ab\rangle$)
- For multi-qubit systems

See [Linear Algebra Review](linear-algebra-review.md) for a complete treatment.

**Self-assessment**: What is the adjoint of $\begin{pmatrix} 1 & i \\ 0 & 1 \end{pmatrix}$?

Answer: $\begin{pmatrix} 1 & 0 \\ -i & 1 \end{pmatrix}$

### Essential: Probability

Quantum measurement is inherently probabilistic:

- Probability distributions
- Expected values
- Conditional probability (for understanding measurement)

**Key concept**: Born rule—the probability of measuring outcome $i$ is $|\alpha_i|^2$ where $\alpha_i$ is the amplitude.

**Self-assessment**: If $|\psi\rangle = \frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}|1\rangle$, what's the probability of measuring $|1\rangle$?

Answer: $|\sqrt{2/3}|^2 = 2/3$

### Helpful: Discrete Mathematics

Not strictly required but useful for algorithms:

- Boolean logic
- Graph theory basics (for QAOA)
- Combinatorics (for Grover's algorithm)

### Helpful: Calculus

Useful for variational algorithms (VQE, QAOA):

- Derivatives and gradients
- Optimization concepts
- Taylor series (for understanding gate approximations)

## Programming Background

### Essential: C Programming (for C API)

If using the C library directly:

- Variables and data types
- Functions and pointers
- Memory management (`malloc`, `free`)
- Structs
- Header files and compilation

**Minimum example you should understand**:
```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    double *data;
    size_t size;
} Vector;

int main(void) {
    Vector v;
    v.size = 4;
    v.data = malloc(v.size * sizeof(double));
    // ... use v ...
    free(v.data);
    return 0;
}
```

### Alternative: Python (for Python API)

If using Python bindings:

- Basic Python syntax
- NumPy arrays
- Object-oriented programming basics

**Minimum example**:
```python
import numpy as np

class QuantumState:
    def __init__(self, n_qubits):
        self.amplitudes = np.zeros(2**n_qubits, dtype=complex)
        self.amplitudes[0] = 1.0  # |00...0⟩

state = QuantumState(2)
print(state.amplitudes)
```

### Alternative: JavaScript/TypeScript (for Web API)

If using JavaScript bindings:

- Modern JavaScript (ES6+)
- Promises and async/await
- TypeScript basics (recommended)

**Minimum example**:
```typescript
interface QuantumState {
    numQubits: number;
    amplitudes: Float64Array;
}

function createState(n: number): QuantumState {
    const dim = 2 ** n;
    const amplitudes = new Float64Array(dim * 2); // real, imag pairs
    amplitudes[0] = 1.0; // |0⟩ amplitude
    return { numQubits: n, amplitudes };
}
```

## No Prerequisites Required

You do **not** need prior knowledge of:

- Quantum mechanics or physics
- Previous quantum computing experience
- Specialized mathematics beyond the above

The [Quantum Mechanics Primer](quantum-mechanics-primer.md) covers all necessary physics from scratch.

## Recommended Resources

### Complex Numbers
- Khan Academy: Complex Numbers
- 3Blue1Brown: Euler's formula with introductory group theory

### Linear Algebra
- 3Blue1Brown: Essence of Linear Algebra (YouTube series)
- MIT OpenCourseWare: 18.06 Linear Algebra
- Axler, *Linear Algebra Done Right*

### Quantum Computing Primers
- Nielsen & Chuang, *Quantum Computation and Quantum Information* (the standard reference)
- Mermin, *Quantum Computer Science: An Introduction* (gentler introduction)
- IBM Qiskit Textbook (free online)

### Programming
- K&R, *The C Programming Language* (for C)
- Python official tutorial (for Python)
- TypeScript Handbook (for JavaScript/TypeScript)

## Knowledge Checklist

Before proceeding, ensure you can:

- [ ] Multiply complex numbers
- [ ] Compute the modulus of a complex number
- [ ] Multiply 2×2 matrices
- [ ] Understand what an eigenvector is
- [ ] Calculate simple probabilities
- [ ] Write and compile a basic program in your chosen language

If you're missing any of these, review the relevant resources above or proceed to the [Linear Algebra Review](linear-algebra-review.md) for a focused refresher.

## Next

Proceed to:
- [Linear Algebra Review](linear-algebra-review.md) if you need a math refresher
- [Quantum Mechanics Primer](quantum-mechanics-primer.md) if your math is solid
- [First Simulation](first-simulation.md) if you're ready to code
