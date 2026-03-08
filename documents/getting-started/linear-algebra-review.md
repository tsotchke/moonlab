# Linear Algebra Review

A focused review of linear algebra concepts essential for quantum computing. Each topic includes quantum computing context to illustrate relevance.

## Vector Spaces

### Vectors in $\mathbb{C}^n$

A quantum state is a vector in a complex vector space. For a single qubit, this is $\mathbb{C}^2$:

$$|ψ\rangle = \begin{pmatrix} α \\ β \end{pmatrix}, \quad α, β ∈ \mathbb{C}$$

We use Dirac notation (bra-ket notation):
- **Ket** $|ψ\rangle$: column vector (quantum state)
- **Bra** $\langle ψ|$: row vector (conjugate transpose of ket)

### Basis Vectors

The computational basis for one qubit:

$$|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad |1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

Any qubit state can be written as a linear combination:

$$|ψ\rangle = α|0\rangle + β|1\rangle$$

### Inner Products

The inner product of two vectors $|u\rangle$ and $|v\rangle$ is:

$$\langle u|v\rangle = \sum_i u_i^* v_i$$

Properties:
- $\langle u|v\rangle = \langle v|u\rangle^*$ (conjugate symmetry)
- $\langle u|u\rangle ≥ 0$ (positive semi-definite)
- $\langle u|u\rangle = 0$ if and only if $|u\rangle = 0$

**Quantum interpretation**: $|\langle u|v\rangle|^2$ is the probability of measuring state $|u\rangle$ when the system is in state $|v\rangle$.

**Example**: The inner product of $|0\rangle$ and $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$:

$$\langle 0|+\rangle = \frac{1}{\sqrt{2}}\langle 0|0\rangle + \frac{1}{\sqrt{2}}\langle 0|1\rangle = \frac{1}{\sqrt{2}} · 1 + \frac{1}{\sqrt{2}} · 0 = \frac{1}{\sqrt{2}}$$

### Normalization

Quantum states are unit vectors:

$$\langle ψ|ψ\rangle = |α|^2 + |β|^2 = 1$$

This ensures probabilities sum to 1.

### Orthogonality

Two states are orthogonal if $\langle u|v\rangle = 0$. Orthogonal states are perfectly distinguishable by measurement.

The computational basis states are orthonormal:
- $\langle 0|0\rangle = 1$, $\langle 1|1\rangle = 1$ (normalized)
- $\langle 0|1\rangle = 0$ (orthogonal)

## Matrices

### Matrix Representation

Quantum gates are linear operators represented as matrices. A matrix $A$ acts on a vector:

$$A|ψ\rangle = \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix} \begin{pmatrix} α \\ β \end{pmatrix} = \begin{pmatrix} a_{11}α + a_{12}β \\ a_{21}α + a_{22}β \end{pmatrix}$$

### Conjugate Transpose (Adjoint)

The adjoint $A^†$ (also written $A^*$) is the conjugate transpose:

$$(A^†)_{ij} = A_{ji}^*$$

**Example**:
$$A = \begin{pmatrix} 1 & i \\ 0 & 2 \end{pmatrix} \implies A^† = \begin{pmatrix} 1 & 0 \\ -i & 2 \end{pmatrix}$$

### Unitary Matrices

A matrix $U$ is **unitary** if $U^†U = UU^† = I$.

Properties:
- Preserves inner products: $\langle Uu|Uv\rangle = \langle u|v\rangle$
- Preserves normalization: $\|U|ψ\rangle\| = \||ψ\rangle\|$
- All eigenvalues have modulus 1: $|λ_i| = 1$

**Quantum significance**: All quantum gates are unitary (except measurement). This ensures probability conservation.

**Examples of unitary matrices** (quantum gates):

Pauli-X (NOT gate):
$$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

Hadamard:
$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

Phase gate:
$$S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$$

**Verification**: Check that $H^†H = I$:
$$H^†H = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} = I$$

### Hermitian Matrices

A matrix $H$ is **Hermitian** if $H = H^†$.

Properties:
- All eigenvalues are real
- Eigenvectors of distinct eigenvalues are orthogonal

**Quantum significance**: Observables (measurable quantities) are Hermitian operators. The Pauli matrices are both unitary and Hermitian:

$$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

### Eigenvalues and Eigenvectors

A scalar $λ$ and non-zero vector $|v\rangle$ satisfying $A|v\rangle = λ|v\rangle$ are an eigenvalue-eigenvector pair.

**Example**: Find eigenvalues of $Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$

The eigenvalues are $λ_1 = 1$ and $λ_2 = -1$, with eigenvectors $|0\rangle$ and $|1\rangle$.

**Quantum significance**: Measurement outcomes are eigenvalues of the measured observable. After measurement, the system is in the corresponding eigenstate.

### Spectral Decomposition

A Hermitian matrix can be written as:

$$H = \sum_i λ_i |v_i\rangle\langle v_i|$$

where $λ_i$ are eigenvalues and $|v_i\rangle$ are orthonormal eigenvectors.

**Example**: $Z = (+1)|0\rangle\langle 0| + (-1)|1\rangle\langle 1|$

## Tensor Products

### Definition

The tensor product combines two vector spaces. For vectors:

$$|a\rangle ⊗ |b\rangle = \begin{pmatrix} a_1 \\ a_2 \end{pmatrix} ⊗ \begin{pmatrix} b_1 \\ b_2 \end{pmatrix} = \begin{pmatrix} a_1 b_1 \\ a_1 b_2 \\ a_2 b_1 \\ a_2 b_2 \end{pmatrix}$$

**Notation**: $|a\rangle ⊗ |b\rangle = |a\rangle|b\rangle = |ab\rangle$

### Multi-Qubit States

A 2-qubit state lives in $\mathbb{C}^2 ⊗ \mathbb{C}^2 = \mathbb{C}^4$:

$$|ψ\rangle = α_{00}|00\rangle + α_{01}|01\rangle + α_{10}|10\rangle + α_{11}|11\rangle$$

The computational basis for 2 qubits:

$$|00\rangle = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix}, |01\rangle = \begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix}, |10\rangle = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}, |11\rangle = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}$$

### Tensor Products of Operators

For operators, the tensor product acts componentwise:

$$(A ⊗ B)(|a\rangle ⊗ |b\rangle) = (A|a\rangle) ⊗ (B|b\rangle)$$

Matrix representation (Kronecker product):

$$A ⊗ B = \begin{pmatrix} a_{11}B & a_{12}B \\ a_{21}B & a_{22}B \end{pmatrix}$$

**Example**: $H ⊗ I$ applies Hadamard to qubit 0 and identity to qubit 1:

$$H ⊗ I = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 0 & -1 & 0 \\ 0 & 1 & 0 & -1 \end{pmatrix}$$

### Separable vs. Entangled States

A state is **separable** (product state) if it can be written as $|ψ\rangle = |a\rangle ⊗ |b\rangle$.

A state is **entangled** if it cannot be factored this way.

**Example**: The Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ is entangled.

Proof: Assume $|\Phi^+\rangle = (α|0\rangle + β|1\rangle) ⊗ (γ|0\rangle + δ|1\rangle)$

Expanding: $αγ|00\rangle + αδ|01\rangle + βγ|10\rangle + βδ|11\rangle$

Comparing with $\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$:
- $αγ = 1/\sqrt{2}$
- $αδ = 0$
- $βγ = 0$
- $βδ = 1/\sqrt{2}$

From $αδ = 0$: either $α = 0$ or $δ = 0$.
If $α = 0$, then $αγ = 0 ≠ 1/\sqrt{2}$. Contradiction.
If $δ = 0$, then $βδ = 0 ≠ 1/\sqrt{2}$. Contradiction.

Therefore, no factorization exists—the state is entangled.

## Density Matrices

### Pure States

A pure state $|ψ\rangle$ can be represented as a density matrix:

$$ρ = |ψ\rangle\langle ψ|$$

For $|ψ\rangle = α|0\rangle + β|1\rangle$:

$$ρ = \begin{pmatrix} |α|^2 & αβ^* \\ α^*β & |β|^2 \end{pmatrix}$$

Properties of pure state density matrices:
- $ρ^2 = ρ$ (idempotent)
- $\text{Tr}(ρ) = 1$
- $\text{Tr}(ρ^2) = 1$

### Mixed States

A statistical ensemble of quantum states:

$$ρ = \sum_i p_i |ψ_i\rangle\langle ψ_i|$$

where $p_i ≥ 0$ and $\sum_i p_i = 1$.

For mixed states: $\text{Tr}(ρ^2) < 1$.

**Quantum significance**: Mixed states describe subsystems of entangled systems, decoherence, and thermal states.

### Partial Trace

The partial trace over subsystem $B$ of a bipartite state $ρ_{AB}$:

$$ρ_A = \text{Tr}_B(ρ_{AB}) = \sum_i (I_A ⊗ \langle i|_B) ρ_{AB} (I_A ⊗ |i\rangle_B)$$

**Example**: For the Bell state $|\Phi^+\rangle$:

$$ρ_{AB} = |\Phi^+\rangle\langle \Phi^+| = \frac{1}{2}(|00\rangle\langle 00| + |00\rangle\langle 11| + |11\rangle\langle 00| + |11\rangle\langle 11|)$$

Tracing out $B$:

$$ρ_A = \text{Tr}_B(ρ_{AB}) = \frac{1}{2}(|0\rangle\langle 0| + |1\rangle\langle 1|) = \frac{1}{2}I$$

The reduced density matrix is maximally mixed—this is characteristic of maximal entanglement.

## Summary

| Concept | Quantum Computing Application |
|---------|------------------------------|
| Complex vectors | Quantum states |
| Inner product | Probability amplitudes, overlap |
| Unitary matrices | Quantum gates |
| Hermitian matrices | Observables |
| Eigenvalues | Measurement outcomes |
| Tensor product | Multi-qubit systems |
| Density matrices | Mixed states, subsystems |
| Partial trace | Entanglement entropy |

## Exercises

1. Verify that the Hadamard gate is unitary.

2. Compute $(H ⊗ H)|00\rangle$.

3. Find the eigenvalues and eigenvectors of the Hadamard gate.

4. Prove that $(A ⊗ B)^† = A^† ⊗ B^†$.

5. Calculate the reduced density matrix $ρ_A$ for the state $|ψ\rangle = \frac{1}{\sqrt{2}}|00\rangle + \frac{1}{2}|01\rangle + \frac{1}{2}|10\rangle$.

## Next

Proceed to [Quantum Mechanics Primer](quantum-mechanics-primer.md) to learn how these mathematical structures describe physical quantum systems.
