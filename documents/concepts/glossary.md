# Glossary

A comprehensive reference of terminology used in quantum computing and throughout Moonlab documentation.

---

## A

### Amplitude
A complex number $\alpha_i$ in the state vector expansion $|\psi\rangle = \sum_i \alpha_i |i\rangle$. The probability of measuring outcome $i$ is $|\alpha_i|^2$.

### Ancilla
An auxiliary qubit used to assist computation but not part of the final output. Often used in error correction and phase estimation.

### Ansatz
A parameterized circuit structure used in variational algorithms. From German for "approach" or "starting point." Example: the hardware-efficient ansatz in VQE.

---

## B

### Basis
A set of linearly independent vectors that span a vector space. The **computational basis** for $n$ qubits is $\{|0\rangle, |1\rangle, \ldots, |2^n-1\rangle\}$.

### Bell State
One of four maximally entangled 2-qubit states:
- $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$
- $|\Phi^-\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$
- $|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$
- $|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$

### Bloch Sphere
A geometric representation of single-qubit pure states as points on a unit sphere in 3D. North pole is $|0\rangle$, south pole is $|1\rangle$.

### Bond Dimension
In tensor networks, the dimension of indices connecting tensors. Higher bond dimension captures more entanglement. Denoted $\chi$ in DMRG.

### Born Rule
The postulate that measurement outcome $i$ occurs with probability $P(i) = |\langle i|\psi\rangle|^2$.

### Bra
The conjugate transpose of a ket: $\langle\psi| = (|\psi\rangle)^\dagger$. Together with kets, forms Dirac notation.

---

## C

### CHSH Inequality
A Bell inequality: $|S| \leq 2$ classically, where $S$ combines correlation measurements. Quantum mechanics allows $|S| \leq 2\sqrt{2} \approx 2.828$.

### CNOT Gate
Controlled-NOT gate. Flips target qubit if control is $|1\rangle$:
$$\text{CNOT} = \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\ 0&0&0&1 \\ 0&0&1&0 \end{pmatrix}$$

### Coherence
The property of maintaining quantum superposition. Loss of coherence (decoherence) destroys quantum information.

### Collapse
The projection of a quantum state onto a measurement outcome. Also called wavefunction collapse.

### Computational Basis
The standard basis $\{|0\rangle, |1\rangle\}^{\otimes n}$ for $n$ qubits, corresponding to classical bit strings.

### Controlled Gate
A multi-qubit gate where one qubit (control) determines whether an operation is applied to another qubit (target).

---

## D

### Decoherence
Loss of quantum coherence due to interaction with the environment. A major source of errors in quantum computers.

### Density Matrix
A representation of quantum states (pure or mixed) as $\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$. For pure states, $\rho = |\psi\rangle\langle\psi|$.

### DMRG
Density Matrix Renormalization Group. A tensor network algorithm for finding ground states of 1D systems. Extremely efficient for low-entanglement states.

---

## E

### Eigenvalue
A scalar $\lambda$ such that $A|v\rangle = \lambda|v\rangle$ for some non-zero vector $|v\rangle$ (the eigenvector).

### Entanglement
Quantum correlations between particles that cannot be described by classical physics. Entangled states cannot be written as product states.

### Entanglement Entropy
The von Neumann entropy of the reduced density matrix: $S(\rho_A) = -\text{Tr}(\rho_A \log_2 \rho_A)$. Measures entanglement in pure bipartite states.

---

## F

### Fidelity
A measure of similarity between quantum states: $F(\rho, \sigma) = (\text{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}})^2$. For pure states, $F = |\langle\psi|\phi\rangle|^2$.

---

## G

### Gate
A unitary operation on qubits. Quantum gates are reversible and preserve probability.

### GHZ State
Greenberger-Horne-Zeilinger state. A maximally entangled $n$-qubit state:
$$|GHZ\rangle = \frac{1}{\sqrt{2}}(|00\cdots0\rangle + |11\cdots1\rangle)$$

### Global Phase
A phase factor $e^{i\gamma}$ applied to the entire state. Global phases have no physical effect.

### Grover's Algorithm
A quantum search algorithm achieving $O(\sqrt{N})$ complexity for unstructured search, compared to $O(N)$ classically.

---

## H

### Hadamard Gate
Creates superposition:
$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$
$H|0\rangle = |+\rangle$, $H|1\rangle = |-\rangle$.

### Hamiltonian
A Hermitian operator representing the total energy of a system. Governs time evolution via the Schrödinger equation.

### Hermitian
A matrix equal to its conjugate transpose: $H = H^\dagger$. Observables are Hermitian operators.

### Hilbert Space
A complete inner product space. Quantum states live in Hilbert spaces.

---

## I

### Inner Product
For vectors $|u\rangle$ and $|v\rangle$: $\langle u|v\rangle = \sum_i u_i^* v_i$.

### Interference
The phenomenon where amplitudes add (constructive) or cancel (destructive). Key to quantum algorithm speedups.

---

## K

### Ket
Dirac notation for a column vector: $|ψ\rangle$. Part of bra-ket notation.

### Kraus Operators
Operators $\{E_k\}$ describing quantum channels: $\rho \to \sum_k E_k \rho E_k^\dagger$ with $\sum_k E_k^\dagger E_k = I$.

---

## M

### Maximally Entangled
A bipartite state with maximum entanglement entropy. For 2 qubits, the Bell states have $S = 1$ bit.

### Measurement
Extracting classical information from a quantum system. Causes state collapse and is probabilistic.

### Mixed State
A statistical ensemble of pure states: $\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$. Has $\text{Tr}(\rho^2) < 1$.

### MPS
Matrix Product State. A tensor network representation efficient for 1D systems with limited entanglement.

### MPO
Matrix Product Operator. Tensor network representation of operators, used to represent Hamiltonians in DMRG.

---

## N

### No-Cloning Theorem
Quantum states cannot be perfectly copied. Fundamental to quantum cryptography.

### Normalization
The requirement that $\langle\psi|\psi\rangle = 1$, ensuring probabilities sum to 1.

---

## O

### Observable
A Hermitian operator representing a measurable physical quantity. Eigenvalues are possible measurement outcomes.

### Oracle
A black-box function in quantum algorithms. Typically implemented as a unitary that marks target states.

### Outer Product
For vectors $|u\rangle$ and $|v\rangle$: $|u\rangle\langle v|$, a matrix.

---

## P

### Pauli Matrices
The three 2×2 Hermitian unitary matrices:
$$X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$$

### Phase
The argument of a complex amplitude. Relative phases affect interference.

### Product State
A multi-qubit state that can be factored: $|\psi\rangle = |\phi_1\rangle \otimes |\phi_2\rangle \otimes \cdots$. Not entangled.

### Projector
An operator $P$ with $P^2 = P$. Measurement projectors are $|i\rangle\langle i|$.

### Purity
$\text{Tr}(\rho^2)$. Equals 1 for pure states, less than 1 for mixed states.

---

## Q

### QAOA
Quantum Approximate Optimization Algorithm. A variational algorithm for combinatorial optimization.

### QPE
Quantum Phase Estimation. Estimates eigenvalues of unitary operators.

### Qubit
Quantum bit. The fundamental unit of quantum information, existing in superposition of $|0\rangle$ and $|1\rangle$.

---

## R

### Reduced Density Matrix
The density matrix of a subsystem obtained by partial trace: $\rho_A = \text{Tr}_B(\rho_{AB})$.

### Rotation Gate
Parameterized single-qubit gates:
$$R_x(\theta) = e^{-i\theta X/2}, \quad R_y(\theta) = e^{-i\theta Y/2}, \quad R_z(\theta) = e^{-i\theta Z/2}$$

---

## S

### Separable
A state that can be written as a convex combination of product states. Opposite of entangled.

### State Vector
The vector $|\psi\rangle$ representing a quantum state in Hilbert space.

### Superposition
A quantum state existing in multiple basis states simultaneously: $|\psi\rangle = \sum_i \alpha_i |i\rangle$.

### SWAP Gate
Exchanges two qubits: $\text{SWAP}|ab\rangle = |ba\rangle$.

---

## T

### T Gate
$\pi/8$ phase gate:
$$T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{pmatrix}$$
Important for universal quantum computation.

### Tensor Product
Operation combining vector spaces: $\mathcal{H}_A \otimes \mathcal{H}_B$. For vectors: $(|a\rangle \otimes |b\rangle)$.

### Toffoli Gate
3-qubit controlled-controlled-NOT gate. Universal for classical reversible computation.

### Trace
Sum of diagonal elements: $\text{Tr}(A) = \sum_i A_{ii}$. For density matrices, $\text{Tr}(\rho) = 1$.

---

## U

### Unitary
A matrix satisfying $U^\dagger U = UU^\dagger = I$. All quantum gates are unitary.

### Universal Gate Set
A set of gates from which any unitary can be approximated. Examples: $\{H, T, \text{CNOT}\}$.

---

## V

### Variational Algorithm
Hybrid quantum-classical algorithm using parameterized circuits optimized classically. Examples: VQE, QAOA.

### Von Neumann Entropy
$S(\rho) = -\text{Tr}(\rho \log_2 \rho)$. Measures information content or entanglement.

### VQE
Variational Quantum Eigensolver. Algorithm for finding ground state energies of Hamiltonians.

---

## W

### W State
An entangled $n$-qubit state:
$$|W\rangle = \frac{1}{\sqrt{n}}(|100\cdots0\rangle + |010\cdots0\rangle + \cdots + |000\cdots1\rangle)$$

### Wavefunction
The quantum state $|\psi\rangle$, or in position representation, $\psi(x)$.

---

## Notation Quick Reference

| Notation | Meaning |
|----------|---------|
| $\|ψ\rangle$ | Ket (column vector) |
| $\langle ψ\|$ | Bra (row vector) |
| $\langle φ\|ψ\rangle$ | Inner product |
| $\|φ\rangle\langle ψ\|$ | Outer product |
| $⊗$ | Tensor product |
| $A^†$ | Conjugate transpose |
| $\text{Tr}(A)$ | Trace |
| $[A, B]$ | Commutator $AB - BA$ |
| $\{A, B\}$ | Anticommutator $AB + BA$ |
