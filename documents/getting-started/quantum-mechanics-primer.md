# Quantum Mechanics Primer

This primer introduces the physical principles underlying quantum computation. No prior physics knowledge is required—we build from the mathematical foundations covered in the [Linear Algebra Review](linear-algebra-review.md).

## The Postulates of Quantum Mechanics

Quantum computing rests on four postulates that describe how quantum systems behave.

### Postulate 1: State Space

**The state of a closed quantum system is described by a unit vector in a complex Hilbert space.**

For a single qubit, the Hilbert space is $\mathcal{H} = \mathbb{C}^2$. A general qubit state is:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

where $\alpha, \beta \in \mathbb{C}$ and $|\alpha|^2 + |\beta|^2 = 1$.

**Physical interpretation**:
- $|0\rangle$ and $|1\rangle$ represent two distinguishable physical configurations (e.g., spin up/down, ground/excited state)
- The amplitudes $\alpha$ and $\beta$ encode the "amount" of each basis state
- Normalization ensures total probability is 1

**Key insight**: Unlike classical bits, a qubit can exist in a **superposition**—a coherent combination of both $|0\rangle$ and $|1\rangle$ simultaneously.

### Postulate 2: Evolution

**The evolution of a closed quantum system is described by a unitary transformation.**

$$|\psi(t)\rangle = U(t)|\psi(0)\rangle$$

where $U$ is unitary: $U^\dagger U = I$.

**Physical interpretation**:
- Quantum evolution is deterministic and reversible
- Information is conserved (unitary transformations are invertible)
- Quantum gates are discrete unitary operations

**The Schrödinger equation** governs continuous evolution:

$$i\hbar \frac{d|\psi\rangle}{dt} = H|\psi\rangle$$

For a time-independent Hamiltonian $H$:

$$U(t) = e^{-iHt/\hbar}$$

In quantum computing, we typically work with discrete gates rather than continuous evolution, but understanding this connection is valuable for algorithms like VQE that use parameterized rotations.

### Postulate 3: Measurement

**Quantum measurement is described by a collection of measurement operators $\{M_m\}$ satisfying $\sum_m M_m^\dagger M_m = I$.**

The probability of outcome $m$ is:

$$p(m) = \langle\psi|M_m^\dagger M_m|\psi\rangle$$

The post-measurement state is:

$$|\psi'\rangle = \frac{M_m|\psi\rangle}{\sqrt{p(m)}}$$

**Projective measurement** (the most common case): Measuring an observable $O$ with spectral decomposition $O = \sum_m m|m\rangle\langle m|$:

$$p(m) = |\langle m|\psi\rangle|^2$$

$$|\psi'\rangle = |m\rangle$$

**Physical interpretation**:
- Measurement is probabilistic—we cannot predict individual outcomes
- Measurement **collapses** the superposition to an eigenstate
- The act of measurement irreversibly changes the state
- This is the source of quantum randomness

**Example**: Measuring $|\psi\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$ in the computational basis:

- Probability of $|0\rangle$: $|1/\sqrt{2}|^2 = 1/2$
- Probability of $|1\rangle$: $|1/\sqrt{2}|^2 = 1/2$
- After measurement, state collapses to either $|0\rangle$ or $|1\rangle$

### Postulate 4: Composite Systems

**The state space of a composite system is the tensor product of the component state spaces.**

For two qubits: $\mathcal{H}_{AB} = \mathcal{H}_A \otimes \mathcal{H}_B = \mathbb{C}^2 \otimes \mathbb{C}^2 = \mathbb{C}^4$

A general 2-qubit state:

$$|\psi\rangle = \alpha_{00}|00\rangle + \alpha_{01}|01\rangle + \alpha_{10}|10\rangle + \alpha_{11}|11\rangle$$

**Physical interpretation**:
- Multi-qubit systems have exponentially large state spaces ($2^n$ dimensions for $n$ qubits)
- This exponential scaling is the source of quantum computational power
- Some multi-qubit states cannot be factored—these are **entangled**

## Superposition

Superposition is the principle that a quantum system can exist in multiple basis states simultaneously.

### Mathematical Description

A superposition state:

$$|\psi\rangle = \sum_i \alpha_i |i\rangle, \quad \sum_i |\alpha_i|^2 = 1$$

The complex amplitudes $\alpha_i$ contain both magnitude and phase information.

### Creating Superpositions

The Hadamard gate creates superposition from a basis state:

$$H|0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \equiv |+\rangle$$

$$H|1\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle) \equiv |-\rangle$$

The $|+\rangle$ and $|-\rangle$ states form the **Hadamard basis** (or X-basis).

### Interference

Quantum amplitudes can interfere constructively or destructively:

$$H|+\rangle = H \cdot H|0\rangle = |0\rangle$$

The $|1\rangle$ component cancels due to destructive interference between the $+$ and $-$ amplitudes.

**This is crucial for quantum algorithms**: Grover's algorithm works by arranging constructive interference on the correct answer and destructive interference on wrong answers.

## Entanglement

Entanglement is a uniquely quantum correlation between subsystems that has no classical analog.

### Definition

A state $|\psi\rangle_{AB}$ is **entangled** if it cannot be written as a product state:

$$|\psi\rangle_{AB} \neq |\phi\rangle_A \otimes |\chi\rangle_B$$

### The Bell States

The four maximally entangled 2-qubit states:

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$
$$|\Phi^-\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$$
$$|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$$
$$|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$$

### Properties of Entanglement

**Non-local correlations**: Measurements on entangled particles are correlated regardless of spatial separation.

For $|\Phi^+\rangle$:
- If Alice measures $|0\rangle$, Bob must measure $|0\rangle$
- If Alice measures $|1\rangle$, Bob must measure $|1\rangle$
- These correlations are instantaneous and deterministic (given the measurement outcome)

**Monogamy**: Maximum entanglement between two qubits precludes entanglement with a third.

**No cloning**: Entangled states cannot be copied—a consequence of the no-cloning theorem.

### Quantifying Entanglement

The **von Neumann entropy** of the reduced density matrix measures entanglement:

$$S(\rho_A) = -\text{Tr}(\rho_A \log_2 \rho_A)$$

For a pure bipartite state:
- $S = 0$: Product state (no entanglement)
- $S = 1$: Maximally entangled (for 2 qubits)

The Bell states have $S = 1$ bit of entanglement.

### Bell's Theorem

Bell's theorem proves that quantum correlations cannot be explained by local hidden variables. The CHSH inequality:

$$|S| \leq 2 \quad \text{(classical)}$$

where $S$ is a combination of correlation measurements. Quantum mechanics allows:

$$|S| \leq 2\sqrt{2} \approx 2.828$$

Moonlab simulates violations of Bell's inequality, demonstrating genuine quantum behavior.

## The Born Rule

The Born rule connects the mathematical formalism to experimental predictions.

**Statement**: The probability of measuring outcome $m$ when the system is in state $|\psi\rangle$ is:

$$P(m) = |\langle m|\psi\rangle|^2$$

**Implications**:
- Only probabilities are predicted, not individual outcomes
- Amplitudes (complex numbers) become probabilities (real, non-negative)
- Phase information affects interference but not individual measurement probabilities

## Quantum Gates

Quantum gates are unitary operations that transform quantum states.

### Single-Qubit Gates

The Bloch sphere provides geometric intuition. Any single-qubit state can be written:

$$|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle$$

where $\theta \in [0, \pi]$ and $\phi \in [0, 2\pi)$ are spherical coordinates.

**Pauli gates** are 180° rotations:
- $X$: Rotation about x-axis (bit flip)
- $Y$: Rotation about y-axis
- $Z$: Rotation about z-axis (phase flip)

**Rotation gates** parameterize arbitrary rotations:

$$R_x(\theta) = e^{-i\theta X/2} = \cos\frac{\theta}{2}I - i\sin\frac{\theta}{2}X$$

Similarly for $R_y(\theta)$ and $R_z(\theta)$.

### Two-Qubit Gates

**CNOT (Controlled-NOT)**: Flips target qubit if control is $|1\rangle$:

$$\text{CNOT}|00\rangle = |00\rangle, \quad \text{CNOT}|01\rangle = |01\rangle$$
$$\text{CNOT}|10\rangle = |11\rangle, \quad \text{CNOT}|11\rangle = |10\rangle$$

Creates entanglement from superposition:

$$\text{CNOT}(H \otimes I)|00\rangle = \text{CNOT}\frac{1}{\sqrt{2}}(|00\rangle + |10\rangle) = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

### Universality

A set of gates is **universal** if any unitary can be approximated to arbitrary precision.

**Universal gate sets**:
- $\{H, T, \text{CNOT}\}$
- $\{R_x, R_y, R_z, \text{CNOT}\}$
- $\{H, \text{Toffoli}\}$

Moonlab implements all standard gates with hardware-accelerated performance.

## Quantum Circuits

A quantum circuit is a sequence of gates applied to a register of qubits.

```
|0⟩ ──[H]──●──[M]
           │
|0⟩ ───────⊕──[M]
```

**Reading a circuit**:
- Time flows left to right
- Horizontal lines represent qubits
- Boxes represent gates
- Vertical lines connect multi-qubit gates
- [M] represents measurement

### Circuit Depth and Complexity

**Width**: Number of qubits
**Depth**: Maximum number of sequential gate layers
**Gate count**: Total number of gates

Minimizing depth is crucial for near-term quantum computers where decoherence limits circuit duration.

## Decoherence and Noise

Real quantum systems interact with their environment, causing decoherence.

### Types of Noise

**Bit flip**: $|0\rangle \leftrightarrow |1\rangle$ with probability $p$
**Phase flip**: $|+\rangle \leftrightarrow |-\rangle$ with probability $p$
**Depolarizing**: Replaces state with maximally mixed state with probability $p$
**Amplitude damping**: Models energy relaxation (T1 decay)
**Phase damping**: Models loss of phase coherence (T2 decay)

### Kraus Operators

Noise is described by completely positive trace-preserving (CPTP) maps:

$$\rho \mapsto \sum_k E_k \rho E_k^\dagger, \quad \sum_k E_k^\dagger E_k = I$$

Moonlab supports noise simulation for realistic algorithm testing.

## Summary

| Concept | Key Point |
|---------|-----------|
| State | Unit vector in Hilbert space |
| Superposition | Coherent combination of basis states |
| Entanglement | Non-factorizable correlations |
| Evolution | Unitary transformation |
| Measurement | Probabilistic, collapses state |
| Born rule | $P(m) = \|\langle m\|\psi\rangle\|^2$ |
| Gates | Unitary operations |
| Universality | Arbitrary unitaries from gate set |

## Next

Apply these concepts in practice: [First Simulation](first-simulation.md)

## Further Reading

- Nielsen & Chuang, *Quantum Computation and Quantum Information*, Chapters 1-4
- Preskill, *Lecture Notes for Physics 219*
- Rieffel & Polak, *Quantum Computing: A Gentle Introduction*
