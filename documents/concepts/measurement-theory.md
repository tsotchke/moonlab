# Measurement Theory

Quantum measurement and the Born rule.

## Introduction

Measurement is the interface between quantum and classical worlds. When we measure a quantum system, we extract classical information, but the measurement process irreversibly disturbs the quantum state.

## The Born Rule

### Probability of Outcomes

For a quantum state $|\psi\rangle = \sum_i \alpha_i |i\rangle$, the probability of measuring outcome $i$ is:

$$P(i) = |\langle i | \psi \rangle|^2 = |\alpha_i|^2$$

This is the **Born rule**, the fundamental connection between quantum amplitudes and observed probabilities.

### Normalization

The Born rule requires normalization:

$$\sum_i P(i) = \sum_i |\alpha_i|^2 = 1$$

This is why quantum states must have unit norm.

## Projective Measurement

### Measurement Postulate

Measuring an observable $\hat{O}$ with eigendecomposition:

$$\hat{O} = \sum_i \lambda_i |i\rangle\langle i|$$

- Yields eigenvalue $\lambda_i$ with probability $P(i) = |\langle i|\psi\rangle|^2$
- Collapses state to eigenstate $|i\rangle$

### Computational Basis Measurement

The most common measurement is in the computational basis $\{|0\rangle, |1\rangle\}^{\otimes n}$.

For state $|\psi\rangle = \sum_{i=0}^{2^n-1} \alpha_i |i\rangle$:
- Probability of outcome $|i\rangle$: $P(i) = |\alpha_i|^2$
- Post-measurement state: $|i\rangle$

```c
// Measure all qubits in computational basis
int result = quantum_state_measure_all(state);
// state is now |result⟩
```

### Single-Qubit Measurement

Measuring qubit $q$ in state $|\psi\rangle$:

$$P(q=0) = \sum_{i: \text{bit } q = 0} |\alpha_i|^2$$
$$P(q=1) = \sum_{i: \text{bit } q = 1} |\alpha_i|^2$$

**Post-measurement state**:
If outcome is 0: zero out amplitudes with bit $q = 1$, renormalize
If outcome is 1: zero out amplitudes with bit $q = 0$, renormalize

```c
// Measure single qubit
int outcome = quantum_state_measure(state, qubit);
// state is projected and renormalized
```

## Measurement Operators

### General Measurement

A general quantum measurement is described by measurement operators $\{M_m\}$ satisfying:

$$\sum_m M_m^\dagger M_m = I$$

- Probability of outcome $m$: $P(m) = \langle\psi|M_m^\dagger M_m|\psi\rangle$
- Post-measurement state: $\frac{M_m|\psi\rangle}{\sqrt{P(m)}}$

### POVM Measurement

Positive Operator-Valued Measure (POVM) elements:

$$E_m = M_m^\dagger M_m$$

with $\sum_m E_m = I$.

POVMs describe measurement statistics without specifying post-measurement state.

## Non-Demolition Measurement

### Z-Basis Measurement

Measuring in the computational (Z) basis:
- Eigenstates: $|0\rangle, |1\rangle$
- Projects onto $|0\rangle$ or $|1\rangle$

### X-Basis Measurement

Measuring in the X basis:
- Eigenstates: $|+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}$, $|-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$
- Implemented as: $H \to Z\text{-measure} \to H$

```c
// X-basis measurement on qubit q
quantum_state_h(state, q);
int outcome = quantum_state_measure(state, q);
// outcome: 0 → |+⟩, 1 → |−⟩
```

### Y-Basis Measurement

- Eigenstates: $|R\rangle = \frac{|0\rangle + i|1\rangle}{\sqrt{2}}$, $|L\rangle = \frac{|0\rangle - i|1\rangle}{\sqrt{2}}$
- Implemented as: $S^\dagger H \to Z\text{-measure}$

## Partial Measurement

### Measuring Subsystems

For a bipartite state $|\psi\rangle_{AB}$, measuring only subsystem A:

$$P(a) = \text{Tr}_B[|a\rangle\langle a| \otimes I_B \cdot |\psi\rangle\langle\psi|]$$

Post-measurement state of B (unnormalized):

$$\rho_B^{(a)} = \langle a|_A \, |\psi\rangle\langle\psi| \, |a\rangle_A$$

### Entanglement and Correlation

Measuring one half of an entangled pair instantly determines the other:

For Bell state $|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$:
- Measure qubit 0 → outcome $m$
- Qubit 1 collapses to $|m\rangle$

This correlation is instantaneous but cannot transmit information.

## Expectation Values

### Observable Expectation

The expectation value of observable $\hat{O}$ in state $|\psi\rangle$:

$$\langle \hat{O} \rangle = \langle\psi|\hat{O}|\psi\rangle$$

### Pauli Expectations

For Pauli operators on qubit $q$:

$$\langle Z_q \rangle = P(q=0) - P(q=1)$$
$$\langle X_q \rangle = \langle\psi|H_q Z_q H_q|\psi\rangle$$
$$\langle Y_q \rangle = \langle\psi|S_q^\dagger H_q Z_q H_q S_q|\psi\rangle$$

```c
double z_exp = quantum_state_expectation_z(state, qubit);
double x_exp = quantum_state_expectation_x(state, qubit);
double y_exp = quantum_state_expectation_y(state, qubit);
```

### Multi-Qubit Correlations

For tensor product observables:

$$\langle Z_i Z_j \rangle = \langle\psi|Z_i \otimes Z_j|\psi\rangle$$

```c
double zz_corr = quantum_state_correlation_zz(state, qubit_i, qubit_j);
```

## Measurement Statistics

### Shot-Based Simulation

Real quantum computers don't give probabilities directly. They sample:

```c
// Run 1000 measurements
int counts[1024] = {0};  // For 10 qubits
for (int shot = 0; shot < 1000; shot++) {
    quantum_state_reset(state);
    apply_circuit(state);
    int result = quantum_state_measure_all(state);
    counts[result]++;
}
```

### Statistical Uncertainty

For $N$ shots, the uncertainty in probability estimate:

$$\Delta P = \sqrt{\frac{P(1-P)}{N}}$$

To estimate $P$ within $\epsilon$ with 95% confidence: $N \approx \frac{1}{\epsilon^2}$

### Ideal vs. Noisy Measurement

**Ideal measurement**: Perfect projection, no errors

**Noisy measurement**: Assignment errors
- $P(\text{measure } 0 | \text{state } 1) = \epsilon_1$
- $P(\text{measure } 1 | \text{state } 0) = \epsilon_0$

Mitigation: Readout error correction matrices

## Quantum Non-Demolition

### QND Measurements

A **Quantum Non-Demolition** measurement of observable $\hat{O}$ satisfies:

$$[\hat{O}(t), \hat{O}(t')] = 0$$

The observable commutes with itself at all times, allowing repeated measurement.

### Ancilla-Based Measurement

Measure system indirectly via ancilla:

1. Prepare ancilla in $|0\rangle$
2. Entangle with system: controlled operation
3. Measure ancilla only

```
|ψ⟩ ──●──── (unmeasured)
      │
|0⟩ ──X── [M]
```

## Weak Measurement

### Definition

Weak measurements extract partial information with minimal disturbance:
- Weak coupling to measurement apparatus
- State only slightly disturbed
- Many repetitions needed for signal

### Weak Values

The **weak value** of observable $\hat{A}$ between pre-selected $|i\rangle$ and post-selected $|f\rangle$:

$$A_w = \frac{\langle f|\hat{A}|i\rangle}{\langle f|i\rangle}$$

Weak values can be complex and lie outside the eigenvalue spectrum.

## Implementation

### Measurement in Moonlab

```c
// Non-destructive probability query
double prob_zero = quantum_state_probability_zero(state, qubit);
double prob_one = quantum_state_probability_one(state, qubit);

// Destructive measurement (collapses state)
int outcome = quantum_state_measure(state, qubit);

// Sample multiple shots
int* results = quantum_state_sample(state, num_shots);

// Expectation values (non-destructive)
double exp_z = quantum_state_expectation_z(state, qubit);
```

### Random Number Generation

Measurement outcomes require random sampling. Moonlab uses:

1. **Hardware entropy**: `/dev/urandom`, CPU RDRAND
2. **CSPRNG**: ChaCha20 for simulation reproducibility

```c
// Set seed for reproducibility
quantum_set_seed(12345);

// Or use hardware entropy
quantum_use_hardware_entropy(true);
```

## The Measurement Problem

### Interpretations

The measurement postulate is axiomatic in quantum mechanics. Interpretations differ on the nature of collapse:

| Interpretation | Collapse |
|----------------|----------|
| Copenhagen | Physical, instantaneous |
| Many-Worlds | Apparent (branching) |
| Pilot Wave | Deterministic (hidden variables) |
| QBism | Epistemic (belief update) |

### Decoherence

Environmental decoherence provides a physical mechanism for the appearance of collapse:

$$\rho_{\text{system}} = \text{Tr}_{\text{env}}[\rho_{\text{total}}]$$

Off-diagonal elements decay exponentially, leaving classical probability distribution.

## See Also

- [Quantum Computing Basics](quantum-computing-basics.md) - Foundational concepts
- [Entanglement Measures](entanglement-measures.md) - Entropy and correlations
- [Noise Models](noise-models.md) - Decoherence and errors
- [C API: Measurement](../api/c/measurement.md) - Function reference

