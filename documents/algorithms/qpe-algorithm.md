# QPE Algorithm

Complete guide to Quantum Phase Estimation.

## Overview

Quantum Phase Estimation (QPE) determines the eigenvalue of a unitary operator given an eigenstate. It's a foundational subroutine in many quantum algorithms including Shor's factoring algorithm.

**Discovered**: Kitaev, 1995

**Applications**:
- Shor's algorithm (order finding)
- Quantum chemistry (energy eigenvalues)
- Quantum simulation
- Quantum counting

## Mathematical Foundation

### Problem Statement

Given:
- Unitary operator $U$
- Eigenstate $|u\rangle$ with $U|u\rangle = e^{2\pi i\phi}|u\rangle$

Find: The phase $\phi \in [0, 1)$

### Algorithm Structure

QPE uses $t$ ancilla qubits to estimate $\phi$ to $t$ bits of precision:

$$\phi \approx 0.\phi_1\phi_2\ldots\phi_t = \sum_{j=1}^t \phi_j 2^{-j}$$

### Circuit

```
|0⟩ ─────H───────────●─────────────────────[QFT†]─ φ₁
|0⟩ ─────H───────────│────●────────────────[QFT†]─ φ₂
 ⋮                   │    │         ⋮
|0⟩ ─────H───────────│────│────●───────────[QFT†]─ φₜ
                     │    │    │
|u⟩ ─────────────────U────U²───U^(2^(t-1))─────────
```

### State Evolution

After controlled-$U^{2^j}$ operations:

$$\frac{1}{\sqrt{2^t}}\sum_{k=0}^{2^t-1} e^{2\pi i k\phi}|k\rangle|u\rangle$$

The inverse QFT extracts $\phi$:

$$|\tilde{\phi}\rangle|u\rangle$$

where $\tilde{\phi}$ is the $t$-bit approximation of $\phi$.

## Implementation

### Basic QPE

```c
#include "qpe.h"

int main() {
    // Example: Estimate phase of T gate (φ = 1/8)
    int precision_bits = 4;

    // Configure QPE
    qpe_config_t config = {
        .precision_qubits = precision_bits,
        .eigenstate_qubits = 1
    };

    // Run QPE
    qpe_result_t result = qpe_run_t_gate(&config);

    printf("Estimated phase: %.6f\n", result.phase);
    printf("Exact phase: 0.125 (1/8)\n");
    printf("Binary: %s\n", result.phase_binary);

    return 0;
}
```

### Python Interface

```python
from moonlab.algorithms import QPE
from moonlab import QuantumState

# Create QPE solver
qpe = QPE(precision_qubits=8)

# Unitary to analyze
def controlled_u(state, control, target, power):
    """Apply controlled-U^power."""
    for _ in range(power):
        state.cp(control, target, np.pi/4)  # T gate

# Run QPE
result = qpe.estimate(
    controlled_unitary=controlled_u,
    eigenstate_qubits=1,
    eigenstate_prep=lambda s: s.x(0)  # |1⟩ is eigenstate of T
)

print(f"Estimated phase: {result['phase']:.6f}")
print(f"Binary representation: {result['binary']}")
print(f"Probability: {result['probability']:.4f}")
```

## Circuit Components

### Controlled Unitary Powers

Implement $U^{2^j}$ efficiently:

```python
def controlled_u_power(state, control, targets, power):
    """
    Apply controlled-U^(2^power).

    For many unitaries, U^(2^j) can be computed efficiently
    without repeated squaring.
    """
    if isinstance(power, int):
        # Repeated squaring or direct formula
        angle = base_angle * (2 ** power)
        state.crz(control, targets[0], angle)
    else:
        # General case: apply U 2^power times
        for _ in range(2 ** power):
            apply_controlled_u(state, control, targets)
```

### Inverse QFT

```python
def inverse_qft(state, qubits):
    """Apply inverse QFT to specified qubits."""
    n = len(qubits)

    for j in range(n - 1, -1, -1):
        # Controlled rotations
        for k in range(n - 1, j, -1):
            angle = -np.pi / (2 ** (k - j))
            state.cp(qubits[k], qubits[j], angle)

        # Hadamard
        state.h(qubits[j])

    # Reverse qubit order
    for i in range(n // 2):
        state.swap(qubits[i], qubits[n - 1 - i])
```

### Full QPE Circuit

```python
def qpe_circuit(precision_qubits, target_qubits, controlled_u, eigenstate_prep):
    """
    Full QPE circuit.

    Returns state ready for measurement of precision qubits.
    """
    n_precision = precision_qubits
    n_target = target_qubits
    total = n_precision + n_target

    state = QuantumState(total)

    # Prepare eigenstate on target qubits
    eigenstate_prep(state, range(n_precision, total))

    # Hadamard on precision qubits
    for j in range(n_precision):
        state.h(j)

    # Controlled-U^(2^j) operations
    for j in range(n_precision):
        power = n_precision - 1 - j
        controlled_u(state, j, range(n_precision, total), 2**power)

    # Inverse QFT on precision qubits
    inverse_qft(state, range(n_precision))

    return state
```

## Precision Analysis

### Success Probability

For exact $t$-bit phase, probability of success is 1.

For non-exact phase, probability of measuring closest value:

$$P(\text{closest}) \geq \frac{4}{\pi^2} \approx 0.405$$

With $t + \log(2 + 1/2\epsilon)$ qubits, success probability $\geq 1 - \epsilon$.

### Error Bounds

Phase estimation error:

$$|\phi - \tilde{\phi}| \leq 2^{-t}$$

with high probability.

### Resource Requirements

| Resource | Scaling |
|----------|---------|
| Precision qubits | $t$ |
| Controlled-$U$ calls | $2^t - 1$ |
| Total gates | $O(t^2)$ for QFT + $O(2^t \cdot \text{gates}(U))$ |

## Applications

### Eigenvalue Extraction

For Hamiltonian $H = \sum_j E_j |E_j\rangle\langle E_j|$:

$$U = e^{iHt} \implies \phi_j = \frac{E_j t}{2\pi}$$

```python
def estimate_energy(hamiltonian, eigenstate, precision=8):
    """Use QPE to extract energy eigenvalue."""
    def controlled_evolution(state, control, targets, power):
        t = power * dt
        apply_controlled_hamiltonian_evolution(state, hamiltonian, t, control, targets)

    qpe = QPE(precision_qubits=precision)
    result = qpe.estimate(
        controlled_unitary=controlled_evolution,
        eigenstate_prep=eigenstate
    )

    # Convert phase to energy
    energy = result['phase'] * 2 * np.pi / dt
    return energy
```

### Order Finding

For Shor's algorithm, find smallest $r$ such that $a^r \equiv 1 \mod N$:

$$U|x\rangle = |ax \mod N\rangle$$

Eigenvalues are $e^{2\pi i k/r}$ for $k = 0, 1, \ldots, r-1$.

### Quantum Counting

Count solutions $M$ in Grover's search:

$$\sin^2(\theta) = M/N$$

QPE on Grover operator gives $\theta$, from which $M$ is computed.

## Iterative QPE

For limited qubit count, use iterative phase estimation:

```python
def iterative_qpe(controlled_u, eigenstate_prep, precision, shots_per_bit=100):
    """
    Iterative QPE using single ancilla qubit.

    More robust to noise than standard QPE.
    """
    phase_bits = []

    for k in range(precision - 1, -1, -1):
        # Create state
        state = QuantumState(1 + eigenstate_qubits)

        # Prepare eigenstate
        eigenstate_prep(state, range(1, 1 + eigenstate_qubits))

        # Hadamard on ancilla
        state.h(0)

        # Apply controlled-U^(2^k)
        controlled_u(state, 0, range(1, 1 + eigenstate_qubits), 2**k)

        # Phase correction from previous bits
        correction = sum(phase_bits[j] * 2**(j - precision + k + 1)
                        for j in range(len(phase_bits)))
        state.rz(0, -np.pi * correction)

        # Hadamard and measure
        state.h(0)

        # Statistical sampling
        zeros = sum(state.measure(0) == 0 for _ in range(shots_per_bit))
        phase_bits.insert(0, 0 if zeros > shots_per_bit / 2 else 1)

    # Construct phase
    phase = sum(phase_bits[j] * 2**(-j-1) for j in range(precision))
    return phase
```

## Hadamard Test

Simplified version for estimating $\text{Re}(\langle\psi|U|\psi\rangle)$:

```python
def hadamard_test(state_prep, unitary, shots=1000):
    """
    Estimate Re(⟨ψ|U|ψ⟩).

    Circuit:
    |0⟩ ─H──●──H─ Measure
            │
    |ψ⟩ ────U────
    """
    results = []

    for _ in range(shots):
        state = QuantumState(1 + num_state_qubits)

        # Prepare |ψ⟩
        state_prep(state, range(1, 1 + num_state_qubits))

        # Hadamard test circuit
        state.h(0)
        controlled_unitary(state, 0, range(1, 1 + num_state_qubits))
        state.h(0)

        # Measure ancilla
        results.append(state.measure(0))

    # Re(⟨ψ|U|ψ⟩) = P(0) - P(1)
    p0 = results.count(0) / shots
    return 2 * p0 - 1
```

For imaginary part, add S† before final Hadamard.

## Error Mitigation

### Noise Resilience

Iterative QPE is more noise-resilient than standard QPE:

```python
# Configure noise-resilient QPE
qpe = QPE(
    precision_qubits=8,
    method='iterative',  # More robust
    error_mitigation='bayesian',
    shots_per_bit=500
)
```

### Post-Selection

```python
def qpe_with_verification(controlled_u, eigenstate_prep, precision):
    """QPE with eigenstate verification."""
    result = qpe.estimate(controlled_u, eigenstate_prep)

    # Verify eigenstate
    phase = result['phase']
    verification = verify_eigenstate(eigenstate_prep, controlled_u, phase)

    if verification['fidelity'] > 0.99:
        return result
    else:
        # Retry or report failure
        return qpe_with_verification(controlled_u, eigenstate_prep, precision)
```

## Complexity Analysis

| Variant | Qubits | Gates | Queries to $U$ |
|---------|--------|-------|----------------|
| Standard QPE | $t + n$ | $O(t^2 + 2^t \cdot g)$ | $O(2^t)$ |
| Iterative QPE | $1 + n$ | $O(t \cdot 2^t \cdot g)$ | $O(t \cdot 2^t)$ |
| Bayesian QPE | $1 + n$ | $O(t \cdot g)$ | $O(t)$ |

Where:
- $t$ = precision bits
- $n$ = eigenstate qubits
- $g$ = gates per controlled-$U$

## Example: T Gate Phase

```python
from moonlab.algorithms import QPE
import numpy as np

# T gate has eigenvalue exp(iπ/4) = exp(2πi/8)
# Phase should be 1/8 = 0.125

def controlled_t_power(state, control, target, power):
    """Apply controlled T^(2^power)."""
    angle = np.pi / 4 * (2 ** power)
    state.cp(control, target, angle)

qpe = QPE(precision_qubits=6)
result = qpe.estimate(
    controlled_unitary=controlled_t_power,
    eigenstate_qubits=1,
    eigenstate_prep=lambda s, q: s.x(q[0])  # |1⟩ is eigenstate
)

print(f"Estimated phase: {result['phase']:.6f}")
print(f"Expected: 0.125000")
print(f"Binary: {result['binary']}")  # Should be 001000 (1/8)
```

## See Also

- [C API: QPE](../api/c/qpe.md) - Complete C API reference
- [Quantum Counting](grovers-algorithm.md#quantum-counting) - Application of QPE
- [VQE Algorithm](vqe-algorithm.md) - Alternative for eigenvalue problems

## References

1. Kitaev, A. (1995). "Quantum measurements and the Abelian stabilizer problem." arXiv:quant-ph/9511026.
2. Nielsen, M. A., & Chuang, I. L. (2010). "Quantum Computation and Quantum Information." Cambridge University Press. Chapter 5.2.
3. Svore, K. et al. (2013). "Faster Phase Estimation." Quantum Information & Computation, 14, 306-328.

