# Archived Moonlab Documentation: Bell-CHSH Test

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# Bell-CHSH Test

Complete guide to Bell inequality testing and entanglement verification.

## Overview

The Bell-CHSH test provides experimental verification of quantum entanglement by demonstrating violations of classical correlation bounds. It's the definitive test that quantum mechanics cannot be explained by local hidden variable theories.

**Discovered**:
- Bell inequality: John Bell, 1964
- CHSH form: Clauser, Horne, Shimony, Holt, 1969

**Applications**:
- Entanglement verification
- Device-independent QKD
- Quantum randomness certification
- Foundations of quantum mechanics

## Mathematical Foundation

### Bell's Theorem

No local hidden variable theory can reproduce all predictions of quantum mechanics.

### CHSH Inequality

For measurements $A, A'$ on Alice's qubit and $B, B'$ on Bob's qubit:

$$S = |\langle AB \rangle - \langle AB' \rangle + \langle A'B \rangle + \langle A'B' \rangle|$$

**Classical bound**: $S \leq 2$

**Quantum bound** (Tsirelson): $S \leq 2\sqrt{2} \approx 2.828$

### Optimal Settings

For maximally entangled state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$:

| Measurement | Angle | Operator |
|-------------|-------|----------|
| $A$ | $0°$ | $Z$ |
| $A'$ | $45°$ | $(Z + X)/\sqrt{2}$ |
| $B$ | $22.5°$ | $\cos(22.5°)Z + \sin(22.5°)X$ |
| $B'$ | $-22.5°$ | $\cos(22.5°)Z - \sin(22.5°)X$ |

These settings achieve $S = 2\sqrt{2}$.

## Implementation

### Basic CHSH Test

[archived fence delimiter: ```c]
#include "quantum_sim.h"
#include "bell.h"

int main() {
    // Number of measurement rounds
    int shots = 10000;

    // Create Bell state |Φ+⟩
    quantum_state_t* state = quantum_state_init(2);
    gate_hadamard(state, 0);
    gate_cnot(state, 0, 1);

    // Measurement angles (optimal for CHSH)
    double a1 = 0;              // A
    double a2 = M_PI / 4;       // A'
    double b1 = M_PI / 8;       // B
    double b2 = -M_PI / 8;      // B'

    // Compute correlators
    double E_ab = bell_correlator(state, a1, b1, shots);
    double E_ab_prime = bell_correlator(state, a1, b2, shots);
    double E_a_prime_b = bell_correlator(state, a2, b1, shots);
    double E_a_prime_b_prime = bell_correlator(state, a2, b2, shots);

    // CHSH value
    double S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime;

    printf("CHSH value S = %.4f\n", S);
    printf("Classical bound: 2.0\n");
    printf("Quantum bound: %.4f\n", 2 * sqrt(2));
    printf("Violation: %s\n", fabs(S) > 2.0 ? "YES" : "NO");

    quantum_state_free(state);
    return 0;
}
[archived fence delimiter: ```]

### Python Interface

[archived fence delimiter: ```python]
from moonlab.algorithms import BellTest
from moonlab import QuantumState
import numpy as np

# Create Bell state
state = QuantumState(2)
state.h(0).cnot(0, 1)

# Run CHSH test
bell = BellTest(shots=10000)
result = bell.chsh_test(state)

print(f"CHSH value: {result['S']:.4f}")
print(f"Standard error: {result['std_error']:.4f}")
print(f"Classical bound: 2.0")
print(f"Quantum bound: {2*np.sqrt(2):.4f}")
print(f"Violation: {result['violates_classical']}")
print(f"Significance: {result['sigma']:.1f}σ")
[archived fence delimiter: ```]

## Correlator Measurement

### Single Correlator

$$\langle AB \rangle = P(a=b) - P(a \neq b)$$

where $a, b \in \{+1, -1\}$ are measurement outcomes.

[archived fence delimiter: ```python]
def measure_correlator(state, angle_a, angle_b, shots):
    """
    Measure correlation ⟨A(θ_a)B(θ_b)⟩.

    Measurement in rotated basis:
    - Rotate qubit by angle
    - Measure in Z basis
    """
    correlations = []

    for _ in range(shots):
        # Fresh state copy
        s = state.copy()

        # Rotate to measurement basis
        s.ry(0, -2 * angle_a)
        s.ry(1, -2 * angle_b)

        # Measure both qubits
        outcome_a = s.measure(0)  # 0 → +1, 1 → -1
        outcome_b = s.measure(1)

        # Convert to ±1
        a = 1 - 2 * outcome_a
        b = 1 - 2 * outcome_b

        correlations.append(a * b)

    return np.mean(correlations), np.std(correlations) / np.sqrt(shots)
[archived fence delimiter: ```]

### Exact Correlator

For simulation, compute exactly:

[archived fence delimiter: ```python]
def exact_correlator(state, angle_a, angle_b):
    """Compute exact correlation without sampling."""
    # Measurement operators
    A = np.cos(angle_a) * Z + np.sin(angle_a) * X
    B = np.cos(angle_b) * Z + np.sin(angle_b) * X

    # Tensor product A ⊗ B
    AB = np.kron(A, B)

    return state.expectation(AB)
[archived fence delimiter: ```]

## Bell States

### Four Bell States

| State | Definition | CHSH Value |
|-------|------------|------------|
| $\|\Phi^+\rangle$ | $\frac{1}{\sqrt{2}}(\|00\rangle + \|11\rangle)$ | $+2\sqrt{2}$ |
| $\|\Phi^-\rangle$ | $\frac{1}{\sqrt{2}}(\|00\rangle - \|11\rangle)$ | $+2\sqrt{2}$ |
| $\|\Psi^+\rangle$ | $\frac{1}{\sqrt{2}}(\|01\rangle + \|10\rangle)$ | $-2\sqrt{2}$ |
| $\|\Psi^-\rangle$ | $\frac{1}{\sqrt{2}}(\|01\rangle - \|10\rangle)$ | $-2\sqrt{2}$ |

### Preparation Circuits

[archived fence delimiter: ```python]
def prepare_bell_state(state, variant):
    """
    Prepare Bell state.

    variant: 'phi+', 'phi-', 'psi+', 'psi-'
    """
    state.h(0)
    state.cnot(0, 1)

    if variant in ['phi-', 'psi-']:
        state.z(0)

    if variant in ['psi+', 'psi-']:
        state.x(1)

    return state
[archived fence delimiter: ```]

## Statistical Analysis

### Significance Testing

[archived fence delimiter: ```python]
def chsh_significance(S, std_error, classical_bound=2.0):
    """
    Compute statistical significance of CHSH violation.

    Returns number of standard deviations above classical bound.
    """
    if abs(S) <= classical_bound:
        return 0.0

    return (abs(S) - classical_bound) / std_error

# Example
S = 2.7
std_error = 0.02
sigma = chsh_significance(S, std_error)
print(f"Violation at {sigma:.1f}σ significance")
[archived fence delimiter: ```]

### Error Propagation

CHSH value uncertainty:

$$\sigma_S = \sqrt{\sigma_{E_{AB}}^2 + \sigma_{E_{AB'}}^2 + \sigma_{E_{A'B}}^2 + \sigma_{E_{A'B'}}^2}$$

[archived fence delimiter: ```python]
def chsh_with_error(state, shots):
    """Compute CHSH with proper error analysis."""
    angles = [
        (0, np.pi/8),           # A, B
        (0, -np.pi/8),          # A, B'
        (np.pi/4, np.pi/8),     # A', B
        (np.pi/4, -np.pi/8)     # A', B'
    ]

    correlators = []
    errors = []

    for a, b in angles:
        E, sigma = measure_correlator(state, a, b, shots)
        correlators.append(E)
        errors.append(sigma)

    S = correlators[0] - correlators[1] + correlators[2] + correlators[3]
    sigma_S = np.sqrt(sum(e**2 for e in errors))

    return S, sigma_S
[archived fence delimiter: ```]

## Advanced Topics

### Loophole-Free Tests

#### Detection Loophole

All particles must be detected:

[archived fence delimiter: ```python]
def detection_efficiency_analysis(eta):
    """
    Analyze required detection efficiency.

    For loophole-free violation: η > 2/(1 + √2) ≈ 82.8%
    """
    threshold = 2 / (1 + np.sqrt(2))
    return eta > threshold, threshold
[archived fence delimiter: ```]

#### Locality Loophole

Measurements must be space-like separated:

[archived fence delimiter: ```python]
def locality_analysis(distance, measurement_time):
    """Check if measurements are space-like separated."""
    c = 3e8  # Speed of light
    required_separation = c * measurement_time
    return distance > required_separation
[archived fence delimiter: ```]

### Non-Maximally Entangled States

For state $|\psi\rangle = \cos\theta|00\rangle + \sin\theta|11\rangle$:

Maximum CHSH value:
$$S_{max} = 2\sqrt{1 + \sin^2(2\theta)}$$

[archived fence delimiter: ```python]
def partial_entanglement_chsh(theta):
    """
    CHSH test for partially entangled state.
    """
    state = QuantumState(2)
    state.ry(0, 2 * theta)
    state.cnot(0, 1)

    # Optimal angles depend on entanglement
    optimal_angles = compute_optimal_angles(theta)

    return bell.chsh_test(state, angles=optimal_angles)
[archived fence delimiter: ```]

### Multi-Party Bell Tests

#### GHZ State

Three-party correlation:

[archived fence delimiter: ```python]
def ghz_test(shots=10000):
    """Mermin inequality test for GHZ state."""
    state = QuantumState(3)
    state.h(0)
    state.cnot(0, 1)
    state.cnot(0, 2)

    # Mermin operator expectation
    # M = XXX - XYY - YXY - YYX
    M = measure_ghz_correlators(state, shots)

    print(f"Mermin value: {M:.4f}")
    print(f"Classical bound: 2")
    print(f"Quantum value for GHZ: 4")
[archived fence delimiter: ```]

#### W State

[archived fence delimiter: ```python]
def w_state_test():
    """Bell test for W state."""
    state = QuantumState(3)
    prepare_w_state(state)

    # W state has different entanglement properties
    result = bell.multiparty_test(state, parties=3)
    return result
[archived fence delimiter: ```]

## Visualization

### Correlation Functions

[archived fence delimiter: ```python]
import matplotlib.pyplot as plt
import numpy as np

def plot_correlation_function(state, angle_a, shots=1000):
    """Plot correlation vs Bob's angle."""
    angles_b = np.linspace(0, 2*np.pi, 100)
    correlations = []

    for angle_b in angles_b:
        E, _ = measure_correlator(state, angle_a, angle_b, shots)
        correlations.append(E)

    plt.figure(figsize=(10, 6))
    plt.plot(angles_b, correlations, label=f'θ_A = {angle_a:.2f}')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.xlabel("Bob's angle θ_B")
    plt.ylabel("Correlation ⟨AB⟩")
    plt.title("Bell Correlation Function")
    plt.legend()
    plt.grid(True)
[archived fence delimiter: ```]

### CHSH Landscape

[archived fence delimiter: ```python]
def plot_chsh_landscape(state, resolution=50):
    """Plot CHSH value vs measurement angles."""
    a_prime = np.linspace(0, np.pi/2, resolution)
    b_prime = np.linspace(-np.pi/4, np.pi/4, resolution)

    S = np.zeros((resolution, resolution))

    for i, ap in enumerate(a_prime):
        for j, bp in enumerate(b_prime):
            # Fixed: A at 0, B at optimal
            S[i, j] = compute_chsh(state, 0, ap, np.pi/8, bp)

    plt.figure(figsize=(10, 8))
    plt.contourf(b_prime, a_prime, S, levels=50, cmap='RdBu_r')
    plt.colorbar(label='CHSH Value S')
    plt.contour(b_prime, a_prime, S, levels=[2.0], colors='black', linewidths=2)
    plt.xlabel("B' angle")
    plt.ylabel("A' angle")
    plt.title("CHSH Landscape (black line = classical bound)")
[archived fence delimiter: ```]

## Practical Considerations

### Finite Statistics

[archived fence delimiter: ```python]
def required_shots(target_sigma, expected_S, expected_std=0.1):
    """
    Calculate shots needed for target significance.

    target_sigma: desired significance (e.g., 5 for 5σ)
    expected_S: expected CHSH value
    """
    violation = abs(expected_S) - 2.0

    if violation <= 0:
        return float('inf')

    # Standard error scales as 1/√shots
    required_std = violation / target_sigma
    shots = (expected_std / required_std) ** 2

    return int(np.ceil(shots))

# Example: 5σ violation
shots = required_shots(5.0, expected_S=2.7)
print(f"Required shots for 5σ: {shots}")
[archived fence delimiter: ```]

### Noise Effects

[archived fence delimiter: ```python]
def noisy_chsh_test(state, noise_model, shots=10000):
    """CHSH test with realistic noise."""

    # Apply noise during preparation
    noisy_state = noise_model.apply(state)

    # Test with noisy state
    result = bell.chsh_test(noisy_state, shots=shots)

    # Check if violation survives
    if not result['violates_classical']:
        print("Warning: Noise destroyed Bell violation!")

    return result
[archived fence delimiter: ```]

## Complete Example

[archived fence delimiter: ```python]
from moonlab.algorithms import BellTest
from moonlab import QuantumState
import numpy as np

def full_bell_experiment():
    """Complete Bell-CHSH experiment."""

    # Prepare maximally entangled state
    state = QuantumState(2)
    state.h(0)
    state.cnot(0, 1)

    # Verify entanglement
    entropy = state.entanglement_entropy(0)
    print(f"Entanglement entropy: {entropy:.4f} (max = 1.0)")

    # Run CHSH test
    bell = BellTest(shots=100000)
    result = bell.chsh_test(state)

    print("\n=== CHSH Test Results ===")
    print(f"S value: {result['S']:.4f} ± {result['std_error']:.4f}")
    print(f"Classical bound: 2.0")
    print(f"Quantum bound: {2*np.sqrt(2):.4f}")
    print(f"Violation: {result['violates_classical']}")
    print(f"Significance: {result['sigma']:.1f}σ")

    # Individual correlators
    print("\n=== Correlators ===")
    for name, value in result['correlators'].items():
        print(f"⟨{name}⟩ = {value:.4f}")

    return result

if __name__ == "__main__":
    full_bell_experiment()
[archived fence delimiter: ```]

## See Also

- [Tutorial: Bell Test](../tutorials/05-bell-test-verification.md) - Step-by-step tutorial
- [Entanglement Measures](../concepts/entanglement-measures.md) - Theory background
- [Creating Bell States](../tutorials/03-creating-bell-states.md) - State preparation

## References

1. Bell, J. S. (1964). "On the Einstein Podolsky Rosen Paradox." Physics, 1, 195-200.
2. Clauser, J. F. et al. (1969). "Proposed experiment to test local hidden-variable theories." Physical Review Letters, 23, 880.
3. Aspect, A. et al. (1982). "Experimental Realization of Einstein-Podolsky-Rosen-Bohm Gedankenexperiment." Physical Review Letters, 49, 91.
4. Hensen, B. et al. (2015). "Loophole-free Bell inequality violation using electron spins separated by 1.3 kilometres." Nature, 526, 682-686.

```
