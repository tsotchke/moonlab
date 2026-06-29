# Archived Moonlab Documentation: QPE API

This local Moonlab document is retained as archived vendor text for the QGTL integration audit; current supported claims are measured by `scripts/moonlab_doc_claim_audit.py` and grounded against `external/moonlab/README.md`, `external/moonlab/CMakeLists.txt`, and `docs/MOONLAB_OPEN_CORE_INTEGRATION.md`.

The historical text below is preserved as an archival snapshot, not as current release documentation.

```text
# QPE API

Complete reference for Quantum Phase Estimation (QPE) in the C library.

**Header**: `src/algorithms/qpe.h`

## Overview

Quantum Phase Estimation (QPE) is a fundamental quantum algorithm that estimates eigenvalues of unitary operators. It forms the foundation for:

- **Shor's factoring algorithm**: Integer factorization
- **HHL algorithm**: Quantum linear system solver
- **Quantum chemistry**: Computing excited state energies
- **Period finding**: Cryptographic applications

## Algorithm

Given a unitary $U$ and its eigenstate $|\psi\rangle$ where $U|\psi\rangle = e^{2\pi i \phi}|\psi\rangle$, QPE estimates the phase $\phi$ with precision determined by the number of precision qubits.

**Algorithm Steps** (Kitaev 1995, Cleve et al. 1998):

1. **Prepare**: $|0\rangle^{\otimes m}|\psi\rangle$ where $U|\psi\rangle = e^{2\pi i\phi}|\psi\rangle$
2. **Apply** $H^{\otimes m}$ to first $m$ qubits
3. **Apply** controlled-$U^{2^k}$ operations
4. **Inverse QFT** on first $m$ qubits
5. **Measure** to get $\phi$ estimate ($m$-bit precision)

**Precision**: $m$ qubits give $\phi$ within $2^{-m}$ accuracy

**32-qubit simulator**: 16 precision qubits + 16 system qubits

## Unitary Operator

### unitary_operator_t

Unitary operator for QPE.

[archived fence delimiter: ```c]
typedef struct {
    size_t num_qubits;           // Qubits operator acts on
    void *operator_data;         // Opaque operator specification

    // Apply U to state
    qs_error_t (*apply)(quantum_state_t *state, void *data);

    // Apply U^k to state (for controlled-U^(2^k))
    qs_error_t (*apply_power)(quantum_state_t *state, void *data, uint64_t power);

    // Optional: eigenvalue for validation
    complex_t eigenvalue;
} unitary_operator_t;
[archived fence delimiter: ```]

**Fields**:
- `num_qubits`: Number of qubits the operator acts on
- `operator_data`: Opaque pointer to operator-specific data
- `apply`: Function pointer to apply $U$ once
- `apply_power`: Function pointer to apply $U^k$
- `eigenvalue`: Known eigenvalue (for validation)

### unitary_operator_create

Create unitary operator.

[archived fence delimiter: ```c]
unitary_operator_t* unitary_operator_create(size_t num_qubits);
[archived fence delimiter: ```]

**Parameters**:
- `num_qubits`: Number of qubits

**Returns**: Initialized operator (caller must set function pointers)

### unitary_operator_free

Free unitary operator.

[archived fence delimiter: ```c]
void unitary_operator_free(unitary_operator_t *op);
[archived fence delimiter: ```]

## Eigenstate

### eigenstate_t

Eigenstate of unitary operator.

[archived fence delimiter: ```c]
typedef struct {
    quantum_state_t *state;      // Eigenstate |ψ⟩
    complex_t eigenvalue;        // e^(2πiφ)
    double phase;                // φ ∈ [0,1)
} eigenstate_t;
[archived fence delimiter: ```]

**Mathematical Relationship**:
$$U|\psi\rangle = e^{2\pi i \phi}|\psi\rangle$$

where:
- $|\psi\rangle$ is the eigenstate
- $e^{2\pi i \phi}$ is the eigenvalue
- $\phi \in [0, 1)$ is the phase

### eigenstate_create

Create eigenstate.

[archived fence delimiter: ```c]
eigenstate_t* eigenstate_create(size_t num_qubits);
[archived fence delimiter: ```]

**Parameters**:
- `num_qubits`: Number of qubits

**Returns**: Initialized eigenstate structure

### eigenstate_free

Free eigenstate.

[archived fence delimiter: ```c]
void eigenstate_free(eigenstate_t *es);
[archived fence delimiter: ```]

## QPE Algorithm

### qpe_config_t

QPE configuration.

[archived fence delimiter: ```c]
typedef struct {
    size_t precision_qubits;     // m qubits for phase estimation
    size_t system_qubits;        // n qubits for system
    double phase_accuracy;       // Target accuracy (2^(-m))
} qpe_config_t;
[archived fence delimiter: ```]

### qpe_result_t

QPE result.

[archived fence delimiter: ```c]
typedef struct {
    double estimated_phase;      // φ estimate in [0, 1)
    complex_t estimated_eigenvalue;  // e^(2πiφ)
    uint64_t phase_bitstring;   // m-bit measurement outcome
    double confidence;           // Probability of correct estimation
    size_t precision_bits;       // Number of precision bits used

    // Reference values (if known)
    double true_phase;           // Known phase (for validation)
    double phase_error;          // |estimated - true|
} qpe_result_t;
[archived fence delimiter: ```]

**Fields**:
- `estimated_phase`: Estimated phase $\phi \in [0, 1)$
- `estimated_eigenvalue`: Computed eigenvalue $e^{2\pi i\phi}$
- `phase_bitstring`: Raw measurement outcome
- `confidence`: Probability estimate was correct
- `precision_bits`: Precision used
- `true_phase`: Known true phase (for validation)
- `phase_error`: Absolute error $|\hat{\phi} - \phi|$

### qpe_estimate_phase

Execute QPE algorithm.

[archived fence delimiter: ```c]
qpe_result_t qpe_estimate_phase(
    const unitary_operator_t *unitary,
    const eigenstate_t *eigenstate,
    size_t precision_qubits,
    quantum_entropy_ctx_t *entropy
);
[archived fence delimiter: ```]

**Parameters**:
- `unitary`: Unitary operator $U$
- `eigenstate`: Eigenstate $|\psi\rangle$ of $U$
- `precision_qubits`: Number of precision bits $m$
- `entropy`: Entropy source for measurement

**Returns**: QPE result with phase estimate

**Precision Guarantee**:
With $m$ precision qubits, the algorithm returns $\phi$ with:
- Absolute error $\leq 2^{-m}$
- Success probability $\geq 1 - \epsilon$ (can be amplified)

**Example**:
[archived fence delimiter: ```c]
// Estimate phase of T gate (should give φ = 1/8)
unitary_operator_t *T = qpe_create_t_gate();
eigenstate_t *es = eigenstate_create(1);

// Prepare |1⟩ as eigenstate of T
gate_x(es->state, 0);
es->phase = 0.125;  // True phase: 1/8

// Run QPE with 4 precision qubits
quantum_entropy_ctx_t entropy;
quantum_entropy_init(&entropy, NULL, NULL);

qpe_result_t result = qpe_estimate_phase(T, es, 4, &entropy);

printf("Estimated phase: %.4f\n", result.estimated_phase);
printf("True phase: %.4f\n", result.true_phase);
printf("Error: %.4f\n", result.phase_error);
[archived fence delimiter: ```]

### qpe_apply_controlled_unitary_power

Apply controlled-$U^k$ operation.

[archived fence delimiter: ```c]
qs_error_t qpe_apply_controlled_unitary_power(
    quantum_state_t *state,
    int control,
    int target_start,
    const unitary_operator_t *unitary,
    uint64_t power
);
[archived fence delimiter: ```]

**Parameters**:
- `state`: Full quantum state
- `control`: Control qubit index
- `target_start`: First target qubit
- `unitary`: Unitary operator
- `power`: $k$ (apply $U^k$)

**Returns**: `QS_SUCCESS` or error

**Action**: Applies $U^k$ to target qubits if control qubit is $|1\rangle$

### qpe_bitstring_to_phase

Convert measured bitstring to phase estimate.

[archived fence delimiter: ```c]
double qpe_bitstring_to_phase(uint64_t bitstring, size_t precision_bits);
[archived fence delimiter: ```]

**Parameters**:
- `bitstring`: Measured $m$-bit string
- `precision_bits`: Number of bits $m$

**Returns**: Phase $\phi \in [0, 1)$

**Formula**:
$$\phi = \frac{\text{bitstring}}{2^m}$$

### qpe_print_result

Print QPE result.

[archived fence delimiter: ```c]
void qpe_print_result(const qpe_result_t *result);
[archived fence delimiter: ```]

## Pre-built Unitaries

### qpe_create_phase_gate

Create phase gate unitary.

[archived fence delimiter: ```c]
unitary_operator_t* qpe_create_phase_gate(double theta);
[archived fence delimiter: ```]

**Parameters**:
- `theta`: Phase angle

**Returns**: Unitary with $U|\psi\rangle = e^{i\theta}|\psi\rangle$

### qpe_create_t_gate

Create T gate unitary.

[archived fence delimiter: ```c]
unitary_operator_t* qpe_create_t_gate(void);
[archived fence delimiter: ```]

**Returns**: T gate with eigenvalue $e^{i\pi/4}$ for $|1\rangle$

**Phase**: $\phi = 1/8$ for eigenstate $|1\rangle$

### qpe_create_rz_gate

Create $R_Z(\theta)$ rotation unitary.

[archived fence delimiter: ```c]
unitary_operator_t* qpe_create_rz_gate(double theta);
[archived fence delimiter: ```]

**Parameters**:
- `theta`: Rotation angle

**Returns**: $R_Z(\theta)$ gate

**Eigenstates**:
- $|0\rangle$ with eigenvalue $e^{-i\theta/2}$, phase $\phi = -\theta/(4\pi) \mod 1$
- $|1\rangle$ with eigenvalue $e^{i\theta/2}$, phase $\phi = \theta/(4\pi) \mod 1$

## Mathematical Background

### Phase Estimation Theory

For a unitary $U$ with eigenstate $|\psi\rangle$ and eigenvalue $e^{2\pi i\phi}$:

$$U|\psi\rangle = e^{2\pi i\phi}|\psi\rangle$$

QPE uses the quantum Fourier transform to encode $\phi$ in a measurement outcome.

### Circuit Structure

[archived fence delimiter: ```]
|0⟩ ─H─────•─────────────────────QFT†─ M → φ₁
|0⟩ ─H─────┼──────•──────────────QFT†─ M → φ₂
|0⟩ ─H─────┼──────┼──────•───────QFT†─ M → φ₃
     ...   │      │      │       ...
|ψ⟩ ───────U¹─────U²─────U⁴──────────────────
[archived fence delimiter: ```]

### Precision Analysis

With $m$ precision qubits:
- Bit precision: $2^{-m}$
- Success probability: $\geq 4/\pi^2 \approx 0.405$ (for exact phases)
- Can be boosted to $1 - \epsilon$ with $O(\log(1/\epsilon))$ additional qubits

### Complexity

| Resource | Complexity |
|----------|------------|
| Qubits | $m + n$ |
| Controlled-U gates | $2^m - 1$ |
| QFT operations | $O(m^2)$ |
| Total gates | $O(2^m + m^2)$ |

## Complete Example

[archived fence delimiter: ```c]
#include "src/algorithms/qpe.h"
#include "src/quantum/state.h"
#include "src/quantum/gates.h"
#include "src/utils/quantum_entropy.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    // Initialize entropy
    quantum_entropy_ctx_t entropy;
    quantum_entropy_init(&entropy, NULL, NULL);

    // Test QPE on phase gate with known phase
    double true_phase = 0.375;  // 3/8
    double theta = 2 * M_PI * true_phase;

    printf("=== QPE Test: Phase = %.4f ===\n", true_phase);

    // Create phase gate unitary
    unitary_operator_t *U = qpe_create_phase_gate(theta);

    // Create eigenstate (|1⟩ for phase gate)
    eigenstate_t *es = eigenstate_create(1);
    gate_x(es->state, 0);  // Prepare |1⟩
    es->phase = true_phase;

    // Test with increasing precision
    for (size_t m = 2; m <= 8; m++) {
        qpe_result_t result = qpe_estimate_phase(U, es, m, &entropy);

        printf("m=%zu: estimated=%.6f, error=%.6f, confidence=%.3f\n",
               m,
               result.estimated_phase,
               result.phase_error,
               result.confidence);
    }

    // Cleanup
    unitary_operator_free(U);
    eigenstate_free(es);

    return 0;
}
[archived fence delimiter: ```]

**Expected Output**:
[archived fence delimiter: ```]
=== QPE Test: Phase = 0.3750 ===
m=2: estimated=0.250000, error=0.125000, confidence=0.850
m=3: estimated=0.375000, error=0.000000, confidence=1.000
m=4: estimated=0.375000, error=0.000000, confidence=1.000
m=5: estimated=0.375000, error=0.000000, confidence=1.000
...
[archived fence delimiter: ```]

## Precision Requirements by Application

| Application | Typical Precision | Qubits Needed |
|-------------|-------------------|---------------|
| Period finding | $O(\log N)$ | $2\log_2 N$ |
| Shor's algorithm | $2n + 3$ | $3n$ total |
| HHL eigenvalues | $O(\kappa/\epsilon)$ | Depends on condition number |
| Chemistry (0.1 mHa) | $\sim 10^{-4}$ Ha | 13-14 bits |

## Applications

### Period Finding (Shor's Algorithm)

QPE finds the period $r$ in $f(x) = a^x \mod N$:

1. Create unitary $U|y\rangle = |ay \mod N\rangle$
2. Use $|\psi\rangle = \frac{1}{\sqrt{r}}\sum_{k=0}^{r-1}|a^k \mod N\rangle$
3. QPE gives $\phi = s/r$ for some $s$
4. Continued fractions extract $r$

### Quantum Chemistry

QPE estimates eigenvalues of molecular Hamiltonians:
1. Prepare approximate ground state $|\psi_0\rangle$
2. Trotterize evolution $U = e^{-iHt}$
3. QPE gives energy: $E = -\phi \cdot 2\pi/t$

## See Also

- [Algorithms: QPE](../../algorithms/qpe-algorithm.md) - Full theory
- [Tutorial: Phase Estimation](../../tutorials/advanced/phase-estimation.md)
- [VQE API](vqe.md) - Alternative for ground states
```
