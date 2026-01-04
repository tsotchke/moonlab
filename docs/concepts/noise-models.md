# Noise Models

Simulating realistic quantum hardware.

## Introduction

Real quantum computers are noisy. Gates have errors, qubits decohere, and measurements are imperfect. Noise simulation is essential for:

- Validating algorithms on realistic hardware
- Developing error mitigation strategies
- Benchmarking quantum devices
- Understanding performance limits

## Types of Quantum Noise

### Coherent vs. Incoherent Errors

**Coherent errors**: Systematic, reversible
- Over-rotation in gates
- Calibration errors
- Cross-talk between qubits

**Incoherent errors**: Random, irreversible
- Decoherence (T1, T2)
- Depolarization
- Measurement errors

### Error Channels

Noise is described by **quantum channels** (completely positive trace-preserving maps):

$$\mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger$$

where $\{E_k\}$ are Kraus operators satisfying $\sum_k E_k^\dagger E_k = I$.

## Common Noise Channels

### Bit Flip

Flips $|0\rangle \leftrightarrow |1\rangle$ with probability $p$:

$$\mathcal{E}(\rho) = (1-p)\rho + p X\rho X$$

**Kraus operators**:
$$E_0 = \sqrt{1-p} \, I, \quad E_1 = \sqrt{p} \, X$$

```c
quantum_noise_bit_flip(state, qubit, probability);
```

### Phase Flip

Applies $Z$ with probability $p$:

$$\mathcal{E}(\rho) = (1-p)\rho + p Z\rho Z$$

**Kraus operators**:
$$E_0 = \sqrt{1-p} \, I, \quad E_1 = \sqrt{p} \, Z$$

```c
quantum_noise_phase_flip(state, qubit, probability);
```

### Depolarizing Channel

Completely randomizes state with probability $p$:

$$\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

**Kraus operators**:
$$E_0 = \sqrt{1-p} \, I, \quad E_1 = \sqrt{p/3} \, X, \quad E_2 = \sqrt{p/3} \, Y, \quad E_3 = \sqrt{p/3} \, Z$$

**Interpretation**: With probability $p$, apply random Pauli error.

```c
quantum_noise_depolarizing(state, qubit, probability);
```

### Amplitude Damping

Models energy relaxation (T1 decay):

$$\mathcal{E}(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger$$

**Kraus operators**:
$$E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad E_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

**Physical meaning**: Excited state $|1\rangle$ decays to $|0\rangle$ with probability $\gamma$.

```c
quantum_noise_amplitude_damping(state, qubit, gamma);
```

### Phase Damping (Dephasing)

Models T2 decay without energy loss:

$$\mathcal{E}(\rho) = (1-\lambda)\rho + \lambda Z\rho Z$$

Equivalently, off-diagonal elements decay:

$$\rho_{01} \to \sqrt{1-\lambda} \, \rho_{01}$$

```c
quantum_noise_phase_damping(state, qubit, lambda);
```

### Generalized Amplitude Damping

Includes thermal excitation (finite temperature):

$$\gamma_\uparrow = n_{\text{th}} \gamma, \quad \gamma_\downarrow = (1 + n_{\text{th}}) \gamma$$

where $n_{\text{th}} = 1/(e^{\hbar\omega/k_B T} - 1)$ is thermal population.

```c
quantum_noise_thermal(state, qubit, gamma, temperature);
```

## Characteristic Times

### T1: Energy Relaxation Time

Time constant for $|1\rangle \to |0\rangle$ decay:

$$P(1, t) = P(1, 0) \, e^{-t/T_1}$$

### T2: Dephasing Time

Time constant for coherence decay:

$$|\rho_{01}(t)| = |\rho_{01}(0)| \, e^{-t/T_2}$$

**Relation**: $T_2 \leq 2T_1$

### T2*: Inhomogeneous Dephasing

Includes low-frequency noise and inhomogeneity:

$$T_2^* \leq T_2 \leq 2T_1$$

### Typical Values

| Platform | T1 | T2 |
|----------|----|----|
| Superconducting | 50-300 μs | 50-200 μs |
| Trapped ion | seconds | 1-10 s |
| Neutral atom | seconds | ms-s |
| NV center | 1-6 ms | 1-2 ms |

## Gate Errors

### Error Models

**Depolarizing gate error**:
```c
// Apply noisy CNOT
quantum_state_cnot(state, control, target);
quantum_noise_depolarizing_2q(state, control, target, error_rate);
```

**Over/under-rotation**:
```c
// Gate with calibration error
double noisy_angle = ideal_angle * (1.0 + calibration_error);
quantum_state_rx(state, qubit, noisy_angle);
```

### Gate Error Rates

| Gate Type | Typical Error Rate |
|-----------|-------------------|
| Single-qubit | 0.01% - 0.1% |
| Two-qubit | 0.1% - 1% |
| Measurement | 0.5% - 5% |

### Error Insertion

```c
// Create noise model
noise_model_t* noise = noise_model_create();

// Add gate errors
noise_model_add_gate_error(noise, "cx", 0.01, NOISE_DEPOLARIZING);
noise_model_add_gate_error(noise, "u3", 0.001, NOISE_DEPOLARIZING);

// Add decoherence
noise_model_add_t1_t2(noise, 50e-6, 100e-6);  // T1=50μs, T2=100μs

// Apply noisy circuit
for (int i = 0; i < num_gates; i++) {
    apply_gate(state, circuit[i]);
    noise_model_apply(noise, state, circuit[i]);
}
```

## Measurement Errors

### Readout Error

Probability of misidentifying qubit state:

$$P(\text{measure } 0 | \text{state } 1) = \epsilon_{0|1}$$
$$P(\text{measure } 1 | \text{state } 0) = \epsilon_{1|0}$$

**Confusion matrix**:
$$M = \begin{pmatrix} 1-\epsilon_{1|0} & \epsilon_{0|1} \\ \epsilon_{1|0} & 1-\epsilon_{0|1} \end{pmatrix}$$

```c
// Apply measurement error
measurement_error_t merr = {
    .p_0_given_1 = 0.02,
    .p_1_given_0 = 0.01
};
int result = quantum_state_measure_noisy(state, qubit, &merr);
```

### Error Mitigation

**Matrix inversion**:
$$P_{\text{ideal}} = M^{-1} P_{\text{measured}}$$

```c
// Mitigate readout errors
double mitigated_probs[1024];
readout_error_mitigation(measured_counts, mitigated_probs, confusion_matrix);
```

## Correlated Errors

### Cross-Talk

Gates on one qubit affect neighbors:

```c
// Model cross-talk
crosstalk_model_t* ct = crosstalk_model_create(num_qubits);
crosstalk_add_coupling(ct, qubit_a, qubit_b, coupling_strength);

// Apply gate with cross-talk
quantum_state_rx(state, qubit_a, theta);
crosstalk_apply(ct, state, qubit_a);
```

### Spatially Correlated Noise

Errors may be correlated between nearby qubits:

$$\mathcal{E}_{i,j}(\rho) = (1-p)\rho + p Z_i Z_j \rho Z_i Z_j$$

## Noise Simulation Methods

### Monte Carlo

Sample error realizations:

```c
int counts[1024] = {0};
for (int shot = 0; shot < num_shots; shot++) {
    quantum_state_reset(state);
    apply_circuit_with_noise(state, circuit, noise_model);
    int result = quantum_state_measure_all(state);
    counts[result]++;
}
```

### Density Matrix

Track full mixed state:

$$\rho \to \mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger$$

**Advantage**: Exact noise evolution
**Disadvantage**: Memory scales as $O(4^n)$

### Trajectory Method

Stochastic Schrödinger equation:
- Apply deterministic evolution
- Randomly apply jump operators
- Average over trajectories

## Noise Characterization

### Randomized Benchmarking

Measure average gate fidelity:

1. Apply random Clifford sequence
2. Append inverse Clifford
3. Measure return to initial state
4. Fit decay to extract error rate

```c
// Randomized benchmarking
rb_result_t result = randomized_benchmarking(
    state,
    qubit,
    sequence_lengths,
    num_sequences,
    num_shots
);
printf("Error per Clifford: %.4f\n", result.error_per_clifford);
```

### Process Tomography

Fully characterize quantum channel:

1. Prepare complete set of input states
2. Apply channel
3. Perform state tomography on outputs
4. Reconstruct process matrix $\chi$

### Gate Set Tomography

Self-consistent characterization of preparation, gates, and measurement.

## Implementing Noise Models

### Kraus Representation

```c
// Define Kraus operators for amplitude damping
void amplitude_damping_kraus(double gamma, complex_t E0[4], complex_t E1[4]) {
    E0[0] = (complex_t){1.0, 0.0};
    E0[1] = (complex_t){0.0, 0.0};
    E0[2] = (complex_t){0.0, 0.0};
    E0[3] = (complex_t){sqrt(1.0 - gamma), 0.0};

    E1[0] = (complex_t){0.0, 0.0};
    E1[1] = (complex_t){sqrt(gamma), 0.0};
    E1[2] = (complex_t){0.0, 0.0};
    E1[3] = (complex_t){0.0, 0.0};
}

// Apply to density matrix
void apply_kraus(complex_t* rho, int dim, complex_t** kraus, int num_kraus) {
    complex_t* rho_new = calloc(dim * dim, sizeof(complex_t));
    for (int k = 0; k < num_kraus; k++) {
        // rho_new += E_k * rho * E_k^dagger
        matrix_multiply_conjugate(kraus[k], rho, kraus[k], rho_new, dim);
    }
    memcpy(rho, rho_new, dim * dim * sizeof(complex_t));
    free(rho_new);
}
```

### Pauli Twirling

Convert coherent errors to incoherent:

```c
// Pauli twirl: P_i U P_i^† where P_i is random Pauli
int pauli = random_pauli();
apply_pauli(state, qubit, pauli);
apply_noisy_gate(state, qubit, gate);
apply_pauli(state, qubit, pauli);  // Same Pauli to undo
```

## References

**Textbooks and Reviews**:
1. Nielsen, M.A. & Chuang, I.L. (2010). *Quantum Computation and Quantum Information*. Cambridge University Press. Chapter 8.
2. Preskill, J. (2018). "Quantum Computing in the NISQ era and beyond." *Quantum* 2, 79.

**Quantum Error Correction**:
3. Knill, E., Laflamme, R., & Zurek, W.H. (1998). "Resilient quantum computation: error models and thresholds." *Proc. R. Soc. Lond. A* 454, 365-384.
4. Gottesman, D. (2010). "An introduction to quantum error correction and fault-tolerant quantum computation." *Quantum Information Science and Its Contributions to Mathematics*, 13-58.
5. Preskill, J. (1998). "Reliable quantum computers." *Proc. R. Soc. Lond. A* 454, 385-410.

**Hardware Characterization**:
6. Krantz, P. et al. (2019). "A quantum engineer's guide to superconducting qubits." *Appl. Phys. Rev.* 6, 021318.

**Error Mitigation**:
7. Temme, K., Bravyi, S., & Gambetta, J.M. (2017). "Error mitigation for short-depth quantum circuits." *Phys. Rev. Lett.* 119, 180509.

## See Also

- [Measurement Theory](measurement-theory.md) - Measurement fundamentals
- [Variational Algorithms](variational-algorithms.md) - Noise-resilient algorithms
- [C API: Noise](../api/c/noise.md) - Noise function reference
- [Noise Simulation Guide](../guides/noise-simulation.md) - Practical guide

