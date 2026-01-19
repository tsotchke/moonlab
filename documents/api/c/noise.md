# Noise API

Complete reference for quantum noise models and error channels in the C library.

**Header**: `src/quantum/noise.h`

## Overview

The noise module provides realistic noise simulation for NISQ (Noisy Intermediate-Scale Quantum) devices, including:

- Depolarizing noise
- Amplitude damping (T1 relaxation)
- Phase damping (T2 dephasing)
- Bit flip, phase flip channels
- Thermal relaxation
- Readout errors

## Noise Model Structure

### noise_model_t

Configuration structure for quantum noise.

```c
typedef struct {
    int enabled;                    // Whether noise is active

    // Single-qubit error rates
    double depolarizing_rate;       // Depolarizing probability
    double amplitude_damping_rate;  // Amplitude damping (T1)
    double phase_damping_rate;      // Phase damping (T2 pure)

    // Thermal relaxation
    double t1;                      // T1 time (energy relaxation)
    double t2;                      // T2 time (dephasing)
    double gate_time;               // Gate duration for thermal calc

    // Two-qubit error rates
    double two_qubit_depolarizing_rate;

    // Readout errors
    double readout_error_0;         // P(1|0) - measure 1 when state is 0
    double readout_error_1;         // P(0|1) - measure 0 when state is 1
} noise_model_t;
```

## Noise Model Management

### noise_model_create

Create a new noise model with default parameters.

```c
noise_model_t* noise_model_create(void);
```

**Returns**: Pointer to new noise model, or NULL on error

**Default Values**: All rates set to 0 (ideal simulation)

### noise_model_destroy

Free a noise model.

```c
void noise_model_destroy(noise_model_t *model);
```

### noise_model_copy

Create a copy of a noise model.

```c
noise_model_t* noise_model_copy(const noise_model_t *model);
```

### noise_model_create_realistic

Create a noise model from typical hardware parameters.

```c
noise_model_t* noise_model_create_realistic(
    double t1_us,
    double t2_us,
    double gate_error,
    double readout_error
);
```

**Parameters**:
- `t1_us`: T1 relaxation time in microseconds (typical: 50-200 µs)
- `t2_us`: T2 dephasing time in microseconds (typical: 20-100 µs)
- `gate_error`: Single-qubit gate error probability (typical: 0.001-0.01)
- `readout_error`: Measurement error probability (typical: 0.01-0.05)

**Example**:
```c
// IBM-like superconducting qubit parameters
noise_model_t *model = noise_model_create_realistic(
    100.0,  // T1 = 100 µs
    50.0,   // T2 = 50 µs
    0.001,  // 0.1% gate error
    0.02    // 2% readout error
);
```

## Noise Channels

### noise_depolarizing_single

Apply depolarizing channel to a single qubit.

```c
void noise_depolarizing_single(
    quantum_state_t *state,
    int qubit,
    double probability,
    double random_value
);
```

**Parameters**:
- `state`: Quantum state (modified)
- `qubit`: Target qubit
- `probability`: Depolarizing probability $p$
- `random_value`: Random number in [0, 1) for stochastic application

**Mathematical Definition**:
$$\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

With probability $p$, applies a random Pauli error; otherwise leaves state unchanged.

### noise_depolarizing_two_qubit

Apply depolarizing channel to two qubits.

```c
void noise_depolarizing_two_qubit(
    quantum_state_t *state,
    int qubit1,
    int qubit2,
    double probability,
    double random_value
);
```

**Mathematical Definition**:
$$\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{15}\sum_{P \in \{I,X,Y,Z\}^{\otimes 2} \setminus I\otimes I} P\rho P$$

### noise_amplitude_damping

Apply amplitude damping channel (energy relaxation, T1 decay).

```c
void noise_amplitude_damping(
    quantum_state_t *state,
    int qubit,
    double gamma,
    double random_value
);
```

**Parameters**:
- `state`: Quantum state (modified)
- `qubit`: Target qubit
- `gamma`: Damping parameter $\gamma = 1 - e^{-t/T_1}$
- `random_value`: Random number for stochastic simulation

**Mathematical Definition (Kraus operators)**:
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}$$

**Physical Meaning**: Models spontaneous emission from $|1\rangle$ to $|0\rangle$.

### noise_phase_damping

Apply phase damping channel (pure dephasing).

```c
void noise_phase_damping(
    quantum_state_t *state,
    int qubit,
    double gamma,
    double random_value
);
```

**Parameters**:
- `gamma`: Dephasing parameter $\gamma = 1 - e^{-t/T_\phi}$ where $1/T_\phi = 1/T_2 - 1/(2T_1)$

**Mathematical Definition (Kraus operators)**:
$$K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\gamma} \end{pmatrix}$$

**Physical Meaning**: Destroys phase coherence without changing populations.

### noise_pure_dephasing

Apply pure dephasing with random phase.

```c
void noise_pure_dephasing(
    quantum_state_t *state,
    int qubit,
    double sigma,
    double random_phase
);
```

**Parameters**:
- `sigma`: Standard deviation of phase fluctuation
- `random_phase`: Random phase value (Gaussian-distributed)

### noise_bit_flip

Apply bit flip channel.

```c
void noise_bit_flip(
    quantum_state_t *state,
    int qubit,
    double probability,
    double random_value
);
```

**Mathematical Definition**:
$$\mathcal{E}(\rho) = (1-p)\rho + p \cdot X\rho X$$

### noise_phase_flip

Apply phase flip channel.

```c
void noise_phase_flip(
    quantum_state_t *state,
    int qubit,
    double probability,
    double random_value
);
```

**Mathematical Definition**:
$$\mathcal{E}(\rho) = (1-p)\rho + p \cdot Z\rho Z$$

### noise_bit_phase_flip

Apply bit-phase flip channel (Y error).

```c
void noise_bit_phase_flip(
    quantum_state_t *state,
    int qubit,
    double probability,
    double random_value
);
```

**Mathematical Definition**:
$$\mathcal{E}(\rho) = (1-p)\rho + p \cdot Y\rho Y$$

### noise_thermal_relaxation

Apply combined thermal relaxation (T1 and T2 processes).

```c
void noise_thermal_relaxation(
    quantum_state_t *state,
    int qubit,
    double t1,
    double t2,
    double time,
    const double *random_values
);
```

**Parameters**:
- `t1`: T1 relaxation time
- `t2`: T2 dephasing time (must satisfy $T_2 \leq 2T_1$)
- `time`: Gate/idle duration
- `random_values`: Array of random numbers for stochastic simulation

**Physical Meaning**: Combines amplitude damping and phase damping to model realistic qubit decoherence.

### noise_readout_error

Simulate measurement readout error.

```c
int noise_readout_error(
    int outcome,
    double error_0_to_1,
    double error_1_to_0,
    double random_value
);
```

**Parameters**:
- `outcome`: True measurement outcome (0 or 1)
- `error_0_to_1`: Probability of reading 1 when true state is 0
- `error_1_to_0`: Probability of reading 0 when true state is 1
- `random_value`: Random number for stochastic flip

**Returns**: Potentially flipped measurement outcome

## Applying Noise Models

### noise_apply_model

Apply noise model to a single qubit after a gate.

```c
void noise_apply_model(
    quantum_state_t *state,
    int qubit,
    const noise_model_t *model,
    const double *random_values
);
```

**Parameters**:
- `state`: Quantum state
- `qubit`: Target qubit
- `model`: Noise model configuration
- `random_values`: Array of random values for stochastic channels

### noise_apply_model_two_qubit

Apply noise model after a two-qubit gate.

```c
void noise_apply_model_two_qubit(
    quantum_state_t *state,
    int qubit1,
    int qubit2,
    const noise_model_t *model,
    const double *random_values
);
```

## Configuration Functions

### noise_model_set_depolarizing

```c
void noise_model_set_depolarizing(noise_model_t *model, double rate);
```

### noise_model_set_amplitude_damping

```c
void noise_model_set_amplitude_damping(noise_model_t *model, double rate);
```

### noise_model_set_phase_damping

```c
void noise_model_set_phase_damping(noise_model_t *model, double rate);
```

### noise_model_set_thermal

```c
void noise_model_set_thermal(noise_model_t *model, double t1, double t2);
```

### noise_model_set_gate_time

```c
void noise_model_set_gate_time(noise_model_t *model, double time);
```

### noise_model_set_readout_error

```c
void noise_model_set_readout_error(
    noise_model_t *model,
    double error_0,
    double error_1
);
```

### noise_model_set_enabled

```c
void noise_model_set_enabled(noise_model_t *model, int enabled);
```

## Example: Noisy Simulation

```c
#include "src/quantum/state.h"
#include "src/quantum/gates.h"
#include "src/quantum/noise.h"
#include "src/utils/entropy.h"

int main(void) {
    // Create noise model matching IBM hardware
    noise_model_t *noise = noise_model_create_realistic(
        100.0,   // T1 = 100 µs
        50.0,    // T2 = 50 µs
        0.001,   // 0.1% single-qubit error
        0.02     // 2% readout error
    );
    noise->two_qubit_depolarizing_rate = 0.01;  // 1% two-qubit error

    // Create quantum state
    quantum_state_t state;
    quantum_state_init(&state, 2);

    // Initialize entropy source
    entropy_ctx_t *entropy = entropy_create();

    // Noisy Bell state preparation
    gate_hadamard(&state, 0);

    // Apply noise after Hadamard
    double randoms[4];
    for (int i = 0; i < 4; i++) randoms[i] = entropy_double(entropy);
    noise_apply_model(&state, 0, noise, randoms);

    gate_cnot(&state, 0, 1);

    // Apply noise after CNOT
    for (int i = 0; i < 4; i++) randoms[i] = entropy_double(entropy);
    noise_apply_model_two_qubit(&state, 0, 1, noise, randoms);

    // Measure with readout error
    double p_one = measurement_probability_one(&state, 0);
    double r = entropy_double(entropy);
    int outcome = (r < p_one) ? 1 : 0;

    // Apply readout error
    r = entropy_double(entropy);
    outcome = noise_readout_error(outcome, noise->readout_error_0,
                                   noise->readout_error_1, r);

    printf("Measured qubit 0: %d\n", outcome);

    // Cleanup
    noise_model_destroy(noise);
    entropy_destroy(entropy);
    quantum_state_free(&state);

    return 0;
}
```

## Typical Hardware Parameters

| Platform | T1 (µs) | T2 (µs) | 1Q Error | 2Q Error | Readout |
|----------|---------|---------|----------|----------|---------|
| IBM Eagle (2023) | 100-200 | 50-100 | 0.1% | 1% | 1-2% |
| Google Sycamore | 20-50 | 10-30 | 0.1% | 0.5% | 3% |
| Rigetti Aspen | 30-80 | 20-50 | 0.5% | 2% | 5% |
| IonQ | 10,000+ | 1,000+ | 0.01% | 0.3% | 0.5% |

## See Also

- [VQE API](vqe.md) - VQE with noise simulation
- [Guides: Noise Simulation](../../guides/noise-simulation.md) - Detailed guide
- [Concepts: Noise Models](../../concepts/noise-models.md) - Theory
