# Grover's Algorithm API

Complete reference for Grover's search algorithm in the C library.

**Header**: `src/algorithms/grover.h`

## Overview

Grover's algorithm provides quadratic speedup for unstructured search, finding a marked item in $N$ elements using $O(\sqrt{N})$ queries instead of classical $O(N)$. This implementation includes:

- Standard Grover search
- Adaptive iteration optimization
- Multi-target oracles
- Amplitude amplification
- Quantum sampling applications

## Core Types

### grover_config_t

Algorithm configuration.

```c
typedef struct {
    size_t num_qubits;          // Number of qubits (search space = 2^n)
    uint64_t marked_state;      // State to search for
    size_t num_iterations;      // Number of Grover iterations
    int use_optimal_iterations; // Auto-calculate optimal iterations
} grover_config_t;
```

### grover_result_t

Algorithm result.

```c
typedef struct {
    uint64_t found_state;        // Measured state
    double success_probability;  // P(measuring marked state)
    size_t oracle_calls;         // Number of oracle queries
    size_t iterations_performed; // Actual iterations
    double fidelity;             // |⟨target|final⟩|²
    int found_marked_state;      // 1 if found, 0 if not
} grover_result_t;
```

## Main Algorithm

### grover_search

Execute Grover's search algorithm.

```c
grover_result_t grover_search(
    quantum_state_t *state,
    const grover_config_t *config,
    quantum_entropy_ctx_t *entropy
);
```

**Parameters**:
- `state`: Quantum state (will be modified)
- `config`: Algorithm configuration
- `entropy`: Cryptographically secure entropy source for measurement

**Returns**: Result with found state and statistics

**Algorithm**:
1. **Initialize**: $|s\rangle = H^{\otimes n}|0\rangle^n$ (uniform superposition)
2. **Repeat** $\approx \frac{\pi}{4}\sqrt{N}$ times:
   - **Oracle**: $O|x\rangle = (-1)^{f(x)}|x\rangle$ where $f(\text{marked}) = 1$
   - **Diffusion**: $D = 2|s\rangle\langle s| - I$ (amplitude amplification)
3. **Measure**: Returns marked state with high probability

**Complexity**: $O(\sqrt{N})$ oracle queries

**Example**:
```c
#include "src/algorithms/grover.h"
#include "src/utils/quantum_entropy.h"

int main(void) {
    quantum_entropy_ctx_t entropy;
    quantum_entropy_init(&entropy, NULL, NULL);

    quantum_state_t state;
    quantum_state_init(&state, 8);  // 256-element search space

    grover_config_t config = {
        .num_qubits = 8,
        .marked_state = 42,  // Find |42⟩
        .use_optimal_iterations = 1
    };

    grover_result_t result = grover_search(&state, &config, &entropy);

    printf("Found: %llu (target was 42)\n", result.found_state);
    printf("Success probability: %.4f\n", result.success_probability);
    printf("Oracle calls: %zu\n", result.oracle_calls);

    quantum_state_free(&state);
    return 0;
}
```

### grover_optimal_iterations

Calculate optimal number of Grover iterations.

```c
size_t grover_optimal_iterations(size_t num_qubits);
```

**Parameters**:
- `num_qubits`: Number of qubits

**Returns**: Optimal iterations $\approx \lfloor \frac{\pi}{4}\sqrt{2^n} \rfloor$

**Mathematical Derivation**:
After $k$ iterations, the amplitude of the marked state is:
$$\sin((2k+1)\theta)$$
where $\sin(\theta) = 1/\sqrt{N}$. Maximum occurs at $k = \lfloor \frac{\pi}{4\theta} \rfloor$.

## Building Blocks

### grover_oracle

Apply oracle operator (phase flip on marked state).

```c
qs_error_t grover_oracle(quantum_state_t *state, uint64_t marked_state);
```

**Parameters**:
- `state`: Quantum state
- `marked_state`: Index of state to mark

**Returns**: `QS_SUCCESS` or error code

**Action**: $O|x\rangle = (-1)^{f(x)}|x\rangle$ where $f(m) = 1$ for marked state $m$

### grover_diffusion

Apply diffusion operator (inversion about average).

```c
qs_error_t grover_diffusion(quantum_state_t *state);
```

**Parameters**:
- `state`: Quantum state

**Returns**: `QS_SUCCESS` or error code

**Mathematical Definition**:
$$D = 2|s\rangle\langle s| - I = H^{\otimes n}(2|0\rangle\langle 0| - I)H^{\otimes n}$$

**Implementation**: $H^{\otimes n} \to$ Conditional phase flip $\to H^{\otimes n}$

### grover_iteration

Single Grover iteration (oracle + diffusion).

```c
qs_error_t grover_iteration(quantum_state_t *state, uint64_t marked_state);
```

**Parameters**:
- `state`: Quantum state
- `marked_state`: Target state

**Returns**: `QS_SUCCESS` or error code

## Random Sampling

### grover_random_sample

Generate random number using Grover-based quantum sampling.

```c
uint64_t grover_random_sample(
    quantum_state_t *state,
    size_t num_qubits,
    quantum_entropy_ctx_t *entropy
);
```

**Parameters**:
- `state`: Quantum state (reset and used for sampling)
- `num_qubits`: Bits of randomness (range: 0 to $2^n - 1$)
- `entropy`: Secure entropy source

**Returns**: Random sample from quantum distribution

### grover_random_samples

Generate multiple random samples.

```c
qs_error_t grover_random_samples(
    quantum_state_t *state,
    size_t num_qubits,
    uint64_t *samples,
    size_t num_samples,
    quantum_entropy_ctx_t *entropy
);
```

## Performance Analysis

### grover_analysis_t

Performance statistics structure.

```c
typedef struct {
    double success_rate;         // Fraction of successful searches
    double avg_iterations;       // Average iterations used
    double quantum_speedup;      // √N (theoretical)
    double measured_speedup;     // Observed speedup
    size_t total_oracle_calls;   // Total oracle queries
} grover_analysis_t;
```

### grover_analyze_performance

Run performance analysis.

```c
grover_analysis_t grover_analyze_performance(
    size_t num_qubits,
    size_t num_trials,
    quantum_entropy_ctx_t *entropy
);
```

**Example**:
```c
quantum_entropy_ctx_t entropy;
quantum_entropy_init(&entropy, NULL, NULL);

grover_analysis_t stats = grover_analyze_performance(10, 1000, &entropy);

printf("Success rate: %.2f%%\n", stats.success_rate * 100);
printf("Quantum speedup: %.1fx\n", stats.quantum_speedup);
```

### grover_print_result

Print formatted result.

```c
void grover_print_result(
    const grover_result_t *result,
    const grover_config_t *config
);
```

## Advanced Operations

### grover_adaptive_search

Adaptive Grover search with automatic iteration optimization.

```c
grover_result_t grover_adaptive_search(
    quantum_state_t *state,
    uint64_t marked_state,
    quantum_entropy_ctx_t *entropy
);
```

**Features**:
- Automatically determines optimal iterations
- Adjusts based on success probability feedback
- Handles unknown number of solutions

### grover_oracle_multi_phase

Oracle with multiple marked states and custom phases.

```c
qs_error_t grover_oracle_multi_phase(
    quantum_state_t *state,
    const uint64_t *marked_states,
    const double *phases,
    size_t num_marked
);
```

**Parameters**:
- `marked_states`: Array of states to mark
- `phases`: Phase angle for each marked state (radians)
- `num_marked`: Number of marked states

**Use Case**: Fine-grained amplitude control, custom distributions

### grover_amplitude_amplification

Generalized amplitude amplification.

```c
qs_error_t grover_amplitude_amplification(
    quantum_state_t *state,
    const double *target_amplitudes,
    size_t num_iterations
);
```

**Parameters**:
- `target_amplitudes`: Desired amplitude distribution
- `num_iterations`: Number of amplification iterations

### grover_importance_sampling

Quantum importance sampling with Grover speedup.

```c
qs_error_t grover_importance_sampling(
    quantum_state_t *state,
    double (*importance_function)(uint64_t),
    size_t num_samples,
    uint64_t *samples,
    quantum_entropy_ctx_t *entropy
);
```

**Parameters**:
- `importance_function`: Weight function for importance sampling
- `num_samples`: Number of samples to generate
- `samples`: Output array

### grover_mcmc_step

Quantum-enhanced MCMC step.

```c
uint64_t grover_mcmc_step(
    quantum_state_t *state,
    double (*target_distribution)(uint64_t),
    uint64_t current_state,
    quantum_entropy_ctx_t *entropy
);
```

**Use Case**: Quantum speedup for sampling from peaked distributions

## Mathematical Background

### Success Probability

After $k$ iterations, the probability of measuring the marked state is:
$$P_k = \sin^2((2k+1)\theta)$$

where $\sin(\theta) = \sqrt{M/N}$ for $M$ marked states among $N$ total.

### Optimal Iterations

For single marked state ($M = 1$):
$$k_{\text{opt}} = \left\lfloor \frac{\pi}{4}\sqrt{N} \right\rfloor$$

giving success probability $P \geq 1 - O(1/N)$.

### Multiple Solutions

For $M$ marked states:
$$k_{\text{opt}} = \left\lfloor \frac{\pi}{4}\sqrt{N/M} \right\rfloor$$

Unknown $M$ requires adaptive or amplitude estimation approaches.

## Thread Safety

- `grover_oracle` and `grover_diffusion`: Not thread-safe (modify state)
- Analysis functions: Thread-safe (create own states)

## Performance Notes

1. **GPU Acceleration**: Use Metal backend for 20-40× speedup
2. **Optimal qubits**: 8-12 qubits give best demonstration of speedup
3. **Memory**: State vector requires $2^n \times 16$ bytes

## See Also

- [Algorithms: Grover's Algorithm](../../algorithms/grovers-algorithm.md) - Full theory
- [Tutorial: Grover's Search](../../tutorials/04-grovers-search.md) - Step-by-step guide
- [GPU Metal API](gpu-metal.md) - Hardware acceleration
