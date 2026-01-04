# Variational Algorithms

Hybrid quantum-classical optimization.

## Introduction

Variational Quantum Algorithms (VQAs) are a class of hybrid algorithms that use a parameterized quantum circuit (ansatz) optimized by a classical optimizer. They are designed for near-term quantum devices with limited coherence times and gate fidelities.

## The Variational Principle

### Quantum Variational Principle

For any trial state $|\psi(\theta)\rangle$ and Hamiltonian $H$:

$$E(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle \geq E_0$$

where $E_0$ is the ground state energy.

Equality holds when $|\psi(\theta)\rangle = |E_0\rangle$.

### Optimization Objective

Find parameters $\theta^*$ that minimize the cost function:

$$\theta^* = \arg\min_\theta C(\theta)$$

where $C(\theta)$ depends on the application:
- VQE: $C(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle$
- QAOA: $C(\theta) = \langle\psi(\theta)|C|\psi(\theta)\rangle$

## Parameterized Quantum Circuits

### Ansatz Design

An **ansatz** is a parameterized circuit $U(\theta)$:

$$|\psi(\theta)\rangle = U(\theta)|0\rangle^{\otimes n}$$

Good ansätze are:
1. **Expressive**: Can represent the solution
2. **Trainable**: Gradients don't vanish
3. **Hardware-efficient**: Matches device connectivity

### Common Ansatz Types

#### Hardware-Efficient Ansatz

Alternating rotation and entanglement layers:

```
Layer 1:    Ry──Rz────●────────────
            Ry──Rz────X───●────────
            Ry──Rz────────X───●────
            Ry──Rz────────────X────

Layer 2:    Ry──Rz────●────────────
            Ry──Rz────X───●────────
            ...
```

```c
// Create hardware-efficient ansatz
for (int layer = 0; layer < num_layers; layer++) {
    for (int q = 0; q < num_qubits; q++) {
        quantum_state_ry(state, q, theta[layer][q][0]);
        quantum_state_rz(state, q, theta[layer][q][1]);
    }
    for (int q = 0; q < num_qubits - 1; q++) {
        quantum_state_cnot(state, q, q + 1);
    }
}
```

#### UCCSD Ansatz (Chemistry)

Unitary Coupled Cluster Singles and Doubles:

$$U(\theta) = e^{T(\theta) - T^\dagger(\theta)}$$

where $T = T_1 + T_2$ includes single and double excitations.

```c
// Apply UCCSD excitation operator
vqe_apply_uccsd_ansatz(state, theta, molecular_orbitals);
```

#### QAOA Ansatz

Alternating cost and mixer unitaries:

$$|\gamma, \beta\rangle = \prod_{p=1}^{P} e^{-i\beta_p B} e^{-i\gamma_p C} |+\rangle^{\otimes n}$$

```c
// QAOA layer
for (int p = 0; p < num_layers; p++) {
    // Cost unitary
    qaoa_apply_cost_unitary(state, cost_hamiltonian, gamma[p]);
    // Mixer unitary
    qaoa_apply_mixer_unitary(state, beta[p]);
}
```

## Gradient Computation

### Parameter Shift Rule

For gates of the form $e^{-i\theta G}$ with $G^2 = I$:

$$\frac{\partial C}{\partial \theta} = \frac{C(\theta + \pi/2) - C(\theta - \pi/2)}{2}$$

This is exact (not finite difference approximation).

```c
// Compute gradient via parameter shift
double gradient = 0.5 * (
    compute_cost(theta + M_PI/2) -
    compute_cost(theta - M_PI/2)
);
```

### General Parameter Shift

For generators with eigenvalues $\pm r$:

$$\frac{\partial C}{\partial \theta} = r \cdot \frac{C(\theta + \pi/(4r)) - C(\theta - \pi/(4r))}{2}$$

### Simultaneous Perturbation

**SPSA** (Simultaneous Perturbation Stochastic Approximation):

$$g(\theta) \approx \frac{C(\theta + c\Delta) - C(\theta - c\Delta)}{2c} \Delta^{-1}$$

where $\Delta$ is a random perturbation vector.

Requires only 2 circuit evaluations per iteration (vs. $2p$ for parameter shift with $p$ parameters).

## Classical Optimizers

### Gradient-Based

| Optimizer | Description |
|-----------|-------------|
| **Gradient Descent** | $\theta \leftarrow \theta - \eta \nabla C$ |
| **Adam** | Adaptive learning rate with momentum |
| **L-BFGS** | Quasi-Newton, good for smooth landscapes |
| **Natural Gradient** | Uses quantum Fisher information |

```c
// Adam optimizer
typedef struct {
    double learning_rate;
    double beta1, beta2;
    double epsilon;
    double* m;  // First moment
    double* v;  // Second moment
} adam_optimizer_t;

void adam_step(adam_optimizer_t* opt, double* theta, double* grad, int n) {
    for (int i = 0; i < n; i++) {
        opt->m[i] = opt->beta1 * opt->m[i] + (1 - opt->beta1) * grad[i];
        opt->v[i] = opt->beta2 * opt->v[i] + (1 - opt->beta2) * grad[i] * grad[i];
        double m_hat = opt->m[i] / (1 - pow(opt->beta1, opt->t));
        double v_hat = opt->v[i] / (1 - pow(opt->beta2, opt->t));
        theta[i] -= opt->learning_rate * m_hat / (sqrt(v_hat) + opt->epsilon);
    }
    opt->t++;
}
```

### Gradient-Free

| Optimizer | Description |
|-----------|-------------|
| **COBYLA** | Constrained optimization |
| **Nelder-Mead** | Simplex method |
| **Powell** | Direction set method |
| **Bayesian** | Gaussian process surrogate |

```c
// COBYLA configuration
cobyla_config_t config = {
    .max_evaluations = 1000,
    .tolerance = 1e-6,
    .initial_step = 0.5
};

cobyla_minimize(cost_function, theta, num_params, &config);
```

## VQE (Variational Quantum Eigensolver)

### Algorithm

1. Choose ansatz $U(\theta)$
2. Prepare state $|\psi(\theta)\rangle = U(\theta)|0\rangle$
3. Measure $\langle H \rangle$ via Pauli decomposition
4. Update $\theta$ using optimizer
5. Repeat until convergence

### Hamiltonian Measurement

Decompose Hamiltonian into Pauli strings:

$$H = \sum_i c_i P_i$$

Measure each $\langle P_i \rangle$ separately:

```c
// Measure Hamiltonian expectation
double energy = 0.0;
for (int i = 0; i < num_terms; i++) {
    double exp_val = measure_pauli_string(state, paulis[i], qubits[i]);
    energy += coefficients[i] * exp_val;
}
```

### Example: H2 Molecule

```c
// Create VQE solver
vqe_solver_t* vqe = vqe_create(4, 2, VQE_OPTIMIZER_ADAM);

// Set molecular Hamiltonian (H2 at 0.74 Å)
vqe_set_h2_hamiltonian(vqe, 0.74);

// Run optimization
vqe_result_t result = vqe_solve(vqe);

printf("Ground state energy: %.6f Hartree\n", result.energy);
printf("Converged in %d iterations\n", result.num_iterations);
```

## QAOA (Quantum Approximate Optimization)

### Algorithm

Solve combinatorial optimization $\max_x C(x)$:

1. Encode cost function as diagonal Hamiltonian $\hat{C}$
2. Define mixer $\hat{B} = \sum_i X_i$
3. Prepare QAOA state with $p$ layers
4. Optimize $(\gamma_1, \beta_1, \ldots, \gamma_p, \beta_p)$
5. Sample solution from final state

### MaxCut Example

Cost function:

$$C = \sum_{(i,j) \in E} \frac{1 - Z_i Z_j}{2}$$

```c
// Create QAOA solver
qaoa_solver_t* qaoa = qaoa_create(num_vertices, num_layers);

// Add graph edges
for (int e = 0; e < num_edges; e++) {
    qaoa_add_edge(qaoa, edges[e].u, edges[e].v, edges[e].weight);
}

// Solve
qaoa_result_t result = qaoa_solve(qaoa);

printf("Best cut value: %d\n", result.best_cost);
printf("Best solution: %s\n", result.best_bitstring);
```

### Approximation Ratio

QAOA with $p$ layers achieves approximation ratio:

$$r_p = \frac{\langle C \rangle_{\text{QAOA}}}{\max C}$$

For MaxCut on 3-regular graphs:
- $p=1$: $r_1 \geq 0.6924$
- $p \to \infty$: $r \to 1$

## Barren Plateaus

### The Problem

Random parameterized circuits suffer from **barren plateaus**: exponentially vanishing gradients.

$$\text{Var}[\partial_\theta C] \sim O(1/2^n)$$

### Mitigation Strategies

1. **Shallow circuits**: Limit depth
2. **Local cost functions**: Avoid global observables
3. **Layer-wise training**: Train one layer at a time
4. **Warm starting**: Initialize near known good state
5. **Structured ansätze**: Exploit problem symmetry

```c
// Layer-wise training
for (int layer = 0; layer < num_layers; layer++) {
    // Only train parameters in current layer
    for (int iter = 0; iter < max_iter; iter++) {
        double grad[PARAMS_PER_LAYER];
        compute_layer_gradient(layer, grad);
        update_layer_params(layer, grad);
    }
}
```

## Noise Effects

### Error Mitigation

Variational algorithms can be made noise-resilient:

**Zero-noise extrapolation**:
1. Run circuit at multiple noise levels
2. Extrapolate to zero noise

**Probabilistic error cancellation**:
1. Decompose ideal gates as noisy gate combinations
2. Apply quasi-probability corrections

```c
// Zero-noise extrapolation
double noise_levels[] = {1.0, 1.5, 2.0};
double energies[3];

for (int i = 0; i < 3; i++) {
    set_noise_scale(noise_levels[i]);
    energies[i] = compute_energy(theta);
}

double extrapolated = richardson_extrapolate(noise_levels, energies, 3);
```

## Convergence

### Stopping Criteria

1. **Energy convergence**: $|E_k - E_{k-1}| < \epsilon$
2. **Gradient norm**: $\|\nabla C\| < \epsilon$
3. **Parameter change**: $\|\theta_k - \theta_{k-1}\| < \epsilon$
4. **Maximum iterations**: $k > k_{\max}$

### Typical Behavior

```
Iteration | Energy    | Gradient Norm
----------|-----------|---------------
0         | -0.5000   | 0.8234
10        | -0.9823   | 0.2341
50        | -1.1234   | 0.0523
100       | -1.1364   | 0.0089
150       | -1.1371   | 0.0012
200       | -1.1372   | 0.0001  ✓ Converged
```

## References

**Foundational Papers**:
1. Peruzzo, A. et al. (2014). "A variational eigenvalue solver on a photonic quantum processor." *Nat. Commun.* 5, 4213.
2. Farhi, E., Goldstone, J., & Gutmann, S. (2014). "A Quantum Approximate Optimization Algorithm." *arXiv:1411.4028*.
3. McClean, J.R. et al. (2016). "The theory of variational hybrid quantum-classical algorithms." *New J. Phys.* 18, 023023.

**Reviews**:
4. Cerezo, M. et al. (2021). "Variational quantum algorithms." *Nat. Rev. Phys.* 3, 625-644.
5. Bharti, K. et al. (2022). "Noisy intermediate-scale quantum algorithms." *Rev. Mod. Phys.* 94, 015004.

**Training Challenges**:
6. McClean, J.R. et al. (2018). "Barren plateaus in quantum neural network training landscapes." *Nat. Commun.* 9, 4812.
7. Stokes, J. et al. (2020). "Quantum natural gradient." *Quantum* 4, 269.

## See Also

- [VQE Algorithm](../algorithms/vqe-algorithm.md) - Detailed VQE guide
- [QAOA Algorithm](../algorithms/qaoa-algorithm.md) - Detailed QAOA guide
- [Quantum Gates](quantum-gates.md) - Gate mathematics
- [C API: VQE](../api/c/vqe.md) - VQE function reference
- [C API: QAOA](../api/c/qaoa.md) - QAOA function reference

