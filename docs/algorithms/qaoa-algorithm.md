# QAOA Algorithm

Complete guide to the Quantum Approximate Optimization Algorithm.

## Overview

The Quantum Approximate Optimization Algorithm (QAOA) is a hybrid quantum-classical algorithm for solving combinatorial optimization problems. It provides approximate solutions to NP-hard problems with potential quantum advantage.

**Discovered**: Farhi, Goldstone, Gutmann, 2014

**Applications**:
- MaxCut
- Graph coloring
- Traveling salesman
- Portfolio optimization
- Scheduling problems
- SAT solving

## Mathematical Foundation

### Problem Formulation

QAOA solves problems of the form:

$$\max_{\mathbf{z} \in \{0,1\}^n} C(\mathbf{z})$$

where $C$ is a cost function encoded as a diagonal Hamiltonian:

$$C = \sum_\alpha c_\alpha C_\alpha$$

### QAOA Ansatz

The QAOA state at depth $p$ is:

$$|\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle = \prod_{l=1}^{p} e^{-i\beta_l B} e^{-i\gamma_l C} |+\rangle^{\otimes n}$$

where:
- **Cost unitary**: $e^{-i\gamma C}$ encodes the objective
- **Mixer unitary**: $e^{-i\beta B}$ enables exploration
- **Standard mixer**: $B = \sum_i X_i$

### Expectation Value

The objective is to maximize:

$$\langle C \rangle = \langle\boldsymbol{\gamma}, \boldsymbol{\beta}|C|\boldsymbol{\gamma}, \boldsymbol{\beta}\rangle$$

## MaxCut Problem

### Problem Definition

Given graph $G = (V, E)$, partition vertices to maximize cut edges:

$$C = \sum_{(i,j) \in E} \frac{1 - Z_i Z_j}{2}$$

Each term equals 1 if vertices $i, j$ are in different partitions.

### Implementation

```c
#include "qaoa.h"

int main() {
    // Define graph (pentagon)
    int edges[][2] = {{0,1}, {1,2}, {2,3}, {3,4}, {4,0}};
    int n_edges = 5;
    int n_vertices = 5;

    // Create MaxCut Hamiltonian
    hamiltonian_t* C = maxcut_hamiltonian(edges, n_edges, n_vertices);

    // Configure QAOA
    qaoa_config_t config = {
        .num_qubits = n_vertices,
        .depth = 3,
        .optimizer = QAOA_OPTIMIZER_COBYLA,
        .max_iterations = 200
    };

    // Run QAOA
    qaoa_result_t result = qaoa_solve(C, &config);

    printf("Best cut: %llu\n", result.best_bitstring);
    printf("Cut value: %d\n", result.best_cost);
    printf("Approximation ratio: %.4f\n", result.approximation_ratio);

    hamiltonian_destroy(C);
    return 0;
}
```

### Python Interface

```python
from moonlab.algorithms import QAOA
import numpy as np

# Define graph
edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
n_vertices = 5

# Create and run QAOA
qaoa = QAOA(num_qubits=n_vertices, depth=3)
result = qaoa.solve_maxcut(edges)

print(f"Best solution: {result['best_bitstring']:05b}")
print(f"Cut value: {result['best_cost']}")
print(f"Expected value: {result['expectation']:.4f}")
print(f"Optimal γ: {result['optimal_gamma']}")
print(f"Optimal β: {result['optimal_beta']}")
```

## Circuit Structure

### Cost Unitary

For MaxCut, apply $e^{-i\gamma Z_i Z_j}$ for each edge:

```python
def apply_cost_unitary(state, edges, gamma):
    """Apply exp(-iγC) for MaxCut."""
    for i, j in edges:
        # ZZ rotation: CNOT - Rz - CNOT
        state.cnot(i, j)
        state.rz(j, 2 * gamma)  # Factor of 2 from ZZ decomposition
        state.cnot(i, j)
```

Circuit diagram for one edge:
```
q_i: ───●───────●───
        │       │
q_j: ───X──Rz───X───
```

### Mixer Unitary

Standard transverse-field mixer:

```python
def apply_mixer_unitary(state, n_qubits, beta):
    """Apply exp(-iβB) where B = Σ X_i."""
    for i in range(n_qubits):
        state.rx(i, 2 * beta)
```

### Full QAOA Circuit

```python
def qaoa_circuit(n_qubits, edges, gamma, beta, depth):
    """Create complete QAOA circuit."""
    state = QuantumState(n_qubits)

    # Initial state: |+⟩^n
    for i in range(n_qubits):
        state.h(i)

    # p layers
    for l in range(depth):
        apply_cost_unitary(state, edges, gamma[l])
        apply_mixer_unitary(state, n_qubits, beta[l])

    return state
```

## Parameter Optimization

### Landscape Analysis

For $p=1$, the energy landscape is 2D and can be visualized:

```python
def qaoa_landscape(edges, n_vertices, resolution=100):
    """Compute QAOA energy landscape for p=1."""
    gamma = np.linspace(0, 2*np.pi, resolution)
    beta = np.linspace(0, np.pi, resolution)

    E = np.zeros((resolution, resolution))

    for i, g in enumerate(gamma):
        for j, b in enumerate(beta):
            state = qaoa_circuit(n_vertices, edges, [g], [b], 1)
            E[j, i] = compute_expectation(state, edges)

    return gamma, beta, E
```

### Optimization Strategies

#### Gradient-Free

```python
from scipy.optimize import minimize

def qaoa_cost(params, n, edges, p):
    gamma = params[:p]
    beta = params[p:]
    state = qaoa_circuit(n, edges, gamma, beta, p)
    return -compute_expectation(state, edges)  # Negative for minimization

result = minimize(qaoa_cost, initial_params,
                  args=(n, edges, p),
                  method='COBYLA')
```

#### Gradient-Based

Using parameter shift rule:

```python
def qaoa_gradient(params, n, edges, p):
    """Compute gradient using parameter shift."""
    gradient = np.zeros_like(params)
    shift = np.pi / 2

    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += shift

        params_minus = params.copy()
        params_minus[i] -= shift

        gradient[i] = (qaoa_cost(params_plus, n, edges, p) -
                       qaoa_cost(params_minus, n, edges, p)) / 2

    return gradient
```

### Parameter Initialization

#### Random Initialization

```python
gamma_init = np.random.uniform(0, 2*np.pi, p)
beta_init = np.random.uniform(0, np.pi, p)
```

#### Warm Starting

From smaller depth:

```python
def warm_start(optimal_p, new_p):
    """Interpolate parameters from depth p to p+1."""
    gamma_new = np.zeros(new_p)
    beta_new = np.zeros(new_p)

    for i in range(new_p):
        # Linear interpolation
        t = i / (new_p - 1) * (len(optimal_p['gamma']) - 1)
        idx = int(t)
        frac = t - idx

        if idx < len(optimal_p['gamma']) - 1:
            gamma_new[i] = (1-frac) * optimal_p['gamma'][idx] + \
                           frac * optimal_p['gamma'][idx+1]
            beta_new[i] = (1-frac) * optimal_p['beta'][idx] + \
                          frac * optimal_p['beta'][idx+1]
        else:
            gamma_new[i] = optimal_p['gamma'][-1]
            beta_new[i] = optimal_p['beta'][-1]

    return gamma_new, beta_new
```

## Theoretical Analysis

### Approximation Ratio

For MaxCut on 3-regular graphs at $p=1$:

$$r \geq 0.6924$$

This exceeds the classical Goemans-Williamson bound of 0.878 for some instances.

### Depth-Performance Tradeoff

| Depth $p$ | Parameters | Approximation |
|-----------|------------|---------------|
| 1 | 2 | ~0.69 (3-reg) |
| 2 | 4 | ~0.75 |
| 5 | 10 | ~0.85 |
| $\to \infty$ | $\to \infty$ | 1.0 (exact) |

### Concentration

For typical instances, QAOA parameters concentrate:
- Optimal parameters depend weakly on specific instance
- Can transfer parameters between similar problems

## Advanced Topics

### Constrained Optimization

#### Penalty Method

Add penalty for violated constraints:

```python
def constrained_cost(C, constraints, penalty_weight):
    """Add penalty terms for constraints."""
    H = C.copy()
    for constraint in constraints:
        # constraint = 0 when satisfied
        H += penalty_weight * constraint**2
    return H
```

#### Custom Mixers

Preserve feasibility with XY-mixer:

```python
def xy_mixer_unitary(state, beta, edges):
    """XY mixer for constrained subspace."""
    for i, j in edges:
        # Preserves particle number
        state.rxx(i, j, beta)
        state.ryy(i, j, beta)
```

### Weighted Problems

```python
# Weighted MaxCut
weighted_edges = [(0, 1, 1.5), (1, 2, 2.0), (2, 3, 0.5)]

def weighted_cost_unitary(state, edges, gamma):
    for i, j, weight in edges:
        state.cnot(i, j)
        state.rz(j, 2 * gamma * weight)
        state.cnot(i, j)
```

### QUBO Problems

Any QUBO (Quadratic Unconstrained Binary Optimization):

$$\min_{\mathbf{x}} \mathbf{x}^T Q \mathbf{x}$$

can be converted to Ising form:

```python
def qubo_to_ising(Q):
    """Convert QUBO matrix to Ising Hamiltonian."""
    n = Q.shape[0]
    J = np.zeros((n, n))
    h = np.zeros(n)
    offset = 0

    for i in range(n):
        h[i] = Q[i, i] / 2
        offset += Q[i, i] / 2
        for j in range(i+1, n):
            J[i, j] = Q[i, j] / 4
            h[i] += Q[i, j] / 4
            h[j] += Q[i, j] / 4
            offset += Q[i, j] / 4

    return J, h, offset
```

## Problem Encodings

### Graph Coloring

$k$-coloring with $n \cdot k$ qubits:

```python
def graph_coloring_hamiltonian(edges, n_vertices, k_colors):
    """
    Encode k-coloring as QAOA problem.

    x_{v,c} = 1 if vertex v has color c
    """
    terms = []

    # Each vertex has exactly one color
    for v in range(n_vertices):
        # Penalty for not exactly one color
        for c1 in range(k_colors):
            for c2 in range(c1+1, k_colors):
                terms.append((qubit_idx(v, c1), qubit_idx(v, c2), 1.0))

    # Adjacent vertices have different colors
    for u, v in edges:
        for c in range(k_colors):
            terms.append((qubit_idx(u, c), qubit_idx(v, c), 1.0))

    return terms
```

### Traveling Salesman

Using one-hot encoding:

```python
def tsp_hamiltonian(distances, n_cities):
    """
    Encode TSP as QAOA problem.

    x_{i,t} = 1 if city i is visited at time t
    """
    n = n_cities
    terms = []

    # Distance objective
    for t in range(n):
        next_t = (t + 1) % n
        for i in range(n):
            for j in range(n):
                if i != j:
                    weight = distances[i][j]
                    terms.append((
                        qubit_idx(i, t),
                        qubit_idx(j, next_t),
                        weight
                    ))

    # Constraints: each city once, each time once
    # (Added as penalty terms)

    return terms
```

## Performance Optimization

### Moonlab Optimizations

```python
# Enable GPU acceleration
qaoa = QAOA(num_qubits=20, depth=5, backend='metal')

# Use optimized cost unitary
qaoa.use_fused_cost_layer(True)  # Fuses ZZ rotations

# Parallel parameter evaluation
results = qaoa.evaluate_parallel([params1, params2, params3])
```

### Batched Evaluation

```python
def batched_qaoa_cost(param_batch, n, edges, p):
    """Evaluate multiple parameter sets in parallel."""
    states = []
    for params in param_batch:
        gamma, beta = params[:p], params[p:]
        states.append(qaoa_circuit(n, edges, gamma, beta, p))

    return [compute_expectation(s, edges) for s in states]
```

## Example: Portfolio Optimization

```python
from moonlab.algorithms import QAOA
import numpy as np

# Expected returns and covariance
returns = np.array([0.1, 0.15, 0.08, 0.12])
cov_matrix = np.array([
    [0.1, 0.02, 0.01, 0.03],
    [0.02, 0.15, 0.02, 0.01],
    [0.01, 0.02, 0.08, 0.02],
    [0.03, 0.01, 0.02, 0.12]
])

# Markowitz objective: maximize return - λ * risk
lambda_risk = 0.5

def portfolio_cost(selection):
    """Compute portfolio value for binary selection."""
    selected_returns = returns[selection]
    selected_cov = cov_matrix[np.ix_(selection, selection)]

    return np.sum(selected_returns) - lambda_risk * np.sum(selected_cov)

# Convert to QUBO and solve with QAOA
qaoa = QAOA(num_qubits=4, depth=3)
result = qaoa.solve_qubo(Q_matrix)

print(f"Optimal portfolio: {result['best_bitstring']:04b}")
print(f"Expected value: {portfolio_cost(result['best_bitstring']):.4f}")
```

## Complexity Analysis

| Component | Complexity |
|-----------|------------|
| Circuit depth | $O(p \cdot m)$ for $m$ edges |
| Parameters | $2p$ |
| Measurements | $O(1/\epsilon^2)$ per evaluation |
| Optimization | $O(2p)$ dimensions |

## See Also

- [Tutorial: QAOA](../tutorials/07-qaoa-optimization.md) - Step-by-step tutorial
- [C API: QAOA](../api/c/qaoa.md) - Complete C API reference
- [Variational Algorithms](../concepts/variational-algorithms.md) - Theory background
- [VQE Algorithm](vqe-algorithm.md) - Related variational algorithm

## References

**Foundational Papers**:
1. Farhi, E., Goldstone, J., & Gutmann, S. (2014). "A Quantum Approximate Optimization Algorithm." *arXiv:1411.4028*.
2. Farhi, E. & Harrow, A.W. (2016). "Quantum supremacy through the quantum approximate optimization algorithm." *arXiv:1602.07674*.

**Analysis and Implementation**:
3. Zhou, L. et al. (2020). "Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices." *Phys. Rev. X* 10, 021067.
4. Crooks, G.E. (2018). "Performance of the quantum approximate optimization algorithm on the maximum cut problem." *arXiv:1811.08419*.
5. Guerreschi, G.G. & Matsuura, A.Y. (2019). "QAOA for Max-Cut requires hundreds of qubits for quantum speed-up." *Sci. Rep.* 9, 6903.

**Extensions and Variants**:
6. Hadfield, S. et al. (2019). "From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz." *Algorithms* 12(2), 34.
7. Harrigan, M.P. et al. (2021). "Quantum approximate optimization of non-planar graph problems on a planar superconducting processor." *Nat. Phys.* 17, 332-336.

