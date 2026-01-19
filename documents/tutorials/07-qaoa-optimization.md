# Tutorial 07: QAOA Optimization

Solve combinatorial optimization problems with QAOA.

**Duration**: 45 minutes
**Prerequisites**: [Tutorial 06](06-vqe-molecular-simulation.md)
**Difficulty**: Intermediate

## Learning Objectives

By the end of this tutorial, you will:

- Understand the QAOA algorithm structure
- Encode optimization problems as Hamiltonians
- Implement and optimize QAOA circuits
- Solve MaxCut and other combinatorial problems

## The Problem: MaxCut

Given a graph $G = (V, E)$, partition vertices into two sets to maximize edges between them.

**Applications**:
- Network design
- VLSI circuit layout
- Clustering
- Machine learning

**Complexity**: NP-hard (no efficient classical algorithm known)

## QAOA Overview

**Quantum Approximate Optimization Algorithm**:

1. Encode cost function as Hamiltonian $C$
2. Prepare initial superposition $|+\rangle^{\otimes n}$
3. Apply $p$ layers of:
   - Cost unitary: $e^{-i\gamma C}$
   - Mixer unitary: $e^{-i\beta B}$
4. Measure and compute cost
5. Optimize parameters $(\gamma, \beta)$

## Step 1: Problem Encoding

### MaxCut Hamiltonian

For edge $(i, j)$, the cost contribution is:

$$C_{ij} = \frac{1 - Z_i Z_j}{2}$$

This equals 1 if $i$ and $j$ are in different partitions, 0 otherwise.

Total cost:
$$C = \sum_{(i,j) \in E} \frac{1 - Z_i Z_j}{2}$$

```python
import numpy as np
from moonlab import QuantumState

# Example: 5-node graph (pentagon)
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
n_vertices = 5

def create_maxcut_hamiltonian(edges):
    """Create MaxCut Hamiltonian as list of ZZ terms."""
    terms = []
    for i, j in edges:
        terms.append((i, j, 0.5))  # coefficient 0.5 for (1 - ZZ)/2
    return terms

hamiltonian = create_maxcut_hamiltonian(edges)
```

## Step 2: QAOA Circuit

```python
def apply_cost_unitary(state, edges, gamma):
    """
    Apply cost unitary: exp(-i * gamma * C)
    For MaxCut: applies exp(-i * gamma * ZZ/2) for each edge
    """
    for i, j in edges:
        # ZZ rotation: exp(-i * gamma * Z_i Z_j / 2)
        state.cnot(i, j)
        state.rz(j, gamma)
        state.cnot(i, j)

def apply_mixer_unitary(state, n, beta):
    """
    Apply mixer unitary: exp(-i * beta * B)
    Standard mixer: B = sum_i X_i
    """
    for i in range(n):
        state.rx(i, 2 * beta)

def qaoa_circuit(n, edges, gamma, beta, p):
    """
    Create QAOA circuit with p layers.

    gamma: array of p phase parameters
    beta: array of p mixer parameters
    """
    state = QuantumState(n)

    # Initial superposition
    for i in range(n):
        state.h(i)

    # p layers
    for layer in range(p):
        apply_cost_unitary(state, edges, gamma[layer])
        apply_mixer_unitary(state, n, beta[layer])

    return state
```

## Step 3: Cost Function

```python
def evaluate_cut(bitstring, edges):
    """Evaluate MaxCut cost for a given bitstring."""
    cut_value = 0
    for i, j in edges:
        bit_i = (bitstring >> i) & 1
        bit_j = (bitstring >> j) & 1
        if bit_i != bit_j:
            cut_value += 1
    return cut_value

def compute_expectation(state, edges, shots=1000):
    """Compute expected cut value from QAOA state."""
    total_cost = 0

    for _ in range(shots):
        # Need to reset and rebuild state for each shot
        result = state.measure_all()
        cost = evaluate_cut(result, edges)
        total_cost += cost

    return total_cost / shots

def qaoa_cost(params, n, edges, p):
    """
    QAOA cost function for optimization.
    We minimize negative expectation (to maximize cut).
    """
    gamma = params[:p]
    beta = params[p:]

    state = qaoa_circuit(n, edges, gamma, beta, p)
    expectation = compute_expectation(state, edges)

    return -expectation  # Negative because we minimize
```

## Step 4: Optimization

```python
from scipy.optimize import minimize

# Parameters
n = 5
p = 2  # Number of QAOA layers

# Initial random parameters
initial_gamma = np.random.uniform(0, 2*np.pi, p)
initial_beta = np.random.uniform(0, np.pi, p)
initial_params = np.concatenate([initial_gamma, initial_beta])

# Optimize
result = minimize(
    qaoa_cost,
    initial_params,
    args=(n, edges, p),
    method='COBYLA',
    options={'maxiter': 200}
)

optimal_gamma = result.x[:p]
optimal_beta = result.x[p:]

print(f"Optimal gamma: {optimal_gamma}")
print(f"Optimal beta: {optimal_beta}")
print(f"Expected cut value: {-result.fun:.4f}")
```

## Step 5: Sample Solutions

```python
def sample_qaoa(n, edges, gamma, beta, p, shots=1000):
    """Sample from QAOA and return best solution."""
    state = qaoa_circuit(n, edges, gamma, beta, p)

    solutions = {}
    for _ in range(shots):
        state_copy = qaoa_circuit(n, edges, gamma, beta, p)
        result = state_copy.measure_all()
        cost = evaluate_cut(result, edges)

        bitstring = format(result, f'0{n}b')
        if bitstring not in solutions:
            solutions[bitstring] = {'count': 0, 'cost': cost}
        solutions[bitstring]['count'] += 1

    # Find best solution
    best = max(solutions.items(), key=lambda x: x[1]['cost'])

    return solutions, best

solutions, best = sample_qaoa(n, edges, optimal_gamma, optimal_beta, p)

print("\nTop solutions:")
sorted_solutions = sorted(solutions.items(),
                         key=lambda x: x[1]['cost'],
                         reverse=True)
for bitstring, data in sorted_solutions[:5]:
    print(f"  {bitstring}: cut = {data['cost']}, count = {data['count']}")

print(f"\nBest solution: {best[0]} with cut value {best[1]['cost']}")
```

## Using the Built-in QAOA

```python
from moonlab.algorithms import QAOA

# Define graph
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (1, 3)]
n_vertices = 5

# Create QAOA solver
qaoa = QAOA(num_qubits=n_vertices, num_layers=3)

# Solve MaxCut
result = qaoa.solve_maxcut(edges)

print(f"Best cut: {result['best_bitstring']:05b}")
print(f"Cut value: {result['best_cost']}")
print(f"Expectation: {result['expectation']:.4f}")
print(f"Optimal gamma: {result['optimal_gamma']}")
print(f"Optimal beta: {result['optimal_beta']}")
```

## Visualizing the Landscape

```python
import matplotlib.pyplot as plt

def qaoa_landscape(n, edges, p=1, resolution=50):
    """Compute QAOA landscape for p=1."""
    gamma_range = np.linspace(0, 2*np.pi, resolution)
    beta_range = np.linspace(0, np.pi, resolution)

    landscape = np.zeros((resolution, resolution))

    for i, gamma in enumerate(gamma_range):
        for j, beta in enumerate(beta_range):
            state = qaoa_circuit(n, edges, [gamma], [beta], 1)
            # Use exact expectation for speed
            exp_val = 0
            for edge in edges:
                exp_val += 0.5 * (1 - state.expectation_zz(edge[0], edge[1]))
            landscape[j, i] = exp_val

    return gamma_range, beta_range, landscape

gamma_range, beta_range, landscape = qaoa_landscape(n, edges)

plt.figure(figsize=(10, 8))
plt.contourf(gamma_range, beta_range, landscape, levels=50, cmap='viridis')
plt.colorbar(label='Expected Cut Value')
plt.xlabel('γ')
plt.ylabel('β')
plt.title('QAOA Landscape (p=1)')
plt.savefig('qaoa_landscape.png')
plt.show()
```

## Scaling QAOA

### Increasing Depth

```python
# Compare different depths
for p in [1, 2, 3, 4, 5]:
    qaoa = QAOA(num_qubits=5, num_layers=p)
    result = qaoa.solve_maxcut(edges)

    # Approximation ratio
    max_cut = 5  # Maximum possible cut for pentagon
    approx_ratio = result['best_cost'] / max_cut

    print(f"p={p}: Cut={result['best_cost']}, Approx ratio={approx_ratio:.3f}")
```

### Warm Starting

Use classical solution to initialize QAOA:

```python
def warm_start_params(edges, n, p):
    """Initialize QAOA from classical approximate solution."""
    # Use greedy algorithm for initial guess
    partition = np.zeros(n, dtype=int)
    for v in range(n):
        # Put in partition that maximizes cut
        cost_0 = sum(1 for i, j in edges
                    if (i == v and partition[j] == 1) or
                       (j == v and partition[i] == 1))
        cost_1 = sum(1 for i, j in edges
                    if (i == v and partition[j] == 0) or
                       (j == v and partition[i] == 0))
        partition[v] = 1 if cost_1 > cost_0 else 0

    # Set gamma, beta to bias toward this solution
    gamma = np.ones(p) * 0.5
    beta = np.ones(p) * 0.2

    return np.concatenate([gamma, beta])
```

## Other Optimization Problems

### Ising Model

```python
def ising_qaoa(J, h, p):
    """
    QAOA for Ising model: H = sum_ij J_ij Z_i Z_j + sum_i h_i Z_i
    """
    n = len(h)
    qaoa = QAOA(num_qubits=n, num_layers=p)
    result = qaoa.solve_ising(J, h)
    return result

# Example: 4-spin chain
J = np.array([
    [0, -1, 0, 0],
    [-1, 0, -1, 0],
    [0, -1, 0, -1],
    [0, 0, -1, 0]
])
h = np.array([0.1, 0, -0.1, 0])

result = ising_qaoa(J, h, p=3)
print(f"Ground state: {result['best_bitstring']:04b}")
print(f"Energy: {result['best_cost']:.4f}")
```

### Graph Coloring

```python
# Encode graph coloring as QUBO
def graph_coloring_qaoa(edges, n_vertices, n_colors):
    """
    Encode k-coloring as QAOA problem.
    Uses n_vertices * n_colors qubits.
    """
    # Implementation would convert to Ising form
    pass
```

## Exercises

### Exercise 1: Larger Graphs

Solve MaxCut on a 10-vertex random graph. How does the solution quality scale?

### Exercise 2: Weighted MaxCut

Modify the code to handle weighted edges.

### Exercise 3: Compare with Classical

Implement a classical greedy/local-search algorithm and compare with QAOA.

### Exercise 4: Traveling Salesman

Encode a small TSP instance (4-5 cities) and solve with QAOA.

## Key Takeaways

1. **QAOA** approximates solutions to combinatorial optimization
2. **Cost Hamiltonian** encodes the objective function
3. **Mixer Hamiltonian** enables exploration
4. **Increasing p** improves approximation quality
5. **Warm starting** accelerates convergence

## Next Steps

Scale to larger systems with tensor networks:

**[08. Tensor Network Simulation →](08-tensor-network-simulation.md)**

## Further Reading

- [QAOA Algorithm](../algorithms/qaoa-algorithm.md) - Full mathematical treatment
- [Variational Algorithms](../concepts/variational-algorithms.md) - Theory background
- [C API: QAOA](../api/c/qaoa.md) - Low-level implementation
- Farhi et al. (2014). "A Quantum Approximate Optimization Algorithm." arXiv:1411.4028.

