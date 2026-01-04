# QAOA API

Complete reference for the Quantum Approximate Optimization Algorithm (QAOA) in the C library.

**Header**: `src/algorithms/qaoa.h`

## Overview

QAOA is a hybrid quantum-classical algorithm for combinatorial optimization problems (Farhi et al., 2014). It provides quantum advantage for NP-hard problems including:

- **MaxCut**: Graph partitioning
- **Traveling Salesman Problem (TSP)**: Route optimization
- **Portfolio optimization**: Financial asset allocation
- **Job scheduling**: Manufacturing optimization
- **Graph coloring**: Network design

The 32-qubit capability enables real business-scale problems: 32-city TSP, 32-stock portfolio, 32-node MaxCut.

## Algorithm

QAOA proceeds as follows:

1. **Encode problem** as Ising Hamiltonian: $H_C = \sum_{ij} J_{ij} Z_i Z_j + \sum_i h_i Z_i$
2. **Initialize**: $|\psi_0\rangle = H^{\otimes n}|0\rangle^n$ (uniform superposition)
3. **Alternate** $p$ layers of:
   - **Cost Hamiltonian**: $\exp(-i\gamma_k H_C)$
   - **Mixer Hamiltonian**: $\exp(-i\beta_k H_M)$ where $H_M = \sum_i X_i$
4. **Measure** and evaluate solution quality
5. **Classical optimization**: minimize $\langle H_C \rangle$ over angles $(\gamma, \beta)$

## Ising Model

### ising_model_t

Ising model for combinatorial optimization.

```c
typedef struct {
    size_t num_qubits;          // Number of qubits (problem size)
    double **J;                 // Coupling matrix [num_qubits × num_qubits]
    double *h;                  // Local fields [num_qubits]
    double offset;              // Energy offset (constant term)
    char *problem_name;         // Problem description
} ising_model_t;
```

**Mathematical Form**:
$$H = \sum_{i<j} J_{ij} Z_i Z_j + \sum_i h_i Z_i + \text{offset}$$

Where:
- $J_{ij}$: Coupling coefficients (interaction strengths)
- $h_i$: Local field coefficients (bias terms)
- $Z_i$: Pauli-Z operators (binary variables $\pm 1$)

### ising_model_create

Create Ising model.

```c
ising_model_t* ising_model_create(size_t num_qubits);
```

**Parameters**:
- `num_qubits`: Number of qubits (problem size)

**Returns**: Initialized Ising model with allocated coupling matrix and fields

### ising_model_free

Free Ising model.

```c
void ising_model_free(ising_model_t *model);
```

### ising_model_set_coupling

Set coupling coefficient $J[i][j]$.

```c
int ising_model_set_coupling(ising_model_t *model, size_t i, size_t j, double value);
```

**Parameters**:
- `model`: Ising model
- `i`, `j`: Qubit indices
- `value`: Coupling strength

**Returns**: 0 on success, -1 on error

**Note**: Coupling is symmetric; setting $J[i][j]$ also sets $J[j][i]$

### ising_model_set_field

Set local field $h[i]$.

```c
int ising_model_set_field(ising_model_t *model, size_t i, double value);
```

**Parameters**:
- `model`: Ising model
- `i`: Qubit index
- `value`: Field strength

**Returns**: 0 on success, -1 on error

### ising_model_evaluate

Evaluate Ising energy for a bit string.

```c
double ising_model_evaluate(const ising_model_t *model, uint64_t bitstring);
```

**Parameters**:
- `model`: Ising model
- `bitstring`: Bit string (0/1 for each qubit)

**Returns**: Energy value

**Mathematical Definition**:
$$E(z) = \sum_{i<j} J_{ij} z_i z_j + \sum_i h_i z_i + \text{offset}$$

where $z_i \in \{-1, +1\}$ (mapped from qubit measurement outcomes)

### ising_model_print

Print Ising model.

```c
void ising_model_print(const ising_model_t *model);
```

## Problem Encodings

### MaxCut

#### graph_t

Graph structure for MaxCut problem.

```c
typedef struct {
    size_t num_vertices;        // Number of vertices
    size_t num_edges;           // Number of edges
    int **edges;                // Edge list [num_edges][2]
    double *weights;            // Edge weights (NULL for unweighted)
} graph_t;
```

#### graph_create

Create graph.

```c
graph_t* graph_create(size_t num_vertices, size_t num_edges);
```

#### graph_free

Free graph.

```c
void graph_free(graph_t *graph);
```

#### graph_add_edge

Add edge to graph.

```c
int graph_add_edge(graph_t *graph, size_t edge_idx, int u, int v, double weight);
```

**Parameters**:
- `graph`: Graph
- `edge_idx`: Edge index
- `u`, `v`: Vertex indices
- `weight`: Edge weight (1.0 for unweighted)

**Returns**: 0 on success, -1 on error

#### ising_encode_maxcut

Encode MaxCut problem as Ising model.

```c
ising_model_t* ising_encode_maxcut(const graph_t *graph);
```

**Parameters**:
- `graph`: Input graph

**Returns**: Ising model encoding

**Encoding**:
$$H = -\sum_{(i,j) \in E} w_{ij} \frac{1 - Z_i Z_j}{2}$$

**MaxCut Objective**: Partition graph into two sets to maximize cut edges.

**Example**:
```c
// Create 4-node graph
graph_t *graph = graph_create(4, 4);
graph_add_edge(graph, 0, 0, 1, 1.0);
graph_add_edge(graph, 1, 1, 2, 1.0);
graph_add_edge(graph, 2, 2, 3, 1.0);
graph_add_edge(graph, 3, 3, 0, 1.0);

// Encode as Ising model
ising_model_t *ising = ising_encode_maxcut(graph);
```

### Portfolio Optimization

#### portfolio_problem_t

Portfolio optimization problem.

```c
typedef struct {
    size_t num_assets;          // Number of assets
    double *expected_returns;   // Expected return for each asset
    double **covariance;        // Risk covariance matrix
    double risk_aversion;       // Risk tolerance parameter λ
    double *budget_constraint;  // Budget allocation constraints
} portfolio_problem_t;
```

#### portfolio_problem_create

Create portfolio problem.

```c
portfolio_problem_t* portfolio_problem_create(size_t num_assets);
```

#### portfolio_problem_free

Free portfolio problem.

```c
void portfolio_problem_free(portfolio_problem_t *problem);
```

#### ising_encode_portfolio

Encode portfolio optimization as Ising model.

```c
ising_model_t* ising_encode_portfolio(const portfolio_problem_t *problem);
```

**Objective**: Maximize Return - $\lambda$ · Risk subject to budget constraints

### Number Partition

#### partition_problem_t

Number partition problem.

```c
typedef struct {
    size_t num_numbers;         // Number of elements
    int64_t *numbers;           // Numbers to partition
} partition_problem_t;
```

**Objective**: Partition numbers into two sets with equal sums.

#### ising_encode_partition

Encode number partition as Ising model.

```c
ising_model_t* ising_encode_partition(const partition_problem_t *problem);
```

## QAOA Algorithm

### qaoa_config_t

QAOA configuration.

```c
typedef struct {
    size_t num_qubits;          // Number of qubits
    size_t num_layers;          // Circuit depth p (typically 1-10)
    double *gamma;              // Cost Hamiltonian angles [num_layers]
    double *beta;               // Mixer Hamiltonian angles [num_layers]
} qaoa_config_t;
```

### qaoa_result_t

QAOA result.

```c
typedef struct {
    uint64_t best_bitstring;         // Best solution found
    double best_energy;              // Energy of best solution
    double *energy_history;          // Energy at each iteration
    size_t num_iterations;           // Optimization iterations
    int converged;                   // Convergence flag
    double approximation_ratio;      // Solution quality vs optimal
    double *optimal_gamma;           // Optimal cost angles
    double *optimal_beta;            // Optimal mixer angles
    size_t num_layers;              // Number of QAOA layers used
    size_t total_measurements;       // Total quantum measurements
    double optimization_time;        // Time in seconds
} qaoa_result_t;
```

### qaoa_solver_t

QAOA solver context.

```c
typedef struct {
    ising_model_t *ising;            // Problem encoding
    qaoa_config_t config;            // QAOA configuration
    quantum_entropy_ctx_t *entropy;   // Entropy for measurements
    double *current_gamma;           // Current cost angles
    double *current_beta;            // Current mixer angles
    double current_energy;           // Current best energy
    int optimizer_type;              // 0=COBYLA, 1=L-BFGS, 2=Gradient
    double learning_rate;            // For gradient methods
    size_t max_iterations;           // Max optimization iterations
    double tolerance;                // Convergence tolerance
    int verbose;                     // Print progress
    size_t total_measurements;
    double total_time;
    double *energy_history;
    size_t history_size;
} qaoa_solver_t;
```

### qaoa_solver_create

Create QAOA solver.

```c
qaoa_solver_t* qaoa_solver_create(
    ising_model_t *ising_model,
    size_t num_layers,
    quantum_entropy_ctx_t *entropy
);
```

**Parameters**:
- `ising_model`: Problem as Ising model
- `num_layers`: QAOA depth $p$ (typically 1-5)
- `entropy`: Entropy context

**Returns**: QAOA solver context

### qaoa_solver_free

Free QAOA solver.

```c
void qaoa_solver_free(qaoa_solver_t *solver);
```

### qaoa_compute_expectation

Compute expectation value for QAOA parameters.

```c
double qaoa_compute_expectation(
    qaoa_solver_t *solver,
    const double *gamma,
    const double *beta
);
```

**Parameters**:
- `solver`: QAOA solver
- `gamma`: Cost Hamiltonian angles
- `beta`: Mixer Hamiltonian angles

**Returns**: Expected energy $\langle H_C \rangle(\gamma, \beta) = \langle\psi(\gamma,\beta)|H_C|\psi(\gamma,\beta)\rangle$

### qaoa_solve

Execute QAOA optimization.

```c
qaoa_result_t qaoa_solve(qaoa_solver_t *solver);
```

**Parameters**:
- `solver`: QAOA solver

**Returns**: QAOA result with best solution

**Algorithm**:
1. Initialize angles (random or informed)
2. Loop:
   - Prepare QAOA state $|\psi(\gamma,\beta)\rangle$
   - Measure energy expectation $\langle H_C \rangle$
   - Classical optimization step
   - Check convergence
3. Return best solution

### qaoa_apply_circuit

Apply QAOA circuit for given parameters.

```c
qs_error_t qaoa_apply_circuit(
    quantum_state_t *state,
    const ising_model_t *ising,
    const double *gamma,
    const double *beta,
    size_t num_layers
);
```

**Parameters**:
- `state`: Quantum state
- `ising`: Ising model (cost Hamiltonian)
- `gamma`: Cost angles
- `beta`: Mixer angles
- `num_layers`: Number of QAOA layers

**Returns**: `QS_SUCCESS` or error

**Circuit**:
$$|\psi(\gamma,\beta)\rangle = \hat{U}(\beta_p)\hat{U}(\gamma_p)\cdots\hat{U}(\beta_1)\hat{U}(\gamma_1)|+\rangle^{\otimes n}$$

where:
- $\hat{U}(\gamma) = \exp(-i\gamma H_C)$ - Cost Hamiltonian evolution
- $\hat{U}(\beta) = \exp(-i\beta H_M)$ - Mixer Hamiltonian evolution

### qaoa_apply_cost_hamiltonian

Apply cost Hamiltonian evolution: $\exp(-i\gamma H_C)$.

```c
qs_error_t qaoa_apply_cost_hamiltonian(
    quantum_state_t *state,
    const ising_model_t *ising,
    double gamma
);
```

**Implementation**: Decomposes into $ZZ$ and $Z$ rotations

### qaoa_apply_mixer_hamiltonian

Apply mixer Hamiltonian evolution: $\exp(-i\beta H_M)$.

```c
qs_error_t qaoa_apply_mixer_hamiltonian(
    quantum_state_t *state,
    double beta
);
```

**Standard Mixer**: $H_M = \sum_i X_i$

**Implementation**: $\exp(-i\beta H_M) = \prod_i R_X(2\beta)$

## Gradient Computation

### qaoa_compute_gradient

Compute gradient of energy with respect to QAOA parameters.

```c
int qaoa_compute_gradient(
    qaoa_solver_t *solver,
    const double *gamma,
    const double *beta,
    double *grad_gamma,
    double *grad_beta
);
```

**Parameters**:
- `solver`: QAOA solver
- `gamma`: Cost angles
- `beta`: Mixer angles
- `grad_gamma`: Output gradient w.r.t. gamma
- `grad_beta`: Output gradient w.r.t. beta

**Returns**: 0 on success, -1 on error

**Method**: Uses parameter shift rule:
$$\frac{\partial\langle H \rangle}{\partial \gamma_k} = \frac{\langle H \rangle(\gamma_k + \frac{\pi}{2}) - \langle H \rangle(\gamma_k - \frac{\pi}{2})}{2}$$

## Solution Extraction

### qaoa_sample_solution

Sample solution from QAOA state.

```c
uint64_t qaoa_sample_solution(
    quantum_state_t *state,
    quantum_entropy_ctx_t *entropy
);
```

**Parameters**:
- `state`: QAOA state
- `entropy`: Entropy context

**Returns**: Measured bitstring

### qaoa_get_top_solutions

Get top k solutions from QAOA state.

```c
int qaoa_get_top_solutions(
    quantum_state_t *state,
    const ising_model_t *ising,
    size_t k,
    uint64_t *solutions,
    double *energies,
    quantum_entropy_ctx_t *entropy,
    size_t num_samples
);
```

**Parameters**:
- `state`: QAOA state
- `ising`: Problem Ising model
- `k`: Number of solutions to return
- `solutions`: Output: top k bitstrings
- `energies`: Output: energies of solutions
- `entropy`: Entropy context
- `num_samples`: Number of samples to take

**Returns**: 0 on success, -1 on error

### qaoa_approximation_ratio

Compute approximation ratio.

```c
double qaoa_approximation_ratio(
    double best_energy,
    double optimal_energy,
    double worst_energy
);
```

**Parameters**:
- `best_energy`: Best energy found by QAOA
- `optimal_energy`: Known optimal energy
- `worst_energy`: Worst possible energy

**Returns**: Approximation ratio (0 to 1, higher is better)

**Formula**:
$$\text{ratio} = \frac{\text{best} - \text{worst}}{\text{optimal} - \text{worst}}$$

## Utility Functions

### qaoa_print_result

Print QAOA result.

```c
void qaoa_print_result(const qaoa_result_t *result);
```

### qaoa_bitstring_to_binary

Convert bitstring to binary array.

```c
void qaoa_bitstring_to_binary(
    uint64_t bitstring,
    size_t num_qubits,
    int *binary
);
```

### qaoa_bitstring_to_spins

Convert bitstring to spin values ($\pm 1$).

```c
void qaoa_bitstring_to_spins(
    uint64_t bitstring,
    size_t num_qubits,
    int *spins
);
```

## Complete Example: MaxCut

```c
#include "src/algorithms/qaoa.h"
#include "src/utils/quantum_entropy.h"

int main(void) {
    // Initialize entropy
    quantum_entropy_ctx_t entropy;
    quantum_entropy_init(&entropy, NULL, NULL);

    // Create graph: 4 vertices, 5 edges (K4 minus one edge)
    graph_t *graph = graph_create(4, 5);
    graph_add_edge(graph, 0, 0, 1, 1.0);
    graph_add_edge(graph, 1, 0, 2, 1.0);
    graph_add_edge(graph, 2, 1, 2, 1.0);
    graph_add_edge(graph, 3, 1, 3, 1.0);
    graph_add_edge(graph, 4, 2, 3, 1.0);

    // Encode as Ising model
    ising_model_t *ising = ising_encode_maxcut(graph);
    ising_model_print(ising);

    // Create QAOA solver with p=3 layers
    qaoa_solver_t *solver = qaoa_solver_create(ising, 3, &entropy);
    solver->max_iterations = 100;
    solver->tolerance = 1e-5;
    solver->verbose = 1;

    // Run QAOA
    qaoa_result_t result = qaoa_solve(solver);
    qaoa_print_result(&result);

    // Extract solution
    int partition[4];
    qaoa_bitstring_to_binary(result.best_bitstring, 4, partition);

    printf("\n=== MaxCut Solution ===\n");
    printf("Partition: {");
    for (int i = 0; i < 4; i++) {
        if (partition[i] == 0) printf(" %d", i);
    }
    printf(" } vs {");
    for (int i = 0; i < 4; i++) {
        if (partition[i] == 1) printf(" %d", i);
    }
    printf(" }\n");
    printf("Cut size: %.0f edges\n", -result.best_energy);
    printf("Approximation ratio: %.3f\n", result.approximation_ratio);

    // Cleanup
    free(result.energy_history);
    free(result.optimal_gamma);
    free(result.optimal_beta);
    qaoa_solver_free(solver);
    graph_free(graph);

    return 0;
}
```

## Performance Considerations

### Layer Depth Selection

| $p$ | Typical Use | Trade-off |
|-----|-------------|-----------|
| 1 | Quick approximation | Fast but lower quality |
| 2-3 | Standard applications | Good balance |
| 5+ | High-quality solutions | More parameters, slower |

### Problem Scaling

| Qubits | Variables | Applications |
|--------|-----------|--------------|
| 8 | 8 | Small demos |
| 16 | 16 | Medium problems |
| 24 | 24 | Large real-world |
| 32 | 32 | Business-scale |

## References

- Farhi, E., Goldstone, J., & Gutmann, S. (2014). A Quantum Approximate Optimization Algorithm. arXiv:1411.4028

## See Also

- [VQE API](vqe.md) - Molecular simulation
- [Ising Models](../../concepts/variational-algorithms.md) - Theory
- [Algorithms: QAOA](../../algorithms/qaoa-algorithm.md) - Full theory
- [Tutorial: QAOA Optimization](../../tutorials/07-qaoa-optimization.md)
