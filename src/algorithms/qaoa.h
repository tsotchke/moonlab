#ifndef QAOA_H
#define QAOA_H

#include "../quantum/state.h"
#include "../utils/quantum_entropy.h"
#include <stddef.h>
#include <stdint.h>

/**
 * @file qaoa.h
 * @brief Quantum Approximate Optimization Algorithm (Farhi-Goldstone-Gutmann 2014).
 *
 * OVERVIEW
 * --------
 * QAOA is a hybrid quantum-classical algorithm that produces
 * approximate ground states of diagonal cost Hamiltonians
 * @f$H_C@f$.  An @f$n@f$-bit combinatorial optimisation problem
 * (MaxCut, Max-@f$k@f$-SAT, Max-Independent-Set, etc.) is first
 * encoded into an Ising Hamiltonian
 * @f[
 *   H_C \;=\; \sum_{i<j} J_{ij}\,Z_i Z_j + \sum_i h_i\,Z_i,
 * @f]
 * whose computational-basis ground state encodes the optimum.  The
 * @f$p@f$-level QAOA ansatz prepares
 * @f[
 *   |\psi_p(\gamma, \beta)\rangle \;=\;
 *   \prod_{k=1}^{p} e^{-i\beta_k H_M}\,e^{-i\gamma_k H_C}\,|+\rangle^{\otimes n},
 * @f]
 * with the *mixer* @f$H_M = \sum_i X_i@f$; the @f$2p@f$ angles
 * @f$(\gamma, \beta)@f$ are then optimised classically to minimise
 * @f$\langle\psi_p|H_C|\psi_p\rangle@f$.  As @f$p \to \infty@f$ the
 * ansatz tends to the adiabatic ground state (Farhi et al.).  At
 * fixed @f$p@f$, approximation-ratio guarantees are known on
 * specific problem classes (e.g. Farhi et al.'s 0.6924 bound for
 * MaxCut on 3-regular graphs at @f$p = 1@f$).
 *
 * GENERALISATIONS
 * ---------------
 * Hadfield et al. (2019) extend the mixer beyond @f$\sum X_i@f$ to
 * any unitary that preserves a problem-specific feasible subspace
 * ("Quantum Alternating Operator Ansatz"); this framework covers
 * constrained problems (TSP, scheduling, graph colouring) where the
 * Ising encoding uses large penalty terms.  The built-in encoders
 * in `ising_encode_maxcut`, `ising_encode_tsp`, etc. use the
 * penalty-based encoding; a future iteration can swap in the
 * constrained QAOA mixer without changing the user-facing API.
 *
 * SIZE AND NOISE CAVEATS
 * ----------------------
 * At @f$n = 32@f$ qubits the simulator can exercise the algorithm on
 * graphs up to 32 nodes for MaxCut and comparably sized TSP /
 * portfolio instances.  These numbers describe the *simulator*'s
 * capability; on current NISQ hardware the reachable depth is bound
 * by coherence time and gate fidelity rather than by qubit count, so
 * claims of "practical advantage" on real devices should be read
 * with the Preskill (2018) NISQ caveats in mind.
 *
 * REFERENCES
 * ----------
 *  - E. Farhi, J. Goldstone and S. Gutmann, "A Quantum Approximate
 *    Optimization Algorithm", arXiv:1411.4028 (2014).  Original
 *    paper; MaxCut approximation ratio.
 *  - S. Hadfield, Z. Wang, B. O'Gorman, E. G. Rieffel, D. Venturelli
 *    and R. Biswas, "From the Quantum Approximate Optimization
 *    Algorithm to a Quantum Alternating Operator Ansatz",
 *    Algorithms 12, 34 (2019), arXiv:1709.03489.  Extended mixer
 *    framework for constrained problems.
 *  - J. Preskill, "Quantum Computing in the NISQ era and beyond",
 *    Quantum 2, 79 (2018), arXiv:1801.00862.  Context for practical-
 *    scale claims.
 */

// ============================================================================
// ISING MODEL (Problem Encoding)
// ============================================================================

/**
 * @brief Ising model for combinatorial optimization
 * 
 * Represents problems as:
 * H = Σᵢⱼ Jᵢⱼ ZᵢZⱼ + Σᵢ hᵢZᵢ + offset
 * 
 * Where:
 * - Jᵢⱼ: Coupling coefficients (interaction strengths)
 * - hᵢ: Local field coefficients (bias terms)
 * - Zᵢ: Pauli-Z operators (binary variables ±1)
 */
typedef struct {
    size_t num_qubits;          // Number of qubits (problem size)
    double **J;                 // Coupling matrix [num_qubits × num_qubits]
    double *h;                  // Local fields [num_qubits]
    double offset;              // Energy offset (constant term)
    char *problem_name;         // Problem description
} ising_model_t;

/**
 * @brief Create Ising model
 * @param num_qubits Number of qubits
 * @return Initialized Ising model
 */
ising_model_t* ising_model_create(size_t num_qubits);

/**
 * @brief Free Ising model
 * @param model Ising model to free
 */
void ising_model_free(ising_model_t *model);

/**
 * @brief Set coupling coefficient J[i][j]
 * @param model Ising model
 * @param i First qubit index
 * @param j Second qubit index
 * @param value Coupling strength
 * @return 0 on success, -1 on error
 */
int ising_model_set_coupling(ising_model_t *model, size_t i, size_t j, double value);

/**
 * @brief Set local field h[i]
 * @param model Ising model
 * @param i Qubit index
 * @param value Field strength
 * @return 0 on success, -1 on error
 */
int ising_model_set_field(ising_model_t *model, size_t i, double value);

/**
 * @brief Evaluate Ising energy for bit string
 * 
 * E(z) = Σᵢⱼ Jᵢⱼ zᵢzⱼ + Σᵢ hᵢzᵢ + offset
 * where zᵢ ∈ {-1, +1} (from qubit measurement outcomes)
 * 
 * @param model Ising model
 * @param bitstring Bit string (0/1 for each qubit)
 * @return Energy value
 */
double ising_model_evaluate(const ising_model_t *model, uint64_t bitstring);

/**
 * @brief Print Ising model
 * @param model Ising model
 */
void ising_model_print(const ising_model_t *model);

// ============================================================================
// PROBLEM ENCODINGS
// ============================================================================

/**
 * @brief Graph for MaxCut problem
 */
typedef struct {
    size_t num_vertices;        // Number of vertices
    size_t num_edges;           // Number of edges
    int **edges;                // Edge list [num_edges][2]
    double *weights;            // Edge weights (NULL for unweighted)
} graph_t;

/**
 * @brief Create graph
 * @param num_vertices Number of vertices
 * @param num_edges Number of edges
 * @return Initialized graph
 */
graph_t* graph_create(size_t num_vertices, size_t num_edges);

/**
 * @brief Free graph
 * @param graph Graph to free
 */
void graph_free(graph_t *graph);

/**
 * @brief Add edge to graph
 * @param graph Graph
 * @param edge_idx Edge index
 * @param u First vertex
 * @param v Second vertex
 * @param weight Edge weight (1.0 for unweighted)
 * @return 0 on success, -1 on error
 */
int graph_add_edge(graph_t *graph, size_t edge_idx, int u, int v, double weight);

/**
 * @brief Encode MaxCut problem as Ising model
 * 
 * MaxCut: Partition graph into two sets to maximize cut edges.
 * Encoding: H = -Σ_{(i,j)∈E} wᵢⱼ(1 - ZᵢZⱼ)/2
 * 
 * @param graph Input graph
 * @return Ising model encoding
 */
ising_model_t* ising_encode_maxcut(const graph_t *graph);

/**
 * @brief Portfolio optimization problem
 */
typedef struct {
    size_t num_assets;          // Number of assets
    double *expected_returns;   // Expected return for each asset
    double **covariance;        // Risk covariance matrix
    double risk_aversion;       // Risk tolerance parameter λ
    double *budget_constraint;  // Budget allocation constraints
} portfolio_problem_t;

/**
 * @brief Create portfolio problem
 * @param num_assets Number of assets
 * @return Initialized portfolio problem
 */
portfolio_problem_t* portfolio_problem_create(size_t num_assets);

/**
 * @brief Free portfolio problem
 * @param problem Portfolio problem
 */
void portfolio_problem_free(portfolio_problem_t *problem);

/**
 * @brief Encode portfolio optimization as Ising model
 * 
 * Maximize: Return - λ·Risk
 * Subject to: Budget constraints
 * 
 * @param problem Portfolio problem
 * @return Ising model encoding
 */
ising_model_t* ising_encode_portfolio(const portfolio_problem_t *problem);

/**
 * @brief Number partition problem
 * 
 * Partition numbers into two sets with equal sums.
 */
typedef struct {
    size_t num_numbers;         // Number of elements
    int64_t *numbers;           // Numbers to partition
} partition_problem_t;

/**
 * @brief Encode number partition as Ising model
 * @param problem Partition problem
 * @return Ising model encoding
 */
ising_model_t* ising_encode_partition(const partition_problem_t *problem);

// ============================================================================
// QAOA ALGORITHM
// ============================================================================

/**
 * @brief QAOA configuration
 */
typedef struct {
    size_t num_qubits;          // Number of qubits
    size_t num_layers;          // Circuit depth p (typically 1-10)
    double *gamma;              // Cost Hamiltonian angles [num_layers]
    double *beta;               // Mixer Hamiltonian angles [num_layers]
} qaoa_config_t;

/**
 * @brief QAOA result
 */
typedef struct {
    uint64_t best_bitstring;         // Best solution found
    double best_energy;              // Energy of best solution
    double *energy_history;          // Energy at each iteration
    size_t num_iterations;           // Optimization iterations
    int converged;                   // Convergence flag
    
    // Solution quality metrics
    double approximation_ratio;      // Solution quality vs optimal
    double *optimal_gamma;           // Optimal cost angles
    double *optimal_beta;            // Optimal mixer angles
    size_t num_layers;              // Number of QAOA layers used
    
    // Statistics
    size_t total_measurements;       // Total quantum measurements
    double optimization_time;        // Time in seconds
} qaoa_result_t;

/**
 * @brief QAOA solver context
 */
typedef struct {
    ising_model_t *ising;            // Problem encoding
    qaoa_config_t config;            // QAOA configuration
    quantum_entropy_ctx_t *entropy;   // Entropy for measurements
    
    // Optimization state
    double *current_gamma;           // Current cost angles
    double *current_beta;            // Current mixer angles
    double current_energy;           // Current best energy
    
    // Classical optimizer
    int optimizer_type;              // 0=COBYLA, 1=L-BFGS, 2=Gradient
    double learning_rate;            // For gradient methods
    size_t max_iterations;           // Max optimization iterations
    double tolerance;                // Convergence tolerance
    int verbose;                     // Print progress
    
    // Statistics
    size_t total_measurements;
    double total_time;
    double *energy_history;
    size_t history_size;
} qaoa_solver_t;

/**
 * @brief Create QAOA solver
 * 
 * @param ising_model Problem as Ising model
 * @param num_layers QAOA depth p (typically 1-5)
 * @param entropy Entropy context
 * @return QAOA solver context
 */
qaoa_solver_t* qaoa_solver_create(
    ising_model_t *ising_model,
    size_t num_layers,
    quantum_entropy_ctx_t *entropy
);

/**
 * @brief Free QAOA solver
 * @param solver QAOA solver
 */
void qaoa_solver_free(qaoa_solver_t *solver);

/**
 * @brief Compute expectation value for QAOA parameters
 * 
 * ⟨H_C⟩(γ,β) = ⟨ψ(γ,β)|H_C|ψ(γ,β)⟩
 * 
 * This is the objective function minimized during optimization.
 * 
 * @param solver QAOA solver
 * @param gamma Cost Hamiltonian angles
 * @param beta Mixer Hamiltonian angles
 * @return Expected energy (cost function value)
 */
double qaoa_compute_expectation(
    qaoa_solver_t *solver,
    const double *gamma,
    const double *beta
);

/**
 * @brief Execute QAOA optimization
 * 
 * Main QAOA algorithm:
 * 1. Initialize angles (random or informed)
 * 2. Loop:
 *    a. Prepare QAOA state |ψ(γ,β)⟩
 *    b. Measure energy expectation ⟨H_C⟩
 *    c. Classical optimization step
 *    d. Check convergence
 * 3. Return best solution
 * 
 * @param solver QAOA solver
 * @return QAOA result with best solution
 */
qaoa_result_t qaoa_solve(qaoa_solver_t *solver);

/**
 * @brief Apply QAOA circuit for given parameters
 * 
 * Prepares state: |ψ(γ,β)⟩ = Û(β_p)Û(γ_p)...Û(β₁)Û(γ₁)|+⟩⊗ⁿ
 * where:
 * - Û(γ) = exp(-iγH_C) - Cost Hamiltonian evolution
 * - Û(β) = exp(-iβH_M) - Mixer Hamiltonian evolution
 * 
 * @param state Quantum state
 * @param ising Ising model (cost Hamiltonian)
 * @param gamma Cost angles
 * @param beta Mixer angles
 * @param num_layers Number of QAOA layers
 * @return QS_SUCCESS or error
 */
qs_error_t qaoa_apply_circuit(
    quantum_state_t *state,
    const ising_model_t *ising,
    const double *gamma,
    const double *beta,
    size_t num_layers
);

/**
 * @brief Apply cost Hamiltonian evolution: exp(-iγH_C)
 * 
 * Decomposes into ZZ and Z rotations.
 * 
 * @param state Quantum state
 * @param ising Ising model
 * @param gamma Cost angle
 * @return QS_SUCCESS or error
 */
qs_error_t qaoa_apply_cost_hamiltonian(
    quantum_state_t *state,
    const ising_model_t *ising,
    double gamma
);

/**
 * @brief Apply mixer Hamiltonian evolution: exp(-iβH_M)
 * 
 * Standard mixer: H_M = Σᵢ Xᵢ → exp(-iβH_M) = ∏ᵢ RX(2β)
 * 
 * @param state Quantum state
 * @param beta Mixer angle
 * @return QS_SUCCESS or error
 */
qs_error_t qaoa_apply_mixer_hamiltonian(
    quantum_state_t *state,
    double beta
);

// ============================================================================
// GRADIENT COMPUTATION
// ============================================================================

/**
 * @brief Compute gradient of energy with respect to QAOA parameters
 * 
 * Uses parameter shift rule for exact gradients:
 * ∂⟨H⟩/∂γₖ = [⟨H⟩(γₖ+π/2) - ⟨H⟩(γₖ-π/2)] / 2
 * 
 * @param solver QAOA solver
 * @param gamma Cost angles
 * @param beta Mixer angles
 * @param grad_gamma Output: gradient w.r.t. gamma
 * @param grad_beta Output: gradient w.r.t. beta
 * @return 0 on success, -1 on error
 */
int qaoa_compute_gradient(
    qaoa_solver_t *solver,
    const double *gamma,
    const double *beta,
    double *grad_gamma,
    double *grad_beta
);

// ============================================================================
// SOLUTION EXTRACTION AND ANALYSIS
// ============================================================================

/**
 * @brief Sample solution from QAOA state
 * 
 * Measures all qubits to get candidate solution.
 * Can be repeated multiple times for better solutions.
 * 
 * @param state QAOA state
 * @param entropy Entropy context
 * @return Measured bitstring
 */
uint64_t qaoa_sample_solution(
    quantum_state_t *state,
    quantum_entropy_ctx_t *entropy
);

/**
 * @brief Get top k solutions from QAOA state
 * 
 * Samples multiple times and returns best solutions.
 * 
 * @param state QAOA state
 * @param ising Problem Ising model
 * @param k Number of solutions to return
 * @param solutions Output: top k bitstrings
 * @param energies Output: energies of solutions
 * @param entropy Entropy context
 * @param num_samples Number of samples to take
 * @return 0 on success, -1 on error
 */
int qaoa_get_top_solutions(
    quantum_state_t *state,
    const ising_model_t *ising,
    size_t k,
    uint64_t *solutions,
    double *energies,
    quantum_entropy_ctx_t *entropy,
    size_t num_samples
);

/**
 * @brief Compute approximation ratio
 * 
 * ratio = (best_found - worst) / (optimal - worst)
 * 
 * @param best_energy Best energy found by QAOA
 * @param optimal_energy Known optimal energy
 * @param worst_energy Worst possible energy
 * @return Approximation ratio (0 to 1, higher is better)
 */
double qaoa_approximation_ratio(
    double best_energy,
    double optimal_energy,
    double worst_energy
);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Print QAOA result
 * @param result QAOA result
 */
void qaoa_print_result(const qaoa_result_t *result);

/**
 * @brief Convert bitstring to binary array
 * @param bitstring Input bitstring
 * @param num_qubits Number of qubits
 * @param binary Output: binary array
 */
void qaoa_bitstring_to_binary(
    uint64_t bitstring,
    size_t num_qubits,
    int *binary
);

/**
 * @brief Convert bitstring to spin values (±1)
 * @param bitstring Input bitstring
 * @param num_qubits Number of qubits
 * @param spins Output: spin array
 */
void qaoa_bitstring_to_spins(
    uint64_t bitstring,
    size_t num_qubits,
    int *spins
);

#endif /* QAOA_H */
