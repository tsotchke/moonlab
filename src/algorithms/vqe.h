#ifndef VQE_H
#define VQE_H

#include "../quantum/state.h"
#include "../quantum/noise.h"
#include "../utils/quantum_entropy.h"
#include <stddef.h>
#include <complex.h>

/**
 * @file vqe.h
 * @brief Variational Quantum Eigensolver (VQE) for molecular simulation
 * 
 * VQE is a hybrid quantum-classical algorithm for finding ground state
 * energies of molecular systems. Critical for:
 * - Drug discovery (molecular binding energies)
 * - Materials science (battery chemistry, catalysts)
 * - Chemical reactions (activation energies)
 * 
 * Algorithm:
 * 1. Prepare trial state with variational circuit: |ψ(θ)⟩
 * 2. Measure energy expectation: E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
 * 3. Classical optimization: minimize E(θ) over parameters θ
 * 4. Iterate until convergence
 * 
 * 32-qubit capability enables:
 * - H₂ (2 qubits), LiH (4 qubits), H₂O (8 qubits)
 * - NH₃ (10 qubits), CH₄ (12 qubits), C₂H₄ (16 qubits)
 * - Benzene C₆H₆ (24 qubits), small proteins (28-30 qubits)
 */

// ============================================================================
// MOLECULAR HAMILTONIAN
// ============================================================================

/**
 * @brief Pauli string term in Hamiltonian
 * 
 * Represents a term like 0.5 * Z₀Z₁X₂I₃
 * where coefficient = 0.5 and pauli_string = "ZZXI"
 */
typedef struct {
    double coefficient;           // Coefficient for this term
    char *pauli_string;          // String of Pauli operators (X,Y,Z,I)
    size_t num_qubits;           // Length of Pauli string
} pauli_term_t;

/**
 * @brief Pauli Hamiltonian as sum of Pauli string terms
 *
 * H = Σᵢ cᵢ Pᵢ where Pᵢ are tensor products of Pauli operators
 *
 * Example for H₂ molecule:
 * H = -1.05 II + 0.40 ZI + 0.40 IZ - 0.22 ZZ + 0.18 XX
 *
 * Note: This type is distinct from molecular_hamiltonian_t in chemistry.h
 * which stores raw molecular integrals before Jordan-Wigner transformation.
 */
typedef struct {
    size_t num_qubits;           // Number of qubits needed
    size_t num_terms;            // Number of Pauli terms
    pauli_term_t *terms;         // Array of Pauli terms
    double nuclear_repulsion;    // Nuclear-nuclear repulsion energy
    char *molecule_name;         // Name (e.g., "H2", "LiH")
    double bond_distance;        // Internuclear distance (Angstroms)
} pauli_hamiltonian_t;

/**
 * @brief Create Pauli Hamiltonian
 * @param num_qubits Number of qubits
 * @param num_terms Number of Pauli terms
 * @return Initialized Hamiltonian structure
 */
pauli_hamiltonian_t* pauli_hamiltonian_create(
    size_t num_qubits,
    size_t num_terms
);

/**
 * @brief Free Pauli Hamiltonian
 * @param hamiltonian Hamiltonian to free
 */
void pauli_hamiltonian_free(pauli_hamiltonian_t *hamiltonian);

/**
 * @brief Add Pauli term to Hamiltonian
 * @param hamiltonian Pauli Hamiltonian
 * @param coefficient Term coefficient
 * @param pauli_string Pauli string (e.g., "ZZXI")
 * @param term_index Index where to add term
 * @return 0 on success, -1 on error
 */
int pauli_hamiltonian_add_term(
    pauli_hamiltonian_t *hamiltonian,
    double coefficient,
    const char *pauli_string,
    size_t term_index
);

// ============================================================================
// PRE-BUILT MOLECULAR HAMILTONIANS
// ============================================================================

/**
 * @brief Create H₂ (Hydrogen) molecule Hamiltonian
 * 
 * 2-qubit system for hydrogen molecule at specified bond distance.
 * Uses STO-3G basis, Jordan-Wigner transformation.
 * 
 * @param bond_distance Internuclear distance in Angstroms (0.5-2.0)
 * @return H₂ Hamiltonian
 */
pauli_hamiltonian_t* vqe_create_h2_hamiltonian(double bond_distance);

/**
 * @brief Create LiH (Lithium Hydride) molecule Hamiltonian
 * 
 * 4-qubit system for LiH at specified bond distance.
 * Important for battery research.
 * 
 * @param bond_distance Internuclear distance in Angstroms
 * @return LiH Hamiltonian
 */
pauli_hamiltonian_t* vqe_create_lih_hamiltonian(double bond_distance);

/**
 * @brief Create H₂O (Water) molecule Hamiltonian
 * 
 * 8-qubit system for water molecule.
 * Fixed geometry (O-H bond = 0.958 Å, H-O-H angle = 104.5°)
 * 
 * @return H₂O Hamiltonian
 */
pauli_hamiltonian_t* vqe_create_h2o_hamiltonian(void);

// ============================================================================
// VARIATIONAL ANSATZ
// ============================================================================

/**
 * @brief Ansatz types for trial state preparation
 */
typedef enum {
    VQE_ANSATZ_HARDWARE_EFFICIENT,  // Hardware-efficient ansatz
    VQE_ANSATZ_UCCSD,               // Unitary Coupled Cluster (chemistry)
    VQE_ANSATZ_CUSTOM               // User-defined ansatz
} vqe_ansatz_type_t;

/**
 * @brief Variational circuit ansatz
 * 
 * Defines the parameterized quantum circuit for state preparation.
 * Parameters are optimized by classical optimizer.
 */
typedef struct {
    vqe_ansatz_type_t type;      // Ansatz type
    size_t num_qubits;           // Number of qubits
    size_t num_layers;           // Circuit depth
    size_t num_parameters;       // Total parameters to optimize
    double *parameters;          // Current parameter values
    
    // Circuit specification (for custom ansatz)
    void *circuit_data;          // Opaque circuit description
} vqe_ansatz_t;

/**
 * @brief Create hardware-efficient ansatz
 * 
 * Alternating rotation and entanglement layers.
 * Suitable for near-term quantum devices.
 * 
 * @param num_qubits Number of qubits
 * @param num_layers Circuit depth
 * @return Ansatz structure
 */
vqe_ansatz_t* vqe_create_hardware_efficient_ansatz(
    size_t num_qubits,
    size_t num_layers
);

/**
 * @brief Create UCCSD ansatz (chemistry-inspired)
 * 
 * Unitary Coupled Cluster Singles and Doubles.
 * Provides chemical accuracy for molecular systems.
 * 
 * @param num_qubits Number of qubits
 * @param num_electrons Number of electrons in molecule
 * @return Ansatz structure
 */
vqe_ansatz_t* vqe_create_uccsd_ansatz(
    size_t num_qubits,
    size_t num_electrons
);

/**
 * @brief Free ansatz structure
 * @param ansatz Ansatz to free
 */
void vqe_ansatz_free(vqe_ansatz_t *ansatz);

/**
 * @brief Apply ansatz circuit to quantum state
 *
 * Prepares trial state |ψ(θ)⟩ using current parameters.
 *
 * @param state Quantum state to prepare
 * @param ansatz Variational ansatz
 * @return QS_SUCCESS or error code
 */
qs_error_t vqe_apply_ansatz(
    quantum_state_t *state,
    const vqe_ansatz_t *ansatz
);

/**
 * @brief Apply ansatz circuit with noise simulation
 *
 * Prepares trial state |ψ(θ)⟩ with realistic NISQ noise.
 * Applies single and two-qubit error channels after each gate.
 *
 * @param state Quantum state to prepare
 * @param ansatz Variational ansatz
 * @param noise Noise model (NULL for ideal simulation)
 * @param entropy Random number source for noise
 * @return QS_SUCCESS or error code
 */
qs_error_t vqe_apply_ansatz_noisy(
    quantum_state_t *state,
    const vqe_ansatz_t *ansatz,
    const noise_model_t *noise,
    quantum_entropy_ctx_t *entropy
);

// ============================================================================
// CLASSICAL OPTIMIZERS
// ============================================================================

/**
 * @brief Optimizer types
 */
typedef enum {
    VQE_OPTIMIZER_COBYLA,      // Constrained optimization
    VQE_OPTIMIZER_LBFGS,       // Limited-memory BFGS
    VQE_OPTIMIZER_ADAM,        // Adaptive moment estimation
    VQE_OPTIMIZER_GRADIENT_DESCENT  // Simple gradient descent
} vqe_optimizer_type_t;

/**
 * @brief Classical optimizer configuration
 */
typedef struct {
    vqe_optimizer_type_t type;   // Optimizer type
    size_t max_iterations;       // Maximum iterations
    double tolerance;            // Convergence tolerance
    double learning_rate;        // Learning rate (for gradient methods)
    int verbose;                 // Print progress
} vqe_optimizer_t;

/**
 * @brief Create optimizer
 * @param type Optimizer type
 * @return Optimizer configuration
 */
vqe_optimizer_t* vqe_optimizer_create(vqe_optimizer_type_t type);

/**
 * @brief Free optimizer
 * @param optimizer Optimizer to free
 */
void vqe_optimizer_free(vqe_optimizer_t *optimizer);

// ============================================================================
// VQE ALGORITHM
// ============================================================================

/**
 * @brief VQE solver context
 */
typedef struct {
    pauli_hamiltonian_t *hamiltonian;      // Pauli Hamiltonian
    vqe_ansatz_t *ansatz;                  // Variational ansatz
    vqe_optimizer_t *optimizer;            // Classical optimizer
    quantum_entropy_ctx_t *entropy;        // Entropy for measurements

    // Noise model for NISQ simulation
    noise_model_t *noise_model;            // Hardware noise model (NULL = ideal)

    // Convergence tracking
    size_t iteration;                      // Current iteration
    double *energy_history;                // Energy at each iteration
    size_t max_history;                    // History buffer size

    // Statistics
    size_t total_measurements;             // Total measurements performed
    double total_time;                     // Total optimization time
} vqe_solver_t;

/**
 * @brief VQE result
 */
typedef struct {
    double ground_state_energy;       // Computed ground state energy (Hartree)
    double *optimal_parameters;       // Optimal variational parameters
    size_t num_parameters;           // Number of parameters
    size_t iterations;               // Iterations to convergence
    double convergence_tolerance;    // Final gradient norm
    int converged;                   // 1 if converged, 0 if max iterations
    
    // Reference values (for comparison)
    double fci_energy;              // Full CI energy (if available)
    double hf_energy;               // Hartree-Fock energy (if available)
    double chemical_accuracy;       // Error in kcal/mol (1 kcal/mol = 0.0016 Ha)
} vqe_result_t;

/**
 * @brief Create VQE solver
 * 
 * @param hamiltonian Molecular Hamiltonian
 * @param ansatz Variational ansatz
 * @param optimizer Classical optimizer
 * @param entropy Entropy context for measurements
 * @return VQE solver context
 */
vqe_solver_t* vqe_solver_create(
    pauli_hamiltonian_t *hamiltonian,
    vqe_ansatz_t *ansatz,
    vqe_optimizer_t *optimizer,
    quantum_entropy_ctx_t *entropy
);

/**
 * @brief Free VQE solver
 * @param solver VQE solver to free
 */
void vqe_solver_free(vqe_solver_t *solver);

/**
 * @brief Set noise model for VQE solver
 *
 * Enables NISQ-realistic simulation by applying noise after each gate
 * and readout errors during measurement.
 *
 * @param solver VQE solver context
 * @param noise_model Noise model (solver takes ownership, will free on solver_free)
 */
void vqe_solver_set_noise(vqe_solver_t *solver, noise_model_t *noise_model);

/**
 * @brief Create depolarizing noise model for VQE
 *
 * Convenience function to create a simple depolarizing noise model.
 *
 * @param single_qubit_error Single-qubit gate error probability
 * @param two_qubit_error Two-qubit gate error probability (default: 10x single)
 * @param readout_error Measurement error probability
 * @return Configured noise model
 */
noise_model_t* vqe_create_depolarizing_noise(
    double single_qubit_error,
    double two_qubit_error,
    double readout_error
);

/**
 * @brief Create realistic NISQ noise model
 *
 * Based on typical superconducting qubit parameters.
 *
 * @param t1_us T1 relaxation time in microseconds
 * @param t2_us T2 dephasing time in microseconds
 * @param gate_error Single-qubit gate error rate
 * @param readout_error Readout error rate
 * @return Configured noise model
 */
noise_model_t* vqe_create_nisq_noise(
    double t1_us,
    double t2_us,
    double gate_error,
    double readout_error
);

/**
 * @brief Compute energy expectation for given parameters
 * 
 * E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
 * 
 * This is the objective function minimized by classical optimizer.
 * 
 * @param solver VQE solver context
 * @param parameters Current variational parameters
 * @return Energy expectation value
 */
double vqe_compute_energy(
    vqe_solver_t *solver,
    const double *parameters
);

/**
 * @brief Run VQE optimization to find ground state
 * 
 * Main VQE algorithm:
 * 1. Initialize variational parameters (random or HF)
 * 2. Loop:
 *    a. Prepare state with current parameters
 *    b. Measure energy expectation
 *    c. Classical optimization step
 *    d. Check convergence
 * 3. Return optimal energy and parameters
 * 
 * @param solver VQE solver context
 * @return VQE result with ground state energy
 */
vqe_result_t vqe_solve(vqe_solver_t *solver);

/**
 * @brief Compute gradient of energy with respect to parameters
 * 
 * Uses parameter shift rule:
 * ∂E/∂θᵢ = (E(θ + π/2 eᵢ) - E(θ - π/2 eᵢ)) / 2
 * 
 * @param solver VQE solver context
 * @param parameters Current parameters
 * @param gradient Output: gradient vector
 * @return 0 on success, -1 on error
 */
int vqe_compute_gradient(
    vqe_solver_t *solver,
    const double *parameters,
    double *gradient
);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Measure expectation value of Pauli term
 * 
 * ⟨P⟩ = ⟨ψ|P|ψ⟩ for Pauli operator P
 * 
 * @param state Quantum state
 * @param pauli_term Pauli term to measure
 * @param entropy Entropy for measurements
 * @param num_samples Number of measurement samples
 * @return Expectation value
 */
double vqe_measure_pauli_expectation(
    quantum_state_t *state,
    const pauli_term_t *pauli_term,
    quantum_entropy_ctx_t *entropy,
    size_t num_samples
);

/**
 * @brief Apply Pauli rotation to state
 * 
 * Applies exp(-i θ P) where P is Pauli string.
 * 
 * @param state Quantum state
 * @param pauli_string Pauli operator string
 * @param angle Rotation angle
 * @return QS_SUCCESS or error
 */
qs_error_t vqe_apply_pauli_rotation(
    quantum_state_t *state,
    const char *pauli_string,
    double angle
);

/**
 * @brief Convert energy units
 * 
 * @param energy Energy in Hartree
 * @return Energy in kcal/mol
 */
double vqe_hartree_to_kcalmol(double energy);

/**
 * @brief Print VQE result
 * @param result VQE result
 */
void vqe_print_result(const vqe_result_t *result);

/**
 * @brief Print Hamiltonian
 * @param hamiltonian Molecular Hamiltonian
 */
void vqe_print_hamiltonian(const pauli_hamiltonian_t *hamiltonian);

#endif /* VQE_H */