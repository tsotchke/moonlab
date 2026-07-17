#ifndef VQE_H
#define VQE_H

#include "../quantum/state.h"
#include "../quantum/noise.h"
#include "../utils/quantum_entropy.h"
#include <stddef.h>
#include <complex.h>

#include "../applications/moonlab_api.h"
#ifdef __cplusplus
extern "C" {
#endif


/**
 * @file vqe.h
 * @brief Variational Quantum Eigensolver for electronic-structure problems.
 *
 * OVERVIEW
 * --------
 * The Variational Quantum Eigensolver (Peruzzo et al. 2014) targets
 * the ground-state energy of a Hamiltonian @f$H@f$ by combining a
 * parameterised quantum state preparation with a classical outer
 * loop:
 *
 *   1. Prepare @f$|\psi(\theta)\rangle = U(\theta)|\psi_0\rangle@f$
 *      on the quantum device, where @f$U(\theta)@f$ is a
 *      user-chosen ansatz circuit.
 *   2. Measure the energy expectation
 *      @f$E(\theta) = \langle\psi(\theta)|H|\psi(\theta)\rangle@f$
 *      as a weighted sum over the Pauli terms of @f$H@f$.
 *   3. Update @f$\theta@f$ via a classical optimiser (Nelder-Mead,
 *      COBYLA, SPSA, gradient-based).
 *   4. Iterate until convergence.
 *
 * By the Rayleigh-Ritz variational principle
 * @f$E(\theta) \ge E_{\min}(H)@f$ for every @f$\theta@f$, so VQE's
 * returned energy is an upper bound on the true ground-state
 * energy.  Convergence to the exact ground state depends entirely on
 * the *expressibility* of the ansatz; the accuracy gap relative to
 * the full configuration-interaction (FCI) energy is the primary
 * quality metric.
 *
 * ANSATZ CATALOGUE
 * ----------------
 * Three ansatz families are built in:
 *
 *  - *Hardware-efficient ansatz* (Kandala et al. 2017): alternating
 *    layers of parameterised single-qubit rotations and a fixed
 *    entangling pattern (CNOT ladder or ring).  Good for small
 *    molecules on NISQ hardware; no physical symmetries imposed.
 *  - *Unitary coupled-cluster with singles and doubles* (UCCSD, used
 *    in Peruzzo et al. 2014 and O'Malley et al. 2016): physically
 *    motivated, systematically improvable, particle-number
 *    preserving.  Parameter count grows as @f$O(n_o^2 n_v^2)@f$.
 *  - *Symmetry-preserving Givens ansatz*
 *    (`VQE_ANSATZ_SYMMETRY_PRESERVING`): particle-conserving Givens
 *    rotations; reaches chemical accuracy (<1 kcal/mol) on H2 with
 *    fewer parameters than UCCSD and is the default for small
 *    molecules.  See `documents/api/c/vqe.md` for details.
 *
 * EXACT GROUND-STATE REFERENCE
 * ----------------------------
 * `vqe_exact_ground_state_energy` constructs the full Pauli
 * Hamiltonian matrix and returns the lowest eigenvalue via shifted
 * power iteration, providing an FCI-equivalent reference for unit
 * testing against the variational loop.
 *
 * REFERENCES
 * ----------
 *  - A. Peruzzo, J. McClean, P. Shadbolt, M.-H. Yung, X.-Q. Zhou,
 *    P. J. Love, A. Aspuru-Guzik and J. L. O'Brien, "A variational
 *    eigenvalue solver on a quantum processor", Nat. Commun. 5, 4213
 *    (2014), arXiv:1304.3061.  Original VQE paper.
 *  - P. J. J. O'Malley et al., "Scalable Quantum Simulation of
 *    Molecular Energies", Phys. Rev. X 6, 031007 (2016),
 *    arXiv:1512.06860.  H2 on superconducting qubits with UCCSD
 *    (matches the H2 Hamiltonian built by
 *    `vqe_create_h2_hamiltonian`).
 *  - A. Kandala et al., "Hardware-efficient variational quantum
 *    eigensolver for small molecules and quantum magnets",
 *    Nature 549, 242 (2017), arXiv:1704.05018.  HEA ansatz family.
 *  - S. McArdle, S. Endo, A. Aspuru-Guzik, S. C. Benjamin and X. Yuan,
 *    "Quantum computational chemistry", Rev. Mod. Phys. 92, 015003
 *    (2020), arXiv:1808.10402.  Modern review.
 *
 * @since v0.1.2
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
    uint64_t hf_reference;       // Hartree-Fock bitstring: bit q = 1 means
                                 // qubit q is |1> in the HF reference state.
} pauli_hamiltonian_t;

/**
 * @brief Create Pauli Hamiltonian
 * @param num_qubits Number of qubits
 * @param num_terms Number of Pauli terms
 * @return Initialized Hamiltonian structure
 */
MOONLAB_API pauli_hamiltonian_t* pauli_hamiltonian_create(
    size_t num_qubits,
    size_t num_terms
);

/**
 * @brief Free Pauli Hamiltonian
 * @param hamiltonian Hamiltonian to free
 */
MOONLAB_API void pauli_hamiltonian_free(pauli_hamiltonian_t *hamiltonian);

/**
 * @brief Add Pauli term to Hamiltonian
 * @param hamiltonian Pauli Hamiltonian
 * @param coefficient Term coefficient
 * @param pauli_string Pauli string (e.g., "ZZXI")
 * @param term_index Index where to add term
 * @return 0 on success, -1 on error
 */
MOONLAB_API int pauli_hamiltonian_add_term(
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
MOONLAB_API pauli_hamiltonian_t* vqe_create_h2_hamiltonian(double bond_distance);

/**
 * @brief Create LiH (Lithium Hydride) molecule Hamiltonian
 * 
 * 4-qubit system for LiH at specified bond distance.
 * Important for battery research.
 * 
 * @param bond_distance Internuclear distance in Angstroms
 * @return LiH Hamiltonian
 */
MOONLAB_API pauli_hamiltonian_t* vqe_create_lih_hamiltonian(double bond_distance);

/**
 * @brief Create H₂O (Water) molecule Hamiltonian
 * 
 * 8-qubit system for water molecule.
 * Fixed geometry (O-H bond = 0.958 Å, H-O-H angle = 104.5°)
 * 
 * @return H₂O Hamiltonian
 */
MOONLAB_API pauli_hamiltonian_t* vqe_create_h2o_hamiltonian(void);

/**
 * @brief Exact ground-state energy of a Pauli Hamiltonian by direct
 *        matrix diagonalization. Intended for small (<=~10 qubit)
 *        reference / verification. Complexity O(4^n).
 * @return Ground-state energy including nuclear_repulsion, or NAN on error.
 */
MOONLAB_API double vqe_exact_ground_state_energy(const pauli_hamiltonian_t *hamiltonian);

// ============================================================================
// VARIATIONAL ANSATZ
// ============================================================================

/**
 * @brief Ansatz types for trial state preparation
 */
typedef enum {
    VQE_ANSATZ_HARDWARE_EFFICIENT,  // Hardware-efficient ansatz
    VQE_ANSATZ_UCCSD,               // Unitary Coupled Cluster (chemistry)
    VQE_ANSATZ_CUSTOM,              // User-defined ansatz
    VQE_ANSATZ_SYMMETRY_PRESERVING  // Particle-conserving Givens rotations
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
MOONLAB_API vqe_ansatz_t* vqe_create_hardware_efficient_ansatz(
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
MOONLAB_API vqe_ansatz_t* vqe_create_uccsd_ansatz(
    size_t num_qubits,
    size_t num_electrons
);

/**
 * @brief Particle-conserving (symmetry-preserving) ansatz.
 *        One Givens rotation per occupied-virtual qubit pair, per layer.
 *        Gives chemical accuracy for small molecules (2-6 qubits) where
 *        the GS lives in a fixed-occupation sector.
 */
MOONLAB_API vqe_ansatz_t* vqe_create_symmetry_preserving_ansatz(
    size_t num_qubits,
    size_t num_occupied,
    size_t num_layers
);

/**
 * @brief Free ansatz structure
 * @param ansatz Ansatz to free
 */
MOONLAB_API void vqe_ansatz_free(vqe_ansatz_t *ansatz);

/**
 * @brief Apply ansatz circuit to quantum state
 *
 * Prepares trial state |ψ(θ)⟩ using current parameters.
 *
 * @param state Quantum state to prepare
 * @param ansatz Variational ansatz
 * @return QS_SUCCESS or error code
 */
MOONLAB_API qs_error_t vqe_apply_ansatz(
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
MOONLAB_API qs_error_t vqe_apply_ansatz_noisy(
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
    VQE_OPTIMIZER_GRADIENT_DESCENT, // Simple gradient descent
    VQE_OPTIMIZER_QNG          // Quantum natural gradient (Fubini-Study metric)
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
MOONLAB_API vqe_optimizer_t* vqe_optimizer_create(vqe_optimizer_type_t type);

/**
 * @brief Free optimizer
 * @param optimizer Optimizer to free
 */
MOONLAB_API void vqe_optimizer_free(vqe_optimizer_t *optimizer);

// ============================================================================
// VQE ALGORITHM
// ============================================================================

/**
 * @brief VQE solver context
 *
 * The struct tag exists so the stable ABI header can forward-declare
 * the solver as an opaque handle (moonlab_vqe_solver_t) without
 * pulling in this header.
 */
typedef struct vqe_solver {
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

    // Gradient contract: when a noise model is attached the parameter-shift
    // gradient is stochastic and no longer the exact gradient of the ideal
    // energy.  By default vqe_compute_gradient refuses that case
    // (VQE_GRADIENT_ERR_NOT_EXACT); set this to opt into stochastic PSR.
    int allow_stochastic_gradient;         // 0 = exact-or-error (default)
} vqe_solver_t;

/* Return codes for vqe_compute_gradient (exact-or-error contract). */
#define VQE_GRADIENT_SUCCESS       0   /**< Exact gradient computed. */
#define VQE_GRADIENT_ERR_INVALID  (-1) /**< NULL solver/ansatz/args. */
#define VQE_GRADIENT_ERR_NOT_EXACT (-2)/**< Cannot be exact (noise attached)
                                         *   and stochastic PSR not opted in. */

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
MOONLAB_API vqe_solver_t* vqe_solver_create(
    pauli_hamiltonian_t *hamiltonian,
    vqe_ansatz_t *ansatz,
    vqe_optimizer_t *optimizer,
    quantum_entropy_ctx_t *entropy
);

/**
 * @brief Free VQE solver
 * @param solver VQE solver to free
 */
MOONLAB_API void vqe_solver_free(vqe_solver_t *solver);

/**
 * @brief Set noise model for VQE solver
 *
 * Enables NISQ-realistic simulation by applying noise after each gate
 * and readout errors during measurement.
 *
 * @param solver VQE solver context
 * @param noise_model Noise model (solver takes ownership, will free on solver_free)
 */
MOONLAB_API void vqe_solver_set_noise(vqe_solver_t *solver, noise_model_t *noise_model);

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
MOONLAB_API noise_model_t* vqe_create_depolarizing_noise(
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
MOONLAB_API noise_model_t* vqe_create_nisq_noise(
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
MOONLAB_API double vqe_compute_energy(
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
MOONLAB_API vqe_result_t vqe_solve(vqe_solver_t *solver);

/**
 * @brief Release the heap arrays owned by a vqe_result_t.
 *
 * vqe_solve returns a vqe_result_t by value that owns optimal_parameters.
 * This frees that buffer and zeroes the pointer; it does not free the struct
 * itself (it is caller-owned / stack).
 *
 * @param result VQE result whose owned arrays should be freed (may be NULL)
 */
MOONLAB_API void vqe_result_free(vqe_result_t *result);

/**
 * @brief Compute the exact gradient of the energy with respect to the
 *        variational parameters
 *
 * Dispatches between two exact paths; neither is a finite difference:
 *
 * 1. Reverse-mode adjoint autograd (fast path) for noise-free
 *    hardware-efficient ansaetze: builds a moonlab_diff circuit tape
 *    and back-propagates the Pauli-sum cotangent. Cost is ~2 forward
 *    passes per Hamiltonian term, independent of parameter count.
 * 2. Analytic parameter-shift rule (fallback) for UCCSD, symmetry-
 *    preserving, and custom ansaetze in the NOISE-FREE case:
 *    ∂E/∂θᵢ = (E(θ + π/2 eᵢ) - E(θ - π/2 eᵢ)) / 2.
 *
 * Exact-or-error contract (Eshkol RFC 4.4): when a noise model is attached and
 * enabled the parameter-shift rule runs on a stochastic energy and is no longer
 * the exact gradient of the ideal energy.  In that case this function returns
 * VQE_GRADIENT_ERR_NOT_EXACT and writes nothing, UNLESS the caller has opted in
 * via vqe_solver_set_allow_stochastic_gradient(solver, 1), in which case the
 * stochastic PSR is run and 0 is returned.
 *
 * @param solver VQE solver context
 * @param parameters Current parameters
 * @param gradient Output: gradient vector (ansatz->num_parameters slots)
 * @return VQE_GRADIENT_SUCCESS (0), VQE_GRADIENT_ERR_INVALID (-1), or
 *         VQE_GRADIENT_ERR_NOT_EXACT (-2)
 */
MOONLAB_API int vqe_compute_gradient(
    vqe_solver_t *solver,
    const double *parameters,
    double *gradient
);

/**
 * @brief Opt into (or out of) stochastic parameter-shift gradients under noise.
 *
 * @param solver VQE solver context
 * @param allow  Non-zero to permit stochastic PSR when a noise model is
 *               attached; zero (default) to keep the exact-or-error contract.
 */
MOONLAB_API void vqe_solver_set_allow_stochastic_gradient(vqe_solver_t *solver,
                                                          int allow);

/**
 * @brief Quantum geometric (Fubini-Study) tensor of the ansatz state.
 *
 * Computes g_ij = Re[<d_i psi|d_j psi> - <d_i psi|psi><psi|d_j psi>] for the
 * ideal (noise-free) trial state |psi(parameters)>, via central differences on
 * the statevector.  This is the natural Riemannian metric on the ansatz's
 * parameter space, used by the quantum natural gradient optimizer.
 *
 * @param solver VQE solver context
 * @param parameters Current parameters (num_parameters slots)
 * @param qgt_out Output: symmetric metric, row-major num_parameters x num_parameters
 * @return 0 on success, -1 on error
 * @stability experimental
 */
MOONLAB_API int vqe_compute_qgt(
    vqe_solver_t *solver,
    const double *parameters,
    double *qgt_out
);

/**
 * @brief Natural-gradient direction (g + eps*I)^{-1} * gradient.
 *
 * Preconditions GRADIENT by the regularized quantum geometric tensor.  On
 * success DIRECTION_OUT holds the natural-gradient step direction; on failure
 * (singular metric) callers should fall back to the plain gradient.
 *
 * @param solver VQE solver context
 * @param parameters Current parameters
 * @param gradient Energy gradient (num_parameters slots)
 * @param regularization Tikhonov shift added to the metric diagonal (e.g. 1e-3)
 * @param direction_out Output: natural-gradient direction (num_parameters slots)
 * @return 0 on success, -1 on error
 * @stability experimental
 */
MOONLAB_API int vqe_natural_gradient_direction(
    vqe_solver_t *solver,
    const double *parameters,
    const double *gradient,
    double regularization,
    double *direction_out
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
MOONLAB_API double vqe_measure_pauli_expectation(
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
MOONLAB_API qs_error_t vqe_apply_pauli_rotation(
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
MOONLAB_API double vqe_hartree_to_kcalmol(double energy);

/**
 * @brief Print VQE result
 * @param result VQE result
 */
MOONLAB_API void vqe_print_result(const vqe_result_t *result);

/**
 * @brief Print Hamiltonian
 * @param hamiltonian Molecular Hamiltonian
 */
MOONLAB_API void vqe_print_hamiltonian(const pauli_hamiltonian_t *hamiltonian);

#ifdef __cplusplus
}
#endif

#endif /* VQE_H */
