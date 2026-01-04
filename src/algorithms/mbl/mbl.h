/**
 * @file mbl.h
 * @brief Many-Body Localization (MBL) simulation module
 *
 * This module implements quantum simulation of many-body localized systems:
 * - Disordered Heisenberg spin chains (XXZ model with random fields)
 * - MBL phase detection via level statistics
 * - Entanglement entropy dynamics (logarithmic growth signature)
 * - Imbalance dynamics (memory of initial state)
 * - Local integrals of motion (LIOMs / l-bits)
 * - MBL-thermal phase transition detection
 *
 * Many-Body Localization is a quantum phase where strong disorder prevents
 * thermalization, leading to:
 * - Area-law entanglement (vs volume-law in thermal phase)
 * - Poisson level statistics (vs GOE in thermal phase)
 * - Persistent memory of initial conditions
 * - Emergent local integrals of motion
 *
 * @stability stable
 * @since v1.0.0
 */

#ifndef MBL_H
#define MBL_H

#include "../../quantum/state.h"
#include <stdint.h>
#include <stdbool.h>
#include <complex.h>

// ============================================================================
// DISORDERED HAMILTONIANS
// ============================================================================

/**
 * @brief Random field Heisenberg XXZ model parameters
 *
 * H = J Σᵢ (Sˣᵢ Sˣᵢ₊₁ + Sʸᵢ Sʸᵢ₊₁ + Δ Sᶻᵢ Sᶻᵢ₊₁) + Σᵢ hᵢ Sᶻᵢ
 *
 * where hᵢ are random fields drawn from [-W, W]
 */
typedef struct {
    uint32_t num_sites;         // Number of spin-1/2 sites
    double J;                   // Exchange coupling (typically J=1)
    double delta;               // Anisotropy parameter Δ
    double disorder_strength;   // Disorder strength W
    double *random_fields;      // Random field at each site hᵢ ∈ [-W, W]
    bool periodic_bc;           // Periodic boundary conditions
    uint64_t disorder_seed;     // RNG seed for reproducibility
} xxz_hamiltonian_t;

/**
 * @brief Sparse Hamiltonian representation
 *
 * Stores H in CSR (Compressed Sparse Row) format for efficient
 * matrix-vector multiplication in time evolution.
 */
typedef struct {
    uint32_t dim;               // Matrix dimension (2^num_sites)
    uint32_t nnz;               // Number of non-zero elements
    double complex *values;     // Non-zero values
    uint32_t *col_indices;      // Column indices
    uint32_t *row_ptr;          // Row pointers (size dim+1)
    double *eigenvalues;        // Cached eigenvalues (if computed)
    double complex *eigenvectors; // Cached eigenvectors (if computed)
    bool eigensystem_computed;  // Flag for cached eigensystem
} sparse_hamiltonian_t;

/**
 * @brief Create random field XXZ Hamiltonian
 *
 * Generates a disordered Heisenberg model with random on-site fields.
 * This is the canonical model for studying MBL.
 *
 * @param num_sites Number of spin-1/2 sites (qubits)
 * @param J Exchange coupling (set J=1 for standard units)
 * @param delta Anisotropy: Δ=1 (Heisenberg), Δ=0 (XX), Δ→∞ (Ising)
 * @param disorder_strength Disorder W: fields drawn from [-W, W]
 * @param periodic_bc Use periodic boundary conditions
 * @param seed Random seed for disorder realization
 * @return XXZ Hamiltonian structure
 */
xxz_hamiltonian_t *xxz_hamiltonian_create(uint32_t num_sites,
                                           double J, double delta,
                                           double disorder_strength,
                                           bool periodic_bc,
                                           uint64_t seed);

/**
 * @brief Free XXZ Hamiltonian
 */
void xxz_hamiltonian_free(xxz_hamiltonian_t *h);

/**
 * @brief Build sparse matrix representation
 *
 * Constructs the full Hamiltonian matrix in CSR format.
 * For L sites, matrix is 2^L × 2^L.
 *
 * @param xxz XXZ Hamiltonian parameters
 * @return Sparse Hamiltonian matrix
 */
sparse_hamiltonian_t *xxz_build_sparse(const xxz_hamiltonian_t *xxz);

/**
 * @brief Free sparse Hamiltonian
 */
void sparse_hamiltonian_free(sparse_hamiltonian_t *h);

/**
 * @brief Compute full eigensystem of Hamiltonian
 *
 * Uses LAPACK to compute all eigenvalues and eigenvectors.
 * Results are cached in the sparse_hamiltonian_t structure.
 *
 * @param h Sparse Hamiltonian (modified to cache results)
 * @return QS_SUCCESS or error code
 */
qs_error_t sparse_hamiltonian_diagonalize(sparse_hamiltonian_t *h);

// ============================================================================
// LEVEL STATISTICS
// ============================================================================

/**
 * @brief Level spacing ratio statistics
 *
 * The ratio rₙ = min(sₙ, sₙ₊₁)/max(sₙ, sₙ₊₁) where sₙ = Eₙ₊₁ - Eₙ
 * distinguishes MBL from thermal phases:
 * - Thermal (GOE): ⟨r⟩ ≈ 0.5307 (Wigner-Dyson)
 * - MBL (Poisson): ⟨r⟩ ≈ 0.3863
 */
typedef struct {
    double *ratios;             // Individual r values
    uint32_t num_ratios;        // Number of ratios
    double mean_ratio;          // ⟨r⟩
    double std_ratio;           // Standard deviation
    double poisson_distance;    // |⟨r⟩ - 0.3863|
    double goe_distance;        // |⟨r⟩ - 0.5307|
} level_statistics_t;

/**
 * @brief Compute level spacing ratio statistics
 *
 * Analyzes energy level statistics to detect MBL phase.
 * Works within a fixed symmetry sector (e.g., fixed Sᶻ_total).
 *
 * @param eigenvalues Sorted array of eigenvalues
 * @param num_eigenvalues Number of eigenvalues
 * @param filter_edges Exclude fraction of spectrum edges (typically 0.1)
 * @return Level statistics structure
 */
level_statistics_t *compute_level_statistics(const double *eigenvalues,
                                              uint32_t num_eigenvalues,
                                              double filter_edges);

/**
 * @brief Free level statistics
 */
void level_statistics_free(level_statistics_t *stats);

/**
 * @brief Determine phase from level statistics
 *
 * @param stats Level statistics
 * @return 1 for MBL phase, 0 for thermal phase, -1 for inconclusive
 */
int classify_phase_from_levels(const level_statistics_t *stats);

// ============================================================================
// ENTANGLEMENT DYNAMICS
// ============================================================================

/**
 * @brief Entanglement entropy time series
 *
 * Tracks S(t) = -Tr(ρ_A log₂ ρ_A) for subsystem A over time.
 * MBL signature: logarithmic growth S(t) ~ log(t)
 * Thermal signature: linear growth then saturation S(t) ~ t → L/2
 */
typedef struct {
    double *times;              // Time points
    double *entropies;          // S(t) values
    uint32_t num_points;        // Number of time points
    double saturation_value;    // Asymptotic S_∞
    double growth_exponent;     // Fitted exponent in S ~ t^α
    double log_coefficient;     // Coefficient in S ~ c·log(t)
} entropy_dynamics_t;

/**
 * @brief Simulate entanglement entropy growth
 *
 * Time-evolves an initial product state and tracks bipartite
 * entanglement entropy. Uses exact diagonalization for small systems
 * or Krylov subspace methods for larger ones.
 *
 * @param h Sparse Hamiltonian
 * @param initial_state Initial quantum state (typically product state)
 * @param subsystem_qubits Qubits in subsystem A
 * @param num_subsystem Number of qubits in A
 * @param t_max Maximum time
 * @param num_steps Number of time steps
 * @return Entropy dynamics structure
 */
entropy_dynamics_t *simulate_entropy_dynamics(const sparse_hamiltonian_t *h,
                                               const quantum_state_t *initial_state,
                                               const uint32_t *subsystem_qubits,
                                               uint32_t num_subsystem,
                                               double t_max, uint32_t num_steps);

/**
 * @brief Free entropy dynamics
 */
void entropy_dynamics_free(entropy_dynamics_t *dyn);

/**
 * @brief Fit logarithmic vs linear growth
 *
 * Determines whether entropy grows as log(t) (MBL) or t (thermal).
 *
 * @param dyn Entropy dynamics data
 * @param log_fit_quality Output: R² for log(t) fit
 * @param linear_fit_quality Output: R² for t fit
 * @return 1 if logarithmic dominates, 0 if linear dominates
 */
int fit_entropy_growth(const entropy_dynamics_t *dyn,
                       double *log_fit_quality, double *linear_fit_quality);

// ============================================================================
// IMBALANCE DYNAMICS
// ============================================================================

/**
 * @brief Imbalance order parameter
 *
 * I(t) = (1/L) Σᵢ (-1)ⁱ ⟨Sᶻᵢ(t)⟩
 *
 * For Néel initial state |↑↓↑↓...⟩:
 * - Thermal: I(t) → 0 (system thermalizes, loses memory)
 * - MBL: I(t) → I_∞ > 0 (persistent memory)
 */
typedef struct {
    double *times;              // Time points
    double *imbalance;          // I(t) values
    uint32_t num_points;        // Number of time points
    double initial_imbalance;   // I(0)
    double asymptotic_imbalance; // I_∞ (fitted long-time value)
    double decay_rate;          // Exponential decay rate (if thermal)
} imbalance_dynamics_t;

/**
 * @brief Simulate imbalance dynamics
 *
 * Time-evolves a Néel state and tracks the imbalance order parameter.
 * This is a key experimental observable for detecting MBL.
 *
 * @param h Sparse Hamiltonian
 * @param t_max Maximum time
 * @param num_steps Number of time steps
 * @return Imbalance dynamics structure
 */
imbalance_dynamics_t *simulate_imbalance_dynamics(const sparse_hamiltonian_t *h,
                                                   double t_max,
                                                   uint32_t num_steps);

/**
 * @brief Free imbalance dynamics
 */
void imbalance_dynamics_free(imbalance_dynamics_t *dyn);

/**
 * @brief Determine phase from imbalance
 *
 * @param dyn Imbalance dynamics
 * @param threshold Threshold for persistent imbalance (typically 0.1)
 * @return 1 for MBL, 0 for thermal
 */
int classify_phase_from_imbalance(const imbalance_dynamics_t *dyn,
                                   double threshold);

// ============================================================================
// LOCAL INTEGRALS OF MOTION (LIOMs)
// ============================================================================

/**
 * @brief Local integral of motion (l-bit / LIOM)
 *
 * In the MBL phase, the system has extensively many LIOMs τᶻᵢ
 * that are quasi-local and commute with H and each other:
 * [H, τᶻᵢ] = 0, [τᶻᵢ, τᶻⱼ] = 0
 *
 * Each LIOM is a dressed version of a physical spin:
 * τᶻᵢ = Sᶻᵢ + Σⱼ aᵢⱼ Oⱼ + ...
 */
typedef struct {
    uint32_t site;              // Primary site (center of LIOM)
    double complex *operator;   // Full operator in computational basis
    double *locality_profile;   // |⟨τᶻᵢ|Oⱼ⟩| vs distance |i-j|
    double localization_length; // Exponential decay length
    uint32_t num_sites;         // System size
} liom_t;

/**
 * @brief LIOM system (all LIOMs for a given Hamiltonian)
 */
typedef struct {
    liom_t **lioms;             // Array of LIOMs
    uint32_t num_lioms;         // Number of LIOMs (= num_sites)
    double mean_loc_length;     // Average localization length
    double max_overlap;         // Max inter-LIOM overlap (should be ~0)
} liom_system_t;

/**
 * @brief Construct LIOMs via exact diagonalization
 *
 * Uses the eigenstates to construct LIOMs:
 * τᶻᵢ = Σₙ |n⟩⟨n| ⟨n|Sᶻᵢ|n⟩
 *
 * These operators are exactly diagonal in the energy eigenbasis
 * and inherit locality from the physical spins.
 *
 * @param h Sparse Hamiltonian (must be diagonalized)
 * @return LIOM system
 */
liom_system_t *construct_lioms(const sparse_hamiltonian_t *h);

/**
 * @brief Free LIOM system
 */
void liom_system_free(liom_system_t *sys);

/**
 * @brief Measure LIOM locality
 *
 * Computes the localization length ξ where |⟨τᶻᵢ|Oⱼ⟩| ~ exp(-|i-j|/ξ).
 *
 * @param liom Single LIOM
 * @return Localization length ξ
 */
double liom_localization_length(const liom_t *liom);

// ============================================================================
// TIME EVOLUTION
// ============================================================================

/**
 * @brief Time evolution method
 */
typedef enum {
    EVOLUTION_EXACT,            // Exact: exp(-iHt) via diagonalization
    EVOLUTION_KRYLOV,           // Krylov subspace (Lanczos)
    EVOLUTION_TROTTER           // Trotter-Suzuki decomposition
} evolution_method_t;

/**
 * @brief Time evolve quantum state
 *
 * Applies U(t) = exp(-iHt) to the state.
 *
 * @param state Quantum state (modified in place)
 * @param h Sparse Hamiltonian
 * @param time Evolution time
 * @param method Evolution method
 * @return QS_SUCCESS or error code
 */
qs_error_t mbl_time_evolve(quantum_state_t *state,
                            const sparse_hamiltonian_t *h,
                            double time,
                            evolution_method_t method);

/**
 * @brief Time evolve using exact diagonalization
 *
 * U(t) = Σₙ |n⟩⟨n| exp(-iEₙt)
 *
 * @param state Quantum state
 * @param h Sparse Hamiltonian (must be diagonalized)
 * @param time Evolution time
 * @return QS_SUCCESS or error code
 */
qs_error_t mbl_evolve_exact(quantum_state_t *state,
                             const sparse_hamiltonian_t *h,
                             double time);

/**
 * @brief Time evolve using Krylov subspace
 *
 * Builds orthonormal Krylov basis {|ψ⟩, H|ψ⟩, H²|ψ⟩, ...}
 * and applies time evolution in this reduced space.
 *
 * @param state Quantum state
 * @param h Sparse Hamiltonian
 * @param time Evolution time
 * @param krylov_dim Dimension of Krylov subspace
 * @return QS_SUCCESS or error code
 */
qs_error_t mbl_evolve_krylov(quantum_state_t *state,
                              const sparse_hamiltonian_t *h,
                              double time, uint32_t krylov_dim);

// ============================================================================
// INITIAL STATES
// ============================================================================

/**
 * @brief Prepare Néel state |↑↓↑↓...⟩
 *
 * Standard initial state for MBL imbalance measurements.
 *
 * @param state Quantum state (initialized with correct num_qubits)
 * @return QS_SUCCESS or error code
 */
qs_error_t prepare_neel_state(quantum_state_t *state);

/**
 * @brief Prepare domain wall state |↑↑...↓↓⟩
 *
 * Half up, half down. Used for studying transport.
 *
 * @param state Quantum state
 * @return QS_SUCCESS or error code
 */
qs_error_t prepare_domain_wall_state(quantum_state_t *state);

/**
 * @brief Prepare random product state
 *
 * Each spin randomly |↑⟩ or |↓⟩.
 *
 * @param state Quantum state
 * @param seed Random seed
 * @return QS_SUCCESS or error code
 */
qs_error_t prepare_random_product_state(quantum_state_t *state, uint64_t seed);

// ============================================================================
// PHASE DIAGRAM
// ============================================================================

/**
 * @brief Phase diagram point
 */
typedef struct {
    double disorder_strength;   // W value
    double mean_r;              // Mean level spacing ratio
    double mean_imbalance;      // Long-time imbalance
    double mean_entropy;        // Saturation entropy
    int phase;                  // 0=thermal, 1=MBL, -1=critical
} phase_point_t;

/**
 * @brief MBL phase diagram
 */
typedef struct {
    phase_point_t *points;      // Data points
    uint32_t num_points;        // Number of points
    double critical_disorder;   // Estimated Wc
    double critical_exponent;   // Estimated ν
} phase_diagram_t;

/**
 * @brief Scan disorder strength to map phase transition
 *
 * Computes order parameters across a range of disorder strengths
 * to locate the MBL-thermal transition.
 *
 * @param num_sites System size L
 * @param J Exchange coupling
 * @param delta Anisotropy
 * @param W_min Minimum disorder
 * @param W_max Maximum disorder
 * @param num_W_points Number of disorder values to scan
 * @param num_realizations Disorder realizations to average over
 * @param periodic_bc Periodic boundaries
 * @return Phase diagram structure
 */
phase_diagram_t *scan_phase_diagram(uint32_t num_sites,
                                     double J, double delta,
                                     double W_min, double W_max,
                                     uint32_t num_W_points,
                                     uint32_t num_realizations,
                                     bool periodic_bc);

/**
 * @brief Free phase diagram
 */
void phase_diagram_free(phase_diagram_t *pd);

/**
 * @brief Estimate critical disorder strength
 *
 * Finds Wc where ⟨r⟩ = 0.47 (midpoint between Poisson and GOE).
 *
 * @param pd Phase diagram data
 * @return Critical disorder Wc
 */
double estimate_critical_disorder(const phase_diagram_t *pd);

// ============================================================================
// OBSERVABLES
// ============================================================================

/**
 * @brief Compute expectation value ⟨Sᶻᵢ⟩
 *
 * @param state Quantum state
 * @param site Site index
 * @return ⟨Sᶻᵢ⟩ in range [-0.5, 0.5]
 */
double expectation_sz(const quantum_state_t *state, uint32_t site);

/**
 * @brief Compute total magnetization ⟨Sᶻ_total⟩
 *
 * @param state Quantum state
 * @return Total ⟨Sᶻ⟩
 */
double expectation_sz_total(const quantum_state_t *state);

/**
 * @brief Compute spin-spin correlation ⟨Sᶻᵢ Sᶻⱼ⟩
 *
 * @param state Quantum state
 * @param site_i First site
 * @param site_j Second site
 * @return ⟨Sᶻᵢ Sᶻⱼ⟩
 */
double correlation_sz_sz(const quantum_state_t *state,
                          uint32_t site_i, uint32_t site_j);

/**
 * @brief Compute connected correlation ⟨Sᶻᵢ Sᶻⱼ⟩ - ⟨Sᶻᵢ⟩⟨Sᶻⱼ⟩
 *
 * @param state Quantum state
 * @param site_i First site
 * @param site_j Second site
 * @return Connected correlation
 */
double correlation_connected(const quantum_state_t *state,
                              uint32_t site_i, uint32_t site_j);

/**
 * @brief Compute energy expectation ⟨H⟩
 *
 * @param state Quantum state
 * @param h Sparse Hamiltonian
 * @return ⟨H⟩
 */
double expectation_energy(const quantum_state_t *state,
                           const sparse_hamiltonian_t *h);

/**
 * @brief Compute energy variance ⟨H²⟩ - ⟨H⟩²
 *
 * Small variance indicates eigenstate. Used to verify
 * that time-evolved states remain close to eigenstates in MBL.
 *
 * @param state Quantum state
 * @param h Sparse Hamiltonian
 * @return Energy variance
 */
double energy_variance(const quantum_state_t *state,
                        const sparse_hamiltonian_t *h);

#endif /* MBL_H */
