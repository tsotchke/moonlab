/**
 * @file dmrg.h
 * @brief Density Matrix Renormalization Group for 1D ground states.
 *
 * OVERVIEW
 * --------
 * The Density Matrix Renormalization Group (DMRG), introduced by
 * White in 1992, is a variational algorithm that minimises the
 * energy expectation
 * @f$\langle\psi|H|\psi\rangle/\langle\psi|\psi\rangle@f$ of a 1D
 * quantum Hamiltonian @f$H@f$ over the manifold of matrix-product
 * states (MPS) with bounded bond dimension @f$\chi@f$.  Its modern
 * formulation, clarified after the advent of MPS / MPO notation (see
 * Schollwöck's 2011 review), proceeds by sweeping left-to-right and
 * back, at each step solving a local eigenproblem for the effective
 * Hamiltonian acting on one or two sites, then re-decomposing the
 * updated tensor via SVD.  Because every step minimises the same
 * energy functional, the algorithm is variationally monotonic; the
 * MPS ansatz itself caps the error by @f$O(e^{-c L / \xi})@f$ on
 * gapped systems where the entanglement obeys an area law
 * (Hastings).
 *
 * The implementation here is 2-site DMRG with subspace expansion
 * (adding a small number of additional columns to the left
 * environment at each step so the local problem can explore bond
 * directions that the current @f$\chi@f$ cannot represent) and
 * noise decay (a progressively shrinking random kick to the effective
 * Hamiltonian that prevents lock-in to local minima during the first
 * sweeps).  Both ingredients are standard in modern DMRG codes; see
 * Schollwöck §6 for the full treatment.
 *
 * When to use DMRG vs imaginary-time evolution: DMRG converges in a
 * handful of sweeps whose cost is @f$O(L\,\chi^3\,d^2)@f$ per sweep,
 * while ITE with Trotter steps needs @f$O(1/(\Delta\tau\, \Delta))@f$
 * steps where @f$\Delta@f$ is the spectral gap.  DMRG is the method
 * of choice for gapped 1D ground states; TDVP (see `tdvp.h`) is the
 * complementary method for real-time evolution at fixed bond
 * dimension.
 *
 * REFERENCES
 * ----------
 *  - S. R. White, "Density matrix formulation for quantum
 *    renormalization groups", Phys. Rev. Lett. 69, 2863 (1992),
 *    doi:10.1103/PhysRevLett.69.2863.  Original DMRG paper.
 *  - U. Schollwoeck, "The density-matrix renormalization group in the
 *    age of matrix product states", Ann. Phys. 326, 96 (2011),
 *    arXiv:1008.3477.  Canonical modern treatment; our MPS / MPO
 *    conventions and sweep scheme follow this reference.
 *  - G. Vidal, "Efficient classical simulation of slightly entangled
 *    quantum computations", Phys. Rev. Lett. 91, 147902 (2003),
 *    arXiv:quant-ph/0301063.  MPS as a faithful classical simulation
 *    substrate.
 *
 * @stability evolving
 * @since v0.1.2
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef DMRG_H
#define DMRG_H

#include "tn_state.h"
#include "tensor.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CONFIGURATION
// ============================================================================

/**
 * @brief DMRG algorithm configuration
 */
typedef struct {
    uint32_t max_bond_dim;      /**< Maximum bond dimension */
    double svd_cutoff;          /**< SVD truncation threshold */
    uint32_t max_sweeps;        /**< Maximum number of sweeps */
    double energy_tol;          /**< Energy convergence tolerance */
    uint32_t lanczos_max_iter;  /**< Max Lanczos iterations per site */
    double lanczos_tol;         /**< Lanczos convergence tolerance */
    bool verbose;               /**< Print progress information */
    bool two_site;              /**< Use two-site DMRG (more robust) */

    // Noise perturbation (injects noise into the SVD input to encourage
    // bond-dimension growth and escape local minima). Not true subspace
    // expansion (which would add vectors from the discarded singular space).
    double noise_strength;      /**< Noise strength (0 = disabled) */
    double noise_decay;         /**< Decay factor per sweep (0.5 = halve each sweep) */

    // Density matrix perturbation
    double dm_perturbation;     /**< Perturbation strength for density matrix (0 = disabled) */

    // Warmup parameters
    uint32_t warmup_sweeps;     /**< Number of warmup sweeps with larger noise */
    double warmup_noise;        /**< Noise strength during warmup */
    uint32_t warmup_bond_dim;   /**< Initial bond dimension for warmup */
} dmrg_config_t;

/**
 * @brief Default DMRG configuration
 */
static inline dmrg_config_t dmrg_config_default(void) {
    return (dmrg_config_t){
        .max_bond_dim = 128,
        .svd_cutoff = 1e-10,
        .max_sweeps = 20,
        .energy_tol = 1e-8,
        .lanczos_max_iter = 100,
        .lanczos_tol = 1e-12,
        .verbose = false,
        .two_site = true,
        // Noise-perturbation defaults
        .noise_strength = 1e-4,
        .noise_decay = 0.5,
        // Density matrix perturbation
        .dm_perturbation = 1e-6,
        // Warmup defaults
        .warmup_sweeps = 3,
        .warmup_noise = 1e-3,
        .warmup_bond_dim = 16
    };
}

/**
 * @brief Sweep direction for canonical form handling
 */
typedef enum {
    DMRG_SWEEP_LEFT_TO_RIGHT,   /**< L->R: absorb S into right tensor (left-canonical) */
    DMRG_SWEEP_RIGHT_TO_LEFT    /**< R->L: absorb S into left tensor (right-canonical) */
} dmrg_sweep_direction_t;

/**
 * @brief DMRG result structure
 */
typedef struct {
    double ground_energy;        /**< Ground state energy */
    double energy_variance;      /**< Energy variance (quality measure) */
    uint32_t num_sweeps;         /**< Number of sweeps performed */
    double *sweep_energies;      /**< Energy after each sweep */
    double total_time;           /**< Total computation time (seconds) */
    bool converged;              /**< Whether energy converged */
    double truncation_error;     /**< Total accumulated truncation error */
} dmrg_result_t;

/**
 * @brief Free DMRG result
 */
void dmrg_result_free(dmrg_result_t *result);

// ============================================================================
// HAMILTONIAN DEFINITION
// ============================================================================

/**
 * @brief MPO (Matrix Product Operator) tensor for Hamiltonian
 *
 * H = sum of MPO tensors: H[i]_{b_l, b_r}^{s, s'}
 * where b_l, b_r are bond indices and s, s' are physical indices
 */
typedef struct {
    tensor_t *W;                 /**< MPO tensor [b_l][s][s'][b_r] */
    uint32_t bond_dim_left;      /**< Left MPO bond dimension */
    uint32_t bond_dim_right;     /**< Right MPO bond dimension */
    uint32_t phys_dim;           /**< Physical dimension (usually 2) */
} mpo_tensor_t;

/**
 * @brief MPO representation of Hamiltonian
 */
typedef struct {
    mpo_tensor_t *tensors;       /**< Array of MPO tensors */
    uint32_t num_sites;          /**< Number of sites */
    uint32_t max_mpo_bond;       /**< Maximum MPO bond dimension */
} mpo_t;

/**
 * @brief Create MPO for transverse field Ising model
 *
 * H = -J sum_i Z_i Z_{i+1} - h sum_i X_i
 *
 * @param num_sites Number of lattice sites
 * @param J Coupling strength (ferromagnetic if J > 0)
 * @param h Transverse field strength
 * @return MPO representation or NULL on failure
 */
mpo_t *mpo_tfim_create(uint32_t num_sites, double J, double h);

/**
 * @brief Create MPO for Heisenberg XXZ model
 *
 * H = J sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Delta Z_i Z_{i+1}) - h sum_i Z_i
 *
 * @param num_sites Number of lattice sites
 * @param J Exchange coupling
 * @param Delta Anisotropy (Delta=1 for isotropic Heisenberg)
 * @param h Magnetic field
 * @return MPO representation or NULL on failure
 */
mpo_t *mpo_heisenberg_create(uint32_t num_sites, double J, double Delta, double h);

/**
 * @brief Create MPO for Kitaev chain / XY model
 *
 * H = -J_XX sum_i X_i X_{i+1} - J_YY sum_i Y_i Y_{i+1} - h sum_i Z_i
 *
 * At the Kitaev sweet spot (Delta = t):
 *   J_XX = (t + Delta)/2 = t
 *   J_YY = (t - Delta)/2 = 0
 *   h = mu/2
 *
 * @param num_sites Number of lattice sites
 * @param J_XX XX coupling strength
 * @param J_YY YY coupling strength
 * @param h Transverse Z-field strength
 * @return MPO representation or NULL on failure
 */
mpo_t *mpo_kitaev_create(uint32_t num_sites, double J_XX, double J_YY, double h);

/**
 * @brief Create MPO for 2D magnetic skyrmion Hamiltonian (snake ordering)
 *
 * H = -J Σ S_i·S_j + D Σ d_ij·(S_i × S_j) - B Σ S_i^z - K Σ (S_i^z)²
 *     (exchange)     (DMI)                   (Zeeman)   (anisotropy)
 *
 * The DMI term is what stabilizes skyrmions. For interfacial DMI,
 * D_ij is perpendicular to the bond direction.
 *
 * NOTE: This creates an effective 1D MPO for a 2D lattice using snake ordering.
 * The lattice_2d module provides the coordinate mapping.
 *
 * @param num_sites Total number of sites (Lx * Ly)
 * @param Lx Width of 2D lattice
 * @param Ly Height of 2D lattice
 * @param J Heisenberg exchange (J > 0 ferromagnetic)
 * @param D Dzyaloshinskii-Moriya strength
 * @param B External magnetic field (Zeeman)
 * @param K Easy-axis anisotropy
 * @return MPO representation or NULL on failure
 */
mpo_t *mpo_skyrmion_create(uint32_t num_sites, uint32_t Lx, uint32_t Ly,
                           double J, double D, double B, double K);

/**
 * @brief Create MPO for custom 2D Hamiltonian with neighbor list
 *
 * Allows arbitrary 2D lattice geometry by specifying neighbor pairs
 * and their bond vectors explicitly.
 *
 * @param num_sites Number of sites
 * @param num_bonds Number of bonds (neighbor pairs)
 * @param bonds Array of [site_i, site_j] pairs
 * @param bond_vectors Array of bond direction vectors
 * @param J Exchange coupling
 * @param D DMI strength
 * @param B Zeeman field
 * @param K Anisotropy
 * @return MPO representation or NULL on failure
 */
mpo_t *mpo_2d_heisenberg_dmi_create(uint32_t num_sites,
                                     uint32_t num_bonds,
                                     const uint32_t (*bonds)[2],
                                     const double (*bond_vectors)[3],
                                     double J, double D, double B, double K);

/**
 * @brief Free MPO
 */
void mpo_free(mpo_t *mpo);

// ============================================================================
// LANCZOS EIGENSOLVER
// ============================================================================

/**
 * @brief Lanczos result for ground state
 */
typedef struct {
    double eigenvalue;           /**< Ground state eigenvalue */
    tensor_t *eigenvector;       /**< Ground state eigenvector */
    uint32_t num_iterations;     /**< Iterations used */
    bool converged;              /**< Whether converged */
} lanczos_result_t;

/**
 * @brief Free Lanczos result
 */
void lanczos_result_free(lanczos_result_t *result);

/**
 * @brief Effective Hamiltonian for DMRG local optimization
 *
 * For two-site DMRG at sites (i, i+1):
 * H_eff = L[i-1] @ W[i] @ W[i+1] @ R[i+2]
 *
 * The effective Hamiltonian acts on the theta tensor [chi_l][d][d][chi_r]
 */
typedef struct {
    tensor_t *L;                 /**< Left environment [chi_l][b_l][chi_l] */
    tensor_t *R;                 /**< Right environment [chi_r][b_r][chi_r] */
    mpo_tensor_t *W_left;        /**< Left MPO tensor */
    mpo_tensor_t *W_right;       /**< Right MPO tensor (NULL for 1-site) */
    uint32_t chi_l;              /**< Left MPS bond dimension */
    uint32_t chi_r;              /**< Right MPS bond dimension */
    uint32_t phys_dim;           /**< Physical dimension */
    bool two_site;               /**< Two-site or one-site */
} effective_hamiltonian_t;

/**
 * @brief Persistent scratch buffers used by
 *        @c effective_hamiltonian_apply_ws to avoid per-Lanczos-step
 *        calloc/free of the (chi^2 * bond^2)-sized intermediates.
 *
 * Lanczos typically requests 30-100 H_eff applications per bond, so
 * re-allocating the three interior tensors each time is an O(N_L)
 * drag on every DMRG iteration.  Threading a workspace through means
 * the three buffers allocate once per bond (or once per sweep, if
 * the caller hoists further), and subsequent applications grow the
 * buffer only when the bond dimension actually changes.  Structure
 * is zero-initialised on @c workspace_init so the first call
 * allocates lazily.
 */
typedef struct {
    double complex *temp1;       /**< x contracted with R */
    size_t          temp1_cap;
    double complex *temp2;       /**< + W2 */
    size_t          temp2_cap;
    double complex *temp3;       /**< + W1 */
    size_t          temp3_cap;
} effective_hamiltonian_workspace_t;

/**
 * @brief Zero-initialise a workspace.  Cheap; no allocation.
 */
void effective_hamiltonian_workspace_init(effective_hamiltonian_workspace_t *ws);

/**
 * @brief Release all buffers owned by @p ws and zero the struct.
 */
void effective_hamiltonian_workspace_free(effective_hamiltonian_workspace_t *ws);

/**
 * @brief Apply effective Hamiltonian to vector
 *
 * y = H_eff @ x
 *
 * This is the core operation for Lanczos iteration.
 *
 * @param H_eff Effective Hamiltonian
 * @param x Input vector (flattened theta tensor)
 * @param y Output vector (same shape as x)
 * @return 0 on success
 */
int effective_hamiltonian_apply(const effective_hamiltonian_t *H_eff,
                                const tensor_t *x,
                                tensor_t *y);

/**
 * @brief Like @c effective_hamiltonian_apply but reuses scratch
 *        buffers held in @p ws rather than calloc-ing on every call.
 *
 * Semantics are identical when @p ws is NULL (falls back to per-call
 * allocation).  Otherwise the workspace is grown in place to fit the
 * current bond / physical dimensions and the intermediates live
 * across the full Lanczos sequence.  Safe to share a workspace
 * across every bond in a sweep; not thread-safe.
 */
int effective_hamiltonian_apply_ws(const effective_hamiltonian_t *H_eff,
                                   const tensor_t *x,
                                   tensor_t *y,
                                   effective_hamiltonian_workspace_t *ws);

/**
 * @brief Solve ground state of effective Hamiltonian using Lanczos
 *
 * @param H_eff Effective Hamiltonian
 * @param initial_guess Initial vector (can be NULL for random)
 * @param max_iter Maximum Lanczos iterations
 * @param tol Convergence tolerance
 * @return Lanczos result or NULL on failure
 */
lanczos_result_t *lanczos_ground_state(const effective_hamiltonian_t *H_eff,
                                        const tensor_t *initial_guess,
                                        uint32_t max_iter,
                                        double tol);

// ============================================================================
// DMRG ALGORITHM
// ============================================================================

/**
 * @brief DMRG environment tensors
 *
 * Stores the contracted left and right environments for efficient sweeping.
 */
typedef struct {
    tensor_t **L;                /**< Left environments L[i] for i = 0..n-1 */
    tensor_t **R;                /**< Right environments R[i] for i = 0..n-1 */
    uint32_t num_sites;          /**< Number of sites */
} dmrg_environments_t;

/**
 * @brief Create DMRG environments
 */
dmrg_environments_t *dmrg_environments_create(uint32_t num_sites);

/**
 * @brief Free DMRG environments
 */
void dmrg_environments_free(dmrg_environments_t *env);

/**
 * @brief Initialize right environments from right to left
 */
int dmrg_init_right_environments(dmrg_environments_t *env,
                                  const tn_mps_state_t *mps,
                                  const mpo_t *mpo);

/**
 * @brief Initialize all left environments from MPS and MPO
 *
 * Builds L[0] (left boundary) and L[i] for i=1..n-1
 * by contracting from left to right.
 */
int dmrg_init_left_environments(dmrg_environments_t *env,
                                 const tn_mps_state_t *mps,
                                 const mpo_t *mpo);

/**
 * @brief Update left environment after optimizing site i
 */
int dmrg_update_left_environment(dmrg_environments_t *env,
                                  const tn_mps_state_t *mps,
                                  const mpo_t *mpo,
                                  uint32_t site);

/**
 * @brief Update right environment after optimizing site i
 */
int dmrg_update_right_environment(dmrg_environments_t *env,
                                   const tn_mps_state_t *mps,
                                   const mpo_t *mpo,
                                   uint32_t site);

/**
 * @brief Run DMRG algorithm to find ground state
 *
 * This is the main entry point for ground state preparation.
 *
 * @param mps Initial MPS state (will be modified to ground state)
 * @param mpo Hamiltonian as MPO
 * @param config DMRG configuration
 * @return DMRG result or NULL on failure
 */
dmrg_result_t *dmrg_ground_state(tn_mps_state_t *mps,
                                  const mpo_t *mpo,
                                  const dmrg_config_t *config);

/**
 * @brief Perform one DMRG sweep (left-to-right then right-to-left)
 *
 * @param mps MPS state (modified in place)
 * @param mpo Hamiltonian
 * @param env DMRG environments
 * @param config Configuration
 * @param energy Output: energy after sweep
 * @return 0 on success
 */
int dmrg_sweep(tn_mps_state_t *mps,
               const mpo_t *mpo,
               dmrg_environments_t *env,
               const dmrg_config_t *config,
               double *energy);

/**
 * @brief Optimize two sites and update MPS
 *
 * Core operation for two-site DMRG.
 *
 * @param mps MPS state
 * @param mpo Hamiltonian
 * @param env Environments
 * @param site Left site of the pair
 * @param direction Sweep direction for canonical form
 * @param config Configuration
 * @param energy Output: local energy
 * @return 0 on success
 */
int dmrg_optimize_two_site(tn_mps_state_t *mps,
                            const mpo_t *mpo,
                            dmrg_environments_t *env,
                            uint32_t site,
                            dmrg_sweep_direction_t direction,
                            const dmrg_config_t *config,
                            double *energy);

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/**
 * @brief Find ground state of transverse field Ising model
 *
 * Convenience function that creates MPO and runs DMRG.
 *
 * @param num_sites Number of lattice sites
 * @param g Transverse field ratio h/J (critical point at g=1)
 * @param config DMRG configuration (NULL for defaults)
 * @param result Output: DMRG result
 * @return Ground state MPS or NULL on failure
 */
tn_mps_state_t *dmrg_tfim_ground_state(uint32_t num_sites,
                                        double g,
                                        const dmrg_config_t *config,
                                        dmrg_result_t **result);

/**
 * @brief Compute energy of MPS state with respect to MPO Hamiltonian
 *
 * E = <psi|H|psi> / <psi|psi>
 *
 * @param mps MPS state
 * @param mpo Hamiltonian
 * @return Energy value
 */
double dmrg_compute_energy(const tn_mps_state_t *mps, const mpo_t *mpo);

/**
 * @brief Compute energy variance
 *
 * Var(E) = <H^2> - <H>^2
 *
 * Small variance indicates good approximation to eigenstate.
 *
 * @param mps MPS state
 * @param mpo Hamiltonian
 * @return Energy variance
 */
double dmrg_energy_variance(const tn_mps_state_t *mps, const mpo_t *mpo);

#ifdef __cplusplus
}
#endif

#endif /* DMRG_H */
