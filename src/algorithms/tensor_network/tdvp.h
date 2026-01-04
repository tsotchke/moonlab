/**
 * @file tdvp.h
 * @brief Time-Dependent Variational Principle (TDVP) for MPS dynamics
 *
 * TDVP is a powerful algorithm for time evolution of MPS states that:
 * - Respects the variational manifold structure
 * - Preserves unitarity (norm conservation)
 * - Works for both real and imaginary time evolution
 * - Efficiently handles long-range interactions via MPO
 *
 * ALGORITHM OVERVIEW:
 * ==================
 * Two-site TDVP evolves |ψ(t)⟩ → |ψ(t+dt)⟩ by:
 *
 * 1. Sweep left-to-right:
 *    - Form effective Hamiltonian H_eff for sites (i, i+1)
 *    - Evolve two-site tensor: θ → exp(-iH_eff*dt/2) θ
 *    - SVD split and update environments
 *
 * 2. Sweep right-to-left:
 *    - Same but with remaining dt/2
 *
 * This is a second-order integrator with error O(dt³).
 *
 * APPLICATIONS:
 * =============
 * - Skyrmion dynamics under spin-transfer torque
 * - Quench dynamics in spin systems
 * - Braiding operations for topological qubits
 * - Real-time correlation functions
 *
 * REFERENCES:
 * ===========
 * [1] J. Haegeman et al., "Time-dependent variational principle for quantum
 *     lattices", Phys. Rev. Lett. 107, 070601 (2011)
 *
 * [2] J. Haegeman et al., "Unifying time evolution and optimization with
 *     matrix product states", Phys. Rev. B 94, 165116 (2016)
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef TDVP_H
#define TDVP_H

#include "tn_state.h"
#include "dmrg.h"
#include "tensor.h"
#include "lattice_2d.h"
#include <stdint.h>
#include <stdbool.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CONFIGURATION
// ============================================================================

/**
 * @brief TDVP evolution type
 */
typedef enum {
    TDVP_REAL_TIME,     /**< Real time: exp(-iHt) */
    TDVP_IMAGINARY_TIME /**< Imaginary time: exp(-Ht) for ground state */
} tdvp_evolution_type_t;

/**
 * @brief TDVP algorithm variant
 */
typedef enum {
    TDVP_ONE_SITE,      /**< One-site TDVP (fixed bond dim, faster) */
    TDVP_TWO_SITE       /**< Two-site TDVP (adaptive bond dim, more accurate) */
} tdvp_variant_t;

/**
 * @brief Time integrator type
 */
typedef enum {
    INTEGRATOR_LANCZOS,     /**< Lanczos matrix exponential */
    INTEGRATOR_RUNGE_KUTTA, /**< 4th order Runge-Kutta */
    INTEGRATOR_EXPOKIT      /**< Krylov subspace method */
} integrator_type_t;

/**
 * @brief TDVP configuration
 */
typedef struct {
    tdvp_evolution_type_t evolution_type;   /**< Real or imaginary time */
    tdvp_variant_t variant;                 /**< One-site or two-site */
    integrator_type_t integrator;           /**< Time integration method */

    double dt;                      /**< Time step */
    uint32_t max_bond_dim;          /**< Maximum bond dimension */
    double svd_cutoff;              /**< SVD truncation threshold */

    uint32_t lanczos_max_iter;      /**< Max Lanczos iterations */
    double lanczos_tol;             /**< Lanczos convergence tolerance */

    bool normalize;                 /**< Normalize state after each step */
    bool verbose;                   /**< Print progress information */
} tdvp_config_t;

/**
 * @brief Default TDVP configuration
 */
static inline tdvp_config_t tdvp_config_default(void) {
    return (tdvp_config_t){
        .evolution_type = TDVP_REAL_TIME,
        .variant = TDVP_TWO_SITE,
        .integrator = INTEGRATOR_LANCZOS,
        .dt = 0.01,
        .max_bond_dim = 128,
        .svd_cutoff = 1e-10,
        .lanczos_max_iter = 50,
        .lanczos_tol = 1e-12,
        .normalize = true,
        .verbose = false
    };
}

// ============================================================================
// TDVP RESULT
// ============================================================================

/**
 * @brief TDVP step result
 */
typedef struct {
    double time;                /**< Current time after step */
    double energy;              /**< Energy ⟨H⟩ */
    double norm;                /**< State norm */
    double truncation_error;    /**< Truncation error in this step */
    uint32_t max_bond_dim;      /**< Maximum bond dimension reached */
    double step_time;           /**< Wall time for this step (seconds) */
} tdvp_result_t;

/**
 * @brief TDVP evolution history (for observables)
 */
typedef struct {
    double *times;              /**< Array of times */
    double *energies;           /**< Array of energies */
    double *norms;              /**< Array of norms */
    double *observables;        /**< Array of measured observables */
    uint32_t num_steps;         /**< Number of recorded steps */
    uint32_t capacity;          /**< Allocated capacity */
} tdvp_history_t;

/**
 * @brief Create TDVP history
 */
tdvp_history_t *tdvp_history_create(uint32_t initial_capacity);

/**
 * @brief Free TDVP history
 */
void tdvp_history_free(tdvp_history_t *hist);

/**
 * @brief Add result to history
 */
void tdvp_history_add(tdvp_history_t *hist, const tdvp_result_t *result);

// ============================================================================
// TDVP ENGINE
// ============================================================================

/**
 * @brief TDVP evolution engine
 *
 * Maintains state and environments for efficient multi-step evolution.
 */
typedef struct {
    tn_mps_state_t *mps;            /**< MPS state being evolved */
    mpo_t *mpo;                     /**< Hamiltonian MPO */
    dmrg_environments_t *env;       /**< Left/right environments */
    tdvp_config_t config;           /**< Configuration */
    double current_time;            /**< Current evolution time */
} tdvp_engine_t;

/**
 * @brief Create TDVP engine
 *
 * @param mps MPS state to evolve (will be modified in place)
 * @param mpo Hamiltonian MPO
 * @param config TDVP configuration
 * @return TDVP engine or NULL on failure
 */
tdvp_engine_t *tdvp_engine_create(tn_mps_state_t *mps,
                                   mpo_t *mpo,
                                   const tdvp_config_t *config);

/**
 * @brief Free TDVP engine
 */
void tdvp_engine_free(tdvp_engine_t *engine);

/**
 * @brief Perform one TDVP time step
 *
 * Evolves state by dt using two-site TDVP.
 *
 * @param engine TDVP engine
 * @param result Output result (can be NULL)
 * @return 0 on success
 */
int tdvp_step(tdvp_engine_t *engine, tdvp_result_t *result);

/**
 * @brief Evolve state to target time
 *
 * Performs multiple TDVP steps until t >= target_time.
 *
 * @param engine TDVP engine
 * @param target_time Target evolution time
 * @param history Optional history to record (can be NULL)
 * @return 0 on success
 */
int tdvp_evolve_to(tdvp_engine_t *engine,
                    double target_time,
                    tdvp_history_t *history);

/**
 * @brief Set new time step
 */
void tdvp_set_dt(tdvp_engine_t *engine, double dt);

/**
 * @brief Get current time
 */
double tdvp_get_time(const tdvp_engine_t *engine);

// ============================================================================
// SINGLE-STEP EVOLUTION (STATELESS)
// ============================================================================

/**
 * @brief Perform single TDVP step (stateless version)
 *
 * For simple use cases where engine management is not needed.
 *
 * @param mps MPS state (modified in place)
 * @param mpo Hamiltonian
 * @param dt Time step
 * @param config Configuration
 * @param energy Output: energy after step
 * @return 0 on success
 */
int tdvp_single_step(tn_mps_state_t *mps,
                      const mpo_t *mpo,
                      double dt,
                      const tdvp_config_t *config,
                      double *energy);

// ============================================================================
// MATRIX EXPONENTIAL (LANCZOS)
// ============================================================================

/**
 * @brief Apply matrix exponential to vector using Lanczos
 *
 * Computes y = exp(alpha * H) @ x using Lanczos iteration.
 *
 * For real time evolution: alpha = -i*dt
 * For imaginary time: alpha = -dt
 *
 * @param H_eff Effective Hamiltonian
 * @param x Input vector
 * @param alpha Exponent coefficient
 * @param max_iter Maximum Lanczos iterations
 * @param tol Convergence tolerance
 * @param y Output vector
 * @return 0 on success
 */
int lanczos_expm(const effective_hamiltonian_t *H_eff,
                  const tensor_t *x,
                  double complex alpha,
                  uint32_t max_iter,
                  double tol,
                  tensor_t *y);

// ============================================================================
// SPIN DYNAMICS (SKYRMION SPECIFIC)
// ============================================================================

/**
 * @brief Spin-transfer torque parameters
 *
 * For current-driven skyrmion motion.
 */
typedef struct {
    double jx;      /**< Current density x-component */
    double jy;      /**< Current density y-component */
    double beta;    /**< Non-adiabaticity parameter */
    double alpha;   /**< Gilbert damping */
} stt_params_t;

/**
 * @brief Create MPO for spin-transfer torque Hamiltonian
 *
 * H_STT = sum_i j · ∇S_i (adiabatic) + β j · (S_i × ∇S_i) (non-adiabatic)
 *
 * This is added to the base Hamiltonian to drive skyrmion motion.
 *
 * @param lat 2D lattice
 * @param stt STT parameters
 * @return MPO for STT Hamiltonian
 */
mpo_t *mpo_stt_create(const lattice_2d_t *lat, const stt_params_t *stt);

/**
 * @brief Evolve skyrmion under current drive
 *
 * Combines base Hamiltonian with STT for current-driven dynamics.
 *
 * @param mps MPS state
 * @param mpo_base Base Hamiltonian (exchange + DMI + ...)
 * @param stt STT parameters
 * @param config TDVP configuration
 * @param total_time Total evolution time
 * @param history Output history
 * @return 0 on success
 */
int tdvp_evolve_with_stt(tn_mps_state_t *mps,
                          const mpo_t *mpo_base,
                          const stt_params_t *stt,
                          const tdvp_config_t *config,
                          double total_time,
                          tdvp_history_t *history);

// ============================================================================
// OBSERVABLES DURING EVOLUTION
// ============================================================================

/**
 * @brief Observable callback type
 *
 * Called after each TDVP step to compute observables.
 */
typedef void (*observable_callback_t)(const tn_mps_state_t *mps,
                                       double time,
                                       void *user_data);

/**
 * @brief Evolve with observable measurement
 *
 * @param engine TDVP engine
 * @param target_time Target time
 * @param callback Observable callback
 * @param user_data User data for callback
 * @param measure_interval Steps between measurements
 * @return 0 on success
 */
int tdvp_evolve_with_observables(tdvp_engine_t *engine,
                                  double target_time,
                                  observable_callback_t callback,
                                  void *user_data,
                                  uint32_t measure_interval);

#ifdef __cplusplus
}
#endif

#endif /* TDVP_H */
