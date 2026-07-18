/**
 * @file tdvp.h
 * @brief Time-Dependent Variational Principle (TDVP) for MPS dynamics
 *
 * TDVP is a powerful algorithm for time evolution of MPS states that:
 * - Respects the variational manifold structure
 * - Renormalizes after SVD truncation to maintain unit norm
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
 * MATHEMATICAL UNDERPINNING:
 * ==========================
 * TDVP projects the Schroedinger equation onto the tangent space of
 * the MPS variational manifold; the projected flow respects the
 * manifold exactly, conserves energy for time-independent @f$H@f$,
 * and becomes a symplectic integrator under a symmetric two-site
 * Lie-Trotter split.  Error per step is @f$O(dt^3)@f$ for the
 * two-site integrator used here; the bond dimension @f$\chi@f$ sets
 * the "resolution" with which entanglement growth is tracked.  The
 * Haegeman 2016 paper unifies TDVP with DMRG (imaginary-time TDVP
 * with large @f$dt@f$ converges to the DMRG ground state), which is
 * the theoretical justification for reusing the same MPS / MPO
 * machinery across `dmrg.h` and this file.
 *
 * REFERENCES:
 * ===========
 *  [1] J. Haegeman, J. I. Cirac, T. J. Osborne, I. Pizorn,
 *      H. Verschelde and F. Verstraete, "Time-dependent variational
 *      principle for quantum lattices", Phys. Rev. Lett. 107, 070601
 *      (2011), arXiv:1103.0936.  Introduces TDVP for MPS.
 *  [2] J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken and
 *      F. Verstraete, "Unifying time evolution and optimization with
 *      matrix product states", Phys. Rev. B 94, 165116 (2016),
 *      arXiv:1408.5056.  Two-site / projector variant implemented
 *      here; the "unification" thesis establishes the DMRG link.
 *  [3] U. Schollwoeck, "The density-matrix renormalization group in
 *      the age of matrix product states", Ann. Phys. 326, 96 (2011),
 *      arXiv:1008.3477.  Background for MPS / MPO conventions.
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef TDVP_H
#define TDVP_H
#include "applications/moonlab_api.h"

#include "tn_state.h"
#include "dmrg.h"
#include "tensor.h"
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
 * @brief Adaptive-bond TDVP configuration (since v0.4 / Phase 3B).
 *
 * Replaces the fixed `max_bond_dim` cap with an entropy-feedback PID
 * controller that selects each bond's truncation threshold
 * individually so the post-truncation block entropy meets a target
 * accuracy budget.  Algorithm and validation criteria are documented
 * in `docs/research/adaptive_bond_tdvp.md`.
 *
 * When `enabled == false` (the default produced by
 * `tdvp_config_default`) the legacy fixed-cap path is used and the
 * remaining fields are ignored; this preserves bit-exact behaviour
 * for every v0.3 caller.
 *
 * Reference: arXiv:2604.03960 (entropy-feedback bond control for
 * 2TDVP).
 */
typedef struct {
    bool enabled;                /**< Use PID controller (false = legacy). */
    double target_entropy_error; /**< Target |S - S_chi| budget. */
    double kp;                   /**< Proportional gain. */
    double ki;                   /**< Integral gain (per sweep). */
    double kd;                   /**< Derivative gain (per sweep). */
    uint32_t chi_floor;          /**< Hard lower bound on per-bond chi. */
    uint32_t chi_ceiling;        /**< Hard upper bound on per-bond chi. */
    double alpha;                /**< Entropy-excess -> bond-dim scale. */
} tdvp_adaptive_bond_config_t;

/**
 * @brief Reference-paper PID gains for the adaptive-bond controller.
 *
 * Returns an `enabled = true` configuration with the gains validated
 * in arXiv:2604.03960 for a 24-site Heisenberg chain.  Equivalent to
 * filling the fields by hand with:
 *   - target_entropy_error = eps_S
 *   - kp = 0.5, ki = 0.05, kd = 0.1
 *   - chi_floor = 4, chi_ceiling = 4096
 *   - alpha = 8.0
 *
 * Use this when wiring adaptive bond control into a new TDVP
 * pipeline; tune the gains afterwards if your model has a
 * substantially different entanglement-growth profile.
 */
static inline tdvp_adaptive_bond_config_t
tdvp_adaptive_bond_config_default(double target_entropy_error) {
    return (tdvp_adaptive_bond_config_t){
        .enabled              = true,
        .target_entropy_error = target_entropy_error,
        .kp                   = 0.5,
        .ki                   = 0.05,
        .kd                   = 0.1,
        .chi_floor            = 4,
        .chi_ceiling          = 4096,
        .alpha                = 8.0,
    };
}

/**
 * @brief All-zero, disabled adaptive-bond config.
 *
 * Used inside `tdvp_config_default` to leave the adaptive path off
 * by default so existing v0.3 callers see no behaviour change.
 */
static inline tdvp_adaptive_bond_config_t
tdvp_adaptive_bond_config_disabled(void) {
    return (tdvp_adaptive_bond_config_t){
        .enabled              = false,
        .target_entropy_error = 0.0,
        .kp                   = 0.0,
        .ki                   = 0.0,
        .kd                   = 0.0,
        .chi_floor            = 0,
        .chi_ceiling          = 0,
        .alpha                = 0.0,
    };
}

/**
 * @brief TDVP configuration
 */
typedef struct {
    tdvp_evolution_type_t evolution_type;   /**< Real or imaginary time */
    tdvp_variant_t variant;                 /**< One-site or two-site */
    integrator_type_t integrator;           /**< Time integration method */

    double dt;                      /**< Time step */
    uint32_t max_bond_dim;          /**< Maximum bond dimension (legacy fixed cap) */
    double svd_cutoff;              /**< SVD truncation threshold */

    uint32_t lanczos_max_iter;      /**< Max Lanczos iterations */
    double lanczos_tol;             /**< Lanczos convergence tolerance */

    bool normalize;                 /**< Normalize state after each step */
    bool verbose;                   /**< Print progress information */

    /**
     * Adaptive-bond control (v0.4+).  When `adaptive_bond.enabled`
     * is `true`, `max_bond_dim` and `svd_cutoff` are still honoured
     * as outer-bound safeties but the per-bond truncation threshold
     * is selected by the PID controller.  See
     * `docs/research/adaptive_bond_tdvp.md`.
     */
    tdvp_adaptive_bond_config_t adaptive_bond;
} tdvp_config_t;

/**
 * @brief Default TDVP configuration (legacy fixed-bond path).
 *
 * `adaptive_bond.enabled` is `false`, so callers that built a
 * `tdvp_config_t` via this helper before v0.4 continue to see the
 * exact same behaviour: a fixed `max_bond_dim = 128` cap with
 * `svd_cutoff = 1e-10`.
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
        .verbose = false,
        .adaptive_bond = {
            .enabled              = false,
            .target_entropy_error = 0.0,
            .kp                   = 0.0,
            .ki                   = 0.0,
            .kd                   = 0.0,
            .chi_floor            = 0,
            .chi_ceiling          = 0,
            .alpha                = 0.0,
        },
    };
}

/**
 * @brief Adaptive-bond TDVP configuration (since v0.4).
 *
 * Returns a two-site real-time TDVP config with the entropy-feedback
 * PID controller enabled at the reference-paper gains.  The fixed
 * `max_bond_dim` is retained as an outer-bound safety
 * (`chi_ceiling`).
 *
 * @param target_entropy_error  PID error-signal budget; 1e-3 is a
 *                              sensible default for production runs.
 */
static inline tdvp_config_t tdvp_config_adaptive(double target_entropy_error) {
    tdvp_config_t cfg = tdvp_config_default();
    cfg.adaptive_bond = tdvp_adaptive_bond_config_default(target_entropy_error);
    cfg.max_bond_dim  = cfg.adaptive_bond.chi_ceiling;
    return cfg;
}

// ============================================================================
// TDVP RESULT
// ============================================================================

/**
 * @brief TDVP step result
 *
 * Callers may pass a stack-allocated, zero-initialised
 * `tdvp_result_t` into `tdvp_step`; the engine fills the scalar
 * fields and (only when the adaptive-bond controller is enabled)
 * lazily allocates a heap buffer for `bond_chi_distribution`.  The
 * buffer is reused across calls if its size matches; otherwise it
 * is realloc-ed.  Before discarding the result (or letting it go out
 * of scope) the caller must invoke `tdvp_result_clear` to free the
 * heap allocation.
 */
typedef struct {
    double time;                /**< Current time after step */
    double energy;              /**< Energy ⟨H⟩ */
    double norm;                /**< State norm */
    double truncation_error;    /**< Truncation error in this step */
    uint32_t max_bond_dim;      /**< Maximum bond dimension reached */
    double step_time;           /**< Wall time for this step (seconds) */

    /**
     * Per-bond chi snapshot from the adaptive-bond controller
     * (since v0.4).  Length `n_bonds = num_qubits - 1` when the
     * controller is enabled; otherwise `bond_chi_distribution` is
     * `NULL` and `n_bonds = 0`.  Heap-owned; free via
     * `tdvp_result_clear`.
     */
    uint32_t *bond_chi_distribution;
    uint32_t  n_bonds;
} tdvp_result_t;

/**
 * @brief Free the heap fields of a `tdvp_result_t` and zero the
 *        struct in place.
 *
 * Safe to call on a zero-initialised result (no-op) or repeatedly
 * (subsequent calls are no-ops).
 */
MOONLAB_API void tdvp_result_clear(tdvp_result_t *result);

/**
 * @brief TDVP evolution history (for observables)
 *
 * `bond_chi_history` (since v0.4) is a flat `capacity * n_bonds`
 * buffer storing the per-bond chi snapshot for each recorded step
 * row-major; entry `(step, bond)` lives at index
 * `step * n_bonds + bond`.  Only allocated when the first added
 * result actually carries a non-NULL `bond_chi_distribution`.
 */
typedef struct {
    double *times;              /**< Array of times */
    double *energies;           /**< Array of energies */
    double *norms;              /**< Array of norms */
    double *observables;        /**< Array of measured observables */
    uint32_t num_steps;         /**< Number of recorded steps */
    uint32_t capacity;          /**< Allocated capacity */

    uint32_t *bond_chi_history; /**< Flat capacity * n_bonds buffer, or NULL */
    uint32_t  n_bonds;          /**< Stride of the chi history (0 = unused) */
} tdvp_history_t;

/**
 * @brief Create TDVP history
 */
MOONLAB_API tdvp_history_t *tdvp_history_create(uint32_t initial_capacity);

/**
 * @brief Free TDVP history
 */
MOONLAB_API void tdvp_history_free(tdvp_history_t *hist);

/**
 * @brief Add result to history
 *
 * Records `result->time`, `result->energy`, `result->norm`, and the
 * per-bond chi snapshot from `result->bond_chi_distribution` (lazily
 * allocated on the first result that carries a distribution).  Does
 * not touch `tdvp_history_t::observables`; use
 * ::tdvp_history_add_with_observable for that.
 */
void tdvp_history_add(tdvp_history_t *hist, const tdvp_result_t *result);

/**
 * @brief Add a result to history together with one scalar observable.
 *
 * Equivalent to ::tdvp_history_add but additionally writes
 * @p observable into `tdvp_history_t::observables` at the new step's
 * index.  The `observables` array is lazily allocated on the first
 * call to this function; calls to ::tdvp_history_add on the same
 * history leave the corresponding observable slot at zero.
 */
void tdvp_history_add_with_observable(tdvp_history_t *hist,
                                       const tdvp_result_t *result,
                                       double observable);

// ============================================================================
// TDVP ENGINE
// ============================================================================

/**
 * @brief Opaque per-bond PID state for the adaptive-bond controller.
 *
 * Defined in `tdvp.c`.  The engine owns one slot per inter-site bond
 * when `config.adaptive_bond.enabled` is true and forwards the
 * appropriate slot into the truncation helper on each two-site
 * update; on the legacy path `bond_states` is NULL and the field is
 * never read.
 */
typedef struct tdvp_bond_pid_state tdvp_bond_pid_state_t;

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

    /**
     * Per-bond PID controller state for adaptive-bond TDVP (since
     * v0.4).  Length `num_bond_states = mps->num_qubits - 1` when
     * `config.adaptive_bond.enabled`, NULL on the legacy path.  The
     * engine owns this allocation and frees it in
     * `tdvp_engine_free`.
     */
    tdvp_bond_pid_state_t *bond_states;
    uint32_t num_bond_states;
} tdvp_engine_t;

/**
 * @brief Create TDVP engine
 *
 * @param mps MPS state to evolve (will be modified in place)
 * @param mpo Hamiltonian MPO
 * @param config TDVP configuration
 * @return TDVP engine or NULL on failure
 */
MOONLAB_API tdvp_engine_t *tdvp_engine_create(tn_mps_state_t *mps,
                                   mpo_t *mpo,
                                   const tdvp_config_t *config);

/**
 * @brief Read the current per-bond chi from the adaptive-bond
 *        controller.
 *
 * Returns the target bond dimension the entropy-feedback PID has
 * settled on for the given inter-site bond after the most recent
 * sweep, or `0` if `bond` is out of range, the engine is `NULL`, or
 * the adaptive controller is disabled (in which case no per-bond
 * state is allocated).
 *
 * Primarily intended for tests, instrumentation, and the
 * `bond_chi_distribution` reporting path that the v0.4 result
 * struct will add in a future patch.
 */
MOONLAB_API uint32_t tdvp_bond_chi(const tdvp_engine_t *engine, uint32_t bond);

/**
 * @brief Free TDVP engine
 */
MOONLAB_API void tdvp_engine_free(tdvp_engine_t *engine);

/**
 * @brief Perform one TDVP time step
 *
 * Evolves state by dt using two-site TDVP.
 *
 * @param engine TDVP engine
 * @param result Output result (can be NULL)
 * @return 0 on success
 */
MOONLAB_API int tdvp_step(tdvp_engine_t *engine, tdvp_result_t *result);

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
MOONLAB_API int tdvp_evolve_to(tdvp_engine_t *engine,
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
// OBSERVABLES DURING EVOLUTION
// ============================================================================

/**
 * @brief Observable-side-effect callback type.
 *
 * Called after each TDVP step.  The callback is free to compute any
 * expectation value or write any side effect from `mps`; it does not
 * report a value back to the engine.  Use
 * ::observable_value_callback_t with ::tdvp_evolve_to_with_observable
 * if you want the engine to record the measured value into history.
 */
typedef void (*observable_callback_t)(const tn_mps_state_t *mps,
                                       double time,
                                       void *user_data);

/**
 * @brief Observable-returning callback type.
 *
 * Same shape as ::observable_callback_t, but returns a `double` that
 * ::tdvp_evolve_to_with_observable writes into the history's
 * `observables` array at the corresponding step index.  Typical
 * implementations return an expectation value of some operator, the
 * MPS norm, an entanglement entropy, or similar scalar diagnostic.
 */
typedef double (*observable_value_callback_t)(const tn_mps_state_t *mps,
                                               double time,
                                               void *user_data);

/**
 * @brief Evolve with observable measurement (side-effect callback).
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

/**
 * @brief Evolve to a target time, recording each step + a scalar
 *        observable into history.
 *
 * Mirrors ::tdvp_evolve_to but additionally invokes @p observable_fn
 * after every step (when non-NULL) and stores its return value via
 * ::tdvp_history_add_with_observable.  When @p observable_fn is NULL
 * the call degrades to ::tdvp_evolve_to.
 *
 * @param engine        TDVP engine.
 * @param target_time   Target time (real or imag, sign-agnostic).
 * @param history       Output history; may not be NULL.
 * @param observable_fn Per-step observable callback, or NULL.
 * @param user_data     User data passed verbatim to @p observable_fn.
 * @return 0 on success, negative on integrator failure.
 */
MOONLAB_API int tdvp_evolve_to_with_observable(tdvp_engine_t *engine,
                                    double target_time,
                                    tdvp_history_t *history,
                                    observable_value_callback_t observable_fn,
                                    void *user_data);

#ifdef __cplusplus
}
#endif

#endif /* TDVP_H */
