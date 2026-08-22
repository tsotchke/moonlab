/**
 * @file quantum_annealing.h
 * @brief Closed-system transverse-field quantum annealing for Ising/QUBO.
 *
 * Moonlab evolves
 *   H(s) = -A(s) driver_strength sum_i X_i
 *          + B(s) problem_strength H_problem
 * from |+>^n with a midpoint schedule and first- or second-order product
 * formula.  The implementation is a real statevector evolution; exhaustive
 * classical enumeration is used only after evolution to report the exact
 * optimum, degeneracy, final classical gap, and success probability.
 *
 * QUBO convention: E(x) = offset + sum_{i,j} x_i Q[i,j] x_j for x_i in
 * {0,1}.  Q may be asymmetric; off-diagonal Q[i,j] and Q[j,i] are combined
 * exactly during conversion to Ising spins z_i = 1 - 2 x_i.
 *
 * This is not a hardware-topology/minor-embedding or finite-temperature
 * D-Wave emulator.  Those are provider layers over this exact logical model.
 *
 * @since 1.2.1
 */

#ifndef MOONLAB_QUANTUM_ANNEALING_H
#define MOONLAB_QUANTUM_ANNEALING_H

#include "qaoa.h"
#include "../applications/moonlab_api.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MOONLAB_ANNEAL_OK = 0,
    MOONLAB_ANNEAL_BAD_ARG = -1,
    MOONLAB_ANNEAL_OOM = -2,
    MOONLAB_ANNEAL_STATE_ERROR = -3,
    MOONLAB_ANNEAL_SCHEDULE_ERROR = -4
} moonlab_anneal_status_t;

typedef enum {
    MOONLAB_ANNEAL_SCHEDULE_LINEAR = 0,
    MOONLAB_ANNEAL_SCHEDULE_QUADRATIC = 1,
    MOONLAB_ANNEAL_SCHEDULE_COSINE = 2
} moonlab_anneal_schedule_t;

typedef struct {
    double total_time;             /**< Dimensionless anneal time T (>0). */
    size_t num_steps;              /**< Product-formula steps (>=1). */
    size_t num_samples;            /**< Final computational-basis samples. */
    uint64_t seed;                 /**< 0 assigns and reports a seed. */
    moonlab_anneal_schedule_t schedule;
    double driver_strength;        /**< Multiplier on -sum X_i (>0). */
    double problem_strength;       /**< Multiplier on H_problem (>0). */
    int second_order;              /**< Nonzero selects symmetric Strang steps. */
} moonlab_anneal_config_t;

typedef struct moonlab_anneal_result moonlab_anneal_result_t;

/** Static diagnostic name for a moonlab_anneal_status_t value.
 * @stability beta */
MOONLAB_API const char *moonlab_anneal_status_string(int status);

/** Default: T=10, 1000 steps, 1024 samples, assigned seed, cosine schedule,
 * unit strengths, second-order evolution.
 * @stability beta */
MOONLAB_API moonlab_anneal_config_t moonlab_anneal_config_default(void);

/** Evaluate A(s), B(s).  Enforces s in [0,1], A(0)=B(1)=1 and
 * A(1)=B(0)=0 for every built-in schedule.
 * @stability beta */
MOONLAB_API int moonlab_anneal_schedule_values(
    moonlab_anneal_schedule_t schedule, double s,
    double *driver_out, double *problem_out);

/** Convert full row-major Q in E=x^T Q x + offset to symmetric Ising arrays.
 * out_J is row-major n*n with a zero diagonal and both triangles populated.
 * @stability beta */
MOONLAB_API int moonlab_qubo_to_ising(
    size_t num_variables, const double *Q, double qubo_offset,
    double *out_h, double *out_J, double *out_ising_offset);

/** Evaluate the full-matrix QUBO convention directly. Returns DBL_MAX for an
 * invalid dimension, pointer, offset, or coefficient.
 * @stability beta */
MOONLAB_API double moonlab_qubo_evaluate(
    size_t num_variables, const double *Q, double offset,
    uint64_t bitstring);

/** Anneal a symmetric row-major Ising model.  J diagonal must be zero and
 * J[i,j] must equal J[j,i] within numerical tolerance.
 * @stability beta */
MOONLAB_API int moonlab_quantum_anneal_ising(
    size_t num_qubits, const double *h, const double *J, double offset,
    const moonlab_anneal_config_t *config,
    moonlab_anneal_result_t **out_result);

/** Convert then anneal E(x)=x^TQx+offset.
 * @stability beta */
MOONLAB_API int moonlab_quantum_anneal_qubo(
    size_t num_variables, const double *Q, double offset,
    const moonlab_anneal_config_t *config,
    moonlab_anneal_result_t **out_result);

/** Free an annealing result. @stability beta */
MOONLAB_API void moonlab_anneal_result_free(moonlab_anneal_result_t *result);

/** Result qubit count. @stability beta */
MOONLAB_API size_t moonlab_anneal_result_num_qubits(const moonlab_anneal_result_t *result);
/** Retained sample count. @stability beta */
MOONLAB_API size_t moonlab_anneal_result_num_samples(const moonlab_anneal_result_t *result);
/** Effective non-zero replay seed. @stability beta */
MOONLAB_API uint64_t moonlab_anneal_result_effective_seed(const moonlab_anneal_result_t *result);
/** Best sampled bitstring. @stability beta */
MOONLAB_API uint64_t moonlab_anneal_result_best_bitstring(const moonlab_anneal_result_t *result);
/** Highest-probability final-state bitstring. @stability beta */
MOONLAB_API uint64_t moonlab_anneal_result_most_likely_bitstring(const moonlab_anneal_result_t *result);
/** Representative exact ground bitstring. @stability beta */
MOONLAB_API uint64_t moonlab_anneal_result_ground_bitstring(const moonlab_anneal_result_t *result);
/** Exact ground-manifold degeneracy. @stability beta */
MOONLAB_API size_t moonlab_anneal_result_ground_degeneracy(const moonlab_anneal_result_t *result);
/** Best sampled energy. @stability beta */
MOONLAB_API double moonlab_anneal_result_best_energy(const moonlab_anneal_result_t *result);
/** Exact ground energy. @stability beta */
MOONLAB_API double moonlab_anneal_result_ground_energy(const moonlab_anneal_result_t *result);
/** Final-state expected problem energy. @stability beta */
MOONLAB_API double moonlab_anneal_result_expected_energy(const moonlab_anneal_result_t *result);
/** Probability mass on the exact ground manifold. @stability beta */
MOONLAB_API double moonlab_anneal_result_success_probability(const moonlab_anneal_result_t *result);
/** Expected minus ground energy. @stability beta */
MOONLAB_API double moonlab_anneal_result_residual_energy(const moonlab_anneal_result_t *result);
/** Exact final classical problem gap. @stability beta */
MOONLAB_API double moonlab_anneal_result_problem_gap(const moonlab_anneal_result_t *result);
/** Final statevector norm. @stability beta */
MOONLAB_API double moonlab_anneal_result_final_norm(const moonlab_anneal_result_t *result);

/** Copy one retained sample and its classical problem energy.
 * @stability beta */
MOONLAB_API int moonlab_anneal_result_sample(
    const moonlab_anneal_result_t *result, size_t index,
    uint64_t *bitstring_out, double *energy_out);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_QUANTUM_ANNEALING_H */
