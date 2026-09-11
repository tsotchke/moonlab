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
    MOONLAB_ANNEAL_SCHEDULE_ERROR = -4,
    MOONLAB_ANNEAL_EMBEDDING_ERROR = -5
} moonlab_anneal_status_t;

typedef enum {
    MOONLAB_ANNEAL_SCHEDULE_LINEAR = 0,
    MOONLAB_ANNEAL_SCHEDULE_QUADRATIC = 1,
    MOONLAB_ANNEAL_SCHEDULE_COSINE = 2,
    MOONLAB_ANNEAL_SCHEDULE_PIECEWISE = 3
} moonlab_anneal_schedule_t;

/** Point (t, s) in a piecewise annealing schedule. */
typedef struct {
    double t; /**< Dimensionless time fraction in [0, 1]. */
    double s; /**< Anneal parameter fraction in [0, 1]. */
} moonlab_anneal_schedule_point_t;

typedef struct {
    double total_time;             /**< Dimensionless anneal time T (>0). */
    size_t num_steps;              /**< Product-formula steps (>=1). */
    size_t num_samples;            /**< Final computational-basis samples. */
    uint64_t seed;                 /**< 0 assigns and reports a seed. */
    moonlab_anneal_schedule_t schedule;
    double driver_strength;        /**< Multiplier on -sum X_i (>0). */
    double problem_strength;       /**< Multiplier on H_problem (>0). */
    int second_order;              /**< Nonzero selects symmetric Strang steps. */

    /* 2026 D-Wave extensions */
    /** Array of (t, s) points for PIECEWISE schedule. */
    const moonlab_anneal_schedule_point_t *schedule_points;
    /** Count of points (>=2, t from 0 to 1, monotone in t). */
    size_t num_schedule_points;
    /** Nonzero enables reverse annealing from initial_bitstring. */
    int reverse_anneal;
    /** Starting state for reverse annealing. */
    uint64_t initial_bitstring;
    /** Inversion anneal fraction s in [0, 1]. */
    double reverse_s_target;
    /** Fraction of total_time spent holding at s_target. */
    double reverse_hold_fraction;
    /** Optional per-qubit offsets delta_s_i in [-1, 1] (length num_qubits). */
    const double *anneal_offsets;
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

/* ---- 2026 D-Wave Schedule Helpers & Reverse Annealing ---- */

/** Helper to create a 4-point pause schedule: ramp to s_pause,
 * hold for pause_duration_frac, then ramp to 1.0.
 * @stability beta */
MOONLAB_API int moonlab_anneal_schedule_make_pause(
    double s_pause, double pause_start_frac, double pause_duration_frac,
    moonlab_anneal_schedule_point_t points[4]);

/** Helper to create a 3-point quench schedule: anneal to s_quench,
 * then rapid quench to s=1.0 at t=1.0.
 * @stability beta */
MOONLAB_API int moonlab_anneal_schedule_make_quench(
    double s_quench, double quench_start_frac,
    moonlab_anneal_schedule_point_t points[3]);

/** Execute reverse annealing for an Ising problem starting from a classical bitstring.
 * @stability beta */
MOONLAB_API int moonlab_quantum_reverse_anneal_ising(
    size_t num_qubits, const double *h, const double *J, double offset,
    uint64_t initial_bitstring,
    double s_target,
    double hold_fraction,
    const moonlab_anneal_config_t *config,
    moonlab_anneal_result_t **out_result);

/** Execute reverse annealing for a QUBO problem starting from a classical bitstring.
 * @stability beta */
MOONLAB_API int moonlab_quantum_reverse_anneal_qubo(
    size_t num_variables, const double *Q, double offset,
    uint64_t initial_bitstring,
    double s_target,
    double hold_fraction,
    const moonlab_anneal_config_t *config,
    moonlab_anneal_result_t **out_result);

/* ---- 2026 D-Wave Advantage2 Zephyr Graph Embedding Support ---- */

/**
 * @brief Representation of a D-Wave Advantage2 Zephyr graph topology Z_m.
 *
 * A Zephyr graph Z_m with tile parameter t=4 has:
 * - Number of physical qubits: N = 16 * m * (2*m + 1)
 * - Nominal qubit degree: 20 (16 internal, 2 external, 2 odd couplers)
 */
typedef struct {
    size_t m;                    /**< Grid scale parameter. */
    size_t t;                    /**< Tile parameter (typically 4). */
    size_t num_qubits;           /**< Total number of physical qubits. */
    size_t num_couplers;         /**< Total number of bidirectional physical couplers. */
    uint32_t *coupler_u;         /**< Array of size num_couplers (u < v). */
    uint32_t *coupler_v;         /**< Array of size num_couplers. */
    uint8_t *adjacency_matrix;   /**< Flat adjacency matrix of size num_qubits * num_qubits. */
} moonlab_zephyr_graph_t;

/** Create a Zephyr topology graph Z_m with default t=4.
 * @param m Grid size parameter (m >= 1).
 * @return Allocated Zephyr graph or NULL on error.
 * @stability beta */
MOONLAB_API moonlab_zephyr_graph_t *moonlab_zephyr_graph_create(size_t m);

/** Free a Zephyr graph.
 * @stability beta */
MOONLAB_API void moonlab_zephyr_graph_free(moonlab_zephyr_graph_t *graph);

/** Check if two physical qubits are connected by a coupler in the Zephyr graph.
 * @stability beta */
MOONLAB_API int moonlab_zephyr_has_coupler(const moonlab_zephyr_graph_t *graph,
                                           size_t u, size_t v);

/** Minor embedding mapping from logical variables to physical qubit chains. */
typedef struct {
    size_t num_logical;          /**< Number of logical variables. */
    size_t *chain_lengths;       /**< Length of chain for each logical variable. */
    size_t **chains;             /**< Array of physical qubit indices for each chain. */
} moonlab_zephyr_embedding_t;

/** Free a Zephyr embedding.
 * @stability beta */
MOONLAB_API void moonlab_zephyr_embedding_free(moonlab_zephyr_embedding_t *emb);

/** Generate a deterministic clique (complete graph K_k) embedding into Zephyr Z_m.
 * @param num_logical Number of logical variables k.
 * @param m Zephyr grid size parameter.
 * @return Allocated embedding or NULL if k exceeds the embedding capacity of Z_m.
 * @stability beta */
MOONLAB_API moonlab_zephyr_embedding_t *moonlab_zephyr_find_clique_embedding(
    size_t num_logical, size_t m);

/** Embed a logical Ising problem onto the physical Zephyr graph.
 * @param graph Physical Zephyr graph.
 * @param emb Minor embedding mapping.
 * @param logical_h Array of size num_logical.
 * @param logical_J Row-major array of size num_logical * num_logical.
 * @param chain_strength Strength of ferromagnetic intra-chain coupling (>0).
 * @param out_physical_h Output array of size graph->num_qubits.
 * @param out_physical_J Output row-major array of size graph->num_qubits * graph->num_qubits.
 * @return MOONLAB_ANNEAL_OK on success.
 * @stability beta */
MOONLAB_API int moonlab_zephyr_embed_ising(
    const moonlab_zephyr_graph_t *graph,
    const moonlab_zephyr_embedding_t *emb,
    const double *logical_h,
    const double *logical_J,
    double chain_strength,
    double *out_physical_h,
    double *out_physical_J);

/** Decode physical bitstrings back to logical bitstrings via majority voting on chains.
 * @param emb Minor embedding mapping.
 * @param num_samples Number of samples to unembed.
 * @param physical_samples Array of physical bitstrings (64-bit masks).
 * @param out_logical_samples Output array of logical bitstrings.
 * @param out_chain_break_fractions Optional output array of fraction of broken chains per sample.
 * @stability beta */
MOONLAB_API int moonlab_zephyr_unembed_samples(
    const moonlab_zephyr_embedding_t *emb,
    size_t num_samples,
    const uint64_t *physical_samples,
    uint64_t *out_logical_samples,
    double *out_chain_break_fractions);

#ifdef __cplusplus
}
#endif

#endif /* MOONLAB_QUANTUM_ANNEALING_H */
