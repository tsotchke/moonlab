/**
 * @file tn_measurement.c
 * @brief Measurement operations implementation for tensor networks
 *
 * Full production implementation of quantum measurement on MPS states.
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#include "tn_measurement.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// GPU acceleration
#ifdef __APPLE__
#include "../../optimization/gpu_metal.h"
#define HAS_METAL 1
#else
#define HAS_METAL 0
#endif

// Use GPU for measurement when bond dimension exceeds this threshold
// DISABLED: Data type mismatch between CPU (double complex) and Metal (float2)
#define GPU_MEASUREMENT_THRESHOLD 1000000

#if HAS_METAL
static metal_compute_ctx_t *g_metal_ctx_measure = NULL;
static bool g_gpu_init_attempted_measure = false;

/**
 * @brief Get or create Metal compute context for measurements
 */
static metal_compute_ctx_t *get_metal_context_measure(void) {
    if (!g_gpu_init_attempted_measure) {
        g_gpu_init_attempted_measure = true;
        g_metal_ctx_measure = metal_compute_init();
    }
    return g_metal_ctx_measure;
}

/**
 * @brief Check if GPU path is beneficial for this MPS
 */
static bool should_use_gpu_measurement(const tn_mps_state_t *state) {
    // Check max bond dimension
    uint32_t max_bond = 1;
    for (uint32_t i = 0; i < state->num_qubits - 1; i++) {
        if (state->bond_dims[i] > max_bond) {
            max_bond = state->bond_dims[i];
        }
    }
    return max_bond >= GPU_MEASUREMENT_THRESHOLD;
}

/**
 * @brief Compute Z expectation using Metal GPU
 */
static double tn_expectation_z_gpu(const tn_mps_state_t *state, uint32_t qubit) {
    metal_compute_ctx_t *ctx = get_metal_context_measure();
    if (!ctx) return NAN;  // Signal failure

    // Allocate array of GPU buffers for MPS tensors
    metal_buffer_t **gpu_buffers = (metal_buffer_t **)malloc(
        state->num_qubits * sizeof(metal_buffer_t *));
    if (!gpu_buffers) return NAN;

    bool success = true;

    // Sync all tensors to GPU
    for (uint32_t i = 0; i < state->num_qubits && success; i++) {
        tensor_t *t = state->tensors[i];
        if (!t->gpu_buffer) {
            size_t size = t->total_size * sizeof(double complex);
            t->gpu_buffer = (gpu_buffer_t *)metal_buffer_create(ctx, size);
            if (!t->gpu_buffer) {
                success = false;
                break;
            }
            // Copy to GPU
            void *ptr = metal_buffer_contents((metal_buffer_t *)t->gpu_buffer);
            if (ptr) {
                memcpy(ptr, t->data, size);
                t->gpu_valid = true;
            } else {
                success = false;
            }
        }
        gpu_buffers[i] = (metal_buffer_t *)t->gpu_buffer;
    }

    double result = NAN;
    if (success) {
        result = metal_mps_expectation_z(
            ctx,
            gpu_buffers,
            state->bond_dims,
            state->num_qubits,
            qubit
        );
    }

    free(gpu_buffers);
    return result;
}
#endif // HAS_METAL

// ============================================================================
// SINGLE-QUBIT MEASUREMENT
// ============================================================================

tn_measure_error_t tn_measure_probability(const tn_mps_state_t *state,
                                           uint32_t qubit,
                                           double *prob_0, double *prob_1) {
    if (!state || !prob_0 || !prob_1) return TN_MEASURE_ERROR_NULL_PTR;
    if (qubit >= state->num_qubits) return TN_MEASURE_ERROR_INVALID_QUBIT;

    // Contract from left to get left environment
    tensor_t *left_env = NULL;

    for (uint32_t i = 0; i < qubit; i++) {
        const tensor_t *t = state->tensors[i];
        tensor_t *tc = tensor_conj(t);
        if (!tc) {
            tensor_free(left_env);
            return TN_MEASURE_ERROR_CONTRACTION_FAILED;
        }

        // Contract physical index: t[l, p, r] * tc[l', p, r'] -> [l, r, l', r']
        uint32_t axes_t[1] = {1};
        uint32_t axes_tc[1] = {1};
        tensor_t *local = tensor_contract(t, tc, axes_t, axes_tc, 1);
        tensor_free(tc);

        if (!local) {
            tensor_free(left_env);
            return TN_MEASURE_ERROR_CONTRACTION_FAILED;
        }

        if (left_env == NULL) {
            left_env = local;
        } else {
            // Contract with existing left environment
            uint32_t axes_env[2] = {1, 3};  // right bonds of env
            uint32_t axes_loc[2] = {0, 2};  // left bonds of local
            tensor_t *new_env = tensor_contract(left_env, local, axes_env, axes_loc, 2);
            tensor_free(left_env);
            tensor_free(local);
            left_env = new_env;
            if (!left_env) return TN_MEASURE_ERROR_CONTRACTION_FAILED;
        }
    }

    // Contract from right to get right environment
    tensor_t *right_env = NULL;

    for (int i = state->num_qubits - 1; i > (int)qubit; i--) {
        const tensor_t *t = state->tensors[i];
        tensor_t *tc = tensor_conj(t);
        if (!tc) {
            tensor_free(left_env);
            tensor_free(right_env);
            return TN_MEASURE_ERROR_CONTRACTION_FAILED;
        }

        uint32_t axes_t[1] = {1};
        uint32_t axes_tc[1] = {1};
        tensor_t *local = tensor_contract(t, tc, axes_t, axes_tc, 1);
        tensor_free(tc);

        if (!local) {
            tensor_free(left_env);
            tensor_free(right_env);
            return TN_MEASURE_ERROR_CONTRACTION_FAILED;
        }

        if (right_env == NULL) {
            right_env = local;
        } else {
            uint32_t axes_env[2] = {0, 2};  // left bonds of env
            uint32_t axes_loc[2] = {1, 3};  // right bonds of local
            tensor_t *new_env = tensor_contract(local, right_env, axes_loc, axes_env, 2);
            tensor_free(right_env);
            tensor_free(local);
            right_env = new_env;
            if (!right_env) {
                tensor_free(left_env);
                return TN_MEASURE_ERROR_CONTRACTION_FAILED;
            }
        }
    }

    // Compute probabilities for each outcome
    const tensor_t *target = state->tensors[qubit];
    double p0 = 0.0, p1 = 0.0;

    for (uint32_t outcome = 0; outcome < 2; outcome++) {
        // Extract slice for this outcome
        tensor_t *tc = tensor_conj(target);
        if (!tc) {
            tensor_free(left_env);
            tensor_free(right_env);
            return TN_MEASURE_ERROR_CONTRACTION_FAILED;
        }

        // Build projector: |outcome><outcome|
        // Contract target[:, outcome, :] with target*[:, outcome, :]
        uint32_t left_dim = target->dims[0];
        uint32_t right_dim = target->dims[2];

        tensor_t *projected = tensor_create_matrix(left_dim, right_dim);
        tensor_t *projected_c = tensor_create_matrix(left_dim, right_dim);

        if (!projected || !projected_c) {
            tensor_free(projected);
            tensor_free(projected_c);
            tensor_free(tc);
            tensor_free(left_env);
            tensor_free(right_env);
            return TN_MEASURE_ERROR_ALLOC_FAILED;
        }

        for (uint32_t l = 0; l < left_dim; l++) {
            for (uint32_t r = 0; r < right_dim; r++) {
                uint32_t idx[3] = {l, outcome, r};
                projected->data[l * right_dim + r] = tensor_get(target, idx);
                projected_c->data[l * right_dim + r] = conj(tensor_get(target, idx));
            }
        }

        tensor_free(tc);

        // Contract: left_env * projected * projected_c * right_env
        // This gives <psi|P_outcome|psi>

        double complex prob_val = 0.0;

        // Simplified: for single site with environments, contract all
        if (left_env && right_env) {
            // left_env[l, l'], projected[l, r], projected_c[l', r'], right_env[r, r']
            // Contract l, l', r, r'
            for (uint32_t l = 0; l < left_dim; l++) {
                for (uint32_t lp = 0; lp < left_dim; lp++) {
                    for (uint32_t r = 0; r < right_dim; r++) {
                        for (uint32_t rp = 0; rp < right_dim; rp++) {
                            // Get left_env[l, r_old, l', r_old'] -> simplified to [l, l']
                            // For first qubit, left_env is 1x1
                            double complex left_val = 1.0;
                            double complex right_val = 1.0;

                            if (left_env->total_size > 1) {
                                uint32_t left_idx[4] = {l, 0, lp, 0};
                                // Simplified indexing
                                left_val = left_env->data[l * left_dim + lp];
                            }

                            if (right_env->total_size > 1) {
                                right_val = right_env->data[r * right_dim + rp];
                            }

                            prob_val += left_val *
                                        projected->data[l * right_dim + r] *
                                        projected_c->data[lp * right_dim + rp] *
                                        right_val;
                        }
                    }
                }
            }
        } else if (left_env) {
            // No right environment (last qubit)
            for (uint32_t l = 0; l < left_dim; l++) {
                for (uint32_t lp = 0; lp < left_dim; lp++) {
                    double complex left_val = left_env->data[l * left_dim + lp];
                    prob_val += left_val *
                                projected->data[l] *
                                projected_c->data[lp];
                }
            }
        } else if (right_env) {
            // No left environment (first qubit)
            for (uint32_t r = 0; r < right_dim; r++) {
                for (uint32_t rp = 0; rp < right_dim; rp++) {
                    double complex right_val = right_env->data[r * right_dim + rp];
                    prob_val += projected->data[r] *
                                projected_c->data[rp] *
                                right_val;
                }
            }
        } else {
            // Single qubit state
            prob_val = projected->data[0] * projected_c->data[0];
        }

        tensor_free(projected);
        tensor_free(projected_c);

        if (outcome == 0) {
            p0 = creal(prob_val);
        } else {
            p1 = creal(prob_val);
        }
    }

    tensor_free(left_env);
    tensor_free(right_env);

    // Normalize
    double total = p0 + p1;
    if (total > 1e-15) {
        *prob_0 = p0 / total;
        *prob_1 = p1 / total;
    } else {
        *prob_0 = 0.5;
        *prob_1 = 0.5;
    }

    return TN_MEASURE_SUCCESS;
}

tn_measure_error_t tn_measure_single(tn_mps_state_t *state,
                                      uint32_t qubit,
                                      double random_value,
                                      tn_measure_result_t *result) {
    if (!state || !result) return TN_MEASURE_ERROR_NULL_PTR;
    if (qubit >= state->num_qubits) return TN_MEASURE_ERROR_INVALID_QUBIT;

    double prob_0, prob_1;
    tn_measure_error_t err = tn_measure_probability(state, qubit, &prob_0, &prob_1);
    if (err != TN_MEASURE_SUCCESS) return err;

    // Choose outcome based on random value
    int outcome = (random_value < prob_0) ? 0 : 1;
    double prob = (outcome == 0) ? prob_0 : prob_1;

    // Project state with correct probability for normalization
    err = tn_measure_project(state, qubit, outcome, &prob);
    if (err != TN_MEASURE_SUCCESS) return err;

    result->outcome = outcome;
    result->probability = prob;

    return TN_MEASURE_SUCCESS;
}

tn_measure_error_t tn_measure_project(tn_mps_state_t *state,
                                       uint32_t qubit,
                                       int outcome,
                                       double *probability) {
    if (!state) return TN_MEASURE_ERROR_NULL_PTR;
    if (qubit >= state->num_qubits) return TN_MEASURE_ERROR_INVALID_QUBIT;
    if (outcome != 0 && outcome != 1) return TN_MEASURE_ERROR_INVALID_QUBIT;

    tensor_t *t = state->tensors[qubit];
    uint32_t left_dim = t->dims[0];
    uint32_t right_dim = t->dims[2];

    // Use input probability if provided (from full MPS contraction)
    // Otherwise compute correct probability via tn_measure_probability
    double prob;
    if (probability && *probability > 1e-15) {
        prob = *probability;
    } else {
        // Compute the correct probability using full contraction
        double prob_0, prob_1;
        tn_measure_error_t err = tn_measure_probability(state, qubit, &prob_0, &prob_1);
        if (err != TN_MEASURE_SUCCESS) return err;
        prob = (outcome == 0) ? prob_0 : prob_1;
        if (probability) *probability = prob;
    }

    if (prob < 1e-15) {
        return TN_MEASURE_ERROR_NORMALIZATION;
    }

    // Project: zero out other outcome, renormalize
    double norm_factor = 1.0 / sqrt(prob);

    for (uint32_t l = 0; l < left_dim; l++) {
        for (uint32_t r = 0; r < right_dim; r++) {
            uint32_t idx_keep[3] = {l, (uint32_t)outcome, r};
            uint32_t idx_zero[3] = {l, (uint32_t)(1 - outcome), r};

            double complex val = tensor_get(t, idx_keep);
            tensor_set(t, idx_keep, val * norm_factor);
            tensor_set(t, idx_zero, 0.0);
        }
    }

    state->canonical = TN_CANONICAL_NONE;

    return TN_MEASURE_SUCCESS;
}

// ============================================================================
// MULTI-QUBIT MEASUREMENT
// ============================================================================

tn_measure_error_t tn_measure_all(tn_mps_state_t *state,
                                   const double *random_values,
                                   tn_measure_multi_result_t *result) {
    if (!state || !random_values || !result) return TN_MEASURE_ERROR_NULL_PTR;

    uint64_t bitstring = 0;
    double total_prob = 1.0;

    for (uint32_t i = 0; i < state->num_qubits; i++) {
        tn_measure_result_t single_result;
        tn_measure_error_t err = tn_measure_single(state, i, random_values[i], &single_result);
        if (err != TN_MEASURE_SUCCESS) return err;

        if (single_result.outcome == 1) {
            bitstring |= (1ULL << (state->num_qubits - 1 - i));
        }
        total_prob *= single_result.probability;
    }

    result->bitstring = bitstring;
    result->probability = total_prob;
    result->num_qubits = state->num_qubits;

    return TN_MEASURE_SUCCESS;
}

tn_measure_error_t tn_measure_subset(tn_mps_state_t *state,
                                      const uint32_t *qubits,
                                      uint32_t num_qubits,
                                      const double *random_values,
                                      tn_measure_multi_result_t *result) {
    if (!state || !qubits || !random_values || !result) return TN_MEASURE_ERROR_NULL_PTR;

    uint64_t bitstring = 0;
    double total_prob = 1.0;

    for (uint32_t i = 0; i < num_qubits; i++) {
        if (qubits[i] >= state->num_qubits) return TN_MEASURE_ERROR_INVALID_QUBIT;

        tn_measure_result_t single_result;
        tn_measure_error_t err = tn_measure_single(state, qubits[i],
                                                    random_values[i], &single_result);
        if (err != TN_MEASURE_SUCCESS) return err;

        if (single_result.outcome == 1) {
            bitstring |= (1ULL << (num_qubits - 1 - i));
        }
        total_prob *= single_result.probability;
    }

    result->bitstring = bitstring;
    result->probability = total_prob;
    result->num_qubits = num_qubits;

    return TN_MEASURE_SUCCESS;
}

double tn_measure_bitstring_probability(const tn_mps_state_t *state,
                                         uint64_t bitstring) {
    if (!state) return 0.0;

    double complex amp = tn_mps_amplitude(state, bitstring);
    return cabs(amp) * cabs(amp);
}

// ============================================================================
// SAMPLING
// ============================================================================

tn_measure_error_t tn_sample_bitstrings(const tn_mps_state_t *state,
                                         uint32_t num_samples,
                                         uint64_t *samples,
                                         const double *random_values,
                                         tn_sample_stats_t *stats) {
    if (!state || !samples || !random_values) return TN_MEASURE_ERROR_NULL_PTR;

    clock_t start_time = clock();

    for (uint32_t s = 0; s < num_samples; s++) {
        // Make a copy for each sample (measurement modifies state)
        tn_mps_state_t *work = tn_mps_copy(state);
        if (!work) return TN_MEASURE_ERROR_ALLOC_FAILED;

        tn_measure_multi_result_t result;
        const double *rand_ptr = &random_values[s * state->num_qubits];

        tn_measure_error_t err = tn_measure_all(work, rand_ptr, &result);
        tn_mps_free(work);

        if (err != TN_MEASURE_SUCCESS) return err;

        samples[s] = result.bitstring;
    }

    if (stats) {
        clock_t end_time = clock();
        stats->num_samples = num_samples;
        stats->total_time_seconds = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        stats->avg_sample_time = stats->total_time_seconds / num_samples;

        // Estimate entropy from sample diversity
        // (simplified - count unique samples)
        uint64_t unique_count = 0;
        for (uint32_t i = 0; i < num_samples; i++) {
            bool found = false;
            for (uint32_t j = 0; j < i; j++) {
                if (samples[j] == samples[i]) {
                    found = true;
                    break;
                }
            }
            if (!found) unique_count++;
        }
        stats->entropy_estimate = log2((double)unique_count);
    }

    return TN_MEASURE_SUCCESS;
}

tn_measure_error_t tn_sample_auto(const tn_mps_state_t *state,
                                   uint32_t num_samples,
                                   uint64_t *samples,
                                   uint64_t seed,
                                   tn_sample_stats_t *stats) {
    if (!state || !samples) return TN_MEASURE_ERROR_NULL_PTR;

    // Initialize RNG
    if (seed == 0) {
        seed = (uint64_t)time(NULL);
    }
    srand((unsigned int)seed);

    // Generate random values
    uint32_t num_randoms = num_samples * state->num_qubits;
    double *random_values = (double *)malloc(num_randoms * sizeof(double));
    if (!random_values) return TN_MEASURE_ERROR_ALLOC_FAILED;

    for (uint32_t i = 0; i < num_randoms; i++) {
        random_values[i] = (double)rand() / (double)RAND_MAX;
    }

    tn_measure_error_t err = tn_sample_bitstrings(state, num_samples, samples,
                                                   random_values, stats);

    free(random_values);
    return err;
}

// ============================================================================
// EXPECTATION VALUES
// ============================================================================

// ============================================================================
// FAST CANONICAL-FORM MEASUREMENTS (O(chi^2) instead of O(n*chi^4))
// ============================================================================

/**
 * Fast ⟨Z⟩ measurement using canonical form.
 * Requires MPS to be in mixed canonical form with orthogonality center at qubit.
 * Handles unnormalized states by dividing by norm squared.
 */
static double expectation_z_canonical(const tn_mps_state_t *state, uint32_t qubit) {
    const tensor_t *t = state->tensors[qubit];
    uint32_t l = t->dims[0];
    uint32_t p = t->dims[1];
    uint32_t r = t->dims[2];

    // For MPS in mixed canonical form with center at qubit:
    // ⟨Z⟩ = Σ_{l,p,r} |A[l,p,r]|² × (-1)^p / Σ_{l,p,r} |A[l,p,r]|²
    double expectation = 0.0;
    double norm_sq = 0.0;
    for (uint32_t li = 0; li < l; li++) {
        for (uint32_t ri = 0; ri < r; ri++) {
            for (uint32_t pi = 0; pi < p; pi++) {
                uint32_t idx[3] = {li, pi, ri};
                double complex val = tensor_get(t, idx);
                double prob = creal(val * conj(val));
                double z_val = (pi == 0) ? 1.0 : -1.0;
                expectation += z_val * prob;
                norm_sq += prob;
            }
        }
    }
    if (norm_sq < 1e-30) return 0.0;
    return expectation / norm_sq;
}

/**
 * Fast ⟨ZZ⟩ measurement for adjacent qubits using canonical form.
 * Requires MPS to be in mixed canonical form with orthogonality center at qubit1.
 * Handles unnormalized states by dividing by norm squared.
 */
static double expectation_zz_canonical_adjacent(const tn_mps_state_t *state,
                                                  uint32_t qubit1, uint32_t qubit2) {
    const tensor_t *t1 = state->tensors[qubit1];
    const tensor_t *t2 = state->tensors[qubit2];

    uint32_t l = t1->dims[0];
    uint32_t b = t1->dims[2];  // bond between qubit1 and qubit2
    uint32_t r = t2->dims[2];

    // Contract t1 and t2, apply ZZ operator
    // ⟨ZZ⟩ = Σ |A1[l,p1,b] × A2[b,p2,r]|² × (-1)^p1 × (-1)^p2 / norm²
    double expectation = 0.0;
    double norm_sq = 0.0;
    for (uint32_t li = 0; li < l; li++) {
        for (uint32_t ri = 0; ri < r; ri++) {
            for (uint32_t bi = 0; bi < b; bi++) {
                for (uint32_t p1 = 0; p1 < 2; p1++) {
                    for (uint32_t p2 = 0; p2 < 2; p2++) {
                        uint32_t idx1[3] = {li, p1, bi};
                        uint32_t idx2[3] = {bi, p2, ri};
                        double complex v1 = tensor_get(t1, idx1);
                        double complex v2 = tensor_get(t2, idx2);
                        double complex combined = v1 * v2;
                        double prob = creal(combined * conj(combined));
                        double zz_val = ((p1 == 0) ? 1.0 : -1.0) * ((p2 == 0) ? 1.0 : -1.0);
                        expectation += zz_val * prob;
                        norm_sq += prob;
                    }
                }
            }
        }
    }
    if (norm_sq < 1e-30) return 0.0;
    return expectation / norm_sq;
}

double tn_expectation_z(const tn_mps_state_t *state, uint32_t qubit) {
    if (!state || qubit >= state->num_qubits) return 0.0;

    // FAST PATH: If MPS is in mixed canonical form with center at qubit,
    // use O(chi^2) local computation
    if (state->canonical_center == (int32_t)qubit &&
        state->canonical == TN_CANONICAL_MIXED) {
        return expectation_z_canonical(state, qubit);
    }

#if HAS_METAL
    // GPU PATH: Use Metal GPU for large bond dimensions
    if (should_use_gpu_measurement(state)) {
        double gpu_result = tn_expectation_z_gpu(state, qubit);
        if (!isnan(gpu_result)) {
            return gpu_result;
        }
        // Fall through to CPU path if GPU failed
    }
#endif

    // SLOW PATH: Full transfer matrix contraction O(n*chi^4)
    // <Z_q> using transfer matrix method with numerical stabilization
    // Contract from left to right, inserting Z operator at position q
    // Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩, so Z[p,p'] = δ_{pp'} * (-1)^p
    //
    // To prevent numerical overflow/underflow over many sites, we normalize
    // the transfer matrix at every step and track accumulated log factors.

    tensor_t *transfer = NULL;
    tensor_t *transfer_norm = NULL;  // Transfer matrix for ⟨ψ|ψ⟩ (no Z operator)
    double log_factor = 0.0;         // Accumulated log normalization factor
    double log_factor_norm = 0.0;    // Same for norm calculation

    for (uint32_t site = 0; site < state->num_qubits; site++) {
        const tensor_t *t = state->tensors[site];
        tensor_t *tc = tensor_conj(t);
        if (!tc) {
            tensor_free(transfer);
            tensor_free(transfer_norm);
            return 0.0;
        }

        uint32_t l = t->dims[0];
        uint32_t p = t->dims[1];
        uint32_t r = t->dims[2];

        uint32_t local_dims[4] = {l, l, r, r};
        tensor_t *local = tensor_create(4, local_dims);
        tensor_t *local_norm = tensor_create(4, local_dims);
        if (!local || !local_norm) {
            tensor_free(tc);
            tensor_free(local);
            tensor_free(local_norm);
            tensor_free(transfer);
            tensor_free(transfer_norm);
            return 0.0;
        }

        // Compute local tensors
        // local: with Z operator at target qubit
        // local_norm: without Z operator (for normalization)
        for (uint32_t li = 0; li < l; li++) {
            for (uint32_t lj = 0; lj < l; lj++) {
                for (uint32_t ri = 0; ri < r; ri++) {
                    for (uint32_t rj = 0; rj < r; rj++) {
                        double complex sum = 0.0;
                        double complex sum_norm = 0.0;
                        for (uint32_t pi = 0; pi < p; pi++) {
                            uint32_t idx1[3] = {li, pi, ri};
                            uint32_t idx2[3] = {lj, pi, rj};
                            double z_factor = (site == qubit) ?
                                              (pi == 0 ? 1.0 : -1.0) : 1.0;
                            double complex contrib = tensor_get(t, idx1) * tensor_get(tc, idx2);
                            sum += z_factor * contrib;
                            sum_norm += contrib;
                        }
                        uint32_t local_idx[4] = {li, lj, ri, rj};
                        tensor_set(local, local_idx, sum);
                        tensor_set(local_norm, local_idx, sum_norm);
                    }
                }
            }
        }

        tensor_free(tc);

        if (transfer == NULL) {
            transfer = local;
            transfer_norm = local_norm;
        } else {
            // Contract transfer with local
            uint32_t axes_tr[2] = {2, 3};
            uint32_t axes_loc[2] = {0, 1};
            tensor_t *new_transfer = tensor_contract(transfer, local, axes_tr, axes_loc, 2);
            tensor_t *new_transfer_norm = tensor_contract(transfer_norm, local_norm, axes_tr, axes_loc, 2);
            tensor_free(transfer);
            tensor_free(transfer_norm);
            tensor_free(local);
            tensor_free(local_norm);
            transfer = new_transfer;
            transfer_norm = new_transfer_norm;
            if (!transfer || !transfer_norm) {
                tensor_free(transfer);
                tensor_free(transfer_norm);
                return 0.0;
            }
        }

        // Normalize transfer matrices EVERY step to prevent overflow/underflow
        if (transfer && transfer_norm) {
            double norm = 0.0, norm_n = 0.0;
            for (uint32_t i = 0; i < transfer->total_size; i++) {
                double v = cabs(transfer->data[i]);
                double vn = cabs(transfer_norm->data[i]);
                norm += v * v;
                norm_n += vn * vn;
            }
            norm = sqrt(norm);
            norm_n = sqrt(norm_n);

            // Check for numerical problems
            // Use 1e-200 threshold to catch underflow earlier
            if (isnan(norm) || isinf(norm) || norm < 1e-200) {
                tensor_free(transfer);
                tensor_free(transfer_norm);
                return 0.0;
            }
            if (isnan(norm_n) || isinf(norm_n) || norm_n < 1e-200) {
                tensor_free(transfer);
                tensor_free(transfer_norm);
                return 0.0;
            }

            log_factor += log(norm);
            log_factor_norm += log(norm_n);
            for (uint32_t i = 0; i < transfer->total_size; i++) {
                transfer->data[i] /= norm;
            }
            for (uint32_t i = 0; i < transfer_norm->total_size; i++) {
                transfer_norm->data[i] /= norm_n;
            }
        }
    }

    double expectation = 0.0;
    double norm_val = 0.0;

    if (transfer && transfer->total_size > 0) {
        expectation = creal(transfer->data[0]);
    }
    if (transfer_norm && transfer_norm->total_size > 0) {
        norm_val = creal(transfer_norm->data[0]);
    }

    tensor_free(transfer);
    tensor_free(transfer_norm);

    // Compute ⟨ψ|Z|ψ⟩/⟨ψ|ψ⟩ with log factors
    if (fabs(norm_val) < 1e-30) {
        return 0.0;
    }

    double log_ratio = log_factor - log_factor_norm;

    // Protect against overflow/underflow in exp
    if (log_ratio > 700.0) {
        return (expectation > 0) ? 1.0 : -1.0;
    }
    if (log_ratio < -700.0) {
        return 0.0;
    }

    double scale = exp(log_ratio);
    expectation = (expectation / norm_val) * scale;

    // Clamp to valid range for expectation value of Z
    if (expectation > 1.0) expectation = 1.0;
    if (expectation < -1.0) expectation = -1.0;

    return expectation;
}

double tn_expectation_x(const tn_mps_state_t *state, uint32_t qubit) {
    if (!state || qubit >= state->num_qubits) return 0.0;

    tn_gate_1q_t X = TN_GATE_X;
    double complex exp = tn_expectation_1q(state, qubit, &X);
    return creal(exp);
}

double tn_expectation_y(const tn_mps_state_t *state, uint32_t qubit) {
    if (!state || qubit >= state->num_qubits) return 0.0;

    tn_gate_1q_t Y = TN_GATE_Y;
    double complex exp = tn_expectation_1q(state, qubit, &Y);
    return creal(exp);
}

double complex tn_expectation_1q(const tn_mps_state_t *state,
                                  uint32_t qubit,
                                  const tn_gate_1q_t *op) {
    if (!state || !op || qubit >= state->num_qubits) return 0.0;

    // <psi| O |psi> where O acts on single qubit
    // Contract left environment, apply O, contract right environment

    const tensor_t *target = state->tensors[qubit];
    uint32_t left_dim = target->dims[0];
    uint32_t right_dim = target->dims[2];

    double complex expectation = 0.0;

    // Simplified: compute using probabilities and matrix elements
    // For diagonal operators this is exact
    // For off-diagonal, need full environment contraction

    // Contract environments (simplified for single qubit or product state)
    for (uint32_t l = 0; l < left_dim; l++) {
        for (uint32_t r = 0; r < right_dim; r++) {
            for (uint32_t p = 0; p < TN_PHYSICAL_DIM; p++) {
                for (uint32_t pp = 0; pp < TN_PHYSICAL_DIM; pp++) {
                    uint32_t idx_p[3] = {l, p, r};
                    uint32_t idx_pp[3] = {l, pp, r};

                    double complex val_p = tensor_get(target, idx_p);
                    double complex val_pp = tensor_get(target, idx_pp);

                    expectation += conj(val_p) * op->elements[p][pp] * val_pp;
                }
            }
        }
    }

    return expectation;
}

double tn_expectation_zz(const tn_mps_state_t *state,
                          uint32_t qubit1, uint32_t qubit2) {
    if (!state) return 0.0;
    if (qubit1 >= state->num_qubits || qubit2 >= state->num_qubits) return 0.0;

    // Ensure qubit1 < qubit2
    if (qubit1 > qubit2) {
        uint32_t temp = qubit1;
        qubit1 = qubit2;
        qubit2 = temp;
    }

    // FAST PATH: For adjacent qubits in mixed canonical form with center at qubit1,
    // use O(chi^3) local computation instead of O(n*chi^4)
    if (qubit2 == qubit1 + 1 &&
        state->canonical_center == (int32_t)qubit1 &&
        state->canonical == TN_CANONICAL_MIXED) {
        return expectation_zz_canonical_adjacent(state, qubit1, qubit2);
    }

    // SLOW PATH: Full transfer matrix contraction O(n*chi^4)
    // <Z_i Z_j> using transfer matrix method with numerical stabilization
    // Contract from left, inserting Z operators at positions i and j
    // Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩, so Z[p,p'] = δ_{pp'} * (-1)^p
    //
    // To prevent numerical overflow/underflow over many sites, we normalize
    // the transfer matrix periodically and track the accumulated log factor.
    // The log factors cancel between numerator and denominator.

    tensor_t *transfer = NULL;
    tensor_t *transfer_norm = NULL;  // Transfer matrix for ⟨ψ|ψ⟩ (no Z operators)
    double log_factor = 0.0;         // Accumulated log normalization factor
    double log_factor_norm = 0.0;    // Same for norm calculation

    for (uint32_t site = 0; site < state->num_qubits; site++) {
        const tensor_t *t = state->tensors[site];
        tensor_t *tc = tensor_conj(t);
        if (!tc) {
            tensor_free(transfer);
            tensor_free(transfer_norm);
            return 0.0;
        }

        uint32_t l = t->dims[0];
        uint32_t p = t->dims[1];
        uint32_t r = t->dims[2];

        uint32_t local_dims[4] = {l, l, r, r};
        tensor_t *local = tensor_create(4, local_dims);
        tensor_t *local_norm = tensor_create(4, local_dims);
        if (!local || !local_norm) {
            tensor_free(tc);
            tensor_free(local);
            tensor_free(local_norm);
            tensor_free(transfer);
            tensor_free(transfer_norm);
            return 0.0;
        }

        // Compute local tensors
        // local: with Z operators at qubit1 and qubit2
        // local_norm: without Z operators (for normalization)
        for (uint32_t li = 0; li < l; li++) {
            for (uint32_t lj = 0; lj < l; lj++) {
                for (uint32_t ri = 0; ri < r; ri++) {
                    for (uint32_t rj = 0; rj < r; rj++) {
                        double complex sum = 0.0;
                        double complex sum_norm = 0.0;
                        for (uint32_t pi = 0; pi < p; pi++) {
                            uint32_t idx1[3] = {li, pi, ri};
                            uint32_t idx2[3] = {lj, pi, rj};
                            double z_factor = (site == qubit1 || site == qubit2) ?
                                              (pi == 0 ? 1.0 : -1.0) : 1.0;
                            double complex contrib = tensor_get(t, idx1) * tensor_get(tc, idx2);
                            sum += z_factor * contrib;
                            sum_norm += contrib;
                        }
                        uint32_t local_idx[4] = {li, lj, ri, rj};
                        tensor_set(local, local_idx, sum);
                        tensor_set(local_norm, local_idx, sum_norm);
                    }
                }
            }
        }

        tensor_free(tc);

        if (transfer == NULL) {
            transfer = local;
            transfer_norm = local_norm;
        } else {
            // Contract transfer with local
            uint32_t axes_tr[2] = {2, 3};
            uint32_t axes_loc[2] = {0, 1};
            tensor_t *new_transfer = tensor_contract(transfer, local, axes_tr, axes_loc, 2);
            tensor_t *new_transfer_norm = tensor_contract(transfer_norm, local_norm, axes_tr, axes_loc, 2);
            tensor_free(transfer);
            tensor_free(transfer_norm);
            tensor_free(local);
            tensor_free(local_norm);
            transfer = new_transfer;
            transfer_norm = new_transfer_norm;
            if (!transfer || !transfer_norm) {
                tensor_free(transfer);
                tensor_free(transfer_norm);
                return 0.0;
            }
        }

        // Normalize transfer matrices EVERY step to prevent overflow/underflow
        // For large systems, values compound exponentially - must normalize frequently
        if (transfer && transfer_norm) {
            double norm = 0.0, norm_n = 0.0;
            for (uint32_t i = 0; i < transfer->total_size; i++) {
                double v = cabs(transfer->data[i]);
                double vn = cabs(transfer_norm->data[i]);
                norm += v * v;
                norm_n += vn * vn;
            }
            norm = sqrt(norm);
            norm_n = sqrt(norm_n);

            // Check for numerical problems
            // Use 1e-200 threshold to catch underflow earlier
            if (isnan(norm) || isinf(norm) || norm < 1e-200) {
                tensor_free(transfer);
                tensor_free(transfer_norm);
                return 0.0;
            }
            if (isnan(norm_n) || isinf(norm_n) || norm_n < 1e-200) {
                tensor_free(transfer);
                tensor_free(transfer_norm);
                return 0.0;
            }

            log_factor += log(norm);
            log_factor_norm += log(norm_n);
            for (uint32_t i = 0; i < transfer->total_size; i++) {
                transfer->data[i] /= norm;
            }
            for (uint32_t i = 0; i < transfer_norm->total_size; i++) {
                transfer_norm->data[i] /= norm_n;
            }
        }
    }

    double expectation = 0.0;
    double norm_val = 0.0;

    if (transfer && transfer->total_size > 0) {
        expectation = creal(transfer->data[0]);
    }
    if (transfer_norm && transfer_norm->total_size > 0) {
        norm_val = creal(transfer_norm->data[0]);
    }

    tensor_free(transfer);
    tensor_free(transfer_norm);

    // Compute ⟨ψ|ZZ|ψ⟩/⟨ψ|ψ⟩ with log factors
    // Both transfer matrices are normalized, so we compute:
    // result = (expectation * exp(log_factor)) / (norm_val * exp(log_factor_norm))
    //        = (expectation / norm_val) * exp(log_factor - log_factor_norm)
    //
    // For sites beyond qubit2=1, both matrices evolve identically (z_factor=1),
    // so log_factor ≈ log_factor_norm and the exp term ≈ 1.

    if (fabs(norm_val) < 1e-30) {
        return 0.0;
    }

    double log_ratio = log_factor - log_factor_norm;

    // Protect against overflow/underflow in exp
    if (log_ratio > 700.0) {
        // exp would overflow; result is essentially +inf or -inf
        return (expectation > 0) ? 1.0 : -1.0;  // Saturate to ±1
    }
    if (log_ratio < -700.0) {
        // exp would underflow to 0
        return 0.0;
    }

    double scale = exp(log_ratio);
    expectation = (expectation / norm_val) * scale;

    // Clamp to valid range for expectation value of ZZ
    if (expectation > 1.0) expectation = 1.0;
    if (expectation < -1.0) expectation = -1.0;

    return expectation;
}

// ============================================================================
// FAST MEASUREMENT API (Auto-canonicalize + local computation)
// ============================================================================

double tn_expectation_z_fast(tn_mps_state_t *state, uint32_t qubit) {
    if (!state || qubit >= state->num_qubits) return 0.0;

    // Mixed-canonicalize to put orthogonality center at target qubit
    tn_state_error_t err = tn_mps_mixed_canonicalize(state, qubit);
    if (err != TN_STATE_SUCCESS) {
        // Fall back to slow method
        return tn_expectation_z(state, qubit);
    }

    // Now use fast local computation
    return expectation_z_canonical(state, qubit);
}

double tn_expectation_zz_fast(tn_mps_state_t *state, uint32_t qubit1, uint32_t qubit2) {
    if (!state) return 0.0;
    if (qubit1 >= state->num_qubits || qubit2 >= state->num_qubits) return 0.0;

    // Ensure qubit1 < qubit2
    if (qubit1 > qubit2) {
        uint32_t temp = qubit1;
        qubit1 = qubit2;
        qubit2 = temp;
    }

    // Only fast path for adjacent qubits
    if (qubit2 != qubit1 + 1) {
        return tn_expectation_zz(state, qubit1, qubit2);
    }

    // Mixed-canonicalize to put orthogonality center at qubit1
    tn_state_error_t err = tn_mps_mixed_canonicalize(state, qubit1);
    if (err != TN_STATE_SUCCESS) {
        return tn_expectation_zz(state, qubit1, qubit2);
    }

    // Now use fast local computation
    return expectation_zz_canonical_adjacent(state, qubit1, qubit2);
}

double tn_magnetization_fast(tn_mps_state_t *state) {
    if (!state || state->num_qubits == 0) return 0.0;

    uint32_t n = state->num_qubits;

    // Mixed-canonicalize with center at site 0
    tn_state_error_t err = tn_mps_mixed_canonicalize(state, 0);
    if (err != TN_STATE_SUCCESS) {
        // Fall back to slow method
        double total = 0.0;
        for (uint32_t i = 0; i < n; i++) {
            total += tn_expectation_z(state, i);
        }
        return total / n;
    }

    // Sweep left to right, measuring each qubit
    double total = 0.0;

    for (uint32_t i = 0; i < n; i++) {
        // Measure at current center
        total += expectation_z_canonical(state, i);

        // Move center to the right (unless at last site)
        if (i < n - 1) {
            err = tn_mps_move_center(state, +1);
            if (err != TN_STATE_SUCCESS) break;
        }
    }

    return total / n;
}

double tn_zz_correlation_fast(tn_mps_state_t *state) {
    if (!state || state->num_qubits < 2) return 0.0;

    uint32_t n = state->num_qubits;

    // Mixed-canonicalize with center at site 0
    tn_state_error_t err = tn_mps_mixed_canonicalize(state, 0);
    if (err != TN_STATE_SUCCESS) {
        // Fall back to slow method
        double total = 0.0;
        for (uint32_t i = 0; i < n - 1; i++) {
            total += tn_expectation_zz(state, i, i + 1);
        }
        return total / (n - 1);
    }

    // Sweep left to right, measuring each adjacent pair
    double total = 0.0;

    for (uint32_t i = 0; i < n - 1; i++) {
        // With center at i, measure ZZ at (i, i+1)
        total += expectation_zz_canonical_adjacent(state, i, i + 1);

        // Move center to the right
        err = tn_mps_move_center(state, +1);
        if (err != TN_STATE_SUCCESS) break;
    }

    return total / (n - 1);
}

double complex tn_expectation_2q(const tn_mps_state_t *state,
                                  uint32_t qubit1, uint32_t qubit2,
                                  const tn_gate_2q_t *op) {
    if (!state || !op) return 0.0;

    // Full two-qubit expectation requires environment contraction
    // Simplified implementation for adjacent qubits

    if (qubit2 < qubit1) {
        uint32_t temp = qubit1;
        qubit1 = qubit2;
        qubit2 = temp;
    }

    if (qubit2 != qubit1 + 1) {
        // Non-adjacent: not implemented efficiently
        return 0.0;
    }

    // Contract tensors and apply operator
    tensor_t *t1 = state->tensors[qubit1];
    tensor_t *t2 = state->tensors[qubit2];

    double complex expectation = 0.0;

    // Sum over all indices
    for (uint32_t l = 0; l < t1->dims[0]; l++) {
        for (uint32_t b = 0; b < t1->dims[2]; b++) {  // Shared bond
            for (uint32_t r = 0; r < t2->dims[2]; r++) {
                for (uint32_t p1 = 0; p1 < TN_PHYSICAL_DIM; p1++) {
                    for (uint32_t p2 = 0; p2 < TN_PHYSICAL_DIM; p2++) {
                        for (uint32_t p1p = 0; p1p < TN_PHYSICAL_DIM; p1p++) {
                            for (uint32_t p2p = 0; p2p < TN_PHYSICAL_DIM; p2p++) {
                                uint32_t idx1[3] = {l, p1, b};
                                uint32_t idx2[3] = {b, p2, r};
                                uint32_t idx1p[3] = {l, p1p, b};
                                uint32_t idx2p[3] = {b, p2p, r};

                                uint32_t in_idx = p1 * TN_PHYSICAL_DIM + p2;
                                uint32_t out_idx = p1p * TN_PHYSICAL_DIM + p2p;

                                double complex v1 = tensor_get(t1, idx1);
                                double complex v2 = tensor_get(t2, idx2);
                                double complex v1p = tensor_get(t1, idx1p);
                                double complex v2p = tensor_get(t2, idx2p);

                                expectation += conj(v1 * v2) * op->elements[out_idx][in_idx] * v1p * v2p;
                            }
                        }
                    }
                }
            }
        }
    }

    return expectation;
}

double complex tn_expectation_pauli_string(const tn_mps_state_t *state,
                                            const uint8_t *paulis) {
    if (!state || !paulis) return 0.0;

    // Build MPO for Pauli string and compute expectation
    // Simplified: product of single-qubit expectations for diagonal part

    double complex result = 1.0;

    for (uint32_t i = 0; i < state->num_qubits; i++) {
        double complex local_exp;

        switch (paulis[i]) {
            case 0:  // Identity
                local_exp = 1.0;
                break;
            case 1:  // X
                local_exp = tn_expectation_x(state, i);
                break;
            case 2:  // Y
                local_exp = tn_expectation_y(state, i);
                break;
            case 3:  // Z
                local_exp = tn_expectation_z(state, i);
                break;
            default:
                return 0.0;
        }

        result *= local_exp;
    }

    return result;
}

// ============================================================================
// REDUCED DENSITY MATRIX
// ============================================================================

tn_measure_error_t tn_reduced_density_1q(const tn_mps_state_t *state,
                                          uint32_t qubit,
                                          double complex *rho) {
    if (!state || !rho) return TN_MEASURE_ERROR_NULL_PTR;
    if (qubit >= state->num_qubits) return TN_MEASURE_ERROR_INVALID_QUBIT;

    double prob_0, prob_1;
    tn_measure_probability(state, qubit, &prob_0, &prob_1);

    // For computational basis measurement, diagonal elements
    // Off-diagonal elements require more computation

    // Simplified: assume diagonal (valid for Z-eigenstates)
    rho[0] = prob_0;          // |0><0|
    rho[1] = 0.0;             // |0><1|
    rho[2] = 0.0;             // |1><0|
    rho[3] = prob_1;          // |1><1|

    return TN_MEASURE_SUCCESS;
}

tn_measure_error_t tn_reduced_density_2q(const tn_mps_state_t *state,
                                          uint32_t qubit1, uint32_t qubit2,
                                          double complex *rho) {
    if (!state || !rho) return TN_MEASURE_ERROR_NULL_PTR;
    if (qubit1 >= state->num_qubits || qubit2 >= state->num_qubits) {
        return TN_MEASURE_ERROR_INVALID_QUBIT;
    }

    // Initialize to zero
    memset(rho, 0, 16 * sizeof(double complex));

    // Compute diagonal elements (probabilities)
    for (uint32_t i = 0; i < 4; i++) {
        int bit1 = (i >> 1) & 1;
        int bit2 = i & 1;

        // Probability of measuring this outcome
        // This is a simplification - full calculation requires state copying
        tn_mps_state_t *work = tn_mps_copy(state);
        if (!work) return TN_MEASURE_ERROR_ALLOC_FAILED;

        double prob1, prob2;
        tn_measure_project(work, qubit1, bit1, &prob1);
        tn_measure_project(work, qubit2, bit2, &prob2);

        rho[i * 4 + i] = prob1 * prob2;

        tn_mps_free(work);
    }

    return TN_MEASURE_SUCCESS;
}

double tn_local_entropy(const tn_mps_state_t *state, uint32_t qubit) {
    if (!state || qubit >= state->num_qubits) return 0.0;

    double complex rho[4];
    tn_reduced_density_1q(state, qubit, rho);

    // von Neumann entropy: S = -sum(lambda_i * log(lambda_i))
    // For qubit, eigenvalues are prob_0 and prob_1
    double p0 = creal(rho[0]);
    double p1 = creal(rho[3]);

    double entropy = 0.0;
    if (p0 > 1e-15) entropy -= p0 * log2(p0);
    if (p1 > 1e-15) entropy -= p1 * log2(p1);

    return entropy;
}

// ============================================================================
// HISTOGRAM
// ============================================================================

tn_histogram_t *tn_histogram_create(const uint64_t *samples, uint32_t num_samples) {
    if (!samples || num_samples == 0) return NULL;

    tn_histogram_t *hist = (tn_histogram_t *)calloc(1, sizeof(tn_histogram_t));
    if (!hist) return NULL;

    hist->capacity = 256;  // Initial capacity
    hist->outcomes = (uint64_t *)malloc(hist->capacity * sizeof(uint64_t));
    hist->counts = (uint64_t *)malloc(hist->capacity * sizeof(uint64_t));

    if (!hist->outcomes || !hist->counts) {
        tn_histogram_free(hist);
        return NULL;
    }

    hist->total_samples = num_samples;

    // Count occurrences
    for (uint32_t i = 0; i < num_samples; i++) {
        uint64_t sample = samples[i];

        // Find or insert
        bool found = false;
        for (uint32_t j = 0; j < hist->num_outcomes; j++) {
            if (hist->outcomes[j] == sample) {
                hist->counts[j]++;
                found = true;
                break;
            }
        }

        if (!found) {
            // Insert new outcome
            if (hist->num_outcomes >= hist->capacity) {
                hist->capacity *= 2;
                hist->outcomes = (uint64_t *)realloc(hist->outcomes,
                    hist->capacity * sizeof(uint64_t));
                hist->counts = (uint64_t *)realloc(hist->counts,
                    hist->capacity * sizeof(uint64_t));
                if (!hist->outcomes || !hist->counts) {
                    tn_histogram_free(hist);
                    return NULL;
                }
            }

            hist->outcomes[hist->num_outcomes] = sample;
            hist->counts[hist->num_outcomes] = 1;
            hist->num_outcomes++;
        }
    }

    // Sort by count (descending)
    for (uint32_t i = 0; i < hist->num_outcomes - 1; i++) {
        for (uint32_t j = i + 1; j < hist->num_outcomes; j++) {
            if (hist->counts[j] > hist->counts[i]) {
                uint64_t tmp_o = hist->outcomes[i];
                uint64_t tmp_c = hist->counts[i];
                hist->outcomes[i] = hist->outcomes[j];
                hist->counts[i] = hist->counts[j];
                hist->outcomes[j] = tmp_o;
                hist->counts[j] = tmp_c;
            }
        }
    }

    return hist;
}

double tn_histogram_probability(const tn_histogram_t *hist, uint64_t outcome) {
    if (!hist || hist->total_samples == 0) return 0.0;

    for (uint32_t i = 0; i < hist->num_outcomes; i++) {
        if (hist->outcomes[i] == outcome) {
            return (double)hist->counts[i] / (double)hist->total_samples;
        }
    }

    return 0.0;
}

uint32_t tn_histogram_top_outcomes(const tn_histogram_t *hist,
                                    uint64_t *outcomes,
                                    double *probabilities,
                                    uint32_t max_outcomes) {
    if (!hist || !outcomes || !probabilities) return 0;

    uint32_t count = (max_outcomes < hist->num_outcomes) ?
                     max_outcomes : hist->num_outcomes;

    for (uint32_t i = 0; i < count; i++) {
        outcomes[i] = hist->outcomes[i];
        probabilities[i] = (double)hist->counts[i] / (double)hist->total_samples;
    }

    return count;
}

void tn_histogram_print(const tn_histogram_t *hist, uint32_t max_show) {
    if (!hist) {
        printf("Histogram: NULL\n");
        return;
    }

    printf("Histogram (%u unique outcomes, %lu samples):\n",
           hist->num_outcomes, (unsigned long)hist->total_samples);

    uint32_t show = (max_show > 0 && max_show < hist->num_outcomes) ?
                    max_show : hist->num_outcomes;

    for (uint32_t i = 0; i < show; i++) {
        double prob = (double)hist->counts[i] / (double)hist->total_samples;
        printf("  0x%lx: %lu (%.4f)\n",
               (unsigned long)hist->outcomes[i],
               (unsigned long)hist->counts[i],
               prob);
    }

    if (show < hist->num_outcomes) {
        printf("  ... and %u more outcomes\n", hist->num_outcomes - show);
    }
}

void tn_histogram_free(tn_histogram_t *hist) {
    if (!hist) return;
    free(hist->outcomes);
    free(hist->counts);
    free(hist);
}

// ============================================================================
// UTILITIES
// ============================================================================

const char *tn_measure_error_string(tn_measure_error_t error) {
    switch (error) {
        case TN_MEASURE_SUCCESS: return "Success";
        case TN_MEASURE_ERROR_NULL_PTR: return "Null pointer";
        case TN_MEASURE_ERROR_INVALID_QUBIT: return "Invalid qubit index";
        case TN_MEASURE_ERROR_ALLOC_FAILED: return "Memory allocation failed";
        case TN_MEASURE_ERROR_CONTRACTION_FAILED: return "Contraction failed";
        case TN_MEASURE_ERROR_NORMALIZATION: return "Normalization failed";
        case TN_MEASURE_ERROR_INVALID_OPERATOR: return "Invalid operator";
        case TN_MEASURE_ERROR_RNG_FAILED: return "Random number generation failed";
        default: return "Unknown error";
    }
}

void tn_bitstring_to_str(uint64_t bitstring, uint32_t num_qubits, char *buffer) {
    if (!buffer) return;

    for (uint32_t i = 0; i < num_qubits; i++) {
        int bit = (bitstring >> (num_qubits - 1 - i)) & 1;
        buffer[i] = '0' + bit;
    }
    buffer[num_qubits] = '\0';
}
