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
 * Licensed under the MIT License
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
#define GPU_MEASUREMENT_THRESHOLD 32  // Use GPU for larger bond dimensions

#if HAS_METAL
/**
 * @brief Get Metal compute context from unified GPU context
 *
 * Uses the global GPU context from tensor.c to avoid double initialization.
 */
static metal_compute_ctx_t *get_metal_context_measure(void) {
    tensor_gpu_context_t *gpu_ctx = tensor_gpu_get_context();
    if (!gpu_ctx) return NULL;
    return tensor_gpu_get_metal(gpu_ctx);
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

        // Contract environment tensors with projected state tensors
        // Environment tensor structure after MPS contraction:
        //   left_env has shape [1, 1, chi_l, chi_l] due to MPS left boundary condition
        //   right_env has shape [chi_r, 1, chi_r, 1] due to MPS right boundary condition
        // This allows efficient 2D-style indexing: [0,0,l,lp] -> l*chi + lp
        if (left_env && right_env) {
            // Full contraction: Tr(left_env * |proj><proj| * right_env)
            for (uint32_t l = 0; l < left_dim; l++) {
                for (uint32_t lp = 0; lp < left_dim; lp++) {
                    for (uint32_t r = 0; r < right_dim; r++) {
                        for (uint32_t rp = 0; rp < right_dim; rp++) {
                            double complex left_val = 1.0;
                            double complex right_val = 1.0;

                            if (left_env->total_size > 1) {
                                // left_env[0, 0, l, lp] with boundary dims = 1
                                // Row-major: 0*... + 0*... + l*chi + lp = l*chi + lp
                                left_val = left_env->data[l * left_dim + lp];
                            }

                            if (right_env->total_size > 1) {
                                // right_env[r, 0, rp, 0] with trailing dims = 1
                                // Row-major: r*chi + rp
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

        // Estimate empirical entropy from sample frequencies: H = -Σ p_i log₂(p_i)
        // First count occurrences of each unique sample
        uint64_t* unique_samples = malloc(num_samples * sizeof(uint64_t));
        uint32_t* counts = calloc(num_samples, sizeof(uint32_t));
        uint32_t unique_count = 0;

        if (unique_samples && counts) {
            for (uint32_t i = 0; i < num_samples; i++) {
                bool found = false;
                for (uint32_t j = 0; j < unique_count; j++) {
                    if (unique_samples[j] == samples[i]) {
                        counts[j]++;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    unique_samples[unique_count] = samples[i];
                    counts[unique_count] = 1;
                    unique_count++;
                }
            }

            // Compute empirical entropy
            double entropy = 0.0;
            for (uint32_t j = 0; j < unique_count; j++) {
                double p = (double)counts[j] / (double)num_samples;
                if (p > 0) {
                    entropy -= p * log2(p);
                }
            }
            stats->entropy_estimate = entropy;
        } else {
            // Fallback if allocation fails
            stats->entropy_estimate = log2((double)num_samples);
        }

        free(unique_samples);
        free(counts);
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

    // <psi| O_k |psi> where O acts on single qubit k
    // Use transfer matrix contraction: T[l,l'] = sum_p conj(A[l,p,r]) * A[l',p,r]
    // For the operator site: T_O[l,l'] = sum_{p,p'} conj(A[l,p,r]) * O[p,p'] * A[l',p',r]

    uint32_t n = state->num_qubits;

    // Start with left boundary (identity for first tensor's left index)
    uint32_t chi_0 = state->tensors[0]->dims[0];  // Should be 1
    double complex *L = (double complex *)calloc(chi_0 * chi_0, sizeof(double complex));
    if (!L) return 0.0;
    for (uint32_t i = 0; i < chi_0; i++) {
        L[i * chi_0 + i] = 1.0;  // Identity
    }

    // Contract from left to qubit-1
    for (uint32_t site = 0; site < qubit; site++) {
        const tensor_t *A = state->tensors[site];
        uint32_t chi_l = A->dims[0];
        uint32_t chi_r = A->dims[2];

        double complex *L_new = (double complex *)calloc(chi_r * chi_r, sizeof(double complex));
        if (!L_new) { free(L); return 0.0; }

        // L_new[r,r'] = sum_{l,l',p} L[l,l'] * conj(A[l,p,r]) * A[l',p,r']
        for (uint32_t l = 0; l < chi_l; l++) {
            for (uint32_t lp = 0; lp < chi_l; lp++) {
                double complex L_val = L[l * chi_l + lp];
                if (cabs(L_val) < 1e-15) continue;
                for (uint32_t p = 0; p < TN_PHYSICAL_DIM; p++) {
                    for (uint32_t r = 0; r < chi_r; r++) {
                        for (uint32_t rp = 0; rp < chi_r; rp++) {
                            uint32_t idx1[3] = {l, p, r};
                            uint32_t idx2[3] = {lp, p, rp};
                            double complex A1 = tensor_get(A, idx1);
                            double complex A2 = tensor_get(A, idx2);
                            L_new[r * chi_r + rp] += L_val * conj(A1) * A2;
                        }
                    }
                }
            }
        }
        free(L);
        L = L_new;
    }

    // Now contract with operator site
    const tensor_t *A_op = state->tensors[qubit];
    uint32_t chi_l = A_op->dims[0];
    uint32_t chi_r = A_op->dims[2];

    double complex *L_op = (double complex *)calloc(chi_r * chi_r, sizeof(double complex));
    if (!L_op) { free(L); return 0.0; }

    // L_op[r,r'] = sum_{l,l',p,p'} L[l,l'] * conj(A[l,p,r]) * O[p,p'] * A[l',p',r']
    for (uint32_t l = 0; l < chi_l; l++) {
        for (uint32_t lp = 0; lp < chi_l; lp++) {
            double complex L_val = L[l * chi_l + lp];
            if (cabs(L_val) < 1e-15) continue;
            for (uint32_t p = 0; p < TN_PHYSICAL_DIM; p++) {
                for (uint32_t pp = 0; pp < TN_PHYSICAL_DIM; pp++) {
                    double complex O_val = op->elements[p][pp];
                    if (cabs(O_val) < 1e-15) continue;
                    for (uint32_t r = 0; r < chi_r; r++) {
                        for (uint32_t rp = 0; rp < chi_r; rp++) {
                            uint32_t idx1[3] = {l, p, r};
                            uint32_t idx2[3] = {lp, pp, rp};
                            double complex A1 = tensor_get(A_op, idx1);
                            double complex A2 = tensor_get(A_op, idx2);
                            L_op[r * chi_r + rp] += L_val * conj(A1) * O_val * A2;
                        }
                    }
                }
            }
        }
    }
    free(L);
    L = L_op;

    // Contract from qubit+1 to end
    for (uint32_t site = qubit + 1; site < n; site++) {
        const tensor_t *A = state->tensors[site];
        uint32_t cur_chi_l = A->dims[0];
        uint32_t cur_chi_r = A->dims[2];

        double complex *L_new = (double complex *)calloc(cur_chi_r * cur_chi_r, sizeof(double complex));
        if (!L_new) { free(L); return 0.0; }

        for (uint32_t l = 0; l < cur_chi_l; l++) {
            for (uint32_t lp = 0; lp < cur_chi_l; lp++) {
                double complex L_val = L[l * cur_chi_l + lp];
                if (cabs(L_val) < 1e-15) continue;
                for (uint32_t p = 0; p < TN_PHYSICAL_DIM; p++) {
                    for (uint32_t r = 0; r < cur_chi_r; r++) {
                        for (uint32_t rp = 0; rp < cur_chi_r; rp++) {
                            uint32_t idx1[3] = {l, p, r};
                            uint32_t idx2[3] = {lp, p, rp};
                            double complex A1 = tensor_get(A, idx1);
                            double complex A2 = tensor_get(A, idx2);
                            L_new[r * cur_chi_r + rp] += L_val * conj(A1) * A2;
                        }
                    }
                }
            }
        }
        free(L);
        L = L_new;
    }

    // Final result: trace of L (should be 1x1 at the end)
    double complex result = L[0];
    free(L);

    return result;
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

    // Full two-qubit expectation value using environment contraction
    // For adjacent qubits, direct contraction is used (faster, equally accurate)
    // For non-adjacent qubits, transfer matrix method with proper environment building

    if (qubit2 < qubit1) {
        uint32_t temp = qubit1;
        qubit1 = qubit2;
        qubit2 = temp;
    }

    if (qubit2 != qubit1 + 1) {
        // Non-adjacent qubits: use transfer matrix method
        // ⟨O⟩ = Tr(L · M1 · gap · M2 · R) where L, R are environments
        // and M1, M2 are the operator applications

        uint32_t n = state->num_qubits;

        // Build left environment: contract sites 0 to qubit1-1
        // T[l1, l2] = transfer matrix from identity at left boundary
        const tensor_t *t0 = state->tensors[0];
        uint32_t chi_left = t0->dims[0];
        uint32_t chi_current = chi_left;

        // Start with identity: T[0,0] = 1, else 0
        double complex *T = (double complex *)calloc(chi_current * chi_current, sizeof(double complex));
        if (!T) return 0.0;
        T[0] = 1.0;

        // Contract left environment (sites 0 to qubit1-1)
        for (uint32_t site = 0; site < qubit1; site++) {
            const tensor_t *A = state->tensors[site];
            uint32_t chi_l = A->dims[0];
            uint32_t d = A->dims[1];
            uint32_t chi_r = A->dims[2];

            double complex *T_new = (double complex *)calloc(chi_r * chi_r, sizeof(double complex));
            if (!T_new) { free(T); return 0.0; }

            for (uint32_t r1 = 0; r1 < chi_r; r1++) {
                for (uint32_t r2 = 0; r2 < chi_r; r2++) {
                    double complex sum = 0.0;
                    for (uint32_t l1 = 0; l1 < chi_l; l1++) {
                        for (uint32_t l2 = 0; l2 < chi_l; l2++) {
                            double complex T_val = T[l1 * chi_current + l2];
                            if (cabs(T_val) < 1e-15) continue;
                            for (uint32_t p = 0; p < d; p++) {
                                double complex A_bra = A->data[l1 * d * chi_r + p * chi_r + r1];
                                double complex A_ket = A->data[l2 * d * chi_r + p * chi_r + r2];
                                sum += T_val * conj(A_bra) * A_ket;
                            }
                        }
                    }
                    T_new[r1 * chi_r + r2] = sum;
                }
            }

            free(T);
            T = T_new;
            chi_current = chi_r;
        }

        // Apply first operator at qubit1
        // T_new[r1, r2] = sum_{l,p,p'} T[l1,l2] * conj(A[l1,p,r1]) * O[p1p2, p1'p2'][row1] * A[l2,p',r2]
        {
            const tensor_t *A = state->tensors[qubit1];
            uint32_t chi_l = A->dims[0];
            uint32_t d = A->dims[1];
            uint32_t chi_r = A->dims[2];

            double complex *T_new = (double complex *)calloc(chi_r * chi_r * d * d, sizeof(double complex));
            if (!T_new) { free(T); return 0.0; }

            // T_new stores [r1, r2, p1, p1'] where p1, p1' are operator's first indices
            for (uint32_t r1 = 0; r1 < chi_r; r1++) {
                for (uint32_t r2 = 0; r2 < chi_r; r2++) {
                    for (uint32_t p1_out = 0; p1_out < d; p1_out++) {
                        for (uint32_t p1_in = 0; p1_in < d; p1_in++) {
                            double complex sum = 0.0;
                            for (uint32_t l1 = 0; l1 < chi_l; l1++) {
                                for (uint32_t l2 = 0; l2 < chi_l; l2++) {
                                    double complex T_val = T[l1 * chi_current + l2];
                                    if (cabs(T_val) < 1e-15) continue;
                                    double complex A_bra = A->data[l1 * d * chi_r + p1_out * chi_r + r1];
                                    double complex A_ket = A->data[l2 * d * chi_r + p1_in * chi_r + r2];
                                    sum += T_val * conj(A_bra) * A_ket;
                                }
                            }
                            T_new[(r1 * chi_r + r2) * d * d + p1_out * d + p1_in] = sum;
                        }
                    }
                }
            }

            free(T);
            T = T_new;
            chi_current = chi_r;
        }

        // Contract through the gap (sites qubit1+1 to qubit2-1)
        for (uint32_t site = qubit1 + 1; site < qubit2; site++) {
            const tensor_t *A = state->tensors[site];
            uint32_t chi_l = A->dims[0];
            uint32_t d = A->dims[1];
            uint32_t chi_r = A->dims[2];

            double complex *T_new = (double complex *)calloc(chi_r * chi_r * TN_PHYSICAL_DIM * TN_PHYSICAL_DIM,
                                                             sizeof(double complex));
            if (!T_new) { free(T); return 0.0; }

            for (uint32_t r1 = 0; r1 < chi_r; r1++) {
                for (uint32_t r2 = 0; r2 < chi_r; r2++) {
                    for (uint32_t p1_out = 0; p1_out < TN_PHYSICAL_DIM; p1_out++) {
                        for (uint32_t p1_in = 0; p1_in < TN_PHYSICAL_DIM; p1_in++) {
                            double complex sum = 0.0;
                            for (uint32_t l1 = 0; l1 < chi_l; l1++) {
                                for (uint32_t l2 = 0; l2 < chi_l; l2++) {
                                    double complex T_val = T[(l1 * chi_current + l2) *
                                                             TN_PHYSICAL_DIM * TN_PHYSICAL_DIM +
                                                             p1_out * TN_PHYSICAL_DIM + p1_in];
                                    if (cabs(T_val) < 1e-15) continue;
                                    for (uint32_t p = 0; p < d; p++) {
                                        double complex A_bra = A->data[l1 * d * chi_r + p * chi_r + r1];
                                        double complex A_ket = A->data[l2 * d * chi_r + p * chi_r + r2];
                                        sum += T_val * conj(A_bra) * A_ket;
                                    }
                                }
                            }
                            T_new[(r1 * chi_r + r2) * TN_PHYSICAL_DIM * TN_PHYSICAL_DIM +
                                  p1_out * TN_PHYSICAL_DIM + p1_in] = sum;
                        }
                    }
                }
            }

            free(T);
            T = T_new;
            chi_current = chi_r;
        }

        // Apply second operator at qubit2 and apply the 2-qubit operator matrix
        // Collect into final expectation
        double complex expectation = 0.0;
        {
            const tensor_t *A2 = state->tensors[qubit2];
            uint32_t chi_l2 = A2->dims[0];
            uint32_t d2 = A2->dims[1];
            uint32_t chi_r2 = A2->dims[2];

            for (uint32_t r2_bra = 0; r2_bra < chi_r2; r2_bra++) {
                for (uint32_t r2_ket = 0; r2_ket < chi_r2; r2_ket++) {
                    // Contract right environment
                    double complex R = 0.0;

                    // Build right environment from qubit2+1 to n-1
                    double complex *R_env = (double complex *)calloc(chi_r2 * chi_r2, sizeof(double complex));
                    if (!R_env) { free(T); return 0.0; }
                    R_env[r2_bra * chi_r2 + r2_ket] = 1.0;

                    for (uint32_t site = n - 1; site > qubit2; site--) {
                        const tensor_t *A = state->tensors[site];
                        uint32_t chi_l = A->dims[0];
                        uint32_t d = A->dims[1];
                        uint32_t chi_r = A->dims[2];

                        double complex *R_new = (double complex *)calloc(chi_l * chi_l, sizeof(double complex));
                        if (!R_new) { free(R_env); free(T); return 0.0; }

                        for (uint32_t l1 = 0; l1 < chi_l; l1++) {
                            for (uint32_t l2 = 0; l2 < chi_l; l2++) {
                                double complex sum = 0.0;
                                for (uint32_t r1 = 0; r1 < chi_r; r1++) {
                                    for (uint32_t r2_r = 0; r2_r < chi_r; r2_r++) {
                                        double complex R_val = R_env[r1 * chi_r + r2_r];
                                        if (cabs(R_val) < 1e-15) continue;
                                        for (uint32_t p = 0; p < d; p++) {
                                            double complex A_bra = A->data[l1 * d * chi_r + p * chi_r + r1];
                                            double complex A_ket = A->data[l2 * d * chi_r + p * chi_r + r2_r];
                                            sum += R_val * conj(A_bra) * A_ket;
                                        }
                                    }
                                }
                                R_new[l1 * chi_l + l2] = sum;
                            }
                        }

                        free(R_env);
                        R_env = R_new;
                    }

                    // Now R_env holds the right environment

                    // Contract T with qubit2 tensor and operator
                    for (uint32_t l2_1 = 0; l2_1 < chi_l2; l2_1++) {
                        for (uint32_t l2_2 = 0; l2_2 < chi_l2; l2_2++) {
                            for (uint32_t p1_out = 0; p1_out < TN_PHYSICAL_DIM; p1_out++) {
                                for (uint32_t p1_in = 0; p1_in < TN_PHYSICAL_DIM; p1_in++) {
                                    double complex T_val = T[(l2_1 * chi_current + l2_2) *
                                                             TN_PHYSICAL_DIM * TN_PHYSICAL_DIM +
                                                             p1_out * TN_PHYSICAL_DIM + p1_in];
                                    if (cabs(T_val) < 1e-15) continue;

                                    for (uint32_t p2_out = 0; p2_out < d2; p2_out++) {
                                        for (uint32_t p2_in = 0; p2_in < d2; p2_in++) {
                                            double complex A2_bra = A2->data[l2_1 * d2 * chi_r2 +
                                                                             p2_out * chi_r2 + r2_bra];
                                            double complex A2_ket = A2->data[l2_2 * d2 * chi_r2 +
                                                                             p2_in * chi_r2 + r2_ket];

                                            // Apply 2-qubit operator: O[p1_out*d+p2_out, p1_in*d+p2_in]
                                            uint32_t out_idx = p1_out * TN_PHYSICAL_DIM + p2_out;
                                            uint32_t in_idx = p1_in * TN_PHYSICAL_DIM + p2_in;
                                            double complex O_elem = op->elements[out_idx][in_idx];

                                            expectation += T_val * conj(A2_bra) * O_elem * A2_ket *
                                                          R_env[r2_bra * chi_r2 + r2_ket];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    free(R_env);
                }
            }
        }

        free(T);
        return expectation;
    }

    // Adjacent qubits: use optimized direct contraction
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

/**
 * @brief Get Pauli matrix element P[row, col]
 * Encoding: 0=I, 1=X, 2=Y, 3=Z
 */
static inline double complex get_pauli_element(uint8_t pauli, int row, int col) {
    switch (pauli) {
        case 0:  // Identity
            return (row == col) ? 1.0 : 0.0;
        case 1:  // X = [[0,1],[1,0]]
            return (row != col) ? 1.0 : 0.0;
        case 2:  // Y = [[0,-i],[i,0]]
            if (row == 0 && col == 1) return -I;
            if (row == 1 && col == 0) return I;
            return 0.0;
        case 3:  // Z = [[1,0],[0,-1]]
            if (row == col) return (row == 0) ? 1.0 : -1.0;
            return 0.0;
        default:
            return 0.0;
    }
}

double complex tn_expectation_pauli_string(const tn_mps_state_t *state,
                                            const uint8_t *paulis) {
    if (!state || !paulis) return 0.0;

    uint32_t n = state->num_qubits;
    if (n == 0) return 0.0;

    // Compute <psi|P|psi> using transfer matrix method
    // Transfer matrix T[l1, l2] accumulates contraction from left
    // At each site i: T'[r1, r2] = sum_{l1,l2,p,p'} T[l1,l2] * conj(A[l1,p,r1]) * P[p,p'] * A[l2,p',r2]

    // Get first site dimensions
    const tensor_t *t0 = state->tensors[0];
    if (!t0 || t0->rank < 3) return 0.0;

    // Allocate transfer matrix (starts as 1x1 identity for left boundary)
    double complex *transfer = (double complex *)calloc(1, sizeof(double complex));
    if (!transfer) return 0.0;
    transfer[0] = 1.0;  // Identity

    uint32_t curr_left1 = 1, curr_left2 = 1;

    for (uint32_t site = 0; site < n; site++) {
        const tensor_t *t = state->tensors[site];
        if (!t || t->rank < 3) {
            free(transfer);
            return 0.0;
        }

        uint32_t l_dim = t->dims[0];   // left bond
        uint32_t p_dim = t->dims[1];   // physical (should be 2)
        uint32_t r_dim = t->dims[2];   // right bond

        if (p_dim != 2) {
            free(transfer);
            return 0.0;
        }

        // Allocate new transfer matrix for right bonds
        uint32_t new_size = r_dim * r_dim;
        double complex *new_transfer = (double complex *)calloc(new_size, sizeof(double complex));
        if (!new_transfer) {
            free(transfer);
            return 0.0;
        }

        // Contract: new_transfer[r1, r2] = sum_{l1, l2, p, p'}
        //           transfer[l1, l2] * conj(A[l1,p,r1]) * P[p,p'] * A[l2,p',r2]
        uint8_t pauli = paulis[site];

        for (uint32_t r1 = 0; r1 < r_dim; r1++) {
            for (uint32_t r2 = 0; r2 < r_dim; r2++) {
                double complex sum = 0.0;

                for (uint32_t l1 = 0; l1 < curr_left1; l1++) {
                    for (uint32_t l2 = 0; l2 < curr_left2; l2++) {
                        double complex T_l1l2 = transfer[l1 * curr_left2 + l2];
                        if (cabs(T_l1l2) < 1e-15) continue;

                        for (int p = 0; p < 2; p++) {
                            for (int pp = 0; pp < 2; pp++) {
                                double complex P_elem = get_pauli_element(pauli, p, pp);
                                if (cabs(P_elem) < 1e-15) continue;

                                // A[l, p, r] is stored with index l*p_dim*r_dim + p*r_dim + r
                                uint32_t idx1 = l1 * p_dim * r_dim + p * r_dim + r1;
                                uint32_t idx2 = l2 * p_dim * r_dim + pp * r_dim + r2;

                                double complex A1_conj = conj(t->data[idx1]);
                                double complex A2 = t->data[idx2];

                                sum += T_l1l2 * A1_conj * P_elem * A2;
                            }
                        }
                    }
                }

                new_transfer[r1 * r_dim + r2] = sum;
            }
        }

        free(transfer);
        transfer = new_transfer;
        curr_left1 = r_dim;
        curr_left2 = r_dim;
    }

    // Final result: transfer should be 1x1 (right boundaries are both 1)
    double complex result = 0.0;
    if (curr_left1 == 1 && curr_left2 == 1) {
        result = transfer[0];
    } else if (curr_left1 > 0 && curr_left2 > 0) {
        // Sum diagonal if dimensions don't reduce to 1 (shouldn't happen for proper MPS)
        result = transfer[0];
    }

    free(transfer);

    // Include norm factors
    return result * state->norm * state->norm;
}

// ============================================================================
// REDUCED DENSITY MATRIX
// ============================================================================

tn_measure_error_t tn_reduced_density_1q(const tn_mps_state_t *state,
                                          uint32_t qubit,
                                          double complex *rho) {
    if (!state || !rho) return TN_MEASURE_ERROR_NULL_PTR;
    if (qubit >= state->num_qubits) return TN_MEASURE_ERROR_INVALID_QUBIT;

    // Full reduced density matrix computation including off-diagonal elements
    // ρ[s,s'] = Σ_{l,l',r,r'} left_env[l,l'] * A[l,s,r] * A*[l',s',r'] * right_env[r,r']

    // Contract from left to get left environment
    tensor_t *left_env = NULL;

    for (uint32_t i = 0; i < qubit; i++) {
        const tensor_t *t = state->tensors[i];
        tensor_t *tc = tensor_conj(t);
        if (!tc) {
            tensor_free(left_env);
            return TN_MEASURE_ERROR_CONTRACTION_FAILED;
        }

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
            uint32_t axes_env[2] = {1, 3};
            uint32_t axes_loc[2] = {0, 2};
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
            uint32_t axes_env[2] = {0, 2};
            uint32_t axes_loc[2] = {1, 3};
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

    // Compute full 2x2 reduced density matrix
    const tensor_t *target = state->tensors[qubit];
    uint32_t left_dim = target->dims[0];
    uint32_t right_dim = target->dims[2];

    // Initialize rho to zero
    for (int i = 0; i < 4; i++) rho[i] = 0.0;

    // ρ[s,s'] = Σ_{l,l',r,r'} left_env[l,l'] * A[l,s,r] * A*[l',s',r'] * right_env[r,r']
    for (uint32_t s = 0; s < 2; s++) {
        for (uint32_t sp = 0; sp < 2; sp++) {
            double complex element = 0.0;

            for (uint32_t l = 0; l < left_dim; l++) {
                for (uint32_t lp = 0; lp < left_dim; lp++) {
                    // Get left environment contribution
                    double complex left_val = 1.0;
                    if (left_env && left_env->total_size > 1) {
                        // left_env is [l, r_internal, l', r_internal']
                        // For contracted form, it should be [l, l']
                        if (left_env->rank == 2) {
                            left_val = left_env->data[l * left_dim + lp];
                        } else if (left_env->rank == 4) {
                            // Take trace over internal indices
                            double complex sum = 0.0;
                            uint32_t int_dim = left_env->dims[1];
                            for (uint32_t ri = 0; ri < int_dim; ri++) {
                                uint32_t idx = l * left_env->dims[1] * left_env->dims[2] * left_env->dims[3]
                                             + ri * left_env->dims[2] * left_env->dims[3]
                                             + lp * left_env->dims[3] + ri;
                                if (idx < left_env->total_size) sum += left_env->data[idx];
                            }
                            left_val = sum;
                        }
                    } else if (left_env && l == 0 && lp == 0) {
                        left_val = left_env->data[0];
                    } else if (!left_env && l == 0 && lp == 0) {
                        left_val = 1.0;
                    } else if (!left_env) {
                        left_val = (l == lp) ? 1.0 : 0.0;
                    }

                    for (uint32_t r = 0; r < right_dim; r++) {
                        for (uint32_t rp = 0; rp < right_dim; rp++) {
                            // Get right environment contribution
                            double complex right_val = 1.0;
                            if (right_env && right_env->total_size > 1) {
                                if (right_env->rank == 2) {
                                    right_val = right_env->data[r * right_dim + rp];
                                } else if (right_env->rank == 4) {
                                    double complex sum = 0.0;
                                    uint32_t int_dim = right_env->dims[1];
                                    for (uint32_t li = 0; li < int_dim; li++) {
                                        uint32_t idx = r * right_env->dims[1] * right_env->dims[2] * right_env->dims[3]
                                                     + li * right_env->dims[2] * right_env->dims[3]
                                                     + rp * right_env->dims[3] + li;
                                        if (idx < right_env->total_size) sum += right_env->data[idx];
                                    }
                                    right_val = sum;
                                }
                            } else if (right_env && r == 0 && rp == 0) {
                                right_val = right_env->data[0];
                            } else if (!right_env && r == 0 && rp == 0) {
                                right_val = 1.0;
                            } else if (!right_env) {
                                right_val = (r == rp) ? 1.0 : 0.0;
                            }

                            // Get tensor elements A[l,s,r] and A*[l',s',r']
                            uint32_t idx_A[3] = {l, s, r};
                            uint32_t idx_Ac[3] = {lp, sp, rp};
                            double complex A_val = tensor_get(target, idx_A);
                            double complex Ac_val = conj(tensor_get(target, idx_Ac));

                            element += left_val * A_val * Ac_val * right_val;
                        }
                    }
                }
            }

            rho[s * 2 + sp] = element;
        }
    }

    tensor_free(left_env);
    tensor_free(right_env);

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
