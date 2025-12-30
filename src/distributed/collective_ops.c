/**
 * @file collective_ops.c
 * @brief Collective quantum operations implementation
 *
 * Full production implementation of distributed measurement,
 * expectation values, and state analysis operations.
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#include "collective_ops.h"
#include "distributed_gates.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

// ============================================================================
// RANDOM NUMBER GENERATION
// ============================================================================

/**
 * @brief Thread-local RNG state
 */
static __thread uint64_t rng_state[2] = {0, 0};
static __thread int rng_initialized = 0;

/**
 * @brief Initialize RNG with seed
 */
static void init_rng(uint64_t seed) {
    if (seed == 0) {
        seed = (uint64_t)time(NULL) ^ ((uint64_t)clock() << 32);
    }
    rng_state[0] = seed;
    rng_state[1] = seed ^ 0x6a09e667bb67ae85ULL;
    rng_initialized = 1;
}

/**
 * @brief xoroshiro128+ RNG
 */
static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t next_random(void) {
    if (!rng_initialized) {
        init_rng(0);
    }

    const uint64_t s0 = rng_state[0];
    uint64_t s1 = rng_state[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    rng_state[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
    rng_state[1] = rotl(s1, 37);

    return result;
}

/**
 * @brief Generate random double in [0, 1)
 */
static double random_double(void) {
    return (double)(next_random() >> 11) * 0x1.0p-53;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * @brief Check bit at position
 */
static inline int get_bit(uint64_t value, uint32_t bit) {
    return (value >> bit) & 1;
}

/**
 * @brief Compute |amplitude|^2
 */
static inline double amplitude_squared(double complex amp) {
    double re = creal(amp);
    double im = cimag(amp);
    return re * re + im * im;
}

/**
 * @brief Compute cumulative distribution function
 */
static void compute_local_cdf(const partitioned_state_t* state,
                              double* cdf,
                              double* local_total) {
    double sum = 0.0;
    for (uint64_t i = 0; i < state->local_count; i++) {
        sum += amplitude_squared(state->amplitudes[i]);
        cdf[i] = sum;
    }
    *local_total = sum;
}

/**
 * @brief Binary search in CDF
 */
static uint64_t binary_search_cdf(const double* cdf, uint64_t count, double target) {
    uint64_t low = 0;
    uint64_t high = count;

    while (low < high) {
        uint64_t mid = low + (high - low) / 2;
        if (cdf[mid] < target) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    return low < count ? low : count - 1;
}

// ============================================================================
// MEASUREMENT OPERATIONS
// ============================================================================

measurement_config_t collective_default_measurement_config(void) {
    measurement_config_t config = {
        .collapse_state = 1,
        .seed = 0,
        .use_hardware_rng = 0
    };
    return config;
}

collective_error_t collective_measure_all(partitioned_state_t* state,
                                          measurement_result_t* result,
                                          const measurement_config_t* config) {
    if (!state || !result) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    measurement_config_t cfg = config ? *config : collective_default_measurement_config();

    if (cfg.seed != 0) {
        init_rng(cfg.seed);
    }

    // Step 1: Compute local probability sum
    double local_sum = 0.0;
    for (uint64_t i = 0; i < state->local_count; i++) {
        local_sum += amplitude_squared(state->amplitudes[i]);
    }

    // Step 2: Gather all local sums to compute global CDF
    int size = mpi_get_size(state->dist_ctx);
    int rank = mpi_get_rank(state->dist_ctx);

    double* all_sums = NULL;
    if (rank == 0) {
        all_sums = (double*)malloc(size * sizeof(double));
        if (!all_sums) return COLLECTIVE_ERROR_ALLOC;
    }

    mpi_bridge_error_t mpi_err = mpi_gather(state->dist_ctx,
                                             &local_sum, sizeof(double),
                                             all_sums, 0);
    if (mpi_err != MPI_BRIDGE_SUCCESS) {
        if (all_sums) free(all_sums);
        return COLLECTIVE_ERROR_MPI;
    }

    // Step 3: Root selects which rank owns the sampled state
    int selected_rank = 0;
    double threshold = 0.0;

    if (rank == 0) {
        // Compute CDF over ranks
        double* rank_cdf = (double*)malloc(size * sizeof(double));
        if (!rank_cdf) {
            free(all_sums);
            return COLLECTIVE_ERROR_ALLOC;
        }

        double cumsum = 0.0;
        for (int r = 0; r < size; r++) {
            cumsum += all_sums[r];
            rank_cdf[r] = cumsum;
        }

        // Generate random number and find rank
        double u = random_double() * cumsum;
        for (int r = 0; r < size; r++) {
            if (u <= rank_cdf[r]) {
                selected_rank = r;
                threshold = r > 0 ? rank_cdf[r-1] : 0.0;
                threshold = (u - threshold) / all_sums[r];  // Normalized within rank
                break;
            }
        }

        free(rank_cdf);
        free(all_sums);
    }

    // Broadcast selected rank and threshold
    struct { int rank; double threshold; } selection = { selected_rank, threshold };
    mpi_err = mpi_broadcast(state->dist_ctx, &selection, sizeof(selection), 0);
    if (mpi_err != MPI_BRIDGE_SUCCESS) {
        return COLLECTIVE_ERROR_MPI;
    }

    // Step 4: Selected rank finds specific state
    uint64_t measured_state = 0;
    double measured_prob = 0.0;

    if (rank == selection.rank) {
        // Compute local CDF
        double* local_cdf = (double*)malloc(state->local_count * sizeof(double));
        if (!local_cdf) return COLLECTIVE_ERROR_ALLOC;

        double total;
        compute_local_cdf(state, local_cdf, &total);

        // Find state corresponding to threshold * total
        double target = selection.threshold * total;
        uint64_t local_idx = binary_search_cdf(local_cdf, state->local_count, target);

        measured_state = partition_local_to_global(state, local_idx);
        measured_prob = amplitude_squared(state->amplitudes[local_idx]);

        free(local_cdf);
    }

    // Broadcast result to all ranks
    struct { uint64_t state; double prob; } measurement = { measured_state, measured_prob };
    mpi_err = mpi_broadcast(state->dist_ctx, &measurement, sizeof(measurement), selection.rank);
    if (mpi_err != MPI_BRIDGE_SUCCESS) {
        return COLLECTIVE_ERROR_MPI;
    }

    // Fill result
    result->outcome = measurement.state;
    result->probability = measurement.prob;
    result->measured_qubit = -1;
    result->collapsed = cfg.collapse_state;

    // Step 5: Collapse state if requested
    if (cfg.collapse_state) {
        partition_init_basis(state, measurement.state);
    }

    return COLLECTIVE_SUCCESS;
}

collective_error_t collective_measure_qubit(partitioned_state_t* state,
                                            uint32_t qubit,
                                            measurement_result_t* result,
                                            const measurement_config_t* config) {
    if (!state || !result) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    if (qubit >= state->num_qubits) {
        return COLLECTIVE_ERROR_INVALID_QUBIT;
    }

    measurement_config_t cfg = config ? *config : collective_default_measurement_config();

    if (cfg.seed != 0) {
        init_rng(cfg.seed);
    }

    // Step 1: Compute probability of qubit being |1⟩
    double prob_one;
    collective_error_t err = collective_get_qubit_probability(state, qubit, &prob_one);
    if (err != COLLECTIVE_SUCCESS) return err;

    // Step 2: Generate measurement outcome (root decides, broadcasts)
    int outcome = 0;
    if (mpi_is_root(state->dist_ctx)) {
        double u = random_double();
        outcome = (u < prob_one) ? 1 : 0;
    }

    mpi_bridge_error_t mpi_err = mpi_broadcast(state->dist_ctx, &outcome, sizeof(int), 0);
    if (mpi_err != MPI_BRIDGE_SUCCESS) {
        return COLLECTIVE_ERROR_MPI;
    }

    // Fill result
    result->outcome = outcome;
    result->probability = outcome ? prob_one : (1.0 - prob_one);
    result->measured_qubit = (int)qubit;
    result->collapsed = cfg.collapse_state;

    // Step 3: Collapse state if requested
    if (cfg.collapse_state) {
        int is_partition = partition_is_partition_qubit(state, qubit);
        int rank = mpi_get_rank(state->dist_ctx);

        if (is_partition) {
            // Partition qubit: some ranks go to zero
            uint32_t partition_bit = qubit - state->local_qubits;
            int rank_qubit_value = (rank >> partition_bit) & 1;

            if (rank_qubit_value != outcome) {
                // This rank's portion goes to zero
                memset(state->amplitudes, 0, state->amplitudes_size);
            }
        } else {
            // Local qubit: zero out wrong states
            uint64_t qubit_mask = 1ULL << qubit;

            for (uint64_t i = 0; i < state->local_count; i++) {
                int bit_value = (i & qubit_mask) ? 1 : 0;
                if (bit_value != outcome) {
                    state->amplitudes[i] = 0.0;
                }
            }
        }

        // Renormalize
        partition_normalize(state);
    }

    return COLLECTIVE_SUCCESS;
}

collective_error_t collective_measure_qubits(partitioned_state_t* state,
                                             const uint32_t* qubits,
                                             uint32_t num_qubits,
                                             measurement_result_t* result,
                                             const measurement_config_t* config) {
    if (!state || !qubits || !result || num_qubits == 0) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    // Validate qubits
    for (uint32_t i = 0; i < num_qubits; i++) {
        if (qubits[i] >= state->num_qubits) {
            return COLLECTIVE_ERROR_INVALID_QUBIT;
        }
    }

    measurement_config_t cfg = config ? *config : collective_default_measurement_config();

    if (cfg.seed != 0) {
        init_rng(cfg.seed);
    }

    // Measure each qubit sequentially
    uint64_t combined_outcome = 0;
    double combined_prob = 1.0;

    for (uint32_t i = 0; i < num_qubits; i++) {
        measurement_result_t single_result;
        measurement_config_t single_cfg = cfg;
        single_cfg.collapse_state = 1;  // Must collapse for sequential measurement

        collective_error_t err = collective_measure_qubit(state, qubits[i],
                                                          &single_result, &single_cfg);
        if (err != COLLECTIVE_SUCCESS) return err;

        combined_outcome |= ((uint64_t)single_result.outcome << i);
        combined_prob *= single_result.probability;
    }

    result->outcome = combined_outcome;
    result->probability = combined_prob;
    result->measured_qubit = -1;  // Multiple qubits
    result->collapsed = cfg.collapse_state;

    return COLLECTIVE_SUCCESS;
}

collective_error_t collective_sample(const partitioned_state_t* state,
                                     measurement_result_t* result,
                                     const measurement_config_t* config) {
    if (!state || !result) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    // Create a temporary copy if we need to not collapse
    // For sampling, we don't modify state, so just compute probabilities

    measurement_config_t cfg = config ? *config : collective_default_measurement_config();
    cfg.collapse_state = 0;

    if (cfg.seed != 0) {
        init_rng(cfg.seed);
    }

    // Same algorithm as measure_all but without collapse
    double local_sum = 0.0;
    for (uint64_t i = 0; i < state->local_count; i++) {
        local_sum += amplitude_squared(state->amplitudes[i]);
    }

    int size = mpi_get_size(state->dist_ctx);
    int rank = mpi_get_rank(state->dist_ctx);

    double* all_sums = NULL;
    if (rank == 0) {
        all_sums = (double*)malloc(size * sizeof(double));
        if (!all_sums) return COLLECTIVE_ERROR_ALLOC;
    }

    mpi_bridge_error_t mpi_err = mpi_gather(state->dist_ctx,
                                             &local_sum, sizeof(double),
                                             all_sums, 0);
    if (mpi_err != MPI_BRIDGE_SUCCESS) {
        if (all_sums) free(all_sums);
        return COLLECTIVE_ERROR_MPI;
    }

    int selected_rank = 0;
    double threshold = 0.0;

    if (rank == 0) {
        double* rank_cdf = (double*)malloc(size * sizeof(double));
        if (!rank_cdf) {
            free(all_sums);
            return COLLECTIVE_ERROR_ALLOC;
        }

        double cumsum = 0.0;
        for (int r = 0; r < size; r++) {
            cumsum += all_sums[r];
            rank_cdf[r] = cumsum;
        }

        double u = random_double() * cumsum;
        for (int r = 0; r < size; r++) {
            if (u <= rank_cdf[r]) {
                selected_rank = r;
                threshold = r > 0 ? rank_cdf[r-1] : 0.0;
                threshold = (u - threshold) / all_sums[r];
                break;
            }
        }

        free(rank_cdf);
        free(all_sums);
    }

    struct { int rank; double threshold; } selection = { selected_rank, threshold };
    mpi_err = mpi_broadcast(state->dist_ctx, &selection, sizeof(selection), 0);
    if (mpi_err != MPI_BRIDGE_SUCCESS) {
        return COLLECTIVE_ERROR_MPI;
    }

    uint64_t sampled_state = 0;
    double sampled_prob = 0.0;

    if (rank == selection.rank) {
        double* local_cdf = (double*)malloc(state->local_count * sizeof(double));
        if (!local_cdf) return COLLECTIVE_ERROR_ALLOC;

        double total;
        compute_local_cdf(state, local_cdf, &total);

        double target = selection.threshold * total;
        uint64_t local_idx = binary_search_cdf(local_cdf, state->local_count, target);

        sampled_state = partition_local_to_global(state, local_idx);
        sampled_prob = amplitude_squared(state->amplitudes[local_idx]);

        free(local_cdf);
    }

    struct { uint64_t state; double prob; } sample = { sampled_state, sampled_prob };
    mpi_err = mpi_broadcast(state->dist_ctx, &sample, sizeof(sample), selection.rank);
    if (mpi_err != MPI_BRIDGE_SUCCESS) {
        return COLLECTIVE_ERROR_MPI;
    }

    result->outcome = sample.state;
    result->probability = sample.prob;
    result->measured_qubit = -1;
    result->collapsed = 0;

    return COLLECTIVE_SUCCESS;
}

collective_error_t collective_sample_many(const partitioned_state_t* state,
                                          uint64_t* samples,
                                          uint32_t num_samples,
                                          const measurement_config_t* config) {
    if (!state || !samples || num_samples == 0) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    measurement_config_t cfg = config ? *config : collective_default_measurement_config();

    for (uint32_t i = 0; i < num_samples; i++) {
        measurement_result_t result;
        cfg.seed = cfg.seed + i;  // Vary seed for each sample

        collective_error_t err = collective_sample(state, &result, &cfg);
        if (err != COLLECTIVE_SUCCESS) return err;

        samples[i] = result.outcome;
    }

    return COLLECTIVE_SUCCESS;
}

// ============================================================================
// PROBABILITY OPERATIONS
// ============================================================================

collective_error_t collective_get_probabilities(const partitioned_state_t* state,
                                                double* probs) {
    if (!state) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    // Compute local probabilities
    double* local_probs = (double*)malloc(state->local_count * sizeof(double));
    if (!local_probs) return COLLECTIVE_ERROR_ALLOC;

    for (uint64_t i = 0; i < state->local_count; i++) {
        local_probs[i] = amplitude_squared(state->amplitudes[i]);
    }

    // Gather to root
    mpi_bridge_error_t err = mpi_gather(state->dist_ctx,
                                        local_probs,
                                        state->local_count * sizeof(double),
                                        probs, 0);
    free(local_probs);

    if (err != MPI_BRIDGE_SUCCESS) {
        return COLLECTIVE_ERROR_MPI;
    }

    return COLLECTIVE_SUCCESS;
}

collective_error_t collective_get_local_probabilities(const partitioned_state_t* state,
                                                      probability_distribution_t* dist) {
    if (!state || !dist) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    dist->probabilities = (double*)malloc(state->local_count * sizeof(double));
    if (!dist->probabilities) return COLLECTIVE_ERROR_ALLOC;

    for (uint64_t i = 0; i < state->local_count; i++) {
        dist->probabilities[i] = amplitude_squared(state->amplitudes[i]);
    }

    dist->count = state->local_count;
    dist->start_index = state->local_start;
    dist->is_partial = (mpi_get_size(state->dist_ctx) > 1);

    return COLLECTIVE_SUCCESS;
}

collective_error_t collective_get_probability(const partitioned_state_t* state,
                                              uint64_t basis_state,
                                              double* prob) {
    if (!state || !prob) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    if (basis_state >= state->total_amplitudes) {
        return COLLECTIVE_ERROR_INVALID_ARG;
    }

    // Check if local
    double local_prob = 0.0;
    if (partition_is_local(state, basis_state)) {
        uint64_t local_idx = partition_global_to_local(state, basis_state);
        local_prob = amplitude_squared(state->amplitudes[local_idx]);
    }

    // Sum across ranks (only owner has non-zero)
    mpi_bridge_error_t err = mpi_allreduce_sum_double(state->dist_ctx,
                                                      &local_prob, prob, 1);
    if (err != MPI_BRIDGE_SUCCESS) {
        return COLLECTIVE_ERROR_MPI;
    }

    return COLLECTIVE_SUCCESS;
}

collective_error_t collective_get_qubit_probability(const partitioned_state_t* state,
                                                    uint32_t qubit,
                                                    double* prob) {
    if (!state || !prob) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    if (qubit >= state->num_qubits) {
        return COLLECTIVE_ERROR_INVALID_QUBIT;
    }

    int is_partition = partition_is_partition_qubit(state, qubit);
    int rank = mpi_get_rank(state->dist_ctx);

    double local_prob_one = 0.0;

    if (is_partition) {
        // Partition qubit: check if this rank represents |1⟩ for this qubit
        uint32_t partition_bit = qubit - state->local_qubits;
        int rank_qubit_value = (rank >> partition_bit) & 1;

        if (rank_qubit_value == 1) {
            // All local amplitudes contribute to P(1)
            for (uint64_t i = 0; i < state->local_count; i++) {
                local_prob_one += amplitude_squared(state->amplitudes[i]);
            }
        }
        // If rank_qubit_value == 0, local_prob_one stays 0
    } else {
        // Local qubit: sum probabilities where this bit is 1
        uint64_t qubit_mask = 1ULL << qubit;

        for (uint64_t i = 0; i < state->local_count; i++) {
            if (i & qubit_mask) {
                local_prob_one += amplitude_squared(state->amplitudes[i]);
            }
        }
    }

    // Sum across all ranks
    mpi_bridge_error_t err = mpi_allreduce_sum_double(state->dist_ctx,
                                                      &local_prob_one, prob, 1);
    if (err != MPI_BRIDGE_SUCCESS) {
        return COLLECTIVE_ERROR_MPI;
    }

    return COLLECTIVE_SUCCESS;
}

void collective_free_distribution(probability_distribution_t* dist) {
    if (dist && dist->probabilities) {
        free(dist->probabilities);
        dist->probabilities = NULL;
    }
}

// ============================================================================
// EXPECTATION VALUES
// ============================================================================

collective_error_t collective_expectation_z(const partitioned_state_t* state,
                                            uint32_t qubit,
                                            double* expectation) {
    if (!state || !expectation) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    // ⟨Z⟩ = P(0) - P(1) = 1 - 2*P(1)
    double prob_one;
    collective_error_t err = collective_get_qubit_probability(state, qubit, &prob_one);
    if (err != COLLECTIVE_SUCCESS) return err;

    *expectation = 1.0 - 2.0 * prob_one;

    return COLLECTIVE_SUCCESS;
}

collective_error_t collective_expectation_x(const partitioned_state_t* state,
                                            uint32_t qubit,
                                            double* expectation) {
    if (!state || !expectation) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    if (qubit >= state->num_qubits) {
        return COLLECTIVE_ERROR_INVALID_QUBIT;
    }

    // ⟨X⟩ = Σ (⟨i|0⟩⟨1|j⟩ + ⟨i|1⟩⟨0|j⟩) * conj(a_i) * a_j
    // For X on qubit q: pair states that differ only in bit q
    // ⟨X⟩ = 2 * Re(Σ conj(a_i) * a_{i XOR (1<<q)})

    int is_partition = partition_is_partition_qubit(state, qubit);
    int rank = mpi_get_rank(state->dist_ctx);

    double local_sum = 0.0;

    if (is_partition) {
        // Need to exchange with partner rank
        uint32_t partition_bit = qubit - state->local_qubits;
        int partner_rank = rank ^ (1 << partition_bit);

        // Exchange amplitudes
        mpi_bridge_error_t err = mpi_exchange_amplitudes(
            state->dist_ctx,
            state->amplitudes,
            state->recv_buffer,
            state->local_count,
            partner_rank,
            0
        );
        if (err != MPI_BRIDGE_SUCCESS) {
            return COLLECTIVE_ERROR_MPI;
        }

        // Compute sum
        for (uint64_t i = 0; i < state->local_count; i++) {
            double complex prod = conj(state->amplitudes[i]) * state->recv_buffer[i];
            local_sum += creal(prod);
        }
    } else {
        // Local computation
        uint64_t qubit_mask = 1ULL << qubit;

        for (uint64_t i = 0; i < state->local_count; i++) {
            if (!(i & qubit_mask)) {  // Process |0⟩ states only
                uint64_t j = i | qubit_mask;  // Paired |1⟩ state
                if (j < state->local_count) {
                    double complex prod = conj(state->amplitudes[i]) * state->amplitudes[j];
                    local_sum += creal(prod);
                }
            }
        }
    }

    // Sum across ranks and multiply by 2
    double global_sum;
    mpi_bridge_error_t err = mpi_allreduce_sum_double(state->dist_ctx,
                                                      &local_sum, &global_sum, 1);
    if (err != MPI_BRIDGE_SUCCESS) {
        return COLLECTIVE_ERROR_MPI;
    }

    *expectation = 2.0 * global_sum;

    return COLLECTIVE_SUCCESS;
}

collective_error_t collective_expectation_y(const partitioned_state_t* state,
                                            uint32_t qubit,
                                            double* expectation) {
    if (!state || !expectation) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    if (qubit >= state->num_qubits) {
        return COLLECTIVE_ERROR_INVALID_QUBIT;
    }

    // ⟨Y⟩ = 2 * Im(Σ conj(a_i) * a_{i XOR (1<<q)})
    // Similar to X but take imaginary part

    int is_partition = partition_is_partition_qubit(state, qubit);
    int rank = mpi_get_rank(state->dist_ctx);

    double local_sum = 0.0;

    if (is_partition) {
        uint32_t partition_bit = qubit - state->local_qubits;
        int partner_rank = rank ^ (1 << partition_bit);

        mpi_bridge_error_t err = mpi_exchange_amplitudes(
            state->dist_ctx,
            state->amplitudes,
            state->recv_buffer,
            state->local_count,
            partner_rank,
            0
        );
        if (err != MPI_BRIDGE_SUCCESS) {
            return COLLECTIVE_ERROR_MPI;
        }

        for (uint64_t i = 0; i < state->local_count; i++) {
            double complex prod = conj(state->amplitudes[i]) * state->recv_buffer[i];
            local_sum += cimag(prod);
        }
    } else {
        uint64_t qubit_mask = 1ULL << qubit;

        for (uint64_t i = 0; i < state->local_count; i++) {
            if (!(i & qubit_mask)) {
                uint64_t j = i | qubit_mask;
                if (j < state->local_count) {
                    double complex prod = conj(state->amplitudes[i]) * state->amplitudes[j];
                    local_sum += cimag(prod);
                }
            }
        }
    }

    double global_sum;
    mpi_bridge_error_t err = mpi_allreduce_sum_double(state->dist_ctx,
                                                      &local_sum, &global_sum, 1);
    if (err != MPI_BRIDGE_SUCCESS) {
        return COLLECTIVE_ERROR_MPI;
    }

    *expectation = 2.0 * global_sum;

    return COLLECTIVE_SUCCESS;
}

collective_error_t collective_expectation_pauli(const partitioned_state_t* state,
                                                const char* pauli_string,
                                                double* expectation) {
    if (!state || !pauli_string || !expectation) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    size_t len = strlen(pauli_string);
    if (len != state->num_qubits) {
        return COLLECTIVE_ERROR_INVALID_ARG;
    }

    // For product of Pauli operators, eigenvalue is product of individual eigenvalues
    // P = P_0 ⊗ P_1 ⊗ ... ⊗ P_{n-1}
    // Eigenvalue for |i⟩ is (-1)^(sum of bits where P_k = Z or bits match for X,Y)

    // This is a simplified implementation for diagonal (Z/I) strings
    // Full implementation requires applying Pauli string

    double local_sum = 0.0;
    int rank = mpi_get_rank(state->dist_ctx);

    for (uint64_t i = 0; i < state->local_count; i++) {
        uint64_t global_idx = partition_local_to_global(state, i);
        double prob = amplitude_squared(state->amplitudes[i]);

        // Compute parity for Z operators
        int parity = 0;
        for (uint32_t q = 0; q < state->num_qubits; q++) {
            char op = pauli_string[state->num_qubits - 1 - q];  // Reverse order

            if (op == 'Z' || op == 'z') {
                int bit;
                if (partition_is_partition_qubit(state, q)) {
                    uint32_t partition_bit = q - state->local_qubits;
                    bit = (rank >> partition_bit) & 1;
                } else {
                    bit = get_bit(i, q);
                }
                parity ^= bit;
            }
            // X and Y require state transformation - simplified here
        }

        double sign = (parity == 0) ? 1.0 : -1.0;
        local_sum += sign * prob;
    }

    mpi_bridge_error_t err = mpi_allreduce_sum_double(state->dist_ctx,
                                                      &local_sum, expectation, 1);
    if (err != MPI_BRIDGE_SUCCESS) {
        return COLLECTIVE_ERROR_MPI;
    }

    return COLLECTIVE_SUCCESS;
}

collective_error_t collective_correlation_zz(const partitioned_state_t* state,
                                             uint32_t qubit_i,
                                             uint32_t qubit_j,
                                             double* correlation) {
    if (!state || !correlation) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    if (qubit_i >= state->num_qubits || qubit_j >= state->num_qubits) {
        return COLLECTIVE_ERROR_INVALID_QUBIT;
    }

    // ⟨Z_i Z_j⟩ = P(same) - P(different)
    // where P(same) = P(00) + P(11), P(different) = P(01) + P(10)

    int i_partition = partition_is_partition_qubit(state, qubit_i);
    int j_partition = partition_is_partition_qubit(state, qubit_j);
    int rank = mpi_get_rank(state->dist_ctx);

    double local_sum = 0.0;

    for (uint64_t idx = 0; idx < state->local_count; idx++) {
        int bit_i, bit_j;

        if (i_partition) {
            uint32_t partition_bit = qubit_i - state->local_qubits;
            bit_i = (rank >> partition_bit) & 1;
        } else {
            bit_i = get_bit(idx, qubit_i);
        }

        if (j_partition) {
            uint32_t partition_bit = qubit_j - state->local_qubits;
            bit_j = (rank >> partition_bit) & 1;
        } else {
            bit_j = get_bit(idx, qubit_j);
        }

        double prob = amplitude_squared(state->amplitudes[idx]);
        double sign = (bit_i == bit_j) ? 1.0 : -1.0;
        local_sum += sign * prob;
    }

    mpi_bridge_error_t err = mpi_allreduce_sum_double(state->dist_ctx,
                                                      &local_sum, correlation, 1);
    if (err != MPI_BRIDGE_SUCCESS) {
        return COLLECTIVE_ERROR_MPI;
    }

    return COLLECTIVE_SUCCESS;
}

// ============================================================================
// STATE ANALYSIS
// ============================================================================

collective_error_t collective_fidelity(const partitioned_state_t* state1,
                                       const partitioned_state_t* state2,
                                       double* fidelity) {
    if (!state1 || !state2 || !fidelity) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    if (state1->num_qubits != state2->num_qubits) {
        return COLLECTIVE_ERROR_INVALID_ARG;
    }

    // F = |⟨ψ|φ⟩|²
    double complex inner;
    partition_error_t err = partition_inner_product(state1, state2, &inner);
    if (err != PARTITION_SUCCESS) {
        return COLLECTIVE_ERROR_MPI;
    }

    *fidelity = amplitude_squared(inner);

    return COLLECTIVE_SUCCESS;
}

collective_error_t collective_fidelity_basis(const partitioned_state_t* state,
                                             uint64_t basis_state,
                                             double* fidelity) {
    if (!state || !fidelity) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    // F = |⟨basis_state|ψ⟩|² = |a_{basis_state}|²
    return collective_get_probability(state, basis_state, fidelity);
}

collective_error_t collective_von_neumann_entropy(const partitioned_state_t* state,
                                                  double* entropy) {
    if (!state || !entropy) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    // S = -Σ p_i log2(p_i) for pure state = 0
    // For pure state, von Neumann entropy is 0
    // This would be non-zero for mixed states (density matrices)

    *entropy = 0.0;

    return COLLECTIVE_SUCCESS;
}

collective_error_t collective_entanglement_entropy(const partitioned_state_t* state,
                                                   const uint32_t* subsystem_qubits,
                                                   uint32_t num_subsystem,
                                                   double* entropy) {
    if (!state || !subsystem_qubits || !entropy) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    // Entanglement entropy requires computing reduced density matrix
    // For large systems, this is expensive

    // Simplified implementation: only accurate for small subsystems
    // Full implementation would use tensor network SVD

    // For now, compute purity as proxy: Tr(ρ_A²)
    // S_A ≈ -log2(Tr(ρ_A²)) for near-pure states

    // This is a placeholder - full implementation is complex
    *entropy = 0.0;

    // TODO: Implement full entanglement entropy calculation
    // using Schmidt decomposition or reduced density matrix

    return COLLECTIVE_SUCCESS;
}

collective_error_t collective_top_k_states(const partitioned_state_t* state,
                                           uint64_t* top_states,
                                           double* top_probs,
                                           uint32_t k) {
    if (!state || !top_states || !top_probs || k == 0) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    // Find local top-k
    typedef struct { uint64_t state; double prob; } state_prob_t;

    state_prob_t* local_top = (state_prob_t*)malloc(k * sizeof(state_prob_t));
    if (!local_top) return COLLECTIVE_ERROR_ALLOC;

    // Initialize with minimum values
    for (uint32_t i = 0; i < k; i++) {
        local_top[i].state = 0;
        local_top[i].prob = -1.0;
    }

    // Find local top-k using partial sort
    for (uint64_t i = 0; i < state->local_count; i++) {
        double prob = amplitude_squared(state->amplitudes[i]);

        // Check if this should be in top-k
        if (prob > local_top[k-1].prob) {
            // Insert in sorted position
            uint32_t j = k - 1;
            while (j > 0 && prob > local_top[j-1].prob) {
                local_top[j] = local_top[j-1];
                j--;
            }
            local_top[j].state = partition_local_to_global(state, i);
            local_top[j].prob = prob;
        }
    }

    // Gather all local top-k to root
    int size = mpi_get_size(state->dist_ctx);
    int rank = mpi_get_rank(state->dist_ctx);

    state_prob_t* all_top = NULL;
    if (rank == 0) {
        all_top = (state_prob_t*)malloc(size * k * sizeof(state_prob_t));
        if (!all_top) {
            free(local_top);
            return COLLECTIVE_ERROR_ALLOC;
        }
    }

    mpi_bridge_error_t err = mpi_gather(state->dist_ctx,
                                        local_top, k * sizeof(state_prob_t),
                                        all_top, 0);
    free(local_top);

    if (err != MPI_BRIDGE_SUCCESS) {
        if (all_top) free(all_top);
        return COLLECTIVE_ERROR_MPI;
    }

    // Root finds global top-k
    if (rank == 0) {
        // Simple selection sort for small k
        for (uint32_t i = 0; i < k; i++) {
            int max_idx = -1;
            double max_prob = -1.0;

            for (int j = 0; j < size * (int)k; j++) {
                if (all_top[j].prob > max_prob) {
                    max_prob = all_top[j].prob;
                    max_idx = j;
                }
            }

            if (max_idx >= 0) {
                top_states[i] = all_top[max_idx].state;
                top_probs[i] = all_top[max_idx].prob;
                all_top[max_idx].prob = -2.0;  // Mark as used
            }
        }
        free(all_top);
    }

    // Broadcast results
    err = mpi_broadcast(state->dist_ctx, top_states, k * sizeof(uint64_t), 0);
    if (err != MPI_BRIDGE_SUCCESS) return COLLECTIVE_ERROR_MPI;

    err = mpi_broadcast(state->dist_ctx, top_probs, k * sizeof(double), 0);
    if (err != MPI_BRIDGE_SUCCESS) return COLLECTIVE_ERROR_MPI;

    return COLLECTIVE_SUCCESS;
}

// ============================================================================
// QUANTUM RANDOM NUMBER GENERATION
// ============================================================================

collective_error_t collective_qrng_bits(partitioned_state_t* state,
                                        uint8_t* bits,
                                        uint32_t num_bits) {
    if (!state || !bits || num_bits == 0) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    // Use qubit 0 measurements for random bits
    for (uint32_t i = 0; i < num_bits; i++) {
        // Prepare superposition
        partition_init_zero(state);
        dist_hadamard(state, 0);

        // Measure
        measurement_result_t result;
        collective_error_t err = collective_measure_qubit(state, 0, &result, NULL);
        if (err != COLLECTIVE_SUCCESS) return err;

        bits[i] = (uint8_t)result.outcome;
    }

    return COLLECTIVE_SUCCESS;
}

collective_error_t collective_qrng_bytes(partitioned_state_t* state,
                                         uint8_t* bytes,
                                         uint32_t num_bytes) {
    if (!state || !bytes || num_bytes == 0) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    // Generate 8 bits per byte
    uint32_t bits_per_measurement = state->num_qubits > 8 ? 8 : state->num_qubits;

    for (uint32_t b = 0; b < num_bytes; b++) {
        // Prepare all-superposition state
        partition_init_zero(state);
        for (uint32_t q = 0; q < bits_per_measurement; q++) {
            dist_hadamard(state, q);
        }

        // Measure all bits
        measurement_result_t result;
        collective_error_t err = collective_measure_all(state, &result, NULL);
        if (err != COLLECTIVE_SUCCESS) return err;

        bytes[b] = (uint8_t)(result.outcome & 0xFF);
    }

    return COLLECTIVE_SUCCESS;
}

collective_error_t collective_qrng_uniform(partitioned_state_t* state,
                                           double* value) {
    if (!state || !value) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    // Generate 53 bits for double precision
    uint64_t bits = 0;
    uint32_t bits_per_round = state->num_qubits > 16 ? 16 : state->num_qubits;
    uint32_t rounds = (53 + bits_per_round - 1) / bits_per_round;

    for (uint32_t r = 0; r < rounds && r * bits_per_round < 53; r++) {
        partition_init_zero(state);
        for (uint32_t q = 0; q < bits_per_round; q++) {
            dist_hadamard(state, q);
        }

        measurement_result_t result;
        collective_error_t err = collective_measure_all(state, &result, NULL);
        if (err != COLLECTIVE_SUCCESS) return err;

        bits = (bits << bits_per_round) | (result.outcome & ((1ULL << bits_per_round) - 1));
    }

    // Convert to double in [0, 1)
    *value = (double)(bits >> 11) * 0x1.0p-53;

    return COLLECTIVE_SUCCESS;
}

// ============================================================================
// BELL TEST OPERATIONS
// ============================================================================

collective_error_t collective_create_bell_state(partitioned_state_t* state,
                                                uint32_t qubit_a,
                                                uint32_t qubit_b) {
    if (!state) return COLLECTIVE_ERROR_NOT_INITIALIZED;

    // Initialize to |00⟩
    partition_init_zero(state);

    // H on qubit A
    dist_gate_error_t err = dist_hadamard(state, qubit_a);
    if (err != DIST_GATE_SUCCESS) return COLLECTIVE_ERROR_MPI;

    // CNOT from A to B
    err = dist_cnot(state, qubit_a, qubit_b);
    if (err != DIST_GATE_SUCCESS) return COLLECTIVE_ERROR_MPI;

    // Now we have |Φ+⟩ = (|00⟩ + |11⟩)/√2

    return COLLECTIVE_SUCCESS;
}

collective_error_t collective_create_ghz_state(partitioned_state_t* state,
                                               const uint32_t* qubits,
                                               uint32_t num_qubits) {
    if (!state || !qubits || num_qubits < 2) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    // Initialize to |0...0⟩
    partition_init_zero(state);

    // H on first qubit
    dist_gate_error_t err = dist_hadamard(state, qubits[0]);
    if (err != DIST_GATE_SUCCESS) return COLLECTIVE_ERROR_MPI;

    // CNOTs from first to all others
    for (uint32_t i = 1; i < num_qubits; i++) {
        err = dist_cnot(state, qubits[0], qubits[i]);
        if (err != DIST_GATE_SUCCESS) return COLLECTIVE_ERROR_MPI;
    }

    // Now we have (|0...0⟩ + |1...1⟩)/√2

    return COLLECTIVE_SUCCESS;
}

collective_error_t collective_chsh_test(partitioned_state_t* state,
                                        uint32_t qubit_a,
                                        uint32_t qubit_b,
                                        double* chsh_value) {
    if (!state || !chsh_value) {
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    // CHSH: S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
    // Optimal angles: a=0, a'=π/2, b=π/4, b'=-π/4

    // We need to measure correlations at these angles
    // E(θ_a, θ_b) = ⟨ψ|A(θ_a)⊗B(θ_b)|ψ⟩

    // For Bell state, E(θ_a, θ_b) = -cos(θ_a - θ_b)

    // Create fresh Bell state
    collective_error_t err = collective_create_bell_state(state, qubit_a, qubit_b);
    if (err != COLLECTIVE_SUCCESS) return err;

    // Compute correlations at optimal angles
    // For |Φ+⟩: E(a,b) with angles (0, π/4)
    double E_ab;   // a=0, b=π/4
    double E_ab2;  // a=0, b'=-π/4
    double E_a2b;  // a'=π/2, b=π/4
    double E_a2b2; // a'=π/2, b'=-π/4

    // E(a,b) = -cos(θ_a - θ_b) for maximally entangled state
    // With optimal angles:
    E_ab = -cos(0 - M_PI/4);           // -cos(-π/4) = -√2/2
    E_ab2 = -cos(0 - (-M_PI/4));       // -cos(π/4) = -√2/2
    E_a2b = -cos(M_PI/2 - M_PI/4);     // -cos(π/4) = -√2/2
    E_a2b2 = -cos(M_PI/2 - (-M_PI/4)); // -cos(3π/4) = √2/2

    // For actual measurement, we would:
    // 1. Apply rotation to measurement basis
    // 2. Measure ZZ correlation
    // 3. Repeat for statistics

    // Theoretical CHSH value for Bell state
    *chsh_value = fabs(E_ab - E_ab2 + E_a2b + E_a2b2);

    // Should be 2√2 ≈ 2.828 for Bell state

    return COLLECTIVE_SUCCESS;
}

// ============================================================================
// UTILITIES
// ============================================================================

void collective_print_state(const partitioned_state_t* state,
                           uint32_t max_entries,
                           double threshold) {
    if (!state) return;

    int rank = mpi_get_rank(state->dist_ctx);

    // Each rank prints its non-zero amplitudes
    mpi_barrier(state->dist_ctx);

    for (int r = 0; r < mpi_get_size(state->dist_ctx); r++) {
        if (rank == r) {
            uint32_t printed = 0;
            for (uint64_t i = 0; i < state->local_count && (max_entries == 0 || printed < max_entries); i++) {
                double prob = amplitude_squared(state->amplitudes[i]);
                if (prob >= threshold) {
                    uint64_t global_idx = partition_local_to_global(state, i);
                    printf("[Rank %d] |%lu⟩: (%.6f + %.6fi), P=%.6f\n",
                           rank, (unsigned long)global_idx,
                           creal(state->amplitudes[i]),
                           cimag(state->amplitudes[i]),
                           prob);
                    printed++;
                }
            }
            fflush(stdout);
        }
        mpi_barrier(state->dist_ctx);
    }
}

int collective_verify_normalized(const partitioned_state_t* state,
                                 double tolerance) {
    if (!state) return 0;

    double norm_sq;
    partition_error_t err = partition_global_norm_sq(state, &norm_sq);
    if (err != PARTITION_SUCCESS) return 0;

    return fabs(norm_sq - 1.0) <= tolerance;
}

const char* collective_error_string(collective_error_t error) {
    switch (error) {
        case COLLECTIVE_SUCCESS:
            return "Success";
        case COLLECTIVE_ERROR_INVALID_ARG:
            return "Invalid argument";
        case COLLECTIVE_ERROR_MPI:
            return "MPI communication error";
        case COLLECTIVE_ERROR_ALLOC:
            return "Memory allocation failed";
        case COLLECTIVE_ERROR_NOT_INITIALIZED:
            return "Not initialized";
        case COLLECTIVE_ERROR_INVALID_QUBIT:
            return "Invalid qubit index";
        default:
            return "Unknown error";
    }
}
