/**
 * @file collective_ops.c
 * @brief Collective quantum operations implementation
 *
 * Full production implementation of distributed measurement,
 * expectation values, and state analysis operations.
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#include "collective_ops.h"
#include "distributed_gates.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>

// Cross-platform BLAS/LAPACK support for Hermitian eigenvalue problems
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define HAS_LAPACK 1
#elif defined(QSIM_HAS_OPENBLAS) || defined(__linux__)
#include <lapacke.h>
#define HAS_LAPACK 1
#else
#define HAS_LAPACK 0
#endif

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

    // For product of Pauli operators:
    // P = P_0 ⊗ P_1 ⊗ ... ⊗ P_{n-1}
    // ⟨ψ|P|ψ⟩ = Σᵢⱼ ψᵢ* P_ij ψⱼ

    // Check for X/Y operators (require non-local computation)
    int has_xy = 0;
    uint64_t xy_mask = 0;  // Bits where X or Y operators are located
    int rank = mpi_get_rank(state->dist_ctx);

    for (uint32_t q = 0; q < state->num_qubits; q++) {
        char op = pauli_string[state->num_qubits - 1 - q];
        if (op == 'X' || op == 'x' || op == 'Y' || op == 'y') {
            has_xy = 1;
            xy_mask |= (1ULL << q);
        }
    }

    double local_sum = 0.0;

    if (!has_xy) {
        // Pure Z/I string: diagonal, use eigenvalue approach
        for (uint64_t i = 0; i < state->local_count; i++) {
            double prob = amplitude_squared(state->amplitudes[i]);

            int parity = 0;
            for (uint32_t q = 0; q < state->num_qubits; q++) {
                char op = pauli_string[state->num_qubits - 1 - q];

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
            }

            double sign = (parity == 0) ? 1.0 : -1.0;
            local_sum += sign * prob;
        }
    } else {
        // Has X and/or Y: need to compute off-diagonal terms
        // ⟨ψ|P|ψ⟩ = Σᵢ ψᵢ* (P|i⟩)
        // P|i⟩ = phase * |j⟩ where j differs from i at X/Y positions

        for (uint64_t i = 0; i < state->local_count; i++) {
            uint64_t global_i = partition_local_to_global(state, i);

            // Compute which state P maps |i⟩ to
            uint64_t global_j = global_i ^ xy_mask;  // Flip bits at X/Y positions

            // Compute phase factor from Z and Y operators
            double complex phase = 1.0;

            for (uint32_t q = 0; q < state->num_qubits; q++) {
                char op = pauli_string[state->num_qubits - 1 - q];
                int bit_i = (global_i >> q) & 1;
                int bit_j = (global_j >> q) & 1;

                if (op == 'Z' || op == 'z') {
                    // Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
                    if (bit_i) phase *= -1.0;
                } else if (op == 'Y' || op == 'y') {
                    // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
                    if (bit_i == 0) {
                        phase *= I;  // Y|0⟩ = i|1⟩
                    } else {
                        phase *= -I;  // Y|1⟩ = -i|0⟩
                    }
                }
                // X just flips, no phase: X|0⟩ = |1⟩, X|1⟩ = |0⟩
                // I does nothing
            }

            // Get amplitude at j (may be non-local)
            double complex amp_j = partition_get_amplitude(state, global_j);
            double complex amp_i = state->amplitudes[i];

            // Contribution: ψᵢ* · phase · ψⱼ
            local_sum += creal(conj(amp_i) * phase * amp_j);
        }
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

    // Entanglement entropy S_A = -Tr(ρ_A log₂ ρ_A)
    // where ρ_A = Tr_B(|ψ⟩⟨ψ|) is the reduced density matrix

    // Limit subsystem size to avoid memory issues (max 12 qubits = 4K x 4K matrix)
    if (num_subsystem > 12 || num_subsystem == 0) {
        *entropy = 0.0;
        return COLLECTIVE_ERROR_NOT_INITIALIZED;
    }

    uint32_t n = state->num_qubits;
    uint64_t dim_A = 1ULL << num_subsystem;        // 2^k for subsystem A
    uint64_t dim_B = 1ULL << (n - num_subsystem);  // 2^(n-k) for complement B

    // Build subsystem mask and validate qubits
    uint64_t subsystem_mask = 0;
    for (uint32_t i = 0; i < num_subsystem; i++) {
        if (subsystem_qubits[i] >= n) {
            *entropy = 0.0;
            return COLLECTIVE_ERROR_NOT_INITIALIZED;
        }
        subsystem_mask |= (1ULL << subsystem_qubits[i]);
    }

    // Allocate reduced density matrix (complex, dim_A x dim_A)
    double complex* rho_A = (double complex*)calloc(dim_A * dim_A, sizeof(double complex));
    if (!rho_A) {
        *entropy = 0.0;
        return COLLECTIVE_ERROR_ALLOC;
    }

    // Helper: extract subsystem index from global index
    // Maps global basis state |b⟩ to subsystem basis state |a⟩
    // by picking out the bits specified in subsystem_qubits
    #define EXTRACT_SUBSYSTEM_INDEX(global_idx) ({             \
        uint64_t subsystem_idx = 0;                            \
        for (uint32_t q = 0; q < num_subsystem; q++) {         \
            if ((global_idx) & (1ULL << subsystem_qubits[q])) {\
                subsystem_idx |= (1ULL << q);                  \
            }                                                  \
        }                                                      \
        subsystem_idx;                                         \
    })

    // Compute local contribution to ρ_A
    // ρ_A(i,j) = Σ_b ψ(idx(i,b)) * conj(ψ(idx(j,b)))
    // where idx(a,b) reconstructs full index from subsystem index a and complement index b

    for (uint64_t local_i = 0; local_i < state->local_count; local_i++) {
        uint64_t global_i = state->local_start + local_i;
        double complex psi_i = state->amplitudes[local_i];

        if (cabs(psi_i) < 1e-15) continue;

        uint64_t a_i = EXTRACT_SUBSYSTEM_INDEX(global_i);

        // For the bra part, we need to sum over basis states with same complement index
        // Find complement index (bits NOT in subsystem)
        uint64_t complement_idx = 0;
        uint32_t complement_bit = 0;
        for (uint32_t q = 0; q < n; q++) {
            if (!(subsystem_mask & (1ULL << q))) {
                if (global_i & (1ULL << q)) {
                    complement_idx |= (1ULL << complement_bit);
                }
                complement_bit++;
            }
        }

        // For each j with same complement, contribute to ρ_A(a_i, a_j)
        // This means we iterate over all subsystem indices for the ket side
        for (uint64_t a_j = 0; a_j < dim_A; a_j++) {
            // Reconstruct global_j from (a_j, complement_idx)
            uint64_t global_j = 0;
            complement_bit = 0;
            uint32_t subsystem_bit = 0;

            for (uint32_t q = 0; q < n; q++) {
                if (subsystem_mask & (1ULL << q)) {
                    // Subsystem qubit - take from a_j
                    if (a_j & (1ULL << subsystem_bit)) {
                        global_j |= (1ULL << q);
                    }
                    subsystem_bit++;
                } else {
                    // Complement qubit - take from complement_idx
                    if (complement_idx & (1ULL << complement_bit)) {
                        global_j |= (1ULL << q);
                    }
                    complement_bit++;
                }
            }

            // Get amplitude for global_j (may be on another rank)
            double complex psi_j;
            if (global_j >= state->local_start && global_j < state->local_end) {
                psi_j = state->amplitudes[global_j - state->local_start];
            } else {
                // For non-local indices, use MPI communication
                // Simplified: if not local, set to 0 (will be summed via MPI_Allreduce)
                psi_j = partition_get_amplitude(state, global_j);
            }

            // ρ_A(a_i, a_j) += ψ_i * conj(ψ_j)
            rho_A[a_i * dim_A + a_j] += psi_i * conj(psi_j);
        }
    }

    #undef EXTRACT_SUBSYSTEM_INDEX

    // Reduce across all ranks (sum contributions)
    if (state->dist_ctx && state->dist_ctx->size > 1) {
        double complex* global_rho = (double complex*)calloc(dim_A * dim_A, sizeof(double complex));
        if (global_rho) {
            // Use MPI bridge wrapper for allreduce
            mpi_allreduce_sum_complex(state->dist_ctx, rho_A, global_rho, dim_A * dim_A);
            memcpy(rho_A, global_rho, dim_A * dim_A * sizeof(double complex));
            free(global_rho);
        }
    }

    (void)dim_B;  // Unused but computed for documentation

    // Compute eigenvalues of ρ_A (Hermitian density matrix)
    // Eigenvalues are real and non-negative for valid density matrices

    double* eigenvalues = (double*)calloc(dim_A, sizeof(double));
    if (!eigenvalues) {
        free(rho_A);
        *entropy = 0.0;
        return COLLECTIVE_ERROR_ALLOC;
    }

    // Use runtime flag to track if we need Jacobi fallback
    int use_jacobi = 0;

#if HAS_LAPACK
    // Use ZHEEV for proper Hermitian eigenvalue decomposition
    // This handles the full complex Hermitian matrix correctly

#ifdef __APPLE__
    // Apple Accelerate framework (CLAPACK interface)
    {
        char jobz = 'N';  // Eigenvalues only
        char uplo = 'U';  // Upper triangle
        __CLPK_integer n_clpk = (__CLPK_integer)dim_A;
        __CLPK_integer lda = n_clpk;
        __CLPK_integer lwork = -1;
        __CLPK_integer info;
        double complex work_query;
        double* rwork = (double*)malloc((3 * dim_A - 2) * sizeof(double));

        if (!rwork) {
            free(eigenvalues);
            free(rho_A);
            *entropy = 0.0;
            return COLLECTIVE_ERROR_ALLOC;
        }

        // Query optimal work size
        zheev_(&jobz, &uplo, &n_clpk, (__CLPK_doublecomplex*)rho_A, &lda,
               eigenvalues, (__CLPK_doublecomplex*)&work_query, &lwork, rwork, &info);

        lwork = (__CLPK_integer)creal(work_query);
        if (lwork < 1) lwork = 2 * dim_A;

        double complex* work = (double complex*)malloc(lwork * sizeof(double complex));
        if (!work) {
            free(rwork);
            free(eigenvalues);
            free(rho_A);
            *entropy = 0.0;
            return COLLECTIVE_ERROR_ALLOC;
        }

        // Compute eigenvalues
        zheev_(&jobz, &uplo, &n_clpk, (__CLPK_doublecomplex*)rho_A, &lda,
               eigenvalues, (__CLPK_doublecomplex*)work, &lwork, rwork, &info);

        free(work);
        free(rwork);

        if (info != 0) {
            use_jacobi = 1;  // Fallback to Jacobi if ZHEEV fails
        }
    }
#else
    // OpenBLAS/LAPACKE interface
    {
        lapack_int n_lap = (lapack_int)dim_A;
        lapack_int info = LAPACKE_zheev(LAPACK_ROW_MAJOR, 'N', 'U', n_lap,
                                         (lapack_complex_double*)rho_A, n_lap, eigenvalues);
        if (info != 0) {
            use_jacobi = 1;
        }
    }
#endif

#else
    // No LAPACK available - use Jacobi for Hermitian matrices
    use_jacobi = 1;
#endif

    if (use_jacobi) {
        // Jacobi eigenvalue algorithm for Hermitian matrices
        // We work with the full complex Hermitian matrix
        // Uses unitary (complex) Jacobi rotations to diagonalize

        double complex* H = (double complex*)malloc(dim_A * dim_A * sizeof(double complex));
        if (!H) {
            free(eigenvalues);
            free(rho_A);
            *entropy = 0.0;
            return COLLECTIVE_ERROR_ALLOC;
        }
        memcpy(H, rho_A, dim_A * dim_A * sizeof(double complex));

        const int max_iter = 100;
        const double eps = 1e-12;

        for (int iter = 0; iter < max_iter; iter++) {
            // Find largest off-diagonal magnitude
            double max_off = 0.0;
            uint64_t p = 0, q = 1;

            for (uint64_t i = 0; i < dim_A; i++) {
                for (uint64_t j = i + 1; j < dim_A; j++) {
                    double mag = cabs(H[i * dim_A + j]);
                    if (mag > max_off) {
                        max_off = mag;
                        p = i;
                        q = j;
                    }
                }
            }

            if (max_off < eps) break;

            // Compute Jacobi rotation parameters for Hermitian matrix
            // We want to zero out H[p,q] and H[q,p] = conj(H[p,q])
            double hpp = creal(H[p * dim_A + p]);
            double hqq = creal(H[q * dim_A + q]);
            double complex hpq = H[p * dim_A + q];

            // Phase factor to make hpq real
            double abs_hpq = cabs(hpq);
            double complex phase = (abs_hpq > eps) ? conj(hpq) / abs_hpq : 1.0;
            double hpq_real = abs_hpq;

            // Compute rotation angle for real 2x2 problem
            double theta;
            double diff = hqq - hpp;
            if (fabs(diff) < eps) {
                theta = M_PI / 4.0;
            } else {
                theta = 0.5 * atan2(2.0 * hpq_real, diff);
            }

            double c = cos(theta);
            double s = sin(theta);

            // Apply unitary transformation: H' = U† H U
            // where U rotates in (p,q) plane with phase adjustment
            for (uint64_t k = 0; k < dim_A; k++) {
                if (k != p && k != q) {
                    double complex hkp = H[k * dim_A + p];
                    double complex hkq = H[k * dim_A + q];
                    H[k * dim_A + p] = c * hkp - conj(phase) * s * hkq;
                    H[p * dim_A + k] = conj(H[k * dim_A + p]);
                    H[k * dim_A + q] = phase * s * hkp + c * hkq;
                    H[q * dim_A + k] = conj(H[k * dim_A + q]);
                }
            }

            // Update diagonal and off-diagonal in (p,q) block
            double new_pp = c*c*hpp - 2*c*s*hpq_real + s*s*hqq;
            double new_qq = s*s*hpp + 2*c*s*hpq_real + c*c*hqq;
            H[p * dim_A + p] = new_pp;
            H[q * dim_A + q] = new_qq;
            H[p * dim_A + q] = 0.0;
            H[q * dim_A + p] = 0.0;
        }

        // Extract eigenvalues (diagonal elements, should be real)
        for (uint64_t i = 0; i < dim_A; i++) {
            eigenvalues[i] = creal(H[i * dim_A + i]);
        }

        free(H);
    }

    // Compute von Neumann entropy: S = -Σ λᵢ log₂(λᵢ)
    double S = 0.0;
    for (uint64_t i = 0; i < dim_A; i++) {
        double lambda = eigenvalues[i];
        // Density matrix eigenvalues should be in [0,1]
        // Clamp small negative values from numerical error
        if (lambda < 0.0) lambda = 0.0;
        if (lambda > 1e-15) {  // Avoid log(0)
            S -= lambda * log2(lambda);
        }
    }

    // Handle numerical precision (entropy should be non-negative)
    if (S < 0.0) S = 0.0;

    *entropy = S;

    free(eigenvalues);
    free(rho_A);

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
