/**
 * @file measurement.c
 * @brief Quantum measurement operations
 *
 * Implements projective measurements, partial measurements, and
 * measurement-based state collapse with proper probability handling.
 *
 * Features:
 * - Single qubit measurement with state collapse
 * - Multi-qubit measurement in computational basis
 * - Partial measurement (measure subset of qubits)
 * - Measurement statistics and expectation values
 * - SIMD-accelerated probability computation
 *
 * @stability evolving
 * @since v0.1.2
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#include "measurement.h"
#include "state.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#ifdef HAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

#define MEASUREMENT_SMALL_NORM 1e-15

// ============================================================================
// GPU HOST-BUFFER SYNC
// ============================================================================
//
// For a GPU-backed state (state->gpu_state != NULL) the host `amplitudes`
// buffer is stale until quantum_state_sync_to_host() is called.  Every read
// path here refreshes the host mirror on entry; every path that mutates the
// host amplitudes (measurement collapse) pushes the result back to the GPU on
// exit.  Both sync entry points are no-ops that return QS_SUCCESS when there
// is no GPU backing, so these helpers are free on CPU states.

static inline void measurement_gpu_pull(const quantum_state_t *state) {
    // Reads do not logically mutate the quantum state, but they do need the
    // host mirror refreshed from device memory, so cast away const to sync.
    if (state && state->gpu_state) {
        quantum_state_sync_to_host((quantum_state_t *)state);
    }
}

static inline void measurement_gpu_push(quantum_state_t *state) {
    if (state && state->gpu_state) {
        quantum_state_sync_from_host(state);
    }
}

// ============================================================================
// PROBABILITY COMPUTATION
// ============================================================================

/**
 * @brief Compute probability of measuring qubit in |1⟩ state
 *
 * P(1) = Σ |α_i|² for all i where qubit q is set
 *
 * @param state Quantum state
 * @param qubit Qubit index (0-indexed from LSB)
 * @return Probability in [0, 1]
 */
double measurement_probability_one(const quantum_state_t* state, int qubit) {
    if (!state || !state->amplitudes || qubit < 0 ||
        qubit >= (int)state->num_qubits) {
        return 0.0;
    }

    measurement_gpu_pull(state);

    const uint64_t state_dim = state->state_dim;
    const uint64_t qubit_mask = 1ULL << qubit;
    const complex_t* amp = state->amplitudes;

    double prob = 0.0;

#ifdef HAS_ACCELERATE
    // Use vDSP for SIMD acceleration on Apple platforms
    // Compute |amplitude|² for states where qubit is 1
    for (uint64_t i = 0; i < state_dim; i++) {
        if (i & qubit_mask) {
            double re = creal(amp[i]);
            double im = cimag(amp[i]);
            prob += re * re + im * im;
        }
    }
#else
    // Standard computation
    for (uint64_t i = 0; i < state_dim; i++) {
        if (i & qubit_mask) {
            prob += cabs(amp[i]) * cabs(amp[i]);
        }
    }
#endif

    return prob;
}

/**
 * @brief Compute probability of measuring qubit in |0⟩ state
 *
 * @param state Quantum state
 * @param qubit Qubit index
 * @return Probability in [0, 1]
 */
double measurement_probability_zero(const quantum_state_t* state, int qubit) {
    return 1.0 - measurement_probability_one(state, qubit);
}

/**
 * @brief Compute all single-qubit probabilities
 *
 * @param state Quantum state
 * @param probabilities Output array of size num_qubits (probability of |1⟩)
 */
void measurement_all_probabilities(const quantum_state_t* state, double* probabilities) {
    if (!state || !probabilities) return;

    for (int q = 0; q < (int)state->num_qubits; q++) {
        probabilities[q] = measurement_probability_one(state, q);
    }
}

/**
 * @brief Compute full probability distribution
 *
 * @param state Quantum state
 * @param distribution Output array of size state_dim (|α_i|²)
 */
void measurement_probability_distribution(const quantum_state_t* state,
                                          double* distribution) {
    if (!state || !distribution) return;

    measurement_gpu_pull(state);

    const uint64_t state_dim = state->state_dim;
    const complex_t* amp = state->amplitudes;

#ifdef HAS_ACCELERATE
    // Optimized: compute squared magnitudes
    for (uint64_t i = 0; i < state_dim; i++) {
        double re = creal(amp[i]);
        double im = cimag(amp[i]);
        distribution[i] = re * re + im * im;
    }
#else
    for (uint64_t i = 0; i < state_dim; i++) {
        distribution[i] = cabs(amp[i]) * cabs(amp[i]);
    }
#endif
}

// ============================================================================
// SINGLE QUBIT MEASUREMENT
// ============================================================================

/**
 * @brief Measure single qubit with state collapse
 *
 * Performs projective measurement on a single qubit:
 * 1. Compute probability of |1⟩
 * 2. Generate random outcome based on probability
 * 3. Collapse state by zeroing non-matching amplitudes
 * 4. Renormalize
 *
 * @param state Quantum state (modified in-place)
 * @param qubit Qubit index
 * @param random_value Random value in [0, 1) for determining outcome
 * @return Measurement result (0 or 1)
 */
int measurement_single_qubit(quantum_state_t* state, int qubit, double random_value) {
    if (!state || !state->amplitudes || qubit < 0 ||
        qubit >= (int)state->num_qubits) {
        return -1;
    }

    // Refresh host mirror for a GPU-backed state before reading/collapsing.
    measurement_gpu_pull(state);

    const uint64_t state_dim = state->state_dim;
    const uint64_t qubit_mask = 1ULL << qubit;
    complex_t* amp = state->amplitudes;

    // Compute probability of measuring |1⟩ (host buffer is now fresh).
    double prob_one = measurement_probability_one(state, qubit);

    // Determine outcome
    int result = (random_value < prob_one) ? 1 : 0;

    // Collapse state: zero out amplitudes that don't match outcome
    double norm_factor = 0.0;

    for (uint64_t i = 0; i < state_dim; i++) {
        int qubit_value = (i & qubit_mask) ? 1 : 0;

        if (qubit_value != result) {
            // Zero out non-matching amplitudes
            amp[i] = 0.0;
        } else {
            // Accumulate norm for renormalization
            norm_factor += cabs(amp[i]) * cabs(amp[i]);
        }
    }

    // Renormalize.  A surviving norm below the floor means we collapsed onto
    // an outcome with (numerically) zero weight -- the residual state is not a
    // valid density and cannot be renormalized.  Signal an error rather than
    // silently returning a broken state.
    if (norm_factor <= MEASUREMENT_SMALL_NORM) {
        return -1;
    }
    double inv_norm = 1.0 / sqrt(norm_factor);
    for (uint64_t i = 0; i < state_dim; i++) {
        amp[i] *= inv_norm;
    }

    // Push the collapsed host buffer back to the GPU (no-op on CPU states).
    measurement_gpu_push(state);

    return result;
}

/**
 * @brief Measure single qubit without collapse (weak measurement simulation)
 *
 * Returns the probable outcome without modifying state.
 *
 * @param state Quantum state (not modified)
 * @param qubit Qubit index
 * @param random_value Random value in [0, 1)
 * @return Measurement result (0 or 1)
 */
int measurement_single_qubit_no_collapse(const quantum_state_t* state,
                                         int qubit, double random_value) {
    double prob_one = measurement_probability_one(state, qubit);
    return (random_value < prob_one) ? 1 : 0;
}

// ============================================================================
// MULTI-QUBIT MEASUREMENT
// ============================================================================

/**
 * @brief Measure all qubits with state collapse
 *
 * Measures all qubits simultaneously in computational basis.
 * State collapses to a computational basis state.
 *
 * @param state Quantum state (modified to collapsed state)
 * @param random_value Random value in [0, 1) for sampling
 * @return Measurement result as bit pattern
 */
uint64_t measurement_all_qubits(quantum_state_t* state, double random_value) {
    if (!state || !state->amplitudes) {
        return 0;
    }

    measurement_gpu_pull(state);

    const uint64_t state_dim = state->state_dim;
    const complex_t* amp = state->amplitudes;

    // Find outcome by cumulative probability
    double cumulative = 0.0;
    uint64_t result = 0;
    int found = 0;
    uint64_t last_nonzero = 0;
    int have_nonzero = 0;

    for (uint64_t i = 0; i < state_dim; i++) {
        double prob = cabs(amp[i]) * cabs(amp[i]);
        if (prob > 0.0) { last_nonzero = i; have_nonzero = 1; }
        cumulative += prob;

        if (random_value < cumulative) {
            result = i;
            found = 1;
            break;
        }
    }

    // Rounding can leave random_value >= total probability.  Collapse onto the
    // last basis state that actually carries amplitude -- never a fixed
    // |1...1> that may have zero amplitude.
    if (!found) {
        result = have_nonzero ? last_nonzero : 0;
    }

    // Collapse to computational basis state
    for (uint64_t i = 0; i < state_dim; i++) {
        state->amplitudes[i] = (i == result) ? 1.0 : 0.0;
    }

    measurement_gpu_push(state);

    return result;
}

/**
 * @brief Measure subset of qubits
 *
 * Measures specified qubits, collapsing only those degrees of freedom.
 *
 * @param state Quantum state (modified)
 * @param qubits Array of qubit indices to measure
 * @param num_qubits Number of qubits to measure
 * @param random_value Random value for sampling
 * @return Measurement result as bit pattern for measured qubits
 */
uint64_t measurement_partial(quantum_state_t* state, const int* qubits,
                             int num_measure, double random_value) {
    if (!state || !qubits || num_measure <= 0) {
        return 0;
    }

    /* Validate qubit indices. (A historical `measure_mask` bitmask was
     * computed here but never read; it has been removed.) */
    for (int i = 0; i < num_measure; i++) {
        if (qubits[i] < 0 || qubits[i] >= (int)state->num_qubits) {
            return 0;
        }
    }

    measurement_gpu_pull(state);

    const uint64_t state_dim = state->state_dim;
    complex_t* amp = state->amplitudes;

    // Count possible outcomes and compute probabilities
    uint64_t num_outcomes = 1ULL << num_measure;
    double* outcome_probs = calloc(num_outcomes, sizeof(double));
    if (!outcome_probs) return 0; /* OOM: fall back to a deterministic
                                   * zero outcome rather than crash. */

    for (uint64_t i = 0; i < state_dim; i++) {
        // Extract measured qubit values
        uint64_t outcome = 0;
        for (int j = 0; j < num_measure; j++) {
            if (i & (1ULL << qubits[j])) {
                outcome |= (1ULL << j);
            }
        }
        outcome_probs[outcome] += cabs(amp[i]) * cabs(amp[i]);
    }

    // Sample outcome
    double cumulative = 0.0;
    uint64_t result = 0;
    int found = 0;
    uint64_t last_nonzero = 0;
    int have_nonzero = 0;
    for (uint64_t o = 0; o < num_outcomes; o++) {
        if (outcome_probs[o] > 0.0) { last_nonzero = o; have_nonzero = 1; }
        cumulative += outcome_probs[o];
        if (random_value < cumulative) {
            result = o;
            found = 1;
            break;
        }
    }

    // Rounding can leave random_value >= the cumulative total.  Select the
    // last outcome that actually carries probability rather than defaulting to
    // outcome 0, which may be impossible for this state.
    if (!found) {
        result = have_nonzero ? last_nonzero : 0;
    }

    // Collapse: zero amplitudes that don't match, renormalize rest
    double norm_factor = 0.0;
    for (uint64_t i = 0; i < state_dim; i++) {
        // Extract measured qubit values
        uint64_t outcome = 0;
        for (int j = 0; j < num_measure; j++) {
            if (i & (1ULL << qubits[j])) {
                outcome |= (1ULL << j);
            }
        }

        if (outcome != result) {
            amp[i] = 0.0;
        } else {
            norm_factor += cabs(amp[i]) * cabs(amp[i]);
        }
    }

    // Renormalize.  The outcome selection above guarantees a positive-weight
    // outcome for any state carrying nonzero total probability, so norm_factor
    // is positive for valid input; the guard only trips on an all-zero
    // (degenerate) input, which has no meaningful post-collapse normalization.
    if (norm_factor > MEASUREMENT_SMALL_NORM) {
        double inv_norm = 1.0 / sqrt(norm_factor);
        for (uint64_t i = 0; i < state_dim; i++) {
            amp[i] *= inv_norm;
        }
    }

    // Push the collapsed host buffer back to the GPU (no-op on CPU states).
    measurement_gpu_push(state);

    free(outcome_probs);
    return result;
}

// ============================================================================
// MEASUREMENT STATISTICS
// ============================================================================

/**
 * @brief Compute expectation value of Z operator on qubit
 *
 * <Z> = P(0) - P(1) = 1 - 2*P(1)
 *
 * @param state Quantum state
 * @param qubit Qubit index
 * @return Expectation value in [-1, 1]
 */
double measurement_expectation_z(const quantum_state_t* state, int qubit) {
    double prob_one = measurement_probability_one(state, qubit);
    return 1.0 - 2.0 * prob_one;
}

/**
 * @brief Compute expectation value of X operator on qubit
 *
 * Requires computing <ψ|X|ψ> = Σ (α*_i α_j + α_i α*_j) for X-connected pairs
 *
 * @param state Quantum state
 * @param qubit Qubit index
 * @return Expectation value in [-1, 1]
 */
double measurement_expectation_x(const quantum_state_t* state, int qubit) {
    if (!state || !state->amplitudes) return 0.0;

    measurement_gpu_pull(state);

    const uint64_t state_dim = state->state_dim;
    const uint64_t qubit_mask = 1ULL << qubit;
    const complex_t* amp = state->amplitudes;

    double expectation = 0.0;

    // X flips the qubit, so we sum α*_i α_{i⊕mask}
    for (uint64_t i = 0; i < state_dim; i++) {
        uint64_t j = i ^ qubit_mask;
        if (j > i) {
            // Count each pair once
            complex_t term = conj(amp[i]) * amp[j] + conj(amp[j]) * amp[i];
            expectation += creal(term);
        }
    }

    return expectation;
}

/**
 * @brief Compute expectation value of Y operator on qubit
 *
 * @param state Quantum state
 * @param qubit Qubit index
 * @return Expectation value in [-1, 1]
 */
double measurement_expectation_y(const quantum_state_t* state, int qubit) {
    if (!state || !state->amplitudes) return 0.0;

    measurement_gpu_pull(state);

    const uint64_t state_dim = state->state_dim;
    const uint64_t qubit_mask = 1ULL << qubit;
    const complex_t* amp = state->amplitudes;

    double expectation = 0.0;

    // Y = [[0, -i], [i, 0]]
    for (uint64_t i = 0; i < state_dim; i++) {
        uint64_t j = i ^ qubit_mask;
        if (j > i) {
            int i_bit = (i & qubit_mask) ? 1 : 0;
            // For Y: contribution is i*(-1)^i_bit * (α*_i α_j - α*_j α_i)
            complex_t term = conj(amp[i]) * amp[j] - conj(amp[j]) * amp[i];
            double factor = i_bit ? -1.0 : 1.0;
            expectation += factor * cimag(term);
        }
    }

    return expectation;
}

/**
 * @brief Compute ZZ correlation between two qubits
 *
 * <Z_i Z_j> = P(00) + P(11) - P(01) - P(10)
 *
 * @param state Quantum state
 * @param qubit1 First qubit index
 * @param qubit2 Second qubit index
 * @return Correlation in [-1, 1]
 */
double measurement_correlation_zz(const quantum_state_t* state,
                                  int qubit1, int qubit2) {
    if (!state || !state->amplitudes) return 0.0;

    measurement_gpu_pull(state);

    const uint64_t state_dim = state->state_dim;
    const uint64_t mask1 = 1ULL << qubit1;
    const uint64_t mask2 = 1ULL << qubit2;
    const complex_t* amp = state->amplitudes;

    double p00 = 0.0, p01 = 0.0, p10 = 0.0, p11 = 0.0;

    for (uint64_t i = 0; i < state_dim; i++) {
        double prob = cabs(amp[i]) * cabs(amp[i]);
        int b1 = (i & mask1) ? 1 : 0;
        int b2 = (i & mask2) ? 1 : 0;

        if (!b1 && !b2) p00 += prob;
        else if (!b1 && b2) p01 += prob;
        else if (b1 && !b2) p10 += prob;
        else p11 += prob;
    }

    return p00 + p11 - p01 - p10;
}

// ============================================================================
// SAMPLING
// ============================================================================

/**
 * @brief Sample measurement outcomes without state collapse
 *
 * Generates multiple measurement samples without modifying state.
 * Useful for estimating probability distributions.
 *
 * @param state Quantum state
 * @param outcomes Output array for samples
 * @param num_samples Number of samples to generate
 * @param random_values Array of random values (size num_samples)
 */
void measurement_sample(const quantum_state_t* state, uint64_t* outcomes,
                        int num_samples, const double* random_values) {
    if (!state || !outcomes || !random_values) return;

    measurement_gpu_pull(state);

    const uint64_t state_dim = state->state_dim;
    const complex_t* amp = state->amplitudes;

    // Precompute cumulative distribution
    double* cdf = malloc(state_dim * sizeof(double));
    if (!cdf) {
        /* OOM: zero-fill outcomes and return without sampling. */
        for (int s = 0; s < num_samples; s++) outcomes[s] = 0;
        return;
    }
    double cumulative = 0.0;

    for (uint64_t i = 0; i < state_dim; i++) {
        cumulative += cabs(amp[i]) * cabs(amp[i]);
        cdf[i] = cumulative;
    }

    // Sample using binary search for efficiency
    for (int s = 0; s < num_samples; s++) {
        double r = random_values[s];

        // Binary search in CDF
        uint64_t low = 0, high = state_dim - 1;
        while (low < high) {
            uint64_t mid = (low + high) / 2;
            if (cdf[mid] < r) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        outcomes[s] = low;
    }

    free(cdf);
}

/**
 * @brief Estimate measurement probabilities from samples
 *
 * @param samples Array of measurement outcomes
 * @param num_samples Number of samples
 * @param state_dim Dimension of state space
 * @param probabilities Output probability array
 */
void measurement_estimate_probabilities(const uint64_t* samples, int num_samples,
                                        uint64_t state_dim, double* probabilities) {
    if (!samples || !probabilities || num_samples <= 0) return;

    // Zero initialize
    memset(probabilities, 0, state_dim * sizeof(double));

    // Count occurrences
    for (int i = 0; i < num_samples; i++) {
        if (samples[i] < state_dim) {
            probabilities[samples[i]] += 1.0;
        }
    }

    // Normalize
    double inv_n = 1.0 / num_samples;
    for (uint64_t i = 0; i < state_dim; i++) {
        probabilities[i] *= inv_n;
    }
}

/* =========================================================== */
/* POVM + weak measurement  (v0.2)                              */
/* =========================================================== */

/* Compute out = M * psi for a row-major state_dim x state_dim
 * matrix M.  Overwrites out (no aliasing allowed with psi). */
static void apply_matrix(const complex_t *M, const complex_t *psi,
                          complex_t *out, uint64_t state_dim) {
    for (uint64_t i = 0; i < state_dim; i++) {
        complex_t acc = 0.0;
        const complex_t *row = &M[i * state_dim];
        for (uint64_t j = 0; j < state_dim; j++) {
            acc += row[j] * psi[j];
        }
        out[i] = acc;
    }
}

/* Compute p_k = <psi| K_k^dag K_k |psi> = || K_k psi ||^2. */
static double povm_outcome_prob(const complex_t *Kk,
                                 const complex_t *psi,
                                 complex_t *scratch,
                                 uint64_t state_dim) {
    apply_matrix(Kk, psi, scratch, state_dim);
    double acc = 0.0;
    for (uint64_t i = 0; i < state_dim; i++) {
        double re = creal(scratch[i]);
        double im = cimag(scratch[i]);
        acc += re * re + im * im;
    }
    return acc;
}

/* Verify sum_k K_k^dag K_k = I within tolerance.  This is O(K*D^3);
 * povm_check_completeness_cached() below runs it at most once per POVM. */
static qs_error_t povm_check_completeness_full(const povm_t *povm) {
    if (!povm || !povm->kraus_ops || povm->num_outcomes == 0) {
        return QS_ERROR_INVALID_STATE;
    }
    const uint64_t D = povm->state_dim;
    for (uint64_t a = 0; a < D; a++) {
        for (uint64_t b = 0; b < D; b++) {
            complex_t sum = 0.0;
            for (size_t k = 0; k < povm->num_outcomes; k++) {
                const complex_t *K = povm->kraus_ops[k];
                complex_t entry = 0.0;
                for (uint64_t c = 0; c < D; c++) {
                    entry += conj(K[c * D + a]) * K[c * D + b];
                }
                sum += entry;
            }
            complex_t expected = (a == b) ? 1.0 : 0.0;
            if (cabs(sum - expected) > 1e-9) return QS_ERROR_NOT_NORMALIZED;
        }
    }
    return QS_SUCCESS;
}

/* Memoized completeness check.  Runs the O(K*D^3) verification only on
 * the first call for a given POVM, then caches the verdict on the
 * struct (cast away const: the POVM object itself is caller-owned and
 * not const-qualified in practice; only the pointer parameter is). */
static qs_error_t povm_check_completeness_cached(const povm_t *povm) {
    /* Cache state never overrides the structural invariants. In particular,
     * a stale/forged cached-success verdict with zero outcomes must not reach
     * measurement_povm(), where num_outcomes - 1 would underflow. */
    if (!povm || !povm->kraus_ops || povm->num_outcomes == 0) {
        return QS_ERROR_INVALID_STATE;
    }
    if (povm->completeness_checked) {
        return (qs_error_t)povm->completeness_status;
    }
    qs_error_t rc = povm_check_completeness_full(povm);
    povm_t *mut = (povm_t *)povm;
    mut->completeness_status  = (int)rc;
    mut->completeness_checked = 1;
    return rc;
}

qs_error_t measurement_povm_probabilities(const quantum_state_t *state,
                                           const povm_t *povm,
                                           double *probs_out) {
    if (!state || !state->amplitudes || !povm || !povm->kraus_ops ||
        povm->num_outcomes == 0 ||
        povm->num_outcomes > SIZE_MAX / sizeof(double) || !probs_out ||
        povm->state_dim != state->state_dim) {
        return QS_ERROR_INVALID_STATE;
    }
    measurement_gpu_pull(state);
    complex_t *scratch = (complex_t*)malloc(state->state_dim * sizeof(complex_t));
    if (!scratch) return QS_ERROR_OUT_OF_MEMORY;
    for (size_t k = 0; k < povm->num_outcomes; k++) {
        probs_out[k] = povm_outcome_prob(povm->kraus_ops[k],
                                          state->amplitudes,
                                          scratch, state->state_dim);
    }
    free(scratch);
    return QS_SUCCESS;
}

qs_error_t measurement_povm(quantum_state_t *state,
                             const povm_t *povm,
                             double uniform,
                             size_t *outcome_out) {
    if (!state || !state->amplitudes || !povm || !povm->kraus_ops ||
        povm->num_outcomes == 0 ||
        povm->num_outcomes > SIZE_MAX / sizeof(double) ||
        !isfinite(uniform) || uniform < 0.0 || uniform >= 1.0 ||
        povm->state_dim != state->state_dim) {
        return QS_ERROR_INVALID_STATE;
    }
    qs_error_t cc = povm_check_completeness_cached(povm);
    if (cc != QS_SUCCESS) return cc;

    measurement_gpu_pull(state);

    const uint64_t D = state->state_dim;
    complex_t *scratch = (complex_t*)malloc(D * sizeof(complex_t));
    if (!scratch) return QS_ERROR_OUT_OF_MEMORY;
    double *probs = (double*)malloc(povm->num_outcomes * sizeof(double));
    if (!probs) { free(scratch); return QS_ERROR_OUT_OF_MEMORY; }
    for (size_t k = 0; k < povm->num_outcomes; k++) {
        probs[k] = povm_outcome_prob(povm->kraus_ops[k],
                                      state->amplitudes, scratch, D);
    }

    double cum = 0.0;
    double picked_probability = 0.0;
    size_t picked = 0;
    for (size_t k = 0; k < povm->num_outcomes; k++) {
        const double probability = probs[k];
        cum += probability;
        /* Keep the fallback probability and its index coupled. Both scalars
         * are initialized independently, and the nonzero outcome-count guard
         * above guarantees at least one update. This also avoids asking GCC
         * to correlate a later probs[picked] read with the separate fill loop. */
        picked = k;
        picked_probability = probability;
        if (uniform < cum) break;
    }

    if (picked_probability <= 0.0) {
        free(scratch); free(probs);
        return QS_ERROR_NOT_NORMALIZED;
    }
    if (outcome_out) *outcome_out = picked;
    apply_matrix(povm->kraus_ops[picked], state->amplitudes, scratch, D);
    double inv_norm = 1.0 / sqrt(picked_probability);
    for (uint64_t i = 0; i < D; i++) {
        state->amplitudes[i] = scratch[i] * inv_norm;
    }

    measurement_gpu_push(state);

    free(scratch); free(probs);
    return QS_SUCCESS;
}

qs_error_t measurement_weak_z(quantum_state_t *state,
                               int qubit,
                               double strength,
                               double uniform,
                               int *outcome_out) {
    if (!state || !state->amplitudes || qubit < 0 ||
        (uint32_t)qubit >= state->num_qubits)
        return QS_ERROR_INVALID_STATE;
    if (strength < 0.0) strength = 0.0;
    if (strength > 1.0) strength = 1.0;

    const uint64_t D = state->state_dim;
    const uint64_t mask = (uint64_t)1 << qubit;

    const double theta = (M_PI / 4.0) * (1.0 - strength);
    const double c = cos(theta);
    const double s = sin(theta);

    /* Both Kraus operators are diagonal:
     *   K_+[i,i] = bit ? s : c,   K_-[i,i] = bit ? c : s
     * with bit = (i >> qubit) & 1.  Completeness (K_+^2 + K_-^2 = I) holds
     * exactly per diagonal entry since s^2 + c^2 = 1, so no O(K*D^3) check
     * or dense D x D materialization is needed -- everything is O(D). */
    measurement_gpu_pull(state);
    complex_t *amp = state->amplitudes;

    double p_plus = 0.0, p_minus = 0.0;
    for (uint64_t i = 0; i < D; i++) {
        int bit = (i & mask) ? 1 : 0;
        double kp = bit ? s : c;   /* K_+ diagonal */
        double km = bit ? c : s;   /* K_- diagonal */
        double re = creal(amp[i]);
        double im = cimag(amp[i]);
        double mag2 = re * re + im * im;
        p_plus  += kp * kp * mag2;
        p_minus += km * km * mag2;
    }

    /* Outcome 0 = "+", outcome 1 = "-" (matches the Kraus ordering of the
     * general POVM form documented in the header). */
    int outcome = (uniform < p_plus) ? 0 : 1;
    double p = (outcome == 0) ? p_plus : p_minus;
    if (p <= MEASUREMENT_SMALL_NORM) {
        return QS_ERROR_NOT_NORMALIZED;
    }
    double inv_norm = 1.0 / sqrt(p);
    for (uint64_t i = 0; i < D; i++) {
        int bit = (i & mask) ? 1 : 0;
        double k = (outcome == 0) ? (bit ? s : c) : (bit ? c : s);
        amp[i] *= k * inv_norm;
    }

    measurement_gpu_push(state);

    if (outcome_out) *outcome_out = outcome;
    return QS_SUCCESS;
}
