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
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
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
    if (!state || !state->amplitudes || qubit < 0 || qubit >= state->num_qubits) {
        return 0.0;
    }

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

    for (int q = 0; q < state->num_qubits; q++) {
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
    if (!state || !state->amplitudes || qubit < 0 || qubit >= state->num_qubits) {
        return -1;
    }

    const uint64_t state_dim = state->state_dim;
    const uint64_t qubit_mask = 1ULL << qubit;
    complex_t* amp = state->amplitudes;

    // Compute probability of measuring |1⟩
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

    // Renormalize
    if (norm_factor > 1e-15) {
        double inv_norm = 1.0 / sqrt(norm_factor);
        for (uint64_t i = 0; i < state_dim; i++) {
            amp[i] *= inv_norm;
        }
    }

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

    const uint64_t state_dim = state->state_dim;
    const complex_t* amp = state->amplitudes;

    // Find outcome by cumulative probability
    double cumulative = 0.0;
    uint64_t result = 0;

    for (uint64_t i = 0; i < state_dim; i++) {
        double prob = cabs(amp[i]) * cabs(amp[i]);
        cumulative += prob;

        if (random_value < cumulative) {
            result = i;
            break;
        }
    }

    // Handle edge case (rounding errors)
    if (random_value >= cumulative) {
        result = state_dim - 1;
    }

    // Collapse to computational basis state
    for (uint64_t i = 0; i < state_dim; i++) {
        state->amplitudes[i] = (i == result) ? 1.0 : 0.0;
    }

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

    // Create mask for measured qubits
    uint64_t measure_mask = 0;
    for (int i = 0; i < num_measure; i++) {
        if (qubits[i] >= 0 && qubits[i] < state->num_qubits) {
            measure_mask |= (1ULL << qubits[i]);
        }
    }

    const uint64_t state_dim = state->state_dim;
    complex_t* amp = state->amplitudes;

    // Count possible outcomes and compute probabilities
    uint64_t num_outcomes = 1ULL << num_measure;
    double* outcome_probs = calloc(num_outcomes, sizeof(double));

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
    for (uint64_t o = 0; o < num_outcomes; o++) {
        cumulative += outcome_probs[o];
        if (random_value < cumulative) {
            result = o;
            break;
        }
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

    // Renormalize
    if (norm_factor > 1e-15) {
        double inv_norm = 1.0 / sqrt(norm_factor);
        for (uint64_t i = 0; i < state_dim; i++) {
            amp[i] *= inv_norm;
        }
    }

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

    const uint64_t state_dim = state->state_dim;
    const complex_t* amp = state->amplitudes;

    // Precompute cumulative distribution
    double* cdf = malloc(state_dim * sizeof(double));
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
