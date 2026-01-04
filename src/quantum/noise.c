/**
 * @file noise.c
 * @brief Quantum noise models and error channels
 *
 * Implements common quantum noise channels:
 * - Depolarizing noise
 * - Amplitude damping
 * - Phase damping (dephasing)
 * - Bit flip / Phase flip
 * - Thermal relaxation (T1, T2)
 * - Readout error
 * - Crosstalk
 *
 * Uses Kraus operator representation for channels.
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#include "../utils/config.h"
#include "noise.h"
#include "state.h"
#include "gates.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#define DEFAULT_GATE_TIME_US 0.020  // 20 nanoseconds

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// NOISE MODEL STRUCTURE
// ============================================================================

/**
 * @brief Create a noise model
 */
noise_model_t* noise_model_create(void) {
    noise_model_t* model = calloc(1, sizeof(noise_model_t));
    if (!model) return NULL;

    model->enabled = 1;
    model->depolarizing_rate = 0.0;
    model->amplitude_damping_rate = 0.0;
    model->phase_damping_rate = 0.0;
    model->t1 = 0.0;
    model->t2 = 0.0;
    model->readout_error_0 = 0.0;
    model->readout_error_1 = 0.0;

    return model;
}

/**
 * @brief Destroy noise model
 */
void noise_model_destroy(noise_model_t* model) {
    if (model) {
        free(model);
    }
}

/**
 * @brief Copy noise model
 */
noise_model_t* noise_model_copy(const noise_model_t* model) {
    if (!model) return NULL;

    noise_model_t* copy = malloc(sizeof(noise_model_t));
    if (!copy) return NULL;

    memcpy(copy, model, sizeof(noise_model_t));
    return copy;
}

// ============================================================================
// DEPOLARIZING CHANNEL
// ============================================================================

/**
 * @brief Apply depolarizing channel to single qubit
 *
 * ε(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
 *
 * With probability p, applies random Pauli (X, Y, or Z)
 *
 * @param state Quantum state
 * @param qubit Target qubit
 * @param probability Depolarizing probability p ∈ [0, 1]
 * @param random_value Random value for channel selection
 */
void noise_depolarizing_single(quantum_state_t* state, int qubit,
                               double probability, double random_value) {
    if (!state || probability <= 0.0) return;
    if (random_value >= probability) return;  // No error

    // Apply random Pauli
    double r = random_value / probability;  // Renormalize to [0,1)

    if (r < 1.0/3.0) {
        // Apply X
        gate_pauli_x(state, qubit);
    } else if (r < 2.0/3.0) {
        // Apply Y
        gate_pauli_y(state, qubit);
    } else {
        // Apply Z
        gate_pauli_z(state, qubit);
    }
}

/**
 * @brief Apply depolarizing channel to two qubits
 *
 * 2-qubit depolarizing: applies one of 15 non-identity Paulis
 *
 * @param state Quantum state
 * @param qubit1 First qubit
 * @param qubit2 Second qubit
 * @param probability Depolarizing probability
 * @param random_value Random value
 */
void noise_depolarizing_two_qubit(quantum_state_t* state, int qubit1, int qubit2,
                                  double probability, double random_value) {
    if (!state || probability <= 0.0) return;
    if (random_value >= probability) return;

    // 15 non-identity two-qubit Paulis
    double r = random_value / probability;
    int pauli_idx = (int)(r * 15.0);
    if (pauli_idx > 14) pauli_idx = 14;

    // Decode as (P1, P2) where P ∈ {I, X, Y, Z}
    int p1 = (pauli_idx + 1) / 4;  // Skip (I,I)
    int p2 = (pauli_idx + 1) % 4;

    // Apply Paulis
    switch (p1) {
        case 1: gate_pauli_x(state, qubit1); break;
        case 2: gate_pauli_y(state, qubit1); break;
        case 3: gate_pauli_z(state, qubit1); break;
    }
    switch (p2) {
        case 1: gate_pauli_x(state, qubit2); break;
        case 2: gate_pauli_y(state, qubit2); break;
        case 3: gate_pauli_z(state, qubit2); break;
    }
}

// ============================================================================
// AMPLITUDE DAMPING
// ============================================================================

/**
 * @brief Apply amplitude damping channel
 *
 * Models energy relaxation (T1 decay): |1⟩ → |0⟩
 *
 * Kraus operators:
 *   K0 = [[1, 0], [0, √(1-γ)]]
 *   K1 = [[0, √γ], [0, 0]]
 *
 * @param state Quantum state
 * @param qubit Target qubit
 * @param gamma Damping parameter γ ∈ [0, 1]
 * @param random_value Random value for collapse
 */
void noise_amplitude_damping(quantum_state_t* state, int qubit,
                             double gamma, double random_value) {
    if (!state || !state->amplitudes || gamma <= 0.0 || gamma > 1.0) return;
    if (qubit < 0 || qubit >= state->num_qubits) return;

    const uint64_t state_dim = state->state_dim;
    const uint64_t qubit_mask = 1ULL << qubit;
    complex_t* amp = state->amplitudes;

    double sqrt_1_gamma = sqrt(1.0 - gamma);
    double sqrt_gamma = sqrt(gamma);

    // For each pair of amplitudes differing in qubit
    for (uint64_t i = 0; i < state_dim; i++) {
        if (i & qubit_mask) {
            // i has qubit=1, j=i^mask has qubit=0
            uint64_t j = i ^ qubit_mask;

            complex_t a0 = amp[j];  // |...0...⟩
            complex_t a1 = amp[i];  // |...1...⟩

            // Compute decay probability for this amplitude
            double p_decay = gamma * cabs(a1) * cabs(a1);

            if (random_value < p_decay) {
                // Decay occurred: collapse |1⟩ → |0⟩
                amp[j] = a0 + a1;  // Transfer amplitude
                amp[i] = 0.0;
            } else {
                // No decay: apply damping
                amp[i] *= sqrt_1_gamma;
            }
        }
    }

    // Renormalize
    double norm = 0.0;
    for (uint64_t i = 0; i < state_dim; i++) {
        norm += cabs(amp[i]) * cabs(amp[i]);
    }
    if (norm > 1e-15) {
        double inv_norm = 1.0 / sqrt(norm);
        for (uint64_t i = 0; i < state_dim; i++) {
            amp[i] *= inv_norm;
        }
    }
}

// ============================================================================
// PHASE DAMPING
// ============================================================================

/**
 * @brief Apply phase damping (dephasing) channel
 *
 * Models T2 dephasing without energy loss
 *
 * Kraus operators:
 *   K0 = [[1, 0], [0, √(1-γ)]]
 *   K1 = [[0, 0], [0, √γ]]
 *
 * @param state Quantum state
 * @param qubit Target qubit
 * @param gamma Dephasing parameter
 * @param random_value Random value
 */
void noise_phase_damping(quantum_state_t* state, int qubit,
                         double gamma, double random_value) {
    if (!state || !state->amplitudes || gamma <= 0.0 || gamma > 1.0) return;
    if (qubit < 0 || qubit >= state->num_qubits) return;

    // Phase damping using Monte Carlo trajectory approach
    // K0 = [[1, 0], [0, √(1-γ)]] (no-jump evolution)
    // K1 = [[0, 0], [0, √γ]]     (jump: project to |1⟩)

    const uint64_t state_dim = state->state_dim;
    const uint64_t qubit_mask = 1ULL << qubit;
    complex_t* amp = state->amplitudes;

    double sqrt_1_gamma = sqrt(1.0 - gamma);

    // Compute probability of |1⟩ component (jump probability)
    double prob_one = 0.0;
    for (uint64_t i = 0; i < state_dim; i++) {
        if (i & qubit_mask) {
            prob_one += cabs(amp[i]) * cabs(amp[i]);
        }
    }

    // Jump probability = γ * |⟨1|ψ⟩|²
    double p_jump = gamma * prob_one;

    if (random_value < p_jump && prob_one > 1e-15) {
        // Jump occurred: project to |1⟩ subspace and randomize phase
        // This effectively collapses coherence
        for (uint64_t i = 0; i < state_dim; i++) {
            if (!(i & qubit_mask)) {
                // Zero out |0⟩ components
                amp[i] = 0.0;
            }
        }
    } else {
        // No jump: apply K0 (dampen |1⟩ amplitudes)
        for (uint64_t i = 0; i < state_dim; i++) {
            if (i & qubit_mask) {
                amp[i] *= sqrt_1_gamma;
            }
        }
    }

    // Renormalize
    double norm = 0.0;
    for (uint64_t i = 0; i < state_dim; i++) {
        norm += cabs(amp[i]) * cabs(amp[i]);
    }
    if (norm > 1e-15) {
        double inv_norm = 1.0 / sqrt(norm);
        for (uint64_t i = 0; i < state_dim; i++) {
            amp[i] *= inv_norm;
        }
    }
}

/**
 * @brief Apply pure dephasing (Gaussian)
 *
 * Applies random phase to each amplitude
 *
 * @param state Quantum state
 * @param qubit Target qubit
 * @param sigma Standard deviation of phase
 * @param random_phase Random phase value
 */
void noise_pure_dephasing(quantum_state_t* state, int qubit,
                          double sigma, double random_phase) {
    if (!state || !state->amplitudes || sigma <= 0.0) return;

    const uint64_t state_dim = state->state_dim;
    const uint64_t qubit_mask = 1ULL << qubit;
    complex_t* amp = state->amplitudes;

    // Apply phase to |1⟩ amplitudes
    complex_t phase = cexp(I * sigma * random_phase);

    for (uint64_t i = 0; i < state_dim; i++) {
        if (i & qubit_mask) {
            amp[i] *= phase;
        }
    }
}

// ============================================================================
// BIT FLIP / PHASE FLIP
// ============================================================================

/**
 * @brief Apply bit flip channel
 *
 * With probability p, applies X gate
 */
void noise_bit_flip(quantum_state_t* state, int qubit,
                    double probability, double random_value) {
    if (!state || probability <= 0.0) return;

    if (random_value < probability) {
        gate_pauli_x(state, qubit);
    }
}

/**
 * @brief Apply phase flip channel
 *
 * With probability p, applies Z gate
 */
void noise_phase_flip(quantum_state_t* state, int qubit,
                      double probability, double random_value) {
    if (!state || probability <= 0.0) return;

    if (random_value < probability) {
        gate_pauli_z(state, qubit);
    }
}

/**
 * @brief Apply bit-phase flip channel
 *
 * With probability p, applies Y gate
 */
void noise_bit_phase_flip(quantum_state_t* state, int qubit,
                          double probability, double random_value) {
    if (!state || probability <= 0.0) return;

    if (random_value < probability) {
        gate_pauli_y(state, qubit);
    }
}

// ============================================================================
// THERMAL RELAXATION
// ============================================================================

/**
 * @brief Apply thermal relaxation (T1/T2)
 *
 * Combines amplitude damping (T1) and phase damping (T2)
 *
 * @param state Quantum state
 * @param qubit Target qubit
 * @param t1 T1 time (energy relaxation)
 * @param t2 T2 time (dephasing, T2 ≤ 2*T1)
 * @param time Gate duration
 * @param random_values Array of 2 random values
 */
void noise_thermal_relaxation(quantum_state_t* state, int qubit,
                              double t1, double t2, double time,
                              const double* random_values) {
    if (!state || t1 <= 0.0 || t2 <= 0.0 || time <= 0.0) return;
    if (!random_values) return;

    // Ensure T2 ≤ 2*T1 (physical constraint)
    if (t2 > 2.0 * t1) t2 = 2.0 * t1;

    // Amplitude damping parameter
    double gamma_t1 = 1.0 - exp(-time / t1);

    // Additional dephasing (pure T2 component)
    double gamma_t2 = 0.0;
    if (t2 < 2.0 * t1) {
        double rate_phi = 1.0 / t2 - 1.0 / (2.0 * t1);
        gamma_t2 = 1.0 - exp(-time * rate_phi);
    }

    // Apply amplitude damping
    noise_amplitude_damping(state, qubit, gamma_t1, random_values[0]);

    // Apply additional dephasing
    if (gamma_t2 > 0.0) {
        noise_phase_damping(state, qubit, gamma_t2, random_values[1]);
    }
}

// ============================================================================
// READOUT ERROR
// ============================================================================

/**
 * @brief Simulate readout error
 *
 * Flips measurement outcome with specified probabilities
 *
 * @param outcome Original measurement outcome
 * @param error_0_to_1 P(measure 1 | state is 0)
 * @param error_1_to_0 P(measure 0 | state is 1)
 * @param random_value Random value
 * @return Noisy measurement outcome
 */
int noise_readout_error(int outcome, double error_0_to_1, double error_1_to_0,
                        double random_value) {
    if (outcome == 0) {
        return (random_value < error_0_to_1) ? 1 : 0;
    } else {
        return (random_value < error_1_to_0) ? 0 : 1;
    }
}

// ============================================================================
// APPLY NOISE MODEL
// ============================================================================

/**
 * @brief Apply noise model to qubit after gate
 *
 * @param state Quantum state
 * @param qubit Target qubit
 * @param model Noise model
 * @param random_values Array of random values (need at least 4)
 */
void noise_apply_model(quantum_state_t* state, int qubit,
                       const noise_model_t* model,
                       const double* random_values) {
    if (!state || !model || !model->enabled || !random_values) return;

    int rv_idx = 0;

    // Apply depolarizing noise
    if (model->depolarizing_rate > 0.0) {
        noise_depolarizing_single(state, qubit, model->depolarizing_rate,
                                  random_values[rv_idx++]);
    }

    // Apply amplitude damping
    if (model->amplitude_damping_rate > 0.0) {
        noise_amplitude_damping(state, qubit, model->amplitude_damping_rate,
                                random_values[rv_idx++]);
    }

    // Apply phase damping
    if (model->phase_damping_rate > 0.0) {
        noise_phase_damping(state, qubit, model->phase_damping_rate,
                            random_values[rv_idx++]);
    }

    // Apply thermal relaxation
    if (model->t1 > 0.0 && model->t2 > 0.0 && model->gate_time > 0.0) {
        noise_thermal_relaxation(state, qubit, model->t1, model->t2,
                                 model->gate_time, &random_values[rv_idx]);
    }
}

/**
 * @brief Apply noise model to two-qubit gate
 */
void noise_apply_model_two_qubit(quantum_state_t* state, int qubit1, int qubit2,
                                 const noise_model_t* model,
                                 const double* random_values) {
    if (!state || !model || !model->enabled || !random_values) return;

    // Apply two-qubit depolarizing
    if (model->two_qubit_depolarizing_rate > 0.0) {
        noise_depolarizing_two_qubit(state, qubit1, qubit2,
                                     model->two_qubit_depolarizing_rate,
                                     random_values[0]);
    }

    // Apply single-qubit noise to each
    noise_apply_model(state, qubit1, model, &random_values[1]);
    noise_apply_model(state, qubit2, model, &random_values[5]);
}

// ============================================================================
// NOISE MODEL CONFIGURATION
// ============================================================================

/**
 * @brief Set depolarizing rate
 */
void noise_model_set_depolarizing(noise_model_t* model, double rate) {
    if (model && rate >= 0.0 && rate <= 1.0) {
        model->depolarizing_rate = rate;
    }
}

/**
 * @brief Set amplitude damping rate
 */
void noise_model_set_amplitude_damping(noise_model_t* model, double rate) {
    if (model && rate >= 0.0 && rate <= 1.0) {
        model->amplitude_damping_rate = rate;
    }
}

/**
 * @brief Set phase damping rate
 */
void noise_model_set_phase_damping(noise_model_t* model, double rate) {
    if (model && rate >= 0.0 && rate <= 1.0) {
        model->phase_damping_rate = rate;
    }
}

/**
 * @brief Set T1/T2 times
 */
void noise_model_set_thermal(noise_model_t* model, double t1, double t2) {
    if (model && t1 > 0.0 && t2 > 0.0) {
        model->t1 = t1;
        model->t2 = (t2 <= 2.0 * t1) ? t2 : 2.0 * t1;
    }
}

/**
 * @brief Set gate time for thermal relaxation
 */
void noise_model_set_gate_time(noise_model_t* model, double time) {
    if (model && time >= 0.0) {
        model->gate_time = time;
    }
}

/**
 * @brief Set readout error rates
 */
void noise_model_set_readout_error(noise_model_t* model,
                                   double error_0, double error_1) {
    if (model) {
        if (error_0 >= 0.0 && error_0 <= 1.0) model->readout_error_0 = error_0;
        if (error_1 >= 0.0 && error_1 <= 1.0) model->readout_error_1 = error_1;
    }
}

/**
 * @brief Enable/disable noise model
 */
void noise_model_set_enabled(noise_model_t* model, int enabled) {
    if (model) {
        model->enabled = enabled;
    }
}

/**
 * @brief Create realistic noise model based on hardware specs
 *
 * @param t1_us T1 in microseconds
 * @param t2_us T2 in microseconds
 * @param gate_error Single-qubit gate error rate
 * @param readout_error Readout error rate
 * @return Configured noise model
 */
noise_model_t* noise_model_create_realistic(double t1_us, double t2_us,
                                            double gate_error,
                                            double readout_error) {
    noise_model_t* model = noise_model_create();
    if (!model) return NULL;

    // Use gate time from config if available, otherwise default
    qsim_config_t* cfg = qsim_config_global();
    double gate_time_us = (cfg && cfg->noise.gate_time > 0)
        ? cfg->noise.gate_time
        : DEFAULT_GATE_TIME_US;

    model->t1 = t1_us;
    model->t2 = (t2_us <= 2.0 * t1_us) ? t2_us : 2.0 * t1_us;
    model->gate_time = gate_time_us;

    // Gate error is approximately depolarizing
    model->depolarizing_rate = gate_error;

    // Symmetric readout error
    model->readout_error_0 = readout_error;
    model->readout_error_1 = readout_error;

    return model;
}
