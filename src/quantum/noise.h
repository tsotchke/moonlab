/**
 * @file noise.h
 * @brief Quantum noise models and error channels
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef QUANTUM_NOISE_H
#define QUANTUM_NOISE_H

#include "state.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// NOISE MODEL STRUCTURE
// ============================================================================

/**
 * @brief Quantum noise model configuration
 */
typedef struct {
    int enabled;                    /**< Whether noise is active */

    // Single-qubit error rates
    double depolarizing_rate;       /**< Depolarizing probability */
    double amplitude_damping_rate;  /**< Amplitude damping (T1) */
    double phase_damping_rate;      /**< Phase damping (T2 pure) */

    // Thermal relaxation
    double t1;                      /**< T1 time (energy relaxation) */
    double t2;                      /**< T2 time (dephasing) */
    double gate_time;               /**< Gate duration for thermal calc */

    // Two-qubit error rates
    double two_qubit_depolarizing_rate;

    // Readout errors
    double readout_error_0;         /**< P(1|0) - measure 1 when state is 0 */
    double readout_error_1;         /**< P(0|1) - measure 0 when state is 1 */

} noise_model_t;

// ============================================================================
// NOISE MODEL MANAGEMENT
// ============================================================================

/**
 * @brief Create a noise model
 */
noise_model_t* noise_model_create(void);

/**
 * @brief Destroy noise model
 */
void noise_model_destroy(noise_model_t* model);

/**
 * @brief Copy noise model
 */
noise_model_t* noise_model_copy(const noise_model_t* model);

/**
 * @brief Create realistic noise model from hardware specs
 */
noise_model_t* noise_model_create_realistic(double t1_us, double t2_us,
                                            double gate_error,
                                            double readout_error);

// ============================================================================
// NOISE CHANNELS
// ============================================================================

/**
 * @brief Apply depolarizing channel to single qubit
 */
void noise_depolarizing_single(quantum_state_t* state, int qubit,
                               double probability, double random_value);

/**
 * @brief Apply depolarizing channel to two qubits
 */
void noise_depolarizing_two_qubit(quantum_state_t* state, int qubit1, int qubit2,
                                  double probability, double random_value);

/**
 * @brief Apply amplitude damping channel
 */
void noise_amplitude_damping(quantum_state_t* state, int qubit,
                             double gamma, double random_value);

/**
 * @brief Apply phase damping channel
 */
void noise_phase_damping(quantum_state_t* state, int qubit,
                         double gamma, double random_value);

/**
 * @brief Apply pure dephasing
 */
void noise_pure_dephasing(quantum_state_t* state, int qubit,
                          double sigma, double random_phase);

/**
 * @brief Apply bit flip channel
 */
void noise_bit_flip(quantum_state_t* state, int qubit,
                    double probability, double random_value);

/**
 * @brief Apply phase flip channel
 */
void noise_phase_flip(quantum_state_t* state, int qubit,
                      double probability, double random_value);

/**
 * @brief Apply bit-phase flip channel
 */
void noise_bit_phase_flip(quantum_state_t* state, int qubit,
                          double probability, double random_value);

/**
 * @brief Apply thermal relaxation
 */
void noise_thermal_relaxation(quantum_state_t* state, int qubit,
                              double t1, double t2, double time,
                              const double* random_values);

/**
 * @brief Simulate readout error
 */
int noise_readout_error(int outcome, double error_0_to_1, double error_1_to_0,
                        double random_value);

// ============================================================================
// APPLY NOISE MODEL
// ============================================================================

/**
 * @brief Apply noise model to qubit
 */
void noise_apply_model(quantum_state_t* state, int qubit,
                       const noise_model_t* model,
                       const double* random_values);

/**
 * @brief Apply noise model to two-qubit gate
 */
void noise_apply_model_two_qubit(quantum_state_t* state, int qubit1, int qubit2,
                                 const noise_model_t* model,
                                 const double* random_values);

// ============================================================================
// CONFIGURATION
// ============================================================================

void noise_model_set_depolarizing(noise_model_t* model, double rate);
void noise_model_set_amplitude_damping(noise_model_t* model, double rate);
void noise_model_set_phase_damping(noise_model_t* model, double rate);
void noise_model_set_thermal(noise_model_t* model, double t1, double t2);
void noise_model_set_gate_time(noise_model_t* model, double time);
void noise_model_set_readout_error(noise_model_t* model,
                                   double error_0, double error_1);
void noise_model_set_enabled(noise_model_t* model, int enabled);

#ifdef __cplusplus
}
#endif

#endif /* QUANTUM_NOISE_H */
