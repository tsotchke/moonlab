/**
 * @file bell_test.h
 * @brief Bell inequality test and CHSH experiment
 *
 * Demonstrates quantum entanglement through Bell tests:
 * - CHSH (Clauser-Horne-Shimony-Holt) inequality
 * - Bell state preparation
 * - Correlation measurements
 * - Classical vs quantum comparison
 *
 * Classical limit: |S| ≤ 2
 * Quantum maximum (Tsirelson bound): |S| = 2√2 ≈ 2.828
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#ifndef APPLICATIONS_BELL_TEST_H
#define APPLICATIONS_BELL_TEST_H

#include "../quantum/state.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// BELL TEST CONFIGURATION
// ============================================================================

/**
 * @brief Bell state type
 */
typedef enum {
    BELL_STATE_PHI_PLUS,    /**< |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 */
    BELL_STATE_PHI_MINUS,   /**< |Φ⁻⟩ = (|00⟩ - |11⟩)/√2 */
    BELL_STATE_PSI_PLUS,    /**< |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2 */
    BELL_STATE_PSI_MINUS    /**< |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2 */
} bell_state_type_t;

/**
 * @brief Bell test configuration
 */
typedef struct {
    int num_trials;             /**< Number of measurement trials */
    bell_state_type_t state;    /**< Bell state to test */

    // CHSH angles (in radians)
    double alice_angle_1;       /**< Alice's first measurement angle */
    double alice_angle_2;       /**< Alice's second measurement angle */
    double bob_angle_1;         /**< Bob's first measurement angle */
    double bob_angle_2;         /**< Bob's second measurement angle */

    // Random seed (0 = use system entropy)
    uint64_t seed;
} bell_test_config_t;

/**
 * @brief Bell test results
 */
typedef struct {
    // Raw counts (E(a,b) correlations)
    int pp_count;               /**< Both +1 outcomes */
    int pm_count;               /**< Alice +1, Bob -1 */
    int mp_count;               /**< Alice -1, Bob +1 */
    int mm_count;               /**< Both -1 outcomes */

    // Correlation values
    double E_a1b1;              /**< E(a₁, b₁) */
    double E_a1b2;              /**< E(a₁, b₂) */
    double E_a2b1;              /**< E(a₂, b₁) */
    double E_a2b2;              /**< E(a₂, b₂) */

    // CHSH value
    double S;                   /**< CHSH S = E(a₁,b₁) + E(a₁,b₂) + E(a₂,b₁) - E(a₂,b₂) */
    double S_error;             /**< Statistical error in S */

    // Test results
    int violates_classical;     /**< |S| > 2 (violates classical limit) */
    double sigma_violation;     /**< Standard deviations above 2 */

    // Statistics
    int total_trials;           /**< Total trials performed */
    double execution_time_ms;   /**< Execution time in milliseconds */
} bell_test_results_t;

// ============================================================================
// BELL STATE PREPARATION
// ============================================================================

/**
 * @brief Prepare Bell state
 *
 * @param state 2-qubit quantum state
 * @param type Bell state type
 * @return 0 on success, -1 on error
 */
int bell_state_prepare(quantum_state_t* state, bell_state_type_t type);

/**
 * @brief Verify state is maximally entangled
 *
 * @param state Quantum state
 * @return Concurrence value (1.0 for maximally entangled)
 */
double bell_state_verify_entanglement(const quantum_state_t* state);

/**
 * @brief Get Bell state name
 *
 * @param type Bell state type
 * @return Name string
 */
const char* bell_state_name(bell_state_type_t type);

// ============================================================================
// CHSH TEST
// ============================================================================

/**
 * @brief Create default CHSH test configuration
 *
 * Uses optimal angles for maximum violation:
 * - Alice: 0, π/4
 * - Bob: π/8, 3π/8
 *
 * @return Default configuration
 */
bell_test_config_t bell_test_default_config(void);

/**
 * @brief Run full CHSH Bell test
 *
 * @param config Test configuration
 * @param results Output results
 * @return 0 on success, -1 on error
 */
int bell_test_run_chsh(const bell_test_config_t* config,
                       bell_test_results_t* results);

/**
 * @brief Measure correlation E(a, b) for given angles
 *
 * @param state Bell state (will be reset each trial)
 * @param alice_angle Alice's measurement angle
 * @param bob_angle Bob's measurement angle
 * @param num_trials Number of trials
 * @return Correlation value in [-1, +1]
 */
double bell_test_measure_correlation(quantum_state_t* state,
                                     double alice_angle,
                                     double bob_angle,
                                     int num_trials);

// ============================================================================
// ANALYSIS
// ============================================================================

/**
 * @brief Print Bell test results
 *
 * @param results Test results
 */
void bell_test_print_results(const bell_test_results_t* results);

/**
 * @brief Check if results violate Bell inequality
 *
 * @param results Test results
 * @param confidence_level Confidence level (e.g., 0.99 for 99%)
 * @return 1 if violation is statistically significant, 0 otherwise
 */
int bell_test_is_violation_significant(const bell_test_results_t* results,
                                       double confidence_level);

/**
 * @brief Calculate theoretical CHSH value for given angles
 *
 * @param a1 Alice's first angle
 * @param a2 Alice's second angle
 * @param b1 Bob's first angle
 * @param b2 Bob's second angle
 * @return Theoretical S value
 */
double bell_test_theoretical_S(double a1, double a2, double b1, double b2);

// ============================================================================
// COMPREHENSIVE TESTS
// ============================================================================

/**
 * @brief Run Bell test with all four Bell states
 *
 * @param num_trials Trials per state
 * @param results Array of 4 results (one per Bell state)
 * @return 0 on success, -1 on error
 */
int bell_test_all_states(int num_trials, bell_test_results_t* results);

/**
 * @brief Sweep CHSH value over angle range
 *
 * @param angle_steps Number of angle steps
 * @param S_values Output array (angle_steps elements)
 * @param trials_per_angle Trials per angle pair
 * @return 0 on success, -1 on error
 */
int bell_test_angle_sweep(int angle_steps, double* S_values,
                          int trials_per_angle);

/**
 * @brief Benchmark Bell test throughput
 *
 * @param num_trials Number of trials
 * @return Trials per second
 */
double bell_test_benchmark(int num_trials);

// ============================================================================
// NOISE EFFECTS
// ============================================================================

/**
 * @brief Run noisy Bell test
 *
 * @param config Test configuration
 * @param depolarizing_rate Depolarizing noise rate
 * @param results Output results
 * @return 0 on success, -1 on error
 */
int bell_test_run_noisy(const bell_test_config_t* config,
                        double depolarizing_rate,
                        bell_test_results_t* results);

/**
 * @brief Find threshold noise for visibility loss
 *
 * @param target_S Target CHSH value (e.g., 2.0 for classical limit)
 * @param precision Search precision
 * @return Noise rate at which S drops to target
 */
double bell_test_find_noise_threshold(double target_S, double precision);

// ============================================================================
// COMMAND-LINE INTERFACE
// ============================================================================

/**
 * @brief Run Bell test from command line
 *
 * @param argc Argument count
 * @param argv Arguments
 * @return Exit code
 */
int bell_test_main(int argc, char** argv);

#ifdef __cplusplus
}
#endif

#endif /* APPLICATIONS_BELL_TEST_H */
