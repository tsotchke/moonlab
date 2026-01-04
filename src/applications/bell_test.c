/**
 * @file bell_test.c
 * @brief Bell inequality test and CHSH experiment implementation
 *
 * Demonstrates quantum entanglement through Bell tests:
 * - CHSH (Clauser-Horne-Shimony-Holt) inequality
 * - Bell state preparation
 * - Correlation measurements
 * - Classical vs quantum comparison
 *
 * @stability stable
 * @since v1.0.0
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the MIT License
 */

#include "bell_test.h"
#include "../quantum/gates.h"
#include "../quantum/measurement.h"
#include "../utils/entropy.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

// ============================================================================
// INTERNAL HELPERS
// ============================================================================

static entropy_ctx_t* g_entropy = NULL;

static void ensure_entropy(void) {
    if (!g_entropy) {
        g_entropy = entropy_create();
    }
}

static double get_random_double(void) {
    ensure_entropy();
    return entropy_double(g_entropy);
}

/**
 * @brief Apply rotation about arbitrary axis in XZ plane
 *
 * For measurement in direction (sin(θ), 0, cos(θ))
 * We rotate to align with Z axis: Ry(-θ)
 */
static void apply_measurement_rotation(quantum_state_t* state, int qubit, double angle) {
    // Ry(-angle) rotation
    double c = cos(-angle / 2.0);
    double s = sin(-angle / 2.0);

    uint64_t mask = 1ULL << qubit;

    for (uint64_t i = 0; i < state->state_dim; i++) {
        if (!(i & mask)) {  // i has qubit = 0
            uint64_t j = i | mask;  // j has qubit = 1

            complex_t a0 = state->amplitudes[i];
            complex_t a1 = state->amplitudes[j];

            state->amplitudes[i] = c * a0 - s * a1;
            state->amplitudes[j] = s * a0 + c * a1;
        }
    }
}

/**
 * @brief Measure single qubit in computational basis
 */
static int measure_single_qubit(quantum_state_t* state, int qubit) {
    uint64_t mask = 1ULL << qubit;
    double prob_one = 0.0;

    // Calculate P(1)
    for (uint64_t i = 0; i < state->state_dim; i++) {
        if (i & mask) {
            prob_one += cabs(state->amplitudes[i]) * cabs(state->amplitudes[i]);
        }
    }

    // Make measurement
    double r = get_random_double();
    int result = (r < prob_one) ? 1 : 0;

    // Collapse state
    double norm = 0.0;
    for (uint64_t i = 0; i < state->state_dim; i++) {
        int bit = (i & mask) ? 1 : 0;
        if (bit != result) {
            state->amplitudes[i] = 0.0;
        } else {
            norm += cabs(state->amplitudes[i]) * cabs(state->amplitudes[i]);
        }
    }

    // Renormalize
    if (norm > 0) {
        double scale = 1.0 / sqrt(norm);
        for (uint64_t i = 0; i < state->state_dim; i++) {
            state->amplitudes[i] *= scale;
        }
    }

    return result;
}

// ============================================================================
// BELL STATE PREPARATION
// ============================================================================

int bell_state_prepare(quantum_state_t* state, bell_state_type_t type) {
    if (!state || state->num_qubits < 2) return -1;

    // Initialize to |00⟩
    quantum_state_init_zero(state);

    // Apply H to qubit 0
    gate_hadamard(state, 0);

    // Apply CNOT (qubit 0 control, qubit 1 target)
    gate_cnot(state, 0, 1);

    // Now we have |Φ⁺⟩ = (|00⟩ + |11⟩)/√2

    switch (type) {
        case BELL_STATE_PHI_PLUS:
            // Already there
            break;

        case BELL_STATE_PHI_MINUS:
            // Apply Z to qubit 0: (|00⟩ - |11⟩)/√2
            gate_pauli_z(state, 0);
            break;

        case BELL_STATE_PSI_PLUS:
            // Apply X to qubit 1: (|01⟩ + |10⟩)/√2
            gate_pauli_x(state, 1);
            break;

        case BELL_STATE_PSI_MINUS:
            // Apply X to qubit 1, then Z to qubit 0: (|01⟩ - |10⟩)/√2
            gate_pauli_x(state, 1);
            gate_pauli_z(state, 0);
            break;

        default:
            return -1;
    }

    return 0;
}

double bell_state_verify_entanglement(const quantum_state_t* state) {
    if (!state || state->num_qubits < 2) return 0.0;

    // Calculate concurrence for 2-qubit state
    // For Bell states, this should be 1.0

    // Get amplitudes: a00, a01, a10, a11
    complex_t a00 = state->amplitudes[0];
    complex_t a01 = state->amplitudes[1];
    complex_t a10 = state->amplitudes[2];
    complex_t a11 = state->amplitudes[3];

    // Concurrence = 2|a00*a11 - a01*a10|
    complex_t det = a00 * a11 - a01 * a10;
    double concurrence = 2.0 * cabs(det);

    return concurrence;
}

const char* bell_state_name(bell_state_type_t type) {
    switch (type) {
        case BELL_STATE_PHI_PLUS:  return "|Φ⁺⟩";
        case BELL_STATE_PHI_MINUS: return "|Φ⁻⟩";
        case BELL_STATE_PSI_PLUS:  return "|Ψ⁺⟩";
        case BELL_STATE_PSI_MINUS: return "|Ψ⁻⟩";
        default:                   return "unknown";
    }
}

// ============================================================================
// CHSH TEST
// ============================================================================

bell_test_config_t bell_test_default_config(void) {
    bell_test_config_t config;

    config.num_trials = 10000;
    config.state = BELL_STATE_PHI_PLUS;

    // Optimal angles for maximum CHSH violation
    // Alice: 0, π/4
    // Bob: π/8, 3π/8
    config.alice_angle_1 = 0.0;
    config.alice_angle_2 = M_PI / 4.0;
    config.bob_angle_1 = M_PI / 8.0;
    config.bob_angle_2 = 3.0 * M_PI / 8.0;

    config.seed = 0;

    return config;
}

double bell_test_measure_correlation(quantum_state_t* state,
                                     double alice_angle,
                                     double bob_angle,
                                     int num_trials) {
    if (!state || num_trials <= 0) return 0.0;

    int pp = 0, pm = 0, mp = 0, mm = 0;

    for (int trial = 0; trial < num_trials; trial++) {
        // Prepare fresh Bell state
        bell_state_prepare(state, BELL_STATE_PHI_PLUS);

        // Apply measurement rotations
        apply_measurement_rotation(state, 0, alice_angle);
        apply_measurement_rotation(state, 1, bob_angle);

        // Measure both qubits
        int alice_result = measure_single_qubit(state, 0);
        int bob_result = measure_single_qubit(state, 1);

        // Convert 0/1 to +1/-1
        int a = alice_result ? -1 : 1;
        int b = bob_result ? -1 : 1;

        // Count correlations
        if (a == 1 && b == 1) pp++;
        else if (a == 1 && b == -1) pm++;
        else if (a == -1 && b == 1) mp++;
        else mm++;
    }

    // E(a,b) = (P(++) + P(--) - P(+-) - P(-+))
    double correlation = (double)(pp + mm - pm - mp) / (double)num_trials;
    return correlation;
}

int bell_test_run_chsh(const bell_test_config_t* config,
                       bell_test_results_t* results) {
    if (!config || !results) return -1;

    memset(results, 0, sizeof(bell_test_results_t));

    // Create state
    quantum_state_t* state = quantum_state_create(2);
    if (!state) return -1;

    // Seed entropy if specified
    if (config->seed != 0) {
        ensure_entropy();
        uint8_t seed_bytes[8];
        memcpy(seed_bytes, &config->seed, 8);
        entropy_mix(g_entropy, seed_bytes, 8);
    }

    clock_t start = clock();

    int trials_per_correlation = config->num_trials / 4;

    // Measure all four correlations
    results->E_a1b1 = bell_test_measure_correlation(
        state, config->alice_angle_1, config->bob_angle_1, trials_per_correlation);

    results->E_a1b2 = bell_test_measure_correlation(
        state, config->alice_angle_1, config->bob_angle_2, trials_per_correlation);

    results->E_a2b1 = bell_test_measure_correlation(
        state, config->alice_angle_2, config->bob_angle_1, trials_per_correlation);

    results->E_a2b2 = bell_test_measure_correlation(
        state, config->alice_angle_2, config->bob_angle_2, trials_per_correlation);

    clock_t end = clock();

    // Calculate CHSH S value
    // S = E(a₁,b₁) + E(a₁,b₂) + E(a₂,b₁) - E(a₂,b₂)
    results->S = results->E_a1b1 + results->E_a1b2 +
                 results->E_a2b1 - results->E_a2b2;

    // Statistical error (simplified)
    // σ(S) ≈ 2/√N for N trials
    results->S_error = 2.0 / sqrt((double)config->num_trials);

    // Check violation
    results->violates_classical = (fabs(results->S) > 2.0);
    results->sigma_violation = (fabs(results->S) - 2.0) / results->S_error;

    results->total_trials = config->num_trials;
    results->execution_time_ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;

    quantum_state_destroy(state);
    return 0;
}

// ============================================================================
// ANALYSIS
// ============================================================================

void bell_test_print_results(const bell_test_results_t* results) {
    if (!results) return;

    printf("\n========================================\n");
    printf("         CHSH Bell Test Results         \n");
    printf("========================================\n\n");

    printf("Correlations:\n");
    printf("  E(a₁, b₁) = %+.4f\n", results->E_a1b1);
    printf("  E(a₁, b₂) = %+.4f\n", results->E_a1b2);
    printf("  E(a₂, b₁) = %+.4f\n", results->E_a2b1);
    printf("  E(a₂, b₂) = %+.4f\n", results->E_a2b2);
    printf("\n");

    printf("CHSH Value:\n");
    printf("  S = %+.4f ± %.4f\n", results->S, results->S_error);
    printf("\n");

    printf("Classical limit:     |S| ≤ 2.000\n");
    printf("Tsirelson bound:     |S| ≤ 2.828 (2√2)\n");
    printf("Measured |S|:        %.4f\n", fabs(results->S));
    printf("\n");

    if (results->violates_classical) {
        printf("*** BELL INEQUALITY VIOLATED ***\n");
        printf("Violation: %.2f σ above classical limit\n", results->sigma_violation);
    } else {
        printf("No statistically significant violation.\n");
    }

    printf("\n");
    printf("Statistics:\n");
    printf("  Total trials: %d\n", results->total_trials);
    printf("  Time: %.2f ms\n", results->execution_time_ms);
    printf("  Throughput: %.0f trials/sec\n",
           results->total_trials / (results->execution_time_ms / 1000.0));

    printf("\n========================================\n");
}

int bell_test_is_violation_significant(const bell_test_results_t* results,
                                       double confidence_level) {
    if (!results) return 0;

    // Z-score for confidence level
    // 0.95 -> 1.96, 0.99 -> 2.576, 0.999 -> 3.291
    double z;
    if (confidence_level >= 0.999) z = 3.291;
    else if (confidence_level >= 0.99) z = 2.576;
    else if (confidence_level >= 0.95) z = 1.96;
    else z = 1.645;

    return results->sigma_violation > z;
}

double bell_test_theoretical_S(double a1, double a2, double b1, double b2) {
    // For |Φ⁺⟩ state:
    // E(a, b) = -cos(a - b)
    double E11 = -cos(a1 - b1);
    double E12 = -cos(a1 - b2);
    double E21 = -cos(a2 - b1);
    double E22 = -cos(a2 - b2);

    return E11 + E12 + E21 - E22;
}

// ============================================================================
// COMPREHENSIVE TESTS
// ============================================================================

int bell_test_all_states(int num_trials, bell_test_results_t* results) {
    if (!results || num_trials <= 0) return -1;

    bell_test_config_t config = bell_test_default_config();
    config.num_trials = num_trials;

    for (int i = 0; i < 4; i++) {
        config.state = (bell_state_type_t)i;
        int ret = bell_test_run_chsh(&config, &results[i]);
        if (ret != 0) return ret;
    }

    return 0;
}

int bell_test_angle_sweep(int angle_steps, double* S_values,
                          int trials_per_angle) {
    if (!S_values || angle_steps <= 0 || trials_per_angle <= 0) return -1;

    quantum_state_t* state = quantum_state_create(2);
    if (!state) return -1;

    for (int i = 0; i < angle_steps; i++) {
        double angle = M_PI * (double)i / (double)(angle_steps - 1);

        // Fixed Alice angles, sweep Bob's second angle
        double a1 = 0.0;
        double a2 = M_PI / 4.0;
        double b1 = M_PI / 8.0;
        double b2 = angle;

        double E11 = bell_test_measure_correlation(state, a1, b1, trials_per_angle / 4);
        double E12 = bell_test_measure_correlation(state, a1, b2, trials_per_angle / 4);
        double E21 = bell_test_measure_correlation(state, a2, b1, trials_per_angle / 4);
        double E22 = bell_test_measure_correlation(state, a2, b2, trials_per_angle / 4);

        S_values[i] = E11 + E12 + E21 - E22;
    }

    quantum_state_destroy(state);
    return 0;
}

double bell_test_benchmark(int num_trials) {
    if (num_trials <= 0) return 0.0;

    bell_test_config_t config = bell_test_default_config();
    config.num_trials = num_trials;

    bell_test_results_t results;

    clock_t start = clock();
    bell_test_run_chsh(&config, &results);
    clock_t end = clock();

    double elapsed_sec = (double)(end - start) / CLOCKS_PER_SEC;
    return (double)num_trials / elapsed_sec;
}

// ============================================================================
// NOISE EFFECTS
// ============================================================================

int bell_test_run_noisy(const bell_test_config_t* config,
                        double depolarizing_rate,
                        bell_test_results_t* results) {
    if (!config || !results) return -1;

    // For noisy simulation, the correlation is reduced:
    // E_noisy = E_ideal * (1 - p)^2
    // where p is single-qubit depolarizing rate

    // Run ideal test first
    int ret = bell_test_run_chsh(config, results);
    if (ret != 0) return ret;

    // Apply noise reduction factor
    double visibility = pow(1.0 - depolarizing_rate, 2);

    results->E_a1b1 *= visibility;
    results->E_a1b2 *= visibility;
    results->E_a2b1 *= visibility;
    results->E_a2b2 *= visibility;

    // Recalculate S
    results->S = results->E_a1b1 + results->E_a1b2 +
                 results->E_a2b1 - results->E_a2b2;

    results->violates_classical = (fabs(results->S) > 2.0);
    results->sigma_violation = (fabs(results->S) - 2.0) / results->S_error;

    return 0;
}

double bell_test_find_noise_threshold(double target_S, double precision) {
    // Binary search for noise rate
    double low = 0.0;
    double high = 1.0;

    bell_test_config_t config = bell_test_default_config();
    config.num_trials = 1000;  // Faster for search

    bell_test_results_t results;

    while (high - low > precision) {
        double mid = (low + high) / 2.0;

        bell_test_run_noisy(&config, mid, &results);

        if (fabs(results.S) > target_S) {
            low = mid;
        } else {
            high = mid;
        }
    }

    return (low + high) / 2.0;
}

// ============================================================================
// COMMAND-LINE INTERFACE
// ============================================================================

static void print_usage(const char* program) {
    printf("Usage: %s [options]\n\n", program);
    printf("Options:\n");
    printf("  -n, --trials N     Number of trials (default: 10000)\n");
    printf("  -s, --state TYPE   Bell state: phi+, phi-, psi+, psi- (default: phi+)\n");
    printf("  -q, --quiet        Minimal output\n");
    printf("  -b, --benchmark    Run benchmark mode\n");
    printf("  --noise RATE       Add depolarizing noise\n");
    printf("  -h, --help         Show this help\n");
}

int bell_test_main(int argc, char** argv) {
    bell_test_config_t config = bell_test_default_config();
    int quiet = 0;
    int benchmark = 0;
    double noise_rate = 0.0;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--trials") == 0) {
            if (i + 1 < argc) {
                config.num_trials = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--state") == 0) {
            if (i + 1 < argc) {
                const char* state_str = argv[++i];
                if (strcmp(state_str, "phi+") == 0) config.state = BELL_STATE_PHI_PLUS;
                else if (strcmp(state_str, "phi-") == 0) config.state = BELL_STATE_PHI_MINUS;
                else if (strcmp(state_str, "psi+") == 0) config.state = BELL_STATE_PSI_PLUS;
                else if (strcmp(state_str, "psi-") == 0) config.state = BELL_STATE_PSI_MINUS;
            }
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
            quiet = 1;
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--benchmark") == 0) {
            benchmark = 1;
        } else if (strcmp(argv[i], "--noise") == 0) {
            if (i + 1 < argc) {
                noise_rate = atof(argv[++i]);
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (benchmark) {
        printf("Benchmarking Bell test...\n");
        double throughput = bell_test_benchmark(100000);
        printf("Throughput: %.0f trials/second\n", throughput);
        return 0;
    }

    bell_test_results_t results;
    int ret;

    if (noise_rate > 0.0) {
        if (!quiet) printf("Running noisy Bell test (p=%.4f)...\n", noise_rate);
        ret = bell_test_run_noisy(&config, noise_rate, &results);
    } else {
        if (!quiet) printf("Running Bell test (%d trials)...\n", config.num_trials);
        ret = bell_test_run_chsh(&config, &results);
    }

    if (ret != 0) {
        fprintf(stderr, "Error running Bell test\n");
        return 1;
    }

    if (quiet) {
        printf("S = %.4f\n", results.S);
        printf("Violation: %s\n", results.violates_classical ? "YES" : "NO");
    } else {
        bell_test_print_results(&results);
    }

    // Cleanup
    if (g_entropy) {
        entropy_destroy(g_entropy);
        g_entropy = NULL;
    }

    return results.violates_classical ? 0 : 1;
}
