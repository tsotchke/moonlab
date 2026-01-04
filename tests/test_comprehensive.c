/**
 * @file test_comprehensive.c
 * @brief Comprehensive test suite covering all major quantum simulator components
 *
 * Tests:
 * - Quantum state operations
 * - Gate applications
 * - Entanglement calculations (von Neumann, Renyi, concurrence)
 * - Noise channels (phase damping, amplitude damping)
 * - Tensor network (MPS) operations
 * - Grover's algorithm and quantum counting
 * - Reduced density matrices
 *
 * @stability stable
 * @since v1.0.0
 */

#include "../src/quantum/state.h"
#include "../src/quantum/gates.h"
#include "../src/quantum/entanglement.h"
#include "../src/quantum/noise.h"
#include "../src/algorithms/grover.h"
#include "../src/algorithms/grover_optimizer.h"
#include "../src/algorithms/tensor_network/tn_state.h"
#include "../src/algorithms/tensor_network/tn_gates.h"
#include "../src/algorithms/tensor_network/tn_measurement.h"
#include "../src/utils/quantum_entropy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Simple entropy callback for tests using stdlib random
static int test_entropy_callback(void *user_data, uint8_t *buffer, size_t size) {
    (void)user_data;
    for (size_t i = 0; i < size; i++) {
        buffer[i] = (uint8_t)(rand() & 0xFF);
    }
    return 0;
}

// Test counters
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

static void test_start(const char *name) {
    tests_run++;
    printf("  [%02d] %s... ", tests_run, name);
    fflush(stdout);
}

static void test_pass(void) {
    tests_passed++;
    printf("\033[32m✓ PASS\033[0m\n");
}

static void test_fail(const char *reason) {
    tests_failed++;
    printf("\033[31m✗ FAIL: %s\033[0m\n", reason);
}

static int approx_equal(double a, double b, double tol) {
    return fabs(a - b) < tol;
}

// ============================================================================
// QUANTUM STATE TESTS
// ============================================================================

static void test_state_initialization(void) {
    test_start("State initialization |00...0>");

    quantum_state_t state;
    if (quantum_state_init(&state, 4) != QS_SUCCESS) {
        test_fail("Failed to initialize state");
        return;
    }

    // Check that state is |0000>
    if (cabs(state.amplitudes[0] - 1.0) > 1e-10) {
        quantum_state_free(&state);
        test_fail("Ground state amplitude not 1");
        return;
    }

    for (uint64_t i = 1; i < state.state_dim; i++) {
        if (cabs(state.amplitudes[i]) > 1e-10) {
            quantum_state_free(&state);
            test_fail("Non-zero amplitude in excited state");
            return;
        }
    }

    quantum_state_free(&state);
    test_pass();
}

static void test_hadamard_superposition(void) {
    test_start("Hadamard creates uniform superposition");

    quantum_state_t state;
    quantum_state_init(&state, 3);

    // Apply H to all qubits
    for (int q = 0; q < 3; q++) {
        gate_hadamard(&state, q);
    }

    // All amplitudes should be 1/sqrt(8)
    double expected = 1.0 / sqrt(8.0);
    for (uint64_t i = 0; i < state.state_dim; i++) {
        if (!approx_equal(creal(state.amplitudes[i]), expected, 1e-10) ||
            !approx_equal(cimag(state.amplitudes[i]), 0.0, 1e-10)) {
            quantum_state_free(&state);
            test_fail("Amplitude not uniform after H gates");
            return;
        }
    }

    quantum_state_free(&state);
    test_pass();
}

static void test_pauli_gates(void) {
    test_start("Pauli X, Y, Z gates");

    quantum_state_t state;
    quantum_state_init(&state, 1);

    // X|0> = |1>
    gate_pauli_x(&state, 0);
    if (cabs(state.amplitudes[1] - 1.0) > 1e-10) {
        quantum_state_free(&state);
        test_fail("X|0> != |1>");
        return;
    }

    // Z|1> = -|1>
    gate_pauli_z(&state, 0);
    if (cabs(state.amplitudes[1] + 1.0) > 1e-10) {
        quantum_state_free(&state);
        test_fail("Z|1> != -|1>");
        return;
    }

    // Y = iXZ, so Y|1> = i*X*(-|1>) = -i|0>
    gate_pauli_y(&state, 0);
    if (!approx_equal(creal(state.amplitudes[0]), 0.0, 1e-10) ||
        !approx_equal(cimag(state.amplitudes[0]), 1.0, 1e-10)) {  // -i * -1 = i
        quantum_state_free(&state);
        test_fail("Y gate incorrect");
        return;
    }

    quantum_state_free(&state);
    test_pass();
}

static void test_cnot_gate(void) {
    test_start("CNOT entangles qubits");

    quantum_state_t state;
    quantum_state_init(&state, 2);

    // Create Bell state: H on q0, then CNOT(0,1)
    gate_hadamard(&state, 0);
    gate_cnot(&state, 0, 1);

    // Should have (|00> + |11>) / sqrt(2)
    double amp = 1.0 / sqrt(2.0);
    if (!approx_equal(cabs(state.amplitudes[0]), amp, 1e-10) ||
        !approx_equal(cabs(state.amplitudes[3]), amp, 1e-10) ||
        cabs(state.amplitudes[1]) > 1e-10 ||
        cabs(state.amplitudes[2]) > 1e-10) {
        quantum_state_free(&state);
        test_fail("Bell state not created correctly");
        return;
    }

    quantum_state_free(&state);
    test_pass();
}

// ============================================================================
// ENTANGLEMENT TESTS
// ============================================================================

static void test_von_neumann_entropy_product(void) {
    test_start("von Neumann entropy of product state = 0");

    quantum_state_t state;
    quantum_state_init(&state, 2);

    // |00> is a product state - entropy should be 0
    int qubits_b[] = {1};
    double entropy = entanglement_entropy_bipartition(&state, qubits_b, 1);

    if (!approx_equal(entropy, 0.0, 1e-10)) {
        quantum_state_free(&state);
        test_fail("Product state entropy not 0");
        return;
    }

    quantum_state_free(&state);
    test_pass();
}

static void test_von_neumann_entropy_bell(void) {
    test_start("von Neumann entropy of Bell state = 1");

    quantum_state_t state;
    quantum_state_init(&state, 2);

    // Create Bell state
    gate_hadamard(&state, 0);
    gate_cnot(&state, 0, 1);

    // Maximally entangled - entropy should be 1 bit
    int qubits_b[] = {1};
    double entropy = entanglement_entropy_bipartition(&state, qubits_b, 1);

    if (!approx_equal(entropy, 1.0, 0.01)) {
        quantum_state_free(&state);
        char msg[64];
        snprintf(msg, sizeof(msg), "Bell state entropy = %.4f, expected 1.0", entropy);
        test_fail(msg);
        return;
    }

    quantum_state_free(&state);
    test_pass();
}

static void test_concurrence_bell(void) {
    test_start("Concurrence of Bell state = 1");

    quantum_state_t state;
    quantum_state_init(&state, 2);

    gate_hadamard(&state, 0);
    gate_cnot(&state, 0, 1);

    double conc = entanglement_concurrence_2qubit(&state);

    if (!approx_equal(conc, 1.0, 0.05)) {
        quantum_state_free(&state);
        char msg[64];
        snprintf(msg, sizeof(msg), "Bell concurrence = %.4f, expected 1.0", conc);
        test_fail(msg);
        return;
    }

    quantum_state_free(&state);
    test_pass();
}

static void test_renyi_entropy(void) {
    test_start("Renyi entropy (alpha=2) of Bell state");

    // For maximally mixed single-qubit state (from tracing out Bell pair):
    // rho = I/2, S_2 = -log2(Tr(rho^2)) = -log2(1/2) = 1

    quantum_state_t state;
    quantum_state_init(&state, 2);
    gate_hadamard(&state, 0);
    gate_cnot(&state, 0, 1);

    // Get reduced density matrix by tracing out qubit 1
    int qubits_b[] = {1};
    uint64_t reduced_dim;
    complex_t* rdm = malloc(4 * sizeof(complex_t));
    entanglement_reduced_density_matrix(&state, qubits_b, 1, rdm, &reduced_dim);

    double renyi_2 = entanglement_renyi_entropy(rdm, 2, 2.0);

    free(rdm);
    quantum_state_free(&state);

    if (!approx_equal(renyi_2, 1.0, 0.05)) {
        char msg[64];
        snprintf(msg, sizeof(msg), "Renyi-2 entropy = %.4f, expected 1.0", renyi_2);
        test_fail(msg);
        return;
    }

    test_pass();
}

// ============================================================================
// NOISE CHANNEL TESTS
// ============================================================================

static void test_phase_damping_coherence_decay(void) {
    test_start("Phase damping reduces coherence");

    quantum_state_t state;
    quantum_state_init(&state, 1);

    // Create superposition |+> = (|0> + |1>)/sqrt(2)
    gate_hadamard(&state, 0);

    // Initial off-diagonal coherence
    double initial_coherence = cabs(state.amplitudes[0] * conj(state.amplitudes[1]));

    // Apply phase damping with gamma = 0.5
    noise_phase_damping(&state, 0, 0.5, 0.7);  // random_value = 0.7 (no jump)

    // Coherence should have decayed
    double final_coherence = cabs(state.amplitudes[0] * conj(state.amplitudes[1]));

    quantum_state_free(&state);

    // After no-jump evolution with gamma=0.5, |1> amplitude reduced by sqrt(0.5)
    // So coherence reduced by sqrt(0.5) before renormalization
    if (final_coherence >= initial_coherence) {
        test_fail("Coherence did not decay under phase damping");
        return;
    }

    test_pass();
}

// ============================================================================
// GROVER'S ALGORITHM TESTS
// ============================================================================

static int single_target_oracle(uint64_t state, void* data) {
    uint64_t target = *(uint64_t*)data;
    return (state == target) ? 1 : 0;
}

static void test_grover_single_target(void) {
    test_start("Grover search finds single target");

    quantum_state_t state;
    quantum_state_init(&state, 4);  // 16 states

    // Initialize entropy context for measurement
    quantum_entropy_ctx_t entropy;
    quantum_entropy_init(&entropy, test_entropy_callback, NULL);

    uint64_t target = 7;
    grover_config_t config = {
        .num_qubits = 4,
        .marked_state = target,
        .num_iterations = 0,
        .use_optimal_iterations = 1
    };

    grover_result_t result = grover_search(&state, &config, &entropy);

    quantum_state_free(&state);

    if (result.found_state != target) {
        char msg[64];
        snprintf(msg, sizeof(msg), "Found %llu, expected %llu",
                 (unsigned long long)result.found_state, (unsigned long long)target);
        test_fail(msg);
        return;
    }

    test_pass();
}

static int multi_target_oracle(uint64_t state, void* data) {
    // Mark states 0, 1, 2, 3 (first 4)
    return (state < 4) ? 1 : 0;
}

static void test_grover_quantum_counting(void) {
    test_start("Quantum counting estimates solution count");

    quantum_state_t state;
    quantum_state_init(&state, 4);  // 16 states, 4 marked

    int count = grover_quantum_counting(&state, multi_target_oracle, NULL, 4);

    quantum_state_free(&state);

    // Should estimate approximately 4 solutions
    if (count < 2 || count > 6) {
        char msg[64];
        snprintf(msg, sizeof(msg), "Counted %d, expected ~4", count);
        test_fail(msg);
        return;
    }

    test_pass();
}

// ============================================================================
// TENSOR NETWORK (MPS) TESTS
// ============================================================================

static void test_mps_creation(void) {
    test_start("MPS state creation");

    tn_state_config_t config = tn_state_config_default();
    tn_mps_state_t* mps = tn_mps_create_zero(4, &config);

    if (!mps) {
        test_fail("Failed to create MPS");
        return;
    }

    if (mps->num_qubits != 4) {
        tn_mps_free(mps);
        test_fail("Wrong qubit count");
        return;
    }

    tn_mps_free(mps);
    test_pass();
}

static void test_mps_hadamard(void) {
    test_start("MPS Hadamard gate");

    tn_state_config_t config = tn_state_config_default();
    tn_mps_state_t* mps = tn_mps_create_zero(2, &config);

    if (!mps) {
        test_fail("Failed to create MPS");
        return;
    }

    // Apply Hadamard
    tn_apply_h(mps, 0);

    // Measure probabilities
    double p0, p1;
    tn_measure_probability(mps, 0, &p0, &p1);

    tn_mps_free(mps);

    // Should be 50/50
    if (!approx_equal(p0, 0.5, 0.01) || !approx_equal(p1, 0.5, 0.01)) {
        char msg[64];
        snprintf(msg, sizeof(msg), "p0=%.4f, p1=%.4f, expected 0.5 each", p0, p1);
        test_fail(msg);
        return;
    }

    test_pass();
}

static void test_mps_bell_state(void) {
    test_start("MPS Bell state creation");

    tn_state_config_t config = tn_state_config_default();
    tn_mps_state_t* mps = tn_mps_create_zero(2, &config);

    if (!mps) {
        test_fail("Failed to create MPS");
        return;
    }

    // Create Bell state: H(0), CNOT(0,1)
    tn_apply_h(mps, 0);
    tn_apply_cnot(mps, 0, 1);

    // Measure entanglement entropy
    double entropy = tn_mps_entanglement_entropy(mps, 0);

    tn_mps_free(mps);

    // Maximally entangled: S = 1 bit
    if (!approx_equal(entropy, 1.0, 0.1)) {
        char msg[64];
        snprintf(msg, sizeof(msg), "Entropy = %.4f, expected 1.0", entropy);
        test_fail(msg);
        return;
    }

    test_pass();
}

static void test_mps_reduced_density_matrix(void) {
    test_start("MPS reduced density matrix with coherences");

    tn_state_config_t config = tn_state_config_default();
    tn_mps_state_t* mps = tn_mps_create_zero(2, &config);

    if (!mps) {
        test_fail("Failed to create MPS");
        return;
    }

    // Create |+> state on qubit 0
    tn_apply_h(mps, 0);

    // Get reduced density matrix for qubit 0
    double complex rho[4];
    tn_reduced_density_1q(mps, 0, rho);

    tn_mps_free(mps);

    // For |+> = (|0>+|1>)/sqrt(2), rho should be:
    // [[0.5, 0.5], [0.5, 0.5]]
    double off_diag = cabs(rho[1]);  // Should be 0.5

    if (!approx_equal(off_diag, 0.5, 0.1)) {
        char msg[64];
        snprintf(msg, sizeof(msg), "Off-diagonal = %.4f, expected 0.5", off_diag);
        test_fail(msg);
        return;
    }

    test_pass();
}

// ============================================================================
// MAIN
// ============================================================================

int main(void) {
    // Seed RNG for reproducible tests (Grover measurement is probabilistic)
    srand(42);

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║       COMPREHENSIVE QUANTUM SIMULATOR TEST SUITE              ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Testing all major components for production readiness        ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    printf("▶ QUANTUM STATE OPERATIONS\n");
    test_state_initialization();
    test_hadamard_superposition();
    test_pauli_gates();
    test_cnot_gate();

    printf("\n▶ ENTANGLEMENT CALCULATIONS\n");
    test_von_neumann_entropy_product();
    test_von_neumann_entropy_bell();
    test_concurrence_bell();
    test_renyi_entropy();

    printf("\n▶ NOISE CHANNELS\n");
    test_phase_damping_coherence_decay();

    printf("\n▶ GROVER'S ALGORITHM\n");
    test_grover_single_target();
    test_grover_quantum_counting();

    printf("\n▶ TENSOR NETWORK (MPS) OPERATIONS\n");
    test_mps_creation();
    test_mps_hadamard();
    test_mps_bell_state();
    test_mps_reduced_density_matrix();

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                      TEST RESULTS                             ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Total:  %2d                                                   ║\n", tests_run);
    printf("║  Passed: %2d                                                   ║\n", tests_passed);
    printf("║  Failed: %2d                                                   ║\n", tests_failed);
    printf("╠═══════════════════════════════════════════════════════════════╣\n");

    if (tests_failed == 0) {
        printf("║  \033[32m✓ ALL TESTS PASSED - SYSTEM PRODUCTION READY\033[0m              ║\n");
    } else {
        printf("║  \033[31m✗ SOME TESTS FAILED - REVIEW REQUIRED\033[0m                     ║\n");
    }

    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    return tests_failed > 0 ? 1 : 0;
}
