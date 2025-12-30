/**
 * @file test_quantum_state.c
 * @brief Comprehensive unit tests for quantum state management
 * 
 * Tests all quantum state operations including:
 * - State initialization and memory management
 * - State properties (normalization, entropy, purity, fidelity)
 * - Entanglement measures
 * - Measurement history
 * - Edge cases and error handling
 * 
 * Target: 80%+ code coverage
 */

#include "../../src/quantum/state.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

// Test framework macros
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_START(name) \
    do { \
        tests_run++; \
        printf("\n[%d] Testing: %s... ", tests_run, name); \
        fflush(stdout); \
    } while(0)

#define TEST_PASS() \
    do { \
        tests_passed++; \
        printf("✓ PASS\n"); \
        return 1; \
    } while(0)

#define TEST_FAIL(msg) \
    do { \
        tests_failed++; \
        printf("✗ FAIL: %s\n", msg); \
        return 0; \
    } while(0)

#define ASSERT_TRUE(expr, msg) \
    if (!(expr)) TEST_FAIL(msg)

#define ASSERT_FALSE(expr, msg) \
    if (expr) TEST_FAIL(msg)

#define ASSERT_EQ(a, b, msg) \
    if ((a) != (b)) TEST_FAIL(msg)

#define ASSERT_NEAR(a, b, tol, msg) \
    if (fabs((a) - (b)) > (tol)) { \
        printf("\n  Expected: %.10f, Got: %.10f, Diff: %.10e\n", (double)(b), (double)(a), fabs((double)(a) - (double)(b))); \
        TEST_FAIL(msg); \
    }

// ============================================================================
// INITIALIZATION AND MEMORY MANAGEMENT TESTS
// ============================================================================

int test_state_init_basic() {
    TEST_START("Basic state initialization");
    
    quantum_state_t state;
    qs_error_t err = quantum_state_init(&state, 2);
    
    ASSERT_EQ(err, QS_SUCCESS, "Initialization failed");
    ASSERT_EQ(state.num_qubits, 2, "Incorrect qubit count");
    ASSERT_EQ(state.state_dim, 4, "Incorrect state dimension");
    ASSERT_TRUE(state.amplitudes != NULL, "Amplitudes not allocated");
    ASSERT_TRUE(state.owns_memory, "Should own memory");
    
    // Check initial state is |00⟩
    ASSERT_NEAR(creal(state.amplitudes[0]), 1.0, 1e-10, "Initial state not |00⟩");
    ASSERT_NEAR(cimag(state.amplitudes[0]), 0.0, 1e-10, "Initial state has imaginary part");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_state_init_various_sizes() {
    TEST_START("State initialization with various sizes");
    
    size_t test_sizes[] = {1, 2, 4, 8, 10, 16, 20};
    
    for (size_t i = 0; i < sizeof(test_sizes)/sizeof(test_sizes[0]); i++) {
        quantum_state_t state;
        qs_error_t err = quantum_state_init(&state, test_sizes[i]);
        
        ASSERT_EQ(err, QS_SUCCESS, "Init failed for size");
        ASSERT_EQ(state.num_qubits, test_sizes[i], "Wrong qubit count");
        ASSERT_EQ(state.state_dim, 1ULL << test_sizes[i], "Wrong dimension");
        
        quantum_state_free(&state);
    }
    
    TEST_PASS();
}

int test_state_init_invalid_params() {
    TEST_START("State initialization with invalid parameters");
    
    quantum_state_t state;
    
    // NULL pointer
    qs_error_t err = quantum_state_init(NULL, 4);
    ASSERT_EQ(err, QS_ERROR_INVALID_STATE, "Should reject NULL pointer");
    
    // Zero qubits
    err = quantum_state_init(&state, 0);
    ASSERT_EQ(err, QS_ERROR_INVALID_DIMENSION, "Should reject 0 qubits");
    
    // Too many qubits
    err = quantum_state_init(&state, MAX_QUBITS + 1);
    ASSERT_EQ(err, QS_ERROR_INVALID_DIMENSION, "Should reject excessive qubits");
    
    TEST_PASS();
}

int test_state_clone() {
    TEST_START("State cloning");
    
    quantum_state_t original, clone;
    quantum_state_init(&original, 3);
    
    // Modify original
    original.amplitudes[0] = 0.5 + 0.0*I;
    original.amplitudes[1] = 0.5 + 0.0*I;
    original.amplitudes[2] = 0.5 + 0.0*I;
    original.amplitudes[3] = 0.5 + 0.0*I;
    original.global_phase = 0.5;
    
    qs_error_t err = quantum_state_clone(&clone, &original);
    ASSERT_EQ(err, QS_SUCCESS, "Cloning failed");
    
    // Verify clone
    ASSERT_EQ(clone.num_qubits, original.num_qubits, "Qubit count mismatch");
    ASSERT_NEAR(creal(clone.amplitudes[0]), creal(original.amplitudes[0]), 1e-10, "Amplitude mismatch");
    ASSERT_NEAR(clone.global_phase, original.global_phase, 1e-10, "Phase mismatch");
    
    // Verify deep copy
    clone.amplitudes[0] = 1.0;
    ASSERT_NEAR(creal(original.amplitudes[0]), 0.5, 1e-10, "Not a deep copy");
    
    quantum_state_free(&original);
    quantum_state_free(&clone);
    TEST_PASS();
}

int test_state_reset() {
    TEST_START("State reset");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    // Modify state
    state.amplitudes[0] = 0.0;
    state.amplitudes[1] = 1.0;
    state.global_phase = 1.5;
    
    quantum_state_reset(&state);
    
    // Check reset to |00⟩
    ASSERT_NEAR(creal(state.amplitudes[0]), 1.0, 1e-10, "Not reset to |00⟩");
    ASSERT_NEAR(state.global_phase, 0.0, 1e-10, "Phase not reset");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_state_from_amplitudes() {
    TEST_START("Create state from amplitudes");
    
    complex_t amplitudes[4] = {
        0.5 + 0.0*I,
        0.5 + 0.0*I,
        0.5 + 0.0*I,
        0.5 + 0.0*I
    };
    
    quantum_state_t state;
    qs_error_t err = quantum_state_from_amplitudes(&state, amplitudes, 4);
    
    ASSERT_EQ(err, QS_SUCCESS, "Failed to create from amplitudes");
    ASSERT_EQ(state.num_qubits, 2, "Wrong qubit count");
    ASSERT_NEAR(creal(state.amplitudes[0]), 0.5, 1e-10, "Amplitude mismatch");
    
    quantum_state_free(&state);
    TEST_PASS();
}

// ============================================================================
// NORMALIZATION TESTS
// ============================================================================

int test_state_normalization_check() {
    TEST_START("Normalization check");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    ASSERT_TRUE(quantum_state_is_normalized(&state, 1e-10), "Initial state not normalized");
    
    // Unnormalize
    state.amplitudes[0] = 2.0;
    ASSERT_FALSE(quantum_state_is_normalized(&state, 1e-10), "Should detect unnormalized");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_state_normalize() {
    TEST_START("State normalization");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    // Create unnormalized state
    state.amplitudes[0] = 1.0 + 0.0*I;
    state.amplitudes[1] = 1.0 + 0.0*I;
    state.amplitudes[2] = 1.0 + 0.0*I;
    state.amplitudes[3] = 1.0 + 0.0*I;
    
    qs_error_t err = quantum_state_normalize(&state);
    ASSERT_EQ(err, QS_SUCCESS, "Normalization failed");
    ASSERT_TRUE(quantum_state_is_normalized(&state, 1e-10), "Not normalized after normalize()");
    
    // Check correct normalization
    ASSERT_NEAR(creal(state.amplitudes[0]), 0.5, 1e-10, "Incorrect normalization");
    
    quantum_state_free(&state);
    TEST_PASS();
}

// ============================================================================
// PROPERTY CALCULATION TESTS
// ============================================================================

int test_state_entropy() {
    TEST_START("State entropy calculation");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    // Pure state |00⟩: entropy = 0
    double entropy = quantum_state_entropy(&state);
    ASSERT_NEAR(entropy, 0.0, 1e-10, "Pure state entropy not 0");
    
    // Maximally mixed state
    state.amplitudes[0] = 0.5 + 0.0*I;
    state.amplitudes[1] = 0.5 + 0.0*I;
    state.amplitudes[2] = 0.5 + 0.0*I;
    state.amplitudes[3] = 0.5 + 0.0*I;
    
    entropy = quantum_state_entropy(&state);
    ASSERT_NEAR(entropy, 2.0, 1e-10, "Max entropy not log2(4) = 2");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_state_purity() {
    TEST_START("State purity calculation");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    // Pure state: purity = 1
    double purity = quantum_state_purity(&state);
    ASSERT_NEAR(purity, 1.0, 1e-10, "Pure state purity not 1");
    
    // Mixed state: purity < 1
    state.amplitudes[0] = 0.5;
    state.amplitudes[1] = 0.5;
    state.amplitudes[2] = 0.5;
    state.amplitudes[3] = 0.5;
    
    purity = quantum_state_purity(&state);
    ASSERT_TRUE(purity < 1.0, "Mixed state purity should be < 1");
    ASSERT_NEAR(purity, 0.25, 1e-10, "Incorrect purity for equal superposition");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_state_fidelity() {
    TEST_START("State fidelity calculation");
    
    quantum_state_t state1, state2;
    quantum_state_init(&state1, 2);
    quantum_state_init(&state2, 2);
    
    // Identical states: fidelity = 1
    double fidelity = quantum_state_fidelity(&state1, &state2);
    ASSERT_NEAR(fidelity, 1.0, 1e-10, "Identical states fidelity not 1");
    
    // Orthogonal states: fidelity = 0
    state2.amplitudes[0] = 0.0;
    state2.amplitudes[1] = 1.0;
    
    fidelity = quantum_state_fidelity(&state1, &state2);
    ASSERT_NEAR(fidelity, 0.0, 1e-10, "Orthogonal states fidelity not 0");
    
    quantum_state_free(&state1);
    quantum_state_free(&state2);
    TEST_PASS();
}

// ============================================================================
// AMPLITUDE AND PROBABILITY TESTS
// ============================================================================

int test_get_amplitude() {
    TEST_START("Get amplitude for basis state");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    complex_t amp = quantum_state_get_amplitude(&state, 0);
    ASSERT_NEAR(creal(amp), 1.0, 1e-10, "Wrong amplitude for |00⟩");
    
    amp = quantum_state_get_amplitude(&state, 1);
    ASSERT_NEAR(creal(amp), 0.0, 1e-10, "Non-zero amplitude for |01⟩");
    
    // Out of bounds
    amp = quantum_state_get_amplitude(&state, 100);
    ASSERT_NEAR(creal(amp), 0.0, 1e-10, "Out of bounds should return 0");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_get_probability() {
    TEST_START("Get probability for basis state");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    double prob = quantum_state_get_probability(&state, 0);
    ASSERT_NEAR(prob, 1.0, 1e-10, "Wrong probability for |00⟩");
    
    prob = quantum_state_get_probability(&state, 1);
    ASSERT_NEAR(prob, 0.0, 1e-10, "Non-zero probability for |01⟩");
    
    quantum_state_free(&state);
    TEST_PASS();
}

// ============================================================================
// ENTANGLEMENT TESTS
// ============================================================================

int test_entanglement_entropy_product_state() {
    TEST_START("Entanglement entropy for product state");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    // Product state |00⟩: no entanglement
    int subsystem[] = {0};
    double ent = quantum_state_entanglement_entropy(&state, subsystem, 1);
    ASSERT_NEAR(ent, 0.0, 1e-10, "Product state should have 0 entanglement");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_entanglement_entropy_bell_state() {
    TEST_START("Entanglement entropy for Bell state");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    // Bell state (|00⟩ + |11⟩)/√2
    state.amplitudes[0] = 1.0/sqrt(2.0);
    state.amplitudes[1] = 0.0;
    state.amplitudes[2] = 0.0;
    state.amplitudes[3] = 1.0/sqrt(2.0);
    
    int subsystem[] = {0};
    double ent = quantum_state_entanglement_entropy(&state, subsystem, 1);
    ASSERT_NEAR(ent, 1.0, 1e-9, "Bell state should have max entanglement = 1 bit");
    
    quantum_state_free(&state);
    TEST_PASS();
}

// ============================================================================
// MEASUREMENT HISTORY TESTS
// ============================================================================

int test_measurement_recording() {
    TEST_START("Measurement history recording");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    // Record measurements
    quantum_state_record_measurement(&state, 0);
    quantum_state_record_measurement(&state, 1);
    quantum_state_record_measurement(&state, 3);
    
    ASSERT_EQ(state.num_measurements, 3, "Wrong measurement count");
    
    // Retrieve history
    uint64_t outcomes[3];
    size_t count = quantum_state_get_measurement_history(&state, outcomes, 3);
    
    ASSERT_EQ(count, 3, "Wrong history count");
    ASSERT_EQ(outcomes[0], 0, "Wrong first measurement");
    ASSERT_EQ(outcomes[1], 1, "Wrong second measurement");
    ASSERT_EQ(outcomes[2], 3, "Wrong third measurement");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_measurement_clear() {
    TEST_START("Clear measurement history");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    quantum_state_record_measurement(&state, 0);
    quantum_state_record_measurement(&state, 1);
    
    quantum_state_clear_measurements(&state);
    ASSERT_EQ(state.num_measurements, 0, "Measurements not cleared");
    
    quantum_state_free(&state);
    TEST_PASS();
}

// ============================================================================
// UTILITY FUNCTION TESTS
// ============================================================================

int test_basis_state_string() {
    TEST_START("Basis state string conversion");
    
    char buffer[10];
    
    quantum_basis_state_string(0, 2, buffer, sizeof(buffer));
    ASSERT_TRUE(strcmp(buffer, "00") == 0, "Wrong string for |00⟩");
    
    quantum_basis_state_string(3, 2, buffer, sizeof(buffer));
    ASSERT_TRUE(strcmp(buffer, "11") == 0, "Wrong string for |11⟩");
    
    quantum_basis_state_string(5, 3, buffer, sizeof(buffer));
    ASSERT_TRUE(strcmp(buffer, "101") == 0, "Wrong string for |101⟩");
    
    TEST_PASS();
}

// ============================================================================
// EDGE CASES AND ERROR HANDLING
// ============================================================================

int test_null_pointer_handling() {
    TEST_START("NULL pointer handling");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    // Test NULL state pointers
    quantum_state_free(NULL);  // Should not crash
    
    ASSERT_FALSE(quantum_state_is_normalized(NULL, 1e-10), "NULL should return false");
    
    ASSERT_NEAR(quantum_state_entropy(NULL), 0.0, 1e-10, "NULL should return 0");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_large_state() {
    TEST_START("Large quantum state (16 qubits)");
    
    quantum_state_t state;
    qs_error_t err = quantum_state_init(&state, 16);
    
    ASSERT_EQ(err, QS_SUCCESS, "Failed to create 16-qubit state");
    ASSERT_EQ(state.state_dim, 65536, "Wrong dimension for 16 qubits");
    ASSERT_TRUE(quantum_state_is_normalized(&state, 1e-10), "Large state not normalized");
    
    quantum_state_free(&state);
    TEST_PASS();
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     QUANTUM STATE MANAGEMENT UNIT TEST SUITE             ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    // Initialization tests
    test_state_init_basic();
    test_state_init_various_sizes();
    test_state_init_invalid_params();
    test_state_clone();
    test_state_reset();
    test_state_from_amplitudes();
    
    // Normalization tests
    test_state_normalization_check();
    test_state_normalize();
    
    // Property tests
    test_state_entropy();
    test_state_purity();
    test_state_fidelity();
    
    // Amplitude/probability tests
    test_get_amplitude();
    test_get_probability();
    
    // Entanglement tests
    test_entanglement_entropy_product_state();
    test_entanglement_entropy_bell_state();
    
    // Measurement history tests
    test_measurement_recording();
    test_measurement_clear();
    
    // Utility tests
    test_basis_state_string();
    
    // Edge cases
    test_null_pointer_handling();
    test_large_state();
    
    // Summary
    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║                    TEST SUMMARY                           ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║  Tests Run:    %3d                                        ║\n", tests_run);
    printf("║  Tests Passed: %3d                                        ║\n", tests_passed);
    printf("║  Tests Failed: %3d                                        ║\n", tests_failed);
    printf("║  Success Rate: %3.0f%%                                      ║\n", 
           100.0 * tests_passed / tests_run);
    printf("╚═══════════════════════════════════════════════════════════╝\n\n");
    
    return (tests_failed == 0) ? 0 : 1;
}