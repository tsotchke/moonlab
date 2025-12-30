/**
 * @file test_quantum_gates.c
 * @brief Comprehensive unit tests for quantum gate operations
 * 
 * Tests all quantum gates including:
 * - Single-qubit gates (Pauli, Hadamard, Phase, Rotations)
 * - Two-qubit gates (CNOT, CZ, SWAP)
 * - Three-qubit gates (Toffoli, Fredkin)
 * - Gate correctness and unitary properties
 * - Edge cases and error handling
 * 
 * Target: 80%+ code coverage
 */

#include "../../src/quantum/gates.h"
#include "../../src/quantum/state.h"
#include "../../src/utils/quantum_entropy.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

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

#define ASSERT_EQ(a, b, msg) \
    if ((a) != (b)) TEST_FAIL(msg)

#define ASSERT_NEAR(a, b, tol, msg) \
    if (fabs((a) - (b)) > (tol)) { \
        printf("\n  Expected: %.10f, Got: %.10f\n", (double)(b), (double)(a)); \
        TEST_FAIL(msg); \
    }

// ============================================================================
// PAULI GATES TESTS
// ============================================================================

int test_pauli_x_gate() {
    TEST_START("Pauli-X gate (NOT)");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    // X on |0⟩ → |1⟩
    qs_error_t err = gate_pauli_x(&state, 0);
    ASSERT_EQ(err, QS_SUCCESS, "X gate failed");
    ASSERT_NEAR(creal(state.amplitudes[0]), 0.0, 1e-10, "X didn't flip to |1⟩");
    ASSERT_NEAR(creal(state.amplitudes[2]), 1.0, 1e-10, "X didn't flip to |1⟩");
    
    // X again should return to |0⟩
    gate_pauli_x(&state, 0);
    ASSERT_NEAR(creal(state.amplitudes[0]), 1.0, 1e-10, "X² ≠ I");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_pauli_y_gate() {
    TEST_START("Pauli-Y gate");
    
    quantum_state_t state;
    quantum_state_init(&state, 1);
    
    qs_error_t err = gate_pauli_y(&state, 0);
    ASSERT_EQ(err, QS_SUCCESS, "Y gate failed");
    
    // Y|0⟩ = i|1⟩
    ASSERT_NEAR(cimag(state.amplitudes[1]), 1.0, 1e-10, "Y|0⟩ ≠ i|1⟩");
    ASSERT_NEAR(creal(state.amplitudes[1]), 0.0, 1e-10, "Y|0⟩ has real part");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_pauli_z_gate() {
    TEST_START("Pauli-Z gate");
    
    quantum_state_t state;
    quantum_state_init(&state, 1);
    
    // Z|0⟩ = |0⟩
    qs_error_t err = gate_pauli_z(&state, 0);
    ASSERT_EQ(err, QS_SUCCESS, "Z gate failed");
    ASSERT_NEAR(creal(state.amplitudes[0]), 1.0, 1e-10, "Z changed |0⟩");
    
    // Z|1⟩ = -|1⟩
    state.amplitudes[0] = 0.0;
    state.amplitudes[1] = 1.0;
    gate_pauli_z(&state, 0);
    ASSERT_NEAR(creal(state.amplitudes[1]), -1.0, 1e-10, "Z|1⟩ ≠ -|1⟩");
    
    quantum_state_free(&state);
    TEST_PASS();
}

// ============================================================================
// HADAMARD GATE TESTS
// ============================================================================

int test_hadamard_gate() {
    TEST_START("Hadamard gate (superposition)");
    
    quantum_state_t state;
    quantum_state_init(&state, 1);
    
    // H|0⟩ = (|0⟩ + |1⟩)/√2
    qs_error_t err = gate_hadamard(&state, 0);
    ASSERT_EQ(err, QS_SUCCESS, "Hadamard failed");
    
    double expected = 1.0 / sqrt(2.0);
    ASSERT_NEAR(creal(state.amplitudes[0]), expected, 1e-10, "H|0⟩ amplitude 0 wrong");
    ASSERT_NEAR(creal(state.amplitudes[1]), expected, 1e-10, "H|0⟩ amplitude 1 wrong");
    
    // H² = I
    gate_hadamard(&state, 0);
    ASSERT_NEAR(creal(state.amplitudes[0]), 1.0, 1e-10, "H² ≠ I");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_hadamard_creates_equal_superposition() {
    TEST_START("Hadamard creates equal superposition");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    // Apply H to both qubits
    gate_hadamard(&state, 0);
    gate_hadamard(&state, 1);
    
    // All 4 basis states should have equal amplitude
    double expected = 0.5;
    for (int i = 0; i < 4; i++) {
        double prob = creal(state.amplitudes[i]) * creal(state.amplitudes[i]);
        ASSERT_NEAR(prob, expected, 1e-10, "Not equal superposition");
    }
    
    quantum_state_free(&state);
    TEST_PASS();
}

// ============================================================================
// PHASE GATES TESTS
// ============================================================================

int test_s_gate() {
    TEST_START("S gate (phase gate)");
    
    quantum_state_t state;
    quantum_state_init(&state, 1);
    state.amplitudes[0] = 0.0;
    state.amplitudes[1] = 1.0;
    
    // S|1⟩ = i|1⟩
    gate_s(&state, 0);
    ASSERT_NEAR(cimag(state.amplitudes[1]), 1.0, 1e-10, "S|1⟩ ≠ i|1⟩");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_t_gate() {
    TEST_START("T gate (π/8 gate)");
    
    quantum_state_t state;
    quantum_state_init(&state, 1);
    state.amplitudes[0] = 0.0;
    state.amplitudes[1] = 1.0;
    
    gate_t(&state, 0);
    
    // T|1⟩ = e^(iπ/4)|1⟩
    complex_t expected = cexp(I * M_PI / 4.0);
    ASSERT_NEAR(cabs(state.amplitudes[1] - expected), 0.0, 1e-10, "T gate incorrect");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_custom_phase_gate() {
    TEST_START("Custom phase gate");
    
    quantum_state_t state;
    quantum_state_init(&state, 1);
    state.amplitudes[0] = 0.0;
    state.amplitudes[1] = 1.0;
    
    double theta = M_PI / 3.0;
    gate_phase(&state, 0, theta);
    
    complex_t expected = cexp(I * theta);
    ASSERT_NEAR(cabs(state.amplitudes[1] - expected), 0.0, 1e-10, "Phase gate incorrect");
    
    quantum_state_free(&state);
    TEST_PASS();
}

// ============================================================================
// ROTATION GATES TESTS
// ============================================================================

int test_rx_rotation() {
    TEST_START("RX rotation gate");
    
    quantum_state_t state;
    quantum_state_init(&state, 1);
    
    // RX(π) is equivalent to X
    gate_rx(&state, 0, M_PI);
    ASSERT_NEAR(creal(state.amplitudes[1]), 1.0, 1e-10, "RX(π) ≠ X");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_ry_rotation() {
    TEST_START("RY rotation gate");
    
    quantum_state_t state;
    quantum_state_init(&state, 1);
    
    // RY(π/2) creates superposition like H
    gate_ry(&state, 0, M_PI / 2.0);
    
    double expected = 1.0 / sqrt(2.0);
    ASSERT_NEAR(creal(state.amplitudes[0]), expected, 1e-10, "RY(π/2) incorrect");
    ASSERT_NEAR(creal(state.amplitudes[1]), expected, 1e-10, "RY(π/2) incorrect");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_rz_rotation() {
    TEST_START("RZ rotation gate");
    
    quantum_state_t state;
    quantum_state_init(&state, 1);
    state.amplitudes[0] = 0.0;
    state.amplitudes[1] = 1.0;
    
    gate_rz(&state, 0, M_PI);
    
    // RZ(π)|1⟩ = -i|1⟩
    ASSERT_NEAR(cimag(state.amplitudes[1]), -1.0, 1e-10, "RZ(π) incorrect");
    
    quantum_state_free(&state);
    TEST_PASS();
}

// ============================================================================
// TWO-QUBIT GATES TESTS
// ============================================================================

int test_cnot_gate() {
    TEST_START("CNOT gate (entanglement)");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    // CNOT on |00⟩ → |00⟩
    gate_cnot(&state, 0, 1);
    ASSERT_NEAR(creal(state.amplitudes[0]), 1.0, 1e-10, "CNOT changed |00⟩");
    
    // Prepare |10⟩, CNOT → |11⟩
    quantum_state_reset(&state);
    state.amplitudes[0] = 0.0;
    state.amplitudes[2] = 1.0;
    gate_cnot(&state, 0, 1);
    ASSERT_NEAR(creal(state.amplitudes[3]), 1.0, 1e-10, "CNOT didn't flip target");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_cnot_creates_bell_state() {
    TEST_START("CNOT creates Bell state");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    // H ⊗ I, then CNOT creates |Φ⁺⟩
    gate_hadamard(&state, 0);
    gate_cnot(&state, 0, 1);
    
    // |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    double expected = 1.0 / sqrt(2.0);
    ASSERT_NEAR(creal(state.amplitudes[0]), expected, 1e-10, "Bell state incorrect");
    ASSERT_NEAR(creal(state.amplitudes[3]), expected, 1e-10, "Bell state incorrect");
    ASSERT_NEAR(creal(state.amplitudes[1]), 0.0, 1e-10, "|01⟩ should be 0");
    ASSERT_NEAR(creal(state.amplitudes[2]), 0.0, 1e-10, "|10⟩ should be 0");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_cz_gate() {
    TEST_START("CZ gate");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    // CZ on |11⟩ adds phase
    state.amplitudes[0] = 0.0;
    state.amplitudes[3] = 1.0;
    
    gate_cz(&state, 0, 1);
    ASSERT_NEAR(creal(state.amplitudes[3]), -1.0, 1e-10, "CZ didn't add phase");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_swap_gate() {
    TEST_START("SWAP gate");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    // Prepare |10⟩
    state.amplitudes[0] = 0.0;
    state.amplitudes[2] = 1.0;
    
    // SWAP → |01⟩
    gate_swap(&state, 0, 1);
    ASSERT_NEAR(creal(state.amplitudes[1]), 1.0, 1e-10, "SWAP didn't exchange");
    
    quantum_state_free(&state);
    TEST_PASS();
}

// ============================================================================
// THREE-QUBIT GATES TESTS
// ============================================================================

int test_toffoli_gate() {
    TEST_START("Toffoli gate (CCNOT)");
    
    quantum_state_t state;
    quantum_state_init(&state, 3);
    
    // Toffoli on |110⟩ → |111⟩
    state.amplitudes[0] = 0.0;
    state.amplitudes[6] = 1.0;  // |110⟩
    
    gate_toffoli(&state, 0, 1, 2);
    ASSERT_NEAR(creal(state.amplitudes[7]), 1.0, 1e-10, "Toffoli didn't flip");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_fredkin_gate() {
    TEST_START("Fredkin gate (CSWAP)");
    
    quantum_state_t state;
    quantum_state_init(&state, 3);
    
    // Fredkin on |110⟩ → |101⟩
    state.amplitudes[0] = 0.0;
    state.amplitudes[6] = 1.0;  // |110⟩
    
    gate_fredkin(&state, 0, 1, 2);
    ASSERT_NEAR(creal(state.amplitudes[5]), 1.0, 1e-10, "Fredkin didn't swap");
    
    quantum_state_free(&state);
    TEST_PASS();
}

// ============================================================================
// GATE PROPERTIES TESTS
// ============================================================================

int test_gate_preserves_normalization() {
    TEST_START("Gates preserve normalization");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    // Apply various gates
    gate_hadamard(&state, 0);
    ASSERT_TRUE(quantum_state_is_normalized(&state, 1e-10), "H broke normalization");
    
    gate_cnot(&state, 0, 1);
    ASSERT_TRUE(quantum_state_is_normalized(&state, 1e-10), "CNOT broke normalization");
    
    gate_pauli_x(&state, 0);
    ASSERT_TRUE(quantum_state_is_normalized(&state, 1e-10), "X broke normalization");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_gate_reversibility() {
    TEST_START("Gate reversibility (self-inverse)");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    // X² = I
    gate_pauli_x(&state, 0);
    gate_pauli_x(&state, 0);
    ASSERT_NEAR(creal(state.amplitudes[0]), 1.0, 1e-10, "X² ≠ I");
    
    // H² = I
    gate_hadamard(&state, 0);
    gate_hadamard(&state, 0);
    ASSERT_NEAR(creal(state.amplitudes[0]), 1.0, 1e-10, "H² ≠ I");
    
    quantum_state_free(&state);
    TEST_PASS();
}

// ============================================================================
// ERROR HANDLING TESTS
// ============================================================================

int test_invalid_qubit_index() {
    TEST_START("Invalid qubit index handling");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    // Out of bounds qubit
    qs_error_t err = gate_pauli_x(&state, 10);
    ASSERT_EQ(err, QS_ERROR_INVALID_QUBIT, "Should reject invalid qubit");
    
    err = gate_cnot(&state, 0, 10);
    ASSERT_EQ(err, QS_ERROR_INVALID_QUBIT, "Should reject invalid target");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_null_state_handling() {
    TEST_START("NULL state handling");
    
    qs_error_t err = gate_pauli_x(NULL, 0);
    ASSERT_EQ(err, QS_ERROR_INVALID_STATE, "Should reject NULL state");
    
    err = gate_cnot(NULL, 0, 1);
    ASSERT_EQ(err, QS_ERROR_INVALID_STATE, "Should reject NULL state");
    
    TEST_PASS();
}

// ============================================================================
// COMPLEX GATE SEQUENCES
// ============================================================================

int test_gate_sequence_bell_state() {
    TEST_START("Complex gate sequence (Bell state preparation)");
    
    quantum_state_t state;
    quantum_state_init(&state, 2);
    
    // Create all 4 Bell states
    gate_hadamard(&state, 0);
    gate_cnot(&state, 0, 1);
    
    // Verify entanglement
    int subsystem[] = {0};
    double ent = quantum_state_entanglement_entropy(&state, subsystem, 1);
    ASSERT_NEAR(ent, 1.0, 1e-9, "Bell state not maximally entangled");
    
    quantum_state_free(&state);
    TEST_PASS();
}

int test_gate_sequence_ghz_state() {
    TEST_START("GHZ state preparation (3-qubit entanglement)");
    
    quantum_state_t state;
    quantum_state_init(&state, 3);
    
    // GHZ: H on first, CNOT to others
    gate_hadamard(&state, 0);
    gate_cnot(&state, 0, 1);
    gate_cnot(&state, 0, 2);
    
    // |GHZ⟩ = (|000⟩ + |111⟩)/√2
    double expected = 1.0 / sqrt(2.0);
    ASSERT_NEAR(creal(state.amplitudes[0]), expected, 1e-10, "GHZ state incorrect");
    ASSERT_NEAR(creal(state.amplitudes[7]), expected, 1e-10, "GHZ state incorrect");
    
    quantum_state_free(&state);
    TEST_PASS();
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║      QUANTUM GATES OPERATIONS UNIT TEST SUITE            ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    // Pauli gates
    test_pauli_x_gate();
    test_pauli_y_gate();
    test_pauli_z_gate();
    
    // Hadamard
    test_hadamard_gate();
    test_hadamard_creates_equal_superposition();
    
    // Phase gates
    test_s_gate();
    test_t_gate();
    test_custom_phase_gate();
    
    // Rotation gates
    test_rx_rotation();
    test_ry_rotation();
    test_rz_rotation();
    
    // Two-qubit gates
    test_cnot_gate();
    test_cnot_creates_bell_state();
    test_cz_gate();
    test_swap_gate();
    
    // Three-qubit gates
    test_toffoli_gate();
    test_fredkin_gate();
    
    // Gate properties
    test_gate_preserves_normalization();
    test_gate_reversibility();
    
    // Error handling
    test_invalid_qubit_index();
    test_null_state_handling();
    
    // Complex sequences
    test_gate_sequence_bell_state();
    test_gate_sequence_ghz_state();
    
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