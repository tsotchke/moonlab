/**
 * @file test_stride_gates.c
 * @brief Unit tests for stride-based quantum gate operations
 *
 * Tests verify:
 * 1. Correctness against known quantum state transformations
 * 2. Mathematical properties (unitarity, gate identities)
 * 3. Edge cases (boundary qubits, adjacent qubits)
 * 4. Multi-qubit gate interactions
 *
 * Copyright 2024-2026 tsotchke
 * Licensed under the Apache License, Version 2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#include "../../src/optimization/stride_gates.h"

// ============================================================================
// TEST UTILITIES
// ============================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define EPSILON 1e-10

#define TEST_ASSERT(condition, message) do { \
    if (!(condition)) { \
        printf("  [FAIL] %s\n", message); \
        printf("         at %s:%d\n", __FILE__, __LINE__); \
        tests_failed++; \
        return 0; \
    } \
} while(0)

#define TEST_PASS(message) do { \
    printf("  [PASS] %s\n", message); \
    tests_passed++; \
} while(0)

#define RUN_TEST(test_func) do { \
    printf("\nRunning: %s\n", #test_func); \
    if (test_func()) { \
        printf("  Test passed.\n"); \
    } else { \
        printf("  Test FAILED.\n"); \
    } \
} while(0)

// Check if two complex numbers are approximately equal
static int complex_approx_equal(complex_t a, complex_t b) {
    return cabs(a - b) < EPSILON;
}

// Check if amplitude array has unit norm
static int is_normalized(const complex_t* amplitudes, int num_qubits) {
    uint64_t dim = 1ULL << num_qubits;
    double sum = 0.0;
    for (uint64_t i = 0; i < dim; i++) {
        sum += cabs(amplitudes[i]) * cabs(amplitudes[i]);
    }
    return fabs(sum - 1.0) < EPSILON;
}

// Initialize to |0...0⟩ state
static void init_zero_state(complex_t* amplitudes, int num_qubits) {
    uint64_t dim = 1ULL << num_qubits;
    memset(amplitudes, 0, dim * sizeof(complex_t));
    amplitudes[0] = 1.0;
}

// Initialize to |index⟩ computational basis state
static void init_basis_state(complex_t* amplitudes, int num_qubits, uint64_t index) {
    uint64_t dim = 1ULL << num_qubits;
    memset(amplitudes, 0, dim * sizeof(complex_t));
    amplitudes[index] = 1.0;
}

// ============================================================================
// SINGLE-QUBIT GATE TESTS
// ============================================================================

static int test_hadamard_on_zero(void) {
    // H|0⟩ = (|0⟩ + |1⟩) / √2
    int num_qubits = 3;
    uint64_t dim = 1ULL << num_qubits;
    complex_t* amp = malloc(dim * sizeof(complex_t));

    init_zero_state(amp, num_qubits);
    stride_hadamard(amp, num_qubits, 0);  // H on qubit 0

    // Expected: |000⟩ + |001⟩ (normalized)
    double inv_sqrt2 = 1.0 / sqrt(2.0);
    TEST_ASSERT(complex_approx_equal(amp[0], inv_sqrt2), "H|0⟩: amp[000] = 1/√2");
    TEST_ASSERT(complex_approx_equal(amp[1], inv_sqrt2), "H|0⟩: amp[001] = 1/√2");
    TEST_ASSERT(complex_approx_equal(amp[2], 0.0), "H|0⟩: amp[010] = 0");
    TEST_ASSERT(complex_approx_equal(amp[3], 0.0), "H|0⟩: amp[011] = 0");
    TEST_ASSERT(is_normalized(amp, num_qubits), "H|0⟩: normalized");

    free(amp);
    TEST_PASS("Hadamard on |0⟩ state");
    return 1;
}

static int test_hadamard_on_one(void) {
    // H|1⟩ = (|0⟩ - |1⟩) / √2
    int num_qubits = 2;
    uint64_t dim = 1ULL << num_qubits;
    complex_t* amp = malloc(dim * sizeof(complex_t));

    init_basis_state(amp, num_qubits, 1);  // |01⟩
    stride_hadamard(amp, num_qubits, 0);   // H on qubit 0

    double inv_sqrt2 = 1.0 / sqrt(2.0);
    TEST_ASSERT(complex_approx_equal(amp[0], inv_sqrt2), "H|1⟩: amp[00] = 1/√2");
    TEST_ASSERT(complex_approx_equal(amp[1], -inv_sqrt2), "H|1⟩: amp[01] = -1/√2");
    TEST_ASSERT(is_normalized(amp, num_qubits), "H|1⟩: normalized");

    free(amp);
    TEST_PASS("Hadamard on |1⟩ state");
    return 1;
}

static int test_hadamard_involution(void) {
    // H² = I (Hadamard is its own inverse)
    int num_qubits = 3;
    uint64_t dim = 1ULL << num_qubits;
    complex_t* amp = malloc(dim * sizeof(complex_t));

    init_zero_state(amp, num_qubits);
    stride_hadamard(amp, num_qubits, 1);  // H on qubit 1
    stride_hadamard(amp, num_qubits, 1);  // H again

    // Should be back to |000⟩
    TEST_ASSERT(complex_approx_equal(amp[0], 1.0), "H²|0⟩ = |0⟩: amp[0] = 1");
    for (uint64_t i = 1; i < dim; i++) {
        TEST_ASSERT(complex_approx_equal(amp[i], 0.0), "H²|0⟩ = |0⟩: other amps = 0");
    }

    free(amp);
    TEST_PASS("Hadamard involution (H² = I)");
    return 1;
}

static int test_pauli_x_flip(void) {
    // X|0⟩ = |1⟩, X|1⟩ = |0⟩
    int num_qubits = 3;
    uint64_t dim = 1ULL << num_qubits;
    complex_t* amp = malloc(dim * sizeof(complex_t));

    init_zero_state(amp, num_qubits);  // |000⟩
    stride_pauli_x(amp, num_qubits, 1);  // X on qubit 1 → |010⟩

    TEST_ASSERT(complex_approx_equal(amp[0], 0.0), "X|0⟩: amp[000] = 0");
    TEST_ASSERT(complex_approx_equal(amp[2], 1.0), "X|0⟩: amp[010] = 1");
    TEST_ASSERT(is_normalized(amp, num_qubits), "X|0⟩: normalized");

    free(amp);
    TEST_PASS("Pauli X bit flip");
    return 1;
}

static int test_pauli_x_involution(void) {
    // X² = I
    int num_qubits = 2;
    uint64_t dim = 1ULL << num_qubits;
    complex_t* amp = malloc(dim * sizeof(complex_t));

    init_zero_state(amp, num_qubits);
    stride_pauli_x(amp, num_qubits, 0);
    stride_pauli_x(amp, num_qubits, 0);

    TEST_ASSERT(complex_approx_equal(amp[0], 1.0), "X² = I");

    free(amp);
    TEST_PASS("Pauli X involution (X² = I)");
    return 1;
}

static int test_pauli_y(void) {
    // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
    int num_qubits = 2;
    uint64_t dim = 1ULL << num_qubits;
    complex_t* amp = malloc(dim * sizeof(complex_t));

    init_zero_state(amp, num_qubits);  // |00⟩
    stride_pauli_y(amp, num_qubits, 0);  // Y on qubit 0 → i|01⟩

    TEST_ASSERT(complex_approx_equal(amp[0], 0.0), "Y|0⟩: amp[00] = 0");
    TEST_ASSERT(complex_approx_equal(amp[1], I), "Y|0⟩: amp[01] = i");
    TEST_ASSERT(is_normalized(amp, num_qubits), "Y|0⟩: normalized");

    free(amp);
    TEST_PASS("Pauli Y gate");
    return 1;
}

static int test_pauli_z(void) {
    // Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
    int num_qubits = 2;
    uint64_t dim = 1ULL << num_qubits;
    complex_t* amp = malloc(dim * sizeof(complex_t));

    // Test on |+⟩ state (superposition)
    init_zero_state(amp, num_qubits);
    stride_hadamard(amp, num_qubits, 0);  // H|0⟩ = |+⟩
    stride_pauli_z(amp, num_qubits, 0);   // Z|+⟩ = |-⟩

    double inv_sqrt2 = 1.0 / sqrt(2.0);
    TEST_ASSERT(complex_approx_equal(amp[0], inv_sqrt2), "Z|+⟩ = |-⟩: amp[0]");
    TEST_ASSERT(complex_approx_equal(amp[1], -inv_sqrt2), "Z|+⟩ = |-⟩: amp[1]");

    free(amp);
    TEST_PASS("Pauli Z gate");
    return 1;
}

static int test_phase_gate(void) {
    // P(π/2)|1⟩ = i|1⟩ (S gate)
    int num_qubits = 2;
    uint64_t dim = 1ULL << num_qubits;
    complex_t* amp = malloc(dim * sizeof(complex_t));

    init_basis_state(amp, num_qubits, 1);  // |01⟩
    stride_phase(amp, num_qubits, 0, M_PI / 2.0);  // S on qubit 0

    TEST_ASSERT(complex_approx_equal(amp[1], I), "S|1⟩ = i|1⟩");

    free(amp);
    TEST_PASS("Phase gate (S = P(π/2))");
    return 1;
}

static int test_rx_rotation(void) {
    // RX(π)|0⟩ = -i|1⟩
    int num_qubits = 1;
    complex_t amp[2];

    init_zero_state(amp, num_qubits);
    stride_rx(amp, num_qubits, 0, M_PI);

    TEST_ASSERT(complex_approx_equal(amp[0], 0.0), "RX(π)|0⟩: amp[0] = 0");
    TEST_ASSERT(complex_approx_equal(amp[1], -I), "RX(π)|0⟩: amp[1] = -i");

    TEST_PASS("RX rotation gate");
    return 1;
}

static int test_ry_rotation(void) {
    // RY(π)|0⟩ = |1⟩
    int num_qubits = 1;
    complex_t amp[2];

    init_zero_state(amp, num_qubits);
    stride_ry(amp, num_qubits, 0, M_PI);

    TEST_ASSERT(complex_approx_equal(amp[0], 0.0), "RY(π)|0⟩: amp[0] = 0");
    TEST_ASSERT(complex_approx_equal(amp[1], 1.0), "RY(π)|0⟩: amp[1] = 1");

    TEST_PASS("RY rotation gate");
    return 1;
}

static int test_rz_rotation(void) {
    // RZ(π)|+⟩ = e^(-iπ/2)|+⟩ (global phase on |0⟩, relative on |1⟩)
    int num_qubits = 1;
    complex_t amp[2];

    init_zero_state(amp, num_qubits);
    stride_hadamard(amp, num_qubits, 0);  // |+⟩
    stride_rz(amp, num_qubits, 0, M_PI);  // RZ(π)

    // RZ(π) = [[e^(-iπ/2), 0], [0, e^(iπ/2)]] = [[-i, 0], [0, i]]
    double inv_sqrt2 = 1.0 / sqrt(2.0);
    TEST_ASSERT(complex_approx_equal(amp[0], -I * inv_sqrt2), "RZ(π)|+⟩: amp[0]");
    TEST_ASSERT(complex_approx_equal(amp[1], I * inv_sqrt2), "RZ(π)|+⟩: amp[1]");

    TEST_PASS("RZ rotation gate");
    return 1;
}

// ============================================================================
// TWO-QUBIT GATE TESTS
// ============================================================================

static int test_cnot_control_zero(void) {
    // CNOT|00⟩ = |00⟩ (control is 0, no action)
    int num_qubits = 2;
    complex_t amp[4];

    init_zero_state(amp, num_qubits);
    stride_cnot(amp, num_qubits, 1, 0);  // control=1, target=0

    TEST_ASSERT(complex_approx_equal(amp[0], 1.0), "CNOT|00⟩ = |00⟩");
    TEST_ASSERT(complex_approx_equal(amp[1], 0.0), "CNOT|00⟩: amp[01] = 0");
    TEST_ASSERT(complex_approx_equal(amp[2], 0.0), "CNOT|00⟩: amp[10] = 0");
    TEST_ASSERT(complex_approx_equal(amp[3], 0.0), "CNOT|00⟩: amp[11] = 0");

    TEST_PASS("CNOT with control = 0");
    return 1;
}

static int test_cnot_flip(void) {
    // CNOT|10⟩ = |11⟩ (control is 1, flip target)
    int num_qubits = 2;
    complex_t amp[4];

    init_basis_state(amp, num_qubits, 2);  // |10⟩
    stride_cnot(amp, num_qubits, 1, 0);    // control=1, target=0

    TEST_ASSERT(complex_approx_equal(amp[0], 0.0), "CNOT|10⟩: amp[00] = 0");
    TEST_ASSERT(complex_approx_equal(amp[1], 0.0), "CNOT|10⟩: amp[01] = 0");
    TEST_ASSERT(complex_approx_equal(amp[2], 0.0), "CNOT|10⟩: amp[10] = 0");
    TEST_ASSERT(complex_approx_equal(amp[3], 1.0), "CNOT|10⟩ = |11⟩");

    TEST_PASS("CNOT flip when control = 1");
    return 1;
}

static int test_cnot_reverse(void) {
    // CNOT with swapped control/target
    // CNOT|01⟩ (control=0, target=1) = |11⟩
    int num_qubits = 2;
    complex_t amp[4];

    init_basis_state(amp, num_qubits, 1);  // |01⟩
    stride_cnot(amp, num_qubits, 0, 1);    // control=0, target=1

    // Control = qubit 0 = 1, so flip qubit 1: |01⟩ → |11⟩
    TEST_ASSERT(complex_approx_equal(amp[3], 1.0), "CNOT(0,1)|01⟩ = |11⟩");

    TEST_PASS("CNOT with reversed control/target");
    return 1;
}

static int test_cnot_involution(void) {
    // CNOT² = I
    int num_qubits = 2;
    complex_t amp[4];

    init_basis_state(amp, num_qubits, 2);  // |10⟩
    stride_cnot(amp, num_qubits, 1, 0);    // CNOT → |11⟩
    stride_cnot(amp, num_qubits, 1, 0);    // CNOT → |10⟩

    TEST_ASSERT(complex_approx_equal(amp[2], 1.0), "CNOT² = I");

    TEST_PASS("CNOT involution");
    return 1;
}

static int test_cnot_bell_state(void) {
    // Create Bell state: (H ⊗ I) then CNOT
    // |00⟩ → |+0⟩ → (|00⟩ + |11⟩)/√2
    int num_qubits = 2;
    complex_t amp[4];

    init_zero_state(amp, num_qubits);
    stride_hadamard(amp, num_qubits, 1);  // H on qubit 1 (MSB)
    stride_cnot(amp, num_qubits, 1, 0);   // CNOT (control=1, target=0)

    double inv_sqrt2 = 1.0 / sqrt(2.0);
    TEST_ASSERT(complex_approx_equal(amp[0], inv_sqrt2), "Bell|Φ+⟩: amp[00]");
    TEST_ASSERT(complex_approx_equal(amp[1], 0.0), "Bell|Φ+⟩: amp[01] = 0");
    TEST_ASSERT(complex_approx_equal(amp[2], 0.0), "Bell|Φ+⟩: amp[10] = 0");
    TEST_ASSERT(complex_approx_equal(amp[3], inv_sqrt2), "Bell|Φ+⟩: amp[11]");
    TEST_ASSERT(is_normalized(amp, num_qubits), "Bell state normalized");

    TEST_PASS("CNOT creates Bell state");
    return 1;
}

static int test_cz_gate(void) {
    // CZ|11⟩ = -|11⟩
    int num_qubits = 2;
    complex_t amp[4];

    init_basis_state(amp, num_qubits, 3);  // |11⟩
    stride_cz(amp, num_qubits, 1, 0);

    TEST_ASSERT(complex_approx_equal(amp[3], -1.0), "CZ|11⟩ = -|11⟩");
    TEST_ASSERT(is_normalized(amp, num_qubits), "CZ normalized");

    TEST_PASS("CZ gate");
    return 1;
}

static int test_cz_symmetric(void) {
    // CZ is symmetric: CZ(a,b) = CZ(b,a)
    int num_qubits = 2;
    complex_t amp1[4], amp2[4];

    init_basis_state(amp1, num_qubits, 3);
    init_basis_state(amp2, num_qubits, 3);

    stride_cz(amp1, num_qubits, 0, 1);
    stride_cz(amp2, num_qubits, 1, 0);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT(complex_approx_equal(amp1[i], amp2[i]), "CZ symmetric");
    }

    TEST_PASS("CZ symmetry");
    return 1;
}

static int test_swap_gate(void) {
    // SWAP|01⟩ = |10⟩
    int num_qubits = 2;
    complex_t amp[4];

    init_basis_state(amp, num_qubits, 1);  // |01⟩
    stride_swap(amp, num_qubits, 0, 1);

    TEST_ASSERT(complex_approx_equal(amp[1], 0.0), "SWAP|01⟩: amp[01] = 0");
    TEST_ASSERT(complex_approx_equal(amp[2], 1.0), "SWAP|01⟩ = |10⟩");

    TEST_PASS("SWAP gate");
    return 1;
}

static int test_swap_involution(void) {
    // SWAP² = I
    int num_qubits = 2;
    complex_t amp[4];

    init_basis_state(amp, num_qubits, 1);  // |01⟩
    stride_swap(amp, num_qubits, 0, 1);    // → |10⟩
    stride_swap(amp, num_qubits, 0, 1);    // → |01⟩

    TEST_ASSERT(complex_approx_equal(amp[1], 1.0), "SWAP² = I");

    TEST_PASS("SWAP involution");
    return 1;
}

static int test_iswap_gate(void) {
    // iSWAP|01⟩ = i|10⟩
    int num_qubits = 2;
    complex_t amp[4];

    init_basis_state(amp, num_qubits, 1);  // |01⟩
    stride_iswap(amp, num_qubits, 0, 1);

    TEST_ASSERT(complex_approx_equal(amp[1], 0.0), "iSWAP|01⟩: amp[01] = 0");
    TEST_ASSERT(complex_approx_equal(amp[2], I), "iSWAP|01⟩ = i|10⟩");

    TEST_PASS("iSWAP gate");
    return 1;
}

static int test_controlled_phase(void) {
    // CP(π)|11⟩ = -|11⟩ (same as CZ)
    int num_qubits = 2;
    complex_t amp[4];

    init_basis_state(amp, num_qubits, 3);
    stride_cphase(amp, num_qubits, 0, 1, M_PI);

    TEST_ASSERT(complex_approx_equal(amp[3], -1.0), "CP(π)|11⟩ = -|11⟩");

    TEST_PASS("Controlled phase gate");
    return 1;
}

// ============================================================================
// THREE-QUBIT GATE TESTS
// ============================================================================

static int test_toffoli_basic(void) {
    // Toffoli|110⟩ = |111⟩
    int num_qubits = 3;
    uint64_t dim = 1ULL << num_qubits;
    complex_t* amp = malloc(dim * sizeof(complex_t));

    init_basis_state(amp, num_qubits, 6);  // |110⟩
    stride_toffoli(amp, num_qubits, 2, 1, 0);  // controls=2,1 target=0

    TEST_ASSERT(complex_approx_equal(amp[6], 0.0), "Toffoli|110⟩: amp[110] = 0");
    TEST_ASSERT(complex_approx_equal(amp[7], 1.0), "Toffoli|110⟩ = |111⟩");

    free(amp);
    TEST_PASS("Toffoli gate basic");
    return 1;
}

static int test_toffoli_no_action(void) {
    // Toffoli|100⟩ = |100⟩ (only one control is 1)
    int num_qubits = 3;
    uint64_t dim = 1ULL << num_qubits;
    complex_t* amp = malloc(dim * sizeof(complex_t));

    init_basis_state(amp, num_qubits, 4);  // |100⟩
    stride_toffoli(amp, num_qubits, 2, 1, 0);

    TEST_ASSERT(complex_approx_equal(amp[4], 1.0), "Toffoli|100⟩ = |100⟩");

    free(amp);
    TEST_PASS("Toffoli no action with one control");
    return 1;
}

static int test_toffoli_involution(void) {
    // Toffoli² = I
    int num_qubits = 3;
    uint64_t dim = 1ULL << num_qubits;
    complex_t* amp = malloc(dim * sizeof(complex_t));

    init_basis_state(amp, num_qubits, 6);  // |110⟩
    stride_toffoli(amp, num_qubits, 2, 1, 0);  // → |111⟩
    stride_toffoli(amp, num_qubits, 2, 1, 0);  // → |110⟩

    TEST_ASSERT(complex_approx_equal(amp[6], 1.0), "Toffoli² = I");

    free(amp);
    TEST_PASS("Toffoli involution");
    return 1;
}

static int test_fredkin_basic(void) {
    // Fredkin|101⟩ = |110⟩ (swap targets when control=1)
    int num_qubits = 3;
    uint64_t dim = 1ULL << num_qubits;
    complex_t* amp = malloc(dim * sizeof(complex_t));

    init_basis_state(amp, num_qubits, 5);  // |101⟩
    stride_fredkin(amp, num_qubits, 2, 1, 0);  // control=2, swap 1↔0

    TEST_ASSERT(complex_approx_equal(amp[5], 0.0), "Fredkin|101⟩: amp[101] = 0");
    TEST_ASSERT(complex_approx_equal(amp[6], 1.0), "Fredkin|101⟩ = |110⟩");

    free(amp);
    TEST_PASS("Fredkin gate basic");
    return 1;
}

static int test_fredkin_no_action(void) {
    // Fredkin|010⟩ = |010⟩ (control=0, no swap)
    int num_qubits = 3;
    uint64_t dim = 1ULL << num_qubits;
    complex_t* amp = malloc(dim * sizeof(complex_t));

    init_basis_state(amp, num_qubits, 2);  // |010⟩
    stride_fredkin(amp, num_qubits, 2, 1, 0);

    TEST_ASSERT(complex_approx_equal(amp[2], 1.0), "Fredkin|010⟩ = |010⟩");

    free(amp);
    TEST_PASS("Fredkin no action");
    return 1;
}

// ============================================================================
// BATCH OPERATION TESTS
// ============================================================================

static int test_hadamard_all(void) {
    // H⊗n|0⟩ = |+⟩⊗n = uniform superposition
    int num_qubits = 3;
    uint64_t dim = 1ULL << num_qubits;
    complex_t* amp = malloc(dim * sizeof(complex_t));

    init_zero_state(amp, num_qubits);
    stride_hadamard_all(amp, num_qubits);

    // All amplitudes should be 1/√8 = 1/(2√2)
    double expected = 1.0 / sqrt((double)dim);
    for (uint64_t i = 0; i < dim; i++) {
        TEST_ASSERT(complex_approx_equal(amp[i], expected), "H⊗n uniform");
    }
    TEST_ASSERT(is_normalized(amp, num_qubits), "H⊗n normalized");

    free(amp);
    TEST_PASS("Hadamard all qubits");
    return 1;
}

static int test_batch_single_gate(void) {
    // Apply X to multiple qubits
    int num_qubits = 3;
    uint64_t dim = 1ULL << num_qubits;
    complex_t* amp = malloc(dim * sizeof(complex_t));

    init_zero_state(amp, num_qubits);  // |000⟩

    // X gate matrix
    complex_t X[4] = {0, 1, 1, 0};
    int targets[] = {0, 2};
    stride_batch_single_gate(amp, num_qubits, targets, 2, X);

    // |000⟩ → |001⟩ → |101⟩
    TEST_ASSERT(complex_approx_equal(amp[5], 1.0), "Batch X on qubits 0,2");

    free(amp);
    TEST_PASS("Batch single gate");
    return 1;
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

static int test_adjacent_qubits(void) {
    // Test CNOT on adjacent qubits
    int num_qubits = 4;
    uint64_t dim = 1ULL << num_qubits;
    complex_t* amp = malloc(dim * sizeof(complex_t));

    init_basis_state(amp, num_qubits, 4);  // |0100⟩
    stride_cnot(amp, num_qubits, 2, 1);    // adjacent control=2, target=1

    // Control bit 2 is 1, so flip bit 1: |0100⟩ → |0110⟩ = index 6
    TEST_ASSERT(complex_approx_equal(amp[6], 1.0), "CNOT adjacent qubits");

    free(amp);
    TEST_PASS("Adjacent qubit operations");
    return 1;
}

static int test_boundary_qubits(void) {
    // Test gate on highest and lowest qubits
    int num_qubits = 5;
    uint64_t dim = 1ULL << num_qubits;
    complex_t* amp = malloc(dim * sizeof(complex_t));

    init_basis_state(amp, num_qubits, 16);  // |10000⟩
    stride_cnot(amp, num_qubits, 4, 0);     // control=MSB, target=LSB

    // |10000⟩ → |10001⟩ = index 17
    TEST_ASSERT(complex_approx_equal(amp[17], 1.0), "CNOT boundary qubits");

    free(amp);
    TEST_PASS("Boundary qubit operations");
    return 1;
}

static int test_large_state(void) {
    // Test on larger state space
    int num_qubits = 10;  // 1024 amplitudes
    uint64_t dim = 1ULL << num_qubits;
    complex_t* amp = malloc(dim * sizeof(complex_t));

    init_zero_state(amp, num_qubits);
    stride_hadamard(amp, num_qubits, 5);  // H on middle qubit
    stride_cnot(amp, num_qubits, 5, 3);   // Entangle

    TEST_ASSERT(is_normalized(amp, num_qubits), "Large state normalized");

    free(amp);
    TEST_PASS("Large state operations");
    return 1;
}

// ============================================================================
// GATE IDENTITY TESTS
// ============================================================================

static int test_hzh_equals_x(void) {
    // HZH = X
    int num_qubits = 1;
    complex_t amp1[2], amp2[2];

    // Method 1: HZH
    init_zero_state(amp1, num_qubits);
    stride_hadamard(amp1, num_qubits, 0);
    stride_pauli_z(amp1, num_qubits, 0);
    stride_hadamard(amp1, num_qubits, 0);

    // Method 2: X
    init_zero_state(amp2, num_qubits);
    stride_pauli_x(amp2, num_qubits, 0);

    TEST_ASSERT(complex_approx_equal(amp1[0], amp2[0]), "HZH = X: amp[0]");
    TEST_ASSERT(complex_approx_equal(amp1[1], amp2[1]), "HZH = X: amp[1]");

    TEST_PASS("Gate identity HZH = X");
    return 1;
}

static int test_cnot_decomposition(void) {
    // CNOT = (I⊗H) CZ (I⊗H)
    int num_qubits = 2;
    complex_t amp1[4], amp2[4];

    // Test on |10⟩
    // Method 1: CNOT
    init_basis_state(amp1, num_qubits, 2);
    stride_cnot(amp1, num_qubits, 1, 0);

    // Method 2: (I⊗H) CZ (I⊗H)
    init_basis_state(amp2, num_qubits, 2);
    stride_hadamard(amp2, num_qubits, 0);  // H on target
    stride_cz(amp2, num_qubits, 1, 0);     // CZ
    stride_hadamard(amp2, num_qubits, 0);  // H on target

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT(complex_approx_equal(amp1[i], amp2[i]), "CNOT = HCZ H");
    }

    TEST_PASS("CNOT decomposition");
    return 1;
}

static int test_swap_from_cnot(void) {
    // SWAP = CNOT(a,b) CNOT(b,a) CNOT(a,b)
    int num_qubits = 2;
    complex_t amp1[4], amp2[4];

    // Test on |01⟩
    // Method 1: SWAP
    init_basis_state(amp1, num_qubits, 1);
    stride_swap(amp1, num_qubits, 0, 1);

    // Method 2: Three CNOTs
    init_basis_state(amp2, num_qubits, 1);
    stride_cnot(amp2, num_qubits, 0, 1);
    stride_cnot(amp2, num_qubits, 1, 0);
    stride_cnot(amp2, num_qubits, 0, 1);

    for (int i = 0; i < 4; i++) {
        TEST_ASSERT(complex_approx_equal(amp1[i], amp2[i]), "SWAP = 3 CNOTs");
    }

    TEST_PASS("SWAP from CNOTs");
    return 1;
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main(void) {
    printf("==============================================\n");
    printf("Stride-Based Gate Unit Tests\n");
    printf("==============================================\n");

    // Single-qubit gate tests
    RUN_TEST(test_hadamard_on_zero);
    RUN_TEST(test_hadamard_on_one);
    RUN_TEST(test_hadamard_involution);
    RUN_TEST(test_pauli_x_flip);
    RUN_TEST(test_pauli_x_involution);
    RUN_TEST(test_pauli_y);
    RUN_TEST(test_pauli_z);
    RUN_TEST(test_phase_gate);
    RUN_TEST(test_rx_rotation);
    RUN_TEST(test_ry_rotation);
    RUN_TEST(test_rz_rotation);

    // Two-qubit gate tests
    RUN_TEST(test_cnot_control_zero);
    RUN_TEST(test_cnot_flip);
    RUN_TEST(test_cnot_reverse);
    RUN_TEST(test_cnot_involution);
    RUN_TEST(test_cnot_bell_state);
    RUN_TEST(test_cz_gate);
    RUN_TEST(test_cz_symmetric);
    RUN_TEST(test_swap_gate);
    RUN_TEST(test_swap_involution);
    RUN_TEST(test_iswap_gate);
    RUN_TEST(test_controlled_phase);

    // Three-qubit gate tests
    RUN_TEST(test_toffoli_basic);
    RUN_TEST(test_toffoli_no_action);
    RUN_TEST(test_toffoli_involution);
    RUN_TEST(test_fredkin_basic);
    RUN_TEST(test_fredkin_no_action);

    // Batch operation tests
    RUN_TEST(test_hadamard_all);
    RUN_TEST(test_batch_single_gate);

    // Edge case tests
    RUN_TEST(test_adjacent_qubits);
    RUN_TEST(test_boundary_qubits);
    RUN_TEST(test_large_state);

    // Gate identity tests
    RUN_TEST(test_hzh_equals_x);
    RUN_TEST(test_cnot_decomposition);
    RUN_TEST(test_swap_from_cnot);

    // Summary
    printf("\n==============================================\n");
    printf("Test Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("==============================================\n");

    return tests_failed > 0 ? 1 : 0;
}
