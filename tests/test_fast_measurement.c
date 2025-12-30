/**
 * @file test_fast_measurement.c
 * @brief Test fast measurement functions vs slow ones
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "src/algorithms/tensor_network/tensor.h"
#include "src/algorithms/tensor_network/tn_state.h"
#include "src/algorithms/tensor_network/tn_gates.h"
#include "src/algorithms/tensor_network/tn_measurement.h"

int main(void) {
    printf("Testing fast measurement functions\n");
    printf("==================================\n\n");

    const uint32_t n_qubits = 20;

    tn_state_config_t config = {
        .max_bond_dim = 32,
        .svd_cutoff = 1e-12,
        .max_truncation_error = 1e-10,
        .track_truncation = true,
        .auto_canonicalize = false,
        .target_form = TN_CANONICAL_NONE
    };

    // Create initial state with small tilt
    tn_mps_state_t *mps = tn_mps_create_zero(n_qubits, &config);
    if (!mps) {
        printf("ERROR: Failed to create MPS\n");
        return 1;
    }

    double tilt = 0.05;
    for (uint32_t i = 0; i < n_qubits; i++) {
        tensor_t *t = mps->tensors[i];
        uint32_t idx0[3] = {0, 0, 0};
        uint32_t idx1[3] = {0, 1, 0};
        tensor_set(t, idx0, cos(tilt));
        tensor_set(t, idx1, sin(tilt));
    }

    // Test 1: Initial product state (should be exact)
    printf("Test 1: Initial product state (n=%u)\n", n_qubits);

    // Slow measurement
    double slow_mag = 0.0;
    for (uint32_t i = 0; i < n_qubits; i++) {
        slow_mag += tn_expectation_z(mps, i);
    }
    slow_mag /= n_qubits;

    // Fast measurement (makes a copy since it modifies state)
    tn_mps_state_t *mps_copy = tn_mps_copy(mps);
    double fast_mag = tn_magnetization_fast(mps_copy);
    tn_mps_free(mps_copy);

    printf("  Slow magnetization: %.6f\n", slow_mag);
    printf("  Fast magnetization: %.6f\n", fast_mag);
    printf("  Difference: %.2e\n", fabs(slow_mag - fast_mag));
    printf("  Expected: ~%.6f\n", cos(2*tilt));

    if (fabs(slow_mag - fast_mag) < 1e-6) {
        printf("  PASS: Results match!\n");
    } else {
        printf("  FAIL: Results differ!\n");
    }

    // Test 2: After some evolution
    printf("\nTest 2: After 3 Trotter steps\n");

    // Create imaginary time gates
    double tau = 0.1;
    double J = 1.0, h = 0.5;

    tn_gate_2q_t zz_gate = {{{0}}};
    double ep = exp(tau * J);
    double em = exp(-tau * J);
    zz_gate.elements[0][0] = ep;
    zz_gate.elements[1][1] = em;
    zz_gate.elements[2][2] = em;
    zz_gate.elements[3][3] = ep;

    tn_gate_1q_t x_gate;
    double c = cosh(tau * h);
    double s = sinh(tau * h);
    x_gate.elements[0][0] = c;
    x_gate.elements[0][1] = s;
    x_gate.elements[1][0] = s;
    x_gate.elements[1][1] = c;

    // Apply 3 Trotter steps
    for (int step = 0; step < 3; step++) {
        for (uint32_t i = 0; i < n_qubits - 1; i += 2) {
            double trunc_err = 0.0;
            tn_apply_gate_2q(mps, i, i+1, &zz_gate, &trunc_err);
        }
        for (uint32_t i = 1; i < n_qubits - 1; i += 2) {
            double trunc_err = 0.0;
            tn_apply_gate_2q(mps, i, i+1, &zz_gate, &trunc_err);
        }
        for (uint32_t i = 0; i < n_qubits; i++) {
            tn_apply_gate_1q(mps, i, &x_gate);
        }
    }

    // Slow measurement
    slow_mag = 0.0;
    for (uint32_t i = 0; i < n_qubits; i++) {
        slow_mag += tn_expectation_z(mps, i);
    }
    slow_mag /= n_qubits;

    // Fast measurement
    mps_copy = tn_mps_copy(mps);
    fast_mag = tn_magnetization_fast(mps_copy);
    tn_mps_free(mps_copy);

    printf("  Slow magnetization: %.6f\n", slow_mag);
    printf("  Fast magnetization: %.6f\n", fast_mag);
    printf("  Difference: %.2e\n", fabs(slow_mag - fast_mag));

    if (fabs(slow_mag - fast_mag) < 1e-4) {
        printf("  PASS: Results match!\n");
    } else {
        printf("  FAIL: Results differ!\n");
    }

    // Test ZZ correlation
    printf("\nTest 3: ZZ correlation after evolution\n");

    double slow_zz = 0.0;
    for (uint32_t i = 0; i < n_qubits - 1; i++) {
        slow_zz += tn_expectation_zz(mps, i, i+1);
    }
    slow_zz /= (n_qubits - 1);

    mps_copy = tn_mps_copy(mps);
    double fast_zz = tn_zz_correlation_fast(mps_copy);
    tn_mps_free(mps_copy);

    printf("  Slow ZZ correlation: %.6f\n", slow_zz);
    printf("  Fast ZZ correlation: %.6f\n", fast_zz);
    printf("  Difference: %.2e\n", fabs(slow_zz - fast_zz));

    if (fabs(slow_zz - fast_zz) < 1e-4) {
        printf("  PASS: Results match!\n");
    } else {
        printf("  FAIL: Results differ!\n");
    }

    tn_mps_free(mps);
    printf("\nDone.\n");
    return 0;
}
