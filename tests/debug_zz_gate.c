/*
 * Minimal test case to debug ZZ gate norm collapse issue
 *
 * This tests whether tn_apply_gate_2q() can correctly handle
 * the non-unitary diagonal imaginary time ZZ gate.
 */

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include "../src/algorithms/tensor_network/tn_state.h"
#include "../src/algorithms/tensor_network/tn_gates.h"

int main(void) {
    printf("=== Testing ZZ Gate Application ===\n\n");

    // Create a simple 2-qubit MPS
    tn_state_config_t config = tn_state_config_create(128, 1e-10);
    config.auto_canonicalize = true;  // ENABLE auto-canonicalization to match quantum_spin_chain.c

    tn_mps_state_t *state = tn_mps_create_zero(2, &config);
    if (!state) {
        fprintf(stderr, "Failed to create state\n");
        return 1;
    }

    printf("Initial state |00⟩:\n");
    double norm = tn_mps_norm(state);
    printf("  Norm: %.15e\n\n", norm);

    // Apply small Ry rotation to both qubits to create superposition
    printf("Applying Ry(0.1) to both qubits...\n");
    tn_apply_ry(state, 0, 0.1);
    tn_apply_ry(state, 1, 0.1);
    norm = tn_mps_norm(state);
    printf("  Norm after Ry gates: %.15e\n\n", norm);

    // Normalize
    tn_state_error_t ret = tn_mps_normalize(state);
    if (ret != TN_STATE_SUCCESS) {
        fprintf(stderr, "Normalization failed: %d\n", ret);
        tn_mps_free(state);
        return 1;
    }
    norm = tn_mps_norm(state);
    printf("  Norm after normalization: %.15e\n\n", norm);

    // Create imaginary time ZZ gate: e^{τJ ZZ}
    double tau_J = 0.1;
    double ep = exp(tau_J);
    double em = exp(-tau_J);

    printf("Creating imaginary time ZZ gate with τJ = %.6f:\n", tau_J);
    printf("  e^{+τJ} = %.15f\n", ep);
    printf("  e^{-τJ} = %.15f\n\n", em);

    tn_gate_2q_t zz_gate;

    // Initialize all elements to zero
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            zz_gate.elements[i][j] = 0.0 + 0.0*I;
        }
    }

    // Diagonal elements: diag(e^{τJ}, e^{-τJ}, e^{-τJ}, e^{τJ})
    // Basis ordering: |00⟩, |01⟩, |10⟩, |11⟩
    // ZZ eigenvalues: +1,    -1,    -1,    +1
    zz_gate.elements[0][0] = ep + 0.0*I;  // |00⟩
    zz_gate.elements[1][1] = em + 0.0*I;  // |01⟩
    zz_gate.elements[2][2] = em + 0.0*I;  // |10⟩
    zz_gate.elements[3][3] = ep + 0.0*I;  // |11⟩

    printf("ZZ gate matrix (diagonal):\n");
    for (int i = 0; i < 4; i++) {
        printf("  [");
        for (int j = 0; j < 4; j++) {
            printf(" %.6f%+.6fi", creal(zz_gate.elements[i][j]), cimag(zz_gate.elements[i][j]));
        }
        printf(" ]\n");
    }
    printf("\n");

    // Apply the ZZ gate
    printf("Applying ZZ gate to qubits 0-1...\n");
    norm = tn_mps_norm(state);
    printf("  Norm BEFORE ZZ gate: %.15e\n", norm);

    double trunc_err = 0.0;
    tn_gate_error_t gate_ret = tn_apply_gate_2q(state, 0, 1, &zz_gate, &trunc_err);

    norm = tn_mps_norm(state);
    printf("  Norm AFTER ZZ gate:  %.15e\n", norm);
    printf("  Return code: %d\n", gate_ret);
    printf("  Truncation error: %.15e\n\n", trunc_err);

    if (gate_ret != TN_GATE_SUCCESS) {
        fprintf(stderr, "ERROR: ZZ gate application failed with code %d\n", gate_ret);
        tn_mps_free(state);
        return 1;
    }

    if (norm < 1e-15) {
        fprintf(stderr, "ERROR: Norm collapsed to zero!\n");
        fprintf(stderr, "This confirms the ZZ gate issue.\n\n");

        // Try alternative: apply ZZ as a unitary rotation instead
        printf("=== Testing Alternative: Rzz Rotation ===\n\n");

        tn_mps_state_t *state2 = tn_mps_create_zero(2, &config);
        tn_apply_ry(state2, 0, 0.1);
        tn_apply_ry(state2, 1, 0.1);
        tn_mps_normalize(state2);

        printf("Applying Rzz(%.6f) instead...\n", 2.0 * tau_J);
        norm = tn_mps_norm(state2);
        printf("  Norm BEFORE Rzz: %.15e\n", norm);

        gate_ret = tn_apply_rzz(state2, 0, 1, 2.0 * tau_J);

        norm = tn_mps_norm(state2);
        printf("  Norm AFTER Rzz:  %.15e\n", norm);
        printf("  Return code: %d\n\n", gate_ret);

        tn_mps_free(state2);
        tn_mps_free(state);
        return 1;
    }

    printf("SUCCESS: ZZ gate applied without norm collapse!\n");

    tn_mps_free(state);
    return 0;
}
