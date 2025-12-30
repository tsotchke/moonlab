/*
 * Test the EXACT sequence from quantum_spin_chain.c that fails
 */

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include "../src/algorithms/tensor_network/tn_state.h"
#include "../src/algorithms/tensor_network/tn_gates.h"

static tn_gate_1q_t create_imag_time_x_gate(double tau_h) {
    tn_gate_1q_t gate;
    double c = cosh(tau_h);
    double s = sinh(tau_h);
    gate.elements[0][0] = c + 0.0*I;
    gate.elements[0][1] = s + 0.0*I;
    gate.elements[1][0] = s + 0.0*I;
    gate.elements[1][1] = c + 0.0*I;
    return gate;
}

static tn_gate_2q_t create_imag_time_zz_gate(double tau_J) {
    tn_gate_2q_t gate;
    double ep = exp(tau_J);
    double em = exp(-tau_J);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            gate.elements[i][j] = 0.0 + 0.0*I;
        }
    }

    gate.elements[0][0] = ep + 0.0*I;
    gate.elements[1][1] = em + 0.0*I;
    gate.elements[2][2] = em + 0.0*I;
    gate.elements[3][3] = ep + 0.0*I;

    return gate;
}

int main(void) {
    printf("=== Testing EXACT quantum_spin_chain.c Sequence ===\n\n");

    // Use more qubits to match the failure case
    const uint32_t n = 10;  // Start with 10 qubits
    const double J = 1.0;
    const double h = 0.5;
    const double dt = 0.01;

    tn_state_config_t config = tn_state_config_create(128, 1e-10);
    config.auto_canonicalize = true;

    tn_mps_state_t *state = tn_mps_create_zero(n, &config);
    if (!state) {
        fprintf(stderr, "Failed to create state\n");
        return 1;
    }

    // Initialize with Ry rotations like quantum_spin_chain.c
    printf("Initializing %u qubits with Ry(0.1)...\n", n);
    for (uint32_t i = 0; i < n; i++) {
        tn_apply_ry(state, i, 0.1);
    }
    printf("  Initial norm: %.15e\n\n", tn_mps_norm(state));

    printf("===== STEP 1: First X layer =====\n");
    tn_gate_1q_t x_gate = create_imag_time_x_gate(h * dt / 2.0);
    printf("  tau_h = %f, cosh = %f, sinh = %f\n", h * dt / 2.0, cosh(h * dt / 2.0), sinh(h * dt / 2.0));
    printf("  Pre-X norm: %.15e\n", tn_mps_norm(state));

    for (uint32_t i = 0; i < n; i++) {
        tn_gate_error_t ret = tn_apply_gate_1q(state, i, &x_gate);
        if (ret != TN_GATE_SUCCESS) {
            fprintf(stderr, "ERROR: X gate failed at qubit %u\n", i);
            tn_mps_free(state);
            return 1;
        }
    }

    printf("  Post-X norm (before normalize): %.15e\n", tn_mps_norm(state));

    tn_state_error_t ret = tn_mps_normalize(state);
    if (ret != TN_STATE_SUCCESS) {
        fprintf(stderr, "ERROR: X normalization failed\n");
        tn_mps_free(state);
        return 1;
    }

    printf("  Post-X norm (after normalize): %.15e\n\n", tn_mps_norm(state));

    printf("===== STEP 2: ZZ layer =====\n");
    tn_gate_2q_t zz_gate = create_imag_time_zz_gate(J * dt);
    printf("  tau_J = %f, e^+ = %f, e^- = %f\n", J * dt, exp(J * dt), exp(-J * dt));
    printf("  Pre-ZZ norm: %.15e\n", tn_mps_norm(state));

    // Even bonds - NORMALIZE AFTER EACH BOND
    printf("  Applying even bonds (with normalization after each)...\n");
    for (uint32_t i = 0; i + 1 < n; i += 2) {
        double norm_pre_gate = tn_mps_norm(state);
        printf("    Bond (%u,%u): norm BEFORE gate application = %.15e\n", i, i+1, norm_pre_gate);

        // Print MPS structure before gate
        printf("      MPS structure before gate:\n");
        printf("        Canonical form: %d, center: %d\n", state->canonical, state->canonical_center);
        printf("        Bond dimensions: [");
        for (uint32_t b = 0; b < n - 1; b++) {
            printf("%u", tn_mps_bond_dim(state, b));
            if (b < n - 2) printf(", ");
        }
        printf("]\n");

        // Print individual tensor norms
        printf("        Tensor norms: [");
        for (uint32_t t = 0; t < n; t++) {
            const tensor_t *tensor = tn_mps_get_tensor(state, t);
            double tensor_norm = 0.0;
            for (uint64_t e = 0; e < tensor->total_size; e++) {
                tensor_norm += creal(tensor->data[e] * conj(tensor->data[e]));
            }
            tensor_norm = sqrt(tensor_norm);
            printf("%.6f", tensor_norm);
            if (t < n - 1) printf(", ");
        }
        printf("]\n");

        double trunc_err = 0.0;
        tn_gate_error_t gate_ret = tn_apply_gate_2q(state, i, i + 1, &zz_gate, &trunc_err);
        if (gate_ret != TN_GATE_SUCCESS) {
            fprintf(stderr, "ERROR: ZZ gate failed at bond (%u,%u), error code = %d\n", i, i+1, gate_ret);
            tn_mps_free(state);
            return 1;
        }

        double norm_post_gate = tn_mps_norm(state);
        printf("    Bond (%u,%u): norm AFTER gate application = %.15e\n", i, i+1, norm_post_gate);

        // Print MPS structure after gate
        printf("      MPS structure after gate:\n");
        printf("        Canonical form: %d, center: %d\n", state->canonical, state->canonical_center);
        printf("        Bond dimensions: [");
        for (uint32_t b = 0; b < n - 1; b++) {
            printf("%u", tn_mps_bond_dim(state, b));
            if (b < n - 2) printf(", ");
        }
        printf("]\n");

        // Print actual tensor shapes
        printf("        Tensor shapes: ");
        for (uint32_t t = 0; t < n; t++) {
            const tensor_t *tensor = tn_mps_get_tensor(state, t);
            printf("[%u,%u,%u]", tensor->dims[0], tensor->dims[1], tensor->dims[2]);
            if (t < n - 1) printf(" ");
        }
        printf("\n");

        // Print individual tensor norms after gate
        printf("        Tensor norms: [");
        for (uint32_t t = 0; t < n; t++) {
            const tensor_t *tensor = tn_mps_get_tensor(state, t);
            double tensor_norm = 0.0;
            for (uint64_t e = 0; e < tensor->total_size; e++) {
                tensor_norm += creal(tensor->data[e] * conj(tensor->data[e]));
            }
            tensor_norm = sqrt(tensor_norm);
            printf("%.6f", tensor_norm);
            if (t < n - 1) printf(", ");
        }
        printf("]\n");

        ret = tn_mps_normalize(state);
        if (ret != TN_STATE_SUCCESS) {
            fprintf(stderr, "ERROR: Normalization failed after bond (%u,%u), norm was %.15e\n",
                    i, i+1, norm_post_gate);
            tn_mps_free(state);
            return 1;
        }
        printf("    Bond (%u,%u): norm AFTER normalization = %.15e, trunc_err = %.15e\n\n",
               i, i+1, tn_mps_norm(state), trunc_err);
    }

    printf("  After even bonds norm: %.15e\n", tn_mps_norm(state));

    // Odd bonds - NORMALIZE AFTER EACH BOND
    printf("  Applying odd bonds (with normalization after each)...\n");
    for (uint32_t i = 1; i + 1 < n; i += 2) {
        double trunc_err = 0.0;
        tn_gate_error_t gate_ret = tn_apply_gate_2q(state, i, i + 1, &zz_gate, &trunc_err);
        if (gate_ret != TN_GATE_SUCCESS) {
            fprintf(stderr, "ERROR: ZZ gate failed at bond (%u,%u)\n", i, i+1);
            tn_mps_free(state);
            return 1;
        }
        double norm_before = tn_mps_norm(state);
        ret = tn_mps_normalize(state);
        if (ret != TN_STATE_SUCCESS) {
            fprintf(stderr, "ERROR: Normalization failed after bond (%u,%u)\n", i, i+1);
            tn_mps_free(state);
            return 1;
        }
        printf("    Bond (%u,%u): norm before = %.15e, after = %.15e, trunc_err = %.15e\n",
               i, i+1, norm_before, tn_mps_norm(state), trunc_err);
    }

    printf("  Post-ZZ norm (before normalize): %.15e\n", tn_mps_norm(state));

    ret = tn_mps_normalize(state);
    if (ret != TN_STATE_SUCCESS) {
        fprintf(stderr, "ERROR: ZZ normalization failed! Norm was too small.\n");
        tn_mps_free(state);
        return 1;
    }

    printf("  Post-ZZ norm (after normalize): %.15e\n\n", tn_mps_norm(state));

    printf("SUCCESS: Full Trotter step completed!\n");

    tn_mps_free(state);
    return 0;
}
