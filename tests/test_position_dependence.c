/**
 * @file test_position_dependence.c
 * @brief Test if measurements depend on position correctly
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "src/algorithms/tensor_network/tensor.h"
#include "src/algorithms/tensor_network/tn_state.h"
#include "src/algorithms/tensor_network/tn_gates.h"
#include "src/algorithms/tensor_network/tn_measurement.h"

static tn_gate_1q_t create_imag_time_x(double tau_h) {
    tn_gate_1q_t gate;
    double c = cosh(tau_h);
    double s = sinh(tau_h);
    gate.elements[0][0] = c;
    gate.elements[0][1] = s;
    gate.elements[1][0] = s;
    gate.elements[1][1] = c;
    return gate;
}

static tn_gate_2q_t create_imag_time_zz(double tau_J) {
    tn_gate_2q_t gate = {{{0}}};
    double ep = exp(tau_J);
    double em = exp(-tau_J);
    gate.elements[0][0] = ep;
    gate.elements[1][1] = em;
    gate.elements[2][2] = em;
    gate.elements[3][3] = ep;
    return gate;
}

int main(void) {
    printf("Testing position-dependence of measurements\n");
    printf("============================================\n\n");

    const uint32_t n_qubits = 20;
    const double J = 1.0;
    const double h = 0.5;
    const double tau = 0.1;

    tn_state_config_t config = {
        .max_bond_dim = 64,
        .svd_cutoff = 1e-12,
        .max_truncation_error = 1e-10,
        .track_truncation = true,
        .auto_canonicalize = false,
        .target_form = TN_CANONICAL_NONE
    };

    tn_mps_state_t *mps = tn_mps_create_zero(n_qubits, &config);
    if (!mps) {
        printf("ERROR: Failed to create MPS\n");
        return 1;
    }

    // Initialize with small tilt
    double tilt = 0.05;
    for (uint32_t i = 0; i < n_qubits; i++) {
        tensor_t *t = mps->tensors[i];
        uint32_t idx0[3] = {0, 0, 0};
        uint32_t idx1[3] = {0, 1, 0};
        tensor_set(t, idx0, cos(tilt));
        tensor_set(t, idx1, sin(tilt));
    }

    tn_gate_2q_t zz_gate = create_imag_time_zz(tau * J);
    tn_gate_1q_t x_gate = create_imag_time_x(tau * h);

    printf("N = %u qubits\n\n", n_qubits);

    // Test INITIAL state - should be translationally invariant
    printf("INITIAL STATE (product state - should be uniform):\n");
    printf("  Position   <Z>\n");
    for (uint32_t q = 0; q < n_qubits; q += 2) {
        double z = tn_expectation_z(mps, q);
        printf("  %2u        %.6f\n", q, z);
    }

    // Apply 5 Trotter steps
    for (int step = 1; step <= 5; step++) {
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

    // Test EVOLVED state - should show boundary effects
    printf("\nAFTER 5 TROTTER STEPS (should show boundary effects):\n");
    printf("  Position   <Z>\n");
    for (uint32_t q = 0; q < n_qubits; q += 2) {
        double z = tn_expectation_z(mps, q);
        printf("  %2u        %.6f\n", q, z);
    }

    // Also test ZZ at different positions
    printf("\n  Bond      <ZZ>\n");
    for (uint32_t q = 0; q < n_qubits - 1; q += 2) {
        double zz = tn_expectation_zz(mps, q, q+1);
        printf("  (%2u,%2u)   %.6f\n", q, q+1, zz);
    }

    // Debug: print actual tensor data at different positions
    printf("\nDEBUG: Tensor norms at each site:\n");
    for (uint32_t i = 0; i < n_qubits; i++) {
        tensor_t *t = mps->tensors[i];
        double norm = 0.0;
        for (uint32_t j = 0; j < t->total_size; j++) {
            norm += cabs(t->data[j]) * cabs(t->data[j]);
        }
        printf("  Site %2u: dims=[%u,%u,%u], norm=%.4e\n",
               i, t->dims[0], t->dims[1], t->dims[2], sqrt(norm));
    }

    tn_mps_free(mps);
    printf("\nDone.\n");
    return 0;
}
