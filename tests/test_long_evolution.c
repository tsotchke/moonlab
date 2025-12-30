/**
 * @file test_long_evolution.c
 * @brief Test longer evolution to see system-size dependence
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

void test_size(uint32_t n_qubits, int n_steps) {
    printf("\n=== Testing N=%u qubits, %d steps ===\n", n_qubits, n_steps);

    const double J = 1.0;
    const double h = 0.5;
    const double tau = 0.1;
    const uint32_t max_bond = 64;

    tn_state_config_t config = {
        .max_bond_dim = max_bond,
        .svd_cutoff = 1e-12,
        .max_truncation_error = 1e-10,
        .track_truncation = true,
        .auto_canonicalize = false,
        .target_form = TN_CANONICAL_NONE
    };

    tn_mps_state_t *mps = tn_mps_create_zero(n_qubits, &config);
    if (!mps) {
        printf("  ERROR: Failed to create MPS\n");
        return;
    }

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

    printf("  Step   Z_bulk   ZZ_bulk  max_chi\n");

    for (int step = 0; step <= n_steps; step++) {
        if (step > 0) {
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

        if (step % 10 == 0 || step == n_steps) {
            double z_bulk = tn_expectation_z(mps, n_qubits / 2);
            uint32_t center = n_qubits / 2;
            double zz_bulk = tn_expectation_zz(mps, center, center + 1);

            uint32_t max_bond_dim = 1;
            for (uint32_t i = 0; i < n_qubits - 1; i++) {
                if (mps->bond_dims[i] > max_bond_dim) max_bond_dim = mps->bond_dims[i];
            }

            printf("  %4d   %.4f   %.4f     %u\n", step, z_bulk, zz_bulk, max_bond_dim);
        }
    }

    tn_mps_free(mps);
}

int main(void) {
    printf("Testing longer evolution for system-size dependence\n");
    printf("Running 50 Trotter steps (τ=0.1) for different system sizes\n");
    printf("Finite-size effects should appear when correlation length > N/2\n");

    // Test small systems where finite-size effects should be visible
    test_size(8, 50);
    test_size(16, 50);
    test_size(32, 50);
    test_size(64, 50);

    printf("\nDone.\n");
    return 0;
}
