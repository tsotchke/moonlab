/**
 * @file test_size_scaling.c
 * @brief Find the system size where calculation breaks
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

void test_size(uint32_t n_qubits) {
    printf("\n=== Testing N=%u qubits ===\n", n_qubits);

    const double J = 1.0;
    const double h = 0.5;
    const double tau = 0.1;
    const uint32_t max_bond = 128;

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

    // Initialize with small tilt
    double tilt = 0.05;
    for (uint32_t i = 0; i < n_qubits; i++) {
        tensor_t *t = mps->tensors[i];
        uint32_t idx0[3] = {0, 0, 0};
        uint32_t idx1[3] = {0, 1, 0};
        tensor_set(t, idx0, cos(tilt));
        tensor_set(t, idx1, sin(tilt));
    }

    // Initial measurements
    double z0 = tn_expectation_z(mps, n_qubits / 2);
    double zz0 = tn_expectation_zz(mps, 0, 1);
    printf("  Initial: <Z>=%.4f, <ZZ>=%.4f\n", z0, zz0);

    // Create gates
    tn_gate_2q_t zz_gate = create_imag_time_zz(tau * J);
    tn_gate_1q_t x_gate = create_imag_time_x(tau * h);

    // Do 5 Trotter steps
    for (int step = 1; step <= 5; step++) {
        // Apply ZZ gates on even bonds
        for (uint32_t i = 0; i < n_qubits - 1; i += 2) {
            double trunc_err = 0.0;
            tn_apply_gate_2q(mps, i, i+1, &zz_gate, &trunc_err);
        }

        // Apply ZZ gates on odd bonds
        for (uint32_t i = 1; i < n_qubits - 1; i += 2) {
            double trunc_err = 0.0;
            tn_apply_gate_2q(mps, i, i+1, &zz_gate, &trunc_err);
        }

        // Apply X gates
        for (uint32_t i = 0; i < n_qubits; i++) {
            tn_apply_gate_1q(mps, i, &x_gate);
        }

        // SVD normalization during truncation keeps state approximately normalized.
        // Skip explicit normalization to avoid error-prone transfer matrix computation.
        // (Each SVD normalizes singular values, maintaining ||ψ|| ≈ 1)

        // Measure at BOUNDARY and BULK to show position dependence
        double z_boundary = tn_expectation_z(mps, 0);        // Boundary site
        double z_bulk = tn_expectation_z(mps, n_qubits / 2); // Bulk site (center)
        double zz_boundary = tn_expectation_zz(mps, 0, 1);   // Boundary bond
        uint32_t center = n_qubits / 2;
        double zz_bulk = tn_expectation_zz(mps, center, center + 1); // Bulk bond

        // Compute max bond dimension - this MUST scale with system size for correlated states
        uint32_t max_bond = 1;
        for (uint32_t i = 0; i < n_qubits - 1; i++) {
            if (mps->bond_dims[i] > max_bond) max_bond = mps->bond_dims[i];
        }

        // Compute center bond dimension specifically
        uint32_t center_bond = mps->bond_dims[center];

        printf("  Step %d: Z_bnd=%.4f Z_blk=%.4f  ZZ_bnd=%.4f ZZ_blk=%.4f  max_chi=%u center_chi=%u\n",
               step, z_boundary, z_bulk, zz_boundary, zz_bulk, max_bond, center_bond);
    }

    tn_mps_free(mps);
}

int main(void) {
    printf("Finding system size where calculation breaks\n");
    printf("Parameters: J=1.0, h=0.5, tau=0.1, max_bond=128\n");

    uint32_t sizes[] = {4, 20, 50, 100, 150, 200};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        test_size(sizes[i]);
    }

    printf("\nDone.\n");
    return 0;
}
