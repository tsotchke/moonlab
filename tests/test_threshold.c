/**
 * @file test_threshold.c
 * @brief Find exact system size threshold where calculation breaks
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

int test_size(uint32_t n_qubits, int n_steps) {
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
        fprintf(stderr, "  Failed to create MPS for %u qubits\n", n_qubits);
        return -1;
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

    int broken_at = -1;

    for (int step = 1; step <= n_steps; step++) {
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

        // CRITICAL: Mark as left-canonical after TEBD sweep
        // This enables O(1) norm calculation instead of error-prone O(n*chi^4) transfer matrix
        mps->canonical = TN_CANONICAL_LEFT;

        // Normalize (now uses fast path since we marked canonical form)
        tn_mps_normalize(mps);

        // Check norm
        double norm = tn_mps_norm(mps);
        if (norm < 0.001) {
            broken_at = step;
            break;
        }
    }

    tn_mps_free(mps);
    return broken_at;
}

int main(void) {
    printf("Finding exact threshold where 100-qubit simulation breaks\n");
    printf("Testing sizes: 60, 70, 80, 90, 100 qubits\n\n");

    // Test specific sizes - run each independently
    uint32_t sizes[] = {60, 70, 80, 90, 100};
    int num_sizes = 5;

    for (int i = 0; i < num_sizes; i++) {
        uint32_t n = sizes[i];
        printf("Testing N=%u...\n", n);
        fflush(stdout);

        int broken = test_size(n, 5);
        if (broken < 0) {
            printf("  N=%3u: FAILED TO CREATE\n", n);
        } else if (broken > 0) {
            printf("  N=%3u: BROKEN at step %d\n", n, broken);
        } else {
            printf("  N=%3u: OK (all 5 steps)\n", n);
        }
        fflush(stdout);
    }

    printf("\nDone.\n");
    return 0;
}
